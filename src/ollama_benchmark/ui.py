"""Progress reporting: a plain logger and an interactive ``rich`` dashboard.

The runner emits lifecycle events to a Reporter. Two implementations:

* ``PlainReporter`` — line-oriented logging (non-TTY / ``--ui plain``).
* ``RichReporter``  — a live multi-panel dashboard with a keyboard listener:
  header + overall progress, a "current model" panel with streaming metrics,
  a live GPU panel, and a scrollable / sortable summary table of completed
  models.

Interactive keys (rich dashboard, on a TTY):
  SHIFT+<key>  sort the completed table by a column (see legend)
  r            reverse the current sort
  j / k, ↑ / ↓ scroll the completed table
  SPACE, PgUp/PgDn  page the completed table
  g / b        jump to top / bottom (bottom re-enables auto-follow)
  f            toggle auto-follow (stick to newest row)
  q            quit — asks y/n confirmation (during run)
  SHIFT+Q      quit immediately (after run completes)
  CTRL+C       quit immediately (after run completes)
"""

from __future__ import annotations

import os
import select
import sys
import threading
import time
from collections import deque, namedtuple


class Reporter:
    """No-op base class; override the hooks you care about."""

    def run_start(self, total: int, card: str) -> None: ...
    def model_start(self, meta: dict) -> None: ...
    def model_loaded(self, placement: dict | None) -> None: ...
    def token(self, text: str) -> None: ...
    def gpu(self, reading: dict) -> None: ...
    def model_done(self, record: dict) -> None: ...
    def run_done(self, results: list[dict]) -> None: ...
    def should_abort(self) -> bool: return False
    def set_local_metrics(self, flag: bool) -> None: ...

    def __enter__(self): return self
    def __exit__(self, *exc): ...


class PlainReporter(Reporter):
    def __init__(self):
        self._total = 0
        self._idx = 0

    def run_start(self, total: int, card: str) -> None:
        self._total = total
        print(f"[info] benchmarking {total} model(s), GPU card={card}")

    def model_start(self, meta: dict) -> None:
        self._idx += 1
        print(f"[{self._idx}/{self._total}] {meta['model']} ...", flush=True)

    def model_done(self, record: dict) -> None:
        perf = record.get("performance") or {}
        gpu = record.get("gpu") or {}
        if record["ok"]:
            if record["workload"] == "completion":
                print(f"    ok  {perf.get('tokens_per_s')} tok/s"
                      f"  vram_delta={gpu.get('vram_delta_mb')}MB"
                      f"  peak_power={gpu.get('power_peak_w')}W")
            else:
                print(f"    ok  embedding_dim={perf.get('embedding_dim')}"
                      f"  vram_delta={gpu.get('vram_delta_mb')}MB")
        else:
            print(f"    FAIL: {record['error']}")


# --------------------------------------------------------------------------- #
# Column model for the completed table
# --------------------------------------------------------------------------- #

Col = namedtuple("Col", "id header key numeric justify style")

# ``key`` is the UPPERCASE letter that sorts by this column (SHIFT+key).
COLUMNS = [
    Col("model",     "model",     "M", False, "left",  "cyan"),
    Col("params",    "params",    "P", True,  "right", None),
    Col("quant",     "quant",     "Q", False, "left",  "dim"),
    Col("ctx",       "ctx",       "C", True,  "right", None),
    Col("disk_mb",   "disk MB",   "D", True,  "right", None),
    Col("placement", "placement", "G", False, "left",  "blue"),
    Col("tok_s",     "tok/s",     "T", True,  "right", None),
    Col("vram_mb",   "vram MB",   "V", True,  "right", None),
    Col("ram_mb",    "ram MB",    "R", True,  "right", None),
    Col("power",     "pwr W",     "W", True,  "right", None),
    Col("temp",      "°C",        "H", True,  "right", None),
    Col("load_s",    "load s",    "L", True,  "right", None),
    Col("caps",      "caps",      "S", False, "left",  "dim"),
]
_COL_BY_KEY = {c.key: c for c in COLUMNS}


def _mb(v):
    return f"{v/1e6:,.0f}" if v else "-"


def _fmt(v, suffix="", nd=1):
    if v is None:
        return "-"
    return f"{v:.{nd}f}{suffix}"


def _to_int(v):
    if v is None:
        return None
    try:
        return int(str(v).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _params_to_num(v) -> float:
    """Turn '31B' / '268.10M' / '14.7B' into a comparable number."""
    if not v:
        return -1.0
    s = str(v).strip().upper()
    mult = 1.0
    if s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("K"):
        mult, s = 1e3, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return -1.0


# --------------------------------------------------------------------------- #
# Keyboard listener (raw single-key reads in a background thread)
# --------------------------------------------------------------------------- #

_ESCAPE_MAP = {
    b"\x1b[A": "UP", b"\x1b[B": "DOWN", b"\x1b[C": "RIGHT", b"\x1b[D": "LEFT",
    b"\x1b[5~": "PGUP", b"\x1b[6~": "PGDN", b"\x1b[H": "HOME", b"\x1b[F": "END",
    b" ": "SPACE", b"\r": "ENTER", b"\n": "ENTER",
    b"\x03": "CTRL_C",  # CTRL+C in cbreak mode (belt-and-suspenders for SIGINT)
}


class KeyListener(threading.Thread):
    def __init__(self, on_key):
        super().__init__(daemon=True)
        self.on_key = on_key
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno()
        self._old = None

    def run(self) -> None:
        import termios
        import tty
        try:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:  # noqa: BLE001 - no controlling tty
            return
        try:
            while not self._stop.is_set():
                r, _, _ = select.select([self._fd], [], [], 0.1)
                if not r:
                    continue
                key = self._read_key()
                if key:
                    try:
                        self.on_key(key)
                    except Exception:  # noqa: BLE001 - a bad key must not crash
                        pass
        finally:
            self._restore()

    def _read_key(self) -> str:
        ch = os.read(self._fd, 1)
        if ch == b"\x1b":
            seq = ch
            while True:
                r, _, _ = select.select([self._fd], [], [], 0.02)
                if not r:
                    break
                seq += os.read(self._fd, 1)
            return _ESCAPE_MAP.get(seq, "ESC")
        if ch in _ESCAPE_MAP:
            return _ESCAPE_MAP[ch]
        try:
            return ch.decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            return ""

    def stop(self) -> None:
        self._stop.set()
        self._restore()

    def _restore(self) -> None:
        if self._old is not None:
            import termios
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:  # noqa: BLE001
                pass
            self._old = None


# --------------------------------------------------------------------------- #
# Dashboard state + rendering
# --------------------------------------------------------------------------- #

class _Dashboard:
    """Holds mutable run state; ``__rich__`` renders it each refresh."""

    def __init__(self, total: int, card: str):
        self.total = total
        self.card = card
        self.start = time.perf_counter()
        self.console = None  # set by the reporter for height-aware scrolling

        self.done = 0
        self.ok = 0
        self.fail = 0

        # current model
        self.cur_name = "-"
        self.cur_workload = "-"
        self.cur_caps: list[str] = []
        self.cur_size_mb = None
        self.cur_details: dict = {}
        self.cur_placement: dict | None = None
        self.cur_status = "starting"
        self.cur_kv_mb: float | None = None

        # live generation
        self.tok_count = 0
        self.first_token_t: float | None = None
        self.preview = deque(maxlen=1200)

        # live resources: payload = {"cards": {card: {...}}, "cpu_pct", "ram_*"}
        self.live: dict = {}
        self.known_cards: list[str] = []      # discovered GPU ids, stable order
        self.card_peaks: dict[str, dict] = {}  # {card: {"vram":, "power":}}
        # live cpu / ram
        self.cpu_peak = 0.0
        self.ram_peak = 0.0
        self.cpu_cores = os.cpu_count() or 1
        self.gpu_vendor = None
        self.local_metrics = True  # False for remote backends → panels show n/d

        # completed rows
        self.rows: list[dict] = []

        # interaction state
        self.sort_key: str | None = "tok_s"
        self.sort_reverse = True
        self.scroll = 0
        self.follow = True
        self._page = 10
        self.quit_prompt = False
        self.paused = False
        self.run_complete = False  # set by reporter when all models done
        self._abort = threading.Event()

    # ---- state mutation ---------------------------------------------------- #

    def set_cards(self, cards: list[str], vendor: str | None) -> None:
        self.known_cards = sorted(cards)
        self.gpu_vendor = vendor

    def _reset_peaks(self) -> None:
        self.card_peaks = {}
        self.cpu_peak = 0.0
        self.ram_peak = 0.0

    def start_model(self, meta: dict) -> None:
        self.cur_name = meta["model"]
        self.cur_workload = meta.get("workload", "-")
        self.cur_caps = meta.get("capabilities") or []
        self.cur_size_mb = meta.get("size_mb")
        self.cur_details = meta.get("details") or {}
        self.cur_placement = None
        self.cur_status = "loading"
        self.cur_kv_mb = None
        self.tok_count = 0
        self.first_token_t = None
        self.preview.clear()
        self._reset_peaks()

    def add_token(self, text: str) -> None:
        if self.first_token_t is None:
            self.first_token_t = time.perf_counter()
            self.cur_status = "generating"
        self.tok_count += 1
        self.preview.extend(text)

    def add_gpu(self, payload: dict) -> None:
        # payload carries all GPU cards plus system-wide CPU/RAM.
        self.live = payload
        cards = payload.get("cards") or {}
        for card, r in cards.items():
            if card not in self.known_cards:
                self.known_cards = sorted(set(self.known_cards) | {card})
            cp = self.card_peaks.setdefault(card, {"vram": 0.0, "power": 0.0})
            if r.get("vram_used_b"):
                cp["vram"] = max(cp["vram"], r["vram_used_b"])
            if r.get("power_w"):
                cp["power"] = max(cp["power"], r["power_w"])
        cpu = payload.get("cpu_pct")
        if cpu is not None:
            self.cpu_peak = max(self.cpu_peak, cpu)
        ram = payload.get("ram_used_b")
        if ram:
            self.ram_peak = max(self.ram_peak, ram)

    def finish_model(self, record: dict) -> None:
        self.done += 1
        if record["ok"]:
            self.ok += 1
        else:
            self.fail += 1
        perf = record.get("performance") or {}
        gpu = record.get("gpu") or {}
        det = record.get("details") or {}
        place = record.get("placement") or {}
        cpu = record.get("cpu") or {}
        # Prefer the *runtime* context actually loaded (from `ollama ps`) over
        # the model's architectural maximum; fall back to the max if unknown.
        runtime_ctx = _to_int(place.get("context"))
        self.rows.append({
            "model": record["model"],
            "workload": record["workload"],
            "params": det.get("parameter_size"),
            "quant": det.get("quantization_level"),
            "ctx": runtime_ctx if runtime_ctx is not None else det.get("context_length"),
            "ctx_max": det.get("context_length"),
            "disk_mb": record.get("size_mb"),
            "placement": place.get("processor"),
            "tok_s": perf.get("tokens_per_s"),
            "dim": perf.get("embedding_dim"),
            "vram_mb": gpu.get("vram_delta_mb"),
            "ram_mb": cpu.get("ram_delta_mb"),
            "power": gpu.get("power_peak_w"),
            "temp": gpu.get("temp_peak_c"),
            "load_s": perf.get("load_duration_s"),
            "ok": record["ok"],
            "caps": ",".join(record.get("capabilities") or []),
        })
        self.cur_kv_mb = record.get("kv_cache_mb")
        self.cur_status = "done"

    # ---- keyboard ---------------------------------------------------------- #

    def handle_key(self, k: str) -> None:
        if self.quit_prompt:
            if k in ("y", "Y"):
                self._abort.set()
                self.quit_prompt = False
            elif k in ("n", "N", "q", "ESC"):
                self.quit_prompt = False
            return

        if k == "q":
            self.quit_prompt = True
            return

        col = _COL_BY_KEY.get(k)  # uppercase letter -> sort column
        if col is not None:
            if self.sort_key == col.id:
                self.sort_reverse = not self.sort_reverse
            else:
                self.sort_key = col.id
                self.sort_reverse = col.numeric  # numbers: high-to-low first
            return

        if k in ("j", "DOWN"):
            self.follow = False
            self.scroll += 1
        elif k in ("k", "UP"):
            self.follow = False
            self.scroll = max(0, self.scroll - 1)
        elif k in ("SPACE", "PGDN"):
            self.follow = False
            self.scroll += self._page
        elif k == "PGUP":
            self.follow = False
            self.scroll = max(0, self.scroll - self._page)
        elif k in ("g", "HOME"):
            self.follow = False
            self.scroll = 0
        elif k in ("b", "END"):
            self.follow = True
        elif k == "f":
            self.follow = not self.follow
        elif k == "r":
            self.sort_reverse = not self.sort_reverse

    # ---- helpers ----------------------------------------------------------- #

    def aborted(self) -> bool:
        return self._abort.is_set()

    def _live_tok_s(self) -> float | None:
        if self.first_token_t is None or self.tok_count == 0:
            return None
        dt = time.perf_counter() - self.first_token_t
        return self.tok_count / dt if dt > 0 else None

    def _elapsed(self) -> str:
        s = int(time.perf_counter() - self.start)
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    def _sorted_rows(self) -> list[dict]:
        if not self.sort_key:
            return self.rows
        key = self.sort_key

        def val(r):
            if key == "params":
                return _params_to_num(r.get("params"))
            v = r.get(key)
            return v.lower() if isinstance(v, str) else v

        present = [r for r in self.rows if r.get(key) is not None]
        missing = [r for r in self.rows if r.get(key) is None]
        try:
            present.sort(key=val, reverse=self.sort_reverse)
        except TypeError:
            present.sort(key=lambda r: str(val(r)), reverse=self.sort_reverse)
        return present + missing  # unknown values always at the end

    # ---- rendering --------------------------------------------------------- #

    def __rich__(self):
        from rich.layout import Layout

        from rich.layout import Layout

        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=4),
            Layout(name="mid", size=15),
            Layout(self._summary(), name="summary"),
        )
        # Top row: current | cpu | gpu0 | gpu1 | ...  The cpu and gpu panels are
        # fixed-width; the current-model panel is flexible and absorbs any extra
        # width (no empty gap on the right). Only the completed table (below)
        # grows vertically / scrolls.
        cards = self.known_cards or list((self.live.get("cards") or {}).keys())
        if not self.local_metrics:
            cards = ["card0"]  # single placeholder panel showing n/d
        children = [
            Layout(self._current_panel(), name="current", ratio=1, minimum_size=48),
            Layout(self._cpu_panel(), name="cpu", size=32),
        ]
        for card in (sorted(cards) or ["card0"]):
            children.append(Layout(self._gpu_panel(card), name=card, size=40))
        layout["mid"].split_row(*children)
        return layout

    def _header(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress_bar import ProgressBar

        pct = self.done / self.total if self.total else 0
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        bar = ProgressBar(total=self.total or 1, completed=self.done, width=None)
        left = (f"[bold]{self.done}/{self.total}[/]  "
                f"[green]ok {self.ok}[/]  [red]fail {self.fail}[/]  "
                f"elapsed [cyan]{self._elapsed()}[/]  card [magenta]{self.card}[/]")
        grid.add_row(left, f"[dim]{pct*100:4.0f}%[/]")
        grid.add_row(bar, "")
        if self.quit_prompt:
            grid.add_row("[bold black on yellow] Quit benchmark? press "
                         "y to confirm · n to cancel [/]", "")
        elif self.paused:
            grid.add_row("[bold black on cyan] PAUSED — select text with the "
                         "mouse to copy, then press p to resume [/]", "")
        border = "yellow" if self.quit_prompt else ("cyan" if self.paused else "blue")
        return Panel(grid, title="🔬 Ollama Benchmark", border_style=border)

    def _current_panel(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Group
        from rich.text import Text

        d = self.cur_details
        meta = Table.grid(padding=(0, 1))
        meta.add_column(style="dim", justify="right")
        meta.add_column()
        placement = "-"
        if self.cur_placement:
            placement = (f"{self.cur_placement.get('processor','-')}"
                         f"  ({self.cur_placement.get('size','?')})")
        live_tps = self._live_tok_s()
        meta.add_row("model", f"[bold yellow]{self.cur_name}[/]")
        meta.add_row("status", _status_text(self.cur_status))
        meta.add_row("caps", ",".join(self.cur_caps) or "-")
        meta.add_row("params", f"{d.get('parameter_size') or '-'}"
                     f"  {d.get('quantization_level') or ''}")
        runtime_ctx = _to_int(self.cur_placement.get("context")) if self.cur_placement else None
        ctx_max = d.get("context_length")
        ctx_disp = runtime_ctx if runtime_ctx is not None else (ctx_max or "-")
        ctx_str = f"{ctx_disp}"
        if ctx_max and str(ctx_disp) != str(ctx_max):
            ctx_str += f"  [dim](max {ctx_max:,})[/]"
        meta.add_row("ctx / disk", f"{ctx_str}"
                     f"  /  {_fmt(self.cur_size_mb, ' MB', 0)}")
        if self.cur_kv_mb is not None:
            meta.add_row("kv cache", f"{_fmt(self.cur_kv_mb, ' MB', 0)}")
        meta.add_row("placement", placement)
        meta.add_row("tokens", f"[bold]{self.tok_count}[/]"
                     f"   live [bold green]{_fmt(live_tps,' tok/s')}[/]")

        preview = "".join(self.preview).replace("\n", " ")
        if len(preview) > 600:
            preview = "…" + preview[-600:]
        body = Group(meta, Text(""),
                     Text(preview or "(waiting for tokens…)",
                          style="dim italic"))
        return Panel(body, title="▶ current model", border_style="yellow")

    def _nd_panel(self, title: str, rows: list[str], style: str):
        from rich.panel import Panel
        from rich.table import Table

        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column()
        for r in rows:
            t.add_row(r, "[dim]n/d[/]")
        return Panel(t, title=f"{title} [dim](remote)[/]", border_style=style)

    def _gpu_panel(self, card: str):
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress_bar import ProgressBar
        from rich.console import Group

        if not self.local_metrics:
            return self._nd_panel("🎮 GPU", ["VRAM", "power", "temp", "util"],
                                  "magenta")

        g = (self.live.get("cards") or {}).get(card) or {}
        peaks = self.card_peaks.get(card) or {}
        used = g.get("vram_used_b")
        total = g.get("vram_total_b")
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("VRAM", f"{_mb(used)} / {_mb(total)} MB")
        t.add_row("peak VRAM", f"{_mb(peaks.get('vram', 0))} MB")
        t.add_row("power", f"{_fmt(g.get('power_w'),' W')}"
                  f"   peak {_fmt(peaks.get('power', 0),' W')}")
        t.add_row("temp", _fmt(g.get("temp_c"), " °C", 0))
        t.add_row("util", _fmt(g.get("gpu_use_pct"), " %", 0))

        frac = (used / total) if (used and total) else 0
        bar = ProgressBar(total=1.0, completed=frac, width=None,
                          complete_style="magenta")
        label = card.replace("card", "GPU")
        vend = f" {self.gpu_vendor}" if self.gpu_vendor else ""
        return Panel(Group(t, bar), title=f"🎮 {label}{vend}",
                     border_style="magenta")

    def _cpu_panel(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress_bar import ProgressBar
        from rich.console import Group

        if not self.local_metrics:
            return self._nd_panel(f"🧠 CPU ({self.cpu_cores} cores)",
                                  ["CPU", "freq", "temp", "load", "RAM"], "cyan")

        cpu = self.live.get("cpu_pct")
        freq = self.live.get("cpu_freq_mhz")
        temp = self.live.get("cpu_temp_c")
        load = self.live.get("load")
        used = self.live.get("ram_used_b")
        total = self.live.get("ram_total_b")
        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("CPU", f"{_fmt(cpu, ' %', 0)}"
                  f"   peak {_fmt(self.cpu_peak, ' %', 0)}")
        cpu_bar = ProgressBar(total=100.0, completed=cpu or 0, width=None,
                              complete_style="cyan")
        t.add_row("freq", f"{freq/1000:.2f} GHz" if freq else "-")
        t.add_row("temp", _fmt(temp, " °C", 0))
        if load:
            t.add_row("load", f"{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}")
        t.add_row("RAM", f"{_mb(used)} / {_mb(total)} MB")
        ram_frac = (used / total) if (used and total) else 0
        ram_bar = ProgressBar(total=1.0, completed=ram_frac, width=None,
                              complete_style="cyan")
        return Panel(Group(t, cpu_bar, ram_bar),
                     title=f"🧠 CPU ({self.cpu_cores} cores)",
                     border_style="cyan")

    def _cell(self, col: Col, r: dict) -> str:
        if col.id == "tok_s":
            if not r["ok"]:
                return "[red]FAIL[/]"
            if r["workload"] == "embedding":
                return f"[dim]dim {r['dim']}[/]"
            return _fmt(r.get("tok_s"))
        if col.id == "placement":
            p = r.get("placement") or "-"
            return f"[{'green' if p == '100% GPU' else 'yellow'}]{p}[/]"
        v = r.get(col.id)
        if v is None:
            return "-"
        if col.id in ("disk_mb", "vram_mb", "ram_mb", "power", "temp"):
            return _fmt(v, nd=0)
        if col.id == "load_s":
            return _fmt(v, nd=2)
        return str(v)

    def _summary(self):
        from rich.panel import Panel
        from rich.table import Table

        rows = self._sorted_rows()
        total = len(rows)

        # Height-aware scroll window sized to the panel's region so that, in
        # follow mode, the newest row is always the last visible one (never
        # cropped). header(4)+mid(15)+panel chrome(3: title/header/subtitle).
        height = self.console.size.height if self.console else 40
        visible = max(1, height - 22)
        self._page = visible
        max_top = max(0, total - visible)
        if self.follow:
            self.scroll = max_top
        self.scroll = min(max(0, self.scroll), max_top)
        window = rows[self.scroll:self.scroll + visible]

        table = Table(expand=True, box=None, pad_edge=False)
        for c in COLUMNS:
            head = c.header
            if self.sort_key == c.id:
                head = f"[bold]{c.header} {'▼' if self.sort_reverse else '▲'}[/]"
            # Uniform left alignment for every header and cell (rich applies the
            # column's justify to both), so nothing looks ragged.
            table.add_column(head, justify="left", style=c.style, no_wrap=True)
        for r in window:
            table.add_row(*[self._cell(c, r) for c in COLUMNS])

        pos = f"{self.scroll + 1}-{self.scroll + len(window)}" if total else "0"
        follow = "[green]follow[/]" if self.follow else "[dim]manual[/]"
        title = f"✔ completed ({total})   rows {pos}   {follow}"
        if self.run_complete:
            legend = ("[dim]SHIFT+[/]"
                      + " ".join(c.key for c in COLUMNS)
                      + " [dim]sort · r reverse · ↑↓/jk scroll · g/b top/bottom · "
                        "f follow · p pause(copy)[/]"
                        "   [bold yellow]SHIFT+Q / q / CTRL+C[/] [dim]exit[/]")
        else:
            legend = ("[dim]SHIFT+[/]"
                      + " ".join(c.key for c in COLUMNS)
                      + " [dim]sort · r reverse · ↑↓/jk scroll · g/b top/bottom · "
                        "f follow · p pause(copy) · q quit[/]")
        return Panel(table, title=title, subtitle=legend, border_style="green")


def _status_text(status: str) -> str:
    colors = {"loading": "blue", "generating": "green",
              "done": "dim", "starting": "dim", "embedding": "green"}
    return f"[{colors.get(status,'white')}]{status}[/]"


class RichReporter(Reporter):
    def __init__(self, total_hint: int = 0, card: str = "card0"):
        self._card = card
        self._dash: _Dashboard | None = None
        self._live = None
        self._keys: KeyListener | None = None
        self._paused = False
        self._local_metrics = True
        self._post_run_exit = threading.Event()  # set by q/Q/CTRL+C after run done

    def set_local_metrics(self, flag: bool) -> None:
        self._local_metrics = flag
        if self._dash:
            self._dash.local_metrics = flag

    def run_start(self, total: int, card: str) -> None:
        from rich.live import Live
        from rich.console import Console

        console = Console()
        self._dash = _Dashboard(total=total, card=card)
        self._dash.local_metrics = self._local_metrics
        self._dash.console = console
        # Discover GPU cards + vendor once so panels render from the start.
        try:
            from .gpu import gpu_vendor, sample_gpu
            self._dash.set_cards(list(sample_gpu().keys()), gpu_vendor())
        except Exception:  # noqa: BLE001
            pass
        self._live = Live(self._dash, console=console, refresh_per_second=8,
                          screen=False, transient=False)
        self._live.start()
        if sys.stdin.isatty():
            self._keys = KeyListener(self._on_key)
            self._keys.start()

    def _on_key(self, key: str) -> None:
        # After the run is done, q / Q / CTRL+C exit immediately — no prompt.
        if self._dash and self._dash.run_complete:
            if key in ("q", "Q", "CTRL_C"):
                self._post_run_exit.set()
                return
        # 'p' pauses the live refresh so the frozen frame can be mouse-selected
        # and copied (constant repainting otherwise wipes the selection).
        if key == "p" and self._dash and not self._dash.quit_prompt:
            self._toggle_pause()
            return
        if self._dash:
            self._dash.handle_key(key)

    def _toggle_pause(self) -> None:
        if not self._live:
            return
        self._paused = not self._paused
        if self._paused:
            self._dash.paused = True
            try:
                self._live.refresh()   # paint the "PAUSED" banner once
                self._live.stop()      # freeze: no more repaints
            except Exception:  # noqa: BLE001
                pass
        else:
            self._dash.paused = False
            try:
                self._live.start()     # resume live updates
            except Exception:  # noqa: BLE001
                pass

    def model_start(self, meta: dict) -> None:
        if self._dash:
            self._dash.start_model(meta)

    def model_loaded(self, placement: dict | None) -> None:
        if self._dash:
            self._dash.cur_placement = placement

    def token(self, text: str) -> None:
        if self._dash:
            self._dash.add_token(text)

    def gpu(self, reading: dict) -> None:
        if self._dash:
            self._dash.add_gpu(reading)

    def model_done(self, record: dict) -> None:
        if self._dash:
            self._dash.finish_model(record)

    def should_abort(self) -> bool:
        return bool(self._dash and self._dash.aborted())

    def run_done(self, results: list[dict]) -> None:
        if self._dash:
            self._dash.run_complete = True
        try:
            self._post_run_exit.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.__exit__(None, None, None)

    def __exit__(self, *exc):
        if self._keys:
            try:
                self._keys.stop()
            except Exception:  # noqa: BLE001
                pass
            self._keys = None
        if self._live:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001
                pass
            self._live = None


def make_reporter(mode: str, card: str) -> Reporter:
    """Pick a reporter. ``mode`` is ``auto`` | ``rich`` | ``plain``."""
    if mode == "plain":
        return PlainReporter()
    if mode == "rich":
        return RichReporter(card=card)
    # auto: rich only on a real terminal
    if sys.stdout.isatty():
        try:
            import rich  # noqa: F401
            return RichReporter(card=card)
        except ImportError:
            pass
    return PlainReporter()
