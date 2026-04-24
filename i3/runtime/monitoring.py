"""Resource sampling helpers for the orchestrator dashboard.

This module is deliberately tiny and has only one hard dependency —
``psutil`` — so it can be imported from the orchestrator bootstrap
before the user's virtualenv is fully provisioned.  GPU telemetry is
best-effort: we prefer ``pynvml`` (richest data) then fall back to
``torch.cuda`` if available, then to ``None`` fields when neither is
present (CPU-only box).

Example
-------
>>> from i3.runtime.monitoring import ResourceSampler
>>> snap = ResourceSampler().sample()
>>> print(snap.format_line())
GPU: (none) | CPU 12% | RAM 14.2/32.0 GiB (44%) | Disk D: 189.4 GiB free
"""

from __future__ import annotations

import os
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import psutil

# ---------------------------------------------------------------------------
# Optional imports — these are all soft so the module loads on any box.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised on CUDA boxes only
    import pynvml  # type: ignore[import-not-found]

    _HAS_NVML = True
except Exception:  # - broad: nvml raises many types
    pynvml = None  # type: ignore[assignment]
    _HAS_NVML = False

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Snapshot record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSnapshot:
    """One-shot read of system + GPU resources.

    All memory values are reported in MiB (1024-based) to keep the format
    stable and easy to threshold against.  ``None`` means "not available
    on this box" — e.g. ``gpu_name`` will be ``None`` on a CPU-only
    laptop.
    """

    #: Human GPU name (e.g. ``"NVIDIA GeForce RTX 4050 Laptop GPU"``), or
    #: ``None`` on CPU-only boxes.
    gpu_name: str | None
    #: GPU utilisation %, 0–100, or ``None`` if no GPU / no nvml.
    gpu_util_pct: float | None
    #: VRAM used, MiB, or ``None``.
    vram_used_mib: float | None
    #: VRAM total, MiB, or ``None``.
    vram_total_mib: float | None
    #: CPU util %, 0–100.  Always populated (psutil never fails here).
    cpu_pct: float
    #: RAM used, MiB.
    ram_used_mib: float
    #: RAM total, MiB.
    ram_total_mib: float
    #: Orchestrator-process RSS, MiB.  Useful for spotting leaks in the
    #: dashboard itself.
    proc_rss_mib: float
    #: Free space on the project drive, MiB.
    disk_free_mib: float
    #: Total size of the project drive, MiB.
    disk_total_mib: float
    #: Which drive / path the disk numbers are for (e.g. ``"D:"``).
    disk_path: str
    #: Instantaneous disk read rate, MiB/s (diff-per-second between
    #: successive samples).  ``0.0`` for the very first sample.
    disk_read_mib_s: float = 0.0
    #: Instantaneous disk write rate, MiB/s.
    disk_write_mib_s: float = 0.0
    #: Instantaneous network tx rate, MiB/s.  Useful for spotting
    #: large LLM-API payloads or HF-hub model downloads that stall
    #: a stage.
    net_tx_mib_s: float = 0.0
    #: Instantaneous network rx rate, MiB/s.
    net_rx_mib_s: float = 0.0
    #: Epoch-seconds when this sample was taken.
    timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mib_to_gib(v: float | None) -> float | None:
        return None if v is None else v / 1024.0

    def format_line(self) -> str:
        """Return a one-line plain-text summary (no rich markup).

        Designed for the ``--quiet`` dashboard mode and for logging.
        """
        ram_used = self._mib_to_gib(self.ram_used_mib) or 0.0
        ram_total = self._mib_to_gib(self.ram_total_mib) or 0.0
        disk_free = self._mib_to_gib(self.disk_free_mib) or 0.0
        if self.gpu_name and self.vram_total_mib:
            vram_u = self._mib_to_gib(self.vram_used_mib) or 0.0
            vram_t = self._mib_to_gib(self.vram_total_mib) or 0.0
            util = self.gpu_util_pct if self.gpu_util_pct is not None else 0.0
            gpu_str = (
                f"GPU {self.gpu_name} util {util:.0f}% "
                f"VRAM {vram_u:.1f}/{vram_t:.1f} GiB"
            )
        else:
            gpu_str = "GPU: (none)"
        return (
            f"{gpu_str} | CPU {self.cpu_pct:.0f}% | "
            f"RAM {ram_used:.1f}/{ram_total:.1f} GiB "
            f"({100 * self.ram_used_mib / max(self.ram_total_mib, 1):.0f}%) | "
            f"Disk {self.disk_path} {disk_free:.1f} GiB free"
        )


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class ResourceSampler:
    """Samples system + GPU telemetry.

    The sampler is stateful: it lazily initialises nvml once (since
    ``nvmlInit`` has non-trivial cost) and caches the disk root.  All
    reads are wrapped in ``try/except`` so a transient driver error
    doesn't crash the dashboard.
    """

    def __init__(self, disk_path: str | os.PathLike[str] = "D:") -> None:
        # ``psutil.disk_usage`` needs a concrete path that exists on this
        # host.  Fall back to the project drive if the caller-supplied
        # path is missing (common in CI where ``D:`` may not exist).
        candidate = os.fspath(disk_path)
        if not os.path.exists(candidate):
            candidate = os.fspath(os.getcwd())
        self._disk_path = candidate
        self._process = psutil.Process()
        self._nvml_ready = False
        self._nvml_handle: object | None = None
        self._gpu_name: str | None = None

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()  # type: ignore[union-attr]
                # Pick GPU 0 — the orchestrator doesn't schedule multi-GPU
                # workloads, so a single handle is enough.
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[union-attr]
                raw = pynvml.nvmlDeviceGetName(self._nvml_handle)  # type: ignore[union-attr]
                self._gpu_name = raw.decode() if isinstance(raw, bytes) else str(raw)
                self._nvml_ready = True
            except Exception:
                self._nvml_ready = False
                self._nvml_handle = None
        if not self._nvml_ready and _HAS_TORCH:
            try:
                if torch.cuda.is_available():  # type: ignore[union-attr]
                    self._gpu_name = torch.cuda.get_device_name(0)  # type: ignore[union-attr]
            except Exception:
                self._gpu_name = None

        # Prime cpu_percent so the first real read returns non-zero.
        psutil.cpu_percent(interval=None)

        # Prime disk / network counters so subsequent reads can produce
        # a per-second delta.  These are CPU-system-wide totals from
        # ``psutil.disk_io_counters`` / ``net_io_counters``; we compute
        # the MiB/s rate by diffing against the previous timestamp.
        self._last_sample_ts: float = time.monotonic()
        try:
            dio = psutil.disk_io_counters()
            self._last_disk_read = float(dio.read_bytes) if dio else 0.0
            self._last_disk_write = float(dio.write_bytes) if dio else 0.0
        except Exception:
            self._last_disk_read = 0.0
            self._last_disk_write = 0.0
        try:
            nio = psutil.net_io_counters()
            self._last_net_sent = float(nio.bytes_sent) if nio else 0.0
            self._last_net_recv = float(nio.bytes_recv) if nio else 0.0
        except Exception:
            self._last_net_sent = 0.0
            self._last_net_recv = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(self) -> ResourceSnapshot:
        """Return a fresh :class:`ResourceSnapshot`.

        Non-blocking; a single call typically costs <2 ms on Windows.
        """
        gpu_util, vram_used, vram_total = self._sample_gpu()
        vm = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage(self._disk_path)
            disk_free = disk.free / (1024 * 1024)
            disk_total = disk.total / (1024 * 1024)
        except Exception:
            disk_free = 0.0
            disk_total = 0.0

        try:
            rss = self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            rss = 0.0

        cpu = psutil.cpu_percent(interval=None)

        # -- I/O rate deltas -----------------------------------------
        # Compute (bytes_now - bytes_prev) / (t_now - t_prev) so we
        # report true MiB/s between successive sample() calls.  Caller
        # controls the sampling cadence; at 500 ms it gives a smooth
        # enough display without quantisation artefacts.
        now = time.monotonic()
        dt = max(now - self._last_sample_ts, 1e-6)

        try:
            dio = psutil.disk_io_counters()
            read_b = float(dio.read_bytes) if dio else 0.0
            write_b = float(dio.write_bytes) if dio else 0.0
            disk_read_rate = (read_b - self._last_disk_read) / dt / (1024 * 1024)
            disk_write_rate = (write_b - self._last_disk_write) / dt / (1024 * 1024)
            self._last_disk_read = read_b
            self._last_disk_write = write_b
        except Exception:
            disk_read_rate = disk_write_rate = 0.0

        try:
            nio = psutil.net_io_counters()
            tx_b = float(nio.bytes_sent) if nio else 0.0
            rx_b = float(nio.bytes_recv) if nio else 0.0
            net_tx_rate = (tx_b - self._last_net_sent) / dt / (1024 * 1024)
            net_rx_rate = (rx_b - self._last_net_recv) / dt / (1024 * 1024)
            self._last_net_sent = tx_b
            self._last_net_recv = rx_b
        except Exception:
            net_tx_rate = net_rx_rate = 0.0

        self._last_sample_ts = now

        return ResourceSnapshot(
            gpu_name=self._gpu_name,
            gpu_util_pct=gpu_util,
            vram_used_mib=vram_used,
            vram_total_mib=vram_total,
            cpu_pct=float(cpu),
            ram_used_mib=vm.used / (1024 * 1024),
            ram_total_mib=vm.total / (1024 * 1024),
            proc_rss_mib=rss,
            disk_free_mib=disk_free,
            disk_total_mib=disk_total,
            disk_path=self._disk_path,
            disk_read_mib_s=max(0.0, disk_read_rate),
            disk_write_mib_s=max(0.0, disk_write_rate),
            net_tx_mib_s=max(0.0, net_tx_rate),
            net_rx_mib_s=max(0.0, net_rx_rate),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_gpu(
        self,
    ) -> tuple[float | None, float | None, float | None]:
        """Return ``(util_pct, vram_used_mib, vram_total_mib)``.

        Prefers nvml for utilisation numbers; falls back to torch if
        nvml isn't available but torch + CUDA is.
        """
        if self._nvml_ready and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(  # type: ignore[union-attr]
                    self._nvml_handle
                )
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)  # type: ignore[union-attr]
                return (
                    float(util.gpu),
                    mem.used / (1024 * 1024),
                    mem.total / (1024 * 1024),
                )
            except Exception:
                pass  # fall through to torch path

        if _HAS_TORCH:
            try:
                if torch.cuda.is_available():  # type: ignore[union-attr]
                    free_b, total_b = torch.cuda.mem_get_info(0)  # type: ignore[union-attr]
                    used_b = total_b - free_b
                    return (
                        None,  # utilisation not known without nvml
                        used_b / (1024 * 1024),
                        total_b / (1024 * 1024),
                    )
            except Exception:
                pass

        return None, None, None

    def close(self) -> None:
        """Shut nvml down cleanly.  Safe to call more than once."""
        if self._nvml_ready and _HAS_NVML:
            try:
                pynvml.nvmlShutdown()  # type: ignore[union-attr]
            except Exception:
                pass
            self._nvml_ready = False


# ---------------------------------------------------------------------------
# Sparkline history — keeps a rolling ring buffer of samples so the dashboard
# can render mini time-series charts next to the point-in-time values.
# ---------------------------------------------------------------------------


# Unicode 1/8-block glyphs for sparkline rendering.  Index 0 = empty,
# index 7 = full block.  Good on every modern terminal font.
_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int | None = None) -> str:
    """Render a values list as a compact unicode sparkline.

    ``width`` — if set, truncates or left-pads the output to that many
    characters.  Without a width we emit one char per value.  Empty
    input returns an empty string.
    """
    if not values:
        return ""
    if width is not None and len(values) > width:
        # Take the most recent ``width`` samples so the rightmost bar
        # reflects the freshest value.
        values = values[-width:]

    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-9:
        # Flat line — render one mid-height bar so the user sees it's
        # reporting data rather than "nothing".
        return _SPARK_BLOCKS[4] * len(values)
    out: list[str] = []
    for v in values:
        norm = (v - lo) / span
        idx = min(len(_SPARK_BLOCKS) - 1, int(round(norm * (len(_SPARK_BLOCKS) - 1))))
        out.append(_SPARK_BLOCKS[idx])
    rendered = "".join(out)
    if width is not None and len(rendered) < width:
        rendered = rendered.rjust(width)
    return rendered


@dataclass
class ResourceHistory:
    """Ring buffer of recent :class:`ResourceSnapshot` values.

    The dashboard renders one sparkline per series (GPU util, VRAM %,
    CPU %, RAM %).  A 120-sample buffer at a 500 ms sampling cadence
    covers the last 60 s — enough context to spot a stage transitioning
    from "idle" to "under load" without being visually noisy.
    """

    capacity: int = 120
    gpu_util: Deque[float] = field(default_factory=deque)
    vram_pct: Deque[float] = field(default_factory=deque)
    cpu_pct: Deque[float] = field(default_factory=deque)
    ram_pct: Deque[float] = field(default_factory=deque)
    disk_read: Deque[float] = field(default_factory=deque)
    disk_write: Deque[float] = field(default_factory=deque)
    net_tx: Deque[float] = field(default_factory=deque)
    net_rx: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        # Cap each series at ``capacity`` so memory use stays flat even
        # on very long runs.
        for d in (
            self.gpu_util,
            self.vram_pct,
            self.cpu_pct,
            self.ram_pct,
            self.disk_read,
            self.disk_write,
            self.net_tx,
            self.net_rx,
        ):
            # deque-with-maxlen needs to be set after construction because
            # dataclass default_factory can't take a maxlen argument.
            while len(d) > self.capacity:
                d.popleft()

    def push(self, snap: ResourceSnapshot) -> None:
        """Append ``snap`` to every series, dropping stale samples."""
        vram_pct = (
            100.0 * (snap.vram_used_mib or 0.0) / max(snap.vram_total_mib or 1.0, 1.0)
            if snap.vram_total_mib
            else 0.0
        )
        ram_pct = 100.0 * snap.ram_used_mib / max(snap.ram_total_mib, 1.0)

        for d, v in (
            (self.gpu_util, snap.gpu_util_pct or 0.0),
            (self.vram_pct, vram_pct),
            (self.cpu_pct, snap.cpu_pct),
            (self.ram_pct, ram_pct),
            (self.disk_read, snap.disk_read_mib_s),
            (self.disk_write, snap.disk_write_mib_s),
            (self.net_tx, snap.net_tx_mib_s),
            (self.net_rx, snap.net_rx_mib_s),
        ):
            d.append(float(v))
            while len(d) > self.capacity:
                d.popleft()

    # ------------------------------------------------------------------
    # Sparklines ready for the dashboard
    # ------------------------------------------------------------------
    def spark(self, series: str, width: int = 24) -> str:
        """Return a unicode sparkline for the named series."""
        mapping: dict[str, Deque[float]] = {
            "gpu": self.gpu_util,
            "vram": self.vram_pct,
            "cpu": self.cpu_pct,
            "ram": self.ram_pct,
            "disk_r": self.disk_read,
            "disk_w": self.disk_write,
            "net_tx": self.net_tx,
            "net_rx": self.net_rx,
        }
        data = mapping.get(series)
        if data is None:
            return ""
        return sparkline(list(data), width=width)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _bar(pct: float, width: int = 10) -> str:
    """Render a mini text bar (no rich markup)."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round((pct / 100.0) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _threshold_colour(pct: float, warn: float, crit: float) -> str:
    """Return a rich colour name for a utilisation percentage.

    ``pct`` is 0–100; ``warn`` and ``crit`` are the thresholds above
    which we shade yellow / red respectively.
    """
    if pct >= crit:
        return "red"
    if pct >= warn:
        return "yellow"
    return "green"


def render_resource_panel(
    snap: ResourceSnapshot,
    history: "ResourceHistory | None" = None,
):  # type: ignore[no-untyped-def]
    """Return a compact ``rich.panel.Panel`` for the dashboard.

    When ``history`` is supplied, each row also gets a 24-char unicode
    sparkline showing the last minute of values — this is what turns
    the panel from a point-in-time readout into a precise, responsive
    telemetry display.

    The import of ``rich`` is deferred so this module remains usable
    in non-rich contexts (e.g. the ``--quiet`` path).
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    def _spark(series: str) -> str:
        return history.spark(series, width=24) if history else ""

    table = Table.grid(padding=(0, 1))
    # label | main reading | sparkline
    table.add_column(justify="left", no_wrap=True)
    table.add_column(justify="left")
    table.add_column(justify="left", style="magenta dim", no_wrap=True)

    # GPU row ------------------------------------------------------------
    if snap.gpu_name and snap.vram_total_mib:
        util = snap.gpu_util_pct if snap.gpu_util_pct is not None else 0.0
        vram_pct = 100.0 * (snap.vram_used_mib or 0.0) / max(snap.vram_total_mib, 1.0)
        util_colour = _threshold_colour(util, 75, 95)
        vram_colour = _threshold_colour(vram_pct, 80, 95)
        gpu_cell = Text.assemble(
            (f"{snap.gpu_name[:28]:<28}", "bold"),
            ("  ", ""),
            (f"{_bar(util)} {util:3.0f}%", util_colour),
            ("  VRAM ", "dim"),
            (
                f"{(snap.vram_used_mib or 0) / 1024:.1f}/"
                f"{snap.vram_total_mib / 1024:.1f} GiB "
                f"({vram_pct:.0f}%)",
                vram_colour,
            ),
        )
    else:
        gpu_cell = Text("(no CUDA GPU detected)", style="dim")
    table.add_row(Text("GPU ", style="bold"), gpu_cell, _spark("gpu"))

    # VRAM sparkline row (only when we actually have VRAM data).
    if snap.gpu_name and snap.vram_total_mib and history:
        table.add_row(Text("VRAM", style="bold"), Text(""), _spark("vram"))

    # CPU row ------------------------------------------------------------
    cpu_colour = _threshold_colour(snap.cpu_pct, 75, 95)
    table.add_row(
        Text("CPU ", style="bold"),
        Text(f"{_bar(snap.cpu_pct)} {snap.cpu_pct:3.0f}%", style=cpu_colour),
        _spark("cpu"),
    )

    # RAM row ------------------------------------------------------------
    ram_pct = 100.0 * snap.ram_used_mib / max(snap.ram_total_mib, 1.0)
    ram_colour = _threshold_colour(ram_pct, 80, 95)
    table.add_row(
        Text("RAM ", style="bold"),
        Text(
            f"{_bar(ram_pct)} "
            f"{snap.ram_used_mib / 1024:.1f}/"
            f"{snap.ram_total_mib / 1024:.1f} GiB "
            f"({ram_pct:.0f}%)",
            style=ram_colour,
        ),
        _spark("ram"),
    )

    # Disk row -----------------------------------------------------------
    disk_free_gib = snap.disk_free_mib / 1024
    # Thresholds are inverted (low free = bad).
    if disk_free_gib < 5:
        disk_colour = "red"
    elif disk_free_gib < 25:
        disk_colour = "yellow"
    else:
        disk_colour = "green"
    table.add_row(
        Text(f"Disk {snap.disk_path} ", style="bold"),
        Text(
            f"free {disk_free_gib:.1f} GiB  "
            f"R {snap.disk_read_mib_s:5.1f} MiB/s  "
            f"W {snap.disk_write_mib_s:5.1f} MiB/s",
            style=disk_colour,
        ),
        _spark("disk_r"),
    )

    # Network row — helpful to spot LLM/API throughput during training
    # or package downloads during install.
    net_desc = (
        f"↑ {snap.net_tx_mib_s:5.2f} MiB/s  ↓ {snap.net_rx_mib_s:5.2f} MiB/s"
    )
    table.add_row(Text("Net ", style="bold"), Text(net_desc, style="dim"), _spark("net_rx"))

    # Orchestrator RSS — low-key, no threshold colour.
    table.add_row(
        Text("Proc ", style="bold"),
        Text(f"RSS {snap.proc_rss_mib:.0f} MiB", style="dim"),
        Text(""),
    )

    return Panel(table, title="Resources (sparklines = last 60 s)", border_style="cyan", padding=(0, 1))


__all__ = [
    "ResourceHistory",
    "ResourceSampler",
    "ResourceSnapshot",
    "render_resource_panel",
    "sparkline",
]


# Silence unused-import lints when neither nvml nor torch is present.
_ = shutil
