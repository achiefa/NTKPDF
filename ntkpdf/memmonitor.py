"""
ntkpdf.memmonitor

A tiny, real-time resident-memory (RSS) monitor for long ``ntkanalysis`` runs.

Enabled with ``ntkpdf scripts/ntkanalysis.py --monitor-memory``. It samples *this*
process's RSS in a daemon thread and renders a compact Unicode sparkline of the
recent history plus the current and peak usage. It exists because the NTK report's
memory profile is the thing we keep fighting (see the OOM case study in CLAUDE.md);
a live trace makes a regression obvious while the run is still going.

The **write style is chosen from what the target actually is** (via ``isatty``, the
same test ``tqdm`` uses) -- not merely from whether ``NTK_MEM_TTY`` is set. This
matters: a carriage-return ``\\r`` only overwrites in place on a *terminal*; writing
it to a regular file just accumulates every frame into one ever-growing line that
looks like garbage under ``cat``/``less``. The three cases:

* **``NTK_MEM_TTY`` -> a terminal device** (e.g. the ``tty`` of a second window):
  a single *in-place* ``\\r`` line is written there, fully isolated from the report's
  stdout/stderr -- in particular it never fights colibri's ``tqdm`` progress bars.
  Usage: in a second terminal run ``tty`` (prints e.g. ``/dev/ttys004``), then launch
  with ``NTK_MEM_TTY=/dev/ttys004 ntkpdf ... --monitor-memory``.
* **``NTK_MEM_TTY`` -> a regular file**: the file is *rewritten* each tick to hold a
  single clean line (truncate + write, no ``\\r``). View it live from anywhere with
  ``watch -n1 cat <file>`` (or ``tail -f <file>``). The sparkline itself carries the
  recent-history trace, so one line is still a graph. This is the no-tty-wrangling way
  to get a separate-window view.
* **otherwise** -> periodic newline-terminated lines on ``stderr``. These scroll with
  the logs (no ``\\r`` fight with tqdm), just less frequently.

Tuning via environment (all optional):

* ``NTK_MEM_TTY``        -- terminal device, or a file path, to write the live line to.
* ``NTK_MEM_INTERVAL``   -- sample period, seconds (default 2 for a device/file, 15 for stderr).
* ``NTK_MEM_CEILING_GB`` -- sparkline full-scale in GB (default 16; top block = ceiling).

The worker is a daemon thread, so it never blocks process exit; :meth:`_Monitor.stop`
gives a clean shutdown (and a trailing newline on the in-place terminal line).
"""

import logging
import os
import resource
import sys
import threading
import time

log = logging.getLogger(__name__)

# Eight one-eighth steps; index by RSS / ceiling.
_BLOCKS = "▁▂▃▄▅▆▇█"

_DEFAULT_CEILING_GB = 16.0
_DEFAULT_HISTORY = 48          # samples kept for the sparkline window
_INTERVAL_TTY = 2.0
_INTERVAL_STDERR = 15.0


def _peak_rss_gb():
    """True peak RSS (GB) of this process. ``ru_maxrss`` is bytes on macOS and
    kilobytes on Linux (mirrors ``ntkpdf.utils.peak_rss_gb``)."""
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1e9 if sys.platform == "darwin" else m / 1e6


def _spark(history, ceiling_gb):
    """Render ``history`` (GB samples) as a block-character sparkline scaled so the
    top block corresponds to ``ceiling_gb`` (fixed scale -> bar height is comparable
    across runs and over time, unlike an auto-rescaled axis)."""
    n = len(_BLOCKS)
    out = []
    for v in history:
        idx = int(v / ceiling_gb * n) if ceiling_gb > 0 else 0
        out.append(_BLOCKS[min(n - 1, max(0, idx))])
    return "".join(out)


class _Monitor:
    """Daemon-thread RSS sampler. Use :func:`start_memory_monitor` to construct."""

    def __init__(self, proc, out, mode, interval, ceiling_gb, history):
        # mode: "tty" (\r in-place on a terminal), "file" (rewrite one line),
        #       "lines" (newline-terminated append, e.g. stderr).
        self._proc = proc
        self._out = out
        self._mode = mode
        self._interval = interval
        self._ceiling_gb = ceiling_gb
        self._history_len = history
        self._history = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="ntk-memmonitor", daemon=True
        )
        # Remember the rendered width so the in-place line can be cleared cleanly.
        self._last_len = 0

    # -- rendering ---------------------------------------------------------
    def _line(self, cur_gb, peak_gb):
        spark = _spark(self._history, self._ceiling_gb)
        return (
            f"[mem] cur {cur_gb:5.1f}G  peak {peak_gb:5.1f}G  "
            f"{spark}  (0-{self._ceiling_gb:g}G)"
        )

    def _write(self, line):
        try:
            if self._mode == "tty":
                # Terminal: pad to erase any leftover from a longer line, then \r.
                pad = max(0, self._last_len - len(line))
                self._out.write("\r" + line + " " * pad)
                self._last_len = len(line)
            elif self._mode == "file":
                # Regular file: keep it at exactly one line so `cat`/`watch cat`/
                # `tail` stay clean (no accumulating \r frames).
                self._out.seek(0)
                self._out.truncate()
                self._out.write(line + "\n")
            else:  # "lines": newline-terminated append (stderr)
                self._out.write(line + "\n")
            self._out.flush()
            return True
        except (OSError, ValueError):
            # Target vanished (e.g. the second terminal was closed) -> stop quietly.
            return False

    # -- thread body -------------------------------------------------------
    def _tick(self):
        try:
            cur_gb = self._proc.memory_info().rss / 1e9
        except Exception:
            # psutil can raise if the process snapshot is briefly unavailable;
            # skip this tick rather than killing the monitor.
            return True
        self._history.append(cur_gb)
        del self._history[: -self._history_len]
        peak_gb = max(_peak_rss_gb(), cur_gb)
        return self._write(self._line(cur_gb, peak_gb))

    def _loop(self):
        while True:
            if not self._tick():
                return
            if self._stop.wait(self._interval):
                break
        if self._mode == "tty":
            # Leave the final line intact and move off it so the shell prompt /
            # any later output starts on a fresh line.
            try:
                self._out.write("\n")
                self._out.flush()
            except (OSError, ValueError):
                pass

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout=2.0):
        self._stop.set()
        self._thread.join(timeout=timeout)


def _resolve_target():
    """Return ``(stream, mode)``. ``NTK_MEM_TTY`` -> that target, with the mode chosen
    by what it *is*: ``"tty"`` for a terminal device (in-place ``\\r`` line),
    ``"file"`` for a regular file (single rewritten line). No ``NTK_MEM_TTY`` -> stderr
    as ``"lines"`` (newline append). The terminal-vs-file split is what stops ``\\r``
    frames from piling up in a piped file."""
    tty = os.environ.get("NTK_MEM_TTY")
    if tty:
        try:
            stream = open(tty, "w", encoding="utf-8")
        except OSError as e:
            print(
                f"[mem] could not open NTK_MEM_TTY={tty!r} ({e}); "
                "writing periodic lines to stderr instead.",
                file=sys.stderr,
            )
            return sys.stderr, "lines"
        try:
            is_tty = os.isatty(stream.fileno())
        except (OSError, ValueError):
            is_tty = False
        return stream, ("tty" if is_tty else "file")
    return sys.stderr, "lines"


def start_memory_monitor():
    """Start the RSS monitor and return the :class:`_Monitor` (or ``None`` if it
    cannot run -- e.g. ``psutil`` is missing). Caller should hold the handle and call
    ``.stop()`` when the run finishes; the worker is a daemon so this is best-effort.
    """
    try:
        import psutil
    except ImportError:
        log.warning(
            "[mem] --monitor-memory requested but psutil is not installed; "
            "skipping the memory monitor."
        )
        return None

    out, mode = _resolve_target()

    def _env_float(name, default):
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            log.warning("[mem] ignoring non-numeric %s=%r", name, raw)
            return default

    interval = _env_float(
        "NTK_MEM_INTERVAL", _INTERVAL_STDERR if mode == "lines" else _INTERVAL_TTY
    )
    ceiling_gb = _env_float("NTK_MEM_CEILING_GB", _DEFAULT_CEILING_GB)

    tty = os.environ.get("NTK_MEM_TTY")
    if mode == "tty":
        log.info(
            "[mem] live RSS monitor -> %s (in-place, every %gs, full-scale %gG)",
            tty, interval, ceiling_gb,
        )
    elif mode == "file":
        log.info(
            "[mem] RSS monitor -> %s (single rewritten line, every %gs, full-scale "
            "%gG). Watch it from another window with `watch -n1 cat %s` (or "
            "`tail -f %s`). Point NTK_MEM_TTY at a real terminal device for an "
            "in-place line instead.",
            tty, interval, ceiling_gb, tty, tty,
        )
    else:
        log.info(
            "[mem] RSS monitor on stderr (every %gs, full-scale %gG). For an isolated "
            "live line, open a second terminal, run `tty`, and re-run with "
            "NTK_MEM_TTY=<that device>; or NTK_MEM_TTY=<a file> + `watch -n1 cat <file>`.",
            interval, ceiling_gb,
        )

    monitor = _Monitor(
        proc=psutil.Process(os.getpid()),
        out=out,
        mode=mode,
        interval=interval,
        ceiling_gb=ceiling_gb,
        history=_DEFAULT_HISTORY,
    )
    return monitor.start()
