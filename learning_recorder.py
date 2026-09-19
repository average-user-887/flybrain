"""Durable, append-only learning records for the NeuroFly daemon.

The daemon keeps its trial ledger (``trial_history``) and learning curve in
memory and only writes small rolling JSON checkpoints, the oldest of which are
deleted.  A multi-day public run would therefore lose most of its history on
restart.  This module writes two append-only JSON Lines files plus a session
manifest under a data directory, with size-based rotation that never rewrites
or truncates an existing line:

``<data_dir>/session.json``            one manifest per daemon start (overwritten
                                       only at the next start; earlier sessions
                                       are also appended to ``sessions.jsonl``).
``<data_dir>/trials.jsonl``            one line per completed trial / milestone.
``<data_dir>/telemetry_summary.jsonl`` one line per summary interval.

The schema is documented in ``docs/DATA_SCHEMA.md``.

The recorder does not touch the simulation loop.  :class:`RecorderThread`
polls a runner (``neurofly_daemon.ContinuousExperimentRunner``) under its
lock, copies out anything new, and writes it outside the lock.  Trials are
identified by their monotonic ``trial`` number, so the recorder tolerates the
runner truncating or replacing ``trial_history``.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_DATA_SUBDIR",
    "resolve_data_dir",
    "JsonlWriter",
    "read_jsonl",
    "summarise_runner",
    "LearningRecorder",
    "RecorderThread",
]

SCHEMA_VERSION = 1
DEFAULT_DATA_SUBDIR = Path("outputs") / "learning"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024     # rotate a JSONL file past 64 MiB


def resolve_data_dir(project_root: Path, explicit: Optional[str] = None,
                     environ: Optional[Dict[str, str]] = None) -> Path:
    """``--data-dir`` > ``NEUROFLY_DATA_DIR`` > ``<project>/outputs/learning``."""
    env = os.environ if environ is None else environ
    raw = explicit or env.get("NEUROFLY_DATA_DIR") or ""
    if raw.strip():
        return Path(raw).expanduser().resolve()
    return (Path(project_root) / DEFAULT_DATA_SUBDIR).resolve()


def _json_default(obj: Any) -> Any:
    """Serialise NumPy scalars/arrays and anything else with a sane fallback."""
    try:
        import numpy as np  # local import keeps this module importable without NumPy
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


class JsonlWriter:
    """Append-only JSON Lines file with size-based rotation.

    When the active file would exceed ``max_bytes`` it is renamed to
    ``name.jsonl.<UTC timestamp>`` and a fresh file is started.  Nothing is
    ever rewritten, truncated or deleted; rotated files sort chronologically.
    Each ``append`` flushes and, when ``fsync=True``, calls ``os.fsync`` so a
    line is durable once ``append`` returns.
    """

    def __init__(self, path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES, fsync: bool = True):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.fsync = bool(fsync)
        self._lock = threading.Lock()
        self._fh = None
        self.lines_written = 0
        self.rotations = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- file management -----------------------------------------------------
    def _open(self) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def rotated_files(self) -> List[Path]:
        return sorted(self.path.parent.glob(self.path.name + ".*"))

    def _rotate(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        target = self.path.with_name(f"{self.path.name}.{stamp}")
        n = 1
        while target.exists():
            target = self.path.with_name(f"{self.path.name}.{stamp}-{n}")
            n += 1
        os.replace(self.path, target)
        self.rotations += 1
        self._open()

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False,
                          default=_json_default) + "\n"
        with self._lock:
            self._open()
            size = self._size()
            if self.max_bytes > 0 and size > 0 and size + len(line.encode("utf-8")) > self.max_bytes:
                self._rotate()
            self._fh.write(line)
            self._fh.flush()
            if self.fsync:
                try:
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass
            self.lines_written += 1

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Convenience loader used by tests and analysis scripts."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class LearningRecorder:
    """Owns the three files under ``data_dir`` and stamps every record."""

    TRIALS_FILE = "trials.jsonl"
    SUMMARY_FILE = "telemetry_summary.jsonl"
    SESSION_FILE = "session.json"
    SESSIONS_LOG = "sessions.jsonl"

    def __init__(self, data_dir: Path, *, session: Optional[Dict[str, Any]] = None,
                 max_bytes: int = DEFAULT_MAX_BYTES, fsync: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}-{secrets.token_hex(3)}"
        self.trials = JsonlWriter(self.data_dir / self.TRIALS_FILE, max_bytes=max_bytes, fsync=fsync)
        self.summaries = JsonlWriter(self.data_dir / self.SUMMARY_FILE, max_bytes=max_bytes, fsync=fsync)
        # Trial numbers restart at 1 on every daemon start, so de-duplication is
        # per session; ``session_id`` on each line disambiguates across restarts.
        self.last_trial_written: int = 0
        self.prior_trial_lines: int = self._count_prior_trial_lines()
        self.session: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": time.time(),
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "prior_trial_lines_on_disk": self.prior_trial_lines,
        }
        if session:
            self.session.update(session)
        self._write_session()

    # -- session manifest ----------------------------------------------------
    def _write_session(self) -> None:
        path = self.data_dir / self.SESSION_FILE
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.session, fh, indent=2, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        with open(self.data_dir / self.SESSIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.session, separators=(",", ":"), default=_json_default) + "\n")

    def _count_prior_trial_lines(self) -> int:
        """Lines already in the active trials file (informational only)."""
        path = self.data_dir / self.TRIALS_FILE
        if not path.exists():
            return 0
        try:
            with open(path, "rb") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0

    # -- records -------------------------------------------------------------
    def record_trials(self, entries: Iterable[Dict[str, Any]]) -> int:
        """Append every entry whose ``trial`` number is new. Returns count."""
        written = 0
        for entry in entries:
            trial = entry.get("trial")
            if not isinstance(trial, int) or trial <= self.last_trial_written:
                continue
            rec = {
                "type": "trial",
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "recorded_at": time.time(),
            }
            rec.update(entry)
            self.trials.append(rec)
            self.last_trial_written = trial
            written += 1
        return written

    def record_summary(self, summary: Dict[str, Any]) -> None:
        rec = {
            "type": "telemetry_summary",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "recorded_at": time.time(),
        }
        rec.update(summary)
        self.summaries.append(rec)

    def close(self) -> None:
        self.trials.close()
        self.summaries.close()

    def __enter__(self) -> "LearningRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def summarise_runner(runner: Any) -> Dict[str, Any]:
    """Build a compact telemetry summary from a runner. Call under runner.lock."""
    telem = runner.latest_telemetry or {}
    fly = telem.get("fly", {})
    plast = telem.get("plasticity", {})
    curve = list(getattr(runner, "learning_curve", []) or [])
    return {
        "timestamp": time.time(),
        "uptime_sec": round(time.time() - runner.start_time, 1),
        "paradigm": runner.active_paradigm_id,
        "step": runner.total_steps,
        "sim_speed": runner.sim_speed,
        # Measured by the scheduler; differs from the requested sim_speed under overload.
        "achieved_speed": (telem.get("timing") or {}).get("achieved_speed"),
        "current_trial": runner.current_trial,
        "trials_completed": len(getattr(runner, "trial_history", []) or []),
        "fly": {k: fly.get(k) for k in ("x", "y", "heading", "speed", "state") if k in fly},
        "mb_weights_mean": plast.get("mb_weights_mean"),
        "mb_weights_std": plast.get("mb_weights_std"),
        "learning_curve_tail": curve[-10:],
        "metrics": telem.get("metrics", {}),
    }


class RecorderThread(threading.Thread):
    """Background poller that drains a runner into a :class:`LearningRecorder`.

    ``poll_interval`` bounds how long a completed trial can sit only in memory;
    ``summary_interval`` sets the cadence of telemetry summary lines.
    """

    def __init__(self, runner: Any, recorder: LearningRecorder, *,
                 poll_interval: float = 1.0, summary_interval: float = 60.0):
        super().__init__(name="NeuroFly-Recorder", daemon=True)
        self.runner = runner
        self.recorder = recorder
        self.poll_interval = max(0.05, float(poll_interval))
        self.summary_interval = max(self.poll_interval, float(summary_interval))
        self._stop_event = threading.Event()
        self._stopped = False
        self._last_summary = 0.0
        self.errors = 0

    def poll_once(self, force_summary: bool = False) -> Dict[str, int]:
        """One drain cycle; safe to call directly from tests."""
        with self.runner.lock:
            pending = list(getattr(self.runner, "trial_history", []) or [])
            now = time.monotonic()
            want_summary = force_summary or (now - self._last_summary >= self.summary_interval)
            summary = summarise_runner(self.runner) if want_summary else None
        written = self.recorder.record_trials(pending)
        if summary is not None:
            self.recorder.record_summary(summary)
            self._last_summary = now
        return {"trials": written, "summaries": 1 if summary is not None else 0}

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as err:  # never let recording kill the daemon
                self.errors += 1
                print(f"[Recorder] error: {err}", file=sys.stderr, flush=True)
            self._stop_event.wait(self.poll_interval)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop polling, flush pending trials plus a final summary, close files."""
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)
        try:
            self.poll_once(force_summary=True)
        except Exception as err:
            self.errors += 1
            print(f"[Recorder] final flush error: {err}", file=sys.stderr, flush=True)
        self.recorder.close()
