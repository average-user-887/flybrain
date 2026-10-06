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

import errno
import functools
import hashlib
import json
import os
import platform
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_DATA_SUBDIR",
    "resolve_data_dir",
    "JsonlWriter",
    "read_jsonl",
    "summarise_runner",
    "ObservationJournalError",
    "ObservationConflictError",
    "RecorderOwnershipError",
    "LearningRecorder",
    "RecorderThread",
]

SCHEMA_VERSION = 1
DEFAULT_DATA_SUBDIR = Path("outputs") / "learning"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024     # rotate a JSONL file past 64 MiB


class ObservationJournalError(ValueError):
    """A strict observation entry cannot be safely read or acknowledged."""


class ObservationConflictError(ObservationJournalError):
    """The ledger already contains a different payload under this full key."""


class RecorderOwnershipError(RuntimeError):
    """The recorder data directory is owned elsewhere or this owner is unusable."""


class _DirectoryOwnerLock:
    """Lifetime, nonblocking OS lock for one recorder data directory."""

    FILE_NAME = ".neurofly-recorder.lock"

    @staticmethod
    def _acquire_error(directory: Path, exc: OSError) -> RecorderOwnershipError:
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            return RecorderOwnershipError(f"recorder data directory is already owned: {directory}")
        return RecorderOwnershipError(
            f"could not acquire recorder ownership for {directory}: {type(exc).__name__}: {exc}")

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / self.FILE_NAME
        self.pid = os.getpid()
        self._fd: Optional[int] = None
        self._backend: Optional[str] = None
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "posix":
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise self._acquire_error(self.directory, exc) from exc
                self._backend = "flock"
            elif os.name == "nt":  # pragma: no cover - exercised only on Windows
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise self._acquire_error(self.directory, exc) from exc
                self._backend = "msvcrt"
            else:  # pragma: no cover - fail closed on an unknown standard-library platform
                raise RecorderOwnershipError(
                    f"no safe recorder directory lock is available on platform {os.name!r}")
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def assert_owner(self) -> None:
        if os.getpid() != self.pid:
            raise RecorderOwnershipError(
                f"recorder owner was created in pid {self.pid} and cannot be used in pid {os.getpid()}")
        if self._fd is None:
            raise RecorderOwnershipError("recorder owner is closed")

    def release(self) -> None:
        self.assert_owner()
        fd, backend = self._fd, self._backend
        if backend == "flock":
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif backend == "msvcrt":  # pragma: no cover - exercised only on Windows
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)
        self._fd = None


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
        # The observation transaction scans and appends under this same lock;
        # its internal helpers are explicitly lock-held and never reacquire it.
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

    # -- strict scientific append -------------------------------------------
    def _fsync_parent(self) -> None:
        """Durably publish directory-entry creation and rotation changes."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(self.path.parent, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _flush_strict(self) -> None:
        self._fh.flush()

    def _fsync_strict(self) -> None:
        os.fsync(self._fh.fileno())

    def _write_strict(self, line: str) -> None:
        written = self._fh.write(line)
        if written != len(line):
            raise OSError(f"short JSONL write: wrote {written} of {len(line)} characters")

    def _sync_path_locked(self, path: Path) -> None:
        """Re-sync an existing complete line before idempotent acknowledgement."""
        if path == self.path and self._fh is not None:
            self._flush_strict()
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._fsync_parent()

    def _rotate_strict_locked(self) -> None:
        if self._fh is not None:
            self._flush_strict()
            self._fsync_strict()
            self._fh.close()
            self._fh = None
        elif self.path.exists():
            self._sync_path_locked(self.path)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        target = self.path.with_name(f"{self.path.name}.{stamp}")
        n = 1
        while target.exists():
            target = self.path.with_name(f"{self.path.name}.{stamp}-{n}")
            n += 1
        os.replace(self.path, target)
        self._fsync_parent()
        self._open()
        self._fsync_parent()
        self.rotations += 1

    def _append_strict_locked(self, line: str) -> None:
        """Append one prevalidated JSON line; caller owns ``_lock``."""
        encoded = line.encode("utf-8")
        if not line.endswith("\n"):
            raise ValueError("strict JSONL append requires a terminating newline")
        self._open()
        # Always re-sync the directory. This also makes a retry after an
        # ambiguous file-creation fsync failure safe.
        self._fsync_parent()
        size = self._size()
        if self.max_bytes > 0 and size > 0 and size + len(encoded) > self.max_bytes:
            self._rotate_strict_locked()
        self._write_strict(line)
        self._flush_strict()
        self._fsync_strict()
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


def _owned_operation(method):
    """Keep a recorder operation inside its lifetime directory ownership."""
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        # A fork can inherit a locked threading.RLock. Refuse that object before
        # acquiring a lock whose owning thread exists only in the parent process.
        if os.getpid() != self._owner.pid:
            self._owner.assert_owner()
        with self._operation_lock:
            return method(self, *args, **kwargs)
    return guarded


class LearningRecorder:
    """Owns the three files under ``data_dir`` and stamps every record."""

    TRIALS_FILE = "trials.jsonl"
    SUMMARY_FILE = "telemetry_summary.jsonl"
    SESSION_FILE = "session.json"
    SESSIONS_LOG = "sessions.jsonl"
    OBSERVATION_KEY_FIELDS = (
        "daemon_run_id", "run_id", "instance_id", "segment_id", "presentation_id",
    )

    def __init__(self, data_dir: Path, *, session: Optional[Dict[str, Any]] = None,
                 max_bytes: int = DEFAULT_MAX_BYTES, fsync: bool = True):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._owner = _DirectoryOwnerLock(self.data_dir)
        self._operation_lock = threading.RLock()
        self._closed = False
        self.trials = None
        self.summaries = None
        try:
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
        except Exception:
            if self.trials is not None:
                self.trials.close()
            if self.summaries is not None:
                self.summaries.close()
            self._owner.release()
            self._closed = True
            raise

    def _assert_owner(self) -> None:
        self._owner.assert_owner()
        if self._closed:
            raise RecorderOwnershipError("recorder is closed")

    # -- session manifest ----------------------------------------------------
    @_owned_operation
    def _write_session(self) -> None:
        self._assert_owner()
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
    @_owned_operation
    def record_trials(self, entries: Iterable[Dict[str, Any]]) -> int:
        """Append every entry whose ``trial`` number is new. Returns count."""
        self._assert_owner()
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

    @staticmethod
    def _canonical_json(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ObservationJournalError(f"observation is not strict finite JSON: {exc}") from exc

    def _prepare_observation(self, envelope: Dict[str, Any]) -> Tuple[
            Dict[str, Any], Tuple[str, ...], Dict[str, str], str]:
        # Kept lazy so legacy recorder startup has its historical import graph.
        from observation_envelopes import validate_observation_envelope

        if not isinstance(envelope, dict):
            raise ObservationJournalError("record_observation requires an envelope object")
        if "last_terminal" in envelope or (
                "observation" in envelope and "schema" not in envelope):
            raise ObservationJournalError(
                "record_observation requires the immutable envelope, not a last_terminal wrapper")
        try:
            validated = validate_observation_envelope(envelope)
        except (TypeError, ValueError) as exc:
            raise ObservationJournalError(f"invalid observation envelope: {exc}") from exc
        records = validated.get("records")
        if not isinstance(records, dict) or not records or not all(
                isinstance(rec, dict) and rec.get("final") is True for rec in records.values()):
            raise ObservationJournalError("record_observation accepts frozen terminal records only")
        if validated.get("completeness") not in ("complete", "incomplete"):
            raise ObservationJournalError("terminal observation requires completeness")
        if not isinstance(validated.get("end_reason"), str) or not validated["end_reason"]:
            raise ObservationJournalError("terminal observation requires end_reason")
        identity = validated.get("identity")
        if not isinstance(identity, dict):
            raise ObservationJournalError("terminal observation requires identity")
        values = {
            "daemon_run_id": identity.get("daemon_run_id"),
            "run_id": identity.get("run_id"),
            "instance_id": identity.get("instance_id"),
            "segment_id": validated.get("segment_id"),
            "presentation_id": validated.get("presentation_id"),
        }
        for field in self.OBSERVATION_KEY_FIELDS:
            if not isinstance(values[field], str) or not values[field]:
                raise ObservationJournalError(f"observation key field {field} must be a nonempty string")
        key = tuple(values[field] for field in self.OBSERVATION_KEY_FIELDS)
        canonical = self._canonical_json(validated)
        return validated, key, values, canonical

    def _strict_observations_locked(self) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        """Scan all physical trial ledgers; caller owns the shared writer lock."""
        if self.trials._fh is not None:
            self.trials._flush_strict()
        paths = self.trials.rotated_files()
        if self.trials.path.exists():
            paths.append(self.trials.path)
        found: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for path in paths:
            with open(path, "rb") as fh:
                line_number = 0
                while True:
                    offset = fh.tell()
                    raw = fh.readline()
                    if not raw:
                        break
                    line_number += 1
                    location = f"{path.name}:line {line_number}:byte {offset}"
                    if not raw.endswith(b"\n"):
                        raise ObservationJournalError(f"partial JSONL row at {location}")
                    if not raw.strip():
                        continue
                    try:
                        text = raw[:-1].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ObservationJournalError(f"invalid UTF-8 at {location}: {exc}") from exc
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ObservationJournalError(f"invalid JSON at {location}: {exc.msg}") from exc
                    if not isinstance(row, dict):
                        raise ObservationJournalError(f"JSONL row is not an object at {location}")
                    if row.get("type") != "observation":
                        continue
                    if not isinstance(row.get("observation_key"), dict) or "observation" not in row:
                        raise ObservationJournalError(f"malformed observation row at {location}")
                    try:
                        _, key, declared, canonical = self._prepare_observation(row["observation"])
                    except ObservationJournalError as exc:
                        raise ObservationJournalError(
                            f"invalid stored observation at {location}: {exc}") from exc
                    if row["observation_key"] != declared:
                        raise ObservationJournalError(
                            f"observation key disagrees with payload at {location}")
                    if key in found:
                        raise ObservationJournalError(
                            f"duplicate observation key at {location}; first seen at "
                            f"{found[key]['file']}:line {found[key]['line']}")
                    found[key] = {
                        "canonical": canonical, "path": path, "file": path.name,
                        "line": line_number, "offset": offset,
                    }
        return found

    @_owned_operation
    def record_observation(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Durably append one validated immutable terminal observation.

        The trials writer is the single in-process writer. Disk is scanned under
        its shared lock on every call, so retries and recorder restarts recover a
        complete write whose earlier acknowledgement was lost.
        """
        self._assert_owner()
        validated, key, declared, canonical = self._prepare_observation(envelope)
        payload_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with self.trials._lock:
            existing = self._strict_observations_locked()
            match = existing.get(key)
            if match is not None:
                if match["canonical"] != canonical:
                    raise ObservationConflictError(
                        f"different observation payload for key {declared} at "
                        f"{match['file']}:line {match['line']}:byte {match['offset']}")
                self.trials._sync_path_locked(match["path"])
                return {
                    "durable": True, "idempotent": True,
                    "observation_key": dict(declared), "file": match["file"],
                    "line": match["line"], "offset": match["offset"],
                    "payload_sha256": payload_sha256,
                }
            row = {
                "type": "observation",
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "recorded_at": time.time(),
                "observation_key": dict(declared),
                "observation": validated,
            }
            line = self._canonical_json(row) + "\n"
            self.trials._append_strict_locked(line)
            written = self._strict_observations_locked().get(key)
            if written is None or written["canonical"] != canonical:
                raise ObservationJournalError("durable observation could not be re-read after append")
            return {
                "durable": True, "idempotent": False,
                "observation_key": dict(declared), "file": written["file"],
                "line": written["line"], "offset": written["offset"],
                "payload_sha256": payload_sha256,
            }

    @_owned_operation
    def record_summary(self, summary: Dict[str, Any]) -> None:
        self._assert_owner()
        rec = {
            "type": "telemetry_summary",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "recorded_at": time.time(),
        }
        rec.update(summary)
        self.summaries.append(rec)

    @_owned_operation
    def close(self) -> None:
        if os.getpid() != self._owner.pid:
            self._owner.assert_owner()
        if self._closed:
            return
        self._owner.assert_owner()
        self.trials.close()
        self.summaries.close()
        self._owner.release()
        self._closed = True

    def __enter__(self) -> "LearningRecorder":
        self._assert_owner()
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
        # Which run/controller produced these numbers, and whether it was driving.
        "identity": telem.get("identity"),
        "motor_source": (telem.get("motor") or {}).get("motor_source"),
        "motor_assists_enabled": (telem.get("motor") or {}).get("motor_assists_enabled"),
        "controller_fault": telem.get("controller_fault"),
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
        self._closed = False
        self._shutdown_clean = False
        self._shutdown_reported = False
        self._last_summary = 0.0
        self.errors = 0
        # Exposed in the daemon's /api/status (audit F, F2): a disk-full recorder is
        # reported as "not saving" instead of only printing to the journal.
        self.last_error: Optional[str] = None
        self.last_errno: Optional[int] = None
        self.last_error_at: Optional[float] = None
        self.failing_since: Optional[float] = None
        self.last_ok_at: Optional[float] = None
        self._liveness_lock = threading.Lock()
        self._start_requested_mono: Optional[float] = None
        self._run_entered_mono: Optional[float] = None
        self._last_progress_mono: Optional[float] = None
        self._active_write_started_mono: Optional[float] = None
        self._active_write: Optional[str] = None

    def start(self) -> None:
        with self._liveness_lock:
            self._start_requested_mono = time.monotonic()
        super().start()

    def _write_started(self, name: str) -> None:
        with self._liveness_lock:
            self._active_write = name
            self._active_write_started_mono = time.monotonic()

    def _write_finished(self) -> None:
        with self._liveness_lock:
            self._active_write = None
            self._active_write_started_mono = None
            self._last_progress_mono = time.monotonic()

    def liveness(self) -> Dict[str, Any]:
        """Detached recorder-thread progress; no runner lock or recorder I/O."""
        now = time.monotonic()
        with self._liveness_lock:
            requested = self._start_requested_mono
            entered = self._run_entered_mono
            progress = self._last_progress_mono
            write_started = self._active_write_started_mono
            active_write = self._active_write
        age = lambda stamp: None if stamp is None else max(0.0, now - stamp)
        return {
            "started": requested is not None,
            "run_entered": entered is not None,
            "alive": self.is_alive(),
            "stop_requested": self._stop_event.is_set(),
            "start_age_s": age(requested),
            "last_progress_age_s": age(progress),
            "active_write": active_write,
            "active_write_age_s": age(write_started),
        }

    def poll_once(self, force_summary: bool = False) -> Dict[str, int]:
        """One drain cycle; safe to call directly from tests."""
        with self.runner.lock:
            pending = list(getattr(self.runner, "trial_history", []) or [])
            now = time.monotonic()
            want_summary = force_summary or (now - self._last_summary >= self.summary_interval)
            summary = summarise_runner(self.runner) if want_summary else None
            # Claim only after diagnostic preparation succeeds. An exception
            # before _drain_observation must not strand an active queue token.
            claim_fn = getattr(self.runner, "claim_terminal_observation", None)
            observation = claim_fn() if claim_fn is not None else None
        # Each write is reported to the runner (``records_failed`` / ``records_ok``), which
        # applies the save policy and keeps the run's invalidity mark: trials.jsonl is part
        # of the scientific record, telemetry_summary.jsonl is diagnostic.  The failing
        # flag below is only this thread's view; it never clears the run's mark.
        observations = self._drain_observation(observation) if observation is not None else 0
        written = self._write("learning_records", self.recorder.record_trials, pending)
        if summary is not None:
            self._write("telemetry_summary", self.recorder.record_summary, summary)
            self._last_summary = now
        result = {"trials": written, "summaries": 1 if summary is not None else 0}
        if observations:
            result["observations"] = observations
        return result

    def _drain_observation(self, claim: Dict[str, Any]) -> int:
        """Write outside runner.lock, then compare-and-commit under it."""
        token = claim["attempt_token"]
        self._write_started("observation")
        try:
            receipt = self.recorder.record_observation(claim["observation"])
        except Exception as err:
            with self.runner.lock:
                self.runner.terminal_observation_failed(token, err)
            raise
        finally:
            self._write_finished()
        try:
            with self.runner.lock:
                self.runner.terminal_observation_succeeded(token, receipt)
        except Exception as err:
            with self.runner.lock:
                self.runner.terminal_observation_failed(token, err)
            raise
        return 1

    def _write(self, channel: str, fn, *args):
        self._write_started(channel)
        try:
            result = fn(*args)
        except Exception as err:
            report = getattr(self.runner, "records_failed", None)
            if report is not None:
                try:
                    report(channel, err)
                except Exception:  # noqa: BLE001 -- reporting must not mask the write error
                    pass
            raise
        finally:
            self._write_finished()
        ok = getattr(self.runner, "records_ok", None)
        if ok is not None:
            ok(channel)
        return result

    def _note_error(self, err: BaseException) -> None:
        self.errors += 1
        now = time.time()
        self.last_error = f"{type(err).__name__}: {err}"
        self.last_errno = getattr(err, "errno", None)
        self.last_error_at = now
        if self.failing_since is None:
            self.failing_since = now

    def describe(self) -> Dict[str, Any]:
        """Recorder health: ``failing`` while the newest attempt failed."""
        return {"errors": self.errors, "last_error": self.last_error, "last_errno": self.last_errno,
                "last_error_at": self.last_error_at, "last_ok_at": self.last_ok_at,
                "failing": self.failing_since is not None, "failing_since": self.failing_since,
                "shutdown_blocked": self._shutdown_reported and self.is_alive(),
                "data_dir": str(getattr(self.recorder, "data_dir", "") or "") or None}

    def run(self) -> None:
        with self._liveness_lock:
            now = time.monotonic()
            self._run_entered_mono = now
            self._last_progress_mono = now
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as err:  # never let recording kill the daemon
                self._note_error(err)
                print(f"[Recorder] error: {err}", file=sys.stderr, flush=True)
            else:
                with self.runner.lock:
                    queue = getattr(self.runner, "observation_publication", None)
                    waiting_for_retry = queue is not None and any(
                        entry["phase"] == "failed"
                        for entry in queue.pending_status()["entries"])
                # A backoff cycle writes no observation. Successful diagnostic
                # writes cannot report recovery of the outstanding failed save.
                if not waiting_for_retry:
                    self.last_ok_at = time.time()
                    self.failing_since = None
            with self._liveness_lock:
                self._last_progress_mono = time.monotonic()
            self._stop_event.wait(self.poll_interval)

    def stop(self, timeout: float = 3.0) -> bool:
        """Stop polling, flush pending trials plus a final summary, close files."""
        if self._closed:
            return self._shutdown_clean
        if not self._stopped:
            self._stopped = True
            self._stop_event.set()
        if self.is_alive():
            self.join(timeout=timeout)
        if self.is_alive():
            if not self._shutdown_reported:
                self._shutdown_reported = True
                err = TimeoutError(
                    f"recorder thread did not stop within {timeout:g}s; active write remains owned")
                self._note_error(err)
                report = getattr(self.runner, "records_failed", None)
                if report is not None:
                    report("learning_records", err)
            return False
        clean = True
        try:
            self.poll_once(force_summary=True)
        except Exception as err:
            clean = False
            self._note_error(err)
            print(f"[Recorder] final flush error: {err}", file=sys.stderr, flush=True)
        status_fn = getattr(self.runner, "observation_publication_status", None)
        if status_fn is not None:
            with self.runner.lock:
                clean = clean and status_fn()["pending"]["pending_count"] == 0
        self.recorder.close()
        self._shutdown_clean = clean
        self._closed = True
        return clean
