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

import base64
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
    "DurabilityUncertainError",
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


class DurabilityUncertainError(ObservationJournalError):
    """A complete row exists, but an fsync covering its bytes already failed.

    After a failed file fsync the kernel may have dropped the dirty pages while
    a later read still returns them from cache, and a later fsync can report
    success. Bytes in such a range are never acknowledged by this process.
    """


QUARANTINE_SUBDIR = "quarantine"
TAIL_DIGEST_BYTES = 4096
_MAX_UNCERTAIN_FILES = 1024


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class JsonlWriter:
    """Append-only JSON Lines file with size-based rotation.

    When the active file would exceed ``max_bytes`` it is renamed to
    ``name.jsonl.<UTC timestamp>`` and a fresh file is started.  Complete lines
    are never rewritten, truncated or deleted; rotated files sort
    chronologically.  Each ``append`` writes unbuffered bytes and, when
    ``fsync=True``, calls ``os.fsync`` so a line is durable once ``append``
    returns.  A failed fsync is raised, never swallowed.

    The only bytes this writer ever removes are an *unterminated* final record
    (a torn write).  Before the first write, and again after one of its own
    writes failed part way, it copies those exact bytes durably into
    ``quarantine/`` and only then truncates the file back to the last newline.
    """

    def __init__(self, path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES, fsync: bool = True,
                 quarantine_dir: Optional[Path] = None):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.fsync = bool(fsync)
        self.quarantine_dir = (Path(quarantine_dir) if quarantine_dir is not None
                               else self.path.parent / QUARANTINE_SUBDIR)
        # Every writer path (legacy append, strict append, tail repair and the
        # observation index) runs under this one lock; lock-held helpers never
        # reacquire it.
        self._lock = threading.Lock()
        self._fh = None
        self.lines_written = 0
        self.rotations = 0
        # Torn-tail repair state: checked once before the first write, and again
        # after one of this writer's own writes failed part way.
        self._tail_checked = False
        self._torn_pending = False
        self.tail_repairs: List[Dict[str, Any]] = []
        # Bytes of the active file proven by this writer's last successful fsync.
        self._synced_size: Optional[int] = None
        # (st_dev, st_ino) -> (start, end, reason) of bytes whose fsync failed.
        self._uncertain: Dict[Tuple[int, int], Tuple[int, int, str]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- file management -----------------------------------------------------
    def _open(self) -> None:
        if self._fh is None:
            # Unbuffered: a failed write can never leave bytes in a user-space
            # buffer that a later flush or close would append out of order.
            self._fh = open(self.path, "a+b", buffering=0)
            self._synced_size = None

    def _size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def rotated_files(self) -> List[Path]:
        return sorted(p for p in self.path.parent.glob(self.path.name + ".*") if p.is_file())

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

    # -- torn final record ----------------------------------------------------
    def _quarantine_locked(self, content: bytes, offset: int, size: int,
                           st: os.stat_result, reason: str) -> Dict[str, Any]:
        """Durably copy torn bytes to an evidence file before any truncation."""
        digest = hashlib.sha256(content).hexdigest()
        now = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now))
        evidence = {
            "schema": "neurofly.jsonl_torn_tail_quarantine.v1",
            "reason": reason,
            "source_file": self.path.name,
            "source_device": st.st_dev,
            "source_inode": st.st_ino,
            "source_size_before": size,
            "offset": offset,
            "length": len(content),
            "sha256": digest,
            "quarantined_at": now,
            "quarantined_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        encoded = (json.dumps(evidence, sort_keys=True, indent=2) + "\n").encode("utf-8")
        qdir = self.quarantine_dir
        created = not qdir.is_dir()
        qdir.mkdir(parents=True, exist_ok=True)
        if created:
            _fsync_directory(qdir.parent)
        target = qdir / f"{self.path.name}.torn-{stamp}-{offset}-{digest[:12]}.json"
        n = 1
        # A failed earlier attempt may have left its unverified .tmp; keep it and
        # pick a fresh name (the source journal was not touched by that attempt).
        while target.exists() or target.with_name(target.name + ".tmp").exists():
            target = qdir / f"{self.path.name}.torn-{stamp}-{offset}-{digest[:12]}-{n}.json"
            n += 1
        tmp = target.with_name(target.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view, total = memoryview(encoded), 0
            while total < len(encoded):
                n = os.write(fd, view[total:])
                if not n:  # no progress: fail before the journal is touched
                    raise OSError(errno.EIO, f"short quarantine write: wrote {total} of "
                                             f"{len(encoded)} bytes")
                total += n
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
        _fsync_directory(qdir)
        # Prove the evidence, as stored, holds exactly the bytes about to be cut.
        stored = json.loads(target.read_bytes().decode("utf-8"))
        if (base64.b64decode(stored["content_base64"]) != content
                or stored["sha256"] != digest or stored["offset"] != offset):
            raise OSError(f"torn-tail quarantine evidence did not verify: {target.name}")
        try:
            shown = str(target.relative_to(self.path.parent))
        except ValueError:
            shown = target.name
        return {"file": self.path.name, "offset": offset, "length": len(content),
                "sha256": digest, "evidence": shown, "reason": reason}

    def _repair_torn_tail_locked(self, *, expected_offset: Optional[int] = None,
                                 reason: str = "unterminated final record") -> Optional[Dict[str, Any]]:
        """Quarantine then truncate bytes after the last newline; caller owns ``_lock``.

        Only an unterminated final record is ever cut. Complete lines, valid or
        not, are left for the reader to accept or refuse.
        """
        try:
            fd = os.open(self.path, os.O_RDWR)
        except FileNotFoundError:
            return None
        try:
            st = os.fstat(fd)
            size = st.st_size
            pos, start = size, 0
            while pos > 0:
                n = min(1 << 16, pos)
                pos -= n
                chunk = os.pread(fd, n, pos)
                idx = chunk.rfind(b"\n")
                if idx >= 0:
                    start = pos + idx + 1
                    break
            if start == size:
                return None
            if expected_offset is not None and start != expected_offset:
                raise ObservationJournalError(
                    f"torn tail of {self.path.name} starts at byte {start}, "
                    f"not at the validated boundary {expected_offset}")
            content = b""
            while len(content) < size - start:
                more = os.pread(fd, size - start - len(content), start + len(content))
                if not more:
                    raise OSError(f"{self.path.name} shrank while reading its torn tail")
                content += more
            if b"\n" in content:  # pragma: no cover - guarded by the backward scan
                raise ObservationJournalError(f"torn tail of {self.path.name} contains a newline")
            receipt = self._quarantine_locked(content, start, size, st, reason)
            if os.fstat(fd).st_size != size or os.pread(fd, len(content), start) != content:
                raise ObservationJournalError(
                    f"{self.path.name} changed while its torn tail was quarantined; not truncated")
            os.ftruncate(fd, start)
            os.fsync(fd)
        finally:
            os.close(fd)
        self.tail_repairs.append(receipt)
        del self.tail_repairs[:-16]
        print(f"[Recorder] quarantined {receipt['length']} torn bytes at "
              f"{self.path.name}:byte {start} to {receipt['evidence']}", file=sys.stderr, flush=True)
        return receipt

    def repair_tail(self) -> Optional[Dict[str, Any]]:
        """Startup repair of an unterminated final record (serialized with writers)."""
        with self._lock:
            receipt = self._repair_torn_tail_locked(reason="unterminated final record found at open")
            self._tail_checked = True
            self._torn_pending = False
            return receipt

    def _prepare_write_locked(self) -> None:
        """Repair any torn tail that is ours, then refuse to extend a foreign one."""
        if not self._tail_checked or self._torn_pending:
            reason = ("this writer's failed write" if self._tail_checked
                      else "unterminated final record found at open")
            self._repair_torn_tail_locked(reason=reason)
            self._tail_checked = True
            self._torn_pending = False
        self._open()
        fd = self._fh.fileno()
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            raise ObservationJournalError(
                f"{self.path.name} ends with an unterminated record at byte {size}; "
                "refusing to append onto it (it is repaired on the next recorder start)")

    def _write_bytes_locked(self, data: bytes) -> None:
        view, total = memoryview(data), 0
        try:
            while total < len(data):
                n = self._fh.write(view[total:])
                if not n:
                    raise OSError(errno.EIO, f"short JSONL write: wrote {total} of {len(data)} bytes")
                total += n
        except BaseException:
            self._torn_pending = True
            raise

    # -- fsync and ambiguity ---------------------------------------------------
    def _mark_uncertain_locked(self, st: os.stat_result, start: int, end: int, reason: str) -> None:
        key = (st.st_dev, st.st_ino)
        if key in self._uncertain:
            old_start, old_end, _ = self._uncertain[key]
            start, end = min(start, old_start), max(end, old_end)
        elif len(self._uncertain) >= _MAX_UNCERTAIN_FILES:
            self._uncertain.pop(next(iter(self._uncertain)))
            self._uncertain_overflow = True
        self._uncertain[key] = (start, end, reason)

    def assert_proven_locked(self, path: Path, offset: int, length: int) -> None:
        """Refuse to acknowledge bytes covered by a failed fsync in this process."""
        if getattr(self, "_uncertain_overflow", False):
            raise DurabilityUncertainError(
                "too many files had fsync failures in this process; durability is uncertain")
        if not self._uncertain:
            return
        st = os.stat(path)
        found = self._uncertain.get((st.st_dev, st.st_ino))
        if found is not None and offset < found[1] and found[0] < offset + length:
            raise DurabilityUncertainError(
                f"durability of {path.name}:byte {offset} is uncertain: an fsync covering "
                f"bytes {found[0]}-{found[1]} failed ({found[2]}); it is not acknowledged "
                "by this process")

    def _fsync_active_locked(self, what: str) -> None:
        """fsync the active file; on any failure mark its unproven bytes uncertain."""
        fd = self._fh.fileno()
        st = os.fstat(fd)
        start = self._synced_size or 0
        try:
            self._fsync_strict()
        except BaseException as exc:
            self._mark_uncertain_locked(st, start, st.st_size, f"{what}: {type(exc).__name__}: {exc}")
            raise
        self._synced_size = st.st_size

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False,
                          default=_json_default) + "\n"
        data = line.encode("utf-8")
        with self._lock:
            self._prepare_write_locked()
            size = self._size()
            if self.max_bytes > 0 and size > 0 and size + len(data) > self.max_bytes:
                self._rotate()
            self._write_bytes_locked(data)
            if self.fsync:
                self._fsync_active_locked("append fsync")
            self.lines_written += 1

    # -- strict scientific append -------------------------------------------
    def _fsync_parent(self) -> None:
        """Durably publish directory-entry creation and rotation changes."""
        _fsync_directory(self.path.parent)

    def _flush_strict(self) -> None:
        self._fh.flush()

    def _fsync_strict(self) -> None:
        os.fsync(self._fh.fileno())

    def _write_strict(self, line: str) -> None:
        self._write_bytes_locked(line.encode("utf-8"))

    def _sync_path_locked(self, path: Path) -> None:
        """Re-sync an existing complete line before idempotent acknowledgement."""
        if path == self.path and self._fh is not None:
            self._flush_strict()
        fd = os.open(path, os.O_RDONLY)
        try:
            st = os.fstat(fd)
            try:
                os.fsync(fd)
            except BaseException as exc:
                self._mark_uncertain_locked(
                    st, 0, st.st_size, f"re-sync fsync: {type(exc).__name__}: {exc}")
                raise
        finally:
            os.close(fd)
        self._fsync_parent()

    def _rotate_strict_locked(self) -> None:
        if self._fh is not None:
            self._flush_strict()
            self._fsync_active_locked("pre-rotation fsync")
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
        self._prepare_write_locked()
        # Always re-sync the directory. This also makes a retry after an
        # ambiguous file-creation fsync failure safe.
        self._fsync_parent()
        size = self._size()
        if self.max_bytes > 0 and size > 0 and size + len(encoded) > self.max_bytes:
            self._rotate_strict_locked()
        self._write_strict(line)
        self._flush_strict()
        self._fsync_active_locked("append fsync")
        self.lines_written += 1

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
                self._synced_size = None

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
        self._sessions_log = None
        try:
            self.session_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}-{secrets.token_hex(3)}"
            self.trials = JsonlWriter(self.data_dir / self.TRIALS_FILE, max_bytes=max_bytes, fsync=fsync)
            self.summaries = JsonlWriter(self.data_dir / self.SUMMARY_FILE, max_bytes=max_bytes, fsync=fsync)
            # Same torn-tail repair and fsync contract as the other journals; never rotated.
            self._sessions_log = JsonlWriter(self.data_dir / self.SESSIONS_LOG, max_bytes=0, fsync=fsync)
            # Trial numbers restart at 1 on every daemon start, so de-duplication is
            # per session; ``session_id`` on each line disambiguates across restarts.
            self.last_trial_written: int = 0
            # Before anything can append: quarantine-then-truncate an unterminated
            # final record left by a crash, so no write is ever merged onto it.
            self.tail_repairs: List[Dict[str, Any]] = [
                receipt for receipt in (
                    self.trials.repair_tail(), self.summaries.repair_tail(),
                    self._sessions_log.repair_tail())
                if receipt is not None]
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
                "torn_tail_repairs": list(self.tail_repairs),
            }
            if session:
                self.session.update(session)
            self._write_session()
        except Exception:
            for writer in (self.trials, self.summaries, self._sessions_log):
                if writer is not None:
                    writer.close()
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
        self._sessions_log.append(self.session)

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
                        "line": line_number, "offset": offset, "length": len(raw),
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
                # A failed fsync in this process may have lost bytes the page
                # cache still shows; a later successful fsync proves nothing.
                self.trials.assert_proven_locked(match["path"], match["offset"], match["length"])
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
        self._sessions_log.close()
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
