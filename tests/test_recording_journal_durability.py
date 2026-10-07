"""Torn-tail repair, fsync failure and incremental-index tests for the JSONL journals.

Every journal written by ``learning_recorder`` (trials.jsonl and its rotated
files, telemetry_summary.jsonl, sessions.jsonl) shares one contract:

* only an unterminated final record is ever removed, and only after its exact
  bytes are durably copied to ``quarantine/`` (file and directory fsynced);
* a complete line that is corrupt is never dropped and fails visibly;
* a failed fsync is raised, and bytes it covered are never acknowledged by the
  same process.
"""

import base64
import copy
import errno
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import learning_recorder  # noqa: E402
from learning_recorder import (  # noqa: E402
    DurabilityUncertainError,
    JsonlWriter,
    LearningRecorder,
    ObservationJournalError,
    read_jsonl,
)
from tests.test_learning_recorder import _next_presentation, _terminal_envelope  # noqa: E402

TORN = b'{"type":"trial","trial":7,"met'


def _evidence(data_dir):
    files = sorted((Path(data_dir) / "quarantine").glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def _fd_path(fd):
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:  # pragma: no cover - non-Linux
        return None


# =============================================================================
# 1. Torn final record: quarantine first, then truncate
# =============================================================================

class TestTornTailRepair:
    @pytest.mark.parametrize("name", ["trials.jsonl", "telemetry_summary.jsonl", "sessions.jsonl"])
    def test_open_quarantines_exact_bytes_truncates_and_writes_resume(self, tmp_path, name):
        path = tmp_path / name
        good = b'{"type":"trial","trial":1}\n{"type":"trial","trial":2}\n'
        path.write_bytes(good + TORN)
        rec = LearningRecorder(tmp_path, fsync=True)
        try:
            [evidence] = _evidence(tmp_path)
            assert evidence["source_file"] == name
            assert base64.b64decode(evidence["content_base64"]) == TORN
            assert evidence["sha256"] == hashlib.sha256(TORN).hexdigest()
            assert evidence["offset"] == len(good) and evidence["length"] == len(TORN)
            assert evidence["source_size_before"] == len(good) + len(TORN)
            assert isinstance(evidence["quarantined_at"], float)
            if name != "sessions.jsonl":
                assert path.read_bytes() == good
            else:  # the new session line was appended after the repaired prefix
                assert path.read_bytes().startswith(good)
                assert read_jsonl(path)[-1]["session_id"] == rec.session_id
            assert any(r["file"] == name and r["sha256"] == evidence["sha256"]
                       for r in rec.session["torn_tail_repairs"])
            rec.record_trials([{"trial": 1, "metric": 0.5}])
            rec.record_summary({"step": 1})
            assert rec.record_observation(_terminal_envelope(run_id="after-repair"))["durable"]
        finally:
            rec.close()
        for journal in ("trials.jsonl", "telemetry_summary.jsonl", "sessions.jsonl"):
            assert all(isinstance(row, dict) for row in read_jsonl(tmp_path / journal))
        # A second start finds nothing to repair and leaves the evidence alone.
        LearningRecorder(tmp_path, fsync=False).close()
        assert len(_evidence(tmp_path)) == 1

    def test_evidence_file_and_directory_are_fsynced_before_truncation(self, tmp_path, monkeypatch):
        path = tmp_path / "trials.jsonl"
        path.write_bytes(b'{"type":"trial"}\n' + TORN)
        events = []
        real_fsync, real_truncate = os.fsync, os.ftruncate

        def spy_fsync(fd):
            events.append(("fsync", _fd_path(fd)))
            return real_fsync(fd)

        def spy_truncate(fd, length):
            events.append(("truncate", _fd_path(fd), length))
            return real_truncate(fd, length)

        monkeypatch.setattr(os, "fsync", spy_fsync)
        monkeypatch.setattr(os, "ftruncate", spy_truncate)
        LearningRecorder(tmp_path, fsync=False).close()
        cut = next(i for i, e in enumerate(events) if e[0] == "truncate")
        assert events[cut][1:] == (str(path), len(b'{"type":"trial"}\n'))
        [evidence_file] = (tmp_path / "quarantine").glob("*.json")
        before = [e[1] for e in events[:cut] if e[0] == "fsync"]
        assert any(p and p.endswith(".json.tmp") for p in before)  # evidence bytes
        assert str(tmp_path / "quarantine") in before                # its directory entry
        assert str(tmp_path) in before                                # the new quarantine dir
        assert ("fsync", str(path)) in events[cut + 1:]                # the truncation itself

    def test_failed_quarantine_leaves_journal_untouched_and_open_fails(self, tmp_path, monkeypatch):
        path = tmp_path / "trials.jsonl"
        before = b'{"type":"trial"}\n' + TORN
        path.write_bytes(before)

        def broken(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "injected quarantine failure")

        monkeypatch.setattr(JsonlWriter, "_quarantine_locked", broken)
        with pytest.raises(OSError, match="injected quarantine failure"):
            LearningRecorder(tmp_path, fsync=False)
        assert path.read_bytes() == before
        monkeypatch.undo()
        LearningRecorder(tmp_path, fsync=False).close()
        assert path.read_bytes() == b'{"type":"trial"}\n'

    def test_whole_file_without_newline_is_quarantined_to_empty(self, tmp_path):
        path = tmp_path / "telemetry_summary.jsonl"
        path.write_bytes(TORN)
        LearningRecorder(tmp_path, fsync=False).close()
        assert path.read_bytes() == b""
        assert base64.b64decode(_evidence(tmp_path)[0]["content_base64"]) == TORN

    def test_complete_corrupt_line_in_diagnostic_journal_is_never_dropped(self, tmp_path):
        path = tmp_path / "telemetry_summary.jsonl"
        before = b'{"ok":1}\n{broken but complete}\n{"ok":2}\n'
        path.write_bytes(before)
        with LearningRecorder(tmp_path, fsync=False) as rec:
            rec.record_summary({"step": 1})
        assert path.read_bytes().startswith(before)
        assert not (tmp_path / "quarantine").exists()

    def test_repair_is_serialized_against_a_concurrent_legacy_append(self, tmp_path, monkeypatch):
        path = tmp_path / "telemetry_summary.jsonl"
        path.write_bytes(b'{"n":0}\n' + TORN)
        writer = JsonlWriter(path, fsync=False)
        entered, release = threading.Event(), threading.Event()
        original = writer._quarantine_locked

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(writer, "_quarantine_locked", blocked)
        repairer = threading.Thread(target=writer.repair_tail)
        appender = threading.Thread(target=writer.append, args=({"n": 1},))
        repairer.start()
        assert entered.wait(2)
        appender.start()
        appender.join(0.2)
        assert appender.is_alive()                      # waits on the writer lock
        assert path.read_bytes() == b'{"n":0}\n' + TORN  # nothing merged onto the torn bytes
        release.set()
        repairer.join(5)
        appender.join(5)
        writer.close()
        assert read_jsonl(path) == [{"n": 0}, {"n": 1}]

    def test_runtime_foreign_torn_tail_is_refused_not_extended(self, tmp_path):
        path = tmp_path / "telemetry_summary.jsonl"
        writer = JsonlWriter(path, fsync=False)
        writer.append({"n": 0})
        with open(path, "ab") as fh:  # another process tears a line while we run
            fh.write(TORN)
        with pytest.raises(ObservationJournalError, match="unterminated record"):
            writer.append({"n": 1})
        assert path.read_bytes() == b'{"n":0}\n' + TORN
        writer.close()
        # The next start repairs it with evidence; nothing complete is lost.
        reopened = JsonlWriter(path, fsync=False)
        reopened.append({"n": 1})
        reopened.close()
        assert read_jsonl(path) == [{"n": 0}, {"n": 1}]
        assert base64.b64decode(_evidence(tmp_path)[0]["content_base64"]) == TORN

    def test_own_partial_write_is_quarantined_before_the_next_write(self, tmp_path):
        path = tmp_path / "trials.jsonl"
        writer = JsonlWriter(path, fsync=False)
        writer.append({"n": 0})

        class HalfThenFail:
            def __init__(self, raw):
                self.raw = raw

            def write(self, data):
                self.raw.write(bytes(data[: len(data) // 2]))
                raise OSError(errno.ENOSPC, "injected short write")

            def __getattr__(self, name):
                return getattr(self.raw, name)

        real = writer._fh
        writer._fh = HalfThenFail(real)
        line = json.dumps({"n": 1}, separators=(",", ":")) + "\n"
        with pytest.raises(OSError, match="injected short write"):
            writer.append({"n": 1})
        torn = line.encode()[: len(line) // 2]
        assert path.read_bytes() == b'{"n":0}\n' + torn
        writer._fh = real
        writer.append({"n": 1})
        writer.close()
        assert read_jsonl(path) == [{"n": 0}, {"n": 1}]
        [evidence] = _evidence(tmp_path)
        assert base64.b64decode(evidence["content_base64"]) == torn
        assert evidence["reason"] == "this writer's failed write"


# =============================================================================
# 2. fsync failures are raised and never re-acknowledged
# =============================================================================

def _eio_on(path_suffix, real_fsync):
    def fail(fd):
        target = _fd_path(fd) or ""
        if target.endswith(path_suffix):
            raise OSError(errno.EIO, "injected EIO")
        return real_fsync(fd)
    return fail


class TestFsyncFailure:
    def test_legacy_append_raises_fsync_errors_instead_of_swallowing(self, tmp_path, monkeypatch):
        writer = JsonlWriter(tmp_path / "trials.jsonl", fsync=True)
        writer.append({"n": 0})
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", os.fsync))
        with pytest.raises(OSError) as caught:
            writer.append({"n": 1})
        assert caught.value.errno == errno.EIO
        assert writer.lines_written == 1
        writer.close()

    def test_record_trials_reports_eio(self, tmp_path, monkeypatch):
        rec = LearningRecorder(tmp_path, fsync=True)
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", os.fsync))
        with pytest.raises(OSError, match="injected EIO"):
            rec.record_trials([{"trial": 1}])
        monkeypatch.undo()
        rec.close()

    def test_eio_on_observation_fsync_is_never_acknowledged_in_process(self, tmp_path, monkeypatch):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-eio")
        real = os.fsync
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", real))
        with pytest.raises(OSError, match="injected EIO"):
            rec.record_observation(envelope)
        # The fsync "recovers": the page cache still shows the line, and a new
        # fsync would succeed. That success proves nothing about the lost write.
        monkeypatch.setattr(os, "fsync", real)
        for _ in range(2):
            with pytest.raises(DurabilityUncertainError, match="uncertain"):
                rec.record_observation(copy.deepcopy(envelope))
        rows = [r for r in read_jsonl(tmp_path / "trials.jsonl") if r.get("type") == "observation"]
        assert len(rows) == 1
        # Bytes written after the failure are proven by their own fsync.
        later = rec.record_observation(_next_presentation(envelope))
        assert later["durable"] is True and later["idempotent"] is False
        rec.close()

    def test_eio_on_idempotent_resync_is_never_acknowledged(self, tmp_path, monkeypatch):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-resync-eio")
        rec.record_observation(envelope)
        real = os.fsync
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", real))
        with pytest.raises(OSError, match="injected EIO"):
            rec.record_observation(envelope)
        monkeypatch.setattr(os, "fsync", real)
        with pytest.raises(DurabilityUncertainError):
            rec.record_observation(envelope)
        rec.close()

    def test_eio_marks_only_unproven_bytes(self, tmp_path, monkeypatch):
        rec = LearningRecorder(tmp_path, max_bytes=128, fsync=False)
        first = _terminal_envelope(run_id="run-rotate-eio")
        second = _next_presentation(first)
        rec.record_observation(first)  # proven by its own fsync
        real = os.fsync
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", real))
        with pytest.raises(OSError, match="injected EIO"):
            rec.record_observation(second)  # the pre-rotation fsync fails
        monkeypatch.setattr(os, "fsync", real)
        assert rec.record_observation(first)["idempotent"] is True
        assert rec.record_observation(second)["idempotent"] is False
        assert rec.record_observation(second)["idempotent"] is True
        rec.close()

    def test_unproven_legacy_rows_become_uncertain(self, tmp_path, monkeypatch):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-legacy-eio")
        real = os.fsync
        rec.record_trials([{"trial": 1}])        # fsync=False: written, not proven
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", real))
        with pytest.raises(OSError, match="injected EIO"):
            rec.record_observation(envelope)
        monkeypatch.setattr(os, "fsync", real)
        start, end, _ = next(iter(rec.trials._uncertain.values()))
        assert start == 0 and end == (tmp_path / "trials.jsonl").stat().st_size
        with pytest.raises(DurabilityUncertainError):
            rec.record_observation(envelope)
        rec.close()

    def test_restart_reproves_with_a_fresh_fsync(self, tmp_path, monkeypatch):
        """Documented contract: a new recorder re-proves on-disk lines by fsync."""
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-eio-restart")
        real = os.fsync
        monkeypatch.setattr(os, "fsync", _eio_on("trials.jsonl", real))
        with pytest.raises(OSError):
            rec.record_observation(envelope)
        monkeypatch.setattr(os, "fsync", real)
        rec.close()
        with LearningRecorder(tmp_path, fsync=False) as restarted:
            assert restarted.record_observation(envelope)["idempotent"] is True
