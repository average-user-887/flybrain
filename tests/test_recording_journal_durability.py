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
    ObservationConflictError,
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


# =============================================================================
# 3. Quarantine evidence writes: no-progress and partial os.write
# =============================================================================

class TestQuarantineWrite:
    def _torn(self, tmp_path):
        path = tmp_path / "telemetry_summary.jsonl"
        before = b'{"n":0}\n' + TORN
        path.write_bytes(before)
        return path, before

    def test_zero_byte_write_raises_eio_leaves_journal_and_releases_lock(self, tmp_path, monkeypatch):
        path, before = self._torn(tmp_path)
        writer = JsonlWriter(path, fsync=False)
        real_write = os.write
        calls = []

        def zero_for_evidence(fd, data):
            if (_fd_path(fd) or "").endswith(".json.tmp"):
                calls.append(len(data))
                return 0
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", zero_for_evidence)
        done = []
        worker = threading.Thread(target=lambda: done.append(
            pytest.raises(OSError, writer.repair_tail)))
        worker.start()
        worker.join(5)
        assert not worker.is_alive(), "zero-byte evidence write must not spin"
        assert done[0].value.errno == errno.EIO and len(calls) == 1
        assert path.read_bytes() == before
        assert writer._lock.acquire(timeout=1)
        writer._lock.release()
        monkeypatch.undo()
        writer.repair_tail()
        writer.close()
        assert path.read_bytes() == b'{"n":0}\n'

    def test_partial_evidence_writes_complete_exactly(self, tmp_path, monkeypatch):
        path, _ = self._torn(tmp_path)
        writer = JsonlWriter(path, fsync=False)
        real_write = os.write
        sizes = []

        def seven_bytes_at_a_time(fd, data):
            if (_fd_path(fd) or "").endswith(".json.tmp"):
                sizes.append(min(7, len(data)))
                return real_write(fd, bytes(data[:7]))
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", seven_bytes_at_a_time)
        receipt = writer.repair_tail()
        writer.close()
        assert len(sizes) > 10
        [evidence] = _evidence(tmp_path)
        assert base64.b64decode(evidence["content_base64"]) == TORN
        assert evidence["sha256"] == receipt["sha256"] == hashlib.sha256(TORN).hexdigest()
        assert path.read_bytes() == b'{"n":0}\n'


# =============================================================================
# 4. Bounded incremental index: same guarantees, cost independent of history
# =============================================================================

def _obs_lines(path):
    return [i for i, line in enumerate(Path(path).read_bytes().splitlines(keepends=True))
            if b'"type":"observation"' in line]


def _count_prepares(monkeypatch, rec):
    calls = []
    original = rec._prepare_observation

    def counted(envelope):
        calls.append(1)
        return original(envelope)

    monkeypatch.setattr(rec, "_prepare_observation", counted)
    return calls


class TestIncrementalIndex:
    def test_mid_file_complete_corrupt_line_fails_open_and_nothing_is_cut(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        rec.record_observation(_terminal_envelope(run_id="run-mid"))
        rec.close()
        path = tmp_path / "trials.jsonl"
        good = path.read_bytes()
        before = good + b'{not json}\n' + good.splitlines(keepends=True)[-1].replace(
            b"run-mid", b"run-mi2") + TORN
        path.write_bytes(before)
        with pytest.raises(ObservationJournalError, match=r"invalid JSON at trials\.jsonl:line 2"):
            LearningRecorder(tmp_path, fsync=False)
        assert path.read_bytes() == before                  # torn tail NOT repaired either
        assert not (tmp_path / "quarantine").exists()

    def test_tampering_is_detected_after_restart(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        rec.record_observation(_terminal_envelope(run_id="run-tamper"))
        rec.close()
        path = tmp_path / "trials.jsonl"
        data = path.read_bytes()
        idx = data.rindex(b'"run_id":"run-tamper"')  # inside observation_key, last occurrence
        path.write_bytes(data[:idx] + b'"run_id":"run-tamperX"' + data[idx + 21:])
        with pytest.raises(ObservationJournalError, match="disagrees|invalid stored|duplicate"):
            LearningRecorder(tmp_path, fsync=False)

    def test_duplicate_key_across_rotated_files_fails_at_open(self, tmp_path):
        rec = LearningRecorder(tmp_path, max_bytes=128, fsync=False)
        envelope = _terminal_envelope(run_id="run-dup")
        rec.record_observation(envelope)
        rec.record_observation(_next_presentation(envelope))
        rec.close()
        [rotated] = sorted(tmp_path.glob("trials.jsonl.*"))
        with open(tmp_path / "trials.jsonl", "ab") as fh:
            fh.write(rotated.read_bytes())
        with pytest.raises(ObservationJournalError, match="duplicate observation key"):
            LearningRecorder(tmp_path, fsync=False)

    def test_in_place_change_at_runtime_forces_full_revalidation(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-inplace")
        rec.record_observation(envelope)
        path = tmp_path / "trials.jsonl"
        data = path.read_bytes()
        st = path.stat()
        idx = data.rindex(b'"run_id":"run-inplace"')
        path.write_bytes(data[:idx] + b'"run_id":"run-inplacX"' + data[idx + 22:])
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # hide it from mtime/size
        with pytest.raises(ObservationJournalError, match="disagrees|invalid stored"):
            rec.record_observation(envelope)
        assert rec._index is None and rec.full_scans >= 2
        path.write_bytes(data)
        assert rec.record_observation(envelope)["idempotent"] is True
        rec.close()

    def test_external_append_forces_full_revalidation(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-ext")
        rec.record_observation(envelope)
        path = tmp_path / "trials.jsonl"
        stored = path.read_bytes().splitlines(keepends=True)[-1]
        with open(path, "ab") as fh:
            fh.write(b'{"type":"trial","trial":99}\n')
        assert rec.record_observation(_next_presentation(envelope))["idempotent"] is False
        assert rec.full_scans == 2  # growth this recorder did not write: re-validated from byte 0
        with open(path, "ab") as fh:  # someone else appends a duplicate of a stored row
            fh.write(stored)
        with pytest.raises(ObservationJournalError, match="duplicate observation key"):
            rec.record_observation(_terminal_envelope(run_id="run-ext-2"))
        rec.close()

    def test_corrupt_prefix_plus_valid_append_is_refused(self, tmp_path):
        """Reviewer reproducer: >6000-byte legacy row, first byte changed, valid row appended."""
        path = tmp_path / "trials.jsonl"
        path.write_bytes(b'{"type":"trial","padding":"' + b"x" * 6000 + b'"}\n')
        rec = LearningRecorder(tmp_path)
        with path.open("r+b") as fh:
            fh.write(b"!")
            fh.seek(0, 2)
            fh.write(b'{"type":"trial","trial":2}\n')
            fh.flush()
            os.fsync(fh.fileno())
        with pytest.raises(ObservationJournalError, match=r"invalid JSON at trials\.jsonl:line 1:byte 0"):
            rec.record_observation(_terminal_envelope(run_id="run-prefix"))
        assert rec.full_scans == 2
        assert not _obs_lines(path)
        rec.close()

    def test_observation_payload_prefix_mutation_plus_append_is_refused(self, tmp_path):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-payload")
        rec.record_observation(envelope)
        rec.record_trials([{"trial": 1, "padding": "y" * 6000}])  # keep the change far from the tail
        path = tmp_path / "trials.jsonl"
        data = path.read_bytes()
        idx = data.index(b'"yaw":0.0')
        with path.open("r+b") as fh:  # same length, still valid JSON, different payload
            fh.seek(idx)
            fh.write(b'"yaw":9.0')
            fh.seek(0, 2)
            fh.write(b'{"type":"trial","trial":2}\n')
        with pytest.raises(ObservationConflictError, match="different observation payload"):
            rec.record_observation(envelope)
        assert rec.full_scans == 2
        rec.close()

    def test_own_partial_strict_write_is_repaired_then_retried(self, tmp_path, monkeypatch):
        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-own-torn")
        rec.record_trials([{"trial": 1}])  # opens the writer's file handle

        class HalfThenFail:
            def __init__(self, raw):
                self.raw = raw

            def write(self, data):
                self.raw.write(bytes(data[: len(data) // 2]))
                raise OSError(errno.ENOSPC, "injected torn strict write")

            def __getattr__(self, name):
                return getattr(self.raw, name)

        real = rec.trials._fh
        rec.trials._fh = HalfThenFail(real)
        with pytest.raises(OSError, match="torn strict write"):
            rec.record_observation(envelope)
        rec.trials._fh = real
        assert rec.record_observation(envelope)["idempotent"] is False
        # A failed write cannot prove how many bytes were its own: re-validated in full.
        assert rec.full_scans == 2
        assert len(_obs_lines(tmp_path / "trials.jsonl")) == 1
        assert _evidence(tmp_path)[0]["reason"] == "this writer's failed write"
        rec.close()

    def test_rotation_restart_and_receipts_without_rescans(self, tmp_path):
        def points_at_row(receipt):
            raw = (tmp_path / receipt["file"]).read_bytes()
            row = json.loads(raw[receipt["offset"]:].split(b"\n", 1)[0])
            assert row["observation_key"] == receipt["observation_key"]
            assert raw[:receipt["offset"]].count(b"\n") + 1 == receipt["line"]

        rec = LearningRecorder(tmp_path, max_bytes=4096, fsync=False)
        envelope = _terminal_envelope(run_id="run-rot")
        receipts, current = [], envelope
        for i in range(12):
            rec.record_trials([{"trial": i + 1}])
            receipts.append(rec.record_observation(current))
            points_at_row(receipts[-1])  # exact at the time it is issued
            current = _next_presentation(current)
        assert len(rec.trials.rotated_files()) >= 3
        assert rec.full_scans == 1
        current = envelope
        for receipt in receipts:  # after rotations: retries follow the renamed files
            retry = rec.record_observation(current)
            assert retry["idempotent"] is True and retry["offset"] == receipt["offset"]
            points_at_row(retry)
            current = _next_presentation(current)
        assert rec.full_scans == 1
        rec.close()
        with LearningRecorder(tmp_path, max_bytes=4096, fsync=False) as again:
            current = envelope
            for receipt in receipts:
                retry = again.record_observation(current)
                assert retry["idempotent"] is True and retry["line"] == receipt["line"]
                points_at_row(retry)
                current = _next_presentation(current)
            assert again.full_scans == 1

    def test_removed_rotated_file_forces_full_revalidation(self, tmp_path):
        rec = LearningRecorder(tmp_path, max_bytes=128, fsync=False)
        envelope = _terminal_envelope(run_id="run-removed")
        rec.record_observation(envelope)
        rec.record_observation(_next_presentation(envelope))
        [rotated] = sorted(tmp_path.glob("trials.jsonl.*"))
        rotated.rename(tmp_path / "moved-away.jsonl")
        assert rec.record_observation(envelope)["idempotent"] is False  # disk truth, re-read
        assert rec.full_scans == 2
        rec.close()

    def test_per_write_validation_work_is_independent_of_history(self, tmp_path, monkeypatch):
        costs = {}
        for history in (5, 60):
            data_dir = tmp_path / f"h{history}"
            rec = LearningRecorder(data_dir, fsync=False)
            current = _terminal_envelope(run_id=f"run-h{history}")
            for _ in range(history):
                rec.record_observation(current)
                current = _next_presentation(current)
            calls = _count_prepares(monkeypatch, rec)
            rec.record_observation(current)          # fresh: input + its own re-read row
            fresh = len(calls)
            rec.record_observation(current)          # idempotent retry: input only
            costs[history] = (fresh, len(calls) - fresh)
            assert rec._index.count() == history + 1
            sizes = rec._index._db.execute(
                "SELECT DISTINCT length(key), length(payload) FROM observations").fetchall()
            assert sizes == [(32, 32)]  # digests only, never payloads
            monkeypatch.undo()
            rec.close()
        assert costs[5] == costs[60] == (2, 1)

    def test_index_file_is_a_derived_cache_rebuilt_at_every_open(self, tmp_path):
        import sqlite3

        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id="run-cache")
        rec.record_observation(envelope)
        rec.close()
        db = sqlite3.connect(str(tmp_path / ".neurofly-observation-index.sqlite"))
        db.execute("DELETE FROM observations")
        db.execute("INSERT INTO observations VALUES (?, ?, 1, 1, 0, 1)",
                   (b"\0" * 32, b"\1" * 32))
        db.commit()
        db.close()
        with LearningRecorder(tmp_path, fsync=False) as again:
            assert again._index.count() == 1
            assert again.record_observation(envelope)["idempotent"] is True
            changed = copy.deepcopy(envelope)
            changed["provenance"]["controller_specific"]["raw_motor_command"]["yaw"] = 2.0
            with pytest.raises(ObservationConflictError):
                again.record_observation(changed)


# =============================================================================
# 5. The SQLite index is derived state: an I/O failure refuses, never acknowledges
# =============================================================================

INDEX_FILE = ".neurofly-observation-index.sqlite"


def _seed_ledger(tmp_path):
    rec = LearningRecorder(tmp_path, fsync=False)
    envelope = _terminal_envelope(run_id="run-cache-fail")
    rec.record_observation(envelope)
    rec.record_observation(_next_presentation(envelope))
    rec.close()
    return envelope, _ledger_bytes(tmp_path)


def _ledger_bytes(tmp_path):
    return {p.name: p.read_bytes() for p in sorted(tmp_path.glob("trials.jsonl*"))}


def _track_connections(monkeypatch):
    import sqlite3

    opened = []
    real = sqlite3.connect

    def spy(*args, **kwargs):
        conn = real(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(learning_recorder.sqlite3, "connect", spy)
    return opened


def _is_closed(conn):
    import sqlite3

    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


class TestDerivedIndexCache:
    @pytest.mark.parametrize("where", ["add", "get"])
    def test_cache_io_failure_refuses_then_rebuilds_without_losing_conflicts(
            self, tmp_path, monkeypatch, where):
        import sqlite3

        rec = LearningRecorder(tmp_path, fsync=False)
        envelope = _terminal_envelope(run_id=f"run-io-{where}")
        rec.record_observation(envelope)
        fresh = _next_presentation(envelope)
        original = getattr(rec._index_store, where)

        def broken(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(rec._index_store, where, broken)
        with pytest.raises(ObservationJournalError, match="index cache failed.*nothing was acknowledged"):
            rec.record_observation(fresh)
        assert rec._index is None
        monkeypatch.setattr(rec._index_store, where, original)
        receipt = rec.record_observation(fresh)  # rebuilt from byte 0
        assert receipt["durable"] is True
        assert rec.full_scans == 2
        assert len(_obs_lines(tmp_path / "trials.jsonl")) == 2
        changed = copy.deepcopy(envelope)
        changed["provenance"]["controller_specific"]["raw_motor_command"]["yaw"] = 4.0
        with pytest.raises(learning_recorder.ObservationConflictError):
            rec.record_observation(changed)
        rec.close()
