"""A recording whose finalisation fails is never published, listed or read as finished.

Codex review of ff251d5, item 2: the rename used to publish the .nfrec (and set
closed=True) before the sidecar was written, so a sidecar failure left a file that
was listed as finished and accepted by the reader.  Now every required step runs on
the hidden .partial first and the rename is the last step; a failure after it
un-publishes the file or marks it INVALID.  Failures are injected before and after
the rename and the sidecar; the last good completed recording is kept.
"""
import errno
import os

import pytest

import neurofly_daemon as nd
from neurofly import recording


def _enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def _runner(tmp_path):
    return nd.ContinuousExperimentRunner(initial_paradigm="t-maze", sim_speed=20.0, checkpoint_interval=3600,
                                         output_dir=tmp_path, continuous=True)


def _record(runner, name, steps=4):
    with runner.lock:
        runner.start_recording(name=name)
        for _ in range(steps):
            runner._advance_one()
        return runner.stop_recording()


@pytest.fixture
def setup(tmp_path):
    runner = _runner(tmp_path)
    good = _record(runner, "good")
    assert good["status"] == "complete"
    with runner.lock:
        runner._clear_error("test")
    return runner, tmp_path / "recordings"


def _listed(directory):
    return {r["name"] for r in recording.list_recordings(directory)}


def _assert_only_good_is_finished(runner, directory, summary):
    assert summary["status"] == "invalid"
    assert _listed(directory) == {"good.nfrec"}                         # last good recording kept
    recording.read_recording(directory / "good.nfrec")
    bad = directory / "bad.nfrec"
    if bad.exists():
        with pytest.raises(ValueError):
            recording.read_recording(bad)
    assert runner.result_validity()["state"] == "incomplete"
    artifacts = summary["artifacts"]
    assert artifacts["completed"] is False
    for key in ("partial", "final", "sidecar", "invalid_marker"):     # reported = actually on disk
        name = artifacts[key]
        assert (name is not None) == (name is not None and (directory / name).exists())
    return artifacts


def test_footer_failure_before_rename(setup, monkeypatch):
    runner, directory = setup
    original = recording._line

    def line(record):
        if record.get("k") == "end":
            _enospc()
        return original(record)
    monkeypatch.setattr(recording, "_line", line)
    summary = _record(runner, "bad")
    artifacts = _assert_only_good_is_finished(runner, directory, summary)
    assert artifacts["final"] is None and artifacts["partial"] == ".bad.nfrec.partial"


def test_sidecar_failure_before_rename(setup, monkeypatch):
    runner, directory = setup
    monkeypatch.setattr(recording, "_replace", _enospc)                # the sidecar's atomic rename
    summary = _record(runner, "bad")
    artifacts = _assert_only_good_is_finished(runner, directory, summary)
    assert artifacts["final"] is None and artifacts["sidecar"] is None and artifacts["partial"]


def test_publish_rename_failure(setup, monkeypatch):
    runner, directory = setup
    calls = []
    original = recording._replace

    def replace(src, dst):
        calls.append(dst)
        if str(dst).endswith(".nfrec"):
            _enospc()
        return original(src, dst)
    monkeypatch.setattr(recording, "_replace", replace)
    summary = _record(runner, "bad")
    artifacts = _assert_only_good_is_finished(runner, directory, summary)
    assert artifacts["final"] is None and artifacts["sidecar"] is None    # sidecar withdrawn
    assert (directory / "bad.nfrec.json.withdrawn").exists()


def test_failure_after_rename_unpublishes(setup, monkeypatch):
    runner, directory = setup
    monkeypatch.setattr(recording, "_fsync_directory", _enospc)
    summary = _record(runner, "bad")
    artifacts = _assert_only_good_is_finished(runner, directory, summary)
    assert artifacts["final"] is None and artifacts["partial"] == ".bad.nfrec.partial"


def test_failure_after_rename_that_cannot_unpublish_is_marked_invalid(setup, monkeypatch):
    runner, directory = setup
    monkeypatch.setattr(recording, "_fsync_directory", _enospc)
    real_replace = os.replace
    published = []

    def replace(src, dst):
        if str(src).endswith("bad.nfrec") and str(dst).endswith(".partial"):
            _enospc()                                               # un-publishing fails too
        published.append(dst)
        return real_replace(src, dst)
    monkeypatch.setattr(recording.os, "replace", replace)
    summary = _record(runner, "bad")
    artifacts = _assert_only_good_is_finished(runner, directory, summary)
    assert artifacts["final"] == "bad.nfrec" and artifacts["invalid_marker"] == "bad.nfrec.INVALID"
    assert "bad.nfrec" not in _listed(directory)


# ---------------------------------------------------------------------------
# Completion fails closed without any cleanup (Codex review of b3d3837, 1)
# ---------------------------------------------------------------------------
def test_dir_sync_failure_in_an_unwritable_directory_is_never_certified(setup, monkeypatch):
    """The exact combined failure: both renames happened, then the directory sync fails
    and the directory is read-only, so neither the rollback nor a marker can be written."""
    runner, directory = setup

    def sync_fails_and_dir_turns_read_only(path):
        os.chmod(directory, 0o555)
        _enospc()
    monkeypatch.setattr(recording, "_fsync_directory", sync_fails_and_dir_turns_read_only)
    try:
        summary = _record(runner, "bad")
        artifacts = summary["artifacts"]
        assert summary["status"] == "invalid" and runner.last_error is not None
        assert artifacts["final"] == "bad.nfrec" and artifacts["sidecar"] == "bad.nfrec.json"
        assert artifacts["completion_record"] is None and artifacts["invalid_marker"] is None
        assert artifacts["certified"] is False and summary["partial_kept"] is False   # only claimed when true
        assert "bad.nfrec" not in _listed(directory)
        with pytest.raises(ValueError, match="completion"):
            recording.read_recording(directory / "bad.nfrec")
    finally:
        os.chmod(directory, 0o755)
    # Restart and replay: a fresh process sees the same files and still refuses them.
    monkeypatch.undo()
    assert _listed(directory) == {"good.nfrec"}
    assert recording.recording_invalid_reason(directory / "bad.nfrec") is not None
    with pytest.raises(ValueError):
        recording.read_recording(directory / "bad.nfrec")
    recording.read_recording(directory / "good.nfrec")


def test_completion_record_failure_is_never_certified(setup, monkeypatch):
    runner, directory = setup
    original = recording._replace

    def replace(src, dst):
        if str(dst).endswith(recording.DONE_SUFFIX):
            _enospc()
        return original(src, dst)
    monkeypatch.setattr(recording, "_replace", replace)
    summary = _record(runner, "bad")
    assert summary["status"] == "invalid" and summary["artifacts"]["certified"] is False
    assert "bad.nfrec" not in _listed(directory)


def test_a_completion_record_that_does_not_match_is_refused(setup):
    runner, directory = setup
    done = directory / ("good.nfrec" + recording.DONE_SUFFIX)
    original = done.read_text()
    try:
        done.write_text(original.replace('"sha256": "', '"sha256": "0'))
        assert "good.nfrec" not in _listed(directory)
        with pytest.raises(ValueError):
            recording.read_recording(directory / "good.nfrec")
    finally:
        done.write_text(original)
    assert _listed(directory) == {"good.nfrec"}


def test_legacy_recordings_without_the_completion_protocol_stay_readable(setup):
    import json as _json
    runner, directory = setup
    sidecar = directory / "good.nfrec.json"
    meta = _json.loads(sidecar.read_text())
    meta.pop("completion_protocol")
    meta.pop("projection")  # Historical v1 predates recording-local projection evidence.
    # Use an actual historical v1 header/frame, rather than downgrading a current v2 sidecar.
    import gzip
    import hashlib
    path = directory / "good.nfrec"
    frame = recording._line({"k": "f", "i": 0, "step": 0})
    with gzip.open(path, "wb") as handle:
        handle.write(recording._line({"k": "header", "format": recording.FORMAT, "version": 1}))
        handle.write(frame)
        handle.write(recording._line({"k": "end", "frames": 1, "events": 0,
                                     "frames_sha256": hashlib.sha256(frame).hexdigest()}))
    meta["recording"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(_json.dumps(meta))
    (directory / ("good.nfrec" + recording.DONE_SUFFIX)).unlink()
    assert _listed(directory) == {"good.nfrec"}
    recording.read_recording(directory / "good.nfrec")


# ---------------------------------------------------------------------------
# The completion acknowledgement must be durable (Codex review of 542a04f)
# ---------------------------------------------------------------------------
def test_final_directory_sync_failure_after_the_completion_record(setup, monkeypatch):
    """Codex's repro: the first directory sync works, the second (after .done) fails."""
    runner, directory = setup
    original = recording._fsync_directory
    calls = []

    def sync(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError(errno.EIO, "injected completion-directory fsync failure")
        return original(path)
    monkeypatch.setattr(recording, "_fsync_directory", sync)
    with runner.lock:
        runner.start_recording(name="failed-ack")
        for _ in range(4):
            runner._advance_one()
        summary = runner.stop_recording()
    assert len(calls) == 2
    assert summary["status"] == "invalid"                       # never "complete"
    assert runner.last_error is not None                        # the run records it (scientific: stops)
    assert runner.result_validity()["state"] == "incomplete"
    assert "Input/output error" in summary["recording_error"]["error"] or "fsync" in summary["recording_error"]["error"]
    assert summary["artifacts"]["certified"] is False           # the record was withdrawn
    assert "failed-ack.nfrec" not in _listed(directory)


@pytest.mark.parametrize("failing_call,what", [(1, "sidecar"), (2, "completion record")])
def test_content_fsync_failure_of_sidecar_or_completion_record(setup, monkeypatch, failing_call, what):
    runner, directory = setup
    calls = []
    original = recording._fsync_file

    def fsync(handle):
        calls.append(handle.name)
        if len(calls) == failing_call:
            raise OSError(errno.EIO, f"injected {what} content fsync failure")
        return original(handle)
    monkeypatch.setattr(recording, "_fsync_file", fsync)
    summary = _record(runner, "bad")
    assert summary["status"] == "invalid" and summary["artifacts"]["certified"] is False
    assert "bad.nfrec" not in _listed(directory)
    assert runner.result_validity()["state"] == "incomplete"
    assert _listed(directory) == {"good.nfrec"}


def test_a_good_recording_fsyncs_sidecar_and_record_contents(tmp_path, monkeypatch):
    seen = []
    original = recording._fsync_file
    monkeypatch.setattr(recording, "_fsync_file", lambda h: (seen.append(os.path.basename(h.name)), original(h)))
    runner = _runner(tmp_path)
    assert _record(runner, "ok")["status"] == "complete"
    assert seen == [".ok.nfrec.json.partial", ".ok.nfrec.done.partial"]
