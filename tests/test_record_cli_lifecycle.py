"""`neurofly record --state-dir` lifecycle: clean exits save and mark, failures stay flagged.

Before the fix a normally finished record run wrote no final checkpoint and no
``session_end`` marker, so the next run on the same state directory restored an
older checkpoint and reported ``interrupted_unclean_shutdown``.
"""
import errno
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurofly.recording import read_recording, record_run  # noqa: E402
from neurofly_daemon import ContinuousExperimentRunner  # noqa: E402

KW = dict(paradigm="t-maze", backend="connectome-fixed", test_synthetic_graph=True)
SAVE_AT_2 = [{"step": 2, "cmd": {"action": "save_checkpoint"}}]


def ledger(state):
    path = Path(state) / "run_validity.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sessions(state):
    """session id -> ordered list of its lifecycle events."""
    out = {}
    for record in ledger(state):
        if record.get("event") in ("session_start", "activate", "session_end", "session_end_failed"):
            out.setdefault(record["session"], []).append(record["event"])
    return out


def pointer(state):
    (path,) = list(Path(state).glob("registry*/**/CURRENT.json"))
    return json.loads(path.read_text())


def checkpoint_hashes(state):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(state).glob("registry*/**/ckpt-*.npz"))}


def initial(summary):
    return read_recording(summary["path"], require_finished=False)["header"]["provenance"]["initial_state"]


def reasons(summary):
    return [i["reason"] for i in summary["result_validity"]["incidents"]]


def test_normal_runs_save_the_final_state_and_continue_cleanly(tmp_path):
    state = tmp_path / "state"
    first = record_run(out=tmp_path / "r1", steps=3, state_dir=state, schedule=SAVE_AT_2, **KW)
    assert first["status"] == "complete" and reasons(first) == []
    assert first["shutdown"] == dict(clean=True, steps_run=3, error=None, graph_step_index=3,
                                     graph_checkpoint=dict(checkpoint_version=2, step_index=3))
    assert pointer(state)["version"] == 2 and pointer(state)["step_index"] == 3
    assert list(sessions(state).values()) == [["session_start", "activate", "session_end"]]

    second = record_run(out=tmp_path / "r2", steps=3, state_dir=state, **KW)
    assert second["status"] == "complete" and reasons(second) == []
    state2 = initial(second)
    assert state2["graph_step_index"] == 3                      # the true final step, not 2
    assert state2["restore_source"]["checkpoint_version"] == 2
    assert second["shutdown"]["graph_checkpoint"] == dict(checkpoint_version=3, step_index=6)

    third = record_run(out=tmp_path / "r3", steps=2, state_dir=state, **KW)
    assert third["status"] == "complete" and reasons(third) == []
    assert initial(third)["graph_step_index"] == 6
    assert initial(third)["restore_source"] == dict(kind="graph_checkpoint", checkpoint_version=3,
                                                    step_index=6, fallback=False)
    assert list(sessions(state).values()) == [["session_start", "activate", "session_end"]] * 3
    assert not any(r.get("reason") for r in ledger(state))


def test_failed_final_save_is_incomplete_and_keeps_the_last_good_checkpoint(tmp_path, monkeypatch):
    state = tmp_path / "state"
    record_run(out=tmp_path / "r1", steps=3, state_dir=state, schedule=SAVE_AT_2, **KW)
    good_pointer, good_files = pointer(state), checkpoint_hashes(state)

    original = ContinuousExperimentRunner.save_checkpoint

    def failing(self, tag="periodic"):
        if tag == "final_shutdown":
            raise OSError(errno.ENOSPC, "No space left on device (test)")
        return original(self, tag)

    monkeypatch.setattr(ContinuousExperimentRunner, "save_checkpoint", failing)
    failed = record_run(out=tmp_path / "r2", steps=3, state_dir=state, **KW)
    monkeypatch.setattr(ContinuousExperimentRunner, "save_checkpoint", original)
    assert failed["status"] == "incomplete"
    assert failed["shutdown"]["clean"] is False and failed["shutdown"]["error"]
    assert failed["shutdown"]["graph_checkpoint"] is None and failed["shutdown"]["graph_step_index"] == 6
    assert failed["path"] and initial(failed)["graph_step_index"] == 3   # the recording itself is kept
    assert "required_save_failed" in reasons(failed)
    assert pointer(state) == good_pointer and checkpoint_hashes(state) == good_files
    second_session = list(sessions(state).values())[1]
    assert "session_end" not in second_session

    # The next run restores the last good state (never a fresh brain) and reports the loss.
    after = record_run(out=tmp_path / "r3", steps=2, state_dir=state, **KW)
    assert initial(after)["graph_step_index"] == 3
    assert initial(after)["restore_source"]["checkpoint_version"] == 2
    assert after["status"] == "incomplete"
    assert "interrupted_unclean_shutdown" in reasons(after) and "required_save_failed" in reasons(after)


def test_killed_record_process_is_still_an_unclean_shutdown(tmp_path):
    state = tmp_path / "state"
    record_run(out=tmp_path / "r1", steps=3, state_dir=state, schedule=SAVE_AT_2, **KW)
    good_pointer = pointer(state)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from neurofly.recording import main; "
            "sys.exit(main(sys.argv[2:]))")
    argv = [sys.executable, "-c", code, str(ROOT), "--paradigm", "t-maze", "--backend", "connectome-fixed",
            "--test-synthetic-graph", "--steps", "1000000", "--state-dir", str(state),
            "--out", str(tmp_path / "killed")]
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and len(sessions(state)) < 2:
            assert proc.poll() is None
            time.sleep(0.05)
        assert len(sessions(state)) == 2
        while time.monotonic() < deadline and "activate" not in list(sessions(state).values())[1]:
            time.sleep(0.05)
        time.sleep(0.5)                                         # let it advance past the checkpoint
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL
    assert list(sessions(state).values())[1] == ["session_start", "activate"]
    assert pointer(state) == good_pointer

    after = record_run(out=tmp_path / "r3", steps=2, state_dir=state, **KW)
    assert initial(after)["graph_step_index"] == 3
    assert after["status"] == "incomplete" and "interrupted_unclean_shutdown" in reasons(after)


def test_fresh_temporary_state_is_not_shut_down(tmp_path):
    summary = record_run(out=tmp_path / "a", steps=3, **KW)
    assert summary["status"] == "complete" and summary["shutdown"] is None
