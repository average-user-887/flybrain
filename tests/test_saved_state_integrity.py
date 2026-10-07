"""Saved-state integrity: a retained brain is never overwritten or silently replaced,
a broken (NaN/inf) state is never checkpointed, and a run's invalidity survives restarts.

Codex review of the freeze candidate (5 Oct 2026), item 3: a missing CURRENT.json on an
established instance used to start it fresh, and numbering restarted at ckpt-000001,
which could replace retained state.  Synthetic test graph, disposable directories only.
"""
import errno
import json
import os

import pytest

import experiment_registry
import neurofly_daemon as nd
from learning_recorder import LearningRecorder, RecorderThread
from tests.test_no_silent_freeze import (_graph_history, _restored, force_checkpoint_due, make_graph_runner,
                                         runners, shutdown, started, status, wait_for)  # noqa: F401
from tests.transition_control_helpers import transition_command

_DURABLE_RECORDERS = []


def durable(runner):
    """Attach the durable observation recorder a clean stop() needs.

    Since 730059e, stop() runs the 'shutdown' lifecycle command, which
    _admit_lifecycle refuses without a durable observation recorder, so
    stop() returns False.  The daemon always attaches one; tests that assert
    a clean shutdown do the same.
    """
    recorder = LearningRecorder(runner.output_dir / f"durable-records-{runner.run_id}",
                                session={"daemon_run_id": runner.run_id})
    with runner.lock:
        runner.attach_learning_records(RecorderThread(runner, recorder, summary_interval=999))
    _DURABLE_RECORDERS.append(recorder)
    return runner


@pytest.fixture(autouse=True)
def _close_durable_recorders():
    yield
    while _DURABLE_RECORDERS:
        _DURABLE_RECORDERS.pop().close()


def test_non_finite_state_is_a_compute_failure():
    assert nd.classify_failure(experiment_registry.NonFiniteState("nan")) == "compute"


def test_non_finite_brain_state_is_never_checkpointed_and_halts(tmp_path, runners, monkeypatch):
    import numpy as np
    runner = started(runners, make_graph_runner(tmp_path))
    with runner.lock:
        runner.checkpoint_now("periodic")
    instance = runner.registry.active
    directory = runner.registry.instance_dir(instance.instance_id)
    pointer = (directory / "CURRENT.json").read_text()
    files = sorted(p.name for p in (directory / "checkpoints").iterdir())
    real = instance.brain.snapshot_state

    def nan_state():
        state = real()
        state = dict(state)
        state["v"] = np.array(state["v"], dtype=float, copy=True)
        state["v"][0] = np.nan
        return state
    monkeypatch.setattr(instance.brain, "snapshot_state", nan_state)
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    monkeypatch.undo()
    st = status(runner)
    assert st["error_detail"]["failure_class"] == "compute" and st["error_detail"]["type"] == "NonFiniteState"
    assert (directory / "CURRENT.json").read_text() == pointer
    assert sorted(p.name for p in (directory / "checkpoints").iterdir()) == files


def test_invalidity_survives_a_restart_and_is_written_to_disk(tmp_path, runners, monkeypatch):
    runner = started(runners, make_graph_runner(tmp_path))
    run_id = status(runner)["result_validity"]["run_id"]
    with runner.lock:
        runner._invalidate("required_save_failed", channel="checkpoint",
                           exc=OSError(errno.ENOSPC, "No space left on device"), failure_class="persistence")
        runner.checkpoint_now("periodic")                     # carries the incident in its meta
    shutdown(runner)
    lines = [r for r in (json.loads(l) for l in (tmp_path / "run_validity.jsonl").read_text().splitlines())
             if r.get("reason")]
    assert lines[0]["run_id"] == run_id and lines[0]["reason"] == "required_save_failed"
    again = make_graph_runner(tmp_path)                        # a new process on the same data
    validity = again.result_validity()
    assert validity["run_id"] == run_id and validity["state"] == "incomplete"
    assert validity["incidents"][0]["restored_from_checkpoint"] is True


def test_absent_pointer_with_retained_checkpoints_recovers_and_never_overwrites(tmp_path):
    directory, _ = _graph_history(tmp_path, versions=3)
    ckpts = directory / "checkpoints"
    before = {p.name: p.read_bytes() for p in ckpts.glob("ckpt-*.npz")}
    assert sorted(before) == ["ckpt-000001.npz", "ckpt-000002.npz", "ckpt-000003.npz"]
    (directory / "CURRENT.json").unlink()                     # the pointer is lost
    runner, event = _restored(tmp_path)
    assert event["fallback"] is True and event["version"] == 3   # newest valid, logged
    assert "CURRENT.json missing" in event["problem"]
    assert runner.registry.active.step_index > 0               # not a fresh brain
    with runner.lock:
        runner.save_checkpoint("periodic")
    after = {p.name: p.read_bytes() for p in ckpts.glob("ckpt-*.npz")}
    assert "ckpt-000004.npz" in after                          # numbering continues above disk
    for name, data in before.items():
        assert after.get(name, data) == data                   # nothing retained was replaced


def test_absent_pointer_and_no_valid_checkpoint_refuses(tmp_path):
    directory, _ = _graph_history(tmp_path, versions=2)
    (directory / "CURRENT.json").unlink()
    for path in (directory / "checkpoints").glob("ckpt-*.npz"):
        path.write_bytes(b"damaged")
    with pytest.raises(experiment_registry.CheckpointCorrupt):
        make_graph_runner(tmp_path)


def test_checkpoint_version_is_always_above_every_file_on_disk(tmp_path):
    directory, _ = _graph_history(tmp_path, versions=1)
    stray = directory / "checkpoints" / "ckpt-000009.npz"     # e.g. an unpublished write
    stray.write_bytes(b"leftover")
    runner = make_graph_runner(tmp_path)
    with runner.lock:
        runner.save_checkpoint("periodic")
    assert (directory / "checkpoints" / "ckpt-000010.npz").exists()
    assert stray.read_bytes() == b"leftover"


def test_pruning_never_removes_the_newest_valid_version(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROFLY_KEEP_CHECKPOINTS", "1")
    directory, instance_id = _graph_history(tmp_path, versions=4)
    current = json.loads((directory / "CURRENT.json").read_text())
    names = sorted(p.name for p in (directory / "checkpoints").glob("ckpt-*.npz"))
    assert names == [current["file"]]
    runner = make_graph_runner(tmp_path)
    meta, _ = runner.registry.read_checkpoint(instance_id)
    assert meta["version"] == current["version"]


# ---------------------------------------------------------------------------
# Validity across restart (Codex review of 2c21cb0, 3): the durable ledger is read back
# ---------------------------------------------------------------------------
def _crash(runner):
    """An abrupt end: no stop(), no final checkpoint, no session_end."""
    shutdown(runner)


def test_incident_after_the_last_checkpoint_survives_a_crash_and_restart(tmp_path, runners):
    runner = started(runners, make_graph_runner(tmp_path))
    run_id = runner.result_validity()["run_id"]
    with runner.lock:
        runner.checkpoint_now("periodic")                     # last good checkpoint: no incident in it
        runner._invalidate("required_save_failed", channel="trial_ledger",
                           exc=OSError(errno.ENOSPC, "No space left on device"), failure_class="persistence")
    _crash(runner)
    again = make_graph_runner(tmp_path)
    validity = again.result_validity()
    assert validity["run_id"] == run_id                       # the same run is resumed ...
    assert validity["state"] == "incomplete"                  # ... and is NOT valid again
    reasons = {i["reason"] for i in validity["incidents"]}
    assert "required_save_failed" in reasons
    assert any(i.get("restored_from_ledger") for i in validity["incidents"])


def test_an_unclean_end_marks_the_resumed_run_interrupted_and_a_clean_one_does_not(tmp_path, runners):
    clean = durable(make_graph_runner(tmp_path / "clean"))
    clean.start()
    assert wait_for(lambda: clean.total_steps > 3)
    assert clean.stop() is True
    assert make_graph_runner(tmp_path / "clean").result_validity()["state"] == "valid_so_far"

    crashed = started(runners, make_graph_runner(tmp_path / "crash"))
    with crashed.lock:
        crashed.checkpoint_now("periodic")
    _crash(crashed)
    restarted = durable(make_graph_runner(tmp_path / "crash"))
    validity = restarted.result_validity()
    assert validity["state"] == "incomplete"
    assert [i["reason"] for i in validity["incidents"]] == ["interrupted_unclean_shutdown"]
    assert restarted.stop() is True                            # this session ends cleanly
    # Reported once per interrupted session, not again on every later start.
    again = make_graph_runner(tmp_path / "crash").result_validity()
    assert [i["reason"] for i in again["incidents"]].count("interrupted_unclean_shutdown") == 1


def test_a_malformed_or_truncated_ledger_is_tolerated_and_still_counts(tmp_path, runners):
    runner = started(runners, make_graph_runner(tmp_path))
    with runner.lock:
        runner._invalidate("required_save_failed", channel="checkpoint",
                           exc=OSError(errno.EIO, "Input/output error"), failure_class="persistence")
    runner.stop()
    ledger = tmp_path / "run_validity.jsonl"
    with open(ledger, "a") as fh:
        fh.write("not json at all\n[1, 2]\n{\"run_id\": \"x\", \"reason\": \"truncat")   # no newline: truncated
    again = make_graph_runner(tmp_path)
    assert again.validity_ledger_skipped == 3
    assert again.result_validity()["state"] == "incomplete"


def test_no_cross_run_contamination(tmp_path, runners):
    runner = started(runners, durable(make_graph_runner(tmp_path)))   # t-maze instance
    with runner.lock:
        runner._invalidate("required_save_failed", channel="checkpoint",
                           exc=OSError(errno.ENOSPC, "No space left on device"), failure_class="persistence")
        # Since 730059e a switch is a queued durable transaction (needs the
        # recorder attached above and completes outside the lock).
        result = transition_command(runner, {"action": "switch_paradigm", "paradigm": "y-maze"},
                                    lock_held=True)
    assert result["status"] == "ok"
    assert runner.result_validity()["state"] == "valid_so_far"        # y-maze run is untouched
    assert runner.stop() is True
    again = make_graph_runner(tmp_path, initial_paradigm="y-maze")
    validity = again.result_validity()
    assert validity["state"] == "valid_so_far" and validity["incidents"] == []
    assert len(validity["other_runs_incomplete"]) == 0               # t-maze not activated here


# ---------------------------------------------------------------------------
# The validity ledger never fails open (Codex review of ff251d5, item 1)
# ---------------------------------------------------------------------------
def _ledger_records(path):
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def test_truncated_tail_then_activation_then_abrupt_restart(tmp_path, runners):
    first = started(runners, make_graph_runner(tmp_path))
    run_id = first.result_validity()["run_id"]
    with first.lock:
        first.checkpoint_now("periodic")
    shutdown(first)                                            # crash 1
    ledger = tmp_path / "run_validity.jsonl"
    with open(ledger, "a") as fh:                              # a record cut off mid-write
        fh.write('{"event": "recovered", "run_id": "' + run_id + '", "at": 17')
    second = make_graph_runner(tmp_path)                       # activation
    v2 = second.result_validity()
    assert v2["run_id"] == run_id and v2["state"] == "incomplete"
    assert v2["history"]["quarantined"] and (tmp_path / v2["history"]["quarantined"]).exists()
    assert {"interrupted_unclean_shutdown", "validity_history_damaged"} <= {i["reason"] for i in v2["incidents"]}
    shutdown(second)                                           # crash 2
    # The second session's own records were not swallowed by the truncated tail ...
    starts = [r for r in _ledger_records(ledger) if r.get("event") == "session_start"]
    assert second.run_id in {r["session"] for r in starts}
    third = make_graph_runner(tmp_path)                        # ... so crash 2 is detected too
    v3 = third.result_validity()
    sessions = {i.get("interrupted_session") for i in v3["incidents"]}
    assert v3["state"] == "incomplete" and {first.run_id, second.run_id} <= sessions


def test_ledger_read_denied_is_unknown_not_valid(tmp_path, runners):
    first = durable(make_graph_runner(tmp_path))
    assert first.stop() is True
    ledger = tmp_path / "run_validity.jsonl"
    os.chmod(ledger, 0)
    try:
        again = make_graph_runner(tmp_path)
        validity = again.result_validity()
        assert validity["history"]["state"] == "unreadable"
        assert validity["state"] == "incomplete"
        assert "validity_history_unreadable" in {i["reason"] for i in validity["incidents"]}
        assert again.last_error is not None          # nor can it write: scientific mode stops
        assert status(again)["status"] == "error"
    finally:
        os.chmod(ledger, 0o644)


@pytest.mark.parametrize("exploratory", [False, True], ids=["scientific", "exploratory"])
def test_ledger_write_denied(tmp_path, runners, exploratory):
    first = durable(make_graph_runner(tmp_path))
    assert first.stop() is True
    ledger = tmp_path / "run_validity.jsonl"
    os.chmod(ledger, 0o444)
    try:
        again = make_graph_runner(tmp_path, exploratory=exploratory)
        st = status(again)
        assert st["result_validity"]["state"] == "incomplete"
        assert st["result_validity"]["unwritten_records"] > 0     # kept in memory, retried
        assert "validity_ledger" in st["persistence"]["failing"]
        if exploratory:
            assert st["halted"] is False and st["status"] == "degraded"
        else:
            assert st["halted"] is True and st["status"] == "error"
            assert st["error_detail"]["channel"] == "validity_ledger"
    finally:
        os.chmod(ledger, 0o644)


def test_malformed_field_types_do_not_crash_and_are_scoped(tmp_path, runners):
    first = durable(make_graph_runner(tmp_path))
    run_id = first.result_validity()["run_id"]
    assert first.stop() is True
    with open(tmp_path / "run_validity.jsonl", "a") as fh:
        fh.write(json.dumps({"run_id": 5, "reason": ["x"], "at": "yesterday"}) + "\n")
        fh.write(json.dumps({"event": "session_start", "session": 3}) + "\n")
        fh.write(json.dumps({"run_id": run_id, "reason": "required_save_failed", "at": "noon"}) + "\n")
        fh.write(json.dumps({"run_id": "other-run", "reason": "x", "at": True}) + "\n")
    again = make_graph_runner(tmp_path)                          # must not raise
    validity = again.result_validity()
    assert again.validity_ledger_skipped == 4
    assert validity["state"] == "incomplete"
    assert [i["reason"] for i in validity["incidents"]] == ["validity_history_damaged"]
    assert "other-run" in again.validity_history["damaged_runs"]   # scoped to its own run only


# ---------------------------------------------------------------------------
# Falling back to an older checkpoint marks the run incomplete (review of ff251d5, 3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_clean_shutdown_then_damaged_newest_checkpoint_is_incomplete(tmp_path, damage):
    runner = durable(make_graph_runner(tmp_path))
    with runner.lock:
        for _ in range(3):
            runner.step_once()
            runner.checkpoint_now("periodic")
        runner.step_once()
        runner.step_once()                                      # two steps only the final checkpoint holds
    run_id = runner.result_validity()["run_id"]
    assert runner.stop() is True                                # a CLEAN shutdown
    directory = runner.registry.instance_dir(runner.registry.active.instance_id)
    current = json.loads((directory / "CURRENT.json").read_text())
    newest = directory / "checkpoints" / current["file"]
    if damage == "corrupt":
        with open(newest, "ab") as fh:
            fh.write(b"junk")
    else:
        newest.unlink()
    again = durable(make_graph_runner(tmp_path))
    validity = again.result_validity()
    assert validity["run_id"] == run_id and validity["state"] == "incomplete"
    incident = [i for i in validity["incidents"] if i["reason"] == "restored_older_checkpoint"][0]
    assert incident["restored_version"] < current["version"] <= incident["newest_version"]
    assert incident["lost_to_step"] >= current["step_index"] > incident["lost_from_step"]
    assert incident["lost_steps"] == incident["lost_to_step"] - incident["lost_from_step"]
    if damage == "corrupt":
        assert newest.exists()                                  # the damaged file is kept
    # The mark survives further clean restarts.
    assert again.stop() is True
    assert make_graph_runner(tmp_path).result_validity()["state"] == "incomplete"


# ---------------------------------------------------------------------------
# Required fields of every ledger schema (Codex review of b3d3837, 2)
# ---------------------------------------------------------------------------
def _clean_history(tmp_path):
    first = durable(make_graph_runner(tmp_path))
    run_id = first.result_validity()["run_id"]
    assert first.stop() is True
    return run_id


def _append(tmp_path, *records):
    with open(tmp_path / "run_validity.jsonl", "a") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def test_activate_without_a_session_is_damage_for_that_run(tmp_path):
    run_id = _clean_history(tmp_path)
    _append(tmp_path, {"event": "session_start", "session": "s-x", "at": 1.0},
            {"event": "activate", "run_id": run_id, "at": 2.0})        # session missing
    again = make_graph_runner(tmp_path)
    validity = again.result_validity()
    assert validity["history"]["state"] == "damaged" and validity["history"]["skipped"] == 1
    assert validity["state"] == "incomplete"
    assert "validity_history_damaged" in {i["reason"] for i in validity["incidents"]}


@pytest.mark.parametrize("record", [
    {"event": "session_start", "at": 1.0},                                   # no session
    {"event": "session_start", "session": "", "at": 1.0},                    # empty identifier
    {"event": "session_end", "session": "s"},                                # no time
    {"event": "activate", "run_id": "", "session": "s", "at": 1.0},          # empty run id
    {"event": "gap_closed", "run_id": "r", "at": 1.0},                       # no channel
    {"event": "something_new", "session": "s", "at": 1.0},                  # unknown schema
    {"run_id": "r", "reason": "", "at": 1.0},                                # empty reason
    {"run_id": "r", "reason": "x", "event": "activate", "at": 1.0},          # two schemas
    {"session": "s", "at": 1.0},                                             # no schema at all
])
def test_incomplete_or_unknown_records_are_never_trusted(record):
    assert nd.ContinuousExperimentRunner._valid_ledger_record(record) is False


def test_well_formed_records_are_accepted():
    ok = nd.ContinuousExperimentRunner._valid_ledger_record
    assert ok({"event": "session_start", "session": "s", "pid": 1, "at": 1.0})
    assert ok({"event": "activate", "run_id": "r", "session": "s", "at": 1.0})
    assert ok({"event": "recovered", "run_id": "r", "by": "switch_paradigm", "step": 3, "at": 1.0})
    assert ok({"event": "gap_closed", "run_id": "r", "channel": "checkpoint", "step": 3, "at": 1.0})
    assert ok({"run_id": "r", "reason": "required_save_failed", "channel": "checkpoint", "at": 1.0,
               "recovered_at": None, "step": 2})


def test_unscoped_damage_marks_runs_with_history_and_scoped_damage_stays_scoped(tmp_path):
    run_id = _clean_history(tmp_path)
    _append(tmp_path, {"event": "activate", "run_id": "another-run", "at": 2.0})   # scoped elsewhere
    again = durable(make_graph_runner(tmp_path))
    assert again.result_validity()["state"] == "valid_so_far"                 # not contaminated
    assert again.stop() is True
    _append(tmp_path, {"event": "session_end", "at": 3.0})                     # no identity at all
    third = make_graph_runner(tmp_path)
    validity = third.result_validity()
    assert validity["run_id"] == run_id and validity["state"] == "incomplete"


def test_a_huge_integer_timestamp_is_damage_and_never_raises(tmp_path):
    first = durable(make_graph_runner(tmp_path))
    run_id = first.result_validity()["run_id"]
    assert first.stop() is True
    with open(tmp_path / "run_validity.jsonl", "a") as fh:
        fh.write('{"event": "session_start", "session": "s-huge", "at": 1' + "0" * 400 + "}\n")
        fh.write('{"run_id": "' + run_id + '", "reason": "x", "at": 1' + "0" * 400 + "}\n")
    assert nd.ContinuousExperimentRunner._valid_ledger_record({"event": "session_start", "session": "s",
                                                                "at": 10 ** 400}) is False
    again = make_graph_runner(tmp_path)                      # must not raise OverflowError
    validity = again.result_validity()
    assert again.validity_ledger_skipped == 2 and validity["state"] == "incomplete"
