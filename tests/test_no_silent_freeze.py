"""No silent freeze: every failure is an honest halt, or (exploratory mode only) a visibly degraded run.

Audit F (docs/receipts/audit-20261005/F-robustness.md) found seven ways in which the
daemon's simulation thread stopped while /api/status still said "online": a disk-full
or GPU error in the periodic checkpoint (#1, #2), in the brain save (#3), in the
events ledger at a trial end (#4), in a --record capture (#5), an exception while
assembling a telemetry frame (#6), and a step error while the disk is full, made
silent by re-selecting the assay (#7).  Each is reproduced here IN PROCESS, against
the real running scheduler thread, with the same fault wrappers as the audit's
faultd.py (tests/fixtures/silent_freeze/).  No disk is ever filled: the wrapper
raises OSError(ENOSPC) exactly as a full disk would.

Also covered: a dead simulation thread (status "dead" within 3 s, commands refused,
a rebuild restarts it), the stall watchdog and --exit-on-stall, the free-space
pre-check, the save policy (scientific stop vs --exploratory), failure classes
(persistence / compute / software) with genuine CUDA error types, immutable result
invalidity, slow CPU versus stall, shutdown idempotence (F5), startup
fallback to the newest checkpoint that verifies (F6), and retention for every
assay (F7).
"""
import errno
import json
import os
import socket
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import experiment_brains
import experiment_registry
import neurofly_daemon as nd
from arena import Arena
from learning_recorder import LearningRecorder, RecorderThread
from tests.transition_control_helpers import transition_command
from neurofly import recording
from stream_gateway import StreamGateway, StreamPolicy

ORIGINAL_ATOMIC_WRITE = experiment_registry.atomic_write_bytes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def cuda_error(*_a, **_k):
    raise RuntimeError("cudaErrorLaunchFailure: unspecified launch failure (injected)")


def wait_for(pred, timeout=15.0, poll=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(poll)
    return False


def status(runner):
    class Handler:
        gateway = type("Gateway", (), {"describe": lambda self: {}})()
    handler = Handler()
    handler.runner = runner
    return nd.NeuroflyHTTPHandler._status_payload(handler)


def advancing(runner, seconds=1.0):
    """Steps advanced over ``seconds`` of real time."""
    before = runner.total_steps
    time.sleep(seconds)
    return runner.total_steps - before


def make_runner(tmp_path, **kw):
    params = dict(initial_paradigm="t-maze", sim_speed=20.0, checkpoint_interval=3600, output_dir=tmp_path,
                  continuous=True)
    params.update(kw)
    return nd.ContinuousExperimentRunner(**params)


def make_graph_runner(tmp_path, **kw):
    return make_runner(tmp_path, backend="connectome-fixed", test_synthetic_graph=True, **kw)


def shutdown(runner):
    runner.running = False
    runner._wake.set()
    runner._watchdog_stop.set()
    thread = getattr(runner, "sim_thread", None)
    if thread is not None:
        thread.join(timeout=10)


@pytest.fixture
def runners():
    made = []
    yield made
    for r in made:
        shutdown(r)
        recorder_thread = getattr(r, "_test_learning_records", None)
        if recorder_thread is not None:
            recorder_thread.stop(timeout=1)


def started(runners, runner):
    runners.append(runner)
    runner.start()
    assert wait_for(lambda: runner.total_steps > 5), "loop never stepped"
    return runner


def with_learning_records(tmp_path, runner):
    recorder = LearningRecorder(tmp_path / "learning-records", fsync=False)
    thread = RecorderThread(runner, recorder, poll_interval=0.01, summary_interval=999)
    with runner.lock:
        runner.attach_learning_records(thread)
    runner._test_learning_records = thread
    thread.start()
    return runner


def force_checkpoint_due(runner):
    runner.last_checkpoint_time = 0.0


class _Kill(BaseException):
    """Not an Exception: escapes the per-iteration handler and ends the thread."""


# ---------------------------------------------------------------------------
# The seven silent freezes of audit F, under the save policy of the Codex
# direction (5 Oct 2026, decision 3): in the default SCIENTIFIC mode a failed
# required save stops the run at a step boundary and marks its result incomplete;
# only explicit --exploratory mode keeps stepping NOT SAVING with a recorded gap.
# ---------------------------------------------------------------------------
def incidents(runner):
    return status(runner)["result_validity"]["incidents"]


def assert_failed_run_stays_marked(runner, st):
    """A modular re-select starts a new run (new run_id); the failed run keeps its mark."""
    failed = {i["run_id"] for i in runner.incidents}
    validity = st["result_validity"]
    assert failed, "the failure was recorded"
    assert all(run == validity["run_id"] and validity["state"] == "incomplete"
               or run in validity["other_runs_incomplete"] for run in failed)
    assert all(i["recovered_at"] is not None for i in runner.incidents)


def assert_stopped_incomplete(runner, channel, failure_class="persistence"):
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True
    assert st["error_detail"]["failure_class"] == failure_class
    assert st["error_detail"].get("channel") == channel
    assert runner.sim_thread.is_alive(), "stopped means halted honestly, not a dead thread"
    assert st["liveness"]["state"] == "halted"
    validity = st["result_validity"]
    assert validity["state"] == "incomplete" and validity["mode"] == "scientific"
    assert any(i["channel"] == channel for i in validity["incidents"])
    steps = runner.total_steps
    time.sleep(0.3)
    assert runner.total_steps == steps                        # stopped at a step boundary
    return st


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO, errno.EROFS], ids=["ENOSPC", "EIO", "EROFS"])
def test_1_checkpoint_storage_failure_stops_the_run_and_marks_it_incomplete(tmp_path, runners, monkeypatch, code):
    runner = started(runners, make_graph_runner(tmp_path))
    with runner.lock:
        good = runner.checkpoint_now("periodic")              # a last good checkpoint exists
    instance_dir = runner.registry.instance_dir(runner.registry.active.instance_id)
    pointer_before = (instance_dir / "CURRENT.json").read_text()

    def failing(*a, **k):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(experiment_registry, "atomic_write_bytes", failing)
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    st = assert_stopped_incomplete(runner, "checkpoint")
    assert st["error_detail"]["phase"] == "persistence"
    assert "required save failed" in st["error_detail"]["recover"]
    assert (instance_dir / "CURRENT.json").read_text() == pointer_before   # last good one kept
    assert good.exists()
    # Recovery while the storage still fails: refused, the halt stays.
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "error" and "saving still fails" in ack["message"]
    assert runner.last_error is not None
    # Storage fixed: the re-select writes a checkpoint first, then resumes...
    monkeypatch.setattr(experiment_registry, "atomic_write_bytes", ORIGINAL_ATOMIC_WRITE)
    steps = runner.total_steps
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok", ack
    assert wait_for(lambda: runner.total_steps > steps + 5)
    st = status(runner)
    assert st["status"] == "online" and st["halted"] is False
    # ...and the run's result stays INCOMPLETE: recovery never erases the failure.
    assert st["result_validity"]["state"] == "incomplete"
    incident = [i for i in st["result_validity"]["incidents"] if i["channel"] == "checkpoint"][0]
    assert incident["recovered_at"] is not None and incident["reason"] == "required_save_failed"
    assert st["cleared_errors"][-1]["failure_class"] == "persistence"


def test_1b_exploratory_mode_keeps_stepping_not_saving_with_a_recorded_gap(tmp_path, runners, monkeypatch):
    runner = make_graph_runner(tmp_path)
    runner.exploratory = True
    started(runners, runner)
    calls = []

    def failing(*a, **k):
        calls.append(1)
        enospc()
    monkeypatch.setattr(experiment_registry, "atomic_write_bytes", failing)
    force_checkpoint_due(runner)
    assert wait_for(lambda: calls)
    assert wait_for(lambda: status(runner)["status"] == "degraded")
    assert runner.sim_thread.is_alive() and advancing(runner, 1.0) > 5
    st = status(runner)
    assert st["mode"] == "exploratory" and st["halted"] is False and st["error"] is None
    failing_ckpt = st["persistence"]["failing"]["checkpoint"]
    assert failing_ckpt["errno"] == errno.ENOSPC and failing_ckpt["backoff_s"] == 30.0
    assert st["persistence"]["reason"] == "disk full"
    gap = [i for i in st["result_validity"]["incidents"] if i["channel"] == "checkpoint"][0]
    assert gap["reason"] == "not_saved_exploratory_gap" and gap["gap_open"] is True
    attempts = len(calls)
    time.sleep(0.5)
    assert len(calls) == attempts                             # bounded retry (back-off)
    monkeypatch.setattr(experiment_registry, "atomic_write_bytes", ORIGINAL_ATOMIC_WRITE)
    runner.persistence.channels["checkpoint"]["next_retry_at"] = 0.0
    force_checkpoint_due(runner)
    assert wait_for(lambda: status(runner)["status"] == "online")
    st = status(runner)
    assert st["result_validity"]["state"] == "incomplete"     # the gap stays on the record
    gap = [i for i in st["result_validity"]["incidents"] if i["channel"] == "checkpoint"][0]
    assert gap["gap_open"] is False and gap["recovered_at"] is not None


def test_3_brain_save_disk_full_stops_the_run(tmp_path, runners, monkeypatch):
    runner = started(runners, make_runner(tmp_path))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    assert_stopped_incomplete(runner, "checkpoint")


def test_4_trial_ledger_disk_full_stops_the_run(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path, continuous=False, trial_length_s=0.1)
    runner = started(runners, with_learning_records(tmp_path, runner))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "log", enospc)
    assert wait_for(lambda: runner.last_error is not None)
    assert_stopped_incomplete(runner, "trial_ledger")


def test_4b_trial_ledger_disk_full_exploratory_keeps_stepping(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path, continuous=False, trial_length_s=0.1)
    runner.exploratory = True
    started(runners, with_learning_records(tmp_path, runner))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "log", enospc)
    trials = len(runner.trial_history)
    assert wait_for(lambda: "trial_ledger" in status(runner)["persistence"]["failing"])
    assert status(runner)["status"] == "degraded" and runner.sim_thread.is_alive()
    assert wait_for(lambda: len(runner.trial_history) > trials + 2)
    assert advancing(runner) > 5
    assert status(runner)["result_validity"]["state"] == "incomplete"


def test_5_failing_record_stops_the_run_and_the_recording_is_invalid(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    with runner.lock:
        runner.start_recording(name="freeze-test")
    started(runners, runner)
    monkeypatch.setattr(recording.RunRecorder, "capture", enospc)
    assert wait_for(lambda: runner.recorder is None and runner.last_error is not None)
    st = assert_stopped_incomplete(runner, "recording")
    assert st["recording"] is None
    assert st["recording_error"]["recording"] == "freeze-test.nfrec"
    assert "INVALID" in st["recording_error"]["message"]
    assert list((tmp_path / "recordings").glob(".freeze-test.nfrec.partial"))   # evidence kept


def test_5b_failing_record_in_exploratory_mode_stops_the_recording_only(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    runner.exploratory = True
    with runner.lock:
        runner.start_recording(name="freeze-test")
    started(runners, runner)
    monkeypatch.setattr(recording.RunRecorder, "capture", enospc)
    assert wait_for(lambda: runner.recorder is None)
    st = status(runner)
    assert st["status"] == "degraded" and st["halted"] is False
    assert runner.sim_thread.is_alive() and advancing(runner) > 5
    assert st["result_validity"]["state"] == "incomplete"


def test_5c_batch_record_run_returns_a_structured_invalid_result(tmp_path, monkeypatch):
    original = recording.RunRecorder.capture
    calls = []

    def capture_fails_later(self, runner, step_result=None):
        calls.append(1)
        if len(calls) > 5:
            enospc()
        return original(self, runner, step_result)
    monkeypatch.setattr(recording.RunRecorder, "capture", capture_fails_later)
    summary = recording.record_run(paradigm="t-maze", out=tmp_path / "rec", steps=40)
    assert summary["status"] == "invalid" and summary["path"] is None
    assert "No space left" in summary["recording_error"]["error"]
    assert summary["result_validity"]["state"] == "incomplete"
    assert list(tmp_path.glob(".rec.nfrec.partial"))
    monkeypatch.setattr(recording, "record_run", lambda **kw: summary)
    assert recording.main(["--paradigm", "t-maze", "--out", str(tmp_path / "x"), "--steps", "5"]) == 1


def test_6_telemetry_error_becomes_an_honest_halt_and_recovers(tmp_path, runners, monkeypatch):
    runner = started(runners, make_runner(tmp_path))
    original = nd.ContinuousExperimentRunner._assemble_telemetry
    armed = {"on": True}

    def flaky(self, step_res):
        if armed["on"]:
            raise RuntimeError("injected fault (telemetry)")
        return original(self, step_res)
    monkeypatch.setattr(nd.ContinuousExperimentRunner, "_assemble_telemetry", flaky)
    assert wait_for(lambda: runner.last_error is not None)
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True
    assert "telemetry publication failed" in st["error"]
    assert st["error_detail"]["phase"] == "publish"
    assert runner.sim_thread.is_alive() and st["liveness"]["state"] == "halted"
    steps = runner.total_steps
    time.sleep(0.3)
    assert runner.total_steps == steps                       # halted: nothing advances
    armed["on"] = False                                      # the fault is gone
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok" and ack["ack"]["halted_by_error"] is None
    assert wait_for(lambda: runner.total_steps > steps + 5)
    st = status(runner)
    assert st["status"] == "online"
    assert_failed_run_stays_marked(runner, st)                # the halt stays on the record


def _step_fails_once(monkeypatch):
    original = Arena.step
    fired = []

    def step_once_fails(self, dt):
        if not fired:
            fired.append(1)
            raise RuntimeError("injected fault (arena step)")
        return original(self, dt)
    monkeypatch.setattr(Arena, "step", step_once_fails)


def test_7_step_error_while_disk_full_then_reselect_assay(tmp_path, runners, monkeypatch):
    """The audit's worst case: the halt bookkeeping write failed (disk full), the
    thread died, and re-selecting the assay CLEARED the halt with nothing stepping."""
    runner = started(runners, make_runner(tmp_path))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "log", enospc)      # disk stays full
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    _step_fails_once(monkeypatch)
    assert wait_for(lambda: runner.last_error is not None)
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True
    assert runner.sim_thread.is_alive(), "the halt path's own write killed the thread"
    assert "halt_log" in st["persistence"]["failing"]          # diagnostic: reported, not fatal
    # Re-select with the disk STILL full: the switch must save the brain -> refused.
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "error" and runner.last_error is not None
    assert status(runner)["status"] == "error"                 # honest, never a silent "online"
    # Disk freed: the documented recovery resumes stepping; the record keeps the halt.
    monkeypatch.undo()
    steps = runner.total_steps
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok", ack
    assert wait_for(lambda: runner.total_steps > steps + 5)
    st = status(runner)
    assert st["status"] == "online"
    assert_failed_run_stays_marked(runner, st)


def test_7b_exploratory_switch_requires_durable_event_then_recovers_after_disk_repair(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    runner.exploratory = True
    started(runners, runner)
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "log", enospc)
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    _step_fails_once(monkeypatch)
    assert wait_for(lambda: runner.last_error is not None)
    steps, old_identity = runner.total_steps, runner.identity()
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "error" and not ack["ack"]["applied"], ack
    assert runner.identity() == old_identity and runner.total_steps == steps
    assert ack["command_id"] == runner.command_acks[-1]["command_id"]
    assert status(runner)["status"] == "error" and runner.last_error is not None
    assert runner.result_validity()["state"] == "incomplete"
    # C2 strengthens the old permissive recovery: an applied switch requires a
    # durable event even in exploratory mode; repair storage before retrying.
    monkeypatch.undo()
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok" and ack["ack"]["applied"], ack
    assert wait_for(lambda: runner.total_steps > steps + 5)
    st = status(runner)
    assert st["status"] == "online" and st["halted"] is False
    assert_failed_run_stays_marked(runner, st)


# ---------------------------------------------------------------------------
# Compute / device failures ALWAYS halt -- genuine CUDA error types, raised at the
# engine layer, not a RuntimeError mocked into a disk write.
# ---------------------------------------------------------------------------
def _cuda_error():
    runtime = pytest.importorskip("cupy_backends.cuda.api.runtime")
    return runtime.CUDARuntimeError(719)     # cudaErrorLaunchFailure


def test_cuda_error_in_the_connectome_step_halts_as_a_device_failure(tmp_path, runners, monkeypatch):
    err = _cuda_error()
    runner = make_graph_runner(tmp_path)
    runner.exploratory = True                                 # even in exploratory mode
    started(runners, runner)
    original = nd.GraphArenaController.__call__

    def engine_fails(self, *a, **k):
        raise err
    monkeypatch.setattr(nd.GraphArenaController, "__call__", engine_fails)
    assert wait_for(lambda: runner.last_error is not None)
    monkeypatch.setattr(nd.GraphArenaController, "__call__", original)
    st = status(runner)
    assert st["status"] == "error" and st["error_detail"]["failure_class"] == "compute"
    assert st["error_detail"]["phase"] == "device" and "cudaErrorLaunchFailure" in st["error"]
    assert st["error_detail"]["type"] == "CUDARuntimeError"
    assert st["result_validity"]["state"] == "incomplete"


def test_runtime_device_fault_with_auto_engine_halts_and_does_not_switch_engine(tmp_path, runners, monkeypatch):
    """'auto' only chooses the engine at set-up; a running brain's device fault still halts."""
    err = _cuda_error()
    runner = make_graph_runner(tmp_path, brain_backend="auto")
    runner.exploratory = True                                 # even in exploratory mode
    started(runners, runner)
    instance = runner.registry.active
    brain, engine = instance.brain, instance.brain.backend

    def device_fails(*a, **k):
        raise err
    monkeypatch.setattr(brain, "step", device_fails)
    assert wait_for(lambda: runner.last_error is not None)
    monkeypatch.undo()
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True
    assert st["error_detail"]["failure_class"] == "compute" and st["error_detail"]["phase"] == "device"
    assert runner.registry.active is instance and instance.brain is brain and brain.backend == engine
    assert runner.registry.brain_backend == "auto"


def test_cuda_error_in_the_checkpoint_device_copy_halts_even_in_exploratory_mode(tmp_path, runners, monkeypatch):
    err = _cuda_error()
    runner = make_graph_runner(tmp_path)
    runner.exploratory = True
    started(runners, runner)
    instance = runner.registry.active

    def device_copy_fails():
        raise err                     # GraphInstance.state_arrays -> brain.snapshot_state()
    monkeypatch.setattr(instance.brain, "snapshot_state", device_copy_fails)
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    monkeypatch.undo()
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True   # not "degraded"
    assert st["error_detail"]["failure_class"] == "compute" and st["error_detail"]["phase"] == "device"
    assert st["error_detail"]["channel"] == "checkpoint"
    assert any(i["reason"] == "compute_failure_while_saving" for i in st["result_validity"]["incidents"])
    # Recovery from a halt raised in a save writes a fresh checkpoint before resuming.
    steps = runner.total_steps
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok", ack
    assert wait_for(lambda: runner.total_steps > steps + 5)
    st = status(runner)
    assert st["status"] == "online" and "checkpoint" not in st["persistence"]["failing"]
    assert st["result_validity"]["state"] == "incomplete"


def test_non_finite_fly_pose_halts_as_a_compute_failure(tmp_path, runners, monkeypatch):
    runner = started(runners, make_runner(tmp_path))
    original = Arena.step

    def poisoned(self, dt):
        result = original(self, dt)
        self.fly.pos.x = float("nan")
        return result
    monkeypatch.setattr(Arena, "step", poisoned)
    assert wait_for(lambda: runner.last_error is not None)
    st = status(runner)
    assert st["error_detail"]["failure_class"] == "compute" and st["error_detail"]["type"] == "NonFiniteStep"


def test_a_bug_in_a_save_is_a_software_failure_and_halts_in_exploratory_mode(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    runner.exploratory = True
    started(runners, runner)

    def bug(self):
        raise ValueError("Out of range float values are not JSON compliant")
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", bug)
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    assert status(runner)["error_detail"]["failure_class"] == "software"


def test_failure_classification_is_explicit():
    cls = nd.classify_failure
    assert cls(OSError(errno.ENOSPC, "x")) == "persistence"
    assert cls(OSError(errno.EIO, "x")) == "persistence"
    assert cls(PermissionError(errno.EACCES, "x", "/data/cuda/ckpt.npz")) == "persistence"  # path ignored
    assert cls(nd.DiskSpaceLow(28, "low")) == "persistence"
    assert cls(_cuda_error()) == "compute"
    assert cls(RuntimeError("CUDA error: an illegal memory access was encountered")) == "compute"
    assert cls(MemoryError()) == "compute"
    assert cls(ValueError("bad value")) == "software"
    assert cls(KeyError("x")) == "software"


def test_slow_cpu_is_slow_not_stalled(tmp_path, runners):
    runner = make_runner(tmp_path, sim_speed=1.0)
    runner.min_stall_s = 1.0
    started(runners, runner)
    runner.step_hook = lambda r: time.sleep(2.5)              # every step takes ~2.5 s
    seen = set()
    end = time.time() + 9
    while time.time() < end:
        seen.add(status(runner)["liveness"]["state"])
        time.sleep(0.1)
    runner.step_hook = None
    assert "slow" in seen and "stalled" not in seen and "dead" not in seen
    assert status(runner)["status"] in ("online",)


# ---------------------------------------------------------------------------
# F1: the thread cannot die silently
# ---------------------------------------------------------------------------
def test_dead_thread_is_reported_within_3_s_and_a_rebuild_restarts_it(tmp_path, runners):
    runner = make_runner(tmp_path)
    with runner.lock:
        runner.checkpoint_now("verified")
    verified_u = runner.active_brain.circuit.u.copy()
    verified_brain_id = runner.active_brain.brain_id
    failed_run = runner.manifest.run_id
    started(runners, runner)

    def kill(_runner):
        runner.step_hook = None
        runner.paused = True
        runner.active_brain.circuit.u[:] = 987654.0
        raise _Kill("injected thread death")
    runner.step_hook = kill
    t0 = time.time()
    assert wait_for(lambda: status(runner)["liveness"]["state"] == "dead", timeout=3.0)
    assert time.time() - t0 < 3.0
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True
    assert st["error"].startswith("simulation thread stopped: _Kill")
    assert st["timing"]["achieved_speed"] == 0.0               # never a frozen 1.6x
    assert st["loop_failure"]["type"] == "_Kill"
    assert st["result_validity"]["state"] == "incomplete"
    incident = st["result_validity"]["incidents"][0]
    assert incident["reason"] == "unexpected_loop_exit"
    assert incident["failure_phase"] == "step"
    assert incident["source_run_id"] == failed_run and incident["source_segment_id"]
    assert len([i for i in runner.incidents if i["reason"] == "unexpected_loop_exit"]) == 1
    # Nothing is applied inline while the loop should be running...
    refused = runner.dispatch_command({"action": "set_speed", "speed": 5})
    assert refused["status"] == "error" and refused["loop_dead"] is True
    # ...but a rebuild restarts the thread and the ack says so.
    steps = runner.total_steps
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok" and ack["loop_restarted"] is True and ack["ack"]["applied"] is True
    assert runner.active_brain.restored and runner.active_brain.brain_id == verified_brain_id
    assert (runner.active_brain.circuit.u == verified_u).all()
    assert runner.dispatch_command({"action": "set_paused", "paused": False})["status"] == "ok"
    assert wait_for(lambda: runner.total_steps > steps + 5)
    recovered = status(runner)
    assert recovered["status"] == "online"
    assert failed_run in recovered["result_validity"]["other_runs_incomplete"]
    assert next(i for i in runner.incidents if i["reason"] == "unexpected_loop_exit")["recovered_at"] is not None


def test_dead_loop_shutdown_keeps_verified_brain_and_restart_keeps_incident(tmp_path, runners):
    runner = make_runner(tmp_path)
    with runner.lock:
        runner.checkpoint_now("verified")
    brain_path = runner.active_brain.path
    verified_bytes = brain_path.read_bytes()
    started(runners, runner)

    def damage_then_die(_runner):
        runner.step_hook = None
        runner.active_brain.circuit.u[:] = -7654321.0
        raise _Kill("damaged state")

    runner.step_hook = damage_then_die
    assert wait_for(lambda: runner._loop_dead())
    assert runner.stop() is False
    assert brain_path.read_bytes() == verified_bytes
    assert not list((tmp_path / "checkpoints").glob("checkpoint_t-maze_final_shutdown_*.json"))
    ledger = [json.loads(line) for line in (tmp_path / "run_validity.jsonl").read_text().splitlines()]
    failures = [r for r in ledger if r.get("reason") == "unexpected_loop_exit"]
    assert len(failures) == 1
    restarted = make_runner(tmp_path)
    runners.append(restarted)
    assert any(r.get("reason") == "unexpected_loop_exit" for r in restarted._validity_ledger)


def test_dead_graph_loop_rebuild_restores_verified_world(tmp_path, runners):
    runner = make_graph_runner(tmp_path)
    with runner.lock:
        runner.checkpoint_now("verified")
    verified_x = runner.arena.fly.pos.x
    started(runners, runner)

    def damage_then_die(_runner):
        runner.step_hook = None
        runner.paused = True
        runner.arena.fly.pos.x = 123456.0
        raise _Kill("damaged graph world")

    runner.step_hook = damage_then_die
    assert wait_for(lambda: runner._loop_dead())
    ack = transition_command(runner, {"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok" and ack["loop_restarted"] is True
    assert runner.arena.fly.pos.x == verified_x
    assert status(runner)["result_validity"]["state"] == "incomplete"


def test_unexplained_scheduler_return_is_incomplete_once(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    monkeypatch.setattr(runner, "_schedule_loop", lambda: None)
    runners.append(runner)
    runner.start()
    assert wait_for(lambda: runner._loop_dead())
    assert runner.loop_failure["type"] == "LoopExit"
    assert runner.error_detail["type"] == "LoopExit"
    assert [i["reason"] for i in runner.result_validity()["incidents"]] == ["unexpected_loop_exit"]


def test_loop_failure_ledger_error_does_not_mask_original(tmp_path, runners, monkeypatch):
    runner = started(runners, make_runner(tmp_path))
    real_open = open
    ledger_path = tmp_path / "run_validity.jsonl"

    def fail_validity(path, mode="r", *args, **kwargs):
        if Path(path) == ledger_path and "a" in mode:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_validity)
    runner.step_hook = lambda _r: (_ for _ in ()).throw(_Kill("original loop failure"))
    assert wait_for(lambda: runner._loop_dead())
    assert runner.error_detail["type"] == "_Kill"
    assert "original loop failure" in runner.last_error
    reasons = [i["reason"] for i in runner.incidents]
    assert reasons.count("unexpected_loop_exit") == 1
    assert reasons.count("validity_ledger_unwritable") == 1


def test_thread_excepthook_backstop_records_uncaught_errors(tmp_path):
    runner = make_runner(tmp_path)
    nd._install_thread_excepthook(runner)

    def boom():
        raise ValueError("uncaught in a NeuroFly thread")
    t = threading.Thread(target=boom, name="NeuroFly-Test")
    t.start()
    t.join()
    assert runner.thread_failures[-1]["thread"] == "NeuroFly-Test"
    assert status(runner)["thread_failures"][-1]["type"] == "ValueError"


def test_a_failing_status_handler_answers_500_instead_of_dropping(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    orig = nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway
    nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway = runner, StreamGateway(StreamPolicy())
    server = ThreadingHTTPServer(("127.0.0.1", 0), nd.NeuroflyHTTPHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(nd.NeuroflyHTTPHandler, "_status_payload", lambda self: 1 / 0)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/status", timeout=5)
        assert err.value.code == 500
        assert "ZeroDivisionError" in json.loads(err.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()
        nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway = orig


# ---------------------------------------------------------------------------
# F2: free-space pre-check and the learning records
# ---------------------------------------------------------------------------
def test_low_disk_stops_before_enospc_without_writing(tmp_path, runners, monkeypatch):
    import shutil
    runner = started(runners, make_runner(tmp_path))
    writes = []
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", lambda self: writes.append(1))
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 10**8, 10**8))
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    entry = status(runner)["persistence"]["failing"]["checkpoint"]
    assert entry["state"] == "disk_low" and entry["failure_class"] == "persistence"
    assert writes == []                                        # nothing was attempted
    assert_stopped_incomplete(runner, "checkpoint")


def _recorder(tmp_path, runner):
    from learning_recorder import LearningRecorder, RecorderThread
    rec = LearningRecorder(tmp_path / "learning")
    thread = RecorderThread(runner, rec, poll_interval=0.05, summary_interval=0.05)
    runner.learning_records = thread
    return rec, thread


def test_diagnostic_summary_failure_only_degrades(tmp_path, runners):
    runner = started(runners, make_runner(tmp_path))
    rec, thread = _recorder(tmp_path, runner)
    rec.record_summary = enospc
    thread.start()
    try:
        assert wait_for(lambda: "telemetry_summary" in status(runner)["persistence"]["failing"])
        st = status(runner)
        assert st["status"] == "degraded" and st["halted"] is False
        assert st["result_validity"]["state"] == "valid_so_far"   # the scientific record is intact
        assert advancing(runner) > 5
    finally:
        thread._stop_event.set()
        thread.join(timeout=5)


def test_required_trial_records_failure_stops_and_the_mark_is_never_cleared(tmp_path, runners):
    runner = started(runners, make_runner(tmp_path))
    rec, thread = _recorder(tmp_path, runner)
    original = rec.record_trials
    rec.record_trials = enospc
    thread.start()
    try:
        assert wait_for(lambda: runner.last_error is not None)
        assert_stopped_incomplete(runner, "learning_records")
        rec.record_trials = original                           # the recorder recovers ...
        assert wait_for(lambda: not thread.describe()["failing"])
        st = status(runner)
        assert st["result_validity"]["state"] == "incomplete"   # ... the run's mark stays
        assert st["halted"] is True
    finally:
        thread._stop_event.set()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# F3: watchdog
# ---------------------------------------------------------------------------
def test_a_step_that_never_returns_is_stalled_after_the_hard_limit(tmp_path, runners):
    runner = make_runner(tmp_path)
    runner.step_hard_limit_s = 0.5
    started(runners, runner)
    release = threading.Event()
    runner.step_hook = lambda r: release.wait(20)
    try:
        assert wait_for(lambda: status(runner)["liveness"]["state"] == "slow" or
                        status(runner)["liveness"]["state"] == "stalled", timeout=5)
        assert wait_for(lambda: status(runner)["liveness"]["state"] == "stalled", timeout=5)
        st = status(runner)
        assert st["status"] == "error" and "one step has run for" in st["error"]
    finally:
        runner.step_hook = None
        release.set()
    assert wait_for(lambda: status(runner)["liveness"]["state"] == "advancing", timeout=5)


def test_no_step_while_running_is_stalled_and_exit_on_stall_fires(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    runner.min_stall_s = 0.5
    runner.watchdog_interval_s = 0.1
    runner.exit_on_stall_s = 0.5
    exits = []
    runner._exit = exits.append
    started(runners, runner)
    gate = threading.Event()
    original = runner._drain_commands
    monkeypatch.setattr(runner, "_drain_commands", lambda: gate.wait(20) or original())
    try:
        assert wait_for(lambda: status(runner)["liveness"]["state"] == "stalled", timeout=5)
        st = status(runner)
        assert st["status"] == "error" and "no step for" in st["error"]
        assert st["timing"]["achieved_speed"] == 0.0
        assert wait_for(lambda: exits == [70], timeout=5)
    finally:
        gate.set()


def test_pause_is_not_a_stall(tmp_path, runners):
    runner = make_runner(tmp_path)
    runner.min_stall_s = 0.3
    started(runners, runner)
    runner.dispatch_command({"action": "set_paused", "paused": True})
    time.sleep(0.8)
    assert status(runner)["liveness"]["state"] == "paused" and status(runner)["status"] == "online"
    runner.dispatch_command({"action": "set_paused", "paused": False})
    time.sleep(0.2)
    assert status(runner)["liveness"]["state"] == "advancing"


def test_heartbeat_carries_step_and_liveness_when_the_loop_is_dead(tmp_path, runners):
    runner = started(runners, make_runner(tmp_path))
    runner.step_hook = lambda r: (_ for _ in ()).throw(_Kill("dead"))
    assert wait_for(lambda: not runner.sim_thread.is_alive())
    orig = nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway
    nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway = runner, StreamGateway(StreamPolicy())
    server = ThreadingHTTPServer(("127.0.0.1", 0), nd.NeuroflyHTTPHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        s = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=10)
        s.sendall(b"GET /api/stream HTTP/1.1\r\nHost: x\r\n\r\n")
        buf, beat = b"", None
        end = time.time() + 8
        while beat is None and time.time() < end:
            buf += s.recv(65536)
            for block in buf.split(b"\n\n"):
                if block.startswith(b"event: heartbeat"):
                    beat = json.loads(block.split(b"data: ", 1)[1])
        s.close()
    finally:
        runner.running = False
        server.shutdown()
        server.server_close()
        nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway = orig
    assert beat is not None
    assert beat["liveness"]["state"] == "dead" and beat["status"] == "error"
    assert beat["step"] == runner.total_steps and beat["halted"] is True


# ---------------------------------------------------------------------------
# F5: shutdown
# ---------------------------------------------------------------------------
@pytest.fixture
def shutdown_recorders():
    records = []
    def attach(runner):
        recorder = LearningRecorder(runner.output_dir / f"shutdown-test-records-{runner.run_id}",
                                    session={"daemon_run_id": runner.run_id})
        runner.attach_learning_records(RecorderThread(runner, recorder, summary_interval=999))
        records.append(recorder)
    yield attach
    for recorder in records:
        recorder.close()


def test_stop_runs_once_and_a_failed_final_checkpoint_is_reported(tmp_path, monkeypatch, shutdown_recorders):
    runner = make_runner(tmp_path)
    shutdown_recorders(runner)
    runner.start()
    assert wait_for(lambda: runner.total_steps > 2)
    assert runner.stop() is True
    assert runner.stop() is True                               # second call: no-op
    finals = list((tmp_path / "checkpoints").glob("checkpoint_t-maze_final_shutdown_*.json"))
    assert len(finals) == 1

    other = make_runner(tmp_path / "b")
    shutdown_recorders(other)
    other.start()
    assert wait_for(lambda: other.total_steps > 2)
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    assert other.stop() is False
    assert other.stop() is False
    assert not other.sim_thread.is_alive()
    assert "checkpoint" in other.persistence.describe()["failing"]


# ---------------------------------------------------------------------------
# F6: startup falls back to the newest checkpoint that verifies
# ---------------------------------------------------------------------------
def _graph_history(tmp_path, versions=3):
    runner = make_graph_runner(tmp_path)
    with runner.lock:
        for _ in range(versions):
            runner.step_once()
            runner.save_checkpoint("periodic")
    inst = runner.registry.active
    directory = runner.registry.instance_dir(inst.instance_id)
    return directory, inst.instance_id


def _restored(tmp_path):
    runner = make_graph_runner(tmp_path)
    events = [json.loads(l) for l in (runner.registry.instance_dir(runner.registry.active.instance_id)
                                       / "events.jsonl").read_text().splitlines()]
    return runner, [e for e in events if e["kind"] == "restore"][-1]


def test_corrupt_current_checkpoint_falls_back_to_the_previous_version(tmp_path):
    directory, _ = _graph_history(tmp_path)
    current = json.loads((directory / "CURRENT.json").read_text())
    with open(directory / "checkpoints" / current["file"], "ab") as fh:
        fh.write(b"junk")
    runner, event = _restored(tmp_path)
    assert event["fallback"] is True and event["version"] == current["version"] - 1
    assert event["sha256_checked"] is True
    assert runner.registry.active.checkpoint_version == current["version"]   # damaged file kept
    assert (directory / "checkpoints" / current["file"]).exists()


@pytest.mark.parametrize("damage", ["missing_file", "empty_current"])
def test_missing_file_or_empty_pointer_falls_back(tmp_path, damage):
    directory, _ = _graph_history(tmp_path)
    current = json.loads((directory / "CURRENT.json").read_text())
    if damage == "missing_file":
        (directory / "checkpoints" / current["file"]).unlink()
        expected = current["version"] - 1
    else:
        (directory / "CURRENT.json").write_text("")
        expected = current["version"]
    _, event = _restored(tmp_path)
    assert event["fallback"] is True and event["version"] == expected


def test_no_verifying_checkpoint_refuses_with_one_plain_sentence(tmp_path):
    directory, _ = _graph_history(tmp_path, versions=2)
    for path in (directory / "checkpoints").glob("ckpt-*.npz"):
        path.write_bytes(b"not a checkpoint")
    with pytest.raises(experiment_registry.CheckpointCorrupt) as err:
        make_graph_runner(tmp_path)
    args = nd.build_arg_parser().parse_args(["--output-dir", str(tmp_path), "--backend", "connectome-fixed"])
    message = nd.startup_refusal(args, err.value)
    assert message.count("\n") == 0 and "Traceback" not in message
    assert message.startswith("[Daemon] Cannot start: saved state in") and "--output-dir" in message


def test_truncated_registry_index_is_a_plain_refusal(tmp_path):
    _graph_history(tmp_path, versions=1)
    index = next(tmp_path.glob("registry*/registry.json"))
    index.write_text(index.read_text()[:40])
    with pytest.raises(experiment_registry.IncompatibleCheckpoint) as err:
        make_graph_runner(tmp_path)
    args = nd.build_arg_parser().parse_args(["--output-dir", str(tmp_path)])
    assert "registry.json is not valid JSON" in nd.startup_refusal(args, err.value)


# ---------------------------------------------------------------------------
# F7: retention for every assay; bounded shutdown records
# ---------------------------------------------------------------------------
def test_json_checkpoints_are_pruned_for_all_assays(tmp_path):
    runner = make_runner(tmp_path)
    runner.keep_checkpoints = 3
    runner.keep_shutdown_checkpoints = 2
    ckpts = runner.checkpoints_dir
    for assay in ("open-arena", "y-maze", "multisensory-sandbox"):
        for i in range(6):
            (ckpts / f"checkpoint_{assay}_periodic_{1000 + i}.json").write_text("{}")
            (ckpts / f"checkpoint_{assay}_final_shutdown_{2000 + i}.json").write_text("{}")
    (ckpts / "checkpoint_y-maze_manual_5.json").write_text("{}")
    with runner.lock:
        runner.save_checkpoint("periodic")                      # active assay: t-maze
    for assay in ("open-arena", "y-maze", "multisensory-sandbox"):
        periodic = sorted(ckpts.glob(f"checkpoint_{assay}_periodic_*.json"))
        finals = sorted(ckpts.glob(f"checkpoint_{assay}_final_shutdown_*.json"))
        assert [p.name for p in periodic] == [f"checkpoint_{assay}_periodic_{1000 + i}.json" for i in (3, 4, 5)]
        assert [p.name for p in finals] == [f"checkpoint_{assay}_final_shutdown_{2000 + i}.json" for i in (4, 5)]
    assert (ckpts / "checkpoint_y-maze_manual_5.json").exists()


def test_checkpoint_interval_is_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv("NEUROFLY_CHECKPOINT_INTERVAL", "300")
    args = nd.build_arg_parser().parse_args([])
    assert args.checkpoint_interval == 300.0
    assert args.exploratory is False and args.exit_on_stall is None


# ---------------------------------------------------------------------------
# Saved-brain overwrite risk (Codex review, item 3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Recording finalisation is part of the required capture (Codex review of 2c21cb0, 1)
# ---------------------------------------------------------------------------
def _failing_close(self):
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def test_record_stop_with_failing_finalisation_is_invalid_and_stops(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    with runner.lock:
        runner.start_recording(name="fin")
    started(runners, runner)
    monkeypatch.setattr(recording.RunRecorder, "close", _failing_close)
    ack = runner.dispatch_command({"action": "record_stop"})
    assert ack["status"] == "error" and ack["recording"]["status"] == "invalid"
    assert runner.recorder is None and runner.recording_error["during"] == "finalise"
    assert_stopped_incomplete(runner, "recording")
    assert list((tmp_path / "recordings").glob(".fin.nfrec.partial"))        # last good data kept


def test_shutdown_with_failing_recording_finalisation_is_recorded_and_nonzero(tmp_path, monkeypatch, shutdown_recorders):
    runner = make_runner(tmp_path)
    shutdown_recorders(runner)
    with runner.lock:
        runner.start_recording(name="fin")
    runner.start()
    assert wait_for(lambda: runner.total_steps > 5)
    monkeypatch.setattr(recording.RunRecorder, "close", _failing_close)
    assert runner.stop() is False
    validity = runner.result_validity()
    assert validity["state"] == "incomplete"
    assert any(i["channel"] == "recording" for i in validity["incidents"])
    ledger = [json.loads(l) for l in (tmp_path / "run_validity.jsonl").read_text().splitlines()]
    assert any(r.get("channel") == "recording" and r.get("reason") for r in ledger)
    assert not any(r.get("event") == "session_end" for r in ledger)            # not a clean end


def test_batch_recording_with_failing_finalisation_returns_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(recording.RunRecorder, "close", _failing_close)
    summary = recording.record_run(paradigm="t-maze", out=tmp_path / "rec", steps=10)
    assert summary["status"] == "invalid" and summary["path"] is None
    assert summary["recording_error"]["during"] == "finalise"
    assert summary["result_validity"]["state"] == "incomplete"


# ---------------------------------------------------------------------------
# The run manifest is required provenance (Codex review of 2c21cb0, 2)
# ---------------------------------------------------------------------------
def _failing_manifest_write(self, path):
    raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def test_manifest_write_failure_stops_a_scientific_run(tmp_path, runners, monkeypatch):
    import provenance
    monkeypatch.setattr(provenance.RunManifest, "write", _failing_manifest_write)
    runner = make_runner(tmp_path)
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True                  # never a healthy status
    assert st["error_detail"]["channel"] == "provenance"
    assert st["result_validity"]["state"] == "incomplete"
    runners.append(runner)
    runner.start()
    time.sleep(0.5)
    assert runner.total_steps == 0 and status(runner)["status"] == "error"


def test_manifest_write_failure_in_exploratory_mode_continues_degraded(tmp_path, runners, monkeypatch):
    import provenance
    monkeypatch.setattr(provenance.RunManifest, "write", _failing_manifest_write)
    runner = started(runners, make_runner(tmp_path, exploratory=True))
    st = status(runner)
    assert st["status"] == "degraded" and st["halted"] is False and st["mode"] == "exploratory"
    assert "provenance" in st["persistence"]["failing"]
    assert any(i["channel"] == "provenance" for i in st["result_validity"]["incidents"])
    assert advancing(runner) > 5
