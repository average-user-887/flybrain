"""No silent freeze: every failure is an honest halt or a degraded-but-stepping run.

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
pre-check, --halt-on-persistence-failure, shutdown idempotence (F5), startup
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


def started(runners, runner):
    runners.append(runner)
    runner.start()
    assert wait_for(lambda: runner.total_steps > 5), "loop never stepped"
    return runner


def force_checkpoint_due(runner):
    runner.last_checkpoint_time = 0.0


class _Kill(BaseException):
    """Not an Exception: escapes the per-iteration handler and ends the thread."""


# ---------------------------------------------------------------------------
# The seven silent freezes of audit F
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fault", [enospc, cuda_error], ids=["1-ckpt-enospc", "2-ckpt-cuda"])
def test_checkpoint_failure_keeps_stepping_visibly_degraded(tmp_path, runners, monkeypatch, fault):
    runner = started(runners, make_graph_runner(tmp_path))
    calls = []

    def failing(*a, **k):
        calls.append(1)
        fault()
    monkeypatch.setattr(experiment_registry, "atomic_write_bytes", failing)
    force_checkpoint_due(runner)
    assert wait_for(lambda: calls), "checkpoint was never attempted"
    assert wait_for(lambda: status(runner)["status"] == "degraded")
    # The thread lives and the run keeps stepping: degraded, not frozen.
    assert runner.sim_thread.is_alive()
    assert advancing(runner, 1.0) > 5
    st = status(runner)
    assert st["status"] == "degraded" and st["halted"] is False and st["error"] is None
    assert st["liveness"]["state"] == "advancing"
    failing_ckpt = st["persistence"]["failing"]["checkpoint"]
    assert failing_ckpt["failures"] == 1 and failing_ckpt["backoff_s"] == 30.0
    if fault is enospc:
        assert failing_ckpt["errno"] == errno.ENOSPC and st["persistence"]["reason"] == "disk full"
    # Back-off: no second 17 MB write attempt on the next step (or the next second).
    attempts = len(calls)
    time.sleep(0.5)
    assert len(calls) == attempts
    # The disk is free again and the retry is due: saving resumes, status online.
    monkeypatch.setattr(experiment_registry, "atomic_write_bytes", ORIGINAL_ATOMIC_WRITE)
    runner.persistence.channels["checkpoint"]["next_retry_at"] = 0.0
    force_checkpoint_due(runner)
    assert wait_for(lambda: status(runner)["status"] == "online")
    assert status(runner)["persistence"]["last_ok_save_at"] is not None



def test_3_brain_save_disk_full_keeps_stepping(tmp_path, runners, monkeypatch):
    runner = started(runners, make_runner(tmp_path))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    force_checkpoint_due(runner)
    assert wait_for(lambda: status(runner)["status"] == "degraded")
    assert runner.sim_thread.is_alive() and advancing(runner) > 5
    assert "checkpoint" in status(runner)["persistence"]["failing"]


def test_4_trial_ledger_disk_full_keeps_stepping(tmp_path, runners, monkeypatch):
    runner = started(runners, make_runner(tmp_path, continuous=False, trial_length_s=0.1))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "log", enospc)
    trials = len(runner.trial_history)
    assert wait_for(lambda: "trial_ledger" in status(runner)["persistence"]["failing"])
    st = status(runner)
    assert st["status"] == "degraded" and runner.sim_thread.is_alive()
    # Trials keep ending (in memory) and the run keeps stepping.
    assert wait_for(lambda: len(runner.trial_history) > trials + 2)
    assert advancing(runner) > 5


def test_5_failing_record_stops_visibly_and_the_run_continues(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    with runner.lock:
        runner.start_recording(name="freeze-test")
    started(runners, runner)
    monkeypatch.setattr(recording.RunRecorder, "capture", enospc)
    assert wait_for(lambda: runner.recorder is None)
    st = status(runner)
    assert st["recording"] is None
    assert st["recording_error"]["recording"] == "freeze-test.nfrec"
    assert "No space left" in st["recording_error"]["error"]
    assert st["status"] == "degraded" and "recording" in st["persistence"]["failing"]
    assert runner.sim_thread.is_alive() and advancing(runner) > 5
    # The partial file is kept as evidence, never deleted.
    assert list((tmp_path / "recordings").glob(".freeze-test.nfrec.partial"))


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
    ack = runner.dispatch_command({"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok" and ack["ack"]["halted_by_error"] is None
    assert wait_for(lambda: runner.total_steps > steps + 5)
    assert status(runner)["status"] == "online"


def test_7_step_error_while_disk_full_then_reselect_assay(tmp_path, runners, monkeypatch):
    """The audit's worst case: the halt bookkeeping write fails (disk full), the thread
    died, and re-selecting the assay CLEARED the halt with nothing stepping."""
    runner = started(runners, make_runner(tmp_path))
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "log", enospc)      # disk stays full
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    original = Arena.step
    fired = []

    def step_once_fails(self, dt):
        if not fired:
            fired.append(1)
            raise RuntimeError("injected fault (arena step)")
        return original(self, dt)
    monkeypatch.setattr(Arena, "step", step_once_fails)
    assert wait_for(lambda: runner.last_error is not None)
    st = status(runner)
    assert st["status"] == "error" and st["halted"] is True
    assert runner.sim_thread.is_alive(), "the halt path's own write killed the thread"
    assert "events_ledger" in st["persistence"]["failing"]          # the failed write is reported
    # Recovery with the disk STILL full: the documented action resumes stepping.
    steps = runner.total_steps
    ack = runner.dispatch_command({"action": "switch_paradigm", "paradigm": "t-maze"})
    assert ack["status"] == "ok", ack
    assert wait_for(lambda: runner.total_steps > steps + 5), "re-select answered ok but nothing steps"
    st = status(runner)
    assert st["status"] == "degraded" and st["halted"] is False   # stepping, visibly not saving
    assert st["liveness"]["state"] == "advancing"


# ---------------------------------------------------------------------------
# F1: the thread cannot die silently
# ---------------------------------------------------------------------------
def test_dead_thread_is_reported_within_3_s_and_a_rebuild_restarts_it(tmp_path, runners):
    runner = started(runners, make_runner(tmp_path))

    def kill(_runner):
        runner.step_hook = None
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
    # Nothing is applied inline while the loop should be running...
    refused = runner.dispatch_command({"action": "set_speed", "speed": 5})
    assert refused["status"] == "error" and refused["loop_dead"] is True
    # ...but a rebuild restarts the thread and the ack says so.
    steps = runner.total_steps
    ack = runner.dispatch_command({"action": "switch_paradigm", "paradigm": "y-maze"})
    assert ack["status"] == "ok" and ack["loop_restarted"] is True and ack["ack"]["applied"] is True
    assert wait_for(lambda: runner.total_steps > steps + 5)
    assert status(runner)["status"] == "online"


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
# F2 options
# ---------------------------------------------------------------------------
def test_halt_on_persistence_failure_halts_honestly(tmp_path, runners, monkeypatch):
    runner = make_runner(tmp_path)
    runner.halt_on_persistence_failure = True
    started(runners, runner)
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", enospc)
    force_checkpoint_due(runner)
    assert wait_for(lambda: runner.last_error is not None)
    st = status(runner)
    assert st["status"] == "error" and st["error_detail"]["phase"] == "persistence"
    assert st["error"].startswith("saving failed: OSError")
    assert runner.sim_thread.is_alive()


def test_low_disk_skips_the_checkpoint_before_enospc(tmp_path, runners, monkeypatch):
    import shutil
    runner = started(runners, make_runner(tmp_path))
    writes = []
    monkeypatch.setattr(experiment_brains.ExperimentBrain, "save", lambda self: writes.append(1))
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 10**8, 10**8))
    force_checkpoint_due(runner)
    assert wait_for(lambda: "checkpoint" in status(runner)["persistence"]["failing"])
    entry = status(runner)["persistence"]["failing"]["checkpoint"]
    assert entry["state"] == "disk_low" and status(runner)["persistence"]["state"] == "disk_low"
    assert writes == []                                        # nothing was attempted
    assert status(runner)["status"] == "degraded" and advancing(runner) > 5


def test_learning_recorder_errors_are_in_status(tmp_path):
    from learning_recorder import LearningRecorder, RecorderThread
    runner = make_runner(tmp_path)
    rec = LearningRecorder(tmp_path / "learning")
    thread = RecorderThread(runner, rec, poll_interval=0.05)
    runner.learning_records = thread
    rec.record_summary = enospc
    thread.start()
    try:
        assert wait_for(lambda: thread.describe()["failing"])
        st = status(runner)
        assert st["status"] == "degraded"
        assert "No space left" in st["persistence"]["failing"]["learning_records"]["error"]
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
def test_stop_runs_once_and_a_failed_final_checkpoint_is_reported(tmp_path, monkeypatch):
    runner = make_runner(tmp_path)
    runner.start()
    assert wait_for(lambda: runner.total_steps > 2)
    assert runner.stop() is True
    assert runner.stop() is True                               # second call: no-op
    finals = list((tmp_path / "checkpoints").glob("checkpoint_t-maze_final_shutdown_*.json"))
    assert len(finals) == 1

    other = make_runner(tmp_path / "b")
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
    assert args.halt_on_persistence_failure is False and args.exit_on_stall is None
