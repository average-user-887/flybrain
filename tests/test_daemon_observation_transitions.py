import json
import sys
import threading
import time

import pytest

from learning_recorder import LearningRecorder, RecorderThread
from neurofly_daemon import ContinuousExperimentRunner, run_daemon
from observation_envelopes import dumps_observation_envelope
from observation_publication import ObservationPublicationError


def instrumented(tmp_path, paradigm, *, window=0.025, **runner_kwargs):
    active = ContinuousExperimentRunner(
        initial_paradigm=paradigm, trial_length_s=window,
        output_dir=tmp_path / "runner", checkpoint_interval=3600, **runner_kwargs)
    recorder = LearningRecorder(tmp_path / "records", session={"daemon_run_id": active.run_id})
    drain = RecorderThread(active, recorder, summary_interval=999)
    with active.lock:
        active.attach_learning_records(drain)
    return active, recorder, drain


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_nonaligned_terminal_waits_for_exact_durable_ack_then_resets_segment(tmp_path):
    active, recorder, drain = instrumented(tmp_path, "t-maze")
    first_segment = active.segment_id
    active.step_once()
    active.step_once()
    terminal = active._observation_terminal
    frozen = dumps_observation_envelope(terminal["observation"], sort_keys=True)
    pose = dict(terminal["observation"]["terminal_pose_post_step"])

    assert terminal["observation"]["measurement_end_rel_s"] == 0.025
    assert pose["step"] == active.total_steps == 2
    assert active.observation_lifecycle_status()["phase"] == "waiting_for_save"
    active.step_once()
    assert active.total_steps == 2 and active.segment_id == first_segment

    assert drain.poll_once()["observations"] == 1
    assert active.segment_id != first_segment
    assert active.current_trial == 2
    durable = active.observation_publication_status()["last_terminal"]
    assert dumps_observation_envelope(durable["observation"], sort_keys=True) == frozen
    assert durable["observation"]["terminal_pose_post_step"] == pose
    assert active.observation_lifecycle_status()["phase"] == "observing"
    assert active.observation_segment_start_sim_s == 0.04
    recorder.close()


def test_presentation_ack_advances_in_place_without_pose_or_segment_reset(tmp_path):
    active, recorder, drain = instrumented(tmp_path, "looming-escape")
    active.step_once(); active.step_once()
    segment = active.segment_id
    pose = (active.arena.fly.pos.x, active.arena.fly.pos.y, active.arena.fly.heading)
    assert drain.poll_once()["observations"] == 1
    assert active.segment_id == segment
    assert (active.arena.fly.pos.x, active.arena.fly.pos.y, active.arena.fly.heading) == pose
    assert active.arena.observation_owner.observation_status()["presentation_index"] == 1
    assert active.current_trial == 1
    recorder.close()


def test_terminal_event_hold_ticks_advance_with_frozen_bytes_then_wait_for_save(tmp_path, monkeypatch):
    active, recorder, drain = instrumented(tmp_path, "wind-tunnel", window=120.0)
    owner = active.arena.observation_owner
    freeze = owner.freeze_observation
    calls = []
    def counted_freeze(*args, **kwargs):
        calls.append(active.total_steps)
        return freeze(*args, **kwargs)
    monkeypatch.setattr(owner, "freeze_observation", counted_freeze)
    # A real source-entry event closes this producer and starts its declared 1 s hold.
    active.arena.fly.pos.x = 180.0
    active.arena.fly.pos.y = 30.0
    active.step_once()
    terminal = active._observation_terminal
    frozen = dumps_observation_envelope(terminal["observation"], sort_keys=True)
    segment = active.segment_id
    assert active.observation_lifecycle_status()["phase"] == "hold"

    while active.arena.observation_owner.observation_status()["hold_remaining_s"] > 0:
        before = active.total_steps
        active.step_once()
        assert active.total_steps == before + 1
        assert active.segment_id == segment
        assert dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True) == frozen
    assert calls == [1]

    assert active.observation_lifecycle_status()["phase"] == "waiting_for_save"
    stopped = active.total_steps
    active.step_once()
    assert active.total_steps == stopped
    drain.poll_once()
    assert active.segment_id != segment
    recorder.close()


def test_wrong_receipt_is_refused_and_cannot_transition(tmp_path):
    active, recorder, _ = instrumented(tmp_path, "t-maze")
    active.step_once(); active.step_once()
    with active.lock:
        claim = active.claim_terminal_observation()
        wrong = {"durable": True, "observation_key": dict(claim["observation_key"]),
                 "payload_sha256": claim["payload_sha256"], "file": "observations.jsonl",
                 "line": 1, "offset": 0, "idempotent": False}
        wrong["observation_key"]["segment_id"] = "unrelated"
        with pytest.raises(ObservationPublicationError, match="does not match"):
            active.terminal_observation_succeeded(claim["attempt_token"], wrong)
        active.terminal_observation_failed(claim["attempt_token"], RuntimeError("wrong receipt"))
    assert active._observation_terminal["durable"] is None
    assert active.current_trial == 1
    recorder.close()


def test_late_ack_after_required_save_failure_does_not_lift_halt(tmp_path):
    active, recorder, _ = instrumented(tmp_path, "t-maze")
    active.step_once(); active.step_once()
    segment = active.segment_id
    with active.lock:
        first = active.claim_terminal_observation()
        active.terminal_observation_failed(first["attempt_token"], OSError("injected save failure"))
        active.observation_publication.retry_failed()
        retry = active.claim_terminal_observation()
    receipt = recorder.record_observation(retry["observation"])
    with active.lock:
        active.terminal_observation_succeeded(retry["attempt_token"], receipt)
    assert active.last_error is not None
    assert active.segment_id == segment and active.current_trial == 1
    assert active.observation_lifecycle_status()["phase"] == "durable_transition_blocked"
    recorder.close()


def test_recorder_ownership_failure_prevents_simulator_start(tmp_path, monkeypatch):
    data_dir = tmp_path / "owned-records"
    owner = LearningRecorder(data_dir, session={"owner": "test"})
    starts = []
    monkeypatch.setattr(ContinuousExperimentRunner, "start", lambda active: starts.append(active))
    pid_file = tmp_path / "daemon.pid"
    monkeypatch.setattr(sys, "argv", [
        "neurofly_daemon.py", "--backend", "modular", "--paradigm", "t-maze",
        "--output-dir", str(tmp_path / "runner"), "--data-dir", str(data_dir),
        "--pid-file", str(pid_file), "--port", "0",
    ])
    try:
        with pytest.raises(SystemExit) as stopped:
            run_daemon()
        assert stopped.value.code == 2
        assert starts == []
        assert not pid_file.exists()
    finally:
        owner.close()


def test_started_recorder_death_halts_and_retains_pending_terminal(tmp_path, monkeypatch):
    active, recorder, drain = instrumented(tmp_path, "t-maze")
    active.recorder_dead_grace_s = 0.03
    active.watchdog_interval_s = 0.005
    monkeypatch.setattr(drain, "run", lambda: None)
    drain.start()
    drain.join(timeout=1)
    assert drain.liveness()["started"] is True and drain.is_alive() is False

    active.start()
    try:
        assert wait_for(lambda: active._observation_terminal is not None)
        frozen = dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True)
        assert wait_for(lambda: active.last_error is not None)
        assert "recorder thread exited" in active.last_error
        assert active.health()["status"] == "error"
        assert active.result_validity()["state"] == "incomplete"
        assert active.observation_publication.pending_status()["pending_count"] == 1
        assert dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True) == frozen
        assert active.observation_lifecycle_status()["phase"] == "durable_transition_blocked"
    finally:
        active.stop()
        recorder.close()


def test_alive_recorder_that_never_enters_poll_loop_halts(tmp_path, monkeypatch):
    active, recorder, drain = instrumented(tmp_path, "t-maze")
    active.recorder_progress_grace_s = 0.03
    active.watchdog_interval_s = 0.005
    release = threading.Event()
    monkeypatch.setattr(drain, "run", lambda: release.wait(2))
    drain.start()
    active.start()
    try:
        assert drain.is_alive() and drain.liveness()["run_entered"] is False
        assert wait_for(lambda: active.last_error is not None)
        assert "did not enter its poll loop" in active.last_error
        assert active.health()["status"] == "error"
        assert active.result_validity()["state"] == "incomplete"
    finally:
        release.set()
        drain.join(timeout=1)
        active.stop()
        recorder.close()


def test_exploratory_dead_recorder_records_one_episode_not_every_watchdog_tick(tmp_path, monkeypatch):
    active, recorder, drain = instrumented(tmp_path, "t-maze", exploratory=True)
    active.recorder_dead_grace_s = 0.02
    active.watchdog_interval_s = 0.005
    monkeypatch.setattr(drain, "run", lambda: None)
    drain.start(); drain.join(timeout=1)
    active.start()
    try:
        assert wait_for(lambda: len(active._recorder_watchdog_failures) == 1)
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(active._recorder_watchdog_failures) == 1
        assert active._recorder_watchdog_failure["active"] is True
        assert active.last_error is None
        assert active.result_validity()["state"] == "incomplete"
    finally:
        active.stop()
        recorder.close()


def test_error_clear_rechecks_recorder_health_at_commit_boundary(tmp_path, monkeypatch):
    active, recorder, _ = instrumented(tmp_path, "t-maze")
    episode = {"episode": 1, "active": True, "error": "fixture"}
    active._recorder_watchdog_failure = episode
    active._recorder_watchdog_failures.append(episode)
    active.last_error = "recorder failed"
    active.error_detail = {"message": active.last_error, "channel": None}
    healthy = iter((True, False))
    monkeypatch.setattr(active, "_learning_recorder_problem", lambda: (None, {"fixture": True}))
    monkeypatch.setattr(active, "_learning_recorder_verified_healthy", lambda state=None: next(healthy))

    before = active.error_detail
    assert active._recovery_gate(before, active._stopping_failures) is None
    assert active._clear_error("race_fixture") is None
    assert active.last_error == "recorder failed"
    assert active.error_detail is before
    assert episode["active"] is True
    recorder.close()


def test_attached_manual_recorder_is_not_mistaken_for_dead_background_thread(tmp_path):
    active, recorder, _ = instrumented(tmp_path, "t-maze")
    active.recorder_dead_grace_s = 0.01
    active.watchdog_interval_s = 0.005
    active.start()
    try:
        assert wait_for(lambda: active._observation_terminal is not None)
        deadline = time.monotonic() + 0.04
        while time.monotonic() < deadline:
            time.sleep(0.005)
        assert active.last_error is None
        assert active.liveness()["state"] == "waiting_for_save"
        assert active.learning_records.liveness()["started"] is False
    finally:
        active.stop()
        recorder.close()


def test_stuck_recorder_write_has_grace_then_halts_without_stealing_owner(tmp_path, monkeypatch):
    active, recorder, drain = instrumented(tmp_path, "t-maze")
    active.recorder_stuck_s = 0.12
    active.recorder_dead_grace_s = 0.03
    active.watchdog_interval_s = 0.005
    entered, release = threading.Event(), threading.Event()
    original = recorder.record_observation

    def blocked(payload):
        entered.set()
        assert release.wait(2)
        return original(payload)

    monkeypatch.setattr(recorder, "record_observation", blocked)
    drain.poll_interval = 0.01
    drain.start()
    active.start()
    segment = active.segment_id
    try:
        assert entered.wait(2)
        frozen = dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True)
        assert drain.liveness()["active_write"] == "observation"
        assert active.last_error is None                 # ordinary in-progress grace
        assert active.lock.acquire(timeout=0.1)          # writer owns no runner lock
        active.lock.release()
        assert wait_for(lambda: active.last_error is not None)
        assert "write has not completed" in active.last_error
        assert active.health()["status"] == "error"
        assert active.result_validity()["state"] == "incomplete"
        assert drain.is_alive() and drain.liveness()["active_write"] == "observation"

        release.set()
        assert wait_for(lambda: active._observation_terminal.get("durable") is not None)
        assert active.last_error is not None             # a late ACK cannot resume science
        assert active.segment_id == segment
        assert dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True) == frozen
        assert active.observation_lifecycle_status()["phase"] == "durable_transition_blocked"

        # The late ACK does not clear the halt. Explicit recovery rearms only
        # after the same recorder has demonstrated healthy polling again.
        assert active._recorder_watchdog_failure["active"] is True
        with active.lock:
            before = active.error_detail
            assert active._recovery_gate(before, active._stopping_failures) is None
            assert active._clear_error("test_verified_recorder_recovery") is not None
        assert active._recorder_watchdog_failure["active"] is False
        assert active._recorder_watchdog_failure["recovered_by"] == "test_verified_recorder_recovery"

        drain._stop_event.set()
        drain.join(timeout=1)
        assert wait_for(lambda: active.last_error is not None)
        assert "recorder thread exited" in active.last_error
        assert [entry["episode"] for entry in active._recorder_watchdog_failures] == [1, 2]
        assert active._recorder_watchdog_failures[0]["active"] is False
        assert active._recorder_watchdog_failures[1]["active"] is True
    finally:
        release.set()
        active.stop()
        drain.stop(timeout=1)


def test_reset_failure_after_exact_ack_retains_confirmed_terminal_and_halts(tmp_path, monkeypatch):
    active, recorder, drain = instrumented(tmp_path, "t-maze")
    active.step_once(); active.step_once()
    segment, trial = active.segment_id, active.current_trial
    frozen = dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True)

    def broken_reset():
        raise RuntimeError("injected physical reset failure")

    monkeypatch.setattr(active.arena.paradigm, "reset_trial", broken_reset)
    assert drain.poll_once()["observations"] == 1
    assert active.last_error is not None
    assert active._state_uncertain is True
    assert active.segment_id == segment and active.current_trial == trial
    assert active.trial_history == []
    assert active._observation_terminal["durable"] is not None
    assert dumps_observation_envelope(active._observation_terminal["observation"], sort_keys=True) == frozen
    assert dumps_observation_envelope(
        active.observation_publication_status()["last_terminal"]["observation"], sort_keys=True) == frozen
    assert active.result_validity()["state"] == "incomplete"
    assert active.observation_lifecycle_status()["phase"] == "durable_transition_blocked"
    recorder.close()


def test_sim_watchdog_exit_check_is_not_blocked_by_runner_lock(tmp_path):
    active = ContinuousExperimentRunner(
        initial_paradigm="t-maze", output_dir=tmp_path / "runner",
        checkpoint_interval=3600)
    active.watchdog_interval_s = 0.005
    active.step_hard_limit_s = 0.01
    active.exit_on_stall_s = 0.0
    exit_called = threading.Event()
    active._exit = lambda code: exit_called.set()

    active.lock.acquire()
    try:
        active.start()
        active._step_started = time.perf_counter() - 1.0
        assert exit_called.wait(1), "watchdog did not evaluate/exit while runner.lock was held"
        assert active.watchdog_thread.is_alive()
    finally:
        active._step_started = None
        active.running = False
        active._wake.set()
        active._watchdog_stop.set()
        active.lock.release()
        active.stop()
