import copy
import threading
from types import SimpleNamespace

import pytest

import maze
from learning_recorder import LearningRecorder, RecorderOwnershipError, RecorderThread
from neurofly_daemon import ContinuousExperimentRunner
from observation_envelopes import build_observation_envelope
from observation_publication import ObservationPublicationError, ObservationPublicationQueue, ObservationQueueFull


def terminal_for(runner, *, producer_name="t_maze", segment_id=None, presentation_suffix=None):
    producer = maze.ExperimentRegistry.get(producer_name)
    fly = SimpleNamespace(
        pos=SimpleNamespace(x=30.0, y=30.0), heading=0.0, speed=1.0,
        angular_velocity=0.0, assay_escape_remaining=0.0, behavioral_state="SURGE")
    producer.step(fly, 0.02)
    snapshot = producer.freeze_observation("manual_reset")
    if presentation_suffix is not None:
        snapshot["presentation_index"] = presentation_suffix
    identity = runner.identity()
    identity.update(brain_id=runner.active_brain.brain_id,
                    graph_io_extension={"version": None, "sha256": None})
    return build_observation_envelope(
        snapshot, identity=identity,
        provenance={
            "gf_source": "geometric", "stimulus_entry_stage": None,
            "motor_assists_enabled": True, "controller_states": ["SURGE", "CAST"],
        },
        segment_id=segment_id or runner.segment_id,
        segment_start_sim_s=100.0, validity="valid",
        terminal_pose_post_step={"x_mm": 30.0, "y_mm": 30.0, "heading_rad": 0.0})


def attached(tmp_path, *, exploratory=False, poll_interval=1.0):
    runner = ContinuousExperimentRunner(
        initial_paradigm="t-maze", output_dir=tmp_path / "run", exploratory=exploratory)
    recorder = LearningRecorder(tmp_path / "records", fsync=False)
    thread = RecorderThread(runner, recorder, poll_interval=poll_interval, summary_interval=999)
    with runner.lock:
        runner.attach_learning_records(thread)
    return runner, recorder, thread


def test_actual_runner_thread_publishes_once_and_exposes_exact_owner(tmp_path):
    runner, recorder, thread = attached(tmp_path)
    envelope = terminal_for(runner)
    with runner.lock:
        runner.enqueue_terminal_observation(envelope)
    result = thread.poll_once()
    assert result["observations"] == 1
    with runner.lock:
        status = runner.observation_publication_status()
    assert status["durability"]["enabled"] is True
    assert status["durability"]["state"] == "enabled"
    assert status["durability"]["reason"] is None
    assert status["durability"]["recorder"]["started"] is False
    assert status["durability"]["watchdog_failure"] is None
    assert status["durability"]["watchdog_failures"] == []
    assert status["pending"]["pending_count"] == 0
    assert status["last_terminal"]["observation"] == envelope
    assert thread.poll_once().get("observations", 0) == 0
    assert thread.stop()


def test_lost_ack_retries_identical_payload_and_preserves_scientific_halt(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path)
    envelope = terminal_for(runner)
    with runner.lock:
        runner.enqueue_terminal_observation(envelope)
    original = recorder.record_observation
    calls = []

    def lose_ack(payload):
        calls.append(copy.deepcopy(payload))
        original(payload)
        raise OSError("lost directory acknowledgement")

    monkeypatch.setattr(recorder, "record_observation", lose_ack)
    with pytest.raises(OSError, match="lost directory"):
        thread.poll_once()
    with runner.lock:
        failed = runner.observation_publication_status()
        assert failed["pending"]["entries"][0]["phase"] == "failed"
        runner.persistence.channels["learning_records"]["next_retry_at"] = 0
    halted = runner.last_error
    assert halted and runner.health()["status"] == "error"

    monkeypatch.setattr(recorder, "record_observation", original)
    assert thread.poll_once()["observations"] == 1
    with runner.lock:
        durable = runner.observation_publication_status()["last_terminal"]
    assert durable["observation"] == envelope
    assert durable["receipt"]["idempotent"] is True
    assert calls == [envelope]
    assert runner.last_error == halted
    assert thread.stop()


def test_exploratory_failure_is_degraded_and_retains_retryable_head(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path, exploratory=True)
    with runner.lock:
        runner.enqueue_terminal_observation(terminal_for(runner))

    def fail(_payload):
        raise OSError(28, "disk full")

    monkeypatch.setattr(recorder, "record_observation", fail)
    with pytest.raises(OSError):
        thread.poll_once()
    assert runner.last_error is None
    assert runner.health()["status"] == "degraded"
    with runner.lock:
        status = runner.observation_publication_status()
    assert status["pending"]["entries"][0]["phase"] == "failed"
    assert status["last_terminal"] is None
    assert thread.stop() is False


def test_wrong_receipt_is_visible_and_does_not_publish_or_drop(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path)
    with runner.lock:
        runner.enqueue_terminal_observation(terminal_for(runner))
    original = recorder.record_observation

    def wrong(payload):
        receipt = original(payload)
        receipt["payload_sha256"] = "0" * 64
        return receipt

    monkeypatch.setattr(recorder, "record_observation", wrong)
    with pytest.raises(ObservationPublicationError, match="digest"):
        thread.poll_once()
    with runner.lock:
        status = runner.observation_publication_status()
    assert status["pending"]["pending_count"] == 1
    assert status["pending"]["entries"][0]["phase"] == "failed"
    assert status["last_terminal"] is None
    assert runner.last_error
    assert thread.stop() is False


def test_blocked_writer_leaves_runner_lock_available_and_owner_open(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path, poll_interval=0.05)
    with runner.lock:
        runner.enqueue_terminal_observation(terminal_for(runner))
    entered, release = threading.Event(), threading.Event()
    original = recorder.record_observation

    def blocked(payload):
        entered.set()
        assert release.wait(5)
        return original(payload)

    monkeypatch.setattr(recorder, "record_observation", blocked)
    thread.start()
    assert entered.wait(2)
    assert runner.lock.acquire(timeout=0.2)
    runner.lock.release()
    assert thread.stop(timeout=0.01) is False
    assert thread.describe()["shutdown_blocked"] is True
    with pytest.raises(RecorderOwnershipError):
        LearningRecorder(tmp_path / "records", fsync=False)
    release.set()
    thread.join(timeout=2)
    assert thread.stop(timeout=1.0) is True


def test_no_record_and_capacity_refuse_without_dropping(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=tmp_path / "run")
    envelope = terminal_for(runner)
    with runner.lock:
        disabled = runner.observation_publication_status()
        with pytest.raises(ObservationPublicationError, match="disabled"):
            runner.enqueue_terminal_observation(envelope)
        assert runner.observation_publication.pending_status()["pending_count"] == 0
    assert disabled["durability"]["state"] == "disabled"

    recorder = LearningRecorder(tmp_path / "records", fsync=False)
    thread = RecorderThread(runner, recorder)
    with runner.lock:
        runner.attach_learning_records(thread)
        runner.observation_publication = ObservationPublicationQueue(capacity=1)
        runner.enqueue_terminal_observation(envelope)
        other = terminal_for(runner, segment_id="segment-other")
        with pytest.raises(ObservationQueueFull):
            runner.enqueue_terminal_observation(other)
        assert runner.observation_publication.pending_status()["pending_count"] == 1
    recorder.close()


def test_summary_failure_does_not_orphan_claimed_observation(tmp_path, monkeypatch):
    import learning_recorder
    runner, recorder, thread = attached(tmp_path)
    with runner.lock:
        runner.enqueue_terminal_observation(terminal_for(runner))
    original = learning_recorder.summarise_runner

    def unavailable_summary(_runner):
        raise RuntimeError("diagnostic summary unavailable")

    monkeypatch.setattr(learning_recorder, "summarise_runner", unavailable_summary)
    with pytest.raises(RuntimeError, match="diagnostic summary unavailable"):
        thread.poll_once(force_summary=True)
    monkeypatch.setattr(learning_recorder, "summarise_runner", original)
    assert thread.poll_once()["observations"] == 1
    assert thread.stop()


def test_failed_final_flush_remains_failed_on_repeated_stop(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path)
    with runner.lock:
        runner.enqueue_terminal_observation(terminal_for(runner))

    def disk_full(_payload):
        raise OSError(28, "disk full")

    monkeypatch.setattr(recorder, "record_observation", disk_full)
    assert thread.stop() is False
    assert thread.stop() is False


def test_retry_backoff_does_not_report_recorder_recovered(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path, poll_interval=0.05)
    with runner.lock:
        runner.enqueue_terminal_observation(terminal_for(runner))

    def disk_full(_payload):
        raise OSError(28, "disk full")

    monkeypatch.setattr(recorder, "record_observation", disk_full)
    with pytest.raises(OSError):
        thread.poll_once()
    # Reproduce the thread's failure bookkeeping and its next backoff cycle.
    thread._note_error(OSError(28, "disk full"))
    first_failure = thread.describe()["failing_since"]
    cycles = 0
    original_poll = thread.poll_once

    def one_backoff_cycle():
        nonlocal cycles
        cycles += 1
        result = original_poll()
        thread._stop_event.set()
        return result

    monkeypatch.setattr(thread, "poll_once", one_backoff_cycle)
    thread.run()
    assert cycles == 1
    assert thread.describe()["failing"] is True
    assert thread.describe()["failing_since"] == first_failure
    # Avoid the monkeypatch's narrowed poll signature during cleanup.
    monkeypatch.setattr(thread, "poll_once", original_poll)
    assert thread.stop() is False


def test_success_callback_cannot_read_queue_without_runner_lock(tmp_path, monkeypatch):
    runner, recorder, thread = attached(tmp_path)
    actual_lock = runner.lock
    monkeypatch.setattr(runner, "lock", SimpleNamespace(acquire=lambda **_kwargs: False))

    def unlocked_read():
        pytest.fail("publication queue read without runner lock")

    monkeypatch.setattr(runner.observation_publication, "pending_status", unlocked_read)
    runner.records_ok("learning_records")
    monkeypatch.setattr(runner, "lock", actual_lock)
    recorder.close()
