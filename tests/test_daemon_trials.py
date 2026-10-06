"""Producer-authoritative trial advancement in the continuous daemon."""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neurofly_daemon import ContinuousExperimentRunner  # noqa: E402
from learning_recorder import LearningRecorder, RecorderThread  # noqa: E402


@pytest.fixture
def make_runner(tmp_path):
    resources = []
    def _make(paradigm: str, trial_length_s: float = 60.0) -> ContinuousExperimentRunner:
        active = ContinuousExperimentRunner(
            initial_paradigm=paradigm,
            sim_speed=100.0,
            checkpoint_interval=3600.0,
            output_dir=tmp_path / "daemon",
            trial_length_s=trial_length_s,
        )
        recorder = LearningRecorder(tmp_path / f"records-{len(resources)}",
                                    session={"daemon_run_id": active.run_id})
        drain = RecorderThread(active, recorder, summary_interval=999)
        with active.lock:
            active.attach_learning_records(drain)
        resources.append(recorder)
        active._test_observation_drain = drain
        return active
    yield _make
    for recorder in resources:
        recorder.close()


def advance(runner, steps):
    """Advance actual ticks and synchronously drain each immutable terminal."""
    for _ in range(steps):
        runner.step_once()
        if runner._observation_terminal is not None \
                and runner._observation_terminal["durable"] is None:
            runner._test_observation_drain.poll_once()


def test_multisensory_sandbox_completes_time_based_trials(make_runner):
    runner = make_runner("multisensory-sandbox", trial_length_s=4.0)
    steps_per_trial = int(round(4.0 / runner.dt))
    for _ in range(steps_per_trial * 3 + 5):
        advance(runner, 1)

    assert len(runner.trial_history) == 3
    assert runner.current_trial == 4
    assert all(t["reason"] == "window_elapsed" for t in runner.trial_history)
    assert [t["trial"] for t in runner.trial_history] == [1, 2, 3]
    assert len(runner.learning_curve) == 3
    assert all(0.0 <= v <= 1.0 for v in runner.learning_curve)   # composite score scaled to [0, 1]
    assert runner.latest_telemetry["trials_completed"] == 3
    assert runner.latest_telemetry["trial"] == 4
    # The paradigm's own trial counter advanced with the daemon's
    assert runner.arena.paradigm.trial_manager.trial_number == 4


def test_legacy_paradigm_max_duration_does_not_override_observation_window(make_runner):
    runner = make_runner("multisensory-sandbox", trial_length_s=600.0)
    max_steps = runner.arena.paradigm.trial_manager.max_duration_steps   # 1500 steps = 30 s
    for _ in range(max_steps + 2):
        advance(runner, 1)
    assert runner.trial_history == []
    assert runner.observation_lifecycle_status()["phase"] == "observing"


def test_natural_endpoint_ends_trial_and_respawns_fly(make_runner):
    runner = make_runner("labyrinth", trial_length_s=600.0)
    fly = runner.arena.fly
    fly.pos.x, fly.pos.y = 130.0, 85.0          # inside the goal chamber
    runner.step_once()
    assert runner._test_observation_drain.poll_once()["observations"] == 1
    for _ in range(110):
        if runner._observation_terminal is None:
            break
        runner.step_once()
    assert len(runner.trial_history) == 1
    assert runner.trial_history[0]["reason"] == "terminal_event:goal_entry"
    # Fly is back at the labyrinth entrance for the next trial
    assert (fly.pos.x, fly.pos.y) == (15.0, 15.0)
    assert runner.arena.paradigm.goal_reached is False
    assert runner.trial_sim_time == 0.0


def test_memory_is_kept_across_trials(make_runner):
    runner = make_runner("multisensory-sandbox", trial_length_s=2.0)
    circuit = runner.arena.fly.circuit
    for _ in range(int(2.0 / runner.dt) + 1):
        advance(runner, 1)
    assert len(runner.trial_history) == 1
    assert runner.arena.fly.circuit is circuit


def test_reset_trial_command_uses_paradigm_spawn(make_runner):
    runner = make_runner("multisensory-sandbox")
    runner.arena.fly.pos.x, runner.arena.fly.pos.y = 50.0, -30.0
    from tests.transition_control_helpers import transition_command
    res = transition_command(runner, {"action": "reset_trial"})
    assert res["status"] == "ok"
    assert (runner.arena.fly.pos.x, runner.arena.fly.pos.y) == (0.0, 0.0)   # not (width/2, height/2)
