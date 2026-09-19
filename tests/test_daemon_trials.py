"""Trial advancement in the continuous daemon.

Before this, ``trials_completed`` stayed at 0 forever: the runner waited for a
``trial_complete`` metric no paradigm ever publishes. Trials now end on a paradigm's
natural endpoint, its own TrialManager duration, or ``trial_length_s`` of simulated
time, and the fly is respawned with its memory intact.
"""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neurofly_daemon import ContinuousExperimentRunner  # noqa: E402


@pytest.fixture
def make_runner(tmp_path):
    def _make(paradigm: str, trial_length_s: float = 60.0) -> ContinuousExperimentRunner:
        return ContinuousExperimentRunner(
            initial_paradigm=paradigm,
            sim_speed=100.0,
            checkpoint_interval=3600.0,
            output_dir=tmp_path / "daemon",
            trial_length_s=trial_length_s,
        )
    return _make


def test_multisensory_sandbox_completes_time_based_trials(make_runner):
    runner = make_runner("multisensory-sandbox", trial_length_s=4.0)
    steps_per_trial = int(round(4.0 / runner.dt))
    for _ in range(steps_per_trial * 3 + 5):
        runner.step_once()

    assert len(runner.trial_history) == 3
    assert runner.current_trial == 4
    assert all(t["reason"] == "time_limit" for t in runner.trial_history)
    assert [t["trial"] for t in runner.trial_history] == [1, 2, 3]
    assert len(runner.learning_curve) == 3
    assert all(0.0 <= v <= 1.0 for v in runner.learning_curve)   # composite score scaled to [0, 1]
    assert runner.latest_telemetry["trials_completed"] == 3
    assert runner.latest_telemetry["trial"] == 4
    # The paradigm's own trial counter advanced with the daemon's
    assert runner.arena.paradigm.trial_manager.trial_number == 4


def test_paradigm_max_duration_ends_trial_before_time_limit(make_runner):
    runner = make_runner("multisensory-sandbox", trial_length_s=600.0)
    max_steps = runner.arena.paradigm.trial_manager.max_duration_steps   # 1500 steps = 30 s
    for _ in range(max_steps + 2):
        runner.step_once()
    assert len(runner.trial_history) == 1
    assert runner.trial_history[0]["reason"] == "max_duration_steps"


def test_natural_endpoint_ends_trial_and_respawns_fly(make_runner):
    runner = make_runner("labyrinth", trial_length_s=600.0)
    fly = runner.arena.fly
    fly.pos.x, fly.pos.y = 130.0, 85.0          # inside the goal chamber
    runner.step_once()
    assert len(runner.trial_history) == 1
    assert runner.trial_history[0]["reason"] == "goal_reached"
    # Fly is back at the labyrinth entrance for the next trial
    assert (fly.pos.x, fly.pos.y) == (15.0, 15.0)
    assert runner.arena.paradigm.goal_reached is False
    assert runner.trial_sim_time == 0.0


def test_memory_is_kept_across_trials(make_runner):
    runner = make_runner("multisensory-sandbox", trial_length_s=2.0)
    circuit = runner.arena.fly.circuit
    for _ in range(int(2.0 / runner.dt) + 1):
        runner.step_once()
    assert len(runner.trial_history) == 1
    assert runner.arena.fly.circuit is circuit


def test_reset_trial_command_uses_paradigm_spawn(make_runner):
    runner = make_runner("multisensory-sandbox")
    runner.arena.fly.pos.x, runner.arena.fly.pos.y = 50.0, -30.0
    res = runner.dispatch_command({"action": "reset_trial"})
    assert res["status"] == "ok"
    assert (runner.arena.fly.pos.x, runner.arena.fly.pos.y) == (0.0, 0.0)   # not (width/2, height/2)
