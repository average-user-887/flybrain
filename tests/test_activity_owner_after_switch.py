"""Brain Activity rates must belong to the displayed graph owner.

One GraphArenaController is shared by every assay. Before this fix its last
spike counts survived a paused assay switch, so the never-stepped target assay
published the previous assay's per-region rates under its own identity.
"""
from neurofly_daemon import ContinuousExperimentRunner
from tests.transition_control_helpers import transition_command


def graph_runner(tmp_path):
    return ContinuousExperimentRunner(initial_paradigm="open-arena", sim_speed=1, checkpoint_interval=3600,
                                      output_dir=tmp_path, backend="connectome-fixed", test_synthetic_graph=True)


def test_stepped_assay_publishes_its_own_rates(tmp_path):
    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(3):
            runner.step_once()
        activity = runner._assemble_telemetry(runner._last_step_result)["activity"]
    assert activity["units"] == "Hz" and activity["rates"] is not None
    assert "unavailable" not in activity
    owner = runner.graph_controller.last_counts_owner
    assert owner == (runner.registry.active.instance_id, "open-arena", runner.registry.active.step_index)


def test_unstepped_switch_target_does_not_inherit_previous_assay_rates(tmp_path):
    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(3):
            runner.step_once()
    source_instance = runner.registry.active.instance_id
    assert runner.graph_controller.last_counts is not None
    ack = transition_command(runner, {"action": "switch_paradigm", "params": {"paradigm": "labyrinth"}})
    assert ack["status"] == "ok", ack
    assert runner.active_paradigm_id == "labyrinth"
    assert runner.registry.active.instance_id != source_instance
    # The shared controller still holds the open-arena counts; they must not be published.
    assert runner.graph_controller.last_counts is not None
    with runner.lock:
        activity = runner._assemble_telemetry({})["activity"]
    assert activity["rates"] is None
    assert activity["unavailable"] == "No graph step for this assay since it was selected"
    # The first real step of the new assay publishes its own rates again.
    with runner.lock:
        runner.step_once()
        activity = runner._assemble_telemetry(runner._last_step_result)["activity"]
    assert activity["rates"] is not None and "unavailable" not in activity
    assert runner.graph_controller.last_counts_owner[:2] == (runner.registry.active.instance_id, "labyrinth")


def test_modular_activity_unchanged(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm="open-arena", sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path, backend="modular")
    with runner.lock:
        runner.step_once()
        activity = runner._assemble_telemetry(runner._last_step_result)["activity"]
    assert activity["grouping"] == "modular-circuit" and "unavailable" not in activity


def test_gait_cadence_withheld_until_the_switch_target_steps(tmp_path):
    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(3):
            runner.step_once()
        stepped = runner._assemble_telemetry(runner._last_step_result)["biomechanics"]
    assert isinstance(stepped["cadence_hz"], float) and "cadence_unavailable" not in stepped
    transition_command(runner, {"action": "switch_paradigm", "params": {"paradigm": "labyrinth"}})
    assert runner.active_paradigm_id == "labyrinth"
    with runner.lock:
        fly = runner.arena.fly
        assert fly.speed > 0                      # constructor default would read as a cadence
        unstepped = runner._assemble_telemetry({})["biomechanics"]
    assert unstepped["cadence_hz"] is None
    assert unstepped["cadence_unavailable"] == "No graph step for this assay since it was selected"
    with runner.lock:
        runner.step_once()
        fly = runner.arena.fly
        expected = round(8.0 * (fly.speed / 12.0) if fly.speed > 0 else 0.0, 1)
        measured = runner._assemble_telemetry(runner._last_step_result)["biomechanics"]
    assert measured["cadence_hz"] == expected and "cadence_unavailable" not in measured


def test_modular_cadence_formula_unchanged(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm="open-arena", sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path, backend="modular")
    with runner.lock:
        bio = runner._assemble_telemetry({})["biomechanics"]
        fly = runner.arena.fly
    assert bio["cadence_hz"] == round(8.0 * (fly.speed / 12.0) if fly.speed > 0 else 0.0, 1)
    assert "cadence_unavailable" not in bio
