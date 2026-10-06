"""Measurement-only segment boundaries on retained C0 producer instances."""
import copy
import json
import pickle

import pytest

import maze
from arena import Arena
from online_metrics import ConfigError, MetricFault, OpenArenaObserver, build_observation_config
from tests.test_metric_contract import ARM, CENTRE, DT, Fly, config, new, rec


def _new_config(owner, config_id="next-segment"):
    return build_observation_config(owner.observation_spec(), config_id=config_id)


def _tick_contact(owner, fly, contact, step=None):
    if step is not None:
        owner.begin_sample(step, DT)
    owner.step(fly, DT)
    owner.observe_contact(owner._v1_last_step, DT, "post_solver", contact)


def _value(owner, name):
    item = rec(owner, name)
    assert item["available"], item
    assert item["interval_rel_s"][0] == 0.0
    return item["value"]


@pytest.mark.parametrize("assay", maze.ExperimentRegistry.list_paradigms() + ["open_arena"])
def test_all_14_producers_start_clean_segment_with_exact_config_and_detached_old_envelope(assay):
    owner = OpenArenaObserver() if assay == "open_arena" else new(assay)
    owner.configure_observation(_new_config(owner, "old-segment"))
    old = owner.freeze_observation("manual_reset")
    old_bytes = json.dumps(old, sort_keys=True, separators=(",", ":"), allow_nan=False)
    next_config = _new_config(owner, f"next-{assay}")

    assert owner.begin_observation_segment(next_config) == next_config

    status = owner.observation_status()
    assert status["state"] == "observing"
    assert status["segment_elapsed_s"] == 0.0
    assert status["presentation_elapsed_s"] == 0.0
    assert status["presentation_index"] == 0
    assert owner.observation_config() == next_config
    assert owner._v1_last_step is None
    assert owner._v1_interventions == []
    assert json.dumps(old, sort_keys=True, separators=(",", ":"), allow_nan=False) == old_bytes


def test_invalid_config_and_fault_reset_are_atomic_at_public_boundary():
    owner = new("t_maze")
    owner.configure_observation(_new_config(owner, "old"))
    owner.step(Fly(float("nan"), 50), DT)
    assert isinstance(owner._v1_fault, MetricFault)
    before = pickle.dumps(vars(owner), protocol=5)
    invalid = _new_config(owner, "invalid")
    invalid["effective_window_s"] = float("nan")

    with pytest.raises(ConfigError):
        owner.begin_observation_segment(invalid)

    assert pickle.dumps(vars(owner), protocol=5) == before
    owner.begin_observation_segment(_new_config(owner, "valid"))
    assert owner._v1_fault is None


def test_t_and_y_maze_keep_rearm_state_without_old_visits():
    t_maze = new("t_maze")
    arm_a, hub = Fly(30, 50), Fly(70, 50)
    t_maze.step(arm_a, DT)
    t_maze.begin_observation_segment(config(t_maze, "t-next"))
    t_maze.step(arm_a, DT)
    assert _value(t_maze, "cs_plus_entries") == 0
    t_maze.step(hub, DT)
    t_maze.step(arm_a, DT)
    assert _value(t_maze, "cs_plus_entries") == 1

    y_maze = new("y_maze")
    arm = Fly(*ARM["A"])
    for _ in range(3):
        y_maze.step(arm, DT)
    y_maze.begin_observation_segment(config(y_maze, "y-next"))
    y_maze.step(arm, DT)
    assert _value(y_maze, "physical_entries") == 0
    for _ in range(3):
        y_maze.step(Fly(*CENTRE), DT)
    y_maze.step(arm, DT)
    assert _value(y_maze, "physical_entries") == 1


def test_labyrinth_keeps_dead_end_and_contact_hysteresis_only():
    owner = new("labyrinth")
    inside, outside = Fly(30, 75), Fly(30, 68.9)
    _tick_contact(owner, inside, True, step=100)
    owner.step(Fly(40, 75), DT)  # establish old path history that must not bridge the boundary
    owner.begin_observation_segment(config(owner, "lab-next"))

    _tick_contact(owner, inside, True, step=7)  # lower, nonconsecutive arena id is a new sample
    assert _value(owner, "dead_end_entries") == 0
    assert _value(owner, "wall_contact_onsets") == 0
    assert _value(owner, "wall_contact_s") == pytest.approx(DT)
    assert _value(owner, "path_length_mm") == 0
    assert owner._lb["contacts"].handoffs == 1

    _tick_contact(owner, outside, False, step=8)
    _tick_contact(owner, inside, True, step=9)
    assert _value(owner, "dead_end_entries") == 1
    assert _value(owner, "wall_contact_onsets") == 1


def test_multisensory_keeps_active_contact_but_resets_path_and_jerk_history():
    owner = new("multisensory")
    fly = Fly(45, 45, speed=3)
    _tick_contact(owner, fly, True)
    _tick_contact(owner, Fly(-40, 40, speed=7), True)
    assert owner._ms["v"] is not None
    owner.begin_observation_segment(config(owner, "multi-next"))

    _tick_contact(owner, fly, True)
    assert _value(owner, "wall_contact_onsets") == 0
    assert _value(owner, "distance_mm") == 0
    assert rec(owner, "mean_abs_speed_jerk_mm_s3")["reason"] == "not_observed"
    _tick_contact(owner, fly, False)
    _tick_contact(owner, fly, True)
    assert _value(owner, "wall_contact_onsets") == 1


def test_open_arena_keeps_food_contact_only_and_resets_distance_baseline():
    owner = OpenArenaObserver()
    owner.observe(Fly(10, 10), True, DT, arena_step=100)
    owner.observe(Fly(20, 10), True, DT, arena_step=101)
    owner.begin_observation_segment(_new_config(owner, "open-next"))

    owner.observe(Fly(20, 10), True, DT, arena_step=7)
    assert _value(owner, "food_contacts") == 0
    assert _value(owner, "distance_mm") == 0
    owner.observe(Fly(20, 10), False, DT, arena_step=8)
    owner.observe(Fly(20, 10), True, DT, arena_step=9)
    assert _value(owner, "food_contacts") == 1


def test_dam_does_not_bridge_position_or_immobility_bout_duration():
    owner = new("circadian_dam")
    owner.step(Fly(10, 5), DT)
    owner._dam.update(cur_us=4_000_000, bout_us=3_000_000, bouts=2, in_bout=True)
    owner.begin_observation_segment(config(owner, "dam-next"))

    owner.step(Fly(40, 5), DT)
    assert _value(owner, "beam_crossings") == 0
    assert _value(owner, "current_immobile_s") == pytest.approx(DT)
    assert owner._dam["bout_us"] == 0
    assert owner._dam["bouts"] == 0


def _non_observer_world(arena):
    snapshot = copy.deepcopy(arena.snapshot_world())
    state = snapshot["state"]
    paradigm = state.pop("paradigm")["state"]
    state.pop("_observation_owner")
    observer_fields = {"_tm", "_ym", "_hm", "_bu", "_vo", "_wt", "_lo", "_om", "_gc",
                       "_dam", "_co", "_lb", "_ms"}
    legacy_paradigm = {key: value for key, value in paradigm.items()
                       if not key.startswith("_v1") and key not in observer_fields}
    return snapshot, legacy_paradigm


def test_actual_arena_retains_physics_controller_and_stimulus_snapshot():
    arena = Arena(paradigm="t-maze", seed=23, num_flies=1, num_predators=0)
    owner = arena.enable_observation(_new_config(arena.paradigm, "arena-old"))
    arena.step(DT)
    before_world, before_paradigm = _non_observer_world(arena)

    owner.begin_observation_segment(_new_config(owner, "arena-next"))

    after_world, after_paradigm = _non_observer_world(arena)
    assert after_world == before_world
    assert after_paradigm == before_paradigm
