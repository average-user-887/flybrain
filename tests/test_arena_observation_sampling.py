"""CARD-12A: real arena samples reach exactly one explicitly enabled observer."""
import copy
import json

import pytest

from arena import Arena, Position
from experiment_brains import PARADIGMS
from online_metrics import MetricFault, OpenArenaObserver, build_observation_config


DT = 0.02


def config_for(arena, name="arena-integration"):
    owner = arena.paradigm or OpenArenaObserver()
    return build_observation_config(owner.observation_spec(), config_id=name)


def observer_snapshot(owner):
    return json.dumps({
        "status": owner.observation_status(),
        "records": owner.get_metric_records(),
        "evidence": owner.get_evidence(),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)


@pytest.mark.parametrize("paradigm", PARADIGMS)
def test_all_14_assays_attach_one_primary_and_preserve_legacy_step_output(paradigm):
    kwargs = dict(paradigm=None if paradigm == "open-arena" else paradigm,
                  seed=17, num_flies=1, num_predators=0)
    legacy = Arena(**kwargs)
    instrumented = Arena(**kwargs)

    owner = instrumented.enable_observation(config_for(instrumented, paradigm))
    assert owner is (instrumented.paradigm or instrumented.observation_owner)
    assert instrumented.observation_owner is owner
    assert owner.observation_status()["segment_elapsed_s"] == 0.0

    for expected_step in range(1, 4):
        legacy_out = legacy.step(DT)
        instrumented_out = instrumented.step(DT)
        assert instrumented_out == legacy_out
        assert instrumented.time_step == expected_step

    status = owner.observation_status()
    assert status["segment_elapsed_s"] == 3 * DT
    assert owner._v1_last_step == instrumented.time_step == 3
    assert owner._v1_last_dt_us == 20_000
    json.dumps(owner.get_metric_records(), allow_nan=False)


def test_effective_config_is_delivered_before_first_sample():
    arena = Arena(paradigm="heat-maze", seed=4, num_flies=1, num_predators=0)
    spec = arena.paradigm.observation_spec()
    config = build_observation_config(spec, config_id="cli-override", override_window_s=1.25,
                                      override_source="cli:--trial-seconds",
                                      set_at_sim_s=8.0, manifest_run_id="run-1")

    owner = arena.enable_observation(config)
    assert owner.observation_config() == config
    assert owner.observation_status()["effective_window_s"] == 1.25
    assert owner._v1_pres_ticks == 0

    arena.step(DT)
    assert owner.observation_config() == config
    assert owner._v1_pres_ticks == 1


def test_multi_subject_dead_primary_and_second_attachment_refuse_before_mutation():
    multi = Arena(paradigm="t-maze", seed=4, num_flies=2, num_predators=0)
    before = observer_snapshot(multi.paradigm)
    with pytest.raises(ValueError, match="exactly one primary fly"):
        multi.enable_observation(config_for(multi))
    assert multi.observation_owner is None
    assert observer_snapshot(multi.paradigm) == before
    assert multi.time_step == 0

    dead = Arena(paradigm="t-maze", seed=4, num_flies=1, num_predators=0)
    dead.fly.alive = False
    before = observer_snapshot(dead.paradigm)
    with pytest.raises(ValueError, match="one live primary fly"):
        dead.enable_observation(config_for(dead))
    assert dead.observation_owner is None
    assert observer_snapshot(dead.paradigm) == before

    single = Arena(paradigm="t-maze", seed=4, num_flies=1, num_predators=0)
    owner = single.enable_observation(config_for(single))
    before = observer_snapshot(owner)
    with pytest.raises(RuntimeError, match="already enabled"):
        single.enable_observation(config_for(single, "second"))
    assert single.observation_owner is owner
    assert observer_snapshot(owner) == before


def test_open_arena_uses_actual_primary_food_contact_pose_step_and_airflow():
    arena = Arena(paradigm=None, seed=4, num_flies=1, num_predators=0)
    arena.wind = (-7.0, 0.0)
    arena.food_positions = [Position(arena.fly.pos.x, arena.fly.pos.y)]
    arena.odor_a.clear()
    arena.odor_a.add_source(arena.fly.pos.x, arena.fly.pos.y, 1.0)
    start = (arena.fly.pos.x, arena.fly.pos.y)
    owner = arena.enable_observation(config_for(arena))

    arena.step(DT)

    records = owner.get_metric_records()
    assert records["food_contacts"]["value"] == 1
    assert records["presentation_food_contacts"]["value"] == 1
    assert records["airflow_mm_s"]["value"] == 7.0
    assert owner._last_xy == start
    event = owner.get_evidence()["contact_events"]["data"]
    assert event == [{"event": "food_contact_onset", "t_rel_s": 0.0, "step": 1,
                      "sample_point": "pre_motor",
                      "attrs": {"presentation_index": 0}}]
    assert owner._v1_pending_step is None


@pytest.mark.parametrize("wind", [
    (float("nan"), 0.0), (float("inf"), 0.0), (float("-inf"), 0.0),
    (-5.0, 1.0), (5.0, 0.0),
])
def test_open_arena_refuses_nonfinite_or_off_axis_condition_before_attachment(wind):
    arena = Arena(paradigm=None, seed=4, num_flies=1, num_predators=0)
    arena.wind = wind
    with pytest.raises(MetricFault, match="airflow") as caught:
        arena.enable_observation(config_for(arena))
    assert caught.value.path == "open_arena.airflow_mm_s"
    assert arena.observation_owner is None
    assert arena.time_step == 0


@pytest.mark.parametrize("wind", [(float("nan"), 0.0), (-5.0, 2.0), (-8.0, 0.0)])
def test_open_arena_condition_fault_propagates_without_observer_mutation(wind):
    arena = Arena(paradigm=None, seed=4, num_flies=1, num_predators=0)
    arena.wind = (-5.0, 0.0)
    arena.food_positions = []
    arena.odor_a.clear()
    owner = arena.enable_observation(config_for(arena))
    arena.step(DT)
    before = observer_snapshot(owner)
    before_step = arena.time_step
    before_pose = (arena.fly.pos.x, arena.fly.pos.y, arena.fly.heading, arena.fly.speed)

    arena.wind = wind
    with pytest.raises(MetricFault):
        arena.step(DT)

    assert observer_snapshot(owner) == before
    assert owner._v1_pending_step is None
    assert arena.time_step == before_step
    assert (arena.fly.pos.x, arena.fly.pos.y, arena.fly.heading, arena.fly.speed) == before_pose


def test_actual_post_solver_contact_handoff_is_once_per_tick_and_preserves_state(monkeypatch):
    arena = Arena(paradigm="labyrinth", seed=4, num_flies=1, num_predators=0)
    arena.fly.pos.x, arena.fly.pos.y = 70.0, 50.0
    owner = arena.enable_observation(config_for(arena))
    collision = {"active": True, "calls": 0}

    def advanced(x, y, vx, vy, **kwargs):
        collision["calls"] += 1
        if collision["active"]:
            return x, y, vx, vy, True, [(0.0, 1.0)]
        return x, y, vx, vy, False, []

    monkeypatch.setattr(arena.paradigm, "check_collisions_advanced", advanced)
    handoffs = []
    original = owner.observe_contact

    def capture(*args, **kwargs):
        handoffs.append((args, copy.deepcopy(kwargs)))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "observe_contact", capture)

    arena.step(0.017)
    assert collision["calls"] == 1
    assert len(handoffs) == 1
    args, kwargs = handoffs[0]
    assert args[:4] == (1, 0.017, "post_solver", True)
    assert kwargs == {"normals": ((0.0, 1.0),), "source": "solver"}
    contact_events = owner.get_evidence()["contact_events"]["data"]
    assert contact_events == [{"event": "contact_onset", "t_rel_s": 0.017, "step": 1,
                               "sample_point": "post_solver", "attrs": {"source": "solver"}}]

    frozen = copy.deepcopy(owner.freeze_observation("re_presentation_user"))
    owner.begin_next_presentation("user_test")
    arena.step(0.017)  # same physical contact: no manufactured onset after presentation change
    records = owner.get_metric_records()
    assert records["wall_contact_onsets"]["value"] == 1
    assert records["wall_contact_s"]["value"] == pytest.approx(0.034)
    assert len(owner.get_evidence()["contact_events"]["data"]) == 1
    assert frozen["records"]["wall_contact_onsets"]["value"] == 1

    collision["active"] = False
    arena.step(0.017)
    assert collision["calls"] == 3
    assert len(handoffs) == 3
    args, kwargs = handoffs[-1]
    assert args[:4] == (3, 0.017, "post_solver", False)
    assert kwargs == {"normals": (), "source": "none"}
    assert owner.get_metric_records()["wall_contact_onsets"]["counts"]["handoffs"] == 3

    owner.reset_trial()
    assert owner.observation_status()["segment_elapsed_s"] == 0.0
    assert owner.get_metric_records()["wall_contact_onsets"]["reason"] == "not_observed"


def test_enabled_paradigm_metric_fault_is_propagated(monkeypatch):
    arena = Arena(paradigm="heat-maze", seed=4, num_flies=1, num_predators=0)
    owner = arena.enable_observation(config_for(arena))
    monkeypatch.setattr(owner.peltier, "get_temperature", lambda x, y: float("nan"))

    with pytest.raises(MetricFault):
        arena.step(DT)
