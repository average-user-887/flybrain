"""Scientific trajectories must have one clock and explicit discontinuities."""
import math

import pytest

from arena import Arena
from experiment_brains import PARADIGMS
from neurofly_daemon import ContinuousExperimentRunner


@pytest.mark.parametrize('pid', PARADIGMS)
def test_continuous_observation_never_auto_respawns(pid, tmp_path):
    r = ContinuousExperimentRunner(initial_paradigm=pid, output_dir=tmp_path,
                                   trial_length_s=1, continuous=True)
    segment = r.segment_id
    previous = r.arena.fly.pos.to_tuple()
    for _ in range(150):
        r.step_once()
        pose = r.arena.fly.pos.to_tuple()
        assert math.dist(previous, pose) <= 3.5 * r.dt + .25
        assert r.segment_id == segment
        assert not r.last_error
        previous = pose
    assert r.current_trial == 1 and not r.trial_history
    assert r.latest_telemetry['continuous']


def test_terminal_outcome_is_not_assigned_to_the_respawn_pose(tmp_path):
    r = ContinuousExperimentRunner(initial_paradigm='labyrinth', output_dir=tmp_path)
    old_segment = r.segment_id
    r.arena.fly.pos.x, r.arena.fly.pos.y = 130, 85
    r.step_once()
    t = r.latest_telemetry
    assert t['segment_id'] != old_segment
    assert t['transition']['ended_segment'] == old_segment
    assert t['transition']['terminal_metrics']['goal_reached']
    assert t['transition']['terminal_pose']['x'] > 120
    assert t['metrics'] == {} and t['fly']['x'] == 15


def test_pause_freezes_time_pose_and_memory_and_reset_marks_a_boundary(tmp_path):
    r = ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=tmp_path, continuous=True)
    r.step_once()
    before = r.latest_telemetry
    r.dispatch_command({'action':'set_paused', 'params':{'paused':True}})
    for _ in range(20):
        r.step_once()
    assert r.total_steps == before['step']
    assert r.latest_telemetry['fly'] == before['fly']
    assert r.latest_telemetry['paused']
    r.dispatch_command({'action':'reset_trial'})
    assert r.latest_telemetry['segment_id'] != before['segment_id']
    assert r.latest_telemetry['transition']['reason'] == 'manual_reset'
    r.dispatch_command({'action':'set_paused', 'paused':False})
    r.step_once()
    assert r.total_steps == before['step'] + 1


def test_step_failure_stops_counting_valid_samples(tmp_path, monkeypatch):
    r = ContinuousExperimentRunner(output_dir=tmp_path)
    def broken(dt):
        raise RuntimeError('broken physics')
    monkeypatch.setattr(r.arena, 'step', broken)
    r.step_once()
    assert r.total_steps == 0 and r.latest_telemetry['error'] == 'broken physics'
    assert r.active_brain.history[-1]['kind'] == 'simulation_error'


@pytest.mark.parametrize('pid', ['visual-operant', 'optomotor'])
def test_tethered_assays_keep_position_while_heading_can_change(pid):
    a = Arena(paradigm=pid, num_flies=1, num_predators=0)
    pose = a.fly.pos.to_tuple()
    for _ in range(100):
        a.step(.02)
        assert a.fly.pos.to_tuple() == pose


def test_locomotion_and_arena_advance_the_same_time():
    a = Arena(paradigm='y-maze', num_flies=1, num_predators=0)
    a.step(.02)
    assert a.fly.surge_cast.dt == .02


def test_circadian_does_not_label_two_seconds_as_hundred_minutes():
    from maze import CircadianDAMParadigm
    dam = CircadianDAMParadigm()
    fly = {'x':10, 'y':5, 'speed':0}
    for _ in range(100):
        result = dam.step(fly, .02)
    assert result['total_sleep_minutes'] == 0
    assert result['stimuli']['minute_of_day'] == pytest.approx(2 / 60)
    for _ in range(5):
        result = dam.step(fly, 60)
    assert result['is_sleeping']
    assert result['total_sleep_minutes'] == pytest.approx(5 + 2 / 60)


def test_looming_assay_gf_event_reaches_the_motor_model():
    a = Arena(paradigm='looming-escape', num_flies=1, num_predators=0)
    events = []
    for _ in range(30):
        r = a.step(.02)
        if r['paradigm_telemetry']['gf_spike']:
            events.append(r)
    assert len(events) == 1
    assert events[0]['state'] == 'ESCAPE'
    assert events[0]['fly_speed'] == pytest.approx(3.5)
    assert a.total_escapes == 1


def test_opposite_drum_motion_causes_opposite_yaw():
    outputs = []
    for velocity in (-30, 30):
        a = Arena(paradigm='optomotor', seed=4, num_flies=1, num_predators=0)
        a.paradigm.drum_velocity_deg_s = velocity
        for _ in range(50):
            a.step(.02)
        outputs.append(a.fly.angular_velocity)
    assert outputs[0] < 0 < outputs[1]


def test_labyrinth_tortuosity_has_no_artificial_one_mm_floor():
    from maze import LabyrinthParadigm
    maze = LabyrinthParadigm()
    maze.path_points = [(0,0),(.01,0),(.02,0)]
    assert maze.get_metrics()['path_tortuosity'] == pytest.approx(1)
    maze.path_points = [(0,0),(0,0)]
    assert maze.get_metrics()['path_tortuosity'] is None
