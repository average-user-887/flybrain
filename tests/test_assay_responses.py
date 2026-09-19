"""Causal checks: changing an input must change the measured motor response."""
import math
import pytest
from arena import Arena
from assay_controls import act, describe, set_parameter
from experiment_brains import PARADIGMS
from neurofly_daemon import ContinuousExperimentRunner


def arena(pid, seed=4):
    return Arena(paradigm=None if pid=='open-arena' else pid, seed=seed,num_flies=1,num_predators=0)


def advance(a, seconds):
    for _ in range(round(seconds/.02)):
        result=a.step(.02)
    return result


def test_mirrored_plume_changes_bilateral_input_and_turn_direction():
    turns=[]
    for y in (25,35):
        a=arena('wind-tunnel');a.fly.pos.y=y
        a.step(.02)
        s=a.fly.sensory_input
        turns.append(a.fly.angular_velocity)
        assert math.copysign(1,s['diff_a'])==math.copysign(1,30-y)
    assert turns[0]>0>turns[1]


def test_heat_moves_toward_local_cooling_without_permanent_reverse():
    a=arena('heat-maze');goal=a.paradigm.refuge_pos
    before=math.dist(a.fly.pos.to_tuple(),goal)
    advance(a,30)
    assert math.dist(a.fly.pos.to_tuple(),goal)<before/2
    assert a.paradigm.refuge_reached
    assert a.fly.food_collected==0  # relief is not sucrose


def test_rotate_landmarks_changes_yaw_with_matched_initial_state():
    turns=[]
    for rotated in (False,True):
        a=arena('buridan');a.fly.heading=.5
        if rotated: act(a,'rotate_stripes')
        a.step(.02);turns.append(a.fly.angular_velocity)
    assert turns[0]<0<turns[1]


def test_heat_drives_tethered_yaw_out_of_punished_sector():
    a=arena('visual-operant');a.paradigm.drum_angle_deg=135
    start=a.fly.pos.to_tuple();advance(a,2)
    assert a.fly.pos.to_tuple()==start
    assert not a.paradigm.laser_heat_active
    assert a.paradigm.get_metrics()['time_safe_pct']>50


def test_receptivity_changes_approach_distance():
    distances=[]
    for virgin in (False,True):
        a=arena('courtship')
        if virgin: act(a,'receptivity')
        advance(a,5)
        distances.append(math.dist(a.fly.pos.to_tuple(),a.paradigm.female_pos))
    assert distances[1]<2<distances[0]


def test_wide_gap_aborts_physically_while_narrow_gap_crosses():
    for width in (3,5):
        a=arena('gap-crossing');set_parameter(a,'gapWidth',width);a.fly.pos.x=42.1
        advance(a,20)
        if width==3:
            assert a.paradigm.crossing_success and a.fly.pos.x>48
        else:
            assert not a.paradigm.crossing_success and a.fly.pos.x<42
            assert a.paradigm.decision_outcome=='ABORT'


def test_grating_contrast_is_a_motor_input():
    turns=[]
    for visible in (False,True):
        a=arena('optomotor')
        if not visible:act(a,'contrast')
        advance(a,1)
        observed=[]
        for _ in range(100):
            a.step(.02);observed.append(a.fly.angular_velocity)
        if not visible:
            assert a.paradigm.hs_firing_history[-1] == 40.0
        turns.append(sum(observed)/len(observed))
    assert turns[1]>turns[0]+.1


def test_repeat_loom_does_not_reset_pose_or_trial_clock():
    a=arena('looming-escape');advance(a,1)
    before=a.fly.pos.to_tuple();elapsed=a.paradigm.time_elapsed_ms
    act(a,'loom')
    assert a.fly.pos.to_tuple()==before and a.paradigm.time_elapsed_ms==elapsed
    advance(a,.4)
    assert a.total_escapes==2


@pytest.mark.parametrize('pid',PARADIGMS)
def test_live_capabilities_have_no_silent_unsupported_parameters(pid):
    a=arena(pid)
    for spec in describe(a)['parameters']:
        set_parameter(a,spec['name'],spec['value'])
    with pytest.raises(ValueError):set_parameter(a,'invented_parameter',1)
    with pytest.raises(ValueError):act(a,'invented_action')


def test_commands_change_physics_and_are_recorded(tmp_path):
    r=ContinuousExperimentRunner(initial_paradigm='optomotor',output_dir=tmp_path,continuous=True)
    assert r.dispatch_command({'action':'set_param','params':{'name':'patternSpeed','value':-60}})['status']=='ok'
    for _ in range(100):r.step_once()
    assert r.arena.fly.angular_velocity<0
    assert r.latest_telemetry['stimuli']['drum_velocity_deg_s']==-60
    assert r.active_brain.history[-1]['kind']=='intervention'
    assert r.dispatch_command({'action':'set_param','name':'fake','value':1})['status']=='error'
    assert r.dispatch_command({'action':'inject_stimulus','type':'unsupported'})['status']=='error'


def test_open_arena_telemetry_contains_the_inputs_actually_seen(tmp_path):
    r=ContinuousExperimentRunner(initial_paradigm='open-arena',output_dir=tmp_path,continuous=True)
    r.step_once()
    assert r.latest_telemetry['stimuli']['odor_a']==r.arena.fly.sensory_input['mean_a']


def test_t_maze_counts_entries_instead_of_occupancy_ticks():
    a=arena('t-maze');p=a.paradigm
    for _ in range(100):p.step({'x':30,'y':50,'heading':0},.02)
    assert p.get_metrics()['cs_plus_choices']==1
    p.step({'x':70,'y':50,'heading':0},.02)
    p.step({'x':30,'y':50,'heading':0},.02)
    assert p.get_metrics()['cs_plus_choices']==2
