"""Cue positions are copied from live model objects, never inferred by the browser."""
import json
from arena import Arena
from neurofly_daemon import assay_cue_telemetry

def test_y_maze_copies_exact_arm_zone_positions_and_names():
    arena=Arena(paradigm='y-maze');p=arena.paradigm
    expected=[{'name':z.name,'position':list(z.bounds[:2])} for z in p.zones if z.zone_type=='arm']
    assert assay_cue_telemetry(arena)['arm_tips']==expected
    arm=next(z for z in p.zones if z.zone_type=='arm');arm.bounds=(12.25,31.5,15.)
    assert assay_cue_telemetry(arena)['arm_tips'][0]['position']==[12.25,31.5]

def test_labyrinth_and_sandbox_use_current_object_coordinates():
    arena=Arena(paradigm='labyrinth');arena.paradigm.goal_pos=(20.25,10.5)
    assert assay_cue_telemetry(arena)['goal_pos']==[20.25,10.5]
    arena=Arena(paradigm='multisensory-sandbox');p=arena.paradigm
    result=assay_cue_telemetry(arena)
    for name in ('food_pos','repellent_pos','pheromone_pos','hotspot_pos','cool_pos'):
        assert result[name]==list(getattr(p,name))
    assert result['pillar_centers']==[list(v) for v in p.pillar_centers]
    result['food_pos'][0]=999
    assert p.food_pos[0]!=999
    json.dumps(assay_cue_telemetry(arena),allow_nan=False)

def test_missing_and_nonfinite_cues_are_not_fabricated():
    assert assay_cue_telemetry(Arena(paradigm='optomotor'))=={}
    assert assay_cue_telemetry(Arena(paradigm='heat-maze'))=={}
    assert assay_cue_telemetry(Arena(paradigm='circadian-dam'))=={}
    arena=Arena(paradigm='labyrinth');arena.paradigm.goal_pos=(float('nan'),1)
    assert 'goal_pos' not in assay_cue_telemetry(arena)

def test_initial_daemon_packets_embed_current_runtime_cues(tmp_path):
    from neurofly_daemon import ContinuousExperimentRunner
    for assay in ('y-maze','labyrinth','multisensory-sandbox'):
        runner=ContinuousExperimentRunner(initial_paradigm=assay,output_dir=tmp_path/assay,checkpoint_interval=3600)
        expected=assay_cue_telemetry(runner.arena)
        scene=runner.latest_telemetry['scene']
        assert expected and all(scene[key]==value for key,value in expected.items())


def test_optomotor_phase_is_declared_scene_state_but_never_a_stimulus_sample():
    # Since ab15fb5 the drum carries a declared external phase (drum_angle_deg,
    # docs/DATA_SCHEMA.md "External stimulus scene phase").  It is presentation
    # state only: the stimuli sampled for the fly stay velocity, wavelength and
    # contrast, and the phase value never changes what the fly is given.
    p=Arena(paradigm='optomotor').paradigm
    assert p.drum_angle_deg==0.0
    sample=p.sample_stimuli(45.,45.,0.)
    assert set(sample)=={'drum_velocity_deg_s','spatial_wavelength_deg','contrast'}
    p.drum_angle_deg=123.0
    assert p.sample_stimuli(45.,45.,0.)==sample
