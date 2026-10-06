"""External drum kinematics only; tiny CPU cases, no graph/device execution."""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from arena import Arena
from brainlab.io_map import OptomotorEncoder
from maze import OptomotorParadigm
from neurofly_daemon import ContinuousExperimentRunner


def make_arena():
    return Arena(paradigm='optomotor', seed=7, num_flies=1, num_predators=0)


def test_phase_wrap_reversal_and_contrast_boundaries_preserve_continuity():
    arena = make_arena()
    p = arena.paradigm
    assert p.drum_angle_deg == 0.0
    p.drum_angle_deg = 359.5
    arena.step(.02)
    assert p.drum_angle_deg == pytest.approx(.1)
    p.drum_velocity_deg_s = -30.0
    p.begin_next_presentation('reverse_grating')
    assert p.drum_angle_deg == pytest.approx(.1)
    arena.step(.02)
    assert p.drum_angle_deg == pytest.approx(359.5)
    p.contrast = 0.0
    p.begin_next_presentation('contrast')
    assert p.drum_angle_deg == pytest.approx(359.5)
    arena.step(.02)
    assert p.drum_angle_deg == pytest.approx(358.9)
    p.drum_velocity_deg_s = 0.0
    p.begin_next_presentation('patternSpeed')
    arena.step(.02)
    assert p.drum_angle_deg == pytest.approx(358.9)


def test_real_runner_pause_holds_phase_and_publishes_the_same_scene_angle(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm='optomotor', output_dir=tmp_path,
        continuous=True, standalone_scheduled_records=False, brain_backend='cpu')
    runner.step_once()
    phase = runner.arena.paradigm.drum_angle_deg
    assert phase == pytest.approx(.6)
    assert runner.latest_telemetry['scene']['drum_angle_deg'] == phase
    runner.paused = True
    for _ in range(4):
        runner.step_once()
        assert runner.arena.paradigm.drum_angle_deg == phase
        assert runner.latest_telemetry['scene']['drum_angle_deg'] == phase
    runner.paused = False
    runner.step_once()
    assert runner.arena.paradigm.drum_angle_deg == pytest.approx(1.2)


def test_nonzero_world_phase_restores_exactly_and_continues_through_reversal():
    arena = make_arena()
    for _ in range(7):
        arena.step(.02)
    snapshot = json.loads(json.dumps(arena.snapshot_world(), allow_nan=False))
    before = copy.deepcopy(snapshot)
    restored = make_arena()
    restored.restore_world(snapshot)
    assert restored.paradigm.drum_angle_deg == arena.paradigm.drum_angle_deg
    assert restored.snapshot_world() == snapshot
    for velocity in (-30.0, -30.0, 30.0):
        for a in (arena, restored):
            a.paradigm.drum_velocity_deg_s = velocity
            a.step(.02)
        assert restored.snapshot_world() == arena.snapshot_world()
    assert snapshot == before


def test_legacy_world_phase_stays_unknown_until_explicit_presentation_or_trial_reset():
    arena = make_arena()
    arena.step(.02)
    legacy = arena.snapshot_world()
    del legacy['state']['paradigm']['state']['drum_angle_deg']
    before = copy.deepcopy(legacy)
    restored = make_arena()
    restored.restore_world(legacy)
    assert restored.paradigm.drum_angle_deg is None
    restored.step(.02)
    assert restored.paradigm.drum_angle_deg is None
    unknown = restored.snapshot_world()
    restored.restore_world(unknown)
    assert restored.paradigm.drum_angle_deg is None
    restored.paradigm.begin_next_presentation('reverse_grating')
    assert restored.paradigm.drum_angle_deg == 0.0
    restored.paradigm.drum_angle_deg = None
    restored.paradigm.reset_trial()
    assert restored.paradigm.drum_angle_deg == 0.0
    assert legacy == before


def test_external_phase_does_not_change_existing_stimuli_step_results_or_encoder_currents():
    known, unknown = OptomotorParadigm(), OptomotorParadigm()
    known.drum_angle_deg = 123.0
    unknown.drum_angle_deg = None
    populations = {name: np.arange(i * 2, i * 2 + 2) for i, name in enumerate(('ftb_L', 'btf_L', 'ftb_R', 'btf_R'))}
    io = SimpleNamespace(populations=populations)
    encoders = [OptomotorEncoder(io, np.random.default_rng(13)) for _ in range(2)]
    fly = SimpleNamespace(pos=SimpleNamespace(x=45.0, y=45.0), heading=0.0,
        speed=0.0, angular_velocity=.1, behavioral_state='FORWARD')
    for index, (velocity, contrast) in enumerate(((30.0, .9), (-30.0, .9), (-30.0, 0.0), (0.0, .9))):
        for p in (known, unknown):
            p.drum_velocity_deg_s = velocity
            p.contrast = contrast
        assert known.step(fly, .02) == unknown.step(fly, .02)
        stimuli = [p.sample_stimuli(45.0, 45.0, 0.0) for p in (known, unknown)]
        assert stimuli[0] == stimuli[1] == {'drum_velocity_deg_s': velocity,
            'spatial_wavelength_deg': 30.0, 'contrast': contrast}
        currents = [np.zeros(8, np.float32) for _ in range(2)]
        totals = [encoder.encode(current, index * 20.0, math.radians(stim['drum_velocity_deg_s']) - .1, stim['contrast'])
            for encoder, current, stim in zip(encoders, currents, stimuli)]
        np.testing.assert_array_equal(currents[0], currents[1])
        assert totals[0] == totals[1]
