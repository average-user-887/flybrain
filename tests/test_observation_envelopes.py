import copy
import json
from types import SimpleNamespace

import pytest

import maze
import online_metrics as om
from observation_envelopes import (
    ObservationEnvelopeError,
    build_observation_envelope,
    dumps_observation_envelope,
    validate_observation_envelope,
)
from provenance import RunManifest


PRODUCER_TO_ASSAY_ID = {
    'open_arena': 'open-arena', 't_maze': 't-maze', 'y_maze': 'y-maze',
    'heat_maze': 'heat-maze', 'buridan': 'buridan', 'visual_operant': 'visual-operant',
    'wind_tunnel': 'wind-tunnel', 'looming_escape': 'looming-escape',
    'optomotor': 'optomotor', 'gap_crossing': 'gap-crossing',
    'circadian_dam': 'circadian-dam', 'courtship': 'courtship',
    'labyrinth': 'labyrinth', 'multisensory': 'multisensory-sandbox',
}
PROVENANCE = {
    'gf_source': 'geometric', 'stimulus_entry_stage': None,
    'motor_assists_enabled': True,
    'controller_states': ['SURGE', 'CAST', 'REST'],
    'controller_specific': {'raw_motor_command': {'yaw': 0.0}},
}


class Fly:
    def __init__(self, x=30.0, y=30.0, *, speed=1.0):
        self.pos = SimpleNamespace(x=float(x), y=float(y))
        self.heading = 0.0
        self.speed = speed
        self.angular_velocity = 0.0
        self.assay_escape_remaining = 0.0
        self.behavioral_state = 'SURGE'


def producers():
    return [maze.ExperimentRegistry.get(key) for key in maze.ExperimentRegistry.list_paradigms()] + [
        om.OpenArenaObserver()]


def sample(producer, fly=None, dt=0.02):
    fly = fly or Fly()
    if isinstance(producer, om.OpenArenaObserver):
        producer.observe(fly, False, dt)
    else:
        producer.step(fly, dt)


def identity_for(producer_assay):
    manifest = RunManifest.create(
        backend='modular', assay=PRODUCER_TO_ASSAY_ID[producer_assay],
        instance_id='instance-1', seed=3, graph=None, dynamics={},
        learned_parameter_locations={}, source={'commit': 'fixture'})
    identity = manifest.identity()
    identity.update(activation=7, daemon_run_id='daemon-1', brain_id='brain-1')
    identity['graph_io_extension'] = {'version': None, 'sha256': None}
    return identity


_UNSET = object()


def envelope(snapshot, terminal_pose_post_step=_UNSET, **context):
    frozen = bool(snapshot.get('records')) and all(rec.get('final') is True
                                                   for rec in snapshot['records'].values())
    if terminal_pose_post_step is _UNSET:
        terminal_pose_post_step = ({'x_mm': 30.0, 'y_mm': 30.0, 'heading_rad': 0.0,
                                    'step': 4} if frozen else None)
    supplied_identity = context.pop('identity', None)
    identity = supplied_identity if supplied_identity is not None else identity_for(snapshot['assay'])
    values = dict(identity=identity, provenance=PROVENANCE, segment_id='segment-1',
                  segment_start_sim_s=123.25, validity='valid',
                  terminal_pose_post_step=terminal_pose_post_step)
    values.update(context)
    return build_observation_envelope(snapshot, **values)


def test_all_14_live_and_frozen_snapshots_are_detached_json_and_do_not_advance():
    assert len(producers()) == 14
    for producer in producers():
        assay = producer.observation_spec()['assay']
        before = copy.deepcopy(producer.observation_status())
        live_snapshot = producer.snapshot_observation()
        assert producer.observation_status() == before
        try:
            live = envelope(live_snapshot)
        except ObservationEnvelopeError as exc:
            # Exact base 8976110 predates CARD10E: DAM builds a ratio with
            # numerator/denominator but declares it kind mean.  Do not weaken
            # the envelope contract; this branch becomes ordinary coverage
            # when the separately owned producer correction is integrated.
            assert assay == 'circadian_dam'
            assert live_snapshot['records']['mean_bout_min']['kind'] == 'mean'
            assert 'numerator and denominator' in str(exc)
            continue
        assert live['schema'] == om.SCHEMA
        assert live['presentation_id'] == 'segment-1:0'
        assert live['completeness'] is None
        assert live['terminal_pose_post_step'] is None
        assert all(not rec['final'] for rec in live['records'].values())
        assert validate_observation_envelope(live) == live
        assert json.loads(dumps_observation_envelope(live, sort_keys=True)) == live

        live['identity']['activation'] = -1
        next_snapshot = producer.snapshot_observation()
        assert next_snapshot == live_snapshot

        sample(producer)
        frozen_snapshot = producer.freeze_observation('manual_reset')
        try:
            frozen = envelope(
                frozen_snapshot, validity='exploratory_degraded',
                terminal_pose_post_step={'x_mm': 30.0, 'y_mm': 30.0, 'heading_rad': 0.0})
        except ObservationEnvelopeError as exc:
            assert assay == 'circadian_dam'
            assert frozen_snapshot['records']['mean_bout_min']['kind'] == 'mean'
            assert 'numerator and denominator' in str(exc)
            continue
        assert frozen['completeness'] == 'incomplete'
        assert all(rec['final'] for rec in frozen['records'].values())
        assert frozen['measurement_end_sim_s'] == pytest.approx(
            123.25 + frozen['measurement_end_rel_s'])
        for rec in frozen['records'].values():
            if rec['interval_rel_s'] is None:
                assert rec['interval_sim_s'] is None
            else:
                assert rec['interval_sim_s'] == pytest.approx(
                    [123.25 + value for value in rec['interval_rel_s']])
        json.loads(dumps_observation_envelope(frozen))


def test_completed_dam_bout_obeys_ratio_ownership_after_separate_card10e_fix():
    dam = maze.ExperimentRegistry.get('circadian_dam')
    immobile = Fly(speed=0.0)
    for _ in range(15_000):
        sample(dam, immobile)
    snapshot = dam.freeze_observation('manual_reset')
    mean = snapshot['records']['mean_bout_min']
    assert mean['value'] == pytest.approx(5.0)
    assert mean['numerator'] == pytest.approx(5.0) and mean['denominator'] == 1
    if mean['kind'] == 'mean':
        with pytest.raises(ObservationEnvelopeError, match='numerator and denominator'):
            envelope(snapshot)
    else:
        assert mean['kind'] == 'ratio'
        assert envelope(snapshot)['records']['mean_bout_min']['value'] == pytest.approx(5.0)


def test_later_presentation_nonzero_origin_counts_labels_zero_false_null_and_integral():
    t_maze = maze.ExperimentRegistry.get('t_maze')
    sample(t_maze, Fly(30.0, 50.0), dt=0.1)  # actual arm_a entry
    first = envelope(t_maze.freeze_observation('re_presentation_user'), segment_start_sim_s=500.0)
    assert first['records']['first_choice']['counts']['arm'] == 'arm_a'
    assert first['records']['first_choice']['interval_sim_s'] == pytest.approx([500.0, 500.1])
    t_maze.begin_next_presentation('user_reverse_arms')
    sample(t_maze, Fly(70.0, 50.0), dt=0.2)
    live = envelope(t_maze.snapshot_observation(), segment_id='seg-x', segment_start_sim_s=500.0)
    assert live['presentation_id'] == 'seg-x:1'
    assert live['presentation_start_rel_s'] == pytest.approx(0.1)

    heat = maze.ExperimentRegistry.get('heat_maze')
    sample(heat, Fly(30.0, 30.0))
    heat_env = envelope(heat.snapshot_observation())
    dose = heat_env['records']['thermal_dose_degC_s']
    assert dose['kind'] == 'integral' and dose['unit'] == 'degC*s'
    assert dose['value'] > 0.0
    assert any(rec['value'] == 0 for rec in heat_env['records'].values())
    assert any(rec['value'] is None for rec in heat_env['records'].values())

    gap = maze.ExperimentRegistry.get('gap_crossing')
    gap.configure_observation(om.build_observation_config(
        gap.observation_spec(), config_id='gap-deadline', override_window_s=0.025))
    sample(gap, dt=0.02)
    sample(gap, dt=0.02)
    gap_env = envelope(gap.freeze_observation())
    assert any(rec['value'] is False for rec in gap_env['records'].values())


def test_clipped_window_continuous_unused_override_and_fault_do_not_mutate_prior():
    buridan = maze.ExperimentRegistry.get('buridan')
    cfg = om.build_observation_config(buridan.observation_spec(), config_id='clip', override_window_s=0.025)
    buridan.configure_observation(cfg)
    sample(buridan, dt=0.02)
    sample(buridan, dt=0.02)
    clipped = envelope(buridan.freeze_observation())
    assert clipped['measurement_end_rel_s'] == 0.025
    assert clipped['records']['observation_s']['value'] == 0.025

    open_arena = om.OpenArenaObserver()
    cfg = om.build_observation_config(open_arena.observation_spec(), config_id='unused',
                                      override_window_s=9.0, override_source='api')
    open_arena.configure_observation(cfg)
    continuous = envelope(open_arena.snapshot_observation())
    assert continuous['effective_window_s'] is None
    assert continuous['automatic_end'] is False
    assert continuous['override']['value'] == 9.0

    observed = maze.ExperimentRegistry.get('t_maze')
    sample(observed, Fly(30.0, 50.0))
    prior = envelope(observed.freeze_observation('re_presentation_user'))
    saved = copy.deepcopy(prior)
    observed.begin_next_presentation('user_reverse_arms')
    sample(observed, Fly(70.0, 50.0))  # leave the carried arm and re-arm entry detection
    sample(observed, Fly(120.0, 50.0))  # actual arm_b event in the faulted presentation
    fault = envelope(observed.freeze_observation('fault_halt'), validity='invalidated')
    assert prior == saved
    assert fault['end_reason'] == 'fault_halt' and fault['validity'] == 'invalidated'
    assert fault['evidence']['entries']['data'][0]['attrs']['arm'] == 'arm_b'


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s.update(schema='neurofly.metric/99'), 'schema'),
    (lambda s: s.update(assay='invented'), 'assay'),
    (lambda s: s.update(spec_version='t_maze/99'), 'spec_version'),
    (lambda s: s.update(presentation_index=True), 'presentation_index'),
    (lambda s: s['records'][next(iter(s['records']))].update(available=True, value=None), 'available'),
    (lambda s: s['records'][next(iter(s['records']))].update(capability='unsupported', reason='pending'),
     'unsupported'),
    (lambda s: s['evidence'][next(iter(s['evidence']))].update(type='picture'), 'evidence'),
])
def test_malformed_producer_snapshots_refuse(mutation, match):
    snapshot = maze.ExperimentRegistry.get('t_maze').snapshot_observation()
    mutation(snapshot)
    with pytest.raises((ObservationEnvelopeError, om.ConfigError), match=match):
        envelope(snapshot, identity=identity_for('t_maze'))


def test_malformed_context_and_nonfinite_offset_refuse_without_coercion():
    snapshot = maze.ExperimentRegistry.get('t_maze').snapshot_observation()
    bad_identity = identity_for('t_maze')
    bad_identity.pop('brain_id')
    with pytest.raises(ObservationEnvelopeError, match='brain_id'):
        envelope(snapshot, identity=bad_identity)
    with pytest.raises(ObservationEnvelopeError, match='run_id'):
        envelope(snapshot, identity={'brain_id': 'only'})
    bad_provenance = dict(PROVENANCE)
    bad_provenance.pop('controller_states')
    with pytest.raises(ObservationEnvelopeError, match='controller_states'):
        envelope(snapshot, provenance=bad_provenance)
    with pytest.raises(ObservationEnvelopeError, match='validity'):
        envelope(snapshot, validity='probably')
    with pytest.raises(ObservationEnvelopeError, match='terminal pose'):
        envelope(snapshot, terminal_pose_post_step={'x_mm': 1.0})

    huge = copy.deepcopy(snapshot)
    first = next(iter(huge['records'].values()))
    first['interval_rel_s'] = [1e308, 1e308]
    with pytest.raises(ObservationEnvelopeError, match='non-finite'):
        envelope(huge, segment_start_sim_s=1e308)
    huge = copy.deepcopy(snapshot)
    huge['records'][next(iter(huge['records']))]['counts']['bad'] = float('nan')
    with pytest.raises(ObservationEnvelopeError, match='finite'):
        envelope(huge)


def test_stamped_timing_disagreement_and_strict_json_refuse():
    producer = maze.ExperimentRegistry.get('t_maze')
    sample(producer)
    stamped = envelope(producer.freeze_observation('manual_reset'))
    wrong = copy.deepcopy(stamped)
    first = next(rec for rec in wrong['records'].values() if rec['interval_sim_s'] is not None)
    first['interval_sim_s'][1] += 0.01
    with pytest.raises(ObservationEnvelopeError, match='interval_sim_s'):
        validate_observation_envelope(wrong)
    wrong = copy.deepcopy(stamped)
    wrong['measurement_end_sim_s'] += 0.01
    with pytest.raises(ObservationEnvelopeError, match='measurement_end_sim_s'):
        validate_observation_envelope(wrong)
    wrong = copy.deepcopy(stamped)
    wrong['presentation_id'] = 'guessed'
    with pytest.raises(ObservationEnvelopeError, match='presentation_id'):
        validate_observation_envelope(wrong)
    with pytest.raises(ObservationEnvelopeError, match='fallback coercion'):
        dumps_observation_envelope(stamped, default=str)
    wrong = copy.deepcopy(stamped)
    wrong['identity']['bad'] = float('nan')
    with pytest.raises(ObservationEnvelopeError, match='finite'):
        dumps_observation_envelope(wrong)


def test_actual_nonempty_shapes_cover_all_six_evidence_types():
    t_maze = maze.ExperimentRegistry.get('t_maze')
    sample(t_maze, Fly(30.0, 50.0))
    t_envelope = envelope(t_maze.snapshot_observation())
    assert t_envelope['evidence']['entries']['data'][0]['event'] == 'arm_entry'
    assert t_envelope['evidence']['first_choice_pose']['data']['sample_point'] == 'pre_motor'

    y_maze = maze.ExperimentRegistry.get('y_maze')
    sample(y_maze, Fly(60.0, 100.0))
    y_envelope = envelope(y_maze.snapshot_observation())
    assert y_envelope['evidence']['physical_visit_sequence']['data']['symbols'] == ['A']

    dam = maze.ExperimentRegistry.get('circadian_dam')
    sample(dam, Fly(30.0, 5.0, speed=0.0))
    dam_snapshot = dam.snapshot_observation()
    # Exact base predates accepted CARD10E; apply that one declaration correction
    # only to this detached fixture so the evidence boundary itself is exercised.
    if dam_snapshot['records']['mean_bout_min']['kind'] == 'mean':
        dam_snapshot['records']['mean_bout_min']['kind'] = 'ratio'
    else:
        assert dam_snapshot['records']['mean_bout_min']['kind'] == 'ratio'
    dam_envelope = envelope(dam_snapshot)
    assert dam_envelope['evidence']['activity_bins']['data']['counts'] == [0]
    assert dam_envelope['evidence']['light_phases']['data'][0]['phase'] == 'light'

    open_arena = om.OpenArenaObserver()
    open_arena.log_intervention('food', None)
    intervention = envelope(open_arena.snapshot_observation())
    assert intervention['evidence']['interventions']['data'][0]['name'] == 'food'


@pytest.mark.parametrize('type_,data', [
    ('pose', 'not-a-pose'),
    ('pose', {'x_mm': 1, 'y_mm': 2, 'heading_rad': 0, 't_rel_s': 0,
              'step': 1, 'sample_point': 'after_lunch'}),
    ('event_sequence', {'event': 'wrong-container'}),
    ('event_sequence', [{'event': 'x', 't_rel_s': 0.0, 'step': 1,
                         'sample_point': 'pre_motor', 'attrs': {'seen': True}}]),
    ('sequence', {'symbols': [1], 'collapse': 'none'}),
    ('sequence', {'symbols': ['A'], 'collapse': 'invented'}),
    ('bins', {'bin_s': 60.0, 'start_rel_s': 0.0, 'quantity': 'events', 'counts': 'many'}),
    ('phase_boundaries', 4),
    ('phase_boundaries', [{'phase': 'light', 'start_rel_s': 2.0, 'end_rel_s': 1.0}]),
    ('intervention_log', [{'value': 1.0}]),
])
def test_each_malformed_fixed_evidence_shape_refuses(type_, data):
    snapshot = maze.ExperimentRegistry.get('t_maze').snapshot_observation()
    snapshot['evidence']['bad'] = {'type': type_, 'data': data}
    with pytest.raises(ObservationEnvelopeError, match='evidence.bad'):
        envelope(snapshot)


def test_record_scalar_types_and_ordered_intervals_refuse():
    producer = maze.ExperimentRegistry.get('t_maze')
    sample(producer, Fly(30.0, 50.0))
    snapshot = producer.snapshot_observation()
    record = snapshot['records']['cs_plus_entries']
    record.update(kind='event', unit='bool', value=1)
    with pytest.raises(ObservationEnvelopeError, match='event values must be bools'):
        envelope(snapshot)

    snapshot = producer.snapshot_observation()
    record = snapshot['records']['cs_plus_entries']
    record.update(kind='label', unit='label', value=True)
    with pytest.raises(ObservationEnvelopeError, match='label values must be strings'):
        envelope(snapshot)

    snapshot = producer.snapshot_observation()
    snapshot['records']['cs_plus_entries']['interval_rel_s'] = [2.0, 1.0]
    with pytest.raises(ObservationEnvelopeError, match='greater than or equal'):
        envelope(snapshot)


def test_real_manifest_identity_core_assay_mapping_and_extensions():
    for producer_assay, assay_id in PRODUCER_TO_ASSAY_ID.items():
        identity = identity_for(producer_assay)
        assert identity['assay'] == assay_id
        producer = (om.OpenArenaObserver() if producer_assay == 'open_arena'
                    else maze.ExperimentRegistry.get(producer_assay))
        built = envelope(producer.snapshot_observation(), identity=identity)
        assert built['identity'] == identity
        assert 'controller_id' not in built['identity']
        assert built['identity']['graph_io'] is None
        assert built['identity']['graph_io_extension']['version'] is None

    snapshot = maze.ExperimentRegistry.get('t_maze').snapshot_observation()
    for field in ('run_id', 'instance_id', 'assay', 'backend', 'controller_version',
                  'daemon_run_id', 'brain_id'):
        identity = identity_for('t_maze')
        identity.pop(field)
        with pytest.raises(ObservationEnvelopeError, match=field):
            envelope(snapshot, identity=identity)
    identity = identity_for('t_maze')
    identity['assay'] = 'heat-maze'
    with pytest.raises(ObservationEnvelopeError, match='identity.assay'):
        envelope(snapshot, identity=identity)
    identity = identity_for('t_maze')
    identity['activation'] = True
    with pytest.raises(ObservationEnvelopeError, match='activation'):
        envelope(snapshot, identity=identity)
    identity = identity_for('t_maze')
    identity['synthetic'] = 0
    with pytest.raises(ObservationEnvelopeError, match='synthetic'):
        envelope(snapshot, identity=identity)


def test_frozen_lifecycle_relationships_and_fault_validity_refuse_contradictions():
    producer = maze.ExperimentRegistry.get('t_maze')
    sample(producer, Fly(30.0, 50.0))
    fault = producer.freeze_observation('fault_halt')
    assert envelope(fault, validity='invalidated')['completeness'] == 'incomplete'

    wrong = copy.deepcopy(fault)
    wrong['completeness'] = 'complete'
    with pytest.raises(ObservationEnvelopeError, match='requires.*incomplete'):
        envelope(wrong, validity='invalidated')
    with pytest.raises(ObservationEnvelopeError, match='requires invalidated'):
        envelope(wrong, validity='valid')
    with pytest.raises(ObservationEnvelopeError, match='requires invalidated'):
        envelope(fault, validity='valid')

    interrupted = maze.ExperimentRegistry.get('t_maze').freeze_observation('manual_reset')
    interrupted['completeness'] = 'complete'
    with pytest.raises(ObservationEnvelopeError, match='requires.*incomplete'):
        envelope(interrupted)

    completed = maze.ExperimentRegistry.get('gap_crossing')
    completed.configure_observation(om.build_observation_config(
        completed.observation_spec(), config_id='deadline', override_window_s=0.025))
    sample(completed, dt=0.02)
    sample(completed, dt=0.02)
    completed_snapshot = completed.freeze_observation()
    completed_snapshot['completeness'] = 'incomplete'
    with pytest.raises(ObservationEnvelopeError, match='requires.*complete'):
        envelope(completed_snapshot)

    terminal = maze.ExperimentRegistry.get('t_maze').freeze_observation('manual_reset')
    terminal.update(end_reason='terminal_event:goal', terminal_event='other', completeness='complete')
    with pytest.raises(ObservationEnvelopeError, match='must agree'):
        envelope(terminal)

    continuous = om.OpenArenaObserver().freeze_observation('re_presentation_user')
    assert continuous['completeness'] == 'complete'
    assert envelope(continuous)['completeness'] == 'complete'


def test_frozen_terminal_pose_requires_finite_core_and_retains_extensions():
    frozen = maze.ExperimentRegistry.get('t_maze').freeze_observation('manual_reset')
    for pose in (None, 'not-a-pose', {'x_mm': 1.0, 'y_mm': 2.0},
                 {'x_mm': 1.0, 'y_mm': 2.0, 'heading_rad': float('inf')}):
        with pytest.raises(ObservationEnvelopeError, match='terminal_pose_post_step'):
            envelope(frozen, terminal_pose_post_step=pose)
    pose = {'x_mm': 1.0, 'y_mm': 2.0, 'heading_rad': 0.5,
            'z_mm': 0.4, 'step': 9, 'sample_point': 'post_solver'}
    assert envelope(frozen, terminal_pose_post_step=pose)['terminal_pose_post_step'] == pose


def test_absolute_timestamps_require_exact_reconstructed_values_at_large_origin():
    producer = maze.ExperimentRegistry.get('t_maze')
    sample(producer, Fly(30.0, 50.0))
    stamped = envelope(producer.freeze_observation('manual_reset'), segment_start_sim_s=1e12)
    assert validate_observation_envelope(json.loads(json.dumps(stamped, allow_nan=False))) == stamped
    wrong = copy.deepcopy(stamped)
    record = next(rec for rec in wrong['records'].values() if rec['interval_sim_s'] is not None)
    record['interval_sim_s'][1] += 0.5
    with pytest.raises(ObservationEnvelopeError, match='interval_sim_s'):
        validate_observation_envelope(wrong)
    wrong = copy.deepcopy(stamped)
    wrong['measurement_end_sim_s'] += 0.5
    with pytest.raises(ObservationEnvelopeError, match='measurement_end_sim_s'):
        validate_observation_envelope(wrong)


def test_window_elapsed_requires_actual_rounded_deadline_build_validate_and_dumps():
    early = maze.ExperimentRegistry.get('t_maze')
    sample(early, dt=0.02)
    interrupted = early.freeze_observation('manual_reset')
    false_deadline = copy.deepcopy(interrupted)
    false_deadline.update(end_reason='window_elapsed', completeness='complete')
    with pytest.raises(ObservationEnvelopeError, match='deadline to be reached'):
        envelope(false_deadline)

    stamped = envelope(interrupted)
    stamped.update(end_reason='window_elapsed', completeness='complete')
    with pytest.raises(ObservationEnvelopeError, match='deadline to be reached'):
        validate_observation_envelope(stamped)
    with pytest.raises(ObservationEnvelopeError, match='deadline to be reached'):
        dumps_observation_envelope(stamped)


def test_actual_live_frozen_clipped_and_nongrid_later_deadlines_pass():
    clipped = maze.ExperimentRegistry.get('buridan')
    clipped.configure_observation(om.build_observation_config(
        clipped.observation_spec(), config_id='clip-25ms', override_window_s=0.025))
    sample(clipped, dt=0.02)
    sample(clipped, dt=0.02)
    live = envelope(clipped.snapshot_observation())
    assert live['state'] == 'closed'
    assert live['measurement_end_rel_s'] == live['window_end_rel_s'] == 0.025
    frozen = envelope(clipped.freeze_observation())
    assert frozen['completeness'] == 'complete'

    later = maze.ExperimentRegistry.get('buridan')
    sample(later, dt=0.02)
    first = later.freeze_observation('policy_change')
    assert envelope(first)['completeness'] == 'incomplete'
    config = om.build_observation_config(
        later.observation_spec(), config_id='non-grid', override_window_s=0.0250004)
    later.begin_next_presentation('policy_change', config)
    sample(later, dt=0.02)
    sample(later, dt=0.02)
    later_live = envelope(later.snapshot_observation(), segment_start_sim_s=900.125)
    assert later_live['presentation_index'] == 1
    assert later_live['presentation_start_rel_s'] == 0.02
    assert later_live['effective_window_s'] == 0.0250004
    assert later_live['window_end_rel_s'] == later_live['measurement_end_rel_s'] == 0.045
    already_complete = later.freeze_observation('policy_change')
    assert already_complete['end_reason'] == 'window_elapsed'
    assert envelope(already_complete)['completeness'] == 'complete'


@pytest.mark.parametrize('mutation,match', [
    (lambda s: s.update(automatic_end=False), 'automatic finite window'),
    (lambda s: s.update(window_end_rel_s=s['window_end_rel_s'] + 0.001), 'window_end_rel_s'),
    (lambda s: s.update(measurement_end_rel_s=s['measurement_end_rel_s'] - 0.001),
     'cutoff must equal'),
    (lambda s: s.update(presentation_elapsed_s=0.02), 'observed presentation clock range|deadline to be reached'),
    (lambda s: s.update(state='observing'), 'closed state'),
    (lambda s: s.update(terminal_event='fabricated'), 'cannot carry a terminal event'),
])
def test_mutated_deadline_policy_endpoint_cutoff_and_state_refuse(mutation, match):
    producer = maze.ExperimentRegistry.get('buridan')
    producer.configure_observation(om.build_observation_config(
        producer.observation_spec(), config_id='deadline', override_window_s=0.025))
    sample(producer, dt=0.02)
    sample(producer, dt=0.02)
    snapshot = producer.freeze_observation()
    mutation(snapshot)
    if snapshot['automatic_end'] is False:
        snapshot['effective_window_s'] = snapshot['window_s'] = None
    with pytest.raises((ObservationEnvelopeError, om.ConfigError), match=match):
        envelope(snapshot)


def test_presentation_clock_range_terminal_hold_and_continuous_interruption_pass():
    bad = maze.ExperimentRegistry.get('t_maze')
    sample(bad, dt=0.02)
    snapshot = bad.freeze_observation('manual_reset')
    snapshot['measurement_end_rel_s'] = 0.03
    with pytest.raises(ObservationEnvelopeError, match='observed presentation clock range'):
        envelope(snapshot)
    snapshot = bad.freeze_observation('manual_reset')
    snapshot['presentation_start_rel_s'] = -0.001
    with pytest.raises(ObservationEnvelopeError, match='nonnegative'):
        envelope(snapshot)

    heat = maze.ExperimentRegistry.get('heat_maze')
    sample(heat, Fly(82.0, 78.0), dt=0.02)
    live_hold = envelope(heat.snapshot_observation())
    assert live_hold['state'] == 'measurement_ended'
    assert live_hold['terminal_event'] == 'refuge_entry'
    assert live_hold['measurement_end_rel_s'] == 0.0
    assert envelope(heat.freeze_observation())['completeness'] == 'complete'

    continuous = om.OpenArenaObserver()
    sample(continuous, dt=0.02)
    interrupted = envelope(continuous.freeze_observation('re_presentation_user'))
    assert interrupted['state'] == 'observing'
    assert interrupted['automatic_end'] is False
    assert interrupted['completeness'] == 'complete'


def test_provisional_fault_requires_invalidated_before_early_return_and_on_revalidation():
    producer = maze.ExperimentRegistry.get('t_maze')
    sample(producer, dt=0.02)
    snapshot = producer.snapshot_observation()
    snapshot.update(state='closed', end_reason='fault_halt', measurement_end_rel_s=0.02)
    with pytest.raises(ObservationEnvelopeError, match='requires invalidated'):
        envelope(snapshot, validity='valid')

    stamped = envelope(snapshot, validity='invalidated')
    assert all(rec['final'] is False for rec in stamped['records'].values())
    stamped['validity'] = 'valid'
    with pytest.raises(ObservationEnvelopeError, match='requires invalidated'):
        validate_observation_envelope(stamped)
    with pytest.raises(ObservationEnvelopeError, match='requires invalidated'):
        dumps_observation_envelope(stamped)


def test_re_presentation_completeness_uses_factual_rounded_elapsed_not_reason_label():
    before = maze.ExperimentRegistry.get('buridan')
    before.configure_observation(om.build_observation_config(
        before.observation_spec(), config_id='before', override_window_s=0.025))
    sample(before, dt=0.02)
    before_snapshot = before.freeze_observation('re_presentation_user')
    assert before_snapshot['completeness'] == 'incomplete'
    assert envelope(before_snapshot)['completeness'] == 'incomplete'

    after = maze.ExperimentRegistry.get('buridan')
    after.configure_observation(om.build_observation_config(
        after.observation_spec(), config_id='after', override_window_s=0.025))
    sample(after, dt=0.02)
    sample(after, dt=0.02)
    after_snapshot = after.freeze_observation('re_presentation_user')
    assert after_snapshot['end_reason'] == 're_presentation_user'
    assert after_snapshot['completeness'] == 'complete'
    assert after_snapshot['measurement_end_rel_s'] == 0.025
    assert envelope(after_snapshot)['completeness'] == 'complete'

    later = maze.ExperimentRegistry.get('buridan')
    sample(later, dt=0.02)
    later.freeze_observation('policy_change')
    config = om.build_observation_config(
        later.observation_spec(), config_id='later-rounded', override_window_s=0.0250004)
    later.begin_next_presentation('policy_change', config)
    sample(later, dt=0.02)
    sample(later, dt=0.02)
    later_snapshot = later.freeze_observation('re_presentation_user')
    assert later_snapshot['presentation_start_rel_s'] == 0.02
    assert later_snapshot['measurement_end_rel_s'] == 0.045
    assert later_snapshot['completeness'] == 'complete'
    assert envelope(later_snapshot)['completeness'] == 'complete'

    continuous = om.OpenArenaObserver()
    sample(continuous, dt=0.02)
    continuous_snapshot = continuous.freeze_observation('re_presentation_user')
    assert continuous_snapshot['automatic_end'] is False
    assert continuous_snapshot['completeness'] == 'complete'
    assert envelope(continuous_snapshot)['completeness'] == 'complete'
