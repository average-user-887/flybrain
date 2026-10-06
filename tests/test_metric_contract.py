"""Metric contract C0 producer vectors (metric-contract/1.2).

Every vector drives the producer directly: synthetic flies, ``begin_sample`` / ``step`` /
``observe_contact`` and an injected ``provenance``. No daemon is involved. Vector ids
are in the test names (``test_TM_7_...``); docs/ASSAY_SEMANTICS.md has the map.
"""
import copy
import hashlib
import json
import math
import os
import pickle
from types import SimpleNamespace

import pytest

import assay_controls
import maze
import online_metrics as om
from arena import Arena
from assay_controls import act, preflight, set_parameter
from experiment_brains import PARADIGMS
from online_metrics import (ConfigError, MetricFault, OpenArenaObserver, build_observation_config,
                            validate_finite)

DT = 0.02
ALL_STATES = ['SURGE', 'CAST', 'REST', 'SOCIAL_APPROACH', 'SOCIAL_AVOID', 'COURTSHIP', 'PROBE', 'ABORT']
GRAPH_STATES = ['FORWARD', 'TURN', 'STOP']


class Fly:
    """A synthetic fly with the I-10 fields; speed defaults to 0 (never the legacy 1.2 default)."""

    def __init__(self, x=0.0, y=0.0, heading=0.0, speed=0.0, angular_velocity=0.0, **kw):
        self.pos = SimpleNamespace(x=float(x), y=float(y))
        self.heading = heading
        self.speed = speed
        self.angular_velocity = angular_velocity
        self.assay_escape_remaining = 0.0
        self.behavioral_state = ''
        for k, v in kw.items():
            setattr(self, k, v)

    def at(self, x, y):
        self.pos.x, self.pos.y = float(x), float(y)
        return self


def new(key, **kw):
    return maze.ExperimentRegistry.get(key, **kw)


def run(p, fly, seconds=None, n=None, dt=DT, each=None):
    n = n if n is not None else int(round(seconds / dt))
    for i in range(n):
        if each:
            each(i, fly)
        p.step(fly, dt)


def rec(p, name):
    return p.get_metric_records()[name]


def t(p):
    return p.observation_status()['segment_elapsed_s']


def fake_arena(p, fly=None):
    return SimpleNamespace(paradigm=p, fly=fly or Fly(), paradigm_key=Arena.paradigm_key, wind=(0.0, 0.0))


def represent(p, reason, end_reason='re_presentation_user', mutate=None, config=None):
    """The daemon's order: freeze, apply, begin the next presentation in place."""
    env = p.freeze_observation(end_reason)
    if mutate:
        mutate()
    p.begin_next_presentation(reason, config)
    return env


def config(p, cid, **kw):
    return build_observation_config(p.observation_spec(), config_id=cid, **kw)


def assert_unavailable(r, reason):
    assert r['value'] is None and r['available'] is False and r['reason'] == reason, r


def assert_value(r, value, approx=False):
    assert r['available'] is True and r['reason'] is None, r
    if approx:
        assert r['value'] == pytest.approx(value)
    else:
        assert r['value'] == value, r


def json_ok(*objs):
    for o in objs:
        json.dumps(o, allow_nan=False)
        assert validate_finite(o) is None


# ============================================================================
# Schema vectors S-1 ... S-13
# ============================================================================
EXPECTED = {  # assay: (mode, window_s, hold_s, re_presentation_triggers) -- v1 §9.1
    'open_arena': ('continuous', None, 0.0, ['windStrength']),
    't_maze': ('fixed', 120.0, 0.0, ['reverse_arms']),
    'y_maze': ('fixed', 300.0, 0.0, []),
    'heat_maze': ('until_terminal', 300.0, 2.0, ['floorTemp']),
    'buridan': ('fixed', 300.0, 0.0, ['rotate_stripes', 'contrast']),
    'visual_operant': ('presentation', 120.0, 0.0, ['reverse_heat']),
    'wind_tunnel': ('until_terminal', 120.0, 1.0, ['windVelocity', 'plumeWidth', 'shift_plume']),
    'looming_escape': ('presentation', 2.0, 0.0, ['loom']),
    'optomotor': ('presentation', 60.0, 0.0, ['patternSpeed', 'reverse_grating', 'contrast']),
    'gap_crossing': ('until_terminal', 120.0, 1.0, ['gapWidth']),
    'circadian_dam': ('continuous', None, 0.0, []),
    'courtship': ('fixed', 600.0, 0.0, ['receptivity']),
    'labyrinth': ('until_terminal', 300.0, 2.0, []),
    'multisensory': ('fixed', 120.0, 0.0, []),
}


def all_producers():
    return [new(k) for k in maze.ExperimentRegistry.list_paradigms()] + [OpenArenaObserver()]


def test_S_1_all_14_specs_static_finite_and_windows_exact():
    specs = {s['assay']: s for s in (p.observation_spec() for p in all_producers())}
    assert set(specs) == set(EXPECTED)
    again = {s['assay']: s for s in (p.observation_spec() for p in all_producers())}
    assert specs == again
    for assay, (mode, window, hold, triggers) in EXPECTED.items():
        s = specs[assay]
        json_ok(s)
        assert (s['mode'], s['window_s'], s['hold_s'], s['re_presentation_triggers']) == (mode, window, hold, triggers)
        assert s['learning_claim'] == 'none' and s['schema'] == 'neurofly.metric/1.2'
        assert s['contract'] == 'metric-contract/1.2'
        assert s['spec_version'] == f'{assay}/1.2'
        assert s['headline_metric'] and isinstance(s['companion_metrics'], list)
    # every connected live control that is a declared trigger is a real control of that assay
    for pid in PARADIGMS:
        a = Arena(paradigm=None if pid == 'open-arena' else pid, seed=4, num_flies=1, num_predators=0)
        names = {c['name'] for c in assay_controls.describe(a)['parameters'] + assay_controls.describe(a)['actions']}
        spec = (a.paradigm or OpenArenaObserver()).observation_spec()
        assert set(spec['re_presentation_triggers']) <= names

    for producer in all_producers():
        envelope = producer.freeze_observation('manual_reset')
        assay = envelope['assay']
        assert envelope['schema'] == 'neurofly.metric/1.2'
        assert envelope['contract'] == 'metric-contract/1.2'
        assert envelope['spec_version'] == f'{assay}/1.2'


def test_S_2_vocabularies_are_closed():
    with pytest.raises(ValueError):
        om.record(1, kind='bogus', unit='s')
    with pytest.raises(ValueError):
        om.record(1, kind='count', unit='furlongs')
    with pytest.raises(ValueError):
        om.record(kind='count', unit='count', reason='maybe')
    with pytest.raises(ValueError):
        om.record(1, kind='count', unit='count', capability='guessed')
    with pytest.raises(ValueError):
        om.record(1, kind='count', unit='count', scope='run')
    with pytest.raises(ValueError):
        om.evidence('picture', {})
    with pytest.raises(ValueError):
        om.completeness_for('tired')
    with pytest.raises(ConfigError):
        om.validate_observation_config({'config_id': 'x', 'effective_window_s': 1.0, 'automatic_end': True,
                                        'window_source': 'vibes', 'override': None, 'hold_s': 0.0})
    assert om.completeness_for('policy_change') == 'incomplete'
    integral = om.record(1.0, kind='integral', unit='degC*s')
    assert integral['kind'] == 'integral' and integral['unit'] == 'degC*s'

    unknown = new('t_maze').observation_spec()
    unknown['schema'] = 'neurofly.metric/99'
    with pytest.raises(ValueError, match='unsupported metric schema/contract'):
        om.check_spec(unknown)
    old = new('t_maze').observation_spec()
    old.update(schema='neurofly.metric/1.1', contract='metric-contract/1.1',
               spec_version='t_maze/1.1')
    with pytest.raises(ValueError, match='unsupported metric schema/contract'):
        om.check_spec(old)
    fabricated = new('t_maze').observation_spec()
    fabricated.update(assay='fabricated_assay', spec_version='fabricated_assay/1.2')
    with pytest.raises(ValueError, match='unsupported assay identity'):
        om.check_spec(fabricated)
    for invalid_assay in (None, '', 12, ['t_maze'], {'name': 't_maze'}):
        nonstring = new('t_maze').observation_spec()
        nonstring['assay'] = invalid_assay
        with pytest.raises(ValueError, match='unsupported assay identity'):
            om.check_spec(nonstring)

    wrong_assay_version = new('t_maze').observation_spec()
    wrong_assay_version['spec_version'] = 't_maze/99'
    with pytest.raises(ValueError, match='unsupported assay spec version'):
        om.check_spec(wrong_assay_version)
    nonstring_version = new('t_maze').observation_spec()
    nonstring_version['spec_version'] = 12
    with pytest.raises(ValueError, match='unsupported assay spec version'):
        om.check_spec(nonstring_version)


def test_S_3_validate_finite_returns_nested_paths():
    nan = float('nan')
    assert validate_finite({'counts': {'a': 1, 'b': nan}}) == 'counts.b'
    ev = {'data': [{'attrs': {'theta': 1.0}}, {'attrs': {'theta': float('inf')}}]}
    assert validate_finite(ev) == 'data[1].attrs.theta'
    assert validate_finite({'interval_rel_s': [0.0, nan]}) == 'interval_rel_s[1]'
    assert validate_finite({'ok': [1, 2.0, None, 'x', True]}) is None


def test_S_4_unexpected_nonfinite_raises_metricfault():
    with pytest.raises(MetricFault):
        om.record(float('nan'), kind='mean', unit='s')
    with pytest.raises(MetricFault) as e:
        om.record(1, kind='count', unit='count', counts={'n': float('inf')})
    assert e.value.path == 'counts.n'
    with pytest.raises(MetricFault):
        om.evidence('event_sequence', [om.event('x', 0.0, 1, theta=float('nan'))])
    p = new('t_maze')
    out = p.step(Fly(float('nan'), 20.0), DT)  # the legacy step still returns normally
    assert isinstance(out, dict)
    with pytest.raises(MetricFault):
        p.get_metric_records()


def degenerate_states():
    """(label, producer) for every assay in fresh / one-sample / long / interrupted states."""
    out = []
    for make in [lambda: new(k) for k in maze.ExperimentRegistry.list_paradigms()] + [OpenArenaObserver]:
        p = make()
        out.append(('fresh', p))
        p = make()
        step(p, Fly(30, 30))
        out.append(('one_sample', p))
        p = make()
        for _ in range(60):
            step(p, Fly(30, 30, speed=1.0))
        out.append(('sixty_samples', p))
    return out


def step(p, fly, dt=DT, contact=None):
    if isinstance(p, OpenArenaObserver):
        p.observe(fly, bool(contact), dt)
    else:
        p.step(fly, dt)
        if contact is not None:
            p.observe_contact(p._v1_last_step, dt, 'post_solver', contact)


def check_record_invariants(recs):
    for name, r in recs.items():
        assert (r['value'] is not None) == r['available'] == (r['reason'] is None), (name, r)   # S-6
        assert not isinstance(r['value'], (list, dict, tuple, set)), name                         # S-7
        assert r['scope'] in om.SCOPES, name                                                       # I-20
        assert r['unit'] in om.UNITS and r['kind'] in om.KINDS
        if r['capability'] == 'unsupported':
            assert r['reason'] == 'unsupported'
        if name in om.OUTCOME_BOOLEANS:                                                            # S-13
            assert not (r['value'] is False and r['reason'] is not None)


@pytest.mark.parametrize('end_reason', [None, 'manual_reset', 'fault_halt', 'experiment_selected'])
def test_S_5_S_6_S_7_S_8_degenerate_states_are_finite_json_and_consistent(end_reason):
    for label, p in degenerate_states():
        recs, ev, status, spec = p.get_metric_records(), p.get_evidence(), p.observation_status(), p.observation_spec()
        json_ok(recs, ev, status, spec)
        check_record_invariants(recs)
        assert all(r['final'] is False for r in recs.values())                                    # S-8
        for item in ev.values():
            assert item['type'] in om.EVIDENCE_TYPES
        if end_reason is not None:
            env = p.freeze_observation(end_reason)
            json_ok(env)
            check_record_invariants(env['records'])
            assert all(r['final'] is True for r in env['records'].values())
            assert env['completeness'] == 'incomplete'
            assert all(r['final'] is False for r in p.get_metric_records().values())  # live view untouched


def test_S_8_final_is_set_only_by_the_freeze_helper():
    recs = {'a': om.record(1, kind='count', unit='count')}
    assert recs['a']['final'] is False
    frozen = om.freeze_records(recs)
    assert frozen['a']['final'] is True and recs['a']['final'] is False


def test_S_9_state_derived_records_unsupported_without_provenance():
    for key, names in (('wind_tunnel', ['surge_s', 'cast_s', 'rest_s', 'other_state_s', 'surge_cast_ratio']),
                       ('courtship', ['approach_s', 'avoid_s', 'courtship_state_s', 'courtship_state_fraction'])):
        for provenance in (None, {}, {'controller_states': GRAPH_STATES}):
            p = new(key)
            p.provenance = provenance
            run(p, Fly(20, 50), n=10)
            recs = p.get_metric_records()
            for name in names:
                assert recs[name]['capability'] == 'unsupported' and recs[name]['reason'] == 'unsupported', name
                assert recs[name]['note'].startswith('controller does not emit')
            assert set(p.observation_spec()['required_states']) == set(names)


def test_S_10_begin_next_presentation_resets_timing_and_only_presentation_state():
    p = OpenArenaObserver()
    for i in range(50):
        p.observe(Fly(i * 0.1, 0), i == 10, DT)
    p.log_intervention('food', None)
    p.begin_next_presentation('user_windStrength')
    st = p.observation_status()
    assert st['presentation_index'] == 1 and st['presentation_start_rel_s'] == 1.0
    assert st['presentation_elapsed_s'] == 0.0 and st['segment_elapsed_s'] == 1.0
    recs = p.get_metric_records()
    assert recs['food_contacts']['value'] == 1                       # segment-scoped, kept
    assert_unavailable(recs['presentation_food_contacts'], 'not_observed')  # presentation-scoped, cleared
    assert [e['name'] for e in p.get_evidence()['interventions']['data']] == ['food']
    tm = new('t_maze')
    run(tm, Fly(30, 50), n=5)
    tm.begin_next_presentation('user_reverse_arms')
    assert tm.observation_status()['presentation_start_rel_s'] == 0.1
    assert rec(tm, 'cs_plus_entries')['reason'] == 'not_observed'


def test_S_11_premotor_times_before_dt_and_postsolver_contacts_plus_dt():
    p = new('labyrinth')
    fly = Fly(30, 75)  # inside dead_end_1
    for i, step_id in enumerate((501, 502, 503)):
        p.begin_sample(step_id, DT)
        p.step(fly, DT)
        p.observe_contact(step_id, DT, 'post_solver', i == 1)
    ev = p.get_evidence()['contact_events']['data']
    assert ev == [{'event': 'contact_onset', 't_rel_s': 0.04, 'step': 502, 'sample_point': 'post_solver',
                   'attrs': {'source': 'solver'}}]
    tm = new('t_maze')
    for i, step_id in enumerate((7, 8, 9)):
        tm.begin_sample(step_id, DT)
        tm.step(Fly(70, 20) if i < 2 else Fly(30, 50), DT)
    entry = tm.get_evidence()['entries']['data'][0]
    assert entry['t_rel_s'] == 0.04 and entry['step'] == 9 and entry['sample_point'] == 'pre_motor'
    with pytest.raises(ValueError):
        p.observe_contact(999, DT, 'post_solver', True)  # not the step just sampled


def test_S_12_legacy_get_metrics_unchanged_against_713ba82_fixture():
    from tests.fixtures.metric_c0_legacy_driver import drive
    path = os.path.join(os.path.dirname(__file__), 'fixtures', 'metric_c0_legacy_713ba82.json')
    with open(path, 'rb') as fh:
        fixture_bytes = fh.read()
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        'c2f0274cac53648398429436fb1997b25c7618652c5e48d0befbe7b936915fc9')
    fixture = json.loads(fixture_bytes)
    assert fixture['source'] == '713ba82:maze.py'
    current = json.loads(json.dumps(drive(maze), sort_keys=True, allow_nan=False))
    for key, expected in fixture['paradigms'].items():
        assert current[key]['metrics'] == expected['metrics'], key
        assert current[key]['checkpoints'] == expected['checkpoints'], key


# ---- S-13: the deadline-outcome invariant over every outcome boolean -------------
def short_window(p, seconds):
    p.configure_observation(config(p, 'short', override_window_s=seconds, override_source='api'))


def outcome_cases():
    """(assay, record, situation) -> record, built from the five v1.1 A5 situations."""
    cases = {}

    def five(assay, name, make, occur, nothing):
        p = make(); nothing(p, 5); cases[(assay, name, 'pending')] = rec(p, name)
        p = make(); occur(p); cases[(assay, name, 'occurred')] = rec(p, name)
        p = make(); short_window(p, 1.0); nothing(p, 60); cases[(assay, name, 'completed')] = rec(p, name)
        p = make(); nothing(p, 5); cases[(assay, name, 'interrupted')] = p.freeze_observation('manual_reset')['records'][name]
        p = make(); nothing(p, 5); cases[(assay, name, 'fault')] = p.freeze_observation('fault_halt')['records'][name]

    five('heat_maze', 'refuge_reached', lambda: new('heat_maze'),
         lambda p: run(p, Fly(82, 78), n=1), lambda p, n: run(p, Fly(40, 40), n=n))
    five('wind_tunnel', 'source_reached', lambda: new('wind_tunnel'),
         lambda p: run(p, Fly(180, 30), n=1), lambda p, n: run(p, Fly(20, 50), n=n))
    five('labyrinth', 'goal_reached', lambda: new('labyrinth'),
         lambda p: run(p, Fly(130, 85), n=1), lambda p, n: run(p, Fly(10, 10), n=n))

    def looming():
        p = new('looming_escape')
        p.gf_source = 'connectome'
        return p
    five('looming_escape', 'escape_initiated', lambda: new('looming_escape'),
         lambda p: run(p, Fly(40, 40), n=25), lambda p, n: (setattr(p, 'gf_source', 'connectome'),
                                                             run(p, Fly(40, 40), n=n)))

    def escape_done(p):
        fly = Fly(40, 40)
        run(p, fly, n=20)
        fly.assay_escape_remaining = 0.1
        run(p, fly, n=3)
        fly.assay_escape_remaining = 0.0
        run(p, fly, n=1)
    five('looming_escape', 'escape_completed', lambda: new('looming_escape'), escape_done,
         lambda p, n: run(p, Fly(40, 40), n=n))
    five('gap_crossing', 'crossing_success', lambda: new('gap_crossing'),
         lambda p: run(p, Fly(50, 10), n=1), lambda p, n: run(p, Fly(20, 10), n=n))

    def turned(p):
        p2 = p
        p2.gap_width_mm = 5.0
        p2.zones[1].bounds = (45.0, 0.0, 50.0, 20.0)
        p2.zones[2].bounds = (50.0, 7.5, 100.0, 12.5)
        run(p2, Fly(43.5, 10), n=2)
        run(p2, Fly(43.5, 10, heading=math.pi), n=1)
    five('gap_crossing', 'turn_complete', lambda: new('gap_crossing', gap_width_mm=5.0), turned,
         lambda p, n: run(p, Fly(20, 10), n=n))
    # Completed at ANOTHER declared terminal event (v1.1 A5 row 3).
    p = new('gap_crossing', gap_width_mm=5.0)
    turned(p)
    cases[('gap_crossing', 'crossing_success', 'other_terminal')] = rec(p, 'crossing_success')
    p = new('gap_crossing')
    run(p, Fly(50, 10), n=1)
    cases[('gap_crossing', 'turn_complete', 'other_terminal')] = rec(p, 'turn_complete')
    return cases


def test_S_13_deadline_outcome_invariant_across_all_outcome_booleans():
    cases = outcome_cases()
    names = {(a, n) for a, n, _ in cases}
    assert {n for _, n in names} == set(om.OUTCOME_BOOLEANS)
    for (assay, name, situation), r in cases.items():
        assert not (r['value'] is False and r['reason'] is not None), (assay, name, situation)
        expected = {'pending': (None, False, 'pending'), 'occurred': (True, True, None),
                    'completed': (False, True, None), 'other_terminal': (False, True, None),
                    'interrupted': (None, False, 'censored'), 'fault': (None, False, 'invalidated')}[situation]
        assert (r['value'], r['available'], r['reason']) == expected, (assay, name, situation, r)


# ============================================================================
# open_arena (OpenArenaObserver)
# ============================================================================
def oa_run(o, contacts, dt=DT, airflow=None):
    for c in contacts:
        o.observe(Fly(10, 10), c, dt, airflow_mm_s=airflow)


def test_OA_1_consecutive_contact_samples_count_once():
    o = OpenArenaObserver()
    oa_run(o, [True, True, True])
    assert_value(rec(o, 'food_contacts'), 1)


def test_OA_2_contact_release_contact_counts_two():
    o = OpenArenaObserver()
    oa_run(o, [True, False, True])
    assert_value(rec(o, 'food_contacts'), 2)


def test_OA_3_no_contact_is_a_valid_negative_and_latency_pending():
    o = OpenArenaObserver()
    oa_run(o, [False] * 1200, dt=0.1)
    assert_value(rec(o, 'food_contacts'), 0)
    assert_unavailable(rec(o, 'time_to_first_contact_s'), 'pending')


def test_OA_4_observation_s_is_sum_of_variable_dt():
    o = OpenArenaObserver()
    dts = [0.01, 0.03, 0.02, 0.017] * 25
    for d in dts:
        o.observe(Fly(0, 0), False, d)
    assert rec(o, 'observation_s')['value'] == pytest.approx(sum(dts), abs=1e-9)


def test_OA_5a_continuous_never_ends_automatically():
    o = OpenArenaObserver()
    spec = o.observation_spec()
    assert spec['mode'] == 'continuous' and spec['window_s'] is None
    assert o.observation_config()['automatic_end'] is False
    for _ in range(36000):
        o.observe(Fly(0, 0), False, 0.1)
        assert o._v1_end is None
    st = o.observation_status()
    assert st['state'] == 'observing' and st['presentation_index'] == 0
    assert st['segment_elapsed_s'] == pytest.approx(3600.0) and st['window_end_rel_s'] is None


def test_OA_5b_airflow_change_closes_presentation_complete():
    o = OpenArenaObserver()
    oa_run(o, [False] * 1500, airflow=0.0)
    env = represent(o, 'user_windStrength', mutate=lambda: setattr(o, 'airflow_mm_s', 20.0))
    assert env['completeness'] == 'complete' and env['end_reason'] == 're_presentation_user'
    assert env['records']['presentation_observation_s']['interval_rel_s'] == [0.0, 30.0]
    assert env['records']['airflow_mm_s']['value'] == 0.0
    oa_run(o, [False] * 10, airflow=20.0)
    st = o.observation_status()
    assert st['presentation_index'] == 1 and st['presentation_start_rel_s'] == 30.0
    assert_value(rec(o, 'airflow_mm_s'), 20.0)


def test_OA_6_interruption_without_contact_censors_latency():
    o = OpenArenaObserver()
    oa_run(o, [False] * 250)
    r = o.freeze_observation('manual_reset')['records']['time_to_first_contact_s']
    assert_unavailable(r, 'censored')
    assert r['counts']['censored_at_rel_s'] == 5.0 and r['counts']['censored_after_s'] == 5.0


def test_OA_7_segment_and_presentation_scopes_are_separate():
    o = OpenArenaObserver()
    def seq(t0, t1):
        return [round(t0 / DT) + i in (500, 2000) for i in range(round((t1 - t0) / DT))]
    oa_run(o, seq(0, 30))
    env0 = represent(o, 'user_windStrength')
    oa_run(o, seq(30, 50))
    recs = o.get_metric_records()
    assert env0['records']['presentation_food_contacts']['value'] == 1
    assert recs['presentation_food_contacts']['value'] == 1
    assert recs['food_contacts']['value'] == 2 and recs['food_contacts']['scope'] == 'segment'
    assert recs['presentation_food_contacts']['scope'] == 'presentation'
    assert_value(recs['time_to_first_contact_s'], 10.0)
    assert env0['records']['food_contacts']['value'] == 1  # segment snapshot at the freeze


def test_OA_8_food_placement_is_logged_and_closes_nothing():
    a = Arena(paradigm=None, seed=4, num_flies=1, num_predators=0)
    pf = preflight(a, 'assay_action', 'food')
    assert pf['accepted'] and pf['closes_presentation'] is False and pf['intervention_only'] is True
    o = OpenArenaObserver()
    oa_run(o, [False] * 1000)
    o.log_intervention('food', None)
    oa_run(o, [False] * 10)
    assert o.observation_status()['presentation_index'] == 0
    assert o.get_evidence()['interventions']['data'] == [{'name': 'food', 'value': None, 't_rel_s': 20.0,
                                                          'step': 1000}]


# ============================================================================
# t_maze
# ============================================================================
STEM, HUB, CSP, CSM = (70, 20), (70, 50), (30, 50), (110, 50)


def test_TM_1_no_entry_in_120_s():
    p = new('t_maze')
    run(p, Fly(*STEM), seconds=121, dt=0.1)
    st = p.observation_status()
    assert st['state'] == 'closed' and st['measurement_end_rel_s'] == 120.0
    r = rec(p, 'arm_entry_preference_index')
    assert_unavailable(r, 'zero_denominator')
    assert r['counts'] == {'cs_plus': 0, 'cs_minus': 0} and r['denominator'] == 0
    fc = rec(p, 'first_choice')
    assert_unavailable(fc, 'censored')
    assert fc['counts']['censored_at_rel_s'] == 120.0 and fc['counts']['censored_after_s'] == 120.0
    assert_unavailable(rec(p, 'first_choice_latency_s'), 'censored')


def test_TM_2_and_TM_3_single_entries():
    p = new('t_maze')
    run(p, Fly(*CSP), n=1)
    assert_value(rec(p, 'arm_entry_preference_index'), 1.0)
    p = new('t_maze')
    run(p, Fly(*CSM), n=1)
    assert_value(rec(p, 'arm_entry_preference_index'), -1.0)


def test_TM_4_plus_hub_minus_gives_zero_available():
    p = new('t_maze')
    for xy in (CSP, HUB, CSM):
        run(p, Fly(*xy), n=5)
    r = rec(p, 'arm_entry_preference_index')
    assert_value(r, 0.0)
    assert r['denominator'] == 2


def test_TM_5_dwelling_in_an_arm_is_one_entry():
    p = new('t_maze')
    run(p, Fly(*CSP), seconds=30)
    assert_value(rec(p, 'cs_plus_entries'), 1)


def test_TM_6_first_choice_is_immutable_with_its_own_latency_and_pose():
    p = new('t_maze')
    run(p, Fly(*STEM), n=50)
    run(p, Fly(*CSP), n=5)
    run(p, Fly(*HUB), n=5)
    run(p, Fly(*CSM), n=5)
    assert_value(rec(p, 'first_choice'), 'cs_plus')
    assert_value(rec(p, 'first_choice_latency_s'), 1.0)
    pose = p.get_evidence()['first_choice_pose']['data']
    assert pose['t_rel_s'] == 1.0 and pose['x_mm'] == 30.0 and pose['sample_point'] == 'pre_motor'


def test_TM_7_reversal_closes_presentation_with_condition_specific_counts():
    p = new('t_maze')
    fly = Fly(*STEM)
    run(p, fly, n=100)
    run(p, fly.at(*CSP), n=10)
    run(p, fly.at(*STEM), seconds=50 - t(p))
    assert t(p) == 50.0
    arena = fake_arena(p, fly)
    pf = preflight(arena, 'assay_action', 'reverse_arms')
    assert pf['accepted'] and pf['closes_presentation']
    env = represent(p, 'user_reverse_arms', mutate=lambda: act(arena, 'reverse_arms'))
    assert env['completeness'] == 'incomplete' and env['presentation_index'] == 0
    assert env['records']['arm_entry_preference_index']['value'] == 1.0
    assert env['records']['arm_entry_preference_index']['interval_rel_s'] == [0.0, 50.0]
    run(p, fly, n=1)
    st = p.observation_status()
    assert st['presentation_index'] == 1 and st['presentation_start_rel_s'] == 50.0
    assert st['window_end_rel_s'] == 170.0
    assert rec(p, 'arm_entry_preference_index')['counts'] == {'cs_plus': 0, 'cs_minus': 0}
    assert_unavailable(rec(p, 'first_choice'), 'pending')


# ============================================================================
# y_maze
# ============================================================================
CENTRE = (60.0, 60.0)
ARM = {'A': (60.0, 100.0), 'B': (60 + 40 * math.cos(7 * math.pi / 6), 60 + 40 * math.sin(7 * math.pi / 6)),
       'C': (60 + 40 * math.cos(11 * math.pi / 6), 60 + 40 * math.sin(11 * math.pi / 6))}


def ym_visit(p, *arms):
    for a in arms:
        run(p, Fly(*(CENTRE if a == 'hub' else ARM[a])), n=3)


def test_YM_1_A_hub_B_insufficient_events_two_physical():
    p = new('y_maze')
    ym_visit(p, 'A', 'hub', 'B')
    assert_unavailable(rec(p, 'spontaneous_alternation_rate'), 'insufficient_events')
    assert_value(rec(p, 'physical_entries'), 2)


def test_YM_2_ABC_is_full_alternation():
    p = new('y_maze')
    ym_visit(p, 'A', 'hub', 'B', 'hub', 'C')
    r = rec(p, 'spontaneous_alternation_rate')
    assert_value(r, 1.0)
    assert (r['numerator'], r['denominator']) == (1, 1)


def test_YM_3_ABA_is_zero_available():
    p = new('y_maze')
    ym_visit(p, 'A', 'hub', 'B', 'hub', 'A')
    r = rec(p, 'spontaneous_alternation_rate')
    assert_value(r, 0.0)
    assert (r['numerator'], r['denominator']) == (0, 1)


def test_YM_4_no_entries():
    p = new('y_maze')
    ym_visit(p, 'hub')
    sar = rec(p, 'spontaneous_alternation_rate')
    assert_unavailable(sar, 'insufficient_events')
    assert sar['counts'] == {'collapsed_length': 0, 'physical_entries': 0}
    assert_unavailable(rec(p, 'handedness_index'), 'zero_denominator')


def test_YM_5_boundary_oscillation_is_one_visit():
    p = new('y_maze')
    run(p, Fly(*CENTRE), n=1)
    run(p, Fly(60, 85), seconds=40, each=lambda i, f: f.at(60, 84 if i % 2 else 86))
    assert_value(rec(p, 'physical_entries'), 1)


def test_YM_6_true_exit_and_reentry_is_a_new_physical_visit():
    p = new('y_maze')
    ym_visit(p, 'A')
    run(p, Fly(60, 71.5), n=2)  # 11.5 mm from the centre: re-armed
    ym_visit(p, 'A')
    assert_value(rec(p, 'physical_entries'), 2)
    ev = p.get_evidence()
    assert ev['physical_visit_sequence']['data']['symbols'] == ['A', 'A']
    assert ev['alternation_sequence']['data'] == {'symbols': ['A'], 'collapse': 'consecutive_repeats'}
    assert_unavailable(rec(p, 'spontaneous_alternation_rate'), 'insufficient_events')
    p2 = new('y_maze')
    ym_visit(p2, 'A')
    run(p2, Fly(60, 72.5), n=2)  # 12.5 mm: not re-armed
    ym_visit(p2, 'A')
    assert_value(rec(p2, 'physical_entries'), 1)


# ============================================================================
# heat_maze
# ============================================================================
REFUGE, HOT = (82, 78), (40, 40)


def test_HM_1_refuge_at_14_s_ends_measurement_and_excludes_the_hold():
    p = new('heat_maze')
    fly = Fly(*HOT)
    run(p, fly, n=700)
    run(p, fly.at(*REFUGE), n=1)
    st = p.observation_status()
    assert st['state'] == 'measurement_ended' and st['measurement_end_rel_s'] == 14.0
    assert st['terminal_event'] == 'refuge_entry'
    frozen = p.freeze_observation()
    assert frozen['end_reason'] == 'terminal_event:refuge_entry' and frozen['completeness'] == 'complete'
    assert_value(frozen['records']['escape_latency_s'], 14.0)
    states = []
    for i in range(99):  # the rest of the 2 s hold (it ends at 16.00), moving around the hot floor
        p.step(fly.at(30 + i * 0.2, 40), DT)
        states.append(p.observation_status()['state'])
    assert states[:-1] == ['measurement_ended'] * 98 and states[-1] == 'closed' and t(p) == 16.0
    live = p.get_metric_records()
    for name, r in frozen['records'].items():
        assert {**r, 'final': False} == live[name], name


def test_HM_2_no_refuge_in_300_s():
    p = new('heat_maze')
    run(p, Fly(*HOT), seconds=300, dt=0.1)
    lat = rec(p, 'escape_latency_s')
    assert_unavailable(lat, 'censored')
    assert lat['counts']['censored_at_rel_s'] == 300.0
    assert_value(rec(p, 'refuge_reached'), False)
    assert rec(p, 'thermal_dose_degC_s')['value'] > 0


def test_HM_3_quadrant_fraction_not_observed_before_any_sample():
    assert_unavailable(rec(new('heat_maze'), 'target_quadrant_fraction'), 'not_observed')


def test_HM_3b_thermal_dose_is_integral_with_unchanged_numeric_definition():
    p = new('heat_maze')
    unavailable = rec(p, 'thermal_dose_degC_s')
    assert_unavailable(unavailable, 'not_observed')
    assert (unavailable['kind'], unavailable['unit']) == ('integral', 'degC*s')

    fly = Fly(*HOT)
    excess = max(0.0, p.peltier.get_temperature(fly.pos.x, fly.pos.y) - 25.0)
    p.step(fly, DT)
    available = rec(p, 'thermal_dose_degC_s')
    assert (available['kind'], available['unit']) == ('integral', 'degC*s')
    assert_value(available, excess * DT, approx=True)


def test_HM_4_place_learning_unsupported_in_every_state():
    p = new('heat_maze')
    assert rec(p, 'place_learning')['capability'] == 'unsupported'
    run(p, Fly(*REFUGE), n=3)
    assert rec(p, 'place_learning')['capability'] == 'unsupported'
    assert p.freeze_observation()['records']['place_learning']['reason'] == 'unsupported'


def test_HM_5_manual_interruption_at_50_s():
    p = new('heat_maze')
    run(p, Fly(*HOT), seconds=50, dt=0.1)
    env = p.freeze_observation('manual_reset')
    assert env['completeness'] == 'incomplete'
    lat = env['records']['escape_latency_s']
    assert_unavailable(lat, 'censored')
    assert lat['counts']['censored_at_rel_s'] == 50.0
    assert_unavailable(env['records']['refuge_reached'], 'censored')
    assert env['records']['path_length_mm']['available']  # measured prefix kept


# ============================================================================
# buridan
# ============================================================================
def test_BU_1_no_samples_not_observed():
    recs = new('buridan').get_metric_records()
    for name in ('heading_alignment', 'walking_stripe_alignment', 'centre_fraction', 'centrophobism_index',
                 'observation_s', 'walking_s', 'path_length_mm', 'displacement_mm'):
        assert_unavailable(recs[name], 'not_observed')


def test_BU_2_stationary_aligned_300_s():
    p = new('buridan')
    run(p, Fly(60, 60, heading=0.0, speed=0.0), seconds=300, dt=0.1)
    recs = p.get_metric_records()
    assert_value(recs['heading_alignment'], 1.0, approx=True)
    assert_unavailable(recs['walking_stripe_alignment'], 'insufficient_events')
    assert_value(recs['displacement_mm'], 0.0)
    assert recs['stripe_traversals']['capability'] == 'unsupported'


def test_BU_3_walking_60_s_gives_walking_alignment():
    p = new('buridan')
    run(p, Fly(60, 60, heading=0.0, speed=2.0), seconds=60)
    assert rec(p, 'walking_stripe_alignment')['available']
    assert rec(p, 'walking_s')['value'] == pytest.approx(60.0)


def test_BU_4_rotate_stripes_is_a_new_presentation_never_a_traversal():
    p = new('buridan')
    fly = Fly(60, 60)
    run(p, fly, seconds=10)
    arena = fake_arena(p, fly)
    assert preflight(arena, 'assay_action', 'rotate_stripes')['closes_presentation']
    env = represent(p, 'user_rotate_stripes', mutate=lambda: act(arena, 'rotate_stripes'))
    run(p, fly, seconds=10)
    assert env['records']['stripe_traversals']['reason'] == 'unsupported'
    assert rec(p, 'stripe_traversals')['reason'] == 'unsupported'
    assert p.observation_status()['presentation_index'] == 1


# ============================================================================
# visual_operant (VO-1 ... VO-9, L-1 extension)
# ============================================================================
def vo(yaws, sector='safe', dt=DT):
    p = new('visual_operant', coupling_gain=0.0)  # a fixed sector: the drum does not turn
    if sector == 'punished':
        p.invert_sectors = True
    for y in yaws:
        p.step(Fly(40, 40, yaw_torque=y), dt)
    return p


def test_VO_1a_measured_zero_is_available():
    p = vo([0.0] * 6000)
    recs = p.get_metric_records()
    assert p.observation_status()['state'] == 'closed'
    assert_value(recs['safe_occupancy_fraction'], 1.0)
    assert_value(recs['mean_abs_yaw_command_safe_rad_s'], 0.0)
    assert recs['mean_abs_yaw_command_safe_rad_s']['counts']['n'] == 6000
    assert_unavailable(recs['mean_abs_yaw_command_punished_rad_s'], 'not_observed')
    assert recs['mean_abs_yaw_command_punished_rad_s']['counts']['n'] == 0
    assert recs['operant_learning']['capability'] == 'unsupported'


def test_VO_1b_presentation_ending_before_any_sample():
    p = new('visual_operant')
    recs = p.freeze_observation('re_presentation_user')['records']
    for name in ('mean_abs_yaw_command_safe_rad_s', 'mean_abs_yaw_command_punished_rad_s',
                 'mean_yaw_command_safe_rad_s', 'mean_yaw_command_punished_rad_s', 'safe_occupancy_fraction'):
        assert_unavailable(recs[name], 'not_observed')


def test_VO_2_window_end_freezes_complete_and_represents_in_place():
    p = vo([0.1] * 6000)
    env = represent(p, 'window_complete', end_reason=None)
    assert env['completeness'] == 'complete' and env['end_reason'] == 'window_elapsed'
    st = p.observation_status()
    assert st['presentation_index'] == 1 and st['presentation_start_rel_s'] == 120.0 and st['state'] == 'observing'


def test_VO_3_reverse_heat_at_60_s():
    p = vo([0.0] * 3000)
    arena = fake_arena(p)
    env = represent(p, 'user_reverse_heat', mutate=lambda: act(arena, 'reverse_heat'))
    assert env['completeness'] == 'incomplete'
    assert env['records']['safe_occupancy_fraction']['interval_rel_s'] == [0.0, 60.0]
    assert p.observation_status()['window_end_rel_s'] == 180.0
    run(p, Fly(40, 40), n=1)
    assert rec(p, 'safe_occupancy_fraction')['interval_rel_s'][0] == 60.0


def test_VO_4_occupancy_index_not_observed_before_any_sample():
    assert_unavailable(rec(new('visual_operant'), 'occupancy_index'), 'not_observed')


@pytest.mark.parametrize('vid,yaws,signed,absolute', [
    ('VO-5', [0.5] * 500, 0.5, 0.5),
    ('VO-6', [-0.5] * 500, -0.5, 0.5),
    ('VO-7', [0.5, -0.5] * 250, 0.0, 0.5),
    ('VO-8', [0.0] * 500, 0.0, 0.0),
])
def test_VO_5_to_VO_8_signed_and_absolute_yaw_and_L_1_extension(vid, yaws, signed, absolute):
    p = vo(yaws)
    recs = p.get_metric_records()
    s, a = recs['mean_yaw_command_safe_rad_s'], recs['mean_abs_yaw_command_safe_rad_s']
    assert_value(s, signed, approx=True)
    assert_value(a, absolute, approx=True)
    assert s['counts'] == a['counts'] and s['counts']['n'] == 500 and s['unit'] == 'rad/s'
    assert s['note'] == 'signed controller yaw command used as torque proxy'
    # L-1 (extended): the C3 legacy view takes the SIGNED record; today's legacy value agrees.
    assert p.get_metrics()['mean_torque_safe'] == s['value']


def test_VO_9_empty_sector_signed_and_absolute_not_observed():
    p = vo([0.5] * 500)
    recs = p.get_metric_records()
    for name in ('mean_yaw_command_punished_rad_s', 'mean_abs_yaw_command_punished_rad_s'):
        assert_unavailable(recs[name], 'not_observed')
        assert recs[name]['counts'] == {'n': 0, 'sum_signed_rad_s': 0.0, 'sum_abs_rad_s': 0.0}
    # L-1 (extended) at C3: legacy.mean_torque_punished derives to null from this record.
    # At C0 the legacy view is unchanged (S-12) and still reports its 0.0 fallback.
    assert p.get_metrics()['mean_torque_punished'] == 0.0


# ============================================================================
# wind_tunnel
# ============================================================================
def wt(states=ALL_STATES):
    p = new('wind_tunnel')
    p.provenance = {'controller_states': list(states)}
    return p


NO_ODOUR, ODOUR = (20, 50), (20, 30)


def test_WT_1_no_samples_not_observed():
    recs = wt().get_metric_records()
    for name in ('observation_s', 'odor_contact_s', 'surge_s', 'cast_s', 'rest_s', 'surge_cast_ratio',
                 'upwind_displacement_mm', 'path_length_mm'):
        assert_unavailable(recs[name], 'not_observed')


def test_WT_2_all_rest():
    p = wt()
    run(p, Fly(*NO_ODOUR, behavioral_state='REST'), seconds=60)
    assert_unavailable(rec(p, 'surge_cast_ratio'), 'zero_denominator')
    assert_value(rec(p, 'rest_s'), 60.0)


def test_WT_3_surge_only():
    p = wt()
    run(p, Fly(*NO_ODOUR, behavioral_state='SURGE'), seconds=30)
    assert_unavailable(rec(p, 'surge_cast_ratio'), 'zero_denominator')
    assert_value(rec(p, 'surge_s'), 30.0)


def test_WT_4_surge_cast_ratio():
    p = wt()
    run(p, Fly(*NO_ODOUR, behavioral_state='SURGE'), seconds=20)
    run(p, Fly(*NO_ODOUR, behavioral_state='CAST'), seconds=10)
    assert_value(rec(p, 'surge_cast_ratio'), 2.0, approx=True)


def test_WT_5_odour_while_rest():
    p = wt()
    run(p, Fly(*ODOUR, behavioral_state='REST'), seconds=5)
    assert rec(p, 'odor_contact_s')['value'] > 0
    assert_value(rec(p, 'surge_s'), 0.0)


def test_WT_6_graph_states_unsupported_measured_kinematics_available():
    p = wt(GRAPH_STATES)
    run(p, Fly(*ODOUR, speed=1.0), seconds=5, each=lambda i, f: f.at(20 + i * 0.01, 30))
    recs = p.get_metric_records()
    for name in ('surge_s', 'cast_s', 'rest_s', 'other_state_s', 'surge_cast_ratio'):
        assert recs[name]['reason'] == 'unsupported'
    assert recs['upwind_displacement_mm']['available'] and recs['odor_contact_s']['available']


def test_WT_7_no_source_in_120_s():
    p = wt()
    run(p, Fly(*NO_ODOUR), seconds=120, dt=0.1)
    assert_unavailable(rec(p, 'time_to_source_s'), 'censored')
    assert_value(rec(p, 'source_reached'), False)


# ============================================================================
# looming_escape
# ============================================================================
def looming(source='geometric'):
    p = new('looming_escape')
    p.gf_source = source
    return p


def test_LE_1_geometric_threshold_initiates_and_presentation_runs_to_2_s():
    p = looming()
    fly = Fly(40, 40)
    run(p, fly, n=20)                 # tick [0.36, 0.38): delivered θ = 90° ≥ 65°
    assert_value(rec(p, 'escape_initiated'), True)
    gf = [e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'gf_event'][0]
    assert gf['t_rel_s'] == 0.36
    assert_value(rec(p, 'theta_at_initiation_deg'), gf['attrs']['theta_deg'])
    assert_value(rec(p, 'ttc_at_initiation_ms'), gf['attrs']['ttc_ms'])
    assert gf['attrs']['theta_deg'] >= 65.0
    fly.assay_escape_remaining = 0.1
    run(p, fly, n=5, each=lambda i, f: f.at(40 + i, 40))
    fly.assay_escape_remaining = 0.0
    run(p, fly, n=1)
    assert_value(rec(p, 'escape_completed'), True)
    assert rec(p, 'escape_displacement_mm')['value'] > 0
    assert p.observation_status()['state'] == 'observing'
    run(p, fly, n=100 - 26)
    st = p.observation_status()
    assert st['state'] == 'closed' and st['measurement_end_rel_s'] == 2.0


def test_LE_geometric_evidence_matches_actual_step_output_on_the_triggering_tick():
    p = looming('geometric')
    fly = Fly(40, 40)
    trigger = None
    for tick in range(1, 21):
        out = p.step(fly, 0.02)
        events = [e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'gf_event']
        if out['gf_spike']:
            assert trigger is None and len(events) == 1
            trigger = (tick, out, events[0])
            assert events[0]['step'] == tick
            assert events[0]['attrs']['theta_deg'] == pytest.approx(out['stimuli']['theta_deg'])
            assert events[0]['attrs']['ttc_ms'] == pytest.approx(
                out['stimuli']['time_to_collision_s'] * 1000.0)
        elif trigger is None:
            assert events == []
    tick, out, gf = trigger
    assert tick == 19 and out['stimuli']['theta_deg'] == pytest.approx(90.0)
    assert gf['t_rel_s'] == 0.36
    assert len([e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'gf_event']) == 1


def test_LE_2_window_end_freezes_and_next_presentation_starts_pending_in_place():
    p = looming()
    fly = Fly(40, 40)
    run(p, fly, n=100)
    env = represent(p, 'window_complete', end_reason=None)
    assert env['presentation_index'] == 0 and env['records']['escape_initiated']['value'] is True
    assert (fly.pos.x, fly.pos.y) == (40, 40)
    recs = p.get_metric_records()
    assert p.observation_status()['presentation_index'] == 1
    for name in ('escape_initiated', 'escape_completed', 'initiation_latency_s', 'ttc_at_initiation_ms',
                 'theta_at_initiation_deg'):
        assert_unavailable(recs[name], 'pending')


def test_LE_3_reloom_at_1_s_freezes_incomplete_and_drops_stale_values():
    p = looming()
    fly = Fly(40, 40)
    run(p, fly, n=50)
    arena = fake_arena(p, fly)
    env = represent(p, 'user_loom', mutate=lambda: act(arena, 'loom'))
    assert env['completeness'] == 'incomplete' and env['records']['ttc_at_initiation_ms']['available']
    recs = p.get_metric_records()
    assert_unavailable(recs['ttc_at_initiation_ms'], 'pending')
    assert_unavailable(recs['theta_at_initiation_deg'], 'pending')


def test_LE_4_no_gf_event_in_2_s():
    p = looming('connectome')
    run(p, Fly(40, 40), n=100)
    assert_value(rec(p, 'escape_initiated'), False)
    assert_unavailable(rec(p, 'ttc_at_initiation_ms'), 'censored')


def test_LE_T9_3_below_threshold_false_escape_cannot_initiate_or_end():
    p = looming('connectome')
    p.escape_initiated = True   # a stale legacy flag must not count
    fly = Fly(40, 40, behavioral_state='ESCAPE')
    run(p, fly, n=15)           # samples t = 0 .. 0.28
    p.step(fly, DT)             # the sample at t = 0.300: θ = 22.6°, TTC 0.100 s
    theta, ttc = p._v1_theta(s_us(0.300))
    assert math.degrees(theta) == pytest.approx(22.62, abs=0.01) and ttc == pytest.approx(0.100)
    assert rec(p, 'escape_initiated')['value'] is not True                                    # (a)
    assert not [e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'gf_event']
    for _ in range(6):                                                                          # (b) 0.32 .. 0.42
        p.step(fly, DT)
        assert p.observation_status()['measurement_end_rel_s'] is None
    run(p, fly, n=100 - 22)
    env = p.freeze_observation()
    assert env['end_reason'] == 'window_elapsed' and env['measurement_end_rel_s'] == 2.0
    assert env['records']['escape_initiated']['value'] is False
    # (c) a geometric GF event near 0.37 s: still runs to 2.0 s; θ from that event's sample, never 22.6°
    g = looming('geometric')
    run(g, Fly(40, 40), n=20)
    assert g.observation_status()['measurement_end_rel_s'] is None
    run(g, Fly(40, 40), n=80)
    env = g.freeze_observation()
    assert env['measurement_end_rel_s'] == 2.0
    assert env['records']['theta_at_initiation_deg']['value'] >= 65.0


def s_us(seconds):
    return om.s_to_us(seconds)


def test_LE_6_stimulus_collision_is_not_an_escape():
    p = looming('connectome')
    run(p, Fly(40, 40), n=30)
    names = [e['event'] for e in p.get_evidence()['presentation_events']['data']]
    assert 'stimulus_collision' in names and 'gf_event' not in names
    assert_unavailable(rec(p, 'escape_initiated'), 'pending')


def test_LE_connectome_gf_handoff_uses_actual_geometry_and_presentation_offset():
    p = looming('connectome')
    fly = Fly(40, 40)
    for _ in range(10):
        actual = p.step(fly, DT)
    p.record_gf_spike()
    assert_value(rec(p, 'escape_initiated'), True)
    assert_value(rec(p, 'initiation_latency_s'), 0.18)
    gf = [e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'gf_event'][0]
    assert gf['attrs']['theta_deg'] == pytest.approx(actual['stimuli']['theta_deg'])
    assert gf['attrs']['ttc_ms'] == pytest.approx(actual['stimuli']['time_to_collision_s'] * 1000.0)

    p.freeze_observation('re_presentation_user')
    p.begin_next_presentation('user_loom')
    actual = p.step(fly, DT)
    p.record_gf_spike()
    gf = [e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'gf_event'][0]
    assert gf['t_rel_s'] == 0.2
    assert_value(rec(p, 'initiation_latency_s'), 0.0)
    assert gf['attrs']['theta_deg'] == pytest.approx(actual['stimuli']['theta_deg'])
    assert gf['attrs']['ttc_ms'] == pytest.approx(actual['stimuli']['time_to_collision_s'] * 1000.0)


# ============================================================================
# optomotor
# ============================================================================
def om_run(drum, yaw_deg_s, seconds=1.0, contrast=None):
    p = new('optomotor', drum_velocity_deg_s=drum)
    if contrast is not None:
        p.contrast = contrast
    run(p, Fly(45, 45, angular_velocity=math.radians(yaw_deg_s)), seconds=seconds)
    return p


def test_OM_1_and_OM_2_following_in_either_direction_is_positive():
    assert_value(rec(om_run(30, 15), 'gain'), 0.5, approx=True)
    assert_value(rec(om_run(-30, -15), 'gain'), 0.5, approx=True)


def test_OM_3_static_drum_zero_denominator():
    assert_unavailable(rec(om_run(0, 5), 'gain'), 'zero_denominator')


def test_OM_4_no_contrast_no_yaw_gain_zero_available():
    assert_value(rec(om_run(30, 0, contrast=0.0), 'gain'), 0.0)


def test_OM_5_speed_change_at_59_s_fresh_window_and_unmixed_gains():
    p = new('optomotor')
    fly = Fly(45, 45, angular_velocity=math.radians(15))
    run(p, fly, seconds=59)
    arena = fake_arena(p, fly)
    pf = preflight(arena, 'set_param', 'patternSpeed', 60)
    assert pf['accepted'] and pf['closes_presentation'] and pf['normalized_value'] == 60.0
    env = represent(p, 'user_patternSpeed', mutate=lambda: set_parameter(arena, 'patternSpeed', 60))
    assert env['completeness'] == 'incomplete'
    assert env['records']['gain']['interval_rel_s'] == [0.0, 59.0]
    assert env['records']['gain']['value'] == pytest.approx(0.5)
    run(p, fly, seconds=60)
    st = p.observation_status()
    assert st['measurement_end_rel_s'] == 119.0
    assert_value(rec(p, 'gain'), 0.25, approx=True)


def test_OM_6_window_end_next_presentation_in_place():
    p = new('optomotor')
    fly = Fly(45, 45)
    run(p, fly, seconds=60)
    represent(p, 'window_complete', end_reason=None)
    assert p.observation_status()['presentation_index'] == 1 and (fly.pos.x, fly.pos.y) == (45, 45)


# ============================================================================
# gap_crossing (GC-1 ... GC-7)
# ============================================================================
def gap_walk(p, x0, x1, seconds):
    n = int(round(seconds / DT))
    fly = Fly(x0, 10, heading=0.0, speed=1.0)
    run(p, fly, n=n, each=lambda i, f: f.at(x0 + (x1 - x0) * i / max(1, n - 1), 10))
    return fly


def seq(p):
    return [e['event'] for e in p.get_evidence()['decision_sequence']['data']]


def test_GC_1_width_3_5_probe_cross_landed_then_1_s_hold():
    p = new('gap_crossing', gap_width_mm=3.5)
    gap_walk(p, 30, 50, 10)
    st = p.observation_status()
    assert st['terminal_event'] == 'landed' and st['state'] == 'measurement_ended'
    assert seq(p) == ['probe', 'cross', 'landed']
    run(p, Fly(50, 10), seconds=1.0)
    assert p.observation_status()['state'] == 'closed'


def test_GC_2_and_GC_6_abort_is_not_terminal_turn_complete_is():
    p = new('gap_crossing', gap_width_mm=5.0)
    fly = Fly(20, 10)
    run(p, fly, seconds=20)
    run(p, fly.at(43.5, 10), seconds=20)
    assert p.observation_status()['state'] == 'observing' and seq(p) == ['probe', 'abort']
    assert t(p) == 40.0
    fly.heading = math.pi
    run(p, fly, n=1)
    st = p.observation_status()
    assert st['terminal_event'] == 'turn_complete' and st['measurement_end_rel_s'] == 40.0
    assert seq(p) == ['probe', 'abort', 'turn_complete']
    recs = p.get_metric_records()
    assert_value(recs['crossing_success'], False)
    assert_value(recs['turn_complete'], True)
    run(p, fly, seconds=1.0)
    assert p.observation_status()['state'] == 'closed'


def test_GC_3_window_without_reaching_the_gap():
    p = new('gap_crossing', gap_width_mm=3.5)
    run(p, Fly(20, 10), seconds=120, dt=0.1)
    recs = p.get_metric_records()
    assert_value(recs['crossing_success'], False)
    assert recs['crossing_success']['note'] == 'crossed by the declared deadline'
    assert_unavailable(recs['time_to_cross_s'], 'censored')
    assert recs['time_to_cross_s']['counts']['censored_after_s'] == 120.0
    assert seq(p) == []


def test_GC_4_interruption_after_probe_before_landing():
    p = new('gap_crossing', gap_width_mm=3.5)
    fly = Fly(20, 10)
    run(p, fly, seconds=60)
    run(p, fly.at(44, 10), seconds=10)
    env = p.freeze_observation('manual_reset')
    recs = env['records']
    assert_unavailable(recs['crossing_success'], 'censored')
    assert recs['crossing_success']['counts']['censored_at_rel_s'] == 70.0
    assert_unavailable(recs['time_to_cross_s'], 'censored')
    assert_unavailable(recs['turn_complete'], 'censored')
    assert [e['event'] for e in env['evidence']['decision_sequence']['data']] == ['probe', 'cross']


def test_GC_5_fault_invalidates_the_verdict():
    p = new('gap_crossing', gap_width_mm=3.5)
    run(p, Fly(44, 10), seconds=70, dt=0.1)
    recs = p.freeze_observation('fault_halt')['records']
    assert_unavailable(recs['crossing_success'], 'invalidated')
    assert_unavailable(recs['time_to_cross_s'], 'invalidated')


def test_GC_7_time_to_cross_is_from_each_presentation_start():
    p = new('gap_crossing', gap_width_mm=3.5)
    fly = Fly(20, 10)
    run(p, fly, seconds=61)
    run(p, fly.at(50, 10), n=1)
    assert_value(rec(p, 'crossing_success'), True)
    assert_value(rec(p, 'time_to_cross_s'), 61.0)
    q = new('gap_crossing', gap_width_mm=3.5)
    fly = Fly(20, 10)
    run(q, fly, seconds=20)
    arena = fake_arena(q, fly)
    pf = preflight(arena, 'set_param', 'gapWidth', 3.0)
    assert pf['accepted'] and pf['closes_presentation']
    represent(q, 'user_gapWidth', mutate=lambda: set_parameter(arena, 'gapWidth', 3.0))
    run(q, fly, seconds=61)
    run(q, fly.at(50, 10), n=1)
    assert t(q) == pytest.approx(81.02) and q.observation_status()['measurement_end_rel_s'] == 81.0
    assert_value(rec(q, 'time_to_cross_s'), 61.0)


# ============================================================================
# circadian_dam
# ============================================================================
def dam_run(p, dts, speed=0.0):
    fly = Fly(10, 5, speed=speed)
    for d in dts:
        p.step(fly, d)


def test_DAM_1_to_DAM_3_bout_threshold_in_integer_microseconds():
    for n, bouts, minutes in ((14999, 0, 0.0), (15000, 1, 5.0), (30000, 1, 10.0)):
        p = new('circadian_dam')
        dam_run(p, [DT] * n)
        assert_value(rec(p, 'immobility_bouts_300s'), bouts)
        assert_value(rec(p, 'bout_immobility_min'), minutes, approx=True)
    assert_unavailable(rec(new('circadian_dam'), 'mean_bout_min'), 'not_observed')
    p = new('circadian_dam')
    dam_run(p, [DT] * 100)
    assert_unavailable(rec(p, 'mean_bout_min'), 'zero_denominator')


def test_DAM_mean_bout_is_a_minute_ratio_across_live_freeze_and_presentation():
    fresh = rec(new('circadian_dam'), 'mean_bout_min')
    assert_unavailable(fresh, 'not_observed')
    assert (fresh['kind'], fresh['unit'], fresh['scope']) == ('ratio', 'min', 'segment')
    assert fresh['numerator'] is None and fresh['denominator'] is None

    zero = new('circadian_dam')
    dam_run(zero, [1.0], speed=2.0)
    zero_record = rec(zero, 'mean_bout_min')
    assert_unavailable(zero_record, 'zero_denominator')
    assert (zero_record['kind'], zero_record['unit']) == ('ratio', 'min')
    assert zero_record['numerator'] == 0.0 and zero_record['denominator'] == 0

    p = new('circadian_dam')
    dam_run(p, [300.0])
    one = rec(p, 'mean_bout_min')
    assert_value(one, 5.0)
    assert (one['kind'], one['unit']) == ('ratio', 'min')
    assert one['numerator'] == 5.0 and one['denominator'] == 1
    assert one['value'] == one['numerator'] / one['denominator']

    dam_run(p, [1.0], speed=2.0)
    dam_run(p, [600.0])
    two = rec(p, 'mean_bout_min')
    assert_value(two, 7.5)
    assert (two['kind'], two['unit'], two['scope']) == ('ratio', 'min', 'segment')
    assert two['numerator'] == 15.0 and two['denominator'] == 2
    assert two['value'] == two['numerator'] / two['denominator']

    frozen = p.freeze_observation('policy_change')
    frozen_copy = copy.deepcopy(frozen)
    final = frozen['records']['mean_bout_min']
    assert final['final'] is True
    assert (final['kind'], final['unit'], final['value']) == ('ratio', 'min', 7.5)
    assert (final['numerator'], final['denominator']) == (15.0, 2)

    p.begin_next_presentation('policy_change', config(p, 'dam-ratio-continuation'))
    continued = rec(p, 'mean_bout_min')
    assert (continued['kind'], continued['unit'], continued['value']) == ('ratio', 'min', 7.5)
    assert (continued['numerator'], continued['denominator']) == (15.0, 2)
    assert p.observation_status()['presentation_index'] == 1
    assert frozen == frozen_copy

def test_DAM_4_one_moving_sample_resets_the_interval():
    p = new('circadian_dam')
    dam_run(p, [DT] * 10000)
    dam_run(p, [DT], speed=2.0)
    dam_run(p, [DT] * 10000)
    assert_value(rec(p, 'immobility_bouts_300s'), 0)


def test_DAM_5_variable_dt_exactly_300_s():
    p = new('circadian_dam')
    dam_run(p, [0.01, 0.03] * 7500)
    assert_value(rec(p, 'immobility_bouts_300s'), 1)
    p = new('circadian_dam')
    dam_run(p, [0.01, 0.03] * 7499 + [0.03])
    assert t(p) == 299.99
    assert_value(rec(p, 'immobility_bouts_300s'), 0)


def test_DAM_6_never_ends_at_28_8_s_or_ever():
    p = new('circadian_dam')
    fly = Fly(10, 5)
    for _ in range(3100):
        p.step(fly, DT)
        st = p.observation_status()
        assert st['state'] == 'observing' and st['measurement_end_rel_s'] is None
    assert p.observation_config()['automatic_end'] is False


def test_DAM_7_default_light_phase_keeps_its_12_hour_semantics():
    p = new('circadian_dam')
    dam_run(p, [60.0] * 720)
    assert_value(rec(p, 'light_phase'), 'light')
    dam_run(p, [60.0])
    assert_value(rec(p, 'light_phase'), 'dark')
    assert p.observation_spec()['light_schedule'] == {'mode': 'LD', 'light_s': 43200.0, 'dark_s': 43200.0}


def test_DAM_8_compressed_schedule_drives_actual_input_metadata_and_evidence():
    q = new('circadian_dam')
    q.set_light_schedule(0.1, 0.1)
    sched = q.observation_spec()['light_schedule']
    assert sched['mode'] == 'compressed' and 'no entrainment claim' in sched['note']

    transitions = []
    previous = None
    fly = Fly(10, 5)
    for _ in range(10):
        actual = q.step(fly, 0.02)['stimuli']['is_lights_on']
        direct = q.sample_stimuli(fly.pos, fly.heading)['is_lights_on']
        metadata = rec(q, 'light_phase')['value']
        phase = 'light' if actual else 'dark'
        assert direct is actual
        assert metadata == phase
        if phase != previous:
            transitions.append((q.time_elapsed_ms / 1000.0, phase))
            previous = phase

    assert transitions == [(0.02, 'light'), (0.1, 'dark'), (0.2, 'light')]
    assert q.get_evidence()['light_phases']['data'][:3] == [
        {'phase': 'light', 'start_rel_s': 0.0, 'end_rel_s': 0.1},
        {'phase': 'dark', 'start_rel_s': 0.1, 'end_rel_s': 0.2},
        {'phase': 'light', 'start_rel_s': 0.2, 'end_rel_s': 0.3},
    ]


# ============================================================================
# courtship
# ============================================================================
NEAR, FAR = (11.0, 11.0), (3.0, 10.0)


def test_CO_1_mated_female_avoided_for_600_s():
    p = new('courtship', female_type='mated')
    run(p, Fly(*FAR), seconds=600, dt=0.1)
    assert_value(rec(p, 'proximity_fraction'), 0.0)
    assert rec(p, 'courtship_conditioning')['capability'] == 'unsupported'


def test_CO_2_receptivity_toggle_gives_two_presentations():
    p = new('courtship')
    run(p, Fly(*NEAR), seconds=300, dt=0.1)
    arena = fake_arena(p)
    env = represent(p, 'user_receptivity', mutate=lambda: act(arena, 'receptivity'))
    run(p, Fly(*FAR), seconds=300, dt=0.1)
    assert env['records']['proximity_fraction']['value'] == 1.0
    assert_value(rec(p, 'proximity_fraction'), 0.0)
    assert p.observation_spec()['window_params']['female_type'] == 'virgin'


def test_CO_3_graph_states_unsupported_proximity_available():
    p = new('courtship')
    p.provenance = {'controller_states': GRAPH_STATES}
    run(p, Fly(*NEAR), seconds=5)
    recs = p.get_metric_records()
    for name in ('approach_s', 'avoid_s', 'courtship_state_s', 'courtship_state_fraction'):
        assert recs[name]['reason'] == 'unsupported'
    assert_value(recs['proximity_fraction'], 1.0)


def test_CO_4_before_any_sample_not_observed():
    recs = new('courtship').get_metric_records()
    assert_unavailable(recs['proximity_fraction'], 'not_observed')
    assert_unavailable(recs['min_distance_mm'], 'not_observed')


# ============================================================================
# labyrinth
# ============================================================================
def lb_run(p, fly, n, contact=None):
    for i in range(n):
        p.step(fly, DT)
        if contact is not None:
            p.observe_contact(p._v1_last_step, DT, 'post_solver', contact(i) if callable(contact) else contact)


def test_LB_1_continuous_contact_is_one_onset():
    p = new('labyrinth')
    lb_run(p, Fly(10, 10), 100, contact=True)
    assert_value(rec(p, 'wall_contact_onsets'), 1)
    assert_value(rec(p, 'wall_contact_s'), 2.0, approx=True)


def test_LB_2_contact_release_contact_is_two():
    p = new('labyrinth')
    lb_run(p, Fly(10, 10), 3, contact=lambda i: i != 1)
    assert_value(rec(p, 'wall_contact_onsets'), 2)


def test_LB_3_dead_end_reentry_after_1_mm_exit():
    p = new('labyrinth')
    fly = Fly(10, 10)
    lb_run(p, fly.at(30, 75), 500)
    lb_run(p, fly.at(30, 68.9), 5)   # 1.1 mm outside: re-armed
    lb_run(p, fly.at(30, 75), 5)
    assert_value(rec(p, 'dead_end_entries'), 2)
    q = new('labyrinth')
    lb_run(q, fly.at(30, 75), 5)
    lb_run(q, fly.at(30, 69.5), 5)   # 0.5 mm outside: not re-armed
    lb_run(q, fly.at(30, 75), 5)
    assert_value(rec(q, 'dead_end_entries'), 1)


def test_LB_4_closed_loop_tortuosity_zero_denominator():
    p = new('labyrinth')
    for xy in ((10, 10), (10, 20), (20, 20), (10, 10)):
        lb_run(p, Fly(*xy), 1)
    assert_unavailable(rec(p, 'tortuosity'), 'zero_denominator')
    assert rec(p, 'path_length_mm')['value'] > 0


def test_LB_5_no_goal_in_300_s():
    p = new('labyrinth')
    run(p, Fly(10, 10), seconds=300, dt=0.1)
    assert_unavailable(rec(p, 'time_to_goal_s'), 'censored')
    assert_value(rec(p, 'goal_reached'), False)


def test_LB_6_no_contact_handoff_is_not_observed_not_zero():
    p = new('labyrinth')
    lb_run(p, Fly(10, 10), 50)
    assert_unavailable(rec(p, 'wall_contact_onsets'), 'not_observed')
    assert_unavailable(rec(p, 'wall_contact_s'), 'not_observed')


# ============================================================================
# multisensory
# ============================================================================
def test_MS_1_rest_for_120_s():
    p = new('multisensory')
    run(p, Fly(0, 0, speed=0.0), seconds=120, dt=0.1)
    recs = p.get_metric_records()
    assert_value(recs['distance_mm'], 0.0)
    assert recs['composite_benchmark_score']['capability'] == 'unsupported'
    assert p.get_metrics()['composite_benchmark_score'] > 0  # the legacy view is unchanged at C0


def test_MS_2_empty_history_has_no_default_scores():
    recs = new('multisensory').get_metric_records()
    for name in ('distance_mm', 'odor_a_exposure_s', 'heat_exposure_s', 'mean_abs_speed_jerk_mm_s3',
                 'wall_contact_onsets', 'wall_contact_s'):
        assert_unavailable(recs[name], 'not_observed')
    for name in ('composite_benchmark_score', 'locomotor_coordination_index', 'multisensory_integration_score',
                 'biomechanical_efficiency', 'kinematic_smoothness', 'energy_proxy'):
        assert recs[name]['reason'] == 'unsupported'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_NF_1_heat_rejects_nonfinite_temperature_before_dose_clipping(bad):
    p = new('heat_maze')
    p.peltier.get_temperature = lambda x, y: bad
    with pytest.raises(MetricFault) as fault:
        p.step(Fly(5, 5, speed=0.0), 0.02)
    assert fault.value.path == 'heat_maze.stimuli.temperature'
    json_ok(p.get_metric_records())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_NF_2_wind_rejects_nonfinite_flow_before_vector_math(bad):
    p = new('wind_tunnel')
    p.wind_flow = (bad, 0.0)
    with pytest.raises(MetricFault) as fault:
        p.step(Fly(20, 30), 0.02)
    assert fault.value.path == 'wind_tunnel.stimuli.wind[0]'
    json_ok(p.get_metric_records())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_NF_3_multisensory_rejects_nonfinite_actual_input_before_comparison(bad):
    p = new('multisensory')
    sample = p.sample_stimuli

    def nonfinite_odor(*args, **kwargs):
        stimuli = sample(*args, **kwargs)
        stimuli['odor_a'] = bad
        return stimuli

    p.sample_stimuli = nonfinite_odor
    with pytest.raises(MetricFault) as fault:
        p.step(Fly(0, 0), 0.02)
    assert fault.value.path == 'multisensory.stimuli.odor_a'
    json_ok(p.get_metric_records())


def test_NF_4_finite_zero_stimuli_remain_valid():
    heat = new('heat_maze')
    heat.peltier.get_temperature = lambda x, y: 0.0
    heat_out = heat.step(Fly(5, 5), 0.02)
    assert heat_out['stimuli']['temperature'] == 0.0
    assert_value(rec(heat, 'thermal_dose_degC_s'), 0.0)

    wind = new('wind_tunnel')
    wind.wind_flow = (0.0, 0.0)
    wind_out = wind.step(Fly(20, 30), 0.02)
    assert wind_out['stimuli']['wind_speed'] == 0.0

    multisensory = new('multisensory')
    sample = multisensory.sample_stimuli

    def zero_odor(*args, **kwargs):
        stimuli = sample(*args, **kwargs)
        stimuli['odor_a'] = 0.0
        return stimuli

    multisensory.sample_stimuli = zero_odor
    multi_out = multisensory.step(Fly(0, 0), 0.02)
    assert multi_out['stimuli']['odor_a'] == 0.0
    assert_value(rec(multisensory, 'odor_a_exposure_s'), 0.0)


# ============================================================================
# v1.1 A1: ObservationConfig (CF-1 ... CF-5)
# ============================================================================
def test_CF_non_aligned_window_clips_durations_and_integrals_without_shortening_ticks():
    p = new('buridan')
    p.configure_observation(config(p, 'clip-buridan', override_window_s=0.025))
    run(p, Fly(30, 30, speed=1.0), n=2, dt=0.02)
    env = p.freeze_observation()
    assert env['measurement_end_rel_s'] == 0.025
    assert env['presentation_elapsed_s'] == 0.04
    assert env['dt_s'] == 0.02
    assert_value(env['records']['observation_s'], 0.025)
    assert_value(env['records']['walking_s'], 0.025)

    q = new('heat_maze')
    q.configure_observation(config(q, 'clip-heat', override_window_s=0.025))
    sample = Fly(*HOT)
    excess = max(0.0, q.peltier.get_temperature(sample.pos.x, sample.pos.y) - 25.0)
    run(q, sample, n=2, dt=0.02)
    assert_value(rec(q, 'thermal_dose_degC_s'), excess * 0.025, approx=True)


def test_CF_post_solver_contacts_before_at_and_after_deadline():
    before = new('labyrinth')
    before.configure_observation(config(before, 'contact-before', override_window_s=0.025))
    before.step(Fly(10, 10), 0.02)
    before.observe_contact(before._v1_last_step, 0.02, 'post_solver', True)
    before.step(Fly(10, 10), 0.02)
    before.observe_contact(before._v1_last_step, 0.02, 'post_solver', False)
    assert_value(rec(before, 'wall_contact_onsets'), 1)
    assert_value(rec(before, 'wall_contact_s'), 0.025, approx=True)
    assert [e['t_rel_s'] for e in before.get_evidence()['contact_events']['data']] == [0.02]

    at = new('labyrinth')
    at.configure_observation(config(at, 'contact-at', override_window_s=0.04))
    at.step(Fly(10, 10), 0.02)
    at.observe_contact(at._v1_last_step, 0.02, 'post_solver', False)
    at.step(Fly(10, 10), 0.02)
    at.observe_contact(at._v1_last_step, 0.02, 'post_solver', True)
    assert_value(rec(at, 'wall_contact_onsets'), 1)
    assert [e['t_rel_s'] for e in at.get_evidence()['contact_events']['data']] == [0.04]

    after = new('labyrinth')
    after.configure_observation(config(after, 'contact-after', override_window_s=0.025))
    after.step(Fly(10, 10), 0.02)
    after.observe_contact(after._v1_last_step, 0.02, 'post_solver', False)
    after.step(Fly(10, 10), 0.02)
    after.observe_contact(after._v1_last_step, 0.02, 'post_solver', True)
    assert_value(rec(after, 'wall_contact_onsets'), 0)
    assert_value(rec(after, 'wall_contact_s'), 0.0)
    assert after.get_evidence()['contact_events']['data'] == []


def test_CF_non_aligned_window_keeps_segment_relative_contact_interval():
    p = new('labyrinth')
    p.step(Fly(10, 10), 0.02)
    p.observe_contact(p._v1_last_step, 0.02, 'post_solver', False)
    p.freeze_observation('policy_change')
    p.begin_next_presentation('policy_change', config(p, 'relative', override_window_s=0.025))
    p.step(Fly(10, 10), 0.02)
    p.observe_contact(p._v1_last_step, 0.02, 'post_solver', True)
    p.step(Fly(10, 10), 0.02)
    p.observe_contact(p._v1_last_step, 0.02, 'post_solver', True)
    env = p.freeze_observation()
    assert env['window_end_rel_s'] == 0.045
    assert env['records']['wall_contact_s']['interval_rel_s'] == [0.0, 0.045]
    assert_value(env['records']['wall_contact_s'], 0.025, approx=True)
    assert [e['t_rel_s'] for e in env['evidence']['contact_events']['data']] == [0.04]


def test_CF_1_override_180_accumulates_through_180():
    p = new('t_maze')
    cfg = p.configure_observation(config(p, 'cf1', override_window_s=180, override_source='api',
                                         set_at_sim_s=0.0, manifest_run_id='run-1'))
    assert cfg['window_source'] == 'override' and cfg['effective_window_s'] == 180.0
    fly = Fly(*STEM)
    run(p, fly, seconds=120, dt=0.1)
    assert p.observation_status()['measurement_end_rel_s'] is None
    run(p, fly, seconds=10, dt=0.1)
    run(p, fly.at(*CSP), n=1, dt=0.1)
    run(p, fly.at(*STEM), seconds=50, dt=0.1)
    st = p.observation_status()
    assert st['measurement_end_rel_s'] == 180.0 and st['window_end_rel_s'] == 180.0
    env = p.freeze_observation()
    assert env['records']['arm_entry_preference_index']['interval_rel_s'] == [0.0, 180.0]
    assert env['records']['cs_plus_entries']['value'] == 1
    assert env['config_id'] == 'cf1' and env['window_source'] == 'override' and env['window_s'] == 180.0
    assert env['override'] == {'source': 'api', 'value': 180.0, 'set_at_sim_s': 0.0, 'manifest_run_id': 'run-1'}


def test_CF_2_no_automatic_end_heat_maze_runs_past_300_s():
    p = new('heat_maze')
    p.configure_observation(config(p, 'cf2', continuous=True))
    run(p, Fly(*HOT), seconds=300, dt=0.1)
    dose_300 = rec(p, 'thermal_dose_degC_s')['value']
    run(p, Fly(*HOT), seconds=100, dt=0.1)
    assert p.observation_status()['measurement_end_rel_s'] is None
    assert_unavailable(rec(p, 'escape_latency_s'), 'pending')
    assert rec(p, 'thermal_dose_degC_s')['value'] > dose_300


def test_CF_3_no_automatic_end_records_the_terminal_event_without_ending():
    p = new('heat_maze')
    p.configure_observation(config(p, 'cf3', continuous=True))
    run(p, Fly(*HOT), n=700)
    run(p, Fly(*REFUGE), n=50)
    st = p.observation_status()
    assert st['state'] == 'observing' and st['measurement_end_rel_s'] is None
    assert_value(rec(p, 'escape_latency_s'), 14.0)
    r = rec(p, 'refuge_reached')
    assert r['value'] is True and r['final'] is False   # see the CF-3 ambiguity in ASSAY_SEMANTICS
    assert p.get_evidence()['terminal_events_observed']['data'][0]['t_rel_s'] == 14.0


def test_CF_4_configure_after_first_sample_raises_and_changes_nothing():
    p = new('t_maze')
    run(p, Fly(*STEM), n=3)
    before = (p.observation_config(), p.observation_status(), p.get_metric_records())
    with pytest.raises(ConfigError):
        p.configure_observation(config(p, 'late', override_window_s=180))
    assert (p.observation_config(), p.observation_status(), p.get_metric_records()) == before
    with pytest.raises(ConfigError):
        OpenArenaObserver().configure_observation({'config_id': 'x'})


def test_CF_5_policy_change_at_50_s():
    p = new('t_maze')
    p.configure_observation(config(p, 'cf5-a'))
    run(p, Fly(*STEM), seconds=50, dt=0.1)
    env = represent(p, 'policy_change', end_reason='policy_change',
                    config=config(p, 'cf5-b', override_window_s=180, override_source='api', set_at_sim_s=50.0))
    assert env['completeness'] == 'incomplete' and env['end_reason'] == 'policy_change'
    assert env['records']['arm_entry_preference_index']['interval_rel_s'] == [0.0, 50.0]
    assert env['config_id'] == 'cf5-a' and env['effective_window_s'] == 120.0
    st = p.observation_status()
    assert st['presentation_start_rel_s'] == 50.0 and st['config_id'] == 'cf5-b' and st['window_end_rel_s'] == 230.0
    log = p.get_evidence()['interventions']['data']
    assert {'name': 'policy:effective_window_s', 'value': 180.0, 't_rel_s': 50.0, 'step': 500} in log
    assert any(e['name'] == 'policy:window_source' and e['value'] == 'override' for e in log)


def test_CF_reset_trial_accepts_the_segment_config():
    p = new('optomotor')
    run(p, Fly(45, 45), n=5)
    p.reset_trial(config=config(p, 'seg-2', override_window_s=30, override_source='cli:--trial-seconds'))
    st = p.observation_status()
    assert st['segment_elapsed_s'] == 0.0 and st['config_id'] == 'seg-2' and st['window_end_rel_s'] == 30.0


def test_CF_build_observation_config_rules():
    spec = new('circadian_dam').observation_spec()
    c = build_observation_config(spec, config_id='d', override_window_s=60)
    assert c['automatic_end'] is False and c['effective_window_s'] is None and c['window_source'] == 'spec'
    assert c['override']['value'] == 60.0   # carried for provenance, not applied to a continuous assay
    spec = new('t_maze').observation_spec()
    c = build_observation_config(spec, config_id='c', continuous=True)
    assert (c['automatic_end'], c['window_source'], c['override']['source']) == (False, 'continuous_flag',
                                                                                 'cli:--continuous')


# ============================================================================
# v1.1 A2: preflight (PF-1 ... PF-5)
# ============================================================================
def arenas():
    for pid in PARADIGMS:
        yield pid, Arena(paradigm=None if pid == 'open-arena' else pid, seed=4, num_flies=1, num_predators=0)


def snapshot(a):
    return pickle.dumps(a)


def test_PF_1_preflight_mutates_nothing_for_every_control_of_all_14_assays():
    n = 0
    for pid, a in arenas():
        for _ in range(3):
            a.step(DT)
        info = assay_controls.describe(a)
        before = snapshot(a)
        for spec in info['parameters']:
            for value in (spec['value'], spec['max'] + 1, float('nan'), 'abc'):
                preflight(a, 'set_param', spec['name'], value)
                n += 1
        for action in info['actions']:
            preflight(a, 'assay_action', action['name'])
            n += 1
        preflight(a, 'assay_action', 'invented')
        preflight(a, 'bogus_kind', 'x')
        assert snapshot(a) == before, pid
    assert n >= 30


def test_PF_2_gap_near_edge_is_rejected_with_the_existing_message():
    a = Arena(paradigm='gap-crossing', seed=4, num_flies=1, num_predators=0)
    a.fly.pos.x = 44.0
    before = snapshot(a)
    pf = preflight(a, 'set_param', 'gapWidth', 4.0)
    assert pf == {'accepted': False, 'message': 'Reset the fly before changing the gap underneath it',
                  'normalized_value': None, 'closes_presentation': False, 'intervention_only': False,
                  'kind': 'set_param', 'name': 'gapWidth'}
    assert snapshot(a) == before


def test_PF_3_out_of_range_and_malformed_values_are_rejected():
    a = Arena(paradigm='gap-crossing', seed=4, num_flies=1, num_predators=0)
    a.fly.pos.x = 20.0
    before = snapshot(a)
    for value in (9.0, float('nan'), float('inf'), 'abc', None, True):
        pf = preflight(a, 'set_param', 'gapWidth', value)
        assert not pf['accepted'] and pf['message'], value
    assert not preflight(a, 'set_param', 'invented', 1)['accepted']
    assert snapshot(a) == before
    o = Arena(paradigm='optomotor', seed=4, num_flies=1, num_predators=0)
    before = snapshot(o)
    assert preflight(o, 'set_param', 'patternSpeed', 500)['message'] == 'patternSpeed must be between -120 and 120'
    assert snapshot(o) == before


def test_PF_4_accepted_width_change_then_next_presentation():
    a = Arena(paradigm='gap-crossing', seed=4, num_flies=1, num_predators=0)
    a.fly.pos.x = 20.0
    a.step(DT)
    p = a.paradigm
    pf = preflight(a, 'set_param', 'gapWidth', 4.0)
    assert pf['accepted'] and pf['closes_presentation'] and pf['normalized_value'] == 4.0
    p.freeze_observation('re_presentation_user')
    result = set_parameter(a, 'gapWidth', 4.0, detailed=True)
    assert result == {'applied': True, 'value': 4.0, 'presentation_closed': True}
    assert p.begin_next_presentation('user_gapWidth') == 1
    assert p.zones[1].bounds == (45.0, 0.0, 49.0, 20.0) and p.zones[2].bounds == (49.0, 7.5, 100.0, 12.5)
    assert set_parameter(a, 'gapWidth', 3.5) == 3.5   # the default return stays the float (A1)


def test_PF_5_preflight_and_set_parameter_agree_over_a_grid():
    for pid, a in arenas():
        for spec in assay_controls.describe(a)['parameters']:
            lo, hi = spec['min'], spec['max']
            for value in (lo - 1, lo, (lo + hi) / 2, hi, hi + 0.5, float('nan'), float('-inf'), 'abc', None):
                pf = preflight(a, 'set_param', spec['name'], value)
                try:
                    set_parameter(a, spec['name'], value)
                    applied = True
                except (ValueError, TypeError):
                    applied = False
                assert pf['accepted'] == applied, (pid, spec['name'], value)
        for action in assay_controls.describe(a)['actions']:
            assert preflight(a, 'assay_action', action['name'])['accepted']
            assert act(a, action['name'])['applied'] is True


# ============================================================================
# v1.1 A3: segment-relative timestamps (TS-1 ... TS-5)
# ============================================================================
def test_TS_1_optomotor_second_presentation_window():
    p = new('optomotor')
    fly = Fly(45, 45)
    run(p, fly, seconds=59)
    represent(p, 'user_patternSpeed', mutate=lambda: setattr(p, 'drum_velocity_deg_s', 60.0))
    assert p.observation_status()['window_end_rel_s'] == 119.0
    run(p, fly, seconds=60)
    st = p.observation_status()
    assert st['measurement_end_rel_s'] == 119.0
    assert rec(p, 'gain')['interval_rel_s'] == [59.0, 119.0]


def test_TS_2_looming_second_automatic_presentation_times():
    p = looming('geometric')
    fly = Fly(40, 40)
    run(p, fly, n=200, dt=0.01)
    represent(p, 'window_complete', end_reason=None)
    triggering_output = None
    for _ in range(60):
        out = p.step(fly, 0.01)
        if out['gf_spike']:
            triggering_output = out
    ev = {e['event']: e for e in p.get_evidence()['presentation_events']['data']}
    assert ev['stimulus_onset']['t_rel_s'] == 2.0
    assert ev['stimulus_collision']['t_rel_s'] == 2.4
    assert ev['gf_event']['t_rel_s'] == 2.36
    assert ev['gf_event']['attrs']['theta_deg'] == pytest.approx(triggering_output['stimuli']['theta_deg'])
    assert ev['gf_event']['attrs']['ttc_ms'] == pytest.approx(
        triggering_output['stimuli']['time_to_collision_s'] * 1000.0)
    assert_value(rec(p, 'initiation_latency_s'), 0.36, approx=True)
    assert_value(rec(p, 'ttc_at_initiation_ms'), 30.0, approx=True)
    assert p.freeze_observation('manual_reset')['stimulus_collision_rel_s'] == 2.4


def test_TS_3_reloom_at_59_s():
    p = looming('connectome')
    fly = Fly(40, 40)
    while t(p) < 59.0 - 1e-9:
        p.step(fly, DT)
        if p.observation_status()['state'] == 'closed':
            represent(p, 'window_complete', end_reason=None)
    assert t(p) == 59.0
    arena = fake_arena(p, fly)
    represent(p, 'user_loom', mutate=lambda: act(arena, 'loom'))
    st = p.observation_status()
    assert st['presentation_start_rel_s'] == 59.0 and st['window_end_rel_s'] == 61.0
    run(p, fly, n=25)
    coll = [e for e in p.get_evidence()['presentation_events']['data'] if e['event'] == 'stimulus_collision']
    assert coll[0]['t_rel_s'] == 59.4


def test_TS_4_first_choice_after_reversal():
    p = new('t_maze')
    fly = Fly(*STEM)
    run(p, fly, seconds=50, dt=0.1)
    arena = fake_arena(p, fly)
    represent(p, 'user_reverse_arms', mutate=lambda: act(arena, 'reverse_arms'))
    run(p, fly, seconds=30, dt=0.1)
    run(p, fly.at(*CSM), n=1, dt=0.1)   # arm_b is CS+ after the reversal
    assert_value(rec(p, 'first_choice'), 'cs_plus')
    assert_value(rec(p, 'first_choice_latency_s'), 30.0, approx=True)
    assert p.get_evidence()['first_choice_pose']['data']['t_rel_s'] == 80.0


def test_TS_5_heat_maze_cutoffs_relative_to_segment_and_presentation():
    p = new('heat_maze')
    run(p, Fly(*HOT), seconds=50, dt=0.1)
    lat = p.freeze_observation('manual_reset')['records']['escape_latency_s']
    assert lat['counts'] == {'censored_at_rel_s': 50.0, 'censored_after_s': 50.0}
    q = new('heat_maze')
    run(q, Fly(*HOT), seconds=20, dt=0.1)
    arena = fake_arena(q, Fly(*HOT))
    represent(q, 'user_floorTemp', mutate=lambda: set_parameter(arena, 'floorTemp', 40.0))
    run(q, Fly(*HOT), seconds=30, dt=0.1)
    lat = q.freeze_observation('manual_reset')['records']['escape_latency_s']
    assert lat['counts'] == {'censored_at_rel_s': 50.0, 'censored_after_s': 30.0}


# ============================================================================
# v1 §5.6 (producer side of N1-2): a paradigm without a spec fails visibly
# ============================================================================
def test_paradigm_without_spec_runs_legacy_step_and_refuses_observation():
    from tests.fixtures.wall_progress import FixtureParadigm
    p = FixtureParadigm('no_spec', (20.0, 20.0))
    assert p.step(Fly(5, 5), DT) == {} and p.reset_trial() == {}
    for call in (p.observation_spec, p.observation_status, p.get_metric_records):
        with pytest.raises(maze.MissingObservationSpec, match='assay no_spec has no observation spec'):
            call()

# ============================================================================
# CARD-10A: a fault invalidates the current presentation's deadline verdicts
# ============================================================================
def test_C10A_fault_after_observed_events_invalidates_all_outcomes_and_latencies():
    cases = []

    p = new('heat_maze')
    run(p, Fly(*REFUGE), n=1)
    cases.append((p, 'refuge_reached', ('escape_latency_s',)))

    p = new('wind_tunnel')
    run(p, Fly(180, 30), n=1)
    cases.append((p, 'source_reached', ('time_to_source_s',)))

    p = new('labyrinth')
    run(p, Fly(130, 85), n=1)
    cases.append((p, 'goal_reached', ('time_to_goal_s',)))

    p = looming('geometric')
    run(p, Fly(40, 40), n=20)
    cases.append((p, 'escape_initiated', ('initiation_latency_s', 'ttc_at_initiation_ms')))

    p = looming('geometric')
    fly = Fly(40, 40)
    run(p, fly, n=20)
    fly.assay_escape_remaining = 0.1
    run(p, fly, n=3)
    fly.assay_escape_remaining = 0.0
    run(p, fly, n=1)
    cases.append((p, 'escape_completed', ('initiation_latency_s', 'ttc_at_initiation_ms')))

    p = new('gap_crossing', gap_width_mm=3.5)
    run(p, Fly(50, 10), n=1)
    cases.append((p, 'crossing_success', ('time_to_cross_s',)))

    p = new('gap_crossing', gap_width_mm=5.0)
    p.zones[1].bounds = (45.0, 0.0, 50.0, 20.0)
    p.zones[2].bounds = (50.0, 7.5, 100.0, 12.5)
    run(p, Fly(43.5, 10), n=2)
    run(p, Fly(43.5, 10, heading=math.pi), n=1)
    cases.append((p, 'turn_complete', ('decision_latency_s',)))

    assert {name for _, name, _ in cases} == set(om.OUTCOME_BOOLEANS)
    for paradigm, outcome_name, latency_names in cases:
        env = paradigm.freeze_observation('fault_halt')
        outcome = env['records'][outcome_name]
        assert_unavailable(outcome, 'invalidated')
        assert outcome['final'] is True and outcome['counts']['observed_events'] == 1
        for latency_name in latency_names:
            latency = env['records'][latency_name]
            assert_unavailable(latency, 'invalidated')
            assert latency['final'] is True
            assert ('observed_after_s' in latency['counts']
                    or 'observed_value' in latency['counts'])

    looming_env = cases[3][0].freeze_observation('fault_halt')
    events = looming_env['evidence']['presentation_events']['data']
    assert any(item['event'] == 'gf_event' for item in events)
    assert looming_env['records']['ttc_at_initiation_ms']['evidence_ref'] == 'presentation_events'
    gap_env = cases[5][0].freeze_observation('fault_halt')
    assert [item['event'] for item in gap_env['evidence']['decision_sequence']['data']] == ['landed']


def test_C10A_fault_before_event_and_after_completed_false_are_invalidated():
    before = new('heat_maze')
    run(before, Fly(*HOT), n=2)
    records = before.freeze_observation('fault_halt')['records']
    assert_unavailable(records['refuge_reached'], 'invalidated')
    assert_unavailable(records['escape_latency_s'], 'invalidated')
    assert records['refuge_reached']['counts']['observed_events'] == 0

    completed = new('heat_maze')
    short_window(completed, 0.04)
    run(completed, Fly(*HOT), n=2)
    assert_value(rec(completed, 'refuge_reached'), False)
    records = completed.freeze_observation('fault_halt')['records']
    assert_unavailable(records['refuge_reached'], 'invalidated')
    assert_unavailable(records['escape_latency_s'], 'invalidated')


def test_C10A_segment_goal_true_and_completed_false_are_invalidated_on_fault():
    observed = new('labyrinth')
    lb_run(observed, Fly(10, 10), 1, contact=True)
    run(observed, Fly(130, 85), n=1)
    assert_value(rec(observed, 'goal_reached'), True)
    assert_value(rec(observed, 'time_to_goal_s'), DT)
    evidence_before = copy.deepcopy(observed.get_evidence())

    records = observed.freeze_observation('fault_halt')['records']
    assert_unavailable(records['goal_reached'], 'invalidated')
    assert records['goal_reached']['counts']['observed_events'] == 1
    assert_unavailable(records['time_to_goal_s'], 'invalidated')
    assert records['time_to_goal_s']['counts']['observed_at_rel_s'] == DT
    assert records['time_to_goal_s']['counts']['observed_after_s'] == DT
    assert records['wall_contact_onsets']['value'] == 1
    assert observed.get_evidence() == evidence_before

    absent = new('labyrinth')
    short_window(absent, 2 * DT)
    lb_run(absent, Fly(10, 10), 2, contact=True)
    assert_value(rec(absent, 'goal_reached'), False)
    assert_unavailable(rec(absent, 'time_to_goal_s'), 'censored')
    evidence_before = copy.deepcopy(absent.get_evidence())

    records = absent.freeze_observation('fault_halt')['records']
    assert_unavailable(records['goal_reached'], 'invalidated')
    assert records['goal_reached']['counts']['observed_events'] == 0
    assert_unavailable(records['time_to_goal_s'], 'invalidated')
    assert records['wall_contact_onsets']['value'] == 1
    assert absent.get_evidence() == evidence_before


def test_C10A_fault_does_not_rewrite_an_earlier_frozen_presentation():
    p = looming('geometric')
    run(p, Fly(40, 40), n=20)
    prior = p.freeze_observation('re_presentation_user')
    prior_bytes = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode()
    assert_value(prior['records']['escape_initiated'], True)
    assert_value(prior['records']['initiation_latency_s'], 0.36, approx=True)

    p.begin_next_presentation('user_loom')
    run(p, Fly(40, 40), n=1)
    current = p.freeze_observation('fault_halt')
    assert_unavailable(current['records']['escape_initiated'], 'invalidated')
    assert_unavailable(current['records']['initiation_latency_s'], 'invalidated')
    assert json.dumps(prior, sort_keys=True, separators=(',', ':')).encode() == prior_bytes

# ============================================================================
# CARD-10B: effective observation policy and presentation boundaries
# ============================================================================
def test_C10B_continuous_policy_retains_unused_duration_and_flag_provenance():
    dam = new('circadian_dam')
    unused = config(dam, 'dam-unused-duration', override_window_s=0.04, override_source='api',
                    set_at_sim_s=3.0, manifest_run_id='run-policy')
    applied = dam.configure_observation(unused)
    assert applied['effective_window_s'] is None and applied['automatic_end'] is False
    assert applied['window_source'] == 'spec'
    assert applied['override'] == {'source': 'api', 'value': 0.04, 'set_at_sim_s': 3.0,
                                   'manifest_run_id': 'run-policy'}
    run(dam, Fly(50, 50), n=10)
    assert dam.observation_status()['state'] == 'observing'
    env = dam.freeze_observation('shutdown')
    assert env['effective_window_s'] is None and env['window_s'] is None
    assert env['automatic_end'] is False and env['window_source'] == 'spec'
    assert env['override']['value'] == 0.04

    presentation = looming('connectome')
    flagged = presentation.configure_observation(config(presentation, 'explicit-continuous', continuous=True,
                                                        set_at_sim_s=0.0, manifest_run_id='run-policy'))
    assert flagged['effective_window_s'] is None and flagged['automatic_end'] is False
    assert flagged['window_source'] == 'continuous_flag'
    assert flagged['override']['source'] == 'cli:--continuous' and flagged['override']['value'] is None
    run(presentation, Fly(40, 40), seconds=3.0)
    assert presentation.observation_status()['state'] == 'observing'
    assert presentation.observation_status()['measurement_end_rel_s'] is None


def test_C10B_timed_presentation_override_changes_only_observation_window_at_nonzero_start():
    p = looming('connectome')
    fly = Fly(40, 40)
    run(p, fly, n=10)
    p.freeze_observation('re_presentation_user')
    override = config(p, 'loom-3s', override_window_s=3.0, override_source='api',
                      set_at_sim_s=0.2, manifest_run_id='run-policy')
    p.begin_next_presentation('policy_change', override)

    reference = looming('connectome')
    actual = p.step(fly, DT)
    expected = reference.step(Fly(40, 40), DT)
    assert actual['stimuli'] == expected['stimuli']
    assert p.t_collision_s == reference.t_collision_s == 0.4
    assert p.r_over_v_s == reference.r_over_v_s == 0.02
    status = p.observation_status()
    assert status['presentation_start_rel_s'] == 0.2
    assert status['window_end_rel_s'] == 3.2 and status['effective_window_s'] == 3.0

    run(p, fly, n=149)
    assert p.observation_status()['measurement_end_rel_s'] == 3.2
    env = p.freeze_observation()
    assert env['effective_window_s'] == 3.0 and env['window_source'] == 'override'
    assert env['override']['value'] == 3.0 and env['dt_s'] == DT
    assert env['window_params'] == reference.observation_spec()['window_params']


def test_C10B_open_policy_change_is_incomplete_but_completed_envelope_keeps_its_reason():
    open_observation = new('t_maze')
    open_observation.configure_observation(config(open_observation, 'open-before'))
    run(open_observation, Fly(*STEM), n=5)
    open_env = open_observation.freeze_observation('policy_change')
    assert open_env['completeness'] == 'incomplete' and open_env['end_reason'] == 'policy_change'

    completed = new('t_maze')
    completed.configure_observation(config(completed, 'completed-before', override_window_s=0.04,
                                            override_source='api'))
    run(completed, Fly(*STEM), n=2)
    original = completed.freeze_observation()
    original_bytes = json.dumps(original, sort_keys=True, separators=(',', ':')).encode()
    assert original['end_reason'] == 'window_elapsed' and original['completeness'] == 'complete'

    repeated = completed.freeze_observation('policy_change')
    assert repeated == original
    completed.begin_next_presentation('policy_change',
                                      config(completed, 'after-policy', override_window_s=0.1,
                                             override_source='api', set_at_sim_s=0.04))
    assert json.dumps(original, sort_keys=True, separators=(',', ':')).encode() == original_bytes
    status = completed.observation_status()
    assert status['state'] == 'observing' and status['end_reason'] is None
    assert status['presentation_start_rel_s'] == 0.04 and status['window_end_rel_s'] == 0.14
