"""C0 vectors for the declared open-arena airflow presentation condition."""
import copy
import json

import pytest

from arena import Arena
from assay_controls import preflight, set_parameter
from online_metrics import MetricFault, OpenArenaObserver
from tests.test_metric_contract import DT, Fly, rec


def _snapshot(observer):
    return json.dumps({
        'status': observer.observation_status(),
        'records': observer.get_metric_records(),
        'evidence': observer.get_evidence(),
    }, sort_keys=True, separators=(',', ':'))


def test_zero_and_unchanged_airflow_are_valid_and_first_sample_declares_condition():
    observer = OpenArenaObserver(airflow_mm_s=12.0)
    observer.observe(Fly(0, 0), False, DT, airflow_mm_s=0.0)
    observer.observe(Fly(1, 0), False, DT, airflow_mm_s=-0.0)

    assert rec(observer, 'airflow_mm_s')['value'] == 0.0
    assert rec(observer, 'presentation_observation_s')['value'] == pytest.approx(2 * DT)
    assert observer.observation_spec()['window_params']['airflow_mm_s'] == 0.0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_airflow_is_rejected_before_condition_or_prefix_mutation(bad):
    observer = OpenArenaObserver(airflow_mm_s=5.0)
    observer.observe(Fly(0, 0), True, DT, airflow_mm_s=5.0)
    before = _snapshot(observer)

    with pytest.raises(MetricFault) as caught:
        observer.observe(Fly(1, 0), False, DT, airflow_mm_s=bad)

    assert caught.value.path == 'open_arena.airflow_mm_s'
    assert observer.airflow_mm_s == 5.0
    assert _snapshot(observer) == before


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_initial_airflow_is_refused(bad):
    with pytest.raises(MetricFault) as caught:
        OpenArenaObserver(airflow_mm_s=bad)
    assert caught.value.path == 'open_arena.airflow_mm_s'


def test_unlogged_sampled_change_faults_without_replacing_declared_condition():
    observer = OpenArenaObserver(airflow_mm_s=5.0)
    observer.observe(Fly(0, 0), True, DT, airflow_mm_s=5.0)
    before = _snapshot(observer)

    with pytest.raises(MetricFault, match='freeze and begin a new presentation') as caught:
        observer.observe(Fly(2, 0), True, DT, airflow_mm_s=20.0)

    assert caught.value.path == 'open_arena.airflow_mm_s'
    assert observer.airflow_mm_s == 5.0
    assert _snapshot(observer) == before
    assert rec(observer, 'airflow_mm_s')['value'] == 5.0


def test_external_unlogged_change_keeps_prior_declaration_in_fault_envelope():
    observer = OpenArenaObserver(airflow_mm_s=5.0)
    observer.observe(Fly(0, 0), True, DT)
    prefix = _snapshot(observer)
    observer.airflow_mm_s = 20.0

    with pytest.raises(MetricFault):
        observer.observe(Fly(2, 0), True, DT)

    assert _snapshot(observer) == prefix
    envelope = observer.freeze_observation('fault_halt')
    assert envelope['window_params']['airflow_mm_s'] == 5.0
    assert envelope['records']['airflow_mm_s']['value'] == 5.0
    assert envelope['records']['observation_s']['value'] == DT
    assert envelope['records']['food_contacts']['value'] == 1


def test_logged_control_change_uses_new_presentation_and_preserves_contact_state():
    arena = Arena(paradigm=None, seed=4, num_flies=1, num_predators=0)
    arena.wind = (-5.0, 0.0)
    observer = OpenArenaObserver(airflow_mm_s=5.0)
    fly = Fly(10, 10)
    observer.observe(fly, True, DT, airflow_mm_s=-arena.wind[0])

    arena_before = tuple(arena.wind)
    observer_before = _snapshot(observer)
    accepted = preflight(arena, 'set_param', 'windStrength', 20.0)
    assert accepted['accepted'] and accepted['closes_presentation']
    assert accepted['normalized_value'] == 20.0
    assert tuple(arena.wind) == arena_before
    assert _snapshot(observer) == observer_before

    prior = observer.freeze_observation('re_presentation_user')
    prior_copy = copy.deepcopy(prior)
    applied = set_parameter(arena, 'windStrength', accepted['normalized_value'], detailed=True)
    assert applied == {'applied': True, 'value': 20.0, 'presentation_closed': True}
    observer.airflow_mm_s = applied['value']
    observer.begin_next_presentation('user_windStrength')
    observer.log_intervention('windStrength', applied['value'])
    observer.observe(fly, True, DT, airflow_mm_s=-arena.wind[0])

    records = observer.get_metric_records()
    assert records['airflow_mm_s']['value'] == 20.0
    assert records['food_contacts']['value'] == 1
    assert records['presentation_food_contacts']['value'] == 0
    assert records['observation_s']['value'] == pytest.approx(2 * DT)
    assert records['presentation_observation_s']['value'] == pytest.approx(DT)
    events = observer.get_evidence()['contact_events']['data']
    assert len(events) == 1 and events[0]['attrs']['presentation_index'] == 0
    assert observer.get_evidence()['interventions']['data'][-1]['name'] == 'windStrength'
    assert prior == prior_copy
    assert prior['records']['airflow_mm_s']['value'] == 5.0
    assert prior['records']['presentation_food_contacts']['value'] == 1
