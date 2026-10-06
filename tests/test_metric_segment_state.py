"""C0 vectors for physical state and segment metrics across presentations."""
import copy

import pytest

from online_metrics import OpenArenaObserver
from tests.test_metric_contract import DT, Fly, config, new, rec, ym_visit


def _tick_contact(p, fly, contact):
    p.step(fly, DT)
    p.observe_contact(p._v1_last_step, DT, 'post_solver', contact)


def _assert_value(p, name, value):
    r = rec(p, name)
    assert r['available'] is True, r
    assert r['value'] == pytest.approx(value)
    assert r['interval_rel_s'] == pytest.approx([0.0, p.observation_status()['segment_elapsed_s']])


def test_labyrinth_policy_change_preserves_contact_hysteresis_and_dead_end_state():
    p = new('labyrinth')
    fly = Fly(30, 75)
    _tick_contact(p, fly, True)
    frozen = p.freeze_observation('policy_change')
    frozen_copy = copy.deepcopy(frozen)

    p.begin_next_presentation('policy_change', config(p, 'lb-180', override_window_s=180))
    _tick_contact(p, fly, True)

    _assert_value(p, 'wall_contact_onsets', 1)
    _assert_value(p, 'wall_contact_s', 2 * DT)
    _assert_value(p, 'dead_end_entries', 1)
    _assert_value(p, 'dead_end_s', 2 * DT)
    events = p.get_evidence()['contact_events']['data']
    assert [event['event'] for event in events] == ['contact_onset']
    assert frozen == frozen_copy

    _tick_contact(p, Fly(30, 65), False)
    _tick_contact(p, fly, True)
    _assert_value(p, 'wall_contact_onsets', 2)
    _assert_value(p, 'dead_end_entries', 2)

    p.reset_trial()
    assert p.observation_status()['segment_elapsed_s'] == 0.0
    assert rec(p, 'wall_contact_onsets')['reason'] == 'not_observed'
    assert rec(p, 'dead_end_entries')['reason'] == 'not_observed'
    _tick_contact(p, fly, True)
    _assert_value(p, 'wall_contact_onsets', 1)
    _assert_value(p, 'dead_end_entries', 1)


def test_y_maze_sequence_rearm_and_interval_span_multiple_presentations():
    p = new('y_maze')
    ym_visit(p, 'A')
    first = p.freeze_observation('policy_change')
    first_copy = copy.deepcopy(first)

    for index in range(2):
        p.begin_next_presentation('policy_change', config(p, f'ym-{index}', override_window_s=180))
        ym_visit(p, 'A')
        _assert_value(p, 'physical_entries', 1)

    ym_visit(p, 'hub', 'A')
    _assert_value(p, 'physical_entries', 2)
    assert p.get_evidence()['physical_visit_sequence']['data']['symbols'] == ['A', 'A']
    assert first == first_copy

    p.reset_trial()
    assert rec(p, 'physical_entries')['reason'] == 'not_observed'
    ym_visit(p, 'A')
    _assert_value(p, 'physical_entries', 1)


def test_dam_counts_immobility_and_bins_span_presentations_from_segment_origin():
    p = new('circadian_dam')
    p.step(Fly(10, 5), DT)
    p.step(Fly(40, 5), DT)
    frozen = p.freeze_observation('policy_change')
    frozen_copy = copy.deepcopy(frozen)

    p.begin_next_presentation('policy_change', config(p, 'dam-continuation'))
    p.step(Fly(40, 5), DT)
    _assert_value(p, 'beam_crossings', 1)
    _assert_value(p, 'current_immobile_s', DT)
    bins = p.get_evidence()['activity_bins']['data']
    assert bins == {'bin_s': 60.0, 'start_rel_s': 0.0, 'quantity': 'beam_crossings', 'counts': [1]}
    assert frozen == frozen_copy

    p.reset_trial()
    assert rec(p, 'beam_crossings')['reason'] == 'not_observed'
    assert p.get_evidence()['activity_bins']['data']['counts'] == []


def test_multisensory_contact_and_accumulators_span_presentations():
    p = new('multisensory')
    food = Fly(45, 45, speed=1)
    hotspot = Fly(-40, 40, speed=2)
    _tick_contact(p, food, True)
    frozen = p.freeze_observation('policy_change')
    frozen_copy = copy.deepcopy(frozen)

    p.begin_next_presentation('policy_change', config(p, 'ms-180', override_window_s=180))
    _tick_contact(p, hotspot, True)
    _assert_value(p, 'wall_contact_onsets', 1)
    _assert_value(p, 'wall_contact_s', 2 * DT)
    _assert_value(p, 'odor_a_exposure_s', DT)
    _assert_value(p, 'heat_exposure_s', DT)
    _assert_value(p, 'distance_mm', (85 ** 2 + 5 ** 2) ** 0.5)

    _tick_contact(p, hotspot, False)
    _tick_contact(p, hotspot, True)
    _assert_value(p, 'wall_contact_onsets', 2)
    assert frozen == frozen_copy

    p.reset_trial()
    assert rec(p, 'wall_contact_onsets')['reason'] == 'not_observed'
    assert rec(p, 'odor_a_exposure_s')['reason'] == 'not_observed'


def test_presentation_scoped_open_arena_values_still_reset():
    p = OpenArenaObserver()
    p.observe(Fly(0, 0), True, DT)
    frozen = p.freeze_observation('re_presentation_user')
    frozen_copy = copy.deepcopy(frozen)
    p.begin_next_presentation('user_windStrength')

    assert rec(p, 'food_contacts')['value'] == 1
    assert rec(p, 'presentation_food_contacts')['reason'] == 'not_observed'
    assert frozen == frozen_copy
