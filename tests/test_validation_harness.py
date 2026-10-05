"""Validation harness: specs, statistics, verdict rule and the synthetic code path.

No real graph is needed.  The synthetic runs prove the encoder -> graph ->
decoder -> gate -> receipt path works for every paradigm; they make no
scientific claim, and the receipt says so.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation import harness, stats
from validation.cells import CellTable

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / 'validation/specs'
SPEC_FILES = ('optomotor_v3.json', 'optomotor_v3_2.json', 'looming_gf_v3.json', 'tmaze_odour_naive_v3.json')


# --- specs -----------------------------------------------------------------
@pytest.mark.parametrize('name', SPEC_FILES)
def test_shipped_specs_are_complete(name):
    spec = harness.load_spec(SPECS / name)
    bounds = harness.load_bounds(spec)
    assert spec['dynamics']['version'] == 'v3'
    for check in spec['physiology']['checks']:
        assert check['bound'] in bounds['bounds'], check
    for gate in spec['behaviour']['gates']:
        assert isinstance(gate['verified'], bool)
        # A literature-anchored gate must name a source that was actually read.
        for src in gate['sources']:
            assert 'NOT READ' not in spec['citations'][src]['read'] or gate['verified'] is False, gate['id']


def test_only_fully_verified_specs_are_preregistered():
    for name in SPEC_FILES:
        spec = harness.load_spec(SPECS / name)
        if spec['status'] == 'preregistered':
            assert all(g['verified'] for g in spec['behaviour']['gates']), name


def test_spec_rejects_unknown_citation(tmp_path):
    spec = json.loads((SPECS / 'optomotor_v3.json').read_text())
    spec['behaviour']['gates'][0]['sources'] = ['nobody1999']
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(harness.SpecError):
        harness.load_spec(path)


# --- statistics ------------------------------------------------------------
def test_clopper_pearson_matches_known_values():
    lo, hi = stats.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.30850, abs=1e-4)
    lo, hi = stats.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.18709, abs=1e-4) and hi == pytest.approx(0.81291, abs=1e-4)


def test_bootstrap_is_seeded():
    a = stats.bootstrap_mean([0.1, 0.3, -0.2, 0.5], seed=1)
    b = stats.bootstrap_mean([0.1, 0.3, -0.2, 0.5], seed=1)
    assert a == b and a['ci'][0] <= a['mean'] <= a['ci'][1]


def test_slope_sign():
    groups = {math.log10(8): [1] * 9 + [0], math.log10(40): [1] * 5 + [0] * 5, math.log10(140): [1] + [0] * 9}
    s = stats.bootstrap_slope(groups, seed=3, n_boot=2000)
    assert s['slope'] < 0 and s['ci'][1] < 0


def test_ratio_undefined_when_denominator_can_vanish():
    s = stats.bootstrap_ratio_of_means([1, 0], [1, 0], seed=0, n_boot=200)
    assert s['ci'] == [None, None] and s['undefined_resamples'] > 0


# --- verdict rule ------------------------------------------------------------
def _g(passed, verified=True):
    return dict(id='g', passed=passed, verified=verified)


def test_verdict_needs_behaviour_and_physiology():
    assert harness.decide([_g(True)], [_g(True)], [], synthetic=False)[0] == 'PASS'
    assert harness.decide([_g(True)], [_g(False)], [], synthetic=False)[0] == 'FAIL'
    assert harness.decide([_g(False)], [_g(True)], [], synthetic=False)[0] == 'FAIL'
    assert harness.decide([_g(True)], [_g(None)], [], synthetic=False)[0] == 'INCONCLUSIVE'
    assert harness.decide([_g(True)], [_g(True, verified=False)], [], synthetic=False)[0] == 'PASS_PROVISIONAL'
    assert harness.decide([_g(True)], [_g(True)], ['draft'], synthetic=False)[0] == 'EXPLORATORY'
    assert harness.decide([_g(True)], [_g(True)], [], synthetic=True)[0] == 'SYNTHETIC_PLUMBING_ONLY'


# --- encoders --------------------------------------------------------------
def test_rate_inversion_reproduces_target_rate():
    from brainlab.brain import Brain
    from validation.paradigms.tmaze_odor import drive_for_rate
    arrays = dict(ptr=np.zeros(2, np.int64), post=np.zeros(0, np.int32), weight=np.zeros(0, np.float32),
                  ids=np.array([1], np.int64))
    for target in (10.0, 50.0, 150.0):
        brain = Brain(arrays=arrays, dynamics='v3', backend='cpu')
        counts, _ = brain.step(np.full(1, drive_for_rate(np.array([target]))[0], np.float32), 2000.0)
        assert counts[0] / 2.0 == pytest.approx(target, rel=0.1)


def test_looming_angle_series_reaches_collision():
    from validation.paradigms.looming import theta_series
    stim = dict(step_ms=2.0, theta_start_deg=10.0)
    th = theta_series(40, stim)
    assert th[0] == pytest.approx(10.0, abs=1e-6) and th[-1] > 150 and np.all(np.diff(th) > 0)


def test_looming_angle_series_stops_at_end_size():
    from validation.paradigms.looming import theta_series
    stim = dict(step_ms=2.0, theta_start_deg=5.0, theta_end_deg=90.0)
    th = theta_series(40, stim)
    assert th[0] == pytest.approx(5.0, abs=1e-6) and th[-1] == 90.0 and np.all(np.diff(th) > 0)
    assert th[-2] < 90.0
    back = theta_series(40, stim, receding=True)
    assert back[0] == 90.0 and back[-1] == pytest.approx(5.0, abs=1e-6)


def test_proportion_within_gate():
    gate = dict(id='g', type='proportion_ci_within', claim='', verified=True, sample='s', interval=[0.149, 0.705])
    assert harness.evaluate_gate(gate, dict(s=dict(k=36, n=90)), 1, 100)['passed'] is True
    assert harness.evaluate_gate(gate, dict(s=dict(k=15, n=90)), 1, 100)['passed'] is False   # CI reaches below the floor
    assert harness.evaluate_gate(gate, dict(s=dict(k=85, n=90)), 1, 100)['passed'] is False
    assert harness.evaluate_gate(gate, dict(s=dict(k=0, n=0)), 1, 100)['passed'] is None


def test_cell_table_takes_side_from_root_side_when_soma_side_missing(tmp_path):
    import pyarrow as pa
    import pyarrow.feather as feather
    (tmp_path / 'normalized').mkdir()
    feather.write_feather(pa.table(dict(node_index=[0, 1, 2], source_id=[10, 11, 12],
                                        cell_type=['ORN_DM1', 'ORN_DM1', 'DNa02'])),
                          tmp_path / 'normalized/neurons.feather')
    feather.write_feather(pa.table(dict(bodyId=[10, 11, 12], somaSide=[None, None, 'L'], rootSide=['L', 'R', None])),
                          tmp_path / 'annotations.feather')
    cells = CellTable.from_connectome(tmp_path)
    assert list(cells.select(['ORN_DM1'], 'R').node_index) == [1]
    row = cells.frame.set_index('node_index')
    assert row.loc[0, 'side_source'] == 'rootSide' and row.loc[2, 'side_source'] == 'somaSide'


# --- end to end on the synthetic graph ---------------------------------------
@pytest.mark.parametrize('name,seeds', [('optomotor_v3.json', [0, 1, 2]), ('looming_gf_v3.json', [0, 1, 2]),
                                         ('tmaze_odour_naive_v3.json', [0, 1, 2])])
def test_synthetic_end_to_end_writes_complete_receipt(tmp_path, name, seeds):
    receipt = harness.run(SPECS / name, tmp_path / 'out', synthetic=True, backend='cpu', seeds=seeds,
                          log=lambda *_: None)
    assert receipt['verdict'] == 'SYNTHETIC_PLUMBING_ONLY'
    assert 'synthetic graph' in receipt['not_confirmatory_because']
    saved = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert saved['graph']['identity']['synthetic'] is True
    assert saved['spec']['sha256'] == harness.load_spec(SPECS / name)['_sha256']
    assert saved['dynamics']['version'] == 'v3' and saved['dynamics']['pin']
    assert saved['brain']['runners'][0]['backend'] == 'cpu'
    assert saved['seeds'] == seeds and saved['code']['commit']
    assert len(saved['gates']) == len(harness.load_spec(SPECS / name)['behaviour']['gates'])
    assert all(c['n_trials'] > 0 for c in saved['physiology'])


def test_synthetic_optomotor_controls_behave(tmp_path):
    receipt = harness.run(SPECS / 'optomotor_v3.json', tmp_path / 'o', synthetic=True, backend='cpu',
                          seeds=[0, 1, 2], log=lambda *_: None)
    gates = {g['id']: g for g in receipt['gates']}
    # DNa02 clamped -> yaw exactly zero; nothing delivered -> TI of the sham is 0, so intact-sham = intact.
    assert gates['O7_needs_DNa02']['passed'] is True
    assert gates['O5_needs_input']['statistic']['mean'] == pytest.approx(gates['O1_syndirectional']['statistic']['mean'])


def test_out_dir_is_never_overwritten(tmp_path):
    (tmp_path / 'exists').mkdir()
    with pytest.raises(FileExistsError):
        harness.run(SPECS / 'looming_gf_v3.json', tmp_path / 'exists', synthetic=True, backend='cpu', seeds=[0],
                    log=lambda *_: None)


def test_optomotor_v3_2_amends_only_what_it_declares():
    v1 = harness.load_spec(SPECS / 'optomotor_v3.json')
    v2 = harness.load_spec(SPECS / 'optomotor_v3_2.json')
    for key in ('behaviour', 'stimulus', 'encoder', 'decoder', 'conditions', 'controls', 'decision_rule', 'dynamics'):
        assert v1[key] == v2[key], key
    assert v2['amendment']['prompted_by'] and set(v2['seeds']).isdisjoint(v1['seeds'])
    moved = {c['population'] for c in v2['physiology']['reported']}
    assert moved == {'HS_L', 'HS_R'}
    assert [c for c in v1['physiology']['checks'] if c['population'] not in moved] == v2['physiology']['checks']


def test_reported_physiology_never_enters_the_verdict(tmp_path):
    receipt = harness.run(SPECS / 'optomotor_v3_2.json', tmp_path / 'r', synthetic=True, backend='cpu',
                          seeds=[100], log=lambda *_: None)
    assert {c['id'] for c in receipt['physiology_reported']} == {'R1_HS_L_driven', 'R2_HS_R_driven'}
    assert not {c['id'] for c in receipt['physiology']} & {'R1_HS_L_driven', 'R2_HS_R_driven'}
