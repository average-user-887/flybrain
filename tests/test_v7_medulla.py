"""v7 medulla timing: data loading, observation model, fit harness and prereg consistency.
No fit and no network simulation beyond a 400-cell random graph."""
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / 'scripts' / 'v7'
sys.path.insert(0, str(V7)); sys.path.insert(0, str(ROOT / 'scripts' / 'v6'))
import v7_data as D  # noqa: E402
import fit_v7 as F  # noqa: E402
import heldout_v7 as H  # noqa: E402
import digitise_behnia2014 as G  # noqa: E402

PREREG = ROOT / 'qualification' / 'v7' / 'V7_medulla_prereg.json'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _biphasic(n=D.N_LAGS, peak=50.0, sign=1.0, amp=0.6):
    t = np.arange(n, dtype=float)
    k = (t / peak) ** 3 * np.exp(3 * (1 - t / peak)) - 0.15 * (t / (2 * peak)) ** 3 * np.exp(3 * (1 - t / (2 * peak)))
    return sign * amp * k / np.abs(k).max()


def _data_from(filters, sem=0.03):
    return {c: dict(t_ms=np.arange(D.N_LAGS, dtype=float), mean=filters[c].copy(),
                    sem=np.full(D.N_LAGS, sem), interpolated=np.zeros(D.N_LAGS, bool)) for c in D.RECORDED}


FILTERS = {'Mi1': _biphasic(peak=70, sign=1, amp=0.62), 'Tm3': _biphasic(peak=55, sign=1, amp=0.46),
           'Tm1': _biphasic(peak=54, sign=-1, amp=0.56), 'Tm2': _biphasic(peak=44, sign=-1, amp=0.63)}


# --- stimulus -----------------------------------------------------------------------------

def test_behnia_noise_statistics_and_determinism():
    s = D.behnia_noise(60_000, seed=1)
    assert s.min() >= -1 and s.max() <= 1
    assert 0.44 < s.std() < 0.50                      # s.d. 0.5 before clipping at |s| = 1
    ac = [np.corrcoef(s[:-k], s[k:])[0, 1] for k in (4, 8, 10, 21)]
    # exponential autocovariance with tau 10 ms, sampled on 240 Hz frames (+ frame hold)
    assert abs(ac[2] - math.exp(-1.0)) < 0.08 and ac[3] < ac[2] < ac[1] < ac[0]
    assert np.array_equal(s, D.behnia_noise(60_000, seed=1))
    assert not np.array_equal(s, D.behnia_noise(60_000, seed=2))
    held = D.behnia_noise(100, seed=3)
    assert held[0] == held[3] and held[0] != held[5]     # one 240 Hz frame = 4.17 ms


def test_encoder_mapping_mean_is_s3_gray_level():
    assert D.contrast_to_encoder(0.0) == 10.0
    assert D.contrast_to_encoder(-1.0) == 0.0 and D.contrast_to_encoder(1.0) == 20.0


# --- observation model --------------------------------------------------------------------

def test_filter_estimator_recovers_a_known_filter():
    s = D.behnia_noise(D.NOISE_MS, seed=4)
    k = _biphasic(peak=55, amp=0.5)
    r = np.convolve(s, k)[:len(s)] + np.random.default_rng(0).normal(0, 0.2, len(s))
    est = D.observe_filter(s, r - 60.0)                 # resting offset is irrelevant
    nrmse = np.sqrt(((est - k) ** 2).sum() / (k ** 2).sum())
    assert nrmse < 0.15
    assert abs(D.peak_ms(est, 'Mi1') - 55) <= 3


def test_v7_objective_sees_absolute_mV_where_the_v6_calcium_model_cannot():
    """fail-old / pass-new: a model whose voltage response is twice too large must be penalised.
    The v6 S2 calcium observation model normalises to max |value| and cannot see it; the v7 mV
    observation model (same estimator as the data) does."""
    import fit_s2
    rng = np.random.default_rng(1)
    v = np.cumsum(rng.normal(0, 0.3, 2000))
    old_1, old_2 = fit_s2.observe(v), fit_s2.observe(2 * v)
    assert np.allclose(old_1, old_2)                       # old: amplitude-blind (the defect)
    data = _data_from(FILTERS)
    good = D.objective(FILTERS, data)[0]
    double = D.objective({c: 2 * FILTERS[c] for c in D.RECORDED}, data)[0]
    assert good == 0.0 and double > 50                    # new: a 2x mV error is a large cost


# --- data loading and held-out separation -------------------------------------------------

def _write_csv(path, mean):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['t_ms', 'filter_mV_per_contrast_per_ms', 'sem', 'interpolated'])
        for t, m in enumerate(mean):
            w.writerow([t, f'{m:.5f}', '0.03000', 0])


def test_training_loader_checks_sha_and_refuses_heldout(tmp_path):
    d = tmp_path / 'training'; d.mkdir()
    pins = {}
    for c in D.RECORDED:
        p = d / f'f_{c}.csv'; _write_csv(p, FILTERS[c]); pins[c] = dict(file=p.name, sha256=_sha(p))
    got = D.load_training(d, pins)
    assert np.allclose(got['Tm2']['mean'], FILTERS['Tm2'], atol=1e-5)
    bad = dict(pins); bad['Mi1'] = dict(pins['Mi1'], sha256='0' * 64)
    with pytest.raises(ValueError):
        D.load_training(d, bad)
    with pytest.raises(ValueError):
        D.load_training(d, {'T4a': pins['Mi1']})
    h = tmp_path / 'heldout'; h.mkdir()
    with pytest.raises(ValueError):
        D.load_training(h, {})


def test_fit_harness_never_reads_the_heldout_set():
    for mod in ('v7_data.py', 'fit_v7.py'):
        tree = ast.parse((V7 / mod).read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not any('heldout' in x for x in names), mod
        src = (V7 / mod).read_text()
        assert 'load_heldout' not in src and 'fig2' not in src.lower()


# --- fit harness --------------------------------------------------------------------------

def _tiny_brain():
    from brainlab.v6_calibrated import BrainV6
    from tests.test_graded_v4 import random_graph
    arrays = random_graph(); n = len(arrays['ids'])
    ct = np.array(['R1-R6'] * 20 + ['L1'] * 20 + ['L2'] * 20 + ['Mi1'] * 10 + ['Tm3'] * 10 + ['Tm1'] * 10
                  + ['Tm2'] * 10 + ['X'] * (n - 100))
    p = {'types': {'R1-R6': {'release': {'kind': 'sigmoid', 'gain': 2.0, 'vh_mV': -47.0, 's_mV': 1.0}},
                   'L1': {'tau_m_ms': 199.9, 'e_leak_mV': -40.0}}}
    b = BrainV6(None, p, 'p' * 64, cell_type=ct, arrays=dict(arrays), graded_policy=np.ones(n, np.uint8))
    return b, ct, arrays


def test_apply_params_touches_only_declared_types_and_never_weights():
    b, ct, arrays = _tiny_brain()
    before = {k: getattr(b, k).copy() for k in ('tau_m', 'e_leak', 'rel_gain', 'weight', 'post', 'ptr')}
    idx = {t: np.flatnonzero(ct == t) for t in set(ct.tolist())}
    fitted = {'Mi1': {'tau_m_ms': 33.0, 'e_leak_mV': -60.0}, 'L1': {'release_gain': 4.0, 'tau_m_ms': 12.0}}
    F.apply_params(b, idx, fitted)
    assert np.all(b.tau_m[idx['Mi1']] == 33.0) and np.all(b.e_leak[idx['Mi1']] == -60.0)
    assert np.all(b.rel_gain[idx['L1']] == 4.0) and np.all(b.tau_m[idx['L1']] == 12.0)
    others = np.setdiff1d(np.arange(b.n), np.concatenate([idx['Mi1'], idx['L1']]))
    for k in ('tau_m', 'e_leak', 'rel_gain'):
        assert np.array_equal(getattr(b, k)[others], before[k][others]), k
    for k in ('weight', 'post', 'ptr'):
        assert np.array_equal(getattr(b, k), before[k]), k
    with pytest.raises(ValueError):                       # sigmoid-release types are not refitted
        F.apply_params(b, idx, {'R1-R6': {'release_gain': 3.0}})
    with pytest.raises(ValueError):
        F.apply_params(b, idx, {'Mi1': {'weight': 3.0}})


def test_params_file_changes_only_fitted_keys_and_loads(tmp_path):
    from brainlab.v6_calibrated import BrainV6, load_params
    base = json.loads((ROOT / 'qualification/v6/params_v6_S2A_leak40.json').read_text())
    fitted = {'Mi1': {'tau_m_ms': 40.0, 'e_leak_mV': -58.0}, 'L1': {'release_gain': 3.0}}
    out = F.params_file(base, 'b' * 64, 'p' * 64, fitted, 'test')
    for t in base['types']:
        for k, v in base['types'][t].items():
            if not (t in fitted and k in fitted[t]):
                assert out['types'][t][k] == v, (t, k)
    assert out['types']['L1']['release'] == {'kind': 'linear', 'gain': 3.0}
    assert out['types']['L1']['ih'] == base['types']['L1']['ih']
    assert out['phototransduction'] == base['phototransduction']
    p = tmp_path / 'p.json'; p.write_text(json.dumps(out))
    params, sha = load_params(p, _sha(p))
    b, ct, arrays = _tiny_brain()
    b2 = BrainV6(None, params, sha, cell_type=ct, arrays=dict(arrays), graded_policy=np.ones(b.n, np.uint8))
    assert np.all(b2.rel_gain[ct == 'L1'] == 3.0) and np.all(b2.rel_s[ct == 'L1'] == 0.0)
    assert np.all(b2.tau_m[ct == 'Mi1'] == 40.0)


def test_unpack_clips_to_prereg_bounds_and_starts_are_fixed():
    pre = json.loads(PREREG.read_text())
    spec = F.free_spec(pre)
    big = F.from_x(np.full(len(spec), 50.0), spec)
    small = F.from_x(np.full(len(spec), -500.0), spec)
    for t, k, lo, hi, _ in spec:
        assert big[t][k] == hi and small[t][k] == lo
    base = json.loads((ROOT / 'qualification/v6/params_v6_S2A_leak40.json').read_text())
    s1, s2 = F.starts(pre, base, spec), F.starts(pre, base, spec)
    assert len(s1) == pre['optimiser']['n_starts']
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2))
    x0 = F.from_x(s1[0], spec)
    for (t, k, *_), v in zip(spec, F.base_values(base, spec)):
        assert math.isclose(x0[t][k], v, rel_tol=1e-12)


def test_noise_model_refuses_a_cutout_with_motion_cells():
    ct = np.array(['Mi1', 'Tm3', 'Tm1', 'Tm2', 'T4a'])
    cc = {t: np.array([i]) for i, t in enumerate(['Mi1', 'Tm3', 'Tm1', 'Tm2'])}
    with pytest.raises(ValueError):
        F.NoiseModel(None, ct, np.array([], int), cc, seed=1)


def test_selection_rule_is_lowest_training_cost_then_lowest_start():
    reps = [dict(start=0, cost=2.0), dict(start=1, cost=1.0), dict(start=2, cost=1.0)]
    assert F.select(reps)['start'] == 1


# --- gates --------------------------------------------------------------------------------

def test_train_gates_pass_on_data_and_fail_on_reversed_delay():
    data = _data_from(FILTERS)
    assert D.train_gates(FILTERS, data)['TRAIN_PASS']
    swapped = dict(FILTERS, Mi1=_biphasic(peak=45, sign=1, amp=0.62))
    g = D.train_gates(swapped, data)
    assert not g['A3_order']['ON'] and not g['TRAIN_PASS']
    flipped = dict(FILTERS, Tm1=-FILTERS['Tm1'])
    assert not D.train_gates(flipped, data)['A4_polarity']['Tm1']


def test_heldout_gates():
    held = {'Mi1': dict(onset_peak_mV=30.2, offset_peak_mV=-3.3), 'Tm3': dict(onset_peak_mV=24.7, offset_peak_mV=-8.4),
            'Tm1': dict(onset_peak_mV=-5.9, offset_peak_mV=22.1), 'Tm2': dict(onset_peak_mV=-3.7, offset_peak_mV=20.0)}
    assert H.heldout_gates(held, held)['HELDOUT_PASS']
    bad = dict(held, Tm1=dict(onset_peak_mV=5.9, offset_peak_mV=-22.1))
    assert not H.heldout_gates(bad, held)['H1_polarity']['Tm1']
    small = {c: {k: v / 3 for k, v in held[c].items()} for c in held}
    assert not H.heldout_gates(small, held)['HELDOUT_PASS']
    v = np.full(H.PRE_MS + H.FLASH_MS + H.POST_MS, -55.0)
    v[H.PRE_MS + 30] = -40.0; v[H.PRE_MS + H.FLASH_MS + 50] = -58.0
    assert H.cell_values('Mi1', v) == dict(onset_peak_mV=15.0, offset_peak_mV=-3.0)


# --- digitiser ----------------------------------------------------------------------------

def test_digitiser_recovers_a_synthetic_panel():
    from PIL import Image, ImageDraw
    W, Hh = 260, 260
    im = Image.new('RGB', (W, Hh), (255, 255, 255)); d = ImageDraw.Draw(im)
    panel = dict(types=('Mi1', 'Tm3'), x_axis=20.0, y_zero=200.0, y_half=81.0, half=0.5, x_range=(21, 250),
                 y_range=(10, 240), px_per_50ms=39.0)
    k = _biphasic(peak=60, amp=0.6)[:230]
    upx = (panel['y_zero'] - panel['y_half']) / panel['half']
    pts = [(panel['x_axis'] + t * 39 / 50, panel['y_zero'] - k[t] * upx) for t in range(len(k))]
    d.line([(x, y - 8) for x, y in pts], fill=G.BAND['Mi1'], width=1)
    for off in range(-8, 9):
        d.line([(x, y + off) for x, y in pts], fill=G.BAND['Mi1'], width=1)
    d.line(pts, fill=G.LINE['Mi1'], width=2)
    got = G.digitise_filter(np.asarray(im).astype(float), panel, 'Mi1', t_max_ms=200)
    n = min(len(got['mean']), 180)
    # within the documented uncertainty: 1 px in y (0.0042) and +/-2 ms in time
    for t in range(5, n):
        window = k[max(t - 2, 0):t + 3]
        assert window.min() - 0.0063 <= got['mean'][t] <= window.max() + 0.0063, t
    assert abs(G.peak_time_ms(got, 'Mi1') - 60) <= 5          # flat top: 1 px spans several ms
    assert np.median(got['sem'][20:n]) == pytest.approx(8 / upx, abs=0.01)


# --- prereg -------------------------------------------------------------------------------

def test_prereg_is_consistent_with_the_frozen_code_and_pins():
    pre = json.loads(PREREG.read_text())
    for rel, sha in pre['code_sha256'].items():
        assert _sha(ROOT / rel) == sha, rel
    assert _sha(ROOT / pre['base_params']['file']) == pre['base_params']['sha256']
    allowed_types = {'Mi1', 'Tm3', 'Tm1', 'Tm2', 'L1', 'L2'}
    for f in pre['free_parameters']['list']:
        assert f['type'] in allowed_types and f['key'] in ('tau_m_ms', 'e_leak_mV', 'release_gain')
        assert not f['type'].startswith(('T4', 'T5'))
    g = pre['acceptance']['training']
    assert g['nrmse_max'] == D.GATE_NRMSE and g['peak_slack_ms'] == D.GATE_PEAK_SLACK_MS
    assert g['order_min_ms'] == D.GATE_ORDER_MS
    h = pre['acceptance']['heldout']
    assert h['amp_factor'] == H.GATE_AMP_FACTOR and h['ratio_population_sd'] == H.GATE_RATIO_SD
    s3 = pre['downstream_s3']['unchanged_sha256']
    for rel, sha in s3.items():
        assert _sha(ROOT / rel) == sha, rel
    assert pre['stimulus']['noise_sigma'] == D.NOISE_SIGMA and pre['stimulus']['noise_tau_ms'] == D.NOISE_TAU_MS
    assert pre['stimulus']['frame_hz'] == D.FRAME_HZ and pre['stimulus']['duration_ms'] == D.NOISE_MS
    assert pre['observation_model']['objective_sigma_dig'] == D.SIGMA_DIG
    assert pre['observation_model']['objective_sigma_floor'] == D.SIGMA_FLOOR


class _FakeBrain:
    """Stub with the BrainV6 surface the harness uses: no LIF dynamics, just a known linear
    filter from the mean light encoder level to each recorded cell's V."""

    def __init__(self, ct, light, kernels):
        self.n = len(ct); self.light = light; self.k = kernels; self.ct = ct
        for a in ('tau_m', 'e_leak', 'rel_gain', 'rel_s'):
            setattr(self, a, np.zeros(self.n))
        self.reset_state()

    def reset_state(self):
        self.hist = []; self.v = np.full(self.n, -60.0)

    def step(self, drive, dt):
        c = float(np.mean(drive[self.light])) / 10.0 - 1.0          # back to contrast
        self.hist.append(c)
        h = np.array(self.hist[::-1][:D.N_LAGS])
        for t, kern in self.k.items():
            i = int(np.flatnonzero(self.ct == t)[0])
            self.v[i] = -60.0 + float(np.dot(kern[:len(h)], h))


def test_noise_model_wiring_recovers_the_stub_filters():
    ct = np.array(['R1-R6', 'R1-R6', 'Mi1', 'Tm3', 'Tm1', 'Tm2', 'L1'])
    light = np.array([0, 1])
    b = _FakeBrain(ct, light, FILTERS)
    cc = {t: np.flatnonzero(ct == t) for t in D.RECORDED}
    m = F.NoiseModel(b, ct, light, cc, seed=1)
    K = m.filters({'Mi1': {'tau_m_ms': 30.0}})
    assert b.tau_m[2] == 30.0
    data = _data_from(FILTERS)
    g = D.train_gates(K, data)
    assert g['TRAIN_PASS'], g
