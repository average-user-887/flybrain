"""v8 input timing: logic, harness wiring and prereg consistency.  No fit, no cutout run."""
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / 'scripts' / 'v8'
for p in (V8, ROOT / 'scripts' / 'v7', ROOT / 'scripts' / 'v6'):
    sys.path.insert(0, str(p))
import v8_input as V  # noqa: E402
import fit_v8 as F  # noqa: E402
import heldout_v8 as H  # noqa: E402

PREREG = ROOT / 'qualification' / 'v8' / 'V8_input_prereg.json'
BASE = ROOT / 'qualification' / 'v6' / 'params_v6_S2A_leak40.json'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _meas(r_tp=25.5, l_tp=12.2, lat=6.5, l_lat=6.5, tp4=40.0, tp0=20.0, d4=20.0, d0=10.0):
    m = {'flash': {'R': dict(onset_ms=lat, tp_ms=r_tp, peak_mV=30.0)}}
    for t in V.LMC_TYPES:
        m['flash'][t] = dict(onset_ms=l_lat, tp_ms=l_tp, peak_mV=-30.0)
    m['BG-4'] = {'R': dict(onset_ms=d4, tp_ms=tp4, peak_mV=5.0)}
    m['BG0'] = {'R': dict(onset_ms=d0, tp_ms=tp0, peak_mV=5.0)}
    return m


def test_onset_and_time_to_peak():
    x = np.zeros(100); x[10:30] = np.linspace(0, -20, 20); x[30:] = -20 * np.exp(-np.arange(70) / 10)
    on, tp, pk = V.onset_tp(x)
    assert tp in (29, 30) and pk == pytest.approx(-20) and 10 <= on <= 13


def test_objective_and_gates_at_the_targets():
    m = _meas()
    assert V.objective(m) == 0.0 and V.train_gates(m)['TRAIN_PASS']


@pytest.mark.parametrize('kw,gate', [
    (dict(l_tp=30.0), 'G2_flash_LMC_tp'),            # LMC slower than R: the v7 diagnostic's failure
    (dict(l_lat=14.0), 'G3_latency'),                # synaptic delay
    (dict(tp0=45.0), 'G4_impulse_tp'),               # no light adaptation of time to peak
    (dict(d4=40.0), 'G5_dead_time'),
    (dict(r_tp=40.0), 'G1_flash_R_tp'),
])
def test_each_gate_can_fail(kw, gate):
    g = V.train_gates(_meas(**kw))
    assert not g['TRAIN_PASS']
    v = g[gate]
    assert (v is False) or (isinstance(v, dict) and not all(v.values()))


def test_nan_measurement_is_penalised_not_crashing():
    m = _meas(); m['flash']['L1']['onset_ms'] = math.nan
    assert V.objective(m) >= 1e6 and not V.train_gates(m)['TRAIN_PASS']


def _r(run, cost, ok, status='completed'):
    return dict(run=run, cost=cost, status=status, gates=dict(TRAIN_PASS=ok))


def test_outcome_rules():
    reps = [_r(0, 1.0, False), _r(1, 3.0, True), _r(2, 2.0, True), _r(3, 0.5, False)]
    assert V.outcome(reps, 4) == ('TRAIN_PASS', reps[2])
    assert V.outcome([_r(0, 1, False), _r(1, 1, False)], 4) == ('F1', None)
    assert V.outcome([_r(0, 1, True)] + [dict(run=k, status='capped') for k in range(1, 4)], 4)[0] == 'NOT_EVALUABLE'


def _tiny():
    from brainlab.v6_calibrated import BrainV6
    from tests.test_graded_v4 import random_graph
    arrays = random_graph(); n = len(arrays['ids'])
    ct = np.array(['R1-R6'] * 20 + ['L1'] * 20 + ['L2'] * 20 + ['Mi1'] * 20 + ['X'] * (n - 80))
    base = json.loads(BASE.read_text())
    p = dict(phototransduction=base['phototransduction'], types={k: base['types'][k] for k in ('L1', 'L2', 'Mi1')})
    b = BrainV6(None, p, 'q' * 64, light_nodes=np.arange(20), cell_type=ct, arrays=dict(arrays),
                graded_policy=np.ones(n, np.uint8))
    return b, ct, base


def test_apply_input_params_touches_only_inputs():
    b, ct, base = _tiny()
    idx = {t: np.flatnonzero(ct == t) for t in set(ct.tolist())}
    before = {k: getattr(b, k).copy() for k in ('tau_m', 'gh', 'h_tau', 'h_vh', 'rel_gain', 'rel_vh', 'rel_s', 'e_leak', 'weight')}
    V.apply_input_params(b, idx, np.arange(20), base['phototransduction'],
                         {'R1-R6': {'tau_p0_ms': 2.0, 'dead_time_ms': 6.4}, 'L1': {'tau_m_ms': 9.0, 'ih_g': 0.5}})
    assert b.pt.tau_p0 == 2.0 and b.pt.D == 6 and b.v6_params['phototransduction']['dead_time_ms'] == 6
    assert np.all(b.tau_m[idx['L1']] == 9.0) and np.all(b.gh[idx['L1']] == 0.5)
    others = np.setdiff1d(np.arange(b.n), idx['L1'])
    assert np.array_equal(b.tau_m[others], before['tau_m'][others])
    for k in ('rel_gain', 'rel_vh', 'rel_s', 'e_leak', 'weight'):
        assert np.array_equal(getattr(b, k), before[k]), k            # synapses and leak untouched
    b.reset_state()                                                     # pt state shape follows the new dead time
    with pytest.raises(ValueError):
        V.apply_input_params(b, idx, np.arange(20), base['phototransduction'], {'Mi1': {'tau_m_ms': 5.0}})
    with pytest.raises(ValueError):
        V.apply_input_params(b, idx, np.arange(20), base['phototransduction'], {'R1-R6': {'G': 5.0}})


def test_params_file_changes_only_declared_keys_and_loads(tmp_path):
    from brainlab.v6_calibrated import BrainV6, load_params
    base = json.loads(BASE.read_text())
    fitted = {'R1-R6': {'tau_p0_ms': 3.0, 'Kt': 0.5}, 'L2': {'tau_m_ms': 7.0, 'ih_tau_ms': 50.0}}
    p = F.params_file(base, 'b' * 64, 'p' * 64, fitted, 7, 'test')
    assert p['phototransduction']['dead_time_ms'] == 7 and p['phototransduction']['G'] == base['phototransduction']['G']
    assert p['types']['L2']['ih']['tau_ms'] == 50.0 and p['types']['L2']['ih']['g'] == base['types']['L2']['ih']['g']
    for t in base['types']:
        if t not in ('L2',):
            assert p['types'][t] == base['types'][t], t
    f = tmp_path / 'p.json'; f.write_text(json.dumps(p))
    load_params(f, _sha(f))


def test_runs_grid_and_starts_are_fixed():
    pre = json.loads(PREREG.read_text())
    base = json.loads(BASE.read_text())
    spec = F.free_spec(pre)
    r1, r2 = F.runs(pre, base, spec), F.runs(pre, base, spec)
    assert len(r1) == pre['optimiser']['n_runs'] == len(pre['free_parameters']['dead_time_grid_ms']) * pre['optimiser']['starts_per_dead_time']
    assert all(a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2]) for a, b in zip(r1, r2))
    x0 = F.from_x(r1[0][2], spec)
    for (t, k, *_), v in zip(spec, F.base_values(base, spec)):
        assert math.isclose(x0[t][k], v, rel_tol=1e-12)


def test_heldout_gates():
    d = {'Mi1': 70.0, 'Tm3': 56.0, 'Tm1': 54.0, 'Tm2': 44.0}
    sem = {'Mi1': 3.8, 'Tm3': 5.2, 'Tm1': 3.8, 'Tm2': 2.7}
    assert H.heldout_gates(d, d, sem)['HELDOUT_PASS']
    same_off = dict(d, Tm1=50.0, Tm2=50.0)
    assert not H.heldout_gates(same_off, d, sem)['HELDOUT_PASS']
    slow = {c: v + 40 for c, v in d.items()}
    assert not H.heldout_gates(slow, d, sem)['HELDOUT_PASS']


def test_fit_never_reads_the_heldout():
    for mod in ('v8_input.py', 'fit_v8.py'):
        src = (V8 / mod).read_text()
        tree = ast.parse(src)
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        assert not any('heldout' in x or x in ('v7_data', 'fit_v7') for x in names), mod
        assert 'load_training' not in src and 'behnia' not in src.lower()


def test_prereg_consistency():
    pre = json.loads(PREREG.read_text())
    for rel, s in pre['code_sha256'].items():
        assert _sha(ROOT / rel) == s, rel
    assert _sha(ROOT / pre['base_params']['file']) == pre['base_params']['sha256']
    assert _sha(ROOT / pre['data_pin']['file']) == pre['data_pin']['sha256']
    for f in pre['free_parameters']['continuous']:
        assert f['type'] in ('R1-R6', 'L1', 'L2')
    t = pre['targets']
    assert tuple(t['flash_R_tp']) == V.T_FLASH_R_TP and tuple(t['flash_LMC_tp']) == V.T_FLASH_LMC_TP
    assert tuple(t['flash_latency']) == V.T_FLASH_LATENCY
    assert t['impulse_tp'] == V.T_IMPULSE_TP and t['dead_time'] == V.T_DEAD
    g = pre['acceptance']['training']
    assert g['flash_sd'] == V.G_FLASH_SD and tuple(g['latency_window_ms']) == V.G_LATENCY
    assert g['latency_diff_ms'] == V.G_LATENCY_DIFF and g['tp_frac'] == V.G_TP_FRAC and g['dead_frac'] == V.G_DEAD_FRAC
    h = pre['acceptance']['heldout']
    assert h['peak_slack_ms'] == H.GATE_PEAK_SLACK_MS and h['order_ms'] == H.GATE_ORDER_MS and h['min_peaks_ok'] == H.MIN_PEAKS_OK
    p = pre['protocols']
    assert p['flash_level'] == V.FLASH_LEVEL and p['dark_adapt_ms'] == V.DARK_ADAPT_MS and p['background'] == V.BACKGROUND
