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

PREREG = ROOT / 'qualification' / 'v8' / 'V8_input_prereg_v2.json'
PREREG_V1 = ROOT / 'qualification' / 'v8' / 'V8_input_prereg.json'
import synapse_ceiling_v8 as S  # noqa: E402
BASE = ROOT / 'qualification' / 'v6' / 'params_v6_S2A_leak40.json'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _meas(r_tp=25.5, l_tp=12.2, lat=6.5, l_lat=6.5, k4=40.0, k0=20.0, kpk=0.5, sd=1.0):
    m = {'flash': {'R': dict(onset_ms=lat, tp_ms=r_tp, peak_mV=30.0)}}
    for t in V.LMC_TYPES:
        m['flash'][t] = dict(onset_ms=l_lat, tp_ms=l_tp, peak_mV=-30.0)
    m['BG-4'] = {'R': dict(kernel_tp_ms=k4, kernel_peak=kpk, response_sd_mV=sd)}
    m['BG0'] = {'R': dict(kernel_tp_ms=k0, kernel_peak=kpk, response_sd_mV=sd)}
    return m


def test_onset_and_time_to_peak():
    x = np.zeros(100); x[10:30] = np.linspace(0, -20, 20); x[30:] = -20 * np.exp(-np.arange(70) / 10)
    on, tp, pk = V.onset_tp(x)
    assert tp in (30, 31) and pk == pytest.approx(-20) and 11 <= on <= 14     # +1 ms offset (v2)


def test_objective_and_gates_at_the_targets():
    m = _meas()
    assert V.objective(m) == pytest.approx(0.0, abs=1e-9) and V.train_gates(m, 9)['TRAIN_PASS']


@pytest.mark.parametrize('kw,dead,gate', [
    (dict(l_tp=30.0), 9, 'G2_flash_LMC_tp'),          # LMC slower than R: the v7 diagnostic's failure
    (dict(l_lat=14.0), 9, 'G3_latency'),              # synaptic delay
    (dict(k0=45.0), 9, 'G4_kernel_tp'),               # no light adaptation of the kernel
    (dict(), 4, 'G5_dead_time'),                      # pure delay far from JH2001's BG0 D
    (dict(r_tp=40.0), 9, 'G1_flash_R_tp'),
])
def test_each_gate_can_fail(kw, dead, gate):
    g = V.train_gates(_meas(**kw), dead)
    assert not g['TRAIN_PASS']
    v = g[gate]
    assert (v is False) or (isinstance(v, dict) and not all(v.values()))


def test_nan_measurement_is_penalised_not_crashing():
    m = _meas(); m['flash']['L1']['onset_ms'] = math.nan
    assert V.objective(m) >= 1e6 and not V.train_gates(m, 9)['TRAIN_PASS']


def _r(run, cost, ok, r_ok=True, status='completed'):
    return dict(run=run, cost=cost, status=status, gates=dict(TRAIN_PASS=ok, R_STAGE_PASS=r_ok))


def test_outcome_rules_split_f1_by_stage():
    """fail-old/pass-new (audit 1 item 5b, audit 2 D3): v1 returned one 'F1' blaming R->LMC."""
    reps = [_r(0, 1.0, False), _r(1, 3.0, True), _r(2, 2.0, True), _r(3, 0.5, False)]
    assert V.outcome(reps, 4) == ('TRAIN_PASS', reps[2])
    assert V.outcome([_r(0, 1, False, r_ok=False), _r(1, 1, False, r_ok=False)], 4) == ('F1_R', None)
    assert V.outcome([_r(0, 1, False, r_ok=True), _r(1, 1, False, r_ok=False)], 4) == ('F1_LMC', None)
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
    assert {r[1] for r in r1} == {7, 8, 9, 10}
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
    for rel, s_ in pre['code_sha256'].items():
        assert _sha(ROOT / rel) == s_, rel
    assert _sha(ROOT / pre['base_params']['file']) == pre['base_params']['sha256']
    assert _sha(ROOT / pre['data_pin']['file']) == pre['data_pin']['sha256']
    assert _sha(PREREG_V1) == pre['supersedes']['sha256'] == '6fa584331422598ba694424406cf2fd9ee2499c1f96c8d80588d76ec87471bee'
    for f in pre['free_parameters']['continuous']:
        assert f['type'] in ('R1-R6', 'L1', 'L2')
    t = pre['targets']
    for k in ('mean', 'n', 'lo', 'hi', 'sigma'):
        assert t['flash_R'][k] == V.T_FLASH_R[k] and t['flash_LMC'][k] == V.T_FLASH_LMC[k]
    assert t['window_pad_ms'] == V.WINDOW_PAD_MS and tuple(t['flash_latency']) == V.T_FLASH_LATENCY
    assert t['kernel_tp'] == V.T_KERNEL_TP and t['dead_time_bg0'] == V.T_DEAD_BG0 and t['sigma_tp_frac'] == V.SIG_TP_FRAC
    g = pre['acceptance']['training']
    assert tuple(g['latency_window_ms']) == V.G_LATENCY and g['latency_diff_ms'] == V.G_LATENCY_DIFF
    assert g['tp_frac'] == V.G_TP_FRAC and g['dead_frac'] == V.G_DEAD_FRAC
    h = pre['acceptance']['heldout']
    assert h['peak_slack_ms'] == H.GATE_PEAK_SLACK_MS and h['order_ms'] == H.GATE_ORDER_MS and h['min_peaks_ok'] == H.MIN_PEAKS_OK
    p = pre['protocols']
    assert p['flash_level'] == V.FLASH_LEVEL and p['dark_adapt_ms'] == V.DARK_ADAPT_MS and p['background'] == V.BACKGROUND
    assert p['noise'] == dict(ms=V.NOISE_MS, sd=V.NOISE_SD, cutoff_hz=V.NOISE_CUTOFF_HZ, seed=V.NOISE_SEED)
    d = p['bg_decidability']
    assert d['kernel_peak_min'] == V.K_MIN_MV_PER_CONTRAST_MS and d['response_sd_min_mV'] == V.SD_MIN_MV
    assert len(pre['changes_from_v1']) >= 8


# --- v2 fail-old / pass-new ----------------------------------------------------------------

def test_lmc_dispersion_derivation_and_window_includes_the_15ms_cell():
    """fix (a): v1's window 12.2 +- 2 x 1.1 = [10.0, 14.4] excluded Mansour's recorded 15.0 ms LMC."""
    lo, hi = V.feasible_sd(5, 12.2, 11.0, 15.0, step=0.05)
    assert 1.55 <= lo <= 1.62 and 1.75 <= hi <= 1.80          # +-1.1 is not an s.d. ...
    assert not (lo / math.sqrt(5) <= 1.1 <= hi / math.sqrt(5))  # ... nor an s.e.m.
    rlo, rhi = V.feasible_sd(4, 25.5, 24.0, 29.0, step=0.05)
    assert rlo <= 2.4 + 0.05 and rhi >= 2.4 - 0.05            # R +-2.4 is consistent with an s.d.
    old = (12.2 - 2 * 1.1, 12.2 + 2 * 1.1)
    assert not old[0] <= 15.0 <= old[1]                        # fail-old
    w = V.window(V.T_FLASH_LMC)
    assert w == (10.0, 16.0) and all(w[0] <= x <= w[1] for x in (11.0, 12.2, 15.0))   # pass-new
    assert V.train_gates(_meas(l_tp=15.0), 9)['G2_flash_LMC_tp'] == {'L1': True, 'L2': True}


class _StepStub:
    """BrainV6 surface for measure(): R follows the light with a known delay (end of step 4 = 5 ms)."""

    def __init__(self):
        self.n = 3; self.v = np.zeros(3); self.h = []

    def reset_state(self):
        self.v = np.zeros(3); self.h = []

    def step(self, drive, dt):
        self.h.append(float(drive[0]))
        x = self.h[-5] if len(self.h) >= 5 else 0.0
        self.v = np.array([x, -x, -x]) / 100.0


def test_times_carry_the_end_of_step_offset():
    """fix (d), audit 1 item 5c: v1 reported every time 1 ms early."""
    b = _StepStub()
    m = V.measure(b, np.array([0]), {'R': 0, 'L1': 1, 'L2': 2})
    assert m['flash']['R']['onset_ms'] == 5.0                  # physical time; v1 gave 4.0
    assert m['flash']['R']['tp_ms'] == 5.0


def test_bg_decidability_rule():
    """fix (c), audit 2 D2."""
    m = _meas(kpk=1e-6)
    assert not V.bg_evaluable(m, 'BG-4')
    assert V.objective(m) >= 1e6 and not V.train_gates(m, 9)['G4_kernel_tp']['BG-4']
    m2 = _meas(); m2['BG-4']['R']['kernel_peak'] = 1e-6
    g = V.train_gates(m2, 9, backgrounds=('BG0',))
    assert 'BG-4' not in g['G4_kernel_tp'] and g['R_STAGE_PASS']   # dropped background is not scored


def test_g3_g5_reconcile_only_for_d_7_to_10():
    for d in (7, 8, 9, 10):
        assert V.train_gates(_meas(lat=float(d), l_lat=float(d)), d)['TRAIN_PASS']
    assert V.train_gates(_meas(), 12)['G5_dead_time'] and V.train_gates(_meas(), 6)['G5_dead_time'] is False
    assert not V.train_gates(_meas(lat=12.0, l_lat=12.0), 12)['TRAIN_PASS']    # D 12 forces onset > 10


def test_jh2001_kernel_is_estimated_from_noise():
    """fix (b): the G4 quantity is a noise kernel; a known kernel is recovered."""
    s = V.jh_noise()
    assert abs(s.std() - 0.32) < 0.01 and s.min() >= -1 and s.max() <= 1
    t = np.arange(150.0); k = (t / 20) ** 3 * np.exp(3 * (1 - t / 20)) * 0.4
    v = np.convolve(s, k)[:len(s)]
    est = V.kernel(s, v)
    assert abs(int(np.argmax(est)) - 20) <= 1 and np.sqrt(((est - k) ** 2).sum() / (k ** 2).sum()) < 0.1


def test_graded_cell_parity_between_fit_and_frozen_file(tmp_path):
    """fix (f), audit 1 CANNOT-TELL: v1 wrote L1/L2 values to every cell of the type, BrainV6 loads
    only graded cells; with a mixed graded mask the two differed."""
    from brainlab.v6_calibrated import BrainV6, load_params
    from tests.test_graded_v4 import random_graph
    arrays = random_graph(); n = len(arrays['ids'])
    ct = np.array(['R1-R6'] * 20 + ['L1'] * 20 + ['L2'] * 20 + ['X'] * (n - 60))
    mask = np.ones(n, np.uint8); mask[[25, 26, 45]] = 0                      # some non-graded L1/L2
    base = json.loads(BASE.read_text())
    p0 = dict(phototransduction=base['phototransduction'], types={k: base['types'][k] for k in ('L1', 'L2')})
    fit_b = BrainV6(None, p0, 'a' * 64, light_nodes=np.arange(20), cell_type=ct, arrays=dict(arrays), graded_policy=mask)
    idx = {t: np.flatnonzero(ct == t) for t in set(ct.tolist())}
    fitted = {'L1': {'tau_m_ms': 9.0, 'ih_g': 0.4}, 'L2': {'tau_m_ms': 11.0}}
    V.apply_input_params(fit_b, idx, np.arange(20), base['phototransduction'], fitted)
    pf = F.params_file(dict(base, types=dict(p0['types'])), 'b' * 64, 'p' * 64, fitted, 9, 't')
    pf['phototransduction'] = dict(pf['phototransduction'], dead_time_ms=base['phototransduction']['dead_time_ms'])
    f = tmp_path / 'p.json'; f.write_text(json.dumps(pf))
    params, sha = load_params(f, _sha(f))
    file_b = BrainV6(None, params, sha, light_nodes=np.arange(20), cell_type=ct, arrays=dict(arrays), graded_policy=mask)
    for k in ('tau_m', 'gh'):
        assert np.array_equal(getattr(fit_b, k), getattr(file_b, k)), k
    old = fit_b.tau_m.copy(); old[idx['L1']] = 9.0                             # v1 behaviour
    assert not np.array_equal(old, file_b.tau_m)


def test_ceiling_reading_checks_sign_and_uses_a_derived_threshold():
    """fix (e): v1 read 'structural' for any ceiling below 15 mV, including a NEGATIVE ceiling."""
    assert S.reading('Mi1', -3.0).startswith('NOT APPLICABLE')               # v1: 'structural'
    assert S.threshold('Mi1') == pytest.approx(30.2 - 2 * 0.92)
    assert S.reading('Mi1', 20.0).startswith('BELOW') and 'suggestive' in S.reading('Mi1', 20.0)
    assert S.reading('Mi1', 29.0).startswith('REACHABLE')
    assert 'structural' not in (V8 / 'synapse_ceiling_v8.py').read_text().split('def reading')[1]
