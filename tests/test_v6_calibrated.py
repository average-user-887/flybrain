"""v6 is additive: with no per-type parameters it is v4, bit for bit."""
import numpy as np

from brainlab.brain import Brain
from brainlab.v6_calibrated import BrainV6, Phototransduction
from tests.test_graded_v4 import random_graph


def _run(b, drive, ms=60):
    out = []
    for _ in range(ms):
        b.step(drive, 1.0)
        out.append(b.v.copy())
    return np.array(out)


def test_empty_v6_is_bit_identical_to_v4():
    arrays = random_graph()
    n = len(arrays['ids'])
    rng = np.random.default_rng(5)
    mask = (rng.random(n) < 0.5).astype(np.uint8)
    drive = np.zeros(n, np.float32)
    drive[rng.choice(n, n // 10, replace=False)] = 25.0
    a = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=mask)
    b = BrainV6(None, {'types': {}}, 'x' * 64, cell_type=np.array(['X'] * n), arrays=dict(arrays),
                graded_policy=mask)
    assert np.array_equal(_run(a, drive), _run(b, drive))


def test_sigmoid_release_and_light_change_the_result():
    arrays = random_graph()
    n = len(arrays['ids'])
    mask = np.ones(n, np.uint8)
    ct = np.array(['R1-R6'] * 20 + ['X'] * (n - 20))
    p = {'phototransduction': dict(G=5.0, Ka=0.5, tau_a_ms=200, tau_p0_ms=8, n_stages=4, dead_time_ms=10,
                                   encoder_to_intensity=0.01),
         'types': {'R1-R6': {'release': {'kind': 'sigmoid', 'gain': 2.0, 'vh_mV': -40.0, 's_mV': 4.0}}}}
    b = BrainV6(None, p, 'y' * 64, light_nodes=np.arange(20), cell_type=ct, arrays=dict(arrays), graded_policy=mask)
    drive = np.zeros(n, np.float32); drive[:20] = 100.0
    v = _run(b, drive, 80)
    assert v[-1, :20].mean() > -40.0          # light depolarises R
    s = b.snapshot_state(); x = _run(b, drive, 5); b.restore_state(s); y = _run(b, drive, 5)
    assert np.array_equal(x, y)


def test_phototransduction_adapts():
    pt = Phototransduction(dict(G=5.0, Ka=0.5, tau_a_ms=200, tau_p0_ms=8, n_stages=4, dead_time_ms=10), 1)
    g = [pt.step(np.array([1.0]))[0] for _ in range(2000)]
    assert max(g) > 1.5 * g[-1] > 0


# --- S3 repair: complete, dynamics-label-independent snapshot/restore/reset --------------

def _brain_s3(arrays, n):
    ct = np.array(['R1-R6'] * 20 + ['L1'] * 20 + ['X'] * (n - 40))
    p = {'phototransduction': dict(G=68.8, Ka=0.0072, Kt=1.0, tau_a_ms=200, tau_p0_ms=4.3, n_stages=4,
                                   dead_time_ms=10, encoder_to_intensity=0.01),
         'types': {'R1-R6': {'release': {'kind': 'sigmoid', 'gain': 17.8, 'vh_mV': -47.0, 's_mV': 1.0,
                                         'kappa': 1.0, 'tau_s_ms': 200.0}},
                   'L1': {'tau_m_ms': 50.0, 'e_leak_mV': -40.0,
                          'ih': {'g': 1.2, 'vh_mV': -60.0, 'k_mV': 5.0, 'tau_ms': 300.0, 'E_mV': -30.0}}}}
    b = BrainV6(None, p, 'z' * 64, light_nodes=np.arange(20), cell_type=ct, arrays=dict(arrays),
                graded_policy=np.ones(n, np.uint8))
    b.dynamics = 'v6'            # exactly what scripts/v6/measure_tuning_v6.py does
    return b


def _cond(b, level, ms=40):
    d = np.zeros(b.n, np.float32); d[:20] = level
    return _run(b, d, ms)


def _mutable(b):
    out = {k: getattr(b, k).copy() for k in ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts', 'active',
                                             'active_flag', 'nactive', 'rel_ring', 'h_state', 'g_light', 'rel_m')}
    out.update(b.pt.state()); out['cursor'] = b.cursor
    return out


def test_v6_snapshot_contains_rel_ring_even_when_relabelled():
    arrays = random_graph(); n = len(arrays['ids'])
    b = _brain_s3(arrays, n)
    _cond(b, 10.0)
    s = b.snapshot_state()
    assert 'rel_ring' in s and s['rel_ring'].any()      # failed before the repair: key absent


def test_v6_restore_is_bit_identical_and_condition_order_independent():
    arrays = random_graph(); n = len(arrays['ids'])
    b = _brain_s3(arrays, n)
    _cond(b, 10.0, 60)                                   # shared gray baseline
    snap = b.snapshot_state(); ref = _mutable(b)
    direct = _cond(b, 20.0)                              # B straight from the snapshot
    b.restore_state(snap)
    _cond(b, 0.0)                                        # A ...
    b.restore_state(snap)                                # ... then restore
    for k, v in _mutable(b).items():
        assert np.array_equal(np.asarray(v), np.asarray(ref[k])), k
    assert np.array_equal(direct, _cond(b, 20.0))        # B after A == B directly


def test_v6_reset_clears_every_mutable_array():
    arrays = random_graph(); n = len(arrays['ids'])
    b = _brain_s3(arrays, n)
    _cond(b, 20.0)
    b.reset_state()
    fresh = _brain_s3(arrays, n)
    for k, v in _mutable(b).items():
        assert np.array_equal(np.asarray(v), np.asarray(_mutable(fresh)[k])), k


def test_v6_incomplete_or_foreign_snapshot_is_refused():
    import pytest
    arrays = random_graph(); n = len(arrays['ids'])
    b = _brain_s3(arrays, n)
    _cond(b, 10.0)
    s = b.snapshot_state()
    for k in ('rel_ring', 'pt_ring', 'v6_params_sha256'):
        bad = dict(s); bad.pop(k)
        with pytest.raises(ValueError):
            b.restore_state(bad)
    bad = dict(s); bad['v6_params_sha256'] = 'q' * 64
    with pytest.raises(ValueError):
        b.restore_state(bad)


def test_v6_two_conditions_either_order_bit_identical():
    arrays = random_graph(); n = len(arrays['ids'])
    res = {}
    for order in (('A', 'B'), ('B', 'A')):
        b = _brain_s3(arrays, n)
        _cond(b, 10.0, 60)
        snap = b.snapshot_state()
        for c in order:
            b.restore_state(snap)
            res[(order, c)] = _cond(b, 0.0 if c == 'A' else 20.0)
    assert np.array_equal(res[(('A', 'B'), 'A')], res[(('B', 'A'), 'A')])
    assert np.array_equal(res[(('A', 'B'), 'B')], res[(('B', 'A'), 'B')])
