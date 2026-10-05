"""LIF dynamics v6a: v5 plus a declared per-class graded transfer gain (spec §10).

Holds the locked declaration to account: v1-v5 untouched and their pins
intact, v6a with the v5-linear release table bit-identical to v5 (CPU and GPU),
the release function as declared (value kept at the anchor, slope multiplied by
the class gain, v4's bounds), the derived gain reproduced by a two-cell probe,
and checkpoints that refuse a different release table.
"""
import hashlib
import pathlib

import numpy as np
import pytest

from brainlab import graded_release as gr
from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain
from brainlab.engine import E_EXC_MV, E_INH_MV, R_MAX_HZ
from brainlab.graph_identity import DYNAMICS_VERSIONS, LIF_DYNAMICS_V6A, dynamics_pin

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts', 'active',
         'active_flag', 'nactive', 'rel_ring')
PINS_BEFORE_V6A = {
    'v1': '1b17bfc6cf8c4f54dbda73d715fec53971c369d0495a60d938cd56fa8320f146',
    'v2': '1ea33ba1d05d815cf8f83c0e3a783d3ff8dec101c44703eaad4c667f8cfbeb56',
    'v3': '5739c4f4ffb84979d90ad430d2405bfc0b7736c7c75f2a6e676ef606dffb4a41',
    'v4': '7ae0966578aa3c4a948021b8c38ce8c4de89aad8d67ac0f6d5c9c094b021ed44',
    'v5': '08ff5652c3916d68d85d22ef2e8cd164a9db86f5021100575a27e2c45d5310f4',
}


def random_graph(n=400, k=10, seed=3):
    rng = np.random.default_rng(seed)
    post = rng.integers(0, n, size=n * k).astype(np.int32)
    ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    cls = rng.choice(rk.CLASSES, size=n, p=[0.6, 0.15, 0.15, 0.1])
    sign = np.where(cls == 'nicotinic', 1.0, -1.0)
    weight = (np.repeat(sign, k) * rng.uniform(0.5, 6.0, n * k)).astype(np.float32)
    rcls = np.where(rng.random(n) < 0.2, gr.CLASS_PHOTORECEPTOR, gr.CLASS_DEFAULT)
    return dict(ptr=ptr, post=post, weight=weight, ids=np.arange(1, n + 1, dtype=np.int64)), cls, rcls


def chain(n, edges):
    ptr = np.zeros(n + 1, dtype=np.int64)
    post, w = [], []
    for i in range(n):
        for (a, b), wt in edges:
            if a == i:
                post.append(b)
                w.append(wt)
        ptr[i + 1] = len(post)
    return dict(ptr=ptr, post=np.array(post, dtype=np.int32), weight=np.array(w, dtype=np.float32),
                ids=np.arange(1, n + 1, dtype=np.int64))


# --- declaration and pins ---------------------------------------------------

def test_v1_to_v5_pins_are_unchanged_and_v6a_has_its_own():
    for v, pin in PINS_BEFORE_V6A.items():
        assert dynamics_pin(v) == pin, v
    assert 'v6a' in DYNAMICS_VERSIONS
    assert dynamics_pin('v6a') not in PINS_BEFORE_V6A.values()
    assert LIF_DYNAMICS_V6A['dynamics_version'] == 'v6a'


def test_declaration_lock_matches_the_hash_the_code_cites():
    locked = ROOT / LIF_DYNAMICS_V6A['declaration_lock']
    assert hashlib.sha256(locked.read_bytes()).hexdigest() == LIF_DYNAMICS_V6A['declaration_sha256']


def test_default_dynamics_is_still_v3(monkeypatch):
    monkeypatch.delenv('NEUROFLY_LIF_DYNAMICS', raising=False)
    from brainlab.graph_identity import active_dynamics_version
    assert active_dynamics_version() == 'v3'


def test_v5_linear_table_is_gain_one_everywhere():
    assert {c['gain'] for c in gr.RELEASE[gr.RELEASE_V5_LINEAR].values()} == {1.0}


def test_primary_gain_is_the_declared_derivation():
    d = gr.derive_gain(gr.MEASURED_GAIN_PHOTORECEPTOR, gr.W_REF_PHOTORECEPTOR,
                       gr.ANCHOR_PHOTORECEPTOR_MV)
    assert gr.RELEASE[gr.RELEASE_PRIMARY][gr.CLASS_PHOTORECEPTOR]['gain'] == d['G']
    assert d['G'] > 1.0


def test_classes_come_from_the_presynaptic_cell_type():
    got = gr.classes_from_cell_types(['R1-R6', 'L1', 'R7d', 'Mi1', ''])
    assert list(got) == [gr.CLASS_PHOTORECEPTOR] + [gr.CLASS_DEFAULT] * 4


# --- the release function ----------------------------------------------------

def test_release_keeps_v4_value_at_the_anchor_and_multiplies_the_slope():
    dt = 0.1
    for c in gr.CLASSES:
        spec = gr.RELEASE[gr.RELEASE_PRIMARY][c]
        res = gr.resolve(1, np.ones(1, np.uint8), release_classes=[c], dt=dt)
        a = spec['anchor_mV']
        r_v4 = R_MAX_HZ * (a - E_INH_MV) / (E_EXC_MV - E_INH_MV)
        r_a = gr.release_hz(np.array([a]), res['rel_v0'], res['rel_k'])[0]
        assert r_a == pytest.approx(r_v4, rel=1e-9)
        h = 1e-3
        slope = (gr.release_hz(np.array([a + h]), res['rel_v0'], res['rel_k'])[0]
                 - gr.release_hz(np.array([a - h]), res['rel_v0'], res['rel_k'])[0]) / (2 * h)
        assert slope == pytest.approx(spec['gain'] * R_MAX_HZ / (E_EXC_MV - E_INH_MV), rel=1e-6)
        # v4's bounds
        assert gr.release_hz(np.array([E_INH_MV]), res['rel_v0'], res['rel_k'])[0] >= 0.0
        assert gr.release_hz(np.array([E_EXC_MV]), res['rel_v0'], res['rel_k'])[0] == pytest.approx(R_MAX_HZ)


def test_v5_linear_release_equals_v4_line_for_every_class():
    v = np.linspace(-70, 0, 71)
    for c in gr.CLASSES:
        res = gr.resolve(len(v), np.ones(len(v), np.uint8), release=gr.RELEASE_V5_LINEAR,
                         release_classes=[c] * len(v))
        assert np.all(res['rel_v0'] == E_INH_MV)
        r = gr.release_hz(v, res['rel_v0'], res['rel_k'])
        assert np.allclose(r, R_MAX_HZ * (v - E_INH_MV) / (E_EXC_MV - E_INH_MV), rtol=1e-12)


# --- v6a with the v5 release is v5 --------------------------------------------

def _run(version, arrays, cls, rcls, mask, backend, **kw):
    n = len(arrays['ids'])
    extra = dict(graded_policy=mask, receptor_classes=cls)
    if version == 'v6a':
        extra.update(release_classes=rcls)
    b = Brain(arrays=dict(arrays), dynamics=version, backend=backend, **extra, **kw)
    rng = np.random.default_rng(11)
    drive = np.zeros(n, np.float32)
    drive[rng.choice(n, n // 10, replace=False)] = 25.0
    total = np.zeros(n, np.int64)
    for d in (drive, np.zeros(n, np.float32), -drive * 0.3):
        for _ in range(10):
            c, _ = b.step(d, 5.0)
            total += c
    b._sync_from_gpu()
    return {k: getattr(b, k).copy() for k in STATE}, total


@pytest.mark.parametrize('graded_frac', [0.0, 0.5, 1.0])
@pytest.mark.parametrize('kinetics', [rk.KINETICS_PRIMARY, rk.KINETICS_V4_EQUIVALENT])
def test_v6a_with_v5_release_is_bit_identical_to_v5_on_cpu(graded_frac, kinetics):
    arrays, cls, rcls = random_graph()
    n = len(arrays['ids'])
    mask = (np.random.default_rng(5).random(n) < graded_frac).astype(np.uint8)
    s5, t5 = _run('v5', arrays, cls, rcls, mask, 'cpu', kinetics=kinetics)
    s6, t6 = _run('v6a', arrays, cls, rcls, mask, 'cpu', kinetics=kinetics,
                  release=gr.RELEASE_V5_LINEAR)
    for k in STATE:
        assert np.array_equal(s5[k], s6[k]), k
    assert np.array_equal(t5, t6)


@pytest.mark.parametrize('kinetics', [rk.KINETICS_PRIMARY, rk.KINETICS_V4_EQUIVALENT])
def test_v6a_with_v5_release_is_bit_identical_to_v5_on_gpu(kinetics):
    from brainlab.cupy_v4 import cupy_available
    if not cupy_available():
        pytest.skip('no CuPy GPU on this host')
    arrays, cls, rcls = random_graph(n=600, k=12, seed=7)
    n = len(arrays['ids'])
    mask = (np.random.default_rng(7).random(n) < 0.5).astype(np.uint8)
    s5, t5 = _run('v5', arrays, cls, rcls, mask, 'cuda', kinetics=kinetics)
    s6, t6 = _run('v6a', arrays, cls, rcls, mask, 'cuda', kinetics=kinetics,
                  release=gr.RELEASE_V5_LINEAR)
    for k in ('v', 'g', 'refractory', 'rel_ring'):
        assert np.array_equal(s5[k], s6[k]), k
    assert np.array_equal(t5, t6) and t5.sum() > 0


def test_v6a_calibrated_differs_from_v5():
    arrays, cls, rcls = random_graph()
    n = len(arrays['ids'])
    mask = (np.random.default_rng(5).random(n) < 0.5).astype(np.uint8)
    s5, _ = _run('v5', arrays, cls, rcls, mask, 'cpu')
    s6, _ = _run('v6a', arrays, cls, rcls, mask, 'cpu')
    assert not np.array_equal(s5['v'], s6['v'])


def test_cupy_v6a_matches_the_cpu_reference_with_calibrated_release():
    from brainlab.cupy_v4 import cupy_available
    if not cupy_available():
        pytest.skip('no CuPy GPU on this host')
    arrays, cls, rcls = random_graph(n=600, k=12, seed=9)
    n = len(arrays['ids'])
    mask = (np.random.default_rng(9).random(n) < 0.5).astype(np.uint8)
    sc, tc = _run('v6a', arrays, cls, rcls, mask, 'cpu')
    sg, tg = _run('v6a', arrays, cls, rcls, mask, 'cuda')
    assert np.array_equal(tc, tg)
    assert np.max(np.abs(sc['v'] - sg['v'])) < 1e-3
    assert np.max(np.abs(sc['g'] - sg['g'])) < 1e-4


# --- the declared transfer, at unit level ---------------------------------------

def transfer_probe(release, delta=0.05, kinetics=rk.KINETICS_PRIMARY, w_ref=None):
    """Two graded cells: a photoreceptor-class cell held at its anchor +- delta by
    drive, an inhibitory (histaminergic) edge of weight W_ref onto a default-class
    target with nothing else.  Returns the small-signal DC voltage gain."""
    w_ref = gr.W_REF_PHOTORECEPTOR if w_ref is None else w_ref
    out = []
    for d in (-delta, +delta):
        b = Brain(arrays=chain(2, [((0, 1), -w_ref)]), dynamics='v6a', backend='cpu',
                  graded_policy=np.array([1, 1], np.uint8), kinetics=kinetics,
                  receptor_classes=['hiscl', 'nicotinic'],
                  release=release, release_classes=[gr.CLASS_PHOTORECEPTOR, gr.CLASS_DEFAULT])
        drive = np.array([gr.ANCHOR_PHOTORECEPTOR_MV - (-52.0) + d, 0.0], np.float32)
        for _ in range(60):
            b.step(drive, 10.0)
        out.append((float(b.v[0]), float(b.v[1])))
    (p0, q0), (p1, q1) = out
    return (q1 - q0) / (p1 - p0)


@pytest.mark.parametrize('kinetics', [rk.KINETICS_PRIMARY, rk.KINETICS_V4_EQUIVALENT])
def test_two_cell_probe_reproduces_the_declared_photoreceptor_gain(kinetics):
    g = transfer_probe(gr.RELEASE_PRIMARY, kinetics=kinetics)
    assert g == pytest.approx(-gr.MEASURED_GAIN_PHOTORECEPTOR, rel=0.02)
    g5 = transfer_probe(gr.RELEASE_V5_LINEAR, kinetics=kinetics)
    d = gr.derive_gain(gr.MEASURED_GAIN_PHOTORECEPTOR, gr.W_REF_PHOTORECEPTOR,
                       gr.ANCHOR_PHOTORECEPTOR_MV)
    assert g5 == pytest.approx(-d['v4_small_signal_gain'], rel=0.02)


# --- checkpoints ------------------------------------------------------------------

def test_checkpoints_refuse_another_release_table_and_version():
    arrays, cls, rcls = random_graph(n=50, k=4)
    mask = np.ones(50, np.uint8)
    a = Brain(arrays=dict(arrays), dynamics='v6a', backend='cpu', graded_policy=mask,
              receptor_classes=cls, release_classes=rcls)
    a.step(np.zeros(50, np.float32), 1.0)
    snap = a.snapshot_state()
    assert 'release_sha256' in snap
    b = Brain(arrays=dict(arrays), dynamics='v6a', backend='cpu', graded_policy=mask,
              receptor_classes=cls, release_classes=rcls, release=gr.RELEASE_V5_LINEAR)
    with pytest.raises(ValueError, match='release'):
        b.restore_state(snap)
    c = Brain(arrays=dict(arrays), dynamics='v5', backend='cpu', graded_policy=mask,
              receptor_classes=cls)
    with pytest.raises(ValueError, match='v6a'):
        c.restore_state(snap)
    a2 = Brain(arrays=dict(arrays), dynamics='v6a', backend='cpu', graded_policy=mask,
               receptor_classes=cls, release_classes=rcls)
    a2.restore_state(snap)


def test_release_arguments_are_refused_outside_v6a():
    arrays, cls, rcls = random_graph(n=20, k=2)
    with pytest.raises(ValueError, match='v6a'):
        Brain(arrays=dict(arrays), dynamics='v5', backend='cpu', receptor_classes=cls,
              graded_policy=np.ones(20, np.uint8), release=gr.RELEASE_PRIMARY)


# --- the real graph ------------------------------------------------------------------

def test_reference_weight_is_rederived_from_the_pinned_graph():
    from brainlab.graph_identity import resolve_graph_dir
    gdir, _ = resolve_graph_dir()
    path = pathlib.Path(gdir) / 'graph.npz'
    if not path.exists():
        pytest.skip('real graph not linked')
    from brainlab.brain import _v3_policy_weight
    from brainlab.graded_policy import load_cell_classes
    with np.load(path) as g:
        arr = {k: g[k] for k in ('ptr', 'post', 'weight', 'ids')}
    w = _v3_policy_weight(arr)
    ct, _ = load_cell_classes()
    n = len(ct)
    pre = np.repeat(np.arange(n), np.diff(arr['ptr']))
    post = arr['post']
    isr = ct == 'R1-R6'
    lmc = np.isin(ct, ['L1', 'L2'])
    sel = isr[pre] & lmc[post]
    assert np.all(w[sel] < 0)
    W = np.bincount(post[sel], weights=-w[sel].astype(np.float64), minlength=n)[lmc]
    W = W[W > 0]
    assert len(W) == 1623
    assert W.mean() == pytest.approx(gr.W_REF_PHOTORECEPTOR, abs=5e-4)
