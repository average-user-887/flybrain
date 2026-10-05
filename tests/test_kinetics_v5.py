"""LIF dynamics v5: v4 plus declared per-receptor-class kinetics (spec §9).

Holds the locked declaration to account: v1-v4 untouched and their pins
intact, v5 with v4's kinetics bit-identical to v4 (CPU and GPU), distinct
kinetics per class, the charge-preserving quantum, and checkpoints that refuse
a different version or kinetics table.
"""
import hashlib
import math
import pathlib

import numpy as np
import pytest

from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain
from brainlab.graph_identity import DYNAMICS_VERSIONS, LIF_DYNAMICS_V5, dynamics_pin

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts', 'active',
         'active_flag', 'nactive', 'rel_ring')
PINS_BEFORE_V5 = {
    'v1': '1b17bfc6cf8c4f54dbda73d715fec53971c369d0495a60d938cd56fa8320f146',
    'v2': '1ea33ba1d05d815cf8f83c0e3a783d3ff8dec101c44703eaad4c667f8cfbeb56',
    'v3': '5739c4f4ffb84979d90ad430d2405bfc0b7736c7c75f2a6e676ef606dffb4a41',
    'v4': '7ae0966578aa3c4a948021b8c38ce8c4de89aad8d67ac0f6d5c9c094b021ed44',
}


def random_graph(n=400, k=10, seed=3):
    rng = np.random.default_rng(seed)
    post = rng.integers(0, n, size=n * k).astype(np.int32)
    ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    # one sign per presynaptic neuron, as under the v3 policy
    cls = rng.choice(rk.CLASSES, size=n, p=[0.6, 0.15, 0.15, 0.1])
    sign = np.where(cls == 'nicotinic', 1.0, -1.0)
    weight = (np.repeat(sign, k) * rng.uniform(0.5, 6.0, n * k)).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=weight, ids=np.arange(1, n + 1, dtype=np.int64)), cls


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

def test_v1_to_v4_pins_are_unchanged_and_v5_has_its_own():
    for v, pin in PINS_BEFORE_V5.items():
        assert dynamics_pin(v) == pin, v
    assert 'v5' in DYNAMICS_VERSIONS
    assert dynamics_pin('v5') not in PINS_BEFORE_V5.values()
    assert LIF_DYNAMICS_V5['dynamics_version'] == 'v5'


def test_declaration_lock_matches_the_hash_the_code_cites():
    locked = ROOT / 'docs/receipts/receptor_kinetics_v5_declaration.locked.md'
    assert hashlib.sha256(locked.read_bytes()).hexdigest() == LIF_DYNAMICS_V5['declaration_sha256']


def test_default_dynamics_is_still_v3(monkeypatch):
    monkeypatch.delenv('NEUROFLY_LIF_DYNAMICS', raising=False)
    from brainlab.graph_identity import active_dynamics_version
    assert active_dynamics_version() == 'v3'


def test_primary_table_has_every_class_and_the_v4_table_is_all_5ms():
    for name in (rk.KINETICS_PRIMARY, rk.KINETICS_V4_EQUIVALENT):
        assert set(rk.KINETICS[name]) == set(rk.CLASSES)
    assert set(rk.KINETICS[rk.KINETICS_V4_EQUIVALENT].values()) == {5.0}


def test_classes_come_from_the_presynaptic_transmitter():
    got = rk.classes_from_transmitters(['acetylcholine', 'gaba', 'glutamate', 'histamine',
                                        'unclear', 'nan', 'dopamine'])
    assert list(got) == ['nicotinic', 'gaba_a', 'glucl', 'hiscl', 'nicotinic', 'nicotinic', 'nicotinic']


def test_quantum_factor_preserves_discrete_time_mean_conductance():
    dt = 0.1
    for tau in (1.0, 5.0, 12.0, 40.0):
        q = rk.quantum_factor(tau, dt)
        # steady state of g <- g*a + q*u for constant u is q*u/(1-a)
        ss = q / (1 - math.exp(-dt / tau))
        ref = 1.0 / (1 - math.exp(-dt / rk.TAU_REF_MS))
        assert ss == pytest.approx(ref, rel=1e-12)
    assert rk.quantum_factor(5.0, dt) == 1.0


# --- v5 with v4 kinetics is v4 ------------------------------------------------

def _run(version, arrays, cls, mask, backend, **kw):
    n = len(arrays['ids'])
    extra = dict(graded_policy=mask)
    if version == 'v5':
        extra.update(receptor_classes=cls, **kw)
    b = Brain(arrays=dict(arrays), dynamics=version, backend=backend, **extra)
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


@pytest.mark.parametrize('graded_frac', [0.0, 0.5])
def test_v5_with_v4_kinetics_is_bit_identical_to_v4_on_cpu(graded_frac):
    arrays, cls = random_graph()
    n = len(arrays['ids'])
    mask = (np.random.default_rng(5).random(n) < graded_frac).astype(np.uint8)
    s4, t4 = _run('v4', arrays, cls, mask, 'cpu')
    s5, t5 = _run('v5', arrays, cls, mask, 'cpu', kinetics=rk.KINETICS_V4_EQUIVALENT)
    for k in STATE:
        assert np.array_equal(s4[k], s5[k]), k
    assert np.array_equal(t4, t5) and t4.sum() > 0


def test_v5_with_v4_kinetics_is_bit_identical_to_v4_on_gpu():
    from brainlab.cupy_v4 import cupy_available
    if not cupy_available():
        pytest.skip('no CuPy GPU on this host')
    arrays, cls = random_graph(n=600, k=12, seed=7)
    n = len(arrays['ids'])
    mask = (np.random.default_rng(7).random(n) < 0.5).astype(np.uint8)
    s4, t4 = _run('v4', arrays, cls, mask, 'cuda')
    s5, t5 = _run('v5', arrays, cls, mask, 'cuda', kinetics=rk.KINETICS_V4_EQUIVALENT)
    for k in ('v', 'g', 'refractory', 'rel_ring'):
        assert np.array_equal(s4[k], s5[k]), k
    assert np.array_equal(t4, t5) and t4.sum() > 0


def test_cupy_v5_matches_the_cpu_reference_with_declared_kinetics():
    from brainlab.cupy_v4 import cupy_available
    if not cupy_available():
        pytest.skip('no CuPy GPU on this host')
    arrays, cls = random_graph(n=600, k=12, seed=9)
    n = len(arrays['ids'])
    mask = (np.random.default_rng(9).random(n) < 0.5).astype(np.uint8)
    sc, tc = _run('v5', arrays, cls, mask, 'cpu')
    sg, tg = _run('v5', arrays, cls, mask, 'cuda')
    assert np.array_equal(tc, tg)
    assert np.max(np.abs(sc['v'] - sg['v'])) < 1e-3
    assert np.max(np.abs(sc['g'] - sg['g'])) < 1e-4


# --- the declared kinetics ------------------------------------------------------

def unitary(cls_name, kinetics=None, steps=1500):
    """One presynaptic spike onto a passive target; returns the target's V trace."""
    w = 4.0 if cls_name == 'nicotinic' else -4.0
    arrays = chain(2, [((0, 1), w)])
    classes = np.array([cls_name, 'nicotinic'], dtype=object)
    b = Brain(arrays=arrays, dynamics='v5', backend='cpu', graded_policy=np.array([0, 1], np.uint8),
              receptor_classes=classes, kinetics=kinetics)
    trace = []
    drive = np.zeros(2, np.float32)
    drive[0] = 200.0
    b.step(drive, 2.0)       # one spike at ~0.7 ms; refractory covers the rest
    assert b.total_spikes == 1
    for _ in range(steps):
        b.step(np.zeros(2, np.float32), 0.1)
        trace.append(float(b.v[1]))
    return np.array(trace) + 52.0  # deflection from V_rest, mV


def test_each_class_has_its_declared_decay_and_charge_is_preserved():
    tau = rk.KINETICS[rk.KINETICS_PRIMARY]
    ref = {c: unitary(c, rk.KINETICS_V4_EQUIVALENT) for c in rk.CLASSES}
    got = {c: unitary(c) for c in rk.CLASSES}
    for c in rk.CLASSES:
        # same PSP area (charge preserved), to the membrane's own small nonlinearity
        assert np.sum(got[c]) == pytest.approx(np.sum(ref[c]), rel=0.05), c
    # slower class -> later peak
    peak = {c: int(np.argmax(np.abs(got[c]))) for c in rk.CLASSES}
    order = sorted(rk.CLASSES, key=lambda c: tau[c])
    for a, b_ in zip(order, order[1:]):
        if tau[a] < tau[b_]:
            assert peak[a] < peak[b_], (a, b_, peak)


# --- checkpoints ------------------------------------------------------------------

def test_v4_and_v5_checkpoints_refuse_each_other():
    arrays, cls = random_graph(n=50, k=4)
    mask = np.zeros(50, np.uint8)
    v4 = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=mask)
    v5 = Brain(arrays=dict(arrays), dynamics='v5', backend='cpu', graded_policy=mask,
               receptor_classes=cls, kinetics=rk.KINETICS_V4_EQUIVALENT)
    with pytest.raises(ValueError, match='v4'):
        v5.restore_state(v4.snapshot_state())
    with pytest.raises(ValueError, match='v5'):
        v4.restore_state(v5.snapshot_state())


def test_a_v5_checkpoint_refuses_a_different_kinetics_table():
    arrays, cls = random_graph(n=50, k=4)
    mask = np.zeros(50, np.uint8)
    a = Brain(arrays=dict(arrays), dynamics='v5', backend='cpu', graded_policy=mask,
              receptor_classes=cls, kinetics=rk.KINETICS_V4_EQUIVALENT)
    b = Brain(arrays=dict(arrays), dynamics='v5', backend='cpu', graded_policy=mask,
              receptor_classes=cls)
    with pytest.raises(ValueError):
        b.restore_state(a.snapshot_state())


def test_v5_snapshot_round_trips():
    arrays, cls = random_graph(n=200, k=6)
    mask = (np.random.default_rng(2).random(200) < 0.5).astype(np.uint8)
    b = Brain(arrays=dict(arrays), dynamics='v5', backend='cpu', graded_policy=mask, receptor_classes=cls)
    drive = np.full(200, 6.0, np.float32)
    b.step(drive, 20.0)
    snap = b.snapshot_state()
    b.step(drive, 20.0)
    after = {k: getattr(b, k).copy() for k in STATE}
    b.restore_state(snap)
    b.step(drive, 20.0)
    for k in STATE:
        assert np.array_equal(after[k], getattr(b, k)), k


def test_kinetics_arguments_are_refused_outside_v5():
    arrays, cls = random_graph(n=20, k=2)
    with pytest.raises(ValueError):
        Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=np.zeros(20, np.uint8),
              kinetics=rk.KINETICS_PRIMARY)


# --- the encoder additions ----------------------------------------------------------

def test_per_eye_encoder_with_equal_directions_is_the_v4_encoder():
    from brainlab.graph_identity import GraphUnavailable
    from brainlab.photoreceptor_io import PhotoreceptorGratingEncoder, resolve_photoreceptor_io
    try:
        io = resolve_photoreceptor_io()
    except (GraphUnavailable, FileNotFoundError, OSError):
        pytest.skip('real graph not present')
    enc = PhotoreceptorGratingEncoder(io)
    n = max(int(v.max()) for v in io.r_nodes.values()) + 1
    for theta, t in ((0.0, 0.0), (135.0, 37.0), (300.0, 1234.0)):
        a = np.zeros(n, np.float32)
        b = np.zeros(n, np.float32)
        enc.encode(a, t, theta, 1.0)
        enc.encode_per_eye(b, t, {'L': theta, 'R': theta}, 1.0)
        assert np.array_equal(a, b)
    f = np.zeros(n, np.float32)
    enc.encode_flicker(f, 0.0, 1.0)
    driven = np.concatenate(list(io.r_nodes.values()))
    assert np.allclose(f[driven], 20.0)
