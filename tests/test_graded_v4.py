"""LIF dynamics v4: hybrid graded/spiking transmission (docs/LIF_DYNAMICS_SPEC.md §7).

The claims these tests hold to account are the ones the locked declaration
makes: subthreshold transmission works, a graded cell never spikes, the release
scale is the derived one, v1-v3 are untouched and bit-reproducible, and
checkpoints keep mutually refusing each other.
"""
import hashlib
import pathlib

import numpy as np
import pytest

from brainlab import graded_policy
from brainlab.brain import Brain
from brainlab.engine import E_EXC_MV, E_INH_MV, R_MAX_HZ, TAU_SYN_MS, V_REST_MV
from brainlab.graph_identity import (DYNAMICS_VERSIONS, LIF_DYNAMICS_V4,
                                     dynamics_pin)

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts', 'active',
         'active_flag', 'nactive')


def chain(n, edges):
    ptr = np.zeros(n + 1, dtype=np.int64)
    post, w = [], []
    for i in range(n):
        for (a, b), wt in edges:
            if a == i:
                post.append(b)
                w.append(wt)
        ptr[i + 1] = len(post)
    return dict(ptr=ptr, post=np.array(post, dtype=np.int32),
                weight=np.array(w, dtype=np.float32),
                ids=np.arange(1, n + 1, dtype=np.int64))


def random_graph(n=400, k=10, seed=3):
    rng = np.random.default_rng(seed)
    post = rng.integers(0, n, size=n * k).astype(np.int32)
    ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    sign = np.where(rng.random(n * k) < 0.75, 1.0, -1.0)
    weight = (sign * rng.uniform(0.5, 6.0, n * k)).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=weight, ids=np.arange(1, n + 1, dtype=np.int64))


# --- declaration ------------------------------------------------------------

def test_v4_is_a_declared_version_with_its_own_pin():
    assert 'v4' in DYNAMICS_VERSIONS
    pins = {v: dynamics_pin(v) for v in ('v1', 'v2', 'v3', 'v4')}
    assert len(set(pins.values())) == 4, pins
    assert LIF_DYNAMICS_V4['dynamics_version'] == 'v4'


def test_declaration_lock_matches_the_hash_the_code_cites():
    locked = ROOT / 'docs/receipts/graded_transmission_v4_declaration.locked.md'
    digest = hashlib.sha256(locked.read_bytes()).hexdigest()
    assert digest == LIF_DYNAMICS_V4['declaration_sha256']
    assert digest == graded_policy.describe()['declaration_sha256']


def test_release_function_anchors_are_the_already_declared_constants():
    g = LIF_DYNAMICS_V4['graded_release']
    assert g['r_max_hz'] == pytest.approx(1000.0 / 2.2)
    assert g['r_at_rest_hz'] == pytest.approx(R_MAX_HZ * (V_REST_MV - E_INH_MV)
                                             / (E_EXC_MV - E_INH_MV))
    # The maintained baseline is not zero: that is the point of §7.4.
    assert g['r_at_rest_hz'] > 0


def test_graded_policy_none_declares_nothing_graded():
    mask, report = graded_policy.resolve(500, policy=graded_policy.POLICY_NONE)
    assert mask.sum() == 0
    assert report['graded_neurons'] == 0


def test_graded_policy_resolves_by_class_not_row_order():
    ct = np.array(['R1-R6', 'L1', 'DNa02', 'LC4', 'HSE', 'Tm5a'], dtype=object)
    sc = np.array(['ol_sensory', 'ol_intrinsic', 'descending_neuron', 'visual_projection',
                   'visual_projection', 'ol_intrinsic'], dtype=object)
    primary = graded_policy.graded_mask(ct, sc, policy=graded_policy.POLICY_PRIMARY)
    assert list(primary) == [1, 1, 0, 0, 1, 1]      # LC4 spikes; Tm5a is tier 2
    tier1 = graded_policy.graded_mask(ct, sc, policy=graded_policy.POLICY_TIER1)
    assert list(tier1) == [1, 1, 0, 0, 1, 0]        # Tm5a is not a named type
    # The digest is over the resolved node set, so a different list re-pins.
    assert (graded_policy.graded_set_sha256(primary)
            != graded_policy.graded_set_sha256(tier1))


# --- v1-v3 are untouched ----------------------------------------------------

def test_v4_with_no_graded_class_is_bit_identical_to_v3():
    arrays = random_graph()
    n = len(arrays['ids'])
    rng = np.random.default_rng(11)
    drive = np.zeros(n, np.float32)
    drive[rng.choice(n, n // 10, replace=False)] = 25.0
    out = {}
    for version, kw in (('v3', {}), ('v4', dict(graded_policy=np.zeros(n, np.uint8)))):
        b = Brain(arrays=dict(arrays), dynamics=version, backend='cpu', **kw)
        b.step(drive, 50.0)
        b.step(np.zeros(n, np.float32), 50.0)
        out[version] = {k: getattr(b, k).copy() for k in STATE}
        out[version]['total'] = b.total_spikes
    for k in STATE:
        assert np.array_equal(out['v3'][k], out['v4'][k]), k
    assert out['v3']['total'] == out['v4']['total'] > 0


# --- the graded mechanism ---------------------------------------------------

def test_subthreshold_input_is_transmitted_under_v4_and_not_under_v3():
    arrays = chain(3, [((0, 1), 10.0), ((1, 2), 10.0)])
    read = {}
    for label, version, kw, drive1 in (
            ('v3', 'v3', {}, 4.0),
            ('v4_base', 'v4', dict(graded_policy=np.array([0, 1, 0], np.uint8)), 0.0),
            ('v4_up', 'v4', dict(graded_policy=np.array([0, 1, 0], np.uint8)), 4.0),
            ('v4_down', 'v4', dict(graded_policy=np.array([0, 1, 0], np.uint8)), -4.0)):
        b = Brain(arrays=dict(arrays), dynamics=version, backend='cpu', **kw)
        drive = np.zeros(3, np.float32)
        drive[1] = drive1
        for _ in range(60):
            b.step(drive, 5.0)
        read[label] = (float(b.v[1]), float(b.v[2]), int(b.total_spikes))
    # Node 1 stays strictly subthreshold in every condition.
    assert all(v1 < -45.0 for v1, _, _ in read.values())
    # Under v3 nothing at all reaches node 2.
    assert read['v3'][1] == pytest.approx(V_REST_MV, abs=1e-6)
    # Under v4 node 2 follows node 1 in BOTH directions around the baseline.
    assert read['v4_up'][1] > read['v4_base'][1] > read['v4_down'][1]
    assert all(spikes == 0 for _, _, spikes in read.values())


def test_a_graded_cell_never_spikes_but_drives_a_spiking_one():
    arrays = chain(3, [((0, 1), 400.0), ((1, 2), 4000.0)])
    b = Brain(arrays=arrays, dynamics='v4', backend='cpu',
              graded_policy=np.array([0, 1, 0], np.uint8))
    drive = np.zeros(3, np.float32)
    drive[0] = 30.0
    total = np.zeros(3, np.int64)
    for _ in range(20):
        counts, _ = b.step(drive, 25.0)
        total += counts
    assert total[0] > 0            # spiking presynaptic cell fires
    assert total[1] == 0           # the graded cell never does
    assert total[2] > 0            # and it drives the spiking cell downstream
    assert b.release_rate_hz()[1] > 0
    assert b.release_rate_hz()[2] == 0


@pytest.mark.parametrize('v_clamp', [-52.0, -45.0, -20.0, 0.0])
def test_release_scale_is_the_derived_one(v_clamp):
    """A graded cell at V delivers what a v3 spiking cell at r(V) Hz delivers."""
    arrays = chain(2, [((0, 1), 1.0)])
    b = Brain(arrays=arrays, dynamics='v4', backend='cpu',
              graded_policy=np.array([1, 0], np.uint8))
    drive = np.zeros(2, np.float32)
    drive[0] = v_clamp - V_REST_MV       # at rest g_tot = 1, so V_inf = V_rest + I
    b.step(drive, 500.0)
    r = R_MAX_HZ * (v_clamp - E_INH_MV) / (E_EXC_MV - E_INH_MV)
    # float32 settling leaves V a few 1e-4 mV off the clamp target.
    assert b.release_rate_hz()[0] == pytest.approx(r, rel=1e-4)
    # Steady conductance of a train at rate r, in this engine's discrete time.
    decay = np.exp(-0.1 / TAU_SYN_MS)
    expected = (1.0 / 52.0) * r * 1e-4 / (1.0 - decay)
    assert float(b.g[0, 1]) == pytest.approx(expected, rel=1e-3)


def test_graded_cells_ignore_threshold_reset_and_refractory():
    """Driven far above threshold, a graded cell sits there instead of firing."""
    arrays = chain(2, [((0, 1), 1.0)])
    b = Brain(arrays=arrays, dynamics='v4', backend='cpu',
              graded_policy=np.array([1, 0], np.uint8))
    drive = np.zeros(2, np.float32)
    drive[0] = 40.0
    counts, _ = b.step(drive, 200.0)
    assert counts[0] == 0
    assert float(b.v[0]) == pytest.approx(-12.0, abs=0.1)
    assert int(b.refractory[0]) == 0


# --- checkpoints ------------------------------------------------------------

def test_v3_and_v4_checkpoints_refuse_each_other():
    arrays = random_graph(n=100, k=4)
    n = len(arrays['ids'])
    v3 = Brain(arrays=dict(arrays), dynamics='v3', backend='cpu')
    v4 = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu',
               graded_policy=np.zeros(n, np.uint8))
    with pytest.raises(ValueError, match='v3'):
        v4.restore_state(v3.snapshot_state())
    with pytest.raises(ValueError, match='v4'):
        v3.restore_state(v4.snapshot_state())


def test_a_v4_checkpoint_refuses_a_different_graded_class_set():
    arrays = random_graph(n=100, k=4)
    n = len(arrays['ids'])
    mask_a = np.zeros(n, np.uint8)
    mask_b = mask_a.copy()
    mask_b[:5] = 1
    a = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=mask_a)
    b = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=mask_b)
    with pytest.raises(ValueError, match='graded cell-class set'):
        b.restore_state(a.snapshot_state())


def test_v4_snapshot_carries_the_release_ring_and_round_trips():
    arrays = random_graph(n=120, k=5)
    n = len(arrays['ids'])
    mask = np.zeros(n, np.uint8)
    mask[::3] = 1
    b = Brain(arrays=dict(arrays), dynamics='v4', backend='cpu', graded_policy=mask)
    drive = np.zeros(n, np.float32)
    drive[1] = 20.0
    b.step(drive, 20.0)
    snap = b.snapshot_state()
    assert 'rel_ring' in snap and snap['rel_ring'].any()
    after = b.step(drive, 20.0)[0].copy()
    b.restore_state(snap)
    assert np.array_equal(b.step(drive, 20.0)[0], after)


# --- GPU --------------------------------------------------------------------

def test_cupy_v4_matches_the_cpu_reference():
    from brainlab.cupy_v4 import cupy_available
    if not cupy_available():
        pytest.skip('no CuPy GPU on this host')
    arrays = random_graph(n=600, k=12, seed=7)
    n = len(arrays['ids'])
    rng = np.random.default_rng(7)
    mask = (rng.random(n) < 0.5).astype(np.uint8)
    drive = np.zeros(n, np.float32)
    drive[rng.choice(n, 60, replace=False)] = 12.0
    got = {}
    for backend in ('cpu', 'cuda'):
        b = Brain(arrays=dict(arrays), dynamics='v4', graded_policy=mask, backend=backend)
        total = np.zeros(n, np.int64)
        for _ in range(20):
            counts, _ = b.step(drive, 5.0)
            total += counts
        b._sync_from_gpu()
        got[backend] = (total, b.v.copy())
    assert np.array_equal(got['cpu'][0], got['cuda'][0])
    assert np.max(np.abs(got['cpu'][1] - got['cuda'][1])) < 1e-3


# --- the photoreceptor stimulus --------------------------------------------

def test_photoreceptor_io_matches_its_pin_and_drives_only_r1_r6():
    from brainlab.graph_identity import GraphUnavailable
    from brainlab.photoreceptor_io import (PHOTORECEPTOR_IO_PIN,
                                           PhotoreceptorGratingEncoder,
                                           resolve_photoreceptor_io)
    try:
        io = resolve_photoreceptor_io()
    except GraphUnavailable as exc:
        pytest.skip(f'real graph not available: {exc}')
    assert io.sha256 == PHOTORECEPTOR_IO_PIN
    assert len(io.r_nodes['L']) == 1107 and len(io.r_nodes['R']) == 2228
    driven = np.concatenate([io.r_nodes['L'], io.r_nodes['R']])
    currents = np.zeros(166_700, np.float32)
    PhotoreceptorGratingEncoder(io).encode(currents, 0.0, 0.0, 1.0)
    assert np.count_nonzero(currents) > 0
    others = np.ones(len(currents), bool)
    others[driven] = False
    assert not currents[others].any(), 'the encoder must drive R1-R6 and nothing else'
    # Retinotopy must actually vary across the eye, or there is no grating.
    assert np.ptp(io.r_position_deg['R'][:, 0]) > 50.0
