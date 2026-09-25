"""CUDA v3 backend against the numba CPU reference.

Runs on a real GPU, or on numba's CUDA simulator with NUMBA_ENABLE_CUDASIM=1
(slow, so the graphs here are tiny).  Skipped when neither is available.
"""
import os

import numpy as np
import pytest

from brainlab.brain import Brain
from brainlab.cuda_engine import cuda_available

pytestmark = pytest.mark.skipif(
    not (cuda_available() or os.environ.get('NUMBA_ENABLE_CUDASIM') == '1'),
    reason='needs an NVIDIA GPU or NUMBA_ENABLE_CUDASIM=1')


def _graph(n=120, k=12, seed=3):
    rng = np.random.default_rng(seed)
    degree = rng.poisson(k, size=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=ptr[1:])
    post = rng.integers(0, n, size=int(ptr[-1]), dtype=np.int32)
    weight = (rng.geometric(0.3, size=len(post)) * np.where(rng.random(len(post)) < 0.3, -1, 1)
              * 0.275 * 6).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=np.ascontiguousarray(weight), ids=np.arange(n, dtype=np.int64))


def _run(brain, ms=12.0, window=2.0):
    drive = np.zeros(brain.n, dtype=np.float32)
    drive[:brain.n // 8] = 20.0
    trains = []
    for _ in range(int(ms / window)):
        counts, _ = brain.step(drive, window)
        trains.append(counts)
    return np.array(trains)


def test_cuda_matches_cpu_reference():
    arrays = _graph()
    cpu = _run(Brain(arrays=arrays, dynamics='v3'))
    gpu_brain = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    gpu = _run(gpu_brain)
    assert cpu.sum() > 0, 'workload must spike'
    assert np.array_equal(cpu, gpu)
    ref = Brain(arrays=arrays, dynamics='v3')
    _run(ref)
    state = gpu_brain.snapshot_state()
    np.testing.assert_allclose(state['v'], ref.v, atol=1e-4)
    np.testing.assert_allclose(state['g'], ref.g, atol=1e-5)
    assert np.array_equal(state['refractory'], ref.refractory)
    assert set(state['active'][:state['nactive'][0]]) == set(ref.active[:ref.nactive[0]])


def test_cuda_is_deterministic_and_restores():
    arrays = _graph(seed=5)
    a = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    first = _run(a, ms=6.0)
    snap = a.snapshot_state()
    tail_a = _run(a, ms=6.0)
    b = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    assert np.array_equal(_run(b, ms=6.0), first)
    b.restore_state(snap)
    assert np.array_equal(_run(b, ms=6.0), tail_a)


def test_cuda_refuses_other_dynamics_and_weight_edits():
    arrays = _graph()
    with pytest.raises(ValueError):
        Brain(arrays=arrays, dynamics='v1', backend='cuda')
    brain = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    with pytest.raises(ValueError):
        brain.weight[0] = 1.0
    arrays['weight'][0] = arrays['weight'][0]  # the shared graph itself stays writable


def test_plastic_edge_updates_match_cpu():
    arrays = _graph(seed=9)
    edges = np.arange(0, len(arrays['post']), 7)
    new = (arrays['weight'][edges] * 1.5).astype(np.float32)
    results = []
    for backend in ('cpu', 'cuda'):
        brain = Brain(arrays=dict(arrays, weight=arrays['weight'].copy()), dynamics='v3', backend=backend)
        _run(brain, ms=4.0)
        brain.set_edge_weights(edges, new)
        results.append(_run(brain, ms=8.0))
    assert results[0].sum() > 0
    assert np.array_equal(results[0], results[1])


def test_weight_assignment_and_shared_device_graph():
    arrays = _graph(seed=11)
    for value in arrays.values():
        value.flags.writeable = False            # like SharedGraph
    a = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    b = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    assert a._gpu.d_post is b._gpu.d_post      # one device copy of the graph
    assert a._gpu.d_edge_inc is b._gpu.d_edge_inc
    working = arrays['weight'].copy()
    working[:50] *= 2.0
    b.weight = working                           # registry-style materialize
    assert a._gpu.d_edge_inc is not b._gpu.d_edge_inc
    ref = Brain(arrays=dict(arrays, weight=working), dynamics='v3', backend='cpu')
    assert np.array_equal(_run(b), _run(ref))
    b.update_weights(np.arange(10))              # in-place edit path of the registry
    working[:10] = 0.0
    b.update_weights(np.arange(10))
    ref.set_edge_weights(np.arange(10), np.zeros(10, dtype=np.float32))
    assert np.array_equal(_run(b), _run(ref))


def test_auto_backend_selects_gpu_for_v3_only(monkeypatch):
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'auto')
    arrays = _graph()
    assert Brain(arrays=arrays, dynamics='v3').backend == 'cuda'
    assert Brain(arrays=arrays, dynamics='v1').backend == 'cpu'


def test_cuda_reset_state_replays_like_a_new_brain():
    arrays = _graph(seed=7)
    brain = Brain(arrays=arrays, dynamics='v3', backend='cuda')
    first = _run(brain)
    assert first.sum() > 0, 'workload must spike'
    brain.reset_state()
    assert np.array_equal(_run(brain), first)
    assert np.array_equal(_run(Brain(arrays=arrays, dynamics='v3', backend='cuda')), first)
