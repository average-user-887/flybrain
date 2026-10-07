"""GpuCohortEngine against the CPU-loop reference, and its GPU identities.

The checks live in ``brainlab.cohort.contract`` (also used by
``neurofly cohort verify``); the stimulus, seeds and tolerances are
preregistered in ``tests/cohort_contract.json``, shipped byte-identical as
``brainlab/cohort/cohort_contract.json``.  Real-graph tests need the MaleCNS
graph (``NEUROFLY_GRAPH_DIR`` / ``NEUROFLY_CONNECTOME_DIR`` or the checkout
defaults) and skip without it; everything GPU skips without a CUDA device.

``NEUROFLY_COHORT_GPU`` selects the device (index or name substring, e.g.
``1660``); ``NEUROFLY_COHORT_EVIDENCE`` names a directory for JSON reports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

from brainlab.cohort import contract as C  # noqa: E402

TESTS_CONTRACT = Path(__file__).with_name('cohort_contract.json')


def _gpu_ok() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_ok(), reason='needs a CUDA device with CuPy')


def test_packaged_contract_is_byte_identical_to_the_preregistered_one():
    assert C.CONTRACT_PATH.read_bytes() == TESTS_CONTRACT.read_bytes()


def _make(arrays, n_flies):
    engine = C.gpu_factory()(arrays, n_flies)
    wanted = os.environ.get('NEUROFLY_COHORT_GPU')
    if wanted and not wanted.isdigit():
        assert wanted in engine.device_name, engine.device_name
    return engine


def _synthetic(n=3000, k=20, seed=3):
    rng = np.random.default_rng(seed)
    degree = rng.poisson(k, size=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=ptr[1:])
    post = rng.integers(0, n, size=int(ptr[-1]), dtype=np.int32)
    weight = (rng.geometric(0.3, size=len(post)) * np.where(rng.random(len(post)) < 0.3, -1, 1)
              * 0.275 * 6).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=np.ascontiguousarray(weight),
                ids=np.arange(n, dtype=np.int64))


_REAL = {}


def real_graph() -> dict:
    if 'arrays' not in _REAL:
        try:
            _REAL['arrays'] = C.real_graph()
        except Exception as exc:   # graph or transmitter table not installed
            _REAL['arrays'], _REAL['why'] = None, f'{type(exc).__name__}: {exc}'
    if _REAL['arrays'] is None:
        pytest.skip(f'real MaleCNS graph unavailable ({_REAL["why"]})')
    return _REAL['arrays']


def _evidence(name, report):
    out = os.environ.get('NEUROFLY_COHORT_EVIDENCE')
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / name).write_text(json.dumps(report, indent=2))


# ------------------------------------------------------------ fast (synthetic)
@needs_gpu
def test_graph_is_read_only_and_uploaded_once():
    arrays = _synthetic()
    a, b = _make(arrays, 2), _make(arrays, 3)
    assert not arrays['weight'].flags.writeable
    assert a.d_edge_inc is b.d_edge_inc and a.d_post is b.d_post


@needs_gpu
def test_refusals_are_atomic_synthetic():
    assert C.refusals(_synthetic(), _make) == []


@needs_gpu
def test_identities_synthetic():
    arrays = _synthetic()
    assert C.batch_invariance(arrays, _make) == []
    assert C.cross_fly_isolation(arrays, _make) == []
    assert C.resume(arrays, _make) == []


@needs_gpu
def test_state_round_trips_with_cpu_brain_format():
    from brainlab.cohort.api import CpuLoopCohortEngine
    arrays = _synthetic()
    cpu = CpuLoopCohortEngine(arrays, 1)
    C.run_calls(cpu, ticks=60)
    engine = _make(arrays, 1)
    engine.write_state(0, cpu.read_state(0))       # CPU -> GPU
    back = engine.read_state(0)
    cpu2 = CpuLoopCohortEngine(arrays, 1)
    cpu2.write_state(0, back)                       # GPU -> CPU
    ref = cpu.read_state(0)
    for key in ('v', 'g', 'refractory', 'queue_count', 'active_flag', 'counts'):
        assert np.array_equal(cpu2.read_state(0)[key], ref[key]), key
    for key in ('cursor', 'total_spikes', 'sim_ms'):
        assert back[key] == ref[key]


@needs_gpu
def test_multi_tick_calls_equal_single_ticks():
    arrays = _synthetic()
    a, b = _make(arrays, 2), _make(arrays, 2)
    C.run_calls(a, ticks=200, call=20)
    C.run_calls(b, ticks=200, call=1)
    for k in range(2):
        sa, sb = a.read_state(k), b.read_state(k)
        for key in ('counts', 'sim_ms'):   # last-call counts; sim_ms sums 0.1 vs 2.0 steps
            sa.pop(key), sb.pop(key)
        assert C.same_state(sa, sb)


def test_cpu_engine_passes_identity_and_refusal_checks():
    arrays = _synthetic(n=600)
    make = C.cpu_factory()
    assert C.refusals(arrays, make) == []
    assert C.cross_fly_isolation(arrays, make) == []
    assert C.resume(arrays, make) == []


# ----------------------------------------------------------------- real graph
@needs_gpu
def test_gpu_matches_cpu_reference_on_real_graph():
    arrays = real_graph()
    report = C.reference_comparison(arrays, _make)
    if report['verdict'] == 'FAIL' and os.environ.get('NEUROFLY_COHORT_EVIDENCE'):
        report['descriptive'] = C.descriptive_full_run(arrays, _make, report['n_flies'], report['ticks'])
    _evidence('reference_comparison.json', report)
    assert sum(report['spikes_per_fly']) > 0, 'workload must spike'
    assert report['first_divergence'] is None, f"FAIL at {report['first_divergence']}"


@needs_gpu
def test_cross_engine_restore_matches_on_real_graph():
    report = C.cross_engine_restore(real_graph(), _make)
    _evidence('cross_engine_restore.json', report)
    for direction, rep in report.items():
        assert sum(rep['spikes_per_fly']) > 0, direction
        assert rep['first_divergence'] is None, f"{direction} FAIL at {rep['first_divergence']}"


@needs_gpu
def test_batch_invariance_real_graph():
    assert C.batch_invariance(real_graph(), _make) == []


@needs_gpu
def test_cross_fly_isolation_real_graph():
    assert C.cross_fly_isolation(real_graph(), _make) == []


@needs_gpu
def test_resume_is_byte_identical_real_graph():
    assert C.resume(real_graph(), _make) == []
