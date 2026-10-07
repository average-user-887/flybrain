"""GpuCohortEngine against the CPU-loop reference, and its GPU identities.

The stimulus, seeds and tolerances are preregistered in
``tests/cohort_contract.json``; this file reads them from there.  Real-graph
tests need the MaleCNS graph (``NEUROFLY_GRAPH_DIR`` /
``NEUROFLY_CONNECTOME_DIR`` or the checkout defaults) and skip without it.
Everything here skips cleanly without a usable CUDA device.

``NEUROFLY_COHORT_GPU`` selects the device (index or name substring, e.g.
``1660``); ``NEUROFLY_COHORT_EVIDENCE`` names a directory that receives the
reference-comparison report as JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

CONTRACT = json.loads((Path(__file__).with_name('cohort_contract.json')).read_text())
SEED = 20261007


def _gpu_ok() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_ok(), reason='needs a CUDA device with CuPy')


def _engine(arrays, n_flies):
    from brainlab.cohort.gpu import GpuCohortEngine
    engine = GpuCohortEngine(arrays, n_flies, validate=False)
    wanted = os.environ.get('NEUROFLY_COHORT_GPU')
    if wanted and not wanted.isdigit():
        assert wanted in engine.device_name, engine.device_name
    return engine


def contract_drive(n: int, n_flies: int, tick: int, perturb_fly=None) -> np.ndarray:
    """The preregistered tonic drive for every fly at ``tick``."""
    base = SEED if tick < 100 else SEED + 1000
    drive = np.zeros((n_flies, n), dtype=np.float32)
    for k in range(n_flies):
        seed = SEED + 5000 if k == perturb_fly else base + k
        idx = np.random.default_rng(seed).choice(n, n // 50, replace=False)
        drive[k, idx] = 20.0
    return drive


def _drive_cache(n, n_flies, perturb_fly=None):
    return {phase: contract_drive(n, n_flies, phase, perturb_fly) for phase in (0, 100)}


def _run(engine, ticks=200, call=20, perturb_fly=None):
    """Run in calls of ``call`` ticks (never crossing tick 100); per-call counts."""
    drives = _drive_cache(engine.n, engine.n_flies, perturb_fly)
    out, t = [], 0
    while t < ticks:
        out.append(engine.step(drives[0 if t < 100 else 100], call))
        t += call
    return out


def _same_state(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                return False
        elif x != y or type(x) is not type(y):
            return False
    return True


# ------------------------------------------------------------------ graphs
def _synthetic(n=3000, k=20, seed=3):
    rng = np.random.default_rng(seed)
    degree = rng.poisson(k, size=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=ptr[1:])
    post = rng.integers(0, n, size=int(ptr[-1]), dtype=np.int32)
    weight = (rng.geometric(0.3, size=len(post)) * np.where(rng.random(len(post)) < 0.3, -1, 1)
              * 0.275 * 6).astype(np.float32)
    arrays = dict(ptr=ptr, post=post, weight=np.ascontiguousarray(weight),
                  ids=np.arange(n, dtype=np.int64))
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


_REAL = {}


def real_graph() -> dict:
    """The real MaleCNS graph with the v3 transmitter policy, loaded once."""
    if 'arrays' not in _REAL:
        try:
            from experiment_registry import SharedGraph
            _REAL['arrays'] = SharedGraph.load_for_dynamics(dynamics='v3').arrays
        except Exception as exc:   # graph or transmitter table not installed
            _REAL['arrays'] = None
            _REAL['why'] = f'{type(exc).__name__}: {exc}'
    if _REAL['arrays'] is None:
        pytest.skip(f'real MaleCNS graph unavailable ({_REAL["why"]})')
    return _REAL['arrays']


# ------------------------------------------- refusals (mirror test_cohort_api)
def _snap_all(engine):
    return [engine.read_state(k) for k in range(engine.n_flies)]


def test_graph_is_read_only_and_uploaded_once():
    arrays = _synthetic()
    a = _engine(arrays, 2)
    b = _engine(arrays, 3)
    assert not arrays['weight'].flags.writeable
    assert a.d_edge_inc is b.d_edge_inc and a.d_post is b.d_post


def test_nan_in_one_fly_advances_no_fly():
    engine = _engine(_synthetic(), 2)
    _run(engine, ticks=20)
    before = _snap_all(engine)
    drive = contract_drive(engine.n, 2, 0)
    drive[1, 0] = np.nan
    with pytest.raises(ValueError):
        engine.step(drive, 10)
    assert all(_same_state(x, y) for x, y in zip(before, _snap_all(engine)))


@pytest.mark.parametrize('ticks', [1.5, 0, -1, True, '3'])
def test_non_integer_ticks_refused(ticks):
    engine = _engine(_synthetic(), 1)
    before = engine.read_state(0)
    with pytest.raises(ValueError):
        engine.step(contract_drive(engine.n, 1, 0), ticks)
    assert _same_state(before, engine.read_state(0))


def test_write_state_refusals_change_nothing():
    arrays = _synthetic()
    engine = _engine(arrays, 2)
    _run(engine, ticks=40)
    good = engine.read_state(0)
    engine.step(contract_drive(engine.n, 2, 0), 7)
    before = _snap_all(engine)

    def refused(state, fly=1):
        with pytest.raises(ValueError):
            engine.write_state(fly, state)
        assert all(_same_state(x, y) for x, y in zip(before, _snap_all(engine))), \
            'a refused write changed live state'

    for dyn in ('v2', 'v4', None):
        refused(dict(good, dynamics=dyn))
    for key in ('queue', 'v', 'cursor', 'active_flag'):
        missing = dict(good)
        del missing[key]
        refused(missing)
    partial = dict(good, v=np.full_like(good['v'], -30.0), g=np.zeros(3, dtype=np.float32))
    refused(partial)                                       # valid field + invalid field
    refused(dict(good, v=good['v'][:-1]))
    refused(dict(good, g=good['g'][0]))
    refused(dict(good, queue=good['queue'][:-1]))
    refused(dict(good, v=good['v'].astype(np.float64)))
    refused(dict(good, refractory=good['refractory'].astype(np.int32)))
    refused(dict(good, active_flag=good['active_flag'].astype(bool)))
    refused(dict(good, v=np.where(np.arange(engine.n) == 3, np.nan, good['v']).astype(np.float32)))
    refused(dict(good, cursor=-1))
    refused(dict(good, sim_ms=float('nan')))
    refused(dict(good, queue_count=np.full_like(good['queue_count'], engine.n + 1)))
    bad_q = {**good, 'queue': good['queue'].copy(), 'queue_count': good['queue_count'].copy()}
    bad_q['queue_count'][0] = 1
    bad_q['queue'][0, 0] = engine.n                       # out-of-range neuron
    refused(bad_q)
    refused(dict(good, active_flag=np.zeros_like(good['active_flag'])))   # list/flag disagree
    refused(good, fly=2)
    refused(good, fly=-1)
    engine.write_state(1, good)
    assert _same_state(engine.read_state(1), good)
    assert _same_state(engine.read_state(0), before[0])


def test_state_round_trips_with_cpu_brain_format():
    from brainlab.cohort.api import CpuLoopCohortEngine
    arrays = _synthetic()
    cpu = CpuLoopCohortEngine(arrays, 1)
    _run(cpu, ticks=60)
    engine = _engine(arrays, 1)
    engine.write_state(0, cpu.read_state(0))       # CPU -> GPU
    back = engine.read_state(0)
    cpu2 = CpuLoopCohortEngine(arrays, 1)
    cpu2.write_state(0, back)                       # GPU -> CPU
    ref = cpu.read_state(0)
    for key in ('v', 'g', 'refractory', 'queue_count', 'active_flag', 'counts'):
        assert np.array_equal(cpu2.read_state(0)[key], ref[key]), key
    for key in ('cursor', 'total_spikes', 'sim_ms'):
        assert back[key] == ref[key]


# ------------------------------------------------- CPU reference, real graph
def _compare_tick(ref, gpu, tol):
    """Return None if within the preregistered bounds, else the first breach."""
    dv = float(np.abs(gpu['v'].astype(np.float64) - ref['v'].astype(np.float64)).max())
    if not dv <= tol['max_abs_dV_mV']:
        return dict(kind='dV', value=dv)
    gr, gg = ref['g'].astype(np.float64), gpu['g'].astype(np.float64)
    zero = gr == 0.0
    if zero.any():
        za = float(np.abs(gg[zero]).max())
        if not za <= 1e-12:
            return dict(kind='g_zero_abs', value=za)
    if (~zero).any():
        rel = float((np.abs(gg[~zero] - gr[~zero]) / np.abs(gr[~zero])).max())
        if not rel <= tol['g_rel_error_max']:
            return dict(kind='g_rel', value=rel)
    if not np.array_equal(ref['refractory'], gpu['refractory']):
        return dict(kind='refractory', value=int((ref['refractory'] != gpu['refractory']).sum()))
    return None


def _lockstep(cpu, gpu, t0, t1, report):
    """Advance both engines tick by tick from ``t0`` to ``t1``; stop at the first breach."""
    tol = CONTRACT['reference_comparison']
    n_flies = cpu.n_flies
    drives = _drive_cache(cpu.n, n_flies)
    total = np.zeros(n_flies, dtype=np.int64)
    t = t0
    for t in range(t0, t1):
        drive = drives[0 if t < 100 else 100]
        c_ref, c_gpu = cpu.step(drive, 1), gpu.step(drive, 1)
        total += c_ref.sum(axis=1)
        if not np.array_equal(c_ref, c_gpu):
            flies = sorted(set(np.nonzero((c_ref != c_gpu).any(axis=1))[0].tolist()))
            report['first_divergence'] = dict(tick=t, kind='spikes', flies=flies,
                                              neurons=int((c_ref != c_gpu).sum()))
            break
        for k in range(n_flies):
            ref, got = cpu.read_state(k), gpu.read_state(k)
            dv = float(np.abs(got['v'].astype(np.float64) - ref['v']).max())
            report['worst']['dV'] = max(report['worst']['dV'], dv)
            nz = ref['g'] != 0
            if nz.any():
                rel = float((np.abs(got['g'][nz].astype(np.float64) - ref['g'][nz])
                             / np.abs(ref['g'][nz].astype(np.float64))).max())
                report['worst']['g_rel'] = max(report['worst']['g_rel'], rel)
            breach = _compare_tick(ref, got, tol)
            if breach is not None:
                report['first_divergence'] = dict(tick=t, fly=k, **breach)
                break
        if report['first_divergence'] is not None:
            break
    report['ticks_compared'] = report.get('ticks_compared', 0) + (t - t0 + 1)
    report['spikes_per_fly'] = (np.asarray(report.get('spikes_per_fly') or np.zeros(n_flies, int))
                                + total).tolist()
    return report


def _new_report(n_flies, ticks, device):
    return dict(n_flies=n_flies, ticks=ticks, device=device, first_divergence=None,
                worst=dict(dV=0.0, g_rel=0.0), spikes_per_fly=None)


def _verdict(report):
    spiked = sum(report['spikes_per_fly']) > 0
    report['verdict'] = 'PASS' if report['first_divergence'] is None and spiked else 'FAIL'
    if not spiked:
        report['void'] = 'workload did not spike'
    return report


def reference_comparison(arrays, n_flies=8, ticks=200):
    from brainlab.cohort.api import CpuLoopCohortEngine
    cpu = CpuLoopCohortEngine(arrays, n_flies)
    gpu = _engine(arrays, n_flies)
    report = _new_report(n_flies, ticks, gpu.device_name)
    return _verdict(_lockstep(cpu, gpu, 0, ticks, report))


def cross_engine_restore(arrays, n_flies=2, split=100, ticks=200):
    """CPU state -> GPU and GPU state -> CPU at ``split``, then the same continuation."""
    from brainlab.cohort.api import CpuLoopCohortEngine
    drives = _drive_cache(arrays['ptr'].size - 1, n_flies)
    out = {}
    # CPU -> GPU
    cpu = CpuLoopCohortEngine(arrays, n_flies)
    for t in range(0, split, 20):
        cpu.step(drives[0 if t < 100 else 100], 20)
    gpu = _engine(arrays, n_flies)
    for k in range(n_flies):
        gpu.write_state(k, cpu.read_state(k))
    out['cpu_to_gpu'] = _verdict(_lockstep(cpu, gpu, split, ticks,
                                           _new_report(n_flies, ticks - split, gpu.device_name)))
    del cpu, gpu
    # GPU -> CPU
    gpu = _engine(arrays, n_flies)
    for t in range(0, split, 20):
        gpu.step(drives[0 if t < 100 else 100], 20)
    cpu = CpuLoopCohortEngine(arrays, n_flies)
    for k in range(n_flies):
        cpu.write_state(k, gpu.read_state(k))
    out['gpu_to_cpu'] = _verdict(_lockstep(cpu, gpu, split, ticks,
                                           _new_report(n_flies, ticks - split, gpu.device_name)))
    return out


def _evidence(name, report):
    out = os.environ.get('NEUROFLY_COHORT_EVIDENCE')
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / name).write_text(json.dumps(report, indent=2))


def test_cross_engine_restore_matches_on_real_graph():
    arrays = real_graph()
    spec = CONTRACT['cross_engine_restore']
    report = cross_engine_restore(arrays, spec['n_flies'], spec['split_tick'], spec['ticks'])
    _evidence('cross_engine_restore.json', report)
    for direction, rep in report.items():
        assert sum(rep['spikes_per_fly']) > 0, direction
        assert rep['first_divergence'] is None, f"{direction} FAIL at {rep['first_divergence']}"


def test_gpu_matches_cpu_reference_on_real_graph():
    arrays = real_graph()
    report = reference_comparison(arrays, CONTRACT['reference_comparison']['n_flies'],
                                  CONTRACT['reference_comparison']['ticks'])
    _evidence('reference_comparison.json', report)
    assert sum(report['spikes_per_fly']) > 0, 'workload must spike'
    assert report['first_divergence'] is None, f"FAIL at {report['first_divergence']}"


# ------------------------------------------------------ GPU identity checks
def test_batch_invariance_real_graph():
    arrays = real_graph()
    results = {}
    for B in (8, 32):
        engine = _engine(arrays, B)
        calls = _run(engine)
        results[B] = ([c.copy() for c in calls], [engine.read_state(k) for k in range(B)])
        del engine
    solo = _engine(arrays, 1)
    drives32 = _drive_cache(solo.n, 32)
    for k in range(32):
        solo.reset([0])
        calls, t = [], 0
        while t < 200:
            calls.append(solo.step(drives32[0 if t < 100 else 100][k:k + 1], 20))
            t += 20
        state = solo.read_state(0)
        for B, (bcalls, bstates) in results.items():
            if k >= B:
                continue
            assert all(np.array_equal(c[k:k + 1], s) for c, s in zip(bcalls, calls)), (B, k)
            assert _same_state(bstates[k], state), (B, k)
    assert sum(int(c.sum()) for c in results[32][0]) > 0


def test_cross_fly_isolation_real_graph():
    arrays = real_graph()
    engine = _engine(arrays, 8)
    base_calls = _run(engine)
    base = [engine.read_state(k) for k in range(8)]
    engine.reset(range(8))
    pert_calls = _run(engine, perturb_fly=3)
    pert = [engine.read_state(k) for k in range(8)]
    for k in range(8):
        if k == 3:
            continue
        assert all(np.array_equal(a[k], b[k]) for a, b in zip(base_calls, pert_calls)), k
        assert _same_state(base[k], pert[k]), k
    assert not _same_state(base[3], pert[3]), 'the perturbation must reach fly 3'


def test_resume_is_byte_identical_real_graph():
    arrays = real_graph()
    whole = _engine(arrays, 8)
    whole_calls = _run(whole)
    whole_states = [whole.read_state(k) for k in range(8)]
    del whole
    first = _engine(arrays, 8)
    drives = _drive_cache(first.n, 8)
    calls = [first.step(drives[0 if t < 100 else 100], 20) for t in range(0, 120, 20)]
    saved = [first.read_state(k) for k in range(8)]
    del first
    second = _engine(arrays, 8)
    for k in range(8):
        second.write_state(k, saved[k])
    calls += [second.step(drives[0 if t < 100 else 100], 20) for t in range(120, 200, 20)]
    assert all(np.array_equal(a, b) for a, b in zip(whole_calls, calls))
    for k in range(8):
        assert _same_state(second.read_state(k), whole_states[k]), k
