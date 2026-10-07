"""The preregistered cohort contract (``cohort_contract.json``) as runnable checks.

``run_contract(engine='gpu')`` yields one ``{name, passed, detail}`` dict per
check, in contract order:

* ``cpu_reference``: GPU vs the CPU-loop reference, tick by tick; the first
  breach of any bound is a FAIL (its descriptive all-ticks summary is in
  ``detail`` and never changes the verdict);
* ``cross_engine_restore``: CPU state into the GPU and back, same continuation;
* ``batch_invariance``, ``cross_fly_isolation``, ``resume``: byte identity;
* ``refusals``: bad drives, tick counts and states change nothing.

``engine='cpu'`` runs the identity and refusal checks on the CPU-loop engine
(the reference and cross-engine checks compare against it, so they are not
run). ``engine='gpu'`` raises ``RuntimeError`` when no CUDA device is usable.
The contract JSON shipped in this package is byte-identical to
``tests/cohort_contract.json`` (checked by the tests).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np

CONTRACT_PATH = Path(__file__).with_name('cohort_contract.json')
SEED = 20261007


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


# ------------------------------------------------------------------ inputs
def contract_drive(n: int, n_flies: int, tick: int, perturb_fly: Optional[int] = None) -> np.ndarray:
    """The preregistered tonic drive for every fly at ``tick``."""
    base = SEED if tick < 100 else SEED + 1000
    drive = np.zeros((n_flies, n), dtype=np.float32)
    for k in range(n_flies):
        seed = SEED + 5000 if k == perturb_fly else base + k
        idx = np.random.default_rng(seed).choice(n, n // 50, replace=False)
        drive[k, idx] = 20.0
    return drive


def drive_phases(n: int, n_flies: int, perturb_fly: Optional[int] = None) -> Dict[int, np.ndarray]:
    return {phase: contract_drive(n, n_flies, phase, perturb_fly) for phase in (0, 100)}


def run_calls(engine, ticks: int = 200, call: int = 20, perturb_fly=None, t0: int = 0) -> List[np.ndarray]:
    """Run ``t0..ticks`` in calls of ``call`` ticks (never crossing tick 100)."""
    drives = drive_phases(engine.n, engine.n_flies, perturb_fly)
    out, t = [], t0
    while t < ticks:
        out.append(engine.step(drives[0 if t < 100 else 100], call))
        t += call
    return out


def same_state(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            if not isinstance(y, np.ndarray) or x.dtype != y.dtype or x.shape != y.shape \
                    or x.tobytes() != y.tobytes():
                return False
        elif x != y or type(x) is not type(y):
            return False
    return True


def real_graph() -> dict:
    """The real MaleCNS graph with the v3 transmitter policy."""
    from experiment_registry import SharedGraph
    return SharedGraph.load_for_dynamics(dynamics='v3').arrays


def gpu_factory(device=None) -> Callable:
    try:
        import cupy as cp
        usable = cp.cuda.runtime.getDeviceCount() > 0
    except Exception as exc:
        raise RuntimeError(f'No usable CUDA device for the GPU cohort contract ({exc})') from exc
    if not usable:
        raise RuntimeError('No usable CUDA device for the GPU cohort contract')
    from .gpu import GpuCohortEngine
    return lambda arrays, n_flies: GpuCohortEngine(arrays, n_flies, device=device, validate=False)


def cpu_factory() -> Callable:
    from .api import CpuLoopCohortEngine
    return lambda arrays, n_flies: CpuLoopCohortEngine(arrays, n_flies)


# --------------------------------------------------------- CPU reference
def _compare_tick(ref, got, tol):
    """None if within the preregistered bounds, else the first breach."""
    dv = float(np.abs(got['v'].astype(np.float64) - ref['v'].astype(np.float64)).max())
    if not dv <= tol['max_abs_dV_mV']:
        return dict(kind='dV', value=dv)
    gr, gg = ref['g'].astype(np.float64), got['g'].astype(np.float64)
    zero = gr == 0.0
    if zero.any():
        za = float(np.abs(gg[zero]).max())
        if not za <= 1e-12:
            return dict(kind='g_zero_abs', value=za)
    if (~zero).any():
        rel = float((np.abs(gg[~zero] - gr[~zero]) / np.abs(gr[~zero])).max())
        if not rel <= tol['g_rel_error_max']:
            return dict(kind='g_rel', value=rel)
    if not np.array_equal(ref['refractory'], got['refractory']):
        return dict(kind='refractory', value=int((ref['refractory'] != got['refractory']).sum()))
    return None


def _new_report(n_flies, ticks, device):
    return dict(n_flies=n_flies, ticks=ticks, device=device, first_divergence=None,
                worst=dict(dV=0.0, g_rel=0.0), spikes_per_fly=None, ticks_compared=0)


def _lockstep(cpu, gpu, t0, t1, report, tol):
    """Advance both engines tick by tick from ``t0`` to ``t1``; stop at the first breach."""
    n_flies = cpu.n_flies
    drives = drive_phases(cpu.n, n_flies)
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
    report['ticks_compared'] += t - t0 + 1
    report['spikes_per_fly'] = total.tolist()
    return report


def _verdict(report):
    spiked = sum(report['spikes_per_fly']) > 0
    report['verdict'] = 'PASS' if report['first_divergence'] is None and spiked else 'FAIL'
    if not spiked:
        report['void'] = 'workload did not spike'
    return report


def reference_comparison(arrays, make_gpu, contract=None) -> dict:
    from .api import CpuLoopCohortEngine
    contract = contract or load_contract()
    spec = contract['reference_comparison']
    cpu = CpuLoopCohortEngine(arrays, spec['n_flies'])
    gpu = make_gpu(arrays, spec['n_flies'])
    report = _new_report(spec['n_flies'], spec['ticks'], getattr(gpu, 'device_name', None))
    return _verdict(_lockstep(cpu, gpu, 0, spec['ticks'], report, spec))


def descriptive_full_run(arrays, make_gpu, n_flies=8, ticks=200) -> dict:
    """All ticks, no early stop. Descriptive only: never changes a verdict."""
    from .api import CpuLoopCohortEngine
    cpu = CpuLoopCohortEngine(arrays, n_flies)
    gpu = make_gpu(arrays, n_flies)
    drives = drive_phases(cpu.n, n_flies)
    d = dict(label='descriptive, does not change the FAIL verdict', ticks=ticks, n_flies=n_flies,
             spike_mismatches=0, first_spike_mismatch_tick=None, max_abs_dV_mV=0.0,
             max_dV_tick_fly=None, max_g_rel=0.0, max_g_rel_tick_fly=None,
             refractory_mismatches=0, g_rel_points_over_1e_6=0, g_rel_ticks_over_1e_6=[],
             g_zero_points_nonzero=0)
    for t in range(ticks):
        drive = drives[0 if t < 100 else 100]
        c_ref, c_gpu = cpu.step(drive, 1), gpu.step(drive, 1)
        mism = int((c_ref != c_gpu).sum())
        d['spike_mismatches'] += mism
        if mism and d['first_spike_mismatch_tick'] is None:
            d['first_spike_mismatch_tick'] = t
        for k in range(n_flies):
            ref, got = cpu.read_state(k), gpu.read_state(k)
            dv = float(np.abs(got['v'].astype(np.float64) - ref['v']).max())
            if dv > d['max_abs_dV_mV']:
                d['max_abs_dV_mV'], d['max_dV_tick_fly'] = dv, [t, k]
            gr, gg = ref['g'].astype(np.float64), got['g'].astype(np.float64)
            nz = gr != 0
            d['g_zero_points_nonzero'] += int((gg[~nz] != 0).sum())
            rel = np.abs(gg[nz] - gr[nz]) / np.abs(gr[nz])
            if rel.size and float(rel.max()) > d['max_g_rel']:
                d['max_g_rel'], d['max_g_rel_tick_fly'] = float(rel.max()), [t, k]
            over = int((rel > 1e-6).sum())
            if over:
                d['g_rel_points_over_1e_6'] += over
                d['g_rel_ticks_over_1e_6'].append([t, k, over])
            d['refractory_mismatches'] += int((ref['refractory'] != got['refractory']).sum())
    d['g_rel_note'] = ('points are (tick, fly, g row, neuron) entries with |g_ref| > 0; '
                       'max excess is max_g_rel')
    return d


def cross_engine_restore(arrays, make_gpu, contract=None) -> dict:
    """CPU state -> GPU and GPU state -> CPU at the split, then the same continuation."""
    from .api import CpuLoopCohortEngine
    contract = contract or load_contract()
    spec, tol = contract['cross_engine_restore'], contract['reference_comparison']
    n_flies, split, ticks = spec['n_flies'], spec['split_tick'], spec['ticks']
    out = {}
    cpu = CpuLoopCohortEngine(arrays, n_flies)
    run_calls(cpu, ticks=split)
    gpu = make_gpu(arrays, n_flies)
    for k in range(n_flies):
        gpu.write_state(k, cpu.read_state(k))
    out['cpu_to_gpu'] = _verdict(_lockstep(cpu, gpu, split, ticks,
                                           _new_report(n_flies, ticks - split, gpu.device_name), tol))
    del cpu, gpu
    gpu = make_gpu(arrays, n_flies)
    run_calls(gpu, ticks=split)
    cpu = CpuLoopCohortEngine(arrays, n_flies)
    for k in range(n_flies):
        cpu.write_state(k, gpu.read_state(k))
    out['gpu_to_cpu'] = _verdict(_lockstep(cpu, gpu, split, ticks,
                                           _new_report(n_flies, ticks - split, gpu.device_name), tol))
    return out


# ------------------------------------------------------- identity checks
def batch_invariance(arrays, make, sizes=(8, 32)) -> List[str]:
    """Problems found (empty = byte-identical)."""
    problems, results = [], {}
    for B in sizes:
        engine = make(arrays, B)
        calls = run_calls(engine)
        results[B] = ([c.copy() for c in calls], [engine.read_state(k) for k in range(B)])
        del engine
    solo = make(arrays, 1)
    widest = max(sizes)
    drives = drive_phases(solo.n, widest)
    for k in range(widest):
        solo.reset([0])
        calls, t = [], 0
        while t < 200:
            calls.append(solo.step(drives[0 if t < 100 else 100][k:k + 1], 20))
            t += 20
        state = solo.read_state(0)
        for B, (bcalls, bstates) in results.items():
            if k >= B:
                continue
            if not all(np.array_equal(c[k:k + 1], s) for c, s in zip(bcalls, calls)):
                problems.append(f'B={B} fly {k}: spike counts differ from B=1')
            if not same_state(bstates[k], state):
                problems.append(f'B={B} fly {k}: state differs from B=1')
    if sum(int(c.sum()) for c in results[widest][0]) == 0:
        problems.append('void: workload did not spike')
    return problems


def cross_fly_isolation(arrays, make, n_flies=8, perturbed=3) -> List[str]:
    problems = []
    engine = make(arrays, n_flies)
    base_calls = [c.copy() for c in run_calls(engine)]
    base = [engine.read_state(k) for k in range(n_flies)]
    engine.reset(range(n_flies))
    pert_calls = run_calls(engine, perturb_fly=perturbed)
    pert = [engine.read_state(k) for k in range(n_flies)]
    for k in range(n_flies):
        if k == perturbed:
            continue
        if not all(np.array_equal(a[k], b[k]) for a, b in zip(base_calls, pert_calls)):
            problems.append(f'fly {k}: spike counts changed')
        if not same_state(base[k], pert[k]):
            problems.append(f'fly {k}: state changed')
    if same_state(base[perturbed], pert[perturbed]):
        problems.append(f'void: the perturbation did not reach fly {perturbed}')
    return problems


def resume(arrays, make, n_flies=8, split=120) -> List[str]:
    problems = []
    whole = make(arrays, n_flies)
    whole_calls = [c.copy() for c in run_calls(whole)]
    whole_states = [whole.read_state(k) for k in range(n_flies)]
    del whole
    first = make(arrays, n_flies)
    calls = [c.copy() for c in run_calls(first, ticks=split)]
    saved = [first.read_state(k) for k in range(n_flies)]
    del first
    second = make(arrays, n_flies)
    for k in range(n_flies):
        second.write_state(k, saved[k])
    calls += run_calls(second, t0=split)
    if len(calls) != len(whole_calls) or not all(np.array_equal(a, b) for a, b in zip(whole_calls, calls)):
        problems.append('resumed spike counts differ from the uninterrupted run')
    for k in range(n_flies):
        if not same_state(second.read_state(k), whole_states[k]):
            problems.append(f'fly {k}: resumed state differs')
    return problems


def refusals(arrays, make) -> List[str]:
    """Every bad drive, tick count and state is refused and changes nothing."""
    problems = []
    engine = make(arrays, 2)
    run_calls(engine, ticks=40)
    good = engine.read_state(0)
    engine.step(contract_drive(engine.n, 2, 0), 7)
    snap = lambda: [engine.read_state(k) for k in range(2)]   # noqa: E731
    before = snap()

    def expect_refused(label, call):
        try:
            call()
        except ValueError:
            pass
        except Exception as exc:   # refused, but not with the contract's ValueError
            problems.append(f'{label}: raised {type(exc).__name__}, not ValueError')
        else:
            problems.append(f'{label}: accepted')
            return
        if not all(same_state(x, y) for x, y in zip(before, snap())):
            problems.append(f'{label}: refused but live state changed')

    nan_drive = contract_drive(engine.n, 2, 0)
    nan_drive[1, 0] = np.nan
    expect_refused('NaN drive in one fly', lambda: engine.step(nan_drive, 10))
    for ticks in (1.5, 0, -1, True, '3'):
        expect_refused(f'ticks={ticks!r}', lambda t=ticks: engine.step(contract_drive(engine.n, 2, 0), t))
    bad_states = {f'dynamics={d!r}': dict(good, dynamics=d) for d in ('v2', 'v4', None)}
    for key in ('queue', 'v', 'cursor', 'active_flag'):
        bad_states[f'missing {key}'] = {k: v for k, v in good.items() if k != key}
    bad_states.update({
        'extra key': dict(good, graded_set_sha256='x'),
        'valid v with invalid g': dict(good, v=np.full_like(good['v'], -30.0), g=np.zeros(3, np.float32)),
        'short v': dict(good, v=good['v'][:-1]),
        'g wrong shape': dict(good, g=good['g'][0]),
        'v float64': dict(good, v=good['v'].astype(np.float64)),
        'refractory int32': dict(good, refractory=good['refractory'].astype(np.int32)),
        'active_flag bool': dict(good, active_flag=good['active_flag'].astype(bool)),
        'NaN in v': dict(good, v=np.where(np.arange(engine.n) == 3, np.nan, good['v']).astype(np.float32)),
        'negative cursor': dict(good, cursor=-1),
        'NaN sim_ms': dict(good, sim_ms=float('nan')),
        'queue_count beyond n': dict(good, queue_count=np.full_like(good['queue_count'], engine.n + 1)),
    })
    for label, state in bad_states.items():
        expect_refused(label, lambda s=state: engine.write_state(1, s))
    expect_refused('fly out of range', lambda: engine.write_state(2, good))
    engine.write_state(1, good)
    if not same_state(engine.read_state(1), good):
        problems.append('a valid state did not round-trip')
    return problems


# ------------------------------------------------------------------ runner
def run_contract(engine: str = 'gpu', arrays: Optional[dict] = None, device=None,
                 evidence_dir: Optional[os.PathLike] = None) -> Iterator[dict]:
    """Yield ``{name, passed, detail}`` for every preregistered check."""
    if engine not in ('gpu', 'cpu'):
        raise ValueError("engine must be 'gpu' or 'cpu'")
    contract = load_contract()
    make = gpu_factory(device) if engine == 'gpu' else cpu_factory()
    arrays = real_graph() if arrays is None else arrays

    def save(name, obj):
        if evidence_dir:
            Path(evidence_dir).mkdir(parents=True, exist_ok=True)
            (Path(evidence_dir) / name).write_text(json.dumps(obj, indent=2))

    if engine == 'gpu':
        report = reference_comparison(arrays, make, contract)
        if report['verdict'] == 'FAIL':
            report['descriptive'] = descriptive_full_run(
                arrays, make, report['n_flies'], report['ticks'])
        save('reference_comparison.json', report)
        detail = (f"first divergence {report['first_divergence']}; worst before it {report['worst']}; "
                  f"spikes per fly {report['spikes_per_fly']}")
        if 'descriptive' in report:
            detail += f"; {json.dumps(report['descriptive'])}"
        yield dict(name='cpu_reference', passed=report['verdict'] == 'PASS', detail=detail)

        cross = cross_engine_restore(arrays, make, contract)
        save('cross_engine_restore.json', cross)
        yield dict(name='cross_engine_restore',
                   passed=all(r['verdict'] == 'PASS' for r in cross.values()),
                   detail='; '.join(f"{d}: {r['verdict']} first_divergence={r['first_divergence']} "
                                    f"worst={r['worst']}" for d, r in cross.items()))

    for name, check in (('batch_invariance', batch_invariance),
                        ('cross_fly_isolation', cross_fly_isolation),
                        ('resume', resume), ('refusals', refusals)):
        problems = check(arrays, make)
        yield dict(name=name, passed=not problems,
                   detail='byte-identical' if not problems and name != 'refusals'
                   else ('all refused atomically' if not problems else '; '.join(problems[:20])))
