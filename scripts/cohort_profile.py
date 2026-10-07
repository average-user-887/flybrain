"""Per-tick, per-phase profile of the GPU cohort kernel on the real cohort workload.

Runs the actual ``neurofly cohort run`` loop (real MaleCNS graph, optomotor
assay, Arena, encoder, decoder, recording and checkpoints) with the GPU engine
swapped for an instrumented copy of the SAME kernel source.  Two modes:

* ``timing``: thread 0 reads ``%globaltimer`` after every grid barrier, so each
  tick is split into phase 1 (integrate/threshold/enqueue), phase 2a (stamp due
  spikers and their targets) and phase 2b+3 (per-target delivery scan, reset).
  No counters, so the phase times are not perturbed by atomics.
* ``counters``: per tick, due spikers, due out-edges, marked receivers, incoming
  edges scanned by the marked receivers, arrivals actually applied and the
  largest single-target scan.  Its phase times are perturbed and not reported.

The whole-cohort wall is split into engine.step, the rest of Cohort.step
(encode, decode, Arena, recording) and checkpoints.  The kernel's arithmetic is
untouched: instrumentation only adds stores to a separate profile buffer.

    CUDA_VISIBLE_DEVICES=<1660 uuid> CUDA_DEVICE_ORDER=PCI_BUS_ID \
        python scripts/cohort_profile.py --mode timing --store /scratch/new-dir --out prof.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

NSLOT = 12   # per-tick profile words
T0, T1, T2, T3, DUE_SPK, DUE_EDGES, MARKED, SCANNED, APPLIED, MAX_SCAN = range(10)
FIELDS = ['call', 'tick', 'phase1_us', 'phase2a_us', 'phase2b3_us', 'tick_us', 'due_spikers', 'due_edges',
          'marked_receivers', 'in_edges_scanned', 'arrivals_applied', 'max_target_scan']

_TIMER = ('if (prof_mode == 1 && tid == 0) { unsigned long long _t; '
          'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); prof[(long long)step * 12 + SLOT] = _t; }')


def _sub(src: str, old: str, new: str, count: int = 1) -> str:
    if src.count(old) != count:
        raise SystemExit(f'profile instrumentation anchor {old!r} found {src.count(old)} times, expected {count}')
    return src.replace(old, new)


def instrument(source: str) -> str:
    """Add a profile buffer to the cohort kernel without touching its arithmetic."""
    s = _sub(source, 'const double g_unit_exc, const double g_unit_inh)',
             'const double g_unit_exc, const double g_unit_inh,\n'
             '    unsigned long long* prof, const int prof_mode)')
    s = _sub(s, 'for (int step = 0; step < steps; ++step) {',
             'for (int step = 0; step < steps; ++step) {\n        ' + _TIMER.replace('SLOT', '0'))
    parts = s.split('grid.sync();')
    if len(parts) != 4:
        raise SystemExit(f'expected 3 grid.sync() in the kernel, found {len(parts) - 1}')
    s = parts[0]
    for k in (1, 2, 3):
        s += 'grid.sync();\n        ' + _TIMER.replace('SLOT', str(k)) + parts[k]
    if 'in_pre' not in s:      # counters need the full-scan kernel's incoming-edge sources
        return s
    # Counters (mode 2).
    s = _sub(s, 'const int pre = qf[q];',
             'const int pre = qf[q];\n'
             '                if (prof_mode == 2 && lane == 0) {\n'
             '                    atomicAdd(&prof[(long long)step * 12 + 4], 1ULL);\n'
             '                    atomicAdd(&prof[(long long)step * 12 + 5], (unsigned long long)(ptr[pre + 1] - ptr[pre]));\n'
             '                }')
    s = _sub(s, 'if (recv_stamp[x] == stamp) {',
             'if (recv_stamp[x] == stamp) {\n'
             '                if (prof_mode == 2) {\n'
             '                    const long long _deg = in_ptr[i + 1] - in_ptr[i];\n'
             '                    unsigned long long _hit = 0;\n'
             '                    for (long long e = in_ptr[i]; e < in_ptr[i + 1]; ++e)\n'
             '                        if (due_stamp[f * n + in_pre[e]] == stamp) ++_hit;\n'
             '                    atomicAdd(&prof[(long long)step * 12 + 6], 1ULL);\n'
             '                    atomicAdd(&prof[(long long)step * 12 + 7], (unsigned long long)_deg);\n'
             '                    atomicAdd(&prof[(long long)step * 12 + 8], _hit);\n'
             '                    atomicMax(&prof[(long long)step * 12 + 9], (unsigned long long)_deg);\n'
             '                }')
    return s


def profiling_engine_class(mode: int, sink: list, kernel_source: str = None):
    import cupy as cp
    from brainlab.cohort import gpu as G
    from brainlab.cuda_engine import THREADS_PER_BLOCK

    class ProfilingEngine(G.GpuCohortEngine):
        def __init__(self, arrays, n_flies, **kw):
            super().__init__(arrays, n_flies, **kw)
            src = instrument(kernel_source or G._SOURCE)
            with self.device:
                self.kernel = cp.RawKernel(src, 'advance_v3_cohort', options=('-std=c++17',),
                                           enable_cooperative_groups=True)
                self.kernel.compile()
            self._blocks = None
            self.engine_s = 0.0

        def step(self, drive, ticks):
            drive = self._check_step_args(drive, ticks)
            ticks = int(ticks)
            clock = time.perf_counter()
            with self.device:
                if self._last_drive is None or not np.array_equal(self._last_drive, drive):
                    self.d_drive.set(drive)
                    self._last_drive = drive.copy()
                self.d_counts.fill(0)
                self.d_cursor.set(np.asarray(self.cursor, dtype=np.int64))
                prof = cp.zeros(ticks * NSLOT, dtype=cp.uint64)
                args = (self.d_ptr, self.d_post, self.d_v, self.d_g,
                        self.d_refractory, self.d_drive, self.d_queue, self.d_queue_count,
                        self.d_counts, self.d_active_flag, self.d_spiked, self.d_cursor,
                        self.d_in_ptr, self.d_in_pre, self.d_in_w, self.d_due, self.d_recv,
                        np.int64(self.n), np.int32(self.n_flies), np.int32(self.delay_slots),
                        np.int32(ticks), np.float64(self.dt), np.float64(self.e_inh),
                        np.float64(self.g_unit_exc), np.float64(self.g_unit_inh), prof, np.int32(mode))
                self.kernel((self._grid_blocks(),), (THREADS_PER_BLOCK,), args)
                counts = self.d_counts.get()
                sink.append(prof.get().reshape(ticks, NSLOT))
            self.engine_s += time.perf_counter() - clock
            for k in range(self.n_flies):
                self.cursor[k] += ticks
                self.total_spikes[k] += int(counts[k].sum())
                self.sim_ms[k] += ticks * self.dt
            return counts

    return ProfilingEngine


def guard_device():
    import cupy as cp
    names = [cp.cuda.runtime.getDeviceProperties(k)['name'].decode()
             for k in range(cp.cuda.runtime.getDeviceCount())]
    if len(names) != 1 or '1660' not in names[0]:
        raise SystemExit(f'refusing: visible CUDA devices {names}; expose only the GTX 1660 Ti')
    return names[0]


# ----------------------------------------------------------------- identity
RAW = ('d_v', 'd_g', 'd_refractory', 'd_queue_count', 'd_counts', 'd_active_flag', 'd_spiked', 'd_due', 'd_recv')


def baseline_module(ref: str):
    """The GPU engine module at git ``ref`` (the strict full-scan kernel), importable beside the live one."""
    import importlib.util
    import subprocess
    import tempfile
    src = subprocess.run(['git', '-C', str(ROOT), 'show', f'{ref}:brainlab/cohort/gpu.py'], check=True,
                         capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp(prefix='cohort-gpu-baseline-')) / 'gpu_baseline.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location('brainlab.cohort.gpu_baseline', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _diff(a, b, tag: str) -> list:
    """Byte-level differences between two engines' live state (canonical + raw device arrays)."""
    out = []
    for k in range(a.n_flies):
        sa, sb = a.read_state(k), b.read_state(k)
        for key in sa:
            x, y = sa[key], sb[key]
            if isinstance(x, np.ndarray):
                if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                    out.append(f'{tag} fly {k}: {key}')
            elif x != y:
                out.append(f'{tag} fly {k}: {key}')
    for name in RAW:
        if getattr(a, name).get().tobytes() != getattr(b, name).get().tobytes():
            out.append(f'{tag}: raw {name}')
    return out


def identity(ref: str, out: Path) -> int:
    """Baseline (``ref``) and live kernel on the same device and inputs: byte-identical, check by check."""
    from brainlab.cohort import contract as C
    from brainlab.cohort import gpu as G
    base_mod = baseline_module(ref)
    arrays = C.real_graph()
    mk = {'baseline': lambda B: base_mod.GpuCohortEngine(arrays, B, validate=False),
          'candidate': lambda B: G.GpuCohortEngine(arrays, B, validate=False)}
    report = {'baseline_ref': ref, 'device': guard_device(), 'checks': {}}

    def pair(B):
        return mk['baseline'](B), mk['candidate'](B)

    def lock(engines, fn, tag):
        """Run fn(engine) on both, compare returned spikes and state after it."""
        r = [fn(e) for e in engines]
        probs = []
        if r[0] is not None:
            ra = r[0] if isinstance(r[0], list) else [r[0]]
            rb = r[1] if isinstance(r[1], list) else [r[1]]
            if len(ra) != len(rb) or any(x.tobytes() != y.tobytes() for x, y in zip(ra, rb)):
                probs.append(f'{tag}: spikes')
        return probs + _diff(engines[0], engines[1], tag)

    def record(name, probs, **extra):
        report['checks'][name] = dict(byte_identical=not probs, problems=probs[:20], **extra)
        print(f'[identity] {name}: {"BYTE-IDENTICAL" if not probs else "DIFFERS"} {probs[:3]}', flush=True)

    # 1. cpu_reference workload: B8, per-tick lockstep, contract drive phases (state compared every tick).
    contract = C.load_contract()
    B, ticks = 8, 200
    eng = pair(B)
    drives = C.drive_phases(eng[0].n, B)
    probs, spikes = [], 0
    for t in range(ticks):
        d = drives[max(k for k in drives if k <= t)]
        probs += lock(eng, lambda e: e.step(d, 1), f'tick {t}')
        spikes += int(eng[1].d_counts.sum().get())
    record('cpu_reference_workload', probs, ticks=ticks, n_flies=B, candidate_spikes=spikes)
    del eng

    # 2. cross_engine_restore: CPU state at the split written into both, then lockstep.
    cpu = C.cpu_factory()(arrays, B)
    split = int(contract['cross_engine_restore']['split_tick'])
    C.run_calls(cpu, ticks=split)
    eng = pair(B)
    for e in eng:
        for k in range(B):
            e.write_state(k, cpu.read_state(k))
    probs = _diff(eng[0], eng[1], 'after restore')
    for t0 in range(split, 200, 20):
        d = drives[max(k for k in drives if k <= t0)]
        probs += lock(eng, lambda e: e.step(d, 20), f'call at {t0}')
    record('cross_engine_restore_workload', probs, split=split)
    del eng, cpu

    # 3. batch invariance B1/8/32 (each B baseline vs candidate; candidate B1/B8 vs B32 slices).
    for B_ in (1, 8, 32):
        eng = pair(B_)
        probs = lock(eng, lambda e: [c.copy() for c in C.run_calls(e)], f'B{B_}')
        record(f'batch_B{B_}', probs)
        del eng
    # Cross-B invariance of the candidate itself is the contract's batch_invariance check (below).

    # 4. isolation: perturbed fly 3.
    eng = pair(8)
    probs = lock(eng, lambda e: [c.copy() for c in C.run_calls(e, perturb_fly=3)], 'perturbed')
    record('isolation_perturbed_run', probs)
    del eng

    # 5. resume: 120 ticks, state saved, fresh engines restored, continued.
    first = pair(8)
    probs = lock(first, lambda e: [c.copy() for c in C.run_calls(e, ticks=120)], 'first segment')
    saved = [first[0].read_state(k) for k in range(8)]
    del first
    second = pair(8)
    for e in second:
        for k in range(8):
            e.write_state(k, saved[k])
    probs += lock(second, lambda e: [c.copy() for c in C.run_calls(e, t0=120)], 'resumed segment')
    record('resume_split_120', probs)
    del second

    # 6. the full preregistered contract on the candidate (CPU reference and tolerances unchanged).
    results = list(C.run_contract('gpu', arrays=arrays, evidence_dir=out.parent / 'contract_candidate'))
    for r in results:
        print(f"[contract] {r['name']}: {'PASS' if r['passed'] else 'FAIL'} {r['detail'][:160]}", flush=True)
    report['contract_candidate'] = results
    report['all_byte_identical'] = all(c['byte_identical'] for c in report['checks'].values())
    report['contract_pass'] = all(r['passed'] for r in results)
    out.write_text(json.dumps(report, indent=2, default=str))
    return 0 if report['all_byte_identical'] and report['contract_pass'] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--mode', choices=('timing', 'counters', 'identity'), required=True)
    p.add_argument('--store', type=Path, help='new, empty cohort directory (scratch); timing/counters')
    p.add_argument('--out', type=Path, required=True, help='per-tick CSV, or the identity JSON report')
    p.add_argument('--baseline-ref', default='v0.5.0rc1', help='identity: git ref of the strict kernel')
    p.add_argument('--flies', type=int, default=8)
    p.add_argument('--seconds', type=float, default=2.0)
    p.add_argument('--checkpoint-every-ms', type=float, default=None)
    a = p.parse_args(argv)
    dev = guard_device()
    if a.mode == 'identity':
        a.out.parent.mkdir(parents=True, exist_ok=True)
        return identity(a.baseline_ref, a.out)
    if a.store is None:
        p.error('--store is required for timing/counters')
    from brainlab.cohort import runner
    mode = 1 if a.mode == 'timing' else 2
    sink: list = []
    engines: list = []
    Engine = profiling_engine_class(mode, sink)

    def make_engine(name, arrays, n_flies):
        assert name == 'gpu'
        engines.append(Engine(arrays, n_flies))
        return engines[-1]

    ck = {'s': 0.0, 'n': 0}
    orig_ckpt = runner.Cohort.checkpoint

    def checkpoint(self):
        clock = time.perf_counter()
        orig_ckpt(self)
        ck['s'] += time.perf_counter() - clock
        ck['n'] += 1

    runner.make_engine = make_engine
    runner.Cohort.checkpoint = checkpoint
    clock = time.perf_counter()
    rows = runner.run_cohort(a.store, flies=a.flies, seconds=a.seconds, engine='gpu',
                             checkpoint_every_ms=a.checkpoint_every_ms, progress=lambda m: None)
    total_s = time.perf_counter() - clock
    manifest = json.loads((a.store / 'cohort_manifest.json').read_text())
    seg = manifest['segments'][-1]
    eng = engines[0]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tick = 0
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for call, block in enumerate(sink):
            for r in block:
                row = dict(call=call, tick=tick)
                tick += 1
                if mode == 1:
                    row.update(phase1_us=(r[T1] - r[T0]) / 1e3, phase2a_us=(r[T2] - r[T1]) / 1e3,
                               phase2b3_us=(r[T3] - r[T2]) / 1e3, tick_us=(r[T3] - r[T0]) / 1e3)
                else:
                    row.update(due_spikers=int(r[DUE_SPK]), due_edges=int(r[DUE_EDGES]),
                               marked_receivers=int(r[MARKED]), in_edges_scanned=int(r[SCANNED]),
                               arrivals_applied=int(r[APPLIED]), max_target_scan=int(r[MAX_SCAN]))
                w.writerow(row)
    arr = np.concatenate(sink)
    summary = dict(mode=a.mode, device=dev, flies=a.flies, seconds=a.seconds, ticks_per_fly=int(len(arr)),
                   grid_blocks=eng._grid_blocks(), run_wall_s=seg['wall_s'], total_s=total_s,
                   graph_load_s=seg['graph_load_s'], setup_s=seg['setup_s'],
                   engine_step_s=eng.engine_s, checkpoint_s=ck['s'], checkpoints=ck['n'],
                   other_loop_s=seg['wall_s'] - eng.engine_s - ck['s'],
                   fly_steps_per_s=rows[0]['steps_per_s'] * a.flies if rows else None)
    if mode == 1:
        p1, p2a, p2b = ((arr[:, b] - arr[:, a_]).astype(np.float64) / 1e9 for a_, b in ((T0, T1), (T1, T2), (T2, T3)))
        tot = p1.sum() + p2a.sum() + p2b.sum()
        summary.update(phase1_s=p1.sum(), phase2a_s=p2a.sum(), phase2b3_s=p2b.sum(), in_kernel_s=tot,
                       phase2b3_share=p2b.sum() / tot, phase2a_share=p2a.sum() / tot, phase1_share=p1.sum() / tot)
    else:
        c = arr.astype(np.float64)
        summary.update(due_spikers=c[:, DUE_SPK].sum(), due_edges=c[:, DUE_EDGES].sum(),
                       marked_receivers=c[:, MARKED].sum(), in_edges_scanned=c[:, SCANNED].sum(),
                       arrivals_applied=c[:, APPLIED].sum(), max_target_scan=int(arr[:, MAX_SCAN].max()),
                       scanned_per_applied=c[:, SCANNED].sum() / max(c[:, APPLIED].sum(), 1),
                       mean_scanned_per_tick=c[:, SCANNED].mean(), mean_applied_per_tick=c[:, APPLIED].mean())
    a.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
