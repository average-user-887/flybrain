"""Measured cohort throughput: CPU loop, serial GPU and batched GPU.

Same graph (real MaleCNS, v3 policy) and the same inputs everywhere: the
preregistered tonic drive of ``brainlab/cohort/cohort_contract.json`` (the
tick-0 pattern for fly k), held for the whole run and stepped in calls of
``--call`` ticks.  Writes one raw CSV row per measured run; prints nothing
but what was measured.  No speed claim beyond these numbers.

    CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/cohort_bench.py --device 1660 \
        --out runs/cohort_bench.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FIELDS = ['config', 'engine', 'n_flies', 'rep', 'ticks', 'call_ticks', 'wall_s',
          'fly_steps_per_s', 'sim_s_per_wall_s_per_fly', 'startup_s', 'compile_s', 'upload_s',
          'state_alloc_s', 'warmup_s', 'peak_vram_bytes', 'engine_device_bytes', 'total_spikes',
          'device', 'graph_prepare_s']


def drive_for(n, flies):
    from brainlab.cohort.contract import contract_drive
    flies = list(flies)
    return np.ascontiguousarray(contract_drive(n, max(flies) + 1, 0)[np.asarray(flies)])


def vram_used(cp) -> int:
    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free)


def timed(engine, drive, ticks, call):
    spikes = 0
    clock = time.perf_counter()
    for _ in range(ticks // call):
        spikes += int(engine.step(drive, call).sum())
    return time.perf_counter() - clock, spikes


def bench_cpu(arrays, B, ticks, call, reps, graph_s):
    from brainlab.cohort.api import CpuLoopCohortEngine
    clock = time.perf_counter()
    engine = CpuLoopCohortEngine(arrays, B)
    startup = time.perf_counter() - clock
    drive = drive_for(engine.n, range(B))
    clock = time.perf_counter()
    engine.step(drive, call)                        # numba already compiled at import/cache
    warm = time.perf_counter() - clock
    rows = []
    for rep in range(reps):
        engine.reset(range(B))
        wall, spikes = timed(engine, drive, ticks, call)
        rows.append(dict(config=f'cpu_B{B}', engine='cpu-loop-v3', n_flies=B, rep=rep, ticks=ticks,
                         call_ticks=call, wall_s=wall, fly_steps_per_s=B * ticks / wall,
                         sim_s_per_wall_s_per_fly=ticks * 1e-4 / wall, startup_s=startup,
                         warmup_s=warm, total_spikes=spikes, device=platform.processor() or 'cpu',
                         graph_prepare_s=graph_s))
    return rows


def _gpu_engine(arrays, B, device, limit):
    import cupy as cp
    from brainlab.cohort import gpu as G
    import gc
    G._GRAPH_CACHE.clear()                          # measure a real upload every time
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    clock = time.perf_counter()
    engine = G.GpuCohortEngine(arrays, B, device=device, validate=False)
    startup = time.perf_counter() - clock
    if '1660' not in engine.device_name and device == '1660':
        raise SystemExit(f'refusing: selected {engine.device_name}')
    if engine.device_bytes() > limit:
        raise SystemExit(f'refusing: {engine.device_bytes()} bytes exceeds the {limit} limit')
    return engine, startup


def bench_gpu_batched(arrays, B, ticks, call, reps, device, limit, graph_s):
    import cupy as cp
    engine, startup = _gpu_engine(arrays, B, device, limit)
    drive = drive_for(engine.n, range(B))
    with engine.device:
        clock = time.perf_counter()
        engine.step(drive, call)                    # first launch: module load, occupancy query
        warm = time.perf_counter() - clock
        peak = vram_used(cp)
        rows = []
        for rep in range(reps):
            engine.reset(range(B))
            wall, spikes = timed(engine, drive, ticks, call)
            peak = max(peak, vram_used(cp))
            if peak > limit:
                raise SystemExit(f'device memory {peak} exceeded the {limit} limit')
            t = engine.timings
            rows.append(dict(config=f'gpu_batched_B{B}', engine=engine.backend_id, n_flies=B, rep=rep,
                             ticks=ticks, call_ticks=call, wall_s=wall, fly_steps_per_s=B * ticks / wall,
                             sim_s_per_wall_s_per_fly=ticks * 1e-4 / wall, startup_s=startup,
                             compile_s=t['compile_s'], upload_s=t['graph_upload_s'],
                             state_alloc_s=t['state_alloc_s'], warmup_s=warm, peak_vram_bytes=peak,
                             engine_device_bytes=engine.device_bytes(), total_spikes=spikes,
                             device=engine.device_name, graph_prepare_s=graph_s))
    del engine
    return rows


def bench_gpu_serial(arrays, n_serial, ticks, call, reps, device, limit, graph_s):
    """One B=1 engine run once per fly, one after the other."""
    import cupy as cp
    engine, startup = _gpu_engine(arrays, 1, device, limit)
    with engine.device:
        clock = time.perf_counter()
        engine.step(drive_for(engine.n, [0]), call)
        warm = time.perf_counter() - clock
        peak = vram_used(cp)
        rows = []
        for rep in range(reps):
            wall, spikes = 0.0, 0
            for k in range(n_serial):
                engine.reset([0])
                w, s = timed(engine, drive_for(engine.n, [k]), ticks, call)
                wall, spikes = wall + w, spikes + s
            peak = max(peak, vram_used(cp))
            t = engine.timings
            rows.append(dict(config=f'gpu_serial_{n_serial}xB1', engine=engine.backend_id,
                             n_flies=n_serial, rep=rep, ticks=ticks, call_ticks=call, wall_s=wall,
                             fly_steps_per_s=n_serial * ticks / wall,
                             sim_s_per_wall_s_per_fly=ticks * 1e-4 / wall, startup_s=startup,
                             compile_s=t['compile_s'], upload_s=t['graph_upload_s'],
                             state_alloc_s=t['state_alloc_s'], warmup_s=warm, peak_vram_bytes=peak,
                             engine_device_bytes=engine.device_bytes(), total_spikes=spikes,
                             device=engine.device_name, graph_prepare_s=graph_s))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--device', default=os.environ.get('NEUROFLY_COHORT_GPU'))
    p.add_argument('--ticks', type=int, default=2000, help='ticks per measured run (0.1 ms each)')
    p.add_argument('--call', type=int, default=20, help='ticks per step call')
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--cpu-batches', default='1,8')
    p.add_argument('--gpu-batches', default='1,8,32')
    p.add_argument('--serial', type=int, default=8)
    p.add_argument('--vram-limit', type=float, default=4.5e9)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    for B in [int(x) for x in a.gpu_batches.split(',') if x]:
        if B not in (1, 8, 32):
            raise SystemExit('B is limited to 1, 8 and 32')

    from brainlab.cohort.contract import real_graph
    clock = time.perf_counter()
    arrays = real_graph()
    graph_s = time.perf_counter() - clock
    print(f'[bench] graph prepared in {graph_s:.1f} s: {len(arrays["ids"]):,} neurons, '
          f'{len(arrays["post"]):,} edges', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()

        def emit(rows):
            for row in rows:
                writer.writerow(row)
                print(f"[bench] {row['config']} rep {row['rep']}: {row['wall_s']:.2f} s, "
                      f"{row['fly_steps_per_s']:.0f} fly-steps/s", flush=True)
            fh.flush()

        for B in [int(x) for x in a.gpu_batches.split(',') if x]:
            emit(bench_gpu_batched(arrays, B, a.ticks, a.call, a.reps, a.device, a.vram_limit, graph_s))
        if a.serial:
            emit(bench_gpu_serial(arrays, a.serial, a.ticks, a.call, a.reps, a.device, a.vram_limit,
                                  graph_s))
        for B in [int(x) for x in a.cpu_batches.split(',') if x]:
            emit(bench_cpu(arrays, B, a.ticks, a.call, a.reps, graph_s))
    meta = dict(ticks=a.ticks, call=a.call, reps=a.reps, graph_prepare_s=graph_s,
                drive='contract tick-0 tonic pattern per fly, held constant', python=sys.version)
    a.out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
