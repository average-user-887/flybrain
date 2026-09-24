"""Host benchmark: how fast does each part of the simulation run on this machine?

Runs the same fixed, seeded workloads on any host so results from different
machines (e.g. the laptop and the Ryzen workstation) can be compared directly,
and so a GPU backend has a CPU baseline to beat.

Stages (each is skipped, with the reason recorded, when its inputs are missing):

* ``brain-synthetic`` -- the LIF kernel on a seeded random graph sized like
  MaleCNS v1.0 (166,700 neurons, ~25.6M edges). Needs no dataset, so it runs
  identically everywhere.
* ``brain-synthetic-cuda`` / ``brain-malecns-cuda`` -- the same workloads on
  the GPU backend (``brainlab.cuda_engine``), skipped without a CUDA device.
* ``brain-malecns``   -- the same kernel on the pinned MaleCNS graph with the v3
  transmitter policy, if the dataset is present.
* ``body``            -- FlyGym/MuJoCo physics alone (FlyGym 2.1.0 required).
* ``daemon-<backend>`` -- ``ContinuousExperimentRunner.step_once`` for the
  modular controller and, with the dataset, the connectome backend.

The headline number of every stage is ``sim_s_per_wall_s`` (1.0 = real time).

    python scripts/benchmark.py --out runs/bench-$(hostname).json
    python scripts/benchmark.py --quick          # smaller workloads, smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MALECNS_NEURONS = 166_700
MALECNS_EDGES = 25_582_938
STEP_MS = 2.0


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def _windows_ram_gib():
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / 2**30, 1)
    except Exception:
        return None


def host_facts() -> dict:
    facts = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'cpu_count': os.cpu_count(),
        'git_sha': _run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
        'git_dirty': bool(_run(['git', '-C', str(ROOT), 'status', '--porcelain'])),
    }
    lscpu = _run(['lscpu']) or ''
    for line in lscpu.splitlines():
        if line.startswith('Model name:'):
            facts['cpu_model'] = line.split(':', 1)[1].strip()
    if not facts.get('cpu_model'):
        facts['cpu_model'] = platform.processor() or None
    try:
        with open('/proc/meminfo') as fh:
            facts['ram_gib'] = round(int(fh.readline().split()[1]) / 2**20, 1)
    except OSError:
        facts['ram_gib'] = _windows_ram_gib()
    facts['nvidia_smi'] = _run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                '--format=csv,noheader'])
    for mod in ('numpy', 'numba', 'mujoco', 'flygym', 'cupy'):
        try:
            facts[f'{mod}_version'] = __import__(mod).__version__
        except Exception:
            facts[f'{mod}_version'] = None
    try:
        from brainlab.cuda_engine import numba_cuda_available
        from brainlab.cupy_engine import cupy_available
        facts['numba_cuda_available'] = numba_cuda_available()
        facts['cupy_gpu_available'] = cupy_available()
    except Exception as error:
        facts['numba_cuda_available'] = f'no ({type(error).__name__})'
    return facts


def synthetic_malecns_like(n: int, edges: int, seed: int) -> dict:
    """Seeded CSR graph with MaleCNS-like size and a mixed-sign weight scale.

    Not biology: a fixed workload so hosts and backends can be compared.
    """
    rng = np.random.default_rng(seed)
    degree = rng.poisson(edges / n, size=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(degree, out=ptr[1:])
    m = int(ptr[-1])
    post = rng.integers(0, n, size=m, dtype=np.int32)
    synapses = rng.geometric(0.3, size=m).astype(np.float32)
    sign = np.where(rng.random(m) < 0.3, -1.0, 1.0).astype(np.float32)
    weight = np.ascontiguousarray(synapses * sign * np.float32(0.275))
    ids = np.arange(n, dtype=np.int64)
    return dict(ptr=ptr, post=post, weight=weight, ids=ids)


def time_brain(arrays: dict, *, sim_ms: float, drive_fraction: float, drive: float,
               seed: int, dynamics: str = 'v3', backend: str = 'cpu') -> dict:
    from brainlab.brain import Brain
    brain = Brain(arrays=arrays, validate=False, dynamics=dynamics, backend=backend)
    rng = np.random.default_rng(seed)
    currents = np.zeros(brain.n, dtype=np.float32)
    driven = rng.choice(brain.n, size=max(1, int(brain.n * drive_fraction)), replace=False)
    currents[driven] = drive

    compile_clock = time.perf_counter()
    brain.step(currents, STEP_MS)           # JIT compile (or load the numba cache)
    compile_s = time.perf_counter() - compile_clock

    steps = int(round(sim_ms / STEP_MS))
    spikes = 0
    kernel_s = 0.0
    wall = time.perf_counter()
    for _ in range(steps):
        counts, elapsed = brain.step(currents, STEP_MS)
        spikes += int(counts.sum())
        kernel_s += elapsed
    wall = time.perf_counter() - wall
    sim_s = steps * STEP_MS / 1000.0
    return dict(backend=backend, dynamics=dynamics, neurons=int(brain.n), edges=int(len(brain.post)),
                driven_neurons=int(len(driven)), drive=drive, sim_s=sim_s, wall_s=round(wall, 3),
                kernel_s=round(kernel_s, 3), first_step_s=round(compile_s, 3),
                sim_s_per_wall_s=round(sim_s / wall, 4),
                network_spikes_per_sim_s=round(spikes / sim_s, 1),
                active_neurons_end=int(brain.nactive[0]))


def stage_brain_synthetic(args, backend: str = 'cpu') -> dict:
    if backend == 'cuda':
        from brainlab.cuda_engine import cuda_available
        if not cuda_available():
            return {'skipped': 'no CUDA device visible to numba'}
    n = MALECNS_NEURONS if not args.quick else 20_000
    edges = MALECNS_EDGES if not args.quick else 20_000 * 153
    clock = time.perf_counter()
    arrays = synthetic_malecns_like(n, edges, seed=args.seed)
    build_s = time.perf_counter() - clock
    result = time_brain(arrays, sim_ms=args.brain_ms, drive_fraction=0.02, drive=20.0, seed=args.seed,
                        backend=backend)
    result['graph'] = f'synthetic seed={args.seed}'
    result['graph_build_s'] = round(build_s, 2)
    return result


def stage_brain_malecns(args, backend: str = 'cpu') -> dict:
    if backend == 'cuda':
        from brainlab.cuda_engine import cuda_available
        if not cuda_available():
            return {'skipped': 'no CUDA device visible to numba'}
    from brainlab.graph_identity import GraphUnavailable
    from brainlab.transmitter_policy import apply_to_shared
    from experiment_registry import SharedGraph
    try:
        clock = time.perf_counter()
        shared = SharedGraph.load()
        shared, _ = apply_to_shared(shared)
        load_s = time.perf_counter() - clock
    except (GraphUnavailable, FileNotFoundError, OSError) as error:
        return {'skipped': f'MaleCNS graph not available: {error}'}
    result = time_brain(shared.arrays, sim_ms=args.brain_ms, drive_fraction=0.02, drive=20.0,
                        seed=args.seed, backend=backend)
    result['graph'] = shared.identity.graph_sha256
    result['graph_load_s'] = round(load_s, 2)
    return result


def stage_body(args) -> dict:
    try:
        from neurofly_body.flygym_body import FlyGymBody
        body = FlyGymBody()
    except Exception as error:
        return {'skipped': f'FlyGym body unavailable: {type(error).__name__}: {error}'}
    try:
        body.reset(seed=args.seed)
        substeps = int(round(STEP_MS / 1000.0 / body.physics_dt_s))
        steps = int(round(args.body_ms / STEP_MS))
        wall = time.perf_counter()
        for _ in range(steps):
            body.step((1.0, 1.0), substeps)
        wall = time.perf_counter() - wall
        sim_s = steps * STEP_MS / 1000.0
        return dict(sim_s=sim_s, wall_s=round(wall, 3), physics_dt_s=body.physics_dt_s,
                    sim_s_per_wall_s=round(sim_s / wall, 4))
    finally:
        body.close()


def stage_daemon(args, backend: str) -> dict:
    import tempfile
    from neurofly_daemon import ContinuousExperimentRunner
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            clock = time.perf_counter()
            runner = ContinuousExperimentRunner(initial_paradigm='open-arena', sim_speed=1.0,
                                                checkpoint_interval=1e9, output_dir=Path(out_dir),
                                                backend=backend)
            init_s = time.perf_counter() - clock
        except Exception as error:
            return {'skipped': f'{backend} runner unavailable: {type(error).__name__}: {error}'}
        steps = int(round(args.daemon_s / runner.dt))
        with runner.lock:
            runner.step_once(publish=False)  # warm-up / JIT
            wall = time.perf_counter()
            for _ in range(steps):
                runner.step_once(publish=False)
            wall = time.perf_counter() - wall
        sim_s = steps * runner.dt
        return dict(backend=backend, paradigm='open-arena', dt_s=runner.dt, sim_s=round(sim_s, 3),
                    wall_s=round(wall, 3), init_s=round(init_s, 2),
                    sim_s_per_wall_s=round(sim_s / wall, 4))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, help='write the JSON receipt here')
    parser.add_argument('--stages', default='brain-synthetic,brain-synthetic-cuda,brain-malecns,brain-malecns-cuda,'
                                'body,daemon-modular,daemon-connectome',
                        help='comma-separated stages to run')
    parser.add_argument('--brain-ms', type=float, default=500.0, help='simulated ms per brain stage')
    parser.add_argument('--body-ms', type=float, default=1000.0, help='simulated ms for the body stage')
    parser.add_argument('--daemon-s', type=float, default=5.0, help='simulated s per daemon stage')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--quick', action='store_true', help='small workloads for a smoke test')
    args = parser.parse_args(argv)
    if args.quick:
        args.brain_ms, args.body_ms, args.daemon_s = 20.0, 40.0, 0.2

    stages = {
        'brain-synthetic': stage_brain_synthetic,
        'brain-synthetic-cuda': lambda a: stage_brain_synthetic(a, 'cuda'),
        'brain-malecns': stage_brain_malecns,
        'brain-malecns-cuda': lambda a: stage_brain_malecns(a, 'cuda'),
        'body': stage_body,
        'daemon-modular': lambda a: stage_daemon(a, 'modular'),
        'daemon-connectome': lambda a: stage_daemon(a, 'connectome-fixed'),
    }
    receipt = {'schema': 'neurofly.host-benchmark.v1', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               'host': host_facts(), 'stages': {}}
    for name in [s.strip() for s in args.stages.split(',') if s.strip()]:
        if name not in stages:
            parser.error(f'unknown stage {name!r}; choose from {sorted(stages)}')
        print(f'[bench] {name} ...', flush=True)
        try:
            result = stages[name](args)
        except Exception as error:
            result = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        receipt['stages'][name] = result
        headline = result.get('sim_s_per_wall_s', result.get('skipped') or result.get('error'))
        print(f'[bench] {name}: {headline}', flush=True)
    receipt['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')

    text = json.dumps(receipt, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
        print(f'[bench] wrote {args.out}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    # Process-wide defaults for the daemon stages; set only when run as a script
    # so importing this module (e.g. from tests) never changes the environment.
    os.environ.setdefault('NEUROFLY_LIF_DYNAMICS', 'v3')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    raise SystemExit(main())
