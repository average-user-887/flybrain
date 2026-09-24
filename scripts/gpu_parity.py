"""GPU vs CPU accuracy check for the v3 LIF kernel.

Runs the same seeded workload three times:

* ``cpu``      -- the numba reference kernel;
* ``gpu``      -- the CUDA backend (``Brain(backend='cuda')``);
* ``cpu-perturbed`` -- the CPU kernel again with every initial membrane
  potential nudged by float32 epsilon-scale noise (``--perturb-mV``).

The perturbed run is the yardstick: a recurrent spiking network amplifies
any rounding difference, so GPU-vs-CPU divergence should be judged against
how far the CPU diverges from ITSELF under a perturbation of the same size
as float32 rounding.  If the GPU is no further from the CPU than that, its
difference is within the model's own numerical sensitivity.

Reported per comparison: total-spike relative difference, the first 2 ms
window whose spike vector differs, the Pearson correlation of per-neuron
rates, the mean absolute per-neuron rate difference, and the same numbers
for the population-rate time series.

    python scripts/gpu_parity.py --graph malecns --ms 1000 --out runs/gpu-parity.json
    python scripts/gpu_parity.py --graph synthetic --ms 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark import MALECNS_EDGES, MALECNS_NEURONS, STEP_MS, synthetic_malecns_like  # noqa: E402


def load_graph(kind: str, seed: int) -> tuple[dict, str]:
    if kind == 'synthetic':
        return synthetic_malecns_like(MALECNS_NEURONS, MALECNS_EDGES, seed), f'synthetic seed={seed}'
    if kind == 'small':
        return synthetic_malecns_like(5_000, 5_000 * 60, seed), f'small synthetic seed={seed}'
    from brainlab.transmitter_policy import apply_to_shared
    from experiment_registry import SharedGraph
    shared, _ = apply_to_shared(SharedGraph.load())
    return shared.arrays, shared.identity.graph_sha256


def run(arrays: dict, *, backend: str, ms: float, seed: int, perturb_mV: float = 0.0) -> dict:
    from brainlab.brain import Brain
    brain = Brain(arrays=arrays, validate=False, dynamics='v3', backend=backend)
    if perturb_mV:
        rng = np.random.default_rng(seed + 1)
        state = brain.snapshot_state()
        state['v'] = (state['v'] + rng.uniform(-perturb_mV, perturb_mV, brain.n)).astype(np.float32)
        brain.restore_state(state)
    rng = np.random.default_rng(seed)
    drive = np.zeros(brain.n, dtype=np.float32)
    drive[rng.choice(brain.n, size=max(1, brain.n // 50), replace=False)] = 20.0
    windows = int(round(ms / STEP_MS))
    trains = np.zeros((windows, brain.n), dtype=np.int16)
    clock = time.perf_counter()
    for w in range(windows):
        counts, _ = brain.step(drive, STEP_MS)
        trains[w] = counts
    return dict(trains=trains, wall_s=time.perf_counter() - clock)


def compare(ref: np.ndarray, other: np.ndarray, ms: float) -> dict:
    sec = ms / 1000.0
    total_ref, total_other = int(ref.sum()), int(other.sum())
    differs = np.flatnonzero((ref != other).any(axis=1))
    rate_ref = ref.sum(axis=0) / sec
    rate_other = other.sum(axis=0) / sec
    active = (rate_ref > 0) | (rate_other > 0)
    pop_ref = ref.sum(axis=1).astype(float)
    pop_other = other.sum(axis=1).astype(float)

    def corr(a, b):  # None when undefined (a constant series), keeping the JSON valid
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None

    return dict(
        total_spikes=[total_ref, total_other],
        total_rel_diff=round((total_other - total_ref) / max(1, total_ref), 6),
        identical=bool(differs.size == 0),
        first_differing_window_ms=None if differs.size == 0 else float(differs[0] * STEP_MS),
        neurons_active=int(active.sum()),
        rate_pearson=corr(rate_ref[active], rate_other[active]) if active.any() else None,
        rate_mean_abs_diff_hz=float(np.abs(rate_ref - rate_other)[active].mean()) if active.any() else 0.0,
        population_pearson=corr(pop_ref, pop_other),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--graph', choices=('malecns', 'synthetic', 'small'), default='malecns')
    parser.add_argument('--ms', type=float, default=1000.0)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--perturb-mV', type=float, default=1e-5,
                        help='initial-voltage noise for the CPU self-divergence yardstick '
                             '(float32 epsilon at -50 mV is ~4e-6 mV)')
    parser.add_argument('--skip-gpu', action='store_true', help='yardstick only (no CUDA device needed)')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)

    arrays, graph_id = load_graph(args.graph, args.seed)
    print(f'[parity] graph {graph_id}: {len(arrays["ids"]):,} neurons, {len(arrays["post"]):,} edges', flush=True)
    cpu = run(arrays, backend='cpu', ms=args.ms, seed=args.seed)
    print(f'[parity] cpu done in {cpu["wall_s"]:.1f} s', flush=True)
    pert = run(arrays, backend='cpu', ms=args.ms, seed=args.seed, perturb_mV=args.perturb_mV)
    print(f'[parity] cpu-perturbed done in {pert["wall_s"]:.1f} s', flush=True)
    report = {'schema': 'neurofly.gpu-parity.v1', 'graph': graph_id, 'ms': args.ms, 'seed': args.seed,
              'perturb_mV': args.perturb_mV,
              'cpu_vs_cpu_perturbed': compare(cpu['trains'], pert['trains'], args.ms)}
    if not args.skip_gpu:
        gpu = run(arrays, backend='cuda', ms=args.ms, seed=args.seed)
        print(f'[parity] gpu done in {gpu["wall_s"]:.1f} s', flush=True)
        report['cpu_vs_gpu'] = compare(cpu['trains'], gpu['trains'], args.ms)
        gpu2 = run(arrays, backend='cuda', ms=args.ms, seed=args.seed)
        report['gpu_repeat_identical'] = bool(np.array_equal(gpu['trains'], gpu2['trains']))
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
