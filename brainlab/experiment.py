"""Logged full-brain baseline experiment with randomized, reset trials."""
import argparse
import numpy as np
import pyarrow.feather as feather
from .brain import Brain
from .connectome import ROOT, file_digest
from .runs import Run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--runs-dir', type=str)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    graph = ROOT/'outputs/brainlab/malecns_v1/graph.npz'
    nodes = feather.read_table(ROOT/'connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
    retina = np.flatnonzero(nodes.cell_type.eq('R1-R6').to_numpy())
    lamina = np.flatnonzero(nodes.cell_type.isin(['L1','L2','L3','L5']).to_numpy())
    schedule = np.tile(['silent','tonic_only','retina_plus_tonic'], args.repeats)
    np.random.default_rng(args.seed).shuffle(schedule)
    config = dict(experiment='stimulation-baseline-v1', learning=False, reset_each_trial=True,
        repeats=args.repeats, schedule=schedule.tolist(), bin_ms=10., stimulus_ms=100.,
        recovery_ms=50., retinal_current=20., lamina_current=12.,
        seed_scope='Trial order only; neural dynamics deterministic',
        neuron_order_source_sha256=file_digest(
            ROOT/'connectome_data/malecns_v1/normalized/neurons.feather'))
    with Run(graph, config, args.seed, root=args.runs_dir) as run:
        print(f'Run directory: {run.path}', flush=True)
        for trial_id, condition in enumerate(schedule):
            brain = Brain(graph)
            np.testing.assert_array_equal(brain.ids, nodes.source_id.to_numpy(dtype=np.int64))
            currents = np.zeros(brain.n,dtype=np.float32)
            if condition != 'silent':
                currents[lamina] = 12.
            if condition == 'retina_plus_tonic':
                currents[retina] = 20.
            run.event('trial_start', trial_id=trial_id, condition=str(condition), reset=True)
            total = 0
            for phase, bins in [('stimulus',10),('recovery',5)]:
                drive = currents if phase == 'stimulus' else np.zeros_like(currents)
                for _ in range(bins):
                    counts,_ = run.step(brain,drive,10.,trial_id=trial_id,phase=phase)
                    if phase == 'stimulus':
                        total += int(counts.sum())
            run.trial(trial_id=trial_id, condition=str(condition), stimulus_spikes=total,
                success=None, reward=None, split='baseline', learning=False)
            print(f'{trial_id+1}/{len(schedule)} {condition}: {total} stimulus spikes', flush=True)
    print('Completed. Use python -m brainlab.report to compare recorded runs.')

if __name__ == '__main__':
    main()
