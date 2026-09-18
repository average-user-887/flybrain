"""Run a synthetic smoke demo or stimulate an explicit neuron in a real graph."""
import argparse
import json
from pathlib import Path
import numpy as np
from .brain import Brain
from .connectome import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--neuron-id', type=int)
    parser.add_argument('--current', type=float, default=20.)
    parser.add_argument('--duration-ms', type=float, default=100.)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/brainlab/demo.json')
    args = parser.parse_args()
    if not args.graph:
        graph = ROOT/'outputs/brainlab/synthetic.npz'
        graph.parent.mkdir(parents=True, exist_ok=True)
        np.savez(graph, ptr=np.array([0,1,2,2],dtype=np.int64),
            post=np.array([1,2],dtype=np.int32), weight=np.array([100,100],dtype=np.float32),
            ids=np.array([0,1,2],dtype=np.int64))
        neuron_id = 0 if args.neuron_id is None else args.neuron_id
    else:
        graph = args.graph
        if args.neuron_id is None:
            parser.error('--neuron-id is required with --graph')
        neuron_id = args.neuron_id
    brain = Brain(graph)
    selected = np.flatnonzero(brain.ids == neuron_id)
    if len(selected) != 1:
        parser.error('Neuron ID not found in graph')
    drive = np.zeros(brain.n,dtype=np.float32)
    drive[selected[0]] = args.current
    counts, wall = brain.step(drive,args.duration_ms)
    result = dict(dataset='user-supplied graph' if args.graph else 'SYNTHETIC three-neuron chain (not fly data)',
        neurons=brain.n, edges=len(brain.post), stimulated_id=str(neuron_id),
        sim_ms=brain.sim_ms, wall_seconds=wall, total_spikes=int(counts.sum()),
        active_neurons=int(np.count_nonzero(counts)),
        spikes={str(brain.ids[i]):int(counts[i]) for i in np.flatnonzero(counts)})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
