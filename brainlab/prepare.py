"""Prepare every retained connection without game-specific cell selections."""
import json
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from .connectome import ROOT
from .transmitters import transmitter_signs


def main():
    source = ROOT/'connectome_data/malecns_v1/normalized'
    nodes = feather.read_table(source/'neurons.feather').to_pandas()
    with pa.memory_map(str(source/'edges.arrow'), 'r') as mapping:
        edges = ipc.open_file(mapping).read_all()
    pre, post, count = [edges.column(k).to_numpy() for k in
                        ['pre_index', 'post_index', 'synapse_count']]
    order = np.argsort(pre, kind='stable')
    signs, uncertain = transmitter_signs(nodes.neurotransmitter)
    out = ROOT/'outputs/brainlab/malecns_v1'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'graph.npz',
        ptr=np.r_[0, np.cumsum(np.bincount(pre, minlength=len(nodes)))].astype(np.int64),
        post=post[order].astype(np.int32),
        weight=(count[order].astype(np.float32)*signs[pre[order]]*.275).astype(np.float32),
        ids=nodes.source_id.to_numpy(dtype=np.int64))
    manifest = dict(dataset='MaleCNS v1.0', neurons=len(nodes), edges=len(pre),
        uncertain_sign_neurons=int(uncertain.sum()), synaptic_scale=.275,
        dynamics='Upstream fixed-weight LIF proxy, dt=0.1 ms; not biologically validated',
        input='Explicit per-neuron current', output='Per-neuron spike counts',
        learning=False, game_interface=False)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
