"""Full retained-graph integrity and numerical smoke checks; not biological validation."""
import json
import time
import resource
import numpy as np
import pyarrow.feather as feather
from .brain import Brain
from .connectome import ROOT, file_digest


def main():
    folder = ROOT/'outputs/brainlab/malecns_v1'
    graph = folder/'graph.npz'
    imported = json.loads((ROOT/'connectome_data/malecns_v1/normalized/report.json').read_text())
    started = time.perf_counter()
    brain = Brain(graph)
    assert brain.n == imported['retained_neuron_candidates'] == 166700
    assert len(brain.post) == imported['graph']['retained_edge_rows'] == 25582938
    assert imported['graph']['retained_synaptic_contacts'] == 124177617
    weights = brain.weight.copy()
    zero = np.zeros(brain.n, dtype=np.float32)
    silent, silence_wall = brain.step(zero, 10.)
    assert not silent.any(), 'Unexpected spontaneous spikes without current'
    assert np.all(brain.v == -52) and not brain.g.any()
    nodes = feather.read_table(ROOT/'connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
    retina = np.flatnonzero(nodes.cell_type.eq('R1-R6').to_numpy())
    assert len(retina) > 0
    drive = zero.copy()
    drive[retina] = 20.
    retina_only, retina_wall = brain.step(drive, 100.)
    retina_outside = retina_only.copy()
    retina_outside[retina] = 0
    # Upstream's graded-cell proxy uses explicit tonic lamina drive. Record
    # retina-only response separately, rather than hiding its lack of propagation.
    lamina = np.flatnonzero(nodes.cell_type.isin(['L1','L2','L3','L5']).to_numpy())
    tonic = zero.copy()
    tonic[lamina] = 12.
    baseline = Brain(graph)
    baseline_counts, baseline_wall = baseline.step(tonic, 100.)
    brain = Brain(graph)
    brain.step(zero, 10.)
    drive[lamina] = 12.
    counts, stimulus_wall = brain.step(drive, 100.)
    assert counts[retina].sum() > 0, 'Stimulated cells did not fire'
    outside = counts.copy()
    outside[np.r_[retina, lamina]] = 0
    assert outside.sum() > 0, 'No propagated spikes beyond stimulated cells'
    assert np.isfinite(brain.v).all() and np.isfinite(brain.g).all()
    assert not np.array_equal(counts, baseline_counts), 'Retinal input had no effect'
    np.testing.assert_array_equal(brain.weight, weights)
    print('Full-graph stimulation passed; checking chunked replay.', flush=True)
    replay = Brain(graph)
    replay.step(zero, 10.)
    repeat = np.zeros(brain.n, dtype=np.int64)
    replay_wall = 0.
    for _ in range(10):
        chunk, wall = replay.step(drive, 10.)
        repeat += chunk
        replay_wall += wall
    np.testing.assert_array_equal(counts, repeat)
    np.testing.assert_array_equal(brain.v, replay.v)
    np.testing.assert_array_equal(brain.g, replay.g)
    np.testing.assert_array_equal(brain.refractory, replay.refractory)
    result = dict(dataset='MaleCNS v1.0', neurons=brain.n, edges=len(brain.post),
        synaptic_contacts=imported['graph']['retained_synaptic_contacts'],
        graph_sha256=file_digest(graph), stimulated_type='R1-R6',
        stimulated_neurons=len(retina), current=20., stimulus_ms=100.,
        tonic_lamina_neurons=len(lamina), tonic_lamina_current=12.,
        retina_only_spikes=int(retina_only.sum()),
        retina_only_propagated_spikes=int(retina_outside.sum()),
        retina_only_wall_seconds=retina_wall,
        tonic_only_spikes=int(baseline_counts.sum()), tonic_only_wall_seconds=baseline_wall,
        neurons_changed_vs_tonic_only=int(np.count_nonzero(counts != baseline_counts)),
        silence_ms=10., silence_spikes=int(silent.sum()),
        stimulus_spikes=int(counts.sum()), firing_neurons=int(np.count_nonzero(counts)),
        propagated_spikes=int(outside.sum()),
        propagated_firing_neurons=int(np.count_nonzero(outside)),
        silence_wall_seconds=silence_wall, stimulus_wall_seconds=stimulus_wall,
        chunked_replay_wall_seconds=replay_wall,
        runtime_seconds=time.perf_counter()-started,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        checks=dict(graph_size=True, silent_control=True, stimulated_response=True,
            network_propagation=True, finite_state=True, fixed_weights=True,
            chunked_replay_exact=True, response_differs_from_tonic_only=True),
        limitations='Uniform artificial current; no calibrated vision, behavior, learning, or biological validation.')
    (folder/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
