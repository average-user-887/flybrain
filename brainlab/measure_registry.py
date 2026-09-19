"""Measure real-graph registry costs on this machine (WP4 acceptance item 5).

Loads the verified graph ONCE, then records: load time, RSS, sustained LIF
throughput under a fixed uniform retinal drive, checkpoint size / write time /
load time, the A->B->A switch cost, and the cost of materializing a plastic
working weight copy.  Writes JSON to --out.  Use /usr/bin/time -v for peak RSS.

The drive is the declared validation stimulus (all R1-R6 at 20 units), not a
behavioral stimulus; this measures compute, not behavior.
"""
import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np


def rss_mib():
    with open('/proc/self/status') as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sustained-ms', type=float, default=1000.0)
    parser.add_argument('--step-ms', type=float, default=2.0)
    args = parser.parse_args()

    from experiment_registry import ExperimentRegistry, SharedGraph
    import pyarrow.feather as feather

    report = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                  started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), rss_mib_start=rss_mib())
    clock = time.perf_counter()
    shared = SharedGraph.load()
    report['graph_verify_and_load_s'] = time.perf_counter() - clock
    report['rss_mib_after_graph_load'] = rss_mib()
    report['identity'] = shared.identity.to_dict()
    report['graph_array_bytes'] = {k: int(v.nbytes) for k, v in shared.arrays.items()}

    nodes = feather.read_table(shared.identity.neuron_map_path, columns=['cell_type']).to_pandas()
    retinal = np.flatnonzero(nodes.cell_type.eq('R1-R6').to_numpy())
    currents = np.zeros(shared.n, dtype=np.float32)
    currents[retinal] = 20.0
    report['drive'] = dict(cells='R1-R6', count=int(len(retinal)), current=20.0, step_ms=args.step_ms)

    registry = ExperimentRegistry(shared, args.registry_root)
    results = {}
    for backend in ('connectome-fixed', 'connectome-with-trained-readout'):
        entry = {}
        clock = time.perf_counter()
        a = registry.activate('optomotor', backend)
        entry['activate_fresh_s'] = time.perf_counter() - clock
        a.step(currents, args.step_ms)            # numba compile / warm-up, excluded
        steps = int(round(args.sustained_ms / args.step_ms))
        spikes = 0
        clock = time.perf_counter()
        for i in range(steps):
            result = a.step(currents, args.step_ms, target=(i % 2) if backend.endswith('readout') else None)
            spikes += int(result.counts.sum())
        wall = time.perf_counter() - clock
        entry.update(sustained_sim_ms=steps * args.step_ms, sustained_wall_s=wall,
                     sim_ms_per_wall_s=steps * args.step_ms / wall,
                     realtime_factor=(steps * args.step_ms / 1000.0) / wall, spikes=spikes,
                     rss_mib_while_active=rss_mib())
        clock = time.perf_counter()
        path = registry.checkpoint(world_state={'note': 'measurement; no arena attached'})
        entry['checkpoint_write_s'] = time.perf_counter() - clock
        entry['checkpoint_bytes'] = path.stat().st_size
        entry['checkpoint_path'] = str(path)
        clock = time.perf_counter()
        registry.activate('buridan', backend)
        entry['switch_to_fresh_other_s'] = time.perf_counter() - clock
        clock = time.perf_counter()
        restored = registry.activate('optomotor', backend)
        entry['switch_back_restore_s'] = time.perf_counter() - clock
        entry['restored_step_index'] = restored.step_index
        clock = time.perf_counter()
        registry.read_checkpoint(restored.instance_id)
        entry['checkpoint_read_verify_s'] = time.perf_counter() - clock
        results[backend] = entry
    report['backends'] = results

    clock = time.perf_counter()
    working = shared.arrays['weight'].copy()
    report['plastic_working_copy'] = dict(bytes=int(working.nbytes), copy_s=time.perf_counter() - clock,
        note='connectome-plastic materializes one dense working weight copy for the ACTIVE instance only')
    del working
    edges = len(shared.arrays['weight'])
    report['plastic_delta_storage'] = dict(
        dense_bytes=edges * 4, sparse_bytes_per_edge=12,
        sparse_breakeven_edges=edges * 4 // 12,
        note='Sparse (int64 index + float32 delta) is smaller only below the break-even edge count.')
    report['rss_mib_end'] = rss_mib()
    report['ru_maxrss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
