#!/usr/bin/env python3
"""Audited adapter around unchanged pathA_run.run_one; launched only after root approval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pathA_escape_asym_common as common


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('contract', 'contract-sha256', 'prereg', 'prereg-sha256', 'expected-code-sha', 'out', 'accepted-runs'):
        ap.add_argument('--' + flag, required=True)
    return ap


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')


def check_runtime_controls(condition,total,out,seed,reference):
    """Pure count checks; called after retaining a row and before the next run."""
    if (condition['rate_hz']==0 and np.any(total)) or np.any(total[condition['clamp_nodes']]) \
            or np.any(total[condition['activate_nodes']]<=0):
        raise ValueError('runtime no-input/clamp/direct-activation control failed')
    if condition['arm']=='empty_deletion_sham':
        sham_reference=out/'counts'/(condition['id'].replace('empty_deletion_sham','intact')+'_s0.npz')
        nz=np.flatnonzero(total)
        with np.load(sham_reference,allow_pickle=False) as z:
            if not np.array_equal(z['node_index'],nz.astype(np.int32)) or not np.array_equal(z['counts'],total[nz]):
                raise ValueError('runtime empty-deletion sham differs from intact')

    if condition['arm'] in ('intact','direct_activation_control'):
        key=condition['id']+'_s'+str(seed)
        actual={int(i):int(total[i]) for i in np.flatnonzero(total)}
        if actual!=reference[key]: raise ValueError('intact/direct control differs from accepted raw counts')


def main():
    a = parser().parse_args()
    common.require_environment()
    c, p = common.load_plan(a.contract, a.contract_sha256, a.prereg, a.prereg_sha256, a.expected_code_sha)
    out = Path(a.out).resolve()
    launch = common.strict_json(out / 'launch.json')
    if launch['runner_argv'] != sys.argv or launch['source_sha'] != a.expected_code_sha:
        raise ValueError('runner not bound to its exclusive launcher receipt')
    pins = p['pins']
    data,static_base,static_deleted,reference=common.static_assets(c,p,os.environ['NEUROFLY_GRAPH_DIR'],os.environ['NEUROFLY_CONNECTOME_DIR'],a.accepted_runs)
    gdir, cdir = Path(os.environ['NEUROFLY_GRAPH_DIR']), Path(os.environ['NEUROFLY_CONNECTOME_DIR'])
    data = dict(graph_npz=gdir / 'graph.npz', neurons_feather=cdir / 'normalized/neurons.feather',
                annotations_feather=cdir / 'annotations.feather', edges_arrow=cdir / 'normalized/edges.arrow')
    for name, path in data.items():
        if common.sha(path) != pins[name + '_sha256']:
            raise ValueError('data identity mismatch: ' + name)
    import pathA_run as frozen_run
    import pathA_provenance as provenance
    import pathA_gf_edge_analyse as fixture_check
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    arrays = frozen_run.load_v3_arrays(gdir, cdir, pins['neurons_feather_sha256'])
    if hashlib.sha256(arrays['weight'].tobytes()).hexdigest() != pins['v3_weight_sha256']:
        raise ValueError('reconstructed v3 weights differ')
    for name, s in c['sets'].items():
        if arrays['ids'][s['node_index']].tolist() != s['body_ids']:
            raise ValueError('frozen set body identity mismatch: ' + name)
    fixture = common.ROOT / pins['reset_fixture_file']
    digest, problems = fixture_check.fixture_digest(fixture, pins['state_arrays_order'], 'v3', len(arrays['ids']))
    if common.sha(fixture) != pins['reset_fixture_sha256'] or problems or digest != pins['reset_state_sha256']:
        raise ValueError('independent reset fixture check failed')
    from brainlab.brain import Brain
    frozen_run.check_direct_edges(arrays,{k:np.array(c['sets'][k]['node_index'],np.int64) for k in ('GF','TTMn')},p)
    deleted=frozen_run.deleted_weight(arrays,[0,14539])
    common.check_weight_copy(arrays['weight'],deleted,pins)
    brain = Brain(arrays=arrays, validate=True, dynamics='v3', backend='cpu')
    deleted_brain=Brain(arrays=dict(arrays,weight=deleted),validate=True,dynamics='v3',backend='cpu')
    (out / 'inputs').mkdir()
    (out / 'counts').mkdir()
    shutil.copyfile(fixture, out / 'reset_fixture.npz')
    np.save(out / 'weights.npy', arrays['weight'], allow_pickle=False)
    if common.sha(out / 'weights.npy') != pins['weights_npy_sha256']:
        raise ValueError('saved full weights differ from static pin')
    np.save(out/'deleted_weights.npy',deleted,allow_pickle=False)
    if common.sha(out/'deleted_weights.npy')!=pins['deleted_weights_npy_sha256']: raise ValueError('saved deleted weight pin mismatch')
    write_json(out / 'startup.json', dict(pid=os.getpid(), parent_pid=os.getppid(), source_sha=a.expected_code_sha,
                                         argv=sys.argv, python=sys.executable,
                                         thread_environment={k: os.environ[k] for k in common.THREAD_ENV},
                                         contract_sha256=a.contract_sha256, prereg_sha256=a.prereg_sha256))
    base = dict(contract_sha256=a.contract_sha256, prereg_sha256=a.prereg_sha256,
                code_sha=a.expected_code_sha, graph_npz_sha256=pins['graph_npz_sha256'],
                engine_sha256=provenance.engine_sha256(), dynamics='v3', backend='cpu')
    if base['engine_sha256'] != pins['engine_sha256']:
        raise ValueError('engine differs from static pin')
    readouts = {name: np.asarray(c['sets'][name]['node_index'], np.int64) for name in c['readouts']}
    with (out / 'runs.jsonl').open('x') as log:
        for condition in c['conditions']:
            if condition['arm']=='empty_deletion_sham':
                sham_weight=frozen_run.deleted_weight(arrays,[])
                if sham_weight.tobytes()!=arrays['weight'].tobytes(): raise ValueError('empty deletion changed weights')
                current=Brain(arrays=dict(arrays,weight=sham_weight),validate=True,dynamics='v3',backend='cpu')
            else:
                current=deleted_brain if condition['zero_edges'] else brain
            wanted_weight=pins['deleted_weight_sha256'] if condition['zero_edges'] else pins['v3_weight_sha256']
            for seed in condition['seeds']:
                key = condition['id'] + '_s' + str(seed)
                retained, expected = common.input_data(condition, seed, c['protocol'])
                input_path = out / 'inputs' / (key + '.npz')
                np.savez_compressed(input_path, **retained)
                # run_one performs its own canonical reset and regenerates its input train.
                audit = {}
                res, total, wall, stim_rate, event_ticks = frozen_run.run_one(
                    current, retained['activation_nodes'], condition['rate_hz'],
                    np.asarray(condition['clamp_nodes'], np.int64), readouts, c, seed, audit)
                for field in ('input_sha256', 'rng_seed_seq', 'rng_state_sha256'):
                    if audit[field] != expected[field]:
                        raise ValueError('retained input differs from unchanged runner')
                weight_digest = hashlib.sha256(current.weight.tobytes()).hexdigest()
                parent_digest = hashlib.sha256(arrays['weight'].tobytes()).hexdigest()
                if audit['initial_state_clean'] is not True or audit['state_sha256'] != pins['reset_state_sha256'] \
                        or weight_digest != wanted_weight or parent_digest != pins['v3_weight_sha256'] \
                        or event_ticks != expected['event_ticks']:
                    raise ValueError('runtime input/reset/weight integrity failure')
                nz = np.flatnonzero(total)
                counts_path = out / 'counts' / (key + '.npz')
                np.savez_compressed(counts_path, node_index=nz.astype(np.int32), counts=total[nz].astype(np.int64))
                row = dict(condition=condition['id'], seed=seed, arm=condition['arm'],
                           rate_hz=condition['rate_hz'], activate_nodes=condition['activate_nodes'],
                           clamp_nodes=condition['clamp_nodes'],zero_edges=condition['zero_edges'],weight_copy=condition['weight_copy'], provenance=dict(base, seed=seed),
                           audit=dict(audit, weight_sha256=weight_digest, parent_weight_unchanged=True),
                           input_file_sha256=common.sha(input_path), counts_file_sha256=common.sha(counts_path),
                           event_ticks=event_ticks, events_per_cell=expected['events_per_cell'],
                           activated_mean_rate_hz=stim_rate, wall_s=wall, total_spikes=int(total.sum()),
                           firing_neurons=int(len(nz)), readouts=res)
                line = json.dumps(row, sort_keys=True)
                log.write(line + '\n')
                log.flush()
                print(json.dumps(dict(condition=condition['id'], seed=seed,
                                      row_sha256=hashlib.sha256(line.encode()).hexdigest()), sort_keys=True), flush=True)
                # Retain the failing row, then stop immediately; never continue a broken control.
                check_runtime_controls(condition,total,out,seed,reference)
    for name, path in data.items():
        if common.sha(path) != pins[name + '_sha256']:
            raise ValueError('data changed during run: ' + name)
    common.accepted_references(p,a.accepted_runs)


if __name__ == '__main__':
    main()
