#!/usr/bin/env python3
"""Freeze the accepted 75-row design and exact inputs, without any dynamics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import platform
from pathlib import Path

import numpy as np

import pathA_escape_common as common

SOURCE_FILES = ('brainlab/engine.py', 'brainlab/brain.py', 'brainlab/transmitter_policy.py',
                'brainlab/transmitters.py', 'scripts/pathA_run.py', 'scripts/pathA_provenance.py',
                'scripts/pathA_gf_edge_analyse.py', 'scripts/pathA_cf_analyse.py',
                'scripts/pathA_escape_common.py', 'scripts/pathA_escape_run.py',
                'scripts/pathA_escape_launch.py', 'scripts/pathA_escape_analyse.py',
                'scripts/pathA_make_escape_prereg.py', 'scripts/pathA_escape_inventory.py',
                'qualification/pathA/contract.json', 'qualification/pathA/pvlp151_cf_prereg.json',
                'qualification/pathA/pvlp151_gf_edge_prereg.json',
                'qualification/pathA/escape_transmission_inventory.json',
                'qualification/pathA/escape_transmission_proposal.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--graph-dir',required=True,type=Path)
    ap.add_argument('--connectome-dir',required=True,type=Path)
    ap.add_argument('--out-dir',required=True,type=Path)
    a=ap.parse_args()
    common.require_environment()
    q=common.ROOT/'qualification/pathA'
    proposal=common.strict_json(q/'escape_transmission_proposal.json')
    original=common.strict_json(q/'contract.json')
    cf=common.strict_json(q/'pvlp151_cf_prereg.json')
    p=copy.deepcopy(proposal)
    sets=dict(original['sets'],**cf['sets'])
    contract=dict(schema='neurofly.pathA.escape_contract/1',
                  label=p['label'],status='FROZEN PRELAUNCH PREPARATION; ROOT LAUNCH DECISION REQUIRED',
                  parent_contract_sha256=common.sha(q/'contract.json'),data=original['data'],
                  protocol=p['protocol'],sets={k:sets[k] for k in p['readouts']},
                  readouts=p['readouts'],conditions=p['conditions'])
    common.validate_manifest(contract,p)
    if any(a.out_dir.joinpath(f).exists() for f in ('escape_transmission_contract.json','escape_transmission_prereg.json')):
        raise ValueError('refusing to overwrite a frozen derived contract/prereg')
    paths=dict(graph_npz=a.graph_dir/'graph.npz',neurons_feather=a.connectome_dir/'normalized/neurons.feather',
               annotations_feather=a.connectome_dir/'annotations.feather',edges_arrow=a.connectome_dir/'normalized/edges.arrow')
    for name,path in paths.items():
        if common.sha(path)!=p['pins'][name+'_sha256']:
            raise ValueError('data pin mismatch: '+name)
    import sys
    sys.path.insert(0,str(common.ROOT))
    from brainlab.transmitter_policy import apply_policy
    import pyarrow as pa
    import pyarrow.feather as feather
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    with np.load(paths['graph_npz'],allow_pickle=False) as z:
        ptr,post,raw,ids=z['ptr'],z['post'],z['weight'],z['ids']
    table=feather.read_table(paths['neurons_feather']).to_pandas().sort_values('node_index')
    if not np.array_equal(table.node_index,np.arange(len(ids))) or not np.array_equal(table.source_id,ids):
        raise ValueError('neuron/body ordering mismatch')
    for name,s in contract['sets'].items():
        if ids[s['node_index']].tolist()!=s['body_ids']:
            raise ValueError('frozen body identity mismatch')
    weights,_=apply_policy(ptr,post,raw,table.neurotransmitter.to_numpy())
    weights=np.ascontiguousarray(weights,dtype=np.float32)
    if hashlib.sha256(weights.tobytes()).hexdigest()!=p['pins']['v3_weight_sha256']:
        raise ValueError('v3 weight mismatch')
    buffer=io.BytesIO()
    np.save(buffer,weights,allow_pickle=False)
    p['pins']['weights_npy_sha256']=hashlib.sha256(buffer.getvalue()).hexdigest()
    p['pins']['reset_fixture_file']='qualification/pathA/fixtures/pvlp151_gf_reset_state_v3.npz'
    if common.sha(common.ROOT/p['pins']['reset_fixture_file'])!=p['pins']['reset_fixture_sha256']:
        raise ValueError('reset fixture pin mismatch')
    p['pins']['source_files_sha256']={name:common.sha(common.ROOT/name) for name in SOURCE_FILES}
    engine=hashlib.sha256()
    for name in SOURCE_FILES[:3]:
        engine.update(name.encode()+b'\0'+(common.ROOT/name).read_bytes()+b'\0')
    if engine.hexdigest()!=p['pins']['engine_sha256']:
        raise ValueError('engine differs from accepted identity')
    p['runtime_versions']=dict(python=platform.python_version(),numpy=np.__version__,numba=importlib.metadata.version('numba'))
    p['expected_inputs']={c['id']+'_s'+str(seed):common.input_data(c,seed,contract['protocol'])[1]
                          for c in contract['conditions'] for seed in c['seeds']}
    p['schema']='neurofly.pathA.escape_prereg/1'
    p['status']='FROZEN PRELAUNCH PREPARATION ONLY; NO DYNAMICS AUTHORIZED'
    p['root_launch_decision']['requested']='Review final clean source, derived contract/prereg/analyzer external pins and focused negative fixtures; authorize exactly one 75-row CPU launch or reject. Preparation does not authorize execution.'
    p['root_launch_decision']['required_before_launch']=[
        'Root independently review candidate and record external source/contract/prereg/analyzer pins.',
        'Root record fresh CPU/RAM availability and preserve owner services.',
        'Root issue explicit GO, start the exclusive launcher and capture actual timeout-child/wrapper PIDs.',
        'Root audit complete output with the fixed independent checker before interpreting.']
    p['launch_receipts']='Exclusive one-process launch; timeout --signal=TERM 3500 is inside child. Analyzer requires externally observed wrapper and timeout-child PIDs, exact argv, startup environment, integer exit0, ordered row-byte hashes and full log hash.'
    for name,path in paths.items():
        if common.sha(path)!=p['pins'][name+'_sha256']:
            raise ValueError('data changed while freezing')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    cpath=a.out_dir/'escape_transmission_contract.json'
    with cpath.open('x') as f:
        f.write(json.dumps(contract,indent=1)+'\n')
    p['pins']['derived_contract_sha256']=common.sha(cpath)
    ppath=a.out_dir/'escape_transmission_prereg.json'
    with ppath.open('x') as f:
        f.write(json.dumps(p,indent=1)+'\n')
    print(json.dumps(dict(contract_sha256=common.sha(cpath),prereg_sha256=common.sha(ppath),
                          analyzer_sha256=p['pins']['source_files_sha256']['scripts/pathA_escape_analyse.py'])))


if __name__=='__main__':
    main()
