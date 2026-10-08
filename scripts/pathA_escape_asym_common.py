"""Additive static deletion plan; reuses frozen escape input/source helpers; no dynamics."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from pathA_escape_common import (ROOT, THREAD_ENV, sha, strict_json, integer,
                                require_environment, input_data, verify_sources)
DIRECT_EDGES = [dict(edge_index=0,pre_node=0,post_node=144432,pre_body=10001,post_body=800146,contacts=70,weight=19.25),
                dict(edge_index=14539,pre_node=6,post_node=147865,pre_body=10010,post_body=804642,contacts=20,weight=5.5)]

def weight_hash(weight):
    return hashlib.sha256(weight.tobytes()).hexdigest()

def npy_hash(weight):
    stream=io.BytesIO(); np.save(stream,weight,allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()

def check_edges(ptr,post,ids,weight,edges):
    if json.dumps(edges,sort_keys=True)!=json.dumps(DIRECT_EDGES,sort_keys=True):
        raise ValueError('typed edge identities/weights differ from fixed design')
    actual=sorted(int(ptr[i])+int(j) for i in (0,6) for j in np.flatnonzero(np.isin(post[ptr[i]:ptr[i+1]],[144432,147865])))
    if actual!=[0,14539]: raise ValueError('not exactly the two GF-to-TTMn chemical edges')
    for r in edges:
        e,pre,target=r['edge_index'],r['pre_node'],r['post_node']
        if not (ptr[pre]<=e<ptr[pre+1]) or int(post[e])!=target or int(ids[pre])!=r['pre_body'] or int(ids[target])!=r['post_body'] or float(weight[e])!=r['weight']:
            raise ValueError('graph edge/body/weight identity mismatch')

def check_weight_copy(base,deleted,pins):
    if base.dtype!=np.float32 or deleted.dtype!=np.float32 or base.shape!=deleted.shape:
        raise ValueError('weight array type/shape mismatch')
    if np.flatnonzero(base!=deleted).tolist()!=[0,14539] or np.any(deleted[[0,14539]]!=0):
        raise ValueError('weight copy must zero exactly edges 0 and 14539')
    if weight_hash(base)!=pins['v3_weight_sha256'] or weight_hash(deleted)!=pins['deleted_weight_sha256'] or npy_hash(base)!=pins['weights_npy_sha256'] or npy_hash(deleted)!=pins['deleted_weights_npy_sha256']:
        raise ValueError('external base/deleted weight pins mismatch')

def validate_manifest(contract, prereg):
    if json.dumps(contract['conditions'],sort_keys=True) != json.dumps(prereg['conditions'],sort_keys=True):
        raise ValueError('contract/prereg conditions disagree')
    p = contract['protocol']
    if any(p.get(k) != v for k, v in dict(dt_ms=0.1, duration_ms=1000.0, dynamics='v3', backend='cpu',
                                        pulse_drive=10000.0, silence_drive=-10000.0,
                                        synaptic_scale_mV_per_contact=0.275).items()):
        raise ValueError('frozen dynamics/drive changed')
    if 'shuffled_input' in p or 'timing_probe_set' in p or p['compute']['timeout_s'] != 3500 or p['compute']['budget_wall_hours'] != 1.0:
        raise ValueError('irrelevant protocol or wrong budget')
    if contract['readouts'] != ['GF','TTMn','GFC2','PSI','DLMn','PVLP151']:
        raise ValueError('readouts differ from fixed design')
    accepted = strict_json(ROOT/'qualification/pathA/escape_transmission_contract.json')
    if any(json.dumps(contract[k],sort_keys=True)!=json.dumps(accepted[k],sort_keys=True) for k in ('protocol','sets','readouts','data')):
        raise ValueError('accepted protocol/sets/readouts/data changed')
    if prereg['direct_edges'] != DIRECT_EDGES or any(not integer(r[k]) for r in prereg['direct_edges'] for k in ('edge_index','pre_body','post_body','pre_node','post_node','contacts')):
        raise ValueError('exact typed direct-edge manifest differs')
    gf, ttm, gfc = (contract['sets'][k]['node_index'] for k in ('GF','TTMn','GFC2'))
    if len(gf)!=2 or len(ttm)!=2 or len(gfc)!=10:
        raise ValueError('target set cardinality differs from frozen design')
    fixed=[]
    for name,nodes in (('GF10001',[gf[0]]),('GF10010',[gf[1]]),('GF_BOTH',gf)):
        for arm,edges in (('intact',[]),('delete_direct',[0,14539])):
            fixed.append(dict(id=name+'_200_'+arm,activate_nodes=nodes,rate_hz=200,clamp_nodes=[],zero_edges=edges,seeds=list(range(8)),arm=arm,weight_copy=bool(edges)))
        fixed.append(dict(id=name+'_200_empty_deletion_sham',activate_nodes=nodes,rate_hz=200,clamp_nodes=[],
                          seeds=[0],arm='empty_deletion_sham',zero_edges=[],weight_copy=True))
    for arm,edges in (('intact',[]),('delete_direct',[0,14539])):
        fixed.append(dict(id='NO_INPUT_'+arm,activate_nodes=[],rate_hz=0,clamp_nodes=[],zero_edges=edges,seeds=list(range(8)),arm=arm,weight_copy=bool(edges)))
    fixed.append(dict(id='TTMn_BOTH_200_direct_control',activate_nodes=ttm,rate_hz=200,clamp_nodes=[],seeds=list(range(8)),arm='direct_activation_control',zero_edges=[],weight_copy=False))
    if json.dumps(contract['conditions'],sort_keys=True)!=json.dumps(fixed,sort_keys=True):
        raise ValueError('conditions differ from accepted fixed design')
    expected = {}
    for c in contract['conditions']:
        if c['id'] in {k[0] for k in expected}:
            raise ValueError('duplicate condition')
        if c['rate_hz'] not in (0, 200) or isinstance(c['rate_hz'], bool):
            raise ValueError('rate outside fixed design')
        for field in ('activate_nodes', 'clamp_nodes', 'zero_edges'):
            nodes = c[field]
            if not all(integer(i) and i >= 0 for i in nodes) or len(set(nodes)) != len(nodes):
                raise ValueError('invalid node identity')
        if set(c['activate_nodes']) & set(c['clamp_nodes']):
            raise ValueError('activation/clamp overlap')
        if c['seeds'] != ([0] if c['arm'] == 'empty_deletion_sham' else list(range(8))):
            raise ValueError('seeds differ from fixed design')
        for seed in c['seeds']:
            if not integer(seed):
                raise ValueError('seed must be integer, never bool/float')
            expected[c['id'], seed] = c
    if len(expected) != 75:
        raise ValueError('expected exactly 75 rows')
    if 'expected_inputs' not in prereg: raise ValueError('missing frozen exact input manifest')
    if 'expected_inputs' in prereg:
        inputs={c['id']+'_s'+str(seed):input_data(c,seed,p)[1] for (name,seed),c in expected.items()}
        if prereg['expected_inputs']!=inputs:
            raise ValueError('frozen exact input manifest differs')
        for record in prereg['expected_inputs'].values():
            if not integer(record['event_ticks']) or not all(integer(v) for v in record['rng_seed_seq']+record['events_per_cell']):
                raise ValueError('input manifest count/seed types malformed')
    return expected


def load_plan(contract_path, contract_sha, prereg_path, prereg_sha, expected_code):
    if sha(contract_path) != contract_sha or sha(prereg_path) != prereg_sha:
        raise ValueError('external contract/prereg identity mismatch')
    contract, prereg = strict_json(contract_path), strict_json(prereg_path)
    if prereg['pins']['derived_contract_sha256'] != contract_sha:
        raise ValueError('prereg contract pin mismatch')
    validate_manifest(contract, prereg)
    verify_sources(prereg, expected_code)
    import importlib.metadata
    import platform
    if prereg['runtime_versions']!=dict(python=platform.python_version(),numpy=np.__version__,numba=importlib.metadata.version('numba')):
        raise ValueError('runtime library versions differ from frozen preparation')
    return contract, prereg

def accepted_references(prereg, directory):
    """Externally frozen existing evidence only; no recomputation or dynamic inference."""
    from pathA_escape_analyse import sparse
    directory=Path(directory)
    ref=prereg['accepted_reference']
    for file,digest in ref['files_sha256'].items():
        if sha(directory/file)!=digest: raise ValueError('accepted evidence identity mismatch: '+file)
    if strict_json(directory/'analysis.json')['status']!='VALID': raise ValueError('accepted reference is not VALID')
    result={}
    for key,digest in ref['counts_sha256'].items():
        file=directory/'counts'/(key+'.npz')
        if sha(file)!=digest: raise ValueError('accepted raw count pin mismatch: '+key)
        result[key]=sparse(file,prereg['pins']['n_neurons'])
    return result

def check_reference_coverage(contract,prereg):
    keys={c['id']+'_s'+str(seed) for c in contract['conditions'] if c['arm'] in ('intact','direct_activation_control') for seed in c['seeds']}
    if keys!=set(prereg['accepted_reference']['counts_sha256']) or len(keys)!=40:
        raise ValueError('accepted intact/direct reference coverage must be exactly 40')

def static_assets(contract,prereg,gdir,cdir,reference):
    """Independent chemical-weight reconstruction; no Brain/engine import or kernels."""
    import pyarrow as pa
    import pyarrow.feather as feather
    from pathA_gf_edge_analyse import fixture_digest
    pa.set_cpu_count(1); pa.set_io_thread_count(1)
    pins=prereg['pins']
    data=dict(graph_npz=Path(gdir)/'graph.npz',neurons_feather=Path(cdir)/'normalized/neurons.feather',
              annotations_feather=Path(cdir)/'annotations.feather',edges_arrow=Path(cdir)/'normalized/edges.arrow')
    for name,file in data.items():
        if sha(file)!=pins[name+'_sha256']: raise ValueError('external data pin mismatch: '+name)
    with np.load(data['graph_npz'],allow_pickle=False) as z:
        ptr,post,ids,weight=z['ptr'],z['post'],z['ids'],z['weight'].copy()
    nt=feather.read_table(data['neurons_feather']).to_pandas().sort_values('node_index')
    if len(ids)!=pins['n_neurons'] or not np.array_equal(nt.node_index,np.arange(len(ids))) or not np.array_equal(nt.source_id,ids):
        raise ValueError('neuron/body ordering mismatch')
    for name,v in contract['sets'].items():
        if ids[v['node_index']].tolist()!=v['body_ids']: raise ValueError('set body identity mismatch: '+name)
    for i,label in enumerate(nt.neurotransmitter):
        if str(label).strip().lower() in ('dopamine','octopamine','serotonin'): weight[ptr[i]:ptr[i+1]]=0
    weight=np.ascontiguousarray(weight,dtype=np.float32)
    check_edges(ptr,post,ids,weight,prereg['direct_edges'])
    deleted=weight.copy(); deleted[[0,14539]]=0
    check_weight_copy(weight,deleted,pins)
    fixture=ROOT/pins['reset_fixture_file']
    digest,problems=fixture_digest(fixture,pins['state_arrays_order'],'v3',len(ids))
    if sha(fixture)!=pins['reset_fixture_sha256'] or problems or digest!=pins['reset_state_sha256']: raise ValueError('reset fixture identity mismatch')
    check_reference_coverage(contract,prereg)
    refs=accepted_references(prereg,reference)
    for name,file in data.items():
        if sha(file)!=pins[name+'_sha256']: raise ValueError('data changed during static verification: '+name)
    return data,weight,deleted,refs
