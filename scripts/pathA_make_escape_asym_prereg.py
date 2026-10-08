#!/usr/bin/env python3
"""Freeze the ONE approved 75-row direct chemical-edge deletion design; no dynamics."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pathA_escape_asym_common as common

NEW_SOURCES=('scripts/pathA_escape_asym_common.py','scripts/pathA_escape_asym_run.py',
             'scripts/pathA_escape_asym_launch.py','scripts/pathA_escape_asym_analyse.py',
             'scripts/pathA_make_escape_asym_prereg.py','tests/test_pathA_escape_asym_preparation.py')
ACCEPTED_FILES={'analysis.json':'d4e1458f9d9b7c9954ef58c5dfa22c7eaa27c2e63760ddf5ad128e604b24638e',
                'runs.jsonl':'0ab96c300fd267ef7fea19126d9119992f482240ee9c3c3f8152e23b4f1e59e0',
                'EVIDENCE_SHA256.json':'273c4d975df5194b2d9ad93c83441761f6b2c2335b8a4bd6b49e7884324e991c'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('graph-dir','connectome-dir','accepted-runs','out-dir'): ap.add_argument('--'+key,required=True,type=Path)
    a=ap.parse_args(); common.require_environment()
    q=common.ROOT/'qualification/pathA'
    c=common.strict_json(q/'escape_transmission_contract.json')
    p=common.strict_json(q/'escape_transmission_prereg.json')
    gf,ttm=(c['sets'][name]['node_index'] for name in ('GF','TTMn'))
    conditions=[]
    for name,nodes in (('GF10001',[gf[0]]),('GF10010',[gf[1]]),('GF_BOTH',gf)):
        for arm,edges in (('intact',[]),('delete_direct',[0,14539])):
            conditions.append(dict(id=name+'_200_'+arm,activate_nodes=nodes,rate_hz=200,clamp_nodes=[],zero_edges=edges,seeds=list(range(8)),arm=arm,weight_copy=bool(edges)))
        conditions.append(dict(id=name+'_200_empty_deletion_sham',activate_nodes=nodes,rate_hz=200,clamp_nodes=[],seeds=[0],arm='empty_deletion_sham',zero_edges=[],weight_copy=True))
    for arm,edges in (('intact',[]),('delete_direct',[0,14539])):
        conditions.append(dict(id='NO_INPUT_'+arm,activate_nodes=[],rate_hz=0,clamp_nodes=[],zero_edges=edges,seeds=list(range(8)),arm=arm,weight_copy=bool(edges)))
    conditions.append(dict(id='TTMn_BOTH_200_direct_control',activate_nodes=ttm,rate_hz=200,clamp_nodes=[],seeds=list(range(8)),arm='direct_activation_control',zero_edges=[],weight_copy=False))
    c.update(schema='neurofly.pathA.escape_asym_contract/1',label='ESCAPE-ASYM-02: EXPLORATORY OFFLINE CHEMICAL-EDGE DELETION DIAGNOSTIC',conditions=conditions,
             parent_contract_sha256=common.sha(q/'escape_transmission_contract.json'))
    p.update(schema='neurofly.pathA.escape_asym_prereg/1',label=c['label'],conditions=copy.deepcopy(conditions),direct_edges=copy.deepcopy(common.DIRECT_EDGES),
             direct_edges_from='GF',direct_edges_to='TTMn',direct_edge_count=2,
             question='At unchanged three GF inputs, how do each TTMn and other frozen readouts change when ONLY chemical edges 0/14539 are zeroed in a v3 weight copy?',
             rationale='Discriminate fixed-model sensitivity to the two measured direct chemical edges. Counts cannot isolate recurrent routes or establish physiological transmission.')
    p.pop('expected_inputs',None)
    p['controls']=dict(no_input='Eight seeds per intact/deletion arm: full network zero.',
                       direct_ttmn='Eight unchanged direct-TTMn control rows reproduce accepted full sparse counts exactly.',
                       empty_deletion='Three seed-0 shams exercise deleted_weight(arrays,[]) and a separate weight-copy Brain; full counts equal intact.',
                       accepted_reproduction='All 40 intact/no-input/direct-control rows reproduce matching accepted full sparse counts exactly; abort on the first difference.',
                       matched_inputs='Same ordered activated nodes, exact packed events/RNG and full canonical reset for each pair; no clamp nodes.')
    p['analysis']=dict(primary='Per-cell absolute counts/Hz first, then arithmetic population mean; across eight seeds mean and sample SD; paired deletion-minus-intact per cell/population, lower/equal/higher seed counts.',
                       asymmetry='Report TTMn 800146 and 804642 separately. All eight lower/equal, or mixed, are descriptive sensitivity labels only; no fitted threshold.',
                       network_controls='Full sparse network: changed-neuron count and sum absolute count difference, including changes outside activated GF/GFC2.',
                       interpretation='Deletion sensitivity in the full recurrent chemical-only proxy. No biological success, pathway fraction, necessity or mediation inference; missing electrical wiring stays missing, no new wiring; A2 FAIL unchanged.')
    p['gates'].update(G1_external_identity='External source, contract, prereg, accepted evidence, graph/labels/edges pins; typed edge/body identities and exact full GF->TTMn edge coverage.',
                      G2_state_weight='Full reset fixture unchanged. Full base/deleted float32 weight artifacts match external byte and NPY pins; exactly edges 0/14539 differ, both zero; immutable parent retains base hash every row.',
                      G4_controls='All 40 matching accepted full-network count records reproduce exactly; every no-input row zero; three empty-deletion shams equal intact; every activated cell fires.',
                      success='All gates pass: VALID descriptive diagnostic, including zero or mixed deletion effects. No biological pass threshold.',
                      failure='Any identity/input/state/weight/control/count/log/coverage/exit failure: INVALID, retain partial evidence, stop interpretation, no retry/expansion/fitting.')
    p['compute']['estimate']='Previous accepted 75 rows 97.987614 seconds; estimate 2-5 minutes plus audit, one CPU thread/nice10; one hour overall, 3500-second timeout child.'
    p['status']='FROZEN PRELAUNCH PREPARATION ONLY; ROOT GO REQUIRED; NO DYNAMICS AUTHORIZED'
    p['accepted_reference']=dict(source_sha='a1456f3383b8ea87bc0c089ef9ef05492a746efb',contract_sha256=common.sha(q/'escape_transmission_contract.json'),prereg_sha256=common.sha(q/'escape_transmission_prereg.json'),files_sha256=ACCEPTED_FILES,counts_sha256={})
    for file,digest in ACCEPTED_FILES.items():
        if common.sha(a.accepted_runs/file)!=digest: raise ValueError('accepted evidence changed: '+file)
    manifest=common.strict_json(a.accepted_runs/'EVIDENCE_SHA256.json')
    for cc in conditions:
        if cc['arm'] in ('intact','direct_activation_control'):
            for seed in cc['seeds']:
                key=cc['id']+'_s'+str(seed)
                p['accepted_reference']['counts_sha256'][key]=manifest['counts/'+key+'.npz']
    with np.load(a.graph_dir/'graph.npz',allow_pickle=False) as z:
        p['pins']['n_neurons']=len(z['ids'])
    # Derive copy once, statically, from the already accepted full weight artifact.
    base=np.load(a.accepted_runs/'weights.npy',allow_pickle=False)
    if common.weight_hash(base)!=p['pins']['v3_weight_sha256'] or common.npy_hash(base)!=p['pins']['weights_npy_sha256']: raise ValueError('accepted base weight changed')
    deleted=base.copy(); deleted[[0,14539]]=0
    p['pins']['deleted_weight_sha256']=common.weight_hash(deleted)
    p['pins']['deleted_weights_npy_sha256']=common.npy_hash(deleted)
    p['pins']['source_files_sha256'].update({name:common.sha(common.ROOT/name) for name in NEW_SOURCES})
    for name in ('escape_transmission_contract.json','escape_transmission_prereg.json'):
        p['pins']['source_files_sha256']['qualification/pathA/'+name]=common.sha(q/name)
    p['expected_inputs']={cc['id']+'_s'+str(seed):common.input_data(cc,seed,c['protocol'])[1] for cc in conditions for seed in cc['seeds']}
    common.validate_manifest(c,p)
    common.static_assets(c,p,a.graph_dir,a.connectome_dir,a.accepted_runs)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    cpath=a.out_dir/'escape_asymmetry_contract.json'; ppath=a.out_dir/'escape_asymmetry_prereg.json'
    if cpath.exists() or ppath.exists(): raise ValueError('never overwrite frozen derived documents')
    with cpath.open('x') as f: f.write(json.dumps(c,indent=1)+'\n')
    p['pins']['derived_contract_sha256']=common.sha(cpath)
    with ppath.open('x') as f: f.write(json.dumps(p,indent=1)+'\n')
    print(json.dumps(dict(contract_sha256=common.sha(cpath),prereg_sha256=common.sha(ppath),deleted_weight_sha256=p['pins']['deleted_weight_sha256'],analyzer_sha256=common.sha(common.ROOT/'scripts/pathA_escape_asym_analyse.py'))))

if __name__=='__main__': main()
