"""SYNTHETIC static evidence only: no Brain, kernels, simulation, services or GPU."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pathA_escape_asym_analyse as audit
import pathA_escape_asym_common as common
import pathA_escape_common as base
import pathA_escape_asym_run as prepared_runner

CODE='a'*40
PSHA='b'*64
# Frozen engine identity: both escape preregs pin engine_sha256 to the engine at the
# published commit below (7847aab4, ancestor of HEAD; d42985cd carries the same engine).
# Master db62e700 later added the FlyWire-783 guard to brainlab/brain.py, so the current
# engine hashes to CURRENT_ENGINE_SHA256 and must never pass the frozen pin.
PINNED_ENGINE_COMMIT='7847aab46e8a1025ea87b296c3acdf7efdcf6605'
FROZEN_ENGINE_SHA256='473946a0b5206b6991d183e2bbac0b199b76bd44ee1321075b43b45fb561803e'
CURRENT_ENGINE_SHA256='ce64bddd073bafa750f4443844ac9034418d22bca20c590720482dfa3ec2faa4'
ENGINE_FILES=('brainlab/engine.py','brainlab/brain.py','brainlab/transmitter_policy.py')


def engine_source_root(dest,rev):
    """Fixture source root: engine files exactly as committed at `rev`, plus pathA_run.py."""
    for name in ENGINE_FILES:
        data=subprocess.run(['git','-C',str(ROOT),'show',rev+':'+name],check=True,capture_output=True).stdout
        (dest/name).parent.mkdir(parents=True,exist_ok=True)
        (dest/name).write_bytes(data)
    (dest/'scripts').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'scripts/pathA_run.py',dest/'scripts/pathA_run.py')
    return dest


def engine_digest(root):
    h=hashlib.sha256()
    for name in ENGINE_FILES:
        h.update(name.encode()+b'\0'+(root/name).read_bytes()+b'\0')
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


@pytest.fixture
def evidence(tmp_path):
    c=json.loads((ROOT/'qualification/pathA/escape_asymmetry_contract.json').read_text())
    p=json.loads((ROOT/'qualification/pathA/escape_asymmetry_prereg.json').read_text())
    n=p['pins']['n_neurons']
    p['conditions']=copy.deepcopy(c['conditions'])
    p['expected_inputs']={cc['id']+'_s'+str(s):common.input_data(cc,s,c['protocol'])[1]
                          for cc in c['conditions'] for s in cc['seeds']}
    out=tmp_path
    (out/'inputs').mkdir();(out/'counts').mkdir()
    rows=[]
    for cc in c['conditions']:
        for seed in cc['seeds']:
            base=cc['id']+'_s'+str(seed)+'.npz'
            inp, rec=common.input_data(cc,seed,c['protocol'])
            np.savez_compressed(out/'inputs'/base,**inp)
            sp={i:1 for i in cc['activate_nodes']}
            if cc['rate_hz'] and cc['arm']!='direct_activation_control':
                sp.update({i:2 for i in c['sets']['TTMn']['node_index']})
                if not cc['clamp_nodes']:
                    sp.update({i:3 for i in c['sets']['GFC2']['node_index']})
            nn=sorted(sp)
            np.savez_compressed(out/'counts'/base,node_index=np.array(nn,np.int32),counts=np.array([sp[i] for i in nn],np.int64))
            r=dict(condition=cc['id'],seed=seed,arm=cc['arm'],rate_hz=cc['rate_hz'],
                   activate_nodes=cc['activate_nodes'],clamp_nodes=cc['clamp_nodes'],zero_edges=cc['zero_edges'],weight_copy=cc['weight_copy'],
                   provenance=dict(contract_sha256=p['pins']['derived_contract_sha256'],prereg_sha256=PSHA,
                       code_sha=CODE,graph_npz_sha256=p['pins']['graph_npz_sha256'],engine_sha256=p['pins']['engine_sha256'],
                       dynamics='v3',backend='cpu',seed=seed),
                   audit={k:rec[k] for k in ('input_sha256','rng_seed_seq','rng_state_sha256')},
                   event_ticks=rec['event_ticks'],events_per_cell=rec['events_per_cell'],
                   input_file_sha256=common.sha(out/'inputs'/base),counts_file_sha256=common.sha(out/'counts'/base),
                   total_spikes=sum(sp.values()),firing_neurons=len(sp),wall_s=.001,
                   activated_mean_rate_hz=float(np.mean([sp[i] for i in cc['activate_nodes']])) if cc['activate_nodes'] else 0.,
                   readouts={name:dict(counts=[sp.get(i,0) for i in c['sets'][name]['node_index']],
                       mean_rate_hz=float(np.mean([sp.get(i,0) for i in c['sets'][name]['node_index']]))) for name in c['readouts']})
            r['audit'].update(initial_state_clean=True,state_sha256=p['pins']['reset_state_sha256'],
                              weight_sha256=p['pins']['deleted_weight_sha256' if cc['zero_edges'] else 'v3_weight_sha256'],parent_weight_unchanged=True)
            rows.append(r)
    refresh_rows(out,rows)
    argv=['timeout','--signal=TERM','3500',sys.executable,'-B',str(ROOT/'scripts/pathA_escape_asym_run.py'),
          '--contract',str(out/'contract.json'),'--contract-sha256',p['pins']['derived_contract_sha256'],
          '--prereg',str(out/'prereg.json'),'--prereg-sha256',PSHA,'--expected-code-sha',CODE,'--out',str(out)]
    launch=dict(argv=argv,runner_argv=argv[5:],source_sha=CODE,wrapper_pid=101,
                start_utc='2026-10-08T00:00:00+00:00',thread_environment=audit.ENV)
    startup=dict(pid=102,parent_pid=100,source_sha=CODE,argv=argv[5:],python=sys.executable,
                 thread_environment=audit.ENV,contract_sha256=p['pins']['derived_contract_sha256'],prereg_sha256=PSHA)
    final=dict(argv=argv,child_pid=100,wrapper_pid=101,start_utc=launch['start_utc'],
               end_utc='2026-10-08T00:00:01+00:00',wall_s=1.,exit_code=0,log='run.log',log_sha256=common.sha(out/'run.log'))
    for file,value in [('launch.json',launch),('startup.json',startup),('exit.json',final)]:write(out/file,value)
    return c,p,out,rows,argv,n


def refresh_rows(out,rows):
    lines=[json.dumps(r,sort_keys=True) for r in rows]
    (out/'runs.jsonl').write_text('\n'.join(lines)+'\n')
    log=[dict(condition=r['condition'],seed=r['seed'],row_sha256=hashlib.sha256(line.encode()).hexdigest())
         for r,line in zip(rows,lines)]
    (out/'run.log').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in log))


def audited(e):
    c,p,out,_,_,n=e
    return audit.audit_rows(c,p,PSHA,CODE,out,n)


def test_positive_static_fixture_and_reporting(evidence):
    c,p,out,_,argv,n=evidence
    common.validate_manifest(c,p)
    rows,counts=audited(evidence)
    audit.check_receipts(out,argv,CODE,100,101,rows)
    report=audit.describe(c,counts)
    assert len(rows)==75 and len(report['absolute_rows'])==75
    assert report['paired']['GF10001']['GFC2']['per_cell_change'][0]['mean']==0
    assert report['paired']['GF10001']['TTMn']['intact_presence'][0]['label']=='consistently observed'


@pytest.mark.parametrize('bad',['bool_seed','float_seed','provenance_seed','source','duplicate','missing','order',
                              'state','weight','input_hash','rng_float','events','readout_bool','population'])
def test_row_identity_state_input_count_failures(evidence,bad):
    _,_,out,rows,_,_=evidence
    r=rows[0]
    if bad=='bool_seed':r['seed']=False
    elif bad=='float_seed':r['seed']=0.0
    elif bad=='provenance_seed':r['provenance']['seed']=False
    elif bad=='source':r['provenance']['code_sha']='c'*40
    elif bad=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif bad=='missing':rows.pop()
    elif bad=='order':rows[0],rows[1]=rows[1],rows[0]
    elif bad=='state':r['audit']['state_sha256']='0'*64
    elif bad=='weight':r['audit']['weight_sha256']='0'*64
    elif bad=='input_hash':r['audit']['input_sha256']='0'*64
    elif bad=='rng_float':r['audit']['rng_seed_seq'][0]=0.0
    elif bad=='events':r['events_per_cell'][0]+=1
    elif bad=='readout_bool':r['readouts']['GF']['counts'][0]=True
    elif bad=='population':r['readouts']['GF']['mean_rate_hz']+=1
    refresh_rows(out,rows)
    with pytest.raises(ValueError):audited(evidence)


@pytest.mark.parametrize('bad',['input_bytes','float_counts','baseline_fired','sham_changed','foreign_file'])
def test_artifact_and_control_failures(evidence,bad):
    c,_,out,rows,_,_=evidence
    r=rows[0]
    if bad=='foreign_file':
        (out/'counts'/'foreign').write_text('foreign')
    elif bad=='input_bytes':
        path=out/'inputs'/(r['condition']+'_s0.npz')
        with np.load(path) as z:arr={k:z[k] for k in z.files}
        arr['packed_events'][0]^=1
        np.savez_compressed(path,**arr);r['input_file_sha256']=common.sha(path)
    else:
        if bad=='baseline_fired':r=next(r for r in rows if r['rate_hz']==0)
        if bad=='sham_changed':r=next(r for r in rows if r['arm']=='empty_deletion_sham')
        path=out/'counts'/(r['condition']+'_s'+str(r['seed'])+'.npz')
        with np.load(path) as z:nn,cc=z['node_index'],z['counts']
        if bad=='float_counts':cc=cc.astype(float)
        else:
            sp=dict(zip(nn.tolist(),cc.tolist()))
            node=c['sets']['TTMn']['node_index'][0]
            sp[node]=sp.get(node,0)+1
            nn=np.array(sorted(sp),np.int32);cc=np.array([sp[i] for i in nn],np.int64)
            r['total_spikes']=sum(sp.values());r['firing_neurons']=len(sp)
            for name in c['readouts']:
                actual=[sp.get(i,0) for i in c['sets'][name]['node_index']]
                r['readouts'][name]=dict(counts=actual,mean_rate_hz=float(np.mean(actual)))
        np.savez_compressed(path,node_index=nn,counts=cc);r['counts_file_sha256']=common.sha(path)
    refresh_rows(out,rows)
    with pytest.raises(ValueError):audited(evidence)


@pytest.mark.parametrize('bad',['float_exit','bool_exit','timeout','argv','pid','log_bytes','log_rows','environment','time'])
def test_receipt_failures(evidence,bad):
    _,_,out,_,argv,_=evidence
    rows,_=audited(evidence)
    final=json.loads((out/'exit.json').read_text())
    if bad=='float_exit':final['exit_code']=0.0
    elif bad=='bool_exit':final['exit_code']=False
    elif bad=='timeout':final['exit_code']=124
    elif bad=='argv':final['argv'][2]='3600'
    elif bad=='pid':final['child_pid']=999
    elif bad=='log_bytes':(out/'run.log').write_text('different')
    elif bad=='log_rows':
        lines=(out/'run.log').read_text().splitlines();(out/'run.log').write_text('\n'.join(lines[:-1])+'\n')
        final['log_sha256']=common.sha(out/'run.log')
    elif bad=='environment':
        startup=json.loads((out/'startup.json').read_text());startup['thread_environment']['CUDA_VISIBLE_DEVICES']='0'
        write(out/'startup.json',startup)
    elif bad=='time':final['wall_s']=3501
    write(out/'exit.json',final)
    with pytest.raises(ValueError):audit.check_receipts(out,argv,CODE,100,101,rows)


def test_no_input_rng_and_external_input_manifest(evidence):
    c,p,_,_,_,_=evidence
    common.validate_manifest(c,p)
    no=next(cc for cc in c['conditions'] if cc['rate_hz']==0)
    arrays,record=common.input_data(no,3,c['protocol'])
    exact,independent=audit.independent_input(no,3,c['protocol'])
    assert record==independent and record['rng_seed_seq']==[3,0,0]
    assert record['input_sha256']==hashlib.sha256(b'none').hexdigest()
    assert len(arrays['packed_events'])==0
    p['expected_inputs'][no['id']+'_s3']['rng_seed_seq'][1]=2000
    with pytest.raises(ValueError):common.validate_manifest(c,p)


def test_changed_design_or_irrelevant_protocol_is_refused(evidence):
    c,p,*_=evidence
    for field,value in [('shuffled_input','forbidden'),('timing_probe_set','forbidden')]:
        bad=copy.deepcopy(c);bad['protocol'][field]=value
        with pytest.raises(ValueError):common.validate_manifest(bad,p)
    bad=copy.deepcopy(c);bad['protocol']['compute']['budget_wall_hours']=4
    with pytest.raises(ValueError):common.validate_manifest(bad,p)
    bad=copy.deepcopy(c);bad['conditions'][0]['rate_hz']=199
    pp=copy.deepcopy(p);pp['conditions']=bad['conditions']
    with pytest.raises(ValueError):common.validate_manifest(bad,pp)


def test_duplicate_json_keys_refused(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"seed":0,"seed":1}')
    with pytest.raises(ValueError):common.strict_json(p)
    with pytest.raises(ValueError):audit.parse(p.read_text())


def test_external_source_and_document_pin_failure_before_launch(evidence,monkeypatch,tmp_path):
    c,p,out,*_=evidence
    with pytest.raises(ValueError,match='clean externally pinned'):
        common.verify_sources(p,CODE)
    # The frozen pin is checked against the historical engine it was accepted on.
    pinned=engine_source_root(tmp_path/'pinned_engine_src',PINNED_ENGINE_COMMIT)
    assert engine_digest(pinned)==FROZEN_ENGINE_SHA256==p['pins']['engine_sha256']
    monkeypatch.setattr(base,'ROOT',pinned)
    monkeypatch.setattr('subprocess.check_output',lambda cmd,**kwargs: CODE+'\n' if 'rev-parse' in cmd else '')
    pp=copy.deepcopy(p)
    pp['pins']['source_files_sha256']={'scripts/pathA_run.py':common.sha(pinned/'scripts/pathA_run.py')}
    common.verify_sources(pp,CODE)
    wrong_engine=copy.deepcopy(pp)
    wrong_engine['pins']['engine_sha256']='0'*64
    with pytest.raises(ValueError,match='canonical engine identity differs'):
        common.verify_sources(wrong_engine,CODE)
    pp['pins']['source_files_sha256']['scripts/pathA_run.py']='0'*64
    with pytest.raises(ValueError,match='source file differs'):
        common.verify_sources(pp,CODE)
    write(out/'contract.json',c);write(out/'prereg.json',p)
    with pytest.raises(ValueError,match='external contract/prereg'):
        common.load_plan(out/'contract.json','0'*64,out/'prereg.json',common.sha(out/'prereg.json'),CODE)



def test_current_head_engine_fails_closed_against_frozen_pin(evidence,monkeypatch,tmp_path):
    """Any escape run must use the pinned engine commit; today's HEAD engine is refused."""
    c,p,out,*_=evidence
    head=engine_source_root(tmp_path/'head_engine_src','HEAD')
    assert engine_digest(head)==CURRENT_ENGINE_SHA256!=FROZEN_ENGINE_SHA256==p['pins']['engine_sha256']
    monkeypatch.setattr(base,'ROOT',head)
    monkeypatch.setattr('subprocess.check_output',lambda cmd,**kwargs: CODE+'\n' if 'rev-parse' in cmd else '')
    pp=copy.deepcopy(p)
    pp['pins']['source_files_sha256']={'scripts/pathA_run.py':common.sha(head/'scripts/pathA_run.py')}
    with pytest.raises(ValueError,match='canonical engine identity differs'):
        common.verify_sources(pp,CODE)

def test_input_manifest_integer_types_are_independently_checked(evidence):
    c,p,*_=evidence
    first=next(iter(p['expected_inputs'].values()))
    first['rng_seed_seq'][0]=0.0
    with pytest.raises(ValueError,match='types malformed'):
        common.validate_manifest(c,p)
    with pytest.raises(ValueError,match='types malformed'):
        audited(evidence)


@pytest.mark.parametrize('bad',['baseline','accepted','activation','sham'])
def test_runtime_stops_on_control_failure_without_a_brain(evidence,bad):
    c,_,out,_,_,n=evidence
    selector={'baseline':lambda x:x['rate_hz']==0,'accepted':lambda x:x['id']=='GF10001_200_intact',
              'activation':lambda x:x['id']=='GF10001_200_intact',
              'sham':lambda x:x['arm']=='empty_deletion_sham'}[bad]
    condition=next(x for x in c['conditions'] if selector(x))
    raw=audit.sparse(out/'counts'/(condition['id']+'_s0.npz'),n)
    total=np.zeros(n,np.int64)
    for i,count in raw.items():total[i]=count
    prepared_runner.check_runtime_controls(condition,total,out,0,{condition['id']+'_s0':raw})
    if bad=='baseline':total[0]=1
    elif bad=='accepted':total[c['sets']['TTMn']['node_index'][0]]+=1
    elif bad=='activation':total[condition['activate_nodes'][0]]=0
    else:total[c['sets']['TTMn']['node_index'][0]]+=1
    with pytest.raises(ValueError):prepared_runner.check_runtime_controls(condition,total,out,0,{condition['id']+'_s0':raw})


@pytest.mark.parametrize('bad',['extra_edge','missing_edge','nonzero','wrong_base','wrong_deleted','shape','dtype'])
def test_exact_weight_copy_failures(bad):
    base=np.ones(14541,np.float32);base[0]=19.25;base[14539]=5.5
    deleted=base.copy();deleted[[0,14539]]=0
    pins=dict(v3_weight_sha256=common.weight_hash(base),deleted_weight_sha256=common.weight_hash(deleted),
              weights_npy_sha256=common.npy_hash(base),deleted_weights_npy_sha256=common.npy_hash(deleted))
    common.check_weight_copy(base,deleted,pins)
    audit.check_deleted_weights(base,deleted,pins)
    if bad=='extra_edge':deleted[1]=0
    elif bad=='missing_edge':deleted[0]=base[0]
    elif bad=='nonzero':deleted[0]=.1
    elif bad=='wrong_base':pins['v3_weight_sha256']='0'*64
    elif bad=='wrong_deleted':pins['deleted_weight_sha256']='0'*64
    elif bad=='shape':deleted=deleted[:-1]
    elif bad=='dtype':deleted=deleted.astype(np.float64)
    with pytest.raises(ValueError):common.check_weight_copy(base,deleted,pins)
    if bad!='wrong_base':
        with pytest.raises(ValueError):audit.check_deleted_weights(base,deleted,pins)


@pytest.mark.parametrize('bad',['bool_edge','float_edge','body','contacts','weight','zero_subset','sham_copy','reference_coverage','reference_counts'])
def test_typed_edges_and_control_binding(evidence,bad):
    c,p,out,rows,_,_=evidence
    if bad=='bool_edge':p['direct_edges'][0]['edge_index']=False
    elif bad=='float_edge':p['direct_edges'][0]['edge_index']=0.0
    elif bad=='body':p['direct_edges'][0]['pre_body']=10010
    elif bad=='contacts':p['direct_edges'][0]['contacts']=71
    elif bad=='weight':p['direct_edges'][0]['weight']=19
    elif bad=='zero_subset':
        cc=next(x for x in c['conditions'] if x['zero_edges']);cc['zero_edges']=[0];p['conditions']=copy.deepcopy(c['conditions'])
    elif bad=='sham_copy':
        cc=next(x for x in c['conditions'] if x['arm']=='empty_deletion_sham');cc['weight_copy']=False;p['conditions']=copy.deepcopy(c['conditions'])
    elif bad=='reference_coverage':
        p['accepted_reference']['counts_sha256'].pop(next(iter(p['accepted_reference']['counts_sha256'])))
        with pytest.raises(ValueError):common.check_reference_coverage(c,p)
        return
    elif bad=='reference_counts':
        _,counts=audited(evidence)
        refs={cc['id']+'_s'+str(seed):copy.deepcopy(counts[cc['id'],seed]) for cc in c['conditions'] if cc['arm'] in ('intact','direct_activation_control') for seed in cc['seeds']}
        audit.check_accepted_controls(c,counts,refs)
        refs['GF10001_200_intact_s0'][c['sets']['TTMn']['node_index'][0]]+=1
        with pytest.raises(ValueError):audit.check_accepted_controls(c,counts,refs)
        return
    with pytest.raises(ValueError):common.validate_manifest(c,p)


def test_static_dry_run_cannot_launch_or_create_output(evidence,monkeypatch,capsys):
    import pathA_escape_asym_launch as launcher
    c,p,out,*_=evidence
    target=out/'never-created'
    monkeypatch.setattr(sys,'argv',['launcher','--contract','c','--contract-sha256','csha','--prereg','p','--prereg-sha256','psha','--expected-code-sha',CODE,'--out',str(target),'--accepted-runs',str(out),'--dry-run'])
    monkeypatch.setattr(common,'load_plan',lambda *args:(c,p))
    monkeypatch.setattr(common,'static_assets',lambda *args:None)
    monkeypatch.setattr(common,'require_environment',lambda:None)
    monkeypatch.setenv('NEUROFLY_GRAPH_DIR','g');monkeypatch.setenv('NEUROFLY_CONNECTOME_DIR','n')
    def forbidden(*args,**kwargs):raise AssertionError('dynamic process forbidden in dry run')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    assert launcher.main()==0 and not target.exists()
    assert 'NO DYNAMICS' in capsys.readouterr().out


@pytest.mark.parametrize('bad',['destination','source_body','target_body','weight','extra_direct','typed_edge'])
def test_static_graph_edge_gate(bad):
    ids=np.arange(147867,dtype=np.int64);ids[0]=10001;ids[6]=10010;ids[144432]=800146;ids[147865]=804642
    ptr=np.full(147868,14540,dtype=np.int64);ptr[0]=0;ptr[1:7]=1
    post=np.full(14540,2,dtype=np.int32);post[0]=144432;post[14539]=147865
    weight=np.ones(14540,np.float32);weight[0]=19.25;weight[14539]=5.5
    edges=copy.deepcopy(common.DIRECT_EDGES)
    common.check_edges(ptr,post,ids,weight,edges)
    if bad=='destination':post[14539]=144432
    elif bad=='source_body':ids[0]=10010
    elif bad=='target_body':ids[147865]=800146
    elif bad=='weight':weight[0]=19
    elif bad=='extra_direct':post[1]=144432
    elif bad=='typed_edge':edges[0]['edge_index']=False
    with pytest.raises(ValueError):common.check_edges(ptr,post,ids,weight,edges)


def test_missing_input_manifest_and_reference_artifact_are_rejected(evidence):
    c,p,out,*_=evidence
    bad=copy.deepcopy(p);bad.pop('expected_inputs')
    with pytest.raises(ValueError):common.validate_manifest(c,bad)
    with pytest.raises(OSError):common.accepted_references(p,out/'missing-reference')
