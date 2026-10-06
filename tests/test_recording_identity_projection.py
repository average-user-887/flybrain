"""Recording identity namespaces preserve scientific values and original proof."""
import copy,gzip,hashlib,json,stat
from pathlib import Path
import pytest
from neurofly.recording import (_Ordinals,_RecordingProjection,_line,RunRecorder,
                                frame_from_telemetry,source_telemetry_from_frame,
                                read_recording,list_recordings,public_recording_artifacts,FORMAT)
from neurofly_daemon import ContinuousExperimentRunner
from observation_envelopes import validate_observation_envelope


def packet(tmp_path):
    runner=ContinuousExperimentRunner(initial_paradigm='t-maze',output_dir=tmp_path,
                                     checkpoint_interval=1e9,trial_length_s=60)
    return runner,runner._assemble_telemetry(runner._last_step_result)

def test_typed_observation_aliases_values_roundtrip_and_no_alias_mutation(tmp_path):
    runner,p=packet(tmp_path/'runner');before=copy.deepcopy(p)
    projection=_RecordingProjection(_Ordinals('s'))
    f=frame_from_telemetry(p,_Ordinals('unused'),projection,0);f.update(i=0,k='f',activity=None,spikes=None)
    assert p==before
    obs=f['observation'];validate_observation_envelope(obs)
    assert obs['segment_id']==f['segment_id']==f['observation_lifecycle']['segment_id']
    assert obs['presentation_id']==obs['segment_id']+':0'
    assert obs['config_id']==f['observation_lifecycle']['producer']['config_id']=='segment:'+obs['segment_id']
    assert obs['override']['manifest_run_id']==obs['identity']['run_id']==f['identity']['run_id']
    assert obs['records']==p['observation']['records'] and obs['evidence']==p['observation']['evidence']
    assert source_telemetry_from_frame(f,projection.sidecar())==p
    f['observation']['records'].clear();assert p==before

@pytest.mark.parametrize('field,value',[('zero',0),('null',None),('false',False),('unavailable',{'value':None,'available':False,'reason':'not_observed'})])
def test_null_zero_false_unavailable_survive(field,value):
    p={'segment_id':'random','observation':{'records':{field:value}}}
    f=frame_from_telemetry(p,_Ordinals('s'));assert f['observation']['records'][field]==value
    assert type(f['observation']['records'][field]) is type(value)

@pytest.mark.parametrize('change',['metric','provenance','invalidity','lifecycle','scientific_timestamp','brain_science'])
def test_actual_science_or_provenance_changes_projected_bytes(tmp_path,change):
    _,p=packet(tmp_path/'runner');q=copy.deepcopy(p)
    if change=='metric':q['observation']['records']['arm_entry_preference_index']['counts']['cs_plus']=1
    if change=='provenance':q['observation']['provenance']['gf_source']='connectome'
    if change=='invalidity':q['result_validity']['state']='invalidated'
    if change=='lifecycle':q['observation_lifecycle']['phase']='hold'
    if change=='scientific_timestamp':q['observation']['evidence']['entries']['data'].append({'timestamp':123})
    if change=='brain_science':q['brain']['new_science_field']=0
    assert _line(frame_from_telemetry(p,_Ordinals('s')))!=_line(frame_from_telemetry(q,_Ordinals('s')))


def publication(p,path=None):
    original=copy.deepcopy(p['observation'])
    if path:original['provenance']['source_location']=str(path)
    raw=json.dumps(original,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    digest=hashlib.sha256(raw).hexdigest()
    key={k:original['identity'][k] for k in ('daemon_run_id','run_id','instance_id')}
    key.update(segment_id=original['segment_id'],presentation_id=original['presentation_id'])
    return {'observation':original,'identity':original['identity'],'observation_key':key,
            'payload_sha256':digest,'receipt':{'durable':True,'payload_sha256':digest,
              'observation_key':key,'file':'learning.jsonl','line':1,'offset':0,'idempotent':False}}

@pytest.mark.parametrize('redacted',[False,True])
def test_receipt_pairing_source_private_archive_and_public_exclusion(tmp_path,redacted):
    runner,p=packet(tmp_path/'runner');terminal=publication(p,tmp_path/'private-name' if redacted else None)
    p['observation_publication']['last_terminal']=terminal
    runner._assemble_telemetry=lambda _:copy.deepcopy(p)
    recorder=RunRecorder(runner,tmp_path/'public'/'actual.nfrec');summary=recorder.close()
    sidecar=json.loads((tmp_path/'public'/'actual.nfrec.json').read_text());source=sidecar['projection']['source_evidence'][0]
    assert sidecar['recording']['sha256']==summary['sha256']
    assert source['source_payload_redacted'] is redacted and source['verified_for_replay'] is False
    f=read_recording(tmp_path/'public'/'actual.nfrec')['frames'][0]
    projected=f['observation_publication']['last_terminal']
    assert 'receipt' not in projected and 'payload_sha256' not in projected
    assert projected['recording_source_evidence']['verified_for_replay'] is False
    assert source_telemetry_from_frame(f,sidecar['projection'])['observation_publication']['last_terminal']['receipt']==terminal['receipt']
    if redacted:
        directory=tmp_path/'public'/'.source-evidence-private';archive=directory/'actual.nfrec.source.json'
        assert stat.S_IMODE(directory.stat().st_mode)==0o700 and stat.S_IMODE(archive.stat().st_mode)==0o600
        private=json.loads(archive.read_text());assert private['recording_sha256']==summary['sha256']
        exact=private['source_evidence'][0]['source'];assert exact==terminal
        assert str(tmp_path) not in (tmp_path/'public'/'actual.nfrec.json').read_text()
    else:exact=source['source'];assert not (tmp_path/'public'/'.source-evidence-private').exists()
    digest=hashlib.sha256(json.dumps(exact['observation'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert digest==exact['payload_sha256']==exact['receipt']['payload_sha256']
    assert [r['name'] for r in list_recordings(tmp_path/'public')]==['actual.nfrec']
    assert [p.name for p in public_recording_artifacts(tmp_path/'public'/'actual.nfrec')]==['actual.nfrec','actual.nfrec.json','actual.nfrec.done']


def test_old_v1_still_readable(tmp_path):
    frame={'k':'f','i':0,'step':0};line=_line(frame)
    path=tmp_path/'legacy.nfrec'
    with gzip.open(path,'wb') as f:
        f.write(_line({'k':'header','format':FORMAT,'version':1}));f.write(line)
        f.write(_line({'k':'end','frames':1,'events':0,'frames_sha256':hashlib.sha256(line).hexdigest()}))
    assert read_recording(path,require_finished=False)['frames']==[frame]

def test_transitions_and_source_alias_crosswalk_preserve_relationships():
    projection=_RecordingProjection(_Ordinals('s'))
    first={'segment_id':'old','identity':{'brain_id':'brain1','instance_id':'brain1','run_id':'run1','daemon_run_id':'daemon'},
           'observation':{'config_id':'segment:old','segment_id':'old','presentation_id':'old:3'}}
    a=projection.project(first,0)
    second={'segment_id':'new','identity':{'brain_id':'brain2','instance_id':'instance2','run_id':'run2','daemon_run_id':'daemon'},
            'transition':{'ended_segment':'old'},'observation':{'config_id':'segment:new','segment_id':'new','presentation_id':'new:0'},
            'observation_lifecycle':{'segment_id':'new','producer':{'config_id':'segment:new'}}}
    b=projection.project(second,1)
    assert b['segment_id']=='s1' and b['transition']['ended_segment']==a['segment_id']=='s0'
    assert a['identity']['brain_id']==a['identity']['instance_id']
    assert b['identity']['brain_id']!=b['identity']['instance_id']
    assert a['identity']['daemon_run_id']==b['identity']['daemon_run_id']
    b.update(k='f',i=1);assert source_telemetry_from_frame(b,projection.sidecar())==second


def test_repeated_receipt_stores_original_once_and_no_full_telemetry_copy(tmp_path):
    _,p=packet(tmp_path/'runner');p['observation_publication']['last_terminal']=publication(p)
    projection=_RecordingProjection(_Ordinals('s'))
    projection.project(p,0);projection.project(p,1)
    assert len(projection.source_evidence)==1
    assert all('telemetry' not in row for row in projection.sidecar()['source_frames'])


def test_known_wall_clock_moves_but_probe_scientific_timestamp_stays():
    a={'timestamp':1,'brain':{'last_saved':1,'probe':{'timestamp':12}}}
    b=copy.deepcopy(a);b['timestamp']=2;b['brain']['last_saved']=2
    assert _line(frame_from_telemetry(a,_Ordinals('s')))==_line(frame_from_telemetry(b,_Ordinals('s')))
    b['brain']['probe']['timestamp']=13
    assert _line(frame_from_telemetry(a,_Ordinals('s')))!=_line(frame_from_telemetry(b,_Ordinals('s')))

def test_sparse_operational_deltas_restore_changes_and_removed_paths():
    projection=_RecordingProjection(_Ordinals('s'));packets=[];frames=[]
    for i in range(4):
        p={'timestamp':i,'timing':{'step':i,'constant':10},'segment_id':'segment','observation_publication':{'last_terminal':None}}
        if i<2:p['brain']={'last_saved':i}
        if i==0:p['status']={'phase':'ready','message':'initial'}
        elif i==1:p['status']={'phase':'running'}
        packets.append(p);f=projection.project(p,i);f.update(i=i,k='f');frames.append(f)
    sidecar=projection.sidecar()
    for p,f in zip(packets,frames):assert source_telemetry_from_frame(f,sidecar)==p
    assert any(p.get('remove_patch') for row in sidecar['source_frames'] for p in row['operational_fields'])
    paths=sidecar['operational_paths'];constant=paths.index(['timing','constant'])
    assert sum(p['p']==constant for row in sidecar['source_frames'] for p in row['operational_fields'])==1
