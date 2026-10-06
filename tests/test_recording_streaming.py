"""Bounded provenance writer: lossless rows, mutation-safe dedup and failure markers."""
import copy,hashlib,json,stat,tracemalloc,gc
from pathlib import Path
import pytest
from neurofly import recording as r
from tests.test_recording_identity_projection import packet,publication


def _writer(tmp_path):
    runner,p=packet(tmp_path/'runner')
    p['observation_publication']['last_terminal']=publication(p,tmp_path/'secret')
    runner._assemble_telemetry=lambda _:copy.deepcopy(p)
    return runner,p,r.RunRecorder(runner,tmp_path/'public'/'stream.nfrec')


def test_writer_streams_all_rows_and_roundtrips_original(tmp_path):
    runner,p,w=_writer(tmp_path)
    packets=[copy.deepcopy(p)]
    for i in range(1,13):
        p['timestamp']=i;p['timing']['new_wall']=i;p['observation']['records']['test_science']={'value':i,'available':False}
        packets.append(copy.deepcopy(p));runner.total_steps=i;assert w.capture(runner)
    assert isinstance(w._projection.sources,r._DiskRows)
    assert len(w._projection.sources)==13 and w._projection.wrapper_hashes==1 and w._projection.payload_hashes==1
    assert len(w._projection.source_evidence)==len(w._projection.private_evidence)==1
    before=list(w._projection.sources);w.close()
    meta=json.loads(w._sidecar_path.read_text());frames=r.read_recording(w.path)['frames']
    assert meta['projection']['source_frames']==before
    for p,f in zip(packets,frames):
        assert r.source_telemetry_from_frame(f,meta['projection'])==r._json_safe(r.redact_local(p))
    state=json.loads((w._projection.spool.path/'status.json').read_text())
    assert state['state']=='complete' and state['recording_sha256']==r._file_digest(w.path)
    assert {x.name for x in r.public_recording_artifacts(w.path)}=={'stream.nfrec','stream.nfrec.json','stream.nfrec.done'}
    assert stat.S_IMODE(w._projection.spool.path.stat().st_mode)==0o700
    assert all(stat.S_IMODE(x.stat().st_mode)==0o600 for x in w._projection.spool.path.iterdir() if x.is_file())


def test_mutating_wrapper_with_same_claimed_digest_requires_new_proof(tmp_path):
    runner,p,w=_writer(tmp_path);terminal=p['observation_publication']['last_terminal'];old=copy.deepcopy(terminal)
    old_digest=terminal['payload_sha256'];terminal['observation']['records']['real_changed_science']=0
    assert terminal['payload_sha256']==old_digest  # malicious/stale digest cannot be dedup authority
    runner.total_steps=1;w.capture(runner);assert terminal!=old
    assert w._projection.wrapper_hashes==2 and len(w._projection.source_evidence)==2
    w.close();archive=w.path.parent/'.source-evidence-private'/(w.path.name+'.source.json')
    proof=json.loads(archive.read_text())['source_evidence']
    assert [x['source'] for x in proof]==[old,terminal]


@pytest.mark.parametrize('stage',['append','index','sync','sidecar','private','status'])
def test_failures_keep_partial_history_and_never_certify(tmp_path,monkeypatch,stage):
    runner,p,w=_writer(tmp_path)
    def fail(*a,**k):raise OSError('injected evidence failure')
    if stage=='append':
        monkeypatch.setattr(w._projection.sources,'append',fail)
        runner.total_steps=1
        with pytest.raises(OSError):w.capture(runner)
    elif stage=='index':
        monkeypatch.setattr(w._projection.spool,'note',fail)
        p['observation_publication']['last_terminal']['receipt']['line']=2;runner.total_steps=1
        with pytest.raises(OSError):w.capture(runner)
    else:
        if stage=='sync':monkeypatch.setattr(w._projection.spool,'sync',fail)
        if stage in ('sidecar','private'):
            original=r._stream_json
            def stream(handle,value):
                if (stage=='sidecar' and 'recording' in value if isinstance(value,dict) else False):fail()
                if (stage=='private' and 'recording_sha256' in value if isinstance(value,dict) else False):fail()
                return original(handle,value)
            monkeypatch.setattr(r,'_stream_json',stream)
        if stage=='status':
            original=w._projection.spool.status
            def status(state,**kw):
                if state=='complete':fail()
                return original(state,**kw)
            monkeypatch.setattr(w._projection.spool,'status',status)
        with pytest.raises(OSError):w.close()
    assert not w._done_path.exists() and not r.list_recordings(w.path.parent)
    assert w._projection.sources.path.is_file()
    if w._cleanup_thread is not None: w._cleanup_thread.join(timeout=5)
    marker=json.loads((w._projection.spool.path/'status.json').read_text())
    assert marker['state']=='failed'
    w.abort('test cleanup')
    if w._cleanup_thread is not None: w._cleanup_thread.join(timeout=5)
    assert w._raw.closed and w.closed
    with pytest.raises(RuntimeError):w.close()


def test_abort_preserves_rows_and_marks_failed_without_receipt_change(tmp_path):
    _,_,w=_writer(tmp_path);w.abort('explicit cancellation')
    w._cleanup_thread.join(timeout=5)
    assert json.loads((w._projection.spool.path/'status.json').read_text())['state']=='failed'
    assert len(list(w._projection.sources))==1 and not w._done_path.exists()


def test_distinct_wrappers_disk_index_and_memory_bounded(tmp_path):
    projection=r._RecordingProjection(r._Ordinals('s'),r._ProjectionSpool(tmp_path/'fake.nfrec'))
    p={'timestamp':0,'segment_id':'stable','observation_publication':{'last_terminal':{
        'observation':{'records':{'metric':0}},'payload_sha256':'unchanged_stale_digest','receipt':{'durable':True}}}}
    tracemalloc.start()
    for i in range(2000):
        p['timestamp']=i;p['observation_publication']['last_terminal']['observation']['records']['metric']=i
        projection.project(p,i)
        if i==199:gc.collect();early=tracemalloc.get_traced_memory()[0]
    gc.collect();late,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert late-early<350_000
    assert len(projection.sources)==len(projection.source_evidence)==2000
    assert len(projection._wrapper_cache)==8 and projection._evidence_seen=={}
    assert len(projection._last_patches)<=4
    assert len(list(projection.source_evidence))==2000


def test_content_cache_and_delta_equality_preserve_false_zero_null(tmp_path):
    projection=r._RecordingProjection(r._Ordinals('s'),r._ProjectionSpool(tmp_path/'types.nfrec'))
    for i,value in enumerate((False,0,None,0.,-0.)):
        p={'timing':{'flag':value},'observation_publication':{'last_terminal':{
            'observation':{'records':{'metric':value}},'payload_sha256':'same','receipt':{'durable':True}}}}
        projection.project(p,i)
    assert len(projection.source_evidence)==projection.wrapper_hashes==projection.payload_hashes==5
    rows=list(projection.sources)
    assert len(rows)==5 and all(row['operational_fields'] for row in rows)
    assert [row['source']['observation']['records']['metric'] for row in projection.source_evidence]==[False,0,None,0.,-0.]


def test_private_permissions_unsupported_refuses_before_proof_append(tmp_path,monkeypatch):
    original=r._require_private
    def refusal(path,mode):
        if mode==0o700:raise PermissionError('unsupported private permissions')
        original(path,mode)
    monkeypatch.setattr(r,'_require_private',refusal)
    with pytest.raises(PermissionError):r._ProjectionSpool(tmp_path/'bad.nfrec')
    assert not list(tmp_path.rglob('private.jsonl'))


def test_disk_digest_connection_closes_on_failure(tmp_path):
    import sqlite3
    with pytest.raises(RuntimeError):
        with r._digest_db(tmp_path/'index.sqlite') as db:
            db.execute('CREATE TABLE sample (id INTEGER)')
            raise RuntimeError('index transaction failed')
    with pytest.raises(sqlite3.ProgrammingError):db.execute('SELECT 1')


def test_valid_json_row_loss_or_tamper_refuses_completion(tmp_path):
    _,_,w=_writer(tmp_path)
    data=w._projection.sources.path.read_bytes()
    w._projection.sources.path.write_bytes(data.replace(b'"frame":0',b'"frame":1'))
    with pytest.raises(RuntimeError,match='spool count/digest'):w.close()
    assert not w._done_path.exists()
    if w._cleanup_thread is not None:w._cleanup_thread.join(timeout=5)
