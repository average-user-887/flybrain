"""Regression probes for authenticated proof refs and abort/close linearization."""
import copy
import hashlib
import json
import sqlite3
import threading
import pytest
from neurofly import recording as r
from tests.test_recording_streaming import _writer


def _evict_wrapper(runner, packet, writer):
    original = copy.deepcopy(packet['observation_publication']['last_terminal'])
    digest = hashlib.sha256(r._line(original)).hexdigest()
    for step in range(1, 10):
        packet['observation_publication']['last_terminal']['observation']['records']['audit_index_variant'] = step
        runner.total_steps = step
        writer.capture(runner)
    assert len(writer._projection._wrapper_cache) == 8
    packet['observation_publication']['last_terminal'] = original
    runner.total_steps = 10
    return digest


@pytest.mark.parametrize('ref', ['e-does-not-exist', 'e1'])
def test_tampered_sqlite_ref_cannot_select_absent_or_wrong_proof(tmp_path, ref):
    runner, packet, writer = _writer(tmp_path)
    digest = _evict_wrapper(runner, packet, writer)
    with sqlite3.connect(writer._projection.spool.index) as db:
        db.execute('UPDATE evidence SET ref=? WHERE digest=?', (ref, digest))
    with pytest.raises(RuntimeError, match='index authentication'):
        writer.capture(runner)
    if writer._cleanup_thread is not None:
        writer._cleanup_thread.join(5)
    assert writer.closed and not writer.completed and not writer._done_path.exists()
    assert json.loads((writer._projection.spool.path/'status.json').read_text())['state'] == 'failed'
    with pytest.raises(RuntimeError, match='aborted'):
        writer.close()


def test_tampered_index_digest_or_auth_cannot_impersonate_another_proof(tmp_path):
    runner, packet, writer = _writer(tmp_path)
    digest = _evict_wrapper(runner, packet, writer)
    with sqlite3.connect(writer._projection.spool.index) as db:
        other_auth = db.execute('SELECT auth FROM evidence WHERE ref=?', ('e1',)).fetchone()[0]
        db.execute('UPDATE evidence SET ref=?,auth=? WHERE digest=?', ('e1', other_auth, digest))
    with pytest.raises(RuntimeError, match='index authentication'):
        writer.capture(runner)
    if writer._cleanup_thread is not None:
        writer._cleanup_thread.join(5)
    assert not writer._done_path.exists()


def test_legitimate_evicted_index_ref_still_resolves_exact_proof(tmp_path):
    runner, packet, writer = _writer(tmp_path)
    _evict_wrapper(runner, packet, writer)
    original = copy.deepcopy(packet['observation_publication']['last_terminal'])
    writer.capture(runner)
    writer.close()
    frame = r.read_recording(writer.path)['frames'][-1]
    meta = json.loads(writer._sidecar_path.read_text())
    assert r.source_telemetry_from_frame(frame, meta['projection'])['observation_publication']['last_terminal'] == r.redact_local(original)
    assert frame['observation_publication']['last_terminal']['recording_source_evidence']['ref'] == 'e0'
    assert writer.artifacts()['certified']


@pytest.mark.parametrize('stage', ['before_status', 'after_status'])
@pytest.mark.parametrize('reason', ['deterministic concurrent cancellation', ''])
def test_abort_wins_before_final_publication_and_status_never_recovers_complete(tmp_path, monkeypatch, stage, reason):
    _, _, writer = _writer(tmp_path)
    original_status = writer._projection.spool.status
    entered, release = threading.Event(), threading.Event()
    errors = []
    def status(state, **fields):
        if state == 'complete':
            if stage == 'after_status':
                original_status(state, **fields)
            entered.set()
            assert release.wait(5)
            if stage == 'after_status':
                return
        return original_status(state, **fields)
    monkeypatch.setattr(writer._projection.spool, 'status', status)
    def close():
        try:
            writer.close()
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=close)
    thread.start()
    try:
        assert entered.wait(5)
        writer.abort(reason)
        assert writer.closed and writer.aborted == reason and not writer.completed
        assert r.recording_writer_invalid_reason(writer.path) is not None
    finally:
        release.set()
        thread.join(5)
        writer._cleanup_thread.join(5)
    assert not thread.is_alive()
    assert len(errors) == 1 and 'aborted' in str(errors[0])
    assert not writer.completed and not writer._done_path.exists() and not writer.artifacts()['certified']
    assert json.loads((writer._projection.spool.path/'status.json').read_text())['state'] == 'failed'
    assert not r.list_recordings(writer.path.parent)
    with pytest.raises(ValueError):
        r.public_recording_artifacts(writer.path)


def test_abort_wins_during_done_write_without_waiting_for_filesystem(tmp_path, monkeypatch):
    _, _, writer = _writer(tmp_path)
    original_write = r._write_durably
    entered, release, aborted = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def write(path, data):
        if path.name.endswith('.done.partial'):
            entered.set()
            assert release.wait(5)
        return original_write(path, data)
    monkeypatch.setattr(r, '_write_durably', write)
    def close():
        try:
            writer.close()
        except Exception as exc:
            errors.append(exc)
    closing = threading.Thread(target=close)
    closing.start()
    assert entered.wait(5)
    aborting = threading.Thread(target=lambda: (writer.abort(''), aborted.set()))
    aborting.start()
    try:
        assert aborted.wait(.5), 'abort waited for finalization I/O'
        assert writer.closed and writer.aborted == '' and not writer.completed
        assert r.recording_writer_invalid_reason(writer.path) is not None
    finally:
        release.set()
        closing.join(5)
        aborting.join(5)
        writer._cleanup_thread.join(5)
    assert not closing.is_alive() and not aborting.is_alive()
    assert len(errors) == 1 and 'aborted' in str(errors[0])
    assert not writer.completed and not writer.artifacts()['certified']
    assert json.loads((writer._projection.spool.path/'status.json').read_text())['state'] == 'failed'


def test_simultaneous_finalizers_write_one_footer_and_return_same_summary(tmp_path):
    _, _, writer = _writer(tmp_path)
    barrier = threading.Barrier(3)
    results, errors = [], []
    def close():
        barrier.wait(timeout=5)
        try:
            results.append(writer.close())
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(5)
    assert all(not thread.is_alive() for thread in threads) and not errors
    assert len(results) == 2 and results[0] == results[1]
    recording = r.read_recording(writer.path)
    assert recording['end']['frames'] == len(recording['frames']) == 1


@pytest.mark.parametrize('stage', ['status', 'done_write', 'done_rename', 'done_dir_fsync'])
def test_revocation_gate_contains_no_stalled_filesystem_work(tmp_path, monkeypatch, stage):
    _, _, writer = _writer(tmp_path)
    entered, release, aborted = threading.Event(), threading.Event(), threading.Event()
    errors = []
    status, write, replace, sync = writer._projection.spool.status, r._write_durably, r._replace, r._fsync_directory
    def pause():
        entered.set()
        assert release.wait(5)
    def wrapped_status(state, **fields):
        if stage == 'status' and state == 'complete':
            pause()
        return status(state, **fields)
    def wrapped_write(path, data):
        if stage == 'done_write' and path.name.endswith('.done.partial'):
            pause()
        return write(path, data)
    def wrapped_replace(src, dst):
        result = replace(src, dst)
        if stage == 'done_rename' and dst == writer._done_path:
            pause()  # an in-flight rename may already be visible on disk
        return result
    def wrapped_sync(directory):
        if stage == 'done_dir_fsync' and writer._done_path.exists():
            pause()
        return sync(directory)
    monkeypatch.setattr(writer._projection.spool, 'status', wrapped_status)
    monkeypatch.setattr(r, '_write_durably', wrapped_write)
    monkeypatch.setattr(r, '_replace', wrapped_replace)
    monkeypatch.setattr(r, '_fsync_directory', wrapped_sync)
    def close():
        try:
            writer.close()
        except Exception as exc:
            errors.append(exc)
    closing = threading.Thread(target=close)
    closing.start()
    assert entered.wait(5)
    aborting = threading.Thread(target=lambda: (writer.abort('', defer_cleanup=True), aborted.set()))
    aborting.start()
    try:
        assert aborted.wait(.5), 'revocation waited for a stalled filesystem operation'
        assert writer._completion_lock.acquire(blocking=False), 'filesystem owns revocation gate'
        writer._completion_lock.release()
        assert writer.closed and not writer.completed and writer.aborted == ''
        assert 'revoked' in r.recording_writer_invalid_reason(writer.path)
    finally:
        release.set()
        closing.join(5)
        aborting.join(5)
        writer._cleanup_thread.join(5)
    assert not closing.is_alive() and not writer._cleanup_thread.is_alive()
    assert len(errors) == 1 and 'aborted' in str(errors[0])
    assert not writer.completed and not writer.artifacts()['certified']
    assert not writer._done_path.exists() and not r.list_recordings(writer.path.parent)
    with pytest.raises(ValueError):
        r.public_recording_artifacts(writer.path)
    assert json.loads((writer._projection.spool.path/'status.json').read_text())['state'] == 'failed'


def test_completed_ack_precedes_later_abort_without_changing_durable_success(tmp_path):
    _, _, writer = _writer(tmp_path)
    summary = writer.close()
    writer.abort('too late', defer_cleanup=True)
    assert writer.completed and not hasattr(writer, 'aborted')
    assert r.recording_writer_invalid_reason(writer.path) is None
    assert writer.artifacts()['certified'] and summary == writer.close()
