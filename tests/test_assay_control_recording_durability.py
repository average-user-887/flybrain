"""Real NF command durability and canceled-writer ownership at C1 boundaries."""
import json
import os
import threading
import time
import zlib

import pytest

from neurofly.recording import RunRecorder
from tests.test_assay_control_transactions import attached, airflow


def with_nf(tmp_path, **kwargs):
    runner, records, drain = attached(tmp_path, **kwargs)
    nf = RunRecorder(runner, tmp_path / 'nf' / 'actual.nfrec')
    runner.recorder = nf
    return runner, records, drain, nf


def finish_threads(tx, nf):
    if tx.get('writer') is not None:
        tx['writer'].join(2)
        assert not tx['writer'].is_alive()
    if nf._cleanup_thread is not None:
        nf._cleanup_thread.join(2)
        assert not nf._cleanup_thread.is_alive()


def test_applied_ack_has_real_nf_fsync_bytes_and_exact_receipt(tmp_path, monkeypatch):
    runner, records, drain, nf = with_nf(tmp_path)
    synced = []
    original = os.fsync
    def spy(fd):
        synced.append(os.readlink(f'/proc/self/fd/{fd}'))
        return original(fd)
    monkeypatch.setattr(os, 'fsync', spy)
    reply = runner.dispatch_command(airflow(23))
    tx = runner._pending_assay_control
    drain.poll_once()
    assert tx['entry']['done'].wait(2)
    assert runner.command_acks[-1]['ack']['applied']
    assert str(nf._partial) in synced
    lines = zlib.decompressobj(31).decompress(nf._partial.read_bytes()).splitlines()
    events = [json.loads(line) for line in lines if json.loads(line).get('k') == 'e']
    assert events == [{'k': 'e', 'kind': 'command', 'step': 0, 'cmd': airflow(23)}]
    assert tx['recording_receipt']['operation_id'] == reply['command_id']
    assert tx['recording_receipt']['command_sha256'] == tx['recording_request']['command_sha256']
    assert tx['recording_receipt']['partial_bytes'] > 0
    nf.abort('test cleanup'); runner.recorder = None; records.close()


@pytest.mark.parametrize('phase', ['before_write', 'during_fsync', 'after_write'])
@pytest.mark.parametrize('exploratory', [False, True])
def test_independent_abort_revokes_command_without_lock_wait(tmp_path, monkeypatch, phase, exploratory):
    runner, records, drain, nf = with_nf(tmp_path, exploratory=exploratory)
    entered, release = threading.Event(), threading.Event()
    if phase == 'before_write':
        original = runner.active_brain.log
        def blocked(kind, **fields):
            if kind == 'assay_control':
                entered.set(); assert release.wait(3)
            return original(kind, **fields)
        monkeypatch.setattr(runner.active_brain, 'log', blocked)
    elif phase == 'during_fsync':
        original = os.fsync
        def blocked(fd):
            if os.readlink(f'/proc/self/fd/{fd}') == str(nf._partial):
                entered.set(); assert release.wait(3)
            return original(fd)
        monkeypatch.setattr(os, 'fsync', blocked)
    else:
        original = nf.durable_command
        def blocked(request):
            receipt = original(request)
            entered.set(); assert release.wait(3)
            return receipt
        monkeypatch.setattr(nf, 'durable_command', blocked)
    reply = runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(1)
    try:
        start = time.monotonic()
        with runner.lock:
            runner._recording_failed(OSError('independent NF failure'))
        assert time.monotonic() - start < 0.5
        assert nf.closed and runner.recorder is None
        assert runner.last_error is not None and runner._state_uncertain
        assert tx['entry']['done'].is_set()
        assert len(runner.command_acks) == 1
        assert runner.command_acks[0]['command_id'] == reply['command_id']
        assert not runner.command_acks[0]['ack']['applied']
        assert nf._last_command_receipt is None
        assert nf._cleanup_thread is not None
        assert runner.lock.acquire(timeout=0.2); runner.lock.release()
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 3})['status'] == 'ok'
    finally:
        release.set(); finish_threads(tx, nf); records.close()
    assert len(runner.command_acks) == 1 and not runner.command_acks[0]['ack']['applied']


def test_blocked_fsync_freezes_sampling_and_refuses_record_ownership_changes(tmp_path, monkeypatch):
    runner, records, drain, nf = with_nf(tmp_path)
    entered, release = threading.Event(), threading.Event(); original = os.fsync
    def blocked(fd):
        if os.readlink(f'/proc/self/fd/{fd}') == str(nf._partial):
            entered.set(); assert release.wait(3)
        return original(fd)
    monkeypatch.setattr(os, 'fsync', blocked)
    runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(1)
    try:
        assert runner.lock.acquire(timeout=0.2); runner.lock.release()
        assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 3})['status'] == 'ok'
        for action in ('record_stop', 'record_start'):
            assert runner.dispatch_command({'action': action, 'name': 'other'})['status'] == 'error'
        assert runner.recorder is nf and not nf.closed
        before = runner.total_steps; runner.step_once(); assert runner.total_steps == before
        assert not tx['entry']['done'].is_set() and not runner.command_acks
    finally:
        release.set(); finish_threads(tx, nf)
    assert runner.command_acks[-1]['ack']['applied']
    nf.abort('test cleanup'); runner.recorder = None; records.close()


def test_timeout_rejects_late_receipt_and_preserves_canceled_name(tmp_path, monkeypatch):
    runner, records, drain, nf = with_nf(tmp_path, exploratory=True)
    entered, release = threading.Event(), threading.Event(); original = os.fsync
    def blocked(fd):
        if os.readlink(f'/proc/self/fd/{fd}') == str(nf._partial):
            entered.set(); assert release.wait(3)
        return original(fd)
    monkeypatch.setattr(os, 'fsync', blocked)
    runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(1)
    try:
        runner.assay_control_timeout_s = 0
        start = time.monotonic(); runner.step_once()
        assert time.monotonic() - start < 0.5
        assert runner._pending_assay_control is None and tx['entry']['done'].is_set()
        assert nf.closed and nf._cleanup_thread.is_alive()
        before = nf._partial.read_bytes()
        with pytest.raises(FileExistsError): RunRecorder(runner, nf.path)
        assert nf._partial.read_bytes() == before
        assert len(runner.command_acks) == 1 and not runner.command_acks[0]['ack']['applied']
    finally:
        release.set(); finish_threads(tx, nf); records.close()
    assert nf._last_command_receipt is None
    assert len(runner.command_acks) == 1 and runner.last_error is not None


@pytest.mark.parametrize('artifact', ['partial', 'final'])
def test_existing_recording_artifacts_are_never_overwritten(tmp_path, artifact):
    runner, records, _, nf = with_nf(tmp_path)
    nf.abort('test cleanup'); runner.recorder = None
    if artifact == 'final':
        nf._partial.rename(nf.path)
        target = nf.path
    else: target = nf._partial
    before = target.read_bytes()
    with pytest.raises(FileExistsError): RunRecorder(runner, nf.path)
    assert target.read_bytes() == before
    records.close()


@pytest.mark.parametrize('field,value', [('operation_id','wrong'), ('command_sha256','wrong'), ('step',999)])
def test_mismatched_real_command_receipt_cannot_acknowledge_applied(tmp_path, monkeypatch, field, value):
    runner, records, drain, nf = with_nf(tmp_path)
    original = nf.durable_command
    def corrupted(request):
        return {**original(request), field: value}
    monkeypatch.setattr(nf, 'durable_command', corrupted)
    runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert tx['entry']['done'].wait(2)
    finish_threads(tx, nf)
    assert len(runner.command_acks) == 1 and not runner.command_acks[0]['ack']['applied']
    assert runner.last_error is not None and runner._state_uncertain
    records.close()


class ComparisonTrapInt(int):
    def __eq__(self, other):
        raise AssertionError('custom receipt comparison must not run')


class ComparisonTrapDict(dict):
    def __eq__(self, other):
        raise AssertionError('custom receipt comparison must not run')


class ComparisonTrapStr(str):
    __hash__ = str.__hash__

    def __eq__(self, other):
        raise AssertionError('custom receipt comparison must not run')


@pytest.mark.parametrize('change', [
    lambda r: {**r, 'step': True},
    lambda r: {**r, 'step': 1.0},
    lambda r: {**r, 'events': True},
    lambda r: {**r, 'events': 1.0},
    lambda r: {**r, 'partial_bytes': float(r['partial_bytes'])},
    lambda r: {**r, 'step': ComparisonTrapInt(r['step'])},
    lambda r: ComparisonTrapDict(r),
    lambda r: {**r, 'operation_id': ComparisonTrapStr(r['operation_id'])},
    lambda r: {**{k: v for k, v in r.items() if k != 'operation_id'},
               ComparisonTrapStr('operation_id'): r['operation_id']},
    lambda r: {k: v for k, v in r.items() if k != 'events'},
    lambda r: {**r, 'unexpected': 0},
    lambda r: {**r, 'command_sha256': None},
], ids=['bool-step', 'float-step', 'bool-events', 'float-events', 'float-bytes',
        'custom-int', 'custom-dict', 'custom-str-value', 'custom-str-key',
        'missing-field', 'extra-field', 'null-digest'])
def test_malformed_real_receipt_is_refused_before_comparison(tmp_path, monkeypatch, change):
    runner, records, drain, nf = with_nf(tmp_path)
    runner.step_once()  # A bool/float alias would equal this valid step of one.
    original = nf.durable_command
    monkeypatch.setattr(nf, 'durable_command', lambda request: change(original(request)))
    runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert tx['entry']['done'].wait(2)
    finish_threads(tx, nf)
    assert len(runner.command_acks) == 1 and not runner.command_acks[0]['ack']['applied']
    assert runner._state_uncertain and runner.last_error is not None
    assert 'invalid types or fields' in runner.recording_error['error']
    records.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_ordinary_recording_failure_message_matches_policy(tmp_path, exploratory):
    runner, records, _, nf = with_nf(tmp_path, exploratory=exploratory)
    with runner.lock:
        runner._recording_failed(OSError('ordinary capture failure'))
    message = runner.recording_error['message']
    assert ('permits continuing' in message) is exploratory
    assert (runner.last_error is None) is exploratory
    assert not runner._state_uncertain and nf.closed
    records.close()


def test_cleanup_start_failure_is_reported_and_never_revives_ack(tmp_path, monkeypatch):
    runner, records, drain, nf = with_nf(tmp_path, exploratory=True)
    entered, release = threading.Event(), threading.Event(); original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'assay_control': entered.set(); assert release.wait(3)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(1)
    def fail_start(*args): raise RuntimeError('injected cleanup startup failure')
    monkeypatch.setattr(threading.Thread, 'start', fail_start)
    try:
        with runner.lock: runner._recording_failed(OSError('independent failure'))
        assert runner.recording_error['cleanup_error']
        assert nf._cleanup_error and nf.closed
        assert not runner.command_acks[0]['ack']['applied']
    finally:
        release.set(); tx['writer'].join(2)
        # No cleanup worker started; release handles off the runner lock in test.
        nf._abort_handles(); records.close()
    assert len(runner.command_acks) == 1


@pytest.mark.parametrize('phase', ['write', 'fsync', 'missing_receipt'])
def test_real_recording_io_or_missing_receipt_failure_never_applied(tmp_path, monkeypatch, phase):
    runner, records, drain, nf = with_nf(tmp_path, exploratory=True)
    if phase == 'write':
        def fail(*args): raise OSError('NF command write failed')
        monkeypatch.setattr(nf._gz, 'write', fail)
    elif phase == 'fsync':
        original = os.fsync
        def fail(fd):
            if os.readlink(f'/proc/self/fd/{fd}') == str(nf._partial):
                raise OSError('NF command fsync failed')
            return original(fd)
        monkeypatch.setattr(os, 'fsync', fail)
    else:
        original = nf.durable_command
        def fail(request): original(request); return None
        monkeypatch.setattr(nf, 'durable_command', fail)
    runner.dispatch_command(airflow(23)); tx = runner._pending_assay_control
    drain.poll_once(); assert tx['entry']['done'].wait(2)
    finish_threads(tx, nf)
    assert len(runner.command_acks) == 1 and not runner.command_acks[0]['ack']['applied']
    assert runner._state_uncertain and runner.last_error is not None and nf.closed
    assert nf._partial.exists() and runner.recording_error
    assert 'keeps running' not in runner.recording_error['message']
    assert 'halted' in runner.recording_error['message']
    assert 'uncertain' in runner.recording_error['message']
    records.close()
