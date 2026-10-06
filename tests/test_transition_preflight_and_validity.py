"""Barrier regressions for off-lock graph validation and owned validity batches."""
import json
import os
import threading
import time

import pytest

from experiment_registry import SharedGraph
from tests.test_assay_control_transactions import attached, airflow, state


def finish(tx):
    for key in ('writer', 'failure_writer'):
        if tx.get(key):
            tx[key].join(10)
            assert not tx[key].is_alive()


def poll_until_done(tx, drain):
    deadline = time.monotonic() + 10
    while not tx['entry']['done'].is_set() and time.monotonic() < deadline:
        drain.poll_once()
        tx['entry']['done'].wait(0.01)
    assert tx['entry']['done'].is_set()
    finish(tx)


def test_graph_validation_waits_off_lock_before_freezing_source(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path, test_synthetic_graph=True)
    before = state(runner, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = SharedGraph.synthetic
    def blocked(**kwargs):
        entered.set(); assert release.wait(10)
        return original(**kwargs)
    monkeypatch.setattr(SharedGraph, 'synthetic', blocked)
    reply = runner.dispatch_command({'action': 'switch_controller', 'backend': 'connectome-fixed'})
    tx = runner._pending_assay_control
    assert reply['status'] == 'queued' and entered.wait(2)
    validator = tx['writer']
    try:
        assert tx['phase'] == 'validating_target' and 'terminal' not in tx
        assert state(runner, tmp_path) == before and runner._observation_terminal is None
        assert runner.lock.acquire(timeout=0.2); runner.lock.release()
        assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 3})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'reset_trial'})['status'] == 'error'
        assert runner.observation_lifecycle_status()['phase'] == 'waiting_for_save'
    finally:
        release.set(); validator.join(10)
    poll_until_done(tx, drain)
    assert tx['entry']['result']['ack']['applied'] and runner.backend == 'connectome-fixed'
    assert tx['entry']['result']['command_id'] == reply['command_id']
    assert tx['terminal']['durable'] and len(runner.command_acks) == 1
    records.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_unavailable_graph_is_refused_without_prefix_identity_or_store_change(tmp_path, monkeypatch, exploratory):
    runner, records, _ = attached(tmp_path, exploratory=exploratory, test_synthetic_graph=True)
    before = state(runner, tmp_path)
    entered, release = threading.Event(), threading.Event()
    def unavailable(**kwargs):
        entered.set(); assert release.wait(10)
        raise RuntimeError('requested graph is unavailable')
    monkeypatch.setattr(SharedGraph, 'synthetic', unavailable)
    reply = runner.dispatch_command({'action': 'switch_backend', 'backend': 'connectome-fixed'})
    tx = runner._pending_assay_control
    assert reply['status'] == 'queued' and entered.wait(2)
    release.set(); finish(tx)
    assert tx['entry']['done'].is_set() and not tx['entry']['result']['ack']['applied']
    assert 'unavailable' in tx['entry']['result']['message']
    assert state(runner, tmp_path) == before
    assert runner._observation_terminal is None and not runner._state_uncertain and runner.last_error is None
    assert runner.observation_publication.pending_status()['pending_count'] == 0
    records.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_validation_timeout_and_late_graph_cannot_freeze_or_switch(tmp_path, monkeypatch, exploratory):
    runner, records, _ = attached(tmp_path, exploratory=exploratory, test_synthetic_graph=True)
    before = state(runner, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = SharedGraph.synthetic
    def late(**kwargs):
        entered.set(); assert release.wait(10)
        return original(**kwargs)
    monkeypatch.setattr(SharedGraph, 'synthetic', late)
    runner.dispatch_command({'action': 'switch_backend', 'backend': 'connectome-fixed'})
    tx = runner._pending_assay_control
    assert entered.wait(2)
    try:
        runner.assay_control_timeout_s = 0
        with runner.lock:
            runner._check_assay_control_timeout()
        assert tx['entry']['done'].is_set() and not tx['entry']['result']['ack']['applied']
        assert runner._observation_terminal is None and runner._pending_assay_control is None
    finally:
        release.set(); finish(tx)
    assert state(runner, tmp_path) == before and runner.backend == 'modular'
    assert len(runner.command_acks) == 1 and runner.last_error is None and not runner._state_uncertain
    records.close()


@pytest.mark.parametrize('problem', ['missing_recorder', 'full_queue'])
def test_recorder_and_queue_preflight_never_start_heavy_graph_loading(tmp_path, monkeypatch, problem):
    runner, records, _ = attached(tmp_path, exploratory=True, test_synthetic_graph=True)
    if problem == 'missing_recorder':
        runner.learning_records = None
    else:
        runner.observation_publication.capacity = 1
        runner.dispatch_command(airflow(7))
        claim = runner.claim_terminal_observation()
        runner.terminal_observation_failed(claim['attempt_token'], OSError('retained evidence'))
    before = state(runner, tmp_path)
    def forbidden(**kwargs): raise AssertionError('preflight must precede graph loading')
    monkeypatch.setattr(SharedGraph, 'synthetic', forbidden)
    reply = runner.dispatch_command({'action': 'switch_backend', 'backend': 'connectome-fixed'})
    assert reply['status'] == 'error' and not reply['ack']['applied']
    assert state(runner, tmp_path) == before and runner._pending_assay_control is None
    records.close()


@pytest.mark.parametrize('second_writer', [False, True])
def test_validity_fsync_retains_concurrent_append_and_serializes_writers(tmp_path, monkeypatch, second_writer):
    runner, records, _ = attached(tmp_path)
    first = {'event': 'activate', 'run_id': 'owned-prefix', 'session': runner.run_id, 'at': 1.0}
    later = {'event': 'activate', 'run_id': 'during-fsync', 'session': runner.run_id, 'at': 2.0}
    runner._pending_validity[:] = [first]
    entered, release, second_done = threading.Event(), threading.Event(), threading.Event()
    original = os.fsync
    path = runner.output_dir / 'run_validity.jsonl'
    def blocked(fd):
        if os.readlink(f'/proc/self/fd/{fd}') == str(path) and not entered.is_set():
            entered.set(); assert release.wait(10)
        return original(fd)
    monkeypatch.setattr(os, 'fsync', blocked)
    writer = threading.Thread(target=lambda: runner._flush_validity(wait=True))
    writer.start(); assert entered.wait(2)
    with runner.lock:
        runner._pending_validity.append(later)
        start = time.monotonic(); runner._flush_validity()
        assert time.monotonic() - start < 0.2  # Locked callers never wait for disk/writer.
    other = None
    if second_writer:
        def save_later(): runner._flush_validity(wait=True); second_done.set()
        other = threading.Thread(target=save_later); other.start()
        assert not second_done.wait(0.05)
    release.set(); writer.join(10)
    assert not writer.is_alive()
    if other:
        other.join(10); assert second_done.is_set() and not other.is_alive()
    else:
        assert runner._pending_validity == [later]
        assert [json.loads(x) for x in path.read_text().splitlines()][-1] == first
        runner._flush_validity(wait=True)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows.count(first) == rows.count(later) == 1
    assert not runner._pending_validity
    records.close()


def test_failed_validity_batch_keeps_owned_and_concurrently_appended_records(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path, exploratory=True)
    first = {'event': 'activate', 'run_id': 'failed-prefix', 'session': runner.run_id, 'at': 1.0}
    later = {'event': 'activate', 'run_id': 'failed-during-fsync', 'session': runner.run_id, 'at': 2.0}
    runner._pending_validity[:] = [first]
    entered, release = threading.Event(), threading.Event()
    original = os.fsync; path = runner.output_dir / 'run_validity.jsonl'
    def failed(fd):
        if os.readlink(f'/proc/self/fd/{fd}') == str(path):
            entered.set(); assert release.wait(10)
            raise OSError('batch fsync unavailable')
        return original(fd)
    monkeypatch.setattr(os, 'fsync', failed)
    writer = threading.Thread(target=lambda: runner._flush_validity(wait=True))
    writer.start(); assert entered.wait(2)
    with runner.lock: runner._pending_validity.append(later)
    release.set(); writer.join(10)
    assert not writer.is_alive()
    assert first in runner._pending_validity and later in runner._pending_validity
    assert runner.result_validity()['state'] == 'incomplete'
    monkeypatch.setattr(os, 'fsync', original)
    runner._flush_validity(wait=True)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert first in rows and later in rows
    assert first not in runner._pending_validity and later not in runner._pending_validity
    records.close()


@pytest.mark.parametrize('fail_transition_batch', [False, True])
def test_transition_ack_waits_for_its_validity_batch_behind_existing_writer(tmp_path, monkeypatch, fail_transition_batch):
    runner, records, drain = attached(tmp_path, exploratory=True)
    before = {'event': 'activate', 'run_id': 'existing-writer', 'session': runner.run_id, 'at': 1.0}
    runner._pending_validity[:] = [before]
    entered, release, transition_write = threading.Event(), threading.Event(), threading.Event()
    original = os.fsync
    path = runner.output_dir / 'run_validity.jsonl'
    calls = []
    def blocked(fd):
        if os.readlink(f'/proc/self/fd/{fd}') == str(path):
            calls.append(fd)
            if len(calls) == 1:
                entered.set(); assert release.wait(10)
            else:
                transition_write.set()
                if fail_transition_batch: raise OSError('transition validity unavailable')
        return original(fd)
    monkeypatch.setattr(os, 'fsync', blocked)
    existing = threading.Thread(target=lambda: runner._flush_validity(wait=True))
    existing.start(); assert entered.wait(2)
    reply = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 't-maze'})
    tx = runner._pending_assay_control
    drain.poll_once()
    deadline = time.monotonic() + 5
    while not tx.get('committed') and time.monotonic() < deadline:
        tx['entry']['done'].wait(0.01)
    try:
        assert tx.get('committed') and not tx['entry']['done'].is_set()
        assert not transition_write.is_set()
        assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 2})['status'] == 'ok'
        assert runner._pending_validity[-1]['event'] == 'activate'
        transition_run = runner._pending_validity[-1]['run_id']
    finally:
        release.set(); existing.join(10)
    assert tx['entry']['done'].wait(10); finish(tx)
    assert transition_write.is_set()
    assert tx['entry']['result']['ack']['applied'] is (not fail_transition_batch)
    if fail_transition_batch:
        assert any(row.get('run_id') == transition_run for row in runner._pending_validity)
    else:
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert any(row.get('run_id') == transition_run and row['event'] == 'activate' for row in rows)
    monkeypatch.setattr(os, 'fsync', original)
    records.close()
