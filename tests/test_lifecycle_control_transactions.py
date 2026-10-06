"""Teaching, policy and shutdown preserve exact old-prefix ownership and durability."""
import copy
import json
import os
import threading
import time

import pytest

from tests.test_assay_control_transactions import attached, state
from tests.transition_control_helpers import transition_command


def finish(tx):
    assert tx['entry']['done'].wait(5)
    for key in ('writer', 'failure_writer'):
        if tx.get(key):
            tx[key].join(5)
            assert not tx[key].is_alive()
    return tx['entry']['result']


def complete(runner, drain, cmd):
    reply = runner.dispatch_command(cmd)
    assert reply['status'] == 'queued', reply
    tx = runner._pending_assay_control
    drain.poll_once()
    return finish(tx), tx


@pytest.mark.parametrize('cmd', [
    {'action': 'teach_brain', 'pairs': True}, {'action': 'teach_brain', 'pairs': 0},
    {'action': 'teach_brain', 'reverse': 1}, {'action': 'set_learning', 'enabled': 1},
    {'action': 'set_observation_policy', 'continuous': 1},
    {'action': 'set_observation_policy', 'trial_seconds': -1},
    {'action': 'set_observation_policy', 'trial_seconds': float('nan')},
    {'action': 'return_from_teaching'},
])
def test_rejected_lifecycle_is_byte_identical(tmp_path, cmd):
    runner, records, _ = attached(tmp_path)
    before = state(runner, tmp_path)
    reply = runner.dispatch_command(cmd)
    assert reply['status'] == 'error' and not reply['ack']['applied']
    assert state(runner, tmp_path) == before
    records.close()


@pytest.mark.parametrize('cmd', [
    {'action': 'set_learning', 'enabled': False},
    {'action': 'set_observation_policy', 'continuous': True, 'trial_seconds': 7.5},
    {'action': 'set_observation_policy', 'continuous': False, 'trial_seconds': 3.0},
    {'action': 'teach_brain', 'pairs': 1},
])
def test_lifecycle_waits_for_exact_prefix_before_mutation(tmp_path, cmd):
    runner, records, drain = attached(tmp_path)
    runner.step_once()
    pose = copy.deepcopy(runner._terminal_pose())
    learning, policy, segment = runner.active_brain.learning_enabled, runner.continuous, runner.segment_id
    reply = runner.dispatch_command(cmd)
    tx = runner._pending_assay_control
    assert reply['status'] == 'queued'
    assert runner.active_brain.learning_enabled == learning and runner.continuous == policy
    assert not runner.active_brain.teaching and runner.segment_id == segment
    runner.step_once()
    assert runner._terminal_pose() == pose
    drain.poll_once(); result = finish(tx)
    assert result['ack']['applied'] and tx['terminal']['durable']
    assert tx['terminal']['observation']['end_reason'] == 'policy_change'
    assert runner._terminal_pose() == pose
    events = [json.loads(x) for x in (runner.active_brain.directory / 'open-arena.events.jsonl').read_text().splitlines()]
    assert any(x['kind'] == 'assay_control' and x['action'] == cmd['action'] for x in events)
    assert len(runner.command_acks) == 1
    records.close()


def test_teaching_owns_clock_then_returns_fresh_segment_in_place(tmp_path):
    runner, records, drain = attached(tmp_path)
    runner.step_once()
    pose = copy.deepcopy(runner._terminal_pose())
    oldsegment, oldstep, oldtrial = runner.segment_id, runner.total_steps, runner.trial_sim_time
    _, entry = complete(runner, drain, {'action': 'teach_brain', 'pairs': 1})
    frozen = copy.deepcopy(entry['terminal'])
    assert runner.latest_telemetry['observation'] is None
    assert runner.observation_lifecycle_status()['phase'] == 'teaching'
    assert runner.dispatch_command({'action': 'set_learning', 'enabled': False})['status'] == 'error'
    for _ in range(100):
        with runner.lock: runner.step_once()
        pending = runner._pending_assay_control
        if pending: finish(pending)
    assert runner.total_steps == oldstep + 100
    assert {k: v for k, v in runner._terminal_pose().items() if k != 'step'} == {k: v for k, v in pose.items() if k != 'step'} and runner.trial_sim_time == oldtrial
    assert not runner.active_brain.teaching and runner._teaching_suspended
    with runner.lock: runner.step_once()
    finish(runner._pending_assay_control) if runner._pending_assay_control else None
    assert not runner._teaching_suspended and runner.segment_id != oldsegment
    assert runner.observation_segment_start_sim_s == runner.total_steps * runner.dt
    assert {k: v for k, v in runner._terminal_pose().items() if k != 'step'} == {k: v for k, v in pose.items() if k != 'step'}
    assert entry['terminal'] == frozen
    assert runner._live_observation()['segment_id'] == runner.segment_id
    records.close()


def test_teaching_event_io_is_off_lock_and_transport_remains_responsive(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'teaching_started':
            entered.set(); assert release.wait(5)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    runner.dispatch_command({'action': 'teach_brain', 'pairs': 1})
    tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(2)
    try:
        assert not tx['entry']['done'].is_set()
        assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 2})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'reset_trial'})['status'] == 'error'
    finally:
        release.set()
    assert finish(tx)['ack']['applied']
    records.close()


def test_restored_teaching_has_no_typed_assay_sample_until_return(tmp_path):
    runner, records, _ = attached(tmp_path)
    # Fixture stores genuine existing protocol state; restart must suspend it.
    runner.active_brain.start_teaching(1)
    runner.active_brain.save()
    records.close()
    resumed, records, drain = attached(tmp_path)
    assert resumed.active_brain.teaching and resumed._teaching_suspended
    assert resumed._live_observation() is None and resumed._observation_terminal is None
    for _ in range(100):
        with resumed.lock: resumed.step_once()
        if resumed._pending_assay_control: finish(resumed._pending_assay_control)
    with resumed.lock: resumed.step_once()
    if resumed._pending_assay_control: finish(resumed._pending_assay_control)
    assert resumed._live_observation() is not None
    assert resumed.observation_segment_start_sim_s == 2.0
    records.close()


def test_shutdown_freezes_and_saves_before_clean_marker(tmp_path):
    runner, records, _ = attached(tmp_path)
    runner.step_once()
    pose, step = copy.deepcopy(runner._terminal_pose()), runner.total_steps
    assert runner.stop() and runner.stop()
    assert runner.total_steps == step and runner._terminal_pose() == pose
    rows = [json.loads(x) for x in (runner.output_dir / 'run_validity.jsonl').read_text().splitlines()]
    assert sum(x.get('event') == 'session_end' for x in rows) == 1
    assert not runner.observation_publication.pending_status()['pending_count']
    assert runner.command_acks[-1]['ack']['applied']
    last = runner.observation_publication_status()['last_terminal']
    assert last['observation']['end_reason'] == 'shutdown'
    records.close()


@pytest.mark.parametrize('channel', ['prefix', 'event', 'checkpoint', 'marker'])
def test_shutdown_save_failure_never_reports_clean(tmp_path, monkeypatch, channel):
    runner, records, drain = attached(tmp_path)
    if channel == 'prefix':
        monkeypatch.setattr(records, 'record_observation', lambda *a, **k: (_ for _ in ()).throw(OSError('prefix unavailable')))
    elif channel == 'event':
        original = runner.active_brain.log
        def fail(kind, **fields):
            if kind == 'assay_control': raise OSError('event unavailable')
            return original(kind, **fields)
        monkeypatch.setattr(runner.active_brain, 'log', fail)
    elif channel == 'checkpoint':
        monkeypatch.setattr(runner, 'save_checkpoint', lambda *a, **k: (_ for _ in ()).throw(OSError('checkpoint unavailable')))
    else:
        original = os.fsync
        path = runner.output_dir / 'run_validity.jsonl'
        def fail(fd):
            if os.readlink(f'/proc/self/fd/{fd}') == str(path) and 'session_end' in path.read_text():
                raise OSError('marker acknowledgement unavailable')
            return original(fd)
        monkeypatch.setattr(os, 'fsync', fail)
    assert runner.stop() is False and runner.stop() is False
    assert not runner.command_acks[-1]['ack']['applied']
    rows = [json.loads(x) for x in (runner.output_dir / 'run_validity.jsonl').read_text().splitlines()]
    if channel != 'marker': assert not any(x.get('event') == 'session_end' for x in rows)
    else:
        assert any(x.get('event') == 'session_end_failed' for x in rows)
        assert runner.run_id in runner._interrupted_sessions(rows)
        assert not runner._stop_ok
        assert any(row.get('event') == 'session_end_failed' for row in runner._pending_validity)
        assert runner.persistence.describe()['failing']['validity_ledger']
    if channel == 'prefix': assert runner.observation_publication.pending_status()['pending_count'] == 1
    records.close()


def test_graph_toy_teaching_stays_purely_refused(tmp_path):
    runner, records, _ = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    before = state(runner, tmp_path)
    assert runner.dispatch_command({'action': 'teach_brain'})['status'] == 'error'
    assert state(runner, tmp_path) == before
    records.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_policy_event_failure_halts_uncertain_without_applied_ack(tmp_path, monkeypatch, exploratory):
    runner, records, drain = attached(tmp_path, exploratory=exploratory)
    original = runner.active_brain.log
    def failed(kind, **fields):
        if kind == 'assay_control': raise OSError('policy evidence unavailable')
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', failed)
    reply, tx = complete(runner, drain, {'action': 'set_observation_policy', 'continuous': True})
    assert not reply['ack']['applied'] and runner.last_error and runner._state_uncertain
    assert tx['terminal']['durable'] and tx['terminal']['observation']['end_reason'] == 'policy_change'
    assert len(runner.command_acks) == 1
    records.close()


def test_teaching_pair_and_completion_persistence_wait_off_lock(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    complete(runner, drain, {'action': 'teach_brain', 'pairs': 1})
    for _ in range(99):
        with runner.lock: runner.step_once()
    entered, release = threading.Event(), threading.Event()
    original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'teaching_pair': entered.set(); assert release.wait(5)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    with runner.lock: runner.step_once()
    tx = runner._pending_assay_control
    assert entered.wait(2) and runner.total_steps == 100
    try:
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 3})['status'] == 'ok'
        assert runner._live_observation() is None
        with runner.lock: runner.step_once()
        assert runner.total_steps == 100 and not tx['entry']['done'].is_set()
    finally:
        release.set()
    assert finish(tx)['ack']['applied']
    with runner.lock: runner.step_once()
    if runner._pending_assay_control: finish(runner._pending_assay_control)
    assert runner.observation_segment_start_sim_s == 2.0
    records.close()


def test_shutdown_finalises_real_nf_before_clean_marker_and_ack(tmp_path):
    from neurofly.recording import RunRecorder
    runner, records, _ = attached(tmp_path)
    nf = RunRecorder(runner, tmp_path / 'nf' / 'shutdown.nfrec')
    runner.recorder = nf
    assert runner.stop() and nf.closed and not getattr(nf, "aborted", None)
    assert runner.command_acks[-1]['ack']['applied']
    assert nf.path.exists() and runner.recorder is None
    records.close()


def test_shutdown_checkpoint_wait_is_responsive_and_late_completion_cannot_mark_clean(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.15
    entered, release = threading.Event(), threading.Event()
    original = runner.save_checkpoint
    def blocked(*a, **k): entered.set(); assert release.wait(5); return original(*a, **k)
    monkeypatch.setattr(runner, 'save_checkpoint', blocked)
    outcome = []
    stopper = threading.Thread(target=lambda: outcome.append(runner.stop()))
    stopper.start(); assert entered.wait(2)
    tx = runner._pending_assay_control
    try:
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 2})['status'] == 'ok'
        assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
        stopper.join(2); assert outcome == [False]
        assert tx['entry']['done'].is_set() and not tx['entry']['result']['ack']['applied']
    finally:
        release.set()
    finish(tx)
    rows = [json.loads(x) for x in (runner.output_dir / 'run_validity.jsonl').read_text().splitlines()]
    assert not any(x.get('event') == 'session_end' for x in rows)
    assert len(runner.command_acks) == 1
    records.close()


@pytest.mark.parametrize('action', ['teach_brain', 'set_learning', 'set_observation_policy', 'shutdown'])
def test_missing_observation_recorder_refuses_lifecycle_without_mutation(tmp_path, action):
    runner, records, _ = attached(tmp_path)
    runner.learning_records = None
    before = state(runner, tmp_path)
    reply = runner.dispatch_command({'action': action, 'enabled': False})
    assert reply['status'] == 'error' and not reply['ack']['applied']
    assert state(runner, tmp_path) == before
    assert runner._pending_assay_control is None
    records.close()


def test_shutdown_prefix_disk_wait_is_bounded_and_retains_late_owned_write(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.15
    entered, release = threading.Event(), threading.Event()
    original = records.record_observation
    def blocked(*a, **k): entered.set(); assert release.wait(5); return original(*a, **k)
    monkeypatch.setattr(records, 'record_observation', blocked)
    outcome = []
    stopper = threading.Thread(target=lambda: outcome.append(runner.stop()))
    stopper.start(); assert entered.wait(2)
    tx = runner._pending_assay_control
    try:
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 2})['status'] == 'ok'
        stopper.join(2); assert outcome == [False]
        assert not tx['entry']['result']['ack']['applied']
        assert runner.observation_publication.pending_status()['pending_count'] == 1
    finally:
        release.set(); tx['shutdown_drain'].join(5)
    finish(tx)
    assert not runner.command_acks[-1]['ack']['applied'] and len(runner.command_acks) == 1
    rows = [json.loads(x) for x in (runner.output_dir / 'run_validity.jsonl').read_text().splitlines()]
    assert not any(x.get('event') == 'session_end' for x in rows)
    assert runner.last_error
    records.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_lifecycle_event_timeout_never_accepts_late_completion(tmp_path, monkeypatch, exploratory):
    runner, records, drain = attached(tmp_path, exploratory=exploratory)
    entered, release = threading.Event(), threading.Event()
    original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'assay_control': entered.set(); assert release.wait(5)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    runner.dispatch_command({'action': 'set_learning', 'enabled': False})
    tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(2)
    frozen = copy.deepcopy(tx['terminal'])
    try:
        runner.assay_control_timeout_s = 0
        with runner.lock: runner._check_assay_control_timeout()
        assert tx['entry']['done'].is_set() and not tx['entry']['result']['ack']['applied']
        assert runner._state_uncertain and runner.last_error
    finally:
        release.set()
    finish(tx)
    assert tx['terminal'] == frozen and len(runner.command_acks) == 1
    records.close()


def test_fresh_policy_begin_failure_retains_prior_prefix_and_halts(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    def failed(): raise RuntimeError('fresh segment rejected')
    monkeypatch.setattr(runner, '_configure_observation_owner', failed)
    result, tx = complete(runner, drain, {'action': 'set_observation_policy', 'continuous': True})
    assert not result['ack']['applied'] and tx['terminal']['durable']
    assert runner.last_error and runner._state_uncertain
    records.close()


def test_shutdown_during_teaching_keeps_checkpoint_protocol_and_original_prefix(tmp_path):
    runner, records, drain = attached(tmp_path)
    _, tx = complete(runner, drain, {'action': 'teach_brain', 'pairs': 1})
    for _ in range(17):
        with runner.lock: runner.step_once()
    assert runner.stop()
    assert runner.active_brain.teaching['tick'] == 17
    assert tx['terminal']['observation']['end_reason'] == 'policy_change'
    assert runner.observation_publication.pending_status()['pending_count'] == 0
    records.close()


def test_failed_shutdown_history_remains_interrupted_after_later_end_record(tmp_path):
    runner, records, _ = attached(tmp_path)
    rows = [{'event': 'session_start', 'session': 'failed'},
            {'event': 'activate', 'session': 'failed', 'run_id': 'affected'},
            {'event': 'session_end_failed', 'session': 'failed'},
            {'event': 'session_end', 'session': 'failed'}]
    assert runner._interrupted_sessions(rows) == {'failed': {'affected'}}
    records.close()


def test_shutdown_waits_for_existing_control_without_superseding_prefix(tmp_path):
    runner, records, _ = attached(tmp_path)
    runner.dispatch_command({'action': 'set_learning', 'enabled': False})
    prior = runner._pending_assay_control
    assert runner.stop()
    assert prior['entry']['result']['ack']['applied'] and prior['terminal']['durable']
    assert len(runner.command_acks) == 2
    assert runner.command_acks[-1]['ack']['applied']
    assert runner.observation_publication.pending_status()['pending_count'] == 0
    records.close()


def test_completed_teaching_cannot_silently_freeze_when_recorder_is_detached(tmp_path):
    runner, records, drain = attached(tmp_path)
    complete(runner, drain, {'action': 'teach_brain', 'pairs': 1})
    for _ in range(100):
        with runner.lock: runner.step_once()
        if runner._pending_assay_control: finish(runner._pending_assay_control)
    runner.learning_records = None
    with runner.lock: runner.step_once()
    if runner._pending_assay_control: finish(runner._pending_assay_control)
    assert runner.last_error and runner._state_uncertain
    assert runner._live_observation() is None
    assert not runner.command_acks[-1]['ack']['applied']
    records.close()
