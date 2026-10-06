"""Real prefix/event durability for manual reset and ready-target transitions."""
import copy
import json
import threading
import time
import zlib

import numpy as np
import pytest

from experiment_brains import PARADIGMS
from neurofly.recording import RunRecorder
from tests.test_assay_control_transactions import attached, state


def switch(target):
    return {'action': 'switch_paradigm', 'paradigm': target}


def complete(runner, records, drain, cmd):
    reply = runner.dispatch_command(cmd)
    assert reply['status'] == 'queued', reply
    tx = runner._pending_assay_control
    deadline = time.monotonic() + 10
    while not tx['entry']['done'].is_set() and time.monotonic() < deadline:
        drain.poll_once()
        tx['entry']['done'].wait(0.01)
    assert tx['entry']['done'].wait(10), 'transition writer did not complete'
    if tx.get('writer'):
        tx['writer'].join(10)
        assert not tx['writer'].is_alive()
    if tx.get('failure_writer'):
        tx['failure_writer'].join(10)
        assert not tx['failure_writer'].is_alive()
    return tx['entry']['result'], tx


def test_reset_differs_from_pause_resume_and_retains_exact_durable_prefix(tmp_path):
    runner, records, drain = attached(tmp_path)
    runner.step_once(); runner.arena.fly.pos.x += 8
    brain, identity, segment = runner.active_brain, runner.identity(), runner.segment_id
    weights = brain.circuit.w.copy()
    runner.dispatch_command({'action': 'set_paused', 'paused': True})
    runner.dispatch_command({'action': 'set_paused', 'paused': False})
    assert runner.segment_id == segment and runner.identity() == identity
    reply = runner.dispatch_command({'action': 'reset_trial', 'advance': False})
    tx = runner._pending_assay_control
    frozen = copy.deepcopy(tx['terminal']['observation'])
    assert reply['status'] == 'queued' and frozen['end_reason'] == 'manual_reset'
    assert frozen['identity']['run_id'] == identity['run_id']
    before = runner.total_steps; runner.step_once(); assert runner.total_steps == before
    assert runner.segment_id == segment
    drain.poll_once(); assert tx['entry']['done'].wait(10)
    assert tx['entry']['result']['ack']['applied']
    assert runner.active_brain is brain and runner.identity() == identity
    assert runner.segment_id != segment and runner.current_trial == 1
    assert runner.arena.observation_owner.observation_status()['presentation_index'] == 0
    assert np.array_equal(brain.circuit.w, weights)
    saved = runner.observation_publication_status()['last_terminal']['observation']
    assert saved == frozen
    records.close()


def test_switch_away_back_preserves_modular_brain_world_and_new_observation_identity(tmp_path):
    runner, records, drain = attached(tmp_path)
    runner.step_once(); runner.arena.fly.pos.x += 5
    source, world, identity = runner.active_brain, runner.arena.snapshot_world(), runner.identity()
    source.circuit.w[0, 0] += 0.25
    weights = source.circuit.w.copy()
    reply, tx = complete(runner, records, drain, switch('t-maze'))
    assert reply['ack']['applied'] and runner.active_brain is not source
    assert tx['terminal']['observation']['end_reason'] == 'experiment_selected'
    assert tx['terminal']['observation']['identity']['run_id'] == identity['run_id']
    assert runner._live_observation()['identity']['run_id'] == runner.manifest.run_id
    reply, _ = complete(runner, records, drain, switch('open-arena'))
    assert reply['ack']['applied'] and runner.active_brain is source
    assert runner.arena.snapshot_world()['state']['fly'] == world['state']['fly']
    assert np.array_equal(source.circuit.w, weights)
    records.close()


@pytest.mark.parametrize('target', PARADIGMS)
def test_all_modular_targets_begin_ready_fresh_segment(tmp_path, target):
    runner, records, drain = attached(tmp_path)
    reply, tx = complete(runner, records, drain, switch(target))
    assert reply['ack']['applied'] and reply['ack']['identity']['assay'] == target
    assert runner._live_observation()['identity']['assay'] == target
    assert runner._pending_assay_control is None and runner._observation_terminal is None
    before = runner.total_steps; runner.step_once()
    assert runner.total_steps == before + 1 and runner.last_error is None
    assert tx['terminal']['durable'] is not None
    records.close()


@pytest.mark.parametrize('action', ['switch_backend', 'switch_controller'])
def test_backend_and_alias_preserve_graph_state_away_back(tmp_path, action):
    runner, records, drain = attached(tmp_path, assay='t-maze', backend='connectome-fixed', test_synthetic_graph=True)
    runner.step_once()
    source, source_id = runner.registry.active, runner.registry.active.instance_id
    first, tx = complete(runner, records, drain, {'action': action, 'backend': 'modular'})
    assert first['ack']['applied'] and runner.backend == 'modular'
    assert source.brain is None and tx['terminal']['observation']['end_reason'] == 'backend_switch'
    second, _ = complete(runner, records, drain, {'action': action, 'backend': 'connectome-fixed'})
    assert second['ack']['applied'] and runner.registry.active.instance_id == source_id
    assert runner.registry.active is not source and runner.registry.active.brain is not None
    records.close()


@pytest.mark.parametrize('cmd', [switch(None), switch([]), switch('unavailable'),
    {'action': 'switch_backend', 'backend': {}}, {'action': 'switch_controller', 'backend': False},
    {'action': 'reset_trial', 'advance': 1}, {'action': 'reset_trial', 'keep_memory': 'yes'}])
def test_preflight_rejection_is_byte_identical(tmp_path, cmd):
    runner, records, _ = attached(tmp_path)
    before = state(runner, tmp_path)
    reply = runner.dispatch_command(cmd)
    assert reply['status'] == 'error' and not reply['ack']['applied']
    assert state(runner, tmp_path) == before and runner._pending_assay_control is None
    records.close()


def test_missing_recorder_and_same_backend_noop_do_not_fake_transition(tmp_path):
    runner, records, _ = attached(tmp_path)
    runner.learning_records = None
    before = state(runner, tmp_path)
    for cmd in [switch('t-maze'), {'action': 'reset_trial'}]:
        reply = runner.dispatch_command(cmd)
        assert reply['status'] == 'error' and 'recorder is required' in reply['message']
        assert state(runner, tmp_path) == before
    reply = runner.dispatch_command({'action': 'switch_controller', 'backend': runner.backend})
    assert reply['status'] == 'ok' and state(runner, tmp_path) == before
    records.close()


def test_failed_actual_graph_restore_keeps_source_usable(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    source, brain = runner.registry.active, runner.registry.active.brain
    def fail(target):
        assert runner.registry.active is source and source.brain is brain
        raise RuntimeError('target restore refused')
    monkeypatch.setattr(runner.registry, '_restore', fail)
    reply, tx = complete(runner, records, drain, switch('t-maze'))
    assert not reply['ack']['applied'] and runner.last_error and runner._state_uncertain
    assert runner.registry.active is source and source.brain is brain
    assert runner.active_brain is tx['source_brain'] and tx['terminal']['durable']
    records.close()


@pytest.mark.parametrize('phase', ['target', 'source_save', 'begin', 'event'])
def test_transition_failure_never_acknowledges_applied(tmp_path, monkeypatch, phase):
    runner, records, drain = attached(tmp_path)
    source = runner.active_brain
    def fail(*args, **kwargs): raise OSError('injected '+phase)
    if phase == 'target': monkeypatch.setattr(runner.brains, 'get', fail)
    elif phase == 'source_save': monkeypatch.setattr(source, 'save', fail)
    elif phase == 'begin': monkeypatch.setattr(runner, '_configure_observation_owner', fail)
    else:
        target = runner.brains.get('t-maze')
        original = target.log
        def failed_event(kind, **fields):
            if kind == 'assay_control': fail()
            return original(kind, **fields)
        monkeypatch.setattr(target, 'log', failed_event)
    reply, tx = complete(runner, records, drain, switch('t-maze'))
    assert not reply['ack']['applied'] and runner.last_error and runner._state_uncertain
    assert tx['terminal']['durable'] and len(runner.command_acks) == 1
    if phase in ('target', 'source_save'): assert runner.active_brain is source
    records.close()


def test_pending_preparation_is_responsive_and_late_target_is_released(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    source = runner.registry.active
    entered, release = threading.Event(), threading.Event(); prepared = []
    original = runner._prepare_transition_target
    def blocked(tx):
        result = original(tx); prepared.append(result)
        entered.set(); assert release.wait(10)
        return result
    monkeypatch.setattr(runner, '_prepare_transition_target', blocked)
    reply = runner.dispatch_command(switch('t-maze')); tx = runner._pending_assay_control
    drain.poll_once(); assert entered.wait(10)
    try:
        assert runner.lock.acquire(timeout=0.2); runner.lock.release()
        before = state(runner, tmp_path)
        assert runner.dispatch_command(switch('y-maze'))['status'] == 'error'
        assert state(runner, tmp_path) == before
        for cmd in [{'action': 'set_paused', 'paused': True}, {'action': 'set_speed', 'speed': 3}]:
            assert runner.dispatch_command(cmd)['status'] == 'ok'
        assert runner.observation_lifecycle_status()['phase'] == 'waiting_for_save'
        runner.assay_control_timeout_s = 0; runner.step_once()
        assert tx['entry']['done'].is_set() and not runner.command_acks[-1]['ack']['applied']
        assert runner.registry.active is source and source.brain is not None
    finally:
        release.set(); tx['writer'].join(10)
    assert prepared[0]['activation']['target'].brain is None
    assert runner.registry.active is source and runner.active_paradigm_id == 'open-arena'
    assert len(runner.command_acks) == 1 and runner.last_error and runner._state_uncertain
    records.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_prefix_save_failure_and_late_receipt_never_switch(tmp_path, exploratory):
    runner, records, _ = attached(tmp_path, exploratory=exploratory)
    source = runner.active_brain
    reply = runner.dispatch_command(switch('t-maze')); tx = runner._pending_assay_control
    claim = runner.claim_terminal_observation()
    runner.terminal_observation_failed(claim['attempt_token'], OSError('prefix unavailable'))
    assert not runner.command_acks[-1]['ack']['applied'] and runner.active_brain is source
    runner.observation_publication.retry_failed()
    claim = runner.claim_terminal_observation()
    durable = records.record_observation(claim['observation'])
    runner.terminal_observation_succeeded(claim['attempt_token'], durable)
    assert runner.active_brain is source and len(runner.command_acks) == 1
    assert tx['terminal']['observation']['identity']['assay'] == 'open-arena'
    records.close()


def test_real_nf_switch_event_is_durable_before_applied_ack(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    nf = RunRecorder(runner, tmp_path / 'actual.nfrec'); runner.recorder = nf
    original = nf.durable_command
    def inspect(request):
        assert not runner.command_acks
        receipt = original(request)
        rows = [json.loads(x) for x in zlib.decompressobj(31).decompress(nf._partial.read_bytes()).splitlines()]
        assert any(x.get('kind') == 'command' and x['cmd'] == switch('t-maze') for x in rows)
        return receipt
    monkeypatch.setattr(nf, 'durable_command', inspect)
    reply, tx = complete(runner, records, drain, switch('t-maze'))
    assert reply['ack']['applied'] and tx['recording_receipt']['operation_id'] == reply['command_id']
    nf.abort('cleanup'); runner.recorder = None; records.close()


def test_real_nf_event_failure_halts_uncertain_without_success(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    nf = RunRecorder(runner, tmp_path / 'actual.nfrec'); runner.recorder = nf
    def fail(*args): raise OSError('NF transition fsync unavailable')
    monkeypatch.setattr(nf, 'durable_command', fail)
    reply, _ = complete(runner, records, drain, {'action': 'reset_trial'})
    assert not reply['ack']['applied'] and nf.closed and runner._state_uncertain
    assert runner.last_error and 'halted' in runner.recording_error['message']
    if nf._cleanup_thread: nf._cleanup_thread.join(10)
    records.close()


def test_full_queue_refuses_transition_without_losing_evidence(tmp_path):
    runner, records, _ = attached(tmp_path, exploratory=True)
    runner.observation_publication.capacity = 1
    runner.dispatch_command(switch('t-maze'))
    tx = runner._pending_assay_control
    claim = runner.claim_terminal_observation()
    runner.terminal_observation_failed(claim['attempt_token'], OSError('retained prefix'))
    tx['failure_writer'].join(10)
    assert not tx['failure_writer'].is_alive()
    before = state(runner, tmp_path)
    reply = runner.dispatch_command({'action': 'reset_trial'})
    assert reply['status'] == 'error' and 'queue is full' in reply['message']
    assert state(runner, tmp_path) == before
    assert runner.observation_publication.pending_status()['pending_count'] == 1
    records.close()


def test_halted_recovery_waits_for_existing_exact_prefix(tmp_path):
    runner, records, drain = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    runner.step_once()
    with runner.lock:
        runner._state_uncertain = True
        runner._halt_on_error(RuntimeError('failed controller'), phase='step')
        runner._capture_terminal_observation(runner._last_step_result, 'fault_halt')
    terminal = runner._observation_terminal
    frozen = copy.deepcopy(terminal['observation'])
    source = runner.registry.active
    result, tx = complete(runner, records, drain, switch('t-maze'))
    assert result['ack']['applied'] and result['cleared_error']['message'] == 'failed controller'
    assert tx['terminal'] is terminal and terminal['observation'] == frozen
    assert terminal['durable'] and source.brain is None
    assert runner.last_error is None and not runner._state_uncertain
    assert runner._live_observation()['identity']['assay'] == 't-maze'
    records.close()


def test_modular_to_graph_controller_resolves_actual_prepared_graph(tmp_path):
    runner, records, drain = attached(tmp_path, test_synthetic_graph=True)
    result, _ = complete(runner, records, drain, {'action': 'switch_controller', 'backend': 'connectome-fixed'})
    assert result['ack']['applied']
    assert runner.graph_controller._prepared_graph is runner.shared_graph
    assert runner.graph_controller.dn_indices == runner.shared_graph.io_map
    records.close()


@pytest.mark.parametrize('phase', ['restore', 'constructor', 'index', 'checkpoint'])
def test_registry_activation_failure_never_releases_usable_source(tmp_path, monkeypatch, phase):
    import experiment_registry as registry_module
    runner, records, _ = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    registry, source = runner.registry, runner.registry.active
    def fail(*args, **kwargs): raise RuntimeError('activation '+phase+' failed')
    if phase == 'restore': monkeypatch.setattr(registry, '_restore', fail)
    elif phase == 'constructor': monkeypatch.setattr(registry_module, 'GraphInstance', fail)
    elif phase == 'index': monkeypatch.setattr(registry, '_write_index', fail)
    else: monkeypatch.setattr(registry, 'checkpoint', fail)
    with pytest.raises(RuntimeError): registry.activate('t-maze', 'connectome-fixed')
    assert registry.active is source and source.brain is not None
    records.close()


def test_prepared_activation_cannot_replace_a_changed_source(tmp_path):
    runner, records, _ = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    registry, source = runner.registry, runner.registry.active
    prepared = registry.prepare_activation('t-maze', 'connectome-fixed')
    registry.active = None
    with pytest.raises(RuntimeError, match='source changed'): registry.commit_activation(prepared)
    registry.cancel_activation(prepared)
    assert source.brain is not None and prepared['target'].brain is None
    registry.active = source
    records.close()


def test_successful_transition_fsync_and_source_cleanup_never_wait_under_runner_lock(tmp_path, monkeypatch):
    import os
    runner, records, drain = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    source = runner.registry.active
    original_sync, original_release = os.fsync, source.release
    checked = []
    def off_lock(operation, *args):
        assert runner.lock.acquire(timeout=0.2), 'persistence or cleanup owns runner.lock'
        runner.lock.release(); checked.append(operation.__name__)
        return operation(*args)
    monkeypatch.setattr(os, 'fsync', lambda fd: off_lock(original_sync, fd))
    monkeypatch.setattr(source, 'release', lambda: off_lock(original_release))
    result, _ = complete(runner, records, drain, switch('t-maze'))
    assert result['ack']['applied'] and 'fsync' in checked and 'release' in checked
    records.close()


def test_save_halt_recovery_requires_target_checkpoint_and_retains_invalidity(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path, backend='connectome-fixed', test_synthetic_graph=True)
    with runner.lock:
        runner._persistence_failed('checkpoint', OSError('source checkpoint failed'))
    original = runner.registry.checkpoint
    def fail(*args, **kwargs): raise OSError('target checkpoint still fails')
    monkeypatch.setattr(runner.registry, 'checkpoint', fail)
    result, tx = complete(runner, records, drain, switch('t-maze'))
    assert not result['ack']['applied'] and runner.last_error and runner._state_uncertain
    assert tx['terminal']['durable'] and runner.result_validity()['state'] == 'incomplete'
    monkeypatch.setattr(runner.registry, 'checkpoint', original)
    result, _ = complete(runner, records, drain, switch('open-arena'))
    assert result['ack']['applied'] and runner.last_error is None
    assert 'checkpoint' not in runner.persistence.describe()['failing']
    records.close()


def test_failed_target_preparation_resolves_ack_while_failure_log_is_blocked(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    source, identity = runner.active_brain, runner.identity()
    entered, release = threading.Event(), threading.Event()
    target = runner.brains.get('t-maze')
    original_source_log, original_target_log = source.log, target.log
    def refused(kind, **fields):
        if kind == 'transition_prepared': raise OSError('preparation event unavailable')
        return original_target_log(kind, **fields)
    def blocked(kind, **fields):
        if kind == 'control_transition_failed':
            entered.set(); assert release.wait(10)
        return original_source_log(kind, **fields)
    monkeypatch.setattr(target, 'log', refused)
    monkeypatch.setattr(source, 'log', blocked)
    runner.dispatch_command(switch('t-maze')); tx = runner._pending_assay_control
    drain.poll_once()
    try:
        assert tx['entry']['done'].wait(2) and entered.wait(2)
        assert not tx['entry']['result']['ack']['applied']
        assert runner.active_brain is source and runner.identity() == identity
        assert runner.lock.acquire(timeout=0.2); runner.lock.release()
        assert runner.dispatch_command({'action': 'set_speed', 'speed': 3})['status'] == 'ok'
        assert runner.last_error and runner._state_uncertain
    finally:
        release.set(); tx['writer'].join(10); tx['failure_writer'].join(10)
        records.close()


def test_target_event_directory_fsync_failure_cannot_acknowledge_applied(tmp_path, monkeypatch):
    import os
    import stat
    runner, records, drain = attached(tmp_path)
    target = runner.brains.get('t-maze')
    original = os.fsync
    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) and os.readlink(f'/proc/self/fd/{fd}') == str(target.directory):
            raise OSError('target event directory durability failed')
        return original(fd)
    monkeypatch.setattr(os, 'fsync', fail_directory)
    result, tx = complete(runner, records, drain, switch('t-maze'))
    assert not result['ack']['applied'] and runner.last_error and runner._state_uncertain
    assert tx['terminal']['durable'] and len(runner.command_acks) == 1
    assert runner._finish_assay_control() is False
    assert len(runner.command_acks) == 1
    records.close()
