"""Command-owned terminal barriers preserve measured prefixes and physical state."""
import copy
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

import assay_controls
import neurofly_daemon
from learning_recorder import LearningRecorder, RecorderThread
from neurofly_daemon import ContinuousExperimentRunner
from observation_envelopes import dumps_observation_envelope


def attached(tmp_path, assay='open-arena', **kwargs):
    runner = ContinuousExperimentRunner(initial_paradigm=assay, trial_length_s=None,
                                        output_dir=tmp_path / 'runner', checkpoint_interval=3600,
                                        **kwargs)
    recorder = LearningRecorder(tmp_path / 'records', session={'daemon_run_id': runner.run_id})
    drain = RecorderThread(runner, recorder, summary_interval=999)
    runner.attach_learning_records(drain)
    return runner, recorder, drain


def poll_control(runner, drain):
    transaction = runner._pending_assay_control
    result = drain.poll_once()
    if transaction is not None and transaction.get('writer') is not None:
        assert transaction['entry']['done'].wait(2), 'control event writer did not finish'
    return result


def state(runner, root):
    return (copy.deepcopy(runner.arena.snapshot_world()),
            copy.deepcopy(runner.arena.observation_owner.snapshot_observation()),
            copy.deepcopy(runner.identity()),
            {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()})


def airflow(value):
    return {'action': 'set_param', 'name': 'windStrength', 'value': value}


def test_airflow_7_to_23_preserves_contact_pose_identity_and_exact_prefix(tmp_path):
    runner, recorder, drain = attached(tmp_path)
    runner.arena.wind = (-7.0, 0.0)
    owner = runner.arena.observation_owner
    owner.airflow_mm_s = 7.0
    runner.step_once()
    owner._in_contact = True
    identity, segment = runner.identity(), runner.segment_id
    pose = (runner.arena.fly.pos.x, runner.arena.fly.pos.y, runner.arena.fly.heading)
    count = owner._seg['contacts']
    reply = runner.dispatch_command(airflow(23))
    assert reply['status'] == 'queued' and not reply['ack']['applied']
    transaction = runner._pending_assay_control
    terminal = runner._observation_terminal
    frozen = dumps_observation_envelope(terminal['observation'], sort_keys=True)
    assert runner.arena.wind == (-7.0, 0.0)
    assert not transaction['entry']['done'].is_set()
    assert runner.observation_lifecycle_status()['phase'] == 'waiting_for_save'
    step = runner.total_steps
    runner.step_once()
    assert runner.total_steps == step
    assert poll_control(runner, drain)['observations'] == 1
    assert runner.arena.wind == (-23.0, 0.0)
    assert runner.identity() == identity and runner.segment_id == segment
    assert (runner.arena.fly.pos.x, runner.arena.fly.pos.y, runner.arena.fly.heading) == pose
    assert owner._in_contact is True and owner._seg['contacts'] == count
    assert owner.observation_status()['presentation_index'] == 1
    assert owner.snapshot_observation()['window_params']['airflow_mm_s'] == 23.0
    assert transaction['entry']['done'].is_set()
    assert transaction['entry']['result']['ack']['applied'] is True
    assert dumps_observation_envelope(runner.observation_publication_status()['last_terminal']['observation'], sort_keys=True) == frozen
    runner.step_once()
    assert runner.last_error is None and runner.total_steps == step + 1
    recorder.close()


@pytest.mark.parametrize('cmd', [airflow(float('nan')), airflow(True), airflow(99),
                                {'action': 'assay_action', 'name': 'not_connected'},
                                {'action': 'place_stimulus', 'type': 'wind', 'x': 20, 'y': 20},
                                {'action': 'place_stimulus', 'type': 'food', 'x': -1, 'y': 20}])
def test_rejection_has_no_world_owner_identity_or_ledger_changes(tmp_path, cmd):
    runner, recorder, _ = attached(tmp_path)
    before = state(runner, tmp_path)
    reply = runner.dispatch_command(cmd)
    assert reply['status'] == 'error' and not reply['ack']['applied']
    assert state(runner, tmp_path) == before
    recorder.close()


def test_action_boundary_and_intervention_only_spatial_provenance(tmp_path):
    runner, recorder, drain = attached(tmp_path, 'optomotor')
    runner.step_once()
    reply = runner.dispatch_command({'action': 'assay_action', 'name': 'reverse_grating'})
    assert reply['status'] == 'queued'
    old = runner.arena.paradigm.drum_velocity_deg_s
    poll_control(runner, drain)
    assert runner.arena.paradigm.drum_velocity_deg_s == -old
    assert runner.command_acks[-1]['ack']['applied']
    recorder.close()
    runner, recorder, _ = attached(tmp_path / 'spatial')
    before = runner.arena.observation_owner.observation_status()['presentation_index']
    reply = runner.dispatch_command({'action': 'place_stimulus', 'type': 'food', 'x': 20, 'y': 21})
    assert reply['status'] == 'queued'
    transaction = runner._pending_assay_control
    if transaction is not None:
        assert transaction['entry']['done'].wait(2)
    assert runner.command_acks[-1]['command_id'] == reply['command_id']
    assert runner.command_acks[-1]['ack']['applied']
    assert runner.arena.observation_owner.observation_status()['presentation_index'] == before
    assert runner.observation_publication.pending_status()['pending_count'] == 0
    rows = [json.loads(line) for line in (runner.active_brain.directory / 'open-arena.events.jsonl').read_text().splitlines()]
    assert rows[-1]['coordinates'] == {'x': 20.0, 'y': 21.0}
    assert rows[-1]['intervention_only'] is True
    recorder.close()


@pytest.mark.parametrize('phase', ['apply', 'begin', 'log'])
def test_partial_failures_halt_and_retain_old_durable_prefix(tmp_path, monkeypatch, phase):
    runner, recorder, drain = attached(tmp_path)
    runner.step_once()
    runner.dispatch_command(airflow(23))
    terminal = runner._observation_terminal
    frozen = dumps_observation_envelope(terminal['observation'], sort_keys=True)
    def fail(*args, **kwargs):
        if phase == 'apply':
            runner.arena.wind = (-19.0, 0.0)
        raise OSError('injected partial control failure')
    if phase == 'apply':
        monkeypatch.setattr(assay_controls, 'set_parameter', fail)
    elif phase == 'begin':
        monkeypatch.setattr(runner.arena.observation_owner, 'begin_next_presentation', fail)
    else:
        monkeypatch.setattr(runner.active_brain, 'log', fail)
    poll_control(runner, drain)
    assert runner.last_error is not None and runner._state_uncertain
    assert runner.result_validity()['state'] == 'incomplete'
    assert runner.command_acks[-1]['status'] == 'error' and not runner.command_acks[-1]['ack']['applied']
    assert runner._observation_terminal is terminal
    assert dumps_observation_envelope(terminal['observation'], sort_keys=True) == frozen
    assert terminal['durable'] is not None
    recorder.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_save_failure_error_once_late_ack_never_executes_control(tmp_path, monkeypatch, exploratory):
    runner, recorder, drain = attached(tmp_path, exploratory=exploratory)
    initial_wind = runner.arena.wind
    runner.step_once()
    reply = runner.dispatch_command(airflow(23))
    transaction = runner._pending_assay_control
    terminal = runner._observation_terminal
    frozen = dumps_observation_envelope(terminal['observation'], sort_keys=True)
    original = recorder.record_observation
    def fail(payload):
        raise OSError('save unavailable')
    monkeypatch.setattr(recorder, 'record_observation', fail)
    with pytest.raises(OSError):
        poll_control(runner, drain)
    assert transaction['entry']['done'].is_set()
    assert runner._pending_assay_control is None
    assert len(runner.command_acks) == 1 and runner.command_acks[-1]['status'] == 'error'
    assert runner.arena.wind == initial_wind
    assert runner.observation_publication.pending_status()['pending_count'] == 1
    if exploratory:
        assert runner.last_error is None
        assert 'unchanged condition' in runner.command_acks[-1]['message']
        assert runner.arena.observation_owner.observation_status()['presentation_index'] == 1
        assert runner.health()['status'] == 'degraded'
        runner.step_once()
    else:
        assert runner.last_error is not None and runner._observation_terminal is terminal
    assert runner.result_validity()['state'] == 'incomplete'
    monkeypatch.setattr(recorder, 'record_observation', original)
    runner.persistence.channels['learning_records']['next_retry_at'] = 0
    poll_control(runner, drain)
    assert runner.arena.wind == initial_wind
    assert len(runner.command_acks) == 1
    assert dumps_observation_envelope(runner.observation_publication_status()['last_terminal']['observation'], sort_keys=True) == frozen
    assert runner.result_validity()['state'] == 'incomplete'
    recorder.close()


def test_timeout_and_late_matching_ack_remain_failed(tmp_path):
    runner, recorder, _ = attached(tmp_path)
    initial_wind = runner.arena.wind
    runner.dispatch_command(airflow(23))
    transaction = runner._pending_assay_control
    claim = runner.claim_terminal_observation()
    runner.assay_control_timeout_s = 0
    runner.step_once()
    assert transaction['entry']['done'].is_set()
    assert not runner.command_acks[-1]['ack']['applied']
    receipt = recorder.record_observation(claim['observation'])
    runner.terminal_observation_succeeded(claim['attempt_token'], receipt)
    assert runner.arena.wind == initial_wind and len(runner.command_acks) == 1
    assert runner.last_error is not None
    recorder.close()


@pytest.mark.parametrize('exploratory', [False, True])
def test_control_timeout_preserves_failed_save_deadline_and_actual_attempts(tmp_path, monkeypatch, exploratory):
    runner, recorder, drain = attached(tmp_path, exploratory=exploratory)
    clock = SimpleNamespace(wall=1000.0, mono=1000.0)
    monkeypatch.setattr(neurofly_daemon.time, 'time', lambda: clock.wall)
    monkeypatch.setattr(neurofly_daemon.time, 'monotonic', lambda: clock.mono)
    original = recorder.record_observation

    def disk_full(_payload):
        raise OSError(28, 'disk full')

    monkeypatch.setattr(recorder, 'record_observation', disk_full)
    runner.dispatch_command(airflow(7))
    with pytest.raises(OSError):
        drain.poll_once()
    clock.wall = clock.mono = 1030.0
    with pytest.raises(OSError):
        drain.poll_once()
    failure = copy.deepcopy(runner.persistence.channels['learning_records'])
    pending = runner.observation_publication.pending_status()['entries'][0]
    assert failure['failures'] == pending['attempts'] == 2
    assert failure['next_retry_at'] == 1090.0
    wind, step = runner.arena.wind, runner.total_steps
    canceled = []
    # Each 30-second command timeout expires while the same actual retry waits.
    for start in (1030.0, 1060.0):
        clock.wall = clock.mono = start
        reply = runner.dispatch_command(airflow(23) if exploratory else
                                        {'action': 'switch_paradigm', 'paradigm': 'open-arena'})
        assert reply['status'] == 'queued'
        transaction = runner._pending_assay_control
        clock.wall = clock.mono = start + 30.0
        runner._check_assay_control_timeout()
        if transaction.get('failure_writer') is not None:
            transaction['failure_writer'].join(2)
            assert not transaction['failure_writer'].is_alive()
        assert transaction['entry']['done'].is_set()
        assert transaction['entry']['result']['ack']['applied'] is False
        canceled.append(reply['command_id'])
        assert runner.persistence.channels['learning_records'] == failure
        head = runner.observation_publication.pending_status()['entries'][0]
        assert head['attempts'] == 2 and head['payload_sha256'] == pending['payload_sha256']
        assert runner.arena.wind == wind and runner.total_steps == step
        if clock.wall < failure['next_retry_at']:
            assert runner.claim_terminal_observation() is None
    # An actual third write at the unchanged deadline still advances backoff.
    with pytest.raises(OSError):
        drain.poll_once()
    third = runner.persistence.channels['learning_records']
    assert third['failures'] == 3 and third['next_retry_at'] == 1210.0
    assert runner.observation_publication.pending_status()['entries'][0]['attempts'] == 3
    monkeypatch.setattr(recorder, 'record_observation', original)
    clock.wall = clock.mono = 1210.0
    assert drain.poll_once()['observations'] == 1
    durable = runner.observation_publication_status()['last_terminal']
    assert durable['receipt']['payload_sha256'] == pending['payload_sha256']
    assert runner.arena.wind == wind
    assert all(not ack['ack']['applied'] for ack in runner.command_acks
               if ack['command_id'] in canceled)
    if not exploratory:
        assert runner.last_error is not None
        reply = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 'open-arena'})
        transaction = runner._pending_assay_control
        if transaction is not None:
            assert transaction['entry']['done'].wait(2)
        assert runner.command_acks[-1]['command_id'] == reply['command_id']
        assert runner.command_acks[-1]['ack']['applied'] and runner.last_error is None
    assert runner.result_validity()['state'] == 'incomplete' or runner.result_validity()['other_runs_incomplete']
    recorder.close()


@pytest.mark.parametrize('claimed', [False, True])
def test_control_timeout_still_diagnoses_unclaimed_or_active_writer(tmp_path, monkeypatch, claimed):
    runner, recorder, _ = attached(tmp_path)
    clock = SimpleNamespace(wall=1000.0, mono=1000.0)
    monkeypatch.setattr(neurofly_daemon.time, 'time', lambda: clock.wall)
    monkeypatch.setattr(neurofly_daemon.time, 'monotonic', lambda: clock.mono)
    runner.dispatch_command(airflow(23))
    transaction = runner._pending_assay_control
    claim = runner.claim_terminal_observation() if claimed else None
    # A previous channel fault alone must not conceal a currently active write.
    if claimed:
        runner.persistence.failed('learning_records', OSError(28, 'previous disk full'), now=990.0)
    clock.wall = clock.mono = 1030.0
    runner._check_assay_control_timeout()
    failure = runner.persistence.channels['learning_records']
    assert failure['failures'] == (2 if claimed else 1)
    assert failure['last_failure_at'] == 1030.0 and failure['errno'] is None
    assert failure['error'].startswith('TimeoutError:')
    assert transaction['entry']['result']['ack']['applied'] is False
    assert runner.last_error is not None
    if claim is not None:
        receipt = recorder.record_observation(claim['observation'])
        runner.terminal_observation_succeeded(claim['attempt_token'], receipt)
        assert runner.arena.wind != (-23.0, 0.0)
    recorder.close()


def test_failed_observation_does_not_hide_active_legacy_recorder_write(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path)
    clock = SimpleNamespace(wall=1000.0, mono=1000.0)
    monkeypatch.setattr(neurofly_daemon.time, 'time', lambda: clock.wall)
    monkeypatch.setattr(neurofly_daemon.time, 'monotonic', lambda: clock.mono)
    runner.dispatch_command(airflow(7))
    claim = runner.claim_terminal_observation()
    runner.terminal_observation_failed(claim['attempt_token'], OSError(28, 'disk full'))
    runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 'open-arena'})
    transaction = runner._pending_assay_control
    # A legacy trials write can be active with no observation claim.
    drain._write_started('learning_records')
    clock.wall = clock.mono = 1030.0
    runner._check_assay_control_timeout()
    drain._write_finished()
    transaction['failure_writer'].join(2)
    failure = runner.persistence.channels['learning_records']
    assert failure['failures'] == 2 and failure['error'].startswith('TimeoutError:')
    assert failure['last_failure_at'] == 1030.0
    assert runner.observation_publication.pending_status()['entries'][0]['attempts'] == 1
    assert transaction['entry']['result']['ack']['applied'] is False
    recorder.close()


def test_rapid_controls_order_and_transport_responsiveness(tmp_path):
    runner, recorder, drain = attached(tmp_path)
    first = runner.dispatch_command(airflow(7))
    before = state(runner, tmp_path)
    rejected = runner.dispatch_command(airflow(23))
    assert rejected['status'] == 'error' and state(runner, tmp_path) == before
    assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
    assert runner.dispatch_command({'action': 'set_speed', 'speed': 3})['status'] == 'ok'
    poll_control(runner, drain)
    second = runner.dispatch_command(airflow(23))
    poll_control(runner, drain)
    assert [ack['command_id'] for ack in runner.command_acks] == [first['command_id'], second['command_id']]
    assert [ack['ack']['applied'] for ack in runner.command_acks] == [True, True]
    assert runner.arena.wind == (-23.0, 0.0)
    recorder.close()


def test_full_queue_refuses_without_replacing_unsaved_observation(tmp_path):
    runner, recorder, drain = attached(tmp_path, exploratory=True)
    runner.observation_publication.capacity = 1
    runner.dispatch_command(airflow(7))
    claim = runner.claim_terminal_observation()
    runner.terminal_observation_failed(claim['attempt_token'], OSError('retained save gap'))
    before = state(runner, tmp_path)
    reply = runner.dispatch_command(airflow(23))
    assert reply['status'] == 'error' and 'queue is full' in reply['message']
    assert state(runner, tmp_path) == before
    assert runner.observation_publication.pending_status()['pending_count'] == 1
    recorder.close()


def test_running_dispatch_waits_outside_runner_lock_for_final_ack(tmp_path):
    runner, recorder, drain = attached(tmp_path)
    runner.command_reply_wait_s = 0.02
    runner.start()
    try:
        reply = runner.dispatch_command(airflow(23))
        assert reply['status'] == 'queued'
        assert runner._pending_assay_control is not None
        assert runner.liveness()['state'] == 'waiting_for_save'
        assert not runner.command_acks
        poll_control(runner, drain)
        assert runner.command_acks[-1]['ack']['applied']
        assert runner.arena.wind == (-23.0, 0.0)
    finally:
        runner.stop()
        recorder.close()


def test_exploratory_retry_retains_two_prefixes_and_applies_only_retry(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path, exploratory=True)
    runner.step_once()
    first = runner.dispatch_command(airflow(7))
    frozen = dumps_observation_envelope(runner._observation_terminal['observation'], sort_keys=True)
    original = recorder.record_observation
    def fail(payload):
        raise OSError('first prefix unavailable')
    monkeypatch.setattr(recorder, 'record_observation', fail)
    with pytest.raises(OSError):
        poll_control(runner, drain)
    runner.step_once()
    second = runner.dispatch_command(airflow(23))
    pending = runner.observation_publication.pending_status()
    assert pending['pending_count'] == 2
    assert len({e['observation_key']['presentation_id'] for e in pending['entries']}) == 2
    assert runner.arena.wind != (-7.0, 0.0)
    monkeypatch.setattr(recorder, 'record_observation', original)
    runner.persistence.channels['learning_records']['next_retry_at'] = 0
    poll_control(runner, drain)
    poll_control(runner, drain)
    assert runner.observation_publication.pending_status()['pending_count'] == 0
    assert runner.arena.wind == (-23.0, 0.0)
    assert [a['command_id'] for a in runner.command_acks] == [first['command_id'], second['command_id']]
    assert [a['ack']['applied'] for a in runner.command_acks] == [False, True]
    assert runner.arena.observation_owner.observation_status()['presentation_index'] == 2
    assert runner.result_validity()['state'] == 'incomplete'
    recorder.close()


def test_blocked_prefix_write_yields_runner_lock_and_delays_applied_ack(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path)
    runner.dispatch_command(airflow(23))
    transaction = runner._pending_assay_control
    entered, release = threading.Event(), threading.Event()
    original = recorder.record_observation
    def blocked(payload):
        entered.set()
        assert release.wait(2)
        return original(payload)
    monkeypatch.setattr(recorder, 'record_observation', blocked)
    writer = threading.Thread(target=drain.poll_once)
    writer.start()
    try:
        assert entered.wait(1)
        assert runner.lock.acquire(timeout=0.5)
        try:
            assert runner.arena.wind != (-23.0, 0.0)
            assert not transaction['entry']['done'].is_set()
            assert runner._apply_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
            runner.step_once()
            assert runner.total_steps == 0
        finally:
            runner.lock.release()
    finally:
        release.set()
        writer.join(2)
        assert transaction['entry']['done'].wait(2)
        recorder.close()
    assert not writer.is_alive()
    assert transaction['entry']['done'].is_set() and runner.command_acks[-1]['ack']['applied']


def test_exploratory_failed_restart_halts_and_retains_failed_prefix(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path, exploratory=True)
    runner.dispatch_command(airflow(23))
    terminal = runner._observation_terminal
    def fail(*args, **kwargs):
        raise OSError('injected failure')
    monkeypatch.setattr(recorder, 'record_observation', fail)
    monkeypatch.setattr(runner.arena.observation_owner, 'begin_next_presentation', fail)
    with pytest.raises(OSError):
        poll_control(runner, drain)
    assert runner.last_error is not None and runner._state_uncertain
    assert runner._observation_terminal is terminal
    assert runner.observation_publication.pending_status()['pending_count'] == 1
    assert len(runner.command_acks) == 1 and not runner.command_acks[-1]['ack']['applied']
    recorder.close()


@pytest.mark.parametrize('cmd', [airflow(23), {'action': 'place_stimulus', 'type': 'food', 'x': 20, 'y': 21}])
def test_blocked_control_event_writer_is_responsive_and_owns_final_ack(tmp_path, monkeypatch, cmd):
    runner, recorder, drain = attached(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'assay_control':
            entered.set()
            assert release.wait(2)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    reply = runner.dispatch_command(cmd)
    transaction = runner._pending_assay_control
    if cmd['action'] == 'set_param':
        drain.poll_once()
    assert entered.wait(1)
    try:
        assert not transaction['entry']['done'].is_set()
        assert not runner.command_acks
        assert runner.observation_lifecycle_status()['phase'] == 'waiting_for_save'
        assert runner.lock.acquire(timeout=0.2), 'event fsync must not hold runner.lock'
        runner.lock.release()
        assert runner.dispatch_command({'action': 'set_paused', 'paused': True})['status'] == 'ok'
        assert runner.dispatch_command(airflow(7))['status'] == 'error'
        before = runner.total_steps
        runner.step_once()
        assert runner.total_steps == before
    finally:
        release.set()
        transaction['writer'].join(2)
    assert transaction['entry']['done'].is_set()
    assert len(runner.command_acks) == 1
    assert runner.command_acks[-1]['command_id'] == reply['command_id']
    assert runner.command_acks[-1]['ack']['applied']
    recorder.close()


def test_control_event_timeout_late_completion_never_acknowledges_applied(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path, exploratory=True)
    entered, release = threading.Event(), threading.Event()
    original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'assay_control':
            entered.set()
            assert release.wait(2)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    runner.dispatch_command(airflow(23))
    transaction = runner._pending_assay_control
    terminal = runner._observation_terminal
    drain.poll_once()
    assert entered.wait(1)
    try:
        runner.assay_control_timeout_s = 0
        runner.step_once()
        assert runner.last_error is not None and runner._state_uncertain
        assert runner.arena.wind == (-23.0, 0.0)
        assert runner._observation_terminal is terminal and terminal['durable'] is not None
        assert len(runner.command_acks) == 1 and not runner.command_acks[-1]['ack']['applied']
    finally:
        release.set()
        transaction['writer'].join(2)
    assert len(runner.command_acks) == 1
    assert runner.result_validity()['state'] == 'incomplete'
    recorder.close()


def test_event_writer_start_failure_halts_without_final_applied_ack(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path)
    runner.dispatch_command(airflow(23))
    def fail_start(self):
        raise RuntimeError('cannot start control event writer')
    monkeypatch.setattr(threading.Thread, 'start', fail_start)
    drain.poll_once()
    assert runner.last_error is not None and runner._state_uncertain
    assert len(runner.command_acks) == 1 and not runner.command_acks[-1]['ack']['applied']
    assert runner._observation_terminal['durable'] is not None
    recorder.close()


def test_optional_command_recording_failure_has_no_false_applied_ack(tmp_path):
    runner, recorder, drain = attached(tmp_path)
    class FailingRecording:
        def command(self, *args):
            raise OSError('command recording unavailable')
        def abort(self, *args):
            pass
    runner.recorder = FailingRecording()
    runner.dispatch_command(airflow(23))
    poll_control(runner, drain)
    assert runner.last_error is not None and runner._state_uncertain
    assert len(runner.command_acks) == 1 and not runner.command_acks[-1]['ack']['applied']
    assert runner._observation_terminal['durable'] is not None
    recorder.close()


def test_no_recorder_cannot_acknowledge_presentation_change(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm='open-arena', output_dir=tmp_path)
    before = state(runner, tmp_path)
    reply = runner.dispatch_command(airflow(23))
    assert reply['status'] == 'error' and not reply['ack']['applied']
    assert 'durable observation recorder is required' in reply['message']
    assert state(runner, tmp_path) == before


def test_halt_while_saving_event_cannot_be_acknowledged_as_applied(tmp_path, monkeypatch):
    runner, recorder, drain = attached(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = runner.active_brain.log
    def blocked(kind, **fields):
        if kind == 'assay_control':
            entered.set()
            assert release.wait(2)
        return original(kind, **fields)
    monkeypatch.setattr(runner.active_brain, 'log', blocked)
    runner.dispatch_command(airflow(23))
    transaction = runner._pending_assay_control
    drain.poll_once()
    assert entered.wait(1)
    with runner.lock:
        runner._state_uncertain = True
        runner._halt_on_error(RuntimeError('independent halt while saving control'))
    release.set()
    transaction['writer'].join(2)
    assert transaction['entry']['done'].is_set()
    assert len(runner.command_acks) == 1 and not runner.command_acks[-1]['ack']['applied']
    assert runner.last_error is not None
    recorder.close()
