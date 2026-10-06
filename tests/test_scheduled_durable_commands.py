"""Scheduled inputs use the central exact-owner durability transaction."""
import copy
import json
import threading

import pytest

import neurofly_daemon as daemon
from learning_recorder import LearningRecorder
from neurofly.recording import (_Ordinals, _RecordingProjection, RunRecorder,
                                frame_from_telemetry, read_recording, record_run,
                                source_telemetry_from_frame)
from tests.test_assay_control_transactions import attached


def tick(runner):
    with runner.lock:
        return runner._advance_one()


def test_local_writer_initialization_off_lock_and_owned_cleanup(tmp_path, monkeypatch):
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path)
    original = daemon.LearningRecorder
    def create(*args, **kwargs):
        assert runner.lock.acquire(blocking=False)
        runner.lock.release()
        return original(*args, **kwargs)
    monkeypatch.setattr(daemon, 'LearningRecorder', create)
    runner.schedule_command(0, {'action': 'set_speed', 'speed': 2})
    owner = runner._scheduled_records_owner
    assert owner is runner.learning_records and owner.is_alive()
    assert runner._close_scheduled_records() and not owner.is_alive() and owner._closed
    assert json.loads((owner.recorder.data_dir / 'sessions.jsonl').read_text().splitlines()[0])['source'] == 'standalone_scheduled_api'


@pytest.mark.parametrize('failure', ['create', 'start'])
def test_startup_failure_has_no_entry_and_releases_owned_recorder(tmp_path, monkeypatch, failure):
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path)
    def fail(*args, **kwargs):
        assert runner.lock.acquire(blocking=False)
        runner.lock.release()
        raise OSError('scheduled startup failure')
    if failure == 'create':
        monkeypatch.setattr(daemon, 'LearningRecorder', fail)
    else:
        monkeypatch.setattr(daemon.RecorderThread, 'start', fail)
    with pytest.raises(OSError, match='scheduled startup failure'):
        runner.schedule_command(0, {'action': 'teach_brain', 'pairs': 1})
    assert runner.learning_records is None and runner._scheduled_records_owner is None
    assert runner._scheduled == {} and runner.total_steps == 0
    if failure == 'start':
        again = LearningRecorder(tmp_path / 'scheduled-records' / runner.run_id)
        again.close()


def test_configured_opt_out_never_enables_a_writer_and_refuses_truthfully(tmp_path):
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path, standalone_scheduled_records=False)
    entry = runner.schedule_command(0, {'action': 'teach_brain', 'pairs': 1})
    tick(runner)
    assert entry['done'].wait(3)
    assert entry['result']['status'] == 'error' and not entry['result']['ack']['applied']
    assert runner.learning_records is None and runner._scheduled_records_owner is None
    assert not (tmp_path / 'scheduled-records').exists()
    tick(runner)
    assert runner.total_steps == 0 and runner.last_error


def test_attached_writer_is_never_replaced_or_closed(tmp_path):
    runner, records, drain = attached(tmp_path)
    try:
        entry = runner.schedule_command(0, {'action': 'set_speed', 'speed': 2})
        tick(runner)
        assert entry['done'].is_set() and entry['result']['status'] == 'ok'
        assert runner.learning_records is drain and runner._scheduled_records_owner is None
        assert runner._close_scheduled_records() and not drain._closed
        assert not (runner.output_dir / 'scheduled-records').exists()
    finally:
        records.close()


@pytest.mark.parametrize('fail', [False, True])
def test_exact_step_barrier_same_boundary_order_and_failure_ack(tmp_path, monkeypatch, fail):
    runner, records, drain = attached(tmp_path, 't-maze')
    writer = RunRecorder(runner, tmp_path / 'actual.nfrec')
    runner.recorder = writer
    entered, release = threading.Event(), threading.Event()
    original = writer.durable_command
    def stalled(request):
        entered.set()
        assert release.wait(3)
        if fail:
            raise OSError('scheduled command fsync failure')
        return original(request)
    monkeypatch.setattr(writer, 'durable_command', stalled)
    first = runner.schedule_command(1, {'action': 'reset_trial', 'advance': True})
    second = runner.schedule_command(1, {'action': 'reset_trial', 'advance': True})
    try:
        tick(runner); tick(runner)
        assert runner.total_steps == 1 and first['result'] is None and not first['done'].is_set()
        drain.poll_once()
        assert entered.wait(2)
        assert runner.lock.acquire(blocking=False)
        runner.lock.release()
        for _ in range(3): tick(runner)
        assert runner.total_steps == 1 and second['result'] is None and not second['done'].is_set()
        release.set()
        assert first['done'].wait(3)
        assert first['result']['ack']['applied_step'] == 1
        if fail:
            assert first['result']['status'] == 'error' and not first['result']['ack']['applied']
            tick(runner)
            assert runner.total_steps == 1 and second['result'] is None
            assert runner.last_error and not writer.artifacts()['certified']
        else:
            assert first['result']['status'] == 'ok' and first['result']['ack']['applied']
            tick(runner)
            drain.poll_once()
            assert second['done'].wait(3)
            assert second['result']['status'] == 'ok' and second['result']['ack']['applied_step'] == 1
            assert first['id'] != second['id']
            tick(runner)
            assert runner.total_steps == 2
    finally:
        release.set()
        if not writer.closed: writer.abort('test cleanup')
        if writer._cleanup_thread: writer._cleanup_thread.join(3)
        records.close()


def test_refused_input_is_durable_evidence_not_applied_command(tmp_path):
    command = {'action': 'teach_brain', 'pairs': 0}
    summary = record_run(paradigm='t-maze', out=tmp_path / 'refused', steps=10,
                         schedule=[{'step': 3, 'cmd': command}])
    rec = read_recording(tmp_path / 'refused.nfrec')
    refused = [e for e in rec['events'] if e.get('cmd', {}).get('action') == 'scheduled_command_refused']
    assert len(refused) == 1 and refused[0]['step'] == 3
    assert refused[0]['cmd']['requested_command'] == command and refused[0]['cmd']['applied'] is False
    assert summary['status'] == 'incomplete' and summary['error']
    assert rec['end']['last_step'] == 3


def test_operational_ages_roundtrip_both_paths_without_stripping_science_or_status():
    health = {'started': False, 'alive': False, 'active_write': None,
              'start_age_s': 1.5, 'last_progress_age_s': 0, 'active_write_age_s': None}
    source = {k: {'durability': {'enabled': False, 'state': 'disabled', 'reason': 'unavailable',
                               'recorder': copy.deepcopy(health)}}
              for k in ('observation_lifecycle', 'observation_publication')}
    source['observation'] = {'records': {'latency_s': 1.5}, 'sim_time_s': 0}
    projection = _RecordingProjection(_Ordinals('s'))
    frame = frame_from_telemetry(source, _Ordinals('unused'), projection, 0)
    frame.update(k='f', i=0, activity=None, spikes=None)
    for key in ('observation_lifecycle', 'observation_publication'):
        durability = frame[key]['durability']
        assert durability['enabled'] is False and durability['state'] == 'disabled'
        assert durability['reason'] == 'unavailable' and durability['recorder']['started'] is False
        assert durability['recorder']['alive'] is False and durability['recorder']['active_write'] is None
        assert all(durability['recorder'][age] is None for age in ('start_age_s', 'last_progress_age_s', 'active_write_age_s'))
    assert frame['observation']['records']['latency_s'] == 1.5 and frame['observation']['sim_time_s'] == 0
    assert source_telemetry_from_frame(frame, projection.sidecar()) == source


def test_stop_closes_only_owned_writer_and_schedule_cannot_reopen(tmp_path):
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path)
    runner.schedule_command(10, {'action': 'teach_brain', 'pairs': 1})
    owner = runner._scheduled_records_owner
    assert runner.stop()
    assert owner._closed and not owner.is_alive()
    with pytest.raises(RuntimeError, match='after stop'):
        runner.schedule_command(0, {'action': 'set_speed', 'speed': 2})
    assert runner.learning_records is owner


def test_stop_deadline_does_not_wait_for_stalled_owned_recorder_close(tmp_path, monkeypatch):
    import time
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path)
    runner.schedule_command(10, {'action': 'teach_brain', 'pairs': 1})
    owner = runner._scheduled_records_owner
    entered, release = threading.Event(), threading.Event()
    original = owner.recorder.close
    def close():
        entered.set()
        assert release.wait(3)
        original()
    monkeypatch.setattr(owner.recorder, 'close', close)
    runner.assay_control_timeout_s = 0.1
    started = time.monotonic()
    try:
        assert runner.stop() is False
        assert time.monotonic() - started < 0.4
        assert entered.is_set() and not owner._closed
        assert runner.lock.acquire(blocking=False)
        runner.lock.release()
    finally:
        release.set()
        runner._scheduled_records_cleanup_thread.join(3)
    assert owner._closed and not owner.is_alive()


@pytest.mark.parametrize('step', [-1, 0.5, True, '0', 0])
def test_invalid_or_past_step_creates_no_writer(tmp_path, step):
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path)
    runner.total_steps = 1
    with pytest.raises(ValueError):
        runner.schedule_command(step, {'action': 'set_speed', 'speed': 2})
    assert runner._scheduled == {} and runner.learning_records is None
    assert runner._scheduled_records_owner is None and not (tmp_path / 'scheduled-records').exists()


def test_startup_race_rejection_closes_unused_new_owner_off_lock(tmp_path, monkeypatch):
    runner = daemon.ContinuousExperimentRunner(output_dir=tmp_path)
    original = daemon.LearningRecorder
    created = []
    def advance_during_creation(*args, **kwargs):
        with runner.lock:
            runner.total_steps = 1
        value = original(*args, **kwargs)
        created.append(value)
        return value
    monkeypatch.setattr(daemon, 'LearningRecorder', advance_during_creation)
    original_stop = daemon.RecorderThread.stop
    def stop(owner, *args, **kwargs):
        assert runner.lock.acquire(blocking=False)
        runner.lock.release()
        return original_stop(owner, *args, **kwargs)
    monkeypatch.setattr(daemon.RecorderThread, 'stop', stop)
    with pytest.raises(ValueError, match='past'):
        runner.schedule_command(0, {'action': 'set_speed', 'speed': 2})
    assert len(created) == 1 and created[0]._closed
    assert runner.learning_records is None and runner._scheduled_records_owner is None
    assert runner._scheduled == {} and not runner._scheduled_records_retiring
    # Historical scratch is retained; only ownership/resources are retired.
    assert (tmp_path / 'scheduled-records' / runner.run_id / 'sessions.jsonl').exists()


def test_identically_named_scientific_provenance_ages_remain_canonical_science():
    from neurofly.recording import _line
    first = {'observation': {'provenance': {'durability': {'recorder': {'start_age_s': 1.0}}}}}
    second = copy.deepcopy(first)
    second['observation']['provenance']['durability']['recorder']['start_age_s'] = 2.0
    a = frame_from_telemetry(first, _Ordinals('s'))
    b = frame_from_telemetry(second, _Ordinals('s'))
    assert a['observation']['provenance']['durability']['recorder']['start_age_s'] == 1.0
    assert _line(a) != _line(b)
