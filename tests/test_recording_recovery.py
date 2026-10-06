"""Requested raw-recording recovery on a local modular fixture; no service/UI."""
import copy
import errno
import json

import pytest

import neurofly_daemon as nd
from neurofly import recording
from tests.transition_control_helpers import transition_command


@pytest.fixture
def runner(tmp_path):
    r = nd.ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=tmp_path,
                                      checkpoint_interval=3600)
    yield r
    if r.recorder is not None:
        r.stop_recording()


def enospc(*args, **kwargs):
    raise OSError(errno.ENOSPC, 'injected capture failure')


def fail_capture(runner, monkeypatch):
    runner.start_recording(name='original')
    writer = runner.recorder
    original = recording.RunRecorder.capture
    monkeypatch.setattr(recording.RunRecorder, 'capture', enospc)
    runner.step_once()
    if writer._cleanup_thread:
        writer._cleanup_thread.join(3)
        assert not writer._cleanup_thread.is_alive()
    assert runner.last_error and runner.recorder is None
    assert writer._partial.is_file()
    return writer, original


def owner(runner):
    return dict(identity=copy.deepcopy(runner.identity()), step=runner.total_steps,
                activation=runner.activation, segment=runner.segment_id,
                brain=runner.active_brain, arena=runner.arena,
                world=repr(runner.arena.snapshot_world()),
                saved_brain=(runner.active_brain.path.read_bytes()
                             if runner.active_brain.path.exists() else None))


@pytest.mark.parametrize('command', [
    {'action': 'switch_paradigm', 'paradigm': 't-maze'},
    {'action': 'switch_paradigm', 'paradigm': 'y-maze'},
    {'action': 'switch_backend', 'backend': 'modular'},
    {'action': 'reset_trial'},
])
def test_armed_capture_failure_refuses_before_owner_or_brain_mutation(runner, monkeypatch, command):
    writer, _ = fail_capture(runner, monkeypatch)
    before, prefix = owner(runner), writer._partial.read_bytes()
    reply = runner.dispatch_command(command)
    assert reply['status'] == 'error' and 'new name' in reply['message']
    assert owner(runner) == before
    assert runner._pending_assay_control is None
    runner.step_once()
    assert runner.total_steps == before['step']
    assert writer._partial.read_bytes() == prefix
    assert runner.health()['result_validity']['state'] == 'incomplete'
    assert runner.persistence.describe()['failing']['recording']['required']


def test_clear_fault_alone_is_not_recovery_and_existing_prefix_cannot_be_reused(runner, monkeypatch):
    writer, capture = fail_capture(runner, monkeypatch)
    prefix = writer._partial.read_bytes()
    monkeypatch.setattr(recording.RunRecorder, 'capture', capture)
    # Healthy checkpoint storage does not prove an absent raw recorder works.
    assert runner.checkpoint_now('recovery-probe') is not None
    before = owner(runner)
    reply = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 't-maze'})
    assert reply['status'] == 'error' and owner(runner) == before
    reply = runner.dispatch_command({'action': 'record_start', 'name': 'original'})
    assert reply['status'] == 'error' and 'already exists' in reply['message']
    assert runner.recorder is None and runner.last_error
    assert writer._partial.read_bytes() == prefix
    assert runner._clear_error('unverified-test') is None
    assert runner._recovery_gate(runner.error_detail, runner._stopping_failures,
                                 verified_checkpoint=True)['status'] == 'error'


def test_replacement_requires_actual_capture_then_reselection_recovers(runner, monkeypatch):
    writer, capture = fail_capture(runner, monkeypatch)
    failed_run = runner.identity()['run_id']
    prefix = writer._partial.read_bytes()
    # Fresh header construction alone cannot bypass the still-armed capture fault.
    rejected = []
    def fail_initial(self, *args, **kwargs):
        rejected.append(self)
        enospc()
    monkeypatch.setattr(recording.RunRecorder, 'capture', fail_initial)
    reply = runner.dispatch_command({'action': 'record_start', 'name': 'replacement-failed'})
    assert reply['status'] == 'error' and runner.recorder is None and runner.last_error
    assert 'recording' in runner.persistence.describe()['failing']
    assert rejected[0].closed and rejected[0].aborted
    rejected[0]._cleanup_thread.join(3)
    assert not rejected[0]._cleanup_thread.is_alive()
    # Returning without producing frame0 is also not recovery evidence.
    def skip_initial(self, *args, **kwargs):
        rejected.append(self)
        return False
    monkeypatch.setattr(recording.RunRecorder, 'capture', skip_initial)
    reply = runner.dispatch_command({'action': 'record_start', 'name': 'replacement-skipped'})
    assert reply['status'] == 'error' and 'Initial recording capture' in reply['message']
    assert runner.recorder is None and 'recording' in runner.persistence.describe()['failing']
    rejected[-1]._cleanup_thread.join(3)
    assert not rejected[-1]._cleanup_thread.is_alive()
    monkeypatch.setattr(recording.RunRecorder, 'capture', capture)
    reply = runner.dispatch_command({'action': 'record_start', 'name': 'replacement'})
    assert reply['status'] == 'ok'
    assert runner.recorder.frames >= 1 and runner.recorder.last_step == runner.total_steps
    assert 'recording' not in runner.persistence.describe()['failing']
    assert runner.last_error  # recording repair does not itself lift the halt
    reply = transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 't-maze'})
    assert reply['status'] == 'ok' and runner.last_error is None
    recovered = runner.health()['result_validity']
    assert failed_run in recovered['other_runs_incomplete']
    assert recovered['run_id'] != failed_run
    assert any(i['run_id'] == failed_run and i['channel'] == 'recording' for i in runner.incidents)
    assert writer._partial.read_bytes() == prefix
    step = runner.total_steps
    runner.step_once()
    assert runner.total_steps == step + 1 and runner.recorder is not None


def test_exploratory_capture_failure_can_continue_and_reselect(runner, monkeypatch):
    runner.exploratory = True
    runner.start_recording(name='exploratory')
    monkeypatch.setattr(recording.RunRecorder, 'capture', enospc)
    runner.step_once()
    assert runner.recorder is None and runner.last_error is None
    step = runner.total_steps
    runner.step_once()
    assert runner.total_steps == step + 1
    reply = transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 'y-maze'})
    assert reply['status'] == 'ok' and runner.last_error is None
    assert runner.health()['status'] == 'degraded'


def test_unrecorded_scientific_run_has_no_raw_recording_requirement(runner):
    reply = transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 'y-maze'})
    assert reply['status'] == 'ok' and runner.recorder is None
    assert runner._raw_recording_recovery_refusal() is None
    step = runner.total_steps
    runner.step_once()
    assert runner.total_steps == step + 1


def test_recording_error_reports_artifact_history_not_frozen_execution_state(runner, monkeypatch):
    fail_capture(runner, monkeypatch)
    error = runner.recording_error
    assert error['run_id'] == runner.identity()['run_id']
    assert 'The run stopped' not in error['message'] and 'keeps running' not in error['message']
    assert 'verified initial capture' in error['message']
    assert 'record_start' in runner.error_detail['recover']
    assert '--record NEW_NAME' in runner.error_detail['recover']


@pytest.mark.parametrize('stage,close_error', [('header', False), ('header', True), ('gzip', False)])
def test_rejected_header_setup_closes_handles_keeps_prefix_and_cleanup_error(runner, monkeypatch, stage, close_error):
    original_writer, capture = fail_capture(runner, monkeypatch)
    prefix = original_writer._partial.read_bytes()
    monkeypatch.setattr(recording.RunRecorder, 'capture', capture)
    rejected = []
    initialize = recording.RunRecorder.__init__
    def track(self, *args, **kwargs):
        rejected.append(self)
        return initialize(self, *args, **kwargs)
    monkeypatch.setattr(recording.RunRecorder, '__init__', track)
    def fail_header(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'injected header setup failure')
    if stage == 'gzip':
        monkeypatch.setattr(recording.gzip, 'GzipFile', fail_header)
    else:
        monkeypatch.setattr(recording.gzip.GzipFile, 'write', fail_header)
        if close_error:
            close = recording.gzip.GzipFile.close
            def close_then_fail(self):
                close(self)
                raise OSError(errno.EIO, 'injected cleanup close failure')
            monkeypatch.setattr(recording.gzip.GzipFile, 'close', close_then_fail)
    reply = runner.dispatch_command({'action': 'record_start', 'name': 'rejected-header'})
    assert reply['status'] == 'error' and 'injected header setup failure' in reply['message']
    assert runner.recorder is None and runner.last_error
    assert 'recording' in runner.persistence.describe()['failing']
    writer = rejected[0]
    assert writer.closed and 'injected header setup failure' in writer.aborted
    writer._cleanup_thread.join(3)
    assert not writer._cleanup_thread.is_alive() and writer._raw.closed
    assert writer._gz is None or writer._gz.closed
    assert writer._partial.is_file() and not writer.path.exists()
    assert json.loads((writer._projection.spool.path / 'status.json').read_text())['state'] == 'failed'
    assert original_writer._partial.read_bytes() == prefix
    if close_error:
        assert 'injected cleanup close failure' in writer._cleanup_error
    else:
        assert writer._cleanup_error is None
