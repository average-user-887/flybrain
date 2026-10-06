"""Off-loop validity delivery preserves the last trustworthy frame; no services/GPU."""
import copy
import io
import itertools
import json
from types import SimpleNamespace

import pytest

import neurofly_daemon as nd
from tests.transition_control_helpers import transition_command


def runner_at_frame(tmp_path, **kwargs):
    runner = nd.ContinuousExperimentRunner(initial_paradigm='t-maze',
        output_dir=tmp_path, checkpoint_interval=3600, **kwargs)
    runner.step_once()
    runner._publish_snapshot()
    return runner


@pytest.mark.parametrize('exploratory,validity', [(False, 'invalidated'), (True, 'exploratory_degraded')])
def test_unexpected_baseexception_delivers_detached_owner_policy_without_engine_reads(tmp_path, monkeypatch, exploratory, validity):
    runner = runner_at_frame(tmp_path, exploratory=exploratory)
    published = runner.published
    packet = copy.deepcopy(runner.latest_telemetry)
    world = copy.deepcopy(runner.arena.snapshot_world())
    def damaged_engine(*args, **kwargs):
        raise AssertionError('fault delivery queried the damaged engine')
    monkeypatch.setattr(runner.arena.observation_owner, 'snapshot_observation', damaged_engine)
    monkeypatch.setattr(runner, '_assemble_telemetry', damaged_engine)
    def exit_loop():
        runner._loop_phase = 'step'
        raise SystemExit('bounded CARD28 injected exit')
    monkeypatch.setattr(runner, '_schedule_loop', exit_loop)
    runner.running = True
    runner.sim_thread = SimpleNamespace(is_alive=lambda: False)
    with pytest.raises(SystemExit, match='CARD28'):
        runner._run_loop()
    health = runner.health()
    notice = health['observation_validity_update']
    assert notice == {
        'schema': 'neurofly-observation-validity-update/1',
        'identity': {**packet['identity'], 'brain_id': packet['brain_id']},
        'segment_id': packet['segment_id'], 'validity': validity,
        'reason': 'unexpected_loop_exit',
    }
    assert health['result_validity']['state'] == 'incomplete'
    assert health['liveness']['state'] == 'dead'
    assert runner.published is published and runner.latest_telemetry == packet
    assert runner.arena.snapshot_world() == world
    assert runner.total_steps == packet['step']
    notice['identity']['run_id'] = 'caller-corruption'
    assert runner.health()['observation_validity_update']['identity']['run_id'] == packet['identity']['run_id']
    runner.running = False


def test_sse_heartbeat_carries_fault_notice_when_published_frame_cannot_advance(tmp_path, monkeypatch):
    runner = runner_at_frame(tmp_path)
    published = runner.published
    runner._loop_phase = 'step'
    runner._record_loop_failure(SystemExit('bounded heartbeat diagnostic'))
    runner.running = True
    runner.sim_thread = SimpleNamespace(is_alive=lambda: False)
    ticks = itertools.count(0, 2)
    monkeypatch.setattr(nd.time, 'monotonic', lambda: next(ticks))
    class Sink(io.BytesIO):
        def flush(self):
            if b'event: heartbeat\n' in self.getvalue():
                runner.running = False
    handler = object.__new__(nd.NeuroflyHTTPHandler)
    handler.runner, handler.wfile = runner, Sink()
    handler.gateway = SimpleNamespace(policy=SimpleNamespace(stream_hz=30),
        record_delivery=lambda *args: None, pace=lambda previous: previous)
    handler._stream_snapshots()
    wire = handler.wfile.getvalue().decode()
    beat = json.loads(wire.split('event: heartbeat\ndata: ', 1)[1].split('\n', 1)[0])
    assert beat['seq'] == published.seq
    assert beat['identity'] == beat['observation_validity_update']['identity']
    assert beat['brain_id'] == runner.latest_telemetry['brain_id']
    assert beat['segment_id'] == runner.latest_telemetry['segment_id']
    assert beat['observation_validity_update']['validity'] == 'invalidated'
    assert runner.published is published
    assert 'observation' not in beat and 'fly' not in beat


def test_ordinary_halt_and_verified_rebuild_use_normal_frames_and_preserve_failed_run(tmp_path):
    runner = runner_at_frame(tmp_path)
    failed_run = runner.manifest.run_id
    runner._loop_phase = 'step'
    runner._on_loop_exception(RuntimeError('bounded ordinary halt'))
    runner._publish_snapshot()
    assert runner.latest_telemetry['observation']['validity'] == 'invalidated'
    assert runner.health()['observation_validity_update'] is None
    reply = transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 't-maze'})
    assert reply['status'] == 'ok' and reply['ack']['applied'] is True
    runner.step_once()
    runner._publish_snapshot()
    assert runner.latest_telemetry['observation']['validity'] == 'valid'
    assert failed_run in runner.result_validity()['other_runs_incomplete']
    assert runner.health()['observation_validity_update'] is None
