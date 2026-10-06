"""Deterministic public fault snapshots across construction, recovery and delivery."""
import threading

import neurofly_daemon as nd
from tests.test_assay_control_transactions import attached
from tests.test_no_silent_freeze import status


def assert_fault_consistent(payload):
    assert not payload['halted'] or isinstance(payload['error_detail'], dict), 'halt exposed without complete fault detail'
    if payload['halted']:
        assert payload['error_detail']['message'] == payload['error']
        assert payload['error_detail']['failure_class'] == 'persistence'
        assert payload['error_detail']['channel'] == 'trial_ledger'
        assert payload['error_detail']['recover'] == nd.ContinuousExperimentRunner.SAVE_RECOVERY


def test_status_never_exposes_halt_during_incomplete_detail_construction(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = nd._traceback_tail
    def blocked(exc):
        entered.set(); assert release.wait(5)
        return original(exc)
    monkeypatch.setattr(nd, '_traceback_tail', blocked)
    def fault():
        with runner.lock:
            runner._halt_on_error(OSError('trial disk full'), phase='persistence',
                                  channel='trial_ledger', failure_class='persistence')
    writer = threading.Thread(target=fault)
    writer.start(); assert entered.wait(2)
    try:
        assert_fault_consistent(status(runner))
    finally:
        release.set(); writer.join(5)
        records.close()


def test_status_health_and_detail_are_one_snapshot_across_recovery(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    with runner.lock:
        runner._halt_on_error(OSError('trial disk full'), phase='persistence',
                              channel='trial_ledger', failure_class='persistence')
    entered, release = threading.Event(), threading.Event()
    original = runner.health
    def blocked():
        snapshot = original()
        entered.set(); assert release.wait(5)
        return snapshot
    monkeypatch.setattr(runner, 'health', blocked)
    results = []
    reader = threading.Thread(target=lambda: results.append(status(runner)))
    reader.start(); assert entered.wait(2)
    try:
        with runner.lock:
            runner._clear_error('verified-test-recovery', defer_persistence=True)
    finally:
        release.set(); reader.join(5)
    assert_fault_consistent(results[0])
    records.close()


def test_health_fault_snapshot_is_detached_from_repeats_and_recovery(tmp_path):
    runner, records, _ = attached(tmp_path)
    with runner.lock:
        runner._halt_on_error(OSError('first cause'), phase='persistence',
                              channel='trial_ledger', failure_class='persistence')
    original = runner.health()
    assert_fault_consistent(original)
    original['error_detail']['channel'] = 'caller-corruption'
    assert runner.health()['error_detail']['channel'] == 'trial_ledger'
    saved = runner.health()
    internal = runner.error_detail
    with runner.lock:
        runner._halt_on_error(ValueError('second cause'))
    assert runner.error_detail is internal
    assert runner.health()['error_detail']['repeats'] == 1
    assert 'repeats' not in saved['error_detail']
    assert runner.health()['error'] == saved['error']
    with runner.lock: runner._clear_error('verified-test-recovery', defer_persistence=True)
    assert_fault_consistent(saved)
    assert runner.health()['error'] is None and runner.health()['error_detail'] is None
    records.close()


def test_telemetry_uses_captured_health_fault_after_recovery(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    with runner.lock:
        runner._halt_on_error(OSError('trial disk full'), phase='persistence',
                              channel='trial_ledger', failure_class='persistence')
    original = runner.health
    def recovery_between_health_and_packet():
        captured = original()
        runner._clear_error('verified-test-recovery', defer_persistence=True)
        return captured
    monkeypatch.setattr(runner, 'health', recovery_between_health_and_packet)
    packet = runner._assemble_telemetry(runner._last_step_result)
    assert_fault_consistent(packet)
    assert packet['halted'] and runner.last_error is None
    records.close()


def test_sse_heartbeat_uses_captured_health_fault_after_recovery(tmp_path, monkeypatch):
    import io
    import json
    from types import SimpleNamespace
    runner, records, _ = attached(tmp_path)
    with runner.lock:
        runner._halt_on_error(OSError('trial disk full'), phase='persistence',
                              channel='trial_ledger', failure_class='persistence')
    original = runner.health
    def captured_then_cleared():
        captured = original()
        runner._clear_error('verified-test-recovery', defer_persistence=True)
        return captured
    monkeypatch.setattr(runner, 'health', captured_then_cleared)
    runner.running, runner.published = True, None
    reads = iter((0.0, 2.0))
    monkeypatch.setattr(nd.time, 'monotonic', lambda: next(reads, 2.0))
    class Sink(io.BytesIO):
        def flush(self): runner.running = False
    handler = object.__new__(nd.NeuroflyHTTPHandler)
    handler.runner, handler.wfile = runner, Sink()
    handler.gateway = SimpleNamespace(policy=SimpleNamespace(stream_hz=30),
                                      record_delivery=lambda *a: None, pace=lambda *a: 0)
    handler._stream_snapshots()
    text = handler.wfile.getvalue().decode()
    heartbeat = json.loads(text.split('event: heartbeat\ndata: ', 1)[1].split('\n', 1)[0])
    assert_fault_consistent(heartbeat)
    assert heartbeat['halted'] and runner.last_error is None
    records.close()


def test_recorded_fault_cause_survives_separate_dead_loop_diagnostic(tmp_path):
    from types import SimpleNamespace
    runner, records, _ = attached(tmp_path)
    with runner.lock:
        runner._halt_on_error(OSError('trial disk full'), phase='persistence',
                              channel='trial_ledger', failure_class='persistence')
    runner.running = True
    runner.sim_thread = SimpleNamespace(is_alive=lambda: False)
    runner.loop_failure = {'type': 'RuntimeError', 'message': 'separate loop death'}
    payload = runner.health()
    assert payload["error"].startswith("simulation thread stopped: RuntimeError")
    assert payload["error_detail"]["message"] == runner.error_detail["message"]
    assert payload["error_detail"]["channel"] == "trial_ledger"
    assert payload["error_detail"]["recover"] == runner.SAVE_RECOVERY
    assert payload['status'] == 'error' and payload['liveness']['state'] == 'dead'
    assert payload['loop_failure']['message'] == 'separate loop death'
    records.close()
