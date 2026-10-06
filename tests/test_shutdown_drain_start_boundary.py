"""Drain startup failures cannot block stop or retain unstarted join targets."""
import threading
import time
import pytest
from tests.test_assay_control_transactions import attached


@pytest.mark.parametrize('busy,publisher_fails', [(False, False), (True, False), (True, True)])
def test_drain_start_failure_is_bounded_revoked_and_exactly_once(tmp_path, monkeypatch, busy, publisher_fails):
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.1
    requested, held, release = (threading.Event() for _ in range(3))
    def hold():
        assert requested.wait(2)
        with runner.lock:
            held.set(); assert release.wait(5)
    holder = threading.Thread(target=hold)
    if busy: holder.start()
    original_start = threading.Thread.start
    captured = []
    def startup(thread):
        if thread.name == 'NeuroFly-ShutdownDrain':
            captured.append(runner._pending_assay_control)
            if busy:
                requested.set(); assert held.wait(2)
            raise RuntimeError('owned drain startup failed')
        if publisher_fails and thread.name == 'NeuroFly-ShutdownCancellation':
            raise RuntimeError('failure publisher startup unavailable')
        return original_start(thread)
    monkeypatch.setattr(threading.Thread, 'start', startup)
    outcomes, errors = [], []
    def stop():
        try: outcomes.append(runner.stop())
        except Exception as exc: errors.append(exc)
    stopper = threading.Thread(target=stop)
    begin = time.monotonic(); stopper.start()
    try:
        stopper.join(0.5)
        assert not stopper.is_alive() and outcomes == [False] and not errors
        assert time.monotonic() - begin < 0.5
        tx = captured[0]
        assert tx['cancel_requested'] and str(tx['cancellation_error']) == 'owned drain startup failed'
        assert tx.get('shutdown_drain') is None
        assert runner.observation_publication.pending_status()['pending_count'] == 1
        assert runner.total_steps == 0 and runner.active_brain.learning_enabled is True
        if busy:
            assert not tx['entry']['done'].is_set()
            if publisher_fails:
                assert tx.get('cancellation_writer') is None
                assert 'publisher startup' in tx['cancellation_start_error']
            else:
                assert tx['cancellation_writer'].ident is not None
        release.set()
        if busy: holder.join(2)
        if publisher_fails:
            with runner.lock: runner.step_once()
        if tx.get('cancellation_writer'): tx['cancellation_writer'].join(2)
        assert tx['entry']['done'].wait(2)
        assert not tx['entry']['result']['ack']['applied']
        assert 'owned drain startup failed' in tx['entry']['result']['message']
        assert len([x for x in runner.command_acks if x['ack']['action'] == 'shutdown']) == 1
        assert runner.stop() is False
        assert not (runner.output_dir / 'run_validity.jsonl').read_text().count('"event": "session_end"')
    finally:
        release.set(); stopper.join(5)
        if busy: holder.join(5)
        if captured and captured[0].get('failure_writer'): captured[0]['failure_writer'].join(5)
        records.close()


def test_stop_never_joins_unstarted_failure_writer_when_all_worker_starts_fail(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.1
    captured = []
    def no_workers(thread):
        captured.append((thread.name, runner._pending_assay_control))
        raise RuntimeError('all worker startups unavailable')
    monkeypatch.setattr(threading.Thread, 'start', no_workers)
    try:
        assert runner.stop() is False
        tx = next(tx for name, tx in captured if name == 'NeuroFly-ShutdownDrain')
        assert tx['entry']['done'].is_set() and not tx['entry']['result']['ack']['applied']
        assert tx.get('shutdown_drain') is None
        assert tx['failure_writer'].ident is None
        assert runner.stop() is False
        assert runner.observation_publication.pending_status()['pending_count'] == 1
    finally:
        records.close()
