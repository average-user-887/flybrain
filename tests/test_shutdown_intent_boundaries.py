"""Deterministic admission barrier for stop while a prior control completes."""
import threading

from tests.test_assay_control_transactions import attached


def test_shutdown_intent_prevents_next_step_between_prior_ack_and_shutdown_admission(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    runner.running = True
    runner.assay_control_timeout_s = 0.15
    runner.dispatch_command({'action': 'set_learning', 'enabled': False})
    prior = runner._pending_assay_control
    want_step, decided, stepped, release_step = (threading.Event() for _ in range(4))
    acquired = []
    original_acquire, original_step = runner.lock.acquire, runner.arena.step
    def acquire(*args, **kwargs):
        if threading.current_thread().name == 'Stopper':
            acquired.append(True)
            if len(acquired) == 2:
                want_step.set()
                assert decided.wait(2)
        return original_acquire(*args, **kwargs)
    def stalled(dt):
        stepped.set(); decided.set()
        assert release_step.wait(5)
        return original_step(dt)
    monkeypatch.setattr(runner.lock, 'acquire', acquire)
    monkeypatch.setattr(runner.arena, 'step', stalled)
    def scheduled_successor():
        assert want_step.wait(2)
        with runner.lock:
            if runner._can_step(): runner._advance_one()
            decided.set()
    successor = threading.Thread(target=scheduled_successor, name='ScheduledSuccessor')
    outcomes = []
    stopper = threading.Thread(target=lambda: outcomes.append(runner.stop()), name='Stopper')
    successor.start(); stopper.start()
    try:
        assert decided.wait(2)
        assert prior['entry']['done'].is_set() and runner._stopped
        assert not stepped.is_set(), 'shutdown intent did not gate the next scheduled arena step'
        stopper.join(1)
        assert not stopper.is_alive(), 'stop exceeded its deadline reacquiring the runner lock'
        assert outcomes == [True] and runner.total_steps == 0
    finally:
        release_step.set()
        successor.join(5); stopper.join(5)
        if prior.get('shutdown_drain'): prior['shutdown_drain'].join(5)
        if prior.get('writer'): prior['writer'].join(5)
        records.close()


def test_stop_reacquisition_deadline_with_competing_lock_holder(tmp_path, monkeypatch):
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.1
    runner.dispatch_command({'action': 'set_learning', 'enabled': False})
    prior = runner._pending_assay_control
    requested, held, release = (threading.Event() for _ in range(3))
    acquire, count = runner.lock.acquire, []
    def barrier(*a, **k):
        if threading.current_thread().name == 'Stopper':
            count.append(True)
            if len(count) == 2:
                requested.set(); assert held.wait(2)
        return acquire(*a, **k)
    monkeypatch.setattr(runner.lock, 'acquire', barrier)
    def contender():
        assert requested.wait(2)
        with runner.lock:
            held.set(); assert release.wait(5)
    holder = threading.Thread(target=contender)
    outcomes = []
    stopper = threading.Thread(target=lambda: outcomes.append(runner.stop()), name='Stopper')
    holder.start(); stopper.start()
    try:
        assert held.wait(2)
        stopper.join(0.5)
        assert outcomes == [False] and not stopper.is_alive()
        assert prior['entry']['done'].is_set()
        assert not any(x['ack']['action'] == 'shutdown' for x in runner.command_acks)
        assert runner.total_steps == 0
    finally:
        release.set(); holder.join(5); stopper.join(5); records.close()


def test_late_shutdown_marker_is_revoked_without_fsync_failure(tmp_path, monkeypatch):
    import json
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.1
    entered, release = threading.Event(), threading.Event()
    original = runner._save_shutdown
    def marker_window(tx, recorder):
        original(tx, recorder)
        entered.set(); assert release.wait(5)
    monkeypatch.setattr(runner, '_save_shutdown', marker_window)
    outcome = []
    stopper = threading.Thread(target=lambda: outcome.append(runner.stop()))
    stopper.start()
    try:
        assert entered.wait(2)
        tx = runner._pending_assay_control
        stopper.join(1); assert outcome == [False]
        assert not tx['entry']['result']['ack']['applied']
        tx['failure_writer'].join(2)
        rows = [json.loads(x) for x in (runner.output_dir / 'run_validity.jsonl').read_text().splitlines()]
        assert sum(x.get('event') == 'session_end' for x in rows) == 1
        assert sum(x.get('event') == 'session_end_failed' for x in rows) == 1
        assert tx['marker_revocation_durable'] is True
        assert runner.run_id in runner._interrupted_sessions(rows)
        assert any(x.get('reason') == 'assay_control_failed' for x in rows)
        incidents = list(runner.incidents)
        with runner.lock: runner._clear_error('verified-test-recovery', defer_persistence=True)
        assert runner.incidents == incidents and incidents
        assert runner.result_validity()['state'] == 'incomplete'
    finally:
        release.set(); stopper.join(5)
        if 'tx' in locals(): tx['writer'].join(5)
        records.close()


def test_stop_cancels_pending_owner_without_waiting_for_stuck_lock(tmp_path, monkeypatch):
    import json
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.15
    saving, release_save, held, release_lock = (threading.Event() for _ in range(4))
    original = runner.save_checkpoint
    def checkpoint(*a, **k):
        saving.set(); assert release_save.wait(5); return original(*a, **k)
    monkeypatch.setattr(runner, 'save_checkpoint', checkpoint)
    outcome = []
    stopper = threading.Thread(target=lambda: outcome.append(runner.stop()))
    stopper.start(); assert saving.wait(2)
    tx = runner._pending_assay_control
    def stuck_reader():
        with runner.lock:
            held.set(); assert release_lock.wait(5)
    holder = threading.Thread(target=stuck_reader)
    holder.start()
    try:
        assert held.wait(2)
        stopper.join(0.5)
        assert outcome == [False] and tx['cancel_requested']
        assert not tx['entry']['done'].is_set()
        release_save.set(); release_lock.set()
        holder.join(2); tx['cancellation_writer'].join(2)
        assert tx['entry']['done'].wait(2)
        tx['writer'].join(2)
        assert not tx['entry']['result']['ack']['applied']
        assert len([x for x in runner.command_acks if x['ack']['action'] == 'shutdown']) == 1
        rows = [json.loads(x) for x in (runner.output_dir / 'run_validity.jsonl').read_text().splitlines()]
        assert not any(x.get('event') == 'session_end' for x in rows)
    finally:
        release_save.set(); release_lock.set(); holder.join(5); stopper.join(5)
        if tx.get('failure_writer'): tx['failure_writer'].join(5)
        records.close()


def test_initial_stop_lock_timeout_revokes_existing_transaction(tmp_path):
    runner, records, _ = attached(tmp_path)
    runner.assay_control_timeout_s = 0.1
    runner.dispatch_command({'action': 'set_learning', 'enabled': False})
    tx = runner._pending_assay_control
    held, release = threading.Event(), threading.Event()
    def holder():
        with runner.lock:
            held.set(); assert release.wait(5)
    owner = threading.Thread(target=holder); owner.start()
    try:
        assert held.wait(2)
        assert runner.stop() is False
        assert tx['cancel_requested'] and not tx['entry']['done'].is_set()
        release.set(); owner.join(2); tx['cancellation_writer'].join(2)
        assert tx['entry']['done'].is_set() and not tx['entry']['result']['ack']['applied']
        assert runner.active_brain.learning_enabled is True
        assert runner.observation_publication.pending_status()['pending_count'] == 1
    finally:
        release.set(); owner.join(5)
        if tx.get('failure_writer'): tx['failure_writer'].join(5)
        records.close()
