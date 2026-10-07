"""A stop after a clean API shutdown returns that result; failures stay honest (v0.4.1)."""
import hashlib
import json

from tests.test_assay_control_transactions import attached
from tests.test_lifecycle_control_transactions import complete


def rows(runner):
    path = runner.output_dir / 'run_validity.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def hashes(runner):
    """Every file the runner owns: checkpoints, CURRENT/registry state and ledgers."""
    roots = {runner.output_dir, getattr(runner.active_brain, 'directory', runner.output_dir)}
    found = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for root in roots for p in root.rglob('*') if p.is_file()}
    assert any('final_shutdown' in name or 'checkpoint' in name for name in found)
    return found


def test_signal_stop_after_clean_api_shutdown_returns_earlier_success(tmp_path):
    runner, records, drain = attached(tmp_path)
    runner.assay_control_timeout_s = 1.0
    runner.step_once()
    result, _ = complete(runner, drain, {'action': 'shutdown'})
    assert result['ack']['applied']
    before_rows, before_hashes = rows(runner), hashes(runner)
    assert sum(r.get('event') == 'session_end' for r in before_rows) == 1
    acks = len(runner.command_acks)
    assert runner.stop() is True and runner.stop() is True
    assert runner._pending_assay_control is None and len(runner.command_acks) == acks
    assert rows(runner) == before_rows and hashes(runner) == before_hashes
    assert not any(r.get('event') in ('required_save_failed', 'session_end_failed') for r in rows(runner))
    records.close()


def test_stop_after_failed_api_shutdown_still_reports_failure(tmp_path, monkeypatch):
    runner, records, drain = attached(tmp_path)
    runner.assay_control_timeout_s = 1.0
    monkeypatch.setattr(runner, 'save_checkpoint',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('checkpoint unavailable')))
    result, _ = complete(runner, drain, {'action': 'shutdown'})
    assert not result['ack']['applied']
    assert runner.stop() is False and runner.stop() is False
    assert not any(r.get('event') == 'session_end' for r in rows(runner))
    records.close()


def test_single_direct_stop_is_unchanged(tmp_path):
    runner, records, _ = attached(tmp_path)
    runner.step_once()
    assert runner.stop() is True
    assert sum(r.get('event') == 'session_end' for r in rows(runner)) == 1
    assert runner.command_acks[-1]['ack']['applied']
    records.close()
