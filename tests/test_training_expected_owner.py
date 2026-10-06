"""Displayed-owner preconditions are checked when queued writes actually apply."""
import copy
import pytest
from neurofly_daemon import ContinuousExperimentRunner


def owner(runner):
    return dict(runner.identity(), brain_id=runner.active_brain.brain_id)


@pytest.mark.parametrize('action', ['teach_brain', 'probe_brain', 'set_learning', 'save_checkpoint'])
def test_stale_two_tab_training_owner_has_no_effect(tmp_path, action, monkeypatch):
    runner = ContinuousExperimentRunner(output_dir=tmp_path)
    expected = owner(runner)
    runner.activation += 1  # Another tab completed a switch, even back to the same retained brain.
    before = copy.deepcopy(runner.active_brain.summary(details=True))
    def forbidden(*args, **kwargs):
        raise AssertionError('stale training action reached mutation/persistence')
    monkeypatch.setattr(runner.active_brain, 'probe', forbidden)
    monkeypatch.setattr(runner, 'checkpoint_now', forbidden)
    result = runner.dispatch_command({'action': action, 'params': {'expected_owner': expected, 'enabled': False, 'pairs': 1}})
    assert result['status'] == 'error' and result['ack']['applied'] is False
    assert 'stale' in result['message']
    monkeypatch.undo()  # summary itself computes a read-only probe.
    assert runner.active_brain.summary(details=True) == before
    assert list(tmp_path.rglob('*.npz')) == []


def test_queued_owner_rechecked_after_intervening_switch(tmp_path, monkeypatch):
    runner = ContinuousExperimentRunner(output_dir=tmp_path)
    monkeypatch.setattr(runner, '_loop_active', lambda: True)
    runner.command_reply_wait_s = 0
    queued = runner.dispatch_command({'action': 'probe_brain', 'params': {'expected_owner': owner(runner)}})
    assert queued['status'] == 'queued' and queued['applied'] is False
    runner.activation += 1
    monkeypatch.setattr(runner.active_brain, 'probe', lambda: pytest.fail('stale queued probe applied'))
    runner._drain_commands()
    final = runner.command_acks[-1]
    assert final['command_id'] == queued['command_id']
    assert final['status'] == 'error' and final['ack']['applied'] is False


@pytest.mark.parametrize('invalid', [{}, {'activation': True}, 'not an identity', None])
def test_invalid_owner_fails_closed(tmp_path, invalid):
    runner = ContinuousExperimentRunner(output_dir=tmp_path)
    reply = runner.dispatch_command({'action': 'probe_brain', 'params': {'expected_owner': invalid}})
    assert reply['status'] == 'error' and reply['ack']['applied'] is False


def test_matching_owner_and_legacy_probe_remain_supported(tmp_path):
    runner = ContinuousExperimentRunner(output_dir=tmp_path)
    for params in ({}, {'expected_owner': owner(runner)}):
        reply = runner.dispatch_command({'action': 'probe_brain', 'params': params})
        assert reply['status'] == 'ok' and reply['ack']['applied'] is True
