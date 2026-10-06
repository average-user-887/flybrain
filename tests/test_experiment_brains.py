"""Prove memory isolation, genuine plasticity, controls and durable restoration."""
import json
import numpy as np
import pytest
from experiment_brains import ExperimentBrain, PARADIGMS
from neurofly_daemon import ContinuousExperimentRunner
from learning_recorder import LearningRecorder, RecorderThread
from circuit import MushroomBodyCircuit
from tests.transition_control_helpers import transition_command


_recorders = []


@pytest.fixture(autouse=True)
def close_durable_recorders():
    yield
    while _recorders:
        _recorders.pop().close()


def runner(path):
    active = ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=path,
                                      checkpoint_interval=3600, trial_length_s=2)
    records = LearningRecorder(path / f'test-records-{active.run_id}', session={'daemon_run_id': active.run_id})
    active.attach_learning_records(RecorderThread(active, records, summary_interval=999))
    _recorders.append(records)
    return active


def teach(r, pairs=6, reverse=False):
    assert transition_command(r, {'action': 'teach_brain', 'pairs': pairs, 'reverse': reverse})['status'] == 'ok'
    while r._teaching_suspended or r._pending_assay_control is not None:
        with r.lock:
            r.step_once()
        tx = r._pending_assay_control
        if tx is not None:
            assert tx['entry']['done'].wait(10)
            assert tx['entry']['result']['ack']['applied']
            tx['writer'].join(10)
            assert not tx['writer'].is_alive()


def test_each_experiment_has_a_separate_brain(tmp_path):
    r = runner(tmp_path)
    identities = set()
    for p in PARADIGMS:
        assert transition_command(r, {'action': 'switch_paradigm', 'paradigm': p})['status'] == 'ok'
        identities.add(r.active_brain.brain_id)
        r.step_once()
    assert len(identities) == 14
    brains = list(r.brains.instances.values())
    for a, b in zip(brains, brains[1:]):
        assert not np.shares_memory(a.circuit.w, b.circuit.w)
        assert a.arena.fly.cx is not b.arena.fly.cx


def test_train_switch_return_and_restart_preserve_memory(tmp_path):
    r = runner(tmp_path)
    first = r.active_brain
    teach(r)
    assert first.probe()['discrimination'] > 0.05
    weights = first.circuit.w.copy()
    probe = first.probe()
    transition_command(r, {'action': 'switch_paradigm', 'paradigm': 'y-maze'})
    np.testing.assert_array_equal(r.active_brain.circuit.w, 0)
    teach(r, reverse=True)
    assert r.active_brain.probe()['discrimination'] < -0.05
    transition_command(r, {'action': 'switch_paradigm', 'paradigm': 't-maze'})
    assert r.active_brain is first
    np.testing.assert_array_equal(first.circuit.w, weights)
    r.save_checkpoint('test')
    restarted = runner(tmp_path)
    assert restarted.active_brain.brain_id == first.brain_id
    assert restarted.active_brain.restored
    assert restarted.active_brain.probe() == probe
    np.testing.assert_array_equal(restarted.active_brain.circuit.w, weights)
    assert len([e for e in first.history if e['kind'] == 'teaching_pair']) == 6


def test_frozen_control_and_probe_do_not_change_memory(tmp_path):
    r = runner(tmp_path)
    transition_command(r, {'action': 'set_learning', 'enabled': False})
    before = r.active_brain.circuit.w.copy()
    teach(r)
    np.testing.assert_array_equal(r.active_brain.circuit.w, before)
    assert r.active_brain.probe()['discrimination'] == 0
    for _ in range(100):
        r.step_once()
    np.testing.assert_array_equal(r.active_brain.circuit.w, before)
    arrays = {k: v.copy() for k, v in vars(r.active_brain.circuit).items() if isinstance(v, np.ndarray)}
    r.dispatch_command({'action': 'probe_brain'})
    for k, v in arrays.items():
        np.testing.assert_array_equal(getattr(r.active_brain.circuit, k), v)


def test_bad_commands_leave_active_brain_untouched(tmp_path):
    r = runner(tmp_path)
    original = r.active_brain
    for cmd in ({'action': 'switch_paradigm', 'paradigm': '../../outside'},
                {'action': 'teach_brain', 'pairs': -1}, {'action': 'set_learning', 'enabled': 'no'}):
        assert r.dispatch_command(cmd)['status'] == 'error'
    assert r.active_brain is original
    path = r.save_checkpoint('../../outside')
    assert path.parent == tmp_path / 'checkpoints'


def test_corrupt_checkpoint_is_rejected_without_overwrite(tmp_path):
    b = ExperimentBrain('t-maze', tmp_path)
    b.save()
    data = json.loads(b.path.read_text())
    data['circuit']['w'] = [[0, 0]]
    b.path.write_text(json.dumps(data))
    before = b.path.read_bytes()
    with pytest.raises(ValueError, match='invalid circuit array'):
        ExperimentBrain('t-maze', tmp_path)
    assert b.path.read_bytes() == before


def test_trial_history_and_learning_curve_are_experiment_specific(tmp_path):
    r = runner(tmp_path)
    recorder = LearningRecorder(tmp_path / 'records', session={'daemon_run_id': r.run_id})
    drain = RecorderThread(r, recorder, summary_interval=999)
    with r.lock:
        r.attach_learning_records(drain)
    for _ in range(101):
        r.step_once()
        drain.poll_once()
    assert r.active_brain.trials == 1
    curve = r.learning_curve.copy()
    transition_command(r, {'action': 'switch_paradigm', 'paradigm': 'buridan'})
    assert r.learning_curve == []
    for _ in range(101):
        r.step_once()
        drain.poll_once()
    assert [t['trial'] for t in r.trial_history] == [1, 2]  # recorder global sequence
    assert [t['brain_trial'] for t in r.trial_history] == [1, 1]
    assert len({t['brain_id'] for t in r.trial_history}) == 2
    transition_command(r, {'action': 'switch_paradigm', 'paradigm': 't-maze'})
    assert r.learning_curve == curve
    recorder.close()


def test_plasticity_advances_kc_trace_once_per_time_bin():
    c = MushroomBodyCircuit()
    _, kc = c.encode_odor(1, 0)
    c.advance(1, 0, dt_seconds=0.02)
    np.testing.assert_allclose(c.y_kc, kc * (1 - np.exp(-0.02)), atol=1e-12)
    assert c.step_count == 2


def test_teaching_checkpoint_resumes_the_same_protocol(tmp_path):
    b = ExperimentBrain('t-maze', tmp_path)
    b.start_teaching(2)
    for _ in range(47):
        b.teaching_step(0.02)
    b.save()
    resumed = ExperimentBrain('t-maze', tmp_path)
    assert resumed.teaching['tick'] == 47
    while b.teaching:
        b.teaching_step(0.02)
    while resumed.teaching:
        resumed.teaching_step(0.02)
    np.testing.assert_array_equal(b.circuit.w, resumed.circuit.w)


def test_tmaze_keeps_physical_cues_and_antennae_separate(tmp_path, monkeypatch):
    r = runner(tmp_path)
    a = r.arena
    a.fly.pos.x, a.fly.pos.y, a.fly.heading = 25., 50., 1.57
    captured = []
    original = a.fly.circuit.advance
    def capture(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(a.fly.circuit, 'advance', capture)
    a.step(0.02)
    first = captured[-1]
    assert first['odor_a'] > first['odor_b']
    assert first['dt_seconds'] == 0.02
    # Changing which arm is rewarded must not relabel the odor identity.
    a.paradigm.cs_plus_arm = 'arm_b'
    a.fly.pos.x, a.fly.pos.y, a.fly.heading = 25., 50., 1.57
    a.step(0.02)
    assert captured[-1]['odor_a'] > captured[-1]['odor_b']
