"""Trial clock continuity across daemon restarts (CPU only; no services).

The per-assay trial elapsed time and counter are saved with the state they belong
to: the selected verified graph checkpoint, or the modular brain JSON.  A restart
restores them from exactly that state.  State saved without them stays explicitly
unknown until a trial starts in front of the process; nothing is inferred from
brain, world or daemon counters.  The packet step/sim time are the daemon session
clock and are labelled as such.

The observation producer owns when a trial ends (metric-contract/1.2).  A restart
begins a new, explicit observation segment with the full window, so a restored
trial runs one more window; its recorded duration includes the restored elapsed
time.  These tests pin that behaviour; they do not claim measurement continuation.
"""
import json

import numpy as np
import pytest

import experiment_brains
from tests.transition_control_helpers import transition_command
from experiment_registry import IncompatibleCheckpoint, load_checkpoint_file
from learning_recorder import LearningRecorder, RecorderThread
from neurofly_daemon import ContinuousExperimentRunner
from provenance import (LEGACY_TRIAL_CLOCK_REASON, TRIAL_CLOCK_SCHEMA, trial_clock,
                        unknown_trial_clock, validate_trial_clock)
from tests.test_provenance_registry import make_registry

DT = 0.02


@pytest.fixture
def daemon(tmp_path):
    """Start a daemon process on one output directory; each call is a restart."""
    recorders = []

    def start(paradigm='multisensory-sandbox', *, graph=False, trial_length_s=600.0):
        runner = ContinuousExperimentRunner(
            initial_paradigm=paradigm, sim_speed=100.0, checkpoint_interval=3600.0,
            output_dir=tmp_path / 'daemon', trial_length_s=trial_length_s,
            backend='connectome-fixed' if graph else 'modular', test_synthetic_graph=graph,
            brain_backend='cpu', keep_checkpoints=0)
        recorder = LearningRecorder(tmp_path / f'records-{len(recorders)}',
                                    session={'daemon_run_id': runner.run_id})
        drain = RecorderThread(runner, recorder, summary_interval=999)
        with runner.lock:
            runner.attach_learning_records(drain)
        recorders.append(recorder)
        runner._test_drain = drain
        return runner

    yield start
    for recorder in recorders:
        recorder.close()


def advance(runner, steps):
    """Actual ticks; each immutable terminal is drained so trials can end."""
    for _ in range(steps):
        runner.step_once()
        if runner._observation_terminal is not None \
                and runner._observation_terminal['durable'] is None:
            runner._test_drain.poll_once()


def clock_of(runner):
    """The active clock without its segment_id (checked separately below)."""
    return sans(validate_trial_clock(runner.active_brain.trial_clock))


def sans(clock):
    return {k: v for k, v in clock.items() if k != 'segment_id'}


def tc(*args, **kwargs):
    return sans(trial_clock(*args, **kwargs))


def saved_brain_json(runner):
    return json.loads(runner.active_brain.path.read_text())


def newest_graph_meta(runner):
    registry = runner.registry
    return registry.read_checkpoint(registry.active.instance_id)[0]


def make_legacy_brain_json(runner):
    """Rewrite the brain JSON exactly as code before trial clocks wrote it."""
    data = saved_brain_json(runner)
    data.pop('trial_clock')
    runner.active_brain.path.write_text(json.dumps(data))


# -- known clocks ------------------------------------------------------------------
def test_fresh_assay_starts_a_known_first_trial(daemon):
    runner = daemon()
    assert clock_of(runner) == tc()
    assert runner.trial_sim_time == 0.0 and runner.current_trial == 1


def test_known_modular_clock_continues_across_restart(daemon):
    first = daemon(trial_length_s=0.2)
    advance(first, 2 * 10 + 4)                     # two whole trials, then 4 steps of the third
    assert len(first.trial_history) == 2 and first.current_trial == 3
    assert first.trial_sim_time == pytest.approx(4 * DT)
    elapsed, steps, trials = first.trial_sim_time, first.active_brain.steps, first.active_brain.trials
    first.save_checkpoint('clock')
    assert sans(saved_brain_json(first)['trial_clock']) == tc(elapsed, 3)

    restarted = daemon(trial_length_s=0.2)
    assert restarted.total_steps == 0                          # a new daemon session
    assert restarted.trial_sim_time == elapsed and restarted.current_trial == 3
    assert restarted.active_brain.steps == steps and restarted.active_brain.trials == trials
    ack = restarted.dispatch_command({'action': 'set_paused', 'paused': True})['ack']
    assert ack['applied_step'] == 0 and ack['applied_sim_time_scope'] == 'daemon_session'
    assert ack['clocks']['session'] == {'scope': 'daemon_session', 'daemon_run_id': restarted.run_id,
                                        'step': 0, 'elapsed_s': 0.0}
    assert sans(ack['clocks']['trial']) == tc(elapsed, 3)
    assert 'graph' not in ack['clocks']
    restarted.dispatch_command({'action': 'set_paused', 'paused': False})

    # Trial 3 continues: it ends after the restarted observation window and its
    # recorded duration includes the 4 steps observed before the restart.
    advance(restarted, 9)
    assert restarted.trial_history == []
    assert restarted.trial_sim_time == pytest.approx(elapsed + 9 * DT)
    advance(restarted, 1)
    assert [t['assay_trial'] for t in restarted.trial_history] == [3]
    assert [t['trial'] for t in restarted.trial_history] == [1]     # session record key
    assert restarted.trial_history[0]['sim_seconds'] == pytest.approx(elapsed + 0.2)
    assert restarted.trial_history[0]['trial_known'] is True
    assert restarted.current_trial == 4 and restarted.trial_sim_time == 0.0


def test_known_graph_clock_comes_only_from_the_selected_checkpoint(daemon):
    first = daemon('t-maze', graph=True)
    advance(first, 12)
    instance = first.registry.active
    assert instance.trial_clock is first.active_brain.trial_clock      # one record
    elapsed, step_index = first.trial_sim_time, instance.step_index
    sim_ms = float(instance.brain.sim_ms)
    ident = first.identity()
    first.save_checkpoint('clock')
    assert sans(newest_graph_meta(first)['trial_clock']) == tc(elapsed, 1)
    assert 'trial_clock' not in newest_graph_meta(first)['world_state']
    # A newer, contradictory helper JSON is never the authority for a graph run.
    helper = saved_brain_json(first)
    helper['trial_clock'] = trial_clock(99.0, 42)
    first.active_brain.path.write_text(json.dumps(helper))

    restarted = daemon('t-maze', graph=True)
    assert restarted.identity()['run_id'] == ident['run_id']
    assert restarted.identity()['instance_id'] == ident['instance_id']
    assert restarted.identity()['daemon_run_id'] != ident['daemon_run_id']
    assert restarted.trial_sim_time == elapsed and restarted.current_trial == 1
    assert restarted.registry.active.step_index == step_index
    clocks = restarted.clock_status()
    assert clocks['session']['step'] == 0 and clocks['session']['elapsed_s'] == 0.0
    assert clocks['graph'] == {'scope': 'retained_graph_instance', 'instance_id': ident['instance_id'],
                               'step': step_index, 'elapsed_s': round(sim_ms / 1000.0, 5)}
    assert sans(clocks['trial']) == tc(elapsed, 1)
    packet = restarted.latest_telemetry
    assert packet['sim_time_scope'] == 'daemon_session' and packet['clocks'] == clocks
    assert packet['timing']['sim_time_scope'] == 'daemon_session'
    advance(restarted, 1)
    assert restarted.trial_sim_time == pytest.approx(elapsed + DT)
    assert restarted.registry.active.trial_clock['elapsed_s'] == restarted.trial_sim_time


def test_switching_assays_restores_each_counter_and_keeps_records_monotonic(daemon, tmp_path):
    runner = daemon(trial_length_s=0.2)
    advance(runner, 2 * 10 + 3)                     # sandbox: trials 1-2 done, 3 running
    sandbox = clock_of(runner)
    transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 'buridan'})
    assert runner.current_trial == 1 and runner.trial_sim_time == 0.0
    advance(runner, 10)                             # buridan's own trial 1
    transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 'multisensory-sandbox'})
    assert clock_of(runner) == sandbox and runner.trial_sim_time == sandbox['elapsed_s']
    advance(runner, 10)
    history = runner.trial_history
    assert [(t['paradigm'], t['assay_trial']) for t in history] == [
        ('multisensory-sandbox', 1), ('multisensory-sandbox', 2), ('buridan', 1),
        ('multisensory-sandbox', 3)]
    # The learning recorder appends only increasing session trial numbers.
    assert [t['trial'] for t in history] == [1, 2, 3, 4]
    recorder = LearningRecorder(tmp_path / 'switch-records', session={'daemon_run_id': runner.run_id})
    try:
        assert recorder.record_trials(history) == 4
    finally:
        recorder.close()


# -- the observation window restart is an explicit, recorded discontinuity -------------
def observation_records(tmp_path):
    rows = []
    for path in sorted(tmp_path.glob('records-*/trials.jsonl')):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return [row for row in rows if row['type'] == 'observation']


def ledger(runner):
    path = runner.active_brain.directory / f'{runner.active_paradigm_id}.events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_restart_opens_a_new_segment_with_lineage_and_records_the_old_one_interrupted(daemon, tmp_path):
    first = daemon(trial_length_s=0.2)
    advance(first, 4)
    parent = first.segment_id
    first.save_checkpoint('mid-trial')
    assert saved_brain_json(first)['trial_clock']['segment_id'] == parent
    assert first.current_segment_lineage() is None                 # an ordinary segment

    restarted = daemon(trial_length_s=0.2)
    child = restarted.segment_id
    assert child != parent
    lineage = restarted.current_segment_lineage()
    assert lineage == {
        'segment_id': child, 'parent_segment_id': parent, 'reason': 'daemon_restart',
        'parent_observation': 'interrupted_incomplete_no_terminal',
        'daemon_run_id': restarted.run_id, 'trial': 1, 'trial_known': True,
        'restored_trial_elapsed_s': pytest.approx(4 * DT), 'trial_elapsed_known': True,
        'observation_window': 'restarted', 'metric_accumulators': 'not_restored'}
    # Durable, append-only incomplete marker for the interrupted parent.
    interrupted = [e for e in ledger(restarted) if e['kind'] == 'observation_interrupted']
    assert len(interrupted) == 1
    assert {k: interrupted[0][k] for k in lineage} == lineage
    # The new window is a different presentation identity with fresh accumulators.
    packet = restarted.latest_telemetry
    assert packet['observation']['segment_id'] == child
    assert packet['observation']['presentation_id'] == f'{child}:0'
    assert packet['observation']['config_id'].startswith(f'segment:{child}')
    window = packet['clocks']['observation']
    assert window['segment_id'] == child and window['lineage'] == lineage
    assert window['segment_elapsed_s'] == 0.0 and window['effective_window_s'] == 0.2
    assert packet['clocks']['trial']['elapsed_s'] == pytest.approx(4 * DT)   # trial clock apart
    assert packet['observation_lifecycle']['segment_lineage'] == lineage
    owner = restarted.arena.observation_owner
    assert owner.observation_status()['segment_elapsed_s'] == 0.0

    advance(restarted, 10)                          # the restarted window, not the remainder
    record = restarted.trial_history[0]
    assert record['observation_segment_id'] == child and record['segment_lineage'] == lineage
    assert record['observation_measured_s'] == pytest.approx(0.2)            # measurement window
    assert record['sim_seconds'] == pytest.approx(4 * DT + 0.2)              # trial clock
    # Only the new segment has a terminal observation; the parent is never combined.
    keys = [row['observation_key'] for row in observation_records(tmp_path)]
    assert [k['segment_id'] for k in keys] == [child]
    assert [k['presentation_id'] for k in keys] == [f'{child}:0']
    # The next segment is ordinary again.
    assert restarted.segment_id != child and restarted.current_segment_lineage() is None
    assert restarted.clock_status()['observation']['lineage'] is None


def test_lineage_chains_across_repeated_restarts(daemon):
    first = daemon(trial_length_s=600.0)
    advance(first, 3)
    first.save_checkpoint('one')
    second = daemon(trial_length_s=600.0)
    advance(second, 3)
    second.save_checkpoint('two')
    third = daemon(trial_length_s=600.0)
    assert third.current_segment_lineage()['parent_segment_id'] == second.segment_id
    assert second.current_segment_lineage()['parent_segment_id'] == first.segment_id
    assert third.current_segment_lineage()['restored_trial_elapsed_s'] == pytest.approx(6 * DT)
    events = [e for e in ledger(third) if e['kind'] == 'observation_interrupted']
    assert [e['parent_segment_id'] for e in events] == [first.segment_id, second.segment_id]


def test_legacy_restart_lineage_names_an_unknown_parent(daemon):
    first = daemon(trial_length_s=600.0)
    advance(first, 3)
    first.save_checkpoint('pre-clock')
    make_legacy_brain_json(first)
    restarted = daemon(trial_length_s=600.0)
    lineage = restarted.current_segment_lineage()
    assert lineage['parent_segment_id'] is None and lineage['reason'] == 'daemon_restart'
    assert lineage['trial_elapsed_known'] is False and lineage['trial_known'] is False
    assert lineage['metric_accumulators'] == 'not_restored'


def test_graph_restart_lineage_comes_from_the_checkpoint(daemon):
    first = daemon('t-maze', graph=True)
    advance(first, 5)
    parent = first.segment_id
    first.save_checkpoint('mid-trial')
    assert newest_graph_meta(first)['trial_clock']['segment_id'] == parent
    restarted = daemon('t-maze', graph=True)
    lineage = restarted.current_segment_lineage()
    assert lineage['parent_segment_id'] == parent and lineage['reason'] == 'daemon_restart'
    assert restarted.latest_telemetry['clocks']['observation']['segment_elapsed_s'] == 0.0


def test_switching_back_marks_the_reactivated_segment(daemon):
    runner = daemon(trial_length_s=600.0)
    advance(runner, 3)
    sandbox_segment = runner.segment_id
    transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 'buridan'})
    assert runner.current_segment_lineage() is None             # buridan starts fresh
    transition_command(runner, {'action': 'switch_paradigm', 'paradigm': 'multisensory-sandbox'})
    lineage = runner.current_segment_lineage()
    assert lineage['parent_segment_id'] == sandbox_segment
    assert lineage['reason'] == 'assay_reactivated'
    assert lineage['parent_observation'] == 'ended_before_assay_switch'
    assert not [e for e in ledger(runner) if e['kind'] == 'observation_interrupted']


# -- legacy state stays unknown --------------------------------------------------------
def test_legacy_graph_checkpoint_stays_unknown_across_repeated_save_and_restart(daemon):
    first = daemon('t-maze', graph=True)
    advance(first, 15)
    first.save_checkpoint('helper')
    first.registry.active.trial_clock = None           # the pre-clock checkpoint schema
    first.registry.checkpoint(world_state=first.arena.snapshot_world())
    legacy = newest_graph_meta(first)
    assert 'trial_clock' not in legacy and legacy['step_index'] > 0
    assert legacy['brain_scalars']['sim_ms'] > 0
    make_legacy_brain_json(first)

    observed = 0.0
    for cycle, steps in enumerate((3, 4, 5)):
        restarted = daemon('t-maze', graph=True)
        clock = clock_of(restarted)
        assert clock['elapsed_known'] is False and clock['trial_known'] is False
        assert clock['reason'] == LEGACY_TRIAL_CLOCK_REASON
        # Never inferred from the restored brain, world or step counters.
        assert restarted.trial_sim_time == pytest.approx(observed)
        assert restarted.current_trial == 1
        assert restarted.registry.active.step_index >= legacy['step_index']
        advance(restarted, steps)
        observed += steps * DT
        restarted.save_checkpoint(f'unknown-{cycle}')
        saved = newest_graph_meta(restarted)['trial_clock']
        assert saved['elapsed_known'] is False and saved['trial_known'] is False
        assert saved['elapsed_s'] == pytest.approx(observed)   # observed lower bound only

    restarted = daemon('t-maze', graph=True)
    assert clock_of(restarted)['elapsed_known'] is False


def test_legacy_modular_brain_stays_unknown_until_a_new_trial_starts(daemon):
    first = daemon(trial_length_s=0.2)
    advance(first, 7)
    first.save_checkpoint('pre-clock')
    make_legacy_brain_json(first)
    trials_before = first.active_brain.trials

    for cycle in range(3):
        restarted = daemon(trial_length_s=0.2)
        clock = clock_of(restarted)
        assert clock == dict(sans(unknown_trial_clock()), elapsed_s=clock['elapsed_s'])
        assert restarted.active_brain.trials == trials_before
        advance(restarted, 2)
        restarted.save_checkpoint(f'unknown-{cycle}')
        assert saved_brain_json(restarted)['trial_clock']['elapsed_known'] is False
    assert clock_of(restarted)['elapsed_s'] == pytest.approx(6 * DT)

    # The trial that starts in front of this process has a known elapsed time.
    restarted = daemon(trial_length_s=0.2)
    advance(restarted, 10)
    assert len(restarted.trial_history) == 1
    assert restarted.trial_history[0]['sim_seconds'] == pytest.approx(6 * DT + 0.2)
    assert restarted.trial_history[0]['trial_known'] is False
    clock = clock_of(restarted)
    assert clock['elapsed_known'] is True and clock['elapsed_s'] == 0.0
    assert clock['trial_known'] is False and clock['reason'] == LEGACY_TRIAL_CLOCK_REASON
    after = daemon(trial_length_s=0.2)
    assert clock_of(after) == clock and after.trial_sim_time == 0.0


def test_reset_trial_command_makes_elapsed_known_and_keeps_the_ordinal_unknown(daemon):
    first = daemon(trial_length_s=600.0)
    advance(first, 3)
    first.save_checkpoint('pre-clock')
    make_legacy_brain_json(first)
    restarted = daemon(trial_length_s=600.0)
    assert clock_of(restarted)['elapsed_known'] is False
    advance(restarted, 2)
    assert transition_command(restarted, {'action': 'reset_trial'})['status'] == 'ok'
    clock = clock_of(restarted)
    assert restarted.trial_sim_time == 0.0 and clock['elapsed_s'] == 0.0
    assert clock['elapsed_known'] is True and clock['trial_known'] is False
    restarted.save_checkpoint('after-reset')
    assert clock_of(daemon(trial_length_s=600.0)) == clock


def test_invalid_modular_clock_refuses_restore_and_leaves_the_file(daemon, tmp_path):
    first = daemon()
    first.save_checkpoint('clock')
    data = saved_brain_json(first)
    data['trial_clock']['elapsed_s'] = -1
    first.active_brain.path.write_text(json.dumps(data))
    before = first.active_brain.path.read_bytes()
    with pytest.raises(ValueError, match='trial clock elapsed_s'):
        experiment_brains.ExperimentBrain('multisensory-sandbox', first.active_brain.directory)
    assert first.active_brain.path.read_bytes() == before


# -- fallback restores the older checkpoint's own clock ---------------------------------
def test_corrupt_newest_graph_checkpoint_falls_back_with_its_matching_clock(daemon):
    first = daemon('t-maze', graph=True)
    advance(first, 6)
    older_clock, older_step = clock_of(first), first.registry.active.step_index
    older_world = first.arena.snapshot_world()
    older = first.save_checkpoint('older')
    advance(first, 9)
    first.save_checkpoint('newest')
    meta = newest_graph_meta(first)
    assert meta['trial_clock']['elapsed_s'] == pytest.approx(15 * DT)
    newest = first.registry.instance_dir(first.registry.active.instance_id) / 'checkpoints' / \
        f'ckpt-{meta["version"]:06d}.npz'
    newest.write_bytes(b'corrupt retained newest checkpoint')

    restarted = daemon('t-maze', graph=True)
    instance = restarted.registry.active
    fallback = [i for i in restarted.incidents if i['reason'] == 'restored_older_checkpoint']
    assert [i['restored_version'] for i in fallback] == [meta['version'] - 1]
    assert instance.step_index == older_step
    assert clock_of(restarted) == older_clock
    assert restarted.trial_sim_time == older_clock['elapsed_s'] == pytest.approx(6 * DT)
    assert instance.world_state == json.loads(json.dumps(older_world))
    assert newest.read_bytes() == b'corrupt retained newest checkpoint'   # evidence kept
    assert json.loads(older.read_text())['graph_checkpoint']


def test_registry_fallback_restores_each_versions_own_clock(tmp_path):
    reg = make_registry(tmp_path, brain_backend='cpu', keep_checkpoints=0)
    inst = reg.activate('t-maze', 'connectome-fixed')
    inst.trial_clock, inst.step_index = trial_clock(4.34, 7), 379
    first = reg.checkpoint(world_state={'marker': 'older-world'})
    inst.trial_clock, inst.step_index = trial_clock(9.99, 9), 999
    newest = reg.checkpoint(world_state={'marker': 'newest-world'})
    original_older = first.read_bytes()
    newest.write_bytes(b'corrupt retained newest checkpoint')
    fresh = make_registry(tmp_path, brain_backend='cpu', keep_checkpoints=0)
    restored = fresh.activate('t-maze', 'connectome-fixed')
    assert restored.trial_clock == trial_clock(4.34, 7)
    assert restored.step_index == 379 and restored.world_state == {'marker': 'older-world'}
    assert restored.restore_fallback['restored_version'] == 1
    assert first.read_bytes() == original_older


def test_malformed_clock_makes_a_checkpoint_incompatible(tmp_path):
    reg = make_registry(tmp_path, brain_backend='cpu', keep_checkpoints=0)
    inst = reg.activate('t-maze', 'connectome-fixed')
    meta = inst.meta()
    meta.update(version=1, trial_clock=dict(trial_clock(), elapsed_s=-1.0))
    path = tmp_path / 'bad-clock.npz'
    np.savez(path, meta=np.array(json.dumps(meta)), **inst.state_arrays())
    with pytest.raises(IncompatibleCheckpoint, match='trial clock elapsed_s'):
        load_checkpoint_file(path)
    meta.pop('trial_clock')
    np.savez(path, meta=np.array(json.dumps(meta)), **inst.state_arrays())
    assert 'trial_clock' not in load_checkpoint_file(path)[0]


# -- the trial-completion boundary around a save ---------------------------------------
def test_trial_completion_boundary_saves_a_matching_count_and_clock(daemon, monkeypatch):
    first = daemon(trial_length_s=0.2)
    advance(first, 9)                               # one step before the window ends
    first.save_checkpoint('before-boundary')
    before = saved_brain_json(first)
    assert before['trials'] == 0 and sans(before['trial_clock']) == tc(9 * DT, 1)

    saves = []
    real_save = experiment_brains.ExperimentBrain.save

    def recording_save(brain):
        saves.append((brain.trials, sans(brain.trial_clock)))
        return real_save(brain)

    monkeypatch.setattr(experiment_brains.ExperimentBrain, 'save', recording_save)
    advance(first, 1)                               # the boundary: trial 1 completes
    monkeypatch.undo()
    assert [t['assay_trial'] for t in first.trial_history] == [1]
    # The boundary save holds the advanced count and the new trial's clock, never
    # "1 completed" with trial 1 still running at its old elapsed time.
    assert saves == [(1, tc(0.0, 2))]
    on_disk = saved_brain_json(first)
    assert on_disk['trials'] == 1 and sans(on_disk['trial_clock']) == tc(0.0, 2)

    after = daemon(trial_length_s=0.2)
    assert after.current_trial == 2 and after.trial_sim_time == 0.0
    assert after.active_brain.trials == 1


def test_restart_just_before_a_trial_boundary_keeps_that_trial(daemon):
    first = daemon(trial_length_s=0.2)
    advance(first, 9)
    first.save_checkpoint('before-boundary')
    restarted = daemon(trial_length_s=0.2)
    assert restarted.current_trial == 1 and restarted.trial_sim_time == pytest.approx(9 * DT)
    assert restarted.active_brain.trials == 0
    advance(restarted, 10)                          # a fresh observation window
    assert [t['assay_trial'] for t in restarted.trial_history] == [1]
    assert restarted.trial_history[0]['sim_seconds'] == pytest.approx(9 * DT + 0.2)
    assert restarted.current_trial == 2 and restarted.active_brain.trials == 1


def test_graph_restore_after_a_later_trial_boundary_uses_the_checkpoints_clock(daemon):
    first = daemon('t-maze', graph=True, trial_length_s=0.2)
    advance(first, 7)
    first.save_checkpoint('before-boundary')
    saved = newest_graph_meta(first)
    advance(first, 5)                               # trial 1 ends without a graph checkpoint
    assert len(first.trial_history) == 1 and first.current_trial == 2

    restarted = daemon('t-maze', graph=True, trial_length_s=0.2)
    assert restarted.registry.active.step_index == saved['step_index']
    assert clock_of(restarted) == sans(saved['trial_clock']) == tc(saved['trial_clock']['elapsed_s'], 1)
    assert restarted.trial_sim_time == pytest.approx(7 * DT)


@pytest.mark.parametrize('updates', [
    {'elapsed_s': float('nan')}, {'elapsed_s': -0.1}, {'current_trial': True}, {'current_trial': 0},
    {'elapsed_known': 'yes'}, {'elapsed_known': False, 'reason': None}, {'reason': 'stale'},
    {'schema': 'other'}, {'extra': 1}])
def test_malformed_trial_clock_is_refused(updates):
    with pytest.raises(ValueError):
        validate_trial_clock(dict(trial_clock(), **updates))
    assert trial_clock()['schema'] == TRIAL_CLOCK_SCHEMA
