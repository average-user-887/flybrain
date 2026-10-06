"""Linked-run migration preserves graph state while changing its I/O identity."""
import hashlib
import io
import json

import numpy as np
import pytest

from arena import Arena
from experiment_registry import (CheckpointCorrupt, ExperimentRegistry, IncompatibleCheckpoint,
                                  IOContinuationRequired, SharedGraph, TestOnlyCoactivityRule)
from neurofly_daemon import ContinuousExperimentRunner, build_arg_parser
from provenance import restore_rng


def _registry(root, *, continuation=False, keep=20):
    shared = SharedGraph.synthetic(allow_synthetic=True, n=64, k_out=6, seed=17)
    rule = TestOnlyCoactivityRule(edges=np.arange(0, 384, 7), eta=0.05)
    return ExperimentRegistry(shared, root, test_mode=True, plasticity_rule=rule,
                              keep_checkpoints=keep,
                              continue_io_state=continuation)


def _drive(instance, steps):
    for _ in range(steps):
        current = np.zeros(instance.shared.n, dtype=np.float32)
        current[instance.rng.choice(instance.shared.n, 8, replace=False)] = 19.0
        instance.step(current, 5.0)


def _hash_tree(path):
    return {str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in path.rglob('*') if item.is_file()}


def _rewrite_checkpoint_graph_io(parent_dir, version, graph_io):
    checkpoint = parent_dir / 'checkpoints' / f'ckpt-{version:06d}.npz'
    with np.load(checkpoint, allow_pickle=False) as archive:
        meta = json.loads(str(archive['meta']))
        arrays = {name: archive[name] for name in archive.files if name != 'meta'}
    if graph_io is None:
        meta.pop('graph_io', None)
    else:
        meta['graph_io'] = graph_io
    buffer = io.BytesIO()
    np.savez(buffer, meta=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    checkpoint.write_bytes(buffer.getvalue())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    pointer_path = parent_dir / 'CURRENT.json'
    pointer = json.loads(pointer_path.read_text())
    if int(pointer['version']) == version:
        pointer.update(sha256=digest, bytes=checkpoint.stat().st_size)
        pointer_path.write_text(json.dumps(pointer, indent=2) + '\n')
    events_path = parent_dir / 'events.jsonl'
    events = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') == 'checkpoint' and int(event['version']) == version:
            event['sha256'] = digest
        events.append(event)
    events_path.write_text(''.join(json.dumps(event) + '\n' for event in events))


def _select_parent_version(registry, parent_id, version):
    parent_dir = registry.instance_dir(parent_id)
    path = parent_dir / 'checkpoints' / f'ckpt-{version:06d}.npz'
    pointer = {
        'version': version, 'file': path.name,
        'sha256': registry._recorded_hashes(parent_id)[version],
        'bytes': path.stat().st_size,
        'step_index': registry._recorded_steps(parent_id)[version],
        'format': 'neurofly.graph-instance-checkpoint.v1',
    }
    (parent_dir / 'CURRENT.json').write_text(json.dumps(pointer, indent=2) + '\n')


def _two_completed_lineages(root):
    shared, parent_id, parent_run, parent_dir = _make_legacy_parent(root, checkpoints=2)
    first = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    child1 = first.activate('t-maze', 'connectome-fixed')

    second = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    key = 't-maze|connectome-fixed'
    second.index['current_instances'][key] = parent_id
    second._write_index()
    _select_parent_version(second, parent_id, 1)
    child2 = second.activate('t-maze', 'connectome-fixed')
    return shared, parent_id, parent_run, parent_dir, child1, child2


def _make_legacy_parent(root, *, invalid=False, checkpoints=1, world_state=None,
                        backend='connectome-fixed'):
    registry = _registry(root)
    parent = registry.activate('t-maze', backend)
    parent.world_state = (world_state if world_state is not None else
                          {'pose': [4.0, 5.0, 0.25], 'arena_rng': {'counter': 9}})
    if invalid:
        parent.invalidity = [{'reason': 'required_save_failed', 'run_id': parent.manifest.run_id,
                              'at': 123.0}]
        parent.manifest.intervention_schedule.append({'kind': 'ablation', 'target': 'example'})
        parent.manifest.write(registry.instance_dir(parent.instance_id) / 'manifest.json')
    for _ in range(checkpoints):
        _drive(parent, 3)
        registry.checkpoint()
    parent_id = parent.instance_id
    parent_run = parent.manifest.run_id
    parent_dir = registry.instance_dir(parent_id)

    # Model a pre-I/O-identity store: neither manifest nor checkpoint claims the
    # new method, while its digest records remain internally correct.
    manifest = registry.manifest(parent_id)
    manifest.graph_io = None
    manifest.write(parent_dir / 'manifest.json')
    for checkpoint in sorted((parent_dir / 'checkpoints').glob('ckpt-*.npz')):
        with np.load(checkpoint, allow_pickle=False) as archive:
            meta = json.loads(str(archive['meta']))
            arrays = {name: archive[name] for name in archive.files if name != 'meta'}
        meta.pop('graph_io', None)
        buffer = io.BytesIO()
        np.savez(buffer, meta=np.array(json.dumps(meta, allow_nan=False)), **arrays)
        checkpoint.write_bytes(buffer.getvalue())
    pointer_path = parent_dir / 'CURRENT.json'
    pointer = json.loads(pointer_path.read_text())
    current_path = parent_dir / 'checkpoints' / pointer['file']
    pointer['sha256'] = hashlib.sha256(current_path.read_bytes()).hexdigest()
    pointer['bytes'] = current_path.stat().st_size
    pointer_path.write_text(json.dumps(pointer, indent=2) + '\n')
    events_path = parent_dir / 'events.jsonl'
    events = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') == 'checkpoint':
            version_path = parent_dir / 'checkpoints' / f'ckpt-{int(event["version"]):06d}.npz'
            event['sha256'] = hashlib.sha256(version_path.read_bytes()).hexdigest()
        events.append(event)
    events_path.write_text(''.join(json.dumps(event) + '\n' for event in events))
    return registry.shared, parent_id, parent_run, parent_dir


def _assert_same_payload(parent_meta, parent_arrays, child):
    assert child.step_index == parent_meta['step_index']
    assert child.rng.bit_generator.state == restore_rng(parent_meta['rng_state']).bit_generator.state
    assert child.world_state == parent_meta['world_state']
    child_arrays = child.state_arrays()
    assert set(child_arrays) == set(parent_arrays)
    for key in parent_arrays:
        np.testing.assert_array_equal(child_arrays[key], parent_arrays[key], err_msg=key)


def test_default_refuses_legacy_before_writes_and_explicit_option_links_child(tmp_path):
    shared, parent_id, parent_run, parent_dir = _make_legacy_parent(tmp_path / 'registry')
    before = _hash_tree(parent_dir)

    refused = ExperimentRegistry(shared, tmp_path / 'registry', test_mode=True)
    with pytest.raises(IOContinuationRequired, match='--continue-io-state'):
        refused.activate('t-maze', 'connectome-fixed')
    assert _hash_tree(parent_dir) == before

    migrating = ExperimentRegistry(shared, tmp_path / 'registry', test_mode=True,
                                   continue_io_state=True)
    parent_meta, parent_arrays, _ = migrating._select_parent_checkpoint(parent_id)
    child = migrating.activate('t-maze', 'connectome-fixed')
    assert child.instance_id != parent_id and child.manifest.run_id != parent_run
    _assert_same_payload(parent_meta, parent_arrays, child)
    assert _hash_tree(parent_dir) == before
    assert child.manifest.parent_run_id == parent_run
    assert child.manifest.lineage['prior_graph_io'] == {'version': 'unknown', 'sha256': 'unknown'}
    assert child.manifest.lineage['parent_checkpoint']['sha256']
    assert migrating.instance_id_for_run(parent_run) == parent_id


def test_restart_reuses_current_child_and_preserves_inherited_evidence(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, parent_run, parent_dir = _make_legacy_parent(root, invalid=True)
    before = _hash_tree(parent_dir)
    first = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    child = first.activate('t-maze', 'connectome-fixed')
    child_id = child.instance_id
    assert child.manifest.intervention_schedule[0] == {
        'kind': 'ablation', 'target': 'example', 'source_run_id': parent_run,
    }
    assert len(child.manifest.intervention_schedule) == 5
    assert child.invalidity[0]['source_run_id'] == parent_run
    first.checkpoint()

    restarted = ExperimentRegistry(shared, root, test_mode=True)
    restored = restarted.activate('t-maze', 'connectome-fixed')
    assert restored.instance_id == child_id
    assert restarted.activate('t-maze', 'connectome-fixed').instance_id == child_id
    assert len([entry for entry in restarted.index['instances'].values()
                if entry['assay'] == 't-maze' and entry['backend'] == 'connectome-fixed']) == 2
    assert restarted.instance_id_for_run(parent_run) == parent_id
    assert _hash_tree(parent_dir) == before


@pytest.mark.parametrize('backend', ['connectome-plastic', 'connectome-with-trained-readout'])
def test_linked_child_preserves_learned_arrays(tmp_path, backend):
    root = tmp_path / backend
    original = _registry(root)
    parent = original.activate('t-maze', backend)
    if backend == 'connectome-plastic':
        parent.plastic_delta[:] = np.linspace(-0.25, 0.25, len(parent.plastic_delta), dtype=np.float32)
        parent._materialize()
    else:
        parent.readout.weights[:] = np.linspace(-1.0, 1.0, len(parent.readout.weights))
    _drive(parent, 2)
    original.checkpoint(world_state={'learned': backend})
    parent_id = parent.instance_id
    parent_dir = original.instance_dir(parent_id)
    manifest = original.manifest(parent_id)
    manifest.graph_io = None
    manifest.write(parent_dir / 'manifest.json')

    # Checkpoint graph I/O is additive evidence; the parent manifest controls
    # method compatibility and is explicitly unknown in this legacy fixture.
    before = _hash_tree(parent_dir)
    rule = TestOnlyCoactivityRule(edges=np.arange(0, 384, 7), eta=0.05)
    migrated = ExperimentRegistry(original.shared, root, test_mode=True, plasticity_rule=rule,
                                  continue_io_state=True)
    child = migrated.activate('t-maze', backend)
    _, parent_arrays = migrated.read_checkpoint(parent_id)
    child_arrays = child.state_arrays()
    learned_key = 'plastic.delta' if backend == 'connectome-plastic' else 'readout.weights'
    np.testing.assert_array_equal(child_arrays[learned_key], parent_arrays[learned_key])
    assert child.world_state == {'learned': backend}
    assert _hash_tree(parent_dir) == before


def test_corrupt_newest_uses_recorded_fallback_and_no_valid_checkpoint_refuses(tmp_path):
    root = tmp_path / 'fallback'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root, checkpoints=2)
    current = json.loads((parent_dir / 'CURRENT.json').read_text())
    newest = parent_dir / 'checkpoints' / current['file']
    damaged = bytearray(newest.read_bytes())
    damaged[-8] ^= 0xFF
    newest.write_bytes(damaged)
    before = _hash_tree(parent_dir)

    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    child = registry.activate('t-maze', 'connectome-fixed')
    selected = child.manifest.lineage['parent_checkpoint']
    assert selected['fallback'] is True and selected['version'] == 1
    assert selected['lost_steps'] == 3
    assert child.invalidity[-1]['reason'] == 'io_migration_restored_older_checkpoint'
    assert _hash_tree(parent_dir) == before

    broken_root = tmp_path / 'none-valid'
    shared2, _, _, parent_dir2 = _make_legacy_parent(broken_root)
    pointer2 = json.loads((parent_dir2 / 'CURRENT.json').read_text())
    (parent_dir2 / 'checkpoints' / pointer2['file']).write_bytes(b'broken')
    before2 = _hash_tree(parent_dir2)
    broken = ExperimentRegistry(shared2, broken_root, test_mode=True, continue_io_state=True)
    with pytest.raises(CheckpointCorrupt, match='no saved checkpoint'):
        broken.activate('t-maze', 'connectome-fixed')
    assert _hash_tree(parent_dir2) == before2


@pytest.mark.parametrize('stage', ['child_payload_write', 'store_publication', 'index_publication'])
def test_publication_failures_preserve_parent_and_restart_never_selects_partial(tmp_path, stage):
    root = tmp_path / stage
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)

    def fail(at):
        if at == stage:
            raise OSError(f'injected {stage} failure')

    registry._migration_fault = fail
    with pytest.raises(OSError, match='injected'):
        registry.activate('t-maze', 'connectome-fixed')
    assert _hash_tree(parent_dir) == before

    restarted = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    child = restarted.activate('t-maze', 'connectome-fixed')
    assert child.instance_id != parent_id
    current = restarted.index['current_instances']['t-maze|connectome-fixed']
    assert current == child.instance_id
    assert (restarted.instance_dir(current) / 'CURRENT.json').is_file()
    assert _hash_tree(parent_dir) == before


def test_child_retention_cannot_remove_parent_checkpoint(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, parent_run, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True,
                                  keep_checkpoints=1)
    child = registry.activate('t-maze', 'connectome-fixed')
    for _ in range(3):
        _drive(child, 1)
        registry.checkpoint()
    assert _hash_tree(parent_dir) == parent_before
    assert registry.instance_id_for_run(parent_run) == parent_id


def test_daemon_requires_and_retains_explicit_continuation_option(tmp_path):
    root = tmp_path / 'registry'
    world = Arena(paradigm='t-maze', seed=3, num_flies=1, num_predators=0).snapshot_world()
    shared, parent_id, _, parent_dir = _make_legacy_parent(root, world_state=world)
    before = _hash_tree(parent_dir)
    args = build_arg_parser().parse_args(['--continue-io-state'])
    assert args.continue_io_state is True

    common = dict(initial_paradigm='t-maze', output_dir=tmp_path / 'outputs',
                  backend='connectome-fixed', test_synthetic_graph=True,
                  shared_graph=shared, registry_root=root)
    with pytest.raises(IOContinuationRequired):
        ContinuousExperimentRunner(**common)
    runner = ContinuousExperimentRunner(**common, continue_io_state=True)
    assert runner.registry.active.instance_id != parent_id
    assert runner.identity()['graph_io'] == runner.registry.graph_io
    assert _hash_tree(parent_dir) == before


def test_same_process_retry_never_uses_an_unpublished_in_memory_child(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    real_write = registry._write_index
    calls = []

    def fail_index_write():
        calls.append('write')
        raise OSError('injected index publication failure')

    registry._write_index = fail_index_write
    for _ in range(2):
        with pytest.raises(OSError, match='index publication'):
            registry.activate('t-maze', 'connectome-fixed')
        assert registry.index['current_instances']['t-maze|connectome-fixed'] == parent_id
        assert json.loads((root / 'registry.json').read_text())['current_instances'][
            't-maze|connectome-fixed'] == parent_id
    assert calls == ['write', 'write']

    registry._write_index = real_write
    child = registry.activate('t-maze', 'connectome-fixed')
    assert child.instance_id != parent_id
    assert len(registry.index['instances']) == 2
    assert _hash_tree(parent_dir) == parent_before


def test_directory_sync_failure_after_store_rename_recovers_one_child(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    real_sync = registry._sync_dir
    failed = []

    def fail_after_store_rename(path):
        path = type(root)(path)
        if (not failed and path == root / 't-maze' / 'connectome-fixed'
                and len([item for item in path.iterdir() if item.is_dir()]) == 2):
            failed.append(path)
            raise OSError('injected post-rename directory sync failure')
        return real_sync(path)

    registry._sync_dir = fail_after_store_rename
    with pytest.raises(OSError, match='post-rename'):
        registry.activate('t-maze', 'connectome-fixed')
    assert registry.index['current_instances']['t-maze|connectome-fixed'] == parent_id

    registry._sync_dir = real_sync
    child = registry.activate('t-maze', 'connectome-fixed')
    assert child.instance_id != parent_id
    assert len(registry.index['instances']) == 2
    assert _hash_tree(parent_dir) == parent_before


def test_persistent_store_publication_sync_failure_refuses_same_process_retry(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    parent_meta, parent_arrays, _ = registry._select_parent_checkpoint(parent_id)
    critical = root / 't-maze' / 'connectome-fixed'
    real_sync = registry._sync_dir
    critical_calls = []

    def fail_critical_sync(path):
        path = type(root)(path)
        if path == critical and len([item for item in path.iterdir() if item.is_dir()]) == 2:
            critical_calls.append(path)
            raise OSError('persistent child-store publication sync failure')
        return real_sync(path)

    registry._sync_dir = fail_critical_sync
    for expected_calls in (1, 2):
        with pytest.raises(OSError, match='persistent child-store'):
            registry.activate('t-maze', 'connectome-fixed')
        assert len(critical_calls) == expected_calls
        assert registry.index['current_instances']['t-maze|connectome-fixed'] == parent_id
        assert json.loads((root / 'registry.json').read_text())['current_instances'][
            't-maze|connectome-fixed'] == parent_id
        assert len([item for item in critical.iterdir() if item.is_dir()]) == 2
        assert _hash_tree(parent_dir) == parent_before
    failed_child = next(item.name for item in critical.iterdir()
                        if item.is_dir() and item.name != parent_id)

    events = []

    def track_sync(path):
        path = type(root)(path)
        if path == critical:
            events.append('store_sync')
        return real_sync(path)

    real_write_index = registry._write_index
    real_write_intent = registry._write_intent

    def track_index():
        events.append('index')
        return real_write_index()

    def track_intent(path, intent):
        if intent.get('phase') == 'complete':
            events.append('complete')
        return real_write_intent(path, intent)

    registry._sync_dir = track_sync
    registry._write_index = track_index
    registry._write_intent = track_intent
    child = registry.activate('t-maze', 'connectome-fixed')
    assert events.index('store_sync') < events.index('index') < events.index('complete')
    assert child.instance_id == failed_child
    _assert_same_payload(parent_meta, parent_arrays, child)
    assert len(registry.index['instances']) == 2
    assert len([item for item in critical.iterdir() if item.is_dir()]) == 2
    assert _hash_tree(parent_dir) == parent_before


def test_persistent_store_publication_sync_failure_refuses_restart_recovery(tmp_path,
                                                                            monkeypatch):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    parent_meta, parent_arrays, _ = registry._select_parent_checkpoint(parent_id)
    critical = root / 't-maze' / 'connectome-fixed'
    real_sync = ExperimentRegistry._sync_dir
    critical_calls = []

    def fail_critical_sync(path):
        path = type(root)(path)
        if path == critical and len([item for item in path.iterdir() if item.is_dir()]) == 2:
            critical_calls.append(path)
            raise OSError('persistent child-store publication sync failure')
        return real_sync(path)

    registry._sync_dir = fail_critical_sync
    with pytest.raises(OSError, match='persistent child-store'):
        registry.activate('t-maze', 'connectome-fixed')
    assert len(critical_calls) == 1

    monkeypatch.setattr(ExperimentRegistry, '_sync_dir', staticmethod(fail_critical_sync))
    with pytest.raises(OSError, match='persistent child-store'):
        ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    assert len(critical_calls) == 2
    assert json.loads((root / 'registry.json').read_text())['current_instances'][
        't-maze|connectome-fixed'] == parent_id
    assert len([item for item in critical.iterdir() if item.is_dir()]) == 2
    assert _hash_tree(parent_dir) == parent_before
    failed_child = next(item.name for item in critical.iterdir()
                        if item.is_dir() and item.name != parent_id)

    events = []

    class TrackingRegistry(ExperimentRegistry):
        @staticmethod
        def _sync_dir(path):
            path = type(root)(path)
            if path == critical:
                events.append('store_sync')
            return real_sync(path)

        def _write_index(self):
            events.append('index')
            return super()._write_index()

        def _write_intent(self, path, intent):
            if intent.get('phase') == 'complete':
                events.append('complete')
            return super()._write_intent(path, intent)

    recovered = TrackingRegistry(shared, root, test_mode=True, continue_io_state=True)
    assert events.index('store_sync') < events.index('index') < events.index('complete')
    child = recovered.activate('t-maze', 'connectome-fixed')
    assert child.instance_id == failed_child
    _assert_same_payload(parent_meta, parent_arrays, child)
    assert len(recovered.index['instances']) == 2
    assert len([item for item in critical.iterdir() if item.is_dir()]) == 2
    assert _hash_tree(parent_dir) == parent_before


def test_index_directory_sync_failure_republishes_before_same_process_success(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    real_sync = registry._sync_dir
    failed_child = []

    def fail_after_index_rename(path):
        if type(root)(path) == root and not failed_child:
            disk = json.loads((root / 'registry.json').read_text())
            selected = disk['current_instances']['t-maze|connectome-fixed']
            if selected != parent_id:
                failed_child.append(selected)
                raise OSError('injected index directory sync failure')
        return real_sync(path)

    registry._sync_dir = fail_after_index_rename
    with pytest.raises(OSError, match='index directory sync'):
        registry.activate('t-maze', 'connectome-fixed')
    assert registry.index['current_instances']['t-maze|connectome-fixed'] == parent_id
    assert json.loads((root / 'registry.json').read_text())['current_instances'][
        't-maze|connectome-fixed'] == failed_child[0]

    registry._sync_dir = real_sync
    child = registry.activate('t-maze', 'connectome-fixed')
    assert child.instance_id == failed_child[0]
    assert len(registry.index['instances']) == 2
    assert _hash_tree(parent_dir) == parent_before


def test_failure_marking_completed_intent_recovers_without_another_child(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _make_legacy_parent(root)
    parent_before = _hash_tree(parent_dir)
    registry = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    real_write_intent = registry._write_intent
    failed_child = []

    def fail_completed_intent(path, intent):
        if intent.get('phase') == 'complete' and not failed_child:
            failed_child.append(intent['child_instance_id'])
            raise OSError('injected completed-intent failure')
        return real_write_intent(path, intent)

    registry._write_intent = fail_completed_intent
    with pytest.raises(OSError, match='completed-intent'):
        registry.activate('t-maze', 'connectome-fixed')
    assert registry.index['current_instances']['t-maze|connectome-fixed'] == failed_child[0]

    registry._write_intent = real_write_intent
    child = registry.activate('t-maze', 'connectome-fixed')
    assert child.instance_id == failed_child[0]
    assert len(registry.index['instances']) == 2
    assert _hash_tree(parent_dir) == parent_before


def test_restart_does_not_replay_completed_intents_over_newer_selection(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, parent_run, parent_dir, child1, child2 = _two_completed_lineages(root)
    parent_before = _hash_tree(parent_dir)
    assert child1.instance_id != child2.instance_id

    restarted = ExperimentRegistry(shared, root, test_mode=True)
    assert restarted.index['current_instances']['t-maze|connectome-fixed'] == child2.instance_id
    assert restarted.instance_id_for_run(parent_run) == parent_id
    assert restarted.instance_id_for_run(child1.manifest.run_id) == child1.instance_id
    assert restarted.instance_id_for_run(child2.manifest.run_id) == child2.instance_id
    assert _hash_tree(parent_dir) == parent_before


def test_pending_older_intent_cannot_supersede_later_current_child(tmp_path):
    root = tmp_path / 'registry'
    shared, _, _, _, child1, child2 = _two_completed_lineages(root)
    for path in (root / '.migration-intents').glob('*.json'):
        intent = json.loads(path.read_text())
        if intent['child_instance_id'] == child1.instance_id:
            intent['phase'] = 'store_published'
            path.write_text(json.dumps(intent, indent=2) + '\n')

    with pytest.raises(IncompatibleCheckpoint, match='expected current instance'):
        ExperimentRegistry(shared, root, test_mode=True)
    assert json.loads((root / 'registry.json').read_text())['current_instances'][
        't-maze|connectome-fixed'] == child2.instance_id


def test_competing_pending_intents_refuse_before_selecting_one(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, _, _, _ = _two_completed_lineages(root)
    for path in (root / '.migration-intents').glob('*.json'):
        intent = json.loads(path.read_text())
        intent['phase'] = 'store_published'
        path.write_text(json.dumps(intent, indent=2) + '\n')
    index_path = root / 'registry.json'
    index = json.loads(index_path.read_text())
    index['current_instances']['t-maze|connectome-fixed'] = parent_id
    index_path.write_text(json.dumps(index, indent=2) + '\n')

    with pytest.raises(IncompatibleCheckpoint, match='ambiguous pending'):
        ExperimentRegistry(shared, root, test_mode=True)
    assert json.loads(index_path.read_text())['current_instances'][
        't-maze|connectome-fixed'] == parent_id


def test_known_manifest_checkpoint_io_contradiction_refuses_before_publication(tmp_path):
    root = tmp_path / 'registry'
    original = _registry(root)
    parent = original.activate('t-maze', 'connectome-fixed')
    _drive(parent, 1)
    original.checkpoint(world_state={'marker': 'known-mismatch'})
    parent_dir = original.instance_dir(parent.instance_id)
    manifest = original.manifest(parent.instance_id)
    manifest.graph_io = {'version': 'prior-v0', 'sha256': '0' * 64}
    manifest.write(parent_dir / 'manifest.json')
    parent_before = _hash_tree(parent_dir)

    continued = ExperimentRegistry(original.shared, root, test_mode=True, continue_io_state=True)
    with pytest.raises(IncompatibleCheckpoint, match='graph_io contradicts'):
        continued.activate('t-maze', 'connectome-fixed')
    assert _hash_tree(parent_dir) == parent_before
    assert not (root / '.migration-intents').exists()


def test_one_missing_io_source_remains_compatible_and_lineage_keeps_known_source(tmp_path):
    manifest_missing = tmp_path / 'manifest-missing'
    original = _registry(manifest_missing)
    parent = original.activate('t-maze', 'connectome-fixed')
    _drive(parent, 1)
    original.checkpoint()
    parent_dir = original.instance_dir(parent.instance_id)
    checkpoint_io = original.graph_io.copy()
    manifest = original.manifest(parent.instance_id)
    manifest.graph_io = None
    manifest.write(parent_dir / 'manifest.json')
    child = ExperimentRegistry(original.shared, manifest_missing, test_mode=True,
                               continue_io_state=True).activate('t-maze', 'connectome-fixed')
    assert child.manifest.lineage['prior_graph_io'] == checkpoint_io
    assert child.manifest.lineage['prior_graph_io_sources'] == {
        'manifest': {'version': 'unknown', 'sha256': 'unknown'},
        'checkpoint': checkpoint_io,
    }

    checkpoint_missing = tmp_path / 'checkpoint-missing'
    original = _registry(checkpoint_missing)
    parent = original.activate('t-maze', 'connectome-fixed')
    _drive(parent, 1)
    original.checkpoint()
    parent_dir = original.instance_dir(parent.instance_id)
    prior_io = {'version': 'prior-v0', 'sha256': '1' * 64}
    manifest = original.manifest(parent.instance_id)
    manifest.graph_io = prior_io
    manifest.write(parent_dir / 'manifest.json')
    _rewrite_checkpoint_graph_io(parent_dir, 1, None)
    child = ExperimentRegistry(original.shared, checkpoint_missing, test_mode=True,
                               continue_io_state=True).activate('t-maze', 'connectome-fixed')
    assert child.manifest.lineage['prior_graph_io'] == prior_io
    assert child.manifest.lineage['prior_graph_io_sources'] == {
        'manifest': prior_io,
        'checkpoint': {'version': 'unknown', 'sha256': 'unknown'},
    }
