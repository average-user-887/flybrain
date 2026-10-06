"""Upgrade refusal preserves learned state; app revisions do not force retraining."""
import hashlib
import io
import json

import numpy as np
import pytest

from experiment_registry import (ExperimentRegistry, IncompatibleCheckpoint,
                                 SharedGraph, TestOnlyCoactivityRule)


def registry(root, eta=0.05, continuation=False):
    shared = SharedGraph.synthetic(allow_synthetic=True, n=64, k_out=6, seed=17)
    rule = TestOnlyCoactivityRule(edges=np.arange(0, 384, 7), eta=eta)
    return ExperimentRegistry(shared, root, test_mode=True, plasticity_rule=rule,
                              continue_io_state=continuation)


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def saved(root):
    original = registry(root)
    instance = original.activate('t-maze', 'connectome-plastic')
    instance.plastic_delta[:] = np.linspace(-0.1, 0.1, len(instance.plastic_delta))
    original.checkpoint(world_state={'retained': True})
    return original, instance


def rewrite_meta(original, instance, update):
    directory = original.instance_dir(instance.instance_id)
    pointer_path = directory / 'CURRENT.json'
    pointer = json.loads(pointer_path.read_text())
    path = directory / 'checkpoints' / pointer['file']
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['meta']))
        arrays = {key: archive[key] for key in archive.files if key != 'meta'}
    update(meta)
    buffer = io.BytesIO()
    np.savez(buffer, meta=np.array(json.dumps(meta)), **arrays)
    path.write_bytes(buffer.getvalue())
    pointer['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    pointer['bytes'] = path.stat().st_size
    pointer_path.write_text(json.dumps(pointer))


@pytest.mark.parametrize('continuation', [False, True])
def test_changed_eta_refuses_before_selection_or_active_release(tmp_path, continuation):
    original, instance = saved(tmp_path)
    if continuation:
        manifest = original.manifest(instance.instance_id)
        manifest.graph_io = None
        manifest.write(original.instance_dir(instance.instance_id) / 'manifest.json')
    # Simulate a legacy index whose selection would otherwise be written on load.
    original.index.pop('current_instances')
    original._write_index()
    requested = registry(tmp_path, eta=0.2, continuation=continuation)
    active = requested.activate('open-arena', 'connectome-fixed')
    before = hashes(tmp_path)
    with pytest.raises(IncompatibleCheckpoint, match='plasticity rule differs'):
        requested.activate('t-maze', 'connectome-plastic')
    assert requested.active is active
    assert active.brain is not None
    assert hashes(tmp_path) == before
    assert requested.index.get('current_instances', {}).get('t-maze|connectome-plastic') is None


def test_same_rule_source_upgrade_restores_learned_state_and_ids(tmp_path):
    original, instance = saved(tmp_path)
    manifest = original.manifest(instance.instance_id)
    manifest.source['commit'] = '0' * 40
    manifest.source['app_version'] = 'older-app-version'
    manifest.write(original.instance_dir(instance.instance_id) / 'manifest.json')
    before = hashes(tmp_path)
    events = next(tmp_path.rglob('events.jsonl')).read_bytes()
    requested = registry(tmp_path)
    restored = requested.activate('t-maze', 'connectome-plastic')
    assert restored.instance_id == instance.instance_id
    assert restored.manifest.run_id == instance.manifest.run_id
    np.testing.assert_array_equal(restored.plastic_delta, instance.plastic_delta)
    assert restored.world_state == {'retained': True}
    after = hashes(tmp_path)
    assert {k: v for k, v in after.items() if not k.endswith('events.jsonl')} == {
        k: v for k, v in before.items() if not k.endswith('events.jsonl')}
    assert next(tmp_path.rglob('events.jsonl')).read_bytes().startswith(events)
    currents = np.full(instance.shared.n, 19.0, dtype=np.float32)
    instance.step(currents, 5.0)
    restored.step(currents, 5.0)
    for key, value in instance.state_arrays().items():
        np.testing.assert_array_equal(restored.state_arrays()[key], value)


@pytest.mark.parametrize('missing', ['manifest', 'checkpoint'])
def test_one_legacy_descriptor_is_sufficient(tmp_path, missing):
    original, instance = saved(tmp_path)
    if missing == 'manifest':
        manifest = original.manifest(instance.instance_id)
        manifest.dynamics.pop('plasticity_rule')
        manifest.write(original.instance_dir(instance.instance_id) / 'manifest.json')
    else:
        rewrite_meta(original, instance, lambda meta: meta.pop('rule'))
    before = hashes(tmp_path)
    events = next(tmp_path.rglob('events.jsonl')).read_bytes()
    restored = registry(tmp_path).activate('t-maze', 'connectome-plastic')
    np.testing.assert_array_equal(restored.plastic_delta, instance.plastic_delta)
    after = hashes(tmp_path)
    assert {k: v for k, v in after.items() if not k.endswith('events.jsonl')} == {
        k: v for k, v in before.items() if not k.endswith('events.jsonl')}
    assert next(tmp_path.rglob('events.jsonl')).read_bytes().startswith(events)


@pytest.mark.parametrize('problem', ['missing', 'contradictory', 'malformed', 'incomplete'])
def test_unknown_or_conflicting_rule_evidence_preserves_originals(tmp_path, problem):
    original, instance = saved(tmp_path)
    if problem == 'missing':
        manifest = original.manifest(instance.instance_id)
        manifest.dynamics.pop('plasticity_rule')
        manifest.write(original.instance_dir(instance.instance_id) / 'manifest.json')
        rewrite_meta(original, instance, lambda meta: meta.pop('rule'))
        message = 'evidence is missing'
    elif problem == 'contradictory':
        rewrite_meta(original, instance, lambda meta: meta['rule'].update(eta=0.2))
        message = 'contradicts'
    elif problem == 'malformed':
        rewrite_meta(original, instance, lambda meta: meta.update(rule=[]))
        message = 'evidence is malformed'
    else:
        rewrite_meta(original, instance, lambda meta: meta['rule'].pop('eta'))
        message = 'contradicts'
    before = hashes(tmp_path)
    requested = registry(tmp_path)
    with pytest.raises(IncompatibleCheckpoint, match=message):
        requested.activate('t-maze', 'connectome-plastic')
    assert requested.active is None
    assert hashes(tmp_path) == before


def test_direct_payload_refusal_precedes_brain_mutation(tmp_path):
    original, instance = saved(tmp_path)
    meta, arrays = original.read_checkpoint(instance.instance_id)
    before = instance.state_arrays()
    meta['rule']['eta'] = 0.2
    arrays['brain.v'][:] = -1.0
    with pytest.raises(IncompatibleCheckpoint, match='contradicts'):
        instance.load_payload(meta, arrays)
    after = instance.state_arrays()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
