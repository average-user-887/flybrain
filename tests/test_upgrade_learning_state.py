"""Checkpoint continuation uses real WP6/readout updates in isolated stores."""
import copy
import hashlib
import io
import json

import numpy as np
import pytest

from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
from experiment_registry import ExperimentRegistry, IncompatibleCheckpoint, SharedGraph


def registry(root):
    shared = SharedGraph.synthetic(allow_synthetic=True, n=64, k_out=6, seed=17)
    edges = np.flatnonzero(shared.arrays['weight'] < 0)[::7]
    rule = VisualHeadingPlasticityRule(edges, shared.arrays['weight'][edges],
                                      el_nodes=np.arange(8), test_only=True)
    return ExperimentRegistry(shared, root, test_mode=True, plasticity_rule=rule)


def trained(root, backend='connectome-plastic'):
    reg = registry(root)
    inst = reg.activate('t-maze', backend)
    if inst.readout is not None:
        inst.readout.rate = 0.375
    currents = np.full(reg.shared.n, 50.0, dtype=np.float32)
    for _ in range(12):
        inst.step(currents, 4.0, target=1)
    reg.checkpoint(world_state={'retained': True})
    return reg, inst, currents


def payload(inst):
    return copy.deepcopy(inst.meta()), {k: v.copy() for k, v in inst.state_arrays().items()}


def rewrite_checkpoint(reg, inst, mutate):
    directory = reg.instance_dir(inst.instance_id)
    pointer_path = directory / 'CURRENT.json'
    pointer = json.loads(pointer_path.read_text())
    path = directory / 'checkpoints' / pointer['file']
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['meta']))
        arrays = {k: archive[k] for k in archive.files if k != 'meta'}
    mutate(meta, arrays)
    buffer = io.BytesIO()
    np.savez(buffer, meta=np.array(json.dumps(meta)), **arrays)
    path.write_bytes(buffer.getvalue())
    pointer.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    pointer_path.write_text(json.dumps(pointer))


@pytest.mark.parametrize('backend', ['connectome-plastic', 'connectome-with-trained-readout'])
def test_learned_checkpoint_continues_exactly(tmp_path, backend):
    original, inst, currents = trained(tmp_path, backend)
    if inst.rule is not None:
        assert np.any(inst.plastic_delta != 0)
        assert np.any(inst.rule.pre_trace != 0) and inst.rule.mod_trace != 0
    else:
        assert np.any(inst.readout.weights != 0)
    requested = registry(tmp_path)
    restored = requested.activate('t-maze', backend)
    assert restored.manifest.run_id == inst.manifest.run_id
    assert restored.instance_id == inst.instance_id
    assert restored.world_state == {'retained': True}
    assert restored.learning_state_restore['status'] == 'exact'
    if inst.readout is not None:
        assert restored.readout.rate == 0.375
    for _ in range(3):
        left = inst.step(currents, 4.0, target=1)
        right = restored.step(currents, 4.0, target=1)
        assert (left.action, left.reward) == (right.action, right.reward)
        for key, value in inst.state_arrays().items():
            np.testing.assert_array_equal(restored.state_arrays()[key], value)
        assert restored.meta()['rng_state'] == inst.meta()['rng_state']
        assert restored.meta()['brain_scalars'] == inst.meta()['brain_scalars']


@pytest.mark.parametrize('backend', ['connectome-plastic', 'connectome-with-trained-readout',
                                     'connectome-fixed'])
def test_historical_omission_has_durable_reset_provenance(tmp_path, backend):
    original, inst, _ = trained(tmp_path, backend)
    def historical(meta, arrays):
        for key in ('learning_state_format', 'learning_state_restore', 'learning_state_resets'):
            meta.pop(key)
        if backend == 'connectome-plastic':
            arrays.pop('plastic.pre_trace')
            arrays.pop('plastic.mod_trace')
        elif backend == 'connectome-with-trained-readout':
            meta.pop('readout_rate')
    rewrite_checkpoint(original, inst, historical)
    requested = registry(tmp_path)
    # Nonzero stale traces must not leak into a historical restore.
    requested.plasticity_rule.pre_trace.fill(0.7)
    requested.plasticity_rule.mod_trace = 0.8
    restored = requested.activate('t-maze', backend)
    expected = [] if backend == 'connectome-fixed' else (
        ['plastic.pre_trace', 'plastic.mod_trace'] if backend == 'connectome-plastic'
        else ['readout_rate'])
    assert restored.learning_state_restore['reset_fields'] == expected
    assert restored.learning_state_restore['status'] == ('reset' if expected else 'exact')
    if restored.rule is not None:
        np.testing.assert_array_equal(restored.plastic_delta, inst.plastic_delta)
        assert not np.any(restored.rule.pre_trace) and restored.rule.mod_trace == 0
    if restored.readout is not None:
        np.testing.assert_array_equal(restored.readout.weights, inst.readout.weights)
        assert restored.readout.rate == 4.0
    events = [json.loads(line) for line in (requested.instance_dir(inst.instance_id) /
                                           'events.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'restore'
    assert events[-1]['learning_state'] == restored.learning_state_restore
    history = copy.deepcopy(restored.learning_state_resets)
    requested.checkpoint()
    again = registry(tmp_path).activate('t-maze', backend)
    assert again.learning_state_restore['status'] == 'exact'
    assert again.learning_state_resets == history


@pytest.mark.parametrize('name,bad', [
    ('plastic.pre_trace', np.zeros(1)),
    ('plastic.pre_trace', np.array(['oops'])),
    ('plastic.pre_trace', 'bool'),
    ('plastic.pre_trace', 'nan'),
    ('plastic.pre_trace', 'overflow'),
    ('plastic.mod_trace', np.array([0.5])),
    ('plastic.mod_trace', np.array(True)),
    ('plastic.mod_trace', np.array(float('inf'))),
    ('plastic.delta', np.zeros(1)),
    ('plastic.delta', 'nan'),
    ('plastic.pre_trace', None),
])
def test_malformed_plastic_learning_refuses_before_any_mutation(tmp_path, name, bad):
    _, inst, _ = trained(tmp_path)
    meta, arrays = payload(inst)
    before_meta, before_arrays = payload(inst)
    # Ensure an accidental brain restore is detectable as well.
    meta['brain_scalars']['sim_ms'] += 100
    if isinstance(bad, str):
        value = {'bool': True, 'nan': np.nan, 'overflow': 1e300}[bad]
        arrays[name] = np.full(arrays[name].shape, value)
    elif bad is None:
        arrays.pop(name)
    else:
        arrays[name] = bad
    with pytest.raises(IncompatibleCheckpoint):
        inst.load_payload(meta, arrays)
    assert inst.meta() == before_meta
    for key, value in before_arrays.items():
        np.testing.assert_array_equal(inst.state_arrays()[key], value)


@pytest.mark.parametrize('bad', [None, True, '0.375', float('nan'), float('inf'), 10 ** 400, [], {}])
def test_malformed_rate_refuses_before_mutation(tmp_path, bad):
    _, inst, _ = trained(tmp_path, 'connectome-with-trained-readout')
    meta, arrays = payload(inst)
    before, saved_arrays = payload(inst)
    meta['readout_rate'] = bad
    meta['brain_scalars']['sim_ms'] += 100
    with pytest.raises(IncompatibleCheckpoint, match='readout_rate'):
        inst.load_payload(meta, arrays)
    assert inst.meta() == before
    for key, value in saved_arrays.items():
        np.testing.assert_array_equal(inst.state_arrays()[key], value)


def test_partial_historical_traces_are_not_silently_reset(tmp_path):
    _, inst, _ = trained(tmp_path)
    meta, arrays = payload(inst)
    meta.pop('learning_state_format')
    arrays.pop('plastic.mod_trace')
    with pytest.raises(IncompatibleCheckpoint, match='plastic.mod_trace'):
        inst.load_payload(meta, arrays)


@pytest.mark.parametrize('marker', ['future', None, 1])
def test_unsupported_marker_refused(tmp_path, marker):
    _, inst, _ = trained(tmp_path)
    meta, arrays = payload(inst)
    meta['learning_state_format'] = marker
    with pytest.raises(IncompatibleCheckpoint, match='format'):
        inst.load_payload(meta, arrays)


@pytest.mark.parametrize('name,bad', [
    ('readout.weights', np.zeros(1)),
    ('readout.weights', 'bool'),
    ('readout.weights', 'nan'),
    ('readout.weights', 'string'),
    ('readout_rate', 'missing'),
])
def test_malformed_readout_payload_refuses_before_mutation(tmp_path, name, bad):
    _, inst, _ = trained(tmp_path, 'connectome-with-trained-readout')
    meta, arrays = payload(inst)
    before, saved_arrays = payload(inst)
    meta['brain_scalars']['sim_ms'] += 100
    if name == 'readout_rate':
        meta.pop(name)
    elif isinstance(bad, str):
        arrays[name] = np.full(arrays[name].shape,
                              {'bool': True, 'nan': np.nan, 'string': 'oops'}[bad])
    else:
        arrays[name] = bad
    with pytest.raises(IncompatibleCheckpoint):
        inst.load_payload(meta, arrays)
    assert inst.meta() == before
    for key, value in saved_arrays.items():
        np.testing.assert_array_equal(inst.state_arrays()[key], value)


def test_zero_traces_and_rate_are_preserved(tmp_path):
    _, plastic, _ = trained(tmp_path / 'plastic')
    meta, arrays = payload(plastic)
    arrays['plastic.pre_trace'].fill(0)
    arrays['plastic.mod_trace'] = np.array(0.0)
    plastic.load_payload(meta, arrays)
    assert not np.any(plastic.rule.pre_trace) and plastic.rule.mod_trace == 0
    assert plastic.learning_state_restore['status'] == 'exact'
    _, readout, _ = trained(tmp_path / 'readout', 'connectome-with-trained-readout')
    meta, arrays = payload(readout)
    meta['readout_rate'] = 0
    readout.load_payload(meta, arrays)
    assert readout.readout.rate == 0
    assert readout.learning_state_restore['status'] == 'exact'


def test_historical_rate_reset_uses_constructor_default_not_live_rate(tmp_path):
    _, inst, _ = trained(tmp_path, 'connectome-with-trained-readout')
    meta, arrays = payload(inst)
    meta.pop('learning_state_format')
    meta.pop('readout_rate')
    inst.readout.rate = 9.0
    inst.load_payload(meta, arrays)
    assert inst.readout.rate == 4.0
    assert inst.learning_state_restore['reset_fields'] == ['readout_rate']


def test_existing_historical_readout_rate_is_restored(tmp_path):
    original, inst, _ = trained(tmp_path, 'connectome-with-trained-readout')
    rewrite_checkpoint(original, inst, lambda meta, arrays: meta.pop('learning_state_format'))
    restored = registry(tmp_path).activate('t-maze', 'connectome-with-trained-readout')
    assert restored.readout.rate == 0.375
    assert restored.learning_state_restore['status'] == 'exact'


def test_fixed_backend_rejects_unowned_trace_fields(tmp_path):
    _, inst, _ = trained(tmp_path, 'connectome-fixed')
    meta, arrays = payload(inst)
    arrays['plastic.mod_trace'] = np.array(0.5)
    with pytest.raises(IncompatibleCheckpoint, match='unsupported'):
        inst.load_payload(meta, arrays)


@pytest.mark.parametrize('backend', ['connectome-plastic', 'connectome-with-trained-readout'])
@pytest.mark.parametrize('continuation', [False, True])
def test_malformed_checkpoint_preflight_preserves_active_and_store(tmp_path, monkeypatch,
                                                                  backend, continuation):
    original, target, _ = trained(tmp_path, backend)
    def corrupt(meta, arrays):
        if backend == 'connectome-plastic':
            arrays['plastic.pre_trace'] = np.zeros(1)
        else:
            meta['readout_rate'] = 'bad'
    rewrite_checkpoint(original, target, corrupt)
    if continuation:
        manifest = original.manifest(target.instance_id)
        manifest.graph_io = None
        manifest.write(original.instance_dir(target.instance_id) / 'manifest.json')
    requested = registry(tmp_path)
    requested.continue_io_state = continuation
    active = requested.activate('open-arena', 'connectome-fixed')
    # A legacy unselected index must also remain untouched on refusal.
    requested.index.pop('current_instances')
    requested._write_index()
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tmp_path.rglob('*') if p.is_file()}
    def no_brain_allocation(*args, **kwargs):
        pytest.fail('Read-only learning preflight must not allocate a brain')
    monkeypatch.setattr('experiment_registry.Brain', no_brain_allocation)
    with pytest.raises(IncompatibleCheckpoint):
        requested.activate('t-maze', backend)
    assert requested.active is active and active.brain is not None
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


def test_wp6_traces_belong_to_assay_and_survive_switching(tmp_path):
    reg, first, currents = trained(tmp_path)
    meta, arrays = payload(first)
    template = reg.plasticity_rule
    assert first.rule is not template
    assert not np.shares_memory(first.rule.pre_trace, template.pre_trace)
    assert not np.any(template.pre_trace) and template.mod_trace == 0
    second = reg.activate('open-arena', 'connectome-plastic')
    assert second.rule is not first.rule
    assert not np.shares_memory(second.rule.pre_trace, first.rule.pre_trace)
    assert not np.any(second.rule.pre_trace) and second.rule.mod_trace == 0
    for _ in range(4):
        second.step(currents, 4.0)
    np.testing.assert_array_equal(first.rule.pre_trace, arrays['plastic.pre_trace'])
    assert first.rule.mod_trace == float(arrays['plastic.mod_trace'])
    restored = reg.activate('t-maze', 'connectome-plastic')
    for key, value in arrays.items():
        np.testing.assert_array_equal(restored.state_arrays()[key], value)
    assert restored.meta()['rng_state'] == meta['rng_state']


def test_linked_child_retains_traces_without_mutating_active_parent(tmp_path):
    reg, parent, _ = trained(tmp_path)
    _, arrays = payload(parent)
    manifest = reg.manifest(parent.instance_id)
    manifest.graph_io = None
    manifest.write(reg.instance_dir(parent.instance_id) / 'manifest.json')
    rewrite_checkpoint(reg, parent, lambda meta, arrays: meta.update(graph_io=None))
    reg.continue_io_state = True
    child_id = reg._migrate_instance(parent.instance_id)
    assert child_id != parent.instance_id
    assert reg.active is parent and parent.brain is not None
    for key, value in arrays.items():
        np.testing.assert_array_equal(parent.state_arrays()[key], value)
    child = reg.activate('t-maze', 'connectome-plastic')
    for key, value in arrays.items():
        np.testing.assert_array_equal(child.state_arrays()[key], value)
