"""graph-arena-io-v3 (sensory-delivery repair): a saved v2 lineage never changes silently.

The v3 declaration restores the multisensory wind input that v2 dropped.  A store saved
under v2 must refuse an ordinary resume, and the existing ``--continue-io-state`` route
must create a linked child that carries the v3 declaration with the parent's verified
brain state copied unchanged.  The parent store is never modified.  Fixture hashes only.
"""
import json

import pytest

import provenance
from brainlab.runtime_backend import UnsupportedRuntimeCapability, require_fixed_amd
from experiment_registry import ExperimentRegistry, IOContinuationRequired
from tests.test_graph_io_migration import (_assert_same_payload, _drive, _hash_tree, _registry,
                                           _rewrite_checkpoint_graph_io)

# The released v0.5.0 (33242c7) declaration, as recorded by every saved v2 store.
V2_GRAPH_IO = {'version': 'graph-arena-io-v2-unassisted',
               'sha256': '23d421f937ed485e5cf8d70835c2a50d56b8f88be217e4e66e5ad3909eef6191'}
# Pinned so a later edit of the hashed mapping cannot pass silently.
V3_GRAPH_IO = {'version': 'graph-arena-io-v3-unassisted',
               'sha256': '220a31c25c6c236957bb0fbab316111223c978d8d810ecef88910ca8c838fedb'}
ASSAY, BACKEND = 'multisensory-sandbox', 'connectome-fixed'


def test_v3_declaration_states_wind_keys_units_precedence_and_invalid_input():
    declared = provenance.graph_io_declaration(include_config=True)
    assert {k: declared[k] for k in ('version', 'sha256')} == V3_GRAPH_IO
    assert declared['sha256'] != V2_GRAPH_IO['sha256']
    wind = next(p for p in declared['config']['input_probes'] if p['name'] == 'jon_wind')
    assert wind['accepted_keys'] == ['wind_speed', 'wind_magnitude']
    assert wind['units'].startswith('mm/s')
    assert 'explicit 0' in wind['precedence'] and 'one injection' in wind['precedence']
    assert 'NOT DELIVERED' in wind['invalid_input'] and 'non-finite' in wind['invalid_input']
    assert 'NOT DELIVERED' in wind['vector_only_input']
    assert declared['config']['supersedes']['version'] == V2_GRAPH_IO['version']


def _v2_parent(root):
    """A multisensory connectome-fixed store whose manifest and checkpoints record v2."""
    registry = _registry(root)
    parent = registry.activate(ASSAY, BACKEND)
    parent.world_state = {'pose': [45.0, 45.0, 0.5], 'arena_rng': {'counter': 3}}
    _drive(parent, 3)
    registry.checkpoint()
    parent_id = parent.instance_id
    parent_dir = registry.instance_dir(parent_id)
    manifest = registry.manifest(parent_id)
    manifest.graph_io = dict(V2_GRAPH_IO)
    manifest.write(parent_dir / 'manifest.json')
    for checkpoint in sorted((parent_dir / 'checkpoints').glob('ckpt-*.npz')):
        _rewrite_checkpoint_graph_io(parent_dir, int(checkpoint.stem.split('-')[1]), dict(V2_GRAPH_IO))
    return registry.shared, parent_id, manifest.run_id, parent_dir


def test_ordinary_resume_of_v2_lineage_refuses_before_any_write(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, _, parent_dir = _v2_parent(root)
    before = _hash_tree(parent_dir)
    index_before = (root / 'index.json').read_bytes() if (root / 'index.json').exists() else None
    with pytest.raises(IOContinuationRequired) as refused:
        ExperimentRegistry(shared, root, test_mode=True).activate(ASSAY, BACKEND)
    message = str(refused.value)
    for text in (parent_id, V2_GRAPH_IO['version'], V2_GRAPH_IO['sha256'],
                 V3_GRAPH_IO['version'], V3_GRAPH_IO['sha256'], '--continue-io-state'):
        assert text in message
    assert _hash_tree(parent_dir) == before
    if index_before is not None:
        assert (root / 'index.json').read_bytes() == index_before


def test_explicit_continuation_links_child_with_v3_and_unchanged_state(tmp_path):
    root = tmp_path / 'registry'
    shared, parent_id, parent_run, parent_dir = _v2_parent(root)
    before = _hash_tree(parent_dir)
    migrating = ExperimentRegistry(shared, root, test_mode=True, continue_io_state=True)
    parent_meta, parent_arrays, _ = migrating._select_parent_checkpoint(parent_id)
    child = migrating.activate(ASSAY, BACKEND)

    assert child.instance_id != parent_id and child.manifest.run_id != parent_run
    assert child.manifest.parent_run_id == parent_run
    _assert_same_payload(parent_meta, parent_arrays, child)      # brain state copied unchanged
    lineage = child.manifest.lineage
    assert lineage['kind'] == 'graph_io_state_continuation'
    assert lineage['prior_graph_io'] == V2_GRAPH_IO
    assert lineage['new_graph_io'] == V3_GRAPH_IO
    assert child.manifest.graph_io == V3_GRAPH_IO
    assert lineage['parent_checkpoint']['graph_io'] == V2_GRAPH_IO
    assert lineage['parent_checkpoint']['sha256']
    assert _hash_tree(parent_dir) == before                        # parent store untouched

    # A later ordinary restart resumes the v3 child, never the v2 parent.
    restarted = ExperimentRegistry(shared, root, test_mode=True)
    assert restarted.activate(ASSAY, BACKEND).instance_id == child.instance_id
    assert restarted.instance_id_for_run(parent_run) == parent_id
    assert _hash_tree(parent_dir) == before
    json.dumps(lineage)                                            # serialisable as recorded


def test_amd_keeps_refusing_explicit_io_continuation():
    """Known limitation, unchanged: AMD cannot carry a v2 lineage into v3."""
    with pytest.raises(UnsupportedRuntimeCapability, match='graph-I/O state migration'):
        require_fixed_amd('wgpu-amd', 'connectome-fixed', continue_io_state=True)
