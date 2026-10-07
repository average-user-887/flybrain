"""Cohort persistence: atomic deterministic checkpoints, refusals, lock, v0.4 refusal.

SYNTHETIC TEST GRAPH with the CPU reference engine; no scientific claim.
"""
import json
import os

import numpy as np
import pytest

from brainlab.cohort import store
from brainlab.cohort.runner import CohortError, resume_cohort, run_cohort, synthetic_graph


def quiet(_):
    pass


def snapshot_tree(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name == store.LOCK_NAME:
                continue
            path = os.path.join(dirpath, name)
            with open(path, 'rb') as fh:
                out[os.path.relpath(path, root)] = fh.read()
    return out


@pytest.fixture
def cohort_dir(tmp_path):
    root = tmp_path / 'c'
    run_cohort(root, flies=3, seconds=0.2, seed_base=1, checkpoint_every_ms=100,
               graph=synthetic_graph(), progress=quiet)
    return root


def last_ckpt(root, fly):
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    return root / manifest['flies'][fly]['checkpoints'][-1]['file']


def test_checkpoint_encoding_is_deterministic_and_self_verifying():
    state = {'v': np.arange(4, dtype=np.float32), 'cursor': 3, 'total_spikes': 1, 'sim_ms': 0.3, 'dynamics': 'v3'}
    a = store.encode_checkpoint({'fly_id': 0, 'tick': 3}, state)
    b = store.encode_checkpoint({'fly_id': 0, 'tick': 3}, dict(state))
    assert a == b
    meta, back = store.decode_checkpoint(a)
    assert np.array_equal(back['v'], state['v']) and back['v'].dtype == np.float32
    assert {k: back[k] for k in ('cursor', 'total_spikes', 'sim_ms', 'dynamics')} == \
        {k: state[k] for k in ('cursor', 'total_spikes', 'sim_ms', 'dynamics')}


def test_atomic_write_leaves_no_temp(tmp_path):
    digest = store.atomic_write_bytes(tmp_path / 'x.bin', b'abc')
    assert (tmp_path / 'x.bin').read_bytes() == b'abc'
    assert digest == store.sha256_bytes(b'abc')
    assert [p.name for p in tmp_path.iterdir()] == ['x.bin']


@pytest.mark.parametrize('damage', ['corrupt', 'truncate', 'schema', 'graph'])
def test_damaged_checkpoint_is_refused_and_preserved(cohort_dir, damage, monkeypatch):
    path = last_ckpt(cohort_dir, 1)
    if damage == 'corrupt':
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(bytes(data))
    elif damage == 'truncate':
        path.write_bytes(path.read_bytes()[:-100])
    elif damage == 'schema':
        # A self-consistent checkpoint under another schema, chain sha updated too:
        # the schema check itself must refuse it.
        meta, state = store.decode_checkpoint(path.read_bytes())
        # Stored members, so the schema string can be edited in place (as rc1 archives are).
        import zipfile
        monkeypatch.setattr(store, 'CHECKPOINT_COMPRESSION', zipfile.ZIP_STORED)
        monkeypatch.setattr(store, 'CHECKPOINT_COMPRESSLEVEL', None)
        data = store.encode_checkpoint({k: v for k, v in meta.items() if k != 'content_sha256'}, state)
        data = data.replace(b'neurofly.cohort.v1', b'neurofly.cohort.v0')
        path.write_bytes(data)
        manifest = json.loads((cohort_dir / store.MANIFEST_NAME).read_text())
        manifest['flies'][1]['checkpoints'][-1]['sha256'] = store.sha256_bytes(data)
        (cohort_dir / store.MANIFEST_NAME).write_text(json.dumps(manifest))
    before = snapshot_tree(cohort_dir)
    if damage == 'graph':
        with pytest.raises(CohortError, match='graph_sha256 mismatch'):
            resume_cohort(cohort_dir, seconds=0.1, graph=synthetic_graph(seed=5), progress=quiet)
    else:
        with pytest.raises(store.CohortStoreError):
            resume_cohort(cohort_dir, seconds=0.1, graph=synthetic_graph(), progress=quiet)
    assert snapshot_tree(cohort_dir) == before


def test_failure_on_last_fly_leaves_every_fly_and_file_untouched(tmp_path):
    root = tmp_path / 'c8'
    run_cohort(root, flies=8, seconds=0.1, seed_base=1, graph=synthetic_graph(), progress=quiet)
    # Fly 7's final checkpoint carries a state that passes the file checks but
    # fails the engine's state validation (non-finite membrane).
    path = last_ckpt(root, 7)
    meta, state = store.decode_checkpoint(path.read_bytes())
    state['v'] = state['v'].copy()
    state['v'][0] = np.nan
    data = store.encode_checkpoint({k: v for k, v in meta.items() if k != 'content_sha256'}, state)
    path.write_bytes(data)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    manifest['flies'][7]['checkpoints'][-1]['sha256'] = store.sha256_bytes(data)
    (root / store.MANIFEST_NAME).write_text(json.dumps(manifest))
    before = snapshot_tree(root)
    with pytest.raises(CohortError, match='fly 7'):
        resume_cohort(root, seconds=0.1, graph=synthetic_graph(), progress=quiet)
    assert snapshot_tree(root) == before


def test_registry_and_saved_brain_paths_are_refused(tmp_path):
    reg = tmp_path / 'out' / 'registry-v3'
    reg.mkdir(parents=True)
    (reg / 'registry.json').write_text('{"format": "neurofly.experiment-registry.v2"}')
    for path in (reg, tmp_path / 'out', reg / 'optomotor'):
        with pytest.raises(store.CohortStoreError, match='never read'):
            run_cohort(path, flies=1, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    with pytest.raises(store.CohortStoreError, match='never read'):
        resume_cohort(reg, graph=synthetic_graph(), progress=quiet)
    brain = tmp_path / 'saved_brain.npz'
    brain.write_bytes(b'x')
    with pytest.raises(store.CohortStoreError):
        resume_cohort(brain, graph=synthetic_graph(), progress=quiet)
    assert (reg / 'registry.json').read_text().startswith('{"format"')
    assert brain.read_bytes() == b'x'


def test_lock_refuses_a_second_writer(cohort_dir):
    with store.CohortLock(cohort_dir):
        with pytest.raises(store.CohortLocked):
            store.CohortLock(cohort_dir).acquire()
        before = snapshot_tree(cohort_dir)
        with pytest.raises(store.CohortLocked):
            resume_cohort(cohort_dir, seconds=0.1, graph=synthetic_graph(), progress=quiet)
        assert snapshot_tree(cohort_dir) == before
    store.CohortLock(cohort_dir).acquire().release()   # free again afterwards


def test_run_refuses_non_empty_directory(cohort_dir):
    with pytest.raises(CohortError, match='not empty'):
        run_cohort(cohort_dir, flies=1, seconds=0.02, graph=synthetic_graph(), progress=quiet)


def test_cross_engine_resume_is_allowed_and_recorded(tmp_path, monkeypatch):
    """CPU checkpoint resumed on another engine (a CPU stand-in with a different
    backend_id, since tests run without a GPU): exact continuation, both engines
    recorded, every other check still applied."""
    from brainlab.cohort import runner
    from brainlab.cohort.api import CpuLoopCohortEngine

    class OtherEngine(CpuLoopCohortEngine):
        backend_id = 'stand-in-gpu'

    real = runner.make_engine
    monkeypatch.setattr(runner, 'make_engine',
                        lambda name, arrays, n: OtherEngine(arrays, n) if name == 'gpu' else real(name, arrays, n))
    full, split = tmp_path / 'full', tmp_path / 'split'
    run_cohort(full, flies=2, seconds=0.2, checkpoint_every_ms=100, graph=synthetic_graph(), progress=quiet)
    run_cohort(split, flies=2, seconds=0.1, checkpoint_every_ms=100, graph=synthetic_graph(), progress=quiet)
    resume_cohort(split, seconds=0.1, engine='gpu', graph=synthetic_graph(), progress=quiet)
    for k in range(2):
        rel = f'flies/fly-{k:02d}/steps.jsonl'
        assert (full / rel).read_bytes() == (split / rel).read_bytes()
        _, sa, _ = store.read_checkpoint(full / 'ckpt' / store.checkpoint_name(k, 2000))
        mb, sb, _ = store.read_checkpoint(split / 'ckpt' / store.checkpoint_name(k, 2000))
        assert mb['engine_backend_id'] == 'stand-in-gpu'
        assert all(np.array_equal(sa[x], sb[x]) if isinstance(sa[x], np.ndarray) else sa[x] == sb[x] for x in sa)
    manifest = json.loads((split / store.MANIFEST_NAME).read_text())
    seg = manifest['segments'][-1]
    assert (seg['source_engine'], seg['target_engine']) == ('cpu-loop-v3', 'stand-in-gpu')
    assert manifest['engine_backend_id'] == 'stand-in-gpu'
    # Back to CPU: the source is now the recorded stand-in, and the whole chain is verified again.
    resume_cohort(split, seconds=0.1, engine='cpu', graph=synthetic_graph(), progress=quiet)
    assert json.loads((split / store.MANIFEST_NAME).read_text())['segments'][-1]['source_engine'] == 'stand-in-gpu'


def test_checkpoint_engine_disagreeing_with_manifest_is_refused(cohort_dir):
    manifest = json.loads((cohort_dir / store.MANIFEST_NAME).read_text())
    manifest['engine_backend_id'] = 'some-gpu-engine'
    (cohort_dir / store.MANIFEST_NAME).write_text(json.dumps(manifest))
    before = snapshot_tree(cohort_dir)
    with pytest.raises(CohortError, match='checkpoint engine'):
        resume_cohort(cohort_dir, seconds=0.1, engine='cpu', graph=synthetic_graph(), progress=quiet)
    assert snapshot_tree(cohort_dir) == before


def test_resume_refused_when_running_dynamics_differ(cohort_dir, monkeypatch):
    """Same code, different model (TAU_M 20 -> 21 ms): refused, nothing touched."""
    from brainlab import engine as lif
    manifest = json.loads((cohort_dir / store.MANIFEST_NAME).read_text())
    meta, _, _ = store.read_checkpoint(cohort_dir / manifest['flies'][0]['checkpoints'][-1]['file'])
    assert meta['dynamics_signature'] == manifest['dynamics_signature']
    assert manifest['dynamics_signature']['values']['constants']['TAU_M_MS'] == 20.0
    before = snapshot_tree(cohort_dir)
    monkeypatch.setattr(lif, 'TAU_M_MS', 21.0)
    with pytest.raises(CohortError, match='TAU_M_MS: recorded 20.0, running 21.0'):
        resume_cohort(cohort_dir, seconds=0.1, graph=synthetic_graph(), progress=quiet)
    assert snapshot_tree(cohort_dir) == before
    monkeypatch.setattr(lif, 'TAU_M_MS', 20.0)
    resume_cohort(cohort_dir, seconds=0.1, graph=synthetic_graph(), progress=quiet)   # same model: accepted


def test_checkpoint_members_are_explicitly_compressed_and_rc1_archives_still_read():
    """A5: rc1 wrote every member ZIP_STORED (the ZipInfo default overrode the archive's
    deflate setting).  New members are deflated, deterministically; rc1 archives (stored)
    still verify against the sha256 their manifest recorded, so their hashes never change."""
    import io
    import zipfile
    from pathlib import Path
    data = store.encode_checkpoint({}, {'dynamics': 'v3', 'v': np.zeros(100000, np.float32)})
    assert data == store.encode_checkpoint({}, {'dynamics': 'v3', 'v': np.zeros(100000, np.float32)})
    infos = {i.filename: i for i in zipfile.ZipFile(io.BytesIO(data)).infolist()}
    assert {i.compress_type for i in infos.values()} == {zipfile.ZIP_DEFLATED}
    assert infos['state/v.npy'].compress_size < infos['state/v.npy'].file_size // 10
    assert np.array_equal(store.decode_checkpoint(data)[1]['v'], np.zeros(100000, np.float32))
    rc1 = Path(__file__).parent / 'fixtures' / 'cohort_rc1_store'
    manifest = json.loads((rc1 / store.MANIFEST_NAME).read_text())
    for entry in manifest['flies']:
        for link in entry['checkpoints']:
            path = rc1 / link['file']
            assert {i.compress_type for i in zipfile.ZipFile(path).infolist()} == {zipfile.ZIP_STORED}
            store.read_checkpoint(path, link['sha256'])
        store.verify_chain(rc1, entry)
