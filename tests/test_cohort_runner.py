"""Cohort runner: exact resume, per-fly independence, Arena loop, CLI.

SYNTHETIC TEST GRAPH with the CPU reference engine; no scientific claim.
"""
import json

import numpy as np
import pytest

from brainlab.cohort import store
from brainlab.cohort.runner import (CohortError, EngineUnavailable, FlyWorld, main, make_engine,
                                    resume_cohort, run_cohort, synthetic_graph)


def quiet(_):
    pass


def outputs(root, n):
    files = {f'flies/fly-{k:02d}/steps.jsonl': (root / f'flies/fly-{k:02d}/steps.jsonl').read_bytes()
             for k in range(n)}
    files.update({p.relative_to(root).as_posix(): p.read_bytes() for p in sorted((root / 'ckpt').iterdir())})
    return files


def records(root, k):
    return [json.loads(line) for line in (root / f'flies/fly-{k:02d}/steps.jsonl').read_text().splitlines()]


def test_resume_equals_uninterrupted_byte_for_byte(tmp_path):
    full, split, crash = tmp_path / 'full', tmp_path / 'split', tmp_path / 'crash'
    kw = dict(flies=3, seed_base=4, checkpoint_every_ms=100, progress=quiet)
    run_cohort(full, seconds=0.4, graph=synthetic_graph(), **kw)
    # Planned split: 0.2 s, then resume for 0.2 s more.
    run_cohort(split, seconds=0.2, graph=synthetic_graph(), **kw)
    resume_cohort(split, seconds=0.2, graph=synthetic_graph(), progress=quiet)
    # Crash after 13 steps (last checkpoint at step 10), then resume to the target.
    with pytest.raises(KeyboardInterrupt):
        run_cohort(crash, seconds=0.4, graph=synthetic_graph(), stop_after_steps=13, **kw)
    resume_cohort(crash, graph=synthetic_graph(), progress=quiet)
    want = outputs(full, 3)
    assert outputs(split, 3) == want
    got = outputs(crash, 3)
    assert got == want
    orphans = list((crash / 'flies/fly-00').glob('*.orphan.jsonl'))
    assert len(orphans) == 1 and len(orphans[0].read_text().splitlines()) == 13
    # The closed loop actually ran: the brain's yaw moved the fly and changed the slip.
    recs = records(full, 0) + records(full, 1) + records(full, 2)
    assert any(r['motor']['yaw_rad_s'] != 0.0 for r in recs)
    assert len({r['slip_rad_s'] for r in recs}) > 1


def test_resume_state_matches_uninterrupted_engine_state(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    run_cohort(a, flies=2, seconds=0.2, checkpoint_every_ms=100, graph=synthetic_graph(), progress=quiet)
    run_cohort(b, flies=2, seconds=0.1, checkpoint_every_ms=100, graph=synthetic_graph(), progress=quiet)
    resume_cohort(b, seconds=0.1, graph=synthetic_graph(), progress=quiet)
    for k in range(2):
        name = store.checkpoint_name(k, 2000)
        ma, sa, _ = store.read_checkpoint(a / 'ckpt' / name)
        mb, sb, _ = store.read_checkpoint(b / 'ckpt' / name)
        assert ma == mb
        for key in sa:
            assert np.array_equal(sa[key], sb[key]) if isinstance(sa[key], np.ndarray) else sa[key] == sb[key]


def test_flies_are_independent_of_cohort_size_and_position(tmp_path):
    """Fly with seed s gives identical outputs alone or inside a larger cohort."""
    big, solo = tmp_path / 'big', tmp_path / 'solo'
    run_cohort(big, flies=4, seconds=0.2, seed_base=10, graph=synthetic_graph(), progress=quiet)
    run_cohort(solo, flies=1, seconds=0.2, seed_base=12, graph=synthetic_graph(), progress=quiet)
    assert records(big, 2) == records(solo, 0)
    # Each fly has its own encoder RNG and input stream.
    drives = [tuple(r['encoder_totals']['ftb_L'] for r in records(big, k)) for k in range(4)]
    assert len(set(drives)) == 4
    manifest = json.loads((big / store.MANIFEST_NAME).read_text())
    assert [f['seed'] for f in manifest['flies']] == [10, 11, 12, 13]


def test_fly_worlds_do_not_share_rng_or_arena():
    g = synthetic_graph()
    w0, w1 = FlyWorld(0, 1, g, 'optomotor', 20.0), FlyWorld(1, 2, g, 'optomotor', 20.0)
    assert w0.rng is not w1.rng and w0.arena is not w1.arena and w0.decoder is not w1.decoder
    w0.rng.standard_normal(10)
    fresh = FlyWorld(1, 2, g, 'optomotor', 20.0)
    assert w1.rng.bit_generator.state == fresh.rng.bit_generator.state


def test_manifest_and_checkpoint_fields(tmp_path):
    root = tmp_path / 'm'
    run_cohort(root, flies=2, seconds=0.1, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    assert manifest['schema'] == 'neurofly.cohort.v1' and manifest['dynamics'] == 'v3_fixed'
    meta, state, _ = store.read_checkpoint(root / manifest['flies'][1]['checkpoints'][-1]['file'])
    for key in ('graph_sha256', 'io_map_sha256', 'dynamics', 'engine_backend_id', 'payload', 'fly_id', 'seed',
                'rng', 'input_cursor', 'stimulus_schedule_sha256', 'assay', 'world_state', 'tick', 'sim_ms',
                'parent_checkpoint_sha256'):
        assert key in meta, key
    assert meta['seed'] == 2 and meta['input_cursor'] == 5 and meta['tick'] == 1000
    assert meta['parent_checkpoint_sha256'] == manifest['flies'][1]['checkpoints'][0]['sha256']
    assert state['dynamics'] == 'v3' and state['sim_ms'] == pytest.approx(100.0)


def test_unknown_assay_and_gpu_unavailable_fail_clearly(tmp_path, monkeypatch):
    with pytest.raises(CohortError, match='not supported'):
        run_cohort(tmp_path / 'x', flies=1, seconds=0.02, assay='t_maze', graph=synthetic_graph(), progress=quiet)
    import sys
    monkeypatch.setitem(sys.modules, 'brainlab.cohort.gpu', None)
    with pytest.raises(EngineUnavailable, match='--engine gpu is unavailable'):
        make_engine('gpu', synthetic_graph().arrays, 1)


def test_cli_run_and_resume(tmp_path, capsys):
    out = tmp_path / 'cli'
    assert main(['run', '--flies', '2', '--seconds', '0.04', '--out', str(out), '--test-synthetic-graph']) == 0
    text = capsys.readouterr().out
    assert 'x real time' in text and 'SYNTHETIC' in text
    assert main(['resume', str(out), '--seconds', '0.02', '--test-synthetic-graph']) == 0
    assert main(['resume', str(out)]) == 2          # synthetic needs the explicit test option
    assert 'SYNTHETIC TEST GRAPH' in capsys.readouterr().err


def test_neurofly_entry_point_dispatches_cohort(tmp_path):
    from neurofly.cli import main as neurofly_main
    out = tmp_path / 'ep'
    assert neurofly_main(['cohort', 'run', '--flies', '1', '--seconds', '0.02', '--out', str(out),
                          '--test-synthetic-graph']) == 0
    assert (out / store.MANIFEST_NAME).is_file()


def test_manifest_carries_graph_io_declaration_and_notice(tmp_path, capsys):
    import provenance
    out = tmp_path / 'd'
    assert main(['run', '--flies', '1', '--seconds', '0.02', '--out', str(out), '--test-synthetic-graph']) == 0
    assert 'injects motion-selective drive directly into T4/T5' in capsys.readouterr().out
    manifest = json.loads((out / store.MANIFEST_NAME).read_text())
    decl = manifest['scientific_disclosure']
    assert decl['graph_io'] == provenance.graph_io_declaration(include_config=True)
    assert decl['graph_io']['sha256'] == provenance.GRAPH_IO_SHA256
    assert 'engineered' in decl['notice'] and 'No claim of validated' in decl['notice']


def test_writer_identity_is_the_executing_code_and_resume_ignores_code_version(tmp_path, monkeypatch):
    from brainlab.cohort import runner
    root = tmp_path / 'w'
    run_cohort(root, flies=2, seconds=0.04, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    first = manifest['segments'][0]['writer']
    assert first['role'] == 'executing_writer'
    assert 'brainlab.cohort.runner' in first['loaded_module_sha256']
    meta, _, _ = store.read_checkpoint(root / manifest['flies'][0]['checkpoints'][-1]['file'])
    assert meta['writer'] == first and meta['payload']['version'] == first['version']
    # The code "advances": a compatible checkpoint is still resumed, both writers recorded.
    newer = dict(first, version='9.9.9', commit='0' * 40, loaded_module_sha256={'x': 'y'})
    monkeypatch.setattr(runner, 'writer_identity', lambda: dict(newer))
    resume_cohort(root, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    seg = manifest['segments'][-1]
    assert seg['source_writer'] == first and seg['target_writer'] == newer
    assert manifest['segments'][0]['writer'] == first          # earlier segment untouched
    meta, _, _ = store.read_checkpoint(root / manifest['flies'][0]['checkpoints'][-1]['file'])
    assert meta['writer'] == newer and meta['payload']['version'] == '9.9.9'
