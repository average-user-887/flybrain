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


def test_resume_help_states_same_engine_exact_and_cross_engine_bounded(capsys):
    """The CLI must not promise exact CPU<->GPU continuation (only bounded agreement)."""
    with pytest.raises(SystemExit) as exc:
        main(['resume', '--help'])
    assert exc.value.code == 0
    text = ' '.join(capsys.readouterr().out.split())
    assert 'Resume on the same engine (CPU, or GPU on the same device) is byte-identical' in text
    assert 'CPU<->GPU continuation is NOT exact' in text
    assert 'g relative error <= 1e-6 (preregistered), 0 spike mismatches over ticks 100-199' in text
    assert 'observed max g relative error about 4.53e-7' in text
    assert '<= 4.5e-7' not in text
    with pytest.raises(SystemExit):
        main(['--help'])
    listing = ' '.join(capsys.readouterr().out.split())
    assert 'same-engine resume is byte-identical' in listing
    assert 'CPU<->GPU continuation is NOT exact' in listing
    assert 'cohort directory exactly' not in listing


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
    # The cohort's own I/O, referencing the daemon's declaration by version + sha256 only.
    assert decl['graph_io']['daemon_graph_io'] == provenance.graph_io_declaration()
    assert decl['graph_io']['daemon_graph_io']['sha256'] == provenance.GRAPH_IO_SHA256
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


# ---------------------------------------------------------------------------
# rc2 Lane 1 repairs (each test fails on v0.5.0rc1 a4368a2)
# ---------------------------------------------------------------------------
import hashlib
import os
import shutil
import sys
from pathlib import Path

from brainlab.cohort import runner as runner_mod
from brainlab.cohort.api import CpuLoopCohortEngine

RC1_STORE = Path(__file__).parent / 'fixtures' / 'cohort_rc1_store'


class _StandInGpu(CpuLoopCohortEngine):
    """CPU arithmetic under the GPU engine's exact backend id (tests run without a GPU)."""
    backend_id = 'cuda-cohort-v3'


def _tree(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name == store.LOCK_NAME:
                continue
            path = os.path.join(dirpath, name)
            with open(path, 'rb') as fh:
                out[os.path.relpath(path, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _gpu_cohort(tmp_path, monkeypatch, name='g'):
    real = runner_mod.make_engine
    monkeypatch.setattr(runner_mod, 'make_engine',
                        lambda n, arrays, k: _StandInGpu(arrays, k) if n == 'gpu' else real(n, arrays, k))
    root = tmp_path / name
    run_cohort(root, flies=2, seconds=0.04, engine='gpu', checkpoint_every_ms=20, graph=synthetic_graph(),
               progress=quiet)
    monkeypatch.setattr(runner_mod, 'make_engine', real)
    return root


def test_default_resume_uses_the_recorded_gpu_backend(tmp_path, monkeypatch):
    """F1: rc1 asked make_engine for 'cpu' because 'gpu' is not a substring of 'cuda-cohort-v3'."""
    root = _gpu_cohort(tmp_path, monkeypatch)
    asked = []
    monkeypatch.setattr(runner_mod, 'make_engine',
                        lambda n, arrays, k: asked.append(n) or _StandInGpu(arrays, k))
    resume_cohort(root, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    assert asked == ['gpu']
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    seg = manifest['segments'][-1]
    assert seg['source_engine'] == seg['target_engine'] == 'cuda-cohort-v3'
    assert seg['engine_choice'] == 'recorded' and seg['mixed_engine'] is False
    assert 'mixed_engines' not in manifest


def test_default_resume_refuses_when_recorded_gpu_is_unavailable(tmp_path, monkeypatch):
    """F1: no silent GPU->CPU fallback; refused before any persistent change."""
    root = _gpu_cohort(tmp_path, monkeypatch)
    # An interrupted tail must not be cut off by a refused resume either.
    with open(root / 'flies/fly-00/steps.jsonl', 'ab') as fh:
        fh.write(b'{"interrupted": true}\n')
    before = _tree(root)

    def unavailable(n, arrays, k):
        if n == 'gpu':
            raise runner_mod.EngineUnavailable('--engine gpu is unavailable on this machine (no CuPy)')
        return CpuLoopCohortEngine(arrays, k)
    monkeypatch.setattr(runner_mod, 'make_engine', unavailable)
    with pytest.raises(EngineUnavailable, match='cuda-cohort-v3 and that engine is unavailable'):
        resume_cohort(root, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    assert _tree(root) == before
    assert main(['resume', str(root), '--test-synthetic-graph']) == 2
    assert _tree(root) == before


def test_unknown_recorded_backend_is_refused_without_explicit_engine(tmp_path, monkeypatch):
    real = runner_mod.make_engine

    class Odd(CpuLoopCohortEngine):
        backend_id = 'some-gpu-engine'
    monkeypatch.setattr(runner_mod, 'make_engine',
                        lambda n, arrays, k: Odd(arrays, k) if n == 'gpu' else real(n, arrays, k))
    root = tmp_path / 'odd'
    run_cohort(root, flies=1, seconds=0.02, engine='gpu', graph=synthetic_graph(), progress=quiet)
    before = _tree(root)
    with pytest.raises(CohortError, match='does not know'):
        resume_cohort(root, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    assert _tree(root) == before


def test_explicit_engine_switch_is_a_marked_mixed_engine_realisation(tmp_path, monkeypatch):
    """F1: explicit --engine stays allowed; the GPU segment's hardware record survives."""
    root = _gpu_cohort(tmp_path, monkeypatch)
    gpu_description = json.loads((root / store.MANIFEST_NAME).read_text())['engine']
    resume_cohort(root, seconds=0.02, engine='cpu', graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    assert manifest['mixed_engines'] is True
    assert 'Mixed-engine numerical realisation' in manifest['numerical_realisation']
    seg = manifest['segments'][-1]
    assert (seg['source_engine'], seg['target_engine']) == ('cuda-cohort-v3', 'cpu-loop-v3')
    assert seg['engine_choice'] == 'explicit' and seg['mixed_engine'] is True
    assert seg['source_engine_description'] == gpu_description
    assert manifest['segments'][0]['engine_description'] == gpu_description
    # The default now follows the latest recorded engine (cpu), and the mark persists.
    resume_cohort(root, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    assert manifest['mixed_engines'] is True and manifest['segments'][-1]['mixed_engine'] is False


def test_verify_without_cupy_prints_a_clear_failure(monkeypatch, capsys):
    """F3: rc1 raised an uncaught RuntimeError from contract.gpu_factory."""
    monkeypatch.setitem(sys.modules, 'cupy', None)
    assert main(['verify']) == 1
    out = capsys.readouterr().out
    assert out.startswith('FAIL  engine-available:') and 'Traceback' not in out
    assert 'verify --engine cpu' in out


def test_manifest_declares_only_the_cohort_effective_io(tmp_path):
    """F4: rc1 declared the daemon's 9 probes and 6 decoders."""
    import provenance
    root = tmp_path / 'io'
    run_cohort(root, flies=1, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    decl = manifest['scientific_disclosure']['graph_io']
    assert [p['name'] for p in decl['input_probes']] == ['optomotor']
    assert decl['decoders'] == ['yaw']
    assert decl['encoder']['model'].startswith('rectified-sinusoid drive on T4/T5')
    assert decl['decoder']['model'].startswith('yaw = gain*(rate_DNa02_L')
    assert decl['tethered'] == {'forward_speed': 0.0} and decl['step_ms'] == 20.0
    assert decl['io_map_sha256'] == manifest['io_map_sha256']
    assert decl['daemon_graph_io'] == provenance.graph_io_declaration()
    assert 'config' not in decl['daemon_graph_io']
    body = {k: v for k, v in decl.items() if k != 'sha256'}
    assert decl['sha256'] == runner_mod._sha256_json(body)


@pytest.mark.parametrize('step_ms', [20.05, 0.05, 0.0, -20.0, float('nan'), float('inf')])
def test_step_ms_off_the_tick_grid_is_refused_before_any_directory(tmp_path, step_ms):
    """A3: rc1 rounded the brain to 200 ticks while the arena/decoder used 20.05 ms."""
    root = tmp_path / 'never'
    with pytest.raises(CohortError):
        run_cohort(root, flies=1, seconds=0.2005 if step_ms == 20.05 else 0.2, step_ms=step_ms,
                   graph=synthetic_graph(), progress=quiet)
    assert not root.exists()


def test_non_finite_seconds_and_checkpoint_interval_refused(tmp_path):
    for kw in ({'seconds': float('nan')}, {'seconds': float('inf')},
               {'seconds': 0.2, 'checkpoint_every_ms': float('nan')}):
        with pytest.raises(CohortError):
            run_cohort(tmp_path / 'x', flies=1, graph=synthetic_graph(), progress=quiet, **kw)
        assert not (tmp_path / 'x').exists()


def test_rc1_store_still_resumes_and_the_original_is_untouched(tmp_path):
    """A store written by v0.5.0rc1 (a4368a2) resumes exactly; the committed original never changes."""
    assert (RC1_STORE / store.MANIFEST_NAME).is_file()
    before = _tree(RC1_STORE)
    work = tmp_path / 'rc1'
    shutil.copytree(RC1_STORE, work)
    resume_cohort(work, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    assert _tree(RC1_STORE) == before
    fresh = tmp_path / 'fresh'
    run_cohort(fresh, flies=2, seconds=0.06, checkpoint_every_ms=20, graph=synthetic_graph(), progress=quiet)
    for k in range(2):
        rel = f'flies/fly-{k:02d}/steps.jsonl'
        assert (work / rel).read_bytes() == (fresh / rel).read_bytes()
        _, sa, _ = store.read_checkpoint(fresh / 'ckpt' / store.checkpoint_name(k, 600))
        _, sb, _ = store.read_checkpoint(work / 'ckpt' / store.checkpoint_name(k, 600))
        assert all(np.array_equal(sa[x], sb[x]) if isinstance(sa[x], np.ndarray) else sa[x] == sb[x] for x in sa)
    manifest = json.loads((work / store.MANIFEST_NAME).read_text())
    seg = manifest['segments'][-1]
    assert seg['engine_choice'] == 'recorded' and seg['source_engine'] == 'cpu-loop-v3'
    assert seg['source_engine_description']['backend'] == 'cpu-loop-v3'
    # Its recorded rc1 disclosure is kept as written (flagged: see the F4 resume test).
    original = json.loads((RC1_STORE / store.MANIFEST_NAME).read_text())
    assert manifest['scientific_disclosure']['graph_io'] == original['scientific_disclosure']['graph_io']
    assert manifest['scientific_disclosure']['notice'] == original['scientific_disclosure']['notice']


def test_resumed_rc1_store_declares_effective_io_and_flags_the_old_declaration(tmp_path):
    """F4 for old stores: the rc1 manifest's daemon-wide declaration (9 probes, 6 decoders) is kept
    unchanged but flagged as superseded; the resumed segment declares the cohort's actual I/O."""
    import provenance
    original = json.loads((RC1_STORE / store.MANIFEST_NAME).read_text())
    old = original['scientific_disclosure']['graph_io']
    assert len(old['config']['input_probes']) == 9 and len(old['config']['decoders']) == 6
    work = tmp_path / 'rc1'
    shutil.copytree(RC1_STORE, work)
    resume_cohort(work, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((work / store.MANIFEST_NAME).read_text())
    disclosure = manifest['scientific_disclosure']
    assert disclosure['graph_io'] == old                                   # historical, unchanged
    assert disclosure['graph_io_status'].startswith('superseded: declared by the v0.5.0rc1 cohort writer')
    for effective in (disclosure['effective_graph_io'], manifest['segments'][-1]['effective_graph_io']):
        assert effective['scope'] == 'cohort'
        assert [p['name'] for p in effective['input_probes']] == ['optomotor'] and effective['decoders'] == ['yaw']
        assert effective['daemon_graph_io'] == provenance.graph_io_declaration()
    # A second resume keeps the flag and the old declaration; it does not stack another one.
    resume_cohort(work, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    again = json.loads((work / store.MANIFEST_NAME).read_text())['scientific_disclosure']
    assert again == disclosure


def test_resumed_rc2_store_keeps_its_cohort_declaration_unflagged(tmp_path):
    root = tmp_path / 'rc2'
    run_cohort(root, flies=1, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    before = json.loads((root / store.MANIFEST_NAME).read_text())['scientific_disclosure']
    resume_cohort(root, seconds=0.02, graph=synthetic_graph(), progress=quiet)
    manifest = json.loads((root / store.MANIFEST_NAME).read_text())
    assert manifest['scientific_disclosure'] == before
    assert manifest['segments'][-1]['effective_graph_io'] == before['graph_io']
