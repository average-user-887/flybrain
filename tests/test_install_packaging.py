"""Clean-room install findings (Deck and NVIDIA reports, 5 Oct 2026) held as tests.

Default backend choice, version reporting, Docker packaging and the GPU probe that
must not mistake a broken CUDA build for a usable one.
"""
import logging
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import neurofly
import neurofly_daemon as nd
from brainlab import gpu_probe

ROOT = Path(nd.__file__).resolve().parent
PYPROJECT = tomllib.loads((ROOT / 'pyproject.toml').read_text())


# --------------------------------------------------------------- default backend

def _args(**kw):
    base = dict(backend=None, test_synthetic_graph=False, graph_dir=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_explicit_backend_always_wins(tmp_path):
    backend, reason = nd.choose_default_backend(_args(backend='connectome-fixed', graph_dir=str(tmp_path)))
    assert backend == 'connectome-fixed' and 'requested' in reason


def test_no_verified_graph_means_modular_with_next_steps(tmp_path):
    backend, reason = nd.choose_default_backend(_args(graph_dir=str(tmp_path)))
    assert backend == 'modular'
    assert 'no verified MaleCNS graph' in reason and 'neurofly download-data' in reason
    assert 'brainlab.prepare' in reason


def test_synthetic_test_graph_implies_graph_backend(tmp_path):
    assert nd.choose_default_backend(_args(test_synthetic_graph=True))[0] == 'connectome-fixed'


def test_verified_graph_means_connectome(monkeypatch):
    from brainlab.graph_identity import resolve_connectome_dir, resolve_graph_dir
    if not (resolve_graph_dir()[0] / 'graph.npz').is_file() or not resolve_connectome_dir()[0].is_dir():
        pytest.skip('real MaleCNS graph not available')
    backend, reason = nd.choose_default_backend(_args())
    assert backend == 'connectome-fixed' and 'verified' in reason


def test_parser_leaves_backend_unset_without_env(monkeypatch):
    monkeypatch.delenv('NEUROFLY_BACKEND', raising=False)
    assert nd.build_arg_parser().parse_args([]).backend is None


# ----------------------------------------------------------------------- version

def test_package_version_comes_from_pyproject():
    assert neurofly.__version__ == PYPROJECT['project']['version']


def test_dashboard_fallback_badge_matches_package_version():
    html = (ROOT / 'web/index.html').read_text()
    badge = re.search(r'id="appVersionBadge"[^>]*>([^<]+)<', html)
    assert badge, 'version badge missing'
    assert badge.group(1) == f"v{PYPROJECT['project']['version']}", (
        'web/index.html shows a different version than pyproject.toml; update both together')
    assert 'v1.0-release' not in html


def test_status_payload_reports_package_version(tmp_path):
    from stream_gateway import StreamGateway, StreamPolicy
    runner = nd.ContinuousExperimentRunner(initial_paradigm='wind-tunnel', sim_speed=1.0,
                                           checkpoint_interval=3600, output_dir=tmp_path)
    handler = object.__new__(nd.NeuroflyHTTPHandler)
    handler.runner = runner
    handler.gateway = StreamGateway(StreamPolicy())
    assert handler._status_payload()['version'] == PYPROJECT['project']['version']


# ------------------------------------------------------------------- packaging

def test_docker_image_copies_every_declared_package():
    dockerfile = (ROOT / 'Dockerfile').read_text()
    packages = PYPROJECT['tool']['setuptools']['packages']
    for top in sorted({p.split('.')[0] for p in packages}):
        assert re.search(rf'^COPY {top}/ \./{top}/$', dockerfile, re.M), f'Dockerfile does not copy {top}/'
    # download-data reads the source lock; capability prints the matrix.
    assert 'COPY data-provenance/' in dockerfile and 'CAPABILITY_MATRIX.md' in dockerfile


def test_extras_cover_gpu_and_browser_tests_without_numba_cuda():
    extras = PYPROJECT['project']['optional-dependencies']
    assert any(dep.startswith('cupy-cuda12x[ctk]') for dep in extras['gpu'])
    assert not any('numba-cuda' in dep for deps in extras.values() for dep in deps)
    assert any(dep.startswith('selenium') for dep in extras['browser-test'])


# ------------------------------------------------------------------- GPU probe

@pytest.fixture
def fresh_probe():
    gpu_probe.reset()
    yield gpu_probe
    gpu_probe.reset()


def test_broken_gpu_build_is_reported_unusable_with_reason(fresh_probe, caplog):
    def broken():
        raise AttributeError("module 'numpy' has no attribute 'row_stack'")
    with caplog.at_level(logging.WARNING, logger='brainlab'):
        ok, reason = fresh_probe._run('numba.cuda', broken)
    assert ok is False and 'row_stack' in reason
    assert any('unusable' in r.getMessage() for r in caplog.records)


def test_absent_gpu_library_is_quiet(fresh_probe, caplog):
    def absent():
        raise fresh_probe._Absent('CuPy not installed')
    with caplog.at_level(logging.WARNING, logger='brainlab'):
        ok, reason = fresh_probe._run('CuPy', absent)
    assert ok is False and reason == 'CuPy not installed' and not caplog.records


def test_probe_results_are_cached(fresh_probe):
    calls = []
    fresh_probe._run('x', lambda: calls.append(1) or 'ran')
    fresh_probe._run('x', lambda: calls.append(1) or 'ran')
    assert calls == [1]


def test_forced_cpu_is_explained_without_probing(fresh_probe, monkeypatch):
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'cpu')
    monkeypatch.setattr(fresh_probe, 'describe', lambda d: pytest.fail('probed despite forced backend'))
    assert fresh_probe.explain('v3') == ('cpu', 'forced by NEUROFLY_BRAIN_BACKEND=cpu')


def test_auto_brain_falls_back_to_cpu_when_gpu_setup_fails(monkeypatch, caplog):
    import brainlab.brain as brain_mod
    from brainlab.graph_identity import synthetic_test_graph
    monkeypatch.delenv('NEUROFLY_BRAIN_BACKEND', raising=False)
    monkeypatch.setattr(brain_mod, 'resolve_backend', lambda dynamics, backend=None: 'cuda')
    real_setup = brain_mod.Brain._setup_device

    def failing_setup(self):
        if self.backend == 'cuda':
            raise RuntimeError('cooperative launch failed')
        return real_setup(self)
    monkeypatch.setattr(brain_mod.Brain, '_setup_device', failing_setup)
    arrays, _, _ = synthetic_test_graph(n=16, k_out=2)
    with caplog.at_level(logging.WARNING, logger='brainlab'):
        b = brain_mod.Brain(arrays=arrays, dynamics='v3', validate=False)
    assert b.backend == 'cpu' and 'cooperative launch failed' in b.backend_note
    assert any('falls back to the CPU' in r.getMessage() for r in caplog.records)
    counts, _ = b.step(np.zeros(b.n, dtype=np.float32), 1.0)
    assert counts.shape == (b.n,)


def test_explicit_cuda_request_still_fails_loudly(monkeypatch):
    import brainlab.brain as brain_mod
    from brainlab.graph_identity import synthetic_test_graph

    def failing_setup(self):
        raise RuntimeError('no device')
    monkeypatch.setattr(brain_mod.Brain, '_setup_device', failing_setup)
    arrays, _, _ = synthetic_test_graph(n=16, k_out=2)
    with pytest.raises(RuntimeError, match='no device'):
        brain_mod.Brain(arrays=arrays, dynamics='v3', validate=False, backend='cuda')
