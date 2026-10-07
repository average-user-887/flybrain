"""Installed-configuration regression: an explicit --graph-dir survives every
controller rebuild.

Live defect, candidate 7d7abd6 installed from its wheel (Astra dispatch 41,
2026-10-07): the daemon started with ``--graph-dir <checkout>/outputs/brainlab/
malecns_v1`` ran fixed and switched all 14 assays, but selecting the
ConnectomePlastic controller failed with GraphUnavailable because the plastic
rule (``VisualHeadingPlasticityRule.from_shared``) passed ``graph_dir=None`` and
so resolved ``<venv>/site-packages/outputs/brainlab/malecns_v1`` (the packaged
default) instead of the directory the shared graph was verified from.

Here the packaged defaults point at a directory that does not exist (as in an
installed wheel) and the graph lives in a separate explicit directory.  The
graph verifier is replaced by a recording stand-in for a tiny graph (the real
pins describe the 166k-neuron MaleCNS graph); every other load path is the
daemon's own.  No 200 MB graph, no behavioural claim.
"""
import numpy as np
import pandas as pd
import pytest

import experiment_registry
from brainlab import graph_identity, transmitter_policy
from brainlab.graph_identity import (GraphIdentity, GraphUnavailable, resolve_connectome_dir,
                                     resolve_graph_dir, synthetic_test_graph)
from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
from neurofly_daemon import ContinuousExperimentRunner
from tests.transition_control_helpers import transition_command

N = 64


def _write_tiny_graph(tmp_path):
    arrays, _, io_map = synthetic_test_graph(n=N)
    gdir = tmp_path / 'explicit' / 'graph'
    gdir.mkdir(parents=True)
    np.savez(gdir / 'graph.npz', **arrays)
    cdir = tmp_path / 'explicit' / 'connectome'
    (cdir / 'normalized').mkdir(parents=True)
    cell_type = ['other'] * N
    for i in range(0, 20):
        cell_type[i] = 'ER4d' if i % 2 else 'ER2_a'
    for i in range(20, 50):
        cell_type[i] = 'EPG'
    for i in range(50, 54):
        cell_type[i] = 'EL'
    transmitter = ['octopamine' if t == 'EL' else 'gaba' if t.startswith('ER') else 'acetylcholine'
                   for t in cell_type]
    pd.DataFrame({'node_index': np.arange(N), 'source_id': arrays['ids'], 'cell_type': cell_type,
                  'neurotransmitter': transmitter}).to_feather(cdir / 'normalized' / 'neurons.feather')
    return gdir, cdir, io_map


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Packaged defaults absent (installed wheel), explicit graph elsewhere."""
    gdir, cdir, io_map = _write_tiny_graph(tmp_path)
    site = tmp_path / 'venv' / 'site-packages'
    monkeypatch.setattr(graph_identity, 'DEFAULT_GRAPH_DIR', site / 'outputs/brainlab/malecns_v1')
    monkeypatch.setattr(graph_identity, 'DEFAULT_CONNECTOME_DIR', site / 'connectome_data/malecns_v1')
    monkeypatch.delenv('NEUROFLY_GRAPH_DIR', raising=False)
    monkeypatch.setenv('NEUROFLY_CONNECTOME_DIR', str(cdir))
    loads = []

    def verify(graph_dir=None, connectome_dir=None, **_):
        g, _ = resolve_graph_dir(graph_dir)
        c, _ = resolve_connectome_dir(connectome_dir)
        loads.append(('verify_graph', g, c))
        if not (g / 'graph.npz').is_file():
            raise GraphUnavailable(f'Prepared graph not found at {g / "graph.npz"}')
        with np.load(g / 'graph.npz') as data:
            ids = data['ids']
        return GraphIdentity(dataset='tiny-explicit-test', synthetic=False, graph_path=str(g / 'graph.npz'),
                             graph_path_source='argument', graph_sha256='0' * 64,
                             neuron_map_path=str(c / 'normalized/neurons.feather'), neuron_map_sha256='1' * 64,
                             io_map_sha256='2' * 64, neurons=N, edges=N * 6,
                             ids_sha256=graph_identity._ids_digest(ids))

    real_transmitters = transmitter_policy.load_transmitters

    def transmitters(connectome_dir=None):
        loads.append(('transmitter_policy', None, resolve_connectome_dir(connectome_dir)[0]))
        return real_transmitters(connectome_dir)

    original = VisualHeadingPlasticityRule.from_connectome.__func__

    def rule(cls, graph_dir=None, connectome_dir=None, **kwargs):
        loads.append(('plasticity_rule', resolve_graph_dir(graph_dir)[0], resolve_connectome_dir(connectome_dir)[0]))
        return original(cls, graph_dir=graph_dir, connectome_dir=connectome_dir, **kwargs)

    monkeypatch.setattr(experiment_registry, 'verify_graph', verify)
    monkeypatch.setattr(transmitter_policy, 'load_transmitters', transmitters)
    monkeypatch.setattr(VisualHeadingPlasticityRule, 'from_connectome', classmethod(rule))
    real_load = experiment_registry.SharedGraph.load.__func__

    def load(cls, graph_dir=None, connectome_dir=None):
        shared = real_load(cls, graph_dir, connectome_dir)
        shared.io_map = dict(io_map)
        return shared
    monkeypatch.setattr(experiment_registry.SharedGraph, 'load', classmethod(load))
    return dict(gdir=gdir, cdir=cdir, loads=loads, site=site, tmp=tmp_path, monkeypatch=monkeypatch)


def _runner(env):
    return ContinuousExperimentRunner(initial_paradigm='open-arena', sim_speed=1, checkpoint_interval=3600,
                                      output_dir=env['tmp'] / 'state', backend='connectome-fixed',
                                      graph_dir=env['gdir'])


def _switch(runner, backend):
    return transition_command(runner, {'action': 'switch_backend', 'backend': backend}, lock_held=True)


@pytest.mark.parametrize('env_changed_after_start', [False, True])
def test_explicit_graph_dir_is_used_by_every_controller_load(installed, env_changed_after_start):
    runner = _runner(installed)
    if env_changed_after_start:
        # The configuration is captured at startup: a later environment change
        # must not move any controller load.
        installed['monkeypatch'].setenv('NEUROFLY_CONNECTOME_DIR', str(installed['tmp'] / 'elsewhere'))
        installed['monkeypatch'].setenv('NEUROFLY_GRAPH_DIR', str(installed['tmp'] / 'elsewhere'))
    with runner.lock:
        runner.step_once()
        plastic = _switch(runner, 'connectome-plastic')
        assert plastic['status'] == 'ok', plastic.get('message')
        assert runner.backend == 'connectome-plastic' and runner.last_error is None
        assert runner.registry.plasticity_rule.describe()['n_edges'] > 0
        runner.step_once()
        fixed = _switch(runner, 'connectome-fixed')
        assert fixed['status'] == 'ok', fixed.get('message')
        assert runner.backend == 'connectome-fixed' and runner.last_error is None
        runner.step_once()
        again = _switch(runner, 'connectome-plastic')
        assert again['status'] == 'ok', again.get('message')
    kinds = {kind for kind, _, _ in installed['loads']}
    assert {'verify_graph', 'transmitter_policy', 'plasticity_rule'} <= kinds
    for kind, g, c in installed['loads']:
        assert g in (None, installed['gdir']), (kind, g)
        assert c == installed['cdir'], (kind, c)
    assert not installed['site'].exists()       # nothing created at the packaged default
    assert runner.graph_dir == installed['gdir'] and runner.connectome_dir == installed['cdir']


def test_missing_graph_still_refuses_and_halts_with_fixed_controller_retained(installed):
    runner = _runner(installed)
    with runner.lock:
        runner.step_once()
        identity = runner.identity()
    # The explicit graph genuinely disappears after startup.
    (installed['gdir'] / 'graph.npz').rename(installed['gdir'] / 'graph.npz.gone')
    with runner.lock:
        result = _switch(runner, 'connectome-plastic')
    assert result['status'] == 'error' and result.get('applied') is False
    assert 'GraphUnavailable' in (runner.error_detail or {}).get('type', '') or 'missing' in result['message']
    assert str(installed['gdir']) in result['message']           # names the explicit dir, not a default
    assert 'site-packages' not in result['message']
    assert runner.backend == 'connectome-fixed'
    assert runner.last_error is not None                           # halted, not silently resumed
    assert runner.identity()['instance_id'] == identity['instance_id']
    assert not installed['site'].exists()


def test_real_graph_plastic_rule_follows_explicit_location(tmp_path, monkeypatch):
    """Real MaleCNS graph reached only through explicit, non-default paths."""
    real_graph, real_conn = graph_identity.DEFAULT_GRAPH_DIR, graph_identity.DEFAULT_CONNECTOME_DIR
    if not (real_graph / 'graph.npz').is_file() or not (real_conn / 'normalized/neurons.feather').is_file():
        pytest.skip('MaleCNS graph not available at the checkout default')
    gdir, cdir = tmp_path / 'explicit-graph', tmp_path / 'explicit-connectome'
    gdir.symlink_to(real_graph.resolve(), target_is_directory=True)
    cdir.symlink_to(real_conn.resolve(), target_is_directory=True)
    site = tmp_path / 'venv' / 'site-packages'
    monkeypatch.setattr(graph_identity, 'DEFAULT_GRAPH_DIR', site / 'outputs/brainlab/malecns_v1')
    monkeypatch.setattr(graph_identity, 'DEFAULT_CONNECTOME_DIR', site / 'connectome_data/malecns_v1')
    monkeypatch.delenv('NEUROFLY_GRAPH_DIR', raising=False)
    monkeypatch.delenv('NEUROFLY_CONNECTOME_DIR', raising=False)
    shared = experiment_registry.SharedGraph.load(gdir, cdir)
    rule = VisualHeadingPlasticityRule.from_shared(shared)
    assert rule.describe()['n_edges'] == 3081
    assert (rule.initial_weights < 0).all()
    assert not site.exists()
