"""The live daemon defaults to v3 dynamics; v1 stays selectable and separate."""
from pathlib import Path

from experiment_registry import default_registry_dir


def test_registry_root_is_per_dynamics(tmp_path):
    assert default_registry_dir(tmp_path, 'v1') == Path(tmp_path) / 'registry'
    assert default_registry_dir(tmp_path, 'v3') == Path(tmp_path) / 'registry-v3'


def test_registry_manifest_records_active_dynamics(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROFLY_LIF_DYNAMICS', 'v3')
    from tests.test_provenance_registry import make_registry
    registry = make_registry(tmp_path)
    instance = registry.activate('t-maze', 'connectome-fixed')
    assert instance.brain.dynamics == 'v3'
    assert instance.manifest.dynamics['dynamics_version'] == 'v3'


def test_daemon_cli_defaults_to_v3_and_keeps_v1(monkeypatch):
    import neurofly_daemon
    monkeypatch.delenv('NEUROFLY_LIF_DYNAMICS', raising=False)
    assert neurofly_daemon.build_arg_parser().parse_args([]).dynamics == 'v3'
    assert neurofly_daemon.build_arg_parser().parse_args(['--dynamics', 'v1']).dynamics == 'v1'
    monkeypatch.setenv('NEUROFLY_LIF_DYNAMICS', 'v1')
    assert neurofly_daemon.build_arg_parser().parse_args([]).dynamics == 'v1'


def test_full_sim_brain_runs_v3_on_policy_weights(tmp_path, monkeypatch):
    import numpy as np
    from experiment_registry import SharedGraph
    from experiments.full_connectome_simulation import UnifiedConnectomeBrain
    shared = SharedGraph.synthetic(allow_synthetic=True)
    policy_weight = np.full_like(shared.arrays['weight'], 0.25)
    policy = SharedGraph(dict(shared.arrays, weight=policy_weight), shared.identity, shared.io_map)
    calls = []

    def fake_load(cls, graph_dir=None, connectome_dir=None, dynamics=None):
        calls.append(dynamics)
        return policy
    monkeypatch.setattr(SharedGraph, 'load_for_dynamics', classmethod(fake_load))
    brain = UnifiedConnectomeBrain(graph_dir=tmp_path, connectome_dir=tmp_path, enable_plasticity=False)
    assert calls == ['v3']
    assert brain.brain.dynamics == 'v3'
    np.testing.assert_array_equal(brain.brain.weight, policy_weight)
