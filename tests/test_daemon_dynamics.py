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


def test_v3_brain_from_a_path_gets_the_transmitter_policy(tmp_path, monkeypatch):
    import numpy as np
    from brainlab import transmitter_policy
    from brainlab.brain import Brain
    path = tmp_path / 'graph.npz'
    np.savez(path, ptr=np.array([0, 1, 2, 2], dtype=np.int64), post=np.array([1, 2], dtype=np.int32),
             weight=np.array([5.0, 7.0], dtype=np.float32), ids=np.array([10, 11, 12], dtype=np.int64))
    labels = transmitter_policy.normalise(['dopamine', 'acetylcholine', 'gaba'])
    monkeypatch.setattr(transmitter_policy, 'load_transmitters', lambda connectome_dir=None: labels)
    np.testing.assert_array_equal(Brain(path, dynamics='v3').weight, [0.0, 7.0])
    np.testing.assert_array_equal(Brain(path, dynamics='v1').weight, [5.0, 7.0])


def test_registry_refuses_v3_on_pinned_real_weights(monkeypatch):
    import dataclasses
    import pytest
    from experiment_registry import GraphInstance, SharedGraph
    from provenance import BackendError
    shared = SharedGraph.synthetic(allow_synthetic=True)
    real = SharedGraph(shared.arrays, dataclasses.replace(shared.identity, synthetic=False), shared.io_map)

    class Registry:
        pass
    registry = Registry()
    registry.shared = real
    monkeypatch.setenv('NEUROFLY_LIF_DYNAMICS', 'v3')
    with pytest.raises(BackendError, match='transmitter-policy'):
        GraphInstance(registry, 'optomotor', 'connectome-fixed', 'x', 0, manifest=None)
