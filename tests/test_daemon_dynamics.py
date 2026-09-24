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
