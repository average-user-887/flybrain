"""Launcher ownership/idempotence and coherent observatory snapshots."""
import json
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from scripts import observatory
from experiment_brains import ExperimentBrains
from neurofly_daemon import ContinuousExperimentRunner, NeuroflyHTTPHandler
from stream_gateway import StreamGateway, StreamPolicy


def make_project(tmp_path):
    project = tmp_path / 'project with spaces'
    python = project / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(__import__('sys').executable))
    return project


def test_service_install_is_idempotent_and_keeps_venv_path(tmp_path):
    project = make_project(tmp_path)
    units = tmp_path / 'units'
    assert len(observatory.install_units(project, units)) == 3
    before = {p.name: p.stat().st_mtime_ns for p in units.iterdir()}
    assert observatory.install_units(project, units) == []
    assert {p.name: p.stat().st_mtime_ns for p in units.iterdir()} == before
    text = (units / observatory.BRAIN).read_text()
    assert str(project / '.venv' / 'bin' / 'python') in text
    assert '"--host" "127.0.0.1"' in text
    assert 'Restart=always' in text
    assert 'observatory-live' in text


def test_unmanaged_service_is_never_overwritten(tmp_path):
    project = make_project(tmp_path)
    units = tmp_path / 'units'
    units.mkdir()
    target = units / observatory.WEB
    target.write_text('[Service]\nExecStart=/bin/true\n')
    with pytest.raises(RuntimeError, match='unmanaged'):
        observatory.install_units(project, units)
    assert target.read_text() == '[Service]\nExecStart=/bin/true\n'
    assert not (units / observatory.BRAIN).exists()


def test_generated_units_are_accepted_by_systemd(tmp_path):
    if not shutil.which('systemd-analyze'):
        pytest.skip('systemd is not installed on this platform')
    project = make_project(tmp_path)
    units = tmp_path / 'units'
    observatory.install_units(project, units)
    result = subprocess.run(['systemd-analyze', '--user', 'verify',
                             *map(str, units.glob('*.service'))], capture_output=True, text=True)
    if 'Failed to lookup RuntimeDirectory' in result.stderr or 'Failed to initialize manager' in result.stderr:
        pytest.skip('No systemd user environment')
    assert result.returncode == 0, result.stderr


def test_saved_catalog_preserves_progress_without_loading_brains(tmp_path):
    bank = ExperimentBrains(tmp_path)
    b = bank.get('t-maze')
    b.steps, b.trials = 777, 4
    b.save()
    restarted = ExperimentBrains(tmp_path)
    saved = next(x for x in restarted.catalog('y-maze') if x['paradigm'] == 't-maze')
    assert saved['brain_id'] == b.brain_id
    assert saved['steps'] == 777 and saved['trials'] == 4
    assert saved['state'] == 'saved' and restarted.instances == {}
    # Refresh cached metadata when a checkpoint advances.
    b.steps = 999
    b.save()
    assert next(x for x in restarted.catalog('y-maze') if x['paradigm'] == 't-maze')['steps'] == 999


def test_catalog_reports_corruption_instead_of_hiding_or_overwriting_it(tmp_path):
    file = tmp_path / 't-maze.json'
    file.write_text('{broken')
    item = next(x for x in ExperimentBrains(tmp_path).catalog('y-maze') if x['paradigm'] == 't-maze')
    assert item['state'] == 'checkpoint_error'
    assert file.read_text() == '{broken'


def test_http_snapshot_matches_selected_brain_and_public_policy(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=tmp_path)
    class Handler(NeuroflyHTTPHandler):
        pass
    Handler.runner = runner
    Handler.gateway = StreamGateway(StreamPolicy(public=True))
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for pid in ('t-maze', 'y-maze', 't-maze'):
            runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': pid})
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/observatory') as response:
                data = json.load(response)
            assert data['brain']['paradigm'] == data['status']['active_paradigm'] == pid
            assert data['brain']['brain_id'] == data['telemetry']['brain_id']
            assert data['status']['stream']['read_only']
            assert len(data['brains']) == 14
            assert len(data['brain']['weights']) == 120
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
