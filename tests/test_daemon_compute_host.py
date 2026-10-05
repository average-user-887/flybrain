"""Daemon start-up facts a new user needs: where the brain computes, and who can reach the API."""
from pathlib import Path
from types import SimpleNamespace

import neurofly_daemon as nd
from stream_gateway import StreamGateway, StreamPolicy

ROOT = Path(nd.__file__).resolve().parent


def test_daemon_binds_localhost_by_default_and_keeps_host_option():
    parser = nd.build_arg_parser()
    assert parser.parse_args([]).host == '127.0.0.1'
    assert parser.parse_args(['--host', '0.0.0.0']).host == '0.0.0.0'


def test_start_daemon_script_passes_host_explicitly():
    # Local only by default, stated explicitly; NEUROFLY_HOST=0.0.0.0 opts in to the LAN.
    text = (ROOT / 'start_daemon.sh').read_text()
    assert 'HOST="${NEUROFLY_HOST:-127.0.0.1}"' in text
    assert '--host "$HOST"' in text


def test_container_listens_on_all_interfaces_inside_the_container():
    assert '"--host", "0.0.0.0"' in (ROOT / 'Dockerfile').read_text()


def test_cosim_server_binds_localhost_by_default():
    import inspect
    from brainlab import cosim_server
    assert inspect.signature(cosim_server.run_server).parameters['host'].default == '127.0.0.1'


def test_compute_info_modular_reports_cpu_without_touching_a_gpu(monkeypatch):
    import brainlab.brain as brain

    def refuse():
        raise AssertionError('GPU queried for the modular controller')
    monkeypatch.setattr(brain, 'gpu_name', refuse)
    info = nd.ContinuousExperimentRunner.compute_info(SimpleNamespace(graph_mode=False))
    assert info['device'] == 'cpu' and info['brain'] == 'modular' and info['gpu'] is None


def test_compute_info_graph_reports_the_active_brain_backend_and_gpu(monkeypatch):
    import brainlab.brain as brain
    monkeypatch.setattr(brain, 'gpu_name', lambda: 'NVIDIA GeForce GTX 1660 Ti')
    runner = SimpleNamespace(graph_mode=True, registry=SimpleNamespace(
        active=SimpleNamespace(brain=SimpleNamespace(backend='cuda'))))
    info = nd.ContinuousExperimentRunner.compute_info(runner)
    assert info['device'] == 'cuda' and info['gpu'] == 'NVIDIA GeForce GTX 1660 Ti'
    assert 'CUDA' in info['detail'] and info['source'] == 'active brain'
    runner.registry.active.brain.backend = 'cpu'
    info = nd.ContinuousExperimentRunner.compute_info(runner)
    assert info['device'] == 'cpu' and info['gpu'] is None and 'CPU' in info['detail']


def test_compute_info_graph_without_active_brain_resolves_backend(monkeypatch):
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'cpu')
    runner = SimpleNamespace(graph_mode=True, registry=SimpleNamespace(active=None))
    info = nd.ContinuousExperimentRunner.compute_info(runner)
    assert info['device'] == 'cpu' and info['source'].startswith('resolved')


def test_status_payload_carries_compute(tmp_path):
    runner = nd.ContinuousExperimentRunner(initial_paradigm='wind-tunnel', sim_speed=1.0,
                                           checkpoint_interval=3600, output_dir=tmp_path)
    handler = object.__new__(nd.NeuroflyHTTPHandler)
    handler.runner = runner
    handler.gateway = StreamGateway(StreamPolicy())
    payload = handler._status_payload()
    assert payload['compute']['device'] == 'cpu' and payload['compute']['brain'] == 'modular'
