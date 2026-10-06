"""Tiny source contracts only; fake transport never executes WGSL or a service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

import neurofly_daemon as nd
from brainlab import amd_state_adapter
from brainlab.amd_state_adapter import AmdV3StateAdapter, AmdStateUnavailable
from brainlab.runtime_backend import UnsupportedRuntimeCapability, preflight_fixed_store
from experiment_registry import ExperimentRegistry, SharedGraph
from tests.test_amd_state_adapter import FakeDevice, assert_exact, snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def amd(monkeypatch):
    monkeypatch.setenv('NEUROFLY_LIF_DYNAMICS', 'v3')
    monkeypatch.delenv('NEUROFLY_LIF_E_INH_MV', raising=False)
    monkeypatch.setitem(sys.modules, 'wgpu', types.SimpleNamespace(
        BufferUsage=types.SimpleNamespace(STORAGE=1, COPY_SRC=2, COPY_DST=4)))
    devices = []
    def factory(*args, **kwargs):
        device = FakeDevice()
        device.adapter_info = dict(vendor='test-vendor', vendor_id=0x1002, device='FAKE AMD TRANSPORT',
                                   device_id=17, backend_type='Vulkan', adapter_type='IntegratedGPU',
                                   description='test only')
        devices.append(device)
        return AmdV3StateAdapter(*args, **kwargs, device=device)
    monkeypatch.setattr(amd_state_adapter, 'AmdV3StateAdapter', factory)
    return devices


def shared():
    return SharedGraph.synthetic(allow_synthetic=True, n=3, k_out=1)


def registry(path, compute='wgpu-amd'):
    return ExperimentRegistry(shared(), path, test_mode=True, brain_backend=compute)


def tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('origin', ['cpu', 'wgpu-amd'])
def test_registry_restores_complete_nonzero_clock_and_order_from_checkpoint(tmp_path, amd, monkeypatch, origin):
    a = registry(tmp_path, compute=origin)
    instance = a.activate('t-maze', 'connectome-fixed')
    instance.brain.restore_state(snapshot())
    instance.step_index = 7
    saved = instance.brain.snapshot_state()
    a.checkpoint(world_state={'fixture': 'kept'})
    identity = instance.instance_id
    # Environment edits after selector capture cannot redirect the next instance.
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'cpu')
    b = registry(tmp_path)
    restored = b.activate('t-maze', 'connectome-fixed')
    assert restored.instance_id == identity and restored.world_state == {'fixture': 'kept'}
    assert restored.brain.backend == 'wgpu-amd' and not b.learning_enabled
    assert_exact(saved, restored.brain.snapshot_state())
    if origin == 'cpu':
        # Exact same fixed state/identity, no weights/state conversion and no
        # claim that fake transport computes CPU arithmetic.
        np.testing.assert_array_equal(instance.brain.weight, restored.brain.weight)
        return
    # Independent continuation from the same complete state through the real API.
    ca = instance.step(np.array([1, 0, 2], np.float32), .3).counts
    cb = restored.step(np.array([1, 0, 2], np.float32), .3).counts
    np.testing.assert_array_equal(ca, cb)
    assert_exact(instance.brain.snapshot_state(), restored.brain.snapshot_state())


@pytest.mark.parametrize('controller', ['connectome-plastic', 'connectome-with-trained-readout', 'modular'])
def test_registry_controller_refusal_precedes_index_owner_or_weight_mutation(tmp_path, amd, controller):
    r = registry(tmp_path)
    active = r.activate('t-maze', 'connectome-fixed')
    before, weights, writes = tree(tmp_path), active.brain.weight.copy(), amd[-1].writes
    with pytest.raises(UnsupportedRuntimeCapability):
        r.prepare_activation('y-maze', controller)
    assert tree(tmp_path) == before and r.active is active
    np.testing.assert_array_equal(active.brain.weight, weights)
    assert amd[-1].writes == writes


@pytest.mark.parametrize('options', [{'learning_enabled': True}, {'continue_io_state': True},
                                    {'plasticity_rule': object()}])
def test_registry_unsupported_policy_refuses_before_store_creation(tmp_path, amd, options):
    root = tmp_path / 'uncreated'
    with pytest.raises(UnsupportedRuntimeCapability):
        ExperimentRegistry(shared(), root, test_mode=True, brain_backend='wgpu-amd', **options)
    assert not root.exists() and not amd


@pytest.mark.parametrize('kind', ['delta', 'readout', 'trace', 'brain_unknown', 'reset'])
def test_learned_payload_refuses_before_restore_and_all_mutations(tmp_path, amd, kind):
    r = registry(tmp_path)
    instance = r.activate('t-maze', 'connectome-fixed')
    meta, arrays = instance.meta(), instance.state_arrays()
    if kind == 'delta': arrays['plastic.delta'] = np.ones(1, np.float32)
    elif kind == 'readout': arrays['readout.weights'] = np.ones(3)
    elif kind == 'trace': arrays['plastic.pre_trace'] = np.zeros(1)
    elif kind == 'brain_unknown': arrays['brain.learned_weights'] = np.ones(3)
    else: meta['learning_state_resets'] = [{'reason': 'old reset'}]
    before, state, writes = tree(tmp_path), instance.brain.snapshot_state(), amd[-1].writes
    with pytest.raises(UnsupportedRuntimeCapability): instance.load_payload(meta, arrays)
    assert amd[-1].writes == writes and tree(tmp_path) == before
    assert_exact(state, instance.brain.snapshot_state())


def test_saved_learned_state_refused_before_daemon_store_or_pid(tmp_path, amd, monkeypatch):
    out = tmp_path / 'output'
    r = registry(out / 'registry-v3', compute='cpu')
    instance = r.activate('t-maze', 'connectome-fixed')
    r.checkpoint()
    pointer = r.current_pointer(instance.instance_id)
    path = r.instance_dir(instance.instance_id) / 'checkpoints' / pointer['file']
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays['readout.weights'] = np.ones(3)
    np.savez(path, **arrays)
    before = tree(out)
    with pytest.raises(UnsupportedRuntimeCapability):
        nd.ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=out,
            backend='connectome-fixed', brain_backend='wgpu-amd', test_synthetic_graph=True, shared_graph=shared())
    assert tree(out) == before and not amd
    # Run only the startup refusal path: no server or simulation is ever started.
    monkeypatch.setattr(sys, 'argv', ['neurofly_daemon.py', '--backend', 'connectome-fixed',
        '--brain-backend', 'wgpu-amd', '--test-synthetic-graph', '--paradigm', 't-maze',
        '--output-dir', str(out), '--pid-file', str(tmp_path / 'pid' / 'daemon.pid')])
    with pytest.raises(SystemExit) as exit_status: nd.run_daemon()
    assert exit_status.value.code == 2 and not (tmp_path / 'pid').exists()
    assert tree(out) == before and not amd


@pytest.mark.parametrize('backend,dynamics', [('connectome-plastic', 'v3'),
    ('connectome-with-trained-readout', 'v3'), ('modular', 'v3'), ('connectome-fixed', 'v2')])
def test_explicit_unsupported_selector_refuses_before_output_or_pid(tmp_path, amd, monkeypatch, backend, dynamics):
    monkeypatch.setattr(sys, 'argv', ['neurofly_daemon.py', '--backend', backend,
        '--brain-backend', 'wgpu-amd', '--dynamics', dynamics, '--test-synthetic-graph',
        '--output-dir', str(tmp_path / 'out'), '--pid-file', str(tmp_path / 'pid' / 'daemon.pid')])
    with pytest.raises(SystemExit) as exit_status: nd.run_daemon()
    assert exit_status.value.code == 2 and not (tmp_path / 'out').exists()
    assert not (tmp_path / 'pid').exists() and not amd


def test_daemon_capability_refusals_before_owner_store_or_pending_control(tmp_path, amd):
    runner = nd.ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=tmp_path,
        backend='connectome-fixed', brain_backend='wgpu-amd', test_synthetic_graph=True, shared_graph=shared(),
        standalone_scheduled_records=False)
    before, ident, active = tree(tmp_path), runner.identity(), runner.registry.active
    try:
        for cmd in ({'action': 'set_learning', 'enabled': True}, {'action': 'teach_brain'},
                    {'action': 'switch_backend', 'backend': 'connectome-plastic'},
                    {'action': 'switch_backend', 'backend': 'connectome-with-trained-readout'}):
            reply = runner.dispatch_command(cmd)
            assert reply['status'] == 'error'
            assert 'wgpu-amd' in reply['message'] or 'not the controller' in reply['message']
            assert runner.identity() == ident and runner.registry.active is active
            assert tree(tmp_path) == before and runner._pending_assay_control is None
            assert not runner.active_brain.learning_enabled and not runner.arena.fly.learning_enabled
        info = runner.compute_info()
        assert info['device'] == 'wgpu-amd' and info['adapter']['vendor_id'] == 0x1002
        assert info['gpu'] == 'FAKE AMD TRANSPORT' and 'Vulkan' in info['detail']
        amd[-1].adapter_info['device'] = 'CHANGED AFTER CACHE'
        amd[-1].fail = 'write'
        io_counts = (amd[-1].reads, amd[-1].writes, amd[-1].submissions)
        assert runner.compute_info() == info  # cached identity; no device I/O/probe
        assert (amd[-1].reads, amd[-1].writes, amd[-1].submissions) == io_counts
        amd[-1].fail = None
        active.brain._gpu._poisoned = True
        assert runner.compute_info()['state']['poisoned']
        with pytest.raises(AmdStateUnavailable): runner.registry.checkpoint()
        assert tree(tmp_path) == before
        active.brain._gpu._poisoned = False
    finally:
        runner.stop()


def test_cpu_lazy_import_and_selector_capture(tmp_path):
    code = """
import sys, os
from experiment_registry import ExperimentRegistry, SharedGraph
r = ExperimentRegistry(SharedGraph.synthetic(allow_synthetic=True, n=3, k_out=1),
                       sys.argv[1], test_mode=True, brain_backend='cpu')
os.environ['NEUROFLY_BRAIN_BACKEND'] = 'wgpu-amd'
a = r.activate('t-maze', 'connectome-fixed')
b = r.activate('y-maze', 'connectome-fixed')
assert a.brain is None and b.brain.backend == 'cpu'
assert not any(n == 'wgpu' or n.startswith('wgpu.') for n in sys.modules)
assert 'brainlab.amd_state_adapter' not in sys.modules
"""
    proc = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT,
        env=dict(os.environ, NEUROFLY_BRAIN_BACKEND='cpu', NEUROFLY_LIF_DYNAMICS='v3'),
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_device_failure_is_compute_and_cli_status_does_not_probe_cuda(amd, monkeypatch, capsys):
    from neurofly.cli import cmd_status
    from brainlab import graph_identity, gpu_probe
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'wgpu-amd')
    monkeypatch.setitem(sys.modules, 'flygym', types.SimpleNamespace(__version__='fake'))
    monkeypatch.setitem(sys.modules, 'mujoco', types.SimpleNamespace(__version__='fake'))
    def unavailable(*args, **kwargs): raise graph_identity.GraphUnavailable('test only: no graph')
    def forbidden(*args, **kwargs): raise AssertionError('AMD status must not probe CUDA')
    monkeypatch.setattr(graph_identity, 'verify_graph', unavailable)
    monkeypatch.setattr(gpu_probe, 'explain', forbidden)
    assert cmd_status([]) == 0
    output = capsys.readouterr().out
    assert 'wgpu-amd requested' in output and 'Brain backend: CPU' not in output
    gpu_error = type('GPUError', (RuntimeError,), {'__module__': 'wgpu.test'})('fake device error')
    assert nd.classify_failure(gpu_error) == nd.FAILURE_COMPUTE
    assert nd.classify_failure(AmdStateUnavailable('poisoned state')) == nd.FAILURE_COMPUTE
    assert nd.classify_failure(RuntimeError('device arithmetic/overflow failure')) == nd.FAILURE_COMPUTE


def test_metadata_preflight_does_not_read_full_transients(tmp_path, monkeypatch):
    r = registry(tmp_path, compute='cpu')
    instance = r.activate('t-maze', 'connectome-fixed')
    r.checkpoint()
    pointer = r.current_pointer(instance.instance_id)
    path = r.instance_dir(instance.instance_id) / 'checkpoints' / pointer['file']
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    get = archive_type.__getitem__
    def read_only_metadata(self, key):
        assert not key.startswith('brain.'), 'Admission must not load full transients'
        return get(self, key)
    monkeypatch.setattr(archive_type, '__getitem__', read_only_metadata)
    before = tree(tmp_path)
    preflight_fixed_store(tmp_path)
    assert tree(tmp_path) == before


@pytest.mark.parametrize('tamper', ['learned_location', 'dynamics', 'plastic_rule'])
def test_saved_manifest_refusal_keeps_store_and_never_initializes_device(tmp_path, amd, tamper):
    r = registry(tmp_path, compute='cpu')
    instance = r.activate('t-maze', 'connectome-fixed')
    r.checkpoint()
    path = r.instance_dir(instance.instance_id) / 'manifest.json'
    data = json.loads(path.read_text())
    if tamper == 'learned_location': data['learned_parameter_locations']['readout_weights'] = 'retained-state'
    elif tamper == 'dynamics': data['dynamics']['dynamics_version'] = 'v2'
    else: data['dynamics']['plasticity_rule'] = {'name': 'unsupported'}
    path.write_text(json.dumps(data))
    before = tree(tmp_path)
    with pytest.raises(UnsupportedRuntimeCapability): registry(tmp_path)
    assert tree(tmp_path) == before and not amd


def test_explicit_dynamics_wins_during_saved_state_startup_admission(tmp_path, amd, monkeypatch):
    out = tmp_path / 'output'
    r = registry(out / 'registry-v3', compute='cpu')
    r.activate('t-maze', 'connectome-fixed')
    r.checkpoint()
    before = tree(out)
    monkeypatch.setenv('NEUROFLY_LIF_DYNAMICS', 'v1')
    monkeypatch.setattr(sys, 'argv', ['neurofly_daemon.py', '--backend', 'connectome-fixed',
        '--brain-backend', 'wgpu-amd', '--dynamics', 'v3', '--test-synthetic-graph',
        '--paradigm', 't-maze', '--output-dir', str(out), '--pid-file', str(tmp_path / 'pid')])
    class Admitted(Exception): pass
    def stop_before_pid(**kwargs): raise Admitted()
    monkeypatch.setattr(nd.StreamPolicy, 'from_env', stop_before_pid)
    with pytest.raises(Admitted): nd.run_daemon()
    assert tree(out) == before and not (tmp_path / 'pid').exists() and not amd


def test_invalid_environment_selector_refuses_before_pid_or_output(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'unknown-compute')
    monkeypatch.setattr(sys, 'argv', ['neurofly_daemon.py', '--backend', 'connectome-fixed',
        '--test-synthetic-graph', '--output-dir', str(tmp_path / 'out'), '--pid-file', str(tmp_path / 'pid')])
    with pytest.raises(SystemExit) as exit_status: nd.run_daemon()
    assert exit_status.value.code == 2
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'pid').exists()


@pytest.mark.parametrize('duration', ['0', '20.05', '102.5', 'inf', 'nan'])
def test_unsupported_advance_capacity_refuses_before_output_and_pid(tmp_path, amd, monkeypatch, duration):
    monkeypatch.setattr(sys, 'argv', ['neurofly_daemon.py', '--backend', 'connectome-fixed',
        '--brain-backend', 'wgpu-amd', '--test-synthetic-graph', '--graph-step-ms', duration,
        '--output-dir', str(tmp_path / 'out'), '--pid-file', str(tmp_path / 'pid')])
    with pytest.raises(SystemExit) as exit_status: nd.run_daemon()
    assert exit_status.value.code == 2
    assert not (tmp_path / 'out').exists() and not (tmp_path / 'pid').exists() and not amd


@pytest.mark.parametrize('source', ['manifest', 'checkpoint'])
@pytest.mark.parametrize('evidence', ['missing', 'version', 'hash'])
def test_graph_io_refusal_precedes_pid_output_or_device_initialization(tmp_path, amd, monkeypatch, source, evidence):
    out = tmp_path / 'output'
    r = registry(out / 'registry-v3', compute='cpu')
    instance = r.activate('t-maze', 'connectome-fixed')
    r.checkpoint()
    directory = r.instance_dir(instance.instance_id)
    if source == 'manifest':
        path = directory / 'manifest.json'
        data = json.loads(path.read_text())
    else:
        pointer = r.current_pointer(instance.instance_id)
        path = directory / 'checkpoints' / pointer['file']
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        data = json.loads(str(arrays['meta']))
    if evidence == 'missing': data.pop('graph_io')
    elif evidence == 'version': data['graph_io']['version'] = 'old-method'
    else: data['graph_io']['sha256'] = '0' * 64
    if source == 'manifest': path.write_text(json.dumps(data))
    else:
        arrays['meta'] = np.asarray(json.dumps(data))
        np.savez(path, **arrays)
    before = tree(out)
    with pytest.raises(UnsupportedRuntimeCapability, match='exact current graph-I/O') as refusal:
        nd.ContinuousExperimentRunner(initial_paradigm='t-maze', output_dir=out,
            backend='connectome-fixed', brain_backend='wgpu-amd', test_synthetic_graph=True,
            shared_graph=shared())
    assert 'retained' in str(refusal.value) and 'CPU/CUDA' in str(refusal.value)
    assert '--continue-io-state' not in str(refusal.value)
    assert tree(out) == before and not (out / 'checkpoints').exists() and not amd
    monkeypatch.setattr(sys, 'argv', ['neurofly_daemon.py', '--backend', 'connectome-fixed',
        '--brain-backend', 'wgpu-amd', '--test-synthetic-graph', '--paradigm', 't-maze',
        '--output-dir', str(out), '--pid-file', str(tmp_path / 'pid' / 'daemon.pid')])
    with pytest.raises(SystemExit) as exit_status: nd.run_daemon()
    assert exit_status.value.code == 2 and not (tmp_path / 'pid').exists()
    assert tree(out) == before and not amd


@pytest.mark.parametrize('evidence', ['missing', 'version', 'hash'])
def test_payload_graph_io_refusal_precedes_host_or_device_restore(tmp_path, amd, evidence):
    r = registry(tmp_path)
    instance = r.activate('t-maze', 'connectome-fixed')
    instance.brain.restore_state(snapshot())
    meta, arrays = instance.meta(), instance.state_arrays()
    if evidence == 'missing': meta.pop('graph_io')
    elif evidence == 'version': meta['graph_io'] = dict(meta['graph_io'], version='old-method')
    else: meta['graph_io'] = dict(meta['graph_io'], sha256='0' * 64)
    state, before, writes = instance.brain.snapshot_state(), tree(tmp_path), amd[-1].writes
    with pytest.raises(UnsupportedRuntimeCapability, match='exact current graph-I/O'):
        instance.load_payload(meta, arrays)
    assert amd[-1].writes == writes and tree(tmp_path) == before
    assert_exact(state, instance.brain.snapshot_state())
