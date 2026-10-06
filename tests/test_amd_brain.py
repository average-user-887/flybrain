"""Library interface tests with tiny graphs and fake transport, never a GPU."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from brainlab.brain import Brain, resolve_backend
from brainlab.amd_state_adapter import AmdV3StateAdapter, UnsupportedAmdCapability, AmdStateUnavailable
from tests.test_amd_state_adapter import FakeDevice, assert_exact, snapshot


def arrays():
    return dict(ptr=np.array([0, 1, 1, 2], dtype=np.int64),
                post=np.array([2, 0], dtype=np.int32),
                weight=np.array([1, -1], dtype=np.float32),
                ids=np.arange(3, dtype=np.int64))


@pytest.fixture
def amd(monkeypatch):
    import types
    from brainlab import amd_state_adapter
    monkeypatch.setitem(sys.modules, 'wgpu', types.SimpleNamespace(
        BufferUsage=types.SimpleNamespace(STORAGE=1, COPY_SRC=2, COPY_DST=4)))
    devices = []

    def factory(*args, **kwargs):
        device = FakeDevice()
        devices.append(device)
        return AmdV3StateAdapter(*args, **kwargs, device=device)

    monkeypatch.setattr(amd_state_adapter, 'AmdV3StateAdapter', factory)
    return devices


def brain():
    return Brain(arrays=arrays(), dynamics='v3', backend='wgpu-amd')


def host_state(brain):
    # Deliberately inspect host state without initiating export/readback.
    state = {name: getattr(brain, name).copy() for name in brain._state_arrays()}
    state.update(cursor=brain.cursor, total_spikes=brain.total_spikes,
                 sim_ms=brain.sim_ms, dynamics=brain.dynamics)
    return state


def test_full_state_nonzero_clock_continuation_and_reset(amd):
    a = brain()
    a.restore_state(snapshot())
    assert_exact(snapshot(), host_state(a))
    assert_exact(snapshot(), a.snapshot_state())
    a.step([1, 0, 2], 0.2)
    saved = a.snapshot_state()
    b = brain()
    b.restore_state(saved)
    counts_a, _ = a.step([1, 0, 2], 0.3)
    counts_b, _ = b.step([1, 0, 2], 0.3)
    assert_exact(a.snapshot_state(), b.snapshot_state())
    np.testing.assert_array_equal(counts_a, counts_b)
    assert b.cursor == 12 and b.total_spikes == 29 and b.sim_ms == pytest.approx(1.2)
    before = b.snapshot_state()
    with pytest.raises(ValueError): b.reset_state(-60)
    assert_exact(before, b.snapshot_state())
    b.reset_state()
    assert b.cursor == b.total_spikes == b.sim_ms == 0
    np.testing.assert_array_equal(b.v, [-52] * 3)
    assert not b.active_flag.any() and not b.queue_count.any()
    fresh = brain()
    b.step([1, 0, 2], 0.2)
    fresh.step([1, 0, 2], 0.2)
    assert_exact(b.snapshot_state(), fresh.snapshot_state())


@pytest.mark.parametrize('bad', ['dtype', 'missing_clock', 'learned', 'version', 'active', 'scalar'])
def test_restore_refuses_before_host_or_device_mutation(amd, bad):
    a = brain()
    a.restore_state(snapshot())
    before, wrong = host_state(a), snapshot()
    if bad == 'dtype': wrong['nactive'] = wrong['nactive'].astype(np.int64)
    elif bad == 'missing_clock': wrong.pop('sim_ms')
    elif bad == 'learned': wrong['readout_w'] = np.ones(3, dtype=np.float32)
    elif bad == 'version': wrong['dynamics'] = 'v2'
    elif bad == 'active': wrong['active'][:2] = [0, 0]
    elif bad == 'scalar': wrong['cursor'] = True
    writes = amd[0].writes
    with pytest.raises((ValueError, UnsupportedAmdCapability)): a.restore_state(wrong)
    assert amd[0].writes == writes
    assert_exact(before, host_state(a))
    assert_exact(before, a.snapshot_state())


def test_weight_paths_refuse_before_host_mutation(amd):
    a = brain()
    a.restore_state(snapshot())
    before = host_state(a)
    original = a._weight
    saved_weights = original.copy()
    device_before = [bytes(b.data) for b in amd[0].buffers]
    for call in (lambda: setattr(a, 'weight', np.zeros(2, dtype=np.float32)),
                 lambda: a.set_edge_weights([0], [0]), lambda: a.update_weights([0])):
        with pytest.raises(UnsupportedAmdCapability): call()
        assert a._weight is original
        np.testing.assert_array_equal(a._weight, saved_weights)
        assert_exact(before, host_state(a))
        assert [bytes(b.data) for b in amd[0].buffers] == device_before
    exposed = a.weight
    exposed.setflags(write=True)
    exposed.fill(0)
    np.testing.assert_array_equal(a.weight, saved_weights)
    facts = a.compute_capabilities()
    assert facts['full_transient_checkpoint'] and not facts['weight_mutation']
    assert not facts['learning'] and not facts['learned_state_restore']
    facts['weight_mutation'] = True
    assert not a.compute_capabilities()['weight_mutation']


@pytest.mark.parametrize('failure', ['second_write', 'read', 'submit'])
def test_transfer_failure_does_not_publish_partial_host_state(amd, failure):
    a = brain()
    a.restore_state(snapshot())
    before = host_state(a)
    amd[0].fail = failure
    with pytest.raises(RuntimeError):
        if failure == 'submit': a.step([1, 0, 2], 0.1)
        else:
            changed = snapshot()
            changed['v'][0] = -49
            a.restore_state(changed)
    assert_exact(before, host_state(a))
    with pytest.raises(AmdStateUnavailable): a.snapshot_state()
    amd[0].fail = None
    a.restore_state(before)
    assert_exact(before, a.snapshot_state())


@pytest.mark.parametrize('dynamics', ['v1', 'v2', 'v4', 'v5'])
def test_unsupported_dynamics_refuse_before_adapter_import(monkeypatch, dynamics):
    from brainlab import amd_state_adapter
    monkeypatch.setattr(amd_state_adapter, 'AmdV3StateAdapter',
                        lambda *a, **k: pytest.fail('adapter constructor must not run'))
    with pytest.raises(ValueError, match='baseline v3'):
        Brain(arrays=arrays(), dynamics=dynamics, backend='wgpu-amd')


def test_nonbaseline_constants_refuse_before_adapter(amd):
    with pytest.raises(ValueError, match='baseline v3 constants'):
        Brain(arrays=arrays(), dynamics='v3', e_inh_mV=-80, backend='wgpu-amd')
    assert not amd


def test_explicit_argument_and_environment_no_fallback(amd, monkeypatch):
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'wgpu-amd')
    selected = Brain(arrays=arrays(), dynamics='v3')
    assert selected.backend == 'wgpu-amd' and selected.backend_note == 'requested'
    cpu = Brain(arrays=arrays(), dynamics='v3', backend='cpu')
    assert cpu.backend == 'cpu' and cpu._gpu is None
    from brainlab import amd_state_adapter
    def fail(*args, **kwargs): raise RuntimeError('fake device unavailable')
    monkeypatch.setattr(amd_state_adapter, 'AmdV3StateAdapter', fail)
    with pytest.raises(RuntimeError, match='fake device unavailable'): brain()


def test_cpu_and_auto_selection_keep_existing_policy(monkeypatch):
    from brainlab import cuda_engine
    monkeypatch.setenv('NEUROFLY_BRAIN_BACKEND', 'auto')
    monkeypatch.setattr(cuda_engine, 'cuda_available', lambda: False)
    assert Brain(arrays=arrays(), dynamics='v3').backend == 'cpu'
    monkeypatch.setattr(cuda_engine, 'cuda_available', lambda: True)
    assert resolve_backend('v3') == 'cuda'
    assert resolve_backend('v1') == 'cpu'
    assert resolve_backend('v2') == 'cpu'
    a = Brain(arrays=arrays(), dynamics='v3', backend='cpu')
    a.set_edge_weights([0], [2])
    assert a.weight[0] == 2


def test_cuda_interface_keeps_existing_mutable_weights_and_upload_shape(monkeypatch):
    from brainlab import cuda_engine
    class FakeCuda:
        def __init__(self): self.uploads = []; self.weights = []
        def upload_state(self, *args): self.uploads.append(tuple(a.copy() for a in args))
        def set_weights(self, values): self.weights.append(values.copy())
        def update_edges(self, edges, values): self.weights.append(values.copy())
        def advance(self, drive, cursor, steps, counts):
            counts[:] = [steps, 0, steps]
            return cursor + steps
        def download_state(self, *targets):
            saved = self.uploads[-1]
            for destination, source in zip(targets, (saved[0], saved[1], saved[2],
                                                    saved[3], saved[4], saved[6]), strict=True):
                destination[:] = source
    fake = FakeCuda()
    monkeypatch.setattr(cuda_engine, 'make_state', lambda *a, **k: fake)
    a = Brain(arrays=arrays(), dynamics='v3', backend='cuda')
    assert len(fake.uploads[0]) == 7
    a.weight = np.array([2, -2], dtype=np.float32)
    a.set_edge_weights([0], [3])
    assert a.weight[0] == 3 and a.compute_capabilities()['weight_mutation']
    a.restore_state(snapshot())
    counts, _ = a.step([1, 0, 2], 0.3)
    assert a.cursor == 10 and a.total_spikes == 25 and a.sim_ms == 1.0
    np.testing.assert_array_equal(counts, [3, 0, 3])
    assert len(fake.uploads[-1]) == 7


def test_cpu_import_does_not_load_amd_adapter_or_wgpu():
    code = '''import sys
import numpy as np
from brainlab.brain import Brain
a = dict(ptr=np.array([0,0], np.int64), post=np.zeros(0, np.int32),
         weight=np.zeros(0, np.float32), ids=np.array([1], np.int64))
Brain(arrays=a, dynamics="v3", backend="cpu")
assert "brainlab.amd_state_adapter" not in sys.modules
assert "brainlab.wgpu_v3" not in sys.modules
assert "wgpu" not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[1], env=dict(os.environ), timeout=15)
