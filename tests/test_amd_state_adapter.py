"""State/transport contract only: the fake device does not execute WGSL."""
import hashlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from brainlab.amd_state_adapter import (
    AmdStateUnavailable, AmdV3StateAdapter, STATE_ARRAYS,
    UnsupportedAmdCapability,
)
from brainlab import wgpu_v3


class Buffer:
    def __init__(self, data):
        self.data = bytearray(np.asarray(data).tobytes())


class ComputePass:
    def __init__(self, encoder, name):
        self.encoder, self.name = encoder, name

    def set_pipeline(self, pipeline):
        pass

    def set_bind_group(self, *args):
        pass

    def dispatch_workgroups(self, count):
        pass

    def end(self):
        self.encoder.commands.append(self.name)


class Encoder:
    def __init__(self):
        self.commands = []

    def begin_compute_pass(self, label):
        return ComputePass(self, label)

    def finish(self):
        return self


class Pipeline:
    def get_bind_group_layout(self, index):
        return index


class FakeDevice:
    """Byte transport plus deterministic test ticks, never a neural/GPU oracle."""
    features = wgpu_v3.REQUIRED_FEATURES
    limits = {'max-storage-buffers-per-shader-stage': 8,
              'max-buffer-size': 2**31,
              'max-storage-buffer-binding-size': 2**31 - 4,
              'max-compute-workgroups-per-dimension': 65535}

    def __init__(self):
        self.queue = self
        self.buffers = []
        self.writes = self.reads = self.submissions = 0
        self.fail = None

    def create_buffer_with_data(self, data, usage):
        buffer = Buffer(data)
        self.buffers.append(buffer)
        return buffer

    def create_shader_module(self, **kwargs):
        return kwargs

    def create_compute_pipeline(self, **kwargs):
        return Pipeline()

    def create_bind_group(self, **kwargs):
        return kwargs

    def create_command_encoder(self):
        return Encoder()

    def write_buffer(self, buffer, offset, data):
        self.writes += 1
        if self.fail == 'write' or (self.fail == 'second_write' and self.writes % 2 == 0):
            raise RuntimeError('injected transfer failure')
        raw = np.asarray(data).tobytes()
        buffer.data[offset:offset + len(raw)] = raw

    def read_buffer(self, buffer):
        self.reads += 1
        if self.fail == 'read':
            raise RuntimeError('injected read failure')
        return bytes(buffer.data)

    def submit(self, encoders):
        self.submissions += 1
        if self.fail == 'submit':
            raise RuntimeError('injected submit failure')
        n = int(np.frombuffer(self.buffers[0].data, dtype=np.uint32)[0])
        offsets, _ = wgpu_v3.layout(n)
        packed = np.frombuffer(self.buffers[2].data, dtype=np.uint32)
        counts = packed[offsets['counts']:offsets['counts'] + n].view(np.int32)
        drive = packed[offsets['drive']:offsets['drive'] + n].view(np.float32)
        v = packed[offsets['v']:offsets['v'] + n].view(np.float32)
        for encoder in encoders:
            for name in encoder.commands:
                if name == 'setup':
                    counts.fill(0)
                elif name == 'reset':
                    # Representative transport mutations only. This deliberately
                    # does not emulate or verify the retained WGSL arithmetic.
                    packed[1] += 1
                    counts[:] += (drive > 0).astype(np.int32)
                    v[:] += drive / 100


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setitem(sys.modules, 'wgpu', types.SimpleNamespace(
        BufferUsage=types.SimpleNamespace(STORAGE=1, COPY_SRC=2, COPY_DST=4)))
    return FakeDevice()


def adapter(device):
    return AmdV3StateAdapter(np.array([0, 1, 1, 2], dtype=np.int64),
                             np.array([2, 0], dtype=np.int32),
                             np.array([1, -1], dtype=np.float32), n=3, device=device)


def snapshot():
    queue = np.zeros((19, 3), dtype=np.int32)
    queue[3] = [2, 0, 1]
    queue_count = np.zeros(19, dtype=np.int32)
    queue_count[3] = 2
    return dict(v=np.array([-50, -51, -52], dtype=np.float32),
                g=np.arange(6, dtype=np.float32).reshape(2, 3) / 10,
                refractory=np.array([1, 2, 3], dtype=np.int16),
                queue=queue, queue_count=queue_count,
                counts=np.array([1, 0, 2], dtype=np.int32),
                active=np.array([2, 0, 1], dtype=np.int32),
                active_flag=np.array([1, 0, 1], dtype=np.uint8),
                nactive=np.array([2], dtype=np.int32),
                cursor=7, total_spikes=19, sim_ms=0.7, dynamics='v3')


def assert_exact(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], np.ndarray):
            assert left[name].dtype == right[name].dtype
            assert left[name].shape == right[name].shape
            assert left[name].tobytes() == right[name].tobytes(), name
        else:
            assert type(left[name]) is type(right[name])
            assert left[name] == right[name], name


def test_retained_engine_source_hashes():
    root = Path(__file__).resolve().parents[1]
    hashes = {'wgpu_v3.py': 'f2c362496034980091a3cb2b1bb14b7b0dca68d1a3fdf3ab588ff6c83bd7649b',
              'wgpu_v3_math.wgsl': 'a4f666380fc4ccb4b2737462a475484b0c37f14e71470bfb6a4d28355f8e494e'}
    for name, expected in hashes.items():
        assert hashlib.sha256((root / 'brainlab' / name).read_bytes()).hexdigest() == expected


def test_complete_roundtrip_preserves_order_tail_bits_and_ownership(device):
    a = adapter(device)
    original = snapshot()
    original['g'][0, 0] = np.float32(-0.0)
    a.restore_state(original)
    exported = a.snapshot_state()
    assert_exact(original, exported)
    for name in STATE_ARRAYS:
        assert not np.shares_memory(original[name], exported[name])
        original[name].fill(0)
        exported[name].fill(0)
    expected = snapshot()
    expected['g'][0, 0] = np.float32(-0.0)
    assert_exact(expected, a.snapshot_state())


def test_nonzero_clock_restore_continuation(device):
    uninterrupted = adapter(device)
    uninterrupted.restore_state(snapshot())
    drive = np.array([1, 0, 2], dtype=np.float32)
    uninterrupted.step(drive, 0.2)
    saved = uninterrupted.snapshot_state()
    resumed = adapter(FakeDevice())
    resumed.restore_state(saved)
    counts_a, _ = uninterrupted.step(drive, 0.3)
    counts_b, _ = resumed.step(drive, 0.3)
    assert_exact(uninterrupted.snapshot_state(), resumed.snapshot_state())
    np.testing.assert_array_equal(counts_a, [3, 0, 3])
    np.testing.assert_array_equal(counts_a, counts_b)
    state = resumed.snapshot_state()
    assert state['cursor'] == 12
    assert state['total_spikes'] == 29
    assert state['sim_ms'] == pytest.approx(1.2)
    counts_b.fill(0)
    np.testing.assert_array_equal(resumed.snapshot_state()['counts'], [3, 0, 3])


@pytest.mark.parametrize('bad', ['missing', 'version', 'learned', 'dtype', 'shape', 'nan',
                                'negative_g', 'queue_capacity', 'queue_member', 'active_order',
                                'active_flags', 'nactive', 'cursor', 'spikes', 'clock', 'refractory'])
def test_invalid_restore_never_writes_or_changes_valid_state(device, bad):
    a = adapter(device)
    a.restore_state(snapshot())
    before = a.snapshot_state()
    wrong = snapshot()
    if bad == 'missing': wrong.pop('cursor')
    elif bad == 'version': wrong['dynamics'] = 'v2'
    elif bad == 'learned': wrong['plastic_delta'] = np.zeros(2, dtype=np.float32)
    elif bad == 'dtype': wrong['v'] = wrong['v'].astype(np.float64)
    elif bad == 'shape': wrong['nactive'] = np.zeros(2, dtype=np.int32)
    elif bad == 'nan': wrong['v'][0] = np.nan
    elif bad == 'negative_g': wrong['g'][0, 0] = -1
    elif bad == 'queue_capacity': wrong['queue_count'][0] = 4
    elif bad == 'queue_member': wrong['queue'][3, 0] = 3
    elif bad == 'active_order': wrong['active'][:2] = [0, 0]
    elif bad == 'active_flags': wrong['active_flag'][0] = 2
    elif bad == 'nactive': wrong['nactive'][0] = 4
    elif bad == 'cursor': wrong['cursor'] = 2**32 - 19
    elif bad == 'spikes': wrong['total_spikes'] = True
    elif bad == 'clock': wrong['sim_ms'] = float('inf')
    elif bad == 'refractory': wrong['refractory'][0] = 23
    writes, submissions = device.writes, device.submissions
    with pytest.raises((ValueError, UnsupportedAmdCapability)):
        a.restore_state(wrong)
    assert (device.writes, device.submissions) == (writes, submissions)
    assert a.status['can_export']
    assert_exact(before, a.snapshot_state())


def test_all_mutation_learning_refusals_preserve_host_and_device(device):
    a = adapter(device)
    a.restore_state(snapshot())
    before = a.snapshot_state()
    buffers = [bytes(b.data) for b in device.buffers]
    weights = a.weight.copy()
    calls = [lambda: setattr(a, 'weight', np.zeros(2, dtype=np.float32)),
             lambda: a.set_weights(np.zeros(2, dtype=np.float32)),
             lambda: a.update_edges([0], [0]), lambda: a.set_edge_weights([0], [0]),
             lambda: a.update_weights([0]), lambda: a.set_learning(True),
             lambda: a.set_learning(False)]
    for call in calls:
        with pytest.raises(UnsupportedAmdCapability): call()
        np.testing.assert_array_equal(a.weight, weights)
        assert [bytes(b.data) for b in device.buffers] == buffers
        assert_exact(a.snapshot_state(), before)
    exposed = a.weight
    assert not exposed.flags.writeable
    exposed.setflags(write=True)
    exposed.fill(0)
    np.testing.assert_array_equal(a.weight, weights)
    assert not a.capabilities['learning'] and not a.capabilities['weight_mutation']
    with pytest.raises(TypeError): a.capabilities['learning'] = True


@pytest.mark.parametrize('failure', ['write', 'second_write', 'submit', 'read', 'arithmetic'])
def test_failed_transfers_and_arithmetic_block_export_until_explicit_restore(device, failure):
    a = adapter(device)
    saved = snapshot()
    a.restore_state(saved)
    device.fail = failure
    with pytest.raises(RuntimeError):
        if failure in ('write', 'second_write'):
            changed = snapshot()
            changed['v'][0] = -49
            a.restore_state(changed)
        elif failure == 'arithmetic':
            device.buffers[2].data[:4] = np.array([1], dtype=np.uint32).tobytes()
            a.snapshot_state()
        else:
            a.step(np.ones(3, dtype=np.float32), 0.1)
    assert a.status['poisoned'] and not a.status['can_export']
    activity = (device.writes, device.reads, device.submissions)
    with pytest.raises(AmdStateUnavailable): a.snapshot_state()
    with pytest.raises(AmdStateUnavailable): a.step(np.ones(3, dtype=np.float32), 0.1)
    assert activity == (device.writes, device.reads, device.submissions)
    device.fail = None
    a.restore_state(saved)
    assert_exact(saved, a.snapshot_state())


def test_unloaded_state_and_explicit_reset(device):
    a = adapter(device)
    assert not a.status['can_export']
    with pytest.raises(AmdStateUnavailable): a.snapshot_state()
    with pytest.raises(AmdStateUnavailable): a.step(np.zeros(3, dtype=np.float32), 0.1)
    a.restore_state(snapshot())
    device.fail = 'read'
    with pytest.raises(RuntimeError): a.snapshot_state()
    device.fail = None
    a.reset_state()
    state = a.snapshot_state()
    for name in STATE_ARRAYS:
        np.testing.assert_array_equal(state[name], -52 if name == 'v' else 0)
    assert state['cursor'] == state['total_spikes'] == state['sim_ms'] == 0
    assert not a.status['poisoned'] and a.status['can_advance']


@pytest.mark.parametrize('bad', ['drive_dtype', 'drive_nan', 'duration', 'fractional', 'rounded', 'cursor'])
def test_bad_step_preserves_valid_state_before_transfer(device, bad):
    a = adapter(device)
    state = snapshot()
    if bad == 'cursor': state['cursor'] = 2**32 - 20
    a.restore_state(state)
    drive, duration = np.ones(3, dtype=np.float32), 0.1
    if bad == 'drive_dtype': drive = drive.astype(np.float64)
    elif bad == 'drive_nan': drive[0] = np.nan
    elif bad == 'duration': duration = 102.5
    elif bad == 'fractional': duration = 0.15
    elif bad == 'rounded': duration = 100.00000005
    before = (device.writes, device.submissions)
    with pytest.raises(ValueError): a.step(drive, duration)
    assert (device.writes, device.submissions) == before
    assert_exact(state, a.snapshot_state())
