"""WP4: controller provenance, graph identity, explicit backends and the
per-assay experiment registry.

Registry isolation tests run on a SYNTHETIC TEST GRAPH (explicitly labelled,
test mode only).  The real-graph check runs only when NEUROFLY_GRAPH_DIR is set.
"""
import hashlib
import json
import os
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import provenance
from brainlab.brain import Brain
from brainlab.cosim_server import CoSimHTTPHandler, ConnectomeServer
from brainlab.graph_identity import (GraphUnavailable, load_pins, synthetic_test_graph, verify_graph)
from connectome_bridge import ConnectomeBridge, RpcControllerFault
from connectome_client import ConnectomeClient
from experiment_registry import (PARADIGMS, CheckpointCorrupt, ExperimentRegistry, IncompatibleCheckpoint,
                                 SharedGraph, TestOnlyCoactivityRule, load_checkpoint_file)
from provenance import BackendError, RunManifest, get_backend, restore_rng, rng_state

REQUIRED_MANIFEST_FIELDS = {
    'backend', 'assay', 'instance_id', 'run_id', 'seed', 'graph', 'dynamics', 'controller_version',
    'source', 'rng_initial_state', 'learned_parameter_locations', 'intervention_schedule', 'events',
    'synthetic', 'test_mode', 'label',
}


# ---------------------------------------------------------------------------
# Graph identity
# ---------------------------------------------------------------------------
def test_missing_graph_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROFLY_GRAPH_DIR', raising=False)
    with pytest.raises(GraphUnavailable, match='NEUROFLY_GRAPH_DIR'):
        verify_graph(tmp_path)


def test_wrong_graph_hash_is_rejected(tmp_path):
    arrays, _, _ = synthetic_test_graph(n=8, k_out=2)
    np.savez(tmp_path / 'graph.npz', **arrays)
    (tmp_path / 'normalized').mkdir()
    (tmp_path / 'normalized/neurons.feather').write_bytes(b'x')
    with pytest.raises(GraphUnavailable, match='hash mismatch'):
        verify_graph(tmp_path, tmp_path)


@pytest.mark.skipif(not os.environ.get('NEUROFLY_GRAPH_DIR'), reason='real graph location not configured')
def test_real_graph_matches_pins():
    identity = verify_graph()
    pins = load_pins()
    assert identity.graph_sha256 == pins['graph_sha256']
    assert identity.neurons == 166700 and identity.edges == 25582938


def test_server_refuses_missing_graph_and_labels_synthetic(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROFLY_GRAPH_DIR', raising=False)
    with pytest.raises(GraphUnavailable):
        ConnectomeServer(graph_dir=tmp_path)
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    status = server.get_status()
    assert status['synthetic'] is True and status['backend'] == 'synthetic-test-graph'
    assert 'SYNTHETIC' in status['label'] and status['unmapped_channels']
    reply = server.step({}, duration_ms=2.0)
    assert reply['synthetic'] is True and reply['graph_sha256'] == status['graph_sha256']
    assert not list(tmp_path.iterdir()), 'synthetic graph must not be written to disk'


def test_client_rejects_synthetic_server_unless_allowed(tmp_path):
    server = ThreadingHTTPServer(('127.0.0.1', 0), CoSimHTTPHandler)
    server.connectome = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        strict = ConnectomeClient(host='127.0.0.1', port=port, timeout=2.0)
        assert strict.is_connected is False and 'synthetic' in strict.last_error
        assert strict.step({}) is None
        assert any(e['kind'] == 'identity_mismatch' for e in strict.events)
        test_client = ConnectomeClient(host='127.0.0.1', port=port, timeout=2.0, allow_synthetic=True)
        assert test_client.is_connected and test_client.server_identity['synthetic'] is True
        assert test_client.step({}, duration_ms=2.0)['status'] == 'ok'
    finally:
        server.shutdown()
        server.server_close()


def test_client_uses_env_host_when_host_is_none(monkeypatch):
    monkeypatch.setenv('NEUROFLY_CONNECTOME_HOST', '127.0.0.1')
    client = ConnectomeClient(host=None, port=9, timeout=0.05)
    assert client.base_url == 'http://127.0.0.1:9'


# ---------------------------------------------------------------------------
# Backends and manifest
# ---------------------------------------------------------------------------
def test_backends_are_distinct_named_identities():
    for name in ('modular', 'connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout'):
        assert get_backend(name).scientific
    versions = [spec.controller_version for spec in provenance.BACKENDS.values()]
    assert len(versions) == len(set(versions))
    with pytest.raises(BackendError):
        get_backend('connectome')
    with pytest.raises(BackendError):
        get_backend('bridge-surrogate')          # not scientific without a test option
    assert get_backend('bridge-surrogate', allow_test=True).name == 'bridge-surrogate'


def test_manifest_contents_and_synthetic_requires_test_mode(tmp_path):
    _, identity, _ = synthetic_test_graph()
    kwargs = dict(backend='connectome-fixed', assay='optomotor', instance_id='x', seed=3,
                  graph=identity.to_dict(), dynamics={'dt_ms': 0.1}, learned_parameter_locations={},
                  rng=np.random.default_rng(3), source={'commit': None})
    with pytest.raises(BackendError):
        RunManifest.create(**kwargs)
    with pytest.raises(BackendError):
        RunManifest.create(**dict(kwargs, graph=None, test_mode=True))
    manifest = RunManifest.create(**kwargs, test_mode=True)
    assert manifest.synthetic and 'SYNTHETIC' in manifest.label
    manifest.record_event('rpc_fault', step=7, reason='test')
    path = tmp_path / 'manifest.json'
    manifest.write(path)
    data = json.loads(path.read_text())
    assert REQUIRED_MANIFEST_FIELDS <= set(data)
    assert RunManifest.read(path).events[0]['kind'] == 'rpc_fault'
    assert manifest.identity()['graph_sha256'] == identity.graph_sha256


def test_rng_state_roundtrip():
    rng = np.random.default_rng(123)
    rng.random(5)
    clone = restore_rng(json.loads(json.dumps(rng_state(rng))))
    np.testing.assert_array_equal(rng.random(10), clone.random(10))


# ---------------------------------------------------------------------------
# Bridge fault semantics
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.is_connected = True
        self.last_error = None
        self.server_identity = {'graph_sha256': 'f' * 64, 'synthetic': False}

    def step(self, sensory, duration_ms=2.0):
        reply = self.replies.pop(0)
        if reply is None:
            self.last_error = 'step failed: connection refused'
        return reply

    def reset(self):
        return True


def _graph_reply(**rates):
    reply = dict(status='ok', dna02_diff=0.0, dna02_rate_l=0.0, dna02_rate_r=0.0, dnp09_rate=0.0,
                 bpn_rate=0.0, mdn_rate=0.0, dnp01_gf_spikes=0)
    reply.update(rates)
    return reply


def _step(bridge):
    return bridge.step(fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
                       odor_left=0.5, odor_right=0.5, wind_vector=np.array([0.0, 0.0]), dt=0.02)


def test_rpc_disconnect_halts_and_is_reported():
    bridge = ConnectomeBridge(mode='rpc', rpc_client=FakeClient([_graph_reply(), None, None]))
    out = _step(bridge)
    assert out['controller_backend'] == 'hybrid-bridge-rpc-experimental'
    assert out['motor_source'] == 'graph-rpc' and out['controller_fault'] is None
    assert out['dnp09_rate'] == 0.0 and out['bpn_rate'] == 0.0   # graph zeros are applied
    for _ in range(2):
        out = _step(bridge)
        assert out['motor_source'] == 'halted-rpc-fault'
        assert out['forward_speed'] == 0.0 and out['yaw_rate'] == 0.0
        assert 'connection refused' in out['controller_fault']
    faults = [e for e in bridge.controller_events if e['kind'] == 'rpc_fault']
    assert len(faults) == 1 and bridge.rpc_fault_steps == 2


def test_rpc_fault_raise_and_explicit_test_fallback():
    bridge = ConnectomeBridge(mode='rpc', rpc_client=FakeClient([None]), on_rpc_fault='raise')
    with pytest.raises(RpcControllerFault):
        _step(bridge)
    bridge = ConnectomeBridge(mode='rpc', rpc_client=FakeClient([None]), on_rpc_fault='surrogate-test')
    out = _step(bridge)
    assert out['motor_source'] == 'surrogate-fallback-TEST' and out['controller_fault']


def test_offline_rpc_is_not_silently_downgraded():
    bridge = ConnectomeBridge(mode='rpc', rpc_host='127.0.0.1', rpc_port=9)
    assert bridge.mode == 'rpc' and bridge.controller_fault
    assert _step(bridge)['motor_source'] == 'halted-rpc-fault'
    with pytest.raises(RpcControllerFault):
        ConnectomeBridge(mode='rpc', rpc_host='127.0.0.1', rpc_port=9, on_rpc_fault='raise')
    assert ConnectomeBridge().controller_identity()['backend'] == 'bridge-surrogate'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def make_registry(root, **kwargs):
    shared = SharedGraph.synthetic(allow_synthetic=True, n=64, k_out=6, seed=5)
    rule = TestOnlyCoactivityRule(edges=np.arange(0, 384, 7), eta=0.05)
    return ExperimentRegistry(shared, root, test_mode=True, plasticity_rule=rule, **kwargs)


def drive(instance, steps):
    trace = []
    for _ in range(steps):
        currents = np.zeros(instance.shared.n, dtype=np.float32)
        chosen = instance.rng.choice(instance.shared.n, size=12, replace=False)
        currents[chosen] = instance.rng.uniform(8.0, 30.0, size=12)
        result = instance.step(currents, 5.0, target=instance.step_index % 2)
        trace.append(result.counts.copy())
    return trace


def state_of(instance):
    arrays = instance.state_arrays()
    return arrays, instance.rng.bit_generator.state, instance.step_index


def assert_same_state(a, b):
    arrays_a, rng_a, step_a = a
    arrays_b, rng_b, step_b = b
    assert step_a == step_b and rng_a == rng_b and set(arrays_a) == set(arrays_b)
    for key in arrays_a:
        np.testing.assert_array_equal(arrays_a[key], arrays_b[key], err_msg=key)


def test_synthetic_registry_requires_test_mode(tmp_path):
    shared = SharedGraph.synthetic(allow_synthetic=True)
    with pytest.raises(BackendError):
        ExperimentRegistry(shared, tmp_path)
    with pytest.raises(BackendError):
        SharedGraph.synthetic()
    registry = ExperimentRegistry(shared, tmp_path / 'r', test_mode=True)
    with pytest.raises(BackendError, match='WP6'):
        registry.instance_id_for('optomotor', 'connectome-plastic')
    with pytest.raises(BackendError):
        registry.instance_id_for('optomotor', 'modular')


def test_all_14_assays_get_distinct_instances_and_paths(tmp_path):
    from experiment_brains import PARADIGMS as MODULAR_PARADIGMS
    assert PARADIGMS == MODULAR_PARADIGMS and len(PARADIGMS) == 14
    registry = make_registry(tmp_path)
    ids = [registry.instance_id_for(assay, 'connectome-fixed') for assay in PARADIGMS]
    dirs = {registry.instance_dir(i) for i in ids}
    assert len(set(ids)) == 14 and len(dirs) == 14
    for instance_id in ids:
        manifest = registry.manifest(instance_id)
        assert manifest.synthetic and manifest.test_mode and 'SYNTHETIC' in manifest.label
        assert REQUIRED_MANIFEST_FIELDS <= set(manifest.to_dict())
    # Graph arrays are shared and immutable.
    a = registry.activate('optomotor', 'connectome-fixed')
    assert a.brain.weight is registry.shared.arrays['weight'] and not a.brain.weight.flags.writeable


@pytest.mark.parametrize('backend', ['connectome-plastic', 'connectome-with-trained-readout'])
def test_training_a_leaves_b_saved_state_unchanged(tmp_path, backend):
    registry = make_registry(tmp_path)
    b = registry.activate('t-maze', backend)
    drive(b, 6)
    b_id = b.instance_id
    b_dir = registry.instance_dir(b_id)

    a = registry.activate('optomotor', backend)           # checkpoints and releases B
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (b_dir / 'checkpoints').iterdir()}
    pointer = registry.current_pointer(b_id)
    meta_before, arrays_before = registry.read_checkpoint(b_id)
    assert meta_before['step_index'] == 6
    assert b.brain is None                              # released, cannot train in background
    with pytest.raises(RuntimeError):
        b.step(np.zeros(registry.shared.n, dtype=np.float32), 5.0)
    drive(a, 20)
    learned = a.plastic_delta if backend == 'connectome-plastic' else a.readout.weights
    assert np.any(learned != 0), 'instance A must actually learn for this test to mean anything'
    registry.checkpoint()

    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (b_dir / 'checkpoints').iterdir()}
    assert after == before and registry.current_pointer(b_id) == pointer
    meta_after, arrays_after = registry.read_checkpoint(b_id)
    assert meta_after == meta_before
    for key in arrays_before:
        np.testing.assert_array_equal(arrays_before[key], arrays_after[key])
    # Shared graph weights were never modified by A's plasticity.
    np.testing.assert_array_equal(registry.shared.arrays['weight'], synthetic_test_graph(n=64, k_out=6, seed=5)[0]['weight'])


@pytest.mark.parametrize('backend', ['connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout'])
def test_a_b_a_continuation_matches_uninterrupted_reference(tmp_path, backend):
    reference_registry = make_registry(tmp_path / 'reference')
    reference = reference_registry.activate('optomotor', backend)
    reference_trace = drive(reference, 24)
    reference_state = state_of(reference)

    registry = make_registry(tmp_path / 'switched')
    a = registry.activate('optomotor', backend)
    assert a.seed == reference.seed
    trace = drive(a, 10)
    b = registry.activate('buridan', backend)
    drive(b, 15)
    a = registry.activate('optomotor', backend)                   # restored from checkpoint
    trace += drive(a, 14)

    for expected, observed in zip(reference_trace, trace):
        np.testing.assert_array_equal(expected, observed)
    assert_same_state(reference_state, state_of(a))
    kinds = [json.loads(line)['kind'] for line in
             (registry.instance_dir(a.instance_id) / 'events.jsonl').read_text().splitlines()]
    assert kinds == ['checkpoint', 'restore']


def test_world_state_and_stale_packets(tmp_path):
    registry = make_registry(tmp_path)
    a = registry.activate('labyrinth', 'connectome-fixed')
    registry.checkpoint(world_state={'pose': [1.0, 2.0, 0.5], 'arena_rng': 'opaque'})
    identity_a = a.identity
    registry.activate('courtship', 'connectome-fixed')
    assert not registry.is_current_packet(identity_a)
    restored = registry.activate('labyrinth', 'connectome-fixed')
    assert restored.world_state == {'pose': [1.0, 2.0, 0.5], 'arena_rng': 'opaque'}
    assert registry.is_current_packet(restored.identity)


def test_interrupted_write_keeps_last_valid_checkpoint(tmp_path, monkeypatch):
    registry = make_registry(tmp_path)
    a = registry.activate('heat-maze', 'connectome-plastic')
    drive(a, 4)
    registry.checkpoint()
    good = registry.current_pointer(a.instance_id)
    drive(a, 4)
    real_replace = os.replace

    def crash_on_pointer(src, dst):
        if str(dst).endswith('CURRENT.json'):
            raise OSError('simulated power loss')
        return real_replace(src, dst)

    monkeypatch.setattr(provenance.os, 'replace', crash_on_pointer)
    with pytest.raises(OSError):
        registry.checkpoint()
    monkeypatch.setattr(provenance.os, 'replace', real_replace)
    assert registry.current_pointer(a.instance_id) == good
    meta, _ = registry.read_checkpoint(a.instance_id)
    assert meta['version'] == good['version'] and meta['step_index'] == 4
    assert not [p for p in registry.instance_dir(a.instance_id).rglob('*.partial')]


def test_corrupt_and_foreign_checkpoints_are_refused(tmp_path):
    registry = make_registry(tmp_path / 'r')
    a = registry.activate('y-maze', 'connectome-fixed')
    path = registry.checkpoint()
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CheckpointCorrupt):
        registry.read_checkpoint(a.instance_id)

    modular = tmp_path / 'optomotor.json'
    modular.write_text(json.dumps({'w': [0.1, 0.2], 'model': 'modular-mushroom-body'}))
    with pytest.raises(IncompatibleCheckpoint, match='never reinterpreted'):
        load_checkpoint_file(modular)
    weights_only = tmp_path / 'weights.npz'
    np.savez(weights_only, w=np.ones(4))
    with pytest.raises(IncompatibleCheckpoint):
        load_checkpoint_file(weights_only)

    other = SharedGraph.synthetic(allow_synthetic=True, n=32, k_out=4, seed=9)
    with pytest.raises(IncompatibleCheckpoint, match='different graph'):
        ExperimentRegistry(other, tmp_path / 'r', test_mode=True)


def test_brain_snapshot_restore_is_exact():
    arrays, _, _ = synthetic_test_graph(n=40, k_out=5, seed=2)
    one, two = Brain(arrays=arrays), Brain(arrays=arrays)
    drive_vec = np.linspace(0, 25, 40).astype(np.float32)
    one.step(drive_vec, 3.0)
    two.restore_state(one.snapshot_state())
    np.testing.assert_array_equal(one.step(drive_vec, 4.0)[0], two.step(drive_vec, 4.0)[0])
    np.testing.assert_array_equal(one.v, two.v)
