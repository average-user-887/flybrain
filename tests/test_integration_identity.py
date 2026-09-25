"""Integration of WP1-WP4 into the arena, daemon and telemetry.

* every telemetry packet, /api/status and each switch ack carry the run identity;
* an RPC fault under the default ``halt`` policy gives exactly zero motion;
* Arena.snapshot_world / restore_world round-trip and continue deterministically;
* engineered motor assists default by backend (on: modular, off: graph and RPC);
* the modular default never loads the graph;
* packets from before an acknowledged switch are rejected (identity_rejection).

Graph-backend runs use the SYNTHETIC TEST GRAPH (explicit test option, labelled).
"""
import json
import math
import random
import socket
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import experiment_registry
from arena import Arena, WorldStateError
from connectome_bridge import ConnectomeBridge, RpcControllerFault
from experiment_brains import PARADIGMS
from neurofly_daemon import ContinuousExperimentRunner, NeuroflyHTTPHandler, identity_rejection

IDENTITY_KEYS = {'run_id', 'instance_id', 'backend', 'graph_sha256', 'synthetic', 'label', 'activation',
                 'daemon_run_id', 'assay', 'test_mode'}


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _run(arena, steps, dt=0.02):
    return [(r['fly_x'], r['fly_y'], r['fly_heading'], r['fly_speed']) for r in (arena.step(dt) for _ in range(steps))]


def _graph_runner(tmp_path, paradigm='optomotor', backend='connectome-fixed'):
    return ContinuousExperimentRunner(initial_paradigm=paradigm, sim_speed=1, checkpoint_interval=3600,
                                      output_dir=tmp_path, backend=backend, test_synthetic_graph=True)


# ---------------------------------------------------------------------------
# Identity on packets, status and acks
# ---------------------------------------------------------------------------
def test_modular_packets_carry_identity_and_manifest(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm='wind-tunnel', sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path)
    with runner.lock:
        for _ in range(5):
            runner.step_once()
    packet = json.loads(runner._publish_snapshot().data)
    ident = packet['identity']
    assert IDENTITY_KEYS <= set(ident)
    assert ident['backend'] == 'modular' and ident['graph_sha256'] is None and ident['synthetic'] is False
    assert ident['instance_id'] == runner.active_brain.brain_id and ident['daemon_run_id'] == packet['run_id']
    assert packet['motor']['motor_assists_enabled'] is True
    assert packet['motor']['motor_source'] == 'modular' and packet['motor']['motor_source_normal'] is True
    assert packet['controller_fault'] is None and packet['motor']['record']['step'] == 5
    manifest_file = tmp_path / 'manifests' / f"{ident['run_id']}.json"
    assert json.loads(manifest_file.read_text())['backend'] == 'modular'
    # Checkpoint exports carry the same identity and the full manifest.
    saved = json.loads(runner.save_checkpoint('t').read_text())
    assert saved['identity']['run_id'] == ident['run_id'] and saved['manifest']['run_id'] == ident['run_id']


def test_status_and_manifest_endpoints_carry_identity(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm='t-maze', sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path)
    runner._publish_snapshot()
    handler = type('H', (NeuroflyHTTPHandler,), {'runner': runner})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}'
        status = json.loads(urllib.request.urlopen(base + '/api/status', timeout=5).read())
        manifest = json.loads(urllib.request.urlopen(base + '/api/manifest', timeout=5).read())
    finally:
        server.shutdown()
        server.server_close()
    assert IDENTITY_KEYS <= set(status['identity']) and status['backend'] == 'modular'
    assert status['motor']['motor_assists'] == {'wall_avoidance_reflex': True, 'contact_turn': True}
    assert 'controller_fault' in status
    assert manifest['manifest']['run_id'] == status['identity']['run_id']


def test_graph_backend_identity_and_ack_after_ready(tmp_path):
    runner = _graph_runner(tmp_path)
    with runner.lock:
        for _ in range(4):
            runner.step_once()
    before = json.loads(runner._publish_snapshot().data)
    ident = before['identity']
    assert ident['backend'] == 'connectome-fixed' and ident['synthetic'] is True and ident['test_mode'] is True
    assert 'SYNTHETIC' in ident['label'] and ident['graph_sha256']
    assert ident['instance_id'] == runner.registry.active.instance_id
    # No verified sensory/motor mapping in the live arena: zero motion, flagged source.
    assert before['motor']['motor_source'] == 'graph-unmapped-io' and before['motor']['motor_source_normal'] is False
    assert before['motor']['motor_assists_enabled'] is False and before['fly']['speed'] == 0.0

    result = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 't-maze'})
    ack = result['ack']
    assert result['status'] == 'ok' and ack['identity']['assay'] == 't-maze'
    assert ack['identity']['activation'] == ident['activation'] + 1
    # At ack time the registry's active instance IS the acknowledged one.
    assert runner.registry.is_current_packet(ack['identity'])
    after = json.loads(runner._publish_snapshot().data)
    assert after['identity']['instance_id'] == ack['identity']['instance_id']


# ---------------------------------------------------------------------------
# Stale-identity rejection (Python mirror of web/app.js identityRejection)
# ---------------------------------------------------------------------------
def test_stale_identity_packets_are_rejected_after_switch_ack(tmp_path):
    runner = _graph_runner(tmp_path)
    with runner.lock:
        runner.step_once()
    old = json.loads(runner._publish_snapshot().data)
    ack = runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 'buridan'})['ack']
    with runner.lock:
        runner.step_once()
    new = json.loads(runner._publish_snapshot().data)

    assert identity_rejection(old['identity'], old['run_id'], ack).startswith('stale')
    assert identity_rejection(new['identity'], new['run_id'], ack) is None
    forged = dict(new['identity'], instance_id='someone-else')
    assert 'differ' in identity_rejection(forged, new['run_id'], ack)
    # A later switch (e.g. from another dashboard tab) and a restarted daemon are accepted.
    later = dict(new['identity'], activation=ack['identity']['activation'] + 1, run_id='r2', instance_id='i2')
    assert identity_rejection(later, new['run_id'], ack) is None
    assert identity_rejection(old['identity'], 'another-daemon-process', ack) is None
    assert identity_rejection(new['identity'], new['run_id'], None) is None


# ---------------------------------------------------------------------------
# Halt on RPC fault
# ---------------------------------------------------------------------------
class _FlakyClient:
    """Stands in for ConnectomeClient: ``good`` ok steps, then failures."""

    def __init__(self, good):
        self.good = good
        self.is_connected = True
        self.last_error = None

    def step(self, sensory, duration_ms=2.0):
        if self.good > 0:
            self.good -= 1
            return dict(status='ok', dna02_diff=4.0, dna02_rate_l=20.0, dna02_rate_r=16.0, dnp09_rate=40.0,
                        bpn_rate=10.0, mdn_rate=0.0, dnp01_gf_spikes=0)
        self.last_error = 'step failed: injected'
        return None


@pytest.mark.parametrize('paradigm', [None, 'wind-tunnel', 'looming-escape'])
def test_rpc_fault_halt_gives_zero_motion(paradigm):
    arena = Arena(paradigm=paradigm, brain_type='connectome', connectome_mode='rpc',
                  connectome_host='127.0.0.1', connectome_port=_closed_port(), seed=3, num_predators=0)
    fly = arena.fly
    assert fly.connectome_bridge.on_rpc_fault == 'halt'
    assert arena.controller_backend == 'hybrid-bridge-rpc-experimental'
    assert arena.motor_assists == {'wall_avoidance_reflex': False, 'contact_turn': False}
    fly.connectome_bridge.rpc_client = _FlakyClient(good=15)
    moving = _run(arena, 15)
    assert math.hypot(moving[-1][0] - moving[0][0], moving[-1][1] - moving[0][1]) > 0.0 or paradigm == 'looming-escape'
    start = (fly.pos.x, fly.pos.y, fly.heading)
    halted = _run(arena, 60)
    assert all((x, y, h) == start and s == 0.0 for x, y, h, s in halted)
    assert fly.motor_source == 'halted-rpc-fault' and fly.motor_halted
    prov = arena.motor_provenance()
    assert prov['motor_source_normal'] is False and prov['controller_fault'] == 'step failed: injected'
    assert fly.motor_record['halted'] is True and fly.motor_record['realized_mm'] == 0.0


def test_connectome_on_fault_parameter_reaches_bridge():
    with pytest.raises(RpcControllerFault):
        Arena(brain_type='connectome', connectome_mode='rpc', connectome_host='127.0.0.1',
              connectome_port=_closed_port(), connectome_on_fault='raise')
    arena = Arena(brain_type='connectome', connectome_mode='surrogate', connectome_on_fault='surrogate-test')
    assert isinstance(arena.fly.connectome_bridge, ConnectomeBridge)
    assert arena.fly.connectome_bridge.on_rpc_fault == 'surrogate-test'


# ---------------------------------------------------------------------------
# Motor assists by backend
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kwargs,backend,expected', [
    (dict(), 'modular', True),
    (dict(brain_type='connectome', connectome_mode='surrogate'), 'bridge-surrogate', True),
    (dict(controller_backend='connectome-fixed'), 'connectome-fixed', False),
    (dict(controller_backend='connectome-plastic'), 'connectome-plastic', False),
    (dict(controller_backend='connectome-with-trained-readout'), 'connectome-with-trained-readout', False),
])
def test_motor_assists_default_by_backend(kwargs, backend, expected):
    arena = Arena(paradigm='t-maze', **kwargs)
    assert arena.controller_backend == backend
    assert arena.motor_assists == {name: expected for name in Arena.MOTOR_ASSISTS}
    # Explicit settings still win.
    override = Arena(paradigm='t-maze', motor_assists={'contact_turn': not expected}, **kwargs)
    assert override.motor_assists['contact_turn'] is (not expected)


# ---------------------------------------------------------------------------
# World snapshots
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('paradigm', PARADIGMS)
def test_world_snapshot_round_trip_continues_deterministically(paradigm):
    name = None if paradigm == 'open-arena' else paradigm

    def make():
        return Arena(paradigm=name, seed=11, num_flies=1, num_predators=0)

    reference = make()
    _run(reference, 40)
    snapshot = json.loads(json.dumps(reference.snapshot_world(), allow_nan=False))   # JSON-safe
    # Global RNG state must not matter: vision draws from the arena's own generator.
    np.random.seed(1)
    random.seed(1)
    expected = _run(reference, 250)

    np.random.seed(999)
    random.seed(999)
    fresh = make()
    _run(fresh, 7)                                  # a different state before the restore
    fresh.restore_world(snapshot)
    assert _run(fresh, 250) == expected
    assert fresh.fly.vision.rng is fresh.np_rng and fresh.circuit is fresh.fly.circuit

    reference.restore_world(snapshot)               # in place, same object
    assert _run(reference, 250) == expected
    assert snapshot['summary']['flies'][0].keys() >= {'x', 'y', 'heading', 'speed', 'angular_velocity'}
    assert snapshot['summary']['rng']['np_rng']['bit_generator'] == 'PCG64'


def test_world_snapshot_refuses_foreign_paradigm():
    snap = Arena(paradigm='t-maze').snapshot_world()
    with pytest.raises(WorldStateError):
        Arena(paradigm='y-maze').restore_world(snap)
    with pytest.raises(WorldStateError):
        Arena(paradigm='t-maze').restore_world({'format': 'something-else'})


def test_vision_escape_jitter_uses_arena_rng():
    first, second = Arena(seed=5), Arena(seed=5)
    np.random.seed(1)
    a = first.fly.vision.rng.uniform()
    np.random.seed(2)
    b = second.fly.vision.rng.uniform()
    assert a == b and first.fly.vision.rng is first.np_rng


def test_graph_switch_restores_world_and_checkpoint_has_world_state(tmp_path):
    runner = _graph_runner(tmp_path, paradigm='t-maze')
    with runner.lock:
        runner.arena.fly.pos.x += 3.0            # a distinctive world state
        for _ in range(5):
            runner.step_once()
    pose = (runner.arena.fly.pos.x, runner.arena.fly.pos.y, runner.arena.time_step)
    instance_id = runner.registry.active.instance_id
    runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 'y-maze'})
    # A fresh process sees the t-maze world only through the checkpoint.
    runner._graph_arenas.clear()
    runner.dispatch_command({'action': 'switch_paradigm', 'paradigm': 't-maze'})
    assert runner.registry.active.instance_id == instance_id
    assert (runner.arena.fly.pos.x, runner.arena.fly.pos.y, runner.arena.time_step) == pose
    saved = json.loads(runner.save_checkpoint('t').read_text())
    meta, _ = runner.registry.read_checkpoint(instance_id)
    assert saved['graph_checkpoint'] and meta['world_state']['format'] == Arena.WORLD_STATE_FORMAT
    assert meta['world_state']['state']['time_step'] == runner.arena.time_step


def test_periodic_daemon_checkpoints_keep_only_the_newest(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROFLY_KEEP_CHECKPOINTS', '2')
    runner = _graph_runner(tmp_path, paradigm='t-maze')
    assert runner.registry.keep_checkpoints == 2
    with runner.lock:
        manual = runner.save_checkpoint('manual')
        for _ in range(5):
            runner.step_once()
            last = runner.save_checkpoint('periodic')
    instance_id = runner.registry.active.instance_id
    npz = sorted(p.name for p in (runner.registry.instance_dir(instance_id) / 'checkpoints').iterdir())
    assert npz == ['ckpt-000005.npz', 'ckpt-000006.npz']
    periodic = sorted(runner.checkpoints_dir.glob('checkpoint_t-maze_periodic_*.json'))
    assert len(periodic) == 2 and last in periodic and manual.exists()
    saved = json.loads(last.read_text())
    meta, _ = runner.registry.read_checkpoint(instance_id)
    assert saved['graph_checkpoint'].endswith('ckpt-000006.npz') and meta['version'] == 6


# ---------------------------------------------------------------------------
# The modular default never touches the graph
# ---------------------------------------------------------------------------
def test_modular_default_does_not_load_graph(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('the modular default must not load the graph')

    monkeypatch.setattr(experiment_registry.SharedGraph, 'load', classmethod(forbidden))
    monkeypatch.setattr(experiment_registry.SharedGraph, 'synthetic', classmethod(forbidden))
    runner = ContinuousExperimentRunner(initial_paradigm='buridan', sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path)
    with runner.lock:
        runner.step_once()
    assert runner.backend == 'modular' and runner.registry is None and runner.shared_graph is None
    assert runner.arena.graph_controller is None


def test_graph_backend_without_graph_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROFLY_GRAPH_DIR', raising=False)
    from brainlab.graph_identity import GraphUnavailable
    with pytest.raises(GraphUnavailable):
        ContinuousExperimentRunner(initial_paradigm='optomotor', output_dir=tmp_path, backend='connectome-fixed',
                                   graph_dir=tmp_path / 'missing')
