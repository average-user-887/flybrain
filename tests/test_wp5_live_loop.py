"""WP5 optomotor loop wired into the live simulation path (docs/WP5_OPTOMOTOR.md §8).

Covers the arena hooks, the bridge/client forwarding and the daemon's graph
controller.  Everything here runs on the SYNTHETIC TEST GRAPH or on a stub RPC
server: no test loads the 600 MB MaleCNS graph, and none makes a behavioural
claim.  What is asserted is wiring semantics:

* the slip/contrast keys reach the server's sensory packet,
* a graph yaw command REPLACES the modular optomotor bias (never adds to it),
* a missing ``optomotor`` block is an explicitly unsupported mapping with no
  motion, never a silent zero yaw,
* the engineered-assistance flags reach the telemetry packets,
* the modular default path is untouched.
"""
import math

import numpy as np
import pytest

import arena as arena_mod
import connectome_bridge
from arena import Arena
from brainlab.io_map import DNa02YawDecoder, OptomotorEncoder, OptomotorIOMap
from neurofly_daemon import ContinuousExperimentRunner, GraphArenaController

DRUM_DEG_S = 30.0          # maze.OptomotorParadigm default
CONTRAST = 0.9             # maze.OptomotorParadigm default


# ---------------------------------------------------------------------------
# A stub graph server: records the packets it is sent, replies as asked.
# ---------------------------------------------------------------------------
class StubRpcClient:
    """Stands in for ConnectomeClient; never touches a socket."""

    reply_optomotor = True
    engineered_assistance_enabled = False
    yaw_rad_s = 0.37
    packets = []

    def __init__(self, host=None, port=None, allow_synthetic=False, **kwargs):
        self.is_connected = True
        self.last_error = None
        self.last_latency_ms = 0.0
        self.server_identity = {'backend': 'connectome-fixed', 'synthetic': False,
                                'engineered_assistance_enabled': self.engineered_assistance_enabled,
                                'optomotor_io_map_sha256': 'stub-io-map'}

    def check_health(self):
        return True

    def reset(self):
        return True

    def step(self, sensory_packet, duration_ms=2.0):
        type(self).packets.append(dict(sensory_packet))
        reply = {'status': 'ok', 'backend': 'connectome-fixed', 'synthetic': False,
                 'graph_sha256': None, 'dna02_diff': 0.0, 'dna02_rate_l': 0.0, 'dna02_rate_r': 0.0,
                 'dnp09_rate': 0.0, 'bpn_rate': 0.0, 'mdn_rate': 0.0, 'dnp01_gf_spikes': 0,
                 'engineered_assistance_enabled': self.engineered_assistance_enabled,
                 'engineered_assistance_applied': (
                     ['looming_trigger injects 80 directly into DNp01 (not via LC4/LPLC2)']
                     if self.engineered_assistance_enabled else []),
                 'optomotor': None}
        if type(self).reply_optomotor:
            reply['optomotor'] = {'slip_rad_s': sensory_packet.get('optomotor_slip_rad_s'),
                                  'yaw_rad_s': type(self).yaw_rad_s,
                                  'contributions': {'DNa02_L': type(self).yaw_rad_s, 'DNa02_R': 0.0},
                                  'rate_l': 18.5, 'rate_r': 0.0, 'io_map_sha256': 'stub-io-map',
                                  'decoder': 'yaw = 0.02*(rate DNa02_L - rate DNa02_R)'}
        return reply


@pytest.fixture
def stub_rpc(monkeypatch):
    StubRpcClient.packets = []
    StubRpcClient.reply_optomotor = True
    StubRpcClient.engineered_assistance_enabled = False
    StubRpcClient.yaw_rad_s = 0.37
    monkeypatch.setattr(connectome_bridge, 'ConnectomeClient', StubRpcClient)
    return StubRpcClient


def rpc_arena(paradigm='optomotor'):
    return Arena(paradigm=paradigm, brain_type='connectome', connectome_mode='rpc', seed=7)


# ---------------------------------------------------------------------------
# 1. The slip and contrast keys reach the server
# ---------------------------------------------------------------------------
def test_slip_and_contrast_reach_the_server_packet(stub_rpc):
    world = rpc_arena()
    world.fly.angular_velocity = 0.25
    world.step(0.02)
    packet = stub_rpc.packets[0]
    assert packet['optomotor_slip_rad_s'] == pytest.approx(math.radians(DRUM_DEG_S) - 0.25)
    assert packet['optomotor_contrast'] == pytest.approx(CONTRAST)


def test_other_assays_get_no_invented_optomotor_stimulus(stub_rpc):
    world = rpc_arena(paradigm='wind-tunnel')
    world.step(0.02)
    packet = stub_rpc.packets[0]
    assert 'optomotor_slip_rad_s' not in packet and 'optomotor_contrast' not in packet


# ---------------------------------------------------------------------------
# 2. The graph yaw replaces the modular bias, and the assay is tethered at speed 0
# ---------------------------------------------------------------------------
def test_graph_yaw_replaces_the_modular_optomotor_bias(stub_rpc):
    world = rpc_arena()
    fly = world.fly
    stimuli = {'drum_velocity_deg_s': DRUM_DEG_S, 'contrast': CONTRAST}
    dheading, speed, state, _, _ = world.compute_steering(
        sensory=world._sample_odors(fly) if hasattr(world, '_sample_odors') else
        {'left_a': 0.0, 'right_a': 0.0, 'mean_a': 0.0, 'diff_a': 0.0,
         'left_b': 0.0, 'right_b': 0.0, 'mean_b': 0.0, 'diff_b': 0.0},
        fly=fly, dt=0.02, drum_velocity_deg_s=DRUM_DEG_S, visual_contrast=CONTRAST,
        assay_stimuli=stimuli)
    # Exactly the decoded command: not scaled, not added to anything.
    assert dheading == pytest.approx(StubRpcClient.yaw_rad_s)
    assert speed == 0.0 and state == 'OPTOMOTOR-TETHERED'
    assert fly.motor_source == 'graph-rpc' and fly.motor_halted is False
    # The modular bias for the same stimulus is a different, non-zero number, so
    # equality above shows replacement rather than addition.
    modular = Arena(paradigm='optomotor', brain_type='modular', seed=7)
    modular.step(0.02)
    bias = modular.fly.vision.get_optomotor_yaw_bias()
    assert bias != 0.0 and abs(bias - StubRpcClient.yaw_rad_s) > 1e-6


def test_tethered_optomotor_runs_at_speed_zero_and_reports_it(stub_rpc):
    world = rpc_arena()
    for _ in range(5):
        result = world.step(0.02)
    assert result['fly_speed'] == 0.0
    # The COMMANDED speed is zero too: the modular walking drive is never borrowed.
    assert world.fly.motor_record['controller_speed'] == 0.0
    assert world.fly.behavioral_state == 'OPTOMOTOR-TETHERED'
    provenance = world.optomotor_provenance()
    assert provenance['optomotor_yaw_rad_s'] == pytest.approx(StubRpcClient.yaw_rad_s)
    assert provenance['optomotor_unsupported'] is None


# ---------------------------------------------------------------------------
# 3. A missing optomotor block is unsupported, not zero
# ---------------------------------------------------------------------------
def test_missing_optomotor_block_is_unsupported_and_still(stub_rpc):
    stub_rpc.reply_optomotor = False
    world = rpc_arena()
    start = (world.fly.pos.x, world.fly.pos.y, world.fly.heading)
    for _ in range(5):
        world.step(0.02)
    fly = world.fly
    assert fly.motor_source == 'graph-unmapped-io'
    assert fly.motor_source not in Arena.MOTOR_SOURCES_NORMAL
    assert fly.motor_halted is True and fly.speed == 0.0
    assert (fly.pos.x, fly.pos.y, fly.heading) == start
    assert 'no verified' in fly.optomotor_unsupported or 'no optomotor block' in fly.optomotor_unsupported
    assert world.optomotor_provenance()['optomotor_unsupported']


def test_bridge_names_the_unsupported_state_only_when_a_stimulus_was_sent(stub_rpc):
    stub_rpc.reply_optomotor = False
    bridge = connectome_bridge.ConnectomeBridge(mode='rpc')
    out = bridge.step(fly_pos=np.zeros(2), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
                      odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), dt=0.02)
    assert out['optomotor'] is None and out['optomotor_unsupported'] is None
    assert out['motor_source'] == 'graph-rpc'          # no optomotor input: nothing is claimed
    out = bridge.step(fly_pos=np.zeros(2), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
                      odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), dt=0.02,
                      optomotor_slip_rad_s=0.5, optomotor_contrast=1.0)
    assert out['motor_source'] == connectome_bridge.ConnectomeBridge.UNMAPPED_IO
    assert 'no verified WP5 sensory encoder' in out['optomotor_unsupported']


# ---------------------------------------------------------------------------
# 4. Engineered-assistance flags propagate to the packets
# ---------------------------------------------------------------------------
def test_assistance_flags_propagate_to_telemetry(stub_rpc):
    stub_rpc.engineered_assistance_enabled = True
    world = rpc_arena()
    world.step(0.02)
    provenance = world.optomotor_provenance()
    assert provenance['engineered_assistance_enabled'] is True
    assert provenance['engineered_assistance_applied']
    assert provenance['optomotor_io_map_sha256'] == 'stub-io-map'
    identity = world.fly.connectome_bridge.controller_identity()
    assert identity['engineered_assistance_enabled'] is True
    assert identity['optomotor_io_map_sha256'] == 'stub-io-map'


# ---------------------------------------------------------------------------
# 5. The modular default path is unchanged
# ---------------------------------------------------------------------------
def test_modular_path_still_adds_the_vision_optomotor_bias_and_has_no_graph_keys():
    world = Arena(paradigm='optomotor', brain_type='modular', seed=11)
    world.step(0.02)
    provenance = world.motor_provenance()
    assert 'optomotor' not in provenance          # no graph reply: modular telemetry unchanged
    assert world.optomotor_provenance() == {}
    assert provenance['motor_source'] == 'modular'
    fly = world.fly
    sensory = {'left_a': 0.0, 'right_a': 0.0, 'mean_a': 0.0, 'diff_a': 0.0,
               'left_b': 0.0, 'right_b': 0.0, 'mean_b': 0.0, 'diff_b': 0.0}
    baseline = world.compute_steering(sensory=sensory, fly=fly, dt=0.02,
                                      drum_velocity_deg_s=DRUM_DEG_S, visual_contrast=CONTRAST,
                                      assay_stimuli={})[0]
    sentinel = 0.123
    fly.vision.get_optomotor_yaw_bias = lambda: sentinel   # the modular bias is still additive
    shifted = world.compute_steering(sensory=sensory, fly=fly, dt=0.02,
                                     drum_velocity_deg_s=DRUM_DEG_S, visual_contrast=CONTRAST,
                                     assay_stimuli={})[0]
    assert shifted != baseline
    # The modular controller still commands a walking speed (the assay is tethered,
    # so the realized speed is 0 for every backend; the commanded one is what differs).
    assert fly.motor_source == 'modular' and fly.motor_record['controller_speed'] > 0.0


def test_modular_run_is_bit_identical_across_instances():
    """The default path is deterministic and untouched by the graph hooks."""
    def trace():
        world = Arena(paradigm='optomotor', brain_type='modular', seed=3)
        return [(round(r['fly_heading'], 12), round(r['fly_speed'], 12)) for r in
                (world.step(0.02) for _ in range(20))]
    assert trace() == trace()


# ---------------------------------------------------------------------------
# 6. The daemon's graph controller
# ---------------------------------------------------------------------------
def synthetic_io_map(n=64):
    """A stub WP5 map inside a 64-neuron synthetic graph.  Never the real pin."""
    populations = {name: np.arange(20 + k * 5, 25 + k * 5, dtype=np.int64)
                   for k, name in enumerate(('ftb_L', 'btf_L', 'ftb_R', 'btf_R'))}
    populations['DNa02_L'] = np.array([10], np.int64)
    populations['DNa02_R'] = np.array([11], np.int64)
    return OptomotorIOMap(populations=populations,
                          source_ids={k: [int(i) for i in v] for k, v in populations.items()},
                          matched_counts={'T4': 5, 'T5': 0}, available_counts={},
                          monitors={'HS_L': np.array([1]), 'HS_R': np.array([2])},
                          sha256='synthetic-test-optomotor-map')


def graph_runner(tmp_path, paradigm='optomotor'):
    return ContinuousExperimentRunner(initial_paradigm=paradigm, sim_speed=1, checkpoint_interval=3600,
                                      output_dir=tmp_path, backend='connectome-fixed',
                                      test_synthetic_graph=True)


def test_graph_controller_refuses_to_fake_the_map_on_a_synthetic_graph(tmp_path):
    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(3):
            runner.step_once()
    telemetry = runner.arena.fly.last_connectome_telemetry
    assert telemetry['optomotor'] is None
    assert 'could not be resolved' in telemetry['optomotor_unsupported']
    assert runner.arena.fly.motor_source == GraphArenaController.UNMAPPED
    assert runner.arena.fly.speed == 0.0
    provenance = runner.motor_summary()['optomotor']
    assert provenance['optomotor_unsupported'] and provenance['optomotor_io_map_sha256'] is None
    assert provenance['engineered_assistance_enabled'] is False


def test_graph_controller_runs_the_wp5_loop_when_a_map_resolves(tmp_path):
    runner = graph_runner(tmp_path)
    controller = runner.graph_controller
    controller._optomotor_io = synthetic_io_map()      # stub map; no 600 MB graph in tests
    with runner.lock:
        for _ in range(8):
            runner.step_once()
    fly = runner.arena.fly
    telemetry = fly.last_connectome_telemetry
    block = telemetry['optomotor']
    assert block is not None and block['io_map_sha256'] == 'synthetic-test-optomotor-map'
    assert block['slip_rad_s'] == pytest.approx(math.radians(DRUM_DEG_S) - fly.angular_velocity, abs=1e-6)
    assert block['contrast'] == pytest.approx(CONTRAST)
    assert telemetry['engineered_assistance_enabled'] is False
    assert telemetry['engineered_assistance_applied'] == []
    # Yaw is the decoder's command, applied as given; the tethered fly has no forward drive.
    assert fly.motor_source == 'graph' and fly.motor_halted is False
    assert fly.speed == 0.0 and fly.behavioral_state == 'OPTOMOTOR-TETHERED'
    assert telemetry['yaw_rate'] == pytest.approx(block['yaw_rad_s'])
    summary = runner.motor_summary()['optomotor']
    assert summary['optomotor_io_map_sha256'] == 'synthetic-test-optomotor-map'
    assert 'speed 0' in summary['forward_drive']


def test_graph_controller_encoder_drives_only_the_mapped_cells(tmp_path):
    """The loop delivers the encoder's drive and nothing else (no tonic assistance)."""
    runner = graph_runner(tmp_path)
    io_map = synthetic_io_map()
    runner.graph_controller._optomotor_io = io_map
    instance = runner.registry.active
    loop = runner.graph_controller._loop_for(instance)
    assert loop is not None and loop.step_ms == runner.graph_controller.step_ms
    encoder = OptomotorEncoder(io_map, np.random.default_rng(0))
    currents = np.zeros(instance.brain.n, dtype=np.float32)
    encoder.encode(currents, 0.0, math.radians(DRUM_DEG_S), CONTRAST)
    driven = np.flatnonzero(currents)
    allowed = np.concatenate([io_map.populations['ftb_L'], io_map.populations['btf_R']])
    assert set(driven.tolist()) <= set(allowed.tolist())
    assert DNa02YawDecoder(io_map).decode(np.zeros(instance.brain.n, np.int32), 20.0)['yaw_rad_s'] == 0.0
