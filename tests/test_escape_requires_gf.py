"""Escape comes from the connectome: a Giant Fiber (DNp01) spike, never stimulus strength alone."""
import math
from types import SimpleNamespace

import numpy as np

from arena import Arena
from connectome_bridge import ConnectomeBridge
from neurofly_daemon import ContinuousExperimentRunner


def _runner(tmp_path):
    return ContinuousExperimentRunner(initial_paradigm="looming-escape", sim_speed=1, checkpoint_interval=3600,
                                      output_dir=tmp_path, backend="connectome-fixed", test_synthetic_graph=True)


def _clamp_graph(runner, gf_spike):
    """Replace the instance's step with one that fires DNp01 or keeps it silent."""
    controller = runner.graph_controller
    instance = runner.registry.active
    dnp01 = controller.dn_indices["dnp01"]
    real_step = instance.step

    def step(currents, step_ms):
        result = real_step(currents, step_ms)
        counts = np.array(result.counts, copy=True)
        counts[dnp01] = 1 if gf_spike else 0
        return SimpleNamespace(counts=counts)
    instance.step = step
    return controller


def test_maximal_loom_without_gf_spike_is_not_an_escape(tmp_path):
    controller = _clamp_graph(_runner(tmp_path), gf_spike=False)
    # Near-collision disc (170 deg), well above every former stimulus threshold.
    out = controller(fly=controller.runner.arena.fly,
                     sensory={"looming_theta": math.radians(170.0), "looming_detected": True}, dt=0.02)
    assert out["motor_source"] == "graph"
    assert out["state"] != "ESCAPE"
    assert out["dn_rates"]["gf"] == 0.0


def test_gf_spike_is_an_escape_even_without_a_stimulus(tmp_path):
    controller = _clamp_graph(_runner(tmp_path), gf_spike=True)
    out = controller(fly=controller.runner.arena.fly, sensory={}, dt=0.02)
    assert out["state"] == "ESCAPE" and out["forward_speed"] == 35.0
    assert out["dn_rates"]["gf"] > 0.0


def test_paradigm_gf_flag_is_not_a_visual_input(tmp_path):
    controller = _clamp_graph(_runner(tmp_path), gf_spike=False)
    captured = {}
    instance = controller.runner.registry.active
    clamped = instance.step

    def step(currents, step_ms):
        captured["loom"] = float(sum(currents[i] for i in controller.sensory_indices.get("visual_looming", ())))
        return clamped(currents, step_ms)
    instance.step = step
    controller(fly=controller.runner.arena.fly, sensory={"gf_spike": True}, dt=0.02)
    assert captured["loom"] == 0.0


def _looming_arena_at_threshold_crossing():
    a = Arena(paradigm="looming-escape", seed=4, num_flies=1, num_predators=0)
    p = a.paradigm
    # Put the disc just below the paradigm's 65 deg geometric GF threshold.
    t_cross = p.t_collision_s - p.r_over_v_s / math.tan(p.gf_threshold_rad / 2.0)
    p.time_elapsed_ms = p.stimulus_started_ms + t_cross * 1000.0 - 5.0
    return a


def test_graph_backend_ignores_the_geometric_gf_threshold():
    a = _looming_arena_at_threshold_crossing()
    calls = []

    def silent_graph(**kwargs):
        calls.append(kwargs)
        return {"halted": False, "forward_speed": 5.0, "yaw_rate": 0.0, "motor_source": "graph", "state": "GRAPH"}
    a.graph_controller = silent_graph
    for _ in range(10):
        a.step(.02)
    assert calls
    assert a.paradigm.gf_source == "connectome"
    assert not a.paradigm.escape_initiated
    assert a.total_escapes == 0 and a.fly.behavioral_state != "ESCAPE"


def test_graph_backend_escape_follows_the_dnp01_spike():
    a = _looming_arena_at_threshold_crossing()
    a.graph_controller = lambda **kw: {"halted": False, "forward_speed": 35.0, "yaw_rate": 0.0,
                                       "motor_source": "graph", "state": "ESCAPE"}
    for _ in range(3):
        a.step(.02)
    assert a.paradigm.escape_initiated and a.paradigm.time_to_collision_at_jump_ms is not None
    assert a.fly.behavioral_state == "ESCAPE"
    assert a.total_escapes == 1   # one escape per GF volley, not one per spiking step


def test_modular_baseline_keeps_its_geometric_gf():
    a = _looming_arena_at_threshold_crossing()
    for _ in range(10):
        a.step(.02)
    assert a.paradigm.gf_source == "geometric"
    assert a.paradigm.escape_initiated and a.total_escapes == 1


class _Client:
    def __init__(self, gf_spikes):
        self.gf_spikes = gf_spikes
        self.last_error = None
        self.is_connected = True

    def step(self, sensory, duration_ms=20.0):
        return dict(status='ok', dna02_diff=0.0, dna02_rate_l=0.0, dna02_rate_r=0.0, dnp09_rate=0.0,
                    bpn_rate=0.0, mdn_rate=0.0, dnp01_gf_spikes=self.gf_spikes)

    def reset(self):
        return True


def _approaching_predator_step(bridge):
    return bridge.step(fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
                       odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
                       predator_positions=[np.array([60.0, 50.0])], predator_velocities=[np.array([-200.0, 0.0])],
                       dt=0.02)


def test_rpc_bridge_escapes_only_on_remote_dnp01_spikes():
    silent = ConnectomeBridge(mode='rpc', rpc_client=_Client(0))
    _approaching_predator_step(silent)   # the same loom escapes the surrogate (last test)
    assert not silent.escape_active and silent.dnp01_gf_spikes == 0
    spiking = ConnectomeBridge(mode='rpc', rpc_client=_Client(1))
    _approaching_predator_step(spiking)
    assert spiking.escape_active and spiking.dnp01_gf_spikes == 1


def test_surrogate_bridge_keeps_its_hand_built_trigger():
    bridge = ConnectomeBridge(mode='surrogate')
    _approaching_predator_step(bridge)
    assert bridge.escape_active
