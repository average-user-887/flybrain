"""Sensory delivery at the real sender/receiver boundaries (sensory-delivery repair, 2026-10-08).

Two boundaries are checked with small synthetic fixtures (no whole-brain run):

* arena/assay stimuli -> ``GraphArenaController`` (connectome-fixed/-plastic): the JO wind
  probe must read both assay names of the same quantity (``wind_speed`` in the wind tunnel,
  ``wind_magnitude`` in the multisensory sandbox), first present key wins.
* ``ConnectomeBridge`` RPC packet -> JSON -> ``ConnectomeServer.step``: odour, cVA, wind and
  temperature must reach the receptor channels under the keys the server reads.

The tests observe the current actually written into each channel's neuron, not only key
presence.  One distinct synthetic neuron per channel makes every channel observable.  A
delivered current is not evidence of sensing or behaviour.
"""
import json
import math

import numpy as np
import pytest

from neurofly_daemon import ContinuousExperimentRunner

GRAPH_CHANNELS = ("orn_food", "orn_danger", "courtship_cva", "jon_wind", "visual_looming",
                  "thermo_receptors")


def _graph_runner(tmp_path, assay):
    runner = ContinuousExperimentRunner(initial_paradigm=assay, sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path, backend="connectome-fixed",
                                        test_synthetic_graph=True)
    controller = runner.graph_controller
    for i, channel in enumerate(GRAPH_CHANNELS):
        controller.sensory_indices[channel] = [i]
    return runner, controller


def _stage(telemetry, name):
    return next(row for row in telemetry["input_stage"] if row["name"] == name)


def _call(runner, controller, sensory, **kwargs):
    with runner.lock:
        out = controller(fly=runner.arena.fly, sensory=dict(sensory), dt=0.02, **kwargs)
    return out, controller._currents.copy()


# --- graph controller boundary -------------------------------------------------------

def test_multisensory_wind_magnitude_reaches_jo_probe(tmp_path):
    """Was 0.0: the sandbox publishes wind_magnitude (15 mm/s) and the probe read only wind_speed."""
    runner, _ = _graph_runner(tmp_path, "multisensory-sandbox")
    with runner.lock:
        runner.step_once()
    row = _stage(runner.arena.fly.last_connectome_telemetry, "jon_wind")
    assert row["current_injected"] == pytest.approx(0.15 * 15.0)
    assert row["stimulus_key"] == "wind_magnitude"
    assert row["delivery"] == "delivered"


def test_wind_tunnel_wind_speed_path_unchanged(tmp_path):
    runner, _ = _graph_runner(tmp_path, "wind-tunnel")
    with runner.lock:
        runner.step_once()
    row = _stage(runner.arena.fly.last_connectome_telemetry, "jon_wind")
    assert row["current_injected"] == pytest.approx(3.75)      # 25 mm/s * 0.15, as before
    assert row["stimulus_key"] == "wind_speed"


@pytest.fixture
def sandbox(tmp_path):
    return _graph_runner(tmp_path, "multisensory-sandbox")


def test_explicit_zero_wind_speed_overrides_alias(sandbox):
    runner, controller = sandbox
    out, currents = _call(runner, controller, {"wind_speed": 0.0, "wind_magnitude": 30.0},
                          wind_vector=np.array([-30.0, 0.0]))
    row = _stage(out, "jon_wind")
    assert row["stimulus_key"] == "wind_speed"
    assert row["current_injected"] == 0.0 and currents[3] == 0.0


def test_both_wind_keys_inject_once(sandbox):
    runner, controller = sandbox
    out, currents = _call(runner, controller, {"wind_speed": 20.0, "wind_magnitude": 30.0},
                          wind_vector=np.array([-20.0, 0.0]))
    assert _stage(out, "jon_wind")["current_injected"] == pytest.approx(3.0)
    assert currents[3] == pytest.approx(3.0)                   # not 3.0 + 4.5


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -5.0, True, "25"])
def test_non_finite_or_invalid_wind_is_not_delivered(sandbox, bad):
    runner, controller = sandbox
    out, currents = _call(runner, controller, {"wind_speed": bad, "wind_magnitude": 15.0},
                          wind_vector=np.array([-15.0, 0.0]))
    row = _stage(out, "jon_wind")
    assert row["current_injected"] == 0.0 and currents[3] == 0.0
    assert row["delivery"].startswith("NOT DELIVERED")
    assert np.isfinite(currents).all()


def test_vector_only_wind_is_labelled_not_delivered(sandbox):
    """No scalar key: the wind field exists, so it is NOT DELIVERED (mapping undefined), not 'not simulated'."""
    runner, controller = sandbox
    out, currents = _call(runner, controller, {"wind": (15.0, 0.0)}, wind_vector=np.array([15.0, 0.0]))
    row = _stage(out, "jon_wind")
    assert row["current_injected"] == 0.0 and currents[3] == 0.0
    assert row["stimulus_key"] is None
    assert row["delivery"].startswith("NOT DELIVERED")


def test_graph_looming_path_unchanged(sandbox):
    runner, controller = sandbox
    out, currents = _call(runner, controller, {"theta_rad": 0.5})
    assert _stage(out, "looming")["current_injected"] == pytest.approx(min(55.0, 15.0 + 40.0 * 0.5))
    assert currents[4] == pytest.approx(35.0)


# --- RPC bridge -> server boundary ---------------------------------------------------

RPC_CHANNELS = ("orn_food", "courtship_cva", "jon_wind", "thermo_receptors", "visual_looming")


@pytest.fixture
def server(tmp_path):
    from brainlab.cosim_server import ConnectomeServer
    srv = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True, engineered_assistance=False)
    for i, channel in enumerate(RPC_CHANNELS):
        srv.sensory_indices[channel] = [i]
    captured = {}

    def capture(currents, duration_ms):
        captured["currents"] = np.array(currents, copy=True)
        return np.zeros(srv.n_neurons, dtype=np.int64), 0.0
    srv.brain.step = capture

    def inject(packet):
        from connectome_client import _json_safe
        wire = json.loads(json.dumps(packet, default=_json_safe))   # what the HTTP client sends
        reply = srv.step(wire, duration_ms=20.0)
        values = {ch: float(captured["currents"][i]) for i, ch in enumerate(RPC_CHANNELS)}
        return values, reply
    return inject


def _bridge_packet():
    from connectome_bridge import ConnectomeBridge
    bridge = ConnectomeBridge(mode="surrogate")
    return bridge.encode_sensory(
        fly_pos=np.zeros(2), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
        odor_left=0.5, odor_right=0.5, wind_vector=np.array([-25.0, 0.0]), temperature=30.0,
        cva_odor=0.6, dt=0.02, predator_positions=[np.array([3.0, 0.0])],
        predator_velocities=[np.array([-30.0, 0.0])])


def test_bridge_packet_reaches_server_receptor_channels(server):
    """Was 0/0/0/0 (only looming arrived): the packet carried pn_dm1_norm, cva_rate, wind_speed, temperature."""
    values, reply = server(_bridge_packet())
    assert values["orn_food"] == pytest.approx(min(45.0, 0.5 * 30.0))
    assert values["courtship_cva"] == pytest.approx(min(40.0, 0.6 * 35.0))
    assert values["jon_wind"] == pytest.approx(min(35.0, 25.0 * 0.15))
    assert values["thermo_receptors"] == pytest.approx(min(40.0, (30.0 - 25.0) * 4.0))
    assert values["visual_looming"] == pytest.approx(30.0)          # unchanged path
    delivery = reply["sensory_delivery"]
    assert {ch: d["key"] for ch, d in delivery.items()} == {
        "orn_food": "mean_odor", "courtship_cva": "pheromone_cva",
        "jon_wind": "wpn_wind_speed", "thermo_receptors": "temperature_c"}


def test_bridge_packet_receptor_values_finite_and_in_range():
    packet = _bridge_packet()
    for key in ("mean_odor", "pheromone_cva", "wpn_wind_speed", "temperature_c"):
        assert isinstance(packet[key], float) and math.isfinite(packet[key])
    assert 0.0 <= packet["mean_odor"] <= 1.0 and 0.0 <= packet["pheromone_cva"] <= 1.0
    assert packet["wpn_wind_speed"] >= 0.0
    assert packet["temperature_c"] == packet["temperature"]           # absolute degC, not an excess
    assert packet["mean_odor"] != packet["pn_dm1_norm"]               # a concentration, not the PN rate


def test_server_explicit_zero_overrides_fallback(server):
    values, _ = server({"pheromone_cva": 0.0, "courtship_cva": 0.6,
                        "temperature_excess": 0.0, "temperature_c": 30.0})
    assert values["courtship_cva"] == 0.0 and values["thermo_receptors"] == 0.0


def test_server_both_keys_inject_once(server):
    values, _ = server({"pheromone_cva": 0.6, "courtship_cva": 0.6})
    assert values["courtship_cva"] == pytest.approx(21.0)            # not 42


def test_server_non_finite_inputs_not_delivered(server):
    """Was 45/40/35/40 (min(cap, inf)); now nothing is injected and the reason is reported."""
    packet = {"mean_odor": float("inf"), "courtship_cva": float("nan"),
              "wpn_wind_speed": float("inf"), "temperature_c": float("inf")}
    values, reply = server(packet)
    assert all(values[ch] == 0.0 for ch in ("orn_food", "courtship_cva", "jon_wind", "thermo_receptors"))
    assert all(d["status"].startswith("NOT DELIVERED") for d in reply["sensory_delivery"].values())


def test_server_missing_and_unknown_fields(server):
    values, reply = server({"wind_speed": 25.0, "cva_rate": 90.0, "temperature": 30.0,
                            "pn_dm1_norm": 40.0, "bogus": 1})
    assert all(v == 0.0 for v in values.values())
    assert reply["unconsumed_keys"] == ["bogus", "cva_rate", "pn_dm1_norm", "temperature", "wind_speed"]
    assert all(d["key"] is None for d in reply["sensory_delivery"].values())


def test_server_looming_trigger_unchanged(server):
    values, _ = server({"looming_trigger": True})
    assert values["visual_looming"] == pytest.approx(30.0)
    assert values["orn_food"] == values["jon_wind"] == 0.0
