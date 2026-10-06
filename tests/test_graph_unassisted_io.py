"""Lock the unassisted graph-to-arena I/O method and its public evidence."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from neurofly_daemon import ContinuousExperimentRunner
from provenance import GRAPH_IO_CONFIG_JSON, GRAPH_IO_SHA256, graph_io_declaration


def _controlled_graph(tmp_path):
    runner = ContinuousExperimentRunner(
        initial_paradigm="open-arena", sim_speed=1, checkpoint_interval=3600,
        output_dir=tmp_path, backend="connectome-fixed", test_synthetic_graph=True,
    )
    controller = runner.graph_controller
    instance = runner.registry.active
    controller.step_ms = 100.0
    controller.dn_indices = {
        "dna02_l": [0], "dna02_r": [1], "dnp09": [2],
        "dnb01": [3], "mdn": [4], "dnp01": [5],
    }
    controller.sensory_indices = {
        "orn_food": [10], "orn_danger": [11], "visual_l": [12],
        "visual_r": [13], "visual_looming": [14], "courtship_cva": [15],
        "jon_wind": [16], "thermo_receptors": [17],
        "er_ring": [18], "el_modulator": [19],
    }
    state = {"spikes": {}, "currents": None}

    def step(currents, step_ms):
        state["currents"] = np.array(currents, copy=True)
        counts = np.zeros(instance.brain.n, dtype=np.int64)
        for index, count in state["spikes"].items():
            counts[index] = count
        return SimpleNamespace(counts=counts)

    instance.step = step
    return runner, controller, state


def test_silent_graph_has_no_hidden_drive_or_forward_floor(tmp_path):
    runner, controller, state = _controlled_graph(tmp_path)
    out = controller(
        fly=runner.arena.fly,
        sensory={"stripe_contrast": 1.0},
        visual_contrast=1.0,
        dt=0.02,
    )

    assert np.count_nonzero(state["currents"]) == 0
    assert state["currents"][18] == state["currents"][19] == 0.0
    assert out["dn_rates"]["dnb01"] == 0.0
    assert out["raw_motor_command"] == {
        "forward_speed_mm_s": 0.0, "yaw_rate_rad_s": 0.0, "state": "GRAPH",
    }
    assert out["engineered_assistance_enabled"] is False
    assert out["engineered_assistance_applied"] == []
    stages = {row["name"]: row for row in out["input_stage"]}
    assert set(stages) == {"orn_food", "orn_danger", "photoreceptor_l", "photoreceptor_r",
                           "looming", "courtship_cva", "jon_wind", "thermo"}
    assert all(row["entry_stage"] and "status" in row for row in stages.values())


def test_declared_decoder_coefficients_and_overrides_are_applied(tmp_path):
    runner, controller, state = _controlled_graph(tmp_path)

    cases = [
        ({3: 1}, 4.0, 0.0, "GRAPH"),       # 10 Hz DNb01 * 0.4
        ({2: 1}, 15.0, 0.0, "GRAPH"),      # 10 Hz DNp09 * 1.5
        ({0: 1, 1: 2}, 0.0, -0.2, "GRAPH"),
        ({4: 3}, -15.0, 0.0, "REVERSE"),   # 30 Hz MDN > 20 Hz
        ({5: 1}, 35.0, 0.0, "ESCAPE"),
    ]
    for spikes, speed, yaw, motor_state in cases:
        state["spikes"] = spikes
        out = controller(fly=runner.arena.fly, sensory={}, dt=0.02)
        assert out["forward_speed"] == speed
        assert out["yaw_rate"] == yaw
        assert out["state"] == motor_state


def test_absent_dn_pool_is_null_publicly_but_zero_to_decoder(tmp_path):
    runner, controller, state = _controlled_graph(tmp_path)
    controller.dn_indices["dnb01"] = []
    state["spikes"] = {2: 1}

    out = controller(fly=runner.arena.fly, sensory={}, dt=0.02)

    assert out["dn_rates"]["dnb01"] is None
    assert "dnb01" in out["dn_unavailable"]
    assert out["forward_speed"] == 15.0
    json.dumps(out, allow_nan=False)


def test_graph_io_hash_is_canonical_and_callers_receive_copies():
    declaration = graph_io_declaration(include_config=True)
    assert hashlib.sha256(GRAPH_IO_CONFIG_JSON.encode()).hexdigest() == GRAPH_IO_SHA256
    assert declaration["sha256"] == GRAPH_IO_SHA256
    assert declaration["config"]["engineered_assistance"]["enabled"] is False
    declared_names = {probe["name"] for probe in declaration["config"]["input_probes"]}
    assert "er_ring" not in declared_names and "el_modulator" not in declared_names

    declaration["config"]["engineered_assistance"]["enabled"] = True
    assert graph_io_declaration(include_config=True)["config"]["engineered_assistance"]["enabled"] is False
