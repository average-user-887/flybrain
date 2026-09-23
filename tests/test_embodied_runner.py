import json
from pathlib import Path

import pytest

from neurofly_body.runner import EmbodiedConfig, run_embodied, validate_real_v3_status


class FakeGraph:
    def __init__(self):
        self.sim_ms = 0.0
        self.inputs = []

    def reset(self):
        self.sim_ms = 0.0

    def get_status(self):
        return {
            "synthetic": False,
            "lif_dynamics_version": "v3",
            "engineered_assistance_enabled": False,
            "transmitter_policy": "v3-modulatory-only",
            "graph_sha256": "a" * 64,
            "optomotor_io_map_sha256": "b" * 64,
            "effective_graph_sha256": "c" * 64,
        }

    def step(self, sensory, duration_ms=2.0):
        self.inputs.append(dict(sensory))
        self.sim_ms += duration_ms
        return {
            "sim_ms": self.sim_ms,
            "dna02_rate_l": 25.0 if len(self.inputs) == 1 else 0.0,
            "dna02_rate_r": 0.0,
            "total_step_spikes": 1,
        }


class FakeBody:
    physics_dt_s = 0.0001

    def __init__(self):
        self.time = 0.05
        self.yaw = 0.0
        self.commands = []
        self.closed = False

    def reset(self, seed):
        self.time = 0.05
        self.yaw = 0.0
        return self._observation(0.0)

    def step(self, cpg_drive, substeps):
        self.commands.append(tuple(cpg_drive))
        dt = substeps * self.physics_dt_s
        yaw_velocity = cpg_drive[1] - cpg_drive[0]
        self.yaw += yaw_velocity * dt
        self.time += dt
        return self._observation(yaw_velocity)

    def _observation(self, yaw_velocity):
        return {
            "body_sim_time_s": self.time,
            "thorax": {
                "position_mm": [0.0, 0.0, 0.5],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "yaw_rad": self.yaw,
                "yaw_velocity_rad_s": yaw_velocity,
            },
            "joint_angles_rad": [self.yaw],
            "joint_velocities_rad_s": [yaw_velocity],
            "contacts": {"found": [1.0] * 6},
        }

    def describe(self):
        return {"adapter": "fake", "physics_dt_s": self.physics_dt_s, "units": {}}

    def close(self):
        self.closed = True


def _config(path: Path, mode="intact"):
    return EmbodiedConfig(duration_s=0.006, output_dir=path, mode=mode)


def test_fake_graph_body_lockstep_and_artifacts(tmp_path):
    graph, body = FakeGraph(), FakeBody()
    summary = run_embodied(_config(tmp_path / "run"), graph, body)
    assert summary["records"] == 3
    assert body.closed
    assert body.commands[0][1] > 0.0
    rows = [json.loads(line) for line in (tmp_path / "run/telemetry.jsonl").read_text().splitlines()]
    assert [row["run_time_s"] for row in rows] == [0.002, 0.004, 0.006]
    assert [row["neural"]["sim_ms"] for row in rows] == [2.0, 4.0, 6.0]
    assert rows[0]["sensory"]["retinal_slip_rad_s"] == 4.0
    assert json.loads((tmp_path / "run/manifest.json").read_text())["status"] == "complete"


def test_output_disconnected_logs_but_never_applies_decoder(tmp_path):
    body = FakeBody()
    run_embodied(_config(tmp_path / "control", "output-disconnected"), FakeGraph(), body)
    assert body.commands == [(0.0, 0.0)] * 3
    first = json.loads((tmp_path / "control/telemetry.jsonl").read_text().splitlines()[0])
    assert first["motor"]["decoded_cpg_drive"][1] > 0.0
    assert first["motor"]["applied_cpg_drive"] == [0.0, 0.0]


def test_output_directory_is_never_overwritten(tmp_path):
    target = tmp_path / "exists"
    target.mkdir()
    body = FakeBody()
    with pytest.raises(FileExistsError):
        run_embodied(_config(target), FakeGraph(), body)


def test_real_graph_validation_fails_closed():
    status = FakeGraph().get_status()
    validate_real_v3_status(status)
    status["synthetic"] = True
    with pytest.raises(RuntimeError, match="refusing embodied run"):
        validate_real_v3_status(status)
