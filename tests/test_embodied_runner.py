import json
from pathlib import Path

import pytest

from neurofly_body.decoder import DNa02CPGDecoder
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
    summary = run_embodied(_config(tmp_path / "run"), graph, body, decoder=DNa02CPGDecoder())
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
    run_embodied(_config(tmp_path / "control", "output-disconnected"), FakeGraph(), body,
                 decoder=DNa02CPGDecoder())
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


def test_cli_refuses_a_taken_output_directory_before_any_import(tmp_path):
    """D6: the collision is detected before the graph load, not after it."""
    from neurofly_body import cli

    taken = tmp_path / "taken"
    taken.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        cli._check_output_dir(taken)

    args = cli._parser().parse_args(
        ["run", "--duration", "1", "--output", str(taken)]
    )
    with pytest.raises(SystemExit, match="already exists"):
        cli._run(args)

    fresh = tmp_path / "fresh"
    assert cli._check_output_dir(fresh) == fresh
    assert not fresh.exists()


class IdentifiedGraph(FakeGraph):
    """Fake graph that also sends the static identity block the real server sends."""

    IDENTITY = {
        "brain_backend": "cpu",
        "graph_sha256": "a" * 64,
        "synthetic": False,
        "transmitter_policy": "v3-modulatory-only",
        "transmitter_policy_report": {"note": "x" * 400},
    }

    def step(self, sensory, duration_ms=2.0):
        return {**super().step(sensory, duration_ms), **self.IDENTITY}


def test_identity_block_is_hoisted_into_the_manifest(tmp_path):
    """D7: telemetry format 2 writes identity_sha256, not the whole block."""
    run = tmp_path / "run"
    summary = run_embodied(_config(run), IdentifiedGraph(), FakeBody(), decoder=DNa02CPGDecoder())
    assert summary["telemetry_format_version"] == 2
    assert summary["schema"] == "neurofly-embodied-summary-v2"

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["telemetry_format_version"] == 2
    identity = manifest["telemetry_identity"]
    assert identity["values"] == IdentifiedGraph.IDENTITY
    assert len(identity["identity_sha256"]) == 64

    text = (run / "telemetry.jsonl").read_text()
    assert "transmitter_policy_report" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows and all(row["schema"] == "neurofly-embodied-step-v2" for row in rows)
    for row in rows:
        assert row["neural"]["identity_sha256"] == identity["identity_sha256"]
        for key in IdentifiedGraph.IDENTITY:
            assert key not in row["neural"]


def test_identity_that_changes_mid_run_fails_the_run(tmp_path):
    class Drifting(IdentifiedGraph):
        def step(self, sensory, duration_ms=2.0):
            reply = super().step(sensory, duration_ms)
            if len(self.inputs) > 1:
                reply["graph_sha256"] = "b" * 64
            return reply

    with pytest.raises(RuntimeError, match="identity changed mid-run"):
        run_embodied(_config(tmp_path / "drift"), Drifting(), FakeBody(), decoder=DNa02CPGDecoder())
