"""P2 gate: an embodied run replays bit-identically from its seed.

Covers the pieces that make that possible: a server reset that restores the
freshly built state (GPU state and the optomotor noise stream included),
telemetry free of wall-clock fields, and ``replay-check``.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from brainlab.cosim_server import ConnectomeServer
from brainlab.io_map import DNa02YawDecoder, OptomotorEncoder
from neurofly_body import cli
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph
from tests.test_wp5_optomotor import small_io


def _server(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    io = small_io()
    server.optomotor = (io, OptomotorEncoder(io, np.random.default_rng(server.optomotor_seed)),
                        DNa02YawDecoder(io))
    return server


def _drive(server, steps=40):
    replies = []
    for k in range(steps):
        reply = server.step({"optomotor_slip_rad_s": 4.0, "optomotor_contrast": 1.0,
                             "mean_odor": 0.6, "wpn_wind_speed": 15.0}, duration_ms=2.0)
        reply.pop("elapsed_ms")
        replies.append(json.dumps(reply, sort_keys=True))
    return replies, server.brain.v.copy()


def test_server_reset_replays_like_a_new_server(tmp_path):
    fresh = _server(tmp_path)
    first, v_first = _drive(fresh)
    assert any(json.loads(r)["total_step_spikes"] for r in first), "the drive must make spikes"
    fresh.reset()
    again, v_again = _drive(fresh)
    assert again == first
    assert np.array_equal(v_first, v_again)
    other, _ = _drive(_server(tmp_path))
    assert other == first


def test_reset_state_uploads_to_the_gpu_copy(tmp_path):
    server = _server(tmp_path)
    uploads = []

    class Device:
        def upload_state(self, *arrays):
            uploads.append([a.copy() for a in arrays])

    _drive(server, 5)
    server.brain._gpu = Device()
    try:
        server.reset()
    finally:
        server.brain._gpu = None
    assert len(uploads) == 1
    v, g, refractory, queue, queue_count, counts, active_flag = uploads[0]
    assert (v == -52.0).all() and not g.any() and not queue.any() and not active_flag.any()
    assert server.brain.sim_ms == 0.0 and server.brain.cursor == 0 and server.total_steps == 0


class TimedGraph(FakeGraph):
    """Reports a different wall-clock time on every call, like the real server."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def step(self, sensory, duration_ms=2.0):
        self.calls += 1
        reply = super().step(sensory, duration_ms)
        reply["elapsed_ms"] = 1.0 + 0.123 * self.calls * id(self) % 7
        return reply


def _run(path):
    config = EmbodiedConfig(duration_s=0.01, output_dir=path, seed=3)
    return run_embodied(config, TimedGraph(), FakeBody(), invocation={"seed": 3})


def test_telemetry_is_byte_identical_and_wall_clock_goes_to_timing(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    telemetry_a = (tmp_path / "a/telemetry.jsonl").read_bytes()
    assert telemetry_a == (tmp_path / "b/telemetry.jsonl").read_bytes()
    assert a["trajectory_sha256"] == b["trajectory_sha256"]
    assert "elapsed_ms" not in telemetry_a.decode()
    timing = [json.loads(l) for l in (tmp_path / "a/timing.jsonl").read_text().splitlines()]
    assert [t["step"] for t in timing] == [1, 2, 3, 4, 5]
    assert all("neural_elapsed_ms" in t and t["step_wall_ms"] >= 0 for t in timing)
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    assert manifest["trajectory_sha256"] == a["trajectory_sha256"]
    assert manifest["invocation"] == {"seed": 3}


def _fake_cli_run(created):
    def run(args):
        graph, body = TimedGraph(), FakeBody()
        seed_shift = created.get("perturb", 0.0)
        if seed_shift:
            original = body.step

            def step(cpg_drive, substeps):
                return original((cpg_drive[0] + seed_shift, cpg_drive[1]), substeps)

            body.step = step
        config = EmbodiedConfig(duration_s=args.duration, output_dir=args.output, seed=args.seed,
                                mode=args.mode)
        return run_embodied(config, graph, body, invocation=cli._invocation(args))
    return run


def test_replay_check_passes_and_reports_divergence(tmp_path, monkeypatch, capsys):
    state = {}
    monkeypatch.setattr(cli, "_run", _fake_cli_run(state))
    assert cli.main(["run", "--duration", "0.01", "--output", str(tmp_path / "orig"), "--seed", "5"]) == 0
    assert cli.main(["replay-check", str(tmp_path / "orig"), "--output", str(tmp_path / "rep")]) == 0
    receipt = json.loads((tmp_path / "rep/replay_check.json").read_text())
    assert receipt["verdict"] == "BIT_IDENTICAL"
    assert receipt["invocation"]["seed"] == 5 and receipt["records"] == 5

    state["perturb"] = 0.25
    assert cli.main(["replay-check", str(tmp_path / "orig"), "--output", str(tmp_path / "bad")]) == 1
    receipt = json.loads((tmp_path / "bad/replay_check.json").read_text())
    assert receipt["verdict"] == "DIVERGED" and receipt["first_differing_record"] == 1


def test_replay_check_refuses_incomplete_runs(tmp_path):
    run = tmp_path / "failed"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(SystemExit, match="did not complete"):
        cli.main(["replay-check", str(run), "--output", str(tmp_path / "x")])


flygym = pytest.importorskip("flygym")


def test_flygym_body_replays_bit_identically(tmp_path):
    """Real MuJoCo body, fake graph: two short runs give the same bytes."""
    from neurofly_body.flygym_body import FlyGymBody

    digests = []
    for name in ("one", "two"):
        config = EmbodiedConfig(duration_s=0.02, output_dir=tmp_path / name, seed=7)
        summary = run_embodied(config, TimedGraph(), FlyGymBody(warmup_s=0.01))
        digests.append(summary["trajectory_sha256"])
    assert digests[0] == digests[1]
