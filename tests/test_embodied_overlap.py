"""Opt-in motor delay and pipelined (overlapped) brain/body stepping."""
import json
import threading

import pytest

from neurofly_body import cli
from neurofly_body.decoder import DNa02CPGDecoder
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph


class ThreadedBody(FakeBody):
    def __init__(self, fail_at=None):
        super().__init__()
        self.threads = []
        self.fail_at = fail_at

    def step(self, cpg_drive, substeps):
        self.threads.append(threading.current_thread().name)
        if self.fail_at is not None and len(self.commands) == self.fail_at:
            raise RuntimeError("body failed")
        return super().step(cpg_drive, substeps)


def _run(path, **kw):
    body = ThreadedBody()
    config = EmbodiedConfig(duration_s=0.01, output_dir=path, mode="intact", **kw)
    summary = run_embodied(config, FakeGraph(), body, decoder=DNa02CPGDecoder())
    rows = [json.loads(line) for line in (path / "telemetry.jsonl").read_text().splitlines()]
    return summary, body, rows


@pytest.mark.parametrize("kw", [dict(motor_delay_steps=-1), dict(motor_delay_steps=1.0),
                                dict(motor_delay_steps=True), dict(pipeline=True)])
def test_config_rejects_bad_delay(tmp_path, kw):
    with pytest.raises(ValueError):
        EmbodiedConfig(duration_s=0.01, output_dir=tmp_path, **kw).validated()


def test_delay_shifts_the_command_by_whole_steps(tmp_path):
    _, zero, rows0 = _run(tmp_path / "d0")
    _, one, rows1 = _run(tmp_path / "d1", motor_delay_steps=1)
    _, two, _ = _run(tmp_path / "d2", motor_delay_steps=2)
    assert zero.commands[0][1] > 0                      # FakeGraph drives on its first step only
    assert one.commands[0] == (0.0, 0.0) and one.commands[1] == zero.commands[0]
    assert two.commands[:2] == [(0.0, 0.0)] * 2 and two.commands[2] == zero.commands[0]
    assert rows1[0]["motor"]["delay_steps"] == 1 and "delay_steps" not in rows0[0]["motor"]
    assert rows1[0]["motor"]["applied_cpg_drive"] == [0.0, 0.0]
    assert rows1[1]["motor"]["applied_cpg_drive"] == rows0[0]["motor"]["applied_cpg_drive"]
    assert all(name == "MainThread" for name in one.threads)


def test_pipeline_is_bit_identical_to_delayed_sequential(tmp_path):
    seq, seq_body, _ = _run(tmp_path / "seq", motor_delay_steps=1)
    pipe, pipe_body, _ = _run(tmp_path / "pipe", motor_delay_steps=1, pipeline=True)
    assert (tmp_path / "seq/telemetry.jsonl").read_bytes() == (tmp_path / "pipe/telemetry.jsonl").read_bytes()
    assert seq["trajectory_sha256"] == pipe["trajectory_sha256"]
    assert pipe_body.commands == seq_body.commands
    assert all(name.startswith("neurofly-body") for name in pipe_body.threads)
    manifest = json.loads((tmp_path / "pipe/manifest.json").read_text())
    assert manifest["lockstep"]["motor_delay_ms"] == 2.0
    assert "concurrently" in manifest["lockstep"]["ordering"]


def test_pipelined_body_failure_fails_the_run_and_closes_the_body(tmp_path):
    body = ThreadedBody(fail_at=2)
    config = EmbodiedConfig(duration_s=0.01, output_dir=tmp_path / "run", motor_delay_steps=1, pipeline=True)
    with pytest.raises(RuntimeError, match="body failed"):
        run_embodied(config, FakeGraph(), body, decoder=DNa02CPGDecoder())
    assert body.closed
    assert json.loads((tmp_path / "run/manifest.json").read_text())["status"] == "failed"


def test_cli_records_and_replays_the_mode():
    args = cli._parser().parse_args(["run", "--duration", "1", "--output", "x",
                                     "--motor-delay-steps", "1", "--pipeline"])
    inv = cli._invocation(args)
    assert (inv["motor_delay_steps"], inv["pipeline"]) == (1, True)
    default = cli._invocation(cli._parser().parse_args(["run", "--duration", "1", "--output", "x"]))
    assert (default["motor_delay_steps"], default["pipeline"]) == (0, False)
    assert cli.INVOCATION_BACKFILL["pipeline"] is False


pytest.importorskip("flygym")


def test_flygym_pipeline_matches_delayed_sequential_and_replays(tmp_path):
    base = ["run", "--controller", "modular", "--duration", "0.1", "--seed", "1", "--motor-delay-steps", "1"]
    assert cli.main(base + ["--output", str(tmp_path / "seq")]) == 0
    assert cli.main(base + ["--pipeline", "--output", str(tmp_path / "pipe")]) == 0
    assert (tmp_path / "seq/telemetry.jsonl").read_bytes() == (tmp_path / "pipe/telemetry.jsonl").read_bytes()
    assert cli.main(["replay-check", str(tmp_path / "pipe"), "--output", str(tmp_path / "replay")]) == 0
    assert json.loads((tmp_path / "replay/manifest.json").read_text())["invocation"]["pipeline"] is True
