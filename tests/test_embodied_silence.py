"""Opt-in silencing of named cell types in embodied runs (--silence)."""
import json

import numpy as np
import pandas as pd
import pytest

from brainlab.cosim_server import ConnectomeServer
from brainlab.graph_identity import GraphUnavailable
from brainlab.io_map import SILENCE_DRIVE, parse_silence_target, silence_map_from_nodes
from neurofly_body import cli
from neurofly_body.decoder import DNa02CPGDecoder
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph


def _nodes():
    rows = [(0, 30, "DNa02", "L"), (1, 20, "DNa02", "R"), (2, 10, "MDN", "L"),
            (3, 11, "MDN", "R"), (4, 12, "MDN", "L"), (5, 40, "KC", None)]
    return pd.DataFrame(rows, columns=["node_index", "source_id", "cell_type", "somaSide"])


def test_targets_resolve_by_type_and_side():
    m = silence_map_from_nodes(_nodes(), ["DNa02", "MDN:L"])
    assert list(m.populations["DNa02"]) == [1, 0]           # sorted by source_id
    assert list(m.populations["MDN:L"]) == [2, 4]
    assert list(m.nodes) == [0, 1, 2, 4]
    assert m.summary() == {"targets": ["DNa02", "MDN:L"], "map_sha256": m.sha256, "drive": SILENCE_DRIVE,
                           "neurons": {"DNa02": 2, "MDN:L": 2}, "total_neurons": 4}
    assert m.sha256 == silence_map_from_nodes(_nodes(), ["DNa02", "MDN:L"]).sha256
    assert m.sha256 != silence_map_from_nodes(_nodes(), ["DNa02", "MDN"]).sha256
    assert silence_map_from_nodes(_nodes(), ["KC"]).populations["KC"].tolist() == [5]  # unsided types too


def test_targets_fail_closed():
    for bad in ("", ":L", "DNa02:X", "DNa02:"):
        with pytest.raises(ValueError):
            parse_silence_target(bad)
    with pytest.raises(GraphUnavailable, match="no neurons"):
        silence_map_from_nodes(_nodes(), ["DNp09"])
    with pytest.raises(GraphUnavailable, match="no neurons"):
        silence_map_from_nodes(_nodes(), ["KC:L"])
    with pytest.raises(ValueError, match="repeated"):
        silence_map_from_nodes(_nodes(), ["MDN", "MDN"])
    with pytest.raises(ValueError):
        silence_map_from_nodes(_nodes(), [])


def test_synthetic_graph_refuses_silencing(tmp_path):
    with pytest.raises(GraphUnavailable, match="real annotated graph"):
        ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True, silence=("DNa02",))


def test_default_server_reply_has_no_silence_fields(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    reply, status = server.step({}), server.get_status()
    assert "silenced" not in reply and "silence" not in reply
    assert "silence" not in status and "silence_map" not in status


def test_clamp_overrides_all_drive_and_counts_leaks(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    server.silence_map = silence_map_from_nodes(_nodes(), ["DNa02", "MDN:L"])
    server._silence_drive = np.float32(SILENCE_DRIVE)
    seen = {}

    def fake_step(currents, duration_ms):
        seen["currents"] = currents.copy()
        counts = np.zeros(server.n_neurons, dtype=np.int64)
        counts[4] = 3        # a leaking MDN_L neuron
        counts[3] = 7        # MDN_R, not silenced
        return counts, 0.0

    server.brain.step = fake_step
    reply = server.step({"optomotor_slip_rad_s": 4.0}, duration_ms=2.0)
    assert all(seen["currents"][i] == SILENCE_DRIVE for i in (0, 1, 2, 4))
    assert seen["currents"][3] != SILENCE_DRIVE
    assert reply["silenced"] == {"spikes": 3, "by_target": {"DNa02": 0, "MDN:L": 3}}
    assert reply["silence"]["targets"] == ["DNa02", "MDN:L"]
    assert server.get_status()["silence_map"]["source_ids"]["DNa02"] == [20, 30]


class SilencedGraph(FakeGraph):
    def __init__(self, leak):
        super().__init__()
        self.leak = leak

    def get_status(self):
        return {**super().get_status(), "silence": {"targets": ["DNa02"], "total_neurons": 2}}

    def step(self, sensory, duration_ms=2.0):
        return {**super().step(sensory, duration_ms), "silenced": {"spikes": self.leak, "by_target": {}}}


@pytest.mark.parametrize("leak,held", [(0, True), (1, False)])
def test_summary_records_silenced_set_and_clamp(tmp_path, leak, held):
    config = EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "run", mode="intact")
    summary = run_embodied(config, SilencedGraph(leak), FakeBody(), decoder=DNa02CPGDecoder())
    assert summary["silenced"] == {"targets": ["DNa02"], "total_neurons": 2,
                                   "spikes_total": 3 * leak, "clamp_held": held}
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["neural_backend"]["silence"]["targets"] == ["DNa02"]
    assert manifest["invocation"] == {}


def test_unsilenced_summary_has_no_silence_block(tmp_path):
    config = EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "run", mode="intact")
    assert "silenced" not in run_embodied(config, FakeGraph(), FakeBody(), decoder=DNa02CPGDecoder())


def test_cli_flag_is_repeatable_and_replayed(tmp_path, monkeypatch):
    args = cli._parser().parse_args(["run", "--duration", "1", "--output", "x",
                                     "--silence", "DNa02", "--silence", "MDN:L"])
    assert cli._invocation(args)["silence"] == ["DNa02", "MDN:L"]
    assert cli._invocation(cli._parser().parse_args(["run", "--duration", "1", "--output", "x"]))["silence"] is None

    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"status": "complete", "invocation": cli._invocation(args),
                                                   "neural_backend": {}}))
    (run / "summary.json").write_text("{}")
    (run / "telemetry.jsonl").write_text("")
    captured = {}

    def fake_run(replay_args):
        captured["silence"] = replay_args.silence
        out = replay_args.output
        out.mkdir()
        (out / "manifest.json").write_text(json.dumps({"neural_backend": {}}))
        (out / "telemetry.jsonl").write_text("")
        return {"records": 0, "duration_s": 1.0, "wall_time_s": 0.0, "real_time_factor": 0.0}

    monkeypatch.setattr(cli, "_run", fake_run)
    assert cli.main(["replay-check", str(run), "--output", str(tmp_path / "replay")]) == 0
    assert captured["silence"] == ["DNa02", "MDN:L"]


def test_modular_controller_rejects_silencing():
    pytest.importorskip("flygym")
    with pytest.raises(SystemExit, match="connectome controller"):
        cli.main(["run", "--controller", "modular", "--duration", "0.01", "--output", "unused",
                  "--silence", "DNa02"])
