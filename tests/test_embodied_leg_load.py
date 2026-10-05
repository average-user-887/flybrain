"""Opt-in leg-load feedback: FlyGym load measurement, CS afferent map, encoder, wiring."""
import json

import numpy as np
import pandas as pd
import pytest

from brainlab.cosim_server import ConnectomeServer
from brainlab.graph_identity import GraphUnavailable
from brainlab.io_map import (LEG_ORDER, LegLoadEncoder, leg_load_afferent_map_from_nodes)
from neurofly_body import cli
from neurofly_body.decoder import DNa02CPGDecoder
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph

NERVES = {"f": "ProLN", "m": "MesoLN", "h": "MetaLN"}


def _nodes(extra=()):
    rows, k = [], 0
    for leg in LEG_ORDER:
        for j in range(2):
            rows.append((k, 1000 - k, "campaniform sensilla", NERVES[leg[1]], leg[0].upper()))
            k += 1
    rows += [(50, 5, "campaniform sensilla", "ADMN", "L"),      # wing CS: not a leg afferent
             (51, 6, "leg", "ProLN", "L")] + list(extra)          # untyped leg sensory neuron
    return pd.DataFrame(rows, columns=["node_index", "source_id", "subclass", "entryNerve", "rootSide"])


def test_afferent_map_uses_leg_nerve_and_root_side():
    m = leg_load_afferent_map_from_nodes(_nodes())
    assert list(m.populations) == list(LEG_ORDER)
    assert list(m.populations["lf"]) == [1, 0]                  # sorted by source_id
    assert list(m.populations["rh"]) == [11, 10]
    assert 50 not in np.concatenate(list(m.populations.values()))
    assert m.sha256 == leg_load_afferent_map_from_nodes(_nodes()).sha256


def test_afferent_map_fails_closed():
    with pytest.raises(GraphUnavailable, match="root side"):
        leg_load_afferent_map_from_nodes(_nodes([(60, 7, "campaniform sensilla", "MetaLN", None)]))
    nodes = _nodes()
    with pytest.raises(GraphUnavailable, match="rm resolved empty"):
        leg_load_afferent_map_from_nodes(nodes[~((nodes.entryNerve == "MesoLN") & (nodes.rootSide == "R"))])


def _encoder():
    return LegLoadEncoder(leg_load_afferent_map_from_nodes(_nodes()))


def test_encoder_is_rectified_saturating_and_phasic():
    enc, cur = _encoder(), np.zeros(60, np.float32)
    rates = enc.encode(cur, [0.0, 0.5, 3.0, 10.0, 1e6, -3.0], 2.0)
    assert rates["lf"] == 0.0 and rates["lm"] == 0.0 and rates["rh"] == 0.0     # at or below F0, or pulling
    assert 0 < rates["lh"] < rates["rf"] < rates["rm"] <= 200.0
    assert rates["rm"] == pytest.approx(200.0)
    assert cur[4] == pytest.approx(20.0 * rates["lh"] / 200.0) and cur[50] == 0 and cur[51] == 0
    # A step increase in load adds a phasic component that then decays.
    enc = _encoder()
    steady = [enc.encode(np.zeros(60, np.float32), [3.0] * 6, 2.0)["lf"] for _ in range(50)][-1]
    loaded = enc.encode(np.zeros(60, np.float32), [5.0] * 6, 2.0)["lf"]
    after = [enc.encode(np.zeros(60, np.float32), [5.0] * 6, 2.0)["lf"] for _ in range(200)][-1]
    assert loaded > after > steady
    assert after == pytest.approx(200.0 * np.tanh(4.5 / 10.0), rel=1e-6)


def test_encoder_reset_replays_and_rejects_bad_input():
    enc = _encoder()
    loads = [[float(k + i) for i in range(6)] for k in range(10)]
    first = [enc.encode(np.zeros(60, np.float32), x, 2.0) for x in loads]
    enc.reset()
    assert [enc.encode(np.zeros(60, np.float32), x, 2.0) for x in loads] == first
    with pytest.raises(ValueError):
        enc.encode(np.zeros(60, np.float32), [1.0] * 5, 2.0)
    with pytest.raises(ValueError):
        enc.encode(np.zeros(60, np.float32), [float("nan")] * 6, 2.0)


def test_server_default_has_no_leg_load_and_synthetic_refuses(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    assert "leg_load" not in server.step({}) and "leg_load_encoder" not in server.get_status()
    with pytest.raises(GraphUnavailable, match="real annotated graph"):
        ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True, leg_load_feedback=True)


def test_server_drives_afferents_and_reports_spikes(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    server.leg_load = _encoder()
    seen = {}

    def fake_step(currents, duration_ms):
        seen["currents"] = currents.copy()
        counts = np.zeros(server.n_neurons, dtype=np.int64)
        counts[2] = 1   # an lm afferent
        return counts, 0.0

    server.brain.step = fake_step
    with pytest.raises(ValueError, match="leg_load_uN"):
        server.step({}, duration_ms=2.0)
    reply = server.step({"leg_load_uN": [3.0] * 6}, duration_ms=2.0)
    assert seen["currents"][0] > 0 and reply["leg_load"]["afferent_spikes"]["lm"] == 1
    assert reply["leg_load_afferent_map_sha256"] == server.leg_load.map.sha256
    server.reset()
    assert server.leg_load._previous is None


class LoadBody(FakeBody):
    def _observation(self, yaw_velocity):
        return {**super()._observation(yaw_velocity), "leg_load_uN": [1.5] * 6}


class LoadGraph(FakeGraph):
    def get_status(self):
        return {**super().get_status(), "leg_load_afferent_map_sha256": "d" * 64}


def test_runner_passes_load_and_fails_closed(tmp_path):
    config = EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "run", leg_load_feedback=True)
    graph = LoadGraph()
    run_embodied(config, graph, LoadBody(), decoder=DNa02CPGDecoder())
    assert all(s["leg_load_uN"] == [1.5] * 6 for s in graph.inputs)
    row = json.loads((tmp_path / "run/telemetry.jsonl").read_text().splitlines()[0])
    assert row["sensory"]["leg_load_uN"] == [1.5] * 6
    with pytest.raises(RuntimeError, match="no leg-load afferent map"):
        run_embodied(EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "a", leg_load_feedback=True),
                     FakeGraph(), LoadBody(), decoder=DNa02CPGDecoder())
    with pytest.raises(RuntimeError, match="does not report leg_load_uN"):
        run_embodied(EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "b", leg_load_feedback=True),
                     LoadGraph(), FakeBody(), decoder=DNa02CPGDecoder())


def test_runner_default_sends_no_load(tmp_path):
    graph = FakeGraph()
    run_embodied(EmbodiedConfig(duration_s=0.006, output_dir=tmp_path / "run"), graph, LoadBody(),
                 decoder=DNa02CPGDecoder())
    assert all("leg_load_uN" not in s for s in graph.inputs)


def test_cli_flag_round_trip():
    args = cli._parser().parse_args(["run", "--duration", "1", "--output", "x", "--leg-load-feedback"])
    assert cli._invocation(args)["leg_load_feedback"] is True
    assert cli.INVOCATION_BACKFILL["leg_load_feedback"] is False


pytest.importorskip("flygym")


def test_flygym_leg_load_sums_to_body_weight():
    from neurofly_body.flygym_body import FlyGymBody

    body = FlyGymBody(warmup_s=0.05, measure_leg_load=True)
    try:
        body.reset(0)
        totals = [sum(body.step((0.0, 0.0), 20)["leg_load_uN"]) for _ in range(50)]
        mass = sum(body.sim.mj_model.body_mass[i] for i in body._skeleton_ids)
        weight = mass * -body.sim.mj_model.opt.gravity[2]
        assert np.mean(totals) == pytest.approx(weight, rel=0.03)
    finally:
        body.close()
    plain = FlyGymBody(warmup_s=0.0)
    try:
        assert "leg_load_uN" not in plain.reset(0)
    finally:
        plain.close()


def test_modular_controller_rejects_leg_load():
    with pytest.raises(SystemExit, match="connectome controller"):
        cli.main(["run", "--controller", "modular", "--duration", "0.01", "--output", "unused",
                  "--leg-load-feedback"])
