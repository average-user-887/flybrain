"""dn-v2: declared DN -> FlyGym CPG decoder and the locomotion DN map."""
import math

import numpy as np
import pandas as pd
import pytest

from brainlab.cosim_server import ConnectomeServer
from brainlab.graph_identity import GraphUnavailable
from brainlab.io_map import locomotion_dn_map_from_nodes
from neurofly_body.decoder import DNCommandDecoder
from neurofly_body.runner import EmbodiedConfig, run_embodied
from tests.test_embodied_runner import FakeBody, FakeGraph

RATES = ("DNp09_L", "DNp09_R", "DNa02_L", "DNa02_R", "MDN_L", "MDN_R")


def _reply(gf_spikes=0, **rates):
    block = {f"{name}_rate_hz": float(rates.get(name, 0.0)) for name in RATES}
    block.update(GF_L_spikes=gf_spikes, GF_R_spikes=0)
    return {"locomotion_dn": block}


def _settle(decoder, steps=400, **rates):
    for _ in range(steps):
        out = decoder.decode_reply(_reply(**rates), 2.0)
    return out


def test_silence_gives_exactly_zero():
    out = DNCommandDecoder().decode_reply(_reply(), 2.0)
    assert (out["left_cpg_drive"], out["right_cpg_drive"]) == (0.0, 0.0)
    assert out["program"] == "none" and out["events"] == []


def test_dna02_alone_never_makes_the_fly_walk():
    out = _settle(DNCommandDecoder(), DNa02_L=80.0)
    assert (out["left_cpg_drive"], out["right_cpg_drive"]) == (0.0, 0.0)


def test_bilateral_p9_walks_forward_symmetrically():
    out = _settle(DNCommandDecoder(), DNp09_L=50.0, DNp09_R=50.0)
    assert out["program"] == "forward"
    assert out["left_cpg_drive"] == out["right_cpg_drive"] == pytest.approx(1.0, rel=1e-6)


def test_unilateral_p9_turns_toward_its_side():
    # FlyGym turns toward the side with the smaller amplitude (Bidaye 2020: ipsilateral turn).
    out = _settle(DNCommandDecoder(), DNp09_L=50.0)
    assert out["right_cpg_drive"] > out["left_cpg_drive"] == 0.0


def test_dna02_shortens_ipsilateral_strides_only():
    out = _settle(DNCommandDecoder(), DNp09_L=50.0, DNp09_R=50.0, DNa02_R=40.0)
    assert out["left_cpg_drive"] == pytest.approx(1.0, rel=1e-6)
    assert out["right_cpg_drive"] == pytest.approx(0.6, rel=1e-6)   # 1 - 0.01 * 40


def test_mdn_reverses_when_it_beats_forward_drive():
    decoder = DNCommandDecoder()
    out = _settle(decoder, DNp09_L=10.0, DNp09_R=10.0, MDN_L=40.0, MDN_R=40.0)
    assert out["program"] == "reverse"
    assert out["left_cpg_drive"] == out["right_cpg_drive"] == pytest.approx(-0.8, rel=1e-6)
    out = _settle(DNCommandDecoder(), DNp09_L=60.0, DNp09_R=60.0, MDN_L=10.0, MDN_R=10.0)
    assert out["program"] == "forward" and out["left_cpg_drive"] > 0


def test_drive_is_clipped_to_the_nmf_range():
    out = _settle(DNCommandDecoder(), DNp09_L=500.0, DNp09_R=500.0)
    assert out["left_cpg_drive"] == out["right_cpg_drive"] == 1.2
    out = _settle(DNCommandDecoder(), MDN_L=500.0, MDN_R=500.0)
    assert out["left_cpg_drive"] == -1.2


def test_gf_spike_is_a_takeoff_event_reported_unsupported():
    out = DNCommandDecoder().decode_reply(_reply(gf_spikes=1), 2.0)
    assert out["events"][0]["event"] == "takeoff_command"
    assert out["events"][0]["body_action"].startswith("UNSUPPORTED")


def test_missing_locomotion_block_fails_closed():
    with pytest.raises(RuntimeError, match="locomotion_dn"):
        DNCommandDecoder().decode_reply({"locomotion_dn": None}, 2.0)


def test_describe_marks_every_unsourced_gain():
    params = DNCommandDecoder().describe()["parameters"]
    assert {k for k, (_, src) in params.items() if src == "ASSUMPTION"} == {
        "gain_p9_per_hz", "k_dna02_per_hz", "gain_mdn_per_hz", "tau_ms"}


def test_runner_refuses_dn_v2_without_a_locomotion_map(tmp_path):
    config = EmbodiedConfig(duration_s=0.004, output_dir=tmp_path / "run")
    with pytest.raises(RuntimeError, match="locomotion_dn_map_sha256"):
        run_embodied(config, FakeGraph(), FakeBody())


def _nodes(rows):
    return pd.DataFrame(rows, columns=["node_index", "source_id", "cell_type", "type", "somaSide"])


def _full_rows():
    rows, k = [], 0
    for cell_type in ("DNp09", "MDN", "DNp01", "DNa02"):
        for side in ("R", "L"):
            rows.append((k, 1000 - k, cell_type, cell_type, side))
            k += 1
    return rows


def test_locomotion_map_splits_by_soma_side():
    dn = locomotion_dn_map_from_nodes(_nodes(_full_rows()))
    assert set(dn.populations) == {f"{n}_{s}" for n in ("DNp09", "MDN", "GF", "DNa02") for s in "LR"}
    assert list(dn.populations["DNp09_R"]) == [0] and list(dn.populations["DNp09_L"]) == [1]
    assert dn.sha256 == locomotion_dn_map_from_nodes(_nodes(_full_rows())).sha256


def test_locomotion_map_fails_closed():
    rows = _full_rows()
    with pytest.raises(GraphUnavailable, match="without a L/R"):
        locomotion_dn_map_from_nodes(_nodes(rows + [(99, 5, "MDN", "MDN", None)]))
    with pytest.raises(GraphUnavailable, match="MDN_L resolved empty"):
        locomotion_dn_map_from_nodes(_nodes([r for r in rows if not (r[2] == "MDN" and r[4] == "L")]))
    with pytest.raises(GraphUnavailable, match="disagrees"):
        locomotion_dn_map_from_nodes(_nodes(rows + [(98, 6, "DNp09", "DNp10", "L")]))


def test_server_reports_per_side_rates(tmp_path):
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    assert server.step({})["locomotion_dn"] is None  # never faked on a synthetic graph
    server.locomotion_dn = locomotion_dn_map_from_nodes(_nodes(_full_rows()))

    def fake_step(currents, duration_ms):
        counts = np.zeros(server.n_neurons, dtype=np.int64)
        counts[1] = 2   # DNp09_L
        counts[4] = 1   # GF (DNp01) R
        return counts, 0.0

    server.brain.step = fake_step
    block = server.step({}, duration_ms=2.0)["locomotion_dn"]
    assert block["DNp09_L_rate_hz"] == 1000.0 and block["DNp09_R_rate_hz"] == 0.0
    assert block["GF_R_spikes"] == 1 and block["GF_L_spikes"] == 0
    assert server.get_status()["locomotion_dn_map_sha256"] == server.locomotion_dn.sha256


flygym = pytest.importorskip("flygym")


def test_flygym_body_accepts_reverse_drive():
    from neurofly_body.flygym_body import FlyGymBody

    body = FlyGymBody(warmup_s=0.01)
    try:
        start = body.reset(0)["thorax"]["position_mm"][0]
        obs = body.step((-1.0, -1.0), 2000)
        assert obs["thorax"]["position_mm"][0] < start
    finally:
        body.close()
