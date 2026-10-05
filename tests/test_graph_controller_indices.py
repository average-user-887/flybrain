"""The graph controller's node indices belong to the graph that is actually loaded.

Sensory, EPG and DN indices used to be resolved against whatever neuron table the
environment or the default path pointed to, independent of the graph the brain ran
on.  A synthetic test graph run beside real annotations then received real-graph
indices (index 1398 on a 64-neuron brain).  These tests pin the fix: every resolved
index is below the running brain's neuron count, whatever annotations exist on disk.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from brainlab.graph_identity import DEFAULT_CONNECTOME_DIR, DEFAULT_GRAPH_DIR
from neurofly_daemon import ContinuousExperimentRunner


def _all_indices(controller):
    channels = dict(controller.sensory_indices, epg=controller.epg_indices)
    channels.update({f"dn:{k}": v for k, v in controller.dn_indices.items()})
    return channels


def _assert_within(controller, n):
    for name, indices in _all_indices(controller).items():
        assert all(0 <= int(i) < n for i in indices), f"{name} has indices outside 0..{n - 1}: {indices[:5]}"


def test_synthetic_graph_ignores_real_annotations_on_disk(tmp_path, monkeypatch):
    # A neuron table that, like the real MaleCNS one, has LC4/EPG/ORN rows far beyond 64.
    cdir = tmp_path / "connectome"
    (cdir / "normalized").mkdir(parents=True)
    df = pd.DataFrame({"node_index": [1398, 2000, 3000, 4000], "source_id": [1, 2, 3, 4],
                       "cell_type": ["LC4", "EPG", "ORN_DM1", "ER4d"]})
    feather.write_feather(pa.Table.from_pandas(df), cdir / "normalized/neurons.feather")
    monkeypatch.setenv("NEUROFLY_CONNECTOME_DIR", str(cdir))

    runner = ContinuousExperimentRunner(initial_paradigm="looming-escape", sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path / "out", backend="connectome-fixed",
                                        test_synthetic_graph=True)
    controller = runner.graph_controller
    n = runner.registry.active.brain.n
    assert n == runner.shared_graph.n == 64
    _assert_within(controller, n)
    # The synthetic graph has no cell types: no sensory channel is invented for it.
    assert not any(controller.sensory_indices.values()) and not controller.epg_indices
    assert "synthetic" in controller.sensory_unavailable
    # DN channels are the synthetic graph's own io_map, not the MaleCNS node numbers.
    assert controller.dn_indices == {k: list(v) for k, v in runner.shared_graph.io_map.items()}
    # Stepping a looming stimulus through it must not index outside the brain.
    out = controller(fly=runner.arena.fly, sensory={"looming_theta": 2.5, "looming_detected": True}, dt=0.02)
    assert out["motor_source"] == "graph"


def test_real_graph_indices_fit_the_loaded_brain(tmp_path):
    if not (DEFAULT_GRAPH_DIR / "graph.npz").is_file():
        pytest.skip("MaleCNS graph.npz not available in standard path")
    if not (DEFAULT_CONNECTOME_DIR / "normalized/neurons.feather").is_file():
        pytest.skip("MaleCNS normalized/neurons.feather not available in standard path")
    runner = ContinuousExperimentRunner(initial_paradigm="looming-escape", sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path / "out", backend="connectome-fixed")
    controller = runner.graph_controller
    n = runner.registry.active.brain.n
    assert n == runner.shared_graph.n == runner.shared_graph.identity.neurons
    _assert_within(controller, n)
    assert controller.sensory_unavailable is None
    assert controller.sensory_indices["visual_looming"], "LC4/LPLC2 must resolve on the real graph"
    assert controller.epg_indices
