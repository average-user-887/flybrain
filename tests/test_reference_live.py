"""Live Reference fly (not connectome) in the daemon: labelling, assay refusal,
store separation and reference <-> connectome-fixed switching without mixing."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_registry import PARADIGMS
from neurofly.split import SimProxy, sim_state
from neurofly_body.reference import BACKEND_ID, DISPLAY_LABEL, NAMESPACE_DIRNAME
from neurofly_daemon import ContinuousExperimentRunner
from tests.test_reference_fly import FakeBody

CONNECTOME_ONLY_KEYS = ("fly", "metrics", "activity", "neural", "connectome", "motor", "scene",
                        "live_assay", "plasticity", "biomechanics", "descending")


def graph_runner(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm="open-arena", sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path, backend="connectome-fixed",
                                        test_synthetic_graph=True)
    runner.reference_body_factory = FakeBody
    return runner


def switch(runner, backend, **params):
    return runner.dispatch_command({"action": "switch_backend", "params": {"backend": backend, **params}})


def packet(runner):
    runner._publish_snapshot()
    return json.loads(runner.published.data)


def store_digest(root: Path) -> dict:
    """Every file the daemon owns outside the reference namespace."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and NAMESPACE_DIRNAME not in p.parts}


def test_reference_is_never_the_default_and_is_selectable(tmp_path):
    runner = graph_runner(tmp_path)
    assert runner.presented_backend == "connectome-fixed" and runner.reference is None
    result = switch(runner, BACKEND_ID)
    assert result["status"] == "ok", result
    assert result["ack"]["applied"] is True
    ident = result["ack"]["identity"]
    assert ident["backend"] == BACKEND_ID and ident["label"] == DISPLAY_LABEL
    assert ident["graph_sha256"] is None and ident["is_connectome"] is False
    assert ident["daemon_run_id"] == runner.run_id and isinstance(ident["activation"], int)
    runner.stop()


def test_status_packet_and_split_state_carry_the_label(tmp_path):
    runner = graph_runner(tmp_path)
    switch(runner, BACKEND_ID)
    for _ in range(30):
        runner.reference.step_frame()
    pkt = packet(runner)
    assert pkt["type"] == "telemetry" and pkt["backend"] == BACKEND_ID
    assert pkt["paradigm"] == "flat-ground-walking" and pkt["paradigm_title"] == DISPLAY_LABEL
    assert pkt["reference_fly"]["display_label"] == DISPLAY_LABEL
    assert pkt["reference_fly"]["is_connectome"] is False
    assert pkt["reference_fly"]["sim_time_s"] == pytest.approx(0.06)
    assert pkt["liveness"]["state"] == "advancing"
    for key in CONNECTOME_ONLY_KEYS:
        assert key not in pkt, key
    state = sim_state(runner)
    assert state["backend"] == BACKEND_ID and state["identity"]["backend"] == BACKEND_ID
    assert state["reference_fly"]["display_label"] == DISPLAY_LABEL
    proxy = SimProxy(SimpleNamespace(state=state, published=None, problem=lambda: None))
    assert proxy.backend == proxy.presented_backend == BACKEND_ID
    assert proxy.reference_status()["display_label"] == DISPLAY_LABEL
    runner.stop()


@pytest.mark.parametrize("assay", PARADIGMS)
def test_every_assay_is_refused_while_the_reference_fly_is_shown(tmp_path, assay):
    runner = graph_runner(tmp_path)
    switch(runner, BACKEND_ID)
    result = runner.dispatch_command({"action": "switch_paradigm", "params": {"paradigm": assay}})
    assert result["status"] == "error" and result["ack"]["applied"] is False
    assert "unavailable for the reference fly" in result["message"]
    assert runner.reference is not None and runner.identity()["backend"] == BACKEND_ID
    runner.stop()


def test_selecting_the_reference_with_an_assay_is_refused(tmp_path):
    runner = graph_runner(tmp_path)
    result = switch(runner, BACKEND_ID, paradigm="optomotor")
    assert result["status"] == "error" and "unavailable for the reference fly" in result["message"]
    assert runner.reference is None and runner.presented_backend == "connectome-fixed"
    assert not (tmp_path / NAMESPACE_DIRNAME).exists()
    runner.stop()


@pytest.mark.parametrize("cmd", [{"action": "record_start"}, {"action": "save_checkpoint"},
                                 {"action": "reset_trial"}, {"action": "probe_brain"},
                                 {"action": "set_param", "params": {"name": "x", "value": 1}}])
def test_connectome_commands_are_refused_for_the_reference(tmp_path, cmd):
    runner = graph_runner(tmp_path)
    switch(runner, BACKEND_ID)
    result = runner.dispatch_command(cmd)
    assert result["status"] == "error" and "not available for the Reference fly (not connectome)" in result["message"]
    assert runner.recorder is None
    runner.stop()


def test_reference_never_writes_connectome_or_brain_stores(tmp_path):
    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(3):
            runner.step_once()
    assert switch(runner, BACKEND_ID)["status"] == "ok"
    before = store_digest(tmp_path)
    assert any("registry" in name or "graph-bookkeeping" in name for name in before)
    for _ in range(60):
        runner.reference.step_frame()
    packet(runner)
    runner.dispatch_command({"action": "set_paused", "params": {"paused": True}})
    runner.dispatch_command({"action": "set_paused", "params": {"paused": False}})
    runner.dispatch_command({"action": "switch_paradigm", "params": {"paradigm": "t-maze"}})
    assert store_digest(tmp_path) == before
    ref_dir = runner.reference.dir
    assert ref_dir.parent == (tmp_path / NAMESPACE_DIRNAME).resolve()
    switch(runner, "connectome-fixed")
    assert sorted(p.name for p in ref_dir.iterdir()) == ["body.nfbody", "manifest.json", "summary.json",
                                                        "trace.jsonl"]
    summary = json.loads((ref_dir / "summary.json").read_text())
    assert summary["backend_id"] == BACKEND_ID and summary["status"] == "complete"
    assert summary["records"] == 60 and summary["is_connectome"] is False
    assert not any(p.suffix == ".npz" for p in ref_dir.rglob("*"))
    runner.stop()


def test_switching_reference_and_connectome_fixed_does_not_mix_state(tmp_path):
    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(5):
            runner.step_once()
    before = runner.identity()
    steps, instance = runner.total_steps, runner.registry.active.instance_id
    world = json.dumps(runner.arena.snapshot_world(), sort_keys=True, default=str)
    first = switch(runner, BACKEND_ID)["ack"]["identity"]
    for _ in range(40):
        runner.reference.step_frame()
    first_dir = runner.reference.dir
    assert runner.total_steps == steps                      # the connectome run is not stepped
    back = switch(runner, "connectome-fixed")
    assert back["status"] == "ok" and back["ack"]["identity"]["backend"] == "connectome-fixed"
    after = runner.identity()
    assert {k: after[k] for k in ("run_id", "instance_id", "backend")} == \
        {k: before[k] for k in ("run_id", "instance_id", "backend")}
    assert runner.total_steps == steps and runner.registry.active.instance_id == instance
    assert json.dumps(runner.arena.snapshot_world(), sort_keys=True, default=str) == world
    pkt = packet(runner)
    assert pkt["identity"]["backend"] == "connectome-fixed" and "reference_fly" not in pkt
    assert pkt["paradigm"] == "open-arena"
    with runner.lock:
        runner.step_once()
    assert runner.total_steps == steps + 1
    # A second visit is a new, separate reference session starting from its own origin.
    second = switch(runner, BACKEND_ID)["ack"]["identity"]
    assert second["run_id"] != first["run_id"] and runner.reference.dir != first_dir
    assert runner.reference.frames == 0 and runner.reference.trail == [[0.0, 0.0]]
    assert switch(runner, "connectome-fixed")["status"] == "ok"
    runner.stop()


def test_shutdown_closes_the_reference_session(tmp_path):
    runner = graph_runner(tmp_path)
    switch(runner, BACKEND_ID)
    ref_dir = runner.reference.dir
    runner.reference.step_frame()
    runner.stop()
    assert runner.reference is None
    assert json.loads((ref_dir / "manifest.json").read_text())["status"] == "complete"


def test_reference_body_failure_is_shown_not_hidden(tmp_path):
    class Broken(FakeBody):
        def step(self, drive, substeps):
            raise RuntimeError("physics blew up")

    runner = graph_runner(tmp_path)
    runner.reference_body_factory = Broken
    switch(runner, BACKEND_ID)
    runner.reference.step_frame()
    pkt = packet(runner)
    assert pkt["halted"] is True and "physics blew up" in pkt["error"]
    assert pkt["liveness"]["state"] == "halted"
    assert switch(runner, "connectome-fixed")["status"] == "ok"
    runner.stop()


def test_status_and_clock_show_the_reference_clock_not_the_graph_clock(tmp_path):
    from neurofly_daemon import NeuroflyHTTPHandler

    runner = graph_runner(tmp_path)
    with runner.lock:
        for _ in range(4):
            runner.step_once()
    switch(runner, BACKEND_ID)
    for _ in range(25):
        runner.reference.step_frame()
    runner._publish_snapshot()
    payload = NeuroflyHTTPHandler._status_payload(SimpleNamespace(runner=runner, gateway=SimpleNamespace(describe=dict)))
    assert payload["backend"] == BACKEND_ID and payload["active_paradigm"] == "n/a (reference)"
    assert payload["active_paradigm_title"] == DISPLAY_LABEL
    assert payload["current_trial"] is None and payload["trial_elapsed_s"] is None
    assert payload["total_steps"] == 25 and payload["clocks"]["reference"]["sim_time_s"] == pytest.approx(0.05)
    assert runner.presented_step == 25 and runner.total_steps == 4
    assert sim_state(runner)["presented_step"] == 25
    switch(runner, "connectome-fixed")
    payload = NeuroflyHTTPHandler._status_payload(SimpleNamespace(runner=runner, gateway=SimpleNamespace(describe=dict)))
    assert payload["active_paradigm"] == "open-arena" and payload["total_steps"] == 4
    assert runner.presented_step == 4
    runner.stop()


def test_diagnostic_summary_of_a_reference_frame_is_scoped_as_reference():
    """Astra's repro: graph trial/learning fields must not sit beside the reference identity."""
    from learning_recorder import summarise_runner

    r = SimpleNamespace(latest_telemetry={"paradigm": "flat-ground-walking", "step": 615,
                                          "reference_fly": {"sim_time_s": 1.23, "assay": "flat-ground-walking"},
                                          "identity": {"backend": "reference-flygym"},
                                          "timing": {"achieved_speed": 0.1}},
                        start_time=0, active_paradigm_id="multisensory-sandbox", total_steps=73, sim_speed=0.1,
                        current_trial=2, trial_history=[{}], learning_curve=[0.8])
    row = summarise_runner(r)
    assert row["scope"] == "reference" and row["paradigm"] == "n/a (reference)"
    assert row["step"] == 615 and row["reference_sim_time_s"] == 1.23
    assert row["reference_assay"] == "flat-ground-walking"
    for key in ("current_trial", "trials_completed", "learning_curve_tail", "fly", "metrics", "mb_weights_mean"):
        assert row[key] is None, key
    assert row["graph_context"] == {"note": "kept graph run, not stepped while the reference fly is shown",
                                    "paradigm": "multisensory-sandbox", "step": 73, "current_trial": 2,
                                    "trials_completed": 1}
    # A graph frame keeps the unchanged summary shape.
    r.latest_telemetry = {"paradigm": "multisensory-sandbox", "step": 73, "identity": {"backend": "connectome-fixed"},
                          "fly": {"x": 1.0}, "timing": {}}
    row = summarise_runner(r)
    assert "scope" not in row and "graph_context" not in row
    assert row["paradigm"] == "multisensory-sandbox" and row["step"] == 73 and row["current_trial"] == 2
    assert row["trials_completed"] == 1 and row["learning_curve_tail"] == [0.8]
