"""Deterministic run recordings (neurofly/recording.py, docs/RECORDING_FORMAT.md)."""
import gzip
import hashlib
import json
import shutil
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurofly.recording import (FORMAT, RegionMap, RunRecorder, list_recordings, read_recording,  # noqa: E402
                                record_run)
from neurofly_daemon import ContinuousExperimentRunner, NeuroflyHTTPHandler  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_same_inputs_give_identical_file(tmp_path):
    a = record_run(paradigm="t-maze", out=tmp_path / "a", steps=120)
    b = record_run(paradigm="t-maze", out=tmp_path / "b", steps=120)
    assert a["sha256"] == b["sha256"] == sha(tmp_path / "a.nfrec") == sha(tmp_path / "b.nfrec")
    rec = read_recording(tmp_path / "a.nfrec")
    assert rec["header"]["format"] == FORMAT
    assert [f["step"] for f in rec["frames"]] == list(range(121))   # frame 0 = initial state
    assert rec["end"]["frames"] == 121
    frame = rec["frames"][60]
    for key in ("fly", "body_position_mm", "joint_angles_rad", "stimuli", "dn_rates", "descending", "activity"):
        assert key in frame
    for key in ("timestamp", "timing", "path"):
        assert key not in frame
    assert rec["header"]["version"] == 2
    assert frame["identity"]["run_id"].startswith("r")
    assert frame["recording_context"]["mode"] == "replay"
    assert frame["segment_id"] == "s0"


def test_scheduled_inputs_are_recorded_and_reproducible(tmp_path):
    schedule = [{"step": 30, "cmd": {"action": "reset_trial", "advance": True}}]
    a = record_run(paradigm="t-maze", out=tmp_path / "a", steps=60, schedule=schedule)
    b = record_run(paradigm="t-maze", out=tmp_path / "b", steps=60, schedule=schedule)
    c = record_run(paradigm="t-maze", out=tmp_path / "c", steps=60)
    assert a["sha256"] == b["sha256"] != c["sha256"]
    rec = read_recording(tmp_path / "a.nfrec")
    assert rec["header"]["provenance"]["inputs"] == schedule
    assert [(e["kind"], e["step"], e["cmd"]["action"]) for e in rec["events"]] == [("command", 30, "reset_trial")]
    segments = {f["segment_id"] for f in rec["frames"]}
    assert segments == {"s0", "s1"}


def test_synthetic_graph_recording_provenance_and_determinism(tmp_path):
    kw = dict(paradigm="t-maze", steps=40, backend="connectome-fixed", test_synthetic_graph=True)
    a = record_run(out=tmp_path / "a", **kw)
    b = record_run(out=tmp_path / "b", **kw)
    assert a["sha256"] == b["sha256"]
    header = read_recording(tmp_path / "a.nfrec")["header"]
    prov = header["provenance"]
    assert prov["synthetic"] is True and prov["backend"] == "connectome-fixed"
    assert len(prov["graph"]["graph_sha256"]) == 64
    assert "commit" in prov["code"] and "root" not in prov["code"]
    assert prov["seed"] and prov["params"]["dt_s"] == 0.02
    regions = header["channels"]["regions"]
    assert regions["grouping"] == "io-channel" and sum(regions["sizes"]) == 64
    assert header["channels"]["raster"]["n"] > 0


def _initial_state(path):
    return read_recording(path, require_finished=False)["header"]["provenance"]["initial_state"]


def _restore_events(registry_root):
    events = []
    for path in sorted(Path(registry_root).rglob("events.jsonl")):
        events += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in events if e.get("kind") == "restore"]


def test_header_restored_reports_the_graph_checkpoint_restore(tmp_path):
    """initial_state.restored is about the graph instance, not runner bookkeeping.

    A store that holds only the registry (as a migrated linked child, or a copied
    registry, has no graph-bookkeeping file) used to give ``restored: false``
    although the run continued from the checkpoint's step with a restore event.
    """
    kw = dict(paradigm="t-maze", backend="connectome-fixed", test_synthetic_graph=True)
    fresh = _initial_state(record_run(out=tmp_path / "fresh", steps=3, **kw)["path"])
    assert fresh["restored"] is False and fresh["restore_source"] is None
    assert fresh["graph_step_index"] == 0

    learned = tmp_path / "learned"
    record_run(out=tmp_path / "learn", steps=11, state_dir=learned,
               schedule=[{"step": 10, "cmd": {"action": "save_checkpoint"}}], **kw)
    assert (learned / "graph-bookkeeping").is_dir()
    # Registry only: exactly the state layout of the defect report.
    registry_only = tmp_path / "registry-only"
    shutil.copytree(learned / "registry-v3", registry_only / "registry-v3")
    assert not (registry_only / "graph-bookkeeping").exists()
    before = _restore_events(registry_only / "registry-v3")
    state = _initial_state(record_run(out=tmp_path / "continued", steps=2, state_dir=registry_only, **kw)["path"])
    restores = _restore_events(registry_only / "registry-v3")[len(before):]
    assert [(e["version"], e["step"]) for e in restores] == [(1, 10)]
    assert state["graph_step_index"] == 10
    assert state["restored"] is True
    assert state["restore_source"] == dict(kind="graph_checkpoint", checkpoint_version=1, step_index=10,
                                           fallback=False)

    # With the bookkeeping file present the answer is the same (it never decided it).
    full = _initial_state(record_run(out=tmp_path / "continued-full", steps=2, state_dir=learned, **kw)["path"])
    assert full["restored"] is True and full["restore_source"]["step_index"] == 10


def test_header_restored_for_modular_brain_file(tmp_path):
    state_dir = tmp_path / "state"
    naive = _initial_state(record_run(paradigm="t-maze", out=tmp_path / "a", steps=3, state_dir=state_dir,
                                      schedule=[{"step": 2, "cmd": {"action": "save_checkpoint"}}])["path"])
    assert naive["restored"] is False and naive["restore_source"] is None
    assert "graph_step_index" not in naive
    assert (state_dir / "brains" / "t-maze.json").is_file()
    again = _initial_state(record_run(paradigm="t-maze", out=tmp_path / "b", steps=2, state_dir=state_dir)["path"])
    assert again["restored"] is True and again["restore_source"] == {"kind": "modular_brain_file"}
    assert again["brain_steps"] == 2


def test_record_cli_defaults_to_v3_dynamics(tmp_path, monkeypatch):
    from neurofly import recording
    # record_run writes NEUROFLY_LIF_DYNAMICS directly for the explicit v1 call below.
    # delenv on an absent variable registers no undo, so setenv first: monkeypatch then
    # restores the variable's original state after the test instead of leaking v1.
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v3")
    monkeypatch.delenv("NEUROFLY_LIF_DYNAMICS")
    base = ["--paradigm", "t-maze", "--steps", "3", "--backend", "connectome-fixed", "--test-synthetic-graph"]
    assert recording.main(base + ["--out", str(tmp_path / "v3")]) == 0
    assert read_recording(tmp_path / "v3.nfrec")["header"]["provenance"]["lif_dynamics"] == "v3"
    assert recording.main(base + ["--out", str(tmp_path / "v1"), "--dynamics", "v1"]) == 0
    assert read_recording(tmp_path / "v1.nfrec")["header"]["provenance"]["lif_dynamics"] == "v1"
    assert recording.main(["--paradigm", "t-maze", "--steps", "2", "--out", str(tmp_path / "m")]) == 0
    assert read_recording(tmp_path / "m.nfrec")["header"]["provenance"]["lif_dynamics"] is None


def test_recording_names_the_brain_backend(tmp_path, monkeypatch):
    from neurofly import recording
    # record_run writes NEUROFLY_LIF_DYNAMICS (the CLI's default v3); register an undo.
    monkeypatch.setenv("NEUROFLY_LIF_DYNAMICS", "v3")
    monkeypatch.delenv("NEUROFLY_LIF_DYNAMICS")
    base = ["--paradigm", "t-maze", "--steps", "2", "--backend", "connectome-fixed", "--test-synthetic-graph"]
    assert recording.main(base + ["--out", str(tmp_path / "g")]) == 0
    assert read_recording(tmp_path / "g.nfrec")["header"]["provenance"]["brain_backend"] == "cpu"  # conftest pins cpu
    assert recording.main(["--paradigm", "t-maze", "--steps", "2", "--out", str(tmp_path / "m")]) == 0
    assert read_recording(tmp_path / "m.nfrec")["header"]["provenance"]["brain_backend"] is None


def test_region_rates_and_sparse_spikes(tmp_path):
    runner = ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=tmp_path, backend="connectome-fixed",
                                        test_synthetic_graph=True, checkpoint_interval=1e9)
    with runner.lock:
        recorder = RunRecorder(runner, tmp_path / "r", raster="all")
        counts = np.zeros(runner.shared_graph.n, dtype=np.int32)
        counts[[10, 11, 50]] = [2, 1, 3]
        runner.total_steps += 1
        runner.graph_controller.last_counts = counts
        active = runner.registry.active      # injected counts belong to the displayed owner
        runner.graph_controller.last_counts_owner = (active.instance_id, runner.active_paradigm_id, int(active.step_index))
        assert recorder.capture(runner, {})
        recorder.close()
    rec = read_recording(tmp_path / "r.nfrec")
    names = rec["header"]["channels"]["regions"]["names"]
    frame = rec["frames"][-1]
    rates = dict(zip(names, frame["activity"]))
    window = runner.graph_controller.step_ms / 1000.0
    assert rates["dna02_l"] == pytest.approx(2 / window)
    assert rates["dna02_r"] == pytest.approx(1 / window)
    assert rates["other"] == pytest.approx(3 / (52 * window), abs=1e-3)
    assert frame["spikes"] == [10, 2, 11, 1, 50, 3]


def test_region_map_from_labels():
    regions = RegionMap.from_labels("superclass", ["b", "a", None, "b"], "test")
    assert regions.names == ["a", "b", "unknown"]
    assert regions.sizes.tolist() == [1, 2, 1]
    assert regions.rates(np.array([0, 2, 1, 4]), 0.5) == [4.0, 4.0, 2.0]


def test_daemon_record_commands_and_endpoints(tmp_path):
    from tests.transition_control_helpers import transition_command
    runner = ContinuousExperimentRunner(initial_paradigm="t-maze", output_dir=tmp_path, checkpoint_interval=1e9)
    res = runner.dispatch_command({"action": "record_start", "name": "demo"})
    assert res["status"] == "ok", res
    assert runner.dispatch_command({"action": "record_start"})["status"] == "error"
    assert runner.dispatch_command({"action": "record_start", "name": "../x"})["status"] == "error"
    for _ in range(10):
        with runner.lock:
            runner.step_once(publish=False)
    runner.dispatch_command({"action": "set_speed", "speed": 3})          # not an input
    assert transition_command(runner, {"action": "set_learning", "enabled": False})["status"] == "ok"  # an input
    assert runner.dispatch_command({"action": "switch_backend", "backend": "connectome-fixed"})["status"] == "error"
    done = runner.dispatch_command({"action": "record_stop"})
    assert done["status"] == "ok" and done["recording"]["frames"] == 11
    assert runner.dispatch_command({"action": "record_stop"})["status"] == "error"
    rec = read_recording(tmp_path / "recordings" / "demo.nfrec")
    assert [e["cmd"]["action"] for e in rec["events"]] == ["set_learning"]
    sidecar = json.loads((tmp_path / "recordings" / "demo.nfrec.json").read_text())
    assert sidecar["daemon_run_id"] == runner.run_id and sidecar["recording"]["sha256"] == done["recording"]["sha256"]
    assert [r["name"] for r in list_recordings(tmp_path / "recordings")] == ["demo.nfrec"]

    NeuroflyHTTPHandler.runner = runner
    server = ThreadingHTTPServer(("127.0.0.1", 0), NeuroflyHTTPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        listing = json.loads(urllib.request.urlopen(f"{base}/api/recordings").read())
        assert listing["recordings"][0]["name"] == "demo.nfrec" and listing["active"] is None
        body = urllib.request.urlopen(f"{base}/api/recordings/demo.nfrec").read()
        assert hashlib.sha256(body).hexdigest() == done["recording"]["sha256"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/api/recordings/..%2Fcheckpoints")
    finally:
        server.shutdown()
        server.server_close()


def test_corrupt_or_truncated_recording_is_rejected(tmp_path):
    record_run(paradigm="t-maze", out=tmp_path / "a", steps=5)
    lines = gzip.decompress((tmp_path / "a.nfrec").read_bytes()).splitlines(keepends=True)
    tampered = lines[:3] + [lines[3].replace(b'"step":2', b'"step":9')] + lines[4:]
    (tmp_path / "t.nfrec").write_bytes(gzip.compress(b"".join(tampered)))
    with pytest.raises(ValueError, match="digest"):
        read_recording(tmp_path / "t.nfrec", require_finished=False)   # content check only
    (tmp_path / "u.nfrec").write_bytes(gzip.compress(b"".join(lines[:-1])))
    with pytest.raises(ValueError, match="incomplete"):
        read_recording(tmp_path / "u.nfrec", require_finished=False)


def test_predators_are_seeded():
    from arena import Arena

    def trajectory():
        arena = Arena(paradigm=None, seed=7, num_flies=1, num_predators=2)
        out = []
        for _ in range(200):
            arena.step(0.02)
            out.append([(round(p.pos.x, 9), round(p.pos.y, 9), round(p.heading, 9)) for p in arena.predators])
        return out

    assert trajectory() == trajectory()
