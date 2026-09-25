"""Experiment studio (P5): catalog badges, experiment files, the queue API and its guards."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from neurofly_studio import catalog, experiment, metrics
from neurofly_studio.server import Studio, make_server
from tests.studio_fakes import fake_runner

GOOD = {"schema": "neurofly-studio-experiment-v1", "title": "Slow spin", "paradigm": "optomotor",
        "parameters": {"world_angular_velocity_rad_s": 2.0, "contrast": 0.5, "duration_s": 0.5, "seed": 7},
        "repeats": 2, "control": "output-disconnected", "controller": "modular"}


# -- catalog ----------------------------------------------------------------
def test_badges_follow_the_matrix_wording():
    assert catalog.badge_for("Tested on v3 (receipt x)") == "Validated"
    assert catalog.badge_for("**Mapped, untested on v3**") == "Mapped"
    assert catalog.badge_for("Unmapped") == "Exploratory"


def test_catalog_lists_all_14_matrix_paradigms_with_badges():
    data = catalog.catalog()
    ids = [p["id"] for p in data["paradigms"]]
    assert len(ids) == 14 and "optomotor" in ids and "multisensory-sandbox" in ids
    for p in data["paradigms"]:
        assert p["badge"] == catalog.badge_for(p["connectome_status"])
    buildable = [p["id"] for p in data["paradigms"] if p["buildable"]]
    assert buildable == ["optomotor"]


def test_no_paradigm_shows_validated_without_a_tested_row(tmp_path):
    matrix = tmp_path / "m.md"
    matrix.write_text(
        "| # | Paradigm | Sensory | Motor | Modular | IO | Connectome status | Plasticity | Receipts |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| 9 | Optomotor (`optomotor`) | a | b | c | d | Tested on v3: PASS | n/a | r |\n"
        "| 8 | Looming (`looming-escape`) | a | b | c | d | Mapped, untested on v3 | n/a | r |\n",
        encoding="utf-8")
    rows = {r["id"]: r["badge"] for r in catalog.read_matrix(matrix)}
    assert rows == {"optomotor": "Validated", "looming-escape": "Mapped"}


def test_preregistered_specs_are_listed_with_their_hash():
    specs = {s["id"]: s for p in catalog.catalog()["paradigms"] for s in p["preregistered_specs"]}
    assert "optomotor-yaw-v3-2" in specs
    assert len(specs["optomotor-yaw-v3-2"]["sha256"]) == 64


# -- experiment files ---------------------------------------------------------
def test_experiment_is_normalised_and_always_exploratory():
    exp = experiment.validate({"paradigm": "optomotor"})
    assert exp["label"] == "exploratory" and exp["controller"] == "connectome"
    assert exp["parameters"] == {"world_angular_velocity_rad_s": 4.0, "contrast": 1.0,
                                 "duration_s": 5.0, "seed": 1}


@pytest.mark.parametrize("change, message", [
    ({"label": "confirmatory"}, "exploratory"),
    ({"paradigm": "looming-escape"}, "paradigm"),
    ({"neural_dt_ms": 5.0}, "unknown field"),
    ({"parameters": {"physics_dt_s": 0.001}}, "no parameter"),
    ({"parameters": {"contrast": 1.5}}, "between"),
    ({"parameters": {"seed": 1.5}}, "whole number"),
    ({"parameters": {"duration_s": 1.05}}, "0.1 s"),
    ({"parameters": {"contrast": float("nan")}}, "finite"),
    ({"parameters": {"contrast": True}}, "finite"),
    ({"repeats": 7}, "repeats"),
    ({"control": "silence-everything"}, "control"),
    ({"controller": "fast"}, "controller"),
    ({"title": "x" * 121}, "title"),
    ({"silence": ["DNa02"], "controller": "modular"}, "connectome"),
    ({"silence": ["DNa02; rm -rf /"], "controller": "connectome"}, "not a cell type"),
    ({"silence": ["DNa02", "DNa02"], "controller": "connectome"}, "repeated"),
    ({"silence": ["DNa02:X"], "controller": "connectome"}, "not a cell type"),
    ({"silence": "DNa02", "controller": "connectome"}, "list"),
    ({"control": "intact"}, "needs something silenced"),
])
def test_bad_experiments_are_refused(change, message):
    raw = {**GOOD, **change}
    with pytest.raises(experiment.ExperimentError, match=message):
        experiment.validate(raw, silence_supported=True)


def test_silencing_is_refused_when_the_runner_lacks_the_flag():
    with pytest.raises(experiment.ExperimentError, match="no --silence"):
        experiment.validate({**GOOD, "controller": "connectome", "silence": ["DNa02"]},
                            silence_supported=False)


def test_silenced_experiment_pairs_with_the_same_fly_unsilenced():
    exp = experiment.validate({**GOOD, "controller": "connectome", "repeats": 1,
                               "silence": ["DNa02", "MDN:L"], "control": "intact"},
                              silence_supported=True)
    silenced, intact = experiment.plan(exp)
    assert silenced.argv.count("--silence") == 2 and "MDN:L" in silenced.argv
    assert intact.role == "intact" and "--silence" not in intact.argv and intact.silence == []
    assert intact.argv[-2:] == ["--mode", "intact"]
    exp = experiment.validate({**GOOD, "controller": "connectome", "repeats": 1,
                               "silence": ["DNa02"], "control": "output-disconnected"},
                              silence_supported=True)
    _, control = experiment.plan(exp)
    # The disconnected control keeps the same brain, silencing included.
    assert control.silence == ["DNa02"] and control.argv[-2:] == ["--mode", "output-disconnected"]


def test_silence_groups_follow_runner_support(monkeypatch):
    monkeypatch.setattr(catalog, "runner_supports_silence", lambda: False)
    optomotor = next(p for p in catalog.catalog()["paradigms"] if p["id"] == "optomotor")
    assert optomotor["silence_groups"] == []
    monkeypatch.setattr(catalog, "runner_supports_silence", lambda: True)
    optomotor = next(p for p in catalog.catalog()["paradigms"] if p["id"] == "optomotor")
    groups = {g["id"]: g["cell_types"] for g in optomotor["silence_groups"]}
    assert groups["dna02"] == ["DNa02"] and groups["hs"] == ["HSN", "HSE", "HSS"]


def test_metrics_surface_a_leaking_clamp(tmp_path):
    (tmp_path / "telemetry.jsonl").write_text(json.dumps(
        {"run_time_s": 0.002, "sensory": {"body_yaw_velocity_rad_s": 0.0, "world_angular_velocity_rad_s": 4.0},
         "body": {"thorax": {"yaw_rad": 0.0}}, "neural": {"total_step_spikes": 3}}) + "\n")
    assert metrics.run_metrics(tmp_path)["silenced"] is None
    (tmp_path / "summary.json").write_text(json.dumps({"silenced": {
        "targets": ["DNa02"], "total_neurons": 2, "spikes_total": 1, "clamp_held": False, "drive": -200}}))
    assert metrics.run_metrics(tmp_path)["silenced"] == {
        "targets": ["DNa02"], "total_neurons": 2, "spikes_total": 1, "clamp_held": False}


def test_plan_pairs_each_seed_with_its_control():
    runs = experiment.plan(experiment.validate(GOOD), graph_args=["--graph-dir", "/g"])
    assert [(r.seed, r.role) for r in runs] == [(7, "experiment"), (7, "output-disconnected"),
                                                (8, "experiment"), (8, "output-disconnected")]
    assert runs[1].name == runs[0].name + "-control"
    first = runs[0].argv
    assert first[first.index("--seed") + 1] == "7" and first[-2:] == ["--mode", "intact"]
    assert "--graph-dir" not in first  # modular runs do not load the graph
    assert runs[1].argv[-2:] == ["--mode", "output-disconnected"]
    connectome = experiment.plan(experiment.validate({**GOOD, "controller": "connectome"}),
                                 graph_args=["--graph-dir", "/g"])
    assert "--graph-dir" in connectome[0].argv
    for run in runs:  # full-accuracy defaults are never overridden
        assert not {"--neural-dt-ms", "--physics-dt-s", "--record-fps"} & set(run.argv)


# -- server -----------------------------------------------------------------
@pytest.fixture
def served(tmp_path):
    studio = Studio(tmp_path / "queue", tmp_path / "curated", runner=fake_runner)
    server = make_server(studio, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield studio, base
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def _post(url, payload, headers=None):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_page_and_catalog_are_served(served):
    _, base = served
    status, ctype, body = _get(base + "/")
    assert status == 200 and ctype.startswith("text/html") and b"Experiment Studio" in body
    status, _, body = _get(base + "/embodied_replay.html")
    assert status == 200
    assert len(json.loads(_get(base + "/api/studio/catalog")[2])["paradigms"]) == 14
    with pytest.raises(urllib.error.HTTPError) as error:
        _get(base + "/../neurofly_studio/server.py")
    assert error.value.code == 404


def test_submit_run_and_fetch_results(served):
    studio, base = served
    status, reply = _post(base + "/api/studio/experiments", {**GOOD, "repeats": 1})
    assert status == 201 and len(reply["queued"]) == 2
    intact, control = reply["queued"]
    # Not finished yet: files are refused.
    with pytest.raises(urllib.error.HTTPError):
        _get(f"{base}/api/studio/files/queue/{intact}/summary.json")
    studio.start_worker(poll_s=0.05)
    deadline = time.time() + 60
    while time.time() < deadline:
        runs = json.loads(_get(base + "/api/studio/runs")[2])
        if all(job["state"] == "done" for job in runs["queue"]):
            break
        time.sleep(0.1)
    jobs = {job["name"]: job for job in runs["queue"]}
    assert jobs[intact]["state"] == "done" and jobs[intact]["has_recording"]
    assert jobs[intact]["pair"] == control and jobs[control]["pair"] == intact
    assert jobs[intact]["label"] == "exploratory"
    status, ctype, body = _get(f"{base}/api/studio/files/queue/{intact}/body.nfbody")
    assert ctype == "application/gzip" and body[:2] == b"\x1f\x8b"
    moving = json.loads(_get(f"{base}/api/studio/metrics/queue/{intact}")[2])
    still = json.loads(_get(f"{base}/api/studio/metrics/queue/{control}")[2])
    assert moving["mean_body_yaw_velocity_rad_s"] > 0.1
    assert still["mean_body_yaw_velocity_rad_s"] == 0.0  # commands never reach the body
    assert studio.start_worker() != "running"  # one worker per queue


def test_bad_submissions_and_cross_origin_posts_are_refused(served):
    studio, base = served
    status, reply = _post(base + "/api/studio/experiments", {**GOOD, "label": "confirmatory"})
    assert status == 400 and "exploratory" in reply["error"]
    status, _ = _post(base + "/api/studio/experiments", GOOD, {"Origin": "http://evil.example"})
    assert status == 403
    request = urllib.request.Request(base + "/api/studio/experiments", data=b"{}", method="POST",
                                     headers={"Content-Type": "text/plain"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=10)
    assert error.value.code == 415
    assert studio.runs()["queue"] == []


def test_pending_jobs_can_be_cancelled(served):
    studio, base = served
    _, reply = _post(base + "/api/studio/experiments", {**GOOD, "repeats": 1, "control": None})
    name = reply["queued"][0]
    status, _ = _post(f"{base}/api/studio/runs/{name}/cancel", {})
    assert status == 200 and studio.runs()["queue"] == []
    status, _ = _post(f"{base}/api/studio/runs/{name}/cancel", {})
    assert status == 404


def test_curated_runs_are_listed(tmp_path):
    run = tmp_path / "curated" / "optomotor-intro"
    run.mkdir(parents=True)
    (run / "curated.json").write_text(json.dumps({"title": "Intro", "paradigm": "optomotor"}))
    studio = Studio(tmp_path / "queue", tmp_path / "curated")
    curated = studio.runs()["curated"]
    assert [c["name"] for c in curated] == ["optomotor-intro"] and not curated[0]["has_recording"]
    assert studio.run_dir("curated", "optomotor-intro") == run
    with pytest.raises(FileNotFoundError):
        studio.run_dir("curated", "../queue")


@pytest.mark.skipif(not catalog.runner_supports_silence(), reason="neurofly_body run has no --silence yet (PR #31)")
def test_silenced_pair_runs_end_to_end(served):
    studio, base = served
    status, reply = _post(base + "/api/studio/experiments",
                          {**GOOD, "controller": "connectome", "repeats": 1, "silence": ["DNa02"],
                           "control": "intact"})
    assert status == 201, reply
    silenced, intact = reply["queued"]
    studio.start_worker(poll_s=0.05)
    deadline = time.time() + 60
    while time.time() < deadline:
        jobs = {j["name"]: j for j in studio.runs()["queue"]}
        if all(j["state"] == "done" for j in jobs.values()):
            break
        time.sleep(0.1)
    assert jobs[silenced]["silence"] == ["DNa02"] and jobs[silenced]["clamp_held"] is True
    assert jobs[intact]["silence"] == [] and jobs[intact]["clamp_held"] is None
