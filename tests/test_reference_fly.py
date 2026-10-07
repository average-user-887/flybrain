"""Reference fly (not connectome): labelling, namespace separation, store refusal."""
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurofly_body import cli, reference
from neurofly_body.reference import (BACKEND_ID, DASHBOARD_ASSAYS, DISPLAY_LABEL, ReferenceRefused,
                                     check_output_dir, run_reference)

REPO = Path(__file__).resolve().parent.parent


class FakeBody:
    """Moves the thorax forward 0.01 mm per step; no FlyGym needed."""

    def __init__(self):
        self.x = 0.0
        self.closed = False

    def reset(self, seed):
        self.x = 0.0
        return self._obs()

    def step(self, drive, substeps):
        assert tuple(drive) == reference.DEFAULT_DRIVE
        self.x += 0.01
        return self._obs()

    def _obs(self):
        return {"thorax": {"position_mm": [self.x, 0.0, 0.3], "yaw_rad": 0.0},
                "contacts": {"found": [1, 0, 1, 0, 1, 0]}}

    def skeleton(self):
        return {"segments": ["c_thorax"], "parents": [-1], "units": "mm"}

    def segment_positions(self):
        return np.array([[self.x, 0.0, 0.3]])

    def close(self):
        self.closed = True


def _run(tmp_path, name="walk", **kw):
    root = tmp_path / "reference_fly"
    summary = run_reference(name, duration_s=0.1, namespace_root=root, body_factory=FakeBody, **kw)
    return root / name, summary


# --- labelling -------------------------------------------------------------

def test_label_wording():
    assert "not connectome" in DISPLAY_LABEL and "illustrative reference controller" in DISPLAY_LABEL
    for text in (reference.__doc__, reference.DISCLAIMER, DISPLAY_LABEL,
                 (REPO / "neurofly_body/reference.py").read_text(encoding="utf-8")):
        assert "what a real fly would do" not in text.lower().replace('"', "")


def test_every_artifact_is_labelled(tmp_path):
    out, summary = _run(tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    for doc in (summary, manifest, json.loads((out / "summary.json").read_text())):
        assert doc["backend_id"] == BACKEND_ID and doc["display_label"] == DISPLAY_LABEL
        assert doc["is_connectome"] is False and doc["counts_as_connectome_evidence"] is False
        assert doc["qualification_eligible"] is False and doc["graph"] is None
        assert "Assistance OFF" in doc["assistance"] and doc["assistance"].startswith("not applicable")
    lines = (out / "trace.jsonl").read_text().splitlines()
    assert len(lines) == 50 and all(json.loads(l)["backend_id"] == BACKEND_ID for l in lines)
    with gzip.open(out / "body.nfbody", "rt") as stream:
        header = json.loads(stream.readline())
    prov = header["provenance"]
    assert prov["backend_id"] == BACKEND_ID and prov["display_label"] == DISPLAY_LABEL
    assert prov["neural_backend"]["graph_sha256"] is None
    assert prov["neural_backend"]["controller_kind"] == BACKEND_ID
    assert summary["mean_forward_speed_mm_s"] == pytest.approx(5.0)


def test_replay_page_shows_reference_identity():
    js = (REPO / "web/embodied_replay.js").read_text(encoding="utf-8")
    html = (REPO / "web/embodied_replay.html").read_text(encoding="utf-8")
    assert "p.backend_id !== 'reference-flygym'" in js
    assert "'none (not connectome)'" in js and 'id="identity"' in html


def test_body_constructor_failure_marks_manifest_failed(tmp_path):
    def broken_body():
        raise RuntimeError("no FlyGym here")

    root = tmp_path / "reference_fly"
    with pytest.raises(RuntimeError, match="no FlyGym here"):
        run_reference("bad-init", duration_s=0.1, namespace_root=root, body_factory=broken_body)
    manifest = json.loads((root / "bad-init" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "RuntimeError: no FlyGym here"
    assert manifest["backend_id"] == BACKEND_ID
    assert not (root / "bad-init" / "summary.json").exists()


def test_failure_after_construction_closes_the_body(tmp_path):
    bodies = []

    class FailingBody(FakeBody):
        def step(self, drive, substeps):
            raise RuntimeError("physics blew up")

    def factory():
        bodies.append(FailingBody())
        return bodies[-1]

    with pytest.raises(RuntimeError, match="physics blew up"):
        run_reference("bad-step", duration_s=0.1, namespace_root=tmp_path / "reference_fly",
                      body_factory=factory)
    assert bodies[0].closed
    manifest = json.loads((tmp_path / "reference_fly" / "bad-step" / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"] == "RuntimeError: physics blew up"


# --- namespace separation ---------------------------------------------------

def test_default_namespace_is_separate():
    root = reference.default_namespace_root()
    assert root.name == "reference_fly" and root.parent == REPO / "outputs"


@pytest.mark.parametrize("bad", ["../elsewhere", "/"])
def test_refuses_outside_namespace(tmp_path, bad):
    with pytest.raises(ReferenceRefused):
        check_output_dir(Path(bad), tmp_path / "reference_fly")


def test_refuses_namespace_root_with_another_name(tmp_path):
    with pytest.raises(ReferenceRefused, match="named 'reference_fly'"):
        check_output_dir("run", tmp_path / "outputs")


@pytest.mark.parametrize("store", ["connectome_data", "brainlab", "brains", "checkpoints",
                                   "registry", "registry-v3", "learning", "validation"])
def test_refuses_connectome_store_ancestors(tmp_path, store):
    with pytest.raises(ReferenceRefused, match="store"):
        check_output_dir("run", tmp_path / store / "reference_fly")


def test_refuses_directory_holding_a_graph(tmp_path):
    graph_dir = tmp_path / "malecns" / "reference_fly"
    graph_dir.mkdir(parents=True)
    (tmp_path / "malecns" / "graph.npz").write_bytes(b"graph")
    with pytest.raises(ReferenceRefused, match="graph or brain store"):
        check_output_dir("run", graph_dir)


def test_refuses_existing_run_directory(tmp_path):
    _run(tmp_path)
    with pytest.raises(ReferenceRefused, match="already exists"):
        _run(tmp_path)


# --- never touches connectome stores ------------------------------------------

def test_run_leaves_connectome_store_bytes_unchanged(tmp_path):
    store = tmp_path / "outputs" / "brainlab" / "malecns_v1"
    store.mkdir(parents=True)
    (store / "graph.npz").write_bytes(b"weights" * 100)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in store.iterdir()}
    root = tmp_path / "outputs" / "reference_fly"
    run_reference("walk", duration_s=0.05, namespace_root=root, body_factory=FakeBody)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in store.iterdir()}
    assert before == after
    assert sorted(p.name for p in (root / "walk").iterdir()) == [
        "body.nfbody", "manifest.json", "summary.json", "trace.jsonl"]


def test_reference_never_imports_connectome_code(tmp_path):
    code = (
        "import sys, numpy as np\n"
        "from tests.test_reference_fly import FakeBody\n"
        "from neurofly_body.reference import run_reference\n"
        f"run_reference('w', duration_s=0.02, namespace_root={str(tmp_path / 'reference_fly')!r},"
        " body_factory=FakeBody)\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('brainlab', 'experiment_registry', 'experiment_brains', 'circuit', 'validation')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


# --- unsupported assays -----------------------------------------------------

def test_dashboard_assays_match_registry():
    registry = pytest.importorskip("experiment_registry")
    assert DASHBOARD_ASSAYS == registry.PARADIGMS


@pytest.mark.parametrize("assay", DASHBOARD_ASSAYS)
def test_dashboard_assays_are_explicitly_unavailable(tmp_path, assay):
    with pytest.raises(ReferenceRefused, match="unavailable for the reference fly"):
        _run(tmp_path, assay=assay)
    assert not (tmp_path / "reference_fly").exists()
    entry = reference.assay_availability()[assay]
    assert entry == {"available": False, "reason": reference.UNAVAILABLE_REASON}


def test_cli_refuses_unavailable_assay(tmp_path):
    with pytest.raises(SystemExit, match="Reference fly \\(not connectome\\).*optomotor"):
        cli.main(["reference", "--duration", "0.1", "--output", "x", "--assay", "optomotor",
                  "--namespace-root", str(tmp_path / "reference_fly")])


# --- connectome tooling refuses reference runs ------------------------------------

def test_replay_check_and_studio_refuse_reference_runs(tmp_path):
    out, _ = _run(tmp_path)
    with pytest.raises(SystemExit, match="Reference fly"):
        cli.main(["replay-check", str(out), "--output", str(tmp_path / "re")])
    from neurofly_studio.curate import CurationError, check_run
    from neurofly_studio.export import build_bundle

    with pytest.raises(ValueError, match="Reference fly"):
        build_bundle(out)
    with pytest.raises(CurationError, match="Reference fly"):
        check_run(out)


def test_real_flygym_reference_walks(tmp_path):
    pytest.importorskip("flygym")
    pytest.importorskip("flygym_demo")
    root = tmp_path / "reference_fly"
    summary = run_reference("real", duration_s=0.1, namespace_root=root)
    assert summary["backend_id"] == BACKEND_ID and summary["records"] == 50
    assert summary["net_displacement_xy_mm"][0] > 0.2  # the stock controller moves the body
