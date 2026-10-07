"""Reference fly (not connectome): a presentation-only walking demonstration.

Backend id ``reference-flygym``.  It runs FlyGym 2.1's own published walking
controller (``flygym_demo.complex_terrain.HybridTurningController``: a tripod CPG
with leg-retraction and stumbling reflexes, driving preprogrammed single steps
recorded from real flies) on the stock NeuroMechFly body, with a constant
descending command.  Nothing here reads, loads or writes the MaleCNS connectome,
a graph, a saved brain or a learning store.

What it is: an illustrative reference controller, for showing what an
engineered walking controller published with FlyGym does on this body.

What it is not: connectome evidence, a biological qualification result, a
prediction of real fly behaviour, or a run with Assistance OFF.  The
upstream controller is itself engineered (CPG plus hand-designed reflexes).

Results go to their own namespace directory (default
``<repo>/outputs/reference_fly``) and every artifact (manifest, summary,
trace and the ``body.nfbody`` replay header) carries the label below.

Upstream: FlyGym 2.1.0 (https://github.com/NeLy-EPFL/flygym, tag v2.1.0, commit
ca65a510c2afe6ac61c51df4f274c8d190c2f95f), Apache-2.0 for code, model assets
and bundled data (one repository-wide LICENSE; no separate asset or data
licence is declared).  Cite Wang-Chen et al. (2024) Nature Methods 21:2353.
"""

from __future__ import annotations

import json
import math
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

BACKEND_ID = "reference-flygym"
DISPLAY_LABEL = "Reference fly (not connectome) — illustrative reference controller"
SHORT_LABEL = "Reference fly (not connectome)"
DISCLAIMER = (
    "Illustrative reference controller (FlyGym's published CPG + reflex walking "
    "controller). Not the connectome, not connectome evidence, not a biological "
    "qualification result and not a prediction of real fly behaviour."
)
NAMESPACE_DIRNAME = "reference_fly"
SCHEMA = "neurofly-reference-fly-run-v1"
UPSTREAM = {
    "package": "flygym",
    "version": "2.1.0",
    "repository": "https://github.com/NeLy-EPFL/flygym",
    "tag": "v2.1.0",
    "commit": "ca65a510c2afe6ac61c51df4f274c8d190c2f95f",
    "controller": "flygym_demo.complex_terrain.HybridTurningController",
    "step_data": "flygym_demo/complex_terrain/assets/single_steps_untethered.pkl",
    "step_data_sha256": "1e5b28bb6b3f50ac95a04773ea37af28d05e26d03bd90fdba57fa1b3eacfaf8c",
    "licence": {"code": "Apache-2.0", "model_assets": "Apache-2.0 (repository-wide LICENSE)",
                "data": "Apache-2.0 (repository-wide LICENSE; no separate data licence declared)"},
    "citation": "Wang-Chen S et al. (2024) NeuroMechFly v2. Nature Methods 21:2353-2362. "
                "https://doi.org/10.1038/s41592-024-02497-y",
}

# The only assay this lane can demonstrate.  It is not one of the 14 dashboard
# assays: it is open-loop straight walking on flat ground with no stimulus.
SUPPORTED_ASSAYS = ("flat-ground-walking",)
# The 14 dashboard assays (identical to experiment_registry.PARADIGMS; checked by
# tests).  None of them is implemented by the reference controller, and none
# is imitated: asking for one is refused with the reason below.
DASHBOARD_ASSAYS = (
    "open-arena", "t-maze", "y-maze", "heat-maze", "buridan", "visual-operant",
    "wind-tunnel", "looming-escape", "optomotor", "gap-crossing", "circadian-dam",
    "courtship", "labyrinth", "multisensory-sandbox",
)
UNAVAILABLE_REASON = (
    "unavailable for the reference fly: FlyGym's published walking controller has "
    "no sensory pathway for this assay, and no result is imitated"
)

# Directory names that hold connectome graphs, saved brains, checkpoints,
# learning records or qualification output.  The reference lane never writes
# under any of them.
CONNECTOME_STORE_NAMES = frozenset({
    "connectome_data", "brainlab", "brains", "checkpoints", "learning", "registry",
    "graph-bookkeeping", "validation", "manifests", "telemetry", "experiment_data",
})
CONNECTOME_STORE_PREFIXES = ("registry-", "learning-")
# A directory holding one of these files is a graph or brain store.
CONNECTOME_STORE_MARKERS = ("graph.npz", "brain_state.npz", "registry.json")

DEFAULT_DRIVE = (1.0, 1.0)


class ReferenceRefused(ValueError):
    """The reference lane refuses this request; the message says why."""


def assay_availability() -> dict[str, dict[str, Any]]:
    """Every assay the dashboard knows, plus the reference-only one, with status."""
    table = {name: {"available": True, "label": DISPLAY_LABEL} for name in SUPPORTED_ASSAYS}
    for name in DASHBOARD_ASSAYS:
        table[name] = {"available": False, "reason": UNAVAILABLE_REASON}
    return table


def check_assay(assay: str) -> str:
    if assay in SUPPORTED_ASSAYS:
        return assay
    if assay in DASHBOARD_ASSAYS:
        raise ReferenceRefused(f"assay {assay!r} is {UNAVAILABLE_REASON}")
    raise ReferenceRefused(f"unknown assay {assay!r}; the reference fly supports {SUPPORTED_ASSAYS}")


def default_namespace_root() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs" / NAMESPACE_DIRNAME


def _is_store_name(name: str) -> bool:
    return name in CONNECTOME_STORE_NAMES or name.startswith(CONNECTOME_STORE_PREFIXES)


def check_output_dir(output: Path, namespace_root: Path | None = None) -> Path:
    """Return the resolved run directory, or refuse.

    The run directory must be new, must lie inside a namespace root named
    ``reference_fly``, and neither it nor any ancestor may be a connectome,
    brain, checkpoint, learning or qualification store.
    """
    root = (Path(namespace_root) if namespace_root is not None else default_namespace_root())
    root = root.expanduser().resolve()
    if root.name != NAMESPACE_DIRNAME:
        raise ReferenceRefused(
            f"the reference namespace root must be a directory named {NAMESPACE_DIRNAME!r}, got {root}")
    out = Path(output).expanduser()
    out = (root / out if not out.is_absolute() else out).resolve()
    if out == root or root not in out.parents:
        raise ReferenceRefused(f"reference results go under {root}, not {out}")
    for part in (out, *out.parents):
        if _is_store_name(part.name):
            raise ReferenceRefused(f"refusing to write under a connectome/brain store: {part}")
        if part.is_dir() and any((part / marker).exists() for marker in CONNECTOME_STORE_MARKERS):
            raise ReferenceRefused(f"refusing to write under a graph or brain store: {part}")
    if out.exists():
        raise ReferenceRefused(f"output directory already exists: {out} (every run needs a fresh one)")
    return out


def _identity(assay: str) -> dict[str, Any]:
    return {
        "backend_id": BACKEND_ID,
        "display_label": DISPLAY_LABEL,
        "disclaimer": DISCLAIMER,
        "assay": assay,
        "is_connectome": False,
        "counts_as_connectome_evidence": False,
        "qualification_eligible": False,
        "assistance": "not applicable (no connectome; never an Assistance OFF run)",
        "graph": None,
        "upstream": UPSTREAM,
    }


def run_reference(
    output: Path,
    *,
    duration_s: float,
    seed: int = 0,
    assay: str = "flat-ground-walking",
    namespace_root: Path | None = None,
    record_fps: float = 50.0,
    physics_dt_s: float = 0.0001,
    warmup_s: float = 0.05,
    body_factory=None,
) -> dict[str, Any]:
    """Run the reference walking demonstration and write a labelled run directory."""
    assay = check_assay(assay)
    if not (math.isfinite(duration_s) and duration_s > 0):
        raise ReferenceRefused("duration must be positive")
    out = check_output_dir(output, namespace_root)
    frame_dt_s = 0.002  # one trace record and decision tick per 2 ms of simulated time
    substeps = int(round(frame_dt_s / physics_dt_s))
    if substeps < 1 or not math.isclose(substeps * physics_dt_s, frame_dt_s, rel_tol=1e-9):
        raise ReferenceRefused("physics_dt_s must divide 2 ms")
    if body_factory is None:
        from .flygym_body import FlyGymBody

        def body_factory():
            return FlyGymBody(physics_dt_s=physics_dt_s, warmup_s=warmup_s)

    out.mkdir(parents=True, exist_ok=False)
    identity = _identity(assay)
    config = {"duration_s": duration_s, "seed": seed, "physics_dt_s": physics_dt_s,
              "warmup_s": warmup_s, "descending_signal": list(DEFAULT_DRIVE),
              "trace_interval_s": frame_dt_s, "record_fps": record_fps}
    try:
        versions = {name: version(name) for name in ("flygym", "mujoco")}
    except PackageNotFoundError:
        versions = {}
    manifest = {"schema": SCHEMA, **identity, "config": config, "versions": versions,
                "status": "running"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
    body = None
    recorder = None
    try:
        body = body_factory()
        obs = body.reset(seed)
        if record_fps:
            from .body_recording import BodyRecorder

            recorder = BodyRecorder(out / "body.nfbody", fps=record_fps, neural_dt_ms=frame_dt_s * 1000)
            recorder.header(skeleton=body.skeleton(), provenance={
                **identity, "config": config,
                "neural_backend": {"controller_kind": BACKEND_ID, "display_label": DISPLAY_LABEL,
                                   "brain_backend": "none (not connectome)", "graph_sha256": None},
            })
        n = int(round(duration_s / frame_dt_s))
        x0, y0 = obs["thorax"]["position_mm"][:2]
        wall0 = time.perf_counter()
        with (out / "trace.jsonl").open("x", encoding="utf-8") as trace:
            for step in range(1, n + 1):
                obs = body.step(DEFAULT_DRIVE, substeps)
                th = obs["thorax"]
                found = [int(v > 0) for v in obs["contacts"]["found"]]
                trace.write(json.dumps({"backend_id": BACKEND_ID, "t": round(step * frame_dt_s, 6),
                                        "xyz_mm": [round(v, 5) for v in th["position_mm"]],
                                        "yaw_rad": round(th["yaw_rad"], 6), "contacts": found},
                                       sort_keys=True) + "\n")
                if recorder is not None:
                    recorder.step(step, {"neural": {}, "body": obs,
                                         "motor": {"applied_cpg_drive": list(DEFAULT_DRIVE)}},
                                  body.segment_positions(), 0)
        wall = time.perf_counter() - wall0
        recording = recorder.close() if recorder is not None else None
        recorder = None
        x1, y1 = obs["thorax"]["position_mm"][:2]
        summary = {
            "schema": SCHEMA, **identity, "status": "complete",
            "duration_s": duration_s, "records": n,
            "net_displacement_xy_mm": [x1 - x0, y1 - y0],
            "mean_forward_speed_mm_s": (x1 - x0) / duration_s,
            "wall_time_s": wall, "wall_s_per_sim_s": wall / duration_s,
            "recording": recording,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
        manifest["status"] = "complete"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
        return summary
    except BaseException as error:
        if recorder is not None:
            recorder.abort()
        manifest["status"] = "failed"
        from neurofly.privacy import redact_text

        manifest["error"] = redact_text(f"{type(error).__name__}: {error}")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
        raise
    finally:
        if body is not None:
            body.close()
