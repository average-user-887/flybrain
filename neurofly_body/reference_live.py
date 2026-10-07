"""Live Reference fly (not connectome) for the daemon and dashboard.

One route only: flat-ground walking with FlyGym's published controller (see
``neurofly_body/reference.py``).  The daemon holds at most one ``ReferenceLive``
while the dashboard shows the reference fly.  It owns its own body, clock, run
directory (``<daemon output>/reference_fly/live-...``) and labelled recording;
it never reads or writes the daemon's brains, graph, registry, checkpoints or
learning records, and the connectome run it replaced is left exactly as it was
(not stepped) until the reference is closed.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Optional

from .reference import (BACKEND_ID, DASHBOARD_ASSAYS, DEFAULT_DRIVE, DISCLAIMER, DISPLAY_LABEL,
                        NAMESPACE_DIRNAME, SCHEMA, SUPPORTED_ASSAYS, UNAVAILABLE_REASON, _identity,
                        assay_availability, check_output_dir)

ASSAY = SUPPORTED_ASSAYS[0]
FRAME_DT_S = 0.002          # one controller frame (20 physics substeps of 0.1 ms)
TRACE_EVERY = 5             # trace.jsonl row every 10 ms simulated
RECORD_FPS = 50.0           # body.nfbody frames per simulated second
TRAIL_EVERY = 10            # dashboard trail point every 20 ms simulated
TRAIL_MAX = 600             # points kept for the dashboard (12 s)
RECORD_MAX_S = 600.0        # files stop growing after this much simulated time


class ReferenceLive:
    """A live, labelled reference-fly session with its own namespace directory."""

    def __init__(self, namespace_root: Path, *, session_id: str,
                 body_factory: Optional[Callable[[], Any]] = None,
                 physics_dt_s: float = 0.0001, warmup_s: float = 0.05,
                 record_max_s: float = RECORD_MAX_S, seed: int = 0) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        safe_session = "".join(c for c in str(session_id) if c.isalnum())[:12] or "session"
        self.run_id = f"reference-{uuid.uuid4().hex}"
        self.dir = check_output_dir(f"live-{stamp}-{safe_session}-{self.run_id[-6:]}", namespace_root)
        self.namespace_root = Path(namespace_root).resolve()
        self.substeps = int(round(FRAME_DT_S / physics_dt_s))
        if self.substeps < 1 or not math.isclose(self.substeps * physics_dt_s, FRAME_DT_S, rel_tol=1e-9):
            raise ValueError("physics_dt_s must divide 2 ms")
        self.record_max_s = float(record_max_s)
        self.frames = 0
        self.error: Optional[str] = None
        self.closed = False
        self.trail: list[list[float]] = []
        self.started_wall = time.time()
        self._body = None
        self._recorder = None
        self._trace = None
        self._recording_capped = False
        self.dir.mkdir(parents=True, exist_ok=False)
        self.config = {"assay": ASSAY, "seed": seed, "physics_dt_s": physics_dt_s, "warmup_s": warmup_s,
                       "descending_signal": list(DEFAULT_DRIVE), "frame_dt_s": FRAME_DT_S,
                       "trace_interval_s": FRAME_DT_S * TRACE_EVERY, "record_fps": RECORD_FPS,
                       "record_max_s": self.record_max_s, "mode": "live dashboard session"}
        try:
            versions = {name: version(name) for name in ("flygym", "mujoco")}
        except PackageNotFoundError:
            versions = {}
        self.manifest = {"schema": SCHEMA, **_identity(ASSAY), "run_id": self.run_id,
                         "config": self.config, "versions": versions, "status": "running"}
        self._write_manifest()
        try:
            if body_factory is None:
                from .flygym_body import FlyGymBody

                def body_factory():
                    return FlyGymBody(physics_dt_s=physics_dt_s, warmup_s=warmup_s)
            self._body = body_factory()
            self.obs = self._body.reset(seed)
            from .body_recording import BodyRecorder

            self._recorder = BodyRecorder(self.dir / "body.nfbody", fps=RECORD_FPS,
                                          neural_dt_ms=FRAME_DT_S * 1000)
            self._recorder.header(skeleton=self._body.skeleton(), provenance={
                **_identity(ASSAY), "run_id": self.run_id, "config": self.config,
                "neural_backend": {"controller_kind": BACKEND_ID, "display_label": DISPLAY_LABEL,
                                   "brain_backend": "none (not connectome)", "graph_sha256": None},
            })
            self._trace = (self.dir / "trace.jsonl").open("x", encoding="utf-8")
        except BaseException as error:
            self._fail(error)
            self._release()
            raise
        x, y = self.obs["thorax"]["position_mm"][:2]
        self.origin = (float(x), float(y))
        self.trail.append([0.0, 0.0])

    # ------------------------------------------------------------------ state
    @property
    def sim_time_s(self) -> float:
        return round(self.frames * FRAME_DT_S, 6)

    def _write_manifest(self) -> None:
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")

    def _fail(self, error: BaseException) -> None:
        from neurofly.privacy import redact_text

        self.error = redact_text(f"{type(error).__name__}: {error}")
        self.manifest["status"] = "failed"
        self.manifest["error"] = self.error
        self._write_manifest()

    def store_path(self) -> str:
        from neurofly.privacy import portable_path

        return portable_path(self.dir)

    # ------------------------------------------------------------------ stepping
    def step_frame(self) -> None:
        """Advance one 2 ms frame.  A failure stops the session (recorded, never hidden)."""
        if self.closed or self.error is not None:
            return
        try:
            self.obs = self._body.step(DEFAULT_DRIVE, self.substeps)
            self.frames += 1
            th = self.obs["thorax"]
            x, y = (float(v) for v in th["position_mm"][:2])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise FloatingPointError("the reference body produced a non-finite pose")
            if self.frames % TRAIL_EVERY == 0:
                self.trail.append([round(x - self.origin[0], 4), round(y - self.origin[1], 4)])
                if len(self.trail) > TRAIL_MAX:
                    del self.trail[: len(self.trail) - TRAIL_MAX]
            if self.sim_time_s <= self.record_max_s:
                if self.frames % TRACE_EVERY == 0:
                    self._trace.write(json.dumps({
                        "backend_id": BACKEND_ID, "t": self.sim_time_s,
                        "xyz_mm": [round(float(v), 5) for v in th["position_mm"]],
                        "yaw_rad": round(float(th["yaw_rad"]), 6),
                        "contacts": [int(v > 0) for v in self.obs["contacts"]["found"]]},
                        sort_keys=True) + "\n")
                self._recorder.step(self.frames, {"neural": {}, "body": self.obs,
                                                  "motor": {"applied_cpg_drive": list(DEFAULT_DRIVE)}},
                                    self._body.segment_positions(), 0)
            else:
                self._recording_capped = True
        except Exception as error:  # noqa: BLE001 -- shown in the packet and manifest
            self._fail(error)

    # ------------------------------------------------------------------ views
    def identity(self, *, activation: int, daemon_run_id: str) -> dict[str, Any]:
        return {"backend": BACKEND_ID, "label": DISPLAY_LABEL, "run_id": self.run_id,
                "instance_id": f"{BACKEND_ID}:{self.run_id}", "assay": ASSAY,
                "graph_sha256": None, "synthetic": False, "is_connectome": False,
                "counts_as_connectome_evidence": False, "qualification_eligible": False,
                "activation": int(activation), "daemon_run_id": daemon_run_id}

    def status(self) -> dict[str, Any]:
        th = self.obs["thorax"] if getattr(self, "obs", None) else {}
        pos = th.get("position_mm") or [math.nan, math.nan, math.nan]
        dx, dy = float(pos[0]) - self.origin[0], float(pos[1]) - self.origin[1]
        t = self.sim_time_s
        return {
            "backend_id": BACKEND_ID, "display_label": DISPLAY_LABEL, "disclaimer": DISCLAIMER,
            "assay": ASSAY, "run_id": self.run_id, "store": self.store_path(),
            "namespace": NAMESPACE_DIRNAME, "sim_time_s": t, "frames": self.frames,
            "pos_mm": [round(dx, 4), round(dy, 4)],
            "yaw_rad": round(float(th.get("yaw_rad", 0.0)), 6),
            "forward_speed_mm_s": round(dx / t, 3) if t > 0 else None,
            "contacts": [int(v > 0) for v in (self.obs.get("contacts", {}).get("found", []) if getattr(self, "obs", None) else [])],
            "trail_mm": [list(p) for p in self.trail],
            "recording": {"path": "body.nfbody", "fps": RECORD_FPS, "max_s": self.record_max_s,
                          "capped": self._recording_capped},
            "assays": {"available": list(SUPPORTED_ASSAYS),
                       "unavailable": list(DASHBOARD_ASSAYS), "reason": UNAVAILABLE_REASON},
            "error": self.error, "is_connectome": False,
        }

    # ------------------------------------------------------------------ close
    def _release(self) -> None:
        recorder, self._recorder = self._recorder, None
        trace, self._trace = self._trace, None
        body, self._body = self._body, None
        recording = None
        try:
            if recorder is not None:
                recording = recorder.close() if self.error is None else recorder.abort()
        finally:
            try:
                if trace is not None:
                    trace.close()
            finally:
                if body is not None:
                    body.close()
        return recording

    def close(self) -> dict[str, Any]:
        """Finish the files (idempotent).  Returns the summary written to summary.json."""
        if self.closed:
            return self.summary
        self.closed = True
        try:
            recording = self._release()
        except Exception as error:  # noqa: BLE001
            self._fail(error)
            recording = None
        status = self.status()
        status.pop("trail_mm", None)
        self.summary = {"schema": SCHEMA, **_identity(ASSAY), "run_id": self.run_id,
                        "status": "failed" if self.error else "complete", "error": self.error,
                        "duration_s": self.sim_time_s, "records": self.frames,
                        "recorded_s": min(self.sim_time_s, self.record_max_s),
                        "net_displacement_xy_mm": status["pos_mm"],
                        "mean_forward_speed_mm_s": status["forward_speed_mm_s"],
                        "wall_time_s": round(time.time() - self.started_wall, 3),
                        "recording": recording}
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        if self.error is None:
            self.manifest["status"] = "complete"
            self._write_manifest()
        return self.summary


__all__ = ["ReferenceLive", "assay_availability", "BACKEND_ID", "DISPLAY_LABEL"]
