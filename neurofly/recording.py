"""Deterministic run recordings (``.nfrec``) for 1x browser replay and open data.

The simulation runs slower than real time at full accuracy, so a run is recorded
once and played back at 1x in the dashboard (``web/replay.js``).  Format and
guarantees: docs/RECORDING_FORMAT.md.

A recording is one gzip stream (``mtime=0``, no file name) of UTF-8 JSON lines:

* line 1, ``{"k": "header", ...}``: format version, provenance (graph hash, code
  version, seed, parameters) and the channel layout (region names and sizes,
  raster neuron list);
* ``{"k": "f", ...}``: one frame per recorded step, the dashboard telemetry packet
  without its wall-clock fields plus ``activity`` (mean rate per region) and
  ``spikes`` (sparse spike counts of the raster neurons);
* ``{"k": "e", ...}``: an input event (an applied command) at its step;
* last line, ``{"k": "end", ...}``: frame count and the SHA-256 of the frame lines.

Everything wall-clock or random-ID based (timestamps, run UUIDs, host) lives in a
sidecar ``<name>.nfrec.json``, so the same seed, parameters and inputs give a
byte-identical ``.nfrec``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

FORMAT = "neurofly-run-recording"
VERSION = 1
SUFFIX = ".nfrec"

# Telemetry keys that depend on the wall clock, the viewer or random IDs.  The
# replay player re-creates the ones the dashboard needs from the header.
_DROP_KEYS = ("type", "timestamp", "timing", "run_id", "identity", "path", "brain_id", "sim_speed", "activity")
# Deterministic subset of the modular brain summary (``history`` has timestamps).
_BRAIN_KEYS = ("teaching", "trials", "steps", "learning_enabled", "weight_mean", "weight_std",
               "weight_change_l2", "probe", "restored", "seed", "model", "n_kc", "synapses")
# Commands that change what the simulation computes; speed, pause and recording
# control only change when steps run, never their content.
NON_PHYSICAL_ACTIONS = frozenset({"set_speed", "set_paused", "save_checkpoint", "probe_brain",
                                  "record_start", "record_stop"})
RASTER_MODES = ("none", "io", "all")


def _json_safe(value):
    """Non-finite floats become null; numpy scalars and arrays become JSON types."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _line(obj: dict) -> bytes:
    return (json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _sha256_ints(values: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
class RegionMap:
    """A partition of graph neurons into named regions for per-region rates.

    ``index[i]`` is the region of neuron ``i`` or -1.  The real MaleCNS graph is
    grouped by the annotated ``superclass`` (neuprint ROI/neuropil tables are not
    part of the downloaded data); the synthetic test graph by its IO channels.
    """

    def __init__(self, grouping: str, names: List[str], index: np.ndarray, source: str):
        self.grouping = grouping
        self.names = list(names)
        self.index = np.asarray(index, dtype=np.int64)
        self.sizes = np.bincount(self.index[self.index >= 0], minlength=len(self.names)).astype(np.int64)
        self.source = source
        self._valid = self.index >= 0

    def describe(self) -> dict:
        return dict(grouping=self.grouping, names=self.names, sizes=self.sizes.tolist(),
                    source=self.source, membership_sha256=_sha256_ints(self.index),
                    units="Hz, mean per neuron over the graph step that ended at this frame")

    def rates(self, counts: np.ndarray, window_s: float) -> List[float]:
        counts = np.asarray(counts)
        if len(counts) != len(self.index) or window_s <= 0:
            return []
        total = np.bincount(self.index[self._valid], weights=counts[self._valid].astype(np.float64),
                            minlength=len(self.names))
        rates = total / (np.maximum(self.sizes, 1) * window_s)
        return [round(float(r), 3) for r in rates]

    @classmethod
    def from_labels(cls, grouping: str, labels, source: str) -> "RegionMap":
        labels = ["unknown" if (v is None or (isinstance(v, float) and math.isnan(v)) or v == "") else str(v)
                  for v in labels]
        names = sorted(set(labels))
        lookup = {name: i for i, name in enumerate(names)}
        return cls(grouping, names, np.array([lookup[v] for v in labels], dtype=np.int64), source)

    @classmethod
    def from_channels(cls, n: int, channels: Dict[str, Iterable[int]], source: str) -> "RegionMap":
        names = sorted(channels)
        index = np.full(n, -1, dtype=np.int64)
        for i, name in enumerate(names):
            for node in channels[name]:
                if 0 <= int(node) < n and index[int(node)] < 0:
                    index[int(node)] = i
        if np.any(index < 0):
            names.append("other")
            index[index < 0] = len(names) - 1
        return cls("io-channel", names, index, source)


def region_map_for(runner) -> Optional[RegionMap]:
    """Region partition for a graph runner, or None for the modular controller."""
    graph = getattr(runner, "shared_graph", None)
    if graph is None:
        return None
    identity = getattr(graph, "identity", None)
    path = getattr(identity, "neuron_map_path", None)
    if path and not getattr(identity, "synthetic", False):
        try:
            import pyarrow.feather as feather
            nodes = feather.read_table(path, columns=["node_index", "superclass"]).to_pandas()
            nodes = nodes.sort_values("node_index")
            if len(nodes) == graph.n:
                return RegionMap.from_labels("superclass", nodes.superclass.tolist(),
                                             f"neurons.feather superclass ({identity.neuron_map_sha256})")
        except Exception as exc:  # recorded as absent, never faked
            print(f"[recording] superclass regions unavailable: {exc}", flush=True)
    return RegionMap.from_channels(graph.n, getattr(graph, "io_map", {}) or {}, "graph io_map channels")


def raster_neurons(runner, mode: str) -> Tuple[List[int], List[str]]:
    """Neuron indices (and labels) whose spike counts are stored each frame."""
    graph = getattr(runner, "shared_graph", None)
    if graph is None or mode == "none":
        return [], []
    if mode == "all":
        return list(range(graph.n)), []
    groups: Dict[int, str] = {}
    controller = getattr(runner, "graph_controller", None)
    sources = [getattr(graph, "io_map", {}) or {}]
    if controller is not None:
        sources += [controller.dn_indices, controller.sensory_indices, {"epg": controller.epg_indices}]
    for channels in sources:
        for name in sorted(channels):
            for node in channels[name]:
                node = int(node)
                if 0 <= node < graph.n and node not in groups:
                    groups[node] = name
    order = sorted(groups)
    return order, [groups[i] for i in order]


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
class _Ordinals:
    """Maps random IDs (segment UUIDs) to their order of appearance: s0, s1..."""

    def __init__(self, prefix: str):
        self.prefix, self.seen = prefix, {}

    def __call__(self, value):
        if value is None:
            return None
        if value not in self.seen:
            self.seen[value] = f"{self.prefix}{len(self.seen)}"
        return self.seen[value]


def frame_from_telemetry(telemetry: Dict[str, Any], segments: _Ordinals) -> Dict[str, Any]:
    """The telemetry packet without wall-clock and random-ID fields."""
    frame = {k: v for k, v in telemetry.items() if k not in _DROP_KEYS}
    frame["segment_id"] = segments(telemetry.get("segment_id"))
    transition = telemetry.get("transition")
    if isinstance(transition, dict):
        transition = dict(transition)
        if "ended_segment" in transition:
            transition["ended_segment"] = segments(transition["ended_segment"])
        frame["transition"] = transition
    brain = telemetry.get("brain")
    if isinstance(brain, dict):
        frame["brain"] = {k: brain[k] for k in _BRAIN_KEYS if k in brain}
    return frame


def _brain_backend(instance) -> Optional[str]:
    """Where the graph brain ran ('cpu' or 'cuda'); None for modular runs."""
    return getattr(getattr(instance, "brain", None), "backend", None)


def _lif_dynamics(runner) -> Optional[str]:
    """LIF dynamics version of a graph run (``NEUROFLY_LIF_DYNAMICS``); None for modular."""
    if getattr(runner, "shared_graph", None) is None:
        return None
    from brainlab.graph_identity import active_dynamics_version
    return active_dynamics_version()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class RunRecorder:
    """Writes one ``.nfrec`` while a runner steps.  Call ``capture`` after each step."""

    def __init__(self, runner, path, *, record_every: int = 1, raster: str = "io",
                 label: str = "", inputs: Optional[List[dict]] = None):
        if raster not in RASTER_MODES:
            raise ValueError(f"raster must be one of {RASTER_MODES}")
        self.path = Path(path)
        if self.path.suffix != SUFFIX:
            self.path = self.path.with_name(self.path.name + SUFFIX)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record_every = max(1, int(record_every))
        self.raster_mode = raster
        self.frames = 0
        self.events = 0
        self.closed = False
        self._segments = _Ordinals("s")
        self._frame_digest = hashlib.sha256()
        self._regions = runner.region_map()
        self._raster, raster_labels = raster_neurons(runner, raster)
        self._raster_index = np.asarray(self._raster, dtype=np.int64)
        self.start_step = int(runner.total_steps)
        self.last_step = self.start_step
        self._partial = self.path.with_name(f".{self.path.name}.partial")
        self._raw = open(self._partial, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0, compresslevel=6)
        self.header = self._header(runner, label, raster_labels, inputs or [])
        self._gz.write(_line(self.header))
        self._sidecar = dict(created_at=time.time(), host=platform.node(), pid=os.getpid(),
                             daemon_run_id=getattr(runner, "run_id", None),
                             manifest_run_ids=[], label=label)
        self._note_manifest(runner)
        self.capture(runner)          # frame 0: the state the run starts from

    # -- header -------------------------------------------------------------
    def _header(self, runner, label: str, raster_labels: List[str], inputs: List[dict]) -> dict:
        manifest = getattr(runner, "manifest", None)
        graph = dict((manifest.graph or {}) if manifest is not None else {})
        for key in ("graph_path", "neuron_map_path", "root"):   # host paths, not identity
            graph.pop(key, None)
        source = dict(getattr(manifest, "source", None) or getattr(runner, "_source", {}) or {})
        source.pop("root", None)
        brain = getattr(runner, "active_brain", None)
        instance = getattr(getattr(runner, "registry", None), "active", None)
        initial_state = dict(brain_steps=int(getattr(brain, "steps", 0) or 0),
                             restored=bool(getattr(brain, "restored", False)))
        if instance is not None:
            initial_state["graph_step_index"] = int(getattr(instance, "step_index", 0) or 0)
        provenance = dict(
            backend=runner.backend, assay=runner.active_paradigm_id,
            lif_dynamics=_lif_dynamics(runner),
            brain_backend=_brain_backend(instance),
            seed=int(instance.seed) if instance is not None else int(getattr(brain, "seed", 0)),
            controller_version=getattr(manifest, "controller_version", ""),
            label=getattr(manifest, "label", ""), synthetic=bool(getattr(manifest, "synthetic", False)),
            test_mode=bool(getattr(runner, "test_mode", False)),
            graph=graph or None,
            dynamics=getattr(manifest, "dynamics", {}) if manifest is not None else {},
            code=source,
            software=dict(python=platform.python_version(), numpy=np.__version__),
            params=dict(dt_s=runner.dt, graph_step_ms=getattr(runner, "graph_step_ms", None),
                        trial_length_s=runner.trial_length_s, continuous=runner.continuous,
                        motor_assists=dict(getattr(runner.arena, "motor_assists", {}) or {})),
            initial_state=initial_state,
            inputs=list(inputs),
        )
        raster = None
        if self._raster:
            raster = dict(mode=self.raster_mode, neurons=[int(i) for i in self._raster] if self.raster_mode != "all" else None,
                          n=len(self._raster), labels=raster_labels or None,
                          neurons_sha256=_sha256_ints(self._raster),
                          encoding="flat [slot, count, slot, count, ...] of non-zero counts; slot indexes neurons")
        channels = dict(
            regions=self._regions.describe() if self._regions is not None else dict(
                grouping="modular-circuit", source="modular controller telemetry (neural.kc_hz mean, "
                "pam_trace, ppl1_trace, net_valence)", names=list(runner.MODULAR_ACTIVITY), sizes=None,
                units="model units (not spike rates)"),
            raster=raster,
            body=["fly", "body_position_mm", "body_quaternion_wxyz", "joint_angles_rad", "leg_contacts", "biomechanics"],
            stimulus=["stimuli", "sensory", "scene", "assay_state", "live_assay"],
            motor=["dn_rates", "descending", "motor", "motor_drives"],
        )
        return dict(k="header", format=FORMAT, version=VERSION, record_every=self.record_every,
                    start_step=self.start_step, sim_time_start_s=round(self.start_step * runner.dt, 6),
                    frame_dt_s=round(runner.dt * self.record_every, 6),
                    provenance=provenance, channels=channels, label=label)

    def _note_manifest(self, runner):
        manifest = getattr(runner, "manifest", None)
        run_id = getattr(manifest, "run_id", None)
        if run_id and run_id not in self._sidecar["manifest_run_ids"]:
            self._sidecar["manifest_run_ids"].append(run_id)

    # -- frames and events --------------------------------------------------
    @staticmethod
    def _activity(telemetry) -> Optional[List[float]]:
        return (telemetry.get("activity") or {}).get("rates")

    def _spikes(self, runner) -> Optional[List[int]]:
        if not self._raster:
            return None
        counts = getattr(getattr(runner, "graph_controller", None), "last_counts", None)
        if counts is None:
            return None
        sub = np.asarray(counts)[self._raster_index]
        nz = np.flatnonzero(sub)
        out = np.empty(2 * len(nz), dtype=np.int64)
        out[0::2], out[1::2] = nz, sub[nz]
        return out.tolist()

    def capture(self, runner, step_result: Optional[dict] = None) -> bool:
        """Record the state after the step that just ran (every ``record_every`` steps)."""
        if self.closed or runner.total_steps <= self.last_step and self.frames:
            return False
        if (runner.total_steps - self.start_step) % self.record_every:
            return False
        telemetry = runner._assemble_telemetry(step_result if step_result is not None else runner._last_step_result)
        frame = frame_from_telemetry(telemetry, self._segments)
        frame.update(k="f", i=self.frames, activity=self._activity(telemetry),
                     spikes=self._spikes(runner))
        data = _line(frame)
        self._gz.write(data)
        self._frame_digest.update(data)
        self.frames += 1
        self.last_step = runner.total_steps
        self._note_manifest(runner)
        return True

    def event(self, kind: str, step: int, **fields) -> None:
        if self.closed:
            return
        self._gz.write(_line(dict(k="e", kind=kind, step=int(step), **fields)))
        self.events += 1

    def command(self, runner, cmd: dict, result: dict) -> None:
        """Log an applied command as an input event, unless it could not change the run."""
        action = cmd.get("action", "")
        if action in NON_PHYSICAL_ACTIONS or (result or {}).get("status") != "ok":
            return   # refused commands changed nothing
        clean = {k: v for k, v in cmd.items() if k not in ("token",)}
        self.event("command", runner.total_steps, cmd=clean)

    def close(self) -> dict:
        """Finish the file (atomic rename) and write the sidecar; returns the summary."""
        if self.closed:
            return self.summary
        end = dict(k="end", frames=self.frames, events=self.events, first_step=self.start_step,
                   last_step=self.last_step, frames_sha256=self._frame_digest.hexdigest())
        self._gz.write(_line(end))
        self._gz.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        os.replace(self._partial, self.path)
        self.closed = True
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.summary = dict(path=str(self.path), name=self.path.name, bytes=self.path.stat().st_size,
                            sha256=digest, frames=self.frames, events=self.events,
                            first_step=self.start_step, last_step=self.last_step,
                            duration_s=round((self.last_step - self.start_step) * self.header["provenance"]["params"]["dt_s"], 6),
                            assay=self.header["provenance"]["assay"],
                            backend=self.header["provenance"]["backend"],
                            synthetic=self.header["provenance"]["synthetic"])
        sidecar = dict(self._sidecar, closed_at=time.time(), recording=self.summary)
        self.path.with_name(self.path.name + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
        return self.summary


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def read_recording(path) -> dict:
    """Parse a recording and check its frame digest.  Returns header, frames, events, end."""
    header, frames, events, end = None, [], [], None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for raw in fh:
            record = json.loads(raw)
            kind = record.get("k")
            if kind == "header":
                header = record
            elif kind == "f":
                digest.update(raw)
                frames.append(record)
            elif kind == "e":
                events.append(record)
            elif kind == "end":
                end = record
    if header is None or header.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    if header.get("version") != VERSION:
        raise ValueError(f"Unsupported recording version {header.get('version')!r}")
    if end is None:
        raise ValueError(f"{path} is incomplete (no end record)")
    if end["frames_sha256"] != digest.hexdigest() or end["frames"] != len(frames):
        raise ValueError(f"{path} frame digest mismatch")
    return dict(header=header, frames=frames, events=events, end=end)


def list_recordings(directory) -> List[dict]:
    """Finished recordings in ``directory`` (newest first) with their sidecar summaries."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for path in directory.glob(f"*{SUFFIX}"):
        sidecar = path.with_name(path.name + ".json")
        try:
            meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        summary = dict(meta.get("recording") or {}, name=path.name, bytes=path.stat().st_size)
        summary.pop("path", None)
        summary["created_at"] = meta.get("created_at", path.stat().st_mtime)
        out.append(summary)
    return sorted(out, key=lambda s: s["created_at"], reverse=True)


# ---------------------------------------------------------------------------
# Batch recording: run a paradigm headless (as fast as it computes) and record it
# ---------------------------------------------------------------------------
def record_run(*, paradigm: str, out, steps: int, backend: str = "modular", record_every: int = 1,
               raster: str = "io", schedule: Optional[List[dict]] = None, state_dir=None,
               test_synthetic_graph: bool = False, graph_dir=None, graph_step_ms: Optional[float] = None,
               trial_length_s: float = 60.0, continuous: bool = False, label: str = "",
               progress_every: int = 0, dynamics: Optional[str] = None) -> dict:
    """Run ``steps`` fixed-dt steps of a fresh runner and write one recording.

    ``state_dir`` defaults to a new temporary directory, so the run starts from the
    paradigm's naive brain; pass a directory to continue a saved brain (the header
    then records ``initial_state.restored``).  ``schedule`` is a list of
    ``{"step": n, "cmd": {...}}`` applied exactly at step ``n`` (the recorded inputs).
    ``dynamics`` sets the process-wide LIF dynamics (``NEUROFLY_LIF_DYNAMICS``) for
    graph backends, as the daemon's ``--dynamics`` does; None keeps the environment.
    """
    import tempfile
    if dynamics is not None:
        # Process-wide, before any Brain or registry manifest is created (as the daemon).
        os.environ["NEUROFLY_LIF_DYNAMICS"] = dynamics
    from neurofly_daemon import ContinuousExperimentRunner

    schedule = sorted((dict(step=int(e["step"]), cmd=dict(e["cmd"])) for e in (schedule or [])),
                      key=lambda e: e["step"])
    with tempfile.TemporaryDirectory(prefix="nfrec-state-") as tmp:
        runner = ContinuousExperimentRunner(
            initial_paradigm=paradigm, sim_speed=1.0, checkpoint_interval=1e9,
            output_dir=Path(state_dir) if state_dir else Path(tmp), trial_length_s=trial_length_s,
            continuous=continuous, backend=backend, graph_dir=Path(graph_dir) if graph_dir else None,
            test_synthetic_graph=test_synthetic_graph, graph_step_ms=graph_step_ms)
        for entry in schedule:
            runner.schedule_command(entry["step"], entry["cmd"])
        started = time.perf_counter()
        with runner.lock:
            runner.recorder = RunRecorder(runner, out, record_every=record_every, raster=raster,
                                          label=label, inputs=schedule)
            for n in range(int(steps)):
                runner._advance_one()
                if runner.last_error:
                    runner.recorder.event("simulation_error", runner.total_steps, message=runner.last_error)
                    break
                if progress_every and (n + 1) % progress_every == 0:
                    wall = time.perf_counter() - started
                    print(f"[record] step {n + 1}/{steps}  sim {runner.total_steps * runner.dt:.2f} s  "
                          f"{runner.total_steps * runner.dt / wall:.3f}x real time", flush=True)
            summary = runner.stop_recording()
        summary["wall_s"] = round(time.perf_counter() - started, 3)
        summary["brain_backend"] = _brain_backend(getattr(getattr(runner, "registry", None), "active", None))
        summary["error"] = runner.last_error
        return summary


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="neurofly record",
                                     description="Run a paradigm headless and write a deterministic .nfrec "
                                                 "recording for 1x replay in the dashboard.")
    parser.add_argument("--paradigm", required=True)
    parser.add_argument("--out", required=True, help="Output path (.nfrec is appended when missing)")
    length = parser.add_mutually_exclusive_group(required=True)
    length.add_argument("--steps", type=int)
    length.add_argument("--seconds", type=float, help="Simulated seconds (steps = seconds / dt)")
    parser.add_argument("--backend", default="modular")
    parser.add_argument("--test-synthetic-graph", action="store_true",
                        help="Use the labelled synthetic test graph (graph backends, tests only)")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--graph-step-ms", type=float, default=None)
    parser.add_argument("--dynamics", choices=("v1", "v2", "v3"),
                        default=os.environ.get("NEUROFLY_LIF_DYNAMICS") or "v3",
                        help="LIF dynamics of the connectome backends (default: v3, as the daemon; "
                             "env NEUROFLY_LIF_DYNAMICS)")
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--raster", choices=RASTER_MODES, default="io")
    parser.add_argument("--schedule", default=None,
                        help='JSON file: [{"step": n, "cmd": {"action": ...}}, ...] applied at exact steps')
    parser.add_argument("--state-dir", default=None,
                        help="Brain state directory to continue (default: fresh, naive brain)")
    parser.add_argument("--trial-seconds", type=float, default=60.0)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    steps = args.steps if args.steps is not None else int(round(args.seconds / 0.02))
    schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else None
    summary = record_run(paradigm=args.paradigm, out=args.out, steps=steps, backend=args.backend,
                         record_every=args.record_every, raster=args.raster, schedule=schedule,
                         state_dir=args.state_dir, test_synthetic_graph=args.test_synthetic_graph,
                         graph_dir=args.graph_dir, graph_step_ms=args.graph_step_ms,
                         trial_length_s=args.trial_seconds, continuous=args.continuous, label=args.label,
                         progress_every=max(1, steps // 10), dynamics=args.dynamics)
    print(json.dumps(summary, indent=2))
    return 0 if not summary.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
