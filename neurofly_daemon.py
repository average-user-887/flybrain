#!/usr/bin/env python3
"""
Project NeuroFly (v1.0-release) — Continuous Background Learning Daemon
========================================================================
Runs 24/7 headless biological simulation and continuous online learning
on a workstation, a server, or a laptop. The modular backend needs only the Python
standard library and NumPy; the connectome backends also need numba, pandas and
pyarrow, and CuPy or numba.cuda for the GPU brain.

Key Capabilities:
1. 24/7 Continuous Headless Simulation: Steps active neuroethological paradigms
   independently of any connected browser interface.
2. Continuous Plasticity: Mushroom Body synaptic weights (W_KC->MBON) and Central
   Complex orientation landmarks evolve continuously over thousands of trials.
3. Multi-Threaded Dual HTTP/SSE Streaming API (Port 8769):
   - GET  /api/status     : Health, uptime, steps, speed, active paradigm, and metrics.
   - GET  /api/telemetry  : Instant point-in-time state packet (REST polling).
   - GET  /api/stream     : Real-time Server-Sent Events (SSE) stream (30 Hz).
   - GET  /api/paradigms  : Catalog of 14 standard experimental paradigms.
   - POST /api/command    : Bidirectional interventions (stimuli, speed, parameters, switches).
4. Auto-Checkpointing: Periodically writes weight matrices and trial summaries to disk.
5. Light dependencies: Python 3.12 + NumPy for the modular backend (see item above
   for the connectome backends).
6. Public mode (--public / NEUROFLY_PUBLIC=1): read-only stream for untrusted
   viewers -- POST /api/command needs a bearer token equal to NEUROFLY_ADMIN_TOKEN,
   SSE clients are capped and the stream is throttled (stream_gateway.py).
7. Durable learning records: append-only trials.jsonl / telemetry_summary.jsonl
   under NEUROFLY_DATA_DIR or outputs/learning (learning_recorder.py).
8. Timing: fixed dt = 0.02 s at every requested speed; a deadline scheduler runs
   steps in short batches and reports requested vs achieved speed (``timing`` in
   telemetry and /api/status).  Telemetry is published as immutable, pre-serialized
   snapshots, so delivery never holds the simulation lock; commands are applied at a
   step boundary and acknowledged with ``ack.applied_step``.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import assay_controls
from brainlab.runtime_backend import (COMPUTE_BACKENDS, UnsupportedRuntimeCapability,
                                     compute_selector, preflight_fixed_store, require_fixed_amd)

# Resolve project root dynamically without hardcoded machine paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from arena import Arena
    from maze import ExperimentRegistry
except ImportError as e:
    print(f"[Daemon] Error importing arena/maze modules: {e}", file=sys.stderr)
    raise

# Flag-gated public-stream policy and durable learning records.  Both default
# to the historical behaviour (private LAN mode; see docs/PUBLIC_STREAMING.md
# and docs/DATA_SCHEMA.md).  They never touch the simulation loop.
from stream_gateway import MAX_COMMAND_BYTES, StreamGateway, StreamPolicy
from neurofly import __version__ as NEUROFLY_VERSION
from learning_recorder import LearningRecorder, RecorderThread, resolve_data_dir
from observation_publication import (ObservationPublicationError,
                                     ObservationPublicationQueue,
                                     ObservationQueueFull)
from observation_envelopes import build_observation_envelope, dumps_observation_envelope
from online_metrics import (ConfigError, OpenArenaObserver, build_observation_config,
                            s_to_us, us_to_s)
from experiment_brains import ExperimentBrains, PARADIGMS
# Controller identity (WP4).  experiment_registry / brainlab are imported only when
# a graph backend is selected, so the modular default never touches the graph.
from provenance import (GRAPH_BACKENDS, RunManifest, get_backend, graph_io_declaration,
                        resolve_keep_checkpoints, source_revision, trial_clock,
                        unknown_trial_clock, validate_trial_clock)
from neurofly.privacy import redact_local

DAEMON_BACKENDS = ("modular",) + tuple(GRAPH_BACKENDS)
# Packet ``step``/``sim_time_s`` count from zero in every daemon process; they are
# not the retained graph's or the trial's clock (see clock_status).
SESSION_CLOCK_SCOPE = "daemon_session"
# LIF dynamics the daemon supports.  brainlab declares more (research engines, used
# through Brain(..., dynamics=...)); the daemon's registry, manifests and checkpoints
# are defined for these only, so anything else is refused before any file is touched.
DAEMON_DYNAMICS = ("v1", "v2", "v3")
DYNAMICS_ENV = "NEUROFLY_LIF_DYNAMICS"
WEB_BUILD_PLACEHOLDER = "__NEUROFLY_WEB_BUILD__"
DASHBOARD_WEB_ASSETS = (
    "web/index.html",
    "web/training.css",
    "web/vendor/three.min.js",
    "web/vendor/OrbitControls.js",
    "web/live_assays.js",
    "web/metric_records.js",
    "web/observation_display.js",
    "web/observation_renderer.js",
    "web/assay_cues_3d.js",
    "web/app.js",
    "web/replay.js",
    "web/training.js",
)
SPEED_MIN = 0.1
SPEED_MAX = 100.0


def validate_trial_seconds(value: Optional[float]) -> Optional[float]:
    """Validate the optional observation-window override before any filesystem mutation."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("--trial-seconds must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ConfigError("--trial-seconds must be a positive finite number")
    try:
        canonical_us = s_to_us(value)
        canonical_s = us_to_s(canonical_us)
    except (OverflowError, ValueError) as exc:
        raise ConfigError("--trial-seconds is outside the canonical microsecond range") from exc
    if canonical_us <= 0 or not math.isfinite(canonical_s):
        raise ConfigError("--trial-seconds must be at least one canonical microsecond")
    return value


def preflight_observation_policy(paradigm: str, trial_seconds: Optional[float],
                                 continuous: bool) -> Optional[float]:
    """Exercise the actual spec/config builder before a runner creates persistent state."""
    requested = validate_trial_seconds(trial_seconds)
    if not isinstance(continuous, bool):
        raise ConfigError("continuous observation policy must be a bool")
    producer = (OpenArenaObserver() if paradigm == "open-arena"
                else ExperimentRegistry.get(paradigm))
    build_observation_config(
        producer.observation_spec(), config_id="startup-preflight",
        override_window_s=requested, continuous=continuous,
        override_source=("cli:--continuous" if continuous else
                         "cli:--trial-seconds" if requested is not None else None),
        set_at_sim_s=0.0, manifest_run_id="startup-preflight",
    )
    return requested


def dashboard_web_build(root: Path = PROJECT_ROOT) -> str:
    """Content identity for the HTML and application code delivered together."""
    digest = hashlib.sha256()
    for relative in DASHBOARD_WEB_ASSETS:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


DASHBOARD_WEB_BUILD = dashboard_web_build()
DAEMON_SOURCE_REVISION = source_revision()


def delivery_identity() -> Dict[str, Any]:
    """Small source/asset handshake for status and the served dashboard page."""
    commit = DAEMON_SOURCE_REVISION.get("commit")
    return {
        "revision": commit[:12] if commit else "unavailable",
        "source_dirty": DAEMON_SOURCE_REVISION.get("dirty"),
        "web_build": DASHBOARD_WEB_BUILD,
    }


def _containment_geometry(region: Any) -> Dict[str, Any]:
    """JSON description of the exact Arena containment primitive."""
    name = type(region).__name__
    if name == "RectRegion":
        return {"kind": "rectangle", "bounds": [region.xmin, region.ymin, region.xmax, region.ymax]}
    if name == "CircleRegion":
        return {"kind": "circle", "center": [region.cx, region.cy], "radius": region.radius}
    if name == "UnionRegion":
        return {"kind": "union", "members": [_containment_geometry(member) for member in region.members]}
    if name == "HoledRegion":
        return {"kind": "holed", "outer": _containment_geometry(region.outer),
                "holes": [_containment_geometry(hole) for hole in region.holes]}
    raise TypeError(f"unsupported containment geometry {name}")


def _floor_surfaces(region: Any) -> List[Dict[str, Any]]:
    """Renderable floor primitives from containment, without polygon approximation."""
    shape = _containment_geometry(region)
    if shape["kind"] == "rectangle":
        return [{"shape": "rectangle", "bounds": shape["bounds"], "role": "containment"}]
    if shape["kind"] == "circle":
        return [{"shape": "circle", "center": shape["center"], "radius": shape["radius"],
                 "role": "containment"}]
    if shape["kind"] == "union":
        surfaces = []
        for member in region.members:
            surfaces.extend(_floor_surfaces(member))
        return surfaces
    if shape["kind"] == "holed":
        # Obstacles are separately present in the authoritative wall collection.
        return _floor_surfaces(region.outer)
    return []


def assay_geometry_telemetry(arena: Arena) -> Dict[str, Any]:
    """Physical assay geometry used by the Python arena, in arena millimetres."""
    paradigm = getattr(arena, "paradigm", None)
    walls = []
    for wall in getattr(paradigm, "walls", ()) if paradigm is not None else ():
        walls.append({"p1": [float(wall.p1[0]), float(wall.p1[1])],
                      "p2": [float(wall.p2[0]), float(wall.p2[1])]})

    surfaces = _floor_surfaces(arena.containment)
    zones = getattr(paradigm, "zones", ()) if paradigm is not None else ()
    key = Arena.paradigm_key(paradigm)
    if key == "gap_crossing":
        surfaces = [
            {"shape": "rectangle", "bounds": [float(v) for v in zone.bounds], "role": zone.name}
            for zone in zones if zone.name in ("start_track", "landing_track")
        ]
    elif key == "circadian_dam":
        surfaces = [
            {"shape": "rectangle", "bounds": [float(v) for v in zone.bounds], "role": zone.name}
            for zone in zones if str(zone.name).startswith("tube_")
        ]

    return {
        "schema": "neurofly.assay-geometry.v1",
        "coordinate_frame": "arena-mm",
        "source": "python-arena",
        "bounds": [float(v) for v in arena.world_bounds],
        "containment": _containment_geometry(arena.containment),
        "walls": walls,
        "surfaces": surfaces,
    }


def assay_cue_telemetry(arena: Arena) -> Dict[str, Any]:
    """Copy existing runtime cue coordinates; never reconstruct stimulus phase/fields."""
    paradigm = getattr(arena, "paradigm", None)
    def point(value):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value[:2]):
            return None
        return [float(value[0]), float(value[1])]
    result = {}
    for name in ('goal_pos', 'food_pos', 'repellent_pos', 'pheromone_pos', 'hotspot_pos', 'cool_pos'):
        value = point(getattr(paradigm, name, None))
        if value is not None:
            result[name] = value
    pillars = getattr(paradigm, 'pillar_centers', None)
    if isinstance(pillars, (list, tuple)) and all(point(v) is not None for v in pillars):
        result['pillar_centers'] = [point(v) for v in pillars]
    if Arena.paradigm_key(paradigm) == 'y_maze':
        tips = [{'name': zone.name, 'position': point(zone.bounds)}
                for zone in getattr(paradigm, 'zones', ()) if zone.zone_type == 'arm']
        if tips and all(tip['position'] is not None for tip in tips):
            result['arm_tips'] = tips
    return result


def dashboard_index_bytes(index_path: Path) -> bytes:
    """Stamp the served document with the exact asset build reported by status."""
    text = index_path.read_text(encoding="utf-8")
    if text.count(WEB_BUILD_PLACEHOLDER) != 1:
        raise RuntimeError("dashboard index must contain exactly one web-build placeholder")
    return text.replace(WEB_BUILD_PLACEHOLDER, DASHBOARD_WEB_BUILD).encode("utf-8")


class UnsupportedDynamics(ValueError):
    """The requested LIF dynamics version is not one the daemon supports."""


def daemon_dynamics(version: Optional[str] = None) -> str:
    """The daemon's LIF dynamics: ``version``, else NEUROFLY_LIF_DYNAMICS, else v3.

    Raises UnsupportedDynamics for anything outside DAEMON_DYNAMICS -- including a
    version brainlab declares for research (v4, v5) and any unknown string -- so an
    environment default can never bypass the daemon's explicit choices.
    """
    chosen = version or os.environ.get(DYNAMICS_ENV) or "v3"
    if chosen not in DAEMON_DYNAMICS:
        source = f"--dynamics or {DYNAMICS_ENV}" if version else f"{DYNAMICS_ENV} (environment)"
        raise UnsupportedDynamics(
            f"LIF dynamics {chosen!r} from {source} is not supported by the daemon; choose one of "
            f"{', '.join(DAEMON_DYNAMICS)}. Research engines are used through Brain(..., dynamics=...), "
            f"not through the daemon. Nothing was started or written.")
    return chosen


# The inhibitory reversal the daemon runs: the declared v2/v3 baseline only.  The
# E_inh sensitivity variants (docs/EINH_SENSITIVITY.md) are research configurations,
# used through Brain(..., e_inh_mV=...) or NEUROFLY_LIF_E_INH_MV in research scripts;
# the daemon's registry and checkpoints would advertise them as baseline v3.
E_INH_ENV = "NEUROFLY_LIF_E_INH_MV"
DAEMON_E_INH_MV = -70.0


class UnsupportedSensitivityOverride(UnsupportedDynamics):
    """NEUROFLY_LIF_E_INH_MV selects a sensitivity variant the daemon does not run."""


def daemon_e_inh() -> None:
    """Refuse an inhibitory-reversal override other than the declared baseline.

    Unset (or empty) and the explicit baseline (-70 mV) are accepted; anything else
    -- another value, or text that is not a number -- is refused before any file is
    touched, because the daemon would otherwise restore and write baseline-v3
    checkpoints from a different model.
    """
    raw = os.environ.get(E_INH_ENV)
    if raw is None or raw == "":          # exactly as brainlab reads "unset"; whitespace is refused below
        return
    try:
        value = float(raw)
    except ValueError:
        value = None
    if value is None or not math.isfinite(value) or value != DAEMON_E_INH_MV:
        raise UnsupportedSensitivityOverride(
            f"{E_INH_ENV}={raw!r} selects an E_inh sensitivity variant, which the daemon does not run "
            f"(only the declared baseline {DAEMON_E_INH_MV:g} mV, or unset). Run sensitivity variants "
            f"through the research scripts (docs/EINH_SENSITIVITY.md). Nothing was started or written.")


def daemon_configuration(version: Optional[str] = None) -> str:
    """Every process-wide model override the daemon accepts, checked together."""
    chosen = daemon_dynamics(version)
    daemon_e_inh()
    return chosen


def identity_rejection(packet_identity: Optional[Dict[str, Any]], packet_daemon_run: Optional[str],
                       ack: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a telemetry packet must be rejected after a switch acknowledgement, or None.

    Mirrors ``identityRejection`` in web/app.js.  ``ack`` is the last switch ack
    (``ack.identity`` names the activated run/instance and its ``activation``
    counter).  A packet from the same daemon process with an older activation was
    produced before the acknowledged switch (stale); one with the same activation
    must carry exactly the acknowledged run_id and instance_id.  Newer activations
    (another dashboard switched later) and other daemon processes are accepted.
    """
    expected = (ack or {}).get("identity")
    if not expected or not packet_identity:
        return None
    if packet_daemon_run != expected.get("daemon_run_id"):
        return None
    got, want = packet_identity.get("activation"), expected.get("activation")
    if not isinstance(got, int) or not isinstance(want, int):
        return "packet identity has no activation counter"
    if got < want:
        return f"stale: activation {got} precedes acknowledged switch {want}"
    if got == want and (packet_identity.get("run_id") != expected.get("run_id")
                        or packet_identity.get("instance_id") != expected.get("instance_id")):
        return "run_id/instance_id differ from the acknowledged switch"
    return None


class GraphArenaController:
    """Arena motor controller for graph backends (connectome-fixed/-plastic/-readout).

    Each arena step advances the active registry instance by ``step_ms`` of
    simulated brain time.

    The optomotor assay uses the verified WP5 sensory encoder and motor decoder
    (``brainlab/io_map.py``, docs/WP5_OPTOMOTOR.md). Other assays use the explicitly
    declared probes and DN decoder in ``provenance.py``. No tonic graph drive,
    forward floor or surrogate controller is added to their graph output.
    """

    UNMAPPED = "graph-unmapped-io"
    OPTOMOTOR_ASSAY = "optomotor"
    DN_CHANNELS = ("dna02_l", "dna02_r", "dnp09", "dnb01", "mdn", "gf")

    @classmethod
    def unavailable_dn(cls, reason):
        return ({name: None for name in cls.DN_CHANNELS},
                {name: reason for name in cls.DN_CHANNELS})

    def __init__(self, runner: "ContinuousExperimentRunner", step_ms: float, *, shared_graph=None):
        self.runner = runner
        self._prepared_graph = shared_graph
        self.step_ms = float(step_ms)
        self._currents = None
        self.last_total_spikes = 0
        self.last_counts = None              # per-neuron spike counts of the last graph step
        # WP5 optomotor loop, resolved once: an OptomotorIOMap, or False when this
        # graph has none (the reason is kept in ``optomotor_unavailable``).
        self._optomotor_io = None
        # (GraphInstance, OptomotorLoop).  Keyed by the instance OBJECT, not its id:
        # re-activating an assay builds a new GraphInstance with the same id, and a
        # loop bound to the released one can never step (see _loop_for).
        self._optomotor_loop = None
        self.optomotor_unavailable = None
        self.dn_indices = {}
        self.sensory_indices = {}
        self.epg_indices = []
        self._load_indices()

    def _load_indices(self):
        """Resolve DN, sensory and EPG node indices against the graph actually loaded.

        Indices are only meaningful for the neuron table the running graph was
        verified against, so they derive from ``shared_graph`` itself: DN channels
        from its ``io_map``, cell-type channels from the ``neurons.feather`` named
        in its identity.  A synthetic graph has no cell-type annotations, so its
        sensory channels stay empty (``sensory_unavailable`` says why) even when
        real annotation files exist on disk.  Every resolved index is checked
        against the graph's neuron count; a table that does not fit is refused.
        """
        from brainlab.graph_identity import DN_CHANNELS
        self.sensory_indices = {
            "orn_food": [], "orn_danger": [], "visual_l": [], "visual_r": [],
            "visual_looming": [], "jon_wind": [], "feco_proprio": [],
            "courtship_cva": [], "thermo_receptors": []
        }
        self.epg_indices = []
        self.sensory_unavailable = None
        graph = self._prepared_graph or getattr(self.runner, "shared_graph", None)
        identity = getattr(graph, "identity", None)
        io_map = getattr(graph, "io_map", None) if graph is not None else None
        self.dn_indices = {name: list(indices) for name, indices in (io_map or DN_CHANNELS).items()}
        if identity is None or getattr(identity, "synthetic", False) or not getattr(identity, "neuron_map_path", None):
            self.sensory_unavailable = ("synthetic test graph: no cell-type annotations"
                                        if getattr(identity, "synthetic", False)
                                        else "loaded graph names no neuron map")
            return
        n = int(graph.n)
        try:
            neurons_path = Path(identity.neuron_map_path)
            cdir = neurons_path.parent.parent
            if neurons_path.is_file():
                import pyarrow.feather as feather
                df = feather.read_table(neurons_path).to_pandas()
                if len(df) != n:
                    raise ValueError(f"{neurons_path} has {len(df)} rows, the loaded graph {n} neurons")
                if "cell_type" in df.columns:
                    self.sensory_indices["orn_food"] = df[df['cell_type'] == 'ORN_DM1']['node_index'].tolist()[:50]
                    self.sensory_indices["orn_danger"] = df[df['cell_type'] == 'ORN_DA2']['node_index'].tolist()[:50]
                    self.sensory_indices["courtship_cva"] = df[df['cell_type'] == 'ORN_DA1']['node_index'].tolist()[:50]
                    self.sensory_indices["jon_wind"] = df[df['cell_type'].str.contains('JO-', na=False)]['node_index'].tolist()[:50]
                    self.sensory_indices["visual_looming"] = df[df['cell_type'].isin(['LC4', 'LPLC2'])]['node_index'].tolist()
                    self.sensory_indices["er_ring"] = df[df['cell_type'].isin(['ER4d', 'ER2_a', 'ER2_b', 'ER2_c', 'ER2_d'])]['node_index'].tolist()
                    self.sensory_indices["el_modulator"] = df[df['cell_type'] == 'EL']['node_index'].tolist()
                    self.epg_indices = df[df['cell_type'].isin(['EPG', 'EPGt'])]['node_index'].tolist()
                ann_path = cdir / "annotations.feather"
                if ann_path.is_file():
                    ann = feather.read_table(ann_path, columns=['bodyId', 'rootSide']).to_pandas()
                    root_side = dict(zip(ann.bodyId.astype('int64'), ann.rootSide))
                    retina = df[df['cell_type'] == 'R1-R6'].sort_values('source_id')
                    sides = retina['source_id'].map(root_side)
                    self.sensory_indices["visual_l"] = retina[sides == 'L']['node_index'].tolist()[:100]
                    self.sensory_indices["visual_r"] = retina[sides == 'R']['node_index'].tolist()[:100]
                    ann_thermo = feather.read_table(ann_path, columns=['bodyId', 'class']).to_pandas()
                    thermo_ids = set(ann_thermo[ann_thermo['class'] == 'thermosensory']['bodyId'].astype('int64'))
                    self.sensory_indices["thermo_receptors"] = df[df['source_id'].isin(thermo_ids)]['node_index'].tolist()
            else:
                self.sensory_unavailable = f"{neurons_path} missing"
            channels = dict(self.sensory_indices, epg=self.epg_indices)
            bad = {name: [i for i in idx if not 0 <= int(i) < n] for name, idx in channels.items()}
            bad = {name: idx[:3] for name, idx in bad.items() if idx}
            if bad:
                raise ValueError(f"indices outside the loaded graph's {n} neurons: {bad}")
        except Exception as exc:
            for name in self.sensory_indices:
                self.sensory_indices[name] = []
            self.epg_indices = []
            self.sensory_unavailable = str(exc)
            print(f"[GraphArenaController] Note: sensory indices unavailable ({exc})", flush=True)

    @property
    def optomotor_io_map_sha256(self):
        return getattr(self._optomotor_io, "sha256", None) if self._optomotor_io else None

    # This controller exposes every enabled probe and decoder in GRAPH_IO. Tonic
    # DN drive, central-complex navigation constants and a forward floor are not
    # part of that declared method.
    ENGINEERED_ASSISTANCE_ENABLED = False
    GRAPH_IO = graph_io_declaration(include_config=True)
    DECODER_DECLARATION = GRAPH_IO["config"]["decoders"]
    INPUT_CHANNELS = {row["name"]: row for row in GRAPH_IO["config"]["input_probes"]}

    def _loop_for(self, instance):
        """The WP5 optomotor loop for ``instance``, or None with a recorded reason.

        The loop is bound to one GraphInstance object.  Leaving an assay checkpoints
        and releases its instance; returning activates a NEW object with the same
        ``instance_id`` (state restored from the checkpoint).  The cache therefore
        matches on identity: a loop held for the released object is dropped and a
        fresh one is built for the live object, exactly as after a daemon restart
        (the encoder is re-seeded from ``instance.seed``).  Matching on the id alone
        stepped the released instance and froze the 2026-10-04 live observatory.
        """
        held = self._optomotor_loop
        if held is not None and held[0] is instance:
            return held[1]
        self._optomotor_loop = None
        if self._optomotor_io is False:
            return None
        try:
            from brainlab.graph_identity import GraphUnavailable
            from brainlab.io_map import DNa02YawDecoder, OptomotorEncoder, OptomotorLoop, resolve_optomotor_io
            if self._optomotor_io is None:
                if getattr(getattr(self.runner.shared_graph, "identity", None), "synthetic", False):
                    raise GraphUnavailable(
                        "synthetic test graph: the WP5 optomotor map is never faked on one")
                self._optomotor_io = resolve_optomotor_io()
            io_map = self._optomotor_io
            loop = OptomotorLoop(instance, io_map,
                                 OptomotorEncoder(io_map, np.random.default_rng(instance.seed)),
                                 DNa02YawDecoder(io_map), step_ms=self.step_ms)
        except Exception as error:                      # unresolved map: unsupported, not zero
            self._optomotor_io = False
            self.optomotor_unavailable = f"{type(error).__name__}: {error}"
            print(f"[GraphArenaController] optomotor loop unavailable: {self.optomotor_unavailable}",
                  flush=True)
            return None
        self._optomotor_loop = (instance, loop)
        return loop

    def __call__(self, fly=None, sensory=None, dt=0.02, **kwargs):
        self.last_counts = None
        registry = self.runner.registry
        instance = registry.active if registry is not None else None
        if instance is None or instance.assay != self.runner.active_paradigm_id:
            dn_rates, dn_unavailable = self.unavailable_dn("no active graph instance for this assay")
            return {"halted": True, "forward_speed": 0.0, "yaw_rate": 0.0, "motor_source": "halted-no-instance",
                    "controller_fault": "no active graph instance for this assay", "state": "HALTED",
                    "dn_rates": dn_rates, "dn_unavailable": dn_unavailable,
                    "epg_available": False, "epg_resolved_count": 0,
                    "epg_unavailable": "no active graph instance for this assay"}
        if self.runner.active_paradigm_id == self.OPTOMOTOR_ASSAY:
            loop = self._loop_for(instance)
            if loop is not None:
                record = loop.step(float(kwargs.get("optomotor_slip_rad_s", 0.0)),
                                   float(kwargs.get("optomotor_contrast", 1.0)))
                self.last_total_spikes = int(record["total_spikes"])
                self.last_counts = getattr(loop, "last_counts", None)
                epg_wedges = [0.0] * 16
                bump_phase = 0.0
                wp6_mean_delta = 0.0
                wp6_max_delta = 0.0
                if instance.backend == 'connectome-plastic' and len(getattr(instance, "plastic_delta", [])) > 0:
                    wp6_mean_delta = float(np.mean(instance.plastic_delta))
                    wp6_max_delta = float(np.max(np.abs(instance.plastic_delta)))

                return {"halted": False, "forward_speed": 0.0, "yaw_rate": float(record["yaw_rad_s"]),
                        "motor_source": "graph", "controller_fault": None, "state": "OPTOMOTOR-TETHERED",
                        "graph_step": instance.step_index, "graph_step_ms": self.step_ms,
                        "total_spikes": self.last_total_spikes,
                        "dn_rates": {
                            "dna02_l": round(float(record.get("rate_l", 0.0)), 2),
                            "dna02_r": round(float(record.get("rate_r", 0.0)), 2),
                            "dnp09": None, "mdn": None, "gf": None,
                        },
                        "dn_unavailable": {
                            name: "the WP5 optomotor map resolves only DNa02 left/right"
                            for name in ("dnp09", "mdn", "gf")
                        },
                        "epg_wedges": epg_wedges,
                        "epg_bump_phase": bump_phase,
                        "epg_available": False,
                        "epg_resolved_count": 0,
                        "epg_unavailable": "the WP5 optomotor loop does not resolve an EPG population",
                        "wp6": {
                            "mean_delta": round(wp6_mean_delta, 6),
                            "max_delta": round(wp6_max_delta, 6),
                            "n_edges": len(getattr(instance, "plastic_edges", [])),
                        },
                        "engineered_assistance_enabled": self.ENGINEERED_ASSISTANCE_ENABLED,
                        "engineered_assistance_applied": [],
                        "graph_io": graph_io_declaration(),
                        "decoder": dict(self.DECODER_DECLARATION),
                        "raw_motor_command": {"forward_speed_mm_s": 0.0,
                                              "yaw_rate_rad_s": float(record["yaw_rad_s"]),
                                              "state": "OPTOMOTOR-TETHERED"},
                        "input_stage": [{
                            **dict(self.INPUT_CHANNELS["optomotor"]),
                            "current_injected": None,
                            "n_cells": sum(len(v) for k, v in self._optomotor_io.populations.items()
                                           if k.startswith(("ftb_", "btf_"))),
                            "n_spiking_this_step": None,
                            "detail": "WP5 loop does not expose per-cell injected current or T4/T5 spike counts",
                        }],
                        "optomotor": {
                            "slip_rad_s": record["slip_rad_s"], "contrast": record["contrast"],
                            "yaw_rad_s": record["yaw_rad_s"], "contributions": record["contributions"],
                            "rate_l": record["rate_l"], "rate_r": record["rate_r"],
                            "spikes_l": record["spikes_l"], "spikes_r": record["spikes_r"],
                            "io_map_sha256": self.optomotor_io_map_sha256,
                            "decoder": "yaw = 0.02*(rate DNa02_L - rate DNa02_R), + = counter-clockwise; unclipped",
                        }}
            unsupported = (f"The WP5 optomotor map could not be resolved on this graph "
                           f"({self.optomotor_unavailable}); no motor command.")
            dn_rates, dn_unavailable = self.unavailable_dn(unsupported)
            return {"halted": True, "forward_speed": 0.0, "yaw_rate": 0.0, "motor_source": self.UNMAPPED,
                    "controller_fault": None, "state": "NO-MOTOR-MAP",
                    "graph_step": instance.step_index, "graph_step_ms": self.step_ms,
                    "total_spikes": self.last_total_spikes,
                    "dn_rates": dn_rates, "dn_unavailable": dn_unavailable,
                    "epg_available": False, "epg_resolved_count": 0,
                    "epg_unavailable": unsupported,
                    "engineered_assistance_enabled": self.ENGINEERED_ASSISTANCE_ENABLED,
                    "engineered_assistance_applied": [],
                    "optomotor": None, "optomotor_unsupported": unsupported,
                    "unsupported": unsupported}

        n = instance.brain.n
        if self._currents is None or len(self._currents) != n:
            self._currents = np.zeros(n, dtype=np.float32)
        else:
            self._currents.fill(0.0)
        currents = self._currents
        probe_current = {name: 0.0 for name in self.INPUT_CHANNELS if name != "optomotor"}

        if sensory is None:
            sensory = {}
        if "assay_stimuli" in kwargs and isinstance(kwargs["assay_stimuli"], dict):
            s_merged = dict(kwargs["assay_stimuli"])
            s_merged.update(sensory)
            sensory = s_merged

        # 0. Olfactory (food & danger)
        food_stim = float(sensory.get("mean_a", sensory.get("odor_conc", sensory.get("odor_a", 0.0))))
        if food_stim > 0.001:
            i_food = float(min(40.0, food_stim * 35.0))
            probe_current["orn_food"] = i_food
            for idx in self.sensory_indices.get("orn_food", ()):
                if idx < n: currents[idx] += i_food

        danger_stim = float(sensory.get("mean_b", sensory.get("odor_b", 0.0)))
        if danger_stim > 0.001:
            i_danger = float(min(45.0, danger_stim * 40.0))
            probe_current["orn_danger"] = i_danger
            for idx in self.sensory_indices.get("orn_danger", ()):
                if idx < n: currents[idx] += i_danger

        # 1. Visual lateral (photoreceptors) & Ring neurons (ER)
        contrast = float(kwargs.get("visual_contrast", kwargs.get("optomotor_contrast", sensory.get("stripe_contrast", 1.0))))
        retina_l = sensory.get("retina_photoreceptors_l")
        retina_r = sensory.get("retina_photoreceptors_r")
        if retina_l is not None and len(retina_l) > 0:
            mean_l = float(np.mean(retina_l))
            probe_current["photoreceptor_l"] = mean_l * 20.0 * contrast
            for idx in self.sensory_indices.get("visual_l", ()):
                if idx < n: currents[idx] += mean_l * 20.0 * contrast
        if retina_r is not None and len(retina_r) > 0:
            mean_r = float(np.mean(retina_r))
            probe_current["photoreceptor_r"] = mean_r * 20.0 * contrast
            for idx in self.sensory_indices.get("visual_r", ()):
                if idx < n: currents[idx] += mean_r * 20.0 * contrast

        # 2. Visual looming (LC4 / LPLC2)
        raw_loom = sensory.get("looming_theta", sensory.get("theta_rad", kwargs.get("stimulus_theta", 0.0)))
        if sensory.get("theta_deg") is not None and raw_loom == 0.0:
            raw_loom = math.radians(float(sensory["theta_deg"]))
        looming_theta = float(raw_loom)
        # A paradigm's own GF flag ("gf_spike") is an output, never a retinal input.
        looming_detected = bool(sensory.get("looming_detected", False)) or (looming_theta > 0.15)
        i_loom = 0.0
        if looming_detected:
            i_loom = float(min(55.0, looming_theta * 40.0 + 15.0))
            probe_current["looming"] = i_loom
            for idx in self.sensory_indices.get("visual_looming", ()):
                if idx < n: currents[idx] += i_loom

        # 3. Courtship pheromone (DA1 cVA)
        cva_stim = float(sensory.get("cva_concentration", kwargs.get("cva_odor", 0.0)))
        if cva_stim > 0.001:
            i_cva = float(min(35.0, cva_stim * 30.0))
            probe_current["courtship_cva"] = i_cva
            for idx in self.sensory_indices.get("courtship_cva", ()):
                if idx < n: currents[idx] += i_cva

        # 4. Wind mechanoreception (Johnston's organ drag)
        wind_speed = float(sensory.get("wind_speed", 0.0))
        if wind_speed > 3.0:
            i_wind = float(min(35.0, wind_speed * 0.15))
            probe_current["jon_wind"] = i_wind
            for idx in self.sensory_indices.get("jon_wind", ()):
                if idx < n: currents[idx] += i_wind

        # 5. Thermosensory receptors (TRN)
        temp = float(sensory.get("temperature", kwargs.get("temperature", 24.0)))
        if abs(temp - 24.0) > 2.0:
            i_temp = float(min(40.0, abs(temp - 24.0) * 2.5))
            probe_current["thermo"] = i_temp
            for idx in self.sensory_indices.get("thermo_receptors", ()):
                if idx < n: currents[idx] += i_temp

        # Step the graph instance
        result = instance.step(currents, self.step_ms)
        counts = result.counts
        self.last_counts = counts
        self.last_total_spikes = int(counts.sum())

        # Decode descending neuron firing rates (Hz)
        sec = self.step_ms / 1000.0
        dn_unavailable = {}

        def decoded_rate(name, *, per_neuron):
            indices = [int(i) for i in self.dn_indices.get(name, ()) if int(i) < n]
            spikes = sum(counts[i] for i in indices)
            numeric = float(spikes / ((len(indices) if per_neuron else 1) * sec)) if indices else 0.0
            if not indices:
                dn_unavailable[name] = "no neurons resolved for this channel on this graph"
            return spikes, numeric, (round(numeric, 2) if indices else None)

        spk_dna02_l, dna02_rate_l, shown_dna02_l = decoded_rate("dna02_l", per_neuron=False)
        spk_dna02_r, dna02_rate_r, shown_dna02_r = decoded_rate("dna02_r", per_neuron=False)
        spk_dnp09, dnp09_rate, shown_dnp09 = decoded_rate("dnp09", per_neuron=True)
        spk_dnb01, bpn_rate, shown_dnb01 = decoded_rate("dnb01", per_neuron=True)
        spk_mdn, mdn_rate, shown_mdn = decoded_rate("mdn", per_neuron=True)
        spk_dnp01, dnp01_rate, shown_dnp01 = decoded_rate("dnp01", per_neuron=False)

        # Steering & forward drive
        yaw_rate = 0.02 * (dna02_rate_l - dna02_rate_r)
        forward_speed = min(35.0, max(0.0, dnp09_rate * 1.5 + bpn_rate * 0.4))
        state = "GRAPH"

        if mdn_rate > 20.0:
            forward_speed = -15.0
            state = "REVERSE"
        # Escape comes from the connectome alone: a DNp01 (Giant Fiber) spike in
        # this step.  Stimulus strength never triggers it (LC4/LPLC2 drive has to
        # propagate through the graph to the GF).
        if spk_dnp01 > 0:
            forward_speed = 35.0
            state = "ESCAPE"

        # EPG compass bump
        epg_wedges = [0.0] * 16
        bump_phase = 0.0
        if self.epg_indices:
            for k, node in enumerate(self.epg_indices):
                if node < n:
                    w_idx = int(k * 16 / len(self.epg_indices)) % 16
                    epg_wedges[w_idx] += float(counts[node] / sec)
            wedge_angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)
            s_sum = sum(epg_wedges[i] * np.sin(wedge_angles[i]) for i in range(16))
            c_sum = sum(epg_wedges[i] * np.cos(wedge_angles[i]) for i in range(16))
            if abs(s_sum) > 1e-4 or abs(c_sum) > 1e-4:
                bump_phase = float(np.arctan2(s_sum, c_sum))

        wp6_mean_delta = 0.0
        wp6_max_delta = 0.0
        if instance.backend == 'connectome-plastic' and len(getattr(instance, "plastic_delta", [])) > 0:
            wp6_mean_delta = float(np.mean(instance.plastic_delta))
            wp6_max_delta = float(np.max(np.abs(instance.plastic_delta)))

        probe_indices = {
            "orn_food": "orn_food", "orn_danger": "orn_danger",
            "photoreceptor_l": "visual_l", "photoreceptor_r": "visual_r",
            "looming": "visual_looming", "courtship_cva": "courtship_cva",
            "jon_wind": "jon_wind", "thermo": "thermo_receptors",
        }
        input_stage = []
        for name, channel in probe_indices.items():
            indices = [int(i) for i in self.sensory_indices.get(channel, ()) if int(i) < n]
            input_stage.append({
                **dict(self.INPUT_CHANNELS[name]),
                "current_injected": (round(float(probe_current[name]), 6) if indices else None),
                "n_cells": len(indices),
                "n_spiking_this_step": sum(bool(counts[i]) for i in indices),
                "unavailable": (None if indices else
                                "no neurons resolved for this probe on this graph"),
            })

        raw_motor_command = {"forward_speed_mm_s": float(forward_speed),
                             "yaw_rate_rad_s": float(yaw_rate), "state": state}
        return {
            "halted": False,
            "forward_speed": float(forward_speed),
            "yaw_rate": float(yaw_rate),
            "motor_source": "graph",
            "controller_fault": None,
            "state": state,
            "graph_step": instance.step_index,
            "graph_step_ms": self.step_ms,
            "total_spikes": self.last_total_spikes,
            "dn_rates": {
                "dna02_l": shown_dna02_l,
                "dna02_r": shown_dna02_r,
                "dnp09": shown_dnp09,
                "dnb01": shown_dnb01,
                "mdn": shown_mdn,
                "gf": shown_dnp01,
            },
            "dn_unavailable": dn_unavailable,
            "epg_wedges": [round(w, 2) for w in epg_wedges],
            "epg_bump_phase": round(bump_phase, 4),
            "epg_available": bool(self.epg_indices),
            "epg_resolved_count": len(self.epg_indices),
            "epg_unavailable": (None if self.epg_indices else
                                (self.sensory_unavailable or
                                 "no EPG neurons resolved for this graph")),
            "wp6": {
                "mean_delta": round(wp6_mean_delta, 6),
                "max_delta": round(wp6_max_delta, 6),
                "n_edges": len(getattr(instance, "plastic_edges", [])),
            },
            "engineered_assistance_enabled": self.ENGINEERED_ASSISTANCE_ENABLED,
            "engineered_assistance_applied": [],
            "graph_io": graph_io_declaration(),
            "decoder": dict(self.DECODER_DECLARATION),
            "raw_motor_command": raw_motor_command,
            "input_stage": input_stage,
            "optomotor": None,
        }


class TimedLock:
    """The simulation lock, instrumented and polite.

    Records how long each thread role waited to acquire it, and exposes the number
    of waiting threads so the simulation loop can yield between step batches
    instead of re-acquiring immediately (``threading.Lock`` is not fair).
    Supports the ``with`` protocol and ``acquire``/``release`` like a plain lock.
    """

    def __init__(self, samples: int = 4096):
        self._lock = threading.Lock()
        self._meta = threading.Lock()
        self.waiting = 0
        self._waits: Dict[str, deque] = {}
        self._totals: Dict[str, List[float]] = {}  # role -> [count, total_s, max_s]
        self._samples = samples

    @staticmethod
    def _role() -> str:
        name = threading.current_thread().name
        if name.startswith("NeuroFly-SimLoop"):
            return "simulation"
        if name.startswith("NeuroFly-Recorder"):
            return "recorder"
        if name == "MainThread":
            return "main"
        return "http"

    def _record(self, wait_s: float) -> None:
        role = self._role()
        with self._meta:
            self._waits.setdefault(role, deque(maxlen=self._samples)).append(wait_s)
            tot = self._totals.setdefault(role, [0, 0.0, 0.0])
            tot[0] += 1
            tot[1] += wait_s
            tot[2] = max(tot[2], wait_s)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(False):
            self._record(0.0)
            return True
        if not blocking:
            return False
        start = time.perf_counter()
        with self._meta:
            self.waiting += 1
        try:
            ok = self._lock.acquire(True, timeout)
        finally:
            with self._meta:
                self.waiting -= 1
        if ok:
            self._record(time.perf_counter() - start)
        return ok

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def profile(self) -> Dict[str, Any]:
        """Per-role lock wait summary in milliseconds (recent-window percentiles)."""
        out = {}
        with self._meta:
            items = [(role, sorted(w), list(self._totals[role])) for role, w in self._waits.items()]
        for role, waits, (count, total, peak) in items:
            def pct(q):
                return round(waits[min(len(waits) - 1, int(q * len(waits)))] * 1e3, 3) if waits else 0.0
            out[role] = {"acquisitions": count, "mean_wait_ms": round(total / count * 1e3, 3) if count else 0.0,
                         "p50_wait_ms": pct(0.50), "p95_wait_ms": pct(0.95), "p99_wait_ms": pct(0.99),
                         "max_wait_ms": round(peak * 1e3, 3), "window": len(waits)}
        return out


def _percentiles_ms(values) -> Dict[str, float]:
    vals = sorted(values)
    if not vals:
        return {"count": 0}
    pick = lambda q: round(vals[min(len(vals) - 1, int(q * len(vals)))] * 1e3, 2)
    return {"count": len(vals), "p50_ms": pick(0.5), "p95_ms": pick(0.95), "max_ms": round(vals[-1] * 1e3, 2)}


class _Snapshot:
    """One immutable published telemetry frame: serialized once, shared by every reader."""

    __slots__ = ("seq", "step", "data", "wall_time")

    def __init__(self, seq: int, step: int, data: bytes, wall_time: float):
        self.seq = seq
        self.step = step
        self.data = data
        self.wall_time = wall_time


class SlowStepBallast:
    """TEST ONLY: makes every daemon step take about ``wall_ms`` longer on this CPU.

    It runs the real compiled LIF kernel (``Brain.step``, CPU backend) on a private,
    fully driven synthetic graph, so a fast machine behaves like a slow laptop:
    the step holds the simulation lock and runs the same native code, with the
    same GIL behaviour.  The ballast brain is never read, so the simulated
    trajectory is unchanged (tests/test_daemon_responsiveness.py).
    """

    def __init__(self, wall_ms: float, neurons: int = 50000, k_out: int = 40):
        from brainlab.brain import Brain
        from brainlab.graph_identity import synthetic_test_graph
        arrays, _, _ = synthetic_test_graph(n=neurons, k_out=k_out, seed=0)
        self.brain = Brain(arrays=arrays, dynamics="v3", backend="cpu")
        self.drive = np.full(self.brain.n, 30.0, dtype=np.float32)
        self.brain.step(self.drive, 0.1)   # compile before the first timed step
        start = time.perf_counter()
        self.brain.step(self.drive, 2.0)
        per_ms = max(1e-6, (time.perf_counter() - start) / 2.0)
        # ONE long kernel call per step, like the real brain step on a slow laptop.
        self.duration_ms = max(0.1, round(max(0.0, float(wall_ms)) / 1e3 / per_ms, 1))

    def __call__(self, runner=None):
        self.brain.step(self.drive, self.duration_ms)


class DiskSpaceLow(OSError):
    """Free space is below what the next checkpoint needs; the write was skipped."""


class NonFiniteStep(RuntimeError):
    """A step produced NaN/inf state: a compute failure (always halts)."""
    failure_class = "compute"


class LoopExit(RuntimeError):
    """The simulation scheduler ended while the daemon still expected it to run."""


# Failure classes (Codex direction 5 Oct 2026, decision 3).  A save that fails because
# of the storage (disk full, I/O error, read-only, permissions) is a PERSISTENCE
# failure.  A GPU/CUDA or memory error -- even one raised while a checkpoint copies
# device state -- is a COMPUTE failure and always halts.  Anything else raised by a
# save is a SOFTWARE failure (a bug), which also halts: it is never treated as
# ordinary disk I/O.
FAILURE_PERSISTENCE, FAILURE_COMPUTE, FAILURE_SOFTWARE = "persistence", "compute", "software"
_DEVICE_MARKERS = ("cuda", "cupy", "cublas", "cusparse", "curand", "nvrtc", "cudnn", "hip error", "rocm",
                   "device-side assert", "gpu ", "device arithmetic")
_DEVICE_MODULES = ("cupy", "cupy_backends", "numba.cuda", "pycuda", "torch.cuda", "wgpu")


def classify_failure(exc: BaseException) -> str:
    """persistence / compute / software, decided explicitly (see the note above)."""
    declared = getattr(exc, "failure_class", None)   # e.g. NonFiniteState declares "compute"
    if declared in (FAILURE_PERSISTENCE, FAILURE_COMPUTE, FAILURE_SOFTWARE):
        return declared
    module = type(exc).__module__ or ""
    if module.startswith(_DEVICE_MODULES) or isinstance(exc, MemoryError):
        return FAILURE_COMPUTE
    if isinstance(exc, OSError):
        # The errno decides; the message is not searched (a path may contain "cuda").
        return FAILURE_PERSISTENCE
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _DEVICE_MARKERS):
        return FAILURE_COMPUTE
    return FAILURE_SOFTWARE


# Channels whose writes are part of the scientific record.  Their failure stops the
# run (or, in explicit exploratory mode, degrades it with a recorded gap).  Only the
# DIAGNOSTIC channels may fail and merely degrade in the default mode.
REQUIRED_CHANNELS = ("checkpoint", "brain_save", "trial_ledger", "events_ledger", "recording",
                     "learning_records", "provenance")
DIAGNOSTIC_CHANNELS = ("telemetry_summary", "halt_log")


class PersistenceMonitor:
    """What the daemon failed to save, since when, and when it retries (audit F, F2).

    Bookkeeping only; the policy (stop the run, or in exploratory mode degrade) is
    applied by ContinuousExperimentRunner._persistence_failed.  Each channel
    (checkpoint, trial ledger, halt bookkeeping, recording, ...) is tracked
    separately, so a later successful write on one channel cannot hide a failure on
    another.  The periodic checkpoint retries with a back-off (30 s doubling to
    10 min) instead of on every step.
    """

    BACKOFF_START_S = 30.0
    BACKOFF_MAX_S = 600.0

    def __init__(self):
        self._lock = threading.Lock()
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.last_ok: Dict[str, float] = {}      # channel -> wall time of the last good write
        self.total_failures = 0

    def failed(self, channel: str, exc: BaseException, path: Any = None, now: Optional[float] = None,
               failure_class: Optional[str] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self._lock:
            entry = self.channels.get(channel)
            if entry is None:
                entry = self.channels[channel] = {"channel": channel, "since": round(now, 3), "failures": 0,
                                                  "backoff_s": 0.0}
            entry["failures"] += 1
            self.total_failures += 1
            entry["backoff_s"] = (self.BACKOFF_START_S if not entry["backoff_s"]
                                  else min(self.BACKOFF_MAX_S, entry["backoff_s"] * 2))
            entry.update(state="disk_low" if isinstance(exc, DiskSpaceLow) else "failing",
                         error=f"{type(exc).__name__}: {exc}", errno=getattr(exc, "errno", None),
                         path=str(path or getattr(exc, "filename", None) or "") or None,
                         failure_class=failure_class or classify_failure(exc),
                         required=channel not in DIAGNOSTIC_CHANNELS,
                         last_failure_at=round(now, 3), next_retry_at=round(now + entry["backoff_s"], 3))
            return dict(entry)

    def succeeded(self, channel: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.channels.pop(channel, None)
            self.last_ok[channel] = now

    def clear(self, channel: str) -> None:
        with self._lock:
            self.channels.pop(channel, None)

    def retry_due(self, channel: str, now: Optional[float] = None) -> bool:
        entry = self.channels.get(channel)
        return entry is None or (time.time() if now is None else now) >= entry["next_retry_at"]

    def failing(self) -> bool:
        return bool(self.channels)

    def describe(self, extra: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """JSON-safe summary for /api/status, frames and heartbeats."""
        now = time.time()
        with self._lock:
            channels = {k: dict(v) for k, v in self.channels.items()}
            last_ok = dict(self.last_ok)
        channels.update(extra or {})
        state = "ok"
        if channels:
            state = "disk_low" if all(c.get("state") == "disk_low" for c in channels.values()) else "failing"
        last_save = last_ok.get("checkpoint")
        out = {"state": state, "ok": not channels, "failing": channels, "total_failures": self.total_failures,
               "last_ok_save_at": round(last_save, 3) if last_save else None,
               "last_ok_save_age_s": round(now - last_save, 1) if last_save else None}
        if channels:
            first = min(channels.values(), key=lambda c: c.get("since") or now)
            disk = any(c.get("errno") == 28 or c.get("state") == "disk_low" for c in channels.values())
            out["since"] = first.get("since")
            out["reason"] = "disk full" if disk else first.get("error")
            out["summary"] = "; ".join(f"{name}: {c.get('error')}" for name, c in sorted(channels.items()))
        return out


_RUNNERS: "weakref.WeakSet" = None   # set lazily by _install_thread_excepthook


def _install_thread_excepthook(runner) -> None:
    """Backstop (F1): an uncaught exception in any NeuroFly-* thread is recorded on
    the runner, so status reports it instead of a traceback scrolling past on stderr."""
    global _RUNNERS
    import weakref
    if _RUNNERS is None:
        _RUNNERS = weakref.WeakSet()
    _RUNNERS.add(runner)
    if getattr(threading.excepthook, "_neurofly", False):
        return
    previous = threading.excepthook

    def hook(args):
        name = getattr(args.thread, "name", "") or ""
        if name.startswith("NeuroFly"):
            for r in list(_RUNNERS):
                try:
                    r._note_thread_failure(name, args.exc_type, args.exc_value, args.exc_traceback)
                except Exception:  # noqa: BLE001 -- the hook itself must never raise
                    pass
        previous(args)

    hook._neurofly = True
    threading.excepthook = hook


def _traceback_tail(exc: BaseException, lines: int = 8) -> List[str]:
    import traceback
    text = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(text).rstrip().splitlines()[-lines:]


class ContinuousExperimentRunner:
    """Manages the continuous headless simulation loop and online plasticity."""

    # Legacy milestone flags remain for uninstrumented compatibility callers. Once an
    # observation owner is attached, its schema 1.2 status is the sole measurement
    # authority; these flags and the compatibility duration cannot reset that owner.
    TRIAL_END_FLAGS = ("goal_reached", "source_reached", "refuge_reached", "crossing_success", "escape_initiated")
    # Ordered candidates for the per-trial learning-curve value, with a scale to [0, 1].
    TRIAL_METRIC_KEYS = (
        ("performance_index", 1.0), ("pi", 1.0), ("learning_index", 1.0),
        ("operant_learning_index", 1.0), ("spontaneous_alternation_rate", 1.0),
        ("centrophobism_index", 1.0), ("courtship_index", 1.0), ("optomotor_gain", 1.0),
        ("composite_benchmark_score", 0.01), ("surge_to_cast_ratio", 1.0),
    )

    def __init__(
        self,
        initial_paradigm: str = "multisensory-sandbox",
        sim_speed: float = 10.0,
        checkpoint_interval: float = 60.0,
        output_dir: Optional[Path] = None,
        trial_length_s: Optional[float] = None,
        continuous: bool = False,
        backend: str = "modular",
        graph_dir: Optional[Path] = None,
        test_synthetic_graph: bool = False,
        graph_step_ms: Optional[float] = None,
        shared_graph: Any = None,
        registry_root: Optional[Path] = None,
        keep_checkpoints: Optional[int] = None,
        exploratory: bool = False,
        continue_io_state: bool = False,
        standalone_scheduled_records: bool = True,
        brain_backend: Optional[str] = None,
    ):
        self.brain_backend = compute_selector(brain_backend)
        require_fixed_amd(self.brain_backend, backend, continue_io_state=continue_io_state,
                          step_ms=graph_step_ms)
        requested_observation_window_s = preflight_observation_policy(
            initial_paradigm, trial_length_s, continuous)
        if backend in GRAPH_BACKENDS:
            daemon_configuration()     # refuse unsupported dynamics or E_inh before any directory is touched
        if self.brain_backend == 'wgpu-amd':
            from experiment_registry import default_registry_dir
            preflight_fixed_store(Path(registry_root) if registry_root else default_registry_dir(
                Path(output_dir) if output_dir else PROJECT_ROOT / 'outputs'))
        self.output_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "outputs")
        self.standalone_scheduled_records = bool(standalone_scheduled_records and output_dir is not None)
        self._scheduled_records_owner = None
        self._scheduled_records_retiring = False
        self._scheduled_records_lock = threading.Lock()
        self._scheduled_records_cleanup_guard = threading.Lock()
        self._scheduled_records_cleanup_thread = None
        self._scheduled_records_cleanup_ok = False
        self.graph_dir = graph_dir
        self.registry_root = registry_root
        # Newest checkpoints kept per graph instance and per assay (0 keeps all).
        self.keep_checkpoints = resolve_keep_checkpoints(keep_checkpoints)
        self.continue_io_state = bool(continue_io_state)
        self.graph_step_ms = graph_step_ms if graph_step_ms is not None else 20.0
        # Controller backend (provenance.BACKENDS).  A graph backend loads ONE shared
        # immutable graph and ONE ExperimentRegistry; a missing or mismatching graph
        # raises here (GraphUnavailable), never falls back to another controller.
        # ``test_synthetic_graph`` is the explicit, labelled test option.
        if backend not in DAEMON_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose one of {DAEMON_BACKENDS}")
        self.backend = backend
        self.backend_spec = get_backend(backend, scientific=not test_synthetic_graph, allow_test=test_synthetic_graph)
        self.graph_mode = backend in GRAPH_BACKENDS
        self.test_mode = bool(test_synthetic_graph)
        self._source = source_revision(files=self.backend_spec.source_files)
        self.registry = None
        self.shared_graph = None
        self.graph_controller = None
        self._graph_arenas: Dict[str, Any] = {}
        self.activation = 0
        self.manifest: Optional[RunManifest] = None
        if self.graph_mode:
            from experiment_registry import (ExperimentRegistry as GraphRegistry, SharedGraph,
                                             default_registry_dir)
            if shared_graph is None:
                shared_graph = (SharedGraph.synthetic(allow_synthetic=True) if test_synthetic_graph
                                else SharedGraph.load_for_dynamics(graph_dir))
            self.shared_graph = shared_graph
            plasticity_rule = None
            if backend == "connectome-plastic" and not getattr(shared_graph.identity, 'synthetic', False):
                from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
                plasticity_rule = VisualHeadingPlasticityRule.from_shared(shared_graph)
            self.registry = GraphRegistry(shared_graph, Path(registry_root) if registry_root else
                                          default_registry_dir(self.output_dir), test_mode=self.test_mode,
                                          plasticity_rule=plasticity_rule,
                                          keep_checkpoints=self.keep_checkpoints,
                                          continue_io_state=self.continue_io_state,
                                          brain_backend=self.brain_backend)
            self.graph_controller = GraphArenaController(
                self, self.graph_step_ms)
            # Bookkeeping (curves, event logs) for graph runs never shares files with
            # the modular brains: modular weights are not graph weights.
            self.brains = ExperimentBrains(self.output_dir / "graph-bookkeeping" / backend)
        else:
            self.brains = ExperimentBrains(self.output_dir / "brains")
        self._backend_brains = {self.backend: self.brains}
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.telemetry_dir = self.output_dir / "telemetry"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.continuous = bool(continuous)
        self.paused = False
        # Fail-safe halt: an exception inside a step stops the simulation (nothing
        # advances while ``last_error`` is set) instead of stepping a broken state.
        # The halt is reported (status "error", packet ``halted``, the dashboard
        # pill) and lifted only by a command that rebuilds the controller and world
        # (a successful assay or backend switch); see _halt_on_error/_clear_error.
        self.last_error = None
        self.error_detail: Optional[Dict[str, Any]] = None
        # Public readers capture one detached cause/detail pair without taking the runner lock.
        self._fault_snapshot = (None, None)
        # Detached fault metadata can still be delivered after the frame publisher dies.
        self._observation_validity_update = None
        self.cleared_errors: deque = deque(maxlen=16)   # lifted halts, newest last
        self.run_id = uuid.uuid4().hex
        self.transition = None
        self.lock = TimedLock()
        self.running = False
        # Scheduling and delivery (docs: fixed dt; the wall-clock deadline only decides
        # WHEN the next fixed step runs, never its size).  Snapshots are published at
        # ``publish_hz`` while running and ``paused_publish_hz`` while paused; readers
        # never take the simulation lock to deliver them.
        self.publish_hz = 60.0
        self.paused_publish_hz = 2.0
        self.max_batch_wall_s = 0.008   # longest uninterrupted lock hold for a step batch
        self.max_lag_wall_s = 0.25      # schedule debt beyond this is forgiven, not burst
        self.yield_wall_s = 0.001       # GIL hand-over after every batch (see _yield_lock)
        # A command is applied by the simulation thread at the next step boundary.  On
        # a slow computer one step can take seconds, so the HTTP reply waits at most
        # ``command_reply_wait_s``; after that it answers ``queued`` with a command id
        # and the acknowledgement follows in the stream (``command_acks``).  Commands
        # are never dropped and never applied mid-step, so results do not change.
        self.command_reply_wait_s = 1.0
        self.command_acks: deque = deque(maxlen=8)
        self._command_seq = 0
        self._step_started: Optional[float] = None   # perf_counter() while a step runs
        self.last_step_wall_s = 0.0
        # Views that need a consistent state (brain summaries, manifest) are rebuilt
        # at step boundaries while a client asks for them, so a reader never waits
        # behind a long step (read_view).
        self._views: Dict[str, Any] = {}
        self._view_demand: Dict[str, Any] = {}
        self._view_built: Dict[str, float] = {}
        self.published: Optional[_Snapshot] = None
        self._snapshot_seq = 0
        self._publish_due = True
        self._last_step_result: Dict[str, Any] = {}
        self._path: deque = deque(maxlen=240)   # (step, x, y) of every recent step
        self._speed_samples: deque = deque(maxlen=256)
        self._commands: deque = deque()
        self._commands_lock = threading.Lock()
        self._wake = threading.Event()
        self._scheduled: Dict[int, List[Dict[str, Any]]] = {}
        self.command_latency: deque = deque(maxlen=1024)
        self.stop_at_step: Optional[int] = None   # test/replay hook: hold at this step
        self.step_hook = None                     # test hook: called under lock after each step
        self.recorder = None                      # neurofly.recording.RunRecorder while recording
        self.recordings_dir = self.output_dir / "recordings"
        self.sched_stats = {"rebases": 0, "forgiven_wall_s": 0.0, "batches": 0,
                            "max_batch_hold_ms": 0.0, "steps_since_publish": 0, "achieved_speed": 0.0,
                            "published_snapshots": 0}
        # Failure handling (audit F, docs/LEARNING_OBSERVATORY.md "When something fails").
        self.persistence = PersistenceMonitor()
        # Scientific mode (default): a failed REQUIRED save stops the run and marks its
        # result incomplete.  --exploratory: keep stepping, NOT SAVING, gap recorded.
        self.exploratory = bool(exploratory)
        self._stopping_failures = 0
        self.incidents: List[Dict[str, Any]] = []   # append-only result-validity incidents
        self._pending_validity: List[Dict[str, Any]] = []
        self._validity_flushing = False
        self._validity_writer_lock = threading.Lock()
        self.disk_reserve_bytes = 512 * 1024 * 1024
        self._last_checkpoint_bytes: Optional[int] = None
        self.keep_shutdown_checkpoints = 5          # final_shutdown JSON records kept per assay
        self.recording_error: Optional[Dict[str, Any]] = None
        self.learning_records = None                # RecorderThread, set by run_daemon
        self.recorder_dead_grace_s = 2.0
        self.recorder_progress_grace_s = 2.0
        self.recorder_stuck_s = 30.0
        self._recorder_watchdog_failure: Optional[Dict[str, Any]] = None
        self._recorder_watchdog_failures: deque = deque(maxlen=8)
        self._recorder_watchdog_episode = 0
        self.observation_publication = ObservationPublicationQueue()
        self.observation_durability_reason = "durable learning records are disabled"
        self.loop_failure: Optional[Dict[str, Any]] = None
        self._state_uncertain = False
        self.thread_failures: deque = deque(maxlen=8)
        self._loop_phase = "idle"
        self._publish_failures = 0
        self._loop_fault_log = 0.0
        # Watchdog (F3): when a step last completed, when stepping last became
        # expected (after a pause, halt or command), and recent (wall, step) samples
        # from which the achieved speed is computed at read time.
        self.last_advance_wall: Optional[float] = None
        self._steppable_since: float = time.perf_counter()
        self._command_started: Optional[float] = None
        self._advance_samples: deque = deque(maxlen=8192)
        self.step_hard_limit_s = 300.0
        self.min_stall_s = 10.0
        self.exit_on_stall_s: Optional[float] = None   # --exit-on-stall
        self._exit = os._exit                          # replaced in tests
        self.watchdog_interval_s = 2.0
        self._watchdog_stop = threading.Event()
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._stop_ok = True
        self.sim_speed = max(0.1, min(100.0, float(sim_speed)))
        self.dt = 0.02
        self.active_paradigm_id = initial_paradigm
        self.checkpoint_interval = max(5.0, float(checkpoint_interval))
        # Compatibility field: an explicit CLI value is an observation-policy request.
        # None means the assay specification supplies the window; there is no daemon 60 s default.
        self.trial_length_s = requested_observation_window_s
        self._observation_policy_command_id = None
        self._observation_policy_set_at_sim_s = 0.0
        self._observation_policy_source = ("cli:--continuous" if self.continuous else
                                           "cli:--trial-seconds" if self.trial_length_s is not None else None)

        # Runtime state
        self.start_time = time.time()
        self.total_steps = 0
        self.current_trial = 1          # active assay's trial number (its saved trial clock)
        # Session-wide trial number of trial records: starts at 1 in every process and
        # stays monotonic across assay switches (the learning recorder's append key).
        self.session_trial = 1
        self.trial_sim_time = 0.0
        self.last_checkpoint_time = time.time()
        self.trial_history: List[Dict[str, Any]] = []
        self.learning_curve: List[float] = []

        # Intervention override state
        self.stimulus_overrides: Dict[str, Any] = {
            "override_dna02": 0.0,
            "override_thrust": 0.0,
            "override_mdn": 0.0,
            "override_gf": False,
            "flare_odor_a": 0.0,
            "flare_temp": 0.0,
            "active": False
        }

        # Subscriber queues for Server-Sent Events (SSE)
        self.subscribers: List[threading.Event] = []
        self.latest_telemetry: Dict[str, Any] = {}
        self.observation_segment_start_sim_s = 0.0
        self.observation_config = None
        self._observation_terminal: Optional[Dict[str, Any]] = None
        self._pending_assay_control = None
        self._teaching_suspended = False
        self._teaching_prefix = None
        self._teaching_clock_s = 0.0
        self._executing_command_entry = None
        self.assay_control_timeout_s = 30.0
        self.assay_control_failures = deque(maxlen=32)

        # Result-validity ledger written by earlier processes on this output directory:
        # incidents are reconciled into a run when it is activated again, and a session
        # that never shut down cleanly marks the runs it had active as interrupted.
        self._validity_ledger = self._read_validity_ledger()
        self._interrupted = self._interrupted_sessions(self._validity_ledger)
        self._pending_validity.append({"event": "session_start", "session": self.run_id,
                                       "pid": os.getpid(), "at": round(time.time(), 3)})

        # Initialize primary simulation arena
        self._init_arena(self.active_paradigm_id)

    def _switch_backend(self, target_backend: str):
        """Switch controller backend between modular, connectome-fixed, and connectome-plastic under lock."""
        require_fixed_amd(self.brain_backend, target_backend)
        if target_backend == self.backend:
            return
        if target_backend not in DAEMON_BACKENDS:
            raise ValueError(f"Unknown backend {target_backend!r}; choose one of {DAEMON_BACKENDS}")

        target_graph_mode = target_backend in GRAPH_BACKENDS
        if target_graph_mode:
            daemon_configuration()     # refuse unsupported dynamics or E_inh before a registry is created
            from experiment_registry import (ExperimentRegistry as GraphRegistry, SharedGraph,
                                             default_registry_dir)
            if self.shared_graph is None:
                self.shared_graph = (SharedGraph.synthetic(allow_synthetic=True) if self.test_mode
                                     else SharedGraph.load_for_dynamics(self.graph_dir))
            if self.registry is None:
                self.registry = GraphRegistry(self.shared_graph,
                                              Path(self.registry_root) if self.registry_root else default_registry_dir(self.output_dir),
                                              test_mode=self.test_mode,
                                              keep_checkpoints=self.keep_checkpoints,
                                              continue_io_state=self.continue_io_state,
                                              brain_backend=self.brain_backend)
            if self.graph_controller is None:
                self.graph_controller = GraphArenaController(self, self.graph_step_ms)
            if target_backend == "connectome-plastic" and self.registry.plasticity_rule is None:
                if not getattr(self.shared_graph.identity, 'synthetic', False):
                    from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
                    self.registry.plasticity_rule = VisualHeadingPlasticityRule.from_shared(self.shared_graph)

        self.backend = target_backend
        self.backend_spec = get_backend(target_backend, scientific=not self.test_mode, allow_test=self.test_mode)
        self.graph_mode = target_graph_mode
        self._source = source_revision(files=self.backend_spec.source_files)

        if self.graph_mode:
            self.brains = ExperimentBrains(self.output_dir / "graph-bookkeeping" / self.backend)
        else:
            self.brains = ExperimentBrains(self.output_dir / "brains")

        # Activate the active experiment with the new backend
        self._init_arena(self.active_paradigm_id)
        print(f"[Daemon] Switched controller backend to {self.backend} (graph_mode={self.graph_mode})", flush=True)
        print(f"[Daemon] Brain compute: {self.compute_info()['detail']}", flush=True)

    def _init_arena(self, paradigm_name: str):
        """Activate an experiment's own arena and learned state; never share weights."""
        require_fixed_amd(self.brain_backend, self.backend)
        if self.brain_backend == 'wgpu-amd':
            preflight_fixed_store(self.registry.root, paradigm_name)
        # Resolve first so an invalid request cannot disturb the active experiment.
        brain = self.brains.get(paradigm_name)
        manifest_path = None    # graph manifests are written by the registry
        if self.graph_mode:
            # Checkpoint the outgoing instance WITH its world, activate the target
            # (brain snapshot restored), then restore the target's world.  Only after
            # both are ready does this return, so the switch ack names a ready instance.
            active = self.registry.active
            if active is not None and hasattr(self, "arena") and not self._state_uncertain:
                if hasattr(self, "active_brain"):
                    self._sync_trial_clock()
                active.world_state = self.arena.snapshot_world()
            instance = self.registry.activate(paradigm_name, self.backend)
            # Incidents saved with this instance (from an earlier process) stay attached:
            # a restart or re-select never makes an incomplete run look valid again.
            known = {(i.get("run_id"), i.get("at"), i.get("reason")) for i in self.incidents}
            for incident in getattr(instance, "invalidity", None) or []:
                if (incident.get("run_id"), incident.get("at"), incident.get("reason")) not in known:
                    self.incidents.append(dict(incident, restored_from_checkpoint=True))
            arena = self._graph_arenas.get(paradigm_name)
            if arena is None:
                arena = self._build_graph_arena(paradigm_name, instance.seed)
                self._graph_arenas[paradigm_name] = arena
            if instance.world_state:
                arena.restore_world(instance.world_state)
            brain.arena = arena
            manifest = instance.manifest
        else:
            manifest = RunManifest.create(
                backend="modular", assay=paradigm_name, instance_id=brain.brain_id, seed=brain.seed, graph=None,
                dynamics={"model": "modular MB + CX + surge-cast (experiment_brains/arena)", "dt_s": self.dt,
                          "motor_assists": dict(brain.arena.motor_assists)},
                learned_parameter_locations={"mb_weights": str(brain.path),
                                             "events": str(brain.directory / f"{paradigm_name}.events.jsonl")},
                parent_run_id=self.manifest.run_id if self.manifest is not None else None,
                source=self._source)
            manifest_path = self.output_dir / "manifests" / f"{manifest.run_id}.json"
        if hasattr(self, "active_brain") and not self._state_uncertain:
            self._sync_trial_clock()
            # A required save: under the save policy a failure stops the run (scientific
            # mode) or degrades it with a recorded gap (exploratory).
            self._persist("brain_save", self.active_brain.save, path=self.active_brain.path)
        self.manifest = manifest
        self.activation += 1
        manifest.record_event("activate", step=getattr(self, "total_steps", 0), activation=self.activation,
                              daemon_run_id=self.run_id, clock_scope=SESSION_CLOCK_SCOPE)
        self.segment_id = uuid.uuid4().hex
        self.transition = {"reason": "experiment_selected", "step": self.total_steps}
        self.active_brain = brain
        self.arena = brain.arena
        if self.brain_backend == 'wgpu-amd':
            self.active_brain.learning_enabled = self.arena.fly.learning_enabled = False
        self.active_paradigm_id = paradigm_name
        self.active_paradigm_title = getattr(self.arena.paradigm, "name", "Open Arena Assay")
        self._restore_trial_clock(brain, self.registry.active if self.graph_mode else None)
        self.learning_curve = brain.curve
        self._path.clear()
        self._last_step_result = {}
        self._publish_due = True
        self._configure_observation_owner()
        # Validity of this run as recorded on disk by earlier processes (fix 3).
        self._reconcile_validity()
        if manifest_path is not None:
            # The run's provenance record is required (scientific policy), not a log line.
            self._persist("provenance", manifest.write, manifest_path, path=manifest_path)
        self.latest_telemetry = self._assemble_telemetry({})

    def _restore_trial_clock(self, brain, instance=None) -> None:
        """Adopt the trial clock saved with the state that was just restored.

        Graph runs use only the clock in the selected verified checkpoint (including
        an older fallback version), never the helper brain JSON.  State saved without
        a clock stays explicitly unknown; nothing is inferred from brain, world or
        daemon counters.
        """
        if instance is None:
            clock = validate_trial_clock(brain.trial_clock)
        elif instance.trial_clock is not None:
            clock = validate_trial_clock(instance.trial_clock)
        elif instance.checkpoint_version:
            clock = unknown_trial_clock()
        else:
            clock = trial_clock()              # a new instance: nothing saved yet
        brain.trial_clock = clock
        if instance is not None:
            instance.trial_clock = clock       # one record for the NPZ and the helper JSON
        self.trial_sim_time = clock['elapsed_s']
        self.current_trial = clock['current_trial']

    def _sync_trial_clock(self, *, new_trial: bool = False, brain=None) -> None:
        """Write the runner's trial counters into the active assay's clock record.

        ``new_trial`` marks a trial that started in front of this process, so its
        elapsed time is known from zero.  An unknown trial ordinal stays unknown.
        """
        clock = (brain or self.active_brain).trial_clock
        clock['elapsed_s'] = float(self.trial_sim_time)
        clock['current_trial'] = int(self.current_trial)
        if new_trial and not clock['elapsed_known']:
            clock['elapsed_known'] = True
            if clock['trial_known']:
                clock['reason'] = None

    def clock_status(self) -> Dict[str, Any]:
        """The three clocks a display may show, each named for what it counts.

        ``session`` restarts with every daemon process (the packet's ``step`` and
        ``sim_time_s``).  ``trial`` is the active assay's saved trial clock.  ``graph``
        is the retained graph instance's own neural step count and simulated time,
        restored from its checkpoint; it is not the trial's elapsed time.
        """
        clocks: Dict[str, Any] = {
            "session": {"scope": SESSION_CLOCK_SCOPE, "daemon_run_id": self.run_id,
                        "step": self.total_steps, "elapsed_s": round(self.total_steps * self.dt, 5)},
            "trial": dict(self.active_brain.trial_clock)}
        instance = self.registry.active if self.graph_mode and self.registry is not None else None
        sim_ms = getattr(getattr(instance, "brain", None), "sim_ms", None)
        if instance is not None and sim_ms is not None:
            clocks["graph"] = {"scope": "retained_graph_instance", "instance_id": instance.instance_id,
                               "step": int(instance.step_index), "elapsed_s": round(float(sim_ms) / 1000.0, 5)}
        return clocks

    def _observation_provenance(self) -> Dict[str, Any]:
        """Facts declared by the active arena/controller for metric-contract/1.2."""
        graph_states = ["GRAPH", "REVERSE", "ESCAPE", "OPTOMOTOR-TETHERED",
                        "NO-MOTOR-MAP", "HALTED"]
        modular_states = ["SURGE", "CAST", "REST", "FEED", "ESCAPE", "REVERSE", "WANDER"]
        return {
            "gf_source": self.arena._gf_source(self.arena.fly),
            # No common per-packet field currently names one stimulus ingress stage.
            "stimulus_entry_stage": None,
            "motor_assists_enabled": bool(any(self.arena.motor_assists.values())),
            "controller_states": graph_states if self.graph_mode else modular_states,
            "controller_specific": {
                "backend": self.backend,
                "learning_enabled": self.active_brain.learning_enabled,
                "graph_io": self.identity().get("graph_io"),
                "motor_assists": dict(self.arena.motor_assists),
            },
        }

    def _configure_observation_owner(self) -> None:
        """Bind the active arena's producer to a fresh, fully validated segment."""
        self._teaching_suspended = bool(self.active_brain.teaching)
        if self._teaching_suspended:
            self._teaching_clock_s = self.active_brain.teaching['tick'] * self.dt
        origin = float(self.total_steps * self.dt)
        config = build_observation_config(
            (self.arena.observation_owner or
             (self.arena.paradigm if self.arena.paradigm is not None else
              OpenArenaObserver())).observation_spec(),
            config_id=(f"segment:{self.segment_id}" +
                       (f":policy-command:{self._observation_policy_command_id}"
                        if self._observation_policy_command_id else "")),
            override_window_s=self.trial_length_s,
            continuous=self.continuous,
            override_source=self._observation_policy_source,
            set_at_sim_s=self._observation_policy_set_at_sim_s,
            manifest_run_id=self.manifest.run_id,
        )
        owner = self.arena.observation_owner
        if owner is None:
            owner = self.arena.enable_observation(config)
        else:
            owner.begin_observation_segment(config)
        owner.provenance = self._observation_provenance()
        self.observation_config = config
        self.observation_segment_start_sim_s = origin
        self._observation_terminal = None

    def _live_observation(self) -> Optional[Dict[str, Any]]:
        if self._teaching_suspended:
            return None
        owner = self.arena.observation_owner
        if owner is None:
            raise RuntimeError("active arena has no observation owner")
        owner.provenance = self._observation_provenance()
        identity = self.identity()
        identity["brain_id"] = self.active_brain.brain_id
        envelope = build_observation_envelope(
            owner.snapshot_observation(), identity=identity,
            provenance=self._observation_provenance(), segment_id=self.segment_id,
            segment_start_sim_s=self.observation_segment_start_sim_s,
            validity=self._observation_validity(), terminal_pose_post_step=None)
        # Re-parse the strict representation so telemetry never aliases producer state.
        return json.loads(dumps_observation_envelope(envelope, sort_keys=True))

    def _observation_validity(self) -> str:
        """Map the current run policy onto metric-contract/1.2's closed vocabulary."""
        result = self.result_validity()
        if result["state"] == "valid_so_far":
            return "valid"
        return "exploratory_degraded" if result["mode"] == "exploratory" else "invalidated"

    def observation_lifecycle_status(self) -> Dict[str, Any]:
        owner = self.arena.observation_owner
        status = owner.observation_status()
        terminal = self._observation_terminal
        waiting = self._observation_waiting_for_save()
        return {
            "segment_id": self.segment_id,
            "segment_start_sim_s": self.observation_segment_start_sim_s,
            "observation_available": not self._teaching_suspended,
            "teaching_clock_s": self._teaching_clock_s if self._teaching_suspended else None,
            "phase": ("teaching" if self._teaching_suspended and not waiting else
                      "durable_transition_blocked" if terminal is not None
                          and self.last_error is not None else
                      "waiting_for_save" if waiting else
                      "hold" if terminal is not None and status["hold_remaining_s"] > 0 else
                      "measurement_ended" if terminal is not None else "observing"),
            "producer": status,
            "durability": self.observation_publication_status()["durability"],
            "requested_policy": {
                "continuous": self.continuous,
                "trial_seconds": self.trial_length_s,
                "trial_seconds_applied": self.trial_length_s is not None and not self.continuous
                    and self.observation_config["window_source"] == "override",
            },
            "terminal": (None if terminal is None else {
                "observation_key": copy.deepcopy(terminal["observation_key"]),
                "payload_sha256": terminal["payload_sha256"],
                "captured_step": terminal["captured_step"],
                "durable": terminal.get("durable") is not None,
            }),
            "pending_control": (None if self._pending_assay_control is None else {
                "command_id": self._pending_assay_control['entry']['id'],
                "action": self._pending_assay_control['plan']['action'],
                "name": self._pending_assay_control['plan']['name'],
                "persistence_phase": self._pending_assay_control.get('phase', 'saving_prefix'),
            }),
            "control_failures": copy.deepcopy(list(self.assay_control_failures)),
        }

    def _observation_waiting_for_save(self) -> bool:
        if self._pending_assay_control is not None:
            return True
        terminal = self._observation_terminal
        if terminal is None or terminal.get("durable") is not None or self.exploratory:
            return False
        return self.arena.observation_owner.observation_status()["hold_remaining_s"] <= 0.0

    def _terminal_pose(self) -> Dict[str, Any]:
        fly = self.arena.fly
        return {"x_mm": float(fly.pos.x), "y_mm": float(fly.pos.y),
                "heading_rad": float(fly.heading), "step": self.total_steps}

    def _capture_terminal_observation(self, step_result: Dict[str, Any], end_reason=None) -> None:
        """Freeze and enqueue an ended producer once at its actual post-step pose."""
        if self._observation_terminal is not None:
            return
        owner = self.arena.observation_owner
        status = owner.observation_status()
        if status["state"] == "observing" and end_reason is None:
            return
        identity = self.identity()
        identity["brain_id"] = self.active_brain.brain_id
        envelope = build_observation_envelope(
            owner.freeze_observation(end_reason), identity=identity,
            provenance=self._observation_provenance(), segment_id=self.segment_id,
            segment_start_sim_s=self.observation_segment_start_sim_s,
            validity=self._observation_validity(), terminal_pose_post_step=self._terminal_pose())
        envelope = json.loads(dumps_observation_envelope(envelope, sort_keys=True))
        terminal = {
            "observation": envelope,
            "observation_key": {
                "daemon_run_id": identity["daemon_run_id"], "run_id": identity["run_id"],
                "instance_id": identity["instance_id"], "segment_id": self.segment_id,
                "presentation_id": envelope["presentation_id"],
            },
            "payload_sha256": None,
            "captured_step": self.total_steps,
            "step_result": copy.deepcopy(step_result),
            "durable": None,
        }
        # Retain this immutable candidate even when required publication is disabled.
        self._observation_terminal = terminal
        try:
            queued = self.enqueue_terminal_observation(envelope)
        except ObservationPublicationError:
            self._publish_due = True
            return
        terminal["observation_key"] = copy.deepcopy(queued["observation_key"])
        terminal["payload_sha256"] = queued["payload_sha256"]
        self._publish_due = True

    def _maybe_finish_observation_transition(self) -> bool:
        """Transition only after the hold and this exact terminal's durable receipt."""
        terminal = self._observation_terminal
        if terminal is not None and terminal.get("transition_owner") == "assay_control":
            return self._finish_assay_control()
        if terminal is None or terminal.get("durable") is None or self.last_error is not None:
            return False
        if self.arena.observation_owner.observation_status()["hold_remaining_s"] > 0.0:
            return False
        envelope = terminal["observation"]
        try:
            if envelope["mode"] == "presentation":
                self.arena.observation_owner.begin_next_presentation(
                    envelope["end_reason"], self.observation_config)
                self.transition = {"reason": "next_presentation", "step": self.total_steps,
                                   "ended_segment": self.segment_id,
                                   "terminal_pose": copy.deepcopy(envelope["terminal_pose_post_step"])}
                self._observation_terminal = None
            else:
                self._end_trial(terminal["step_result"], envelope["end_reason"])
                self._configure_observation_owner()
        except Exception as exc:  # a reset may have partially mutated physical state
            self._state_uncertain = True
            self._halt_on_error(
                exc, phase="trial bookkeeping", failure_class=FAILURE_SOFTWARE,
                incident_reason="observation_transition_failed",
                incident_extra={"observation_key": copy.deepcopy(terminal["observation_key"]),
                                "payload_sha256": terminal["payload_sha256"]})
            self._publish_due = True
            return False
        self._publish_due = True
        self._wake.set()
        return True

    def _discard_uncertain_state_for_rebuild(self) -> None:
        """Detach a failed loop's state without saving it; the rebuild reloads disk."""
        if not self._state_uncertain:
            return
        source = self.active_paradigm_id
        if self.registry is not None:
            active = self.registry.active
            if active is not None:
                active.release()
                self.registry.active = None
            self._graph_arenas.clear()
        self.brains.instances.pop(source, None)

    def _build_graph_arena(self, paradigm_name: str, seed: int):
        """World for a graph backend: motor assists default OFF (Arena, decision item 4),
        the modular mushroom body is ablated (it is not this run's controller)."""
        return Arena(paradigm=None if paradigm_name == "open-arena" else paradigm_name, brain_type="modular",
                     seed=int(seed), num_flies=1, num_predators=0, fly_ablations=[{"ablate_mb": True}],
                     controller_backend=self.backend, graph_controller=self.graph_controller)

    def compute_info(self) -> Dict[str, Any]:
        """Where the brain computes: the resolved backend and, on CUDA, the GPU name.

        Plain attribute reads plus a cached device-name query; safe without the lock.
        """
        if not self.graph_mode:
            return {"brain": "modular", "device": "cpu", "gpu": None,
                    "detail": "hand-built modular controller (Python/numpy on the CPU); "
                              "no connectome LIF brain is running"}
        from brainlab.brain import gpu_name, resolve_backend
        from brainlab.graph_identity import active_dynamics_version
        dynamics = active_dynamics_version()
        instance = getattr(self.registry, "active", None)
        brain = getattr(instance, "brain", None)
        device = getattr(brain, "backend", None)
        source = "active brain"
        if device == 'wgpu-amd' or self.brain_backend == 'wgpu-amd':
            cached = brain.compute_identity() if brain is not None else {}
            adapter = cached.get('adapter') or {}
            return dict(brain='connectome-lif', dynamics=getattr(brain, 'dynamics', dynamics),
                        device='wgpu-amd', gpu=adapter.get('device'), source=source if brain else 'no active brain',
                        adapter=adapter, capabilities=cached.get('capabilities', {}),
                        state=cached.get('state', {}), reason='explicit fixed-v3 compute; no fallback',
                        detail=f"LIF v3 on {adapter.get('backend_type') or 'unknown API'} "
                               f"({adapter.get('device') or 'unknown device'}); fixed weights, learning unsupported")
        if device is None:
            device = resolve_backend(dynamics)
            source = "resolved (no brain instance active yet)"
        note = getattr(brain, "backend_note", "") or ""
        if "failed" in note:
            reason = note
        else:
            from brainlab.gpu_probe import explain
            reason = explain(dynamics)[1]
        info = {"brain": "connectome-lif", "dynamics": dynamics, "device": device,
                "gpu": gpu_name() if device == "cuda" else None, "source": source, "reason": reason}
        info["detail"] = (f"LIF {dynamics} on CUDA ({info['gpu'] or 'unknown GPU'})" if device == "cuda"
                          else f"LIF {dynamics} on the CPU (numba reference kernel); GPU not used: {reason}")
        return info

    def identity(self) -> Dict[str, Any]:
        """Compact run identity carried by every packet, /api/status and each ack."""
        ident = self.manifest.identity() if self.manifest is not None else {}
        ident.update(activation=self.activation, daemon_run_id=self.run_id)
        return ident

    MOTOR_RECORD_FIELDS = ("step", "state", "controller_yaw", "controller_speed", "wall_reflex_yaw",
                           "controller_yaw_suppressed", "contact_turn_rad", "attempted_mm", "realized_mm",
                           "solver_correction_mm", "failsafe_correction_mm", "overlap_correction_mm",
                           "wall_gap_mm", "near_wall", "in_contact", "tethered", "halted",
                           "raw_motor_command", "applied_motor_command", "engineered_motor_primitive")

    def motor_summary(self) -> Dict[str, Any]:
        """Motor provenance: assists, controller source/fault and this step's motor record."""
        fly = self.arena.fly
        prov = self.arena.motor_provenance(fly)
        record = getattr(fly, "motor_record", None) or {}
        summary = {}
        for key in self.MOTOR_RECORD_FIELDS:
            value = record.get(key)
            if isinstance(value, float):
                value = round(value, 6) if math.isfinite(value) else None
            summary[key] = value
        summary["contacts"] = len(record.get("contact_normals") or [])
        prov["record"] = summary
        # WP5 provenance (empty for the modular controller): whether the graph run's
        # engineered assistance is on, which optomotor IO map it resolved, and the
        # decoded yaw or the named reason the mapping is unsupported.
        optomotor = self.arena.optomotor_provenance(fly)
        if optomotor:
            prov["optomotor"] = optomotor
        return prov

    def start(self):
        """Starts background continuous execution thread (and the watchdog, once)."""
        _install_thread_excepthook(self)
        self.running = True
        self.loop_failure = None
        self._steppable_since = time.perf_counter()
        self.sim_thread = threading.Thread(target=self._run_loop, daemon=True, name="NeuroFly-SimLoop")
        self.sim_thread.start()
        watchdog = getattr(self, "watchdog_thread", None)
        if watchdog is None or not watchdog.is_alive():
            self._watchdog_stop.clear()
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True,
                                                    name="NeuroFly-Watchdog")
            self.watchdog_thread.start()
        print(f"[Daemon] Simulation loop started (Speed: {self.sim_speed}x, Paradigm: {self.active_paradigm_id}).", flush=True)

    def _cancel_shutdown_transaction(self, transaction, exc):
        """Revoke immediately; publish owned failure without waiting for a busy lock."""
        if transaction is None or transaction['entry']['done'].is_set():
            return
        transaction['cancel_requested'] = True
        transaction.setdefault('cancellation_error', exc)
        def fail_locked():
            if self._pending_assay_control is transaction:
                self._fail_assay_control(transaction['cancellation_error'],
                                         before_apply=not transaction.get('applying'))
        if self.lock.acquire(blocking=False):
            try:
                fail_locked()
            finally:
                self.lock.release()
        elif transaction.get('cancellation_writer') is None:
            def fail_owned():
                with self.lock:
                    fail_locked()
            try:
                writer = threading.Thread(target=fail_owned,
                    name='NeuroFly-ShutdownCancellation', daemon=True)
                writer.start()
            except Exception as start_exc:
                # Retain revoked ownership when even failure publication cannot start.
                # The next lock-owned timeout check completes the original failure.
                transaction['cancellation_start_error'] = str(start_exc)
            else:
                transaction['cancellation_writer'] = writer

    def _start_shutdown_drain(self, transaction, drain):
        """Own a non-running recorder drain off lock without blocking stop's deadline."""
        if transaction.get('shutdown_drain') is not None or transaction.get('cancel_requested'):
            return
        def save_prefixes():
            while not transaction['entry']['done'].is_set() and not transaction.get('cancel_requested'):
                try:
                    drain.poll_once()
                except Exception as exc:
                    self._cancel_shutdown_transaction(transaction, exc)
                    return
                transaction['entry']['done'].wait(0.01)
        try:
            worker = threading.Thread(target=save_prefixes,
                name='NeuroFly-ShutdownDrain', daemon=True)
            worker.start()
        except Exception as exc:
            self._cancel_shutdown_transaction(transaction, exc)
        else:
            # Only a successfully started thread belongs in a joinable slot.
            transaction['shutdown_drain'] = worker

    def stop(self) -> bool:
        """Bounded shutdown transaction; failed saves retain evidence and return False."""
        with self._stop_lock:
            if self._stopped:
                return self._stop_ok
            self._stopped, self._stop_ok = True, False
        deadline = time.monotonic() + self.assay_control_timeout_s
        def cancel(transaction):
            self._cancel_shutdown_transaction(transaction, TimeoutError('Shutdown wait expired'))

        if not self.lock.acquire(timeout=min(10.0, max(0.0, self.assay_control_timeout_s))):
            cancel(self._pending_assay_control)
            self.running = False
            self._wake.set()
            self._watchdog_stop.set()
            self._request_scheduled_records_cleanup()
            return False
        try:
            if self._pending_assay_control is not None:
                # Do not supersede a pending prefix or its writer with shutdown.
                result, transaction = None, self._pending_assay_control
            else:
                result = self._apply_command({'action': 'shutdown'})
                transaction = self._pending_assay_control
        finally:
            self.lock.release()
        def acquire_before_deadline():
            return self.lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

        def await_transaction(transaction):
            while transaction is not None and not transaction['entry']['done'].is_set():
                if transaction.get('cancel_requested'):
                    break
                if time.monotonic() >= deadline:
                    cancel(transaction)
                    break
                drain = self.learning_records
                if drain is not None and not drain.liveness()['started']:
                    self._start_shutdown_drain(transaction, drain)
                transaction['entry']['done'].wait(min(0.01, max(0.0, deadline - time.monotonic())))
            return transaction['entry']['result'] if transaction is not None else None

        result = await_transaction(transaction) if transaction is not None else result
        # Shutdown intent gates sampling before a competing command releases its barrier.
        if (transaction is not None and transaction['plan']['action'] != 'shutdown'
                and not transaction.get('cancel_requested')):
            if acquire_before_deadline():
                try:
                    if time.monotonic() < deadline:
                        result = self._apply_command({'action': 'shutdown'})
                        transaction = self._pending_assay_control
                    else:
                        result = None
                finally:
                    self.lock.release()
                if transaction is not None and transaction['plan']['action'] == 'shutdown':
                    result = await_transaction(transaction)
            else:
                result = None
        self.running = False
        self._wake.set()
        self._watchdog_stop.set()
        if transaction is not None:
            for key in ('writer', 'failure_writer', 'shutdown_drain', 'cancellation_writer'):
                writer = transaction.get(key)
                if (writer is not None and writer is not threading.current_thread()
                        and getattr(writer, "ident", None) is not None):
                    writer.join(timeout=min(3.0, max(0.0, deadline - time.monotonic())))
        thread = getattr(self, 'sim_thread', None)
        if (thread is not None and thread is not threading.current_thread()
                and getattr(thread, "ident", None) is not None):
            thread.join(timeout=min(3.0, max(0.0, deadline - time.monotonic())))
        cleanup = self._request_scheduled_records_cleanup()
        if cleanup is not None and getattr(cleanup, 'ident', None) is not None:
            cleanup.join(timeout=max(0.0, deadline - time.monotonic()))
        cleanup_ok = (self._scheduled_records_owner is None or
                      (self._scheduled_records_cleanup_ok and cleanup is not None and not cleanup.is_alive()))
        self._stop_ok = bool(result and result.get('ack', {}).get('applied')
                             and (thread is None or not thread.is_alive()) and cleanup_ok)
        return self._stop_ok

    # ------------------------------------------------------------------ scheduling
    def _can_step(self) -> bool:
        if self._stopped or self.paused or self.last_error or self._observation_waiting_for_save():
            return False
        return self.stop_at_step is None or self.total_steps < self.stop_at_step

    def _loop_active(self) -> bool:
        thread = getattr(self, "sim_thread", None)
        return bool(self.running and thread is not None and thread.is_alive()
                    and threading.current_thread() is not thread)

    def _yield_lock(self, max_wait_s: float = 0.005):
        """Give other threads the GIL and the lock before the next batch.

        Measured (outputs/rethink-audit/lock_profile_receipt.json): a CPU-bound step
        loop that never blocks starves every other Python thread of the GIL -- a
        33 ms ``time.sleep`` in another thread took ~570 ms at 100x.  A real sleep
        hands the GIL over; it costs about ``yield_wall_s / max_batch_wall_s`` of
        throughput and bounds delivery and command latency.
        """
        if self.yield_wall_s is not None:   # None: no hand-over (diagnostics only)
            time.sleep(self.yield_wall_s)
        deadline = time.perf_counter() + max_wait_s
        while self.lock.waiting and time.perf_counter() < deadline:
            time.sleep(0.0002)

    def _run_loop(self):
        """Deadline-aware fixed-dt scheduler.

        Every step integrates exactly ``self.dt`` simulated seconds.  Wall-clock
        deadlines only decide when the next step runs: step ``n`` after the anchor is
        due at ``anchor + n*dt/speed``.  If the machine cannot keep up, the schedule
        debt is capped at ``max_lag_wall_s`` and the remainder forgiven (counted in
        ``sched_stats``), so overload lowers the *achieved* speed visibly instead of
        causing catch-up bursts.  Steps run in batches of at most ``max_batch_wall_s``
        under the lock; between batches the lock is yielded to waiting threads,
        queued commands are applied at a step boundary and a snapshot is published.
        """
        try:
            self._schedule_loop()
        except BaseException as exc:
            # Not an Exception (those are handled per iteration): SystemExit,
            # KeyboardInterrupt or a deliberate kill.  Record it before the thread ends.
            self._record_loop_failure(exc)
            raise
        finally:
            if self.running and self.loop_failure is None:
                self._record_loop_failure(None)
            self._close_scheduled_records()

    def _publish_frame(self) -> None:
        self._loop_phase = "publish"
        self._publish_snapshot()
        self._publish_failures = 0

    def _schedule_loop(self):
        dt = self.dt
        anchor_wall = time.perf_counter()
        anchor_steps = self.total_steps
        anchor_speed = self.sim_speed
        last_publish = 0.0
        while self.running:
            # F1: one failing iteration (a step, a publish, a write, a command) becomes
            # an honest halt naming the phase; it never ends this thread.
            try:
                self._wake.clear()
                self._loop_phase = "command"
                self._drain_commands()
                now = time.perf_counter()
                if not self._can_step():
                    # Not expected to advance (paused, halted, held): restart the stall clock
                    # and the achieved-speed window.
                    self._steppable_since = now
                    if self._advance_samples:
                        self._advance_samples.clear()
                    if self._publish_due or now - last_publish >= 1.0 / self.paused_publish_hz:
                        last_publish = now
                        self._publish_frame()
                    self._loop_phase = "idle"
                    self._wake.wait(0.05)
                    anchor_wall, anchor_steps, anchor_speed = time.perf_counter(), self.total_steps, self.sim_speed
                    continue
                speed = self.sim_speed
                if speed != anchor_speed:
                    anchor_wall, anchor_steps, anchor_speed = now, self.total_steps, speed
                behind = anchor_steps + int((now - anchor_wall) * speed / dt) - self.total_steps
                if behind <= 0:
                    fresh = self.sched_stats["steps_since_publish"] > 0
                    if self._publish_due or (now - last_publish >= 1.0 / self.publish_hz
                                             and (fresh or now - last_publish >= 0.5)):
                        last_publish = now
                        self._publish_frame()
                    next_deadline = anchor_wall + (self.total_steps + 1 - anchor_steps) * dt / speed
                    wait = min(next_deadline, last_publish + 1.0 / self.publish_hz) - time.perf_counter()
                    self._loop_phase = "idle"
                    if wait > 0:
                        self._wake.wait(min(wait, 0.05))
                    continue
                max_lag = max(1, int(self.max_lag_wall_s * speed / dt))
                if behind > max_lag:
                    # Overload: forgive the debt instead of bursting to catch up.
                    self.sched_stats["rebases"] += 1
                    self.sched_stats["forgiven_wall_s"] += (behind - 1) * dt / speed
                    anchor_wall, anchor_steps = now - dt / speed, self.total_steps
                    behind = 1
                batch_start = time.perf_counter()
                with self.lock:
                    taken = 0
                    while taken < behind and self._can_step() and self.running:
                        self._advance_one()
                        taken += 1
                        if time.perf_counter() - batch_start >= self.max_batch_wall_s:
                            break
                hold_ms = (time.perf_counter() - batch_start) * 1e3
                self.sched_stats["batches"] += 1
                self.sched_stats["max_batch_hold_ms"] = max(self.sched_stats["max_batch_hold_ms"], round(hold_ms, 3))
                self._loop_phase = "idle"
                self._yield_lock()
                now = time.perf_counter()
                if self._publish_due or now - last_publish >= 1.0 / self.publish_hz:
                    last_publish = now
                    self._publish_frame()
            except Exception as exc:  # noqa: BLE001 -- see F1 above
                self._on_loop_exception(exc)
                anchor_wall, anchor_steps, anchor_speed = time.perf_counter(), self.total_steps, self.sim_speed

    PUBLISH_RETRIES = 3   # consecutive failed publications before an honest halt

    def _on_loop_exception(self, exc: BaseException) -> None:
        """Turn an exception that escaped one loop iteration into an honest halt."""
        phase = self._loop_phase or "loop"
        self._step_started = None
        now = time.monotonic()
        if now - self._loop_fault_log >= 5.0:
            self._loop_fault_log = now
            print(f"[Daemon] Simulation loop error during {phase}: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        if phase == "publish":
            # A frame that could not be assembled: retry a few times (it may depend on
            # a transient state), then halt instead of silently freezing the display.
            self._publish_failures += 1
            if self._publish_failures < self.PUBLISH_RETRIES and self.last_error is None:
                self._wake.wait(0.05)
                return
        if self.lock.acquire(timeout=5.0):
            try:
                self._halt_on_error(exc, phase=phase)
            finally:
                self.lock.release()
        else:   # cannot happen unless another thread holds the lock forever; still report
            self._halt_on_error(exc, phase=phase)
        self._loop_phase = "idle"
        self._wake.wait(0.2)

    def _record_loop_failure(self, exc: Optional[BaseException]) -> None:
        if self.loop_failure is not None:
            return
        failure = exc or LoopExit("the simulation loop returned while it should run")
        phase = self._loop_phase or "loop"
        self.loop_failure = {"type": type(failure).__name__, "message": str(failure) or type(failure).__name__,
                             "phase": phase, "at": round(time.time(), 3),
                             "traceback": _traceback_tail(failure) if exc is not None else []}
        self._state_uncertain = True
        source = self.identity()
        source["brain_id"] = self.active_brain.brain_id
        source_segment = getattr(self, "segment_id", None)
        # This runs after the scheduler has left its step context, so the simulation
        # lock is no longer held.  Bound the wait: a failure reporter must never join
        # an indefinitely held lock.
        acquired = self.lock.acquire(timeout=1.0)
        try:
            incident_extra = {"source_run_id": source.get("run_id"),
                              "source_instance_id": source.get("instance_id"),
                              "source_activation": source.get("activation"),
                              "source_segment_id": getattr(self, "segment_id", None),
                              "failure_phase": phase}
            if self.last_error is None:
                self._halt_on_error(failure, phase=phase, incident_reason="unexpected_loop_exit",
                                    incident_extra=incident_extra)
            else:
                self._invalidate("unexpected_loop_exit", exc=failure,
                                 failure_class=classify_failure(failure), extra=incident_extra)
            # Do not reassemble telemetry or ask the potentially damaged engine for
            # another observation. Preserve its last frame and deliver only this
            # owner-qualified run-policy downgrade, independently of the scheduler.
            self._observation_validity_update = {
                "schema": "neurofly-observation-validity-update/1",
                "identity": copy.deepcopy(source), "segment_id": source_segment,
                "validity": "exploratory_degraded" if self.exploratory else "invalidated",
                "reason": "unexpected_loop_exit",
            }
        finally:
            if acquired:
                self.lock.release()
        print(f"[Daemon] SIMULATION THREAD STOPPED during {self.loop_failure['phase']}: "
              f"{self.loop_failure['type']}: {self.loop_failure['message']}", file=sys.stderr, flush=True)

    def _note_thread_failure(self, name, exc_type, exc_value, exc_tb) -> None:
        """threading.excepthook backstop: an uncaught exception in a NeuroFly thread."""
        entry = {"thread": name, "type": getattr(exc_type, "__name__", str(exc_type)),
                 "message": str(exc_value), "at": round(time.time(), 3)}
        self.thread_failures.append(entry)
        if name == "NeuroFly-SimLoop" and self.loop_failure is None:
            self._record_loop_failure(exc_value)

    def _loop_dead(self) -> bool:
        """The loop should be running (``running``) but its thread has ended."""
        thread = getattr(self, "sim_thread", None)
        return bool(self.running and thread is not None and not thread.is_alive())

    def _advance_one(self):
        """One scheduled step: step-indexed commands first, then the fixed-dt tick. Holds lock."""
        self._check_assay_control_timeout()
        if not self._can_step():
            return {}
        due = self._scheduled.get(self.total_steps)
        while due:
            entry = due.pop(0)
            self._apply_command_entry_locked(entry)
            if not due:
                self._scheduled.pop(self.total_steps, None)
            if not entry['done'].is_set() or self._pending_assay_control is not None:
                return {}
            if entry['result'].get('status') != 'ok':
                self._halt_on_error(RuntimeError(entry['result'].get('message', 'Scheduled command refused')),
                                    phase='scheduled_command')
                return {}
        if not self._can_step():
            return {}
        before = self.total_steps
        started = self._step_started = time.perf_counter()
        try:
            result = self.step_once(publish=False)
            self.sched_stats["steps_since_publish"] += 1
            if self.step_hook is not None:
                self.step_hook(self)
        finally:
            self._step_started = None
            end = time.perf_counter()
            self.last_step_wall_s = end - started
            if self.total_steps != before:
                self.last_advance_wall = end
                samples = self._advance_samples
                samples.append((end, self.total_steps))
                while samples and end - samples[0][0] > 30.0:
                    samples.popleft()
        return result

    # ------------------------------------------------------------------ watchdog (F3)
    def stall_threshold_s(self) -> float:
        """No step for longer than this, while stepping is expected, is a stall."""
        return max(self.min_stall_s, 20.0 * self.dt / max(self.sim_speed, 1e-6), 3.0 * self.last_step_wall_s)

    def _work_in_progress_s(self) -> float:
        started = self._command_started
        command = 0.0 if started is None else time.perf_counter() - started
        return max(self.step_in_progress_s(), command)

    def liveness(self) -> Dict[str, Any]:
        """Is the simulation advancing?  advancing / paused / halted / slow / stalled / dead."""
        now = time.perf_counter()
        thread = getattr(self, "sim_thread", None)
        alive = bool(thread is not None and thread.is_alive())
        in_progress = self._work_in_progress_s()
        last = self.last_advance_wall
        age = None if last is None else now - last
        threshold = self.stall_threshold_s()
        expected_since = max(last or 0.0, self._steppable_since)
        if thread is None:
            state = "not_started"
        elif self.running and not alive:
            state = "dead"
        elif not self.running:
            state = "stopped"
        elif self.last_error is not None:
            state = "halted"
        elif self._observation_waiting_for_save():
            state = "waiting_for_save"
        elif not self._can_step():
            state = "paused"
        elif in_progress > self.step_hard_limit_s:
            state = "stalled"
        elif in_progress > 2.0:
            state = "slow"
        elif in_progress == 0.0 and now - expected_since > threshold:
            state = "stalled"
        else:
            state = "advancing"
        return {"state": state, "step": self.total_steps,
                "last_advance_age_s": None if age is None else round(age, 2),
                "sim_thread_alive": alive, "step_in_progress_s": round(in_progress, 3),
                "stall_threshold_s": round(threshold, 2), "step_hard_limit_s": self.step_hard_limit_s}

    def health(self) -> Dict[str, Any]:
        """The one honest status: error (halted, stalled, dead) > degraded > online."""
        live = self.liveness()
        error, detail = self._fault_snapshot
        detail = copy.deepcopy(detail)
        halted = error is not None
        if live["state"] == "dead":
            failure = self.loop_failure or {}
            reason = f"{failure.get('type', 'unknown')}: {failure.get('message', '')}".strip(": ")
            error = f"simulation thread stopped: {reason or 'unknown reason'}"
            halted = True
        elif live["state"] == "stalled" and error is None:
            if live["step_in_progress_s"] > self.step_hard_limit_s:
                error = (f"simulation not advancing: one step has run for {live['step_in_progress_s']:.0f} s "
                         f"(limit {self.step_hard_limit_s:.0f} s)")
            else:
                error = (f"simulation not advancing: no step for {live['last_advance_age_s'] or 0:.0f} s "
                         f"while running (limit {live['stall_threshold_s']:.0f} s)")
        persistence = self.persistence_summary()
        if live["state"] in ("dead", "stalled") or halted:
            status = "error"
        elif not persistence["ok"]:
            status = "degraded"
        else:
            status = "online"
        return {"status": status, "error": error, "error_detail": detail, "halted": halted, "liveness": live,
                "mode": "exploratory" if self.exploratory else "scientific",
                "result_validity": self.result_validity(),
                "observation_validity_update": copy.deepcopy(getattr(self, "_observation_validity_update", None)),
                "persistence": persistence, "recording_error": self.recording_error,
                "loop_failure": self.loop_failure, "thread_failures": list(self.thread_failures)}

    def persistence_summary(self) -> Dict[str, Any]:
        out = self.persistence.describe()
        out["mode"] = "exploratory" if self.exploratory else "scientific"
        return out

    def records_failed(self, channel: str, exc: BaseException) -> None:
        """Callback from the learning-records thread (trials.jsonl is required,
        telemetry_summary.jsonl diagnostic).  Applies the same policy as every save."""
        if self.lock.acquire(timeout=10.0):
            try:
                self._persistence_failed(channel, exc, None)
            finally:
                self.lock.release()
        else:
            self._persistence_failed(channel, exc, None)

    def records_ok(self, channel: str) -> None:
        if self.lock.acquire(timeout=10.0):
            try:
                self._records_ok_locked(channel)
            finally:
                self.lock.release()
        else:
            # A success notification may wait for the next drain cycle. Never
            # inspect queue ownership or clear a failure without the runner lock.
            return

    def _records_ok_locked(self, channel: str) -> None:
        if channel == "learning_records":
            pending = self.observation_publication.pending_status()
            if any(entry["phase"] in ("claimed", "failed") for entry in pending["entries"]):
                return
        if channel in self.persistence.channels:
            self._persisted(channel)

    # ------------------------------------------------ observation publication (metric-contract/1.2)
    def attach_learning_records(self, recorder_thread: RecorderThread) -> None:
        """Enable durable observation publication. Caller holds ``self.lock``."""
        if recorder_thread is None or getattr(recorder_thread, "recorder", None) is None:
            raise ValueError("a live RecorderThread with a LearningRecorder is required")
        self.learning_records = recorder_thread
        self.observation_durability_reason = None
        self._publish_due = True

    def _learning_recorder_problem(self):
        """Return the current recorder liveness failure without doing I/O or mutation."""
        recorder = self.learning_records
        if recorder is None:
            return None, None
        state = recorder.liveness()
        if not state["started"]:
            return None, state             # direct/manual poll fixtures are not background owners
        if not state["alive"] and state["start_age_s"] >= self.recorder_dead_grace_s:
            return TimeoutError(
                f"durable recorder thread exited (grace {self.recorder_dead_grace_s:g}s)"), state
        if (state["alive"] and not state["run_entered"]
                and state["start_age_s"] >= self.recorder_progress_grace_s):
            return TimeoutError(
                f"durable recorder did not enter its poll loop within "
                f"{self.recorder_progress_grace_s:g}s"), state
        if (state["active_write_age_s"] is not None
                and state["active_write_age_s"] >= self.recorder_stuck_s):
            return TimeoutError(
                f"durable recorder {state['active_write']} write has not completed for "
                f"{state['active_write_age_s']:.3f}s (limit {self.recorder_stuck_s:g}s)"), state
        progress_limit = self.learning_records.poll_interval + self.recorder_progress_grace_s
        if (state["run_entered"] and state["active_write"] is None
                and state["last_progress_age_s"] is not None
                and state["last_progress_age_s"] >= progress_limit
                and self._work_in_progress_s() == 0.0):
            return TimeoutError(
                f"durable recorder poll made no progress for {state['last_progress_age_s']:.3f}s "
                f"(limit poll interval + grace = {progress_limit:g}s)"), state
        return None, state

    def _learning_recorder_verified_healthy(self, state=None) -> bool:
        if state is None:
            _, state = self._learning_recorder_problem()
        if not state:
            return False
        progress_limit = self.learning_records.poll_interval + self.recorder_progress_grace_s
        return bool(state["started"] and state["run_entered"] and state["alive"]
                    and not state["stop_requested"] and state["active_write"] is None
                    and state["last_progress_age_s"] is not None
                    and state["last_progress_age_s"] < progress_limit)

    def _resolve_recorder_watchdog_locked(self, recovered_by: str, state=None) -> bool:
        failure = self._recorder_watchdog_failure
        if failure is None or not failure.get("active", True):
            return False
        if not self._learning_recorder_verified_healthy(state):
            return False
        failure.update(active=False, recovered_at=round(time.time(), 3), recovered_by=recovered_by)
        self._publish_due = True
        return True

    def _check_learning_recorder_locked(self, problem=None) -> None:
        """Fail required persistence when a started recorder dies or its I/O sticks."""
        current = self._recorder_watchdog_failure
        if not self.running or (current is not None and current.get("active", True)):
            return
        exc, state = problem if problem is not None else self._learning_recorder_problem()
        if exc is None:
            return
        self._recorder_watchdog_episode += 1
        failure = {
            "error": f"{type(exc).__name__}: {exc}", "at": round(time.time(), 3),
            "recorder": state, "active": True,
            "episode": self._recorder_watchdog_episode,
        }
        self._recorder_watchdog_failure = failure
        self._recorder_watchdog_failures.append(failure)
        # This records and halts scientific mode. It deliberately does not stop or
        # close the writer: an in-flight fsync keeps its ownership until it returns.
        self._persistence_failed("learning_records", exc)
        self._publish_due = True

    def enqueue_terminal_observation(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Retain one validated frozen terminal. Caller holds ``self.lock``."""
        if self.learning_records is None:
            exc = ObservationPublicationError(
                "durable observation publication is disabled; required terminal was not queued")
            self._persistence_failed("learning_records", exc)
            raise exc
        try:
            result = self.observation_publication.enqueue(envelope)
        except ObservationQueueFull as exc:
            self._persistence_failed("learning_records", exc)
            raise
        self._publish_due = True
        return result

    def claim_terminal_observation(self) -> Optional[Dict[str, Any]]:
        """Claim or retry the oldest entry. Caller holds ``self.lock``."""
        status = self.observation_publication.pending_status(limit=1)
        if status["entries"] and status["entries"][0]["phase"] == "failed":
            if not self.persistence.retry_due("learning_records"):
                return None
            self.observation_publication.retry_failed()
        claim = self.observation_publication.claim_oldest()
        if claim is not None:
            self._publish_due = True
        return claim

    def terminal_observation_failed(self, token: str, exc: BaseException) -> None:
        """Retain a failed attempt before applying required-save policy. Caller locked."""
        try:
            self.observation_publication.mark_failed(token, {
                "error_type": type(exc).__name__, "message": str(exc),
                "errno": getattr(exc, "errno", None), "at": round(time.time(), 3),
            })
        except Exception as queue_exc:  # preserve the original visible failure and pending work
            print(f"[Recorder] observation failure metadata could not be attached: {queue_exc}",
                  file=sys.stderr, flush=True)
        self._persistence_failed("learning_records", exc)
        if self._pending_assay_control is not None:
            self._fail_assay_control(exc, before_apply=True)
        self._publish_due = True

    def terminal_observation_succeeded(self, token: str, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """Publish only the matching durable receipt. Caller holds ``self.lock``."""
        durable = self.observation_publication.acknowledge(token, receipt)
        self._persisted("learning_records")
        terminal = self._observation_terminal
        if (terminal is not None
                and durable["observation_key"] == terminal["observation_key"]
                and durable["payload_sha256"] == terminal["payload_sha256"]):
            terminal["durable"] = copy.deepcopy(durable)
            self._maybe_finish_observation_transition()
        self._publish_due = True
        return durable

    def observation_publication_status(self) -> Dict[str, Any]:
        """Detached queue and exact-current-owner durable result. Caller locked."""
        identity = self.identity()
        pending = self.observation_publication.pending_status()
        last_terminal = self.observation_publication.last_terminal(
            assay=self.active_paradigm_id,
            backend=self.backend,
            instance_id=identity.get("instance_id") or self.active_brain.brain_id,
            brain_id=self.active_brain.brain_id,
        )
        return {
            "durability": {
                "enabled": self.learning_records is not None,
                "state": "enabled" if self.learning_records is not None else "disabled",
                "reason": self.observation_durability_reason,
                "recorder": (self.learning_records.liveness()
                             if self.learning_records is not None else None),
                "watchdog_failure": copy.deepcopy(self._recorder_watchdog_failure),
                "watchdog_failures": copy.deepcopy(list(self._recorder_watchdog_failures)),
            },
            "pending": pending,
            "last_terminal": last_terminal,
        }

    def _watchdog_loop(self) -> None:
        """NeuroFly-Watchdog: logs stall onset; optionally exits so systemd restarts."""
        previous, bad_since = None, None
        while not self._watchdog_stop.wait(self.watchdog_interval_s):
            try:
                if not self.running:
                    break                       # stopped; start() launches a new watchdog
                recorder_problem = self._learning_recorder_problem()
                current_failure = self._recorder_watchdog_failure
                recorder_action = (recorder_problem[0] is not None
                                   or (self.exploratory and current_failure is not None
                                       and current_failure.get("active", True)
                                       and self._learning_recorder_verified_healthy(recorder_problem[1])))
                if recorder_action and self.lock.acquire(blocking=False):
                    try:
                        if recorder_problem[0] is not None:
                            self._check_learning_recorder_locked(recorder_problem)
                        else:
                            self._resolve_recorder_watchdog_locked(
                                "recorder_progress_restored", recorder_problem[1])
                    finally:
                        self.lock.release()
                # Recorder reporting must never blind the independent simulation
                # liveness/exit watchdog when a step or command owns runner.lock.
                live = self.liveness()
                state = live["state"]
                if state == "dead" and self.loop_failure is None:
                    self._record_loop_failure(None)
                if state in ("stalled", "dead"):
                    if previous not in ("stalled", "dead"):
                        bad_since = time.monotonic()
                        print(f"[Watchdog] Simulation NOT ADVANCING ({state}) at step {live['step']}: "
                              f"last step {live['last_advance_age_s']} s ago, step in progress "
                              f"{live['step_in_progress_s']} s, sim thread alive={live['sim_thread_alive']}. "
                              f"{self.health()['error']}", file=sys.stderr, flush=True)
                    if (self.exit_on_stall_s is not None and bad_since is not None
                            and time.monotonic() - bad_since >= self.exit_on_stall_s):
                        print(f"[Watchdog] Exiting with code 70 after {self.exit_on_stall_s:g} s not advancing "
                              f"(--exit-on-stall); a service manager restarts from the last checkpoint.",
                              file=sys.stderr, flush=True)
                        self._exit(70)
                elif previous in ("stalled", "dead"):
                    print(f"[Watchdog] Simulation advancing again ({state}) at step {live['step']}.", flush=True)
                    bad_since = None
                previous = state
            except Exception as exc:  # noqa: BLE001 -- the watchdog must outlive what it watches
                print(f"[Watchdog] check failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def _ensure_scheduled_records(self):
        """Standalone explicit-output API owns a local writer; configured daemons opt out.

        Called before taking runner.lock. Never replace or close an attached writer.
        """
        if self.learning_records is not None or not self.standalone_scheduled_records:
            return
        with self._scheduled_records_lock:
            if self.learning_records is not None:
                return
            recorder = None
            try:
                recorder = LearningRecorder(self.output_dir / 'scheduled-records' / self.run_id,
                    session={'daemon_run_id': self.run_id, 'backend': self.backend,
                             'source': 'standalone_scheduled_api'})
                worker = RecorderThread(self, recorder, poll_interval=0.05)
                with self.lock:
                    if self._stopped:
                        raise RuntimeError('Cannot initialize scheduled records after stop')
                    if self.learning_records is not None:
                        attached = False
                    else:
                        self.attach_learning_records(worker)
                        self._scheduled_records_owner = worker
                        attached = True
                if not attached:
                    recorder.close()
                    return
                worker.start()
                return worker
            except Exception:
                with self.lock:
                    if self.learning_records is self._scheduled_records_owner:
                        self.learning_records = self._scheduled_records_owner = None
                        self.observation_durability_reason = 'scheduled recorder startup failed'
                if recorder is not None:
                    recorder.close()
                raise

    def _close_scheduled_records(self):
        """Only the locally owned writer; callers must be outside runner.lock."""
        with self._scheduled_records_lock:
            owner = self._scheduled_records_owner
            if owner is not None:
                self._scheduled_records_cleanup_ok = owner.stop(timeout=self.assay_control_timeout_s)
                return self._scheduled_records_cleanup_ok
            return True

    def _request_scheduled_records_cleanup(self):
        """Bounded stop only joins this worker; final writer flush is off caller lock."""
        if self._scheduled_records_owner is None:
            return None
        with self._scheduled_records_cleanup_guard:
            if self._scheduled_records_cleanup_thread is not None:
                return self._scheduled_records_cleanup_thread
            worker = threading.Thread(target=self._close_scheduled_records,
                name='NeuroFly-ScheduledRecorderCleanup', daemon=True)
            self._scheduled_records_cleanup_thread = worker
        try:
            worker.start()
        except Exception:
            self._scheduled_records_cleanup_ok = False
            return None
        return worker

    def _validate_scheduled_input_locked(self, step, cmd):
        if type(step) is not int or step < 0:
            raise ValueError('scheduled step must be a nonnegative integer')
        if type(cmd) is not dict or type(cmd.get('action')) is not str:
            raise ValueError('scheduled command must be a dictionary with an action string')
        if self._stopped:
            raise RuntimeError('Cannot schedule commands after stop')
        if self._scheduled_records_retiring:
            raise RuntimeError('Unused scheduled recorder cleanup is pending; retry afterward')
        if step < self.total_steps:
            raise ValueError(f"step {step} is already in the past (now {self.total_steps})")

    def _retire_unused_scheduled_records(self, owner):
        """Race rejection closes only this call's unused new writer, outside runner.lock."""
        if owner is None:
            return
        with self._scheduled_records_lock:
            with self.lock:
                if (self._scheduled_records_owner is not owner or any(self._scheduled.values())
                        or self._pending_assay_control is not None):
                    return
                self._scheduled_records_retiring = True
            try:
                if not owner.stop(timeout=self.assay_control_timeout_s):
                    raise RuntimeError('Rejected schedule recorder cleanup did not complete')
                with self.lock:
                    if self.learning_records is owner:
                        self.learning_records = None
                        self.observation_durability_reason = 'no scheduled command was admitted'
                    if self._scheduled_records_owner is owner:
                        self._scheduled_records_owner = None
            finally:
                with self.lock:
                    self._scheduled_records_retiring = False

    def schedule_command(self, step: int, cmd: dict) -> dict:
        """Apply at the exact step boundary; result is final only when done is set.

        Standalone callers with explicit output_dir own a durable local writer unless
        standalone_scheduled_records=False. Daemon configuration always opts out of
        automatic ownership and uses its already-attached writer or explicit refusal.
        """
        with self.lock:
            self._validate_scheduled_input_locked(step, cmd)
        new_owner = self._ensure_scheduled_records()
        try:
            with self.lock:
                self._validate_scheduled_input_locked(step, cmd)
                with self._commands_lock:
                    self._command_seq += 1
                    entry = {'cmd': copy.deepcopy(cmd), 'done': threading.Event(), 'result': None,
                             'id': f'{self.run_id[:8]}-{self._command_seq}', 'received': time.perf_counter(),
                             'scheduled_step': step}
                self._scheduled.setdefault(step, []).append(entry)
        except Exception:
            self._retire_unused_scheduled_records(new_owner)
            raise
        return entry

    def _apply_command_entry_locked(self, entry):
        """One command entry for HTTP and scheduled callers; final ACK owns done."""
        self._command_started = time.perf_counter()
        self._executing_command_entry = entry
        try:
            result = self._apply_command(entry['cmd'])
            if ('scheduled_step' in entry and result.get('status') == 'error'
                    and self._pending_assay_control is None and self.last_error is None):
                # A rejected input is evidence, never an applied biological command.
                refusal = {'status': 'error', 'applied': False, 'message': result.get('message', '')}
                plan = {'action': 'scheduled_refusal', 'name': entry['cmd'].get('action', ''),
                        'closes_presentation': False, 'refused_result': refusal}
                result = self._begin_control_transaction(entry['cmd'], plan)
        except Exception as exc:
            result = {'status': 'error', 'message': f'{type(exc).__name__}: {exc}'}
        finally:
            self._command_started = None
            self._executing_command_entry = None
            self._steppable_since = time.perf_counter()
        pending = self._pending_assay_control
        deferred = pending is not None and pending['entry'] is entry
        if not deferred and not entry['done'].is_set():
            entry['result'] = result
            self._note_latency(result, entry['received'])
            if isinstance(result, dict):
                result['command_id'] = entry['id']
                self.command_acks.append(result)
            entry['done'].set()
        return result

    def _drain_commands(self):
        """Apply HTTP commands queued since the last batch, at a step boundary."""
        with self.lock:
            self._check_assay_control_timeout()
        while True:
            with self._commands_lock:
                if not self._commands:
                    return
                entry = self._commands.popleft()
            with self.lock:
                self._apply_command_entry_locked(entry)

    # ------------------------------------------------------------------ publication
    def step_in_progress_s(self) -> float:
        """Wall seconds the current step has been running (0 between steps)."""
        started = self._step_started
        return 0.0 if started is None else round(time.perf_counter() - started, 3)

    def achieved_speed_now(self) -> float:
        """Achieved speed computed at read time from (wall, step) samples (F3).

        The window ends NOW, so a simulation that stopped advancing reads 0 within a
        few seconds instead of repeating the last value a dead thread stored.
        """
        if not self._can_step():
            return 0.0
        thread = getattr(self, "sim_thread", None)
        if thread is not None and not thread.is_alive():
            return 0.0
        samples = list(self._advance_samples)
        if not samples:
            return 0.0
        now = time.perf_counter()
        window = max(2.0, 3.0 * self.last_step_wall_s)
        t_last, s_last = samples[-1]
        if now - t_last > window:
            return 0.0
        # The newest sample at or before the window start anchors a full interval.
        idx = next((i for i, (t, _) in enumerate(samples) if now - t <= window), len(samples) - 1)
        t0, s0 = samples[max(0, idx - 1)]
        if s_last <= s0 or now - t0 <= 0.0:
            return 0.0
        # Steps completed since the anchor sample, over the wall time up to now.
        return round((s_last - s0) * self.dt / (now - t0), 3)

    def timing_snapshot(self) -> Dict[str, Any]:
        """Requested versus achieved speed and delivery counters (JSON-safe)."""
        achieved = self.achieved_speed_now()
        if achieved and getattr(self, "sim_thread", None) is not None \
                and self.liveness()["state"] in ("stalled", "dead"):
            achieved = 0.0
        return {
            "last_step_wall_s": round(self.last_step_wall_s, 4),
            "step_in_progress_s": self.step_in_progress_s(),
            "requested_speed": self.sim_speed,
            "achieved_speed": achieved,
            "integration_dt_s": self.dt,
            "sim_time_s": round(self.total_steps * self.dt, 5),
            "sim_time_scope": SESSION_CLOCK_SCOPE,
            "step": self.total_steps,
            "snapshot_seq": self._snapshot_seq,
            "publish_hz": self.publish_hz,
            "overloaded": bool(self._can_step() and achieved < 0.9 * self.sim_speed
                               and len(self._advance_samples) > 4),
            "schedule_rebases": self.sched_stats["rebases"],
            "forgiven_wall_s": round(self.sched_stats["forgiven_wall_s"], 3),
            "max_batch_hold_ms": self.sched_stats["max_batch_hold_ms"],
            "command_latency": _percentiles_ms(list(self.command_latency)),
        }

    def _measure_achieved(self, now: float):
        self._speed_samples.append((now, self.total_steps))
        while len(self._speed_samples) > 2 and now - self._speed_samples[0][0] > 2.0:
            self._speed_samples.popleft()
        t0, s0 = self._speed_samples[0]
        if now - t0 > 0.2:
            self.sched_stats["achieved_speed"] = round((self.total_steps - s0) * self.dt / (now - t0), 3)

    def _publish_snapshot(self):
        """Assemble, serialize and publish one immutable telemetry frame (takes the lock)."""
        with self.lock:
            now = time.perf_counter()
            if self._can_step():
                self._measure_achieved(now)
            else:
                self._speed_samples.clear()
            self._snapshot_seq += 1
            telemetry = self._assemble_telemetry(self._last_step_result)
            telemetry["timing"]["steps_in_frame"] = self.sched_stats["steps_since_publish"]
            telemetry["command_acks"] = list(self.command_acks)
            self._refresh_views()
            self.sched_stats["steps_since_publish"] = 0
            self.latest_telemetry = telemetry
            data = json.dumps(telemetry).encode("utf-8")
            self.published = _Snapshot(self._snapshot_seq, self.total_steps, data, telemetry["timestamp"])
            self.sched_stats["published_snapshots"] += 1
            self._publish_due = False
        return self.published

    # ------------------------------------------------------------------ consistent reads
    VIEW_DEMAND_S = 10.0     # keep rebuilding a view this long after the last request
    VIEW_REFRESH_S = 0.5     # at most this often, so a fast simulation is not slowed

    def _refresh_views(self):
        """Rebuild recently requested views at a step boundary. Caller holds the lock."""
        now = time.monotonic()
        for key, (build, asked) in list(self._view_demand.items()):
            if now - asked > self.VIEW_DEMAND_S:
                self._view_demand.pop(key, None)
                continue
            if now - self._view_built.get(key, -1e9) < self.VIEW_REFRESH_S:
                continue
            try:
                self._views[key] = build()
                self._view_built[key] = now
            except Exception as exc:  # a view must never stop the simulation
                print(f"[Daemon] view {key!r} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def read_view(self, key: str, build, wait_s: float = 0.2, max_wait_s: float = 3.0):
        """A consistent read of simulation state that never waits behind a long step.

        ``build`` runs under the simulation lock.  When the lock is free within
        ``wait_s`` the view is built now.  Otherwise the copy built at the last step
        boundary is returned (the simulation thread keeps it fresh while clients
        ask).  Only the very first request, with no copy yet, waits up to
        ``max_wait_s``.  Returns None when nothing could be read in time.
        """
        self._view_demand[key] = (build, time.monotonic())
        for timeout, allow_cached in ((wait_s, True), (max_wait_s, False)):
            if self.lock.acquire(timeout=timeout):
                try:
                    view = self._views[key] = build()
                    self._view_built[key] = time.monotonic()
                finally:
                    self.lock.release()
                return view
            if allow_cached and key in self._views:
                return self._views[key]
        return self._views.get(key)

    def step_once(self, publish: bool = True) -> Dict[str, Any]:
        """One simulation tick plus trial bookkeeping. Caller holds ``self.lock``.

        ``publish=False`` (the scheduler) skips telemetry assembly; the snapshot is
        assembled only when published.  Telemetry assembly is read-only, so this
        does not change the simulated trajectory.
        """
        step_dt = self.dt
        self._check_assay_control_timeout()
        if self._stopped or self.paused or self.last_error or self._observation_waiting_for_save():
            # Keep the last measured assay metrics visible while nothing advances.
            self.latest_telemetry = self._assemble_telemetry(self._last_step_result)
            return {}

        # Teaching runs in an explicit cue chamber while the behavioral arena pauses.
        if self._teaching_suspended:
            if not self.active_brain.teaching:
                result = self._admit_lifecycle({'action': 'return_from_teaching'}, {})
                if result['status'] == 'error':
                    plan = dict(action='return_from_teaching', name='return_from_teaching', transition=True,
                                lifecycle=True, closes_presentation=False, end_reason='policy_change',
                                teaching_error=RuntimeError(result['message']))
                    self._begin_control_transaction({'action': 'return_from_teaching'}, plan)
                self.latest_telemetry = self._assemble_telemetry({})
                return {}
            self.total_steps += 1
            self._teaching_clock_s += step_dt
            self._teaching_tick(step_dt)
            self._last_step_result = {}
            if publish:
                self.latest_telemetry = self._assemble_telemetry({})
            if self.recorder is not None:
                self._capture_recording({})
            return {}

        # 1. Step simulation arena
        self._loop_phase = "step"
        try:
            if self.arena.observation_owner is not None:
                self.arena.observation_owner.provenance = self._observation_provenance()
            step_result = self.arena.step(step_dt)
            fly = self.arena.fly
            if not (math.isfinite(float(fly.pos.x)) and math.isfinite(float(fly.pos.y))
                    and math.isfinite(float(fly.heading))):
                raise NonFiniteStep("the step produced a non-finite fly pose (NaN/inf); the compute "
                                    "state is broken")
        except Exception as step_err:
            print(f"[Daemon] Exception in arena.step: {step_err}", file=sys.stderr)
            self._halt_on_error(step_err, phase="step")
            try:
                self.latest_telemetry = self._assemble_telemetry({})
            except Exception as telem_err:  # noqa: BLE001 -- the halt is already recorded
                print(f"[Daemon] telemetry after the halt failed: {telem_err}", file=sys.stderr)
            return {}

        self.total_steps += 1
        self.active_brain.steps += 1
        self.trial_sim_time += step_dt
        self._sync_trial_clock()
        pos = self.arena.fly.pos
        self._path.append((self.total_steps, round(float(pos.x), 4), round(float(pos.y), 4)))

        # Producer status after the actual step is the normal measurement authority.
        self._capture_terminal_observation(step_result)
        terminal_mode = (self._observation_terminal or {}).get("observation", {}).get("mode")
        if self._maybe_finish_observation_transition() and terminal_mode != "presentation":
            # Terminal values belong to the old segment, not the respawned world.
            step_result = {}

        # 2. Trial advancement: natural endpoint or time limit
        self._loop_phase = "trial bookkeeping"
        end_reason = self._trial_end_reason(step_result)
        if end_reason is not None:
            self._end_trial(step_result, end_reason)
            # Terminal outcomes belong to the old segment, never to the respawn pose.
            step_result = {}

        # 3. Assemble telemetry (the scheduler defers this to snapshot publication)
        self._last_step_result = step_result
        if publish:
            self._loop_phase = "publish"
            self.latest_telemetry = self._assemble_telemetry(step_result)
        if self.recorder is not None:
            self._capture_recording(step_result)

        # 4. Periodic checkpoint.  A failed save is handled by the save policy (_persistence_failed).
        self._maybe_periodic_checkpoint()
        self._loop_phase = "step"
        return step_result

    # ------------------------------------------------------------------ persistence (F2)
    def _persist(self, channel: str, fn, *args, path: Any = None, **kwargs):
        """One durable write, never raising.  What a failure does depends on its class
        and channel (_persistence_failed): a required save stops the run; only
        exploratory mode or a diagnostic channel lets the run continue unsaved."""
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- classified and handled in _persistence_failed
            self._persistence_failed(channel, exc, path)
            return None
        self._persisted(channel)
        return result

    def _persisted(self, channel: str) -> None:
        if channel in self.persistence.channels:
            self._close_gaps(channel)
        self.persistence.succeeded(channel)

    def _ledger(self, kind: str, **fields) -> None:
        """Append one event to the active brain's ledger (events.jsonl), guarded."""
        if kind in ("simulation_error", "simulation_error_cleared"):
            channel = "halt_log"    # the halt itself is held in memory, status and the run's validity
        else:
            channel = "trial_ledger" if kind == "trial" else "events_ledger"
        brain = self.active_brain
        self._persist(channel, brain.log, kind, path=getattr(brain, "directory", None), **fields)

    def _persistence_failed(self, channel: str, exc: BaseException, path: Any = None) -> None:
        """Classify a failed write and apply the policy (Codex direction, decision 3).

        compute (GPU/CUDA, memory, non-finite state) -> always halt;
        software (a bug, not the storage)            -> always halt;
        persistence on a DIAGNOSTIC channel          -> degrade;
        persistence on a REQUIRED channel            -> stop the run at this step boundary and
                                                        mark its result incomplete, unless
                                                        --exploratory: degrade with a recorded gap.
        Every case except the diagnostic one leaves an immutable invalidity mark on the run.
        """
        failure_class = classify_failure(exc)
        entry = self.persistence.failed(channel, exc, path, failure_class=failure_class)
        required = channel not in DIAGNOSTIC_CHANNELS
        if failure_class != FAILURE_PERSISTENCE or (required and not self.exploratory):
            self._stopping_failures += 1     # a switch that hit one must not lift the halt
        if failure_class != FAILURE_PERSISTENCE:
            phase = "device" if failure_class == FAILURE_COMPUTE else "persistence"
            self._invalidate(f"{failure_class}_failure_while_saving", channel=channel, exc=exc,
                             failure_class=failure_class)
            if self.last_error is None:
                self._halt_on_error(exc, phase=phase, channel=channel, failure_class=failure_class)
            if channel == 'learning_records' and self._pending_assay_control is not None:
                self._fail_assay_control(exc, before_apply=not self._pending_assay_control.get('applying'))
            return
        if not required:
            if entry["failures"] == 1 or entry["failures"] % 20 == 0:
                print(f"[Daemon] Diagnostic log not written ({channel}): {entry['error']}. The run "
                      f"continues; its scientific record is unaffected.", file=sys.stderr, flush=True)
            return
        if self.exploratory:
            self._invalidate("not_saved_exploratory_gap", channel=channel, exc=exc,
                             failure_class=failure_class, open_gap=True)
            if entry["failures"] == 1 or entry["failures"] % 20 == 0:
                print(f"[Daemon] EXPLORATORY MODE, NOT SAVING ({channel}, failure {entry['failures']}): "
                      f"{entry['error']}. The simulation keeps running and the gap is recorded; next attempt "
                      f"in {entry['backoff_s']:.0f} s.", file=sys.stderr, flush=True)
            if channel == 'learning_records' and self._pending_assay_control is not None:
                self._fail_assay_control(exc, before_apply=not self._pending_assay_control.get('applying'))
            return
        self._invalidate("required_save_failed", channel=channel, exc=exc, failure_class=failure_class)
        if self.last_error is None:
            self._halt_on_error(exc, phase="persistence", channel=channel, failure_class=failure_class)
        if channel == 'learning_records' and self._pending_assay_control is not None:
            self._fail_assay_control(exc, before_apply=not self._pending_assay_control.get('applying'))

    # ------------------------------------------------------------------ result validity
    def _current_run_id(self) -> Optional[str]:
        return self.manifest.run_id if getattr(self, "manifest", None) is not None else None

    def _invalidate(self, reason: str, *, channel: Optional[str] = None, exc: Optional[BaseException] = None,
                    failure_class: Optional[str] = None, open_gap: bool = False,
                    extra: Optional[Dict[str, Any]] = None, flush: bool = True) -> Dict[str, Any]:
        """Mark the active run's result incomplete.  Immutable: incidents are only ever
        appended (memory, the run manifest's events, the graph checkpoint meta and an
        append-only run_validity.jsonl); recovery adds ``recovered_*`` fields, never removes."""
        # Before any run is active the incident belongs to this daemon session.
        run_id = self._current_run_id() or f"session-{self.run_id}"
        for incident in self.incidents:   # one open incident per run and channel
            if (incident["run_id"] == run_id and incident.get("channel") == channel and channel
                    and incident.get("recovered_at") is None and incident["reason"] == reason):
                incident["failures"] = incident.get("failures", 1) + 1
                return incident
        incident = {"run_id": run_id, "paradigm": self.active_paradigm_id, "backend": self.backend,
                    "reason": reason, "channel": channel, "failure_class": failure_class,
                    "error": f"{type(exc).__name__}: {exc}" if exc is not None else None,
                    "errno": getattr(exc, "errno", None), "step": getattr(self, "total_steps", 0),
                    "sim_time_s": round(getattr(self, "total_steps", 0) * getattr(self, "dt", 0.02), 5),
                    "sim_time_scope": SESSION_CLOCK_SCOPE, "daemon_run_id": self.run_id,
                    "at": round(time.time(), 3),
                    "mode": "exploratory" if self.exploratory else "scientific", "gap_open": open_gap,
                    "recovered_at": None, "failures": 1}
        incident.update(extra or {})
        self.incidents.append(incident)
        if self.manifest is not None:
            self.manifest.record_event("result_incomplete", step=incident["step"],
                                       **{k: v for k, v in incident.items() if k not in ("step",)})
        active = self.registry.active if (self.graph_mode and self.registry is not None) else None
        if active is not None:
            active.invalidity = list(getattr(active, "invalidity", []) or []) + [dict(incident)]
        self._pending_validity.append(dict(incident))
        pending = self._pending_assay_control
        if flush and not (pending is not None and pending['plan'].get('transition')):
            self._flush_validity()
        self._publish_due = True
        return incident

    def _close_gaps(self, channel: str) -> None:
        """A channel saves again: the open gap ends (recorded), the run stays incomplete."""
        for incident in self.incidents:
            if incident.get("channel") == channel and incident.get("recovered_at") is None:
                incident.update(recovered_at=round(time.time(), 3), recovered_step=self.total_steps,
                                recovered_by="save succeeded", gap_open=False)
                self._pending_validity.append({"run_id": incident["run_id"], "event": "gap_closed",
                                               "channel": channel, "step": self.total_steps,
                                               "at": incident["recovered_at"]})

    def _mark_recovered(self, cleared_by: str, *, flush=True) -> None:
        for incident in self.incidents:
            if incident.get("recovered_at") is None and not incident.get("gap_open"):
                incident.update(recovered_at=round(time.time(), 3), recovered_step=self.total_steps,
                                recovered_by=cleared_by)
                self._pending_validity.append({"run_id": incident["run_id"], "event": "recovered",
                                               "by": cleared_by, "step": self.total_steps,
                                               "at": incident["recovered_at"]})
        if flush:
            self._flush_validity()

    def _flush_validity(self, *, wait=False) -> bool:
        """Save one snapshot-owned append batch without dropping later records.

        Ordinary locked callers never wait for another writer. C2 off-lock
        workers may serialize behind it with wait=True. Only this exact batch is
        removed after fsync; appends during the write remain queued for retry.
        """
        if not self._pending_validity:
            return True
        if not self._validity_writer_lock.acquire(blocking=wait):
            return False
        self._validity_flushing = True
        owned = self._pending_validity.copy()
        try:
            payload = ''.join(json.dumps(copy.deepcopy(record), default=str) + "\n" for record in owned)
            with open(self.output_dir / "run_validity.jsonl", "a", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            if (len(self._pending_validity) < len(owned) or any(
                    current is not original for current, original in
                    zip(self._pending_validity[:len(owned)], owned))):
                raise RuntimeError('Result-validity batch ownership changed during persistence')
            del self._pending_validity[:len(owned)]
            if "validity_ledger" in self.persistence.channels:
                self._close_gaps("validity_ledger")
                self.persistence.succeeded("validity_ledger")
            return True
        except OSError as exc:
            self._validity_write_failed(exc)
            return False
        finally:
            self._validity_flushing = False
            self._validity_writer_lock.release()

    def _validity_write_failed(self, exc: OSError) -> None:
        entry = self.persistence.failed("validity_ledger", exc, self.output_dir / "run_validity.jsonl")
        if entry["failures"] == 1:
            print(f"[Daemon] Result-validity ledger NOT written: {entry['error']}. Its records are kept "
                  f"in memory and retried; the run is marked incomplete.", file=sys.stderr, flush=True)
        self._invalidate("validity_ledger_unwritable", channel="validity_ledger", exc=exc,
                         failure_class=FAILURE_PERSISTENCE, open_gap=self.exploratory, flush=False)
        if not self.exploratory:
            self._stopping_failures += 1
            if self.last_error is None:
                self._halt_on_error(exc, phase="persistence", channel="validity_ledger",
                                    failure_class=FAILURE_PERSISTENCE)

    # Field types of a ledger record; anything else is damage, never trusted.
    _LEDGER_TYPES = {"run_id": (str, type(None)), "reason": str, "event": str, "session": str,
                     "channel": (str, type(None)), "failure_class": (str, type(None)),
                     "at": (int, float), "recovered_at": (int, float, type(None)),
                     "step": (int, type(None)), "recovered_step": (int, type(None)),
                     "interrupted_session": (str, type(None)), "gap_open": bool, "mode": str}
    # Required fields of every recognised event (all present, identifiers non-empty).
    _LEDGER_EVENTS = {"session_start": ("session", "at"), "session_end": ("session", "at"),
                      "activate": ("run_id", "session", "at"), "recovered": ("run_id", "at"),
                      "gap_closed": ("run_id", "channel", "at"),
                      "session_end_failed": ("session", "at")}
    _RUN_ID_RE = re.compile(r'"run_id"\s*:\s*"([^"\\]{1,200})"')

    @classmethod
    def _valid_ledger_record(cls, record: Any) -> bool:
        """Validate one ledger record.  Never raises: anything unexpected is damage."""
        try:
            return cls._check_ledger_record(record)
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def _check_ledger_record(cls, record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        for key, types in cls._LEDGER_TYPES.items():
            if key in record:
                value, allowed = record[key], (types if isinstance(types, tuple) else (types,))
                if isinstance(value, bool) and bool not in allowed:   # bool is an int subclass
                    return False
                if not isinstance(value, allowed):
                    return False
        def ident(key):     # a required, non-empty identifier
            value = record.get(key)
            return isinstance(value, str) and value.strip() != ""

        def number(key):
            value = record.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            try:
                return math.isfinite(value)
            except OverflowError:          # e.g. the JSON integer 10**400
                return False

        if ("reason" in record) == ("event" in record):
            return False                   # exactly one schema: an incident or an event
        if "reason" in record:             # incident
            return ident("run_id") and ident("reason") and number("at")
        required = cls._LEDGER_EVENTS.get(record["event"])
        if required is None:
            return False                   # unknown event schema: never trusted
        return all(number(k) if k == "at" else ident(k) for k in required)

    def _read_validity_ledger(self) -> List[Dict[str, Any]]:
        """Read run_validity.jsonl without ever failing open.

        * absent (a new output directory): no history, nothing to distrust;
        * unreadable (permissions, I/O): ``validity_history["state"] = "unreadable"`` and
          every run activated in this process is marked incomplete (history unknown);
        * a truncated last line (no newline: a crash mid-write) is copied to a
          quarantine file and separated by a newline before anything is appended,
          so it can never swallow the next record;
        * a line that is not JSON, or has wrong field types, is damage: the run it names
          (when its run_id can be read) is marked incomplete; damage that names no run
          marks every run with earlier history.  Usable records are kept.
        """
        path = self.output_dir / "run_validity.jsonl"
        self.validity_history = {"state": "absent", "skipped": 0, "quarantined": None, "error": None,
                                 "damaged_runs": [], "unscoped_damage": 0}
        self.validity_ledger_skipped = 0
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            self.validity_history.update(state="unreadable", error=f"{type(exc).__name__}: {exc}")
            return []
        self.validity_history["state"] = "ok"
        if raw and not raw.endswith(b"\n"):
            head, _, tail = raw.rpartition(b"\n")
            quarantine = self.output_dir / f"run_validity.quarantine-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
            try:
                with open(quarantine, "ab") as fh:
                    fh.write(tail + b"\n")
                self.validity_history["quarantined"] = quarantine.name
            except OSError:
                pass
            try:
                with open(path, "ab") as fh:     # separate the tail; nothing is removed
                    fh.write(b"\n")
            except OSError as exc:
                self.validity_history["error"] = f"{type(exc).__name__}: {exc}"
        records, damaged_runs = [], set()
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                record = None
            if self._valid_ledger_record(record):
                records.append(record)
                continue
            self.validity_ledger_skipped += 1
            match = self._RUN_ID_RE.search(line)
            if match:
                damaged_runs.add(match.group(1))
            else:
                self.validity_history["unscoped_damage"] += 1
        self.validity_history.update(skipped=self.validity_ledger_skipped, damaged_runs=sorted(damaged_runs))
        if self.validity_ledger_skipped:
            self.validity_history["state"] = "damaged"
        return records

    @staticmethod
    def _interrupted_sessions(records: List[Dict[str, Any]]) -> Dict[str, set]:
        """session id -> run ids it activated, for sessions with no clean ``session_end``."""
        started, ended, failed, active = [], set(), set(), {}
        for record in records:
            event, session = record.get("event"), record.get("session")
            if event == "session_start" and session:
                started.append(session)
            elif event == "session_end" and session:
                ended.add(session)
            elif event == "session_end_failed" and session:
                failed.add(session)
            elif event == "activate" and session and record.get("run_id"):
                active.setdefault(session, set()).add(record["run_id"])
        return {s: active.get(s, set()) for s in started if s not in ended or s in failed}

    def _reconcile_validity(self, *, flush=True) -> None:
        """On activation: bring this run's recorded incidents back (never return an
        interrupted or failed scientific run to valid), and record the activation."""
        run_id = self._current_run_id()
        if run_id is None:
            return
        known = {(i.get("run_id"), i.get("at"), i.get("reason")) for i in self.incidents}
        for record in self._validity_ledger:
            if record.get("run_id") != run_id or not record.get("reason"):
                continue
            if (record.get("run_id"), record.get("at"), record.get("reason")) not in known:
                self.incidents.append(dict(record, restored_from_ledger=True))
                known.add((record.get("run_id"), record.get("at"), record.get("reason")))
        for record in self._validity_ledger:      # recoveries recorded after the incident
            if record.get("run_id") == run_id and record.get("event") in ("recovered", "gap_closed"):
                for incident in self.incidents:
                    if incident["run_id"] == run_id and incident.get("recovered_at") is None \
                            and (incident.get("at") or 0) <= (record.get("at") or 0):
                        incident.update(recovered_at=record.get("at"), recovered_step=record.get("step"),
                                        recovered_by=record.get("by") or record.get("event"), gap_open=False)
        flagged = {i.get("interrupted_session") for i in self.incidents if i["run_id"] == run_id}
        for session, runs in self._interrupted.items():
            if run_id in runs and session not in flagged:
                self._invalidate("interrupted_unclean_shutdown", flush=flush, extra={
                    "interrupted_session": session, "recovered_at": round(time.time(), 3),
                    "recovered_by": "restart from the last checkpoint",
                    "note": "the process ended without a clean shutdown; steps after the last checkpoint "
                            "were lost"})
        # Missing evidence never certifies a run: unreadable or damaged history makes
        # this run's validity unknown, i.e. incomplete.
        history = getattr(self, "validity_history", None) or {}
        reasons = {i["reason"] for i in self.incidents if i["run_id"] == run_id}
        active = self.registry.active if (self.graph_mode and self.registry is not None) else None
        fallback = getattr(active, "restore_fallback", None)
        if fallback:
            # The newest checkpoint could not be used and an older one was restored: the
            # steps committed after it are lost, whatever the previous shutdown looked like.
            active.restore_fallback = None
            self._invalidate("restored_older_checkpoint", flush=flush, extra={
                "restored_version": fallback.get("restored_version"),
                "newest_version": fallback.get("newest_version"),
                "lost_from_step": fallback.get("restored_step"), "lost_to_step": fallback.get("newest_known_step"),
                "lost_steps": fallback.get("lost_steps"), "skipped_versions": fallback.get("skipped_versions"),
                "problem": fallback.get("problem")})
        established = (any(r.get("run_id") == run_id for r in self._validity_ledger)
                       or bool(getattr(active, "checkpoint_version", 0)))
        if history.get("state") == "unreadable" and "validity_history_unreadable" not in reasons:
            self._invalidate("validity_history_unreadable", flush=flush, extra={
                "error": history.get("error"),
                "note": "the validity ledger exists but could not be read; this run's history is unknown"})
        damaged = run_id in (history.get("damaged_runs") or []) or (history.get("unscoped_damage") and established)
        if damaged and "validity_history_damaged" not in reasons:
            self._invalidate("validity_history_damaged", flush=flush, extra={
                "skipped_records": history.get("skipped"), "quarantined": history.get("quarantined"),
                "note": "part of the validity ledger is corrupt or truncated; this run's history may be missing"})
        self._pending_validity.append({"event": "activate", "run_id": run_id, "session": self.run_id,
                                       "at": round(time.time(), 3)})
        if flush:
            self._flush_validity()

    def result_validity(self) -> Dict[str, Any]:
        run_id = self._current_run_id()
        mine = [dict(i) for i in self.incidents if i["run_id"] == run_id]
        return {"run_id": run_id, "state": "incomplete" if mine else "valid_so_far",
                "mode": "exploratory" if self.exploratory else "scientific",
                "incidents": mine, "other_runs_incomplete": sorted({i["run_id"] for i in self.incidents
                                                                    if i["run_id"] != run_id and i["run_id"]}),
                "unwritten_records": len(self._pending_validity),
                "history": {k: v for k, v in (getattr(self, "validity_history", None) or {}).items()
                            if k in ("state", "skipped", "quarantined", "error")}}

    def _checkpoint_bytes_needed(self) -> int:
        last = self._last_checkpoint_bytes
        if last is None:
            last = 0
            if self.graph_mode and self.registry is not None and self.registry.active is not None:
                try:
                    pointer = self.registry.current_pointer(self.registry.active.instance_id)
                    last = int((pointer or {}).get("bytes") or 0)
                except Exception:  # noqa: BLE001 -- an estimate only
                    last = 0
        return 2 * int(last) + int(self.disk_reserve_bytes)

    def _check_free_space(self) -> None:
        """Skip a checkpoint BEFORE the disk is full: free < 2 x last checkpoint + reserve."""
        import shutil
        needed = self._checkpoint_bytes_needed()
        roots = {self.output_dir}
        if self.registry is not None:
            roots.add(Path(self.registry.root))
        for root in roots:
            try:
                free = shutil.disk_usage(root).free
            except OSError:
                continue
            if free < needed:
                raise DiskSpaceLow(28, f"only {free / 2**20:.0f} MB free, a checkpoint needs "
                                       f"{needed / 2**20:.0f} MB (2 x last checkpoint + reserve); "
                                       f"checkpoint skipped", str(root))

    def checkpoint_now(self, tag: str = "periodic", check_space: bool = True) -> Optional[Path]:
        """Guarded checkpoint: returns the path, or None when the save failed (reported)."""
        if self._state_uncertain:
            print(f"[Daemon] Checkpoint {tag!r} skipped: state after the unexpected simulation-loop exit "
                  "is uncertain; the last verified checkpoint is kept.", file=sys.stderr, flush=True)
            return None
        prev_phase, self._loop_phase = self._loop_phase, "persistence"
        try:
            if check_space:
                self._check_free_space()
            path = self.save_checkpoint(tag)
        except Exception as exc:  # noqa: BLE001
            self._persistence_failed("checkpoint", exc, getattr(exc, "filename", None) or self.checkpoints_dir)
            return None
        finally:
            self._loop_phase = prev_phase
        self._persisted("checkpoint")
        self._flush_validity()
        return path

    def _maybe_periodic_checkpoint(self) -> None:
        now = time.time()
        if now - self.last_checkpoint_time < self.checkpoint_interval:
            return
        if not self.persistence.retry_due("checkpoint", now):
            return
        self.last_checkpoint_time = now
        self.checkpoint_now("periodic")

    def _capture_recording(self, step_result: Dict[str, Any]) -> None:
        """Capture one replay frame; a failing --record stops visibly (F2)."""
        self._loop_phase = "recording"
        try:
            self.recorder.capture(self, step_result)
        except Exception as exc:  # noqa: BLE001
            self._recording_failed(exc)

    def _recording_failed(self, exc: BaseException, during: str = "capture") -> None:
        pending = self._pending_assay_control
        recorder, self.recorder = self.recorder, None
        cleanup_error = None
        name = getattr(getattr(recorder, "path", None), "name", None)
        try:
            if recorder is not None and hasattr(recorder, "abort"):
                if pending is not None:
                    recorder.abort(f"{type(exc).__name__}: {exc}", defer_cleanup=True)
                else:
                    recorder.abort(f"{type(exc).__name__}: {exc}")
        except Exception as cleanup_exc:  # noqa: BLE001 -- the recording is already lost; report it
            cleanup_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        self.recording_error = {"recording": name, "run_id": self._current_run_id(),
                                "cleanup_error": cleanup_error, "error": f"{type(exc).__name__}: {exc}",
                                "errno": getattr(exc, "errno", None), "step": self.total_steps,
                                "frames": getattr(recorder, "frames", None), "at": round(time.time(), 3),
                                "during": during,
                                "message": f"Recording {name} " + ("could not be finished" if during == "finalise"
                                           else "stopped: it could not be written") + " "
                                           f"({type(exc).__name__}: {exc}). What was written is kept. If publication completed "
                                           "before the final durability acknowledgement failed, the artifact may remain "
                                           "readable/listable. The save receipt is INVALID. " + (
                                           "The failed control halted that run with uncertain control state; no applied acknowledgement."
                                           if pending is not None else "Exploratory mode permits continuing without this recording."
                                           if self.exploratory else "Scientific recovery requires a new recording with a verified initial capture.")}
        self._persistence_failed("recording", exc, getattr(recorder, "path", None))
        if pending is not None and not pending.get('failed'):
            self._fail_assay_control(exc, before_apply=False)
        print(f"[Daemon] {self.recording_error['message']}", file=sys.stderr, flush=True)

    HALT_RECOVERY = ("The simulation is halted by this error and does not advance. Select an assay "
                     "(selecting the same one retries it) or switch the controller backend to rebuild "
                     "the controller and resume. Pausing, resuming or changing speed does not lift it.")
    SAVE_RECOVERY = ("A required save failed, so the run stopped at a step boundary and its result is marked "
                     "INCOMPLETE. The last good checkpoint and the in-memory state are kept. Free disk space "
                     "(or fix the storage), then select the assay: a checkpoint is written first and the run "
                     "resumes only if it succeeds. Start with --exploratory to keep running unsaved instead.")
    RECORDING_RECOVERY = ("A requested recording failed and its prefix is kept. Repair storage, then start a "
                          "replacement recording with a new name using record_start before selecting the assay. "
                          "Alternatively restart with --record NEW_NAME and the same saved-brain output directory. "
                          "Do not reuse the failed filename. A successful checkpoint alone does not restore recording.")

    def _raw_recording_recovery_refusal(self) -> Optional[dict]:
        """A requested raw recording remains required after its writer detaches.

        Unrecorded runs acquire no obligation; exploratory mode retains its
        explicit unsaved-continuation policy. This check performs no writes.
        """
        failure = self.persistence.describe()['failing'].get('recording')
        if not self.exploratory and failure and failure.get('required'):
            return {'status': 'error', 'message': self.RECORDING_RECOVERY,
                    'halted_by_error': self.last_error}
        return None

    PHASE_LABELS = {"device": "GPU/compute", "step": "simulation step", "publish": "telemetry publication",
                    "persistence": "saving", "command": "command", "recording": "recording",
                    "trial bookkeeping": "trial bookkeeping", "loop": "simulation loop"}

    def _halt_on_error(self, exc: BaseException, phase: str = "step", channel: Optional[str] = None,
                       failure_class: Optional[str] = None, incident_reason: Optional[str] = None,
                       incident_extra: Optional[Dict[str, Any]] = None):
        """Record the failure that halts the simulation (F1). Caller holds the lock.

        Never raises: its own bookkeeping write is guarded, so a full disk cannot turn
        an honest halt into a dead simulation thread (audit F, #7).  A second failure
        while already halted is counted, not allowed to overwrite the first cause.
        """
        if self.last_error is not None:
            if self.error_detail is not None:
                self.error_detail["repeats"] = self.error_detail.get("repeats", 0) + 1
                self._fault_snapshot = (self.last_error, copy.deepcopy(self.error_detail))
            return
        registry = self.registry if self.graph_mode else None
        active = registry.active if registry is not None else None
        message = str(exc) or type(exc).__name__
        failure_class = failure_class or classify_failure(exc)
        if phase == "step" and failure_class == FAILURE_COMPUTE:
            phase = "device"
        if phase not in ("step", "device"):
            message = f"{self.PHASE_LABELS.get(phase, phase)} failed: {type(exc).__name__}: {message}"
        detail = {
            "message": message, "type": type(exc).__name__, "phase": phase,
            "failure_class": failure_class, "step": self.total_steps,
            "sim_time_s": round(self.total_steps * self.dt, 5), "sim_time_scope": SESSION_CLOCK_SCOPE,
            "paradigm": self.active_paradigm_id,
            "backend": self.backend, "instance_id": active.instance_id if active is not None else None,
            "at": round(time.time(), 3), "recover": self.HALT_RECOVERY,
            "traceback": _traceback_tail(exc)}
        if channel:
            detail["channel"] = channel
        if phase == "persistence" and failure_class == FAILURE_PERSISTENCE:
            detail["recover"] = self.RECORDING_RECOVERY if channel == 'recording' else self.SAVE_RECOVERY
        self.error_detail = detail
        self.last_error = message
        self._fault_snapshot = (message, copy.deepcopy(detail))
        if not channel:   # a save failure was already recorded by _persistence_failed
            self._invalidate(incident_reason or f"halted_{phase.replace(' ', '_')}", exc=exc,
                             failure_class=failure_class, extra=incident_extra)
        self._publish_due = True
        print(f"[Daemon] HALTED ({phase}): {message}", file=sys.stderr, flush=True)
        pending = self._pending_assay_control
        if pending is not None and pending['plan'].get('transition'):
            return  # C2 failure evidence is saved off-lock by its completion worker.
        try:
            self._ledger("simulation_error", error=self.last_error, step=self.total_steps,
                         run_id=self.run_id, segment_id=getattr(self, "segment_id", None),
                         error_type=self.error_detail["type"], phase=phase)
        except Exception:  # noqa: BLE001 -- _ledger is guarded; this is belt and braces
            pass

    def _clear_error(self, cleared_by: str, *, defer_persistence=False) -> Optional[Dict[str, Any]]:
        """Lift a halt after a successful rebuild; the cleared error is kept and logged."""
        if self.last_error is None:
            return None
        if self._raw_recording_recovery_refusal() is not None:
            return None
        recorder_recovery_state = None
        recorder_failure = self._recorder_watchdog_failure
        if recorder_failure is not None and recorder_failure.get("active", True):
            recorder_problem, recorder_recovery_state = self._learning_recorder_problem()
            if (recorder_problem is not None
                    or not self._learning_recorder_verified_healthy(recorder_recovery_state)):
                return None
        detail = dict(self.error_detail or {"message": self.last_error})
        detail.update(cleared_by=cleared_by, cleared_at=round(time.time(), 3), cleared_step=self.total_steps,
                      resumed_paradigm=self.active_paradigm_id, resumed_backend=self.backend)
        detail.pop("recover", None)
        detail.pop("traceback", None)
        self.cleared_errors.append(detail)
        self.last_error = None
        self.error_detail = None
        self._fault_snapshot = (None, None)
        self._observation_validity_update = None
        self._publish_failures = 0
        self._steppable_since = time.perf_counter()
        self._mark_recovered(cleared_by, flush=not defer_persistence)
        self._resolve_recorder_watchdog_locked(cleared_by, recorder_recovery_state)
        if not defer_persistence:
            self._ledger("simulation_error_cleared", error=detail["message"], cleared_by=cleared_by,
                         step=self.total_steps, run_id=self.run_id, segment_id=self.segment_id)
        print(f"[Daemon] Halt lifted by {cleared_by}: {detail['message']}", flush=True)
        self._publish_due = True
        return detail

    def _trial_end_reason(self, step_result: Dict[str, Any]) -> Optional[str]:
        """Why the current trial is over, or None while it continues."""
        # Once metric-contract/1.2 owns the active assay, only its status may close
        # the measurement. The normal freeze/hold/ack transition is wired separately.
        if self.arena.observation_owner is not None:
            return None
        if self.continuous:
            return None
        telemetry = step_result.get("paradigm_telemetry") or {}
        for flag in self.TRIAL_END_FLAGS:
            if telemetry.get(flag):
                return flag
        if telemetry.get("first_choice"):
            return "first_choice"
        if telemetry.get("decision_outcome") == "ABORT":
            return "decision_abort"

        paradigm = getattr(self.arena, "paradigm", None)
        manager = getattr(paradigm, "trial_manager", None)
        if manager is not None and not getattr(manager, "trial_active", True):
            return "max_duration_steps"
        if self.trial_length_s is not None and self.trial_sim_time >= self.trial_length_s:
            return "time_limit"
        return None

    def _end_trial(self, step_result: Dict[str, Any], reason: str):
        """Record the milestone, reset the paradigm's trial state and respawn the fly.
        Mushroom-body weights and other plasticity are kept: learning is continuous."""
        transition = {"reason": reason, "step": self.total_steps, "ended_segment": self.segment_id,
                      "terminal_pose": {"x": self.arena.fly.pos.x, "y": self.arena.fly.pos.y},
                      "terminal_metrics": step_result.get("paradigm_metrics", {})}
        paradigm = getattr(self.arena, "paradigm", None)
        if paradigm is not None and hasattr(paradigm, "reset_trial"):
            paradigm.reset_trial()
        self.arena.reset_fly_to_spawn()
        # Commit bookkeeping only after the physical reset succeeds. On an exception
        # the caller retains the old durable terminal and marks the arena uncertain.
        self.transition = transition
        self._record_trial_milestone(step_result, reason)
        self.segment_id = uuid.uuid4().hex
        self.trial_sim_time = 0.0
        self._sync_trial_clock(new_trial=True)
        # Saved after the counter advanced and elapsed reset, so the completed trial
        # count and the trial clock describe the same boundary.
        self._persist("brain_save", self.active_brain.save, path=self.active_brain.path)
        self._path.clear()
        self._publish_due = True

    def _assemble_telemetry(self, step_res: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs standardized JSON telemetry packet for browser streaming."""
        fly = self.arena.fly
        stim = step_res.get("stimuli", {})
        if self.arena.paradigm is None and not stim:
            sensed = self.arena.sample_antennae(fly)
            stim = {**stim, "odor_a": sensed['mean_a'], "odor_b": sensed['mean_b']}
        p_metrics = step_res.get("paradigm_metrics", {})

        # Read actual effective KC→MBON weights (the old .weights field did not exist).
        w = fly.circuit.get_effective_weights()
        weights_mean, weights_std = float(np.mean(w)), float(np.std(w))

        # Joint flexions & cuticular loads (real-time 6-limb biomechanics)
        c_bridge = getattr(fly, "connectome_bridge", getattr(self.arena, "bridge", None))
        if c_bridge and getattr(c_bridge, "joint_angles", None):
            joint_angles = c_bridge.joint_angles
            loads = getattr(c_bridge, "cs_ground_forces", {k: (1.85 if v.get("phase") == "STANCE" else 0.0) for k, v in joint_angles.items()})
        else:
            t = self.total_steps * self.dt
            tripod_phase = (t * 8.0 * 2.0 * math.pi) % (2.0 * math.pi)
            stance_l1 = math.sin(tripod_phase) >= 0.0

            joint_angles = {
                "L1": {"ctr": round(math.sin(tripod_phase) * 25.0, 1), "fti": round(80.0 - math.cos(tripod_phase) * 25.0, 1), "phase": "STANCE" if stance_l1 else "SWING"},
                "L2": {"ctr": round(math.sin(tripod_phase + math.pi) * 25.0, 1), "fti": round(80.0 - math.cos(tripod_phase + math.pi) * 25.0, 1), "phase": "SWING" if stance_l1 else "STANCE"},
                "L3": {"ctr": round(math.sin(tripod_phase) * 25.0, 1), "fti": round(80.0 - math.cos(tripod_phase) * 25.0, 1), "phase": "STANCE" if stance_l1 else "SWING"},
                "R1": {"ctr": round(math.sin(tripod_phase + math.pi) * 25.0, 1), "fti": round(80.0 - math.cos(tripod_phase + math.pi) * 25.0, 1), "phase": "SWING" if stance_l1 else "STANCE"},
                "R2": {"ctr": round(math.sin(tripod_phase) * 25.0, 1), "fti": round(80.0 - math.cos(tripod_phase) * 25.0, 1), "phase": "STANCE" if stance_l1 else "SWING"},
                "R3": {"ctr": round(math.sin(tripod_phase + math.pi) * 25.0, 1), "fti": round(80.0 - math.cos(tripod_phase + math.pi) * 25.0, 1), "phase": "SWING" if stance_l1 else "STANCE"},
            }
            loads = {k: (1.85 if v["phase"] == "STANCE" else 0.0) for k, v in joint_angles.items()}

        # 18-DOF joint angles in radians: order [L1, L2, L3, R1, R2, R3] x [Coxa, Femur, Tibia]
        leg_names = ("L1", "L2", "L3", "R1", "R2", "R3")
        joint_angles_rad = []
        for leg in leg_names:
            info = joint_angles.get(leg, {})
            ctr_deg = float(info.get("ctr", 0.0))
            fti_deg = float(info.get("fti", 80.0))
            phase_sign = 1.0 if info.get("phase") == "STANCE" else -1.0
            thc_rad = round(math.radians(phase_sign * 8.0), 4)
            ctr_rad = round(math.radians(ctr_deg), 4)
            fti_rad = round(math.radians(fti_deg), 4)
            joint_angles_rad.extend([thc_rad, ctr_rad, fti_rad])

        leg_contacts = [bool(joint_angles.get(leg, {}).get("phase") == "STANCE") for leg in leg_names]

        pos_z = float(getattr(fly, "pos_z", 0.5))
        body_position_mm = [round(float(fly.pos.x), 4), round(float(fly.pos.y), 4), round(pos_z, 4)]
        heading = float(fly.heading)
        body_quaternion_wxyz = [
            round(math.cos(heading / 2.0), 5),
            0.0,
            0.0,
            round(math.sin(heading / 2.0), 5),
        ]

        ang_vel = float(getattr(fly, "angular_velocity", 0.0))
        speed = float(fly.speed)
        b_state = str(getattr(fly, "behavioral_state", "FORAGING"))
        conn_telem = getattr(fly, "last_connectome_telemetry", None) or {}
        if conn_telem and "dn_rates" in conn_telem:
            dn_rates = conn_telem["dn_rates"]
        else:
            dn_rates = {
                "dna02_l": round(max(0.0, -ang_vel * 8.0), 2),
                "dna02_r": round(max(0.0, ang_vel * 8.0), 2),
                "dnp09": round(max(0.0, speed * 2.5), 2),
                "mdn": 25.0 if b_state == "REVERSE" else 0.0,
                "gf": 50.0 if b_state == "ESCAPE" else 0.0,
            }
        controller_id = "connectome-v3" if str(self.backend).startswith("connectome") else "modular"

        if c_bridge and hasattr(c_bridge, "last_body_obs") and c_bridge.last_body_obs:
            b_obs = c_bridge.last_body_obs
            if "joint_angles_rad" in b_obs:
                joint_angles_rad = [round(float(v), 4) for v in b_obs["joint_angles_rad"][:18]]
            if "thorax" in b_obs:
                thorax = b_obs["thorax"]
                if "position_mm" in thorax:
                    body_position_mm = [round(float(v), 4) for v in thorax["position_mm"]]
                if "quaternion_wxyz" in thorax:
                    body_quaternion_wxyz = [round(float(v), 5) for v in thorax["quaternion_wxyz"]]
            if "contacts" in b_obs and "found" in b_obs["contacts"]:
                leg_contacts = [bool(f > 0.5) for f in b_obs["contacts"]["found"][:6]]

        motor = self.motor_summary()
        health = self.health()
        packet = {
            "type": "telemetry",
            "run_id": self.run_id,
            "controller_id": controller_id,
            "joint_angles_rad": joint_angles_rad,
            "leg_contacts": leg_contacts,
            "body_position_mm": body_position_mm,
            "body_quaternion_wxyz": body_quaternion_wxyz,
            "dn_rates": dn_rates,
            # Controller identity (same dict as /api/status and the switch ack) and
            # motor provenance; see docs/DATA_SCHEMA.md "Identity and motor provenance".
            "identity": self.identity(),
            "motor": motor,
            "controller_fault": motor.get("controller_fault"),
            "segment_id": self.segment_id,
            "transition": self.transition,
            "continuous": self.continuous,
            "paused": self.paused,
            "error": health["error"],
            "halted": health["halted"],
            "error_detail": health["error_detail"],
            # Audit F: is the simulation advancing, and is it saving?  The page shows
            # "NOT ADVANCING" and "NOT SAVING" from these (docs/DATA_SCHEMA.md).
            "status": health["status"],
            "liveness": health["liveness"],
            "persistence": health["persistence"],
            "recording_error": health["recording_error"],
            "mode": health["mode"],
            "result_validity": health["result_validity"],
            "observation_publication": self.observation_publication_status(),
            "observation": self._live_observation(),
            "observation_lifecycle": self.observation_lifecycle_status(),
            "sim_time_s": round(self.total_steps * self.dt, 5),
            "sim_time_scope": SESSION_CLOCK_SCOPE,
            "clocks": self.clock_status(),
            "brain_id": self.active_brain.brain_id,
            "brain": self.active_brain.summary(),
            "timestamp": round(time.time(), 3),
            "step": self.total_steps,
            "paradigm": self.active_paradigm_id,
            "paradigm_title": self.active_paradigm_title,
            "sim_speed": self.sim_speed,
            # Requested versus achieved speed, dt and delivery counters (see timing_snapshot).
            "timing": self.timing_snapshot(),
            # Measured positions of the most recent steps in this segment, [step, x, y]:
            # lets a decimated display draw the path actually taken between frames.
            "path": [list(p) for p in self._path],
            "stimuli": stim,
            "live_assay": assay_controls.describe(self.arena),
            "motor_drives": getattr(fly, "sensorimotor_drives", {}),
            "assay_state": step_res.get("paradigm_telemetry", {}),
            "scene": {
                "geometry": assay_geometry_telemetry(self.arena),
                **assay_cue_telemetry(self.arena),
                "landmarks": [lm.pos for lm in getattr(self.arena.paradigm,"landmarks",[]) if lm.pos is not None],
                "stripe_contrast": getattr(self.arena.paradigm,"stripe_contrast",1.0),
                "invert_sectors": getattr(self.arena.paradigm,"invert_sectors",False),
                "food": [p.to_tuple() for p in self.arena.food_positions],
                "hazards": [p.to_tuple() for p in self.arena.hazard_positions],
                "predators": [[p.pos.x, p.pos.y, p.get_velocity()[0], p.get_velocity()[1]] for p in self.arena.predators],
                **{name: getattr(self.arena.paradigm, name) for name in
                   ('female_pos', 'female_type', 'refuge_pos', 'refuge_radius', 'drum_angle_deg', 'nozzle_pos', 'filament_sigma', 'cs_plus_arm')
                   if hasattr(self.arena.paradigm, name)},
            },
            "trial": self.current_trial,
            "trial_elapsed_s": round(self.trial_sim_time, 2),
            "trials_completed": len(self.trial_history),
            "world_bounds": list(getattr(self.arena, "world_bounds", (0.0, 0.0, self.arena.width, self.arena.height))),
            "fly": {
                "x": round(float(fly.pos.x), 5),
                "y": round(float(fly.pos.y), 5),
                "heading": round(float(fly.heading), 3),
                "speed": round(float(fly.speed), 2),
                "radius": float(getattr(fly, "radius", 1.5)),
                "state": getattr(fly, "behavioral_state", "FORAGING")
            },
            "sensory": {
                "temp": round(float(stim.get("temperature", 24.0)), 1),
                "odor_a": round(float(stim.get("odor_a", stim.get("odor_conc", 0.0))), 3),
                "odor_b": round(float(stim.get("odor_b", 0.0)), 3),
                "cva": round(float(stim.get("cva_concentration", 0.0)), 3),
                "wind_x": round(float(stim.get("wind", self.arena.wind)[0]), 2),
                "wind_y": round(float(stim.get("wind", self.arena.wind)[1]), 2)
            },
            "descending": {
                "dna02_yaw": round(float(conn_telem.get("yaw_rate", getattr(fly, "angular_velocity", 0.0))), 3),
                "dnp09_thrust": round(float(conn_telem.get("forward_speed", fly.speed)), 2),
                "mdn_reverse": 1.0 if conn_telem.get("state") == "REVERSE" or getattr(fly, "behavioral_state", "") == "REVERSE" else 0.0,
                "gf_escape": 1.0 if conn_telem.get("state") == "ESCAPE" or getattr(fly, "behavioral_state", "") == "ESCAPE" else 0.0
            },
            "biomechanics": {
                "tripod_gait": "TRIPOD_COORDINATED",
                "cadence_hz": round(8.0 * (fly.speed / 12.0) if fly.speed > 0 else 0.0, 1),
                "joint_angles": joint_angles,
                "cuticular_loads": loads
            },
            "neural": {
                "kc_hz": fly.circuit.encode_odor(float(stim.get("odor_a", stim.get("odor_conc", 0))), float(stim.get("odor_b", 0)))[1].tolist() if hasattr(fly, "circuit") and hasattr(fly.circuit, "encode_odor") else [0] * 120,
                "kc_trace": fly.circuit.y_kc.tolist() if hasattr(fly, "circuit") and hasattr(fly.circuit, "y_kc") else [],
                "net_valence": float(fly.circuit.forward(fly.circuit.y_kc)[2]) if hasattr(fly, "circuit") and hasattr(fly.circuit, "forward") else 0.0,
                "pam_trace": float(fly.circuit.y_dan_pam[0]) if hasattr(fly, "circuit") and hasattr(fly.circuit, "y_dan_pam") else 0.0,
                "ppl1_trace": float(fly.circuit.y_dan_ppl1[0]) if hasattr(fly, "circuit") and hasattr(fly.circuit, "y_dan_ppl1") else 0.0,
                "compass_heading": float(conn_telem.get("epg_bump_phase", getattr(fly, "compass_heading", fly.heading))),
                "epg_wedges": conn_telem.get("epg_wedges"),
            },
            "plasticity": {
                "mb_weights_mean": round(weights_mean, 4),
                "mb_weights_std": round(weights_std, 4),
                "learning_curve": self.learning_curve[-30:],
                "wp6": conn_telem.get("wp6"),
            },
            "connectome": conn_telem if conn_telem else None,
            "metrics": p_metrics
        }
        packet["activity"] = self.activity_snapshot(packet)
        return packet

    def _trial_metric(self, metrics: Dict[str, Any]) -> float:
        """The paradigm's headline score for the learning curve, scaled to about [0, 1]."""
        for key, scale in self.TRIAL_METRIC_KEYS:
            val = metrics.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                return float(val) * scale
        return None

    def _record_trial_milestone(self, step_res: Dict[str, Any], reason: str = "milestone"):
        """Records end of trial or adaptation milestone in continuous memory."""
        metrics = step_res.get("paradigm_metrics", {}) or {}
        metric_val = self._trial_metric(metrics)
        self.learning_curve.append(metric_val)
        self.active_brain.trials += 1
        # Guarded (F2): a full disk degrades the run; the trial still counts in memory.
        self._ledger("trial", trial=self.active_brain.trials, metric=metric_val,
                     metric_name=next((k for k, _ in self.TRIAL_METRIC_KEYS if k in metrics), None),
                     reason=reason, metrics=metrics, probe=self.active_brain.probe())
        ident = self.identity()
        self.trial_history.append({
            "run_id": ident.get("run_id"),
            "instance_id": ident.get("instance_id"),
            "backend": ident.get("backend"),
            "brain_id": self.active_brain.brain_id,
            "brain_trial": self.active_brain.trials,
            "trial": self.session_trial,
            "assay_trial": self.current_trial,
            "trial_known": self.active_brain.trial_clock["trial_known"],
            "paradigm": self.active_paradigm_id,
            "step": self.total_steps,
            "sim_seconds": round(self.trial_sim_time, 3),
            "reason": reason,
            "metric": metric_val,
            "timestamp": time.time()
        })
        self.current_trial += 1
        self.session_trial += 1

    def dispatch_command(self, cmd: dict) -> dict:
        """Apply an external command and acknowledge the step at which it took effect.

        While the scheduler runs, the command is queued and applied by the simulation
        thread at the next step boundary (between batches), so a command never waits
        behind an unbounded run of steps and never lands mid-step.  The reply carries
        ``ack.applied_step`` / ``ack.applied_sim_time_s`` and the measured latency.
        Without a running loop (tests, tools) it is applied directly under the lock.
        """
        received = time.perf_counter()
        if not isinstance(cmd, dict):
            return {"status": "error", "message": "Command must be a JSON object"}
        if self._loop_active():
            with self._commands_lock:
                self._command_seq += 1
                entry = {"cmd": cmd, "done": threading.Event(), "result": None,
                         "id": f"{self.run_id[:8]}-{self._command_seq}", "received": received}
                self._commands.append(entry)
            self._wake.set()
            if not entry["done"].wait(self.command_reply_wait_s):
                # A long step is running.  The command stays queued and is applied at
                # the next step boundary; its acknowledgement arrives in the stream.
                return {"status": "queued", "applied": False, "command_id": entry["id"],
                        "action": cmd.get("action", ""),
                        "step_in_progress_s": self.step_in_progress_s(),
                        "message": "Queued: applied when the current simulation step finishes"}
            return entry["result"]
        if self._loop_dead():
            return self._dispatch_with_dead_loop(cmd, received)
        with self.lock:
            result = self._apply_command(cmd)
        self._note_latency(result, received)
        return result

    REBUILD_ACTIONS = ("switch_paradigm", "switch_backend", "switch_controller")

    def _dispatch_with_dead_loop(self, cmd: dict, received: float) -> dict:
        """The loop should run but its thread is gone (F1).  Only a rebuild may run,
        and a successful one restarts the thread; anything else is refused, so no
        reply can say "ok" while nothing steps (audit F, #1-#7 recovery column)."""
        failure = self.loop_failure or {}
        reason = f"{failure.get('type', 'unknown')}: {failure.get('message', '')}"
        if cmd.get("action") not in self.REBUILD_ACTIONS:
            return {"status": "error", "applied": False, "loop_dead": True,
                    "message": f"The simulation thread has stopped ({reason}); nothing can be applied. "
                               f"Select an assay or switch the backend to rebuild and restart it, "
                               f"or restart the daemon."}
        with self.lock:
            result = self._apply_command(cmd)
        restarted = False
        if isinstance(result, dict) and result.get("status") == "ok":
            print(f"[Daemon] Restarting the stopped simulation thread after {cmd.get('action')} "
                  f"(it had stopped: {reason}).", flush=True)
            self.start()
            restarted = self.sim_thread.is_alive()
        self._note_latency(result, received)
        if isinstance(result, dict):
            result["loop_restarted"] = restarted
            if isinstance(result.get("ack"), dict):
                result["ack"]["loop_restarted"] = restarted
                result["ack"]["applied"] = bool(result["ack"].get("applied") and restarted)
        return result

    def _note_latency(self, result, received: float):
        """Record request-to-application latency and put it on the acknowledgement."""
        latency = time.perf_counter() - received
        self.command_latency.append(latency)
        if isinstance(result, dict) and isinstance(result.get("ack"), dict):
            result["ack"]["latency_ms"] = round(latency * 1e3, 3)

    def _admit_lifecycle(self, cmd, params):
        """Pure lifecycle preflight; the C2 transaction owns every accepted boundary."""
        action = cmd['action']
        try:
            require_fixed_amd(self.brain_backend, self.backend,
                              teaching=action == 'teach_brain',
                              learning=action == 'set_learning' and cmd.get('enabled', params.get('enabled')) is True)
        except UnsupportedRuntimeCapability as exc:
            return {'status': 'error', 'message': str(exc)}
        if self.last_error is not None or self._state_uncertain:
            return {'status': 'error', 'message': 'Lifecycle changes require a sound, non-halted controller'}
        if self.learning_records is None:
            return {'status': 'error', 'message': 'A durable observation recorder is required before a lifecycle change'}
        plan = dict(action=action, name=action, transition=True, lifecycle=True,
                    closes_presentation=not self._teaching_suspended, end_reason='policy_change')
        if action == 'teach_brain':
            pairs, reverse = cmd.get('pairs', params.get('pairs', 8)), cmd.get('reverse', params.get('reverse', False))
            if self.graph_mode:
                return {'status': 'error', 'message': 'Toy teaching is unavailable for graph controllers'}
            if self._teaching_suspended or self.active_brain.teaching:
                return {'status': 'error', 'message': 'Teaching is already running'}
            if type(pairs) is not int or not 1 <= pairs <= 50 or type(reverse) is not bool:
                return {'status': 'error', 'message': 'pairs must be an integer from 1 to 50 and reverse must be a boolean'}
            plan.update(pairs=pairs, reverse=reverse)
        elif action == 'return_from_teaching':
            if not self._teaching_suspended or self.active_brain.teaching:
                return {'status': 'error', 'message': 'Return requires completed teaching'}
        elif action == 'set_learning':
            enabled = cmd.get('enabled', params.get('enabled'))
            if type(enabled) is not bool:
                return {'status': 'error', 'message': 'enabled must be a boolean'}
            plan['enabled'] = enabled
        elif action == 'set_observation_policy':
            continuous = cmd.get('continuous', params.get('continuous', self.continuous))
            seconds = cmd.get('trial_seconds', params.get('trial_seconds', self.trial_length_s))
            try:
                seconds = preflight_observation_policy(self.active_paradigm_id, seconds, continuous)
            except (ValueError, TypeError, ConfigError) as exc:
                return {'status': 'error', 'message': str(exc)}
            plan.update(continuous=continuous, trial_seconds=seconds)
        elif action == 'shutdown':
            plan['end_reason'] = 'shutdown'
        else:
            return {'status': 'error', 'message': 'Unknown lifecycle request'}
        if self._teaching_suspended and action not in ('return_from_teaching', 'shutdown'):
            return {'status': 'error', 'message': 'Wait for teaching to return before changing learning or observation policy'}
        if plan['closes_presentation'] and self._observation_terminal is None:
            pending = self.observation_publication.pending_status()
            if pending['pending_count'] >= pending['capacity']:
                return {'status': 'error', 'message': 'Pending observation queue is full; lifecycle unchanged'}
        return self._begin_control_transaction(cmd, plan)

    def _collect_teaching_writes(self, operation):
        """Run the existing biological protocol while deferring its persistence only."""
        brain, events, saves = self.active_brain, [], []
        prior_log, prior_save = brain.log, brain.save
        def log(kind, **fields):
            events.append((kind, copy.deepcopy(fields)))
            return {'kind': kind, 'brain_id': brain.brain_id}
        brain.log, brain.save = log, lambda: saves.append(True)
        try:
            operation()
        finally:
            brain.log, brain.save = prior_log, prior_save
        return events, bool(saves)

    def _teaching_tick(self, dt):
        if not self.active_brain.teaching:
            return
        try:
            events, save = self._collect_teaching_writes(lambda: self.active_brain.teaching_step(dt))
        except Exception as exc:
            # Suppress locked persistence through a real transaction failure owner.
            plan = dict(action='teaching_progress', name='teaching_progress', transition=True,
                        lifecycle=True, closes_presentation=False, end_reason='policy_change')
            self._begin_control_transaction({'action': 'teaching_progress'}, dict(plan, teaching_error=exc))
            return
        if events or save:
            plan = dict(action='teaching_progress', name='teaching_progress', transition=True,
                        lifecycle=True, closes_presentation=False, end_reason='policy_change',
                        teaching_events=events, teaching_save=save)
            self._begin_control_transaction({'action': 'teaching_progress'}, plan)

    def _save_lifecycle(self, transaction):
        """Mutate under lock, persist off lock, then reuse the matching event/ACK barrier."""
        try:
            with self.lock:
                if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                    return
                if (self.identity() != transaction['identity'] or self.active_brain is not transaction['source_brain']
                        or self.error_detail is not transaction['error_before'] or self._state_uncertain):
                    raise RuntimeError('Lifecycle source changed before application')
                plan, action = transaction['plan'], transaction['plan']['action']
                if plan.get('teaching_error') is not None:
                    raise plan['teaching_error']
                events, save = plan.get('teaching_events', []), plan.get('teaching_save', False)
                if action == 'teach_brain':
                    events, save = self._collect_teaching_writes(lambda: self.active_brain.start_teaching(plan['pairs'], plan['reverse']))
                    self._teaching_suspended = True
                    self._teaching_clock_s = 0.0
                    self._teaching_prefix = transaction.get('terminal')
                    save = True
                elif action == 'set_learning':
                    require_fixed_amd(self.brain_backend, self.backend, learning=plan['enabled'])
                    self.active_brain.learning_enabled = self.arena.fly.learning_enabled = plan['enabled']
                    if self.registry is not None:
                        self.registry.learning_enabled = plan['enabled']
                    save = True
                elif action == 'set_observation_policy':
                    self.continuous, self.trial_length_s = plan['continuous'], plan['trial_seconds']
                    self._observation_policy_source = "api"
                    self._observation_policy_command_id = transaction["entry"]["id"]
                    self._observation_policy_set_at_sim_s = self.total_steps * self.dt
                if action in ('set_learning', 'set_observation_policy', 'return_from_teaching'):
                    self.segment_id = uuid.uuid4().hex
                    self._configure_observation_owner()
                    self.transition = {'reason': 'policy_change', 'step': self.total_steps}
                if action == 'shutdown':
                    self.running = False
                    self._wake.set()
                transaction.update(identity=copy.deepcopy(self.identity()), committed=True,
                                   phase='saving_control_event')
                recorder = transaction['recorder']
                if recorder is not None:
                    transaction['recording_request'] = recorder.prepare_durable_command(
                        transaction['entry']['id'], self.total_steps, copy.deepcopy(transaction['entry']['cmd']))
                terminal = transaction.get('terminal')
                event = dict(action=action, reason=plan['end_reason'], step=self.total_steps,
                    command_id=transaction['entry']['id'], identity=self.identity(), segment_id=self.segment_id,
                    learning_enabled=self.active_brain.learning_enabled, continuous=self.continuous,
                    trial_seconds=self.trial_length_s, teaching=copy.deepcopy(self.active_brain.teaching),
                    observation_key=copy.deepcopy(terminal['observation_key']) if terminal else None,
                    payload_sha256=terminal['payload_sha256'] if terminal else None,
                    suspended_observation=self._teaching_suspended)
                result = dict(status='ok', identity=self.identity(), learning_enabled=self.active_brain.learning_enabled,
                              observation_available=not self._teaching_suspended, continuous=self.continuous,
                              trial_seconds=self.trial_length_s)
                brain = self.active_brain
            for kind, fields in events:
                brain.log(kind, **fields)
            if save:
                brain.save()
            if not self._flush_validity(wait=True):
                raise RuntimeError('Required lifecycle validity evidence was not durably saved')
            self._save_assay_control_event(transaction, brain, result, event, recorder)
        except Exception as exc:
            with self.lock:
                if self._pending_assay_control is transaction and not (transaction.get('failed') or transaction.get('cancel_requested')):
                    self._fail_assay_control(exc, before_apply=False)

    def _save_shutdown(self, transaction, recorder):
        """No clean marker until retained observations and all required saves succeed."""
        if self.observation_publication.pending_status()['pending_count']:
            raise RuntimeError('Shutdown still has retained unsaved observations')
        try:
            checkpoint = self.save_checkpoint('final_shutdown')
        except Exception as exc:
            with self.lock:
                self._persistence_failed('checkpoint', exc, self.checkpoints_dir)
            raise
        with open(checkpoint, 'rb') as fh:
            os.fsync(fh.fileno())
        fd = os.open(self.checkpoints_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if recorder is not None:
            with self.lock:
                if self.recorder is not recorder or self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                    raise RuntimeError('Shutdown recording owner changed before finalisation')
            try:
                summary = recorder.close()
            except Exception as exc:
                with self.lock:
                    self._recording_failed(exc, during='finalise')
                raise
            if summary.get('status', 'complete') != 'complete':
                raise RuntimeError('Recording did not finish cleanly')
            with self.lock:
                if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                    raise RuntimeError('Shutdown owner changed during recording finalisation')
                self.recorder = None
                transaction['recording_finalised'] = True
        with self.lock:
            if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                raise RuntimeError('Shutdown owner changed before clean marker')
            marker = {'event': 'session_end', 'session': self.run_id, 'at': round(time.time(), 3)}
            self._pending_validity.append(marker)
        transaction['shutdown_marker'] = marker
        marker_saved = self._flush_validity(wait=True)
        with self.lock:
            owner_current = self._pending_assay_control is transaction and not (
                transaction.get('failed') or transaction.get('cancel_requested'))
        if not marker_saved or not owner_current:
            self._revoke_shutdown_marker(transaction)
            raise RuntimeError('Clean shutdown marker was not acknowledged; disk history may contain an unacknowledged marker')

    def _revoke_shutdown_marker(self, transaction):
        """Append a permanent revocation; never claim a failed write was durable."""
        if transaction.get('shutdown_marker') is None:
            return
        guard = transaction.setdefault('marker_revocation_lock', threading.Lock())
        with guard:
            if transaction.get('marker_revocation_durable'):
                return
            with self._validity_writer_lock:
                with self.lock:
                    marker = transaction['shutdown_marker']
                    self._pending_validity[:] = [row for row in self._pending_validity if row is not marker]
                    if 'marker_revocation' not in transaction:
                        transaction['marker_revocation'] = {
                            'event': 'session_end_failed', 'session': self.run_id,
                            'at': round(time.time(), 3)}
                        self._pending_validity.append(transaction['marker_revocation'])
            transaction['marker_revocation_durable'] = self._flush_validity(wait=True)


    def _admit_transition(self, cmd, params):
        """Validate target/flags without changing source state or freezing evidence."""
        action = cmd['action']
        target = cmd.get('paradigm', params.get('paradigm')) if action == 'switch_paradigm' else self.active_paradigm_id
        backend = cmd.get('backend', params.get('backend')) if action in ('switch_backend', 'switch_controller') else self.backend
        if type(target) is not str or target not in PARADIGMS:
            return {'status': 'error', 'message': 'Choose a known experiment'}
        if type(backend) is not str or backend not in DAEMON_BACKENDS:
            return {'status': 'error', 'message': 'Choose a supported controller backend'}
        try:
            require_fixed_amd(self.brain_backend, backend)
        except (UnsupportedRuntimeCapability, OSError, ValueError) as exc:
            return {'status': 'error', 'message': f'Target unavailable: {exc}'}
        recording_refusal = self._raw_recording_recovery_refusal()
        if recording_refusal is not None:
            return recording_refusal
        if action in ('switch_backend', 'switch_controller') and backend == self.backend:
            if self._state_uncertain:
                return {'status': 'error', 'message': 'The simulation state is uncertain; select an assay or switch to a different backend to rebuild it'}
            return {'status': 'ok', 'backend': self.backend, 'identity': self.identity(), 'cleared_error': None}
        if self.recorder is not None and backend != self.backend:
            return {'status': 'error', 'message': 'Stop the recording before switching backend (a recording covers one graph)'}
        advance, keep = cmd.get('advance', params.get('advance', True)), cmd.get('keep_memory', params.get('keep_memory', True))
        if action == 'reset_trial' and (type(advance) is not bool or type(keep) is not bool):
            return {'status': 'error', 'message': 'advance and keep_memory must be booleans'}
        if action == 'reset_trial' and (self.last_error is not None or self._state_uncertain):
            return {'status': 'error', 'message': 'Manual reset cannot recover an uncertain or halted controller; select an assay'}
        if action == 'reset_trial' and self.graph_mode and not keep:
            return {'status': 'error', 'message': 'Graph memory reset requires an explicitly supported graph-state operation'}
        if self._teaching_suspended or self.active_brain.teaching:
            return {'status': 'error', 'message': 'Finish teaching before changing the measured experiment'}
        graph = self.shared_graph
        try:
            get_backend(backend, scientific=not self.test_mode, allow_test=self.test_mode)
            if backend in GRAPH_BACKENDS:
                daemon_configuration()
                if backend == 'connectome-plastic' and (
                        graph.identity.synthetic if graph is not None else self.test_mode) and (
                        self.registry is None or self.registry.plasticity_rule is None):
                    raise ValueError('Synthetic plastic backend needs a declared plasticity rule')
        except Exception as exc:
            return {'status': 'error', 'message': f'Target unavailable: {type(exc).__name__}: {exc}'}
        if self.learning_records is None:
            return {'status': 'error', 'message': 'A durable observation recorder is required before reset or switching'}
        if self._observation_terminal is None:
            pending = self.observation_publication.pending_status()
            if pending['pending_count'] >= pending['capacity']:
                return {'status': 'error', 'message': 'Pending observation queue is full; no transition or observation was changed'}
        reason = ('manual_reset' if action == 'reset_trial' else
                  'experiment_selected' if action == 'switch_paradigm' else 'backend_switch')
        plan = dict(action=action, name=target if action == 'switch_paradigm' else backend,
                    target=target, backend=backend, graph=graph, advance=advance, keep_memory=keep,
                    transition=True, closes_presentation=True, end_reason=reason,
                    validate_graph=backend in GRAPH_BACKENDS and graph is None,
                    recovering=self.last_error is not None or self._state_uncertain)
        return self._begin_control_transaction(cmd, plan)

    def _prepare_transition_target(self, transaction):
        """Off-lock target restoration and required source saves; never release source."""
        plan = transaction['plan']
        if plan['action'] == 'reset_trial':
            return None
        backend, assay = plan['backend'], plan['target']
        require_fixed_amd(self.brain_backend, backend)
        if self.brain_backend == 'wgpu-amd':
            preflight_fixed_store(self.registry.root, assay)
        spec = get_backend(backend, scientific=not self.test_mode, allow_test=self.test_mode)
        brains = self._backend_brains.get(backend)
        if brains is None or transaction['uncertain_before']:
            directory = self.output_dir / ('brains' if backend == 'modular' else f'graph-bookkeeping/{backend}')
            brains = ExperimentBrains(directory)
            if backend == self.backend and transaction['uncertain_before']:
                brains.instances.update({k: v for k, v in self.brains.instances.items()
                                         if k != self.active_paradigm_id})
        brain = brains.get(assay)
        if self.brain_backend == 'wgpu-amd':
            brain.learning_enabled = False
        if brain.restore_error:
            raise RuntimeError(f'Target checkpoint cannot be restored: {brain.restore_error}')
        prepared = dict(backend=backend, spec=spec, brains=brains, brain=brain,
                        source=source_revision(files=spec.source_files), registry=self.registry,
                        controller=self.graph_controller, graph=plan['graph'], activation=None)
        try:
            if backend in GRAPH_BACKENDS:
                from experiment_registry import ExperimentRegistry as GraphRegistry, default_registry_dir
                registry = self.registry
                if registry is None:
                    registry = GraphRegistry(plan['graph'], Path(self.registry_root) if self.registry_root
                        else default_registry_dir(self.output_dir), test_mode=self.test_mode,
                        keep_checkpoints=self.keep_checkpoints, continue_io_state=self.continue_io_state,
                        brain_backend=self.brain_backend)
                prepared['registry'] = registry
                prepared['activation'] = registry.prepare_activation(
                    assay, backend, force_reload=transaction['uncertain_before'])
                instance = prepared['activation']['target']
                controller = self.graph_controller or GraphArenaController(
                    self, self.graph_step_ms, shared_graph=plan['graph'])
                arena = None if transaction['uncertain_before'] else self._graph_arenas.get(assay)
                if arena is None:
                    arena = Arena(paradigm=None if assay == 'open-arena' else assay, brain_type='modular',
                        seed=int(instance.seed), num_flies=1, num_predators=0, fly_ablations=[{'ablate_mb': True}],
                        controller_backend=backend, graph_controller=controller)
                if instance.world_state and arena is not transaction['source_arena']:
                    arena.restore_world(instance.world_state)
                prepared.update(arena=arena, controller=controller, manifest=instance.manifest,
                                manifest_path=registry.instance_dir(instance.instance_id) / 'manifest.json')
            else:
                manifest = RunManifest.create(backend='modular', assay=assay, instance_id=brain.brain_id,
                    seed=brain.seed, graph=None, dynamics={'model': 'modular MB + CX + surge-cast (experiment_brains/arena)',
                    'dt_s': self.dt, 'motor_assists': dict(brain.arena.motor_assists)},
                    learned_parameter_locations={'mb_weights': str(brain.path),
                        'events': str(brain.directory / f'{assay}.events.jsonl')},
                    parent_run_id=transaction['identity'].get('run_id'), source=prepared['source'])
                prepared.update(arena=brain.arena, manifest=manifest,
                                manifest_path=self.output_dir / 'manifests' / f'{manifest.run_id}.json')
            if not transaction['uncertain_before']:
                source = transaction['source_brain']
                self._sync_trial_clock(brain=source)
                source.save()
                transaction.setdefault('saved_channels', []).append('brain_save')
                if self.graph_mode and self.registry.active is not None:
                    self.registry.checkpoint(world_state=transaction['source_arena'].snapshot_world())
                    transaction['saved_channels'].append('checkpoint')
            record = brain.log('transition_prepared', command_id=transaction['entry']['id'],
                source_identity=copy.deepcopy(transaction['identity']),
                target_identity=prepared['manifest'].identity(),
                observation_key=copy.deepcopy(transaction['terminal']['observation_key']),
                payload_sha256=transaction['terminal']['payload_sha256'])
            if record is None:
                raise RuntimeError('Target preparation evidence could not be saved')
            return prepared
        except Exception:
            self._cancel_transition_target(prepared)
            raise

    @staticmethod
    def _cancel_transition_target(prepared):
        if prepared is not None and prepared.get('activation') is not None:
            prepared['registry'].cancel_activation(prepared['activation'])

    def _commit_transition(self, transaction, prepared):
        """No storage wait: apply only after source and transaction compare-and-commit."""
        plan = transaction['plan']
        if plan['action'] == 'reset_trial':
            if plan['advance']:
                self.current_trial += 1
                self.session_trial += 1
            if not plan['keep_memory']:
                self.arena.fly.circuit.reset_state(keep_memory=False)
            paradigm = self.arena.paradigm
            if paradigm is not None and hasattr(paradigm, 'reset_trial'):
                paradigm.reset_trial()
            self.arena.reset_fly_to_spawn()
            self.trial_sim_time = 0.0
            self._sync_trial_clock(new_trial=True)
            self.segment_id = uuid.uuid4().hex
            self._path.clear()
        else:
            if prepared['activation'] is not None:
                prepared['registry'].commit_activation(prepared['activation'])
            elif self.graph_mode and self.registry is not None and self.registry.active is not None:
                transaction['retired_graph'] = self.registry.active
                self.registry.active = None
            self.backend, self.backend_spec = prepared['backend'], prepared['spec']
            self.graph_mode = self.backend in GRAPH_BACKENDS
            self.shared_graph, self.registry = prepared['graph'], prepared['registry']
            self.graph_controller, self._source = prepared['controller'], prepared['source']
            self.brains = prepared['brains']
            self._backend_brains[self.backend] = self.brains
            self.active_brain, self.arena = prepared['brain'], prepared['arena']
            if self.brain_backend == 'wgpu-amd':
                self.active_brain.learning_enabled = self.arena.fly.learning_enabled = False
            if self.graph_mode:
                self.active_brain.arena = self.arena
                self._graph_arenas[plan['target']] = self.arena
            self.manifest = prepared['manifest']
            self.activation += 1
            self.manifest.record_event('activate', step=self.total_steps, activation=self.activation,
                                       daemon_run_id=self.run_id, clock_scope=SESSION_CLOCK_SCOPE)
            self.active_paradigm_id = plan['target']
            self.active_paradigm_title = getattr(self.arena.paradigm, 'name', 'Open Arena Assay')
            self._restore_trial_clock(self.active_brain,
                                      self.registry.active if self.graph_mode else None)
            self.learning_curve = self.active_brain.curve
            self.segment_id = uuid.uuid4().hex
            self._path.clear()
            self._last_step_result = {}
            self._reconcile_validity(flush=False)
        self.transition = {'reason': plan['end_reason'], 'step': self.total_steps}
        self._configure_observation_owner()
        transaction['identity'] = copy.deepcopy(self.identity())
        transaction['committed'] = True
        transaction['phase'] = 'saving_control_event'
        if self.recorder is not None:
            transaction['recording_request'] = self.recorder.prepare_durable_command(
                transaction['entry']['id'], self.total_steps, copy.deepcopy(transaction['entry']['cmd']))

    def _save_transition(self, transaction):
        prepared = None
        try:
            prepared = self._prepare_transition_target(transaction)
            with self.lock:
                if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                    return
                if (self.identity() != transaction['identity'] or self.active_brain is not transaction['source_brain']
                        or self.error_detail is not transaction['error_before']
                        or self._state_uncertain != transaction['uncertain_before']):
                    raise RuntimeError('Transition source changed during target preparation')
                self._commit_transition(transaction, prepared)
            # Required target evidence and recovery saves stay outside runner.lock.
            if prepared is not None:
                prepared['manifest'].write(prepared['manifest_path'])
            if transaction['plan']['recovering'] and (transaction['error_before'] or {}).get('channel'):
                if self.graph_mode:
                    self.registry.checkpoint(world_state=self.arena.snapshot_world())
                else:
                    self.active_brain.save()
                transaction.setdefault('saved_channels', []).append('checkpoint')
            if transaction['plan']['action'] == 'reset_trial' and not transaction['plan']['keep_memory']:
                self.active_brain.save()
            with self.lock:
                if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                    return
                for channel in transaction.get('saved_channels', []):
                    self._persisted(channel)
                cleared = None
                if transaction['plan']['recovering']:
                    refused = self._recovery_gate(transaction['error_before'], transaction['failures_before'],
                                                  verified_checkpoint=True)
                    if refused is not None:
                        raise RuntimeError(refused['message'])
                    self._state_uncertain = False
                    cleared = self._clear_error('switch_paradigm' if transaction['plan']['action'] == 'switch_paradigm'
                                                else 'switch_backend', defer_persistence=True)
                    if self.last_error is not None:
                        raise RuntimeError('Durable recorder recovery could not be committed')
                plan = transaction['plan']
                result = dict(status='ok', active_paradigm=self.active_paradigm_id, backend=self.backend,
                              current_trial=self.current_trial, identity=self.identity(), cleared_error=cleared)
                event = dict(action=plan['action'], reason=plan['end_reason'], step=self.total_steps,
                    source_identity=copy.deepcopy(transaction['terminal']['observation']['identity']),
                    observation_key=copy.deepcopy(transaction['terminal']['observation_key']),
                    payload_sha256=transaction['terminal']['payload_sha256'], identity=self.identity(),
                    segment_id=self.segment_id, keep_memory=plan['keep_memory'], advance=plan['advance'],
                    cleared_error=cleared)
            if not self._flush_validity(wait=True):
                raise RuntimeError('Required transition validity batch was not durably saved')
            if cleared is not None:
                record = self.active_brain.log('simulation_error_cleared',
                    command_id=transaction['entry']['id'], error=cleared['message'],
                    cleared_by=cleared['cleared_by'], step=self.total_steps,
                    run_id=self.run_id, segment_id=self.segment_id)
                if (type(record) is not dict or record.get('kind') != 'simulation_error_cleared'
                        or record.get('command_id') != transaction['entry']['id']
                        or record.get('brain_id') != self.active_brain.brain_id):
                    raise RuntimeError('Recovery log did not return its matching durable record')
                transaction['halt_log_saved'] = True
            self._save_assay_control_event(transaction, self.active_brain, result, event, transaction['recorder'])
        except Exception as exc:
            with self.lock:
                if self._pending_assay_control is transaction and not (transaction.get('failed') or transaction.get('cancel_requested')):
                    if transaction['plan']['recovering'] and (transaction['error_before'] or {}).get('channel'):
                        exc = RuntimeError(f'Not resumed: saving still fails ({type(exc).__name__}: {exc})')
                    self._fail_assay_control(exc, before_apply=False)
        finally:
            if prepared is not None and prepared.get('activation') is not None:
                if prepared['registry'].active is prepared['activation']['target']:
                    prepared['registry'].release_activation_source(prepared['activation'])
            if transaction.get('retired_graph') is not None:
                transaction['retired_graph'].release()
            if not transaction.get('committed'):
                self._cancel_transition_target(prepared)

    def _admit_assay_control(self, cmd, params):
        """Pure rejection, then one command-owned durability barrier. Caller locked."""
        action = cmd['action']
        name = cmd.get('name', params.get('name', ''))
        plan = {'action': action, 'name': name, 'value': None, 'closes_presentation': False}
        if action == 'place_stimulus':
            if self.arena.paradigm is not None:
                return {'status': 'error', 'message': 'Spatial editing is only available in the open arena'}
            kind = cmd.get('type', params.get('type'))
            if kind == 'wind':
                return {'status': 'error', 'message': 'Spatial wind vectors are unsupported; use windStrength for scalar airflow toward -x'}
            if kind not in ('food', 'alarm'):
                return {'status': 'error', 'message': 'This spatial stimulus is not supported by the live model'}
            try:
                values = [cmd.get(k, params.get(k)) for k in ('x', 'y')]
                if any(isinstance(v, bool) for v in values):
                    raise ValueError('coordinates must be finite numbers')
                x, y = map(float, values)
                if not all(math.isfinite(v) for v in (x, y)) or not (
                        2 <= x <= self.arena.width - 2 and 2 <= y <= self.arena.height - 2):
                    raise ValueError('Place the stimulus inside the arena')
            except (ValueError, TypeError, OverflowError) as exc:
                return {'status': 'error', 'message': str(exc)}
            plan.update(name=kind, x=x, y=y, intervention_only=True)
        else:
            checked = assay_controls.preflight(self.arena, action, name,
                                               cmd.get('value', params.get('value')))
            if not checked['accepted']:
                return {'status': 'error', 'message': checked['message']}
            plan.update(checked, value=checked['normalized_value'])
        if self.last_error is not None or self._state_uncertain or self._teaching_suspended or self.active_brain.teaching:
            return {'status': 'error', 'message': 'Assay controls require a sound, non-teaching active observation'}
        if self._observation_terminal is not None:
            return {'status': 'error', 'message': 'The current observation transition must finish before changing the assay'}
        if plan['closes_presentation']:
            if self.learning_records is None:
                return {'status': 'error', 'message': 'A durable observation recorder is required before changing presentation'}
            pending = self.observation_publication.pending_status()
            if pending['pending_count'] >= pending['capacity']:
                return {'status': 'error', 'message': 'Pending observation queue is full; no control or observation was changed'}
        return self._begin_control_transaction(cmd, plan)

    def _begin_control_transaction(self, cmd, plan):
        entry = self._executing_command_entry
        if entry is None:
            with self._commands_lock:
                self._command_seq += 1
                entry = {'cmd': copy.deepcopy(cmd), 'done': threading.Event(), 'result': None,
                         'id': f'{self.run_id[:8]}-{self._command_seq}', 'received': time.perf_counter()}
        transaction = {'entry': entry, 'plan': plan, 'started': time.monotonic(),
                       'identity': copy.deepcopy(self.identity()), 'failed': False,
                       'source_brain': self.active_brain, 'source_arena': self.arena,
                       'error_before': self.error_detail, 'failures_before': self._stopping_failures,
                       'uncertain_before': self._state_uncertain,
                       'restart_loop': self._loop_dead() and not plan.get('lifecycle')}
        self._pending_assay_control = transaction
        if plan.get('lifecycle') and self._teaching_suspended:
            transaction['terminal'] = self._teaching_prefix
        if not plan['closes_presentation']:
            self._finish_assay_control()
            if entry['done'].is_set():
                return entry['result']
            return {'status': 'queued', 'applied': False, 'command_id': entry['id'],
                    'message': 'Waiting for the intervention event to be durably saved'}
        if plan.get('validate_graph'):
            transaction['phase'] = 'validating_target'
            transaction['writer'] = threading.Thread(target=self._validate_transition_graph,
                args=(transaction,), name='NeuroFly-TargetValidation', daemon=True)
            try:
                transaction['writer'].start()
            except Exception as exc:
                self._complete_assay_control({'status': 'error', 'applied': False,
                    'message': f'Target validation could not start: {exc}'})
                return entry['result']
            return {'status': 'queued', 'applied': False, 'command_id': entry['id'],
                    'message': 'Validating target graph before freezing the source observation'}
        return self._freeze_control_prefix(transaction)

    def _validate_transition_graph(self, transaction):
        """Read/verify the graph off lock; unavailable targets never freeze source."""
        graph, error = None, None
        try:
            from experiment_registry import SharedGraph
            graph = (SharedGraph.synthetic(allow_synthetic=True) if self.test_mode
                     else SharedGraph.load_for_dynamics(self.graph_dir))
        except Exception as exc:
            error = exc
        with self.lock:
            if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                return
            if (self.identity() != transaction['identity']
                    or self.active_brain is not transaction['source_brain']
                    or self.error_detail is not transaction['error_before']
                    or self._state_uncertain != transaction['uncertain_before']):
                error = RuntimeError('Source owner changed during target validation')
            if error is not None:
                self._complete_assay_control({'status': 'error', 'applied': False,
                    'message': f'Target unavailable: {type(error).__name__}: {error}'})
                return
            transaction['plan']['graph'] = graph
            transaction['phase'] = 'saving_prefix'
            self._freeze_control_prefix(transaction)
            self._publish_due = True
            self._wake.set()

    def _freeze_control_prefix(self, transaction):
        entry, plan = transaction['entry'], transaction['plan']
        try:
            self._capture_terminal_observation(self._last_step_result,
                                               plan.get('end_reason', 're_presentation_user'))
            terminal = self._observation_terminal
            terminal['transition_owner'] = 'assay_control'
            transaction['terminal'] = terminal
            if terminal['payload_sha256'] is None or (self.last_error is not None
                                                       and not plan.get('recovering')):
                raise ObservationPublicationError('Control prefix could not be queued for durable publication')
        except Exception as exc:
            if self._observation_terminal is not None:
                self._observation_terminal['transition_owner'] = 'assay_control'
                transaction['terminal'] = self._observation_terminal
            self._fail_assay_control(exc, before_apply=True)
            return entry['result']
        self._publish_due = True
        if terminal.get('durable') is not None:
            self._finish_assay_control()
        return {'status': 'queued', 'applied': False, 'command_id': entry['id'],
                'message': 'Waiting for the exact observation to be saved before applying the control'}

    def _apply_assay_control_plan(self, plan):
        if 'refused_result' in plan:
            return copy.deepcopy(plan['refused_result']), {
                'action': 'scheduled_command_refused', 'requested_action': plan['name'],
                'applied': False, 'message': plan['refused_result']['message'],
                'step': self.total_steps, 'segment_id': self.segment_id}
        owner = self.arena.observation_owner
        action, name, value = plan['action'], plan['name'], plan['value']
        if action == 'set_param':
            assay_controls.set_parameter(self.arena, name, value)
        elif action == 'assay_action':
            assay_controls.act(self.arena, name)
        else:
            from arena import Position
            x, y = plan['x'], plan['y']
            positions, odor = ((self.arena.food_positions, self.arena.odor_a) if name == 'food'
                               else (self.arena.hazard_positions, self.arena.odor_b))
            positions.append(Position(x, y))
            odor.add_source(x, y, 1.0)
        if plan['closes_presentation']:
            if self.arena.paradigm is None:
                owner.airflow_mm_s = self.arena._open_airflow_condition()
            owner.begin_next_presentation(f'user_{name}', self.observation_config)
        if action == 'place_stimulus':
            owner.log_intervention(f'place_{name}:x_mm', plan['x'])
            owner.log_intervention(f'place_{name}:y_mm', plan['y'])
        else:
            owner.log_intervention(name, value)
        event = dict(action=action, name=name, value=value,
                     coordinates=({k: plan[k] for k in ('x', 'y')} if action == 'place_stimulus' else None),
                     intervention_only=bool(plan.get('intervention_only')),
                     run_id=self.run_id, segment_id=self.segment_id,
                     presentation_id=f"{self.segment_id}:{owner.observation_status()['presentation_index']}",
                     step=self.total_steps, sim_time_s=self.total_steps * self.dt)
        result = {'status': 'ok', 'applied': name, 'live_assay': assay_controls.describe(self.arena)}
        return result, event

    def _finish_assay_control(self):
        transaction = self._pending_assay_control
        if transaction is None or (transaction.get('failed') or transaction.get('cancel_requested')) or transaction.get('applying'):
            return False
        terminal = transaction.get('terminal')
        if (self.last_error is not None and not transaction['plan'].get('recovering')) or (transaction['plan']['closes_presentation'] and
                (terminal is None or terminal.get('durable') is None)):
            return False
        try:
            if self.identity() != transaction['identity']:
                raise RuntimeError('Control owner changed while awaiting persistence')
            transaction['applying'] = True
            if transaction['plan'].get('transition'):
                transaction['phase'] = 'preparing_transition'
                transaction['recorder'] = self.recorder
                transaction['writer'] = threading.Thread(target=(self._save_lifecycle if transaction['plan'].get('lifecycle') else self._save_transition),
                    args=(transaction,), name='NeuroFly-Transition', daemon=True)
                transaction['writer'].start()
                return True
            result, event = self._apply_assay_control_plan(transaction['plan'])
            transaction['recorder'] = self.recorder
            if self.recorder is not None:
                command = copy.deepcopy(transaction['entry']['cmd'])
                if 'refused_result' in transaction['plan']:
                    command = {'action': 'scheduled_command_refused', 'requested_command': command,
                               'applied': False, 'message': transaction['plan']['refused_result']['message']}
                transaction['recording_request'] = self.recorder.prepare_durable_command(
                    transaction['entry']['id'], self.total_steps, command)
        except Exception as exc:
            self._fail_assay_control(exc, before_apply=False)
            return False
        # The transaction remains the sampling/mutation barrier during fsync.
        transaction['phase'] = 'saving_control_event'
        transaction['writer'] = threading.Thread(
            target=self._save_assay_control_event,
            args=(transaction, self.active_brain, result, event, self.recorder),
            name='NeuroFly-ControlEvent', daemon=True)
        try:
            transaction['writer'].start()
        except Exception as exc:
            self._fail_assay_control(exc, before_apply=False)
            return False
        self._publish_due = True
        return True

    def _save_assay_control_event(self, transaction, brain, result, event, recorder):
        try:
            self._save_assay_control_event_owned(transaction, brain, result, event, recorder)
        finally:
            if transaction.get('failed') or transaction.get('cancel_requested'):
                self._revoke_shutdown_marker(transaction)

    def _save_assay_control_event_owned(self, transaction, brain, result, event, recorder):
        error = None
        channel = 'events_ledger'
        try:
            record = brain.log('assay_control', **event)
            if record is None:
                raise RuntimeError('Control event was not durably saved after mutation')
            if transaction['plan'].get('transition'):
                # A transition may create this target ledger for the first time.
                # Its directory entry must survive along with the fsynced event.
                directory_fd = os.open(brain.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            if recorder is not None:
                channel = 'recording'
                transaction['recording_receipt'] = recorder.durable_command(transaction['recording_request'])
            if transaction['plan']['action'] == 'shutdown':
                channel = 'shutdown'
                if recorder is not None:
                    recorder.validate_durable_command(transaction['recording_request'], transaction['recording_receipt'])
                self._save_shutdown(transaction, recorder)
        except Exception as exc:
            error = exc
        with self.lock:
            # Late completion cannot revive a failed command or clear invalidity.
            if self._pending_assay_control is not transaction or (transaction.get('failed') or transaction.get('cancel_requested')):
                return
            if self.identity() != transaction['identity'] or self.active_brain is not brain:
                error = RuntimeError('Control owner changed while saving the event')
            elif self.last_error is not None or self._state_uncertain:
                error = RuntimeError('Control state halted or became uncertain while saving the event')
            if recorder is not None and error is None:
                try:
                    if self.recorder is not recorder and transaction['plan']['action'] != 'shutdown':
                        raise RuntimeError('Required command recording owner changed before acknowledgement')
                    if transaction.get('recording_finalised'):
                        if getattr(recorder, 'aborted', None):
                            raise RuntimeError('Shutdown recording was aborted before acknowledgement')
                    else:
                        recorder.validate_durable_command(transaction['recording_request'],
                                                          transaction.get('recording_receipt'))
                except Exception as exc:
                    channel, error = 'recording', exc
            if error is not None:
                if channel == 'recording':
                    self._recording_failed(error)
                else:
                    self._persistence_failed(channel, error, getattr(brain, 'directory', None))
                self._fail_assay_control(error, before_apply=False)
            else:
                self._persisted('events_ledger')
                if transaction.get('halt_log_saved'):
                    self._persisted('halt_log')
                self._observation_terminal = None
                if transaction.get('restart_loop'):
                    try:
                        self.start()
                        if not self.sim_thread.is_alive():
                            raise RuntimeError('Rebuilt simulation thread did not restart')
                        result['loop_restarted'] = True
                    except Exception as exc:
                        self._fail_assay_control(exc, before_apply=False)
                        return
                self._complete_assay_control(result)
                if transaction['plan'].get('lifecycle'):
                    self.latest_telemetry = self._assemble_telemetry(self._last_step_result)
            self._publish_due = True
            self._wake.set()

    def _complete_assay_control(self, result):
        transaction = self._pending_assay_control
        if transaction is None:
            return
        entry = transaction['entry']
        result = self._ack_command_result(entry['cmd'], result)
        result['command_id'] = entry['id']
        terminal = transaction.get('terminal')
        if terminal is not None:
            result['observation_key'] = copy.deepcopy(terminal['observation_key'])
            result['payload_sha256'] = terminal['payload_sha256']
        self._note_latency(result, entry['received'])
        entry['result'] = result
        self._pending_assay_control = None
        self.command_acks.append(result)
        entry['done'].set()
        if 'refused_result' in transaction['plan']:
            self._halt_on_error(RuntimeError(result['message']), phase='scheduled_command')

    def _fail_assay_control(self, exc, *, before_apply):
        transaction = self._pending_assay_control
        if transaction is None or transaction.get('failed'):
            return
        transaction['failed'] = True
        # Revoke the captured command writer before releasing the sampling barrier.
        # _recording_failed may re-enter; failed above makes completion exactly once.
        captured_recorder = transaction.get('recorder')
        if not before_apply and captured_recorder is not None:
            if self.recorder is captured_recorder:
                self._recording_failed(exc)
            elif not getattr(captured_recorder, 'closed', False):
                try:
                    captured_recorder.abort(f'Assay control canceled: {exc}', defer_cleanup=True)
                except Exception as cleanup_exc:
                    exc = RuntimeError(f'{exc}; recording cleanup failed: {cleanup_exc}')
        if 'terminal' not in transaction and self._observation_terminal is not None:
            transaction['terminal'] = self._observation_terminal
            self._observation_terminal['transition_owner'] = 'assay_control'
        restarted = False
        if (before_apply and self.exploratory and self.last_error is None
                and (transaction.get('terminal') or {}).get('payload_sha256') is not None):
            try:
                self.arena.observation_owner.begin_next_presentation(
                    'control_save_failed_unchanged_condition', self.observation_config)
                self._observation_terminal = None
                restarted = True
            except Exception as begin_exc:
                exc = begin_exc
                before_apply = False
        if before_apply and self.exploratory and self.last_error is None and not restarted:
            exc = RuntimeError(f'Control prefix could not be retained safely: {exc}')
            before_apply = False
        if not before_apply:
            self._state_uncertain = True
            if (transaction['plan'].get('transition') and self.last_error is not None
                    and self._current_run_id() != (transaction.get('terminal') or {}).get(
                        'observation', {}).get('identity', {}).get('run_id')):
                self._invalidate('control_transition_failed', exc=exc, flush=False,
                                 extra={'source_observation_key': copy.deepcopy(
                                     (transaction.get('terminal') or {}).get('observation_key'))})
            self._halt_on_error(exc, phase='assay control', failure_class=FAILURE_SOFTWARE,
                                incident_reason='assay_control_failed')
        elif not self.exploratory and self.last_error is None:
            self._persistence_failed('learning_records', exc)
        message = (f'Requested control was NOT applied: {exc}. Exploratory observation restarted '
                   'under unchanged condition; unsaved evidence retained' if restarted else
                   f'Control failed with no applied acknowledgement; frozen prefix retained: {exc}')
        self.assay_control_failures.append({'command_id': transaction['entry']['id'],
                                           'message': message, 'restarted_unchanged': restarted,
                                           'payload_sha256': (transaction.get('terminal') or {}).get('payload_sha256'),
                                           'observation_key': copy.deepcopy((transaction.get('terminal') or {}).get('observation_key'))})
        if transaction['plan'].get('transition'):
            brain, detail = self.active_brain, copy.deepcopy(self.error_detail)
            def save_failure():
                self._revoke_shutdown_marker(transaction)
                self._flush_validity(wait=True)
                try:
                    brain.log('control_transition_failed', command_id=transaction['entry']['id'],
                              message=message, error_detail=detail)
                except Exception as log_exc:
                    with self.lock:
                        self._persistence_failed('halt_log', log_exc, brain.directory)
            transaction['failure_writer'] = threading.Thread(target=save_failure,
                name='NeuroFly-TransitionFailure', daemon=True)
            try:
                transaction['failure_writer'].start()
            except Exception as start_exc:
                message += f'; failure evidence writer could not start: {start_exc}'
        self._complete_assay_control({'status': 'error', 'applied': False, 'message': message})

    def _check_assay_control_timeout(self):
        transaction = self._pending_assay_control
        if transaction is not None and transaction.get('cancel_requested'):
            self._fail_assay_control(transaction.get('cancellation_error') or
                                     TimeoutError('Shutdown canceled'),
                                     before_apply=not transaction.get('applying'))
            return
        if transaction is not None and time.monotonic() - transaction['started'] >= self.assay_control_timeout_s:
            if transaction.get('phase') == 'validating_target':
                transaction['failed'] = True
                self._complete_assay_control({'status': 'error', 'applied': False,
                    'message': 'Target validation timed out; source observation was not changed'})
                return
            exc = TimeoutError('Timed out waiting for the control persistence barrier')
            if not transaction.get('applying'):
                pending = self.observation_publication.pending_status(limit=1)
                failed_save_wait = (not pending['active_claim'] and pending['entries']
                    and pending['entries'][0]['phase'] == 'failed'
                    and 'learning_records' in self.persistence.channels
                    and self.learning_records is not None
                    and self.learning_records.liveness()['active_write'] is None)
                # Canceling a wait is not another save attempt. Keep the actual
                # write failure/deadline, including when its retry just became due.
                # Unattempted or active writes still need timeout diagnostics.
                if not failed_save_wait:
                    self._persistence_failed('learning_records', exc)
            self._fail_assay_control(exc, before_apply=not transaction.get('applying'))

    def _apply_command(self, cmd: dict) -> dict:
        """Apply one command now. Caller holds ``self.lock``; adds the acknowledgement."""
        result = self._apply_command_unacked(cmd)
        return self._ack_command_result(cmd, result)

    def _ack_command_result(self, cmd, result):
        scoped = cmd.get("action") in ("set_param", "assay_action", "place_stimulus",
                                        "reset_trial", "teach_brain", "return_from_teaching", "set_learning",
                                        "set_observation_policy", "shutdown", "teaching_progress", *self.REBUILD_ACTIONS)
        if (self.recorder is not None and isinstance(result, dict)
                and not scoped):
            try:
                self.recorder.command(self, cmd, result)
            except Exception as exc:  # noqa: BLE001 -- a failing recording stops visibly (F2)
                self._recording_failed(exc)
        if isinstance(result, dict):
            result = dict(result)
            result["ack"] = {"action": cmd.get("action", ""), "run_id": self.run_id,
                             "applied": result.get("status") == "ok",
                             "applied_step": self.total_steps,
                             "applied_sim_time_s": round(self.total_steps * self.dt, 5),
                             "applied_sim_time_scope": SESSION_CLOCK_SCOPE,
                             "clocks": self.clock_status(),
                             "paradigm": self.active_paradigm_id,
                             # Built after the command (for a switch: after the target's
                             # brain and world snapshots are restored).  Packets whose
                             # identity is older are stale (identity_rejection).
                             "identity": self.identity(),
                             # Applied, but the simulation still does not advance.
                             "halted_by_error": self.last_error}
        self._publish_due = True
        return result

    def _switch_save_failed(self, exc: BaseException) -> None:
        """A switch refused because the outgoing instance could not be checkpointed (it is
        kept active, unreleased).  A storage or GPU failure there is a failed required save."""
        if isinstance(exc, OSError) or classify_failure(exc) == FAILURE_COMPUTE:
            self._persistence_failed("checkpoint", exc, getattr(exc, "filename", None))

    def _recovery_gate(self, before: Optional[Dict[str, Any]], failures_before: int,
                       *, verified_checkpoint=False) -> Optional[dict]:
        """After a rebuild: refuse to lift a halt when this very switch hit a stopping
        failure (a required save or a compute error), and lift a save-failure halt only
        once a checkpoint has been written (scientific mode)."""
        recording_refusal = self._raw_recording_recovery_refusal()
        if recording_refusal is not None:
            return recording_refusal
        if self.last_error is not None and (self.error_detail is not before
                                            or self._stopping_failures != failures_before):
            return {"status": "error", "message": f"Not resumed: a required save or compute step failed "
                                                  f"during the switch ({self.persistence.describe().get('summary') or self.last_error})",
                    "halted_by_error": self.last_error}
        recorder_problem, recorder_state = self._learning_recorder_problem()
        if recorder_problem is not None:
            return {"status": "error", "message": f"Not resumed: {recorder_problem}",
                    "halted_by_error": self.last_error}
        current_recorder_failure = self._recorder_watchdog_failure
        if (current_recorder_failure is not None and current_recorder_failure.get("active", True)
                and not self._learning_recorder_verified_healthy(recorder_state)):
            return {"status": "error", "message": "Not resumed: durable recorder health has not been "
                                                    "verified after its watchdog failure",
                    "halted_by_error": self.last_error}
        if (before is not None and self.last_error is not None and before.get("channel")
                and not verified_checkpoint):
            # The halt came from a save (storage, GPU copy, NaN state, bug): resume only
            # once a fresh checkpoint of the rebuilt state has been written.
            if self.checkpoint_now("recovery") is None:
                failing = self.persistence.describe()["failing"].get("checkpoint") or {}
                return {"status": "error", "halted_by_error": self.last_error,
                        "message": f"Not resumed: saving still fails ({failing.get('error', 'unknown')}). "
                                   f"Free disk space and select the assay again."}
        return None

    def _apply_command_unacked(self, cmd: dict) -> dict:
        action = cmd.get("action", "")
        # Optional legacy-compatible precondition; training UI always supplies it.
        params = cmd.get("params") if isinstance(cmd.get("params"), dict) else {}
        expected = cmd.get("expected_owner", params.get("expected_owner"))
        if "expected_owner" in cmd or "expected_owner" in params:
            actual = dict(self.identity(), brain_id=self.active_brain.brain_id)
            keys = ("daemon_run_id", "activation", "run_id", "instance_id", "assay", "backend", "brain_id")
            if (not isinstance(expected, dict) or any(key not in expected or type(expected[key]) is not type(actual.get(key))
                    or expected[key] != actual.get(key) for key in keys)):
                return {"status": "error", "applied": False, "message": "Displayed training owner is stale; refresh before issuing this command"}
        if self._pending_assay_control is not None and action not in ("set_paused", "set_speed"):
            return {"status": "error", "applied": False,
                    "message": "An assay control is waiting for its exact durable observation; retry afterward"}
        # Allow both flat arguments and nested 'params' dictionary from client libraries
        p = cmd.get("params", {})
        if not isinstance(p, dict):
            p = {}

        if action in (*self.REBUILD_ACTIONS, "reset_trial"):
            return self._admit_transition(cmd, p)

        elif action in ("probe_brain", "teach_brain") and self.graph_mode:
            return {"status": "error", "message": f"{action} acts on the modular mushroom body, which is not "
                                                  f"the controller of this {self.backend} run"}

        elif action == "probe_brain":
            probe = self.active_brain.probe()
            self._ledger("probe", probe=probe)
            return {"status": "ok", "brain_id": self.active_brain.brain_id, "probe": probe}

        elif action in ('teach_brain', 'return_from_teaching', 'set_learning', 'set_observation_policy', 'shutdown'):
            return self._admit_lifecycle(cmd, p)

        elif action == "set_paused":
            paused = cmd.get("paused", p.get("paused"))
            if not isinstance(paused, bool):
                return {"status": "error", "message": "paused must be a boolean"}
            self.paused = paused
            self.latest_telemetry = self._assemble_telemetry(self._last_step_result)
            return {"status": "ok", "paused": self.paused}

        elif action == "set_speed":
            val = cmd.get("speed") if cmd.get("speed") is not None else p.get("speed", 10.0)
            try:
                new_speed = float(val)
            except (TypeError, ValueError):
                return {"status": "error", "message": "speed must be a number"}
            if not math.isfinite(new_speed):
                return {"status": "error", "message": "speed must be finite"}
            if not SPEED_MIN <= new_speed <= SPEED_MAX:
                return {"status": "error", "message": f"speed must be between {SPEED_MIN}x and {SPEED_MAX}x"}
            self.sim_speed = new_speed
            return {"status": "ok", "sim_speed": self.sim_speed}

        elif action in ('set_param', 'assay_action'):
            return self._admit_assay_control(cmd, p)

        elif action == 'place_stimulus':
            return self._admit_assay_control(cmd, p)

        elif action == 'inject_stimulus':
            return {'status':'error','message':'This preview-only injection is not connected. Use the supported live assay controls.'}

        elif action == "record_start":
            try:
                summary = self.start_recording(name=cmd.get("name", p.get("name")),
                                               record_every=cmd.get("record_every", p.get("record_every", 1)),
                                               raster=cmd.get("raster", p.get("raster", "io")),
                                               label=cmd.get("label", p.get("label", "")))
            except (ValueError, OSError) as exc:
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            self.recording_error = None
            self._persisted("recording")  # constructor established actual frame0 capture
            return {"status": "ok", "recording": summary}

        elif action == "record_stop":
            summary = self.stop_recording()
            if summary is None:
                return {"status": "error", "message": "Not recording"}
            if summary.get("status") != "complete":
                return {"status": "error", "recording": summary,
                        "message": (self.recording_error or {}).get("message", "The recording could not be finished")}
            return {"status": "ok", "recording": summary}

        elif action == "save_checkpoint":
            path = self.checkpoint_now(cmd.get("label", "manual"))
            if path is None:
                failing = self.persistence.describe()["failing"].get("checkpoint") or {}
                return {"status": "error", "message": f"Checkpoint not saved: {failing.get('error', 'unknown error')}"}
            return {"status": "ok", "checkpoint": str(path)}

        else:
            return {"status": "error", "message": f"Unknown action: '{action}'"}

    # ------------------------------------------------------------------ activity
    MODULAR_ACTIVITY = ("KC", "PAM", "PPL1", "MB valence")

    def region_map(self):
        """Region partition of the active graph (cached per graph), None for modular."""
        if self.shared_graph is None:
            return None
        cached = getattr(self, "_region_map_cache", None)
        if cached is None or cached[0] is not self.shared_graph:
            from neurofly.recording import region_map_for
            cached = (self.shared_graph, region_map_for(self))
            self._region_map_cache = cached
        return cached[1]

    def activity_snapshot(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """Per-region activity for the Brain Activity panel and recordings.

        Graph backends: mean spike rate (Hz) per region over the last graph step.
        Modular: the model's own KC / DAN / valence signals (model units).
        """
        regions = self.region_map()
        if regions is not None:
            controller = self.graph_controller
            counts = getattr(controller, "last_counts", None)
            rates = regions.rates(counts, controller.step_ms / 1000.0) if counts is not None else None
            return {"grouping": regions.grouping, "names": regions.names, "sizes": regions.sizes.tolist(),
                    "units": "Hz", "rates": rates}
        neural = packet.get("neural") or {}
        kc = neural.get("kc_hz") or []
        rates = [round(float(np.mean(kc)), 4) if len(kc) else 0.0,
                 round(float(neural.get("pam_trace") or 0.0), 4),
                 round(float(neural.get("ppl1_trace") or 0.0), 4),
                 round(float(neural.get("net_valence") or 0.0), 4)]
        return {"grouping": "modular-circuit", "names": list(self.MODULAR_ACTIVITY), "sizes": None,
                "units": "model units", "rates": rates}

    # ------------------------------------------------------------------ recording
    def start_recording(self, name: Optional[str] = None, record_every: int = 1, raster: str = "io",
                        label: str = "", path: Optional[Path] = None) -> Dict[str, Any]:
        """Start a deterministic run recording (docs/RECORDING_FORMAT.md). Caller holds the lock."""
        from neurofly.recording import RunRecorder
        if self.recorder is not None:
            raise ValueError(f"Already recording to {self.recorder.path.name}")
        if path is None:
            stem = name or f"{time.strftime('%Y%m%dT%H%M%S')}-{self.active_paradigm_id}-{self.backend}"
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", str(stem)) or str(stem).startswith("."):
                raise ValueError("Recording name may use letters, digits, '.', '_' and '-' only")
            path = self.recordings_dir / str(stem)
        self.recorder = RunRecorder(self, path, record_every=int(record_every), raster=str(raster), label=str(label))
        return {"name": self.recorder.path.name, "start_step": self.recorder.start_step,
                "record_every": self.recorder.record_every}

    def stop_recording(self) -> Optional[Dict[str, Any]]:
        """Finish the active recording; returns its summary or None. Caller holds the lock."""
        if self.recorder is None:
            return None
        recorder = self.recorder
        try:
            # Finalisation (footer, gzip close, fsync, rename, sidecar) is part of the
            # required capture: a failure here is a failed recording, not a log line.
            summary = recorder.close()
        except Exception as exc:  # noqa: BLE001 -- classified by the save policy
            self._recording_failed(exc, during="finalise")
            artifacts = recorder.artifacts() if hasattr(recorder, "artifacts") else {}
            if self.recording_error is not None:
                self.recording_error["artifacts"] = artifacts
            return {"status": "invalid", "name": getattr(getattr(recorder, "path", None), "name", None),
                    "path": None, "frames": getattr(recorder, "frames", None),
                    "partial_kept": bool(artifacts.get("partial")), "artifacts": artifacts,
                    "recording_error": self.recording_error}
        self.recorder = None
        summary = dict(summary, status="complete")
        return summary

    def recording_status(self) -> Optional[Dict[str, Any]]:
        rec = self.recorder
        if rec is None:
            return None
        return {"name": rec.path.name, "frames": rec.frames, "start_step": rec.start_step,
                "last_step": rec.last_step, "record_every": rec.record_every}

    def save_checkpoint(self, tag: str = "periodic") -> Path:
        """Saves current continuous synaptic weights and trial ledger to disk."""
        self._sync_trial_clock()
        self.brains.save_all()
        # Labels are display text, never path fragments supplied by an API client.
        safe_tag = "".join(c for c in str(tag) if c.isalnum() or c in "_-")[:60] or "manual"
        filename = f"checkpoint_{self.active_paradigm_id}_{safe_tag}_{time.time_ns()}.json"
        target_file = self.checkpoints_dir / filename
        graph_checkpoint = None
        if self.graph_mode and self.registry.active is not None:
            # Registry checkpoint: brain transients, RNG, learned parameters AND the
            # world snapshot, written atomically and versioned.
            graph_checkpoint = str(self.registry.checkpoint(world_state=self.arena.snapshot_world()))
        data = {
            "identity": self.identity(),
            "manifest": self.manifest.to_dict() if self.manifest is not None else None,
            "graph_checkpoint": graph_checkpoint,
            "tag": str(tag),
            "brain_id": self.active_brain.brain_id,
            "brain_checkpoint": str(self.active_brain.path),
            "timestamp": time.time(),
            "uptime_sec": time.time() - self.start_time,
            "paradigm": self.active_paradigm_id,
            "total_steps": self.total_steps,
            "clock_scope": SESSION_CLOCK_SCOPE,
            "clocks": self.clock_status(),
            "trial": self.current_trial,
            "learning_curve": self.learning_curve[-100:],
            "result_validity": self.result_validity(),
            "weights_mean": self.latest_telemetry.get("plasticity", {}).get("mb_weights_mean", 0.5)
        }
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(redact_local(data), f, indent=2)   # no absolute local paths on disk
        try:
            size = target_file.stat().st_size
            if graph_checkpoint:
                size += Path(graph_checkpoint).stat().st_size
            self._last_checkpoint_bytes = size
        except OSError:
            pass
        if safe_tag in ("periodic", "final_shutdown"):
            try:
                self._prune_json_checkpoints()
            except OSError as exc:   # pruning is housekeeping; never fail a save over it
                print(f"[Daemon] checkpoint pruning failed: {exc}", file=sys.stderr)
        return target_file

    def _prune_json_checkpoints(self) -> List[Path]:
        """Retention for the daemon's JSON records, for EVERY assay (F7).

        Keeps the newest ``keep_checkpoints`` periodic records and the newest
        ``keep_shutdown_checkpoints`` final_shutdown records per assay.  Manual and
        other labelled records are never removed.  Returns the removed paths.
        """
        groups: Dict[tuple, List[tuple]] = {}
        for path in self.checkpoints_dir.glob("checkpoint_*.json"):
            for tag in ("periodic", "final_shutdown"):
                head, sep, stamp = path.stem.rpartition(f"_{tag}_")
                if sep and stamp.isdigit() and head.startswith("checkpoint_"):
                    groups.setdefault((head[len("checkpoint_"):], tag), []).append((int(stamp), path))
                    break
        removed = []
        for (_, tag), stamped in groups.items():
            keep = self.keep_checkpoints if tag == "periodic" else self.keep_shutdown_checkpoints
            if not keep:
                continue
            for _, path in sorted(stamped, reverse=True)[keep:]:
                try:
                    path.unlink()
                    removed.append(path)
                except FileNotFoundError:
                    pass
        return removed

    # Older name, kept for callers outside this file.
    _prune_periodic_checkpoints = _prune_json_checkpoints


class NeuroflyHTTPHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP Server handling REST telemetry and SSE streaming."""

    runner: ContinuousExperimentRunner = None  # Injected on startup
    # Private-mode gateway by default (no auth, unlimited clients, 30 Hz);
    # run_daemon() replaces it when --public / NEUROFLY_PUBLIC is set.
    gateway: StreamGateway = StreamGateway(StreamPolicy())

    def _set_cors_headers(self, content_type: str = "application/json"):
        self.send_header("Access-Control-Allow-Origin", self.gateway.policy.allowed_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def _status_payload(self):
        uptime = time.time() - self.runner.start_time
        # F1/F3: status reads thread liveness and step age, not only flags that a dead
        # simulation thread could never change.  Minimal runners (tests) have no health().
        health = (self.runner.health() if hasattr(self.runner, "health") else
                  {"status": "error" if self.runner.last_error else "online", "error": self.runner.last_error,
                   "halted": self.runner.last_error is not None,
                   "error_detail": getattr(self.runner, "error_detail", None)})
        payload = {
            "status": health["status"],
            "error": health["error"],
            # A halted run is not paused: nothing advances until a rebuild lifts it.
            "halted": health["halted"],
            "error_detail": health.get("error_detail"),
            "liveness": health.get("liveness"),
            "persistence": health.get("persistence"),
            "recording_error": health.get("recording_error"),
            "mode": health.get("mode"),
            "observation_validity_update": health.get("observation_validity_update"),
            "result_validity": health.get("result_validity"),
            "loop_failure": health.get("loop_failure"),
            "thread_failures": health.get("thread_failures"),
            "cleared_errors": list(getattr(self.runner, "cleared_errors", ())),
            "paused": self.runner.paused,
            "continuous": self.runner.continuous,
            "service": "Project NeuroFly Continuous Learning Daemon",
            "version": NEUROFLY_VERSION,
            "delivery": delivery_identity(),
            "uptime_sec": round(uptime, 1),
            "total_steps": self.runner.total_steps,
            "sim_speed": self.runner.sim_speed,
            "active_paradigm": self.runner.active_paradigm_id,
            "active_paradigm_title": self.runner.active_paradigm_title,
            "current_trial": self.runner.current_trial,
            "trials_completed": len(self.runner.trial_history),
            "recording": getattr(self.runner, "recording_status", lambda: None)(),
            "trial_elapsed_s": round(float(getattr(self.runner, "trial_sim_time", 0.0)), 2),
            "clocks": getattr(self.runner, "clock_status", lambda: None)(),
            "trial_length_s": getattr(self.runner, "trial_length_s", None),
            "world_bounds": list(getattr(getattr(self.runner, "arena", None), "world_bounds", ())),
            "stream": self.gateway.describe()
        }
        if hasattr(self.runner, "observation_publication_status"):
            # This endpoint is a delivery view: it must never wait behind a long
            # simulation step or inspect the mutable publication queue unlocked.
            # The SSE snapshot already contains a detached, lock-consistent copy.
            published = getattr(self.runner, "published", None)
            try:
                publication = json.loads(published.data).get("observation_publication") \
                    if published is not None else None
            except (AttributeError, TypeError, ValueError):
                publication = None
            payload["observation_publication"] = publication or {
                "unavailable": True,
                "reason": "observation publication status has not been published yet",
            }
        if hasattr(self.runner, "identity"):
            # Plain reads of the published snapshot (no lock): same identity and motor
            # provenance the stream carries.
            latest = getattr(self.runner, "latest_telemetry", None) or {}
            payload["identity"] = latest.get("identity") or self.runner.identity()
            payload["motor"] = latest.get("motor")
            payload["controller_fault"] = latest.get("controller_fault")
            payload["backend"] = getattr(self.runner, "backend", "modular")
            try:
                payload["compute"] = self.runner.compute_info()
            except Exception as exc:  # never let the status probe fail on a device query
                payload["compute"] = {"device": None, "error": f"{type(exc).__name__}: {exc}"}
            payload["controller_id"] = latest.get("controller_id", "modular")
            payload["joint_angles_rad"] = latest.get("joint_angles_rad", [0.0] * 18)
            payload["leg_contacts"] = latest.get("leg_contacts", [False] * 6)
            payload["body_position_mm"] = latest.get("body_position_mm", [0.0, 0.0, 0.5])
            payload["body_quaternion_wxyz"] = latest.get("body_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
            payload["dn_rates"] = latest.get("dn_rates", {"dna02_l": 0.0, "dna02_r": 0.0, "dnp09": 0.0, "mdn": 0.0, "gf": 0.0})
            payload["body"] = {
                "controller_id": payload["controller_id"],
                "joint_angles_rad": payload["joint_angles_rad"],
                "leg_contacts": payload["leg_contacts"],
                "body_position_mm": payload["body_position_mm"],
                "body_quaternion_wxyz": payload["body_quaternion_wxyz"],
                "dn_rates": payload["dn_rates"],
            }
        if hasattr(self.runner, "timing_snapshot"):
            payload["timing"] = self.runner.timing_snapshot()
        if hasattr(self.runner.lock, "profile"):
            payload["lock_profile"] = self.runner.lock.profile()
        return payload

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _consistent_view(self, key, build):
        """A lock-consistent view that never waits behind a long step, or a 503."""
        runner = self.runner
        if hasattr(runner, "read_view"):
            view = runner.read_view(key, build)
        else:   # minimal runners (tests, tools)
            with runner.lock:
                view = build()
        if view is None:
            self.send_response(503)
            self._set_cors_headers()
            self.send_header("Retry-After", "2")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "simulation step in progress; retry shortly",
                                         "step_in_progress_s": runner.step_in_progress_s()}).encode("utf-8"))
        return view

    def do_GET(self):
        url = self.path.split("?")[0].rstrip("/")

        if url == "" and "text/html" in self.headers.get("Accept", ""):
            index_path = PROJECT_ROOT / "web" / "index.html"
            if index_path.is_file():
                body = dashboard_index_bytes(index_path)
                self.send_response(200)
                self._set_cors_headers("text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        if url in ("", "/status", "/api/status"):
            # Plain attribute reads only: status never waits for the simulation lock,
            # so a busy simulation cannot make the dashboard's reconnect probe time out.
            try:
                resp = self._status_payload()
            except Exception as exc:  # noqa: BLE001 -- answer, never drop the connection
                self._send_json({"status": "error", "error": f"status unavailable: {type(exc).__name__}: {exc}"},
                                status=500)
                return
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))

        elif url == "/api/telemetry":
            snap = getattr(self.runner, "published", None)
            if snap is not None:
                body = snap.data
            else:
                with self.runner.lock:
                    data = self.runner.latest_telemetry
                body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            self.wfile.write(body)

        elif url == "/api/observatory":
            # One consistent view (built at a step boundary) prevents mixed experiment
            # identities during switches without waiting behind a long step.
            runner = self.runner
            view = self._consistent_view("observatory", lambda: {
                "brain": runner.active_brain.summary(details=True),
                "telemetry": runner.latest_telemetry,
                "brains": runner.brains.catalog(runner.active_paradigm_id)})
            if view is not None:
                self._send_json(dict(view, status=self._status_payload()))

        elif url == "/api/manifest":
            # Full machine-readable run manifest of the active run (exports embed it).
            runner = self.runner
            view = self._consistent_view("manifest", lambda: {
                "identity": runner.identity(),
                "manifest": runner.manifest.to_dict() if getattr(runner, "manifest", None) else None})
            if view is not None:
                self._send_json(view)

        elif url in ("/api/brains", "/api/brain"):
            runner = self.runner
            view = self._consistent_view(url, lambda: (
                runner.active_brain.summary(details=True) if url == "/api/brain"
                else {"active": runner.active_paradigm_id,
                      "brains": runner.brains.catalog(runner.active_paradigm_id)}))
            if view is not None:
                self._send_json(view)

        elif url == "/api/recordings":
            from neurofly.recording import list_recordings
            payload = {"recordings": list_recordings(self.runner.recordings_dir),
                       "active": self.runner.recording_status()}
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif url.startswith("/api/recordings/"):
            # Finished recordings only (the .partial of an active one is hidden, and a
            # file whose finalisation failed is never served as a completed recording).
            from neurofly.recording import recording_invalid_reason
            name = url[len("/api/recordings/"):]
            path = (self.runner.recordings_dir / name).resolve()
            ok = (re.fullmatch(r"[A-Za-z0-9._-]+\.nfrec", name) is not None and not name.startswith(".")
                  and path.parent == self.runner.recordings_dir.resolve() and path.is_file()
                  and recording_invalid_reason(path) is None)
            if not ok:
                self.send_response(404)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Recording not found"}')
                return
            data = path.read_bytes()
            self.send_response(200)
            # Served as the gzip file itself (no Content-Encoding): the player inflates it.
            self._set_cors_headers("application/gzip")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(data)
        elif url == "/api/paradigms":
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            all_p = [
                {"id": "open-arena", "title": "0. Open Arena Foraging"},
                {"id": "t-maze", "title": "1. T-Maze Pavlovian Conditioning"},
                {"id": "y-maze", "title": "2. Y-Maze Spontaneous Alternation"},
                {"id": "heat-maze", "title": "3. Thermal Heat-Maze Place Learning"},
                {"id": "buridan", "title": "4. Buridan's Landmark Fixation"},
                {"id": "visual-operant", "title": "5. Visual Operant Flight Simulator"},
                {"id": "wind-tunnel", "title": "6. Wind Tunnel Odor Plume Tracking"},
                {"id": "looming-escape", "title": "7. Visual Looming Giant Fiber Escape"},
                {"id": "optomotor", "title": "8. Optomotor Gaze Stabilization"},
                {"id": "gap-crossing", "title": "9. Gap Crossing & Tactile Probing"},
                {"id": "circadian-dam", "title": "10. Circadian DAM Sleep Monitor"},
                {"id": "courtship", "title": "11. Courtship Conditioning & Wing Song"},
                {"id": "labyrinth", "title": "12. Corridor Obstacle Labyrinth"},
                {"id": "multisensory-sandbox", "title": "13. Multisensory 6-Limb Benchmark"}
            ]
            self.wfile.write(json.dumps({"paradigms": all_p}).encode("utf-8"))

        elif url == "/api/stream":
            # Server-Sent Events (SSE) stream, capped and paced by the gateway
            # (unlimited clients at 30 Hz unless --public is set).
            slot = self.gateway.acquire_stream_slot()
            if not slot:
                self.send_response(503)
                self._set_cors_headers()
                self.send_header("Retry-After", "5")
                self.end_headers()
                self.wfile.write(b'{"error": "Too many stream clients; retry later"}')
                return

            self.send_response(200)
            self._set_cors_headers("text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                self._stream_snapshots()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                slot.release()

        else:
            rel_path = url.lstrip("/")
            candidate = (PROJECT_ROOT / "web" / rel_path).resolve()
            web_dir = (PROJECT_ROOT / "web").resolve()
            if candidate.is_file() and str(candidate).startswith(str(web_dir)):
                ext_map = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".json": "application/json; charset=utf-8",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".svg": "image/svg+xml",
                    ".ico": "image/x-icon",
                    ".woff": "font/woff",
                    ".woff2": "font/woff2",
                }
                content_type = ext_map.get(candidate.suffix.lower(), "application/octet-stream")
                body = candidate.read_bytes()
                self.send_response(200)
                self._set_cors_headers(content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def _stream_snapshots(self):
        """SSE delivery policy: latest-value-wins, never holding the simulation lock.

        Each tick (``stream_hz``) the newest published snapshot is written if it is
        newer than the last one sent to this client; intermediate snapshots are
        skipped, never queued, and counted as decimated.  A slow client therefore
        receives fewer, fresher frames and cannot delay the simulation or other
        clients.  With no new snapshot for one second a ``heartbeat`` event is sent,
        and a ``stream`` event reports this client's delivery counters every second.
        """
        runner = self.runner
        last_send = 0.0
        last_seq = None
        last_write = last_stats = time.monotonic()
        sent = skipped = 0
        while runner.running:
            snap = getattr(runner, "published", None)
            now = time.monotonic()
            if snap is None and not hasattr(runner, "published"):
                # Minimal runners (tests, tools) without snapshot publication.
                with runner.lock:
                    payload = runner.latest_telemetry
                if payload:
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_write = now
            elif snap is not None and snap.seq != last_seq:
                if last_seq is not None and snap.seq > last_seq + 1:
                    skipped += snap.seq - last_seq - 1
                self.wfile.write(b"id: " + str(snap.seq).encode() + b"\ndata: " + snap.data + b"\n\n")
                self.wfile.flush()
                last_seq = snap.seq
                sent += 1
                last_write = now
            elif now - last_write >= 1.0:
                beat = {"server_time": round(time.time(), 3), "seq": last_seq}
                if hasattr(runner, "step_in_progress_s"):
                    # A slow computer, not a dead daemon: say how long this step has run.
                    beat["step_in_progress_s"] = runner.step_in_progress_s()
                    beat["last_step_wall_s"] = round(runner.last_step_wall_s, 4)
                if hasattr(runner, "health"):
                    # F3: the heartbeat says whether the simulation advances, so the page
                    # can tell a dead loop from an idle one even when no frame arrives.
                    try:
                        health = runner.health()
                        # Read owner metadata from a completed packet or the detached
                        # fault notice, never from the stopped engine's world/observer.
                        notice = health.get("observation_validity_update")
                        owner_packet = getattr(runner, "latest_telemetry", None) or {}
                        beat.update(identity=copy.deepcopy((notice or owner_packet).get("identity")),
                                    brain_id=(notice or {}).get("identity", {}).get("brain_id")
                                        if notice else owner_packet.get("brain_id"),
                                    segment_id=(notice or owner_packet).get("segment_id"))
                        beat.update(step=runner.total_steps, paused=runner.paused, status=health["status"],
                                    halted=health["halted"], error=health["error"],
                                    liveness=health["liveness"], persistence=health["persistence"],
                                    mode=health["mode"], result_validity=health["result_validity"],
                                    observation_validity_update=health.get("observation_validity_update"),
                                    error_detail=health.get("error_detail"))
                    except Exception as exc:  # noqa: BLE001 -- a heartbeat must always go out
                        beat["health_error"] = f"{type(exc).__name__}: {exc}"
                self.wfile.write(f"event: heartbeat\ndata: {json.dumps(beat)}\n\n".encode("utf-8"))
                self.wfile.flush()
                last_write = now
            if now - last_stats >= 1.0 and hasattr(runner, "published"):
                stats = {"sent": sent, "decimated_snapshots": skipped, "stream_hz": self.gateway.policy.stream_hz}
                self.wfile.write(f"event: stream\ndata: {json.dumps(stats)}\n\n".encode("utf-8"))
                self.wfile.flush()
                self.gateway.record_delivery(sent, skipped)
                sent = skipped = 0
                last_stats = now
            last_send = self.gateway.pace(last_send)

    def do_POST(self):
        url = self.path.split("?")[0].rstrip("/")
        if url == "/api/command":
            # Public mode: commands need a matching bearer token (or are off).
            if not self.gateway.authorize_command(self.headers):
                self.send_response(403)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(self.gateway.command_rejection()).encode("utf-8"))
                return

            try:
                content_len = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                content_len = -1
            if content_len < 0 or content_len > MAX_COMMAND_BYTES:
                self.send_response(413)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Command body too large"}')
                return
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                cmd = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            result = self.runner.dispatch_command(cmd)
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        elif url == "/api/controller":
            try:
                content_len = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                content_len = -1
            if content_len < 0 or content_len > MAX_COMMAND_BYTES:
                self.send_response(413)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Payload too large"}')
                return
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return
            backend_target = data.get("backend")
            cmd = {"action": "switch_backend", "params": {"backend": backend_target}}
            result = self.runner.dispatch_command(cmd)
            status_code = 200 if (result.get("ack", {}).get("status") == "ok" or result.get("status") == "ok") else 400
            self.send_response(status_code)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging to keep console clean
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project NeuroFly Continuous Headless Learning Daemon")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to bind the HTTP API (default: 127.0.0.1, this machine only). "
                             "Use --host 0.0.0.0 to expose it to your local network.")
    parser.add_argument("--port", type=int, default=8769, help="Port to bind HTTP API (default: 8769)")
    parser.add_argument("--paradigm", default="multisensory-sandbox", help="Initial experimental paradigm")
    parser.add_argument("--speed", type=float, default=5.0, help="Initial simulation speed multiplier (default: 5.0x for real connectome)")
    parser.add_argument("--checkpoint-interval", type=float,
                        default=float(os.environ.get("NEUROFLY_CHECKPOINT_INTERVAL") or 60.0),
                        help="Seconds between periodic checkpoints (env: NEUROFLY_CHECKPOINT_INTERVAL; "
                             "default 60, minimum 5). For a long-running observatory 300 is recommended: "
                             "a connectome checkpoint is ~17 MB, so 30 s writes ~2 GB/hour")
    parser.add_argument("--keep-checkpoints", type=int, default=None,
                        help="Newest periodic checkpoints kept per assay; older ones are deleted "
                             "(env: NEUROFLY_KEEP_CHECKPOINTS; default 20; 0 keeps all)")
    parser.add_argument("--keep-shutdown-checkpoints", type=int, default=5,
                        help="Newest final_shutdown JSON records kept per assay (default 5; 0 keeps all)")

    failure_group = parser.add_argument_group(
        "failure handling", "What happens when saving fails or the simulation stops advancing "
        "(docs/LEARNING_OBSERVATORY.md, 'When something fails').")
    failure_group.add_argument("--exploratory", action="store_true",
                               default=os.environ.get("NEUROFLY_EXPLORATORY", "") not in ("", "0"),
                               help="Exploratory mode (env NEUROFLY_EXPLORATORY=1): when a required save "
                                    "fails (disk full, I/O error) keep stepping with status 'degraded' "
                                    "(NOT SAVING), retry with a back-off (30 s to 10 min) and record the gap; "
                                    "the run is marked incomplete. Default (scientific mode): stop the run "
                                    "at a step boundary and mark its result incomplete. GPU/compute errors "
                                    "always halt")
    failure_group.add_argument("--exit-on-stall", type=float, default=None, metavar="SECONDS",
                               help="Exit with code 70 after the simulation has not advanced (stalled or its "
                                    "thread dead) for this many seconds, so a service manager with "
                                    "Restart=always resumes from the last checkpoint. Default: off")
    failure_group.add_argument("--step-hard-limit", type=float, default=300.0, metavar="SECONDS",
                               help="One step running longer than this counts as stalled (default 300)")
    parser.add_argument("--trial-seconds", type=float, default=None,
                        help="Optional positive finite observation-window override; omitted uses the assay specification")
    parser.add_argument("--continuous", action="store_true", help="Observe continuously without automatic respawns; manual reset starts a new segment")
    parser.add_argument("--output-dir", default=None, help="Brain checkpoints and run outputs directory")
    parser.add_argument("--pid-file", default="", help="Optional path to write daemon PID file")

    backend_group = parser.add_argument_group(
        "controller backend", "Which controller drives the fly (provenance.BACKENDS). Graph backends need the "
        "prepared MaleCNS graph: --graph-dir or NEUROFLY_GRAPH_DIR=/path/to/malecns_v1.")
    backend_group.add_argument("--backend", choices=DAEMON_BACKENDS,
                               default=os.environ.get("NEUROFLY_BACKEND") or None,
                               help="Controller backend (env: NEUROFLY_BACKEND). Default: connectome-fixed "
                                    "when a verified MaleCNS graph is present, otherwise the hand-built "
                                    "modular controller; the choice and its reason are printed at startup. "
                                    "An explicit --backend always wins, and a graph backend without a "
                                    "verified graph is a startup error.")
    backend_group.add_argument("--dynamics", choices=DAEMON_DYNAMICS,
                               default=os.environ.get(DYNAMICS_ENV) or "v3",
                               help="LIF dynamics of the connectome backends (default: v3, with the v3 "
                                    "transmitter policy; env NEUROFLY_LIF_DYNAMICS). v1 brains are kept "
                                    "in outputs/registry, v2/v3 brains in outputs/registry-<version>")
    backend_group.add_argument("--brain-backend", choices=COMPUTE_BACKENDS,
                               default=os.environ.get('NEUROFLY_BRAIN_BACKEND', 'auto'),
                               help="Compute selector captured at startup (env NEUROFLY_BRAIN_BACKEND). "
                                    "wgpu-amd is explicit, requires connectome-fixed baseline v3, "
                                    "supports no learning or learned-state migration and never falls back.")
    backend_group.add_argument("--graph-dir", default=None,
                               help="Prepared graph directory (default: NEUROFLY_GRAPH_DIR, then "
                                    "<checkout>/outputs/brainlab/malecns_v1). Missing graph = startup error.")
    backend_group.add_argument("--graph-step-ms", type=float, default=None,
                               help="Simulated brain milliseconds per 20 ms arena step (default 20)")
    backend_group.add_argument("--test-slow-step-ms", type=float, default=0.0,
                               help="TEST ONLY: add about this many wall ms of real kernel work to every "
                                    "step (a private ballast brain), to reproduce a slow computer")
    backend_group.add_argument("--test-synthetic-graph", action="store_true",
                               help="TEST ONLY: run the graph backend on a small synthetic graph. The run is "
                                    "labelled SYNTHETIC in every packet and in the dashboard.")
    backend_group.add_argument("--continue-io-state", action="store_true",
                               help="Explicitly continue a saved graph brain across a graph I/O method change. "
                                    "Creates a distinct linked child run/store after validating the parent "
                                    "checkpoint; the parent is never rewritten. Without this option an "
                                    "incompatible or legacy saved method is refused before migration writes.")

    public_group = parser.add_argument_group(
        "public streaming",
        "Read-only exposure of the stream (docs/PUBLIC_STREAMING.md). The admin token is read "
        "from NEUROFLY_ADMIN_TOKEN only, never from the command line.")
    public_group.add_argument("--public", action="store_true", default=None,
                              help="Public mode: POST /api/command disabled unless a bearer token "
                                   "matching NEUROFLY_ADMIN_TOKEN is sent; SSE clients capped and "
                                   "throttled (env: NEUROFLY_PUBLIC=1)")
    public_group.add_argument("--max-stream-clients", type=int, default=None,
                              help="Cap on concurrent SSE clients, 0 = unlimited "
                                   "(env: NEUROFLY_MAX_STREAM_CLIENTS; public default 50)")
    public_group.add_argument("--stream-hz", type=float, default=None,
                              help="SSE broadcast rate in Hz (env: NEUROFLY_STREAM_HZ; "
                                   "default 30 private, 10 public)")
    public_group.add_argument("--allowed-origin", default=None,
                              help="Access-Control-Allow-Origin value (env: NEUROFLY_ALLOWED_ORIGIN; default *)")

    record_group = parser.add_argument_group(
        "durable learning records", "Append-only JSONL under the data directory (docs/DATA_SCHEMA.md).")
    record_group.add_argument("--data-dir", default=None,
                              help="Directory for trials.jsonl / telemetry_summary.jsonl "
                                   "(env: NEUROFLY_DATA_DIR; default <project>/outputs/learning)")
    record_group.add_argument("--summary-interval", type=float, default=60.0,
                              help="Seconds between telemetry summary lines (default 60)")
    record_group.add_argument("--no-record", action="store_true",
                              help="Disable the durable JSONL learning records")
    replay_group = parser.add_argument_group(
        "run recording", "Deterministic .nfrec recordings for 1x replay (docs/RECORDING_FORMAT.md).")
    replay_group.add_argument("--record", default=None, metavar="NAME",
                              help="Record from the first step to <output-dir>/recordings/NAME.nfrec "
                                   "(also: POST /api/command record_start / record_stop)")
    replay_group.add_argument("--record-every", type=int, default=1,
                              help="Record one frame every N steps (default 1: every 20 ms step)")
    replay_group.add_argument("--record-raster", choices=("none", "io", "all"), default="io",
                              help="Spike raster: IO/annotated neurons (default), all neurons, or none")
    return parser


def choose_default_backend(args) -> tuple:
    """(backend, reason) when --backend / NEUROFLY_BACKEND was not given.

    connectome-fixed when the MaleCNS graph verifies, otherwise modular.  A
    synthetic test graph run implies the graph backend it tests.
    """
    if args.backend:
        return args.backend, "requested with --backend / NEUROFLY_BACKEND"
    if args.test_synthetic_graph:
        return "connectome-fixed", "--test-synthetic-graph given (TEST ONLY synthetic graph)"
    from brainlab.graph_identity import GraphUnavailable, verify_graph
    try:
        verify_graph(Path(args.graph_dir) if args.graph_dir else None)
    except (GraphUnavailable, OSError, ValueError) as exc:
        return "modular", (
            "no verified MaleCNS graph found, so running the hand-built modular controller "
            f"({str(exc).split('. ')[0]}). For the connectome: `neurofly download-data`, then "
            "`python -m brainlab.connectome` and `python -m brainlab.prepare`, then restart")
    return "connectome-fixed", "verified MaleCNS graph found (pass --backend modular for the hand-built controller)"


def preflight_compute_startup(args):
    """Pure capability/store admission before a PID or runner output is written."""
    selected = compute_selector(args.brain_backend)
    require_fixed_amd(selected, args.backend, dynamics=args.dynamics,
                      continue_io_state=args.continue_io_state, step_ms=args.graph_step_ms)
    if selected == 'wgpu-amd':
        from experiment_registry import default_registry_dir
        root = default_registry_dir(Path(args.output_dir) if args.output_dir else PROJECT_ROOT / 'outputs',
                                    dynamics=args.dynamics)
        preflight_fixed_store(root)
    return selected


def run_daemon():
    parser = build_arg_parser()
    args = parser.parse_args()
    # argparse never checks a default against ``choices``, so the dynamics default is
    # resolved here: an environment value (NEUROFLY_LIF_DYNAMICS=v6a, v4, or any unknown
    # string) is refused before the PID file, data dir or any checkpoint is touched.
    # Validate the value argparse actually produced (an explicit or abbreviated option,
    # or the environment-derived default); never reconstruct intent from the raw flags.
    try:
        args.dynamics = daemon_configuration(args.dynamics)
        args.trial_seconds = validate_trial_seconds(args.trial_seconds)
    except UnsupportedDynamics as exc:
        parser.exit(2, f"[Daemon] Cannot start: {exc}\n")
    except ConfigError as exc:
        parser.exit(2, f"[Daemon] Cannot start: {exc}\n")
    args.backend, backend_reason = choose_default_backend(args)
    # Apply the already-validated dynamics choice before saved-manifest admission.
    # An explicit --dynamics must override the environment during this check too.
    os.environ["NEUROFLY_LIF_DYNAMICS"] = args.dynamics
    try:
        args.brain_backend = preflight_compute_startup(args)
    except (UnsupportedRuntimeCapability, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"[Daemon] Cannot start: {exc}\n")
    print(f"[Daemon] Controller backend: {args.backend} -- {backend_reason}", flush=True)

    stream_policy = StreamPolicy.from_env(
        public=args.public,
        max_stream_clients=args.max_stream_clients,
        stream_hz=args.stream_hz,
        allowed_origin=args.allowed_origin,
    )

    # PID writing if requested
    pid_path = Path(args.pid_file) if args.pid_file else (PROJECT_ROOT / "outputs" / "neurofly_daemon.pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    print("===============================================================================")
    print(f"PROJECT NEUROFLY {NEUROFLY_VERSION} — 24/7 CONTINUOUS REMOTE LEARNING DAEMON")
    print(f"PID: {os.getpid()} | API: http://{args.host}:{args.port} | Speed: {args.speed}x")
    print(f"Active Paradigm: {args.paradigm} | LIF dynamics: {args.dynamics}")
    if stream_policy.public:
        mode = "READ-ONLY (no admin token set)" if stream_policy.read_only else "token-gated commands"
        print(f"Public mode: {mode} | max SSE clients: {stream_policy.max_stream_clients or 'unlimited'}"
              f" | stream {stream_policy.stream_hz:g} Hz")
    print("===============================================================================", flush=True)

    try:
        runner = ContinuousExperimentRunner(
            initial_paradigm=args.paradigm,
            sim_speed=args.speed,
            checkpoint_interval=args.checkpoint_interval,
            keep_checkpoints=args.keep_checkpoints,
            trial_length_s=args.trial_seconds,
            continuous=args.continuous,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            backend=args.backend,
            graph_dir=Path(args.graph_dir) if args.graph_dir else None,
            test_synthetic_graph=args.test_synthetic_graph,
            graph_step_ms=args.graph_step_ms,
            exploratory=args.exploratory,
            continue_io_state=args.continue_io_state,
            standalone_scheduled_records=False,
            brain_backend=args.brain_backend,
        )
    except Exception as exc:  # noqa: BLE001 -- every startup refusal is one plain sentence (F6)
        # Missing/mismatching graph, an unusable backend or saved state that cannot be
        # restored: fail explicitly, no fallback to another controller or fresh weights.
        print(startup_refusal(args, exc), file=sys.stderr, flush=True)
        if os.environ.get("NEUROFLY_DEBUG"):
            import traceback
            traceback.print_exc()
        try:
            pid_path.unlink()
        except OSError:
            pass
        sys.exit(2)
    if runner.exploratory:
        print("[Daemon] EXPLORATORY MODE: a failed save does not stop the run; it continues NOT SAVING, "
              "the gap is recorded and the run is marked incomplete.", flush=True)
    runner.exit_on_stall_s = args.exit_on_stall
    runner.step_hard_limit_s = max(1.0, float(args.step_hard_limit))
    runner.keep_shutdown_checkpoints = max(0, int(args.keep_shutdown_checkpoints))
    ident = runner.identity()
    print(f"[Daemon] Backend: {ident.get('backend')} | label: {ident.get('label')} | "
          f"graph_sha256: {ident.get('graph_sha256')} | synthetic: {ident.get('synthetic')}", flush=True)
    compute = runner.compute_info()
    print(f"[Daemon] Brain compute: {compute['detail']} (device={compute['device']}"
          f"{', gpu=' + compute['gpu'] if compute.get('gpu') else ''})", flush=True)
    if args.record:
        with runner.lock:
            info = runner.start_recording(name=args.record, record_every=args.record_every,
                                          raster=args.record_raster)
        print(f"[Daemon] Recording to {runner.recordings_dir / info['name']}", flush=True)
    if args.test_slow_step_ms > 0:
        runner.step_hook = SlowStepBallast(args.test_slow_step_ms)
        print(f"[Daemon] TEST ONLY: every step slowed by ~{args.test_slow_step_ms:g} ms of ballast kernel work",
              flush=True)
    # Durable ownership must be ready before the first normal arena step. A recorder
    # startup failure therefore cannot leave a background simulator running.
    recorder_thread: Optional[RecorderThread] = None
    if not args.no_record:
        recorder = None
        try:
            data_dir = resolve_data_dir(PROJECT_ROOT, args.data_dir)
            recorder = LearningRecorder(data_dir, session={
                "paradigm": args.paradigm,
                "sim_speed": runner.sim_speed,
                "port": args.port,
                "public": stream_policy.public,
                "backend": runner.backend,
                "daemon_run_id": runner.run_id,
                # Manifest of the run active at start; each switch starts a run whose
                # identity is stamped on every trial line and telemetry summary.
                "manifest": runner.manifest.to_dict() if runner.manifest is not None else None,
            })
            recorder_thread = RecorderThread(runner, recorder, summary_interval=args.summary_interval)
            with runner.lock:
                runner.attach_learning_records(recorder_thread)
            recorder_thread.start()
        except Exception as exc:  # noqa: BLE001 -- startup must fail closed before stepping
            if recorder is not None:
                try:
                    recorder.close()
                except Exception:
                    pass
            print(f"[Daemon] Cannot start durable learning records: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            try:
                pid_path.unlink()
            except OSError:
                pass
            sys.exit(2)
        print(f"[Daemon] Learning records: {data_dir}", flush=True)
    runner.start()

    NeuroflyHTTPHandler.runner = runner
    NeuroflyHTTPHandler.gateway = StreamGateway(stream_policy)
    server = ThreadingHTTPServer((args.host, args.port), NeuroflyHTTPHandler)

    def _cleanup() -> bool:
        """Stop once (F5): final checkpoint, recorder flush and PID file, each guarded,
        so a failed final checkpoint never skips the rest."""
        ok = runner.stop()
        if recorder_thread is not None:
            try:
                if not recorder_thread.stop():
                    ok = False
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"[Daemon] Learning records could not be flushed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
        try:
            if pid_path.exists():
                pid_path.unlink()
        except OSError:
            pass
        return ok

    def _signal_handler(signum, frame):
        print(f"\n[Daemon] Received signal {signum}. Initiating graceful shutdown...", flush=True)
        ok = _cleanup()
        # Do not call server.shutdown() here: the handler runs on the thread
        # that is inside serve_forever(), and shutdown() would wait forever for
        # that loop to exit.  SystemExit unwinds serve_forever() instead and the
        # finally block below closes the socket.
        sys.exit(0 if ok else 1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    exit_code = 0
    try:
        print(f"[Daemon] HTTP API & SSE stream ready at http://{args.host}:{args.port}/", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        if not _cleanup():          # no-op after the signal handler ran it
            exit_code = exit_code or 1
        server.server_close()
        print("[Daemon] Clean shutdown complete." if not exit_code else
              f"[Daemon] Shutdown complete, but not everything was saved (exit code {exit_code}).", flush=True)
    if exit_code:
        sys.exit(exit_code)


def startup_refusal(args, exc: BaseException) -> str:
    """One plain sentence (path plus what to do) for a daemon that cannot start (F6)."""
    name = type(exc).__name__
    text = str(exc) or name
    out_dir = args.output_dir or "outputs"
    if name in ("GraphUnavailable", "BackendError"):
        return f"[Daemon] Cannot start backend {args.backend!r}: {name}: {text}"
    if name in ("CheckpointCorrupt", "IncompatibleCheckpoint"):
        return (f"[Daemon] Cannot start: saved state in {out_dir} cannot be restored ({text}). Nothing was "
                f"changed. To start this assay fresh, move that output directory aside or pass a new "
                f"--output-dir; set NEUROFLY_DEBUG=1 for the traceback.")
    if isinstance(exc, ValueError):   # includes json.JSONDecodeError and 'Cannot restore <path>'
        return (f"[Daemon] Cannot start: a saved state file is unreadable ({name}: {text}). Nothing was "
                f"changed. Restore that file from a backup, or move {out_dir} aside to start fresh; set "
                f"NEUROFLY_DEBUG=1 for the traceback.")
    if isinstance(exc, OSError):
        where = f" ({exc.filename})" if getattr(exc, "filename", None) else ""
        return (f"[Daemon] Cannot start: {exc.strerror or text}{where}. Check that the path exists, is "
                f"readable and that the disk is not full; set NEUROFLY_DEBUG=1 for the traceback.")
    return f"[Daemon] Cannot start backend {args.backend!r}: {name}: {text}"


if __name__ == "__main__":
    run_daemon()
