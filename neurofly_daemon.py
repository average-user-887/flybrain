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
from experiment_brains import ExperimentBrains
# Controller identity (WP4).  experiment_registry / brainlab are imported only when
# a graph backend is selected, so the modular default never touches the graph.
from provenance import GRAPH_BACKENDS, RunManifest, get_backend, resolve_keep_checkpoints, source_revision

DAEMON_BACKENDS = ("modular",) + tuple(GRAPH_BACKENDS)


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

    The optomotor assay is the one assay with a verified sensory encoder and motor
    decoder (WP5, ``brainlab/io_map.py``, docs/WP5_OPTOMOTOR.md): eye-specific
    retinal slip drives T4/T5, and the DNa02 rate difference decodes to a yaw
    command that is applied as given.  Nothing else is injected -- no engineered
    assistance, so there is no graph-derived forward drive and the tethered assay
    runs at speed 0.

    Every other assay has no verified mapping, so no sensory drive is delivered and
    no motor command is decoded: the fly receives zero speed and zero yaw, reported
    as ``motor_source='graph-unmapped-io'``.  The same applies when the optomotor
    map cannot be resolved (synthetic test graph, missing annotations): the state is
    named and the fly is halted, never read as a zero yaw command.  No surrogate
    controller or engineered assist moves the fly instead.
    """

    UNMAPPED = "graph-unmapped-io"
    OPTOMOTOR_ASSAY = "optomotor"

    def __init__(self, runner: "ContinuousExperimentRunner", step_ms: float):
        self.runner = runner
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
        graph = getattr(self.runner, "shared_graph", None)
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

    # The live graph loop injects nothing but the WP5 encoder's drive: the DNp01
    # looming injection and the tonic DNb01 drive of the RPC server are engineered
    # assistance and are not used here (WP5 section 8).
    ENGINEERED_ASSISTANCE_ENABLED = False

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
            return {"halted": True, "forward_speed": 0.0, "yaw_rate": 0.0, "motor_source": "halted-no-instance",
                    "controller_fault": "no active graph instance for this assay", "state": "HALTED"}
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
                            "dnp09": 0.0, "mdn": 0.0, "gf": 0.0,
                        },
                        "epg_wedges": epg_wedges,
                        "epg_bump_phase": bump_phase,
                        "wp6": {
                            "mean_delta": round(wp6_mean_delta, 6),
                            "max_delta": round(wp6_max_delta, 6),
                            "n_edges": len(getattr(instance, "plastic_edges", [])),
                        },
                        "engineered_assistance_enabled": self.ENGINEERED_ASSISTANCE_ENABLED,
                        "engineered_assistance_applied": [],
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
            return {"halted": True, "forward_speed": 0.0, "yaw_rate": 0.0, "motor_source": self.UNMAPPED,
                    "controller_fault": None, "state": "NO-MOTOR-MAP",
                    "graph_step": instance.step_index, "graph_step_ms": self.step_ms,
                    "total_spikes": self.last_total_spikes,
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
            for idx in self.sensory_indices.get("orn_food", ()):
                if idx < n: currents[idx] += i_food

        danger_stim = float(sensory.get("mean_b", sensory.get("odor_b", 0.0)))
        if danger_stim > 0.001:
            i_danger = float(min(45.0, danger_stim * 40.0))
            for idx in self.sensory_indices.get("orn_danger", ()):
                if idx < n: currents[idx] += i_danger

        # 1. Visual lateral (photoreceptors) & Ring neurons (ER)
        contrast = float(kwargs.get("visual_contrast", kwargs.get("optomotor_contrast", sensory.get("stripe_contrast", 1.0))))
        retina_l = sensory.get("retina_photoreceptors_l")
        retina_r = sensory.get("retina_photoreceptors_r")
        if retina_l is not None and len(retina_l) > 0:
            mean_l = float(np.mean(retina_l))
            for idx in self.sensory_indices.get("visual_l", ()):
                if idx < n: currents[idx] += mean_l * 20.0 * contrast
        if retina_r is not None and len(retina_r) > 0:
            mean_r = float(np.mean(retina_r))
            for idx in self.sensory_indices.get("visual_r", ()):
                if idx < n: currents[idx] += mean_r * 20.0 * contrast

        # Central Complex Ring neurons (ER4d/ER2) driven by landmarks / Buridan stripes
        stripes = sensory.get("stripe_bearings", kwargs.get("landmarks"))
        if stripes is not None or "stripe_contrast" in sensory or "stripe_fixation" in sensory:
            i_er = float(min(35.0, 20.0 * contrast))
            for idx in self.sensory_indices.get("er_ring", ()):
                if idx < n: currents[idx] += i_er
            # Tonic modulatory drive for octopaminergic EL neurons during active visual navigation
            for idx in self.sensory_indices.get("el_modulator", ()):
                if idx < n: currents[idx] += 18.0

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
            for idx in self.sensory_indices.get("visual_looming", ()):
                if idx < n: currents[idx] += i_loom

        # 3. Courtship pheromone (DA1 cVA)
        cva_stim = float(sensory.get("cva_concentration", kwargs.get("cva_odor", 0.0)))
        if cva_stim > 0.001:
            i_cva = float(min(35.0, cva_stim * 30.0))
            for idx in self.sensory_indices.get("courtship_cva", ()):
                if idx < n: currents[idx] += i_cva

        # 4. Wind mechanoreception (Johnston's organ drag)
        wind_speed = float(sensory.get("wind_speed", 0.0))
        if wind_speed > 3.0:
            i_wind = float(min(35.0, wind_speed * 0.15))
            for idx in self.sensory_indices.get("jon_wind", ()):
                if idx < n: currents[idx] += i_wind

        # 5. Thermosensory receptors (TRN)
        temp = float(sensory.get("temperature", kwargs.get("temperature", 24.0)))
        if abs(temp - 24.0) > 2.0:
            i_temp = float(min(40.0, abs(temp - 24.0) * 2.5))
            for idx in self.sensory_indices.get("thermo_receptors", ()):
                if idx < n: currents[idx] += i_temp

        # 6. Baseline exploratory drive (tonic BPN/DNb01 current)
        for idx in self.dn_indices.get("dnb01", ()):
            if idx < n: currents[idx] += 12.0

        # Step the graph instance
        result = instance.step(currents, self.step_ms)
        counts = result.counts
        self.last_counts = counts
        self.last_total_spikes = int(counts.sum())

        # Decode descending neuron firing rates (Hz)
        sec = self.step_ms / 1000.0
        spk_dna02_l = sum(counts[i] for i in self.dn_indices.get("dna02_l", ()) if i < n)
        spk_dna02_r = sum(counts[i] for i in self.dn_indices.get("dna02_r", ()) if i < n)
        dna02_rate_l = float(spk_dna02_l / sec)
        dna02_rate_r = float(spk_dna02_r / sec)

        spk_dnp09 = sum(counts[i] for i in self.dn_indices.get("dnp09", ()) if i < n)
        dnp09_n = max(1, len(self.dn_indices.get("dnp09", [])))
        dnp09_rate = float(spk_dnp09 / (dnp09_n * sec))

        spk_dnb01 = sum(counts[i] for i in self.dn_indices.get("dnb01", ()) if i < n)
        dnb01_n = max(1, len(self.dn_indices.get("dnb01", [])))
        bpn_rate = float(spk_dnb01 / (dnb01_n * sec))

        spk_mdn = sum(counts[i] for i in self.dn_indices.get("mdn", ()) if i < n)
        mdn_n = max(1, len(self.dn_indices.get("mdn", [])))
        mdn_rate = float(spk_mdn / (mdn_n * sec))

        spk_dnp01 = sum(counts[i] for i in self.dn_indices.get("dnp01", ()) if i < n)

        # Steering & forward drive
        yaw_rate = 0.02 * (dna02_rate_l - dna02_rate_r)
        forward_speed = min(35.0, max(0.0, dnp09_rate * 1.5 + bpn_rate * 0.4 + 5.0))
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
                "dna02_l": round(dna02_rate_l, 2),
                "dna02_r": round(dna02_rate_r, 2),
                "dnp09": round(dnp09_rate, 2),
                "mdn": round(mdn_rate, 2),
                "gf": round(float(spk_dnp01 / sec), 2),
            },
            "epg_wedges": [round(w, 2) for w in epg_wedges],
            "epg_bump_phase": round(bump_phase, 4),
            "wp6": {
                "mean_delta": round(wp6_mean_delta, 6),
                "max_delta": round(wp6_max_delta, 6),
                "n_edges": len(getattr(instance, "plastic_edges", [])),
            },
            "engineered_assistance_enabled": self.ENGINEERED_ASSISTANCE_ENABLED,
            "engineered_assistance_applied": [],
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


class PersistenceMonitor:
    """What the daemon failed to save, since when, and when it retries (audit F, F2).

    A failed save never stops the simulation (unless --halt-on-persistence-failure):
    the run keeps stepping, visibly degraded.  Each channel (checkpoint, trial ledger,
    halt bookkeeping, recording, ...) is tracked separately, so a later successful
    write on one channel cannot hide a failure on another.  The periodic checkpoint
    retries with a back-off (30 s doubling to 10 min) instead of on every step.
    """

    BACKOFF_START_S = 30.0
    BACKOFF_MAX_S = 600.0

    def __init__(self):
        self._lock = threading.Lock()
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.last_ok: Dict[str, float] = {}      # channel -> wall time of the last good write
        self.total_failures = 0

    def failed(self, channel: str, exc: BaseException, path: Any = None, now: Optional[float] = None) -> Dict[str, Any]:
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

    # Paradigm telemetry flags that mark a natural end of a trial (goal reached, choice
    # made, escape fired). Paradigms without such an endpoint (multisensory sandbox,
    # optomotor, courtship, circadian...) end on time: the paradigm's own TrialManager
    # duration or ``trial_length_s`` of simulated time, whichever comes first.
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
        trial_length_s: float = 60.0,
        continuous: bool = False,
        backend: str = "modular",
        graph_dir: Optional[Path] = None,
        test_synthetic_graph: bool = False,
        graph_step_ms: Optional[float] = None,
        shared_graph: Any = None,
        registry_root: Optional[Path] = None,
        keep_checkpoints: Optional[int] = None
    ):
        self.output_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "outputs")
        self.graph_dir = graph_dir
        self.registry_root = registry_root
        # Newest checkpoints kept per graph instance and per assay (0 keeps all).
        self.keep_checkpoints = resolve_keep_checkpoints(keep_checkpoints)
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
                                          keep_checkpoints=self.keep_checkpoints)
            self.graph_controller = GraphArenaController(
                self, self.graph_step_ms)
            # Bookkeeping (curves, event logs) for graph runs never shares files with
            # the modular brains: modular weights are not graph weights.
            self.brains = ExperimentBrains(self.output_dir / "graph-bookkeeping" / backend)
        else:
            self.brains = ExperimentBrains(self.output_dir / "brains")
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
        self.halt_on_persistence_failure = False   # --halt-on-persistence-failure
        self.disk_reserve_bytes = 512 * 1024 * 1024
        self._last_checkpoint_bytes: Optional[int] = None
        self.keep_shutdown_checkpoints = 5          # final_shutdown JSON records kept per assay
        self.recording_error: Optional[Dict[str, Any]] = None
        self.learning_records = None                # RecorderThread, set by run_daemon
        self.loop_failure: Optional[Dict[str, Any]] = None
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
        # Simulated seconds after which a trial ends even without a natural endpoint
        self.trial_length_s = max(1.0, float(trial_length_s))

        # Runtime state
        self.start_time = time.time()
        self.total_steps = 0
        self.current_trial = 1
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

        # Initialize primary simulation arena
        self._init_arena(self.active_paradigm_id)

    def _switch_backend(self, target_backend: str):
        """Switch controller backend between modular, connectome-fixed, and connectome-plastic under lock."""
        if target_backend == self.backend:
            return
        if target_backend not in DAEMON_BACKENDS:
            raise ValueError(f"Unknown backend {target_backend!r}; choose one of {DAEMON_BACKENDS}")

        target_graph_mode = target_backend in GRAPH_BACKENDS
        if target_graph_mode:
            from experiment_registry import (ExperimentRegistry as GraphRegistry, SharedGraph,
                                             default_registry_dir)
            if self.shared_graph is None:
                self.shared_graph = (SharedGraph.synthetic(allow_synthetic=True) if self.test_mode
                                     else SharedGraph.load_for_dynamics(self.graph_dir))
            if self.registry is None:
                self.registry = GraphRegistry(self.shared_graph,
                                              Path(self.registry_root) if self.registry_root else default_registry_dir(self.output_dir),
                                              test_mode=self.test_mode,
                                              keep_checkpoints=self.keep_checkpoints)
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
        # Resolve first so an invalid request cannot disturb the active experiment.
        brain = self.brains.get(paradigm_name)
        if self.graph_mode:
            # Checkpoint the outgoing instance WITH its world, activate the target
            # (brain snapshot restored), then restore the target's world.  Only after
            # both are ready does this return, so the switch ack names a ready instance.
            active = self.registry.active
            if active is not None and hasattr(self, "arena"):
                active.world_state = self.arena.snapshot_world()
            instance = self.registry.activate(paradigm_name, self.backend)
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
            try:
                manifest.write(self.output_dir / "manifests" / f"{manifest.run_id}.json")
            except OSError as exc:
                print(f"[Daemon] manifest write failed: {exc}", file=sys.stderr)
        if hasattr(self, "active_brain"):
            self.active_brain.elapsed = self.trial_sim_time
            # The outgoing brain stays cached in memory (self.brains), so a failed
            # save here loses nothing yet: it is reported and retried by the next
            # checkpoint's save_all, and the switch goes ahead.
            self._persist("brain_save", self.active_brain.save, path=self.active_brain.path)
        self.manifest = manifest
        self.activation += 1
        manifest.record_event("activate", step=getattr(self, "total_steps", 0), activation=self.activation,
                              daemon_run_id=self.run_id)
        self.segment_id = uuid.uuid4().hex
        self.transition = {"reason": "experiment_selected", "step": self.total_steps}
        self.active_brain = brain
        self.arena = brain.arena
        self.active_paradigm_id = paradigm_name
        self.active_paradigm_title = getattr(self.arena.paradigm, "name", "Open Arena Assay")
        self.trial_sim_time = brain.elapsed
        self.learning_curve = brain.curve
        self._path.clear()
        self._last_step_result = {}
        self._publish_due = True
        self.latest_telemetry = self._assemble_telemetry({})

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
                           "wall_gap_mm", "near_wall", "in_contact", "tethered", "halted")

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

    def stop(self) -> bool:
        """Stop the simulation and save the final checkpoint.  Runs once (F5).

        Returns False when the final checkpoint or the recording could not be saved.
        The caller still completes its own cleanup (recorder flush, PID file) and only
        then exits non-zero.
        """
        with self._stop_lock:
            if self._stopped:
                return self._stop_ok
            self._stopped = True
        print("[Daemon] Stopping simulation loop...", flush=True)
        self.running = False
        self._wake.set()
        self._watchdog_stop.set()
        if hasattr(self, "sim_thread"):
            self.sim_thread.join(timeout=3.0)
        ok = True
        # A step that never returns may still hold the lock: do not wait for it forever.
        if not self.lock.acquire(timeout=10.0):
            print("[Daemon] Final checkpoint skipped: the simulation step did not finish within 10 s "
                  "(its state would be inconsistent). The last periodic checkpoint is kept.",
                  file=sys.stderr, flush=True)
            self._stop_ok = False
            return False
        try:
            try:
                summary = self.stop_recording()
                if summary is not None:
                    print(f"[Daemon] Recording saved: {summary['path']} ({summary['frames']} frames)", flush=True)
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"[Daemon] Recording could not be finished: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            if self.checkpoint_now("final_shutdown", check_space=False) is None:
                ok = False
                entry = self.persistence.describe()["failing"].get("checkpoint") or {}
                print(f"[Daemon] Final checkpoint NOT saved: {entry.get('error')}. The last good checkpoint "
                      f"is kept and will be resumed.", file=sys.stderr, flush=True)
        finally:
            self.lock.release()
        self._stop_ok = ok
        return ok

    # ------------------------------------------------------------------ scheduling
    def _can_step(self) -> bool:
        if self.paused or self.last_error:
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
        if exc is None:
            self.loop_failure = {"type": "LoopExit", "message": "the simulation loop returned while it should run",
                                 "phase": self._loop_phase, "at": round(time.time(), 3), "traceback": []}
        else:
            self.loop_failure = {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__,
                                 "phase": self._loop_phase, "at": round(time.time(), 3),
                                 "traceback": _traceback_tail(exc)}
        print(f"[Daemon] SIMULATION THREAD STOPPED during {self.loop_failure['phase']}: "
              f"{self.loop_failure['type']}: {self.loop_failure['message']}", file=sys.stderr, flush=True)

    def _note_thread_failure(self, name, exc_type, exc_value, exc_tb) -> None:
        """threading.excepthook backstop: an uncaught exception in a NeuroFly thread."""
        entry = {"thread": name, "type": getattr(exc_type, "__name__", str(exc_type)),
                 "message": str(exc_value), "at": round(time.time(), 3)}
        self.thread_failures.append(entry)
        if name == "NeuroFly-SimLoop" and self.loop_failure is None:
            self.loop_failure = dict(entry, phase=self._loop_phase, traceback=[])

    def _loop_dead(self) -> bool:
        """The loop should be running (``running``) but its thread has ended."""
        thread = getattr(self, "sim_thread", None)
        return bool(self.running and thread is not None and not thread.is_alive())

    def _advance_one(self):
        """One scheduled step: step-indexed commands first, then the fixed-dt tick. Holds lock."""
        for entry in self._scheduled.pop(self.total_steps, ()):
            entry["result"] = self._apply_command(entry["cmd"])
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
        error, halted = self.last_error, self.last_error is not None
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
        return {"status": status, "error": error, "halted": halted, "liveness": live,
                "persistence": persistence, "recording_error": self.recording_error,
                "loop_failure": self.loop_failure, "thread_failures": list(self.thread_failures)}

    def persistence_summary(self) -> Dict[str, Any]:
        extra = {}
        records = self.learning_records
        if records is not None and hasattr(records, "describe"):
            info = records.describe()
            if info.get("failing"):
                extra["learning_records"] = {"channel": "learning_records", "state": "failing",
                                             "error": info.get("last_error"), "errno": info.get("last_errno"),
                                             "since": info.get("failing_since"), "failures": info.get("errors"),
                                             "path": info.get("data_dir")}
        return self.persistence.describe(extra)

    def _watchdog_loop(self) -> None:
        """NeuroFly-Watchdog: logs stall onset; optionally exits so systemd restarts."""
        previous, bad_since = None, None
        while not self._watchdog_stop.wait(self.watchdog_interval_s):
            try:
                if not self.running:
                    break                       # stopped; start() launches a new watchdog
                live = self.liveness()
                state = live["state"]
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

    def schedule_command(self, step: int, cmd: dict) -> dict:
        """Apply ``cmd`` exactly when ``total_steps == step`` (before that step's successor).

        Step-indexed interventions make runs at different requested speeds comparable.
        Returns the entry; its ``result`` (with ``ack.applied_step``) is filled once applied.
        """
        entry = {"cmd": dict(cmd), "result": None}
        with self.lock:
            if step < self.total_steps:
                raise ValueError(f"step {step} is already in the past (now {self.total_steps})")
            self._scheduled.setdefault(int(step), []).append(entry)
        return entry

    def _drain_commands(self):
        """Apply HTTP commands queued since the last batch, at a step boundary."""
        while True:
            with self._commands_lock:
                if not self._commands:
                    return
                entry = self._commands.popleft()
            try:
                with self.lock:
                    # A long command (a graph instance switch) is work, not a stall.
                    self._command_started = time.perf_counter()
                    try:
                        entry["result"] = self._apply_command(entry["cmd"])
                    except Exception as exc:  # never kill the loop over one bad command
                        entry["result"] = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
                    finally:
                        self._command_started = None
                        self._steppable_since = time.perf_counter()
                    self._note_latency(entry["result"], entry["received"])
                    if isinstance(entry["result"], dict):
                        entry["result"]["command_id"] = entry["id"]
                        self.command_acks.append(entry["result"])
            finally:
                entry["done"].set()   # the HTTP caller never waits on a command that failed

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
        if self.paused or self.last_error:
            # Keep the last measured assay metrics visible while nothing advances.
            self.latest_telemetry = self._assemble_telemetry(self._last_step_result)
            return {}

        # Teaching runs in an explicit cue chamber while the behavioral arena pauses.
        if self.active_brain.teaching:
            self.active_brain.teaching_step(step_dt)
            self.total_steps += 1
            self._last_step_result = {}
            if publish:
                self.latest_telemetry = self._assemble_telemetry({})
            if self.recorder is not None:
                self._capture_recording({})
            return {}

        # 1. Step simulation arena
        self._loop_phase = "step"
        try:
            step_result = self.arena.step(step_dt)
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
        pos = self.arena.fly.pos
        self._path.append((self.total_steps, round(float(pos.x), 4), round(float(pos.y), 4)))

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

        # 4. Periodic checkpoint.  A failed save never stops the run (F2).
        self._maybe_periodic_checkpoint()
        self._loop_phase = "step"
        return step_result

    # ------------------------------------------------------------------ persistence (F2)
    def _persist(self, channel: str, fn, *args, path: Any = None, **kwargs):
        """One durable write.  A failure degrades the run (reported, retried later)
        instead of stopping it; with --halt-on-persistence-failure it halts honestly."""
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- every write failure is reported, none is fatal
            self._persistence_failed(channel, exc, path)
            return None
        self.persistence.succeeded(channel)
        return result

    def _ledger(self, kind: str, **fields) -> None:
        """Append one event to the active brain's ledger (events.jsonl), guarded."""
        channel = "trial_ledger" if kind == "trial" else "events_ledger"
        brain = self.active_brain
        self._persist(channel, brain.log, kind, path=getattr(brain, "directory", None), **fields)

    def _persistence_failed(self, channel: str, exc: BaseException, path: Any = None) -> None:
        entry = self.persistence.failed(channel, exc, path)
        if entry["failures"] == 1 or entry["failures"] % 20 == 0:
            print(f"[Daemon] NOT SAVING ({channel}, failure {entry['failures']}): {entry['error']}. The "
                  f"simulation keeps running; next attempt in {entry['backoff_s']:.0f} s. Free disk space "
                  f"if the disk is full.", file=sys.stderr, flush=True)
        if self.halt_on_persistence_failure and self.last_error is None:
            self._halt_on_error(exc, phase="persistence", channel=channel)

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
        self.persistence.succeeded("checkpoint")
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

    def _recording_failed(self, exc: BaseException) -> None:
        recorder, self.recorder = self.recorder, None
        name = getattr(getattr(recorder, "path", None), "name", None)
        try:
            if recorder is not None and hasattr(recorder, "abort"):
                recorder.abort(f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 -- the recording is already lost; report it
            pass
        self.recording_error = {"recording": name, "error": f"{type(exc).__name__}: {exc}",
                                "errno": getattr(exc, "errno", None), "step": self.total_steps,
                                "frames": getattr(recorder, "frames", None), "at": round(time.time(), 3),
                                "message": f"Recording {name} stopped: it could not be written "
                                           f"({type(exc).__name__}: {exc}). The simulation keeps running."}
        self._persistence_failed("recording", exc, getattr(recorder, "path", None))
        print(f"[Daemon] {self.recording_error['message']}", file=sys.stderr, flush=True)

    HALT_RECOVERY = ("The simulation is halted by this error and does not advance. Select an assay "
                     "(selecting the same one retries it) or switch the controller backend to rebuild "
                     "the controller and resume. Pausing, resuming or changing speed does not lift it.")
    PHASE_LABELS = {"step": "simulation step", "publish": "telemetry publication",
                    "persistence": "saving", "command": "command", "recording": "recording",
                    "trial bookkeeping": "trial bookkeeping", "loop": "simulation loop"}

    def _halt_on_error(self, exc: BaseException, phase: str = "step", channel: Optional[str] = None):
        """Record the failure that halts the simulation (F1). Caller holds the lock.

        Never raises: its own bookkeeping write is guarded, so a full disk cannot turn
        an honest halt into a dead simulation thread (audit F, #7).  A second failure
        while already halted is counted, not allowed to overwrite the first cause.
        """
        if self.last_error is not None:
            if self.error_detail is not None:
                self.error_detail["repeats"] = self.error_detail.get("repeats", 0) + 1
            return
        registry = self.registry if self.graph_mode else None
        active = registry.active if registry is not None else None
        message = str(exc) or type(exc).__name__
        if phase != "step":
            message = f"{self.PHASE_LABELS.get(phase, phase)} failed: {type(exc).__name__}: {message}"
        self.last_error = message
        self.error_detail = {
            "message": self.last_error, "type": type(exc).__name__, "phase": phase, "step": self.total_steps,
            "sim_time_s": round(self.total_steps * self.dt, 5), "paradigm": self.active_paradigm_id,
            "backend": self.backend, "instance_id": active.instance_id if active is not None else None,
            "at": round(time.time(), 3), "recover": self.HALT_RECOVERY,
            "traceback": _traceback_tail(exc)}
        if channel:
            self.error_detail["channel"] = channel
        self._publish_due = True
        print(f"[Daemon] HALTED ({phase}): {message}", file=sys.stderr, flush=True)
        try:
            self._ledger("simulation_error", error=self.last_error, step=self.total_steps,
                         run_id=self.run_id, segment_id=getattr(self, "segment_id", None),
                         error_type=self.error_detail["type"], phase=phase)
        except Exception:  # noqa: BLE001 -- _ledger is guarded; this is belt and braces
            pass

    def _clear_error(self, cleared_by: str) -> Optional[Dict[str, Any]]:
        """Lift a halt after a successful rebuild; the cleared error is kept and logged."""
        if self.last_error is None:
            return None
        detail = dict(self.error_detail or {"message": self.last_error})
        detail.update(cleared_by=cleared_by, cleared_at=round(time.time(), 3), cleared_step=self.total_steps,
                      resumed_paradigm=self.active_paradigm_id, resumed_backend=self.backend)
        detail.pop("recover", None)
        detail.pop("traceback", None)
        self.cleared_errors.append(detail)
        self.last_error = None
        self.error_detail = None
        self._publish_failures = 0
        self._steppable_since = time.perf_counter()
        self._ledger("simulation_error_cleared", error=detail["message"], cleared_by=cleared_by,
                     step=self.total_steps, run_id=self.run_id, segment_id=self.segment_id)
        print(f"[Daemon] Halt lifted by {cleared_by}: {detail['message']}", flush=True)
        self._publish_due = True
        return detail

    def _trial_end_reason(self, step_result: Dict[str, Any]) -> Optional[str]:
        """Why the current trial is over, or None while it continues."""
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
        if self.trial_sim_time >= self.trial_length_s:
            return "time_limit"
        return None

    def _end_trial(self, step_result: Dict[str, Any], reason: str):
        """Record the milestone, reset the paradigm's trial state and respawn the fly.
        Mushroom-body weights and other plasticity are kept: learning is continuous."""
        self.transition = {"reason": reason, "step": self.total_steps, "ended_segment": self.segment_id,
                           "terminal_pose": {"x": self.arena.fly.pos.x, "y": self.arena.fly.pos.y},
                           "terminal_metrics": step_result.get("paradigm_metrics", {})}
        self._record_trial_milestone(step_result, reason)
        self.segment_id = uuid.uuid4().hex
        paradigm = getattr(self.arena, "paradigm", None)
        if paradigm is not None and hasattr(paradigm, "reset_trial"):
            try:
                paradigm.reset_trial()
            except Exception as reset_err:
                print(f"[Daemon] paradigm.reset_trial failed: {reset_err}", file=sys.stderr)
        self.arena.reset_fly_to_spawn()
        self.trial_sim_time = 0.0
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
            "error": self.last_error,
            "halted": self.last_error is not None,
            "error_detail": self.error_detail,
            # Audit F: is the simulation advancing, and is it saving?  The page shows
            # "NOT ADVANCING" and "NOT SAVING" from these (docs/DATA_SCHEMA.md).
            "status": health["status"],
            "liveness": health["liveness"],
            "persistence": health["persistence"],
            "recording_error": health["recording_error"],
            "sim_time_s": round(self.total_steps * self.dt, 5),
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
        self._persist("brain_save", self.active_brain.save, path=self.active_brain.path)
        ident = self.identity()
        self.trial_history.append({
            "run_id": ident.get("run_id"),
            "instance_id": ident.get("instance_id"),
            "backend": ident.get("backend"),
            "brain_id": self.active_brain.brain_id,
            "brain_trial": self.active_brain.trials,
            "trial": self.current_trial,
            "paradigm": self.active_paradigm_id,
            "step": self.total_steps,
            "sim_seconds": round(self.trial_sim_time, 3),
            "reason": reason,
            "metric": metric_val,
            "timestamp": time.time()
        })
        self.current_trial += 1

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

    def _apply_command(self, cmd: dict) -> dict:
        """Apply one command now. Caller holds ``self.lock``; adds the acknowledgement."""
        result = self._apply_command_unacked(cmd)
        if self.recorder is not None and isinstance(result, dict):
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
                             "paradigm": self.active_paradigm_id,
                             # Built after the command (for a switch: after the target's
                             # brain and world snapshots are restored).  Packets whose
                             # identity is older are stale (identity_rejection).
                             "identity": self.identity(),
                             # Applied, but the simulation still does not advance.
                             "halted_by_error": self.last_error}
        self._publish_due = True
        return result

    def _apply_command_unacked(self, cmd: dict) -> dict:
        action = cmd.get("action", "")
        # Allow both flat arguments and nested 'params' dictionary from client libraries
        p = cmd.get("params", {})
        if not isinstance(p, dict):
            p = {}

        if action == "switch_paradigm":
            target = cmd.get("paradigm") or p.get("paradigm", "open-arena")
            try:
                self._init_arena(target)
            except (ValueError, OSError, RuntimeError) as exc:
                # RuntimeError covers BackendError, GraphUnavailable and checkpoint errors.
                # A failed switch rebuilt nothing, so a standing halt stays.
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            # The target's instance, world and arena are now active: lift a halt.
            cleared = self._clear_error("switch_paradigm")
            return {"status": "ok", "active_paradigm": self.active_paradigm_id, "identity": self.identity(),
                    "cleared_error": cleared}

        elif action in ("switch_backend", "switch_controller"):
            target = cmd.get("backend") or p.get("backend")
            if self.recorder is not None and target != self.backend:
                return {"status": "error", "message": "Stop the recording before switching backend "
                                                      "(a recording covers one graph)"}
            if target not in DAEMON_BACKENDS:
                return {"status": "error", "message": f"Unknown backend {target!r}; choose one of {DAEMON_BACKENDS}"}
            previous = self.backend
            try:
                self._switch_backend(target)
            except Exception as exc:
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            # Only a real change rebuilds the controller; re-selecting it is a no-op.
            cleared = self._clear_error("switch_backend") if self.backend != previous else None
            return {"status": "ok", "backend": self.backend, "identity": self.identity(),
                    "cleared_error": cleared}

        elif action in ("probe_brain", "teach_brain") and self.graph_mode:
            return {"status": "error", "message": f"{action} acts on the modular mushroom body, which is not "
                                                  f"the controller of this {self.backend} run"}

        elif action == "probe_brain":
            probe = self.active_brain.probe()
            self._ledger("probe", probe=probe)
            return {"status": "ok", "brain_id": self.active_brain.brain_id, "probe": probe}

        elif action == "teach_brain":
            try:
                self.active_brain.start_teaching(cmd.get("pairs", 8), cmd.get("reverse", False))
            except ValueError as exc:
                return {"status": "error", "message": str(exc)}
            return {"status": "ok", "brain_id": self.active_brain.brain_id}

        elif action == "set_learning":
            enabled = cmd.get("enabled")
            if not isinstance(enabled, bool):
                return {"status": "error", "message": "enabled must be a boolean"}
            if self.active_brain.teaching:
                return {"status": "error", "message": "Wait for teaching to finish before changing its control"}
            self.active_brain.learning_enabled = enabled
            self.arena.fly.learning_enabled = enabled
            if self.registry is not None:
                self.registry.learning_enabled = enabled
            self._ledger("learning_control", enabled=enabled)
            self._persist("brain_save", self.active_brain.save, path=self.active_brain.path)
            return {"status": "ok", "learning_enabled": enabled}

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
            self.sim_speed = max(0.1, min(100.0, new_speed))
            return {"status": "ok", "sim_speed": self.sim_speed}

        elif action in ('set_param', 'assay_action'):
            name = cmd.get('name', p.get('name', ''))
            try:
                if action == 'set_param':
                    value = assay_controls.set_parameter(self.arena, name, cmd.get('value', p.get('value')))
                else:
                    assay_controls.act(self.arena, name)
                    value = None
            except (ValueError, TypeError) as exc:
                return {'status':'error','message':str(exc)}
            self._ledger('intervention', action=action, name=name, value=value,
                                  run_id=self.run_id, segment_id=self.segment_id, sim_time_s=self.total_steps*self.dt)
            return {'status':'ok','applied':name,'live_assay':assay_controls.describe(self.arena)}

        elif action == 'place_stimulus':
            if self.arena.paradigm is not None:
                return {'status':'error','message':'Spatial editing is only available in the open arena'}
            from arena import Position
            kind = cmd.get('type', p.get('type'))
            try:
                x,y = float(cmd.get('x',p.get('x'))),float(cmd.get('y',p.get('y')))
                if not all(math.isfinite(v) for v in (x,y)) or not (2<=x<=self.arena.width-2 and 2<=y<=self.arena.height-2):
                    raise ValueError('Place the stimulus inside the arena')
                if kind=='food':
                    self.arena.food_positions.append(Position(x,y)); self.arena.odor_a.add_source(x,y,1.0)
                elif kind=='alarm':
                    self.arena.hazard_positions.append(Position(x,y)); self.arena.odor_b.add_source(x,y,1.0)
                elif kind=='wind':
                    angle=math.atan2(y-self.arena.height/2, x-self.arena.width/2)
                    self.arena.wind=(-15*math.cos(angle),-15*math.sin(angle))
                else: raise ValueError('This spatial stimulus is not supported by the live model')
            except (ValueError,TypeError) as exc:
                return {'status':'error','message':str(exc)}
            self._ledger('spatial_intervention', stimulus=kind,x=x,y=y,run_id=self.run_id,step=self.total_steps)
            return {'status':'ok','applied':kind}

        elif action == 'inject_stimulus':
            return {'status':'error','message':'This preview-only injection is not connected. Use the supported live assay controls.'}

        elif action == "reset_trial":
            advance = cmd.get("advance", p.get("advance", True))
            keep_mem = cmd.get("keep_memory", p.get("keep_memory", True))
            if advance:
                self.current_trial += 1
            if not keep_mem and hasattr(self.arena.fly, "circuit"):
                self.arena.fly.circuit.reset_state(keep_memory=False)
                self._ledger("memory_reset")
                self._persist("brain_save", self.active_brain.save, path=self.active_brain.path)
            # Reset paradigm trial state and return the fly to the paradigm spawn
            paradigm = getattr(self.arena, "paradigm", None)
            if paradigm is not None and hasattr(paradigm, "reset_trial"):
                paradigm.reset_trial()
            self.arena.reset_fly_to_spawn()
            self.trial_sim_time = 0.0
            self.segment_id = uuid.uuid4().hex
            self.transition = {"reason": "manual_reset", "step": self.total_steps}
            self._path.clear()
            self._ledger("manual_reset", segment_id=self.segment_id)
            self.latest_telemetry = self._assemble_telemetry({})
            return {"status": "ok", "current_trial": self.current_trial}

        elif action == "record_start":
            try:
                summary = self.start_recording(name=cmd.get("name", p.get("name")),
                                               record_every=cmd.get("record_every", p.get("record_every", 1)),
                                               raster=cmd.get("raster", p.get("raster", "io")),
                                               label=cmd.get("label", p.get("label", "")))
            except (ValueError, OSError) as exc:
                return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            self.recording_error = None
            self.persistence.clear("recording")
            return {"status": "ok", "recording": summary}

        elif action == "record_stop":
            summary = self.stop_recording()
            if summary is None:
                return {"status": "error", "message": "Not recording"}
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
        recorder, self.recorder = self.recorder, None
        return recorder.close() if recorder is not None else None

    def recording_status(self) -> Optional[Dict[str, Any]]:
        rec = self.recorder
        if rec is None:
            return None
        return {"name": rec.path.name, "frames": rec.frames, "start_step": rec.start_step,
                "last_step": rec.last_step, "record_every": rec.record_every}

    def save_checkpoint(self, tag: str = "periodic") -> Path:
        """Saves current continuous synaptic weights and trial ledger to disk."""
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
            "trial": self.current_trial,
            "learning_curve": self.learning_curve[-100:],
            "weights_mean": self.latest_telemetry.get("plasticity", {}).get("mb_weights_mean", 0.5)
        }
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
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
                   "halted": self.runner.last_error is not None})
        payload = {
            "status": health["status"],
            "error": health["error"],
            # A halted run is not paused: nothing advances until a rebuild lifts it.
            "halted": health["halted"],
            "error_detail": getattr(self.runner, "error_detail", None),
            "liveness": health.get("liveness"),
            "persistence": health.get("persistence"),
            "recording_error": health.get("recording_error"),
            "loop_failure": health.get("loop_failure"),
            "thread_failures": health.get("thread_failures"),
            "cleared_errors": list(getattr(self.runner, "cleared_errors", ())),
            "paused": self.runner.paused,
            "continuous": self.runner.continuous,
            "service": "Project NeuroFly Continuous Learning Daemon",
            "version": NEUROFLY_VERSION,
            "uptime_sec": round(uptime, 1),
            "total_steps": self.runner.total_steps,
            "sim_speed": self.runner.sim_speed,
            "active_paradigm": self.runner.active_paradigm_id,
            "active_paradigm_title": self.runner.active_paradigm_title,
            "current_trial": self.runner.current_trial,
            "trials_completed": len(self.runner.trial_history),
            "recording": getattr(self.runner, "recording_status", lambda: None)(),
            "trial_elapsed_s": round(float(getattr(self.runner, "trial_sim_time", 0.0)), 2),
            "trial_length_s": getattr(self.runner, "trial_length_s", None),
            "world_bounds": list(getattr(getattr(self.runner, "arena", None), "world_bounds", ())),
            "stream": self.gateway.describe()
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
                self.send_response(200)
                self._set_cors_headers("text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(index_path.read_bytes())
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
            # Finished recordings only (the .partial of an active one is hidden).
            name = url[len("/api/recordings/"):]
            path = (self.runner.recordings_dir / name).resolve()
            ok = (re.fullmatch(r"[A-Za-z0-9._-]+\.nfrec", name) is not None and not name.startswith(".")
                  and path.parent == self.runner.recordings_dir.resolve() and path.is_file())
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
                self.send_response(200)
                self._set_cors_headers(content_type)
                self.end_headers()
                self.wfile.write(candidate.read_bytes())
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
                        beat.update(step=runner.total_steps, paused=runner.paused, status=health["status"],
                                    halted=health["halted"], error=health["error"],
                                    liveness=health["liveness"], persistence=health["persistence"])
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
    failure_group.add_argument("--halt-on-persistence-failure", action="store_true",
                               help="Halt the simulation when a save fails. Default: keep stepping, report "
                                    "status 'degraded' (NOT SAVING) and retry with a back-off (30 s to 10 min)")
    failure_group.add_argument("--exit-on-stall", type=float, default=None, metavar="SECONDS",
                               help="Exit with code 70 after the simulation has not advanced (stalled or its "
                                    "thread dead) for this many seconds, so a service manager with "
                                    "Restart=always resumes from the last checkpoint. Default: off")
    failure_group.add_argument("--step-hard-limit", type=float, default=300.0, metavar="SECONDS",
                               help="One step running longer than this counts as stalled (default 300)")
    parser.add_argument("--trial-seconds", type=float, default=60.0,
                        help="Simulated seconds per trial for paradigms without a natural endpoint (default: 60)")
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
    backend_group.add_argument("--dynamics", choices=("v1", "v2", "v3"),
                               default=os.environ.get("NEUROFLY_LIF_DYNAMICS") or "v3",
                               help="LIF dynamics of the connectome backends (default: v3, with the v3 "
                                    "transmitter policy; env NEUROFLY_LIF_DYNAMICS). v1 brains are kept "
                                    "in outputs/registry, v2/v3 brains in outputs/registry-<version>")
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


def run_daemon():
    args = build_arg_parser().parse_args()
    args.backend, backend_reason = choose_default_backend(args)
    print(f"[Daemon] Controller backend: {args.backend} -- {backend_reason}", flush=True)
    # Process-wide, before any Brain or registry manifest is created.
    os.environ["NEUROFLY_LIF_DYNAMICS"] = args.dynamics

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
    runner.halt_on_persistence_failure = bool(args.halt_on_persistence_failure)
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
    runner.start()

    # Durable learning records: a poller thread that never touches the sim loop.
    recorder_thread: Optional[RecorderThread] = None
    if not args.no_record:
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
        runner.learning_records = recorder_thread   # its write errors appear in status (F2)
        recorder_thread.start()
        print(f"[Daemon] Learning records: {data_dir}", flush=True)

    NeuroflyHTTPHandler.runner = runner
    NeuroflyHTTPHandler.gateway = StreamGateway(stream_policy)
    server = ThreadingHTTPServer((args.host, args.port), NeuroflyHTTPHandler)

    def _cleanup() -> bool:
        """Stop once (F5): final checkpoint, recorder flush and PID file, each guarded,
        so a failed final checkpoint never skips the rest."""
        ok = runner.stop()
        if recorder_thread is not None:
            try:
                recorder_thread.stop()
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
