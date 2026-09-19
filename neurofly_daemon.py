#!/usr/bin/env python3
"""
Project NeuroFly (v1.0-release) — Continuous Background Learning Daemon
========================================================================
Runs 24/7 headless biological simulation and continuous online learning
on a workstation, a server, or a laptop; it needs only the Python standard library and NumPy.

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
5. Zero External Dependencies: Pure Python 3.12 standard library + NumPy.
6. Public mode (--public / NEUROFLY_PUBLIC=1): read-only stream for untrusted
   viewers -- POST /api/command needs a bearer token equal to NEUROFLY_ADMIN_TOKEN,
   SSE clients are capped and the stream is throttled (stream_gateway.py).
7. Durable learning records: append-only trials.jsonl / telemetry_summary.jsonl
   under NEUROFLY_DATA_DIR or outputs/learning (learning_recorder.py).
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

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
from learning_recorder import LearningRecorder, RecorderThread, resolve_data_dir
from experiment_brains import ExperimentBrains


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
        continuous: bool = False
    ):
        self.output_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "outputs")
        self.brains = ExperimentBrains(self.output_dir / "brains")
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.telemetry_dir = self.output_dir / "telemetry"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.continuous = bool(continuous)
        self.paused = False
        self.last_error = None
        self.run_id = uuid.uuid4().hex
        self.transition = None
        self.lock = threading.Lock()
        self.running = False
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

    def _init_arena(self, paradigm_name: str):
        """Activate an experiment's own arena and learned state; never share weights."""
        # Resolve first so an invalid request cannot disturb the active experiment.
        brain = self.brains.get(paradigm_name)
        if hasattr(self, "active_brain"):
            self.active_brain.elapsed = self.trial_sim_time
            self.active_brain.save()
        self.segment_id = uuid.uuid4().hex
        self.transition = {"reason": "experiment_selected", "step": self.total_steps}
        self.active_brain = brain
        self.arena = brain.arena
        self.active_paradigm_id = paradigm_name
        self.active_paradigm_title = getattr(self.arena.paradigm, "name", "Open Arena Assay")
        self.trial_sim_time = brain.elapsed
        self.learning_curve = brain.curve
        self.latest_telemetry = self._assemble_telemetry({})

    def start(self):
        """Starts background continuous execution thread."""
        self.running = True
        self.sim_thread = threading.Thread(target=self._run_loop, daemon=True, name="NeuroFly-SimLoop")
        self.sim_thread.start()
        print(f"[Daemon] Simulation loop started (Speed: {self.sim_speed}x, Paradigm: {self.active_paradigm_id}).", flush=True)

    def stop(self):
        """Stops simulation and saves final checkpoint."""
        print("[Daemon] Stopping simulation loop...", flush=True)
        self.running = False
        if hasattr(self, "sim_thread"):
            self.sim_thread.join(timeout=3.0)
        self.save_checkpoint("final_shutdown")

    def _run_loop(self):
        """High-speed continuous integration and online learning loop."""
        step_dt = self.dt
        base_interval = step_dt / self.sim_speed

        while self.running:
            loop_start = time.perf_counter()

            with self.lock:
                self.step_once()

            # Sleep to match target speed pacing
            elapsed = time.perf_counter() - loop_start
            sleep_time = (step_dt / self.sim_speed) - elapsed
            if sleep_time > 0.0005:
                time.sleep(sleep_time)

    def step_once(self) -> Dict[str, Any]:
        """One simulation tick plus trial bookkeeping. Caller holds ``self.lock``."""
        step_dt = self.dt
        if self.paused or self.last_error:
            self.latest_telemetry = self._assemble_telemetry({})
            return {}

        # Teaching runs in an explicit cue chamber while the behavioral arena pauses.
        if self.active_brain.teaching:
            self.active_brain.teaching_step(step_dt)
            self.total_steps += 1
            self.latest_telemetry = self._assemble_telemetry({})
            return {}

        # 1. Step simulation arena
        try:
            step_result = self.arena.step(step_dt)
        except Exception as step_err:
            print(f"[Daemon] Exception in arena.step: {step_err}", file=sys.stderr)
            self.last_error = str(step_err)
            self.active_brain.log("simulation_error", error=self.last_error)
            self.latest_telemetry = self._assemble_telemetry({})
            return {}

        self.total_steps += 1
        self.active_brain.steps += 1
        self.trial_sim_time += step_dt

        # 2. Trial advancement: natural endpoint or time limit
        end_reason = self._trial_end_reason(step_result)
        if end_reason is not None:
            self._end_trial(step_result, end_reason)
            # Terminal outcomes belong to the old segment, never to the respawn pose.
            step_result = {}

        # 3. Assemble telemetry snapshot
        self.latest_telemetry = self._assemble_telemetry(step_result)

        # 4. Periodic Checkpointing
        now = time.time()
        if now - self.last_checkpoint_time >= self.checkpoint_interval:
            self.save_checkpoint("periodic")
            self.last_checkpoint_time = now
        return step_result

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

    def _assemble_telemetry(self, step_res: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs standardized JSON telemetry packet for browser streaming."""
        fly = self.arena.fly
        stim = step_res.get("stimuli", {})
        if self.arena.paradigm is None:
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

        return {
            "type": "telemetry",
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "transition": self.transition,
            "continuous": self.continuous,
            "paused": self.paused,
            "error": self.last_error,
            "sim_time_s": round(self.total_steps * self.dt, 5),
            "brain_id": self.active_brain.brain_id,
            "brain": self.active_brain.summary(),
            "timestamp": round(time.time(), 3),
            "step": self.total_steps,
            "paradigm": self.active_paradigm_id,
            "paradigm_title": self.active_paradigm_title,
            "sim_speed": self.sim_speed,
            "stimuli": stim,
            "assay_state": step_res.get("paradigm_telemetry", {}),
            "scene": {
                "food": [p.to_tuple() for p in self.arena.food_positions],
                "hazards": [p.to_tuple() for p in self.arena.hazard_positions],
                "predators": [[p.pos.x, p.pos.y, p.get_velocity()[0], p.get_velocity()[1]] for p in self.arena.predators],
                **{name: getattr(self.arena.paradigm, name) for name in
                   ('female_pos', 'female_type', 'refuge_pos', 'refuge_radius', 'drum_angle_deg')
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
                "wind_x": round(float(self.arena.wind[0]), 2),
                "wind_y": round(float(self.arena.wind[1]), 2)
            },
            "descending": {
                "dna02_yaw": round(float(getattr(fly, "angular_velocity", 0.0)), 3),
                "dnp09_thrust": round(float(fly.speed), 2),
                "mdn_reverse": 1.0 if getattr(fly, "behavioral_state", "") == "REVERSE" else 0.0,
                "gf_escape": 1.0 if getattr(fly, "behavioral_state", "") == "ESCAPE" else 0.0
            },
            "biomechanics": {
                "tripod_gait": "TRIPOD_COORDINATED",
                "cadence_hz": round(8.0 * (fly.speed / 12.0) if fly.speed > 0 else 0.0, 1),
                "joint_angles": joint_angles,
                "cuticular_loads": loads
            },
            "neural": {
                "kc_hz": fly.circuit.encode_odor(float(stim.get("odor_a", stim.get("odor_conc", 0))), float(stim.get("odor_b", 0)))[1].tolist(),
                "kc_trace": fly.circuit.y_kc.tolist(),
                "net_valence": fly.circuit.forward(fly.circuit.y_kc)[2],
                "pam_trace": float(fly.circuit.y_dan_pam[0]),
                "ppl1_trace": float(fly.circuit.y_dan_ppl1[0]),
                "compass_heading": float(getattr(fly, "compass_heading", fly.heading)),
            },
            "plasticity": {
                "mb_weights_mean": round(weights_mean, 4),
                "mb_weights_std": round(weights_std, 4),
                "learning_curve": self.learning_curve[-30:]
            },
            "metrics": p_metrics
        }

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
        self.active_brain.log("trial", trial=self.active_brain.trials, metric=metric_val,
                              metric_name=next((k for k, _ in self.TRIAL_METRIC_KEYS if k in metrics), None),
                              reason=reason, metrics=metrics, probe=self.active_brain.probe())
        self.active_brain.save()
        self.trial_history.append({
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
        """Processes external REST intervention commands with thread-safety."""
        action = cmd.get("action", "")
        # Allow both flat arguments and nested 'params' dictionary from client libraries
        p = cmd.get("params", {})
        if not isinstance(p, dict):
            p = {}

        with self.lock:
            if action == "switch_paradigm":
                target = cmd.get("paradigm") or p.get("paradigm", "open-arena")
                try:
                    self._init_arena(target)
                except (ValueError, OSError) as exc:
                    return {"status": "error", "message": str(exc)}
                return {"status": "ok", "active_paradigm": self.active_paradigm_id}

            elif action == "probe_brain":
                probe = self.active_brain.probe()
                self.active_brain.log("probe", probe=probe)
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
                self.active_brain.log("learning_control", enabled=enabled)
                self.active_brain.save()
                return {"status": "ok", "learning_enabled": enabled}

            elif action == "set_paused":
                paused = cmd.get("paused", p.get("paused"))
                if not isinstance(paused, bool):
                    return {"status": "error", "message": "paused must be a boolean"}
                self.paused = paused
                self.latest_telemetry = self._assemble_telemetry({})
                return {"status": "ok", "paused": self.paused}

            elif action == "set_speed":
                val = cmd.get("speed") if cmd.get("speed") is not None else p.get("speed", 10.0)
                new_speed = float(val)
                self.sim_speed = max(0.1, min(100.0, new_speed))
                return {"status": "ok", "sim_speed": self.sim_speed}

            elif action == "inject_stimulus":
                stim_type = cmd.get("type") or p.get("type", "")
                val = cmd.get("value") if cmd.get("value") is not None else p.get("value", 1.0)
                if stim_type == "optogenetic_dna02":
                    self.arena.fly.angular_velocity += float(val)
                elif stim_type == "optogenetic_dnp09":
                    self.arena.fly.speed = float(val) * 20.0
                elif stim_type == "gf_looming":
                    self.arena.fly.behavioral_state = "ESCAPE"
                    self.arena.fly.speed = 35.0
                elif stim_type == "thermal_flash":
                    if hasattr(self.arena.paradigm, "state"):
                        self.arena.paradigm.state["temp"] = float(val)
                elif stim_type == "odor_puff":
                    if hasattr(self.arena, "odor_a"):
                        self.arena.odor_a.add_source(self.arena.fly.pos.x, self.arena.fly.pos.y, float(val))
                return {"status": "ok", "injected": stim_type}

            elif action == "set_param":
                name = cmd.get("name") or p.get("name", "")
                value = cmd.get("value") if cmd.get("value") is not None else p.get("value")
                if hasattr(self.arena.paradigm, "state") and name in self.arena.paradigm.state:
                    self.arena.paradigm.state[name] = value
                return {"status": "ok", "param": name, "value": value}

            elif action == "reset_trial":
                advance = cmd.get("advance", p.get("advance", True))
                keep_mem = cmd.get("keep_memory", p.get("keep_memory", True))
                if advance:
                    self.current_trial += 1
                if not keep_mem and hasattr(self.arena.fly, "circuit"):
                    self.arena.fly.circuit.reset_state(keep_memory=False)
                    self.active_brain.log("memory_reset")
                    self.active_brain.save()
                # Reset paradigm trial state and return the fly to the paradigm spawn
                paradigm = getattr(self.arena, "paradigm", None)
                if paradigm is not None and hasattr(paradigm, "reset_trial"):
                    paradigm.reset_trial()
                self.arena.reset_fly_to_spawn()
                self.trial_sim_time = 0.0
                self.segment_id = uuid.uuid4().hex
                self.transition = {"reason": "manual_reset", "step": self.total_steps}
                self.active_brain.log("manual_reset", segment_id=self.segment_id)
                self.latest_telemetry = self._assemble_telemetry({})
                return {"status": "ok", "current_trial": self.current_trial}

            elif action == "save_checkpoint":
                path = self.save_checkpoint(cmd.get("label", "manual"))
                return {"status": "ok", "checkpoint": str(path)}

            else:
                return {"status": "error", "message": f"Unknown action: '{action}'"}

    def save_checkpoint(self, tag: str = "periodic") -> Path:
        """Saves current continuous synaptic weights and trial ledger to disk."""
        self.brains.save_all()
        # Labels are display text, never path fragments supplied by an API client.
        safe_tag = "".join(c for c in str(tag) if c.isalnum() or c in "_-")[:60] or "manual"
        filename = f"checkpoint_{self.active_paradigm_id}_{safe_tag}_{time.time_ns()}.json"
        target_file = self.checkpoints_dir / filename
        data = {
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

        return target_file


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
        return {
            "status": "error" if self.runner.last_error else "online",
            "error": self.runner.last_error,
            "paused": self.runner.paused,
            "continuous": self.runner.continuous,
            "service": "Project NeuroFly Continuous Learning Daemon",
            "uptime_sec": round(uptime, 1),
            "total_steps": self.runner.total_steps,
            "sim_speed": self.runner.sim_speed,
            "active_paradigm": self.runner.active_paradigm_id,
            "active_paradigm_title": self.runner.active_paradigm_title,
            "current_trial": self.runner.current_trial,
            "trials_completed": len(self.runner.trial_history),
            "trial_elapsed_s": round(float(getattr(self.runner, "trial_sim_time", 0.0)), 2),
            "trial_length_s": getattr(self.runner, "trial_length_s", None),
            "world_bounds": list(getattr(getattr(self.runner, "arena", None), "world_bounds", ())),
            "stream": self.gateway.describe()
        }

    def do_GET(self):
        url = self.path.split("?")[0].rstrip("/")

        if url in ("", "/status", "/api/status"):
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            with self.runner.lock:
                resp = self._status_payload()
            self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))

        elif url == "/api/telemetry":
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            with self.runner.lock:
                data = self.runner.latest_telemetry
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif url == "/api/observatory":
            # One lock and one response prevent mixed experiment identities during switches.
            with self.runner.lock:
                data = {"status": self._status_payload(),
                        "brain": self.runner.active_brain.summary(details=True),
                        "telemetry": self.runner.latest_telemetry,
                        "brains": self.runner.brains.catalog(self.runner.active_paradigm_id)}
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif url in ("/api/brains", "/api/brain"):
            with self.runner.lock:
                data = (self.runner.active_brain.summary(details=True) if url == "/api/brain"
                        else {"active": self.runner.active_paradigm_id,
                              "brains": self.runner.brains.catalog(self.runner.active_paradigm_id)})
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

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
                last_send = 0.0
                while self.runner.running:
                    with self.runner.lock:
                        payload = self.runner.latest_telemetry
                    if payload:
                        msg = f"data: {json.dumps(payload)}\n\n"
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    last_send = self.gateway.pace(last_send)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                slot.release()

        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

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
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging to keep console clean
        return


def run_daemon():
    parser = argparse.ArgumentParser(description="Project NeuroFly Continuous Headless Learning Daemon")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind HTTP API (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8769, help="Port to bind HTTP API (default: 8769)")
    parser.add_argument("--paradigm", default="multisensory-sandbox", help="Initial experimental paradigm")
    parser.add_argument("--speed", type=float, default=10.0, help="Initial simulation speed multiplier (default: 10.0x)")
    parser.add_argument("--checkpoint-interval", type=float, default=60.0, help="Interval between checkpoints in seconds")
    parser.add_argument("--trial-seconds", type=float, default=60.0,
                        help="Simulated seconds per trial for paradigms without a natural endpoint (default: 60)")
    parser.add_argument("--continuous", action="store_true", help="Observe continuously without automatic respawns; manual reset starts a new segment")
    parser.add_argument("--output-dir", default=None, help="Brain checkpoints and run outputs directory")
    parser.add_argument("--pid-file", default="", help="Optional path to write daemon PID file")

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
    args = parser.parse_args()

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
    print("PROJECT NEUROFLY — 24/7 CONTINUOUS REMOTE LEARNING DAEMON")
    print(f"PID: {os.getpid()} | API Port: {args.port} | Speed: {args.speed}x")
    print(f"Active Paradigm: {args.paradigm}")
    if stream_policy.public:
        mode = "READ-ONLY (no admin token set)" if stream_policy.read_only else "token-gated commands"
        print(f"Public mode: {mode} | max SSE clients: {stream_policy.max_stream_clients or 'unlimited'}"
              f" | stream {stream_policy.stream_hz:g} Hz")
    print("===============================================================================", flush=True)

    runner = ContinuousExperimentRunner(
        initial_paradigm=args.paradigm,
        sim_speed=args.speed,
        checkpoint_interval=args.checkpoint_interval,
        trial_length_s=args.trial_seconds,
        continuous=args.continuous,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
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
        })
        recorder_thread = RecorderThread(runner, recorder, summary_interval=args.summary_interval)
        recorder_thread.start()
        print(f"[Daemon] Learning records: {data_dir}", flush=True)

    NeuroflyHTTPHandler.runner = runner
    NeuroflyHTTPHandler.gateway = StreamGateway(stream_policy)
    server = ThreadingHTTPServer((args.host, args.port), NeuroflyHTTPHandler)

    def _signal_handler(signum, frame):
        print(f"\n[Daemon] Received signal {signum}. Initiating graceful shutdown...", flush=True)
        runner.stop()
        if recorder_thread is not None:
            recorder_thread.stop()
        try:
            if pid_path.exists():
                pid_path.unlink()
        except OSError:
            pass
        # Do not call server.shutdown() here: the handler runs on the thread
        # that is inside serve_forever(), and shutdown() would wait forever for
        # that loop to exit.  SystemExit unwinds serve_forever() instead and the
        # finally block below closes the socket.
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        print(f"[Daemon] HTTP API & SSE stream ready at http://{args.host}:{args.port}/", flush=True)
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        runner.stop()
        if recorder_thread is not None:
            recorder_thread.stop()
        server.server_close()
        print("[Daemon] Clean shutdown complete.", flush=True)


if __name__ == "__main__":
    run_daemon()
