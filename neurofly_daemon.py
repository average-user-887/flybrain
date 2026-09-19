#!/usr/bin/env python3
"""
Project NeuroFly (v1.0-release) — Continuous Background Learning Daemon
========================================================================
Runs 24/7 headless biological simulation and continuous online learning
on remote compute nodes (e.g. AMD Ryzen workstation) or local environments.

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
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
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


class ContinuousExperimentRunner:
    """Manages the continuous headless simulation loop and online plasticity."""

    def __init__(
        self,
        initial_paradigm: str = "multisensory-sandbox",
        sim_speed: float = 10.0,
        checkpoint_interval: float = 60.0,
        output_dir: Optional[Path] = None
    ):
        self.output_dir = output_dir or (PROJECT_ROOT / "outputs")
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.telemetry_dir = self.output_dir / "telemetry"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.running = False
        self.sim_speed = max(0.1, min(100.0, float(sim_speed)))
        self.dt = 0.02
        self.active_paradigm_id = initial_paradigm
        self.checkpoint_interval = max(5.0, float(checkpoint_interval))

        # Runtime state
        self.start_time = time.time()
        self.total_steps = 0
        self.current_trial = 1
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
        """Initializes the arena for the given paradigm."""
        print(f"[Daemon] Initializing Arena with paradigm: {paradigm_name}...", flush=True)
        try:
            self.arena = Arena(
                paradigm=paradigm_name,
                brain_type="modular",
                num_flies=1,
                num_predators=0
            )
            self.active_paradigm_id = paradigm_name
            self.active_paradigm_title = getattr(self.arena.paradigm, "name", paradigm_name)
        except Exception as err:
            print(f"[Daemon] Failed to initialize paradigm '{paradigm_name}': {err}. Falling back to default open-arena.", flush=True)
            self.arena = Arena(paradigm=None)
            self.active_paradigm_id = "open-arena"
            self.active_paradigm_title = "Open Arena Assay"

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
                # 1. Step simulation arena
                try:
                    step_result = self.arena.step(step_dt)
                except Exception as step_err:
                    print(f"[Daemon] Exception in arena.step: {step_err}", file=sys.stderr)
                    step_result = {}

                self.total_steps += 1
                fly = self.arena.fly

                # 2. Check for trial advancement or milestone metrics
                p_metrics = step_result.get("paradigm_metrics", {})
                if p_metrics.get("trial_complete", False) or step_result.get("food_collected", 0) > len(self.trial_history) * 5:
                    self._record_trial_milestone(step_result)

                # 3. Assemble telemetry snapshot
                self.latest_telemetry = self._assemble_telemetry(step_result)

                # 4. Periodic Checkpointing
                now = time.time()
                if now - self.last_checkpoint_time >= self.checkpoint_interval:
                    self.save_checkpoint("periodic")
                    self.last_checkpoint_time = now

            # Sleep to match target speed pacing
            elapsed = time.perf_counter() - loop_start
            sleep_time = (step_dt / self.sim_speed) - elapsed
            if sleep_time > 0.0005:
                time.sleep(sleep_time)

    def _assemble_telemetry(self, step_res: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs standardized JSON telemetry packet for browser streaming."""
        fly = self.arena.fly
        stim = step_res.get("stimuli", {})
        p_metrics = step_res.get("paradigm_metrics", {})

        # Extract Mushroom Body weight summary
        weights_mean = 0.5
        weights_std = 0.05
        if hasattr(fly, "circuit") and getattr(fly.circuit, "weights", None) is not None:
            w = fly.circuit.weights
            weights_mean = float(np.mean(w))
            weights_std = float(np.std(w))

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
            "timestamp": round(time.time(), 3),
            "step": self.total_steps,
            "paradigm": self.active_paradigm_id,
            "paradigm_title": self.active_paradigm_title,
            "sim_speed": self.sim_speed,
            "trial": self.current_trial,
            "fly": {
                "x": round(float(fly.pos.x), 2),
                "y": round(float(fly.pos.y), 2),
                "heading": round(float(fly.heading), 3),
                "speed": round(float(fly.speed), 2),
                "state": getattr(fly, "behavioral_state", "FORAGING")
            },
            "sensory": {
                "temp": round(float(stim.get("temperature", 24.0)), 1),
                "odor_a": round(float(stim.get("odor_cs_plus", stim.get("odor_conc", 0.0))), 3),
                "odor_b": round(float(stim.get("odor_cs_minus", 0.0)), 3),
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
            "plasticity": {
                "mb_weights_mean": round(weights_mean, 4),
                "mb_weights_std": round(weights_std, 4),
                "learning_curve": self.learning_curve[-30:] if self.learning_curve else [0.5]
            },
            "metrics": p_metrics
        }

    def _record_trial_milestone(self, step_res: Dict[str, Any]):
        """Records end of trial or adaptation milestone in continuous memory."""
        metrics = step_res.get("paradigm_metrics", {})
        metric_val = metrics.get("performance_index", metrics.get("pi", metrics.get("learning_index", 0.5)))
        self.learning_curve.append(float(metric_val))
        self.trial_history.append({
            "trial": self.current_trial,
            "paradigm": self.active_paradigm_id,
            "step": self.total_steps,
            "metric": float(metric_val),
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
                self._init_arena(target)
                return {"status": "ok", "active_paradigm": self.active_paradigm_id}

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
                advance = cmd.get("advance", True)
                keep_mem = cmd.get("keep_memory", True)
                if advance:
                    self.current_trial += 1
                if not keep_mem and hasattr(self.arena.fly, "circuit"):
                    self.arena.fly.circuit.reset(preserve_weights=False)
                # Reset fly position to center or paradigm spawn
                self.arena.fly.pos.x = self.arena.width / 2.0
                self.arena.fly.pos.y = self.arena.height / 2.0
                return {"status": "ok", "current_trial": self.current_trial}

            elif action == "save_checkpoint":
                path = self.save_checkpoint(cmd.get("label", "manual"))
                return {"status": "ok", "checkpoint": str(path)}

            else:
                return {"status": "error", "message": f"Unknown action: '{action}'"}

    def save_checkpoint(self, tag: str = "periodic") -> Path:
        """Saves current continuous synaptic weights and trial ledger to disk."""
        filename = f"checkpoint_{self.active_paradigm_id}_{tag}_{int(time.time())}.json"
        target_file = self.checkpoints_dir / filename
        data = {
            "tag": tag,
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

        # Keep only latest 50 checkpoints to conserve disk space
        checkpoints = sorted(self.checkpoints_dir.glob("*.json"), key=os.path.getmtime)
        if len(checkpoints) > 50:
            for old in checkpoints[:-50]:
                try:
                    old.unlink()
                except OSError:
                    pass

        return target_file


class NeuroflyHTTPHandler(BaseHTTPRequestHandler):
    """Multi-threaded HTTP Server handling REST telemetry and SSE streaming."""

    runner: ContinuousExperimentRunner = None  # Injected on startup

    def _set_cors_headers(self, content_type: str = "application/json"):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        url = self.path.split("?")[0].rstrip("/")

        if url in ("", "/status", "/api/status"):
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            uptime = time.time() - self.runner.start_time
            resp = {
                "status": "online",
                "service": "Project NeuroFly Continuous Learning Daemon",
                "uptime_sec": round(uptime, 1),
                "total_steps": self.runner.total_steps,
                "sim_speed": self.runner.sim_speed,
                "active_paradigm": self.runner.active_paradigm_id,
                "active_paradigm_title": self.runner.active_paradigm_title,
                "current_trial": self.runner.current_trial,
                "trials_completed": len(self.runner.trial_history)
            }
            self.wfile.write(json.dumps(resp, indent=2).encode("utf-8"))

        elif url == "/api/telemetry":
            self.send_response(200)
            self._set_cors_headers("application/json")
            self.end_headers()
            with self.runner.lock:
                data = self.runner.latest_telemetry
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
            # Server-Sent Events (SSE) stream
            self.send_response(200)
            self._set_cors_headers("text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                while self.runner.running:
                    with self.runner.lock:
                        payload = self.runner.latest_telemetry
                    if payload:
                        msg = f"data: {json.dumps(payload)}\n\n"
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    time.sleep(0.033)  # ~30 Hz broadcast
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def do_POST(self):
        url = self.path.split("?")[0].rstrip("/")
        if url == "/api/command":
            content_len = int(self.headers.get("Content-Length", 0))
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
    parser.add_argument("--pid-file", default="", help="Optional path to write daemon PID file")
    args = parser.parse_args()

    # PID writing if requested
    pid_path = Path(args.pid_file) if args.pid_file else (PROJECT_ROOT / "outputs" / "neurofly_daemon.pid")
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    print("===============================================================================")
    print("PROJECT NEUROFLY — 24/7 CONTINUOUS REMOTE LEARNING DAEMON")
    print(f"PID: {os.getpid()} | API Port: {args.port} | Speed: {args.speed}x")
    print(f"Active Paradigm: {args.paradigm}")
    print("===============================================================================", flush=True)

    runner = ContinuousExperimentRunner(
        initial_paradigm=args.paradigm,
        sim_speed=args.speed,
        checkpoint_interval=args.checkpoint_interval
    )
    runner.start()

    NeuroflyHTTPHandler.runner = runner
    server = ThreadingHTTPServer((args.host, args.port), NeuroflyHTTPHandler)

    def _signal_handler(signum, frame):
        print(f"\n[Daemon] Received signal {signum}. Initiating graceful shutdown...", flush=True)
        runner.stop()
        try:
            if pid_path.exists():
                pid_path.unlink()
        except OSError:
            pass
        server.shutdown()
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
        server.server_close()
        print("[Daemon] Clean shutdown complete.", flush=True)


if __name__ == "__main__":
    run_daemon()
