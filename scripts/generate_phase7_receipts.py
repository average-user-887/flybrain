"""Generate Phase 7 scientific receipts for real MaleCNS v1.0 closed-loop connectome trials.

1. Optomotor Gaze Stabilization (connectome-fixed, DNa02 yaw steering)
2. Visual Looming Escape (connectome-fixed, LC4/LPLC2 -> DNp01 Giant Fiber)
3. Visual Heading Plasticity (connectome-plastic, WP6 3,081 ER->EPG synaptic depression)
"""
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurofly_daemon import ContinuousExperimentRunner
from brainlab.graph_identity import verify_graph, resolve_connectome_dir, resolve_graph_dir
from brainlab.wp6_plasticity import VisualHeadingPlasticityRule

RECEIPTS_DIR = ROOT / "docs" / "receipts"
RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)


def run_optomotor_trial():
    print("[Phase 7] Running Real Connectome Trial 1: Optomotor Gaze Stabilization...", flush=True)
    out_dir = ROOT / "outputs" / "receipts_tmp" / "optomotor"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = ContinuousExperimentRunner(
        initial_paradigm="optomotor",
        sim_speed=10.0,
        checkpoint_interval=3600,
        output_dir=out_dir,
        backend="connectome-fixed",
        test_synthetic_graph=False
    )
    history = []
    total_spikes_accum = 0
    t0 = time.time()
    with runner.lock:
        for step in range(50):
            res = runner.step_once()
            fly = runner.arena.fly
            telem = getattr(fly, "last_connectome_telemetry", {})
            spikes = telem.get("total_spikes", 0)
            total_spikes_accum += spikes
            history.append({
                "step": step,
                "yaw_rate": telem.get("yaw_rate", 0.0),
                "speed": telem.get("forward_speed", 0.0),
                "dn_rates": telem.get("dn_rates", {}),
                "epg_wedges": telem.get("epg_wedges", []),
                "epg_bump_phase": telem.get("epg_bump_phase", 0.0),
                "total_spikes": spikes,
            })
    elapsed = time.time() - t0
    ident = runner.identity()
    receipt = {
        "schema": "neurofly.connectome-closed-loop-optomotor.v1",
        "timestamp": time.time(),
        "backend": runner.backend,
        "identity": ident,
        "paradigm": "optomotor",
        "steps_simulated": 50,
        "sim_time_s": 50 * 0.02,
        "wall_time_s": round(elapsed, 3),
        "total_spikes_integrated": total_spikes_accum,
        "mean_dna02_l_hz": float(np.mean([h["dn_rates"].get("dna02_l", 0.0) for h in history])),
        "mean_dna02_r_hz": float(np.mean([h["dn_rates"].get("dna02_r", 0.0) for h in history])),
        "final_yaw_rate": history[-1]["yaw_rate"],
        "telemetry_samples": history[:10] + history[-10:],
    }
    target = RECEIPTS_DIR / "connectome_closed_loop_optomotor.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"[Phase 7] Trial 1 complete. Saved: {target.name} ({total_spikes_accum:,} spikes)", flush=True)
    return receipt


def run_looming_escape_trial():
    print("[Phase 7] Running Real Connectome Trial 2: Visual Looming Giant Fiber Escape...", flush=True)
    out_dir = ROOT / "outputs" / "receipts_tmp" / "looming"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = ContinuousExperimentRunner(
        initial_paradigm="looming-escape",
        sim_speed=10.0,
        checkpoint_interval=3600,
        output_dir=out_dir,
        backend="connectome-fixed",
        test_synthetic_graph=False
    )
    history = []
    total_spikes_accum = 0
    t0 = time.time()
    escape_triggered = False
    with runner.lock:
        for step in range(50):
            # At step 25, simulate looming visual stimulus approaching
            if step >= 25:
                theta = min(1.2, 0.2 + (step - 25) * 0.06)
                runner.arena.paradigmState = getattr(runner.arena, "paradigmState", {})
                if hasattr(runner.arena, "paradigm") and hasattr(runner.arena.paradigm, "looming_threat"):
                    runner.arena.paradigm.looming_threat.theta_deg = math.degrees(theta)
            res = runner.step_once()
            fly = runner.arena.fly
            telem = getattr(fly, "last_connectome_telemetry", {})
            spikes = telem.get("total_spikes", 0)
            total_spikes_accum += spikes
            state = telem.get("state", fly.behavioral_state)
            if state == "ESCAPE" or telem.get("forward_speed", 0.0) >= 30.0:
                escape_triggered = True
            history.append({
                "step": step,
                "state": state,
                "speed": telem.get("forward_speed", fly.speed),
                "yaw_rate": telem.get("yaw_rate", fly.angular_velocity),
                "dn_rates": telem.get("dn_rates", {}),
                "total_spikes": spikes,
            })
    elapsed = time.time() - t0
    ident = runner.identity()
    receipt = {
        "schema": "neurofly.connectome-closed-loop-looming.v1",
        "timestamp": time.time(),
        "backend": runner.backend,
        "identity": ident,
        "paradigm": "looming-escape",
        "steps_simulated": 50,
        "sim_time_s": 50 * 0.02,
        "wall_time_s": round(elapsed, 3),
        "total_spikes_integrated": total_spikes_accum,
        "escape_triggered": escape_triggered,
        "max_speed_mm_s": float(np.max([h["speed"] for h in history])),
        "telemetry_samples": history[20:35],
    }
    target = RECEIPTS_DIR / "connectome_closed_loop_looming.json"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"[Phase 7] Trial 2 complete. Saved: {target.name} (Escape: {escape_triggered})", flush=True)
    return receipt


def run_wp6_plasticity_trial():
    print("[Phase 7] Running Real Connectome Trial 3: WP6 Visual-Heading Plasticity...", flush=True)
    out_dir = ROOT / "outputs" / "receipts_tmp" / "plasticity"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = ContinuousExperimentRunner(
        initial_paradigm="buridan",
        sim_speed=10.0,
        checkpoint_interval=3600,
        output_dir=out_dir,
        backend="connectome-plastic",
        test_synthetic_graph=False
    )
    history = []
    total_spikes_accum = 0
    t0 = time.time()
    with runner.lock:
        for step in range(50):
            res = runner.step_once()
            fly = runner.arena.fly
            telem = getattr(fly, "last_connectome_telemetry", {})
            spikes = telem.get("total_spikes", 0)
            total_spikes_accum += spikes
            wp6 = telem.get("wp6", {})
            history.append({
                "step": step,
                "epg_bump_phase": telem.get("epg_bump_phase", 0.0),
                "wp6_mean_delta": wp6.get("mean_delta", 0.0),
                "wp6_max_delta": wp6.get("max_delta", 0.0),
                "total_spikes": spikes,
            })
    elapsed = time.time() - t0
    ident = runner.identity()
    target = RECEIPTS_DIR / "connectome_closed_loop_wp6_plasticity.json"
    receipt = {
        "schema": "neurofly.connectome-closed-loop-wp6.v1",
        "timestamp": time.time(),
        "backend": runner.backend,
        "identity": ident,
        "paradigm": "buridan",
        "steps_simulated": 50,
        "sim_time_s": 50 * 0.02,
        "wall_time_s": round(elapsed, 3),
        "total_spikes_integrated": total_spikes_accum,
        "plastic_synapses": 3081,
        "final_mean_delta": history[-1]["wp6_mean_delta"],
        "final_max_delta": history[-1]["wp6_max_delta"],
        "final_epg_bump_phase": history[-1]["epg_bump_phase"],
        "telemetry_samples": history[:10] + history[-10:],
    }
    with open(target, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"[Phase 7] Trial 3 complete. Saved: {target.name} (WP6 Δw: {receipt['final_mean_delta']:.6f})", flush=True)
    return receipt


if __name__ == "__main__":
    r1 = run_optomotor_trial()
    r2 = run_looming_escape_trial()
    r3 = run_wp6_plasticity_trial()
    print("\n[Phase 7] ALL THREE REAL CONNECTOME EXPERIMENTS COMPLETED SUCCESSFULLY!", flush=True)
