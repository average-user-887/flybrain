"""Requested speed must not change the science (work package 2).

DECLARED TOLERANCE (fixed before the first run, 19 Sep 2026): **exact equality**,
absolute and relative tolerance 0.  Every compared value -- fly pose, heading and
speed, all mushroom-body arrays (``pn_tuning, w_pn_kc, w_kc_mbon_baseline, u, w,
y_kc, y_dan_pam, y_dan_ppl1``), central-complex ``epg``/``goal_vector``, the arena's
``random.Random`` and NumPy generator states, the global NumPy/``random`` states,
trial counters and simulated trial time -- must be bit-identical at equal steps.

Rationale: integration dt is fixed at 0.02 s; the scheduler only decides *when* a
step runs, and telemetry assembly is read-only.  The same float operations run in
the same order in the same process, so any difference is a real coupling between
wall-clock pacing and the simulation, to be inspected -- not absorbed by a looser
threshold.

Protocol: identical seed and initial state (fresh brain directory per run), the same
step-indexed interventions (``schedule_command``), requested 1x, 20x and 100x run by
the real scheduler thread, state captured under the simulation lock at steps 60, 120
and 180.  Run as a script to write a receipt:
``python tests/test_timing_determinism.py --receipt outputs/rethink-audit/determinism_receipt.json``
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_brains import ARRAYS  # noqa: E402
from neurofly_daemon import ContinuousExperimentRunner  # noqa: E402

CHECK_STEPS = (60, 120, 180)
SPEEDS = (1.0, 20.0, 100.0)
PARADIGM = "wind-tunnel"
# Step-indexed interventions: teaching (changes MB weights), then assay controls.
SCHEDULE = (
    (5, {"action": "teach_brain", "pairs": 1}),
    (130, {"action": "set_param", "name": "windVelocity", "value": 30}),
    (150, {"action": "assay_action", "name": "shift_plume"}),
)


def _capture(runner):
    fly = runner.arena.fly
    state = {
        "fly": [float(fly.pos.x), float(fly.pos.y), float(fly.heading), float(fly.speed)],
        "trial_sim_time": runner.trial_sim_time,
        "current_trial": runner.current_trial,
        "brain_steps": runner.active_brain.steps,
        "arena_rng": repr(runner.arena.rng.getstate()),
        "arena_np_rng": repr(runner.arena.np_rng.bit_generator.state),
        "global_np_rng": repr(np.random.get_state()[1][:8].tolist() + [np.random.get_state()[2]]),
        "global_rng": repr(random.getstate()[1][:8]),
        "cx_epg": fly.cx.epg.copy(),
        "cx_goal": fly.cx.goal_vector.copy(),
    }
    for name in ARRAYS:
        state["mb_" + name] = getattr(fly.circuit, name).copy()
    return state


def run_at_speed(speed, directory, timeout_s=60.0):
    random.seed(20260919)
    np.random.seed(20260919)
    runner = ContinuousExperimentRunner(initial_paradigm=PARADIGM, sim_speed=speed, checkpoint_interval=3600,
                                        output_dir=Path(directory))
    captured = {}

    def hook(r):
        if r.total_steps in CHECK_STEPS:
            captured[r.total_steps] = _capture(r)

    runner.step_hook = hook
    runner.stop_at_step = CHECK_STEPS[-1]
    entries = [(step, runner.schedule_command(step, cmd)) for step, cmd in SCHEDULE]
    start = time.perf_counter()
    runner.start()
    try:
        deadline = time.time() + timeout_s
        while len(captured) < len(CHECK_STEPS) and time.time() < deadline:
            time.sleep(0.02)
    finally:
        runner.running = False
        runner._wake.set()
        runner.sim_thread.join(timeout=5)
    wall = time.perf_counter() - start
    acks = [{"scheduled_step": step, "status": e["result"]["status"], "applied_step": e["result"]["ack"]["applied_step"]}
            for step, e in entries]
    return captured, acks, wall


def compare(reference, other):
    """Names of every field that differs at any checked step (exact comparison)."""
    diffs = []
    for step in CHECK_STEPS:
        a, b = reference.get(step), other.get(step)
        if a is None or b is None:
            diffs.append(f"step {step}: missing capture")
            continue
        for key in a:
            va, vb = a[key], b[key]
            same = np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb
            if not same:
                detail = ""
                if isinstance(va, np.ndarray):
                    detail = f" max|diff|={float(np.max(np.abs(va - vb))):.3e}"
                diffs.append(f"step {step}: {key}{detail}")
    return diffs


def run_all(base_dir):
    results = {}
    for speed in SPEEDS:
        results[speed] = run_at_speed(speed, Path(base_dir) / f"speed_{int(speed)}")
    return results


def test_requested_speed_does_not_change_state_or_weights(tmp_path):
    results = run_all(tmp_path)
    ref_states, ref_acks, wall_1x = results[1.0]
    assert set(ref_states) == set(CHECK_STEPS)
    # The interventions took effect exactly at their scheduled steps, at every speed.
    for speed, (_, acks, _) in results.items():
        for ack in acks:
            assert ack["status"] == "ok", (speed, ack)
            assert ack["applied_step"] == ack["scheduled_step"], (speed, ack)
    # Teaching changed the learned weights, so the comparison covers plasticity.
    assert not np.array_equal(ref_states[120]["mb_w"], np.zeros_like(ref_states[120]["mb_w"]))
    for speed in (20.0, 100.0):
        states, _, wall = results[speed]
        assert compare(ref_states, states) == [], f"{speed}x diverged from 1x"
        # The runs really were paced differently.
        assert wall < wall_1x / 3, (speed, wall, wall_1x)


if __name__ == "__main__":
    import argparse
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()
    base = args.work_dir or tempfile.mkdtemp(prefix="neurofly-determinism-")
    results = run_all(base)
    ref = results[1.0][0]
    receipt = {
        "declared_tolerance": "exact equality (atol=0, rtol=0), declared before running; see module docstring",
        "paradigm": PARADIGM, "check_steps": CHECK_STEPS, "speeds": SPEEDS,
        "schedule": [{"step": s, "cmd": c} for s, c in SCHEDULE],
        "runs": {f"{int(s)}x": {"wall_s": round(w, 3), "acks": a,
                                 "divergent_fields_vs_1x": compare(ref, st)}
                 for s, (st, a, w) in results.items()},
        "mb_w_l2_at_180": float(np.linalg.norm(ref[180]["mb_w"])),
        "fly_at_180": ref[180]["fly"],
    }
    receipt["pass"] = all(not r["divergent_fields_vs_1x"] for r in receipt["runs"].values())
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
