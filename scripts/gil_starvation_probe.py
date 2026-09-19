#!/usr/bin/env python3
"""Measure how the simulation loop starves other Python threads, with and without the
scheduler's GIL hand-over (``ContinuousExperimentRunner.yield_wall_s``).

In-process, isolated (temporary brain directory, no HTTP). For each setting it runs the
real scheduler at the requested speed and measures, from another thread, how long a
``time.sleep(0.033)`` (the SSE pacing interval) really takes and how long a command
takes to be acknowledged.  Writes a JSON receipt.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurofly_daemon import ContinuousExperimentRunner  # noqa: E402


def measure(speed, yield_s, work):
    runner = ContinuousExperimentRunner(initial_paradigm="wind-tunnel", sim_speed=speed, checkpoint_interval=3600,
                                        output_dir=tempfile.mkdtemp(dir=work))
    runner.yield_wall_s = yield_s
    runner.start()
    try:
        time.sleep(1.0)
        sleeps = []
        for _ in range(40):
            t0 = time.perf_counter()
            time.sleep(0.033)
            sleeps.append((time.perf_counter() - t0) * 1e3)
        cmds = []
        for _ in range(10):
            t0 = time.perf_counter()
            runner.dispatch_command({"action": "set_speed", "speed": speed})
            cmds.append((time.perf_counter() - t0) * 1e3)
        timing = runner.timing_snapshot()
    finally:
        runner.running = False
        runner._wake.set()
        runner.sim_thread.join(timeout=5)
    sleeps.sort()
    cmds.sort()
    return {"yield_wall_s": yield_s, "requested_speed": speed, "achieved_speed": timing["achieved_speed"],
            "sleep_33ms_actual_p50_ms": round(sleeps[len(sleeps) // 2], 1), "sleep_33ms_actual_max_ms": round(sleeps[-1], 1),
            "command_ack_p50_ms": round(cmds[len(cmds) // 2], 1), "command_ack_max_ms": round(cmds[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--speed", type=float, default=100.0)
    args = ap.parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    rows = [measure(args.speed, y, args.work_dir) for y in (None, 0.0, 0.001)]
    out = {"kind": "GIL starvation probe (in-process, isolated)", "paradigm": "wind-tunnel", "rows": rows}
    Path(args.receipt).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
