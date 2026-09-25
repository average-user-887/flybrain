"""Plain-language numbers for comparing two finished runs, read from telemetry.jsonl.

These are descriptive only.  A studio comparison is exploratory: one seed per
side is an anecdote, and no statistics are claimed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def run_metrics(run_dir: Path) -> dict[str, Any]:
    """Turning and activity summaries of one embodied run."""
    run_dir = Path(run_dir)
    n = 0
    yaw_velocity_sum = 0.0
    spikes = 0
    previous_yaw = None
    turned = 0.0
    world = None
    last_time = 0.0
    with (run_dir / "telemetry.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            sensory = record.get("sensory", {})
            thorax = record.get("body", {}).get("thorax", {})
            yaw_velocity_sum += float(sensory.get("body_yaw_velocity_rad_s", 0.0))
            world = sensory.get("world_angular_velocity_rad_s", world)
            yaw = thorax.get("yaw_rad")
            if yaw is not None:
                if previous_yaw is not None:
                    # Yaw is wrapped to (-pi, pi]; unwrap step by step (steps are 2 ms).
                    turned += math.remainder(float(yaw) - previous_yaw, math.tau)
                previous_yaw = float(yaw)
            spikes += int(record.get("neural", {}).get("total_step_spikes", 0))
            last_time = float(record.get("run_time_s", last_time))
            n += 1
    if n == 0:
        raise ValueError(f"{run_dir}/telemetry.jsonl has no records")
    mean_yaw_velocity = yaw_velocity_sum / n
    gain = (mean_yaw_velocity / world) if world else None
    return {
        "records": n,
        "simulated_s": last_time,
        "world_angular_velocity_rad_s": world,
        "mean_body_yaw_velocity_rad_s": mean_yaw_velocity,
        "net_yaw_change_deg": None if previous_yaw is None else math.degrees(turned),
        "turning_gain": gain,
        "total_graph_spikes": spikes,
    }
