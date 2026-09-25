"""A light stand-in for `python -m neurofly_body run`, for studio tests and headless checks.

It parses the same arguments as the real runner and writes a real run directory
(manifest, telemetry, summary and body.nfbody) through ``run_embodied``, but the
neural side is the modular baseline and the body is a kinematic stick fly, so it
needs neither the MaleCNS graph nor FlyGym.  Nothing here is a simulation result.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from neurofly_body import cli
from neurofly_body.modular import ModularCommandDecoder, ModularOptomotorBackend
from neurofly_body.runner import EmbodiedConfig, run_embodied

# Segment offsets in the fly's own frame (mm): thorax, head, abdomen, six leg tips.
_OFFSETS = [(0.0, 0.0), (0.9, 0.0), (-1.1, 0.0),
            (0.5, 0.9), (0.0, 1.0), (-0.5, 0.9), (0.5, -0.9), (0.0, -1.0), (-0.5, -0.9)]
_NAMES = ["Thorax", "Head", "Abdomen", "LFTarsus", "LMTarsus", "LHTarsus", "RFTarsus", "RMTarsus", "RHTarsus"]


class StickFlyBody:
    physics_dt_s = 0.0001

    def __init__(self) -> None:
        self.reset(0)

    def reset(self, seed):
        self.time, self.x, self.y, self.yaw = 0.05, 0.0, 0.0, 0.0
        return self._observation(0.0)

    def step(self, cpg_drive, substeps):
        left, right = cpg_drive
        dt = substeps * self.physics_dt_s
        yaw_velocity = 2.0 * (right - left)
        speed = 5.0 * (left + right) / 2.0
        self.yaw = math.remainder(self.yaw + yaw_velocity * dt, math.tau)
        self.x += speed * math.cos(self.yaw) * dt
        self.y += speed * math.sin(self.yaw) * dt
        self.time += dt
        return self._observation(yaw_velocity)

    def _observation(self, yaw_velocity):
        half = self.yaw / 2.0
        return {"body_sim_time_s": self.time,
                "thorax": {"position_mm": [self.x, self.y, 0.5],
                           "quaternion_wxyz": [math.cos(half), 0.0, 0.0, math.sin(half)],
                           "yaw_rad": self.yaw, "yaw_velocity_rad_s": yaw_velocity},
                "joint_angles_rad": [0.0], "joint_velocities_rad_s": [0.0],
                "contacts": {"found": [1.0] * 6}}

    def skeleton(self):
        return {"segments": list(_NAMES), "parents": [-1, 0, 0, 0, 0, 0, 0, 0, 0], "units": "mm"}

    def segment_positions(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [[self.x + c * dx - s * dy, self.y + s * dx + c * dy, 0.5 if i < 3 else 0.0]
                for i, (dx, dy) in enumerate(_OFFSETS)]

    def describe(self):
        return {"adapter": "tests.studio_fakes.StickFlyBody (NOT a simulation)",
                "physics_dt_s": self.physics_dt_s, "units": {}}

    def close(self):
        pass


def _split_silence(argv: list[str]) -> tuple[list[str], list[str]]:
    rest, targets, items = [], [], iter(argv)
    for item in items:
        if item == "--silence":
            targets.append(next(items))
        else:
            rest.append(item)
    return rest, targets


def fake_runner(argv: list[str], log_path: Path) -> int:
    """Queue runner with the real argument parser and output format.

    ``--silence`` is stood in for by a modular controller with no turning, and
    summary.json gets a ``silenced`` block shaped like the real runner's.  This
    only exercises the studio; it says nothing about what silencing does.
    """
    argv, silence = _split_silence(list(argv))
    args = cli._parser().parse_args(["run", *argv])
    config = EmbodiedConfig(duration_s=args.duration, output_dir=args.output, mode=args.mode,
                            neural_dt_ms=args.neural_dt_ms, physics_dt_s=args.physics_dt_s,
                            world_angular_velocity_rad_s=args.world_angular_velocity_rad_s,
                            contrast=args.contrast, seed=args.seed, record_fps=args.record_fps)
    backend = ModularOptomotorBackend(turn_gain=0.0 if silence else 1.0)
    run_embodied(config, backend, StickFlyBody(),
                 decoder=ModularCommandDecoder(max_drive=args.max_cpg_drive),
                 invocation=cli._invocation(args))
    if silence:
        summary_path = Path(args.output) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["silenced"] = {"targets": silence, "total_neurons": 0, "spikes_total": 0,
                               "clamp_held": True, "stand_in": "tests/studio_fakes.py"}
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(log_path).write_text("fake runner\n", encoding="utf-8")
    return 0
