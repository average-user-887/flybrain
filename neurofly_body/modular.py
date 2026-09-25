"""Modular (non-connectome) researcher baseline for the embodied FlyGym loop.

The P2 gate asks that the modular and the connectome controllers both walk the
same body.  This backend is the arena's phenomenological optomotor model
(``vision.CompoundEyeVision``: 72 ommatidia, HS-like left/right flow
integration with an 80 ms time constant, ``get_optomotor_yaw_bias``) put behind
the same ``NeuralBackend`` seam as the connectome server.  It is a baseline,
not a biological claim, and nothing in it is derived from the connectome.

Declared engineering choices (all ASSUMPTIONS):

* the drum is at infinity, so the retinal flow is rotation only (world angular
  velocity minus body yaw velocity), exactly the slip the connectome gets;
* tonic forward drive ``forward_drive`` on both sides (the modular fly always
  walks; the connectome fly walks only if DNp09 fires);
* the arena's yaw bias (clipped to +-0.3) becomes an amplitude difference:
  a + (counter-clockwise, leftward) bias shrinks the LEFT legs, and FlyGym turns
  toward the side with the smaller amplitude.  ``turn_gain`` scales it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

YAW_BIAS_LIMIT = 0.3   # vision.CompoundEyeVision.get_optomotor_yaw_bias clip


class ModularOptomotorBackend:
    """The arena's modular optomotor model as an embodied controller."""

    def __init__(self, *, forward_drive: float = 1.0, turn_gain: float = 1.0) -> None:
        from vision import CompoundEyeVision

        if not (math.isfinite(forward_drive) and forward_drive > 0):
            raise ValueError("forward_drive must be finite and positive")
        if not (math.isfinite(turn_gain) and turn_gain >= 0):
            raise ValueError("turn_gain must be finite and non-negative")
        self.forward_drive = float(forward_drive)
        self.turn_gain = float(turn_gain)
        self._vision_cls = CompoundEyeVision
        self.reset()

    def reset(self) -> None:
        self.vision = self._vision_cls(rng=np.random.default_rng(0))
        self.sim_ms = 0.0

    def get_status(self) -> dict[str, Any]:
        return {
            "controller_kind": "modular-baseline",
            "controller_version": "vision.CompoundEyeVision optomotor v1",
            "synthetic": False,
            "connectome": None,
            "forward_drive": self.forward_drive,
            "turn_gain": self.turn_gain,
            "brain_backend": "none",
        }

    def step(self, sensory: dict[str, Any], duration_ms: float = 2.0) -> dict[str, Any]:
        slip = float(sensory["optomotor_slip_rad_s"])
        contrast = float(sensory.get("optomotor_contrast", 1.0))
        dt_s = duration_ms / 1000.0
        # Rotation only: pass the full slip as external rotation with no self-motion.
        self.vision.step(fly_pos=np.zeros(2), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
                         dt=dt_s, external_yaw_rad_s=slip, contrast=contrast)
        bias = self.vision.get_optomotor_yaw_bias()
        turn = self.turn_gain * bias / YAW_BIAS_LIMIT        # in [-turn_gain, turn_gain]
        left = self.forward_drive * max(0.0, 1.0 - max(0.0, turn))
        right = self.forward_drive * max(0.0, 1.0 - max(0.0, -turn))
        self.sim_ms += duration_ms
        return {
            "sim_ms": self.sim_ms,
            "total_step_spikes": 0,
            "delta_hs": float(self.vision.delta_hs),
            "yaw_bias": float(bias),
            "modular_command": {"left": left, "right": right},
        }


@dataclass
class ModularCommandDecoder:
    """Pass-through: the modular controller already emits CPG commands."""

    max_drive: float = 1.2
    name = "modular-passthrough"
    required_status = ("controller_kind",)

    def reset(self) -> None:
        pass

    def decode_reply(self, reply: dict, dt_ms: float) -> dict:
        command = reply["modular_command"]
        cap = self.max_drive
        left = max(-cap, min(cap, float(command["left"])))
        right = max(-cap, min(cap, float(command["right"])))
        return {"left_cpg_drive": left, "right_cpg_drive": right, "events": []}

    def describe(self) -> dict:
        return {
            "name": self.name,
            "classification": "MODULAR BASELINE: phenomenological optomotor controller, no connectome",
            "model": "left = F*(1 - max(0, turn)), right = F*(1 - max(0, -turn)), "
                     "turn = turn_gain * yaw_bias / 0.3 (vision.CompoundEyeVision)",
            "parameters": {"max_drive": [self.max_drive, "NeuroMechFly v2 DN range"]},
            "biological_vnc_claim": False,
        }
