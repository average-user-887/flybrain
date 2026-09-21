"""Declared engineering bridge from descending-neuron rates to leg CPG drive."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DNa02CPGDecoder:
    """Map DNa02 rates to FlyGym's left/right CPG magnitude commands.

    DNa02 activation predicts ipsilateral turning.  The stock FlyGym turning
    controller turns toward the slower (inside) side, so the mapping is crossed:
    left DNa02 drives the right/outer leg oscillators and vice versa.  This is an
    explicit ENGINEERED mapping, not a VNC model or a fitted biological parameter.

    There is intentionally no tonic term, minimum, or positive-speed fallback.
    Exact zero rates from reset yield exact zero commands.
    """

    gain_per_hz: float = 0.04
    tau_ms: float = 50.0
    max_drive: float = 1.2
    rate_l_hz: float = 0.0
    rate_r_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.gain_per_hz <= 0 or self.tau_ms <= 0 or self.max_drive <= 0:
            raise ValueError("decoder gain, tau, and max_drive must be positive")

    def reset(self) -> None:
        self.rate_l_hz = 0.0
        self.rate_r_hz = 0.0

    def decode(self, rate_l_hz: float, rate_r_hz: float, dt_ms: float) -> dict:
        if dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        values = (float(rate_l_hz), float(rate_r_hz), float(dt_ms))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("DNa02 rates and dt must be finite")
        if rate_l_hz < 0 or rate_r_hz < 0:
            raise ValueError("DNa02 rates cannot be negative")

        alpha = 1.0 - math.exp(-dt_ms / self.tau_ms)
        self.rate_l_hz += alpha * (rate_l_hz - self.rate_l_hz)
        self.rate_r_hz += alpha * (rate_r_hz - self.rate_r_hz)

        # FlyGym order is [left-side CPG magnitude, right-side CPG magnitude].
        left_drive = min(self.max_drive, self.gain_per_hz * self.rate_r_hz)
        right_drive = min(self.max_drive, self.gain_per_hz * self.rate_l_hz)
        if self.rate_l_hz == 0.0 and self.rate_r_hz == 0.0:
            left_drive = right_drive = 0.0
        return {
            "raw_rate_l_hz": float(rate_l_hz),
            "raw_rate_r_hz": float(rate_r_hz),
            "filtered_rate_l_hz": self.rate_l_hz,
            "filtered_rate_r_hz": self.rate_r_hz,
            "left_cpg_drive": left_drive,
            "right_cpg_drive": right_drive,
        }

    def describe(self) -> dict:
        return {
            "classification": "ENGINEERED neural-output-to-locomotion mapping",
            "model": "left_cpg=gain*filtered_DNa02_R; right_cpg=gain*filtered_DNa02_L",
            "gain_per_hz": self.gain_per_hz,
            "tau_ms": self.tau_ms,
            "max_drive": self.max_drive,
            "clipping": [0.0, self.max_drive],
            "tonic_drive": 0.0,
            "zero_spikes": "exactly zero CPG drive after reset",
            "biological_vnc_claim": False,
        }
