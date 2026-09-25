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

    LEGACY (``dna02-crossed-v1``): kept so earlier MVP runs stay reproducible.
    It makes DNa02 the only source of propulsion, which the literature
    contradicts: DNa02 shortens ipsilateral strides without changing forward
    speed (Yang et al. 2024, Cell).  New runs use :class:`DNCommandDecoder`.
    """

    name = "dna02-crossed-v1"
    required_status = ()

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

    def decode_reply(self, reply: dict, dt_ms: float) -> dict:
        decoded = self.decode(float(reply["dna02_rate_l"]), float(reply["dna02_rate_r"]), dt_ms)
        decoded["events"] = []
        return decoded

    def describe(self) -> dict:
        return {
            "name": self.name,
            "classification": "ENGINEERED neural-output-to-locomotion mapping (LEGACY)",
            "model": "left_cpg=gain*filtered_DNa02_R; right_cpg=gain*filtered_DNa02_L",
            "gain_per_hz": self.gain_per_hz,
            "tau_ms": self.tau_ms,
            "max_drive": self.max_drive,
            "clipping": [0.0, self.max_drive],
            "tonic_drive": 0.0,
            "zero_spikes": "exactly zero CPG drive after reset",
            "biological_vnc_claim": False,
        }


@dataclass
class DNCommandDecoder:
    """Declared DN -> FlyGym CPG decoder (``dn-v2``).

    Output per side s is the signed HybridTurningController command c_s
    (|c| = oscillator amplitude, c < 0 runs the stepping cycle backwards).
    Every rate is a per-neuron spike rate (Hz), exponentially filtered with tau_ms.

    * Forward: F_s = gain_p9 * r(DNp09 on the OTHER side).  Unilateral P9
      turns the fly toward the active side and bilateral P9 walks it forward
      (Bidaye et al. 2020, Neuron 108:469); routing it to the outer legs and
      the gain are ASSUMPTIONS.
    * Steering: A_s = F_s * max(0, 1 - k_dna02 * r(DNa02 on side s)).  DNa02
      shortens the strides of the three ipsilateral legs and leaves forward
      speed alone (Yang et al. 2024, Cell; Rayshubskiy et al. 2025, eLife),
      so DNa02 alone never makes the fly walk.  k_dna02 is an ASSUMPTION.
    * Reverse: B = gain_mdn * mean r(MDN); if B > max(F) both sides get -B
      (MDN drives backward walking and suppresses forward: Bidaye et al. 2014,
      Science).  The winner-take-all rule and the gain are ASSUMPTIONS, and
      FlyGym's reverse is the forward step replayed backwards, not the
      hindleg-led MDN program (Feng et al. 2020, Nat Commun).
    * Giant fiber: one GF spike is enough for takeoff (von Reyn et al. 2014),
      so any GF spike in a step logs a ``takeoff_command`` event.  The FlyGym
      legs-only body cannot take off; the event is reported as unsupported.
    * |c| is clipped to max_drive = 1.2, the NeuroMechFly v2 descending-drive
      range (Wang-Chen et al. 2024), a controller range rather than biology.

    No published firing-rate calibration exists for any of these gains; they
    are declared placeholders, never fitted to a behavioural outcome.  Zero
    spikes give exactly zero command.
    """

    gain_p9_per_hz: float = 0.02      # ASSUMPTION: 50 Hz DNp09 -> NMF nominal amplitude 1.0
    k_dna02_per_hz: float = 0.01      # ASSUMPTION: 100 Hz DNa02 -> ipsilateral stride 0
    gain_mdn_per_hz: float = 0.02     # ASSUMPTION: same scale as DNp09
    tau_ms: float = 50.0              # ASSUMPTION (FlyGym's own CPG amplitude tau is 50 ms)
    max_drive: float = 1.2            # NeuroMechFly v2 DN range

    name = "dn-v2"
    required_status = ("locomotion_dn_map_sha256",)
    _RATES = ("DNp09_L", "DNp09_R", "DNa02_L", "DNa02_R", "MDN_L", "MDN_R")

    def __post_init__(self) -> None:
        values = (self.gain_p9_per_hz, self.k_dna02_per_hz, self.gain_mdn_per_hz,
                  self.tau_ms, self.max_drive)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("decoder gains, tau and max_drive must be finite and positive")
        self.reset()

    def reset(self) -> None:
        self.rates = {name: 0.0 for name in self._RATES}

    def decode_reply(self, reply: dict, dt_ms: float) -> dict:
        if not (math.isfinite(dt_ms) and dt_ms > 0):
            raise ValueError("dt_ms must be positive")
        block = reply.get("locomotion_dn")
        if not isinstance(block, dict):
            raise RuntimeError("dn-v2 decoder needs the graph's locomotion_dn block; "
                               "the locomotion DN map did not resolve")
        raw = {}
        for name in self._RATES:
            value = float(block[f"{name}_rate_hz"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} rate must be finite and non-negative")
            raw[name] = value
        alpha = 1.0 - math.exp(-dt_ms / self.tau_ms)
        for name, value in raw.items():
            self.rates[name] += alpha * (value - self.rates[name])
        r = self.rates

        forward = {"L": self.gain_p9_per_hz * r["DNp09_R"], "R": self.gain_p9_per_hz * r["DNp09_L"]}
        stride = {s: max(0.0, 1.0 - self.k_dna02_per_hz * r[f"DNa02_{s}"]) for s in ("L", "R")}
        amplitude = {s: forward[s] * stride[s] for s in ("L", "R")}
        backward = self.gain_mdn_per_hz * 0.5 * (r["MDN_L"] + r["MDN_R"])
        reversing = backward > 0.0 and backward > max(forward.values())
        if reversing:
            command = {"L": -backward, "R": -backward}
        else:
            command = amplitude
        cap = self.max_drive
        left = max(-cap, min(cap, command["L"]))
        right = max(-cap, min(cap, command["R"]))
        events = []
        gf_spikes = int(block.get("GF_L_spikes", 0)) + int(block.get("GF_R_spikes", 0))
        if gf_spikes > 0:
            events.append({"event": "takeoff_command", "gf_spikes": gf_spikes,
                           "body_action": "UNSUPPORTED: FlyGym legs-only body cannot take off"})
        return {
            "raw_rates_hz": raw,
            "filtered_rates_hz": dict(r),
            "forward_drive": forward,
            "ipsilateral_stride_factor": stride,
            "backward_drive": backward,
            "program": "reverse" if reversing else ("forward" if max(forward.values()) > 0 else "none"),
            "left_cpg_drive": left,
            "right_cpg_drive": right,
            "events": events,
        }

    def describe(self) -> dict:
        return {
            "name": self.name,
            "classification": "ENGINEERED DN-to-CPG mapping; signs and laterality sourced, gains assumed",
            "model": {
                "forward": "F_s = gain_p9 * r(DNp09 contralateral to s)",
                "steering": "A_s = F_s * max(0, 1 - k_dna02 * r(DNa02_s))",
                "reverse": "B = gain_mdn * mean r(MDN); if B > max(F): c = (-B, -B)",
                "takeoff": "GF spike >= 1 -> takeoff_command event, body action UNSUPPORTED",
                "clip": "|c| <= max_drive",
            },
            "parameters": {
                "gain_p9_per_hz": [self.gain_p9_per_hz, "ASSUMPTION"],
                "k_dna02_per_hz": [self.k_dna02_per_hz, "ASSUMPTION"],
                "gain_mdn_per_hz": [self.gain_mdn_per_hz, "ASSUMPTION"],
                "tau_ms": [self.tau_ms, "ASSUMPTION"],
                "max_drive": [self.max_drive, "NeuroMechFly v2 DN range (Wang-Chen et al. 2024)"],
            },
            "sources": {
                "DNp09": "Bidaye et al. 2020, Neuron 108:469",
                "DNa02": "Yang et al. 2024, Cell; Rayshubskiy et al. 2025, eLife 102230",
                "MDN": "Bidaye et al. 2014, Science 344:97; Feng et al. 2020, Nat Commun 11:6166",
                "GF": "von Reyn et al. 2014, Nat Neurosci 17:962",
            },
            "unsupported": [
                "takeoff/jump/flight (legs-only body)",
                "hindleg-led backward kinematics (FlyGym reverse replays the forward step)",
                "speed via step frequency (amplitude only, 12 Hz base CPG)",
                "active halting (only drive-to-zero)",
            ],
            "tonic_drive": 0.0,
            "zero_spikes": "exactly zero CPG drive",
            "biological_vnc_claim": False,
        }


DECODERS = {"dn-v2": DNCommandDecoder, "dna02-crossed-v1": DNa02CPGDecoder}
