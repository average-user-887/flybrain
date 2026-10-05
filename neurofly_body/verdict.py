"""The component's own headline evidence: does the turn follow the stimulus?

The intact/disconnected pair proves only that the FlyGym CPG moves the body
when it is driven and does not when it is not.  Noise, a constant or a sine
through the same decoder would produce the same gap.  What distinguishes the
neural signal from any other signal of the same size is the **stimulus-sign
reversal**: reverse the sign of the world's angular velocity and the turn must
reverse with it, on more than one seed.

This module runs that experiment as a single subcommand, with the same decoder
choice as ``run`` (``dn-v2`` by default, ``dna02-crossed-v1`` for the legacy
mapping).  The graph and the body are loaded once and reset between
conditions, because the graph load costs tens of seconds; a reuse check at the
end re-runs the first condition from the reused state and requires the same
telemetry bytes as its fresh run.  Only a reuse check over the whole condition
(``--reuse-check-duration`` equal to ``--duration``) counts as the comparability
control; the default 0.1 s window is a smoke comparison and leaves the verdict
UNVERIFIED.  Every intact and
output-disconnected condition also records a ``run``-compatible invocation, so
``python -m neurofly_body replay-check <condition dir>`` re-runs it standalone
on a freshly loaded server.

``--full-controls`` adds the conditions that bound the interpretation:

* zero stimulus, which shows what the fly does with no stimulus at all;
* the output-disconnected control;
* two rate-matched controls, built by rearranging the intact run's own recorded
  drive so that the per-channel mean reaching the CPG is preserved exactly while
  its neural content is not.  ``channel-swapped`` exchanges left and right, which
  tests whether the body follows WHICH side the command came from.
  ``time-shuffled`` permutes the time order of the drive pairs under one fixed
  seed.  It shows only whether same-sign turning survives that one
  permutation; a single permutation cannot show that the time course is
  irrelevant or that the turn is carried by the mean alone.

The stimulus is not seen: the server injects the motion sign directly into the
T4/T5 direction-selective neurons (``brainlab/cosim_server.py``, the WP5
optomotor encoder).  A verdict is therefore about the circuit downstream of
T4/T5 and the body, never about motion computation from photoreceptors.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import __version__
from .decoder import DNa02CPGDecoder, DNCommandDecoder
from .runner import (
    MIN_USEFUL_DNA02_SPIKES,
    TELEMETRY_FORMAT_VERSION,
    EmbodiedConfig,
    run_embodied,
)

# A turn counts as following the stimulus only past this magnitude.  Measured
# zero-stimulus runs drift by a few milliradians, and measured driven runs reach
# 1-3 rad, so this sits two orders of magnitude clear of the noise floor.
YAW_PASS_THRESHOLD_RAD = 0.2

# Fixed permutation seed, so the time-shuffled control is reproducible.
SHUFFLE_SEED = 20260926

# Identities a reproduction must match, copied from the server status.
REPRODUCE_DATA_IDS = (
    "graph_sha256",
    "graph_content_sha256",
    "neuron_map_sha256",
    "neuron_map_content_sha256",
    "io_map_sha256",
    "optomotor_io_map_sha256",
    "locomotion_dn_map_sha256",
    "num_neurons",
)
REPRODUCE_DYNAMICS_IDS = (
    "lif_dynamics_version",
    "lif_dynamics_pin",
    "controller_version",
    "transmitter_policy",
    "engineered_assistance_enabled",
)

# PASS and FAIL are the only validated outcomes.  INCONCLUSIVE: too few DNa02
# spikes to judge.  UNVERIFIED: the comparability control (a reuse check over a
# whole condition) was skipped or ran over only part of a condition, so the
# turns may be read but not relied on.  INVALID: the reuse check found
# divergent telemetry, so the conditions are not comparable at all.
VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "UNVERIFIED", "INVALID")

# What the verdict's decoders are built from, for the receipt.
DECODER_INPUTS = {
    "dn-v2": (
        "the dn-v2 decoder: DNp09 sets the forward drive of the contralateral legs, "
        "DNa02 shortens ipsilateral strides, MDN reverses, GF logs a takeoff event"
    ),
    "dna02-crossed-v1": (
        "the legacy dna02-crossed-v1 decoder: one DNa02 per side drives the "
        "contralateral CPG, and is the only source of propulsion"
    ),
}


@dataclass(frozen=True)
class Condition:
    """One run in the verdict plan."""

    name: str
    seed: int
    world_angular_velocity_rad_s: float
    role: str
    mode: str = "intact"
    drive_from: str | None = None
    drive_construction: str | None = None


@dataclass
class Outcome:
    """What one condition measured."""

    condition: Condition
    summary: dict[str, Any]
    final_yaw_rad: float
    cumulative_yaw_rad: float
    planar_displacement_mm: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def dna02_spikes_l(self) -> int:
        return int(self.summary["total_dna02_spikes_l"])

    @property
    def dna02_spikes_r(self) -> int:
        return int(self.summary["total_dna02_spikes_r"])

    @property
    def dna02_spike_count(self) -> int:
        return int(self.summary["dna02_spike_count"])

    def row(self) -> dict[str, Any]:
        inputs = self.summary.get("decoder_input_spikes") or {}
        return {
            "condition": self.condition.name,
            "role": self.condition.role,
            "mode": self.condition.mode,
            "seed": self.condition.seed,
            "world_angular_velocity_rad_s": self.condition.world_angular_velocity_rad_s,
            "drive_construction": self.condition.drive_construction,
            "dna02_spikes_l": self.dna02_spikes_l,
            "dna02_spikes_r": self.dna02_spikes_r,
            "dna02_spike_count": self.dna02_spike_count,
            "dna02_spike_count_warning": self.summary["dna02_spike_count_warning"],
            "decoder_input_spikes": dict(inputs),
            "mean_applied_drive_l": self.summary["mean_applied_drive_l"],
            "mean_applied_drive_r": self.summary["mean_applied_drive_r"],
            "steps_with_applied_drive_above_1pct_of_max": self.summary[
                "steps_with_applied_drive_above_1pct_of_max"
            ],
            # The turn is judged on the cumulative value.  ``final_yaw_rad`` is
            # ``atan2``-wrapped into (-pi, pi], so a turn past half a revolution
            # comes back with the WRONG SIGN there: this was measured at
            # seed 1, +4 rad/s, whose wrapped yaw read -3.116 rad for a real
            # turn of +3.168 rad.
            "final_yaw_rad": self.final_yaw_rad,
            "cumulative_yaw_rad": self.cumulative_yaw_rad,
            "planar_displacement_mm": self.planar_displacement_mm,
            "decoder_produced_output": self.summary["decoder_produced_output"],
            "motor_output_reached_body": self.summary["motor_output_reached_body"],
            "trajectory_sha256": self.summary["trajectory_sha256"],
            "records": self.summary["records"],
            "wall_time_s": self.summary["wall_time_s"],
            **self.extras,
        }


def plan(
    seeds: Sequence[int], angular_velocity: float, full_controls: bool
) -> list[Condition]:
    """The conditions to run, reversal first so the controls can reuse its drive."""
    if len(seeds) < 2:
        raise SystemExit("the reversal needs at least two seeds (--seeds 0 1)")
    if angular_velocity <= 0:
        raise SystemExit("--angular-velocity must be positive; both signs are run")

    conditions: list[Condition] = []
    for seed in seeds:
        conditions.append(
            Condition(f"seed{seed}-positive-w", seed, +angular_velocity, "reversal")
        )
        conditions.append(
            Condition(f"seed{seed}-negative-w", seed, -angular_velocity, "reversal")
        )
    if full_controls:
        base = seeds[0]
        source = f"seed{base}-positive-w"
        conditions += [
            Condition(f"seed{base}-zero-w", base, 0.0, "baseline"),
            Condition(
                f"seed{base}-output-disconnected",
                base,
                +angular_velocity,
                "control",
                mode="output-disconnected",
            ),
            Condition(
                f"seed{base}-drive-channel-swapped",
                base,
                +angular_velocity,
                "control",
                mode="drive-rate-matched-control",
                drive_from=source,
                drive_construction="channel-swapped",
            ),
            Condition(
                f"seed{base}-drive-time-shuffled",
                base,
                +angular_velocity,
                "control",
                mode="drive-rate-matched-control",
                drive_from=source,
                drive_construction="time-shuffled",
            ),
        ]
    return conditions


def recorded_applied_drive(run_dir: Path) -> list[tuple[float, float]]:
    """The per-step command that actually reached the body in a finished run."""
    drive = []
    with (run_dir / "telemetry.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)["motor"]["applied_cpg_drive"]
            drive.append((float(pair[0]), float(pair[1])))
    return drive


def build_drive_override(
    construction: str, drive: Sequence[tuple[float, float]]
) -> tuple[Callable[[int, tuple[float, float]], tuple[float, float]], dict[str, Any]]:
    """A rate-matched rearrangement of a recorded intact drive sequence."""
    if construction == "channel-swapped":
        sequence = [(right, left) for left, right in drive]
        note = (
            "the recorded intact drive with its left and right channels exchanged. "
            "Every value and every total reaching the CPG is identical; only which "
            "side receives it is mirrored. If the body follows which side the "
            "command came from, the turn reverses."
        )
    elif construction == "time-shuffled":
        order = np.random.default_rng(SHUFFLE_SEED).permutation(len(drive))
        sequence = [drive[int(index)] for index in order]
        note = (
            "the recorded intact drive with its time order permuted (both channels "
            "moved together, so each channel's values and mean are exactly "
            "preserved and the left/right pairing is intact), using one fixed "
            f"permutation (seed {SHUFFLE_SEED}). It tests only whether same-sign "
            "turning survives that one permutation; it cannot show that the time "
            "course is irrelevant or that the turn is carried by the mean alone."
        )
    else:  # pragma: no cover - the CLI restricts the choices
        raise ValueError(f"unknown drive construction {construction!r}")

    zero = (0.0, 0.0)

    def override(step_index: int, _decoded: tuple[float, float]) -> tuple[float, float]:
        return sequence[step_index] if step_index < len(sequence) else zero

    description = {
        "construction": construction,
        "note": note,
        "source_steps": len(drive),
        "shuffle_seed": SHUFFLE_SEED if construction == "time-shuffled" else None,
        "rate_matched": True,
        "mean_left": sum(pair[0] for pair in sequence) / max(len(sequence), 1),
        "mean_right": sum(pair[1] for pair in sequence) / max(len(sequence), 1),
    }
    return override, description


def cumulative_yaw_rad(run_dir: Path) -> float:
    """Total turn over the run, unwrapped.

    ``thorax.yaw_rad`` is an ``atan2`` angle in (-pi, pi], so it is useless as a
    signed turn once the fly turns past half a revolution: a measured +3.168 rad
    turn read -3.116 rad there.  Summing the wrapped per-record differences
    recovers the real total.  The fly spawns with identity rotation, so the first
    record's yaw is itself the first increment.
    """
    total = 0.0
    previous = 0.0
    with (run_dir / "telemetry.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            yaw = float(json.loads(line)["body"]["thorax"]["yaw_rad"])
            delta = yaw - previous
            total += math.atan2(math.sin(delta), math.cos(delta))
            previous = yaw
    return total


def start_position_mm(run_dir: Path) -> tuple[float, float]:
    """Planar thorax position in the first record (one neural step after reset).

    The fly does not spawn at the origin after the warm-up settle, so the
    distance of the final position from the origin overstates travel by about
    0.6 mm; displacement is measured from here instead.
    """
    with (run_dir / "telemetry.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    x, y, _z = first["body"]["thorax"]["position_mm"]
    return float(x), float(y)


def _measure(condition: Condition, run_dir: Path, summary: dict[str, Any]) -> Outcome:
    thorax = summary["final_thorax"]
    x, y, _z = thorax["position_mm"]
    x0, y0 = start_position_mm(run_dir)
    return Outcome(
        condition=condition,
        summary=summary,
        final_yaw_rad=float(thorax["yaw_rad"]),
        cumulative_yaw_rad=cumulative_yaw_rad(run_dir),
        planar_displacement_mm=math.hypot(float(x) - x0, float(y) - y0),
        extras={"run_dir": run_dir.name},
    )


def judge(
    outcomes: dict[str, Outcome],
    seeds: Sequence[int],
    threshold: float,
    reuse_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass only if the turn follows the stimulus sign in both directions, every seed.

    A reuse check that ran and found divergent telemetry makes the verdict
    INVALID whatever the turns did: the conditions were then not run on
    comparable reset states, so no comparison between them can stand.

    Comparability is a required control.  If the reuse check is missing, was
    skipped, or covered only part of a condition (the default 0.1 s window is a
    smoke comparison), the verdict is UNVERIFIED, never PASS or FAIL; what the
    turns alone gave is kept in ``turn_outcome``.
    """
    per_seed = []
    for seed in seeds:
        positive = outcomes[f"seed{seed}-positive-w"]
        negative = outcomes[f"seed{seed}-negative-w"]
        # Cumulative, never the wrapped final yaw: see cumulative_yaw_rad.
        turns_positive = positive.cumulative_yaw_rad > threshold
        turns_negative = negative.cumulative_yaw_rad < -threshold
        undersampled = [
            outcome.condition.name
            for outcome in (positive, negative)
            if outcome.dna02_spike_count < MIN_USEFUL_DNA02_SPIKES
        ]
        per_seed.append(
            {
                "seed": seed,
                "cumulative_yaw_positive_w_rad": positive.cumulative_yaw_rad,
                "cumulative_yaw_negative_w_rad": negative.cumulative_yaw_rad,
                "wrapped_final_yaw_positive_w_rad": positive.final_yaw_rad,
                "wrapped_final_yaw_negative_w_rad": negative.final_yaw_rad,
                "turn_follows_positive_stimulus": turns_positive,
                "turn_follows_negative_stimulus": turns_negative,
                "turn_follows_stimulus_sign": turns_positive and turns_negative,
                # Reported, not gated: the DNa02 asymmetry that should drive it.
                "dna02_asymmetry_positive_w": positive.dna02_spikes_l
                - positive.dna02_spikes_r,
                "dna02_asymmetry_negative_w": negative.dna02_spikes_l
                - negative.dna02_spikes_r,
                "dna02_asymmetry_reverses": (
                    positive.dna02_spikes_l - positive.dna02_spikes_r > 0
                    > negative.dna02_spikes_l - negative.dna02_spikes_r
                ),
                "undersampled_conditions": undersampled,
            }
        )

    all_follow = all(entry["turn_follows_stimulus_sign"] for entry in per_seed)
    undersampled = sorted(
        {name for entry in per_seed for name in entry["undersampled_conditions"]}
    )
    if undersampled:
        turn_outcome = "INCONCLUSIVE"
        turn_reason = (
            "at least one reversal condition produced fewer than "
            f"{MIN_USEFUL_DNA02_SPIKES} DNa02 spikes ({', '.join(undersampled)}); "
            "raise --duration"
        )
    elif all_follow:
        turn_outcome = "PASS"
        turn_reason = (
            "turn direction followed the stimulus sign in both directions on all "
            f"{len(per_seed)} seeds, with |cumulative yaw| > {threshold} rad"
        )
    else:
        turn_outcome = "FAIL"
        turn_reason = "at least one seed did not turn with the stimulus sign"

    check = reuse_check or {}
    if check.get("ran") and not check.get("bit_identical"):
        verdict = "INVALID"
        reason = (
            "the reuse check failed (" + str(check.get("note", "divergent telemetry"))
            + "), so the conditions are not comparable and the turns are not judged"
        )
    elif not check.get("ran"):
        verdict = "UNVERIFIED"
        reason = (
            "the reuse check did not run ("
            + str(check.get("note", "no comparability check supplied"))
            + "), so comparability of the conditions is unverified; the turns alone "
            f"give {turn_outcome}: {turn_reason}"
        )
    elif not check.get("covers_full_condition"):
        verdict = "UNVERIFIED"
        reason = (
            f"the reuse check compared only {check.get('records_compared')} of "
            f"{check.get('condition_records')} records of a condition, a smoke "
            "comparison rather than a full-duration fresh-versus-reused check, so "
            f"comparability is unverified; the turns alone give {turn_outcome}: "
            f"{turn_reason} (rerun with --reuse-check-duration equal to --duration)"
        )
    else:
        verdict, reason = turn_outcome, turn_reason
    return {
        "verdict": verdict,
        "reason": reason,
        "turn_outcome": turn_outcome,
        "yaw_pass_threshold_rad": threshold,
        "min_useful_dna02_spikes": MIN_USEFUL_DNA02_SPIKES,
        "per_seed": per_seed,
    }


def control_findings(
    outcomes: dict[str, Outcome], threshold: float
) -> list[dict[str, Any]]:
    """What each control did, measured against the intact run it was built from."""
    findings = []
    for outcome in outcomes.values():
        condition = outcome.condition
        if condition.role != "control":
            continue
        entry: dict[str, Any] = {
            "condition": condition.name,
            "cumulative_yaw_rad": outcome.cumulative_yaw_rad,
            "planar_displacement_mm": outcome.planar_displacement_mm,
        }
        if condition.mode == "output-disconnected":
            entry["turn_absent"] = abs(outcome.cumulative_yaw_rad) <= threshold
            entry["reading"] = (
                "the turn is absent when the output path is cut"
                if entry["turn_absent"]
                else "the body still turns with the output path cut"
            )
        elif condition.drive_construction is not None:
            source = outcomes[str(condition.drive_from)]
            entry["source_cumulative_yaw_rad"] = source.cumulative_yaw_rad
            same_sign = math.copysign(1.0, outcome.cumulative_yaw_rad) == math.copysign(
                1.0, source.cumulative_yaw_rad
            )
            beyond = abs(outcome.cumulative_yaw_rad) > threshold
            if abs(source.cumulative_yaw_rad) <= threshold:
                entry["informative"] = False
                entry["reading"] = (
                    "uninformative: the intact run it was built from did not turn "
                    f"beyond {threshold} rad, so there was no turn to reverse or keep"
                )
            elif condition.drive_construction == "channel-swapped":
                entry["turn_reversed"] = beyond and not same_sign
                entry["reading"] = (
                    "mirroring the sides reverses the turn: the body follows which "
                    "side the command came from"
                    if entry["turn_reversed"]
                    else "mirroring the sides does NOT reverse the turn"
                )
            else:
                entry["turn_survived"] = beyond and same_sign
                magnitude = (
                    f"|yaw| {abs(source.cumulative_yaw_rad):.3f} -> "
                    f"{abs(outcome.cumulative_yaw_rad):.3f} rad"
                )
                entry["reading"] = (
                    "same-sign turning survives this one time permutation "
                    f"(seed {SHUFFLE_SEED}), with a changed magnitude ({magnitude}); "
                    "one permutation does not show that the time course is irrelevant"
                    if entry["turn_survived"]
                    else "same-sign turning beyond the threshold does not survive this "
                    f"one time permutation (seed {SHUFFLE_SEED}; {magnitude})"
                )
        findings.append(entry)
    return findings


def interpretation(
    decoder_name: str,
    judgement: dict[str, Any],
    outcomes: dict[str, Outcome],
    duration_s: float,
    brain_backend: str | None = None,
) -> list[str]:
    """The receipt's reading, built from the measured numbers, never from memory."""
    reversal = [o for o in outcomes.values() if o.condition.role == "reversal"]
    seeds = len(judgement.get("per_seed", []))
    lines = [
        "The input is imposed, not seen: the stimulus sign is injected directly into "
        "the T4/T5 direction-selective neurons by the optomotor encoder, so no "
        "photoreceptor motion computation is tested. Any result here is about the "
        "circuit downstream of T4/T5 and the body only."
    ]
    if judgement["verdict"] == "PASS":
        lines.append(
            "What it establishes, and only within this scope: stimulus-sign-appropriate "
            f"turning on the `{brain_backend or 'unrecorded'}` brain backend, over "
            f"{seeds} seeds, with T4/T5-imposed input, of an articulated fly driven "
            f"through an engineered decoder ({DECODER_INPUTS.get(decoder_name, decoder_name)}), "
            "on the real v3 MaleCNS graph."
        )
    else:
        lines.append(
            f"The reversal did not pass ({judgement['verdict']}: {judgement['reason']}), "
            "so this run does not establish stimulus-sign-appropriate turning through "
            f"{DECODER_INPUTS.get(decoder_name, decoder_name)}."
        )
    if reversal:
        low = min(o.planar_displacement_mm for o in reversal)
        high = max(o.planar_displacement_mm for o in reversal)
        lines.append(
            "What it does not establish: behavioural competence. In the reversal "
            f"conditions the fly travels {low:.1f}-{high:.1f} mm in {duration_s:g} s, "
            "against 10-20 mm/s for a real walking fly."
        )
    if decoder_name == "dna02-crossed-v1":
        lines.append(
            "The sample is two neurons wide: one DNa02 per side out of 166,700, and "
            "the whole motor command derives from the DNa02 spike counts in this table."
        )
    else:
        lines.append(
            "The sample is a handful of descending neurons: the DNp09 counts set the "
            "forward drive and the DNa02 counts set the turn, so the whole motor "
            "command derives from the spike counts in this table."
        )
    if decoder_name == "dn-v2" and reversal:
        totals: dict[str, int] = {}
        for outcome in reversal:
            for name, count in (outcome.summary.get("decoder_input_spikes") or {}).items():
                totals[name] = totals.get(name, 0) + int(count)
        p9 = totals.get("DNp09_L", 0) + totals.get("DNp09_R", 0)
        mdn = totals.get("MDN_L", 0) + totals.get("MDN_R", 0)
        flips = [e["dna02_asymmetry_reverses"] for e in judgement["per_seed"]]
        lines.append(
            f"Across the {len(reversal)} reversal conditions DNp09 fired {p9} times and "
            f"MDN {mdn} times; the DNa02 L-R asymmetry reversed with the stimulus on "
            f"{sum(flips)} of {len(flips)} seeds. Under dn-v2 DNa02 only shortens the "
            "strides of an existing forward drive, so a DNa02 asymmetry turns the fly "
            "only when DNp09 supplies that drive."
        )
    baseline = next((o for o in outcomes.values() if o.condition.role == "baseline"), None)
    if baseline is not None:
        inputs = baseline.summary.get("decoder_input_spikes") or {}
        spikes = ", ".join(f"{name} {count}" for name, count in sorted(inputs.items()) if count)
        lines.append(
            f"Baseline at zero stimulus (`{baseline.condition.name}`): the fly travels "
            f"{baseline.planar_displacement_mm:.2f} mm and turns "
            f"{baseline.cumulative_yaw_rad:+.3f} rad, from spontaneous graph activity "
            f"({spikes or 'no decoder-input spikes'}); the decoder has no tonic term."
        )
    lines.append(
        "The DN-to-CPG decoder is an engineered bridge, not a ventral nerve cord, and "
        "FlyGym's own leg-retraction, stumbling and adhesion machinery is active in "
        "every condition, including the disconnected control."
    )
    lines.append(
        "No closed-loop DN modulation is demonstrated: body feedback is wired, but its "
        "effect on the DN readout is not resolvable at these spike counts."
    )
    return lines


def _input_columns(receipt: dict[str, Any]) -> list[str]:
    """Decoder-input populations other than DNa02 that fired in any condition."""
    names = set()
    for row in receipt["conditions"]:
        for name, count in (row.get("decoder_input_spikes") or {}).items():
            if not name.startswith("DNa02") and count:
                names.add(name)
    return sorted(names)


def _display_path(path: str) -> str:
    """Never print an absolute host path into a receipt."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"<output>/{candidate.name}"


def _markdown(receipt: dict[str, Any]) -> str:
    extra = _input_columns(receipt)
    decoder = receipt.get("decoder", {}).get("name", "?")
    lines = [
        "# Embodied MVP verdict: stimulus-sign reversal",
        "",
        f"Verdict: **{receipt['judgement']['verdict']}** - "
        f"{receipt['judgement']['reason']}",
        "",
        *(
            [f"Turns alone: {receipt['judgement']['turn_outcome']} (not a validated "
             "verdict, because comparability was not established).", ""]
            if receipt["judgement"].get("turn_outcome")
            not in (None, receipt["judgement"]["verdict"])
            else []
        ),
        f"Generated {receipt['created_at']} by `python -m neurofly_body verdict` "
        f"(neurofly_body {receipt['package_version']}, telemetry format "
        f"{receipt['telemetry_format_version']}, decoder `{decoder}`).",
        "",
        "Every condition is the same real graph and the same articulated FlyGym body, "
        "differing only in the stimulus sign, the seed, or the declared control.",
        "",
        "| condition | role | seed | world w (rad/s) | DNa02 L | DNa02 R | "
        + "".join(f"{name} | " for name in extra)
        + "mean drive L | mean drive R | cumulative yaw (rad) | displacement (mm) |",
        "|---|---|---|---|---|---|" + "---|" * len(extra) + "---|---|---|---|",
    ]
    for row in receipt["conditions"]:
        inputs = row.get("decoder_input_spikes") or {}
        lines.append(
            "| `{condition}` | {role} | {seed} | {world_angular_velocity_rad_s:+.1f} | "
            "{dna02_spikes_l} | {dna02_spikes_r} | ".format(**row)
            + "".join(f"{inputs.get(name, 0)} | " for name in extra)
            + "{mean_applied_drive_l:.4f} | {mean_applied_drive_r:.4f} | "
            "**{cumulative_yaw_rad:+.3f}** | {planar_displacement_mm:.2f} |".format(**row)
        )
    lines += ["", "## Reversal", ""]
    for entry in receipt["judgement"]["per_seed"]:
        lines.append(
            f"- seed {entry['seed']}: +w turns {entry['cumulative_yaw_positive_w_rad']:+.3f} rad, "
            f"-w turns {entry['cumulative_yaw_negative_w_rad']:+.3f} rad - "
            f"follows the stimulus sign: {entry['turn_follows_stimulus_sign']}; "
            f"DNa02 L-R asymmetry reverses: {entry['dna02_asymmetry_reverses']}"
        )
    if receipt.get("controls") or receipt.get("control_findings"):
        lines += ["", "## Controls", ""]
        for note in receipt.get("controls", []):
            lines.append(f"- {note}")
        for finding in receipt.get("control_findings", []):
            lines.append(
                f"- measured, `{finding['condition']}`: cumulative yaw "
                f"{finding['cumulative_yaw_rad']:+.3f} rad"
                + (
                    f" (source run {finding['source_cumulative_yaw_rad']:+.3f} rad)"
                    if "source_cumulative_yaw_rad" in finding
                    else ""
                )
                + f" - {finding['reading']}"
            )
    lines += ["", "## What this does and does not establish", ""]
    for item in receipt["interpretation"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Reuse check",
        "",
        f"- {receipt['reuse_check']['note']}",
    ]
    backend = receipt.get("neural_backend") or {}
    if backend:
        lines += [
            "",
            "## Identity",
            "",
            f"- graph `{str(backend.get('graph_sha256'))[:12]}`, "
            f"{backend.get('num_neurons', '?'):,} neurons"
            if isinstance(backend.get("num_neurons"), int)
            else f"- graph `{str(backend.get('graph_sha256'))[:12]}`",
            f"- brain backend `{backend.get('brain_backend')}`, LIF dynamics "
            f"`{backend.get('lif_dynamics_version')}`, transmitter policy "
            f"`{backend.get('transmitter_policy')}`, engineered assistance "
            f"`{backend.get('engineered_assistance_enabled')}`",
            f"- optomotor IO map `{str(backend.get('optomotor_io_map_sha256'))[:12]}`, "
            f"locomotion DN map `{str(backend.get('locomotion_dn_map_sha256'))[:12]}`",
            f"- {receipt.get('seed_handling', '')}",
            f"- wall time {receipt.get('wall_time_s', 0.0) / 60.0:.1f} min for "
            f"{len(receipt['conditions'])} conditions plus the reuse check",
        ]
    reproduce = receipt.get("reproduce") or {}
    if reproduce.get("command"):
        backend_name = reproduce.get("brain_backend")
        lines += [
            "",
            "## Reproduce",
            "",
            f"The brain backend is pinned to `{backend_name}`, the backend that produced "
            "these numbers. Trajectories depend on the backend: the same condition gives "
            "different trajectories on CPU and CUDA, and whether the two agree as "
            "ensembles has not been tested, so an unpinned (`auto`) backend may not "
            "reproduce them. Every parameter is given "
            "explicitly, defaults included, and floats are written exactly. Replace "
            "`<fresh-dir>` with a new directory. `<repo>` is the root of the checkout "
            "the command runs from (the CLI maps it); a data directory outside the "
            "repository is read from `NEUROFLY_GRAPH_DIR` / `NEUROFLY_CONNECTOME_DIR`. "
            "The files are verified against the sha256 values below when they load. "
            "The command refuses to run if a model override "
            f"({', '.join(f'`{name}`' for name in reproduce.get('refused_environment') or REFUSED_ENVIRONMENT)}) "
            "is set.",
            "",
            "```bash",
            reproduce["command"],
            "```",
            "",
            "It reproduces these numbers only against the same code, data and dynamics:",
            "",
        ]
        code = reproduce.get("code") or {}
        lines.append(
            f"- code: commit `{code.get('commit')}`"
            + (" (working tree had uncommitted changes)" if code.get("dirty") else "")
            + f", neurofly_body {code.get('neurofly_body_version')}"
        )
        for group in ("data", "dynamics"):
            for key, value in (reproduce.get(group) or {}).items():
                lines.append(f"- {group}: `{key}` = `{value}`")
        lines += [
            "",
            "Each intact or output-disconnected condition directory can also be re-run "
            f"standalone with `NEUROFLY_BRAIN_BACKEND={backend_name} python -m neurofly_body "
            "replay-check <condition dir> --output <fresh-dir>`.",
        ]
    lines += [
        "",
        f"Per-condition trajectory digests (telemetry format "
        f"{receipt['telemetry_format_version']}; format-1 digests are not comparable):",
        "",
        "| condition | records | trajectory_sha256 |",
        "|---|---|---|",
    ]
    for row in receipt["conditions"]:
        lines.append(f"| `{row['condition']}` | {row['records']} | `{row['trajectory_sha256']}` |")
    lines += [
        "",
        "## Known measurement traps",
        "",
        "- `thorax.yaw_rad` is an `atan2` angle in (-pi, pi], so it cannot be used as a "
        "signed turn: a real +3.168 rad turn once read -3.116 rad there. The verdict "
        "judges an unwrapped cumulative yaw and reports both.",
        "- A reused `ConnectomeServer` rebuilds its optomotor encoder from "
        "`optomotor_seed` on `reset()`, so that attribute is set per condition (and for "
        "the reuse check); otherwise every condition runs on one seed's noise stream.",
        "",
        "## Run artefacts",
        "",
        f"- output root: `{_display_path(receipt['output_dir'])}` (not committed)",
        "- one directory per condition, each with its own manifest, telemetry, "
        "timing and summary",
        "- `verdict.json` holds these numbers in machine-readable form",
        "",
    ]
    return "\n".join(lines)


# Environment variables that change the model while the server still reports
# the base dynamics pin.  The baseline verdict refuses them rather than
# silently becoming a sensitivity variant.  NEUROFLY_LIF_DYNAMICS is accepted
# only when it says v3, which the verdict pins explicitly anyway.
REFUSED_ENVIRONMENT = (
    "NEUROFLY_LIF_E_INH_MV",      # declared E_inh sensitivity variant (EINH_SENSITIVITY.md)
    "NEUROFLY_LIF_DYNAMICS",      # process-wide dynamics selection
    "NEUROFLY_GRADED_POLICY",     # graded-release class list (v4/v5)
    "NEUROFLY_CUDA_IMPL",         # forces one CUDA implementation
)
BRAIN_BACKENDS = ("auto", "cpu", "cuda")

# How the reproduce command names the graph and neuron-table directories.
#  * inside the repository: ``<repo>/<relative path>``; the verdict CLI maps the
#    ``<repo>`` prefix back to the checkout it runs from (neurofly.privacy).
#  * anywhere else: the documented environment variable, as a shell expansion
#    that stops with a message if it is unset.  The private location is never
#    written; the graph is identified by its sha256, which every run verifies.
LOCATION_ENV = {"graph-dir": "NEUROFLY_GRAPH_DIR", "connectome-dir": "NEUROFLY_CONNECTOME_DIR"}


def refuse_unsupported_environment(environ: Any = None) -> None:
    """Stop before any work if the environment would change the baseline model."""
    environ = os.environ if environ is None else environ
    problems = []
    for name in REFUSED_ENVIRONMENT:
        value = environ.get(name)
        if value in (None, ""):
            continue
        if name == "NEUROFLY_LIF_DYNAMICS" and value == "v3":
            continue
        problems.append(f"{name}={value}")
    backend = environ.get("NEUROFLY_BRAIN_BACKEND")
    if backend not in (None, "") and backend not in BRAIN_BACKENDS:
        problems.append(f"NEUROFLY_BRAIN_BACKEND={backend}")
    if problems:
        raise SystemExit(
            "verdict refuses to run: the baseline verdict uses the clean v3 model, and "
            + ", ".join(problems)
            + " would change it while the receipt still reported the base v3 pin. "
            "Unset it (NEUROFLY_BRAIN_BACKEND may be auto, cpu or cuda)."
        )


def portable_location(path: Any, flag: str) -> str:
    """The shell-ready, portable form of a directory the run actually used."""
    from neurofly.privacy import REPO_ROOT

    absolute = Path(os.path.abspath(os.fspath(path)))
    for root in {Path(os.path.abspath(REPO_ROOT)), Path(REPO_ROOT).resolve()}:
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            continue
        return shlex.quote("<repo>" if str(relative) == "." else f"<repo>/{relative.as_posix()}")
    env = LOCATION_ENV[flag]
    return f'"${{{env}:?set {env} to the directory with the pinned files}}"'


def restore_location(value: Any) -> Path | None:
    """Map a ``<repo>/...`` argument back to this checkout (the CLI side of the convention)."""
    if value is None:
        return None
    from neurofly.privacy import restore_local

    restored = restore_local(os.fspath(value))
    return None if restored is None else Path(restored)


def _exact(value: float) -> str:
    """A float that parses back to the identical value (``repr`` is round-trip safe)."""
    return repr(float(value))


def reproduce_command(invocation: dict[str, Any], brain_backend: str) -> str:
    """The complete command that reproduces a verdict, backend pinned.

    Every parameter is written out, defaults included, so a later change of a
    default cannot silently change what the command runs.  Floats are written
    with ``repr``, which round-trips exactly.  The backend is the one that
    actually ran (``status["brain_backend"]``), never ``auto``: CPU and CUDA
    diverge, so an unpinned backend cannot reproduce the numbers.  The graph
    and neuron-table directories are the ones the server actually loaded, in
    the portable form of :func:`portable_location`.
    """
    parts = [f"NEUROFLY_BRAIN_BACKEND={shlex.quote(str(brain_backend))}",
             "python", "-m", "neurofly_body", "verdict", "--output", "'<fresh-dir>'"]

    def option(name: str, value: Any) -> None:
        text = _exact(value) if isinstance(value, float) else str(value)
        parts.extend([f"--{name}", shlex.quote(text)])

    option("duration", float(invocation["duration_s"]))
    parts.append("--seeds")
    parts.extend(str(int(seed)) for seed in invocation["seeds"])
    option("angular-velocity", float(invocation["angular_velocity_rad_s"]))
    option("contrast", float(invocation["contrast"]))
    option("neural-dt-ms", float(invocation["neural_dt_ms"]))
    option("physics-dt-s", float(invocation["physics_dt_s"]))
    option("warmup-s", float(invocation["warmup_s"]))
    option("decoder", invocation["decoder"])
    option("decoder-tau-ms", float(invocation["decoder_tau_ms"]))
    option("max-cpg-drive", float(invocation["max_cpg_drive"]))
    option("p9-gain-per-hz", float(invocation["p9_gain_per_hz"]))
    option("dna02-stride-k-per-hz", float(invocation["dna02_stride_k_per_hz"]))
    option("mdn-gain-per-hz", float(invocation["mdn_gain_per_hz"]))
    option("cpg-gain-per-hz", float(invocation["cpg_gain_per_hz"]))
    if invocation.get("full_controls"):
        parts.append("--full-controls")
    if invocation.get("reuse_check", True):
        option("reuse-check-duration", float(invocation["reuse_check_duration_s"]))
    else:
        parts.append("--no-reuse-check")
    # Already shell-ready (portable_location); never re-quoted.
    parts.extend(["--graph-dir", invocation["graph_dir"]])
    parts.extend(["--connectome-dir", invocation["connectome_dir"]])
    return " ".join(parts)


def make_decoder(args: Any) -> DNCommandDecoder | DNa02CPGDecoder:
    """The same decoder choice and parameters as ``run``."""
    if getattr(args, "decoder", "dn-v2") == "dn-v2":
        return DNCommandDecoder(
            gain_p9_per_hz=args.p9_gain_per_hz,
            k_dna02_per_hz=args.dna02_stride_k_per_hz,
            gain_mdn_per_hz=args.mdn_gain_per_hz,
            tau_ms=args.decoder_tau_ms,
            max_drive=args.max_cpg_drive,
        )
    return DNa02CPGDecoder(
        gain_per_hz=args.cpg_gain_per_hz,
        tau_ms=args.decoder_tau_ms,
        max_drive=args.max_cpg_drive,
    )


def run_invocation(args: Any, condition: Condition) -> dict[str, Any]:
    """A ``run``-compatible invocation, so ``replay-check`` can re-run the condition.

    Rate-matched controls depend on another run's recorded drive, so they cannot
    be replayed by ``run`` and say so.
    """
    base = {"verdict_condition": condition.name}
    if condition.mode not in ("intact", "output-disconnected"):
        return {**base, "replayable_by_run": False}
    return {
        **base,
        "duration": float(args.duration),
        "mode": condition.mode,
        "graph_dir": None if args.graph_dir is None else str(args.graph_dir),
        "connectome_dir": None if args.connectome_dir is None else str(args.connectome_dir),
        "seed": int(condition.seed),
        "neural_dt_ms": args.neural_dt_ms,
        "physics_dt_s": args.physics_dt_s,
        "warmup_s": args.warmup_s,
        "world_angular_velocity_rad_s": condition.world_angular_velocity_rad_s,
        "contrast": args.contrast,
        "record_fps": 0.0,
        "controller": "connectome",
        "modular_forward_drive": 1.0,
        "modular_turn_gain": 1.0,
        "decoder": args.decoder,
        "decoder_tau_ms": args.decoder_tau_ms,
        "max_cpg_drive": args.max_cpg_drive,
        "p9_gain_per_hz": args.p9_gain_per_hz,
        "dna02_stride_k_per_hz": args.dna02_stride_k_per_hz,
        "mdn_gain_per_hz": args.mdn_gain_per_hz,
        "cpg_gain_per_hz": args.cpg_gain_per_hz,
        "silence": None,
        "motor_delay_steps": 0,
        "pipeline": False,
        "leg_load_feedback": False,
    }


def _config(args: Any, condition: Condition, run_dir: Path, duration: float) -> EmbodiedConfig:
    return EmbodiedConfig(
        duration_s=duration,
        output_dir=run_dir,
        mode=condition.mode,
        neural_dt_ms=args.neural_dt_ms,
        physics_dt_s=args.physics_dt_s,
        world_angular_velocity_rad_s=condition.world_angular_velocity_rad_s,
        contrast=args.contrast,
        seed=condition.seed,
    )


def _set_seed(neural: Any, seed: int) -> None:
    # The reused server rebuilds its optomotor encoder from ``optomotor_seed``
    # on reset(), so the seed must be set per condition.  Without this every
    # condition runs on one seed's noise stream whatever its body seed, which is
    # not what the equivalent standalone `run --seed N` does.
    if hasattr(neural, "optomotor_seed"):
        neural.optomotor_seed = int(seed)


def run_verdict(args: Any) -> int:
    """Execute the verdict plan and write the receipt.  Returns the exit code."""
    import time

    from .cli import _check_output_dir

    started = time.perf_counter()
    refuse_unsupported_environment()
    args.graph_dir = restore_location(args.graph_dir)
    args.connectome_dir = restore_location(args.connectome_dir)
    output_root = _check_output_dir(args.output).resolve()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    conditions = plan(seeds, float(args.angular_velocity), bool(args.full_controls))
    make_decoder(args)  # validate the parameters before the expensive load

    # Imported here so ``--help`` stays free of the heavyweight dependencies.
    from brainlab.cosim_server import ConnectomeServer

    from .flygym_body import FlyGymBody

    output_root.mkdir(parents=True)
    neural = ConnectomeServer(
        graph_dir=args.graph_dir,
        connectome_dir=args.connectome_dir,
        allow_synthetic=False,
        dynamics="v3",
        transmitter_policy="v3-modulatory-only",
        unclear_mode="excitatory",
        engineered_assistance=False,
        optomotor_seed=seeds[0],
    )
    body = FlyGymBody(physics_dt_s=args.physics_dt_s, warmup_s=args.warmup_s)
    equivalence_note = (
        "each condition sets the server's optomotor_seed to its own seed before "
        "the reset, as a standalone `run --seed <seed>` on a freshly loaded server "
        "does. Equivalence to such a run is not established for every condition: it "
        "is checked only for the condition the reuse check repeats, and for any "
        "condition directory re-run with `replay-check`"
    )
    outcomes: dict[str, Outcome] = {}
    control_notes: list[str] = []
    decoder_description: dict[str, Any] = {}
    try:
        for condition in conditions:
            run_dir = output_root / condition.name
            override = None
            description: dict[str, Any] | None = None
            if condition.drive_construction is not None:
                source = output_root / str(condition.drive_from)
                override, description = build_drive_override(
                    condition.drive_construction, recorded_applied_drive(source)
                )
                description["source_run"] = str(condition.drive_from)
                control_notes.append(f"`{condition.name}`: {description['note']}")
            _set_seed(neural, condition.seed)
            print(f"[verdict] {condition.name} ...", file=sys.stderr, flush=True)
            decoder = make_decoder(args)
            decoder_description = decoder.describe()
            summary = run_embodied(
                _config(args, condition, run_dir, float(args.duration)),
                neural,
                body,
                decoder=decoder,
                invocation=run_invocation(args, condition),
                drive_override=override,
                drive_override_description=description,
                close_body=False,
            )
            outcomes[condition.name] = _measure(condition, run_dir, summary)

        reuse_check = _reuse_check(args, conditions[0], output_root, neural, body)
    finally:
        body.close()

    judgement = judge(outcomes, seeds, YAW_PASS_THRESHOLD_RAD, reuse_check)
    status = neural.get_status()
    from brainlab.graph_identity import active_e_inh_mV

    if active_e_inh_mV() is not None:  # pragma: no cover - refused before the load
        raise SystemExit("an E_inh override appeared during the run; the receipt is not written")

    invocation = {
        "duration_s": float(args.duration),
        "seeds": seeds,
        "angular_velocity_rad_s": float(args.angular_velocity),
        "full_controls": bool(args.full_controls),
        "reuse_check": bool(args.reuse_check),
        "reuse_check_duration_s": float(args.reuse_check_duration),
        "decoder": args.decoder,
        "contrast": args.contrast,
        "neural_dt_ms": args.neural_dt_ms,
        "physics_dt_s": args.physics_dt_s,
        "warmup_s": args.warmup_s,
        "decoder_tau_ms": args.decoder_tau_ms,
        "max_cpg_drive": args.max_cpg_drive,
        "p9_gain_per_hz": args.p9_gain_per_hz,
        "dna02_stride_k_per_hz": args.dna02_stride_k_per_hz,
        "mdn_gain_per_hz": args.mdn_gain_per_hz,
        "cpg_gain_per_hz": args.cpg_gain_per_hz,
        # The directories the server actually loaded (argument, environment or
        # default), in the portable form the CLI resolves.
        "graph_dir": portable_location(neural.graph_dir, "graph-dir"),
        "connectome_dir": portable_location(neural.connectome_dir, "connectome-dir"),
        "graph_dir_source": str(neural.graph_dir_source),
    }
    brain_backend = str(status.get("brain_backend"))
    from provenance import source_revision

    revision = source_revision()
    reproduce = {
        "brain_backend": brain_backend,
        "brain_backend_requested": os.environ.get("NEUROFLY_BRAIN_BACKEND") or "auto",
        "refused_environment": list(REFUSED_ENVIRONMENT),
        "command": reproduce_command(invocation, brain_backend),
        # What the command must run against to reproduce the numbers.
        "code": {"commit": revision.get("commit"), "dirty": revision.get("dirty"),
                 "neurofly_body_version": __version__},
        "data": {key: status.get(key) for key in REPRODUCE_DATA_IDS},
        "dynamics": {key: status.get(key) for key in REPRODUCE_DYNAMICS_IDS},
    }
    receipt = {
        "schema": "neurofly-embodied-verdict-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "telemetry_format_version": TELEMETRY_FORMAT_VERSION,
        "output_dir": str(output_root),
        "decoder": decoder_description,
        "invocation": invocation,
        "reproduce": reproduce,
        "provenance": {"python": sys.version.split()[0], "platform": platform.machine()},
        "neural_backend": {
            key: value
            for key, value in status.items()
            if key
            in (
                "graph_sha256",
                "brain_backend",
                "lif_dynamics_version",
                "transmitter_policy",
                "engineered_assistance_enabled",
                "optomotor_io_map_sha256",
                "locomotion_dn_map_sha256",
                "num_neurons",
                "synthetic",
            )
        },
        "judgement": judgement,
        "conditions": [outcomes[c.name].row() for c in conditions],
        "controls": control_notes,
        "control_findings": control_findings(outcomes, YAW_PASS_THRESHOLD_RAD),
        "reuse_check": reuse_check,
        "seed_handling": equivalence_note,
        "interpretation": interpretation(
            str(args.decoder), judgement, outcomes, float(args.duration), brain_backend
        ),
        "wall_time_s": time.perf_counter() - started,
    }
    # Both files are meant to be published: the whole receipt goes through the
    # established sanitiser, so no absolute local path (the output directory,
    # a graph location, a path inside a note) reaches them.  The local paths
    # are printed to the console below and nowhere else.
    from neurofly.privacy import redact_local, redact_text

    (output_root / "verdict.json").write_text(
        json.dumps(redact_local(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "verdict.md").write_text(redact_text(_markdown(receipt)), encoding="utf-8")
    print(json.dumps(receipt["judgement"], indent=2, sort_keys=True))
    print(f"\nreproduce (brain backend pinned):\n  {reproduce['command']}")
    print(f"\nreceipt: {output_root / 'verdict.md'}")
    return 0 if judgement["verdict"] == "PASS" else 1


def _reuse_check(
    args: Any, first: Condition, output_root: Path, neural: Any, body: Any
) -> dict[str, Any]:
    """Compare a fresh run with the same run on the reused graph and body.

    The first condition ran on a freshly built server and body.  Re-running it
    now, after every other condition, must reproduce the same telemetry bytes.
    The claim is scoped to the window compared: only a full-duration repeat
    (``covers_full_condition``) is the comparability control the verdict
    requires; a shorter window is a smoke comparison.

    A prefix comparison is valid because no telemetry record depends on the
    run's duration or invocation: each record holds only that step's simulated
    quantities.  It is only like-with-like if the repeat runs on the first
    condition's optomotor seed, so that is set here.  Without it the repeat ran
    on the LAST condition's noise stream and reported a false divergence (seen
    at record 4 when the plan ended on seed 1).
    """
    if not args.reuse_check:
        return {
            "ran": False,
            "bit_identical": None,
            "note": "skipped (--no-reuse-check)",
        }
    duration = float(args.reuse_check_duration)
    run_dir = output_root / "_reuse-check"
    _set_seed(neural, first.seed)
    summary = run_embodied(
        _config(args, first, run_dir, duration),
        neural,
        body,
        decoder=make_decoder(args),
        invocation={"verdict_condition": "_reuse-check"},
        close_body=False,
    )
    original_all = (output_root / first.name / "telemetry.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    repeat_all = (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    condition_records = len(original_all)
    # A repeat longer than the condition is compared over the condition only.
    records = min(int(summary["records"]), condition_records)
    original = original_all[:records]
    repeat = repeat_all[:records]
    covers_full_condition = records >= condition_records
    identical = original == repeat
    first_difference = next(
        (
            index
            for index, (a, b) in enumerate(zip(original, repeat), start=1)
            if a != b
        ),
        None,
    )
    window = (
        f"all {records} records ({duration:g} s, the whole condition)"
        if covers_full_condition
        else f"the opening {records} of {condition_records} records ({duration:g} s)"
    )
    preamble = (
        f"re-ran {window} of `{first.name}` from the reused graph and body after "
        "every other condition and got "
    )
    if identical and covers_full_condition:
        note = preamble + (
            "byte-identical telemetry to the fresh run. No telemetry difference was "
            "observed over this one full-duration repeat. It does not compare internal "
            "state that the telemetry does not record, and it does not repeat the other "
            "seeds or conditions. It is the comparability check the verdict requires"
        )
    elif identical:
        note = preamble + (
            "byte-identical telemetry over that window only. This is a smoke "
            "comparison, not proof of a complete reset: state that shows up later "
            "in a run would not be seen, so comparability is UNVERIFIED"
        )
    else:
        note = preamble + (
            f"DIVERGENT telemetry at record {first_difference}: state survived a "
            "reset, so the conditions are NOT comparable"
        )
    return {
        "ran": True,
        "bit_identical": identical,
        "records_compared": records,
        "condition_records": condition_records,
        "covers_full_condition": covers_full_condition,
        "duration_s": duration,
        "repeated_condition": first.name,
        "first_differing_record": first_difference,
        "note": note,
    }
