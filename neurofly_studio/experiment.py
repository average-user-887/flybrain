"""Experiment files: one exploratory experiment, expanded into queued runs.

An experiment file is the unit a citizen builds in the browser and a researcher
writes by hand::

    {
      "schema": "neurofly-studio-experiment-v1",
      "title": "Slow rotation, half contrast",
      "paradigm": "optomotor",
      "parameters": {"world_angular_velocity_rad_s": 2.0, "contrast": 0.5,
                     "duration_s": 5.0, "seed": 1},
      "repeats": 1,
      "control": "output-disconnected",
      "controller": "connectome"
    }

``repeats`` queues seeds ``seed .. seed+repeats-1``; ``control`` adds, for each
seed, a paired control run with the same seed.  ``controller`` may be
"modular" (the researcher baseline, never offered in the citizen page).

Every run is labelled exploratory.  The file cannot name full-fidelity knobs
(neural or physics time step, recording rate): the studio always uses the
runner's full-accuracy defaults, and the run's manifest records them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .catalog import STUDIO_PARADIGMS

SCHEMA = "neurofly-studio-experiment-v1"
LABEL = "exploratory"
MAX_REPEATS = 6
CONTROLLERS = ("connectome", "modular")
_KEYS = {"schema", "title", "paradigm", "parameters", "repeats", "control", "controller", "label"}
_SLUG = re.compile(r"[^a-z0-9]+")


class ExperimentError(ValueError):
    """The experiment file is not acceptable; the message says why."""


@dataclass(frozen=True)
class PlannedRun:
    name: str
    role: str          # "experiment" or the control name
    seed: int
    argv: list[str]    # neurofly_body run arguments, without --output


def validate(raw: Any) -> dict[str, Any]:
    """Return a normalised experiment, or raise ExperimentError."""
    if not isinstance(raw, dict):
        raise ExperimentError("an experiment is a JSON object")
    unknown = sorted(set(raw) - _KEYS)
    if unknown:
        raise ExperimentError(f"unknown field(s): {', '.join(unknown)}")
    if raw.get("schema", SCHEMA) != SCHEMA:
        raise ExperimentError(f"schema must be {SCHEMA!r}")
    if raw.get("label", LABEL) != LABEL:
        raise ExperimentError("studio runs are always exploratory; confirmatory runs go through "
                              "`neurofly validate run` with a preregistered spec")
    paradigm = STUDIO_PARADIGMS.get(raw.get("paradigm"))
    if paradigm is None:
        raise ExperimentError(f"paradigm must be one of: {', '.join(sorted(STUDIO_PARADIGMS))}")
    title = raw.get("title") or paradigm.id
    if not isinstance(title, str) or len(title) > 120 or any(ord(c) < 32 for c in title):
        raise ExperimentError("title: text, at most 120 characters")

    given = raw.get("parameters") or {}
    if not isinstance(given, dict):
        raise ExperimentError("parameters must be an object")
    known = {p.name: p for p in paradigm.parameters}
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ExperimentError(f"{paradigm.id} has no parameter(s): {', '.join(unknown)}")
    parameters: dict[str, Any] = {}
    for name, spec in known.items():
        value = given.get(name, spec.default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ExperimentError(f"{name}: a finite number")
        if spec.integer:
            if value != int(value):
                raise ExperimentError(f"{name}: a whole number")
            value = int(value)
        else:
            value = float(value)
        if not spec.minimum <= value <= spec.maximum:
            raise ExperimentError(f"{name}: between {spec.minimum} and {spec.maximum} {spec.unit}".rstrip())
        parameters[name] = value
    duration = parameters.get("duration_s")
    if duration is not None and not math.isclose(duration * 10, round(duration * 10), abs_tol=1e-9):
        raise ExperimentError("duration_s: in steps of 0.1 s")

    repeats = raw.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        raise ExperimentError(f"repeats: a whole number from 1 to {MAX_REPEATS}")
    if "seed" in parameters and parameters["seed"] + repeats - 1 > known["seed"].maximum:
        raise ExperimentError("seed + repeats is out of range")
    control = raw.get("control")
    if control is not None and control not in paradigm.controls:
        raise ExperimentError(f"control must be one of: {', '.join(paradigm.controls) or 'none'}")
    controller = raw.get("controller", "connectome")
    if controller not in CONTROLLERS:
        raise ExperimentError(f"controller must be one of: {', '.join(CONTROLLERS)}")
    return {"schema": SCHEMA, "title": title, "paradigm": paradigm.id, "parameters": parameters,
            "repeats": repeats, "control": control, "controller": controller, "label": LABEL}


def slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:40] or "run"


def plan(experiment: dict[str, Any], *, graph_args: list[str] | None = None,
         now: datetime | None = None) -> list[PlannedRun]:
    """Expand a validated experiment into queue jobs (names are unique per second)."""
    paradigm = STUDIO_PARADIGMS[experiment["paradigm"]]
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    base = f"{stamp}-{slug(experiment['title'])}"
    parameters = experiment["parameters"]
    runs = []
    for offset in range(experiment["repeats"]):
        seed = int(parameters["seed"]) + offset
        argv: list[str] = []
        for spec in paradigm.parameters:
            value = seed if spec.name == "seed" else parameters[spec.name]
            argv += [spec.flag, repr(value) if isinstance(value, float) else str(value)]
        argv += ["--controller", experiment["controller"]]
        if experiment["controller"] == "connectome":
            argv += list(graph_args or [])
        roles = [("experiment", "intact")]
        if experiment["control"]:
            roles.append((experiment["control"], experiment["control"]))
        for role, mode in roles:
            suffix = "" if role == "experiment" else "-control"
            runs.append(PlannedRun(name=f"{base}-s{seed}{suffix}", role=role, seed=seed,
                                   argv=[*argv, "--mode", mode]))
    return runs
