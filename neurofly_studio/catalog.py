"""What the studio offers: the 14 paradigms, their badges, and what can be tweaked.

Badges come straight from docs/CAPABILITY_MATRIX.md, parsed at request time, so
the studio can never show a paradigm as more proven than the matrix does:

* ``Validated``   the matrix's "Connectome status" starts with "Tested on v3";
* ``Mapped``      it starts with "Mapped" (wired, no passing v3 receipt);
* ``Exploratory`` anything else.

Only paradigms with a ``StudioParadigm`` entry can be built and queued.  Today
that is optomotor, the one stimulus the embodied loop (neurofly_body) drives.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "docs" / "CAPABILITY_MATRIX.md"
SPECS_DIR = PROJECT_ROOT / "validation" / "specs"

# Paradigm id in the studio and dashboard -> "paradigm" field of validation specs.
SPEC_PARADIGM = {"optomotor": "optomotor", "looming-escape": "looming", "t-maze": "tmaze_odor"}

_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]*?)\s*\(`([a-z0-9-]+)`\)\s*\|(.*)\|\s*$")


@dataclass(frozen=True)
class Parameter:
    name: str             # key in the experiment file's "parameters"
    flag: str             # neurofly_body run flag
    label: str
    unit: str
    minimum: float
    maximum: float
    default: float
    step: float
    integer: bool = False
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "unit": self.unit, "min": self.minimum,
                "max": self.maximum, "default": self.default, "step": self.step,
                "integer": self.integer, "help": self.help}


@dataclass(frozen=True)
class StudioParadigm:
    id: str
    runner: str
    explanation: str
    parameters: tuple[Parameter, ...]
    controls: dict[str, str] = field(default_factory=dict)


# Input ranges, not validity claims.  Rationale in docs/EXPERIMENT_STUDIO.md.
OPTOMOTOR = StudioParadigm(
    id="optomotor",
    runner="neurofly_body",
    explanation=(
        "The fly stands on a flat floor inside a patterned world that rotates around it. "
        "Real flies turn with the motion to keep their view steady (the optomotor response). "
        "Here the rotation drives direction-selective motion neurons of the MaleCNS connectome, "
        "the whole graph is simulated, and descending neurons such as DNa02 steer the six-legged "
        "FlyGym body."
    ),
    parameters=(
        Parameter("world_angular_velocity_rad_s", "--world-angular-velocity-rad-s",
                  "World rotation speed", "rad/s", -12.0, 12.0, 4.0, 0.5,
                  help="A negative value reverses the direction. 0 is a no-motion control. "
                       "4 rad/s is about 230 degrees per second."),
        Parameter("contrast", "--contrast", "Pattern contrast", "", 0.0, 1.0, 1.0, 0.05,
                  help="From 0 to 1. 0 means the pattern is invisible (a no-stimulus control)."),
        Parameter("duration_s", "--duration", "Simulated time", "s", 0.5, 60.0, 5.0, 0.1,
                  help="The full simulation runs slower than real time, so runs wait in a "
                       "queue and you watch them afterwards at the fly's own speed."),
        Parameter("seed", "--seed", "Random seed", "", 0, 2_147_483_647, 1, 1, integer=True,
                  help="Same seed and settings give a bit-identical run."),
    ),
    controls={
        "output-disconnected": "Brain disconnected from the legs: the connectome runs and is "
                               "recorded, but its commands never reach the body.",
    },
)

STUDIO_PARADIGMS: dict[str, StudioParadigm] = {OPTOMOTOR.id: OPTOMOTOR}


def badge_for(connectome_status: str) -> str:
    text = connectome_status.strip().lstrip("*")
    if text.startswith("Tested on v3"):
        return "Validated"
    if text.startswith("Mapped"):
        return "Mapped"
    return "Exploratory"


def read_matrix(path: Path = MATRIX_PATH) -> list[dict[str, str]]:
    """Rows of the 14-paradigm table: id, title, connectome status, badge."""
    header: list[str] | None = None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None and len(cells) > 3 and cells[:2] == ["#", "Paradigm"]:
            header = cells
            continue
        match = _ROW.match(line)
        if header is None or not match:
            continue
        values = dict(zip(header, cells))
        status = values.get("Connectome status", "")
        rows.append({"number": int(match.group(1)), "title": match.group(2), "id": match.group(3),
                     "connectome_status": status, "badge": badge_for(status),
                     "receipts": values.get("Receipts", "")})
    if not rows:
        raise ValueError(f"{path} has no paradigm matrix")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregistered_specs(specs_dir: Path = SPECS_DIR) -> dict[str, list[dict[str, str]]]:
    """Preregistered validation specs by validation paradigm name (read-only listing)."""
    found: dict[str, list[dict[str, str]]] = {}
    for path in sorted(specs_dir.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("status") != "preregistered" or "paradigm" not in spec:
            continue
        found.setdefault(spec["paradigm"], []).append({
            "id": spec.get("id", path.stem), "path": str(path.relative_to(PROJECT_ROOT)),
            "declared_at": spec.get("declared_at", ""), "sha256": _sha256(path)})
    return found


def catalog(matrix_path: Path = MATRIX_PATH, specs_dir: Path = SPECS_DIR) -> dict[str, Any]:
    specs = preregistered_specs(specs_dir)
    paradigms = []
    for row in read_matrix(matrix_path):
        studio = STUDIO_PARADIGMS.get(row["id"])
        paradigms.append({
            **row,
            "buildable": studio is not None,
            "explanation": studio.explanation if studio else None,
            "parameters": [p.to_dict() for p in studio.parameters] if studio else [],
            "controls": dict(studio.controls) if studio else {},
            "preregistered_specs": specs.get(SPEC_PARADIGM.get(row["id"], ""), []),
        })
    return {"schema": "neurofly-studio-catalog-v1", "matrix": str(matrix_path.name),
            "paradigms": paradigms}
