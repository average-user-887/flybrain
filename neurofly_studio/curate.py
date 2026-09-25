"""Curated runs: finished, replay-verified runs that ship as the studio's entry point.

    python -m neurofly_studio curate RUN_DIR [CONTROL_RUN_DIR] --name optomotor-intro \\
        --title "..." --explanation "..."

A run can be curated only if it completed and ``neurofly_body replay-check``
reproduced it bit for bit (its ``replay_check.json`` says BIT_IDENTICAL), so
every curated run is a checked, reproducible result of the published code.
Curation copies what the gallery needs (manifest.json, summary.json,
body.nfbody, replay_check.json) into ``experiment_data/curated/<name>/`` and
writes ``curated.json`` with the explanation, the comparison numbers
(computed now from telemetry.jsonl, which is left out to keep the install
small unless ``--with-telemetry``) and the SHA-256 of every copied file.
Curated runs keep the exploratory label: curation checks reproducibility, not
scientific validity.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import run_metrics

COPY = ("manifest.json", "summary.json", "body.nfbody", "replay_check.json")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")


class CurationError(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _studio_meta(run_dir: Path) -> dict[str, Any]:
    """The studio's record of a queued run (QUEUE/studio/<name>.json), if there is one."""
    path = run_dir.parent.parent / "studio" / f"{run_dir.name}.json"
    return _json(path) if path.is_file() else {}


def check_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file() or _json(summary_path).get("status") != "complete":
        raise CurationError(f"{run_dir}: not a completed run")
    if not (run_dir / "body.nfbody").is_file():
        raise CurationError(f"{run_dir}: no body.nfbody recording to replay")
    receipt_path = run_dir / "replay_check.json"
    if not receipt_path.is_file():
        raise CurationError(
            f"{run_dir}: no replay_check.json. Run `python -m neurofly_body replay-check {run_dir} "
            f"--output <new dir>` and copy its replay_check.json into the run first")
    receipt = _json(receipt_path)
    if receipt.get("verdict") != "BIT_IDENTICAL":
        raise CurationError(f"{run_dir}: replay check verdict is {receipt.get('verdict')!r}, not BIT_IDENTICAL")
    summary = _json(summary_path)
    if receipt.get("original_trajectory_sha256") not in (None, summary.get("trajectory_sha256")):
        raise CurationError(f"{run_dir}: replay_check.json belongs to a different run")
    return receipt


def _copy_one(run_dir: Path, target: Path, *, info: dict[str, Any], with_telemetry: bool) -> None:
    target.mkdir(parents=True)
    files = list(COPY) + (["telemetry.jsonl"] if with_telemetry else [])
    hashes = {}
    for file in files:
        shutil.copyfile(run_dir / file, target / file)
        hashes[file] = hashlib.sha256((target / file).read_bytes()).hexdigest()
    info = dict(info, files=hashes)
    (target / "curated.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")


def curate(run_dir: Path, curated_dir: Path, *, name: str, title: str, explanation: str,
           control_dir: Path | None = None, control_explanation: str | None = None,
           with_telemetry: bool = False) -> list[Path]:
    """Copy one run (and optionally its control) into the curated gallery."""
    if not _NAME.match(name):
        raise CurationError("name: lower-case letters, digits and '-', at most 61 characters")
    if not title.strip() or not explanation.strip():
        raise CurationError("a curated run needs a title and a plain-language explanation")
    runs = [(Path(run_dir), name)] + ([(Path(control_dir), f"{name}-control")] if control_dir else [])
    targets = [Path(curated_dir) / run_name for _, run_name in runs]
    for target in targets:
        if target.exists():
            raise CurationError(f"{target} already exists; curated runs are never overwritten")
    receipts = [check_run(path) for path, _ in runs]   # check everything before copying anything
    now = datetime.now(timezone.utc).isoformat()
    for index, ((path, run_name), receipt) in enumerate(zip(runs, receipts)):
        meta = _studio_meta(path)
        manifest = _json(path / "manifest.json")
        config = manifest.get("config") or {}
        pair = None if len(runs) == 1 else runs[1 - index][1]
        role = meta.get("role") or ("experiment" if index == 0 else "output-disconnected")
        info = {
            "schema": "neurofly-studio-curated-v1",
            "title": title.strip(),
            "explanation": (explanation if index == 0 else (control_explanation or explanation)).strip(),
            "paradigm": meta.get("paradigm", "optomotor"),
            "role": role, "pair": pair,
            "parameters": meta.get("parameters") or {
                "world_angular_velocity_rad_s": config.get("world_angular_velocity_rad_s"),
                "contrast": config.get("contrast"), "duration_s": config.get("duration_s"),
                "seed": config.get("seed")},
            "controller": meta.get("controller") or (manifest.get("invocation") or {}).get("controller"),
            "silence": meta.get("silence") or ((_json(path / "summary.json").get("silenced") or {})
                                               .get("targets") or []),
            "label": "exploratory",
            "metrics": run_metrics(path),
            "replay_check": {"verdict": receipt["verdict"],
                             "brain_backend": receipt.get("brain_backend")},
            "source_run": path.name, "curated_at": now,
        }
        _copy_one(path, Path(curated_dir) / run_name, info=info, with_telemetry=with_telemetry)
    return targets
