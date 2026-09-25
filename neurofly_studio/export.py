"""Run bundles: one finished run as a self-describing, checksummed zip.

    python -m neurofly_studio export RUN_DIR --out run.zip

A bundle holds the run's own files unchanged (manifest.json, summary.json,
telemetry.jsonl, timing.jsonl, body.nfbody, replay_check.json when present),
``telemetry.parquet`` (one row per 2 ms step, nested fields flattened to
dotted column names), ``studio.json`` (the experiment and its exploratory
label, when the run came from the studio), ``bundle.json`` (format, provenance
and the SHA-256 of every file) and a README.  The zip is deterministic: fixed
member order and timestamps, so the same run always gives the same bytes.

Why Parquet and not NWB (the roadmap asked P5 to choose): a run today records
body kinematics, motor commands, per-type DN rates and whole-graph spike counts,
not per-neuron spike trains or measured data.  Parquet opens directly in
pandas, R (arrow) and Julia, and pyarrow is already a dependency.  NWB earns its
weight once runs export per-neuron rasters; revisit then.  The original JSON
lines stay in the bundle, so nothing is lost in the conversion.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

FORMAT = "neurofly-run-bundle"
VERSION = 1
RUN_FILES = ("manifest.json", "summary.json", "telemetry.jsonl", "timing.jsonl", "body.nfbody",
             "replay_check.json", "curated.json")
MAX_LIST = 128          # longer numeric lists stay only in telemetry.jsonl
_EPOCH = (1980, 1, 1, 0, 0, 0)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    elif isinstance(value, bool):
        out[prefix] = value
    elif isinstance(value, (int, float)):
        out[prefix] = float(value) if isinstance(value, float) else value
    elif isinstance(value, list):
        if 0 < len(value) <= MAX_LIST and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                               for v in value):
            for index, item in enumerate(value):
                out[f"{prefix}.{index}"] = item
    elif isinstance(value, str) and prefix == "schema":
        out[prefix] = value
    # other strings and lists of objects (motor events) stay in telemetry.jsonl


def telemetry_table(lines: Iterable[str]):
    """Flatten telemetry records into a pyarrow Table (columns sorted, missing = null)."""
    import pyarrow as pa

    rows = []
    for line in lines:
        row: dict[str, Any] = {}
        _flatten("", json.loads(line), row)
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    front = [c for c in ("step", "run_time_s") if c in columns]
    columns = front + [c for c in columns if c not in front]
    arrays = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        kinds = {type(v) for v in values if v is not None}
        if kinds == {int}:
            arrays[column] = pa.array(values, pa.int64())
        elif kinds <= {int, float}:
            arrays[column] = pa.array([None if v is None else float(v) for v in values], pa.float64())
        elif kinds == {bool}:
            arrays[column] = pa.array(values, pa.bool_())
        else:
            arrays[column] = pa.array([None if v is None else str(v) for v in values], pa.string())
    return pa.table(arrays)


def _parquet_bytes(table) -> bytes:
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    # No created_by clock, fixed row groups: stable bytes for a given table and pyarrow version.
    pq.write_table(table, sink, compression="zstd", row_group_size=65536)
    return sink.getvalue()


def _readme(name: str, bundle: dict[str, Any]) -> str:
    label = bundle.get("label", "exploratory")
    return f"""NeuroFly run bundle: {name}

Label: {label}. Studio runs are exploratory; they are not results of a
preregistered validation run.

Files
  bundle.json         format, provenance and the SHA-256 of every other file
  manifest.json       the run's full manifest (configuration, graph identity, versions)
  summary.json        totals, trajectory SHA-256, silencing record if any
  telemetry.jsonl     one JSON record per 2 ms neural step (the original)
  telemetry.parquet   the same records flattened to columns (dotted names),
                      numeric lists longer than {MAX_LIST} values kept only in the JSON lines
  timing.jsonl        wall-clock timings (not part of the deterministic result)
  body.nfbody         gzip JSON lines: 3D body frames for web/embodied_replay.html
  studio.json         the studio experiment this run belongs to (if any)
  replay_check.json   bit-identical replay receipt (if one was made)

Reading the table in Python:
  import pandas as pd; df = pd.read_parquet("telemetry.parquet")
"""


def build_bundle(run_dir: Path, *, name: str | None = None,
                 studio_meta: dict[str, Any] | None = None) -> bytes:
    """The zip bytes for one finished run directory."""
    run_dir = Path(run_dir)
    name = name or run_dir.name
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if summary.get("status") != "complete":
        raise ValueError(f"{run_dir} did not complete")

    members: dict[str, bytes] = {}
    for file in RUN_FILES:
        if (run_dir / file).is_file():
            members[file] = (run_dir / file).read_bytes()
    if "telemetry.jsonl" in members:
        text = members["telemetry.jsonl"].decode("utf-8")
        members["telemetry.parquet"] = _parquet_bytes(
            telemetry_table(line for line in text.splitlines() if line.strip()))
    if studio_meta is not None:
        members["studio.json"] = (json.dumps(studio_meta, indent=2, sort_keys=True) + "\n").encode()

    label = (studio_meta or {}).get("label") or "exploratory"
    bundle = {
        "format": FORMAT, "version": VERSION, "name": name, "label": label,
        "trajectory_sha256": summary.get("trajectory_sha256"),
        "recording_frames_sha256": (summary.get("recording") or {}).get("frames_sha256"),
        "provenance": {key: manifest.get(key) for key in (
            "schema", "package_version", "created_at", "completed_at", "config", "invocation",
            "neural_backend", "body_backend", "decoder", "determinism", "provenance")
            if key in manifest},
        "files": {file: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for file, data in sorted(members.items())},
        "parquet": {"flattening": "nested keys joined with '.', list items suffixed .0, .1, ...",
                    "max_list_length": MAX_LIST},
    }
    members["bundle.json"] = (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode()
    members["README.txt"] = _readme(name, bundle).encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in ["README.txt", "bundle.json", *sorted(k for k in members if k not in
                                                         ("README.txt", "bundle.json"))]:
            info = zipfile.ZipInfo(f"{name}/{file}", date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_STORED if file.endswith((".nfbody", ".parquet")) \
                else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, members[file])
    return buffer.getvalue()
