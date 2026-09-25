"""Embodied run recordings (``body.nfbody``) for 1x replay in the browser.

The embodied loop runs slower than real time at full fidelity, so a run is
recorded while it computes and played back at the fly's own speed by
``web/embodied_replay.html``.  One gzip stream (``mtime=0``, no file name) of
UTF-8 JSON lines:

* ``{"k": "header", ...}``: format and version, frame rate, the body skeleton
  (segment names and parent indices), and provenance (config, invocation,
  graph identity, decoder declaration);
* ``{"k": "f", ...}``: one frame per ``1/fps`` s of simulated time: segment
  positions (mm, rounded to 0.1 um), thorax yaw, the decoded CPG command, the
  per-side DN rates when the controller has them, ground contacts per leg,
  graph spikes during the frame window and any motor events;
* ``{"k": "end", ...}``: frame count and the SHA-256 of the frame lines.

Nothing in the file depends on the wall clock, so the same seed and
arguments give a byte-identical recording.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FORMAT = "neurofly-embodied-recording"
VERSION = 1
SUFFIX = ".nfbody"
POSITION_DECIMALS = 4   # mm -> 0.1 um


def _line(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def frame_interval_steps(fps: float, neural_dt_ms: float) -> int:
    """Neural steps per recorded frame; the frame period must be a whole number of steps."""
    if not (math.isfinite(fps) and fps > 0):
        raise ValueError("record fps must be positive")
    steps = 1000.0 / fps / neural_dt_ms
    if steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-9):
        raise ValueError(f"1/fps = {1000.0 / fps} ms is not a whole number of {neural_dt_ms} ms neural steps")
    return int(round(steps))


class BodyRecorder:
    """Writes ``body.nfbody`` during ``run_embodied``."""

    def __init__(self, path: Path, *, fps: float, neural_dt_ms: float) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.every = frame_interval_steps(fps, neural_dt_ms)
        self.neural_dt_ms = float(neural_dt_ms)
        raw = self.path.open("xb")
        self._raw = raw
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        self._digest = hashlib.sha256()
        self.frames = 0
        self._spikes = 0
        self._events: list[dict] = []

    def header(self, *, skeleton: dict, provenance: dict) -> None:
        self._gzip.write(_line({
            "k": "header", "format": FORMAT, "version": VERSION, "fps": self.fps,
            "frame_interval_steps": self.every, "neural_dt_ms": self.neural_dt_ms,
            "skeleton": skeleton, "provenance": provenance,
            "units": {"t": "s simulated", "pos": "mm world frame, z up", "yaw": "rad, + counter-clockwise",
                      "dn": "Hz per neuron", "cmd": "FlyGym CPG command [left, right]"},
        }))

    def step(self, step_index: int, record: dict, positions, total_spikes: int) -> None:
        """Called after every neural step; writes a frame every ``every`` steps."""
        self._spikes += int(total_spikes)
        self._events.extend(record.get("motor", {}).get("decoder", {}).get("events", []) or [])
        if step_index % self.every:
            return
        neural = record["neural"]
        dn = neural.get("locomotion_dn")
        frame = {
            "k": "f",
            "t": round(step_index * self.neural_dt_ms / 1000.0, 9),
            "pos": [round(float(v), POSITION_DECIMALS) for row in positions for v in row],
            "yaw": round(float(record["body"]["thorax"]["yaw_rad"]), 6),
            "cmd": [round(float(v), 6) for v in record["motor"]["applied_cpg_drive"]],
            "contacts": [int(v) for v in record["body"].get("contacts", {}).get("found", [])],
            "spikes": self._spikes,
        }
        if isinstance(dn, dict):
            frame["dn"] = {k[:-len("_rate_hz")]: round(float(v), 3)
                           for k, v in sorted(dn.items()) if k.endswith("_rate_hz")}
        if self._events:
            frame["events"] = self._events
        line = _line(frame)
        self._digest.update(line)
        self._gzip.write(line)
        self.frames += 1
        self._spikes = 0
        self._events = []

    def close(self) -> dict:
        self._gzip.write(_line({"k": "end", "frames": self.frames, "frames_sha256": self._digest.hexdigest()}))
        self._gzip.close()
        self._raw.close()
        return {"path": self.path.name, "frames": self.frames, "fps": self.fps,
                "frames_sha256": self._digest.hexdigest()}

    def abort(self) -> None:
        try:
            self._gzip.close()
        finally:
            self._raw.close()


def read_body_recording(path: Path) -> dict[str, Any]:
    """Parse and verify a recording: returns header, frames and end; raises on corruption."""
    header, frames, end = None, [], None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            record = json.loads(raw)
            kind = record.get("k")
            if kind == "header":
                header = record
            elif kind == "f":
                digest.update(raw)
                frames.append(record)
            elif kind == "end":
                end = record
    if header is None or header.get("format") != FORMAT or end is None:
        raise ValueError(f"{path} is not a complete {FORMAT} file")
    if end["frames"] != len(frames) or end["frames_sha256"] != digest.hexdigest():
        raise ValueError(f"{path} frame count or SHA-256 does not match its end record")
    return {"header": header, "frames": frames, "end": end}
