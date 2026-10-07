"""Deterministic run recordings (``.nfrec``) for 1x browser replay and open data.

The simulation runs slower than real time at full accuracy, so a run is recorded
once and played back at 1x in the dashboard (``web/replay.js``).  Format and
guarantees: docs/RECORDING_FORMAT.md.

A recording is one gzip stream (``mtime=0``, no file name) of UTF-8 JSON lines:

* line 1, ``{"k": "header", ...}``: format version, provenance (graph hash, code
  version, seed, parameters) and the channel layout (region names and sizes,
  raster neuron list);
* ``{"k": "f", ...}``: one frame per recorded step, the dashboard telemetry packet
  without its wall-clock fields plus ``activity`` (mean rate per region) and
  ``spikes`` (sparse spike counts of the raster neurons);
* ``{"k": "e", ...}``: an input event (an applied command) at its step;
* last line, ``{"k": "end", ...}``: frame count and the SHA-256 of the frame lines.

Original wall-clock and random-ID metadata lives in a
sidecar ``<name>.nfrec.json``, so the same seed, parameters and inputs give a
byte-identical ``.nfrec``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
import threading
import uuid
import sqlite3
import hmac
import secrets
import stat
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from neurofly.privacy import host_description, redact_local

FORMAT = "neurofly-run-recording"
VERSION = 2
READABLE_VERSIONS = (1, 2)
SUFFIX = ".nfrec"
INVALID_SUFFIX = ".INVALID"          # marker: a published file whose finalisation failed
DONE_SUFFIX = ".done"                # completion record: written last, required by readers
COMPLETION_PROTOCOL = 2

# Operational/viewer fields stored as reversible sidecar deltas. Random IDs
# remain in frames as explicit recording-local aliases, preserving relationships.
_DROP_KEYS = ("type", "timestamp", "timing", "path", "sim_speed", "activity",
              # Operational health (wall-clock liveness, save failures): not simulation state.
              "status", "liveness", "persistence", "recording_error")
# Commands that change what the simulation computes; speed, pause and recording
# control only change when steps run, never their content.
NON_PHYSICAL_ACTIONS = frozenset({"set_speed", "set_paused", "save_checkpoint", "probe_brain",
                                  "record_start", "record_stop"})
RASTER_MODES = ("none", "io", "all")

# Same-process logical certification; this lock protects scalar memory only.
_writer_publications = {}
_writer_publications_lock = threading.Lock()


def recording_writer_invalid_reason(path):
    key = str(Path(path).resolve())  # filesystem resolution is outside the memory gate
    with _writer_publications_lock:
        state = _writer_publications.get(key)
    if state is None:
        return None
    return ("Recording writer was revoked; completion is unacknowledged"
            if state[1] == "revoked" else "Recording writer completion is not yet acknowledged")



def _json_safe(value):
    """Non-finite floats become null; numpy scalars and arrays become JSON types."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _line(obj: dict) -> bytes:
    return (json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _sha256_ints(values: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
class RegionMap:
    """A partition of graph neurons into named regions for per-region rates.

    ``index[i]`` is the region of neuron ``i`` or -1.  The real MaleCNS graph is
    grouped by the annotated ``superclass`` (neuprint ROI/neuropil tables are not
    part of the downloaded data); the synthetic test graph by its IO channels.
    """

    def __init__(self, grouping: str, names: List[str], index: np.ndarray, source: str):
        self.grouping = grouping
        self.names = list(names)
        self.index = np.asarray(index, dtype=np.int64)
        self.sizes = np.bincount(self.index[self.index >= 0], minlength=len(self.names)).astype(np.int64)
        self.source = source
        self._valid = self.index >= 0

    def describe(self) -> dict:
        return dict(grouping=self.grouping, names=self.names, sizes=self.sizes.tolist(),
                    source=self.source, membership_sha256=_sha256_ints(self.index),
                    units="Hz, mean per neuron over the graph step that ended at this frame")

    def rates(self, counts: np.ndarray, window_s: float) -> List[float]:
        counts = np.asarray(counts)
        if len(counts) != len(self.index) or window_s <= 0:
            return []
        total = np.bincount(self.index[self._valid], weights=counts[self._valid].astype(np.float64),
                            minlength=len(self.names))
        rates = total / (np.maximum(self.sizes, 1) * window_s)
        return [round(float(r), 3) for r in rates]

    @classmethod
    def from_labels(cls, grouping: str, labels, source: str) -> "RegionMap":
        labels = ["unknown" if (v is None or (isinstance(v, float) and math.isnan(v)) or v == "") else str(v)
                  for v in labels]
        names = sorted(set(labels))
        lookup = {name: i for i, name in enumerate(names)}
        return cls(grouping, names, np.array([lookup[v] for v in labels], dtype=np.int64), source)

    @classmethod
    def from_channels(cls, n: int, channels: Dict[str, Iterable[int]], source: str) -> "RegionMap":
        names = sorted(channels)
        index = np.full(n, -1, dtype=np.int64)
        for i, name in enumerate(names):
            for node in channels[name]:
                if 0 <= int(node) < n and index[int(node)] < 0:
                    index[int(node)] = i
        if np.any(index < 0):
            names.append("other")
            index[index < 0] = len(names) - 1
        return cls("io-channel", names, index, source)


def region_map_for(runner) -> Optional[RegionMap]:
    """Region partition for a graph runner, or None for the modular controller."""
    graph = getattr(runner, "shared_graph", None)
    if graph is None:
        return None
    identity = getattr(graph, "identity", None)
    path = getattr(identity, "neuron_map_path", None)
    if path and not getattr(identity, "synthetic", False):
        try:
            import pyarrow.feather as feather
            nodes = feather.read_table(path, columns=["node_index", "superclass"]).to_pandas()
            nodes = nodes.sort_values("node_index")
            if len(nodes) == graph.n:
                return RegionMap.from_labels("superclass", nodes.superclass.tolist(),
                                             f"neurons.feather superclass ({identity.neuron_map_sha256})")
        except Exception as exc:  # recorded as absent, never faked
            print(f"[recording] superclass regions unavailable: {exc}", flush=True)
    return RegionMap.from_channels(graph.n, getattr(graph, "io_map", {}) or {}, "graph io_map channels")


def raster_neurons(runner, mode: str) -> Tuple[List[int], List[str]]:
    """Neuron indices (and labels) whose spike counts are stored each frame."""
    graph = getattr(runner, "shared_graph", None)
    if graph is None or mode == "none":
        return [], []
    if mode == "all":
        return list(range(graph.n)), []
    groups: Dict[int, str] = {}
    controller = getattr(runner, "graph_controller", None)
    sources = [getattr(graph, "io_map", {}) or {}]
    if controller is not None:
        sources += [controller.dn_indices, controller.sensory_indices, {"epg": controller.epg_indices}]
    for channels in sources:
        for name in sorted(channels):
            for node in channels[name]:
                node = int(node)
                if 0 <= node < graph.n and node not in groups:
                    groups[node] = name
    order = sorted(groups)
    return order, [groups[i] for i in order]


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
class _Ordinals:
    """Maps random IDs (segment UUIDs) to their order of appearance: s0, s1..."""

    def __init__(self, prefix: str):
        self.prefix, self.seen = prefix, {}

    def __call__(self, value):
        if value is None:
            return None
        if value not in self.seen:
            self.seen[value] = f"{self.prefix}{len(self.seen)}"
        return self.seen[value]


def _json_equal(a, b):
    """Content proof includes JSON types: false, zero and null are distinct."""
    if type(a) is not type(b): return False
    if isinstance(a, float): return a.hex() == b.hex()  # preserve signed-zero source bytes
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _require_private(path, mode):
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != mode:
        raise PermissionError("Filesystem did not enforce private recording evidence permissions")


@contextmanager
def _digest_db(path):
    db = sqlite3.connect(path)
    try:
        with db: yield db
    finally:
        db.close()  # sqlite transaction context alone does not close the connection


class _DiskRows:
    """Append-only evidence rows; at most one JSON row is resident while iterating."""
    def __init__(self, path):
        self.path = path
        self.count = 0
        self.digest = hashlib.sha256()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        _require_private(path, 0o600)

    def append(self, row):
        data = _line(row)
        with self.path.open("ab") as handle:
            handle.write(data)
        self.digest.update(data)
        self.count += 1

    def __len__(self):
        return self.count

    def __iter__(self):
        digest = hashlib.sha256()
        count = 0
        with self.path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                count += 1
                yield json.loads(raw)
        if count != self.count or digest.digest() != self.digest.digest():
            raise RuntimeError("Recording provenance spool count/digest mismatch")


class _ProjectionSpool:
    """Private writer scratch, never a public artifact or completion receipt."""
    def __init__(self, path):
        self._status_lock = threading.Lock()
        self.path = path.with_name("." + path.name + ".projection.partial")
        self.path.mkdir(mode=0o700)
        _require_private(self.path, 0o700)
        self.sources = _DiskRows(self.path / "frames.jsonl")
        self.evidence = _DiskRows(self.path / "evidence.jsonl")
        self.private = _DiskRows(self.path / "private.jsonl")
        self._index_key = secrets.token_bytes(32)
        self.index = self.path / "digests.sqlite"
        fd = os.open(self.index, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        _require_private(self.index, 0o600)
        with _digest_db(self.index) as db:
            db.execute("CREATE TABLE evidence (digest TEXT PRIMARY KEY, ref TEXT NOT NULL, auth TEXT NOT NULL)")
        self.status("active")

    def _index_auth(self, digest, ref):
        return hmac.new(self._index_key, _line({"digest": digest, "ref": ref}), hashlib.sha256).hexdigest()

    def get(self, digest):
        with _digest_db(self.index) as db:
            row = db.execute("SELECT ref, auth FROM evidence WHERE digest=?", (digest,)).fetchone()
        if row is None:
            return None
        ref, auth = row
        if not isinstance(ref, str) or not isinstance(auth, str) or not hmac.compare_digest(
                auth, self._index_auth(digest, ref)):
            raise RuntimeError("Recording evidence digest index authentication failed")
        return ref

    def note(self, digest, ref):
        # The row can only select the ref assigned after the evidence append.
        with _digest_db(self.index) as db:
            db.execute("INSERT INTO evidence VALUES (?,?,?)", (digest, ref, self._index_auth(digest, ref)))

    def status(self, state, **fields):
        with self._status_lock:
            data = _line({"state": state, "frames": len(self.sources),
                          "evidence": len(self.evidence), "private": len(self.private), **fields})
            target = self.path / "status.json"
            partial = self.path / ".status.partial"
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, target)
            fd = os.open(self.path, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)

    def sync(self):
        for path in (self.sources.path, self.evidence.path, self.private.path, self.index):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def _stream_json(handle, value):
    """Stream existing final JSON schema without accumulating row arrays/bytes."""
    if isinstance(value, dict):
        handle.write(b"{")
        for i, (key, item) in enumerate(value.items()):
            if i: handle.write(b",")
            handle.write(json.dumps(key).encode() + b":")
            _stream_json(handle, item)
        handle.write(b"}")
    elif isinstance(value, (list, tuple, _DiskRows)):
        handle.write(b"[")
        for i, item in enumerate(value):
            if i: handle.write(b",")
            _stream_json(handle, item)
        handle.write(b"]")
    else:
        handle.write(json.dumps(_json_safe(value), separators=(",", ":"), allow_nan=False).encode())


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _projection_failure(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            spool = getattr(getattr(self, "_projection", None), "spool", None)
            if spool is not None:
                # Revocation is memory-only; failure status/withdrawal run off lock.
                self.abort(f"projection I/O failed: {type(exc).__name__}: {exc}", defer_cleanup=True)
            raise
    return guarded


class _RecordingProjection:
    """Detached recording-local identity projection; source evidence stays separate."""
    NAMESPACE = "recording-local/1"
    ID_FAMILIES = {"run_id": "r", "manifest_run_id": "r", "parent_run_id": "r", "daemon_run_id": "d",
                   "brain_id": "b", "instance_id": "b", "parent_instance_id": "b",
                   "segment_id": "s", "ended_segment": "s", "config_id": "c",
                   "presentation_id": "p", "command_id": "q", "operation_id": "q"}

    def __init__(self, segments, spool=None):
        self.spool = spool
        self._wrapper_cache = []  # bounded detached content snapshots, never a mutable source alias
        self.wrapper_hashes = 0
        self._payload_cache = []
        self.payload_hashes = 0
        self.maps = {"s": segments}
        self.sources = spool.sources if spool is not None else []
        self.source_evidence = spool.evidence if spool is not None else []
        self.private_evidence = spool.private if spool is not None else []
        self._evidence_seen = {}
        self._last_patches = {}
        self._paths = {}

    def alias(self, key, value):
        if not isinstance(value, str) or not value:
            return value
        family = self.ID_FAMILIES[key]
        if key == "config_id" and value.startswith("segment:"):
            return "segment:" + self.alias("segment_id", value[len("segment:"):])
        if key == "presentation_id" and ":" in value:
            segment, suffix = value.rsplit(":", 1)
            if suffix.isdecimal():
                return self.alias("segment_id", segment) + ":" + suffix
        return self.maps.setdefault(family, _Ordinals(family))(value)

    def project(self, telemetry, frame_index):
        # Sidecar contains the redacted original packet, not just a hash; source
        # payloads and receipts remain together and cannot certify projected bytes.
        source = _json_safe(redact_local(telemetry))


        def walk(value, key=None, path=()):
            if key in self.ID_FAMILIES:
                return self.alias(key, value)
            if isinstance(value, list):
                return [walk(v, path=path + (str(i),)) for i, v in enumerate(value)]
            if not isinstance(value, dict):
                return value
            result = {}
            for name in sorted(value):
                item = value[name]
                if (path in (("observation_lifecycle", "durability", "recorder"),
                             ("observation_publication", "durability", "recorder")) and name in
                        ("start_age_s", "last_progress_age_s", "active_write_age_s")):
                    # Only operational recorder ages move to reversible sidecar
                    # deltas; keep durability/status and all simulated times intact.
                    result[name] = None
                elif name == "receipt" and isinstance(item, dict):
                    # A terminal publication wrapper has the original immutable
                    # observation and its receipt/digest together in source evidence.
                    original = telemetry
                    for component in path:
                        original = original[int(component)] if isinstance(original, list) else original[component]
                    original = _json_safe(original)  # only this subtree, not the entire telemetry again
                    ref = next((ref for cached, ref in self._wrapper_cache if _json_equal(cached, original)), None)
                    if ref is None:
                        encoded = _line(original)
                        digest = hashlib.sha256(encoded).hexdigest()
                        self.wrapper_hashes += 1
                        ref = self.spool.get(digest) if self.spool is not None else self._evidence_seen.get(digest)
                        if ref is None:
                            ref = f"e{len(self.source_evidence)}"
                            evidence = value  # already privacy-redacted and JSON-safe by the outer pass
                            redacted = _line(evidence) != encoded
                            self.source_evidence.append({"ref": ref, "source": evidence,
                                "source_payload_redacted": redacted, "verified_for_replay": False,
                                "exact_source_wrapper_sha256": digest if redacted else None})
                            if redacted:
                                self.private_evidence.append({"ref": ref, "source": original})
                            if self.spool is not None: self.spool.note(digest, ref)
                            else: self._evidence_seen[digest] = ref
                        self._wrapper_cache.append((original, ref))
                        self._wrapper_cache = self._wrapper_cache[-8:]
                    result["recording_source_evidence"] = {"ref": ref,
                        "verified_for_replay": False,
                        "reason": "original source publication receipt is sidecar evidence only"}
                elif name == "payload_sha256":
                    # This is a new digest/reference explicitly scoped to the
                    # projection, never a forged source publication receipt.
                    observation = value.get("observation")
                    if isinstance(observation, dict):
                        projected = result.get("observation")
                        if projected is None: projected = walk(observation, path=path + ("observation",))
                        digest = next((digest for cached, digest in self._payload_cache
                                       if _json_equal(cached, projected)), None)
                        if digest is None:
                            encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False, allow_nan=False).encode()
                            digest = hashlib.sha256(encoded).hexdigest()
                            self.payload_hashes += 1
                            self._payload_cache.append((_json_safe(projected), digest))
                            self._payload_cache = self._payload_cache[-8:]
                        result["recording_payload_sha256"] = digest
                    else:
                        result["recording_payload_ref"] = self.maps.setdefault("h", _Ordinals("h"))(item) if isinstance(item, str) else item
                elif name in ("timestamp", "created_at", "updated_at", "last_saved", "wall_time", "wall_time_s") and ("durability" in path or "watchdog_failures" in path or "watchdog_failure" in path or (path == ("brain",) and name == "last_saved") or (len(path) == 3 and path[:2] == ("brain", "history") and name == "timestamp")):
                    # Operational absolute wall-clock leaves are in source packet.
                    # Relative/simulated metric timing is never projected out.
                    result[name] = None
                else:
                    result[name] = walk(item, name, path + (name,))
            return result

        frame = walk({k: v for k, v in source.items() if k not in _DROP_KEYS})
        frame["recording_context"] = {"mode": "replay", "identity_namespace": self.NAMESPACE,
                                      "original_durable_evidence_verified": False}
        patches = []
        def missing(value, path):
            if isinstance(value, dict) and value:
                for key, item in value.items():
                    missing(item, path + (key,))
            else:
                patches.append({"path": list(path), "value": value})
        def differences(original, projected, path=()):
            if isinstance(original, dict) and isinstance(projected, dict):
                for name in original:
                    if name in self.ID_FAMILIES:
                        continue  # reversible crosswalk stores each original ID once
                    if name not in projected:
                        if name == "receipt":
                            ref = projected["recording_source_evidence"]["ref"]
                            patches.append({"path": list(path + (name,)), "source_evidence_ref": ref})
                        else:
                            missing(original[name], path + (name,))
                    else:
                        differences(original[name], projected[name], path + (name,))
            elif isinstance(original, list) and isinstance(projected, list) and len(original) == len(projected):
                for i, (a, b) in enumerate(zip(original, projected)):
                    differences(a, b, path + (str(i),))
            elif original != projected:
                patches.append({"path": list(path), "value": original})
        differences(source, frame)
        current = {tuple(patch["path"]): patch for patch in patches}
        delta = []
        for path, patch in current.items():
            if path not in self._paths:
                self._paths[path] = len(self._paths)
            if path not in self._last_patches or not _json_equal(patch, self._last_patches[path]):
                delta.append({"p": self._paths[path], **{k: v for k, v in patch.items() if k != "path"}})
        for path in self._last_patches.keys() - current.keys():
            delta.append({"p": self._paths[path], "remove_patch": True})
        self._last_patches = current
        self.sources.append({"frame": frame_index, "operational_fields": delta})
        return frame

    def sidecar(self):
        return {"identity_namespace": self.NAMESPACE,
                "identity_mapping": {family: {alias: original for original, alias in mapper.seen.items()}
                                     for family, mapper in self.maps.items()},
                "operational_paths": [list(path) for path in self._paths],
                "source_frames": self.sources, "source_evidence": self.source_evidence}


def frame_from_telemetry(telemetry: Dict[str, Any], segments: _Ordinals,
                         projection=None, frame_index=0) -> Dict[str, Any]:
    """Preserve science with explicit replay identities and detached source evidence."""
    return (projection or _RecordingProjection(segments)).project(telemetry, frame_index)


def source_telemetry_from_frame(frame, projection):
    """Reconstruct the privacy-redacted source packet; never verify durability.

    Completion binds this sidecar to the recording. Redacted originals may need
    the separate private archive for exact original receipt/hash verification.
    """
    import copy
    families = _RecordingProjection.ID_FAMILIES
    mapping = projection["identity_mapping"]
    extra = {"recording_context", "recording_payload_sha256", "recording_payload_ref",
             "recording_source_evidence"}
    def original_id(key, value):
        if not isinstance(value, str):
            return value
        if key == "config_id" and value.startswith("segment:"):
            return "segment:" + original_id("segment_id", value[len("segment:"):])
        if key == "presentation_id" and ":" in value:
            segment, index = value.rsplit(":", 1)
            if index.isdecimal():
                return original_id("segment_id", segment) + ":" + index
        return mapping.get(families[key], {}).get(value, value)
    def restore(value, key=None):
        if key in families:
            return original_id(key, value)
        if isinstance(value, dict):
            return {k: restore(v, k) for k, v in value.items() if k not in extra}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return copy.deepcopy(value)
    result = restore({k: v for k, v in frame.items() if k not in ("k", "i", "spikes", "activity")})
    patches = {}
    for row in projection["source_frames"]:
        if row["frame"] > frame["i"]:
            break
        for patch in row["operational_fields"]:
            if patch.get("remove_patch") is True:
                patches.pop(patch["p"], None)
            else:
                patches[patch["p"]] = patch
    evidence = {row["ref"]: row["source"] for row in projection["source_evidence"]}
    for patch in patches.values():
        path = projection["operational_paths"][patch["p"]]
        target = result
        for component in path[:-1]:
            target = target[int(component)] if isinstance(target, list) else target.setdefault(component, {})
        last = int(path[-1]) if isinstance(target, list) else path[-1]
        value = evidence[patch["source_evidence_ref"]]["receipt"] if "source_evidence_ref" in patch else patch["value"]
        target[last] = copy.deepcopy(value)
    return result


def _brain_backend(instance) -> Optional[str]:
    """Where the graph brain ran ('cpu' or 'cuda'); None for modular runs."""
    return getattr(getattr(instance, "brain", None), "backend", None)


def _lif_dynamics(runner) -> Optional[str]:
    """LIF dynamics version of a graph run (``NEUROFLY_LIF_DYNAMICS``); None for modular."""
    if getattr(runner, "shared_graph", None) is None:
        return None
    from brainlab.graph_identity import active_dynamics_version
    return active_dynamics_version()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class RunRecorder:
    """Writes one ``.nfrec`` while a runner steps.  Call ``capture`` after each step."""

    def __init__(self, runner, path, *, record_every: int = 1, raster: str = "io",
                 label: str = "", inputs: Optional[List[dict]] = None):
        if raster not in RASTER_MODES:
            raise ValueError(f"raster must be one of {RASTER_MODES}")
        self.path = Path(path)
        if self.path.suffix != SUFFIX:
            self.path = self.path.with_name(self.path.name + SUFFIX)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record_every = max(1, int(record_every))
        self.raster_mode = raster
        self.frames = 0
        self.events = 0
        self.command_owner = uuid.uuid4().hex  # receipt only; never changes NF bytes
        self._command_io_lock = threading.Lock()
        self._completion_lock = threading.RLock()
        self._finalization_lock = threading.Lock()
        self._cleanup_state_lock = threading.Lock()
        self._cleanup_started = False
        self._cleanup_thread = None
        self._cleanup_error = None
        self._last_command_receipt = None
        self.closed = False
        self.completed = False
        self._segments = _Ordinals("s")
        self._projection = None
        self._frame_digest = hashlib.sha256()
        self._regions = runner.region_map()
        self._raster, raster_labels = raster_neurons(runner, raster)
        self._raster_index = np.asarray(self._raster, dtype=np.int64)
        self.start_step = int(runner.total_steps)
        self.last_step = self.start_step
        self._partial = self.path.with_name(f".{self.path.name}.partial")
        # An interrupted/canceled writer owns its evidence even after detachment.
        for artifact in (self.path, self._partial, self._sidecar_path, self._done_path):
            if artifact.exists() or artifact.is_symlink():
                raise FileExistsError(f"Recording artifact already exists; choose a new name: {artifact.name}")
        self._publication_key = str(self.path.resolve())
        self._projection = _RecordingProjection(self._segments, _ProjectionSpool(self.path))
        self._raw = open(self._partial, "xb")
        self._gz = None  # gzip construction itself can fail after opening the prefix
        with _writer_publications_lock:
            _writer_publications[self._publication_key] = (self.command_owner, "pending")
        try:
            self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0, compresslevel=6)
            self.header = redact_local(self._header(runner, label, raster_labels, inputs or []))
            self._gz.write(_line(self.header))
            self._sidecar = dict(created_at=time.time(), host=host_description(), pid=os.getpid(),
                                 daemon_run_id=getattr(runner, "run_id", None),
                                 manifest_run_ids=[], label=label)
            self._note_manifest(runner)
            # A replacement writer is not healthy merely because its header or
            # a brain checkpoint succeeded: its actual initial frame must exist.
            captured = self.capture(runner)
            if (captured is not True or self.frames < 1 or self.closed
                    or self.last_step != int(runner.total_steps)):
                raise ValueError('Initial recording capture did not complete')
        except BaseException as exc:
            try:
                self.abort(f'Initial recording setup failed: {type(exc).__name__}: {exc}')
            except Exception as cleanup_exc:
                exc.add_note(f'Recording cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}')
            raise

    # -- header -------------------------------------------------------------
    def _header(self, runner, label: str, raster_labels: List[str], inputs: List[dict]) -> dict:
        manifest = getattr(runner, "manifest", None)
        graph = dict((manifest.graph or {}) if manifest is not None else {})
        for key in ("graph_path", "neuron_map_path", "root"):   # host paths, not identity
            graph.pop(key, None)
        # ``code`` is the origin recorded in the run manifest when the run (or the
        # instance it continues) was created; it is never rewritten.  The code that
        # writes this file is ``writer`` (running_code_identity), hashed now.
        manifest_source = getattr(manifest, "source", None)
        source = dict(manifest_source or getattr(runner, "_source", {}) or {})
        source.pop("root", None)
        code_scope = "run_manifest_origin" if manifest_source else "running_runner_source"
        brain = getattr(runner, "active_brain", None)
        instance = getattr(getattr(runner, "registry", None), "active", None)
        initial_state = dict(brain_steps=int(getattr(brain, "steps", 0) or 0))
        if instance is not None:
            # Graph backends: the neural state is the registry instance, so "restored"
            # means it was loaded from a saved checkpoint (the registry's ``restore``
            # event), not whether the runner's graph bookkeeping file was found.
            initial_state["graph_step_index"] = int(getattr(instance, "step_index", 0) or 0)
            restored_from = getattr(instance, "restored_from", None)
            initial_state["restored"] = restored_from is not None
            initial_state["restore_source"] = (dict(kind="graph_checkpoint", **restored_from)
                                               if restored_from is not None else None)
        else:
            restored = bool(getattr(brain, "restored", False))
            initial_state["restored"] = restored
            initial_state["restore_source"] = dict(kind="modular_brain_file") if restored else None
        provenance = dict(
            backend=runner.backend, assay=runner.active_paradigm_id,
            lif_dynamics=_lif_dynamics(runner),
            brain_backend=_brain_backend(instance),
            seed=int(instance.seed) if instance is not None else int(getattr(brain, "seed", 0)),
            controller_version=getattr(manifest, "controller_version", ""),
            label=getattr(manifest, "label", ""), synthetic=bool(getattr(manifest, "synthetic", False)),
            test_mode=bool(getattr(runner, "test_mode", False)),
            graph=graph or None,
            dynamics=getattr(manifest, "dynamics", {}) if manifest is not None else {},
            code=source,
            code_scope=code_scope,
            writer=_running_code_identity(),
            software=dict(python=platform.python_version(), numpy=np.__version__),
            params=dict(dt_s=runner.dt, graph_step_ms=getattr(runner, "graph_step_ms", None),
                        trial_length_s=runner.trial_length_s, continuous=runner.continuous,
                        motor_assists=dict(getattr(runner.arena, "motor_assists", {}) or {})),
            initial_state=initial_state,
            inputs=list(inputs),
        )
        raster = None
        if self._raster:
            raster = dict(mode=self.raster_mode, neurons=[int(i) for i in self._raster] if self.raster_mode != "all" else None,
                          n=len(self._raster), labels=raster_labels or None,
                          neurons_sha256=_sha256_ints(self._raster),
                          encoding="flat [slot, count, slot, count, ...] of non-zero counts; slot indexes neurons")
        channels = dict(
            regions=self._regions.describe() if self._regions is not None else dict(
                grouping="modular-circuit", source="modular controller telemetry (neural.kc_hz mean, "
                "pam_trace, ppl1_trace, net_valence)", names=list(runner.MODULAR_ACTIVITY), sizes=None,
                units="model units (not spike rates)"),
            raster=raster,
            body=["fly", "body_position_mm", "body_quaternion_wxyz", "joint_angles_rad", "leg_contacts", "biomechanics"],
            stimulus=["stimuli", "sensory", "scene", "assay_state", "live_assay"],
            motor=["dn_rates", "descending", "motor", "motor_drives"],
        )
        return dict(k="header", format=FORMAT, version=VERSION, record_every=self.record_every,
                    start_step=self.start_step, sim_time_start_s=round(self.start_step * runner.dt, 6),
                    frame_dt_s=round(runner.dt * self.record_every, 6),
                    provenance=provenance, channels=channels, label=label,
                    projection=dict(identity_namespace=_RecordingProjection.NAMESPACE,
                                    observations="typed replay projections; source evidence in sidecar",
                                    original_durable_evidence_verified=False))

    def _note_manifest(self, runner):
        manifest = getattr(runner, "manifest", None)
        run_id = getattr(manifest, "run_id", None)
        if run_id and run_id not in self._sidecar["manifest_run_ids"]:
            self._sidecar["manifest_run_ids"].append(run_id)

    # -- frames and events --------------------------------------------------
    @staticmethod
    def _activity(telemetry) -> Optional[List[float]]:
        return (telemetry.get("activity") or {}).get("rates")

    def _spikes(self, runner) -> Optional[List[int]]:
        if not self._raster:
            return None
        counts = getattr(getattr(runner, "graph_controller", None), "last_counts", None)
        if counts is None:
            return None
        sub = np.asarray(counts)[self._raster_index]
        nz = np.flatnonzero(sub)
        out = np.empty(2 * len(nz), dtype=np.int64)
        out[0::2], out[1::2] = nz, sub[nz]
        return out.tolist()

    @_projection_failure
    def capture(self, runner, step_result: Optional[dict] = None) -> bool:
        """Record the state after the step that just ran (every ``record_every`` steps)."""
        if self.closed or runner.total_steps <= self.last_step and self.frames:
            return False
        if (runner.total_steps - self.start_step) % self.record_every:
            return False
        telemetry = runner._assemble_telemetry(step_result if step_result is not None else runner._last_step_result)
        frame = frame_from_telemetry(telemetry, self._segments, self._projection, self.frames)
        frame.update(k="f", i=self.frames, activity=self._activity(telemetry),
                     spikes=self._spikes(runner))
        data = _line(frame)
        self._gz.write(data)
        self._frame_digest.update(data)
        self.frames += 1
        self.last_step = runner.total_steps
        self._note_manifest(runner)
        return True

    def event(self, kind: str, step: int, **fields) -> None:
        if self.closed:
            return
        self._gz.write(_line(dict(k="e", kind=kind, step=int(step), **fields)))
        self.events += 1

    def command(self, runner, cmd: dict, result: dict) -> None:
        """Log an applied command as an input event, unless it could not change the run."""
        action = cmd.get("action", "")
        if action in NON_PHYSICAL_ACTIONS or (result or {}).get("status") != "ok":
            return   # refused commands changed nothing
        clean = {k: v for k, v in cmd.items() if k not in ("token",)}
        self.event("command", runner.total_steps, cmd=clean)

    def prepare_durable_command(self, operation_id: str, step: int, cmd: dict) -> dict:
        """Snapshot under runner ownership; no disk access and no mutable input aliases."""
        if self.closed or getattr(self, "aborted", None):
            raise RuntimeError("Required command recording is closed or aborted")
        clean = {k: v for k, v in cmd.items() if k != "token"}
        serialized = _line(clean)
        return {"recording_owner": self.command_owner, "operation_id": operation_id,
                "step": int(step), "command_sha256": hashlib.sha256(serialized).hexdigest(),
                "command": json.loads(serialized)}

    def durable_command(self, request: dict) -> dict:
        """Off-runner-lock command barrier; receipt follows actual recording-fd fsync."""
        with self._command_io_lock:
            self._check_command_owner(request)
            if hashlib.sha256(_line(request["command"])).hexdigest() != request["command_sha256"]:
                raise RuntimeError("Required command recording request digest changed")
            self._gz.write(_line(dict(k="e", kind="command", step=request["step"],
                                     cmd=request["command"])))
            self.events += 1
            self._gz.flush()
            self._raw.flush()
            os.fsync(self._raw.fileno())
            _fsync_directory(self._partial.parent)
            self._check_command_owner(request)  # cancellation during fsync revokes success
            receipt = {k: request[k] for k in ("recording_owner", "operation_id", "step", "command_sha256")}
            receipt.update(events=self.events, partial_bytes=os.fstat(self._raw.fileno()).st_size)
            self._last_command_receipt = dict(receipt)
            return receipt

    def _check_command_owner(self, request: dict) -> None:
        if (self.closed or getattr(self, "aborted", None) or self._raw.closed
                or request["recording_owner"] != self.command_owner):
            raise RuntimeError("Required command recording owner is no longer healthy")

    def validate_durable_command(self, request: dict, receipt: dict) -> None:
        """No disk/lock wait: called at final runner-owned commit of an applied ACK."""
        types = {"recording_owner": str, "operation_id": str, "step": int,
                 "command_sha256": str, "events": int, "partial_bytes": int}
        # Reject shape/types before equality: bool/float numeric aliases and
        # custom values must not impersonate the receipt or run comparisons.
        if (type(receipt) is not dict or len(receipt) != len(types)
                or any(type(k) is not str or k not in types for k in receipt)
                or any(type(receipt[k]) is not kind for k, kind in types.items())):
            raise RuntimeError("Required command recording durable receipt has invalid types or fields")
        expected = ("recording_owner", "operation_id", "step", "command_sha256")
        if (type(request) is not dict
                or any(k not in request or type(request[k]) is not types[k] for k in expected)):
            raise RuntimeError("Required command recording request has invalid types or fields")
        self._check_command_owner(request)
        if (receipt != self._last_command_receipt or receipt["step"] < 0
                or receipt["events"] <= 0 or receipt["partial_bytes"] <= 0
                or any(receipt[k] != request[k] for k in expected)):
            raise RuntimeError("Required command recording durable receipt does not match")

    def _abort_handles(self) -> None:
        with self._command_io_lock:
            for handle in (self._gz, self._raw):
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception as exc:  # keep the partial file and diagnostic failure
                    self._cleanup_error = f"{type(exc).__name__}: {exc}"

    def _cleanup_revoked_recording(self):
        # Do not race finalizer I/O, and never make a runner-lock caller wait here.
        with self._finalization_lock:
            self._abort_handles()
            try:
                if self._done_path.exists():
                    _replace(self._done_path, self._done_path.with_name(self._done_path.name + ".unconfirmed"))
                    _fsync_directory(self.path.parent)
                if self.path.exists():
                    self._unpublish()
                else:
                    self._withdraw_sidecar()
            except Exception as exc:
                self._cleanup_error = f"completion withdrawal failed: {type(exc).__name__}: {exc}"
                self._mark_invalid(self._cleanup_error)
            try:
                self._projection.spool.status("failed", error=self.aborted)
            except Exception as exc:
                self._cleanup_error = f"failure status unconfirmed: {type(exc).__name__}: {exc}"
            # Retain a revoked scalar tombstone if failed disk cleanup can still
            # expose a final file/receipt; external readers cannot use this memory.
            try:
                hidden = not self.path.exists() and not self._done_path.exists()
            except OSError:
                hidden = False
            if hidden:
                with _writer_publications_lock:
                    if _writer_publications.get(self._publication_key, (None,))[0] == self.command_owner:
                        _writer_publications.pop(self._publication_key, None)

    def abort(self, reason: str, *, defer_cleanup: bool = False) -> None:
        """Revoke immediately in memory; all cleanup is asynchronous, at most once.

        ``defer_cleanup`` marks callers that hold a runner lock or a pending control
        transaction (and projection-failure handlers); it does not change what runs.
        For both values abort() never waits on I/O, closes the handles and withdraws
        any published result on one background thread, and never deletes the hidden
        .partial file or its rows, which stay on disk as evidence.
        """
        with self._completion_lock:
            if self.closed:
                return
            self.closed = True
            self.aborted = reason
            self._last_command_receipt = None
            with _writer_publications_lock:
                _writer_publications[self._publication_key] = (self.command_owner, "revoked")
        with self._cleanup_state_lock:
            if self._cleanup_started:
                return
            self._cleanup_started = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_revoked_recording,
            name="NeuroFly-RecordingCleanup", daemon=True)
        try:
            self._cleanup_thread.start()
        except Exception as exc:
            self._cleanup_error = f"cleanup startup failed: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._cleanup_error) from exc

    def _require_completion_owner(self):
        if hasattr(self, "aborted") or (self.closed and not self.completed):
            raise RuntimeError("Cannot complete an aborted recording")

    @_projection_failure
    def close(self) -> dict:
        """Serialize finalizers while allowing revocation during heavy I/O."""
        with self._finalization_lock:
            return self._close()

    def _close(self) -> dict:
        with self._completion_lock:
            self._require_completion_owner()
            if self.closed:
                return self.summary
        # Order matters: every required step (footer, close, fsync, digest, sidecar)
        # completes on the hidden .partial file BEFORE the atomic rename publishes the
        # recording, so a failure can never leave a file that looks finished.  A failure
        # after the rename un-publishes it (or marks it INVALID); see artifacts().
        end = dict(k="end", frames=self.frames, events=self.events, first_step=self.start_step,
                   last_step=self.last_step, frames_sha256=self._frame_digest.hexdigest())
        self._gz.write(_line(end))
        self._gz.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self._require_completion_owner()
        summary = dict(path=str(self.path), name=self.path.name, bytes=self._partial.stat().st_size,
                       sha256=_file_digest(self._partial), frames=self.frames, events=self.events,
                       first_step=self.start_step, last_step=self.last_step,
                       duration_s=round((self.last_step - self.start_step) * self.header["provenance"]["params"]["dt_s"], 6),
                       assay=self.header["provenance"]["assay"],
                       backend=self.header["provenance"]["backend"],
                       synthetic=self.header["provenance"]["synthetic"], status="complete")
        self._projection.spool.sync()
        if len(self._projection.private_evidence):
            private_dir = self.path.parent / ".source-evidence-private"
            if private_dir.is_symlink():
                raise RuntimeError("Private source archive directory must not be a symlink")
            private_dir.mkdir(mode=0o700, exist_ok=True)
            os.chmod(private_dir, 0o700)
            _require_private(private_dir, 0o700)
            archive = private_dir / (self.path.name + ".source.json")
            fd = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                _require_private(archive, 0o600)
                _stream_json(handle, {"recording_sha256": summary["sha256"],
                                      "source_evidence": self._projection.private_evidence})
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(private_dir)
        sidecar = dict(self._sidecar, closed_at=time.time(), recording=summary,
                       completion_protocol=COMPLETION_PROTOCOL,
                       projection=self._projection.sidecar())
        sidecar_tmp = self._sidecar_path.with_name(f".{self._sidecar_path.name}.partial")
        with sidecar_tmp.open("wb") as handle:
            _stream_json(handle, redact_local(sidecar))  # rows already redacted; metadata remains redacted
            handle.write(b"\n")
            handle.flush()
            _fsync_file(handle)
        sidecar_sha = _file_digest(sidecar_tmp)
        self._require_completion_owner()
        _replace(sidecar_tmp, self._sidecar_path)
        self._require_completion_owner()
        try:
            _replace(self._partial, self.path)
        except OSError:
            self._withdraw_sidecar()
            raise
        try:
            # Both renames must be durable BEFORE completion is certified.
            _fsync_directory(self.path.parent)
        except OSError:
            self._unpublish()     # best effort only: without the completion record below,
            raise                 # readers and listings already refuse this recording
        # Completion record, written LAST.  Readers and listings require it and check
        # it against the file and the sidecar, so any failure up to and including its
        # atomic rename leaves an uncertified recording -- no cleanup or later write in
        # this directory is needed for that.
        self._require_completion_owner()
        self._projection.spool.status("complete", recording_sha256=summary["sha256"],
                                      sidecar_sha256=sidecar_sha)
        self._require_completion_owner()
        record = {"name": self.path.name, "sha256": summary["sha256"],
                  "sidecar_sha256": sidecar_sha,
                  "protocol": COMPLETION_PROTOCOL, "completed_at": time.time()}
        self._require_completion_owner()
        done_tmp = self._done_path.with_name(f".{self._done_path.name}.partial")
        _write_durably(done_tmp, (json.dumps(record) + "\n").encode())
        self._require_completion_owner()
        _replace(done_tmp, self._done_path)
        self._require_completion_owner()
        try:
            _fsync_directory(self.path.parent)
        except OSError:
            try:
                os.replace(self._done_path, self._done_path.with_name(self._done_path.name + ".unconfirmed"))
            except OSError:
                pass
            raise
        # Only the logical owner ACK is atomic with abort, never a filesystem call.
        with self._completion_lock:
            self._require_completion_owner()
            self.closed = self.completed = True
            self.summary = summary
            with _writer_publications_lock:
                if _writer_publications.get(self._publication_key, (None,))[0] == self.command_owner:
                    _writer_publications.pop(self._publication_key, None)
        return self.summary

    @property
    def _done_path(self) -> Path:
        return self.path.with_name(self.path.name + DONE_SUFFIX)

    @property
    def _sidecar_path(self) -> Path:
        return self.path.with_name(self.path.name + ".json")

    @property
    def _invalid_marker(self) -> Path:
        return self.path.with_name(self.path.name + INVALID_SUFFIX)

    def _withdraw_sidecar(self) -> None:
        """The recording was not published: its sidecar must not claim it was."""
        try:
            if self._sidecar_path.exists():
                os.replace(self._sidecar_path, self._sidecar_path.with_name(self._sidecar_path.name + ".withdrawn"))
        except OSError:
            self._mark_invalid("sidecar could not be withdrawn")

    def _unpublish(self) -> None:
        """A failure after the rename: move the file back to .partial, else mark it INVALID."""
        try:
            os.replace(self.path, self._partial)
            self._withdraw_sidecar()
        except OSError:
            self._mark_invalid("finalisation failed after the file was published")

    def _mark_invalid(self, reason: str) -> None:
        try:
            self._invalid_marker.write_text(json.dumps({"invalid": True, "reason": reason,
                                                        "at": time.time()}) + "\n")
        except OSError:
            pass

    def artifacts(self) -> dict:
        """Where this recording's files actually are, and what state they are in."""
        def state(p):
            return p.name if p.exists() else None
        return {"partial": state(self._partial), "final": state(self.path), "sidecar": state(self._sidecar_path),
                "completion_record": state(self._done_path), "invalid_marker": state(self._invalid_marker),
                # What a reader concludes from the disk (not what this object hoped):
                "certified": self.path.exists() and recording_invalid_reason(self.path) is None,
                "completed": bool(self.completed and self.path.exists()
                                  and recording_invalid_reason(self.path) is None)}


def _replace(src, dst) -> None:
    os.replace(src, dst)


def _fsync_file(handle) -> None:
    os.fsync(handle.fileno())


def _write_durably(path, data: bytes) -> None:
    """Write, flush and fsync a file's contents (before it is renamed into place)."""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        _fsync_file(handle)


def _fsync_directory(directory) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
def recording_invalid_reason(path) -> Optional[str]:
    """Why a published-looking recording must not be treated as finished, else None:
    an INVALID marker next to it, or a sidecar that says it was not completed."""
    path = Path(path)
    owner_check = globals().get("recording_writer_invalid_reason")
    if owner_check is not None:
        reason = owner_check(path)
        if reason is not None:
            return reason
    marker = path.with_name(path.name + INVALID_SUFFIX)
    if marker.exists():
        return f"{path.name} is marked INVALID ({marker.name})"
    try:
        with gzip.open(path, "rb") as handle:
            header = json.loads(handle.readline())
    except (OSError, EOFError, ValueError):
        return f"{path.name} has an unreadable recording header"
    if (not isinstance(header, dict) or header.get("k") != "header"
            or header.get("format") != FORMAT or type(header.get("version")) is not int
            or header["version"] not in READABLE_VERSIONS):
        return f"{path.name} has an unsupported recording header"
    version = header["version"]
    sidecar = path.with_name(path.name + ".json")
    if not sidecar.is_file():
        return f"{path.name} has no sidecar: its finalisation did not complete"
    try:
        sidecar_bytes = sidecar.read_bytes()
        meta = json.loads(sidecar_bytes)
        status = (meta.get("recording") or {}).get("status", "complete")
    except (OSError, ValueError, AttributeError):
        return f"{sidecar.name} is unreadable"
    if status != "complete":
        return f"{path.name} status is {status!r}"
    protocol = meta.get("completion_protocol")
    if version == 1 and protocol is None and "projection" not in meta:
        return None       # Genuine historical v1 before the completion protocol.
    if type(protocol) is not int or protocol != COMPLETION_PROTOCOL:
        return f"{sidecar.name} has an unsupported completion protocol"
    # Current protocol: certified only by a completion record that matches both files.
    done = path.with_name(path.name + DONE_SUFFIX)
    try:
        record = json.loads(done.read_text())
    except FileNotFoundError:
        return f"{path.name} has no completion record: its completion was never confirmed"
    except (OSError, ValueError):
        return f"{done.name} is unreadable"
    try:
        file_sha = _file_digest(path)
    except OSError:
        return f"{path.name} is unreadable"
    if not isinstance(record, dict) or type(record.get("protocol")) is not int \
            or record.get("protocol") != COMPLETION_PROTOCOL or record.get("name") != path.name \
            or record.get("sha256") != file_sha or record.get("sidecar_sha256") != hashlib.sha256(sidecar_bytes).hexdigest() \
            or (meta.get("recording") or {}).get("sha256") != file_sha:
        return f"{done.name} does not match {path.name} and its sidecar"
    return None


def read_recording(path, require_finished: bool = True) -> dict:
    """Parse a recording and check its frame digest.  Returns header, frames, events, end.

    A recording whose finalisation did not complete (INVALID marker, missing or
    non-complete sidecar) is refused unless ``require_finished=False``.
    """
    if require_finished:
        reason = recording_invalid_reason(path)
        if reason is not None:
            raise ValueError(f"Refusing an unfinished recording: {reason}")
    header, frames, events, end = None, [], [], None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for raw in fh:
            record = json.loads(raw)
            kind = record.get("k")
            if kind == "header":
                header = record
            elif kind == "f":
                digest.update(raw)
                frames.append(record)
            elif kind == "e":
                events.append(record)
            elif kind == "end":
                end = record
    if header is None or header.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    if header.get("version") not in READABLE_VERSIONS:
        raise ValueError(f"Unsupported recording version {header.get('version')!r}")
    if end is None:
        raise ValueError(f"{path} is incomplete (no end record)")
    if end["frames_sha256"] != digest.hexdigest() or end["frames"] != len(frames):
        raise ValueError(f"{path} frame digest mismatch")
    return dict(header=header, frames=frames, events=events, end=end)


def public_recording_artifacts(path):
    """Explicit distributable recording bundle; private source proof never exports."""
    path = Path(path)
    reason = recording_invalid_reason(path)
    if reason is not None:
        raise ValueError(f"Refusing an unfinished recording: {reason}")
    return [p for p in (path, path.with_name(path.name + ".json"),
                        path.with_name(path.name + DONE_SUFFIX)) if p.is_file()]


def list_recordings(directory) -> List[dict]:
    """Finished recordings in ``directory`` (newest first) with their sidecar summaries."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out = []
    for path in directory.glob(f"*{SUFFIX}"):
        if recording_invalid_reason(path) is not None:
            continue            # not finished: never listed (or served) as a completed recording
        sidecar = path.with_name(path.name + ".json")
        try:
            meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        summary = dict(meta.get("recording") or {}, name=path.name, bytes=path.stat().st_size)
        summary.pop("path", None)
        summary["created_at"] = meta.get("created_at", path.stat().st_mtime)
        out.append(summary)
    return sorted(out, key=lambda s: s["created_at"], reverse=True)


# ---------------------------------------------------------------------------
# Batch recording: run a paradigm headless (as fast as it computes) and record it
# ---------------------------------------------------------------------------
def _running_code_identity() -> dict:
    from provenance import running_code_identity   # top-level module of the same distribution
    return running_code_identity()


def _shutdown_state_dir(runner) -> dict:
    """Run the runner's own shutdown transaction (final save, then session_end).

    The lifecycle transaction needs a durable observation recorder; the standalone
    one the scheduled-input API uses is attached here, after the recording is
    closed, so it cannot change recorded frames.
    """
    error = None
    try:
        runner._ensure_scheduled_records()
        clean = bool(runner.stop())
    except Exception as exc:  # noqa: BLE001 -- reported; no clean marker was written
        clean, error = False, f"{type(exc).__name__}: {exc}"
    active = getattr(getattr(runner, "registry", None), "active", None)
    step = int(getattr(active, "step_index", 0) or 0) if active is not None else None
    # Only a clean shutdown saved the final state; after a failure the current
    # checkpoint is the older last good one, whose step this runner does not hold.
    final = (dict(checkpoint_version=int(getattr(active, "checkpoint_version", 0) or 0), step_index=step)
             if clean and active is not None else None)
    return dict(clean=clean, steps_run=int(runner.total_steps), graph_step_index=step, graph_checkpoint=final,
                error=error or (None if clean else runner.last_error or "shutdown was refused or failed"))


def record_run(*, paradigm: str, out, steps: int, backend: str = "modular", record_every: int = 1,
               raster: str = "io", schedule: Optional[List[dict]] = None, state_dir=None,
               test_synthetic_graph: bool = False, graph_dir=None, graph_step_ms: Optional[float] = None,
               trial_length_s: float = 60.0, continuous: bool = False, label: str = "",
               progress_every: int = 0, dynamics: Optional[str] = None) -> dict:
    """Run ``steps`` fixed-dt steps of a fresh runner and write one recording.

    ``state_dir`` defaults to a new temporary directory, so the run starts from the
    paradigm's naive brain; pass a directory to continue a saved brain (the header
    then records ``initial_state.restored``; a normal finish saves the final state and
    writes the clean shutdown marker, see docs/RECORDING_FORMAT.md).  ``schedule`` is a list of
    ``{"step": n, "cmd": {...}}`` applied exactly at step ``n`` (the recorded inputs).
    ``dynamics`` sets the process-wide LIF dynamics (``NEUROFLY_LIF_DYNAMICS``) for
    graph backends, as the daemon's ``--dynamics`` does; None keeps the environment.
    """
    import tempfile
    if dynamics is not None:
        # Process-wide, before any Brain or registry manifest is created (as the daemon).
        os.environ["NEUROFLY_LIF_DYNAMICS"] = dynamics
    from neurofly_daemon import ContinuousExperimentRunner

    schedule = sorted((dict(step=int(e["step"]), cmd=dict(e["cmd"])) for e in (schedule or [])),
                      key=lambda e: e["step"])
    with tempfile.TemporaryDirectory(prefix="nfrec-state-") as tmp:
        runner = ContinuousExperimentRunner(
            initial_paradigm=paradigm, sim_speed=1.0, checkpoint_interval=1e9,
            output_dir=Path(state_dir) if state_dir else Path(tmp), trial_length_s=trial_length_s,
            continuous=continuous, backend=backend, graph_dir=Path(graph_dir) if graph_dir else None,
            test_synthetic_graph=test_synthetic_graph, graph_step_ms=graph_step_ms)
        for entry in schedule:
            runner.schedule_command(entry["step"], entry["cmd"])
        started = time.perf_counter()
        try:
            with runner.lock:
                runner.recorder = RunRecorder(runner, out, record_every=record_every, raster=raster,
                                              label=label, inputs=schedule)
            while runner.total_steps < int(steps):
                with runner.lock:
                    before = runner.total_steps
                    runner._advance_one()
                    stopped = runner.recorder is None or bool(runner.last_error)
                    if runner.last_error and runner.recorder is not None:
                        runner.recorder.event("simulation_error", runner.total_steps, message=runner.last_error)
                if stopped:
                    break
                if runner.total_steps == before:
                    # The central control transaction publishes its final ACK off lock.
                    # Do not count a persistence wait as a simulation tick.
                    runner._wake.wait(0.01)
                    runner._wake.clear()
                if progress_every and runner.total_steps != before and runner.total_steps % progress_every == 0:
                    wall = time.perf_counter() - started
                    print(f"[record] step {runner.total_steps}/{steps}  sim {runner.total_steps * runner.dt:.2f} s  "
                          f"{runner.total_steps * runner.dt / wall:.3f}x real time", flush=True)
            with runner.lock:
                summary = runner.stop_recording() if runner.recorder is not None else None
            # A continued state directory ends like a daemon: the shutdown transaction
            # saves the final state and only then writes the clean session_end marker.
            # Without it the next run would restart from an older checkpoint and
            # (rightly) report the unsaved steps as an unclean shutdown.  A failed or
            # refused shutdown (halted controller, failed save) writes no marker, so
            # it stays flagged; an exception above skips this entirely (crash path).
            shutdown = _shutdown_state_dir(runner) if state_dir else None
        finally:
            runner._close_scheduled_records()
        if summary is None:
            # Structured INVALID result: the file was not finished (its .partial is kept).
            failure = runner.recording_error or {}
            summary = {"path": None, "name": failure.get("recording"), "frames": failure.get("frames"),
                       "partial_kept": True, "recording_error": failure}
        validity = runner.result_validity() if hasattr(runner, "result_validity") else {}
        summary["wall_s"] = round(time.perf_counter() - started, 3)
        summary["brain_backend"] = _brain_backend(getattr(getattr(runner, "registry", None), "active", None))
        summary["error"] = runner.last_error or (summary.get("recording_error") or {}).get("error")
        summary["steps_requested"] = int(steps)
        summary["result_validity"] = validity
        summary["shutdown"] = shutdown
        summary["status"] = ("invalid" if summary.get("recording_error") else
                             "incomplete" if (runner.last_error or validity.get("state") == "incomplete"
                                              or (shutdown is not None and not shutdown["clean"]))
                             else "complete")
        return summary


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="neurofly record",
                                     description="Run a paradigm headless and write a deterministic .nfrec "
                                                 "recording for 1x replay in the dashboard.")
    parser.add_argument("--paradigm", required=True)
    parser.add_argument("--out", required=True, help="Output path (.nfrec is appended when missing)")
    length = parser.add_mutually_exclusive_group(required=True)
    length.add_argument("--steps", type=int)
    length.add_argument("--seconds", type=float, help="Simulated seconds (steps = seconds / dt)")
    parser.add_argument("--backend", default="modular")
    parser.add_argument("--test-synthetic-graph", action="store_true",
                        help="Use the labelled synthetic test graph (graph backends, tests only)")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--graph-step-ms", type=float, default=None)
    parser.add_argument("--dynamics", choices=("v1", "v2", "v3"),
                        default=os.environ.get("NEUROFLY_LIF_DYNAMICS") or "v3",
                        help="LIF dynamics of the connectome backends (default: v3, as the daemon; "
                             "env NEUROFLY_LIF_DYNAMICS)")
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--raster", choices=RASTER_MODES, default="io")
    parser.add_argument("--schedule", default=None,
                        help='JSON file: [{"step": n, "cmd": {"action": ...}}, ...] applied at exact steps')
    parser.add_argument("--state-dir", default=None,
                        help="Brain state directory to continue (default: fresh, naive brain)")
    parser.add_argument("--trial-seconds", type=float, default=60.0)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    steps = args.steps if args.steps is not None else int(round(args.seconds / 0.02))
    schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else None
    summary = record_run(paradigm=args.paradigm, out=args.out, steps=steps, backend=args.backend,
                         record_every=args.record_every, raster=args.raster, schedule=schedule,
                         state_dir=args.state_dir, test_synthetic_graph=args.test_synthetic_graph,
                         graph_dir=args.graph_dir, graph_step_ms=args.graph_step_ms,
                         trial_length_s=args.trial_seconds, continuous=args.continuous, label=args.label,
                         progress_every=max(1, steps // 10), dynamics=args.dynamics)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
