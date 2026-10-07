"""Durable cohort storage: exclusive directory lock, atomic writes, checkpoints.

Layout of a cohort directory ``DIR``::

    DIR/.cohort.lock                    exclusive writer lock (flock)
    DIR/cohort_manifest.json            schema neurofly.cohort.v1, identities, checkpoint chain
    DIR/ckpt/fly-NN-tTTTTTT.npz         one checkpoint per fly per checkpoint tick
    DIR/flies/fly-NN/manifest.json      per-fly identity
    DIR/flies/fly-NN/steps.jsonl        per-step record (population spikes, motor, world)

Every file is written atomically: temporary file in the same directory, fsync,
rename, fsync of the directory.  Checkpoint files are deterministic zip archives
(fixed member timestamps), so identical state gives identical bytes.

A checkpoint is self-describing: its ``meta`` member (JSON) carries the schema,
graph/io/dynamics/engine identity, fly id, seed, RNG state, input cursor,
stimulus-schedule hash, assay and arena world state, tick and sim_ms, the
parent checkpoint's sha256 and a ``content_sha256`` over everything else.
A checkpoint that fails any check is refused and left exactly as found.

v0.4 experiment registries and saved brains are never read or migrated: a
path that looks like one is refused before anything is opened.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .api import COHORT_SCHEMA

MANIFEST_NAME = 'cohort_manifest.json'
LOCK_NAME = '.cohort.lock'
STATE_SCALARS = ('cursor', 'total_spikes', 'sim_ms', 'dynamics')
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class CohortStoreError(RuntimeError):
    """A cohort directory or checkpoint was refused.  Nothing was modified."""


class CohortLocked(CohortStoreError):
    pass


# ---------------------------------------------------------------------------
# Atomic, durable writes
# ---------------------------------------------------------------------------
def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """Write ``data`` to ``path`` via tmp + fsync + rename + fsync(dir); return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.replace(tmp, path)
    fsync_dir(path.parent)
    return hashlib.sha256(data).hexdigest()


def atomic_write_json(path: Path, value) -> str:
    return atomic_write_bytes(path, (json.dumps(value, indent=1, sort_keys=True) + '\n').encode())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Exclusive writer lock
# ---------------------------------------------------------------------------
class CohortLock:
    """Exclusive, non-blocking flock on ``DIR/.cohort.lock``.  A second writer
    (another process or another open in this one) is refused, never queued."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / LOCK_NAME
        self.fd: Optional[int] = None

    def acquire(self) -> 'CohortLock':
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                raise CohortLocked(f'{self.root} is locked by another cohort writer ({self.path}); refused') from exc
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f'{os.getpid()}\n'.encode())
        os.fsync(fd)
        self.fd = fd
        return self

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# v0.4 store refusal (existence checks only; nothing is opened)
# ---------------------------------------------------------------------------
_REGISTRY_NAME = re.compile(r'^registry(-v\d+)?$')


def refuse_legacy_path(path: Path) -> None:
    """Refuse a v0.4 experiment registry, daemon output directory or saved brain.

    They are never read, migrated or written by the cohort tools.
    """
    path = Path(path).expanduser()
    why = None
    if path.is_file() or path.suffix in ('.npz', '.json', '.pkl', '.pt'):
        why = 'it is a file (a saved brain or registry file), not a cohort directory'
    elif _REGISTRY_NAME.match(path.name):
        why = 'it is a v0.4 experiment registry directory'
    else:
        for parent in [path, *path.parents]:
            if (parent / 'registry.json').exists():
                why = f'{parent} holds a v0.4 experiment registry (registry.json)'
                break
            if _REGISTRY_NAME.match(parent.name):
                why = f'{parent} is a v0.4 experiment registry directory'
                break
        if why is None and path.is_dir():
            markers = [m for m in ('registry', 'registry-v2', 'registry-v3', 'brains', 'graph-bookkeeping')
                       if (path / m).exists()]
            if markers:
                why = f'it holds v0.4 saved stores ({", ".join(markers)})'
    if why:
        raise CohortStoreError(
            f'Refused: {path} is not a cohort directory; {why}. v0.4 registries and saved brains '
            'are never read, imported or migrated by "neurofly cohort". Use a new empty --out directory.')


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def checkpoint_name(fly_id: int, tick: int) -> str:
    return f'fly-{fly_id:02d}-t{tick:06d}.npz'


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, np.ascontiguousarray(array), allow_pickle=False)
    return buf.getvalue()


def _content_digest(meta: dict, arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    body = {k: v for k, v in meta.items() if k != 'content_sha256'}
    h.update(json.dumps(body, sort_keys=True, separators=(',', ':')).encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode() + b'\0' + str(a.dtype.str).encode() + b'\0' + repr(a.shape).encode() + b'\0')
        h.update(a.tobytes())
    return h.hexdigest()


def encode_checkpoint(meta: dict, state: Dict[str, object]) -> bytes:
    """Deterministic zip: ``meta.json`` plus one ``state/<name>.npy`` per array.

    ``state`` is exactly ``engine.read_state(fly)``; its scalar fields go into
    ``meta['state_scalars']`` unchanged.
    """
    arrays = {k: np.asarray(v) for k, v in state.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
    meta = dict(meta, schema=COHORT_SCHEMA, state_scalars=scalars,
                state_arrays={k: [a.dtype.str, list(a.shape)] for k, a in sorted(arrays.items())})
    meta['content_sha256'] = _content_digest(meta, arrays)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        def put(name, data):
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
        put('meta.json', json.dumps(meta, sort_keys=True, indent=1).encode())
        for name in sorted(arrays):
            put(f'state/{name}.npy', _npy_bytes(arrays[name]))
    return buf.getvalue()


def decode_checkpoint(data: bytes, where: str = 'checkpoint') -> Tuple[dict, Dict[str, object]]:
    """Parse and self-verify a checkpoint.  Raises CohortStoreError; never repairs."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise CohortStoreError(f'{where}: member {bad} fails its CRC (corrupt)')
            meta = json.loads(zf.read('meta.json'))
            arrays = {}
            for name in zf.namelist():
                if name.startswith('state/') and name.endswith('.npy'):
                    arrays[name[6:-4]] = np.lib.format.read_array(io.BytesIO(zf.read(name)), allow_pickle=False)
    except CohortStoreError:
        raise
    except Exception as exc:   # truncated / not a zip / bad JSON / bad npy
        raise CohortStoreError(f'{where}: unreadable (corrupt or truncated): {type(exc).__name__}: {exc}') from exc
    if meta.get('schema') != COHORT_SCHEMA:
        raise CohortStoreError(f'{where}: schema {meta.get("schema")!r} is not {COHORT_SCHEMA}; refused')
    declared = meta.get('state_arrays') or {}
    if sorted(declared) != sorted(arrays):
        raise CohortStoreError(f'{where}: state arrays {sorted(arrays)} differ from the declared {sorted(declared)}')
    for name, (dtype, shape) in declared.items():
        if arrays[name].dtype.str != dtype or list(arrays[name].shape) != list(shape):
            raise CohortStoreError(f'{where}: array {name} is {arrays[name].dtype}/{arrays[name].shape}, '
                                   f'declared {dtype}/{shape}')
    if _content_digest(meta, arrays) != meta.get('content_sha256'):
        raise CohortStoreError(f'{where}: content sha256 mismatch (corrupt or edited); refused')
    state: Dict[str, object] = dict(arrays)
    state.update(meta.get('state_scalars') or {})
    if state.get('dynamics') != 'v3':
        raise CohortStoreError(f'{where}: dynamics {state.get("dynamics")!r} is not fixed v3; refused')
    return meta, state


def read_checkpoint(path: Path, expected_sha256: Optional[str] = None) -> Tuple[dict, Dict[str, object], str]:
    path = Path(path)
    if not path.is_file():
        raise CohortStoreError(f'{path}: checkpoint missing')
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CohortStoreError(f'{path}: file sha256 {digest} differs from the manifest chain '
                               f'({expected_sha256}); corrupt or replaced, refused')
    meta, state = decode_checkpoint(data, str(path))
    return meta, state, digest


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_NAME


def read_manifest(root: Path) -> dict:
    path = manifest_path(root)
    if not path.is_file():
        raise CohortStoreError(f'{root} has no {MANIFEST_NAME}; not a cohort directory')
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        raise CohortStoreError(f'{path}: unreadable ({type(exc).__name__}: {exc}); refused') from exc
    if manifest.get('schema') != COHORT_SCHEMA:
        raise CohortStoreError(f'{path}: schema {manifest.get("schema")!r} is not {COHORT_SCHEMA}; refused')
    return manifest


def verify_chain(root: Path, fly_entry: dict) -> Tuple[dict, Dict[str, object], str]:
    """Verify every checkpoint of one fly (file sha, self digest, parent links,
    fly id); return the last one's (meta, state, sha)."""
    chain = fly_entry.get('checkpoints') or []
    if not chain:
        raise CohortStoreError(f'fly {fly_entry.get("fly_id")}: no checkpoints recorded')
    parent = None
    last = None
    for link in chain:
        path = Path(root) / link['file']
        meta, state, digest = read_checkpoint(path, link['sha256'])
        if meta.get('parent_checkpoint_sha256') != parent:
            raise CohortStoreError(f'{path}: parent_checkpoint_sha256 {meta.get("parent_checkpoint_sha256")} '
                                   f'breaks the chain (expected {parent}); refused')
        if meta.get('fly_id') != fly_entry['fly_id'] or meta.get('tick') != link['tick']:
            raise CohortStoreError(f'{path}: fly/tick disagree with the manifest; refused')
        parent = digest
        last = (meta, state, digest)
    return last
