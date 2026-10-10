"""Path A run provenance: the smallest immutable record that binds a result row to
the frozen contract, the engine, the graph and the source that produced it.

Every row written by ``scripts/pathA_run.py`` carries ``provenance`` =
``{contract_sha256, graph_npz_sha256, dynamics, backend, engine_sha256,
code_sha, seed}``.  ``check_row`` rejects a row whose record is missing or
differs from the expected one; the runner applies it before resuming and the
analyser before using any row, so a stale or foreign row can never be mixed in.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Files that define the arithmetic of a run: the compiled kernels, the Brain
# wrapper and the v3 weight policy.  Hashed together as the engine identity.
ENGINE_FILES = ('brainlab/engine.py', 'brainlab/brain.py', 'brainlab/transmitter_policy.py')
# Files that must be committed and unmodified for a run to be attributable to code_sha.
SOURCE_FILES = ENGINE_FILES + ('scripts/pathA_run.py', 'scripts/pathA_provenance.py')
FIELDS = ('contract_sha256', 'graph_npz_sha256', 'dynamics', 'backend', 'engine_sha256', 'code_sha', 'seed')


class ProvenanceError(RuntimeError):
    pass


def sha256_bytes(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine_sha256(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for rel in ENGINE_FILES:
        h.update(rel.encode() + b'\0' + (root / rel).read_bytes() + b'\0')
    return h.hexdigest()


def code_sha(root: Path = ROOT) -> str:
    """HEAD of the checkout; refuses if any source file is modified or untracked."""
    git = ['git', '-C', str(root)]
    head = subprocess.run(git + ['rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(git + ['status', '--porcelain', '--', *SOURCE_FILES],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ProvenanceError(f'source files are not committed/clean, run not attributable to {head}:\n{dirty}')
    return head


def expected(contract_sha: str, graph_sha: str, dynamics: str, backend: str, eng_sha: str, code: str) -> dict:
    return dict(contract_sha256=contract_sha, graph_npz_sha256=graph_sha, dynamics=dynamics, backend=backend,
                engine_sha256=eng_sha, code_sha=code)


def stamp(base: dict, seed: int) -> dict:
    return dict(base, seed=int(seed))


HEX_LENGTHS = dict(contract_sha256=64, graph_npz_sha256=64, engine_sha256=64, code_sha=40)


def check_complete(prov: dict, where: str = 'record') -> None:
    """Every field present and well formed: hashes are lowercase hex of the right length,
    dynamics/backend non-empty strings, seed an int.  A missing key never matches."""
    if not isinstance(prov, dict):
        raise ProvenanceError(f'{where}: no provenance record (unverified row)')
    for k, n in HEX_LENGTHS.items():
        v = prov.get(k)
        if not (isinstance(v, str) and len(v) == n and all(c in '0123456789abcdef' for c in v)):
            raise ProvenanceError(f'{where}: provenance {k} missing or not a {n}-hex hash: {v!r}')
    for k in ('dynamics', 'backend'):
        if not (isinstance(prov.get(k), str) and prov.get(k)):
            raise ProvenanceError(f'{where}: provenance {k} missing or empty')
    if not isinstance(prov.get('seed'), int) or isinstance(prov.get('seed'), bool):
        raise ProvenanceError(f'{where}: provenance seed missing or not an int')


def check_row(row: dict, base: dict, *, fields=FIELDS) -> None:
    """Raise ProvenanceError unless ``row`` carries exactly ``base`` plus its own seed."""
    prov = row.get('provenance')
    where = f"row {row.get('condition')!r} seed {row.get('seed')!r}"
    check_complete(prov, where)
    for k in ('engine_sha256', 'code_sha'):
        if k in fields and k not in base:
            raise ProvenanceError(f'{where}: expected identity lacks {k}; refusing')
    want = stamp({k: base[k] for k in base if k in fields}, row.get('seed'))
    for k in fields:
        if k == 'seed':
            if prov.get('seed') != row.get('seed') or not isinstance(row.get('seed'), int):
                raise ProvenanceError(f'{where}: provenance seed {prov.get("seed")!r} != row seed')
            continue
        if k not in want:
            continue
        if want[k] is None:
            raise ProvenanceError(f'{where}: expected {k} is undefined; refusing')
        if prov.get(k) != want[k]:
            raise ProvenanceError(f'{where}: provenance {k} = {prov.get(k)!r}, expected {want[k]!r}')
