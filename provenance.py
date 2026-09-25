"""Controller backends and run identity for NeuroFly.

Every controller path has one explicit, named backend.  They are never
interchangeable: a run records exactly one backend in its manifest, and a
missing graph or mapping is an error in scientific mode, not a reason to swap
in another controller.  Synthetic graphs and surrogate fallbacks are available
only through explicit test options and are labelled in the run identity.

Nothing here talks to running services.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST_SCHEMA = 'neurofly.run-manifest.v1'


@dataclass(frozen=True)
class BackendSpec:
    name: str
    requires_graph: bool
    scientific: bool             # may appear in scientific (non-test) runs
    learning_location: str       # where learned parameters live
    controller_version: str
    source_files: tuple
    description: str


BACKENDS: Dict[str, BackendSpec] = {spec.name: spec for spec in (
    BackendSpec('modular', False, True, 'modular mushroom-body KC->MBON weights (experiment_brains checkpoint JSON)',
                'modular-mb-v1', ('experiment_brains.py', 'circuit.py', 'arena.py'),
                'Hand-built modular controller with explicit assay reflexes. Not the connectome.'),
    BackendSpec('connectome-fixed', True, True, 'none (fixed released-graph weights)',
                'brainlab-lif-v1', ('brainlab/engine.py', 'brainlab/brain.py'),
                'Full prepared MaleCNS graph, fixed weights, LIF proxy dynamics.'),
    BackendSpec('connectome-plastic', True, True,
                'declared plastic edge subset: sparse weight deltas + rule state per instance',
                'brainlab-lif-plastic-v1', ('brainlab/engine.py', 'brainlab/brain.py', 'experiment_registry.py'),
                'Full graph with internal plasticity restricted to a declared edge subset and rule.'),
    BackendSpec('connectome-with-trained-readout', True, True,
                'external readout weights (brainlab.learning.Readout); graph weights fixed',
                'brainlab-lif-readout-v1', ('brainlab/engine.py', 'brainlab/brain.py', 'brainlab/learning.py'),
                'Fixed full graph plus an externally trained decoder. Decoder learning, not synaptic learning.'),
    BackendSpec('bridge-surrogate', False, False, 'modular MB inside ConnectomeBridge',
                'connectome-bridge-surrogate-v1', ('connectome_bridge.py',),
                'In-process hand-built ConnectomeBridge. Named after the connectome but runs no graph.'),
    BackendSpec('hybrid-bridge-rpc-experimental', True, False, 'modular MB inside ConnectomeBridge; graph fixed',
                'connectome-bridge-rpc-hybrid-v1', ('connectome_bridge.py', 'connectome_client.py',
                                                     'brainlab/cosim_server.py'),
                'Experimental hybrid: hand-built bridge whose DN rates are replaced by remote graph readouts.'),
    BackendSpec('synthetic-test-graph', False, False, 'as the wrapped backend, on a synthetic graph',
                'synthetic-test-v1', ('brainlab/graph_identity.py',),
                'SYNTHETIC TEST GRAPH. Test fixture only; never a scientific result.'),
)}

GRAPH_BACKENDS = ('connectome-fixed', 'connectome-plastic', 'connectome-with-trained-readout')


def controller_version_for(spec: 'BackendSpec', dynamics: Optional[dict]) -> str:
    """Re-pin a graph backend's controller version to the LIF dynamics version.

    A dynamics change is a new controller version (docs/LIF_DYNAMICS_SPEC.md):
    a run under the conductance-based v2 engine records ``brainlab-lif-v2``,
    never ``brainlab-lif-v1``, so old and new results can never be confused.
    Runs under v1 dynamics (the default before PR #10; v3 is now the default)
    keep exactly the string they had.
    """
    version = (dynamics or {}).get('dynamics_version')
    if not spec.requires_graph or not version or version == 'v1':
        return spec.controller_version
    if not spec.controller_version.endswith('-v1'):
        raise BackendError(f'Cannot re-pin controller version {spec.controller_version!r} '
                           f'for LIF dynamics {version!r}')
    return spec.controller_version[:-2] + version


class BackendError(RuntimeError):
    """A backend was requested in a mode that cannot honour its identity."""


def get_backend(name: str, *, scientific: bool = True, allow_test: bool = False) -> BackendSpec:
    if name not in BACKENDS:
        raise BackendError(f'Unknown controller backend {name!r}; choose one of {sorted(BACKENDS)}')
    spec = BACKENDS[name]
    if scientific and not spec.scientific and not allow_test:
        raise BackendError(f'Backend {name!r} is not a scientific backend; pass an explicit test option')
    return spec


def _sha(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def source_revision(root: Path = ROOT, files: Optional[tuple] = None) -> dict:
    """Git revision (read-only commands, no index refresh) plus file hashes."""
    info: Dict[str, Any] = {'root': str(root)}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    try:
        info['commit'] = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], env=env,
                                        capture_output=True, text=True, timeout=5).stdout.strip() or None
        status = subprocess.run(['git', '-C', str(root), '--no-optional-locks', 'status', '--porcelain',
                                 '--untracked-files=no'], env=env, capture_output=True, text=True, timeout=10)
        info['dirty'] = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info.update(commit=None, dirty=None)
    if files:
        info['file_sha256'] = {name: _sha(root / name) for name in files}
    return info


def rng_state(generator: np.random.Generator) -> dict:
    """JSON-safe bit generator state (big integers as strings)."""
    def encode(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return {'int': str(int(value))}
        return value
    return encode(generator.bit_generator.state)


def restore_rng(state: dict) -> np.random.Generator:
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'int'}:
                return int(value['int'])
            return {k: decode(v) for k, v in value.items()}
        return value
    decoded = decode(state)
    bit_generator = getattr(np.random, decoded['bit_generator'])()
    bit_generator.state = decoded
    return np.random.Generator(bit_generator)


@dataclass
class RunManifest:
    """Machine-readable identity of one run.  The UI and exports show this same dict."""
    backend: str
    assay: str
    instance_id: str
    seed: int
    graph: Optional[dict]                      # GraphIdentity.to_dict() or None (modular)
    dynamics: dict
    learned_parameter_locations: Dict[str, str]
    run_id: str = field(default_factory=lambda: time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12])
    schema: str = MANIFEST_SCHEMA
    controller_version: str = ''
    synthetic: bool = False
    test_mode: bool = False
    label: str = ''
    source: dict = field(default_factory=dict)
    rng_initial_state: Optional[dict] = None
    intervention_schedule: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    parent_run_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    host: str = field(default_factory=platform.node)

    @classmethod
    def create(cls, *, backend: str, assay: str, instance_id: str, seed: int,
               graph: Optional[dict], dynamics: dict, learned_parameter_locations: Dict[str, str],
               rng: Optional[np.random.Generator] = None, test_mode: bool = False,
               intervention_schedule: Optional[List[dict]] = None, parent_run_id: Optional[str] = None,
               source: Optional[dict] = None) -> 'RunManifest':
        spec = get_backend(backend, scientific=not test_mode, allow_test=test_mode)
        synthetic = bool(graph and graph.get('synthetic'))
        if spec.requires_graph and graph is None:
            raise BackendError(f'Backend {backend} requires a verified graph identity')
        if synthetic and not test_mode:
            raise BackendError('A synthetic graph requires an explicit test mode')
        label = spec.description
        if synthetic:
            label = f'SYNTHETIC TEST GRAPH - {backend} - not a scientific result'
        elif not spec.scientific:
            label = f'NON-SCIENTIFIC BACKEND - {backend}: {spec.description}'
        return cls(backend=backend, assay=assay, instance_id=instance_id, seed=int(seed), graph=graph,
                   dynamics=copy.deepcopy(dynamics), learned_parameter_locations=dict(learned_parameter_locations),
                   controller_version=controller_version_for(spec, dynamics),
                   synthetic=synthetic, test_mode=test_mode,
                   label=label, source=source if source is not None else source_revision(files=spec.source_files),
                   rng_initial_state=rng_state(rng) if rng is not None else None,
                   intervention_schedule=list(intervention_schedule or []), parent_run_id=parent_run_id)

    def record_event(self, kind: str, *, step: Optional[int] = None, **fields) -> dict:
        """Reset, fallback, disconnect, restore and intervention events. Never dropped."""
        event = dict(kind=kind, step=step, wall_time=time.time(), **fields)
        self.events.append(event)
        return event

    def identity(self) -> dict:
        """Compact identity carried on every telemetry packet."""
        graph = self.graph or {}
        return dict(run_id=self.run_id, instance_id=self.instance_id, assay=self.assay,
                    backend=self.backend, controller_version=self.controller_version,
                    graph_sha256=graph.get('graph_sha256'), neuron_map_sha256=graph.get('neuron_map_sha256'),
                    io_map_sha256=graph.get('io_map_sha256'), synthetic=self.synthetic,
                    test_mode=self.test_mode, label=self.label)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: Path) -> None:
        atomic_write_bytes(Path(path), (json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n').encode())

    @classmethod
    def read(cls, path: Path) -> 'RunManifest':
        data = json.loads(Path(path).read_text())
        if data.get('schema') != MANIFEST_SCHEMA:
            raise BackendError(f'Unsupported manifest schema {data.get("schema")!r}')
        return cls(**data)


DEFAULT_KEEP_CHECKPOINTS = 20
KEEP_CHECKPOINTS_ENV = 'NEUROFLY_KEEP_CHECKPOINTS'


def resolve_keep_checkpoints(keep: Optional[int] = None) -> int:
    """Checkpoints kept per instance: explicit value, else the env, else 20. 0 keeps all."""
    if keep is None:
        raw = os.environ.get(KEEP_CHECKPOINTS_ENV, '').strip()
        keep = int(raw) if raw else DEFAULT_KEEP_CHECKPOINTS
    keep = int(keep)
    if keep < 0:
        raise ValueError(f'keep_checkpoints must be >= 0 (0 keeps all), got {keep}')
    return keep


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a same-directory temporary file, fsync, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.partial')
    try:
        with temporary.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        pass
