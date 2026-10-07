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
import math
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from neurofly.privacy import host_description, portable_path, redact_local

ROOT = Path(__file__).resolve().parent
MANIFEST_SCHEMA = 'neurofly.run-manifest.v1'

# Per-assay trial bookkeeping saved with the state it belongs to (the graph
# checkpoint NPZ, or the modular brain JSON).  It is runner bookkeeping, not world
# or neural state, and it is never derived from brain, world or daemon counters.
# When a flag is false the matching number is only what this and later processes
# observed since an origin they could not see: elapsed_s is then a lower bound on
# the trial's true elapsed time, and current_trial is not a trial ordinal.
# segment_id names the observation segment in progress when the clock was saved, so
# a restored trial can name the interrupted measurement it does not continue.
TRIAL_CLOCK_SCHEMA = 'neurofly.assay-trial-clock.v1'
TRIAL_CLOCK_KEYS = frozenset(('schema', 'elapsed_s', 'current_trial', 'elapsed_known',
                              'trial_known', 'reason', 'segment_id'))
LEGACY_TRIAL_CLOCK_REASON = 'checkpoint_predates_trial_clock'


def trial_clock(elapsed_s: float = 0.0, current_trial: int = 1, *, elapsed_known: bool = True,
                trial_known: bool = True, reason: Optional[str] = None,
                segment_id: Optional[str] = None) -> dict:
    """A validated trial clock; the defaults are a fresh first trial."""
    return validate_trial_clock(dict(schema=TRIAL_CLOCK_SCHEMA, elapsed_s=elapsed_s,
                                     current_trial=current_trial, elapsed_known=elapsed_known,
                                     trial_known=trial_known, reason=reason, segment_id=segment_id))


def unknown_trial_clock(reason: str = LEGACY_TRIAL_CLOCK_REASON) -> dict:
    """State saved without a trial clock: both the elapsed time and the ordinal are unknown."""
    return trial_clock(elapsed_known=False, trial_known=False, reason=reason)


def validate_trial_clock(value) -> dict:
    """Return a validated copy, or raise ValueError.  Nothing is inferred or repaired."""
    if not isinstance(value, dict) or value.get('schema') != TRIAL_CLOCK_SCHEMA:
        raise ValueError('Invalid assay trial clock schema')
    extra = set(value) - TRIAL_CLOCK_KEYS
    missing = TRIAL_CLOCK_KEYS - set(value)
    if extra or missing:
        raise ValueError(f'Invalid assay trial clock fields (unexpected {sorted(extra)}, '
                         f'missing {sorted(missing)})')
    elapsed, number = value['elapsed_s'], value['current_trial']
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError('Invalid assay trial clock elapsed_s')
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError('Invalid assay trial clock current_trial')
    if any(type(value[k]) is not bool for k in ('elapsed_known', 'trial_known')):
        raise ValueError('Invalid assay trial clock knowledge flags')
    reason = value['reason']
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError('Invalid assay trial clock reason')
    if (not value['elapsed_known'] or not value['trial_known']) and not reason:
        raise ValueError('Unknown assay trial clock requires a reason')
    segment = value['segment_id']
    if segment is not None and (not isinstance(segment, str) or not segment):
        raise ValueError('Invalid assay trial clock segment_id')
    if value['elapsed_known'] and value['trial_known'] and reason is not None:
        raise ValueError('Known assay trial clock must not carry an unknown reason')
    return dict(value, elapsed_s=float(elapsed))

# Canonical identity of the graph-to-arena input/output method.  The serialized
# bytes are immutable so callers cannot mutate the object after its hash is
# computed.  CARD-02B stores this declaration in a new linked run manifest;
# CARD-02A publishes the version/hash with each graph reply.
GRAPH_IO_VERSION = 'graph-arena-io-v2-unassisted'
_GRAPH_IO_CONFIG = {
    'schema': 'neurofly.graph-io.v2',
    'version': GRAPH_IO_VERSION,
    'engineered_assistance': {'enabled': False, 'applied': []},
    'decoders': {
        'forward': 'min(35, max(0, 1.5*DNp09_hz + 0.4*DNb01_hz)) mm/s',
        'yaw': '0.02*(DNa02_L_hz - DNa02_R_hz) rad/s',
        'reverse': 'MDN_hz > 20 -> -15 mm/s',
        'escape_raw': 'DNp01 spike -> 35 mm/s, state ESCAPE',
        'escape_applied': 'arena primitive -> 3.5 mm/s for 0.2 s',
        'empty_pool_numeric_input_hz': 0.0,
    },
    'input_probes': [
        {'name': 'orn_food', 'cell_types': ['ORN_DM1'], 'entry_stage': 'peripheral receptor',
         'stimulus': 'mean_a | odor_conc | odor_a', 'formula': 'min(40, 35*stimulus)',
         'threshold': '> 0.001', 'status': 'unverified'},
        {'name': 'orn_danger', 'cell_types': ['ORN_DA2'], 'entry_stage': 'peripheral receptor',
         'stimulus': 'mean_b | odor_b', 'formula': 'min(45, 40*stimulus)',
         'threshold': '> 0.001', 'status': 'unverified'},
        {'name': 'photoreceptor_l', 'cell_types': ['R1-R6 left'], 'entry_stage': 'photoreceptor',
         'stimulus': 'retina_photoreceptors_l', 'formula': '20*mean(stimulus)*contrast',
         'threshold': None, 'status': 'unverified'},
        {'name': 'photoreceptor_r', 'cell_types': ['R1-R6 right'], 'entry_stage': 'photoreceptor',
         'stimulus': 'retina_photoreceptors_r', 'formula': '20*mean(stimulus)*contrast',
         'threshold': None, 'status': 'unverified'},
        {'name': 'looming', 'cell_types': ['LC4', 'LPLC2'],
         'entry_stage': 'visual projection neuron; bypasses retina and optic lobe',
         'stimulus': 'looming_detected | looming_theta', 'formula': 'min(55, 15 + 40*theta_rad)',
         'threshold': 'looming_detected or theta_rad > 0.15', 'status': 'unverified'},
        {'name': 'courtship_cva', 'cell_types': ['ORN_DA1'], 'entry_stage': 'peripheral receptor',
         'stimulus': 'cva_concentration | cva_odor', 'formula': 'min(35, 30*stimulus)',
         'threshold': '> 0.001', 'status': 'unverified'},
        {'name': 'jon_wind', 'cell_types': ['JO-* (first 50)'], 'entry_stage': 'peripheral receptor',
         'stimulus': 'wind_speed_mm_s', 'formula': 'min(35, 0.15*wind_speed_mm_s)',
         'threshold': '> 3 mm/s', 'status': 'unverified'},
        {'name': 'thermo', 'cell_types': ['thermosensory'], 'entry_stage': 'peripheral receptor',
         'stimulus': 'temperature_degC', 'formula': 'min(40, 2.5*abs(temperature_degC - 24))',
         'threshold': 'abs(temperature_degC - 24) > 2 degC', 'status': 'unverified'},
        {'name': 'optomotor', 'cell_types': ['T4', 'T5'],
         'entry_stage': 'motion detector; direction imposed by encoder',
         'stimulus': 'retinal_slip_rad_s, contrast', 'formula': 'WP5 OptomotorEncoder (I/O-map pinned)',
         'threshold': None, 'status': 'verified-provisional-input-imposed'},
    ],
    'removed_undisclosed_drive': ['DNb01 +12 tonic current', '+5 mm/s forward floor',
                                   'ER contrast constant', 'EL +18 tonic current'],
}
GRAPH_IO_CONFIG_JSON = json.dumps(_GRAPH_IO_CONFIG, sort_keys=True, separators=(',', ':'),
                                  ensure_ascii=True)
GRAPH_IO_SHA256 = hashlib.sha256(GRAPH_IO_CONFIG_JSON.encode()).hexdigest()


def graph_io_declaration(*, include_config: bool = False) -> dict:
    """Versioned, hashable declaration of the effective graph arena I/O method."""
    out = {'version': GRAPH_IO_VERSION, 'sha256': GRAPH_IO_SHA256}
    if include_config:
        out['config'] = json.loads(GRAPH_IO_CONFIG_JSON)
    return out


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
                'brainlab-lif-v1', ('brainlab/engine.py', 'brainlab/brain.py', 'neurofly_daemon.py',
                                    'arena.py', 'provenance.py'),
                'Full prepared MaleCNS graph, fixed weights, LIF proxy dynamics.'),
    BackendSpec('connectome-plastic', True, True,
                'declared plastic edge subset: sparse weight deltas + rule state per instance',
                'brainlab-lif-plastic-v1', ('brainlab/engine.py', 'brainlab/brain.py', 'experiment_registry.py',
                                            'neurofly_daemon.py', 'arena.py', 'provenance.py'),
                'Full graph with internal plasticity restricted to a declared edge subset and rule.'),
    BackendSpec('connectome-with-trained-readout', True, True,
                'external readout weights (brainlab.learning.Readout); graph weights fixed',
                'brainlab-lif-readout-v1', ('brainlab/engine.py', 'brainlab/brain.py', 'brainlab/learning.py',
                                            'neurofly_daemon.py', 'arena.py', 'provenance.py'),
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
    # 'root' is kept for readers of older manifests; it is written as a placeholder
    # (``<repo>``), never as the absolute checkout path.
    info: Dict[str, Any] = {'root': portable_path(root)}
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


# Modules whose code writes or drives a recording.  Their loaded files are hashed
# as the executing writer's identity (``running_code_identity``).
WRITER_MODULES = ('neurofly.recording', 'neurofly_daemon', 'experiment_registry', 'provenance',
                  'experiment_brains', 'arena', 'brainlab.brain', 'brainlab.engine')
# Files pip writes per installation; they are not part of the distributed code.
_VENV_SPECIFIC = ('INSTALLER', 'REQUESTED', 'direct_url.json', 'RECORD')


def _installed_payload(dist) -> dict:
    """Hash every file the distribution's RECORD lists, from disk, and check it.

    The digest covers the files the wheel ships (paths inside site-packages, minus
    pip's per-install files), so equal code installed anywhere gives the same value.
    """
    import base64
    import csv
    import io
    files, mismatches = {}, []
    for row in csv.reader(io.StringIO(dist.read_text('RECORD') or '')):
        if not row:
            continue
        rel, recorded = row[0], (row[1] if len(row) > 1 else '')
        if (rel.startswith('..') or rel.endswith('.pyc') or '__pycache__' in rel
                or Path(rel).name in _VENV_SPECIFIC):
            continue
        try:
            data = Path(dist.locate_file(rel)).read_bytes()
        except OSError:
            mismatches.append(rel)
            continue
        files[rel] = hashlib.sha256(data).hexdigest()
        algo, _, value = recorded.partition('=')
        if algo != 'sha256' or base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b'=').decode() != value:
            mismatches.append(rel)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(files=len(files), files_sha256=digest, record_mismatches=sorted(mismatches))


def running_code_identity(modules: tuple = WRITER_MODULES) -> dict:
    """The code executing now, never an identity inherited from a saved run.

    ``loaded_module_sha256`` hashes the files the interpreter actually imported.
    When they come from an installed ``neurofly`` distribution, its whole installed
    payload is hashed against RECORD (``ident`` = name-version+files:digest; git is
    not needed).  Otherwise the code runs from a source tree: git commit and dirty
    flag where git is available.
    """
    import sys
    loaded = {}
    for name in modules:
        path = getattr(sys.modules.get(name), '__file__', None)
        if path:
            loaded[name] = _sha(Path(path))
    info: Dict[str, Any] = dict(role='executing_writer', loaded_module_sha256=loaded)
    try:
        from importlib import metadata
        dist = metadata.distribution('neurofly')
    except Exception:  # noqa: BLE001 -- no installed distribution: a source tree
        dist = None
    running = Path(getattr(sys.modules.get('neurofly.recording'), '__file__', None)
                   or ROOT / 'neurofly' / 'recording.py').resolve()
    installed = False
    if dist is not None:
        try:
            installed = Path(dist.locate_file('neurofly/recording.py')).resolve() == running
        except (OSError, TypeError, ValueError):
            installed = False
    if installed:
        payload = _installed_payload(dist)
        name = dist.metadata['Name']
        info.update(kind='installed_distribution', distribution=name, version=dist.version,
                    ident=f"{name}-{dist.version}+files:{payload['files_sha256']}", **payload)
    else:
        from neurofly import __version__
        revision = source_revision(ROOT)
        info.update(kind='source_tree', version=__version__, commit=revision.get('commit'),
                    dirty=revision.get('dirty'))
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
    graph_io: Optional[dict] = None
    lineage: Optional[dict] = None
    created_at: float = field(default_factory=time.time)
    # Non-identifying hardware/software description (neurofly.privacy.host_description).
    # Manifests written before October 2026 hold a string here; both shapes are read.
    host: Any = field(default_factory=host_description)

    @classmethod
    def create(cls, *, backend: str, assay: str, instance_id: str, seed: int,
               graph: Optional[dict], dynamics: dict, learned_parameter_locations: Dict[str, str],
               rng: Optional[np.random.Generator] = None, test_mode: bool = False,
               intervention_schedule: Optional[List[dict]] = None, parent_run_id: Optional[str] = None,
               graph_io: Optional[dict] = None, lineage: Optional[dict] = None,
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
                   intervention_schedule=copy.deepcopy(intervention_schedule or []),
                   parent_run_id=parent_run_id, graph_io=copy.deepcopy(graph_io),
                   lineage=copy.deepcopy(lineage))

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
                    io_map_sha256=graph.get('io_map_sha256'), graph_io=copy.deepcopy(self.graph_io),
                    synthetic=self.synthetic,
                    test_mode=self.test_mode, label=self.label)

    def to_dict(self) -> dict:
        """The manifest as written to disk and shown in the UI: absolute local paths
        (graph files, checkpoints, source root) are replaced by placeholders."""
        return redact_local(asdict(self))

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
