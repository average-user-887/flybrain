"""Per-assay full-graph controller instances with atomic, versioned checkpoints.

One immutable graph (CSR arrays, IDs, stable neuron map) is loaded once and
shared read-only.  Each assay owns an instance with its own instance ID,
checkpoint directory, LIF transients, delay queue, RNG, learned parameters
(sparse plastic deltas or readout weights) and an opaque world snapshot.

Only one instance is active at a time.  Activating another first checkpoints
the active one and releases its mutable state; inactive instances never step.
Checkpoints are NumPy ``.npz`` files without pickle, written to a temporary
file, fsynced and renamed, then published through ``CURRENT.json``.  An
interrupted write leaves ``CURRENT.json`` pointing at the last valid version.

Modular experiment-brain checkpoints (``experiment_brains`` JSON) are a
different format and are refused, never reinterpreted as graph weights.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from brainlab.brain import Brain, STATE_ARRAYS
from brainlab.graph_identity import (GraphIdentity, GraphUnavailable, active_dynamics,
                                     active_dynamics_version, synthetic_test_graph, verify_graph)
from provenance import (BACKENDS, GRAPH_BACKENDS, BackendError, RunManifest, atomic_write_bytes,
                        restore_rng, rng_state, source_revision)

ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_ROOT = ROOT / 'outputs/registry'
CHECKPOINT_FORMAT = 'neurofly.graph-instance-checkpoint.v1'
REGISTRY_FORMAT = 'neurofly.experiment-registry.v1'

# Must stay identical to experiment_brains.PARADIGMS (checked by tests).
PARADIGMS = (
    'open-arena', 't-maze', 'y-maze', 'heat-maze', 'buridan', 'visual-operant',
    'wind-tunnel', 'looming-escape', 'optomotor', 'gap-crossing', 'circadian-dam',
    'courtship', 'labyrinth', 'multisensory-sandbox',
)


class IncompatibleCheckpoint(RuntimeError):
    """A checkpoint belongs to another format, backend or graph."""


class CheckpointCorrupt(RuntimeError):
    """The published checkpoint does not match its recorded hash."""


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _derive_seed(assay: str, backend: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{assay}|{backend}'.encode()).digest()[:4], 'big')


# ---------------------------------------------------------------------------
# Shared immutable graph
# ---------------------------------------------------------------------------
def default_registry_dir(output_dir: Path, dynamics: Optional[str] = None) -> Path:
    """Per-dynamics registry root: checkpoints never cross dynamics versions,
    so v1 brains stay in ``registry/`` and v2/v3 brains get their own root."""
    version = dynamics or active_dynamics_version()
    return Path(output_dir) / ('registry' if version == 'v1' else f'registry-{version}')


class SharedGraph:
    """Read-only graph arrays plus verified identity, loaded once per process."""

    def __init__(self, arrays: Dict[str, np.ndarray], identity: GraphIdentity, io_map: Optional[dict] = None):
        validator = Brain(arrays=arrays, validate=True, backend='cpu')   # validates once, host only
        self.arrays = validator.graph_arrays()
        for value in self.arrays.values():
            value.flags.writeable = False
        self.identity = identity
        self.io_map = io_map or {}
        self.n = validator.n

    @classmethod
    def load(cls, graph_dir=None, connectome_dir=None) -> 'SharedGraph':
        """Verify hashes/IDs, then load the real graph. Raises GraphUnavailable."""
        identity = verify_graph(graph_dir, connectome_dir)
        before = os.stat(identity.graph_path)
        with np.load(identity.graph_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in ('ptr', 'post', 'weight', 'ids')}
        after = os.stat(identity.graph_path)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise GraphUnavailable('Graph changed on disk between verification and load')
        from brainlab.graph_identity import DN_CHANNELS
        return cls(arrays, identity, dict(DN_CHANNELS))

    @classmethod
    def load_for_dynamics(cls, graph_dir=None, connectome_dir=None, dynamics=None) -> 'SharedGraph':
        """Load the real graph with the weights the dynamics version declares.

        v3 is defined together with the v3 transmitter policy (aminergic
        neurons carry no fast weight; docs/LIF_DYNAMICS_SPEC.md §4.3), applied
        in memory with its own graph_sha256.  v1/v2 use the pinned weights.
        """
        shared = cls.load(graph_dir, connectome_dir)
        if (dynamics or active_dynamics_version()) == 'v3':
            from brainlab.transmitter_policy import apply_to_shared
            shared, _ = apply_to_shared(shared, connectome_dir=connectome_dir)
        return shared

    @classmethod
    def synthetic(cls, *, allow_synthetic: bool = False, **kwargs) -> 'SharedGraph':
        if not allow_synthetic:
            raise BackendError('Synthetic graphs need allow_synthetic=True (explicit test option)')
        arrays, identity, io_map = synthetic_test_graph(**kwargs)
        return cls(arrays, identity, io_map)

    def sources_of(self, edges: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.arrays['ptr'], edges, side='right') - 1


# ---------------------------------------------------------------------------
# Learning components
# ---------------------------------------------------------------------------
class TestOnlyCoactivityRule:
    """Bounded pre*post coactivity update on a declared edge subset.

    TEST FIXTURE ONLY.  It exists to exercise checkpoint isolation; it is not
    the WP6 model specification and has no biological claim.
    """
    __test__ = False  # not a pytest class
    name = 'test-only-coactivity'
    test_only = True

    def __init__(self, edges, eta: float = 0.01, bound: float = 2.0):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.eta = float(eta)
        self.bound = float(bound)

    def describe(self) -> dict:
        return dict(name=self.name, test_only=True, n_edges=int(len(self.edges)), eta=self.eta,
                    bound=self.bound, units='weight units per spike-count product per step')

    def update(self, delta: np.ndarray, pre_counts: np.ndarray, post_counts: np.ndarray) -> None:
        delta += self.eta * pre_counts.astype(np.float32) * post_counts.astype(np.float32)
        np.clip(delta, -self.bound, self.bound, out=delta)


@dataclass
class StepResult:
    step: int
    counts: np.ndarray
    action: Optional[int] = None
    reward: Optional[float] = None


class GraphInstance:
    """Mutable state of one assay's controller.  Created only by the registry."""

    def __init__(self, registry: 'ExperimentRegistry', assay: str, backend: str, instance_id: str,
                 seed: int, manifest: RunManifest):
        self.registry = registry
        self.shared = registry.shared
        self.assay, self.backend, self.instance_id, self.seed = assay, backend, instance_id, seed
        self.manifest = manifest
        self.rng = np.random.default_rng(seed)
        self.brain = Brain(arrays=self.shared.arrays, validate=False)
        identity = self.shared.identity
        if (self.brain.dynamics == 'v3' and not identity.synthetic
                and 'v3-modulatory-only' not in identity.dataset):
            raise BackendError('v3 brains need the v3 transmitter-policy weights; load the graph with '
                               'SharedGraph.load_for_dynamics() (or select v1 explicitly)')
        self.step_index = 0
        self.world_state: dict = {}
        self.checkpoint_version = 0
        self.rule = None
        self.plastic_edges = np.zeros(0, dtype=np.int64)
        self.plastic_delta = np.zeros(0, dtype=np.float32)
        self._working_weight = None
        self.readout = None
        if backend == 'connectome-plastic':
            self.rule = registry.plasticity_rule
            self.plastic_edges = self.rule.edges.copy()
            self.plastic_delta = np.zeros(len(self.plastic_edges), dtype=np.float32)
            self._pre = self.shared.sources_of(self.plastic_edges)
            self._post = self.shared.arrays['post'][self.plastic_edges]
            self._materialize()
        elif backend == 'connectome-with-trained-readout':
            from brainlab.learning import Readout
            self.readout = Readout(self.shared.n)

    # -- plastic weights: only the active instance holds a working copy --------
    def _materialize(self):
        working = self.shared.arrays['weight'].copy()
        working[self.plastic_edges] += self.plastic_delta
        self._working_weight = working
        self.brain.weight = working

    def release(self):
        """Drop mutable arrays (after checkpointing). The instance is unusable after."""
        self.brain = None
        self._working_weight = None

    @property
    def identity(self) -> dict:
        ident = self.manifest.identity()
        ident.update(step=self.step_index, checkpoint_version=self.checkpoint_version)
        return ident

    def step(self, currents, duration_ms: float, *, target: Optional[int] = None) -> StepResult:
        if self.brain is None or self.registry.active is not self:
            raise RuntimeError(f'Instance {self.instance_id} is not active; inactive instances never step')
        if self.rule is not None and self.registry.learning_enabled:
            # The rule is declared per ``rule.dt`` of simulated time, so a longer
            # control step is split into rule-sized brain steps, each followed by
            # one rule update and a weight refresh.
            n_sub = self.rule.substeps(duration_ms) if hasattr(self.rule, 'substeps') else 1
            sub_ms = duration_ms / n_sub
            counts = None
            for _ in range(n_sub):
                sub_counts, _ = self.brain.step(currents, sub_ms)
                sub_counts = np.array(sub_counts, copy=True)
                counts = sub_counts if counts is None else counts + sub_counts
                try:
                    self.rule.update(self.plastic_delta, sub_counts[self._pre], sub_counts[self._post],
                                     full_counts=sub_counts)
                except TypeError:
                    self.rule.update(self.plastic_delta, sub_counts[self._pre], sub_counts[self._post])
                self._working_weight[self.plastic_edges] = (
                    self.shared.arrays['weight'][self.plastic_edges] + self.plastic_delta)
                self.brain.update_weights(self.plastic_edges)
        else:
            counts, _ = self.brain.step(currents, duration_ms)
        result = StepResult(step=self.step_index, counts=counts)
        if self.readout is not None:
            from brainlab.learning import features_from_counts
            features = features_from_counts(counts, 0.0, slice(None))
            probability = self.readout.probability(features)
            action = int(self.rng.random() < probability)
            result.action = action
            if target is not None and self.registry.learning_enabled:
                reward = 1.0 if action == int(target) else -1.0
                self.readout.update(features, action, reward)
                result.reward = reward
        self.step_index += 1
        return result

    # -- checkpoint payload -----------------------------------------------------
    def state_arrays(self) -> Dict[str, np.ndarray]:
        state = self.brain.snapshot_state()
        arrays = {f'brain.{name}': state[name] for name in STATE_ARRAYS}
        arrays['plastic.edges'] = self.plastic_edges
        arrays['plastic.delta'] = self.plastic_delta
        if self.readout is not None:
            arrays['readout.weights'] = self.readout.weights
        return arrays

    def meta(self) -> dict:
        state = self.brain.snapshot_state()
        graph = self.shared.identity
        return dict(format=CHECKPOINT_FORMAT, assay=self.assay, backend=self.backend,
                    instance_id=self.instance_id, run_id=self.manifest.run_id, seed=self.seed,
                    graph_sha256=graph.graph_sha256, io_map_sha256=graph.io_map_sha256,
                    synthetic=graph.synthetic, step_index=self.step_index,
                    brain_scalars=dict(cursor=state['cursor'], total_spikes=state['total_spikes'],
                                       sim_ms=state['sim_ms']),
                    rng_state=rng_state(self.rng), world_state=self.world_state,
                    rule=self.rule.describe() if self.rule is not None else None,
                    readout_rate=self.readout.rate if self.readout is not None else None)

    def load_payload(self, meta: dict, arrays: Dict[str, np.ndarray]):
        state = {name: arrays[f'brain.{name}'] for name in STATE_ARRAYS}
        state.update(meta['brain_scalars'])
        self.brain.restore_state(state)
        self.rng = restore_rng(meta['rng_state'])
        self.step_index = int(meta['step_index'])
        self.world_state = meta.get('world_state') or {}
        if self.backend == 'connectome-plastic':
            if not np.array_equal(arrays['plastic.edges'], self.plastic_edges):
                raise IncompatibleCheckpoint('Checkpoint plastic edge subset differs from the declared rule')
            self.plastic_delta = arrays['plastic.delta'].astype(np.float32).copy()
            self._materialize()
        if self.readout is not None:
            self.readout.weights = arrays['readout.weights'].astype(np.float64).copy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ExperimentRegistry:
    def __init__(self, shared: SharedGraph, root: Path = DEFAULT_REGISTRY_ROOT, *, test_mode: bool = False,
                 plasticity_rule=None, learning_enabled: bool = True):
        if shared.identity.synthetic and not test_mode:
            raise BackendError('A synthetic graph can only back a registry in explicit test mode')
        if plasticity_rule is not None and getattr(plasticity_rule, 'test_only', False) and not test_mode:
            raise BackendError('Test-only plasticity rules require test mode')
        self.shared = shared
        self.root = Path(root)
        self.test_mode = test_mode
        self.plasticity_rule = plasticity_rule
        self.learning_enabled = learning_enabled
        self.active: Optional[GraphInstance] = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self._read_index()

    # -- index ----------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.root / 'registry.json'

    def _read_index(self) -> dict:
        if not self.index_path.exists():
            return dict(format=REGISTRY_FORMAT, graph_sha256=self.shared.identity.graph_sha256,
                        synthetic=self.shared.identity.synthetic, instances={})
        index = json.loads(self.index_path.read_text())
        if index.get('format') != REGISTRY_FORMAT:
            raise IncompatibleCheckpoint(f'{self.index_path} is not a {REGISTRY_FORMAT} index')
        if index.get('graph_sha256') != self.shared.identity.graph_sha256:
            raise IncompatibleCheckpoint('Registry was created for a different graph; use a separate root')
        return index

    def _write_index(self):
        atomic_write_bytes(self.index_path, (json.dumps(self.index, indent=2) + '\n').encode())

    def instance_dir(self, instance_id: str) -> Path:
        entry = self.index['instances'][instance_id]
        return self.root / entry['assay'] / entry['backend'] / instance_id

    def instance_id_for(self, assay: str, backend: str) -> str:
        """Return (creating if needed) the instance ID for one assay/backend pair."""
        if assay not in PARADIGMS:
            raise ValueError(f'Unknown assay {assay!r}')
        if backend not in GRAPH_BACKENDS:
            raise BackendError(f'{backend!r} is not a graph backend; modular brains live in experiment_brains')
        for instance_id, entry in self.index['instances'].items():
            if entry['assay'] == assay and entry['backend'] == backend:
                return instance_id
        if backend == 'connectome-plastic' and self.plasticity_rule is None:
            if not self.shared.identity.synthetic:
                from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
                self.plasticity_rule = VisualHeadingPlasticityRule.from_shared(self.shared)
            else:
                raise BackendError('connectome-plastic needs a declared plasticity rule (WP6); none is registered')
        instance_id = uuid.uuid4().hex
        seed = _derive_seed(assay, backend)
        manifest = RunManifest.create(
            backend=backend, assay=assay, instance_id=instance_id, seed=seed,
            graph=self.shared.identity.to_dict(), dynamics=active_dynamics(),
            learned_parameter_locations=self._learned_locations(backend, assay, instance_id),
            rng=np.random.default_rng(seed), test_mode=self.test_mode,
            source=source_revision(files=BACKENDS[backend].source_files))
        if backend == 'connectome-plastic':
            manifest.dynamics['plasticity_rule'] = self.plasticity_rule.describe()
        self.index['instances'][instance_id] = dict(assay=assay, backend=backend, seed=seed,
                                                    run_id=manifest.run_id)
        self._write_index()
        manifest.write(self.instance_dir(instance_id) / 'manifest.json')
        return instance_id

    def _learned_locations(self, backend, assay, instance_id) -> dict:
        base = f'{assay}/{backend}/{instance_id}/checkpoints/ckpt-*.npz'
        locations = {'brain_transients': f'{base} [brain.*]', 'rng': f'{base} [meta.rng_state]',
                     'world_state': f'{base} [meta.world_state]'}
        if backend == 'connectome-plastic':
            locations['plastic_weight_deltas'] = f'{base} [plastic.edges, plastic.delta]'
        if backend == 'connectome-with-trained-readout':
            locations['readout_weights'] = f'{base} [readout.weights]'
        return locations

    def manifest(self, instance_id: str) -> RunManifest:
        return RunManifest.read(self.instance_dir(instance_id) / 'manifest.json')

    # -- activation -------------------------------------------------------------
    def activate(self, assay: str, backend: str) -> GraphInstance:
        """Checkpoint and release the active instance, then load the target.

        Returns only once the target's brain snapshot (and stored world state)
        is restored, so callers may acknowledge the switch on return.
        """
        instance_id = self.instance_id_for(assay, backend)
        if self.active is not None and self.active.instance_id == instance_id:
            return self.active
        if self.active is not None:
            self.checkpoint()
            self.active.release()
            self.active = None
        entry = self.index['instances'][instance_id]
        manifest = self.manifest(instance_id)
        instance = GraphInstance(self, assay, backend, instance_id, entry['seed'], manifest)
        current = self.current_pointer(instance_id)
        if current is not None:
            meta, arrays = self.read_checkpoint(instance_id)
            instance.load_payload(meta, arrays)
            instance.checkpoint_version = current['version']
            self._event(instance, 'restore', version=current['version'], step=instance.step_index)
        self.active = instance
        return instance

    def is_current_packet(self, identity: dict) -> bool:
        """Stale packets (other run/instance) must be rejected by the consumer."""
        return (self.active is not None and identity.get('instance_id') == self.active.instance_id
                and identity.get('run_id') == self.active.manifest.run_id)

    # -- checkpoints --------------------------------------------------------------
    def current_pointer(self, instance_id: str) -> Optional[dict]:
        path = self.instance_dir(instance_id) / 'CURRENT.json'
        return json.loads(path.read_text()) if path.exists() else None

    def checkpoint(self, world_state: Optional[dict] = None) -> Path:
        instance = self.active
        if instance is None:
            raise RuntimeError('No active instance to checkpoint')
        if world_state is not None:
            instance.world_state = world_state
        directory = self.instance_dir(instance.instance_id) / 'checkpoints'
        version = instance.checkpoint_version + 1
        buffer = io.BytesIO()
        meta = instance.meta()
        meta['version'] = version
        np.savez(buffer, meta=np.array(json.dumps(meta, allow_nan=False)), **instance.state_arrays())
        data = buffer.getvalue()
        path = directory / f'ckpt-{version:06d}.npz'
        atomic_write_bytes(path, data)
        pointer = dict(version=version, file=path.name, sha256=_sha_bytes(data), bytes=len(data),
                       step_index=instance.step_index, format=CHECKPOINT_FORMAT)
        atomic_write_bytes(directory.parent / 'CURRENT.json', (json.dumps(pointer, indent=2) + '\n').encode())
        instance.checkpoint_version = version
        self._event(instance, 'checkpoint', version=version, step=instance.step_index, sha256=pointer['sha256'])
        return path

    def read_checkpoint(self, instance_id: str, version: Optional[int] = None):
        entry = self.index['instances'][instance_id]
        directory = self.instance_dir(instance_id)
        pointer = self.current_pointer(instance_id)
        if pointer is None:
            raise FileNotFoundError(f'No checkpoint for {instance_id}')
        if version is None:
            version = pointer['version']
        path = directory / 'checkpoints' / f'ckpt-{version:06d}.npz'
        meta, arrays = load_checkpoint_file(path, expected_sha256=pointer['sha256'] if version == pointer['version'] else None)
        for key, expected in (('backend', entry['backend']), ('assay', entry['assay']),
                              ('instance_id', instance_id), ('graph_sha256', self.shared.identity.graph_sha256),
                              ('io_map_sha256', self.shared.identity.io_map_sha256)):
            if meta.get(key) != expected:
                raise IncompatibleCheckpoint(f'Checkpoint {path.name} {key}={meta.get(key)!r}, expected {expected!r}')
        return meta, arrays

    def _event(self, instance: GraphInstance, kind: str, **fields):
        instance.manifest.record_event(kind, **fields)
        line = json.dumps(dict(kind=kind, instance_id=instance.instance_id, **fields)) + '\n'
        path = self.instance_dir(instance.instance_id) / 'events.jsonl'
        with path.open('a') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def load_checkpoint_file(path: Path, expected_sha256: Optional[str] = None):
    """Load a graph-instance checkpoint. Refuses modular JSON and foreign npz files."""
    path = Path(path)
    if path.suffix != '.npz':
        raise IncompatibleCheckpoint(f'{path.name}: not a graph-instance checkpoint '
                                     '(modular brain JSON is never reinterpreted as graph weights)')
    data = path.read_bytes()
    if expected_sha256 is not None and _sha_bytes(data) != expected_sha256:
        raise CheckpointCorrupt(f'{path} hash does not match CURRENT.json')
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if 'meta' not in archive.files:
                raise IncompatibleCheckpoint(f'{path.name}: no checkpoint metadata')
            meta = json.loads(str(archive['meta']))
            arrays = {name: archive[name] for name in archive.files if name != 'meta'}
    except (ValueError, OSError) as error:
        raise IncompatibleCheckpoint(f'{path.name}: unreadable ({error})') from error
    if meta.get('format') != CHECKPOINT_FORMAT:
        raise IncompatibleCheckpoint(f'{path.name}: format {meta.get("format")!r} is not {CHECKPOINT_FORMAT}')
    return meta, arrays
