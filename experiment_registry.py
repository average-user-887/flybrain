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
After each published checkpoint only the newest ``keep_checkpoints`` versions of
that instance stay on disk (default 20, env ``NEUROFLY_KEEP_CHECKPOINTS``; 0
keeps all).  The version ``CURRENT.json`` names is never removed.

Modular experiment-brain checkpoints (``experiment_brains`` JSON) are a
different format and are refused, never reinterpreted as graph weights.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from brainlab.brain import Brain, STATE_ARRAYS
from brainlab.runtime_backend import (compute_selector, preflight_fixed_store,
                                     require_fixed_amd, require_fixed_manifest, require_fixed_payload)
from brainlab.graph_identity import (GraphIdentity, GraphUnavailable, active_dynamics,
                                     active_dynamics_version, synthetic_test_graph, verify_graph)
from provenance import (BACKENDS, GRAPH_BACKENDS, BackendError, RunManifest, atomic_write_bytes,
                        graph_io_declaration, resolve_keep_checkpoints, restore_rng, rng_state,
                        source_revision)

ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_ROOT = ROOT / 'outputs/registry'
CHECKPOINT_FORMAT = 'neurofly.graph-instance-checkpoint.v1'
REGISTRY_FORMAT = 'neurofly.experiment-registry.v2'
LEGACY_REGISTRY_FORMAT = 'neurofly.experiment-registry.v1'

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


class NonFiniteState(RuntimeError):
    """The brain state holds NaN/inf: a compute failure, never written over a good checkpoint."""
    failure_class = 'compute'


class IOContinuationRequired(IncompatibleCheckpoint):
    """Saved state uses another graph I/O method and needs explicit continuation."""


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
        selected = compute_selector(getattr(registry, 'brain_backend', None))
        require_fixed_amd(selected, backend, learning=getattr(registry, 'learning_enabled', True))
        if selected == 'wgpu-amd':
            require_fixed_manifest(manifest.to_dict())
        self.shared = registry.shared
        self.assay, self.backend, self.instance_id, self.seed = assay, backend, instance_id, seed
        self.manifest = manifest
        self.rng = np.random.default_rng(seed)
        # The registry keeps the operator's selection.  With 'auto', every
        # instance resolves its own engine, so a later GPU set-up failure (for
        # example out of memory) still falls back to the CPU inside Brain
        # instead of being forced onto the engine the first instance resolved.
        self.brain = Brain(arrays=self.shared.arrays, validate=False, backend=selected)
        # Engine provenance: what was requested, what actually runs, and why
        # (Brain's note names a set-up fallback).  Only set-up may fall back; a
        # fault in a running brain propagates and halts, never switches engine.
        self.compute = dict(requested=selected, actual=self.brain.backend,
                            note=getattr(self.brain, 'backend_note', None))
        identity = self.shared.identity
        if (self.brain.dynamics == 'v3' and not identity.synthetic
                and 'v3-modulatory-only' not in identity.dataset):
            raise BackendError('v3 brains need the v3 transmitter-policy weights; load the graph with '
                               'SharedGraph.load_for_dynamics() (or select v1 explicitly)')
        self.step_index = 0
        self.world_state: dict = {}
        self.checkpoint_version = 0
        # Result-validity incidents of this run (append-only; written into every
        # checkpoint's meta and restored with it, so a restart cannot erase them).
        self.invalidity: list = []
        self.restore_fallback: Optional[dict] = None   # set when an older checkpoint was restored
        self.learning_state_restore: Optional[dict] = None
        self.learning_state_resets: list = []
        self.rule = None
        self.plastic_edges = np.zeros(0, dtype=np.int64)
        self.plastic_delta = np.zeros(0, dtype=np.float32)
        self._working_weight = None
        self.readout = None
        if backend == 'connectome-plastic':
            # Descriptors/edge arrays are shared; WP6 traces belong to this instance.
            self.rule = copy.copy(registry.plasticity_rule)
            if hasattr(self.rule, 'pre_trace'):
                self.rule.pre_trace = np.zeros_like(self.rule.pre_trace)
                self.rule.mod_trace = 0.0
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
        require_fixed_amd(self.registry.brain_backend, self.backend, learning=True)
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
        require_fixed_amd(self.registry.brain_backend, self.backend, learning=self.registry.learning_enabled)
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
        if self.rule is not None and hasattr(self.rule, 'pre_trace'):
            arrays['plastic.pre_trace'] = self.rule.pre_trace
            arrays['plastic.mod_trace'] = np.asarray(self.rule.mod_trace, dtype=np.float64)
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
                    graph_io=self.manifest.graph_io,
                    rule=self.rule.describe() if self.rule is not None else None,
                    compute=dict(self.compute),
                    readout_rate=self.readout.rate if self.readout is not None else None,
                    learning_state_format='neurofly.learning-state.v1',
                    learning_state_restore=self.learning_state_restore,
                    learning_state_resets=self.learning_state_resets)

    @staticmethod
    def _learning_payload(backend, rule, readout_size,
                          meta: dict, arrays: Dict[str, np.ndarray]) -> dict:
        """Stage learning state before mutating any live state.

        Unmarked historical payloads may omit both WP6 traces or the readout
        rate. Such omissions explicitly reset that state and leave provenance.
        Marked payloads are complete; missing or partial fields are corruption.
        """
        marker = meta.get('learning_state_format')
        if 'learning_state_format' in meta and marker != 'neurofly.learning-state.v1':
            raise IncompatibleCheckpoint('Unsupported checkpoint learning state format')
        staged = {'resets': []}

        def numeric(name, shape, dtype):
            value = arrays.get(name)
            if (not isinstance(value, np.ndarray) or value.shape != shape
                    or value.dtype.kind not in 'fiu' or not np.all(np.isfinite(value))):
                raise IncompatibleCheckpoint(f'Malformed checkpoint learning state: {name}')
            with np.errstate(over='ignore', invalid='ignore'):
                result = value.astype(dtype).copy()
            if not np.all(np.isfinite(result)):
                raise IncompatibleCheckpoint(f'Malformed checkpoint learning state: {name}')
            return result

        trace_names = ('plastic.pre_trace', 'plastic.mod_trace')
        has_traces = rule is not None and hasattr(rule, 'pre_trace')
        if backend == 'connectome-plastic':
            edges = arrays.get('plastic.edges')
            if (not isinstance(edges, np.ndarray) or edges.dtype.kind not in 'iu'
                    or not np.array_equal(edges, rule.edges)):
                raise IncompatibleCheckpoint('Checkpoint plastic edge subset differs from the declared rule')
            staged['delta'] = numeric('plastic.delta', rule.edges.shape, np.float32)
        if has_traces:
            present = [name in arrays for name in trace_names]
            if not any(present) and marker is None:
                staged['pre_trace'] = np.zeros_like(rule.pre_trace)
                staged['mod_trace'] = 0.0
                staged['resets'].extend(trace_names)
            else:
                staged['pre_trace'] = numeric(trace_names[0], rule.pre_trace.shape, np.float32)
                staged['mod_trace'] = float(numeric(trace_names[1], (), np.float64))
        elif any(name in arrays for name in trace_names):
            raise IncompatibleCheckpoint('Checkpoint traces unsupported by the requested backend/rule')
        if readout_size is not None:
            staged['weights'] = numeric('readout.weights', (readout_size,), np.float64)
            if 'readout_rate' not in meta and marker is None:
                from brainlab.learning import Readout
                staged['rate'] = Readout(0).rate
                staged['resets'].append('readout_rate')
            else:
                rate = meta.get('readout_rate')
                if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                    raise IncompatibleCheckpoint('Malformed checkpoint learning state: readout_rate')
                try:
                    rate = float(rate)
                except OverflowError as error:
                    raise IncompatibleCheckpoint('Malformed checkpoint learning state: readout_rate') from error
                if not np.isfinite(rate):
                    raise IncompatibleCheckpoint('Malformed checkpoint learning state: readout_rate')
                staged['rate'] = rate
        history = meta.get('learning_state_resets', [])
        if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
            raise IncompatibleCheckpoint('Malformed checkpoint learning state reset provenance')
        staged['history'] = [dict(item) for item in history]
        return staged

    def load_payload(self, meta: dict, arrays: Dict[str, np.ndarray]):
        if self.registry.brain_backend == 'wgpu-amd':
            require_fixed_payload(meta, arrays)
        self.registry._check_plasticity_semantics(self.manifest, meta, arrays)
        learning = self._learning_payload(self.backend, self.rule,
                                          self.shared.n if self.readout is not None else None,
                                          meta, arrays)
        state = {name: arrays[f'brain.{name}'] for name in STATE_ARRAYS}
        state.update(meta['brain_scalars'])
        if self.registry.brain_backend == 'wgpu-amd':
            # Version evidence comes from the already-validated saved manifest,
            # not inferred from array shapes or a fresh reinitialization.
            state['dynamics'] = self.manifest.dynamics['dynamics_version']
        self.brain.restore_state(state)
        self.rng = restore_rng(meta['rng_state'])
        self.step_index = int(meta['step_index'])
        self.world_state = meta.get('world_state') or {}
        self.invalidity = list((meta.get('result_validity') or {}).get('incidents') or [])
        self.learning_state_restore = dict(
            status='reset' if learning['resets'] else 'exact',
            reset_fields=learning['resets'], checkpoint_step=self.step_index,
            reason='historical_checkpoint_missing_learning_state' if learning['resets'] else None)
        self.learning_state_resets = learning['history']
        if learning['resets']:
            self.learning_state_resets.append(dict(self.learning_state_restore,
                                                  run_id=self.manifest.run_id))
        if self.backend == 'connectome-plastic':
            self.plastic_delta = learning['delta']
            if 'pre_trace' in learning:
                self.rule.pre_trace[:] = learning['pre_trace']
                self.rule.mod_trace = learning['mod_trace']
            self._materialize()
        if self.readout is not None:
            self.readout.weights = learning['weights']
            self.readout.rate = learning['rate']


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ExperimentRegistry:
    def __init__(self, shared: SharedGraph, root: Path = DEFAULT_REGISTRY_ROOT, *, test_mode: bool = False,
                 plasticity_rule=None, learning_enabled: Optional[bool] = None,
                 keep_checkpoints: Optional[int] = None, continue_io_state: bool = False,
                 brain_backend=None):
        self.brain_backend = compute_selector(brain_backend)
        if learning_enabled is None:
            learning_enabled = self.brain_backend != 'wgpu-amd'
        require_fixed_amd(self.brain_backend, 'connectome-fixed', learning=learning_enabled,
                          continue_io_state=continue_io_state)
        if self.brain_backend == 'wgpu-amd':
            if plasticity_rule is not None:
                require_fixed_amd(self.brain_backend, 'connectome-plastic')
            preflight_fixed_store(root)
        if shared.identity.synthetic and not test_mode:
            raise BackendError('A synthetic graph can only back a registry in explicit test mode')
        if plasticity_rule is not None and getattr(plasticity_rule, 'test_only', False) and not test_mode:
            raise BackendError('Test-only plasticity rules require test mode')
        self.shared = shared
        self.root = Path(root)
        self.test_mode = test_mode
        self.plasticity_rule = plasticity_rule
        self.learning_enabled = learning_enabled
        self.keep_checkpoints = resolve_keep_checkpoints(keep_checkpoints)
        self.continue_io_state = bool(continue_io_state)
        self.graph_io = graph_io_declaration()
        self.active: Optional[GraphInstance] = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self._read_index()
        self._recover_migrations()

    # -- index ----------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.root / 'registry.json'

    def _read_index(self) -> dict:
        if not self.index_path.exists():
            return dict(format=REGISTRY_FORMAT, graph_sha256=self.shared.identity.graph_sha256,
                        synthetic=self.shared.identity.synthetic, instances={})
        try:
            index = json.loads(self.index_path.read_text())
        except ValueError as error:
            raise IncompatibleCheckpoint(f'{self.index_path} is not valid JSON ({error}); it is empty or '
                                         f'truncated') from error
        if index.get('format') not in (LEGACY_REGISTRY_FORMAT, REGISTRY_FORMAT):
            raise IncompatibleCheckpoint(f'{self.index_path} is not a supported registry index')
        if index.get('graph_sha256') != self.shared.identity.graph_sha256:
            raise IncompatibleCheckpoint('Registry was created for a different graph; use a separate root')
        return index

    def _write_index(self):
        self.index['format'] = REGISTRY_FORMAT
        atomic_write_bytes(self.index_path, (json.dumps(self.index, indent=2) + '\n').encode())

    def _commit_index(self, candidate: dict) -> None:
        """Publish an index candidate without retaining an unacknowledged memory view."""
        previous = self.index
        self.index = candidate
        try:
            self._write_index()
            self._sync_dir(self.root)
        except Exception:
            self.index = previous
            raise

    @staticmethod
    def _selection_key(assay: str, backend: str) -> str:
        return f'{assay}|{backend}'

    def _direct_instance_dir(self, assay: str, backend: str, instance_id: str) -> Path:
        return self.root / assay / backend / instance_id

    def instance_dir(self, instance_id: str) -> Path:
        entry = self.index['instances'][instance_id]
        return self.root / entry['assay'] / entry['backend'] / instance_id

    def instance_id_for(self, assay: str, backend: str) -> str:
        """Return the explicitly selected compatible instance, creating or migrating if needed."""
        require_fixed_amd(self.brain_backend, backend, learning=self.learning_enabled)
        if self.brain_backend == 'wgpu-amd':
            preflight_fixed_store(self.root, assay)
        if assay not in PARADIGMS:
            raise ValueError(f'Unknown assay {assay!r}')
        if backend not in GRAPH_BACKENDS:
            raise BackendError(f'{backend!r} is not a graph backend; modular brains live in experiment_brains')
        if backend == 'connectome-plastic' and self.plasticity_rule is None:
            if not self.shared.identity.synthetic:
                from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
                self.plasticity_rule = VisualHeadingPlasticityRule.from_shared(self.shared)
            else:
                raise BackendError('connectome-plastic needs a declared plasticity rule (WP6); none is registered')
        # A previous call in this process may have published a child store but
        # failed before the index or intent completion was durably acknowledged.
        self._recover_migrations()
        key = self._selection_key(assay, backend)
        selected = (self.index.get('current_instances') or {}).get(key)
        candidates = [(instance_id, entry) for instance_id, entry in self.index['instances'].items()
                      if entry['assay'] == assay and entry['backend'] == backend]
        if selected is not None:
            if selected not in self.index['instances']:
                raise IncompatibleCheckpoint(f'Registry current selection {key} names missing instance {selected}')
            candidates = [(selected, self.index['instances'][selected])]
        if candidates:
            # A v1 index has no explicit selection and ordinarily has exactly one
            # instance per pair. If several exist, prefer the newest manifest.
            instance_id, _ = max(candidates, key=lambda item: self.manifest(item[0]).created_at)
            manifest = self.manifest(instance_id)
            if backend in ('connectome-plastic', 'connectome-with-trained-readout'):
                # Check before selecting, checkpointing/releasing another active
                # brain, or publishing a linked child. Never replace incompatible
                # learned state with a fresh instance.
                if self._versions_on_disk(instance_id) or (self.instance_dir(instance_id) / 'CURRENT.json').exists():
                    self._select_parent_checkpoint(instance_id)
                else:
                    self._check_plasticity_semantics(manifest, {})
            if self._io_matches(manifest.graph_io):
                if selected is None:
                    self.index.setdefault('current_instances', {})[key] = instance_id
                    self._write_index()
                    self._sync_dir(self.root)
                return instance_id
            if not self.continue_io_state:
                old = manifest.graph_io or {'version': 'unknown', 'sha256': 'unknown'}
                raise IOContinuationRequired(
                    f'Saved {assay}/{backend} instance {instance_id} uses graph I/O '
                    f'{old.get("version")}/{old.get("sha256")}; this daemon requires '
                    f'{self.graph_io["version"]}/{self.graph_io["sha256"]}. Refusing to alter the saved run. '
                    f'Restart with --continue-io-state to create a linked child run from a verified checkpoint.')
            return self._migrate_instance(instance_id)
        return self._create_instance(assay, backend)

    def _check_plasticity_semantics(self, manifest: RunManifest, meta: dict,
                                   arrays: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Use persisted v1 descriptors; software versions are not rule identity.

        One legacy source may be absent, but available sources must be complete,
        mutually consistent and equal to the requested rule. This is a refusal
        boundary, not authorization to migrate learning rules.
        """
        if manifest.backend != 'connectome-plastic':
            return
        if self.plasticity_rule is None:
            raise IncompatibleCheckpoint('Requested plasticity rule is missing')

        def semantics(value, source):
            if not isinstance(value, dict) or not value:
                raise IncompatibleCheckpoint(f'{source} plasticity rule evidence is malformed')
            # Citations and documentation locations do not change the rule.
            semantic = {key: item for key, item in value.items()
                        if key not in ('spec', 'primary_sources')}
            try:
                return json.dumps(semantic, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise IncompatibleCheckpoint(f'{source} plasticity rule evidence is malformed') from error

        requested = semantics(self.plasticity_rule.describe(), 'Requested')
        evidence = []
        declared = (manifest.dynamics or {}).get('plasticity_rule')
        for source, value in (('Manifest', declared), ('Checkpoint', meta.get('rule'))):
            if value is not None:
                evidence.append((source, semantics(value, source)))
        if not evidence:
            raise IncompatibleCheckpoint('Saved plasticity rule evidence is missing in both manifest and checkpoint')
        if len(evidence) == 2 and evidence[0][1] != evidence[1][1]:
            raise IncompatibleCheckpoint('Checkpoint plasticity rule contradicts the saved manifest')
        for source, saved in evidence:
            if saved != requested:
                old, new = json.loads(saved), json.loads(requested)
                fields = ', '.join(sorted(key for key in set(old) | set(new)
                                          if key not in old or key not in new or old[key] != new[key]))
                raise IncompatibleCheckpoint(
                    f'{source} plasticity rule differs from the requested rule ({fields}). '
                    'Saved state is retained; explicit linked rule migration is a separate operation.')
        if arrays is not None and not np.array_equal(arrays.get('plastic.edges'), self.plasticity_rule.edges):
            raise IncompatibleCheckpoint('Checkpoint plastic edge subset differs from the declared rule')

    def _create_instance(self, assay: str, backend: str) -> str:
        """Durably create a fresh instance whose manifest records the active I/O method."""
        require_fixed_amd(self.brain_backend, backend, learning=self.learning_enabled)
        instance_id = uuid.uuid4().hex
        seed = _derive_seed(assay, backend)
        manifest = RunManifest.create(
            backend=backend, assay=assay, instance_id=instance_id, seed=seed,
            graph=self.shared.identity.to_dict(), dynamics=active_dynamics(),
            learned_parameter_locations=self._learned_locations(backend, assay, instance_id),
            rng=np.random.default_rng(seed), test_mode=self.test_mode,
            graph_io=self.graph_io,
            source=source_revision(files=BACKENDS[backend].source_files))
        if backend == 'connectome-plastic':
            manifest.dynamics['plasticity_rule'] = self.plasticity_rule.describe()
        directory = self._direct_instance_dir(assay, backend, instance_id)
        manifest.write(directory / 'manifest.json')
        self.index['instances'][instance_id] = dict(assay=assay, backend=backend, seed=seed,
                                                    run_id=manifest.run_id)
        self.index.setdefault('current_instances', {})[self._selection_key(assay, backend)] = instance_id
        self._write_index()
        return instance_id

    def _io_matches(self, recorded: Optional[dict]) -> bool:
        return bool(recorded and recorded.get('version') == self.graph_io['version']
                    and recorded.get('sha256') == self.graph_io['sha256'])

    @staticmethod
    def _explicit_graph_io(recorded: Optional[dict]) -> Optional[dict]:
        if (isinstance(recorded, dict) and recorded.get('version')
                and recorded.get('sha256')):
            return {'version': recorded['version'], 'sha256': recorded['sha256']}
        return None

    def instance_id_for_run(self, run_id: str) -> str:
        """Resolve a historical run without changing the current selection."""
        for instance_id, entry in self.index['instances'].items():
            if entry.get('run_id') == run_id:
                return instance_id
        raise KeyError(run_id)

    @property
    def migration_intent_dir(self) -> Path:
        return self.root / '.migration-intents'

    def _migration_fault(self, stage: str) -> None:
        """Test seam for power-loss and publication-failure vectors."""

    @staticmethod
    def _sync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _sync_migrated_store(self, final: Path) -> None:
        """Make a renamed migration store durable before acknowledging it."""
        self._sync_dir(final / 'checkpoints')
        self._sync_dir(final)
        self._sync_dir(final.parent)

    def _write_intent(self, path: Path, intent: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._sync_dir(path.parent.parent)
        atomic_write_bytes(path, (json.dumps(intent, indent=2, allow_nan=False) + '\n').encode())
        self._sync_dir(path.parent)

    def _select_parent_checkpoint(self, instance_id: str):
        """Read-only selection using the accepted newest-valid fallback order."""
        directory = self.instance_dir(instance_id)
        parent_manifest = self.manifest(instance_id)
        entry = self.index['instances'][instance_id]
        for key, expected in (('instance_id', instance_id), ('assay', entry['assay']),
                              ('backend', entry['backend'])):
            if getattr(parent_manifest, key) != expected:
                raise IncompatibleCheckpoint(f'Parent manifest {key} does not match registry index')
        recorded_graph = parent_manifest.graph or {}
        effective_graph = self.shared.identity.to_dict()
        for key in ('graph_sha256', 'neuron_map_sha256', 'io_map_sha256'):
            if recorded_graph.get(key) != effective_graph.get(key):
                raise IncompatibleCheckpoint(f'Parent manifest {key} differs from the active graph')
        effective_dynamics = active_dynamics()
        mismatched_dynamics = [key for key, value in effective_dynamics.items()
                               if parent_manifest.dynamics.get(key) != value]
        if mismatched_dynamics:
            raise IncompatibleCheckpoint('Parent manifest dynamics differ from the active registry '
                                         f'({", ".join(mismatched_dynamics[:3])})')
        pointer = None
        pointer_problem = None
        try:
            pointer = self.current_pointer(instance_id)
        except (ValueError, OSError) as error:
            pointer_problem = f'CURRENT.json unreadable ({type(error).__name__}: {error})'
        versions = sorted(self._versions_on_disk(instance_id), reverse=True)
        if not versions:
            raise CheckpointCorrupt('explicit I/O continuation requires a verified saved checkpoint; none exists')
        hashes = self._recorded_hashes(instance_id)
        candidates = []
        if pointer is not None and pointer.get('version') is not None:
            candidates.append((int(pointer['version']), pointer.get('sha256')))
        candidates.extend((version, hashes.get(version)) for version in versions
                          if pointer is None or version != pointer.get('version'))
        skipped = []
        for version, expected in candidates:
            path = directory / 'checkpoints' / f'ckpt-{version:06d}.npz'
            if not expected:
                skipped.append({'version': version, 'reason': 'no recorded SHA-256 digest'})
                continue
            try:
                meta, arrays = load_checkpoint_file(path, expected_sha256=expected)
                self._check_meta(instance_id, path, meta)
                entry = self.index['instances'][instance_id]
                if meta.get('run_id') != parent_manifest.run_id:
                    raise IncompatibleCheckpoint(f'Checkpoint {path.name} run_id does not match parent manifest')
                if int(meta.get('seed')) != int(entry['seed']):
                    raise IncompatibleCheckpoint(f'Checkpoint {path.name} seed does not match parent instance')
            except (CheckpointCorrupt, IncompatibleCheckpoint, OSError, KeyError, ValueError) as error:
                skipped.append({'version': version, 'reason': f'{type(error).__name__}: {error}'})
                continue
            # A semantic mismatch must not silently fall back to an older rule.
            self._check_plasticity_semantics(parent_manifest, meta, arrays)
            GraphInstance._learning_payload(
                parent_manifest.backend,
                self.plasticity_rule if parent_manifest.backend == 'connectome-plastic' else None,
                self.shared.n if parent_manifest.backend == 'connectome-with-trained-readout' else None,
                meta, arrays)
            manifest_io = self._explicit_graph_io(parent_manifest.graph_io)
            checkpoint_io = self._explicit_graph_io(meta.get('graph_io'))
            if manifest_io is not None and checkpoint_io is not None and manifest_io != checkpoint_io:
                raise IncompatibleCheckpoint(
                    f'Checkpoint {path.name} graph_io contradicts the parent manifest')
            newest_step = max([int((pointer or {}).get('step_index') or 0)]
                              + list(self._recorded_steps(instance_id).values()))
            selected = {
                'version': version, 'step': int(meta['step_index']), 'sha256': expected,
                'file': path.name, 'fallback': bool(skipped), 'skipped': skipped,
                'problem': pointer_problem or (skipped[0]['reason'] if skipped else None),
                'newest_known_step': newest_step,
                'lost_steps': max(0, newest_step - int(meta['step_index'])) if newest_step else 0,
                'graph_io': checkpoint_io or {'version': 'unknown', 'sha256': 'unknown'},
            }
            return meta, arrays, selected
        raise CheckpointCorrupt(f'no saved checkpoint for linked I/O continuation verifies '
                                f'({len(candidates)} retained version(s) tried)')

    def _checkpoint_blob(self, instance: GraphInstance, version: int):
        arrays = instance.state_arrays()
        for name, array in arrays.items():
            if np.issubdtype(np.asarray(array).dtype, np.floating) and not np.isfinite(array).all():
                raise NonFiniteState(f'{name} contains NaN or inf; refusing to checkpoint broken state')
        meta = instance.meta()
        meta['version'] = version
        if getattr(instance, 'invalidity', None):
            meta['result_validity'] = {'state': 'incomplete', 'incidents': instance.invalidity}
        buffer = io.BytesIO()
        np.savez(buffer, meta=np.array(json.dumps(meta, allow_nan=False, default=str)), **arrays)
        return buffer.getvalue(), meta

    def _migrate_instance(self, parent_id: str) -> str:
        parent_manifest = self.manifest(parent_id)
        meta, arrays, selected = self._select_parent_checkpoint(parent_id)
        lineage_key = hashlib.sha256(
            f'{parent_manifest.run_id}|{parent_id}|{selected["sha256"]}|'
            f'{self.graph_io["sha256"]}'.encode()).hexdigest()
        for instance_id, entry in self.index['instances'].items():
            if entry.get('lineage_key') == lineage_key:
                candidate = copy.deepcopy(self.index)
                candidate.setdefault('current_instances', {})[
                    self._selection_key(entry['assay'], entry['backend'])] = instance_id
                self._commit_index(candidate)
                return instance_id

        child_id = uuid.uuid4().hex
        child_seed = int(meta['seed'])
        manifest_io = self._explicit_graph_io(parent_manifest.graph_io)
        checkpoint_io = self._explicit_graph_io(selected.get('graph_io'))
        unknown_io = {'version': 'unknown', 'sha256': 'unknown'}
        old_io = manifest_io or checkpoint_io or unknown_io
        inherited = []
        parent_validity = meta.get('result_validity')
        for incident in ((parent_validity or {}).get('incidents') or []):
            copied = dict(incident)
            copied.setdefault('source_run_id', parent_manifest.run_id)
            inherited.append(copied)
        if not parent_validity:
            inherited.append({'reason': 'parent_validity_unknown',
                              'source_run_id': parent_manifest.run_id,
                              'parent_checkpoint_version': selected['version']})
        elif parent_validity.get('state') != 'valid_so_far' and not inherited:
            inherited.append({'reason': 'parent_state_incomplete',
                              'source_run_id': parent_manifest.run_id,
                              'parent_checkpoint_version': selected['version']})
        if selected['fallback']:
            inherited.append({
                'reason': 'io_migration_restored_older_checkpoint',
                'source_run_id': parent_manifest.run_id,
                'restored_version': selected['version'], 'restored_step': selected['step'],
                'newest_known_step': selected['newest_known_step'],
                'lost_steps': selected['lost_steps'],
                'skipped_versions': [item['version'] for item in selected['skipped']],
            })
        method_interventions = [
            {'kind': 'remove_undisclosed_drive', 'name': name,
             'source_run_id': parent_manifest.run_id}
            for name in graph_io_declaration(include_config=True)['config']['removed_undisclosed_drive']
        ]
        inherited_interventions = []
        for intervention in parent_manifest.intervention_schedule:
            copied = dict(intervention)
            copied.setdefault('source_run_id', parent_manifest.run_id)
            inherited_interventions.append(copied)
        lineage = {
            'kind': 'graph_io_state_continuation', 'idempotence_key': lineage_key,
            'parent_run_id': parent_manifest.run_id, 'parent_instance_id': parent_id,
            'parent_checkpoint': selected, 'prior_graph_io': old_io,
            'prior_graph_io_sources': {
                'manifest': manifest_io or unknown_io,
                'checkpoint': checkpoint_io or unknown_io,
            },
            'new_graph_io': self.graph_io, 'state_continuation': 'explicit',
            'inherited_validity': inherited,
            'inherited_interventions': inherited_interventions,
            'interventions': method_interventions,
        }
        manifest = RunManifest.create(
            backend=parent_manifest.backend, assay=parent_manifest.assay,
            instance_id=child_id, seed=child_seed, graph=self.shared.identity.to_dict(),
            dynamics=parent_manifest.dynamics,
            learned_parameter_locations=self._learned_locations(parent_manifest.backend,
                                                                  parent_manifest.assay, child_id),
            rng=np.random.default_rng(child_seed), test_mode=self.test_mode,
            intervention_schedule=inherited_interventions + method_interventions,
            parent_run_id=parent_manifest.run_id, graph_io=self.graph_io, lineage=lineage,
            source=source_revision(files=BACKENDS[parent_manifest.backend].source_files))
        inherited = [dict(incident, run_id=manifest.run_id) for incident in inherited]
        lineage['inherited_validity'] = inherited
        manifest.lineage = lineage
        manifest.record_event('graph_io_state_continuation', step=selected['step'],
                              **{key: value for key, value in lineage.items() if key != 'kind'})

        child = GraphInstance(self, parent_manifest.assay, parent_manifest.backend,
                              child_id, child_seed, manifest)
        child.load_payload(meta, arrays)
        child.invalidity = inherited
        child.checkpoint_version = 1
        data, _ = self._checkpoint_blob(child, 1)
        pointer = {'version': 1, 'file': 'ckpt-000001.npz', 'sha256': _sha_bytes(data),
                   'bytes': len(data), 'step_index': child.step_index, 'format': CHECKPOINT_FORMAT}

        staging = self.root / '.migration-staging' / f'{lineage_key}.{uuid.uuid4().hex[:8]}.partial'
        final = self._direct_instance_dir(parent_manifest.assay, parent_manifest.backend, child_id)
        intent_path = self.migration_intent_dir / f'{lineage_key}.json'
        intent = {
            'format': 'neurofly.graph-io-migration.v1', 'phase': 'prepared',
            'idempotence_key': lineage_key, 'parent_instance_id': parent_id,
            'child_instance_id': child_id, 'child_run_id': manifest.run_id,
            'assay': parent_manifest.assay, 'backend': parent_manifest.backend,
            'seed': child_seed, 'target_graph_io': self.graph_io,
            'staging': str(staging.relative_to(self.root)), 'final': str(final.relative_to(self.root)),
            'checkpoint': pointer,
        }
        self._write_intent(intent_path, intent)
        manifest.write(staging / 'manifest.json')
        atomic_write_bytes(staging / 'checkpoints' / pointer['file'], data)
        atomic_write_bytes(staging / 'CURRENT.json', (json.dumps(pointer, indent=2) + '\n').encode())
        event = {'kind': 'checkpoint', 'instance_id': child_id, 'version': 1,
                 'step': child.step_index, 'sha256': pointer['sha256'],
                 'migration_parent_run_id': parent_manifest.run_id}
        atomic_write_bytes(staging / 'events.jsonl', (json.dumps(event) + '\n').encode())
        self._migration_fault('child_payload_write')
        staged_manifest = RunManifest.read(staging / 'manifest.json')
        verified_meta, _ = load_checkpoint_file(staging / 'checkpoints' / pointer['file'],
                                                expected_sha256=pointer['sha256'])
        if (staged_manifest.run_id != manifest.run_id or staged_manifest.instance_id != child_id
                or not self._io_matches(staged_manifest.graph_io)
                or (staged_manifest.lineage or {}).get('idempotence_key') != lineage_key):
            raise IncompatibleCheckpoint('staged child manifest does not match the migration intent')
        for key, expected in (('instance_id', child_id), ('run_id', manifest.run_id),
                              ('assay', parent_manifest.assay), ('backend', parent_manifest.backend),
                              ('graph_sha256', self.shared.identity.graph_sha256)):
            if verified_meta.get(key) != expected:
                raise IncompatibleCheckpoint(f'staged child checkpoint {key} mismatch')
        self._sync_dir(staging / 'checkpoints')
        self._sync_dir(staging)
        self._sync_dir(staging.parent)

        self._migration_fault('store_publication')
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        self._sync_migrated_store(final)
        intent['phase'] = 'store_published'
        self._write_intent(intent_path, intent)

        self._migration_fault('index_publication')
        candidate = copy.deepcopy(self.index)
        candidate['instances'][child_id] = {
            'assay': parent_manifest.assay, 'backend': parent_manifest.backend,
            'seed': child_seed, 'run_id': manifest.run_id, 'lineage_key': lineage_key,
            'parent_instance_id': parent_id,
        }
        candidate.setdefault('current_instances', {})[
            self._selection_key(parent_manifest.assay, parent_manifest.backend)] = child_id
        self._commit_index(candidate)
        intent['phase'] = 'complete'
        self._write_intent(intent_path, intent)
        return child_id

    def _recover_migrations(self) -> None:
        """Publish only complete, verified child stores left after index publication failure."""
        if self.brain_backend == 'wgpu-amd':
            return  # fixed AMD restore never publishes a graph-I/O conversion
        if not self.migration_intent_dir.exists():
            return
        recoverable = []
        for path in sorted(self.migration_intent_dir.glob('*.json')):
            try:
                intent = json.loads(path.read_text())
                if intent.get('format') != 'neurofly.graph-io-migration.v1':
                    continue
                if intent.get('phase') == 'complete':
                    continue
                if not self._io_matches(intent.get('target_graph_io')):
                    continue
                child_id = intent['child_instance_id']
                final = self.root / intent['final']
                if not final.is_dir():
                    continue
                manifest = RunManifest.read(final / 'manifest.json')
                pointer = json.loads((final / 'CURRENT.json').read_text())
                expected_pointer = intent['checkpoint']
                if (pointer != expected_pointer or manifest.instance_id != child_id
                        or manifest.run_id != intent['child_run_id']
                        or (manifest.lineage or {}).get('idempotence_key') != intent['idempotence_key']
                        or (manifest.lineage or {}).get('parent_instance_id') != intent['parent_instance_id']):
                    continue
                meta, _ = load_checkpoint_file(final / 'checkpoints' / pointer['file'],
                                               expected_sha256=pointer['sha256'])
                if (meta.get('instance_id') != child_id or meta.get('run_id') != manifest.run_id
                        or meta.get('graph_sha256') != self.shared.identity.graph_sha256
                        or int(meta.get('seed')) != int(intent['seed'])
                        or not self._io_matches(manifest.graph_io)):
                    continue
                # A visible child may be the result of a rename whose containing
                # directory fsync failed.  Verification proves content identity,
                # not publication durability, so recovery must retry the store
                # syncs before it can select or acknowledge this child.
                self._sync_migrated_store(final)
                entry = {'assay': intent['assay'], 'backend': intent['backend'],
                         'seed': int(intent['seed']), 'run_id': manifest.run_id,
                         'lineage_key': intent['idempotence_key'],
                         'parent_instance_id': intent['parent_instance_id']}
                recoverable.append((path, intent, child_id, entry))
            except (ValueError, KeyError, BackendError, IncompatibleCheckpoint, CheckpointCorrupt):
                continue
        by_selection = {}
        for item in recoverable:
            intent = item[1]
            key = self._selection_key(intent['assay'], intent['backend'])
            by_selection.setdefault(key, []).append(item)
        for key, items in by_selection.items():
            if len(items) > 1:
                raise IncompatibleCheckpoint(
                    f'ambiguous pending graph I/O migrations for {key}: {len(items)} verified children')
            path, intent, child_id, entry = items[0]
            selected = (self.index.get('current_instances') or {}).get(key)
            if selected not in (None, intent['parent_instance_id'], child_id):
                raise IncompatibleCheckpoint(
                    f'pending graph I/O migration for {key} expected current instance '
                    f'{intent["parent_instance_id"]}, but registry selects {selected}')
            if selected != child_id:
                candidate = copy.deepcopy(self.index)
                candidate['instances'][child_id] = entry
                candidate.setdefault('current_instances', {})[key] = child_id
                self._commit_index(candidate)
            else:
                recorded = self.index['instances'].get(child_id)
                if recorded != entry:
                    raise IncompatibleCheckpoint(
                        f'pending graph I/O migration child {child_id} contradicts the registry index')
            intent['phase'] = 'complete'
            self._write_intent(path, intent)

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
        """Restore the target, then checkpoint, commit and release the source.

        Returns only once the target's brain snapshot (and stored world state)
        is restored, so callers may acknowledge the switch on return.
        """
        prepared = self.prepare_activation(assay, backend)
        try:
            # A source whose arrays were already released was checkpointed before
            # release (see ``GraphInstance.release``); it has no state to save.
            if (self.active is not None and prepared['target'] is not self.active
                    and self.active.brain is not None):
                self.checkpoint()
            target = self.commit_activation(prepared)
            self.release_activation_source(prepared)
            return target
        except Exception:
            self.cancel_activation(prepared)
            raise

    def prepare_activation(self, assay: str, backend: str, *, force_reload=False) -> dict:
        """Restore the actual prospective instance while retaining the usable source.

        Selection/migration/restore evidence may write the target store. Callers
        must run this off their simulation lock and retain those writes on cancel.
        """
        source = self.active
        instance_id = self.instance_id_for(assay, backend)
        # A released active instance is unusable (its brain is gone), so asking
        # for it again restores it from its checkpoint instead of returning it.
        released = source is not None and source.brain is None
        if (source is not None and source.instance_id == instance_id and not force_reload
                and not released):
            return {'source': source, 'target': source}
        entry = self.index['instances'][instance_id]
        manifest = self.manifest(instance_id)
        instance = GraphInstance(self, assay, backend, instance_id, entry['seed'], manifest)
        try:
            self._restore(instance)
        except Exception:
            instance.release()
            raise
        return {'source': source, 'target': instance}

    def commit_activation(self, prepared: dict) -> GraphInstance:
        """No disk waits: compare the source before publishing the prepared target."""
        if self.active is not prepared['source']:
            raise RuntimeError('Activation source changed while preparing target')
        target = prepared['target']
        if target.brain is None:
            raise RuntimeError('Prepared activation was canceled')
        self.active = target
        return target

    @staticmethod
    def release_activation_source(prepared: dict) -> None:
        """Release retired mutable arrays outside the caller's simulation lock."""
        if prepared['source'] is not None and prepared['source'] is not prepared['target']:
            prepared['source'].release()

    def cancel_activation(self, prepared: dict) -> None:
        if prepared['target'] is not prepared['source'] and self.active is not prepared['target']:
            prepared['target'].release()

    def _restore(self, instance: 'GraphInstance') -> None:
        """Restore the newest checkpoint that verifies (audit F, F6).

        The version CURRENT.json names is tried first.  If CURRENT.json is
        unreadable, or that file is missing, corrupt or incompatible, the retained
        versions are tried newest to oldest (each checked against the hash its
        ``checkpoint`` event recorded, when there is one).  The first that verifies
        is restored and a ``restore`` event names the skipped versions.  Damaged
        files are left in place.  Refuses only when none verifies.
        """
        instance_id = instance.instance_id
        pointer_path = self.instance_dir(instance_id) / 'CURRENT.json'
        problem = None
        if not pointer_path.exists():
            if not self._versions_on_disk(instance_id):
                return                                      # a new instance: fresh start
            # An established instance without its pointer: never start it fresh (that
            # would also overwrite ckpt-000001...); recover the newest valid version.
            current, problem = None, 'CURRENT.json missing although retained checkpoints exist'
        else:
            try:
                current = self.current_pointer(instance_id)
            except (ValueError, OSError) as error:
                current, problem = None, f'CURRENT.json unreadable ({type(error).__name__}: {error})'
        if current is not None:
            try:
                meta, arrays = self.read_checkpoint(instance_id)
            except (CheckpointCorrupt, IncompatibleCheckpoint, OSError, KeyError, ValueError) as error:
                problem = f'version {current.get("version")}: {type(error).__name__}: {error}'
            else:
                instance.load_payload(meta, arrays)
                instance.checkpoint_version = current['version']
                self._event(instance, 'restore', version=current['version'], step=instance.step_index,
                            learning_state=instance.learning_state_restore)
                return
        directory = self.instance_dir(instance_id) / 'checkpoints'
        versions = []
        for path in directory.glob('ckpt-*.npz'):
            try:
                versions.append((int(path.stem[len('ckpt-'):]), path))
            except ValueError:
                continue
        hashes = self._recorded_hashes(instance_id)
        skipped = [{'version': current.get('version'), 'reason': problem}] if current is not None else []
        for version, path in sorted(versions, reverse=True):
            if current is not None and version == current.get('version'):
                continue
            try:
                meta, arrays = load_checkpoint_file(path, expected_sha256=hashes.get(version))
                self._check_meta(instance_id, path, meta)
            except (CheckpointCorrupt, IncompatibleCheckpoint, OSError, KeyError, ValueError) as error:
                skipped.append({'version': version, 'reason': f'{type(error).__name__}: {error}'})
                continue
            instance.load_payload(meta, arrays)
            # New checkpoints are numbered above every file on disk, so the damaged
            # ones are never overwritten (retention prunes them later, as usual).
            highest = max([v for v, _ in versions] + [int((current or {}).get('version') or 0)])
            instance.checkpoint_version = highest
            self._event(instance, 'restore', version=version, step=instance.step_index, fallback=True,
                        problem=problem, skipped=skipped, sha256_checked=version in hashes,
                        learning_state=instance.learning_state_restore)
            # Committed steps after the restored version are lost: the run's result is
            # INCOMPLETE (the daemon records it as an immutable incident; nothing is reset).
            newest_step = max([int((current or {}).get('step_index') or 0)]
                              + list(self._recorded_steps(instance_id).values()))
            instance.restore_fallback = {
                'problem': problem, 'restored_version': version, 'restored_step': instance.step_index,
                'newest_version': highest, 'newest_known_step': newest_step,
                'lost_steps': max(0, newest_step - instance.step_index) if newest_step else None,
                'skipped_versions': [s.get('version') for s in skipped]}
            print(f'[Registry] {instance.assay}/{instance.backend}: {problem}; restored the newest checkpoint '
                  f'that verifies (version {version}, step {instance.step_index}); skipped '
                  f'{len(skipped)} newer version(s).', file=sys.stderr, flush=True)
            return
        raise CheckpointCorrupt(f'no saved checkpoint of {instance.assay}/{instance.backend} verifies '
                                f'({problem}; {len(versions)} retained version(s) tried) in '
                                f'{self.instance_dir(instance_id)}')

    def _recorded_steps(self, instance_id: str) -> dict:
        """version -> step_index from the instance's ``checkpoint`` events (best effort)."""
        steps = {}
        try:
            lines = (self.instance_dir(instance_id) / 'events.jsonl').read_text().splitlines()
        except OSError:
            return steps
        for line in lines:
            try:
                event = json.loads(line)
                if event.get('kind') == 'checkpoint':
                    steps[int(event['version'])] = int(event['step'])
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        return steps

    def _versions_on_disk(self, instance_id: str) -> list:
        """Every ckpt-NNNNNN.npz version number present for the instance (any state)."""
        versions = []
        for path in (self.instance_dir(instance_id) / 'checkpoints').glob('ckpt-*.npz'):
            try:
                versions.append(int(path.stem[len('ckpt-'):]))
            except ValueError:
                continue
        return versions

    def _recorded_hashes(self, instance_id: str) -> dict:
        """version -> sha256 from the instance's ``checkpoint`` events (best effort)."""
        hashes = {}
        path = self.instance_dir(instance_id) / 'events.jsonl'
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return hashes
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('kind') == 'checkpoint' and event.get('sha256') and event.get('version') is not None:
                hashes[int(event['version'])] = event['sha256']
        return hashes

    def _check_meta(self, instance_id: str, path: Path, meta: dict) -> None:
        entry = self.index['instances'][instance_id]
        for key, expected in (('backend', entry['backend']), ('assay', entry['assay']),
                              ('instance_id', instance_id), ('graph_sha256', self.shared.identity.graph_sha256),
                              ('io_map_sha256', self.shared.identity.io_map_sha256)):
            if meta.get(key) != expected:
                raise IncompatibleCheckpoint(f'Checkpoint {path.name} {key}={meta.get(key)!r}, expected {expected!r}')

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
        # Always above every version on disk: a checkpoint never replaces retained state.
        version = max([instance.checkpoint_version] + self._versions_on_disk(instance.instance_id)) + 1
        arrays = instance.state_arrays()     # device-to-host copy for a GPU brain
        for name, array in arrays.items():
            if np.issubdtype(np.asarray(array).dtype, np.floating) and not np.isfinite(array).all():
                raise NonFiniteState(f'{name} contains NaN or inf; refusing to checkpoint a broken state '
                                     f'over the last good one')
        buffer = io.BytesIO()
        meta = instance.meta()
        meta['version'] = version
        if getattr(instance, 'invalidity', None):
            meta['result_validity'] = {'state': 'incomplete', 'incidents': instance.invalidity}
        np.savez(buffer, meta=np.array(json.dumps(meta, allow_nan=False, default=str)), **arrays)
        data = buffer.getvalue()
        path = directory / f'ckpt-{version:06d}.npz'
        atomic_write_bytes(path, data)
        pointer = dict(version=version, file=path.name, sha256=_sha_bytes(data), bytes=len(data),
                       step_index=instance.step_index, format=CHECKPOINT_FORMAT)
        atomic_write_bytes(directory.parent / 'CURRENT.json', (json.dumps(pointer, indent=2) + '\n').encode())
        instance.checkpoint_version = version
        self._event(instance, 'checkpoint', version=version, step=instance.step_index, sha256=pointer['sha256'])
        self.prune_checkpoints(instance.instance_id)
        return path

    def prune_checkpoints(self, instance_id: str) -> list:
        """Delete all but the newest ``keep_checkpoints`` versions of one instance.

        Runs only after ``CURRENT.json`` is published, and never deletes the
        version it names, so resume behaviour is unchanged.  Returns the removed paths.
        """
        if not self.keep_checkpoints:
            return []
        pointer = self.current_pointer(instance_id)
        if pointer is None:
            return []
        directory = self.instance_dir(instance_id) / 'checkpoints'
        versions = []
        for path in directory.glob('ckpt-*.npz'):
            try:
                versions.append((int(path.stem[len('ckpt-'):]), path))
            except ValueError:
                continue
        # Only versions up to the published one: a higher number is an unpublished
        # write that the next checkpoint overwrites.
        published = sorted((v for v in versions if v[0] <= pointer['version']), reverse=True)
        removed = []
        for version, path in published[self.keep_checkpoints:]:
            if version == pointer['version']:
                continue
            try:
                path.unlink()
                removed.append(path)
            except FileNotFoundError:
                pass
        return removed

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
