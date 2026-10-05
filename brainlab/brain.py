"""Direct current input and spike output; no sensory or motor policy.

The dynamics are explicitly versioned (``docs/LIF_DYNAMICS_SPEC.md``).  ``v1``
is the current-based proxy every WP1-WP5 result was produced under; ``v2`` is the conductance-based model with reversal potentials and
one conductance quantum for both signs; ``v3`` is the same conductance model
with the declared per-sign PSP-preserving calibration and is the default.
``v4`` is v3 plus the declared HYBRID graded mode (§7): cell classes listed as
non-spiking in ``brainlab.graded_policy`` are integrated as passive membranes
with no threshold, reset or refractory period, and their subthreshold membrane
potential drives downstream conductance directly, so spiking and graded
populations interoperate on one weight array.  v4 is NOT the default; select it
per instance with ``Brain(..., dynamics='v4')``, choose its class list with
``graded_policy=`` (a declared policy name, or an explicit mask for tests), and
read a graded cell's output with ``release_rate_hz()`` and ``v`` rather than
``counts`` -- a graded cell never spikes.
Select another per instance with ``Brain(..., dynamics='v1')`` or process-wide
with ``NEUROFLY_LIF_DYNAMICS=v1``.

v3 is defined together with its transmitter policy (aminergic neurons carry no
fast weight).  A v3 Brain loaded from a graph *path* applies it when the
MaleCNS transmitter table matches the graph; callers passing ``arrays`` supply
v3 weights themselves (``SharedGraph.load_for_dynamics``).

``backend`` selects where v3 runs: ``'cuda'`` (NVIDIA GPU via
``brainlab.cuda_engine``), ``'cpu'`` (the numba reference kernel) or ``'auto'``,
the default, which uses the GPU for v3 when a CUDA device is usable and the CPU
otherwise.  ``NEUROFLY_BRAIN_BACKEND`` overrides the default process-wide.  The
GPU matches the CPU within the model's own float-rounding sensitivity
(``scripts/gpu_parity.py``); v1 and v2 always run on the CPU.  They are not interchangeable and their
checkpoints are mutually refused: v1's synaptic state array has a different
shape, and every snapshot carries its dynamics version explicitly.
"""
import logging
import math
import os
import time
import numpy as np
from .engine import (E_INH_MV, G_UNIT_EXC_V3, V_REST_MV, advance, advance_v2,
                     advance_v3, advance_v4, advance_v5)
from .graph_identity import (DYNAMICS_VERSIONS, E_INH_ENV, E_INH_VARIANT_VERSIONS,
                             active_dynamics_version, active_e_inh_mV)


MALECNS_NEURONS = 166_700
log = logging.getLogger('brainlab')
_announced = set()

GRAPH_ARRAYS = (('ptr', np.int64), ('post', np.int32), ('weight', np.float32), ('ids', np.int64))
# Every mutable array of the LIF state; together with the scalars below this is
# a complete restartable brain snapshot (the graph arrays are immutable).
STATE_ARRAYS = ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts',
                'active', 'active_flag', 'nactive')
# v4 only: the delayed graded-release ring buffer.  Its presence is what makes a
# v4 snapshot non-interchangeable with a v3 one even by shape.
STATE_ARRAYS_V4 = STATE_ARRAYS + ('rel_ring',)
STATE_SCALARS = ('cursor', 'total_spikes', 'sim_ms')
# Versions with the v4 hybrid graded/spiking mode.  v5 is v4 plus declared
# per-receptor-class synaptic kinetics (docs/LIF_DYNAMICS_SPEC.md §9).
GRADED_VERSIONS = ('v4', 'v5')


def resolve_backend(dynamics: str, backend=None) -> str:
    """Where a Brain with these dynamics runs: 'cuda' or 'cpu' (or an explicit request).

    ``backend`` (else ``NEUROFLY_BRAIN_BACKEND``, else ``'auto'``).  ``'auto'``
    uses the GPU when a usable CUDA build exists for the dynamics: CuPy for the
    graded versions v4/v5, numba.cuda or CuPy for v3; v1/v2 always run on the CPU.
    """
    requested = backend or os.environ.get('NEUROFLY_BRAIN_BACKEND', 'auto')
    if requested == 'auto':
        if dynamics in GRADED_VERSIONS:
            from .cupy_v4 import cupy_available
            requested = 'cuda' if cupy_available() else 'cpu'
        else:
            from .cuda_engine import cuda_available
            requested = 'cuda' if dynamics == 'v3' and cuda_available() else 'cpu'
    return requested


_GPU_NAME = None


def gpu_name():
    """Name of the CUDA device a 'cuda' Brain uses (device 0 of the visible set), or None.

    Queries device properties only; it does not allocate on the device.  Cached.
    """
    global _GPU_NAME
    if _GPU_NAME is None:
        name = None
        try:
            import cupy
            name = cupy.cuda.runtime.getDeviceProperties(0)['name']
        except Exception:
            try:
                from numba import cuda
                devices = cuda.list_devices()
                name = devices[0].name if len(devices) else None
            except Exception:
                name = None
        if isinstance(name, bytes):
            name = name.decode('utf-8', 'replace')
        _GPU_NAME = name or ''
    return _GPU_NAME or None


def _v3_policy_weight(arrays: dict) -> np.ndarray:
    """The v3 fast weights for a graph loaded from a path.

    Applies the declared v3 transmitter policy when the MaleCNS transmitter
    table matches the graph's neuron count.  Graphs without one (synthetic
    test graphs) keep their weights; the real graph without its table is
    refused rather than silently run on v1 weights.
    """
    from .graph_identity import GraphUnavailable
    from .transmitter_policy import apply_policy, load_transmitters
    n = len(arrays['ids'])
    try:
        labels = load_transmitters()
    except (FileNotFoundError, OSError, GraphUnavailable, ImportError):
        labels = None
    if labels is not None and len(labels) == n:
        weight, _ = apply_policy(arrays['ptr'], arrays['post'], arrays['weight'], labels)
        return weight
    if n == MALECNS_NEURONS:
        raise ValueError('v3 needs the MaleCNS transmitter table (connectome_data/.../neurons.feather) '
                         'to apply its transmitter policy; pass dynamics="v1" to run the pinned weights')
    return arrays['weight']


class Brain:
    def _state_arrays(self):
        return STATE_ARRAYS_V4 if self.dynamics in GRADED_VERSIONS else STATE_ARRAYS

    def __init__(self, path=None, *, arrays=None, validate=True, dynamics=None, e_inh_mV=None,
                 backend=None, graded_policy=None, kinetics=None, receptor_classes=None):
        """Load a CSR graph from ``path`` (file or file-like) or reuse ``arrays``.

        ``arrays`` lets several instances share one immutable graph; pass
        ``validate=False`` only for arrays already validated by another Brain.
        ``dynamics`` selects the declared LIF version ('v1', 'v2' or 'v3');
        the default comes from ``NEUROFLY_LIF_DYNAMICS`` and is 'v3'.
        ``e_inh_mV`` overrides the inhibitory reversal potential and exists
        only for the declared sensitivity arms of
        ``docs/LIF_DYNAMICS_SPEC.md``.  Under v3 the inhibitory conductance
        quantum follows it, because the v3 calibration DERIVES that quantum
        from the driving force at rest: ``g_inh = 1/(V_rest - E_inh)``.

        v5 only: ``kinetics`` names a declared receptor-kinetics table
        (``brainlab.receptor_kinetics``; default the primary) and
        ``receptor_classes`` optionally gives each neuron's receptor class
        explicitly (synthetic test graphs).  On the real graph the classes are
        resolved from the released transmitter table.
        """
        self._gpu = None
        self.dynamics = dynamics or active_dynamics_version()
        if self.dynamics not in DYNAMICS_VERSIONS:
            raise ValueError(f'Unknown dynamics version {self.dynamics!r}; '
                             f'declared: {sorted(DYNAMICS_VERSIONS)}')
        if e_inh_mV is None:
            # Declared sensitivity sweep (docs/EINH_SENSITIVITY.md): a
            # process-wide override, so a script that does not build the Brain
            # itself can still select a declared E_inh variant.  Unset means
            # the declared v2/v3 value, so nothing changes by default.
            e_inh_mV = active_e_inh_mV()
            if e_inh_mV is not None and self.dynamics not in E_INH_VARIANT_VERSIONS:
                raise ValueError(f'{E_INH_ENV} selects a declared variant of '
                                 f'{E_INH_VARIANT_VERSIONS} only; dynamics is '
                                 f'{self.dynamics!r}')
        if e_inh_mV is not None and self.dynamics == 'v1':
            raise ValueError('e_inh_mV applies only to the conductance-based dynamics (v2, v3)')
        self.e_inh_mV = float(E_INH_MV if e_inh_mV is None else e_inh_mV)
        if self.e_inh_mV >= V_REST_MV:
            raise ValueError('e_inh_mV must be below V_rest for the v3 calibration to be finite')
        # Declared, derived, not fitted (docs/LIF_DYNAMICS_SPEC.md §4.2).
        self.g_unit_exc = float(G_UNIT_EXC_V3)
        self.g_unit_inh = float(1.0 / (V_REST_MV - self.e_inh_mV))
        if (path is None) == (arrays is None):
            raise ValueError('Provide exactly one of path or arrays')
        if arrays is None:
            with np.load(path, allow_pickle=False) as graph:
                arrays = {name: graph[name] for name, _ in GRAPH_ARRAYS}
            if self.dynamics in ('v3', 'v4', 'v5'):
                arrays['weight'] = _v3_policy_weight(arrays)
        for name, dtype in GRAPH_ARRAYS:
            value = arrays[name]
            if value.ndim != 1 or value.dtype != dtype or not value.flags.c_contiguous:
                raise ValueError(f'Invalid graph array: {name}')
            setattr(self, name, value)
        self.n = len(self.ids)
        if validate and (not self.n or self.ptr.shape != (self.n + 1,) or self.ptr[0] != 0
                or self.ptr[-1] != len(self.post) or np.any(np.diff(self.ptr) < 0)
                or len(self.weight) != len(self.post) or not np.isfinite(self.weight).all()
                or np.any(self.post < 0) or np.any(self.post >= self.n)
                or len(np.unique(self.ids)) != self.n):
            raise ValueError('Invalid CSR graph')
        self.dt = .1
        self.cursor = 0
        self.v = np.full(self.n, -52, dtype=np.float32)
        # v1: one current-like synaptic variable per neuron.
        # v2: (2, n) conductances, row 0 excitatory, row 1 inhibitory.  The
        # differing shape is what makes checkpoints mutually incompatible.
        # v5: (2K, n) for K kinetic channels (row 2k excitatory, 2k+1 inhibitory).
        self.kinetics_report = None
        if self.dynamics == 'v5':
            from . import receptor_kinetics as _rk
            self._kin = _rk.resolve(self.n, kinetics=kinetics, receptor_classes=receptor_classes,
                                    dt=.1, required=(self.n == MALECNS_NEURONS))
            self.kinetics_report = self._kin['report']
            g_rows = 2 * len(self._kin['chan_decay'])
        elif kinetics is not None or receptor_classes is not None:
            raise ValueError('kinetics / receptor_classes apply to LIF dynamics v5 only')
        else:
            g_rows = 2
        self.g = np.zeros(self.n if self.dynamics == 'v1' else (g_rows, self.n), dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.queue = np.zeros((19, self.n), dtype=np.int32)
        self.queue_count = np.zeros(19, dtype=np.int32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        self.total_spikes = 0
        self.sim_ms = 0.
        # v4: resolve the DECLARED graded (non-spiking) cell classes, hash the
        # resolved node set, and allocate the delayed-release ring buffer.
        self.graded = np.zeros(self.n, dtype=np.uint8)
        self.graded_idx = np.zeros(0, dtype=np.int64)
        self.graded_report = None
        self.rel_ring = np.zeros((0, 0), dtype=np.float32)
        if self.dynamics in GRADED_VERSIONS:
            from . import graded_policy as _gp
            if graded_policy is not None and not isinstance(graded_policy, str):
                # An explicit mask, for synthetic test graphs and unit probes.
                self.graded = np.ascontiguousarray(graded_policy, dtype=np.uint8)
                if self.graded.shape != (self.n,):
                    raise ValueError('An explicit graded mask needs one entry per neuron')
                self.graded_report = dict(policy='explicit-mask',
                                          neurons=self.n,
                                          graded_neurons=int(self.graded.sum()),
                                          graded_set_sha256=_gp.graded_set_sha256(self.graded),
                                          note='explicit mask, not a declared class list')
            else:
                policy = graded_policy or os.environ.get('NEUROFLY_GRADED_POLICY') or _gp.POLICY_PRIMARY
                self.graded, self.graded_report = _gp.resolve(
                    self.n, policy=policy, required=(self.n == MALECNS_NEURONS))
            self.graded_idx = np.ascontiguousarray(np.flatnonzero(self.graded), dtype=np.int64)
            self.rel_ring = np.zeros(self.queue.shape, dtype=np.float32)
        requested = backend or os.environ.get('NEUROFLY_BRAIN_BACKEND', 'auto')
        self.backend = resolve_backend(self.dynamics, backend)
        self.backend_note = 'requested' if requested != 'auto' else 'auto'
        try:
            self._setup_device()
        except Exception as exc:
            if requested != 'auto' or self.backend != 'cuda':
                raise
            # 'auto' picked a GPU that passed the probe but failed on the real
            # graph: run on the CPU and say why, never crash or half-initialise.
            reason = f'{type(exc).__name__}: {exc}'.splitlines()[0][:300]
            log.warning('brainlab: GPU set-up failed, LIF %s falls back to the CPU (%s)', self.dynamics, reason)
            self._gpu = None
            self.backend = 'cpu'
            self.backend_note = f'auto; GPU set-up failed ({reason})'
        if self.backend not in _announced:
            _announced.add(self.backend)
            log.info('brainlab: LIF %s running on the %s backend', self.dynamics, self.backend.upper())

    def _setup_device(self):
        if self.backend == 'cuda' and self.dynamics == 'v5':
            from .cupy_v5 import CupyV5State
            self._gpu = CupyV5State(self.ptr, self.post, self.weight, n=self.n,
                                    e_inh=self.e_inh_mV, g_unit_exc=self.g_unit_exc,
                                    g_unit_inh=self.g_unit_inh, dt=self.dt,
                                    delay_slots=self.queue.shape[0], graded=self.graded,
                                    pre_chan=self._kin['pre_chan'],
                                    chan_decay=self._kin['chan_decay'],
                                    chan_q=self._kin['chan_q'])
            self._gpu.upload_state(self.v, self.g, self.refractory, self.queue,
                                   self.queue_count, self.counts, self.active_flag,
                                   self.rel_ring)
            self._refresh_weight_view()
        elif self.backend == 'cuda' and self.dynamics == 'v4':
            from .cupy_v4 import CupyV4State
            self._gpu = CupyV4State(self.ptr, self.post, self.weight, n=self.n,
                                    e_inh=self.e_inh_mV, g_unit_exc=self.g_unit_exc,
                                    g_unit_inh=self.g_unit_inh, dt=self.dt,
                                    delay_slots=self.queue.shape[0], graded=self.graded)
            self._gpu.upload_state(self.v, self.g, self.refractory, self.queue,
                                   self.queue_count, self.counts, self.active_flag,
                                   self.rel_ring)
            self._refresh_weight_view()
        elif self.backend == 'cuda':
            if self.dynamics != 'v3':
                raise ValueError('The CUDA backend implements LIF dynamics v3, v4 and v5 only')
            from .cuda_engine import make_state
            self._gpu = make_state(self.ptr, self.post, self.weight, n=self.n,
                                    e_inh=self.e_inh_mV, g_unit_exc=self.g_unit_exc,
                                    g_unit_inh=self.g_unit_inh, dt=self.dt,
                                    delay_slots=self.queue.shape[0])
            self._gpu.upload_state(self.v, self.g, self.refractory, self.queue,
                                   self.queue_count, self.counts, self.active_flag)
            self._refresh_weight_view()
        elif self.backend != 'cpu':
            raise ValueError(f"Unknown brain backend {self.backend!r}; choose 'cpu' or 'cuda'")

    @property
    def weight(self):
        """Edge weights.  Read-only on the CUDA backend: the device copy is
        authoritative, so change weights by assigning a new array or with
        ``set_edge_weights`` / ``update_weights`` rather than editing in place."""
        return self._weight_view if self._gpu is not None else self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value
        if self._gpu is not None:
            self._gpu.set_weights(value)
            self._refresh_weight_view()

    def _refresh_weight_view(self):
        self._weight_view = self._weight.view()
        self._weight_view.setflags(write=False)

    def set_edge_weights(self, edges, values):
        """Write ``values`` into ``weight[edges]`` on the host and, if active, the GPU."""
        if not self._weight.flags.writeable:
            self._weight = self._weight.copy()
            if self._gpu is not None:
                self._refresh_weight_view()
        self._weight[edges] = values
        self.update_weights(edges)

    def update_weights(self, edges):
        """Push ``weight[edges]`` to the GPU after the caller edited the array it
        assigned to ``weight`` (a no-op on the CPU backend)."""
        if self._gpu is not None:
            edges = np.asarray(edges, dtype=np.int64)
            self._gpu.update_edges(edges, self._weight[edges])

    def graph_arrays(self):
        return {name: getattr(self, name) for name, _ in GRAPH_ARRAYS}

    def snapshot_state(self):
        """Copy every mutable transient (membrane, conductance, refractory,
        delay queue, active set, clocks).  Excludes the immutable graph."""
        self._sync_from_gpu()
        state = {name: getattr(self, name).copy() for name in self._state_arrays()}
        state.update(cursor=int(self.cursor), total_spikes=int(self.total_spikes),
                     sim_ms=float(self.sim_ms), dynamics=self.dynamics)
        if self.dynamics in GRADED_VERSIONS and self.graded_report is not None:
            state['graded_set_sha256'] = self.graded_report['graded_set_sha256']
        if self.dynamics == 'v5':
            state['kinetics_sha256'] = self.kinetics_report['kinetics_sha256']
        return state

    def reset_state(self, v_rest_mV=-52.0):
        """Return every transient to rest: membrane at ``v_rest_mV``, empty
        conductances, refractory counters, delay queue and active set, clocks
        at zero.  On the CUDA backend the device copy is reset too, so a reset
        brain replays exactly like a freshly built one."""
        self.v.fill(v_rest_mV)
        for name in self._state_arrays():
            if name != 'v':
                getattr(self, name).fill(0)
        if self._gpu is not None:
            self._upload_gpu_state()
        self.cursor = 0
        self.total_spikes = 0
        self.sim_ms = 0.0

    def restore_state(self, state):
        # v2 and v3 share the (2, n) synaptic state shape, so the shape check
        # below cannot separate them: the version is carried explicitly.  A
        # checkpoint written before this field existed carries no claim and
        # falls back to the shape check, which still separates v1 from v2.
        written_by = state.get('dynamics')
        if written_by is not None and written_by != self.dynamics:
            raise ValueError(
                f'Snapshot was written under LIF dynamics {written_by!r} and this brain runs '
                f'{self.dynamics!r}. A checkpoint written under a different dynamics version is '
                'refused, never reinterpreted; see docs/LIF_DYNAMICS_SPEC.md.')
        if self.dynamics == 'v5':
            written_kin = state.get('kinetics_sha256')
            if written_kin is not None and written_kin != self.kinetics_report['kinetics_sha256']:
                raise ValueError(
                    'Snapshot was written under a different declared receptor-kinetics table / '
                    f'class assignment ({written_kin}) and this brain resolved '
                    f"{self.kinetics_report['kinetics_sha256']}. Refused, never reinterpreted; "
                    'see docs/LIF_DYNAMICS_SPEC.md §9.')
        if self.dynamics in GRADED_VERSIONS and self.graded_report is not None:
            written_set = state.get('graded_set_sha256')
            if written_set is not None and written_set != self.graded_report['graded_set_sha256']:
                raise ValueError(
                    'Snapshot was written under a different declared graded cell-class set '
                    f'({written_set}) and this brain resolved '
                    f"{self.graded_report['graded_set_sha256']}. Refused, never reinterpreted; "
                    'see docs/LIF_DYNAMICS_SPEC.md §7.6.')
        for name in self._state_arrays():
            value = np.asarray(state[name])
            target = getattr(self, name)
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(
                    f'Snapshot array {name} {value.shape}/{value.dtype} does not fit this '
                    f'brain ({target.shape}/{target.dtype}, LIF dynamics {self.dynamics}). '
                    'A checkpoint written under a different dynamics version is refused, '
                    'never reinterpreted; see docs/LIF_DYNAMICS_SPEC.md.')
            target[...] = value
        if self._gpu is not None:
            self._upload_gpu_state()
        self.cursor = int(state['cursor'])
        self.total_spikes = int(state['total_spikes'])
        self.sim_ms = float(state['sim_ms'])

    def step(self, currents, duration_ms):
        """Apply one finite current per neuron (upstream mV-equivalent units)."""
        drive = np.ascontiguousarray(currents, dtype=np.float32)
        if drive.shape != (self.n,) or not np.isfinite(drive).all():
            raise ValueError('Provide a finite current for every neuron')
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Duration must be finite and positive')
        steps = round(duration_ms / self.dt)
        if steps < 1 or not math.isclose(steps*self.dt, duration_ms, abs_tol=1e-9):
            raise ValueError('Duration must be a multiple of 0.1 ms')
        if self._gpu is not None:
            clock = time.perf_counter()
            self.cursor = self._gpu.advance(drive, self.cursor, steps, self.counts)
            elapsed = time.perf_counter()-clock
            self.total_spikes += int(self.counts.sum())
            self.sim_ms += steps*self.dt
            return self.counts.copy(), elapsed
        newly_active = np.flatnonzero((drive != 0) & (self.active_flag == 0))
        start = int(self.nactive[0])
        self.active[start:start+len(newly_active)] = newly_active
        self.active_flag[newly_active] = 1
        self.nactive[0] += len(newly_active)
        self.counts.fill(0)
        clock = time.perf_counter()
        if self.dynamics == 'v1':
            self.cursor = advance(self.ptr, self.post, self.weight, self.v, self.g,
                self.refractory, drive, self.queue, self.queue_count, self.cursor,
                steps, self.dt, self.counts, self.active, self.active_flag, self.nactive)
        elif self.dynamics == 'v2':
            self.cursor = advance_v2(self.ptr, self.post, self.weight, self.v, self.g,
                self.refractory, drive, self.queue, self.queue_count, self.cursor,
                steps, self.dt, self.counts, self.active, self.active_flag, self.nactive,
                self.e_inh_mV)
        elif self.dynamics == 'v5':
            self.cursor = advance_v5(self.ptr, self.post, self.weight, self.v, self.g,
                self.refractory, drive, self.queue, self.queue_count, self.cursor,
                steps, self.dt, self.counts, self.active, self.active_flag, self.nactive,
                self.e_inh_mV, self.g_unit_exc, self.g_unit_inh,
                self.graded, self.graded_idx, self.rel_ring,
                self._kin['pre_chan'], self._kin['chan_decay'], self._kin['chan_q'])
        elif self.dynamics == 'v4':
            self.cursor = advance_v4(self.ptr, self.post, self.weight, self.v, self.g,
                self.refractory, drive, self.queue, self.queue_count, self.cursor,
                steps, self.dt, self.counts, self.active, self.active_flag, self.nactive,
                self.e_inh_mV, self.g_unit_exc, self.g_unit_inh,
                self.graded, self.graded_idx, self.rel_ring)
        else:
            self.cursor = advance_v3(self.ptr, self.post, self.weight, self.v, self.g,
                self.refractory, drive, self.queue, self.queue_count, self.cursor,
                steps, self.dt, self.counts, self.active, self.active_flag, self.nactive,
                self.e_inh_mV, self.g_unit_exc, self.g_unit_inh)
        elapsed = time.perf_counter()-clock
        self.total_spikes += int(self.counts.sum())
        self.sim_ms += steps*self.dt
        return self.counts.copy(), elapsed

    def _upload_gpu_state(self):
        if self.dynamics in GRADED_VERSIONS:
            self._gpu.upload_state(self.v, self.g, self.refractory, self.queue,
                                   self.queue_count, self.counts, self.active_flag,
                                   self.rel_ring)
        else:
            self._gpu.upload_state(self.v, self.g, self.refractory, self.queue,
                                   self.queue_count, self.counts, self.active_flag)

    def release_rate_hz(self):
        """Release rate per neuron in s^-1 under v4 (0 for spiking cells).

        The graded cells' output quantity, as ``counts`` is the spiking cells'.
        """
        if self.dynamics not in GRADED_VERSIONS:
            raise ValueError('release_rate_hz applies to LIF dynamics v4 and v5 only')
        if self._gpu is not None:
            return self._gpu.release_rate_hz()
        from .engine import E_EXC_MV, R_MAX_HZ
        r = np.where(self.graded.astype(bool),
                     (self.v - self.e_inh_mV) * (R_MAX_HZ / (E_EXC_MV - self.e_inh_mV)), 0.0)
        return r.astype(np.float32)

    def _sync_from_gpu(self):
        """Copy device state into the host arrays (CPU layout, for snapshots and probes)."""
        if self._gpu is None:
            return
        if self.dynamics in GRADED_VERSIONS:
            self._gpu.download_state(self.v, self.g, self.refractory, self.queue,
                                     self.queue_count, self.active_flag, self.rel_ring)
        else:
            self._gpu.download_state(self.v, self.g, self.refractory, self.queue,
                                     self.queue_count, self.active_flag)
        # The GPU keeps an activity mask, not an ordered list; rebuild the list.
        idx = np.flatnonzero(self.active_flag).astype(np.int32)
        self.active[:] = 0
        self.active[:len(idx)] = idx
        self.nactive[0] = len(idx)
