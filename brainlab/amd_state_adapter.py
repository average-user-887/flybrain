"""Explicit fixed-weight v3 state facade for the optional AMD engine.

Brain can select this facade explicitly; registry/daemon admission is separate.
Construction leaves it unloaded until a complete explicit restore or reset.
Array dtypes and ordering are preserved, never converted or reconstructed.
"""
from collections.abc import Mapping
import math
import time
from types import MappingProxyType

import numpy as np

from .wgpu_v3 import MAX_ADVANCE_STEPS, WgpuV3State, validate_graph


STATE_ARRAYS = ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts',
                'active', 'active_flag', 'nactive')
STATE_SCALARS = ('cursor', 'total_spikes', 'sim_ms')
STATE_KEYS = frozenset((*STATE_ARRAYS, *STATE_SCALARS, 'dynamics'))
CAPABILITIES = MappingProxyType({
    'dynamics': 'v3',
    'fixed_weights': True,
    'full_transient_checkpoint': True,
    'weight_mutation': False,
    'learning': False,
    'learned_state_restore': False,
})


class UnsupportedAmdCapability(NotImplementedError):
    """The fixed-weight adapter cannot perform the requested operation."""


class AmdStateUnavailable(RuntimeError):
    """State cannot advance/export until a complete explicit restore or reset."""
    failure_class = 'compute'


class AmdV3StateAdapter:
    """Wrap the pinned ordered engine without changing its shader or arithmetic.

    Device/transfer failure poisons this facade as well as the underlying engine.
    Validation errors leave a previously valid state intact. Recovery is always
    an explicit complete restore/reset, never a fallback or automatic restart.
    Like Brain, the adapter requires exclusive access during each operation.
    """
    capabilities = CAPABILITIES

    def __init__(self, ptr, post, weight, *, n, device=None):
        validate_graph(ptr, post, weight, n)
        # Own the graph values passed to the engine. Accessors expose detached
        # weights, so even changing their write flag cannot change this model.
        ptr, post, weight = (a.copy() for a in (ptr, post, weight))
        self._weight = weight
        self._weight.setflags(write=False)
        self._engine = WgpuV3State(ptr, post, weight, n=n, device=device)
        # Read identity from the actual selected device once, never a second
        # adapter probe during status publication. Missing fields stay unknown.
        info = dict(getattr(self._engine.device, 'adapter_info', {}) or {})
        self._device_info = {key: info.get(key) for key in
                             ('vendor', 'vendor_id', 'device', 'device_id',
                              'description', 'adapter_type', 'backend_type')}
        self.n = n
        self.dt = self._engine.dt
        self._poisoned = False

    @property
    def device_info(self):
        return dict(self._device_info)

    @property
    def status(self):
        poisoned = self._poisoned or self._engine.poisoned
        loaded = self._engine.loaded
        return dict(loaded=loaded, poisoned=poisoned,
                    can_advance=loaded and not poisoned,
                    can_export=loaded and not poisoned)

    @property
    def weight(self):
        detached = self._weight.copy()
        detached.setflags(write=False)
        return detached

    @weight.setter
    def weight(self, value):
        self._refuse_mutation()

    @staticmethod
    def _refuse_mutation(*args, **kwargs):
        raise UnsupportedAmdCapability('Fixed-weight v3 adapter: weight mutation is unsupported')

    set_weights = _refuse_mutation
    update_edges = _refuse_mutation
    set_edge_weights = _refuse_mutation
    update_weights = _refuse_mutation

    @staticmethod
    def set_learning(*args, **kwargs):
        raise UnsupportedAmdCapability('Fixed-weight v3 adapter: learning is unsupported')

    def _require_state(self):
        if not self.status['can_export']:
            raise AmdStateUnavailable('Complete explicit restore/reset required; state is unloaded or poisoned')

    def _detach_and_validate(self, state):
        if not isinstance(state, Mapping):
            raise ValueError('Complete v3 snapshot mapping required')
        extra = state.keys() - STATE_KEYS
        if extra:
            raise UnsupportedAmdCapability('Extra snapshot fields, including learned state, are unsupported')
        if state.keys() != STATE_KEYS or state['dynamics'] != 'v3':
            raise ValueError('Complete explicitly declared v3 snapshot required')
        shapes = ((self.n,), (2, self.n), (self.n,), (19, self.n), (19,),
                  (self.n,), (self.n,), (self.n,), (1,))
        dtypes = (np.float32, np.float32, np.int16, np.int32, np.int32,
                  np.int32, np.int32, np.uint8, np.int32)
        staged = dict(dynamics='v3')
        for name, shape, dtype in zip(STATE_ARRAYS, shapes, dtypes, strict=True):
            a = state[name]
            if type(a) is not np.ndarray or a.shape != shape or a.dtype != dtype:
                raise ValueError(f'Snapshot {name}: exact ndarray shape/dtype required')
            staged[name] = a.copy(order='C')
        for name in ('cursor', 'total_spikes'):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError(f'Snapshot {name}: nonnegative integer required')
            staged[name] = value
        if staged['cursor'] > 2**32 - 20:
            raise ValueError('Snapshot cursor capacity exceeded')
        sim_ms = state['sim_ms']
        if type(sim_ms) is not float or not math.isfinite(sim_ms) or sim_ms < 0:
            raise ValueError('Snapshot sim_ms: finite nonnegative float required')
        staged['sim_ms'] = sim_ms
        if (not np.isfinite(staged['v']).all() or not np.isfinite(staged['g']).all()
                or np.any(staged['g'] < 0)):
            raise ValueError('Snapshot membranes/conductances must be finite with nonnegative conductances')
        if (np.any(staged['refractory'] < 0) or np.any(staged['refractory'] > 22)
                or np.any(staged['counts'] < 0)):
            raise ValueError('Snapshot refractory/counts capacity')
        count = staged['queue_count']
        if np.any(count < 0) or np.any(count > self.n):
            raise ValueError('Snapshot queue capacity')
        for slot, length in enumerate(count):
            members = staged['queue'][slot, :int(length)]
            if (np.any(members < 0) or np.any(members >= self.n)
                    or len(np.unique(members)) != length):
                raise ValueError('Snapshot pending queue membership')
        flags = staged['active_flag']
        length = int(staged['nactive'][0])
        if np.any((flags != 0) & (flags != 1)) or not 0 <= length <= self.n:
            raise ValueError('Snapshot active flags/count capacity')
        members = staged['active'][:length]
        if (np.any(members < 0) or np.any(members >= self.n)
                or len(np.unique(members)) != length
                or not np.array_equal(np.sort(members), np.flatnonzero(flags))):
            raise ValueError('Snapshot active membership')
        return staged

    def restore_state(self, state):
        """Validate a detached complete snapshot before any device write.

        Failed transfers can partially change device buffers: state then remains
        poisoned and cannot export. They never trigger automatic reset.
        """
        staged = self._detach_and_validate(state)
        try:
            self._engine.upload_state(
                *(staged[name] for name in ('v', 'g', 'refractory', 'queue',
                                           'queue_count', 'counts', 'active_flag')),
                active=staged['active'], nactive=int(staged['nactive'][0]),
                cursor=staged['cursor'], total_spikes=staged['total_spikes'],
                sim_ms=staged['sim_ms'])
        except BaseException:
            self._poisoned = True
            raise
        self._poisoned = False

    def snapshot_state(self):
        """Return detached Brain-compatible v3 transients with exact active order."""
        self._require_state()
        try:
            state = self._engine.snapshot_state()
            state['dynamics'] = 'v3'
            return self._detach_and_validate(state)
        except BaseException:
            self._poisoned = True
            raise

    def reset_state(self):
        """Explicit baseline reset; clears all transients and clocks."""
        state = dict(v=np.full(self.n, -52, dtype=np.float32),
                     g=np.zeros((2, self.n), dtype=np.float32),
                     refractory=np.zeros(self.n, dtype=np.int16),
                     queue=np.zeros((19, self.n), dtype=np.int32),
                     queue_count=np.zeros(19, dtype=np.int32),
                     counts=np.zeros(self.n, dtype=np.int32),
                     active=np.zeros(self.n, dtype=np.int32),
                     active_flag=np.zeros(self.n, dtype=np.uint8),
                     nactive=np.zeros(1, dtype=np.int32),
                     cursor=0, total_spikes=0, sim_ms=0.0, dynamics='v3')
        self.restore_state(state)

    def step(self, currents, duration_ms):
        """Brain-style (detached counts, elapsed seconds); exact f32 input only."""
        self._require_state()
        if (type(currents) is not np.ndarray or currents.dtype != np.float32
                or currents.shape != (self.n,) or not np.isfinite(currents).all()):
            raise ValueError('Exact finite float32 current ndarray required')
        drive = currents.copy(order='C')
        if (type(duration_ms) not in (int, float) or not math.isfinite(duration_ms)
                or duration_ms <= 0 or duration_ms > MAX_ADVANCE_STEPS * self.dt):
            raise ValueError('Duration outside bounded v3 advance capacity')
        steps = round(duration_ms / self.dt)
        if steps < 1 or not math.isclose(steps * self.dt, duration_ms, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Duration must be an exact multiple of 0.1 ms')
        cursor = self._engine.cursor
        if cursor + steps > 2**32 - 20:
            raise ValueError('Cursor capacity exceeded')
        counts = np.zeros(self.n, dtype=np.int32)
        started = time.perf_counter()
        try:
            self._engine.advance(drive, cursor, steps, counts)
        except BaseException:
            self._poisoned = True
            raise
        return counts, time.perf_counter() - started
