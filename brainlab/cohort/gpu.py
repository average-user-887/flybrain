"""GPU cohort engine: B independent fixed-v3 brains on one CUDA device.

Built beside the single-brain v3 CUDA kernel (``brainlab.cupy_engine``,
which is not touched) with a fly axis added.  Equations, constants, phase
order (integrate/threshold/enqueue, deliver, reset) and the 1.8 ms delay
queue follow that kernel.  The ARRIVAL arithmetic is different from it:

* single-brain CUDA kernel: arrivals summed as 64-bit integers at 2**-32
  and folded into g once per tick;
* this engine: each receiving target adds its due arrivals one at a time,
  ``g = float32(double(g) + double(w) * g_unit)`` (inhibitory: minus), in
  ascending (pre index, edge index) order over an incoming-edge index;
* the CPU reference (``engine.advance_v3``) does the same per-arrival float32
  rounding, but in its history-dependent active/queue order, which is not in
  general ascending by pre index.

So this engine and the CPU reference are NOT bit-identical.  On the
preregistered contract workload they agree to a max g relative error of
5.35e-7 with 0 spike mismatches over 200 ticks
(``tests/cohort_contract.json``).  The policy is recorded as ``DELIVERY``
and enters the cohort dynamics signature, so cohorts written under the
earlier fixed-point cohort kernel are refused, not reinterpreted.

Subnormals: the GPU flushes subnormal float32 values to zero (CuPy compiles
with -ftz=true); the CPU reference keeps them. Effect on V/spikes not
observed within the 200-tick contract.

Finding the arrivals (``DELIVERY_INDEX``): phase 2a sets one bit per due
arrival at that edge's position in the incoming-edge view (a per-fly arrival
bitmap); phase 2b walks only the bitmap words of each receiving target's own
incoming range, in ascending position, applies the set bits and clears them.
Up to rc1 phase 2b read every incoming edge of each receiving target and
tested its source; it found the same arrivals in the same order, so its OUTPUTS
(states and spikes) are byte-identical to that kernel on the same device; the
kernel source itself differs (checked by
``scripts/cohort_profile.py --mode identity``).  The arithmetic is unchanged,
so the dynamics signature is unchanged; ``describe()`` reports the index.

Layout: every per-fly array carries a leading fly axis (v ``(B, n)``,
g ``(B, 2, n)``, queue ``(B, slots, n)``, ...).  The CSR graph and its
incoming-edge view are uploaded once and shared read-only.  Work is spread
over the flattened ``fly * n + neuron`` index (the flat equivalent of a
``blockIdx.y`` fly axis), which works for any B on a co-resident
cooperative grid.

Determinism and batch invariance: the accumulation order is fixed per
target and fly, with no atomics on g, so a fly's result does not depend on
B, on the other flies or on the grid shape.  The one order-dependent
artefact is the order of spikers inside a delay-queue slot (an atomic
append); it never affects dynamics, and ``read_state`` returns the queue
canonically (live entries sorted, dead tail zeroed) so states are
byte-comparable.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from ..cuda_engine import THREADS_PER_BLOCK
from ..engine import (DELAY_MS, E_EXC_MV, REFRACTORY_MS, TAU_M_MS, TAU_SYN_MS,
                      V_RESET_MV, V_THRESHOLD_MV, V_REST_MV)
from .api import TICK_MS, CohortEngine

_SOURCE = r'''
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void advance_v3_cohort(
    const long long* __restrict__ ptr, const int* __restrict__ post,
    float* v, float* g, short* refractory,
    const float* __restrict__ drive, int* queue, int* queue_count,
    int* counts, unsigned char* active_flag, unsigned char* spiked,
    const long long* __restrict__ cursor0,
    const long long* __restrict__ in_ptr, const int* __restrict__ csc_pos,
    const float* __restrict__ in_w, long long* due_stamp, long long* recv_stamp,
    unsigned int* arrive_bits, const long long n_words,
    const long long n, const int n_flies, const int delay_slots, const int steps,
    const double dt, const double e_inh, const double g_unit_exc, const double g_unit_inh)
{
    cg::grid_group grid = cg::this_grid();
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long nthreads = (long long)gridDim.x * blockDim.x;
    const int delay_ticks = (int)llrint(@DELAY_MS@ / dt);
    const int refractory_ticks = (int)llrint(@REFRACTORY_MS@ / dt);
    const double ag = exp(-dt / @TAU_SYN_MS@);
    const long long warp_id = tid / 32;
    const int lane = (int)(tid % 32);
    const long long nwarps = nthreads / 32;
    const long long total = n * (long long)n_flies;

    for (int step = 0; step < steps; ++step) {
        // Phase 1: integrate, threshold, enqueue (every fly, every neuron).
        for (long long x = tid; x < total; x += nthreads) {
            const long long f = x / n;
            const long long i = x - f * n;
            if (active_flag[x] == 0) {
                if (drive[x] != 0.0f) active_flag[x] = 1; else continue;
            }
            if (refractory[x] > 0) refractory[x] -= 1;
            if (refractory[x] == 0) {
                float* g_exc = g + f * 2 * n;
                float* g_inh = g_exc + n;
                const double ge = g_exc[i], gi = g_inh[i];
                const double gtot = 1.0 + ge + gi;
                const double vinf = (@V_REST_MV@ + ge * @E_EXC_MV@ + gi * e_inh + (double)drive[x]) / gtot;
                double vi = vinf + ((double)v[x] - vinf) * exp(-dt * gtot / @TAU_M_MS@);
                if (vi < e_inh) vi = e_inh; else if (vi > @E_EXC_MV@) vi = @E_EXC_MV@;
                v[x] = (float)vi;
                g_exc[i] = (float)(ge * ag);
                g_inh[i] = (float)(gi * ag);
                if (vi > @V_THRESHOLD_MV@) {
                    const long long cursor = cursor0[f] + step;
                    const int future = (int)((cursor + delay_ticks) % delay_slots);
                    counts[x] += 1;
                    spiked[x] = 1;
                    const long long qrow = f * delay_slots + future;
                    const int k = atomicAdd(&queue_count[qrow], 1);
                    queue[qrow * n + k] = (int)i;
                }
            }
        }
        grid.sync();

        // Phase 2a: stamp this tick's due spikers and the targets they reach,
        // and mark each due arrival's incoming-edge (CSC) position in the
        // fly's arrival bitmap.
        for (int f = 0; f < n_flies; ++f) {
            const long long cursor = cursor0[f] + step;
            const long long stamp = cursor + 1;
            const int slot = (int)(cursor % delay_slots);
            const long long qrow = (long long)f * delay_slots + slot;
            const int nspk = queue_count[qrow];
            const int* qf = queue + qrow * n;
            short* ref_f = refractory + (long long)f * n;
            unsigned int* bits_f = arrive_bits + (long long)f * n_words;
            for (long long q = warp_id; q < nspk; q += nwarps) {
                const int pre = qf[q];
                if (lane == 0) due_stamp[(long long)f * n + pre] = stamp;
                const long long end = ptr[pre + 1];
                for (long long e = ptr[pre] + lane; e < end; e += 32) {
                    const int j = post[e];
                    if (ref_f[j] > 0) continue;
                    recv_stamp[(long long)f * n + j] = stamp;
                    const long long p = csc_pos[e];
                    atomicOr(&bits_f[p >> 5], 1u << (int)(p & 31));
                }
            }
        }
        grid.sync();

        // Phase 2b+3: each receiving target adds its arrivals one by one in
        // float32, in ascending (pre, edge) order, as the CPU kernel's
        // per-arrival accumulation does; then this tick's spikers reset.
        // The arrivals are the set bits of the target's own CSC range, read
        // in ascending position (= ascending (pre, edge)) and cleared, so the
        // bitmap is all zero again when the tick ends.
        for (long long f = tid; f < n_flies; f += nthreads) {
            const long long cursor = cursor0[f] + step;
            queue_count[f * delay_slots + (int)(cursor % delay_slots)] = 0;
        }
        for (long long x = tid; x < total; x += nthreads) {
            const long long f = x / n;
            const long long i = x - f * n;
            float* g_exc = g + f * 2 * n;
            float* g_inh = g_exc + n;
            const long long stamp = cursor0[f] + step + 1;
            if (recv_stamp[x] == stamp) {
                unsigned int* bits_f = arrive_bits + f * n_words;
                float ge = g_exc[i], gi = g_inh[i];
                const long long lo = in_ptr[i], hi = in_ptr[i + 1];   // hi > lo: i was reached
                const long long wlo = lo >> 5, whi = (hi - 1) >> 5;
                for (long long wd = wlo; wd <= whi; ++wd) {
                    unsigned int own = 0xffffffffu;
                    if (wd == wlo) own &= 0xffffffffu << (int)(lo & 31);
                    if (wd == whi) own &= 0xffffffffu >> (31 - (int)((hi - 1) & 31));
                    unsigned int m = bits_f[wd] & own;
                    if (m == 0u) continue;
                    // Words shared with a neighbouring target are cleared bit-wise.
                    if (own == 0xffffffffu) bits_f[wd] = 0u; else atomicAnd(&bits_f[wd], ~m);
                    while (m) {
                        const long long e = (wd << 5) + (__ffs(m) - 1);
                        m &= m - 1u;
                        const float w = in_w[e];
                        if (w > 0.0f) ge = (float)((double)ge + (double)w * g_unit_exc);
                        else gi = (float)((double)gi - (double)w * g_unit_inh);
                    }
                }
                g_exc[i] = ge; g_inh[i] = gi;
                active_flag[x] = 1;
            }
            if (spiked[x]) {
                spiked[x] = 0;
                v[x] = (float)@V_RESET_MV@;
                g_exc[i] = 0.0f; g_inh[i] = 0.0f;
                refractory[x] = (short)refractory_ticks;
            }
        }
        grid.sync();
    }
}
'''
for _name, _value in dict(DELAY_MS=DELAY_MS, REFRACTORY_MS=REFRACTORY_MS, TAU_SYN_MS=TAU_SYN_MS,
                          TAU_M_MS=TAU_M_MS, V_REST_MV=V_REST_MV,
                          E_EXC_MV=E_EXC_MV, V_THRESHOLD_MV=V_THRESHOLD_MV,
                          V_RESET_MV=V_RESET_MV).items():
    # Same literal spelling as the single-brain kernel (``%r`` of the float).
    _SOURCE = _SOURCE.replace(f'@{_name}@', repr(float(_value)))

V_INIT_MV = -52.0
STATE_KEYS = ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts',
              'active', 'active_flag', 'nactive')
SCALAR_KEYS = ('cursor', 'total_spikes', 'sim_ms')
GPU_ENV = 'NEUROFLY_COHORT_GPU'
# Arrival arithmetic of this engine and of the CPU reference (see the module
# docstring).  Part of the cohort dynamics signature.
DELIVERY = 'per-arrival-float32-ascending-pre-edge'
CPU_DELIVERY = 'per-arrival-float32-cpu-active-queue-order'
# How this engine finds a target's due arrivals (not arithmetic: the arrivals,
# their order and their rounding are those of the earlier full incoming-edge
# scan, so it is NOT part of the dynamics signature).  Reported by describe().
DELIVERY_INDEX = 'csc-arrival-bitmap-v1'

# One device copy of each read-only graph per (device, host buffer).
_GRAPH_CACHE: dict = {}


def select_device(device=None) -> int:
    """CUDA device index to use.

    ``device`` (else ``NEUROFLY_COHORT_GPU``) is an int index or a substring
    of the device name (e.g. ``"1660"``, which must match exactly one device;
    a digit string matching no name is taken as an index).
    With neither, the device with the most memory is used.
    """
    import cupy as cp
    device = device if device is not None else os.environ.get(GPU_ENV)
    count = cp.cuda.runtime.getDeviceCount()
    names = []
    for k in range(count):
        name = cp.cuda.runtime.getDeviceProperties(k)['name']
        names.append(name.decode('utf-8', 'replace') if isinstance(name, bytes) else str(name))
    if device is None:
        mems = [cp.cuda.runtime.getDeviceProperties(k)['totalGlobalMem'] for k in range(count)]
        return int(np.argmax(mems))
    hits = [k for k, name in enumerate(names) if str(device) in name]
    if isinstance(device, (int, np.integer)) or (not hits and str(device).isdigit()):
        if not 0 <= int(device) < count:
            raise ValueError(f'No CUDA device {device}; found {names}')
        return int(device)
    if len(hits) != 1:
        raise ValueError(f'CUDA device name {device!r} matches {len(hits)} of {names}')
    return hits[0]


def device_name(index: int) -> str:
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(index)['name']
    return name.decode('utf-8', 'replace') if isinstance(name, bytes) else str(name)


def _graph_on_device(cp, dev: int, arrays: dict, g_unit_exc: float, g_unit_inh: float):
    ptr, post, weight = arrays['ptr'], arrays['post'], arrays['weight']
    key = None
    if not (ptr.flags.writeable or post.flags.writeable or weight.flags.writeable):
        key = (dev, ptr.__array_interface__['data'][0], post.__array_interface__['data'][0],
               weight.__array_interface__['data'][0], weight.nbytes, g_unit_exc, g_unit_inh)
        if key in _GRAPH_CACHE:
            return _GRAPH_CACHE[key][1]
    n = len(ptr) - 1
    # Incoming-edge (CSC) view: per target, edges in ascending (pre, edge) order.
    order = np.argsort(post, kind='stable')
    in_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post, minlength=n), out=in_ptr[1:])
    if len(post) >= 2**31:
        raise ValueError('graph has too many edges for the int32 CSC position map')
    # CSR edge -> its position in the incoming-edge view (the arrival bitmap index).
    csc_pos = np.empty(len(post), dtype=np.int32)
    csc_pos[order] = np.arange(len(post), dtype=np.int32)
    graph = (cp.asarray(ptr), cp.asarray(post),
             cp.asarray(in_ptr), cp.asarray(csc_pos),
             cp.asarray(np.ascontiguousarray(weight[order], dtype=np.float32)))
    if key is not None:
        # Holding the host arrays keeps their buffers alive, so a key never goes stale.
        _GRAPH_CACHE[key] = ((ptr, post, weight), graph)
    return graph


class GpuCohortEngine(CohortEngine):
    """B independent fixed-v3 brains on one CUDA device (CuPy build)."""

    backend_id = 'cuda-cohort-v3'

    def __init__(self, arrays: dict, n_flies: int, *, device=None, blocks: Optional[int] = None,
                 validate: bool = True):
        super().__init__(arrays, n_flies)
        clock = time.perf_counter()
        import cupy as cp
        from ..brain import Brain
        self.cp = cp
        self.timings: Dict[str, float] = {}
        # A host Brain fixes every constant (E_inh, g units, dt, delay slots)
        # and the exact snapshot format, so both engines read the same values.
        template = Brain(arrays=arrays, validate=validate, dynamics='v3', backend='cpu')
        if template.dynamics != 'v3':
            raise ValueError('GpuCohortEngine runs fixed v3 only')
        self.dt = float(template.dt)
        self.e_inh = float(template.e_inh_mV)
        self.g_unit_exc = float(template.g_unit_exc)
        self.g_unit_inh = float(template.g_unit_inh)
        self.delay_slots = int(template.queue.shape[0])
        snap = template.snapshot_state()
        self._format = {name: (snap[name].shape, snap[name].dtype) for name in STATE_KEYS}
        del template, snap
        self.device_index = select_device(device)
        self.device_name = device_name(self.device_index)
        self.device = cp.cuda.Device(self.device_index)
        self.timings['host_setup_s'] = time.perf_counter() - clock

        B, n, S = self.n_flies, self.n, self.delay_slots
        with self.device:
            clock = time.perf_counter()
            self.kernel = cp.RawKernel(_SOURCE, 'advance_v3_cohort', options=('-std=c++17',),
                                       enable_cooperative_groups=True)
            self.kernel.compile()
            self.timings['compile_s'] = time.perf_counter() - clock
            clock = time.perf_counter()
            (self.d_ptr, self.d_post,
             self.d_in_ptr, self.d_csc_pos, self.d_in_w) = _graph_on_device(
                cp, self.device_index, arrays, self.g_unit_exc, self.g_unit_inh)
            cp.cuda.Device().synchronize()
            self.timings['graph_upload_s'] = time.perf_counter() - clock
            clock = time.perf_counter()
            self.d_v = cp.full((B, n), V_INIT_MV, dtype=cp.float32)
            self.d_g = cp.zeros((B, 2, n), dtype=cp.float32)
            self.d_due = cp.zeros((B, n), dtype=cp.int64)
            self.d_recv = cp.zeros((B, n), dtype=cp.int64)
            # One bit per incoming edge per fly; all zero between ticks.
            self.n_words = (len(arrays['post']) + 31) // 32
            self.d_bits = cp.zeros((B, self.n_words), dtype=cp.uint32)
            self.d_refractory = cp.zeros((B, n), dtype=cp.int16)
            self.d_drive = cp.zeros((B, n), dtype=cp.float32)
            self.d_queue = cp.zeros((B, S, n), dtype=cp.int32)
            self.d_queue_count = cp.zeros((B, S), dtype=cp.int32)
            self.d_counts = cp.zeros((B, n), dtype=cp.int32)
            self.d_active_flag = cp.zeros((B, n), dtype=cp.uint8)
            self.d_spiked = cp.zeros((B, n), dtype=cp.uint8)
            self.d_cursor = cp.zeros(B, dtype=cp.int64)
            cp.cuda.Device().synchronize()
            self.timings['state_alloc_s'] = time.perf_counter() - clock
        self.cursor = [0] * B
        self.total_spikes = [0] * B
        self.sim_ms = [0.0] * B
        self._last_drive = None
        self._blocks = blocks

    # ------------------------------------------------------------------ helpers
    def _grid_blocks(self) -> int:
        if self._blocks is None:
            cp = self.cp
            sms = self.device.attributes['MultiProcessorCount']
            try:
                per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                    self.kernel.kernel.ptr, THREADS_PER_BLOCK, 0)
            except Exception:
                per_sm = 1
            needed = (self.n * self.n_flies + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            self._blocks = int(max(1, min(sms * max(1, per_sm), needed)))
        return self._blocks

    def _check_fly(self, fly: int) -> int:
        if isinstance(fly, bool) or not isinstance(fly, (int, np.integer)) \
                or not 0 <= int(fly) < self.n_flies:
            raise ValueError(f'fly must be in [0, {self.n_flies})')
        return int(fly)

    def device_bytes(self) -> int:
        """Bytes held by this engine's state plus the shared graph."""
        arrays = (self.d_ptr, self.d_post, self.d_in_ptr, self.d_csc_pos, self.d_in_w,
                  self.d_due, self.d_recv, self.d_bits, self.d_v, self.d_g,
                  self.d_refractory, self.d_drive, self.d_queue, self.d_queue_count,
                  self.d_counts, self.d_active_flag, self.d_spiked, self.d_cursor)
        return int(sum(a.nbytes for a in arrays))

    def describe(self) -> dict:
        info = super().describe()
        info.update(delivery=DELIVERY, delivery_index=DELIVERY_INDEX, device=self.device_name, device_index=self.device_index,
                    device_bytes=self.device_bytes(), grid_blocks=self._grid_blocks(),
                    threads_per_block=THREADS_PER_BLOCK)
        return info

    # --------------------------------------------------------------- interface
    def reset(self, fly_ids: Sequence[int]) -> None:
        with self.device:
            for k in fly_ids:
                k = self._check_fly(k)
                self.d_v[k].fill(V_INIT_MV)
                for arr in (self.d_g, self.d_due, self.d_recv, self.d_bits, self.d_refractory, self.d_queue,
                            self.d_queue_count, self.d_counts, self.d_active_flag, self.d_spiked):
                    arr[k].fill(0)
                self.cursor[k] = 0
                self.total_spikes[k] = 0
                self.sim_ms[k] = 0.0

    def _kernel_args(self, ticks: int) -> tuple:
        return (self.d_ptr, self.d_post, self.d_v, self.d_g,
                self.d_refractory, self.d_drive, self.d_queue, self.d_queue_count,
                self.d_counts, self.d_active_flag, self.d_spiked, self.d_cursor,
                self.d_in_ptr, self.d_csc_pos, self.d_in_w, self.d_due, self.d_recv,
                self.d_bits, np.int64(self.n_words), np.int64(self.n),
                np.int32(self.n_flies), np.int32(self.delay_slots),
                np.int32(ticks), np.float64(self.dt), np.float64(self.e_inh),
                np.float64(self.g_unit_exc), np.float64(self.g_unit_inh))

    def step(self, drive: np.ndarray, ticks: int) -> np.ndarray:
        drive = self._check_step_args(drive, ticks)   # whole batch, before any fly moves
        ticks = int(ticks)
        with self.device:
            if self._last_drive is None or not np.array_equal(self._last_drive, drive):
                self.d_drive.set(drive)
                self._last_drive = drive.copy()
            self.d_counts.fill(0)
            self.d_cursor.set(np.asarray(self.cursor, dtype=np.int64))
            self.kernel((self._grid_blocks(),), (THREADS_PER_BLOCK,), self._kernel_args(ticks))
            counts = self.d_counts.get()
        for k in range(self.n_flies):
            self.cursor[k] += ticks
            self.total_spikes[k] += int(counts[k].sum())
            self.sim_ms[k] += ticks * self.dt   # as Brain.step: steps * dt
        return counts

    def read_state(self, fly: int) -> Dict[str, object]:
        k = self._check_fly(fly)
        with self.device:
            v = self.d_v[k].get()
            g = self.d_g[k].get()
            refractory = self.d_refractory[k].get()
            queue = self.d_queue[k].get()
            queue_count = self.d_queue_count[k].get()
            counts = self.d_counts[k].get()
            active_flag = self.d_active_flag[k].get()
        # Canonical queue: live entries sorted, dead tail zeroed (never read).
        for s in range(self.delay_slots):
            c = int(queue_count[s])
            queue[s, :c] = np.sort(queue[s, :c])
            queue[s, c:] = 0
        # The GPU keeps an activity mask; rebuild the list as Brain does on CUDA.
        idx = np.flatnonzero(active_flag).astype(np.int32)
        active = np.zeros(self.n, dtype=np.int32)
        active[:len(idx)] = idx
        nactive = np.array([len(idx)], dtype=np.int32)
        return dict(v=v, g=g, refractory=refractory, queue=queue, queue_count=queue_count,
                    counts=counts, active=active, active_flag=active_flag, nactive=nactive,
                    cursor=int(self.cursor[k]), total_spikes=int(self.total_spikes[k]),
                    sim_ms=float(self.sim_ms[k]), dynamics='v3')

    def _validated(self, fly: int, state) -> dict:
        """Base-contract validation against this fly's live snapshot, then the
        extra range checks that keep the kernel's indexing in bounds."""
        if not isinstance(state, dict):
            raise ValueError('state must be a dict in Brain.snapshot_state() format; refused')
        out = self._check_state(fly, state, self.read_state(fly))
        for key in ('active_flag', 'refractory', 'queue', 'active', 'nactive'):
            if out[key].shape != self._format[key][0] or out[key].dtype != self._format[key][1]:
                raise ValueError(f'state array {key} does not fit v3; refused')
        qc = out['queue_count']
        for s in range(self.delay_slots):
            live = out['queue'][s, :int(qc[s])]
            if live.size and (live.min() < 0 or live.max() >= self.n):
                raise ValueError('state queue holds an out-of-range neuron; refused')
        if (out['refractory'] < 0).any() or (out['active_flag'] > 1).any():
            raise ValueError('state refractory/active_flag out of range; refused')
        na = int(out['nactive'][0])
        listed = np.unique(out['active'][:na]) if 0 <= na <= self.n else None
        if listed is None or len(listed) != na or not np.array_equal(
                listed, np.flatnonzero(out['active_flag'])):
            raise ValueError('state active list and active_flag disagree; refused')
        return out

    def write_state(self, fly: int, state: Dict[str, object]) -> None:
        k = self._check_fly(fly)
        arrays = self._validated(k, state)   # staged copies; nothing live changes on refusal
        with self.device:
            self.d_v[k].set(arrays['v'])
            self.d_g[k].set(arrays['g'])
            self.d_refractory[k].set(arrays['refractory'])
            self.d_queue[k].set(arrays['queue'])
            self.d_queue_count[k].set(arrays['queue_count'])
            self.d_counts[k].set(arrays['counts'])
            self.d_active_flag[k].set(arrays['active_flag'])
            self.d_due[k].fill(0)
            self.d_recv[k].fill(0)
            self.d_bits[k].fill(0)
            self.d_spiked[k].fill(0)
            self.device.synchronize()
        self.cursor[k] = int(arrays['cursor'])
        self.total_spikes[k] = int(arrays['total_spikes'])
        self.sim_ms[k] = float(arrays['sim_ms'])
