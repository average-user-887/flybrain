"""CuPy build of the v3 CUDA kernel, for hosts without numba's CUDA target.

A line-for-line CUDA C translation of ``cuda_engine._advance_v3_kernel``;
see that module for the phase structure and the determinism argument.  Both
implementations expose the same state class interface, and
``cuda_engine.make_state`` picks whichever is usable on this host.
"""
import numpy as np

from .cuda_engine import FIXED_SCALE, THREADS_PER_BLOCK, edge_increments
from .engine import (DELAY_MS, E_EXC_MV, REFRACTORY_MS, TAU_M_MS, TAU_SYN_MS,
                     V_RESET_MV, V_THRESHOLD_MV, V_REST_MV)

_SOURCE = r'''
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void advance_v3(
    const long long* __restrict__ ptr, const int* __restrict__ post,
    const long long* __restrict__ edge_inc,
    float* v, float* g, unsigned long long* acc, short* refractory,
    const float* __restrict__ drive, int* queue, int* queue_count,
    int* counts, unsigned char* active_flag, unsigned char* spiked,
    const long long n, const int delay_slots, const long long cursor0, const int steps,
    const double dt, const double e_inh)
{
    cg::grid_group grid = cg::this_grid();
    const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long nthreads = (long long)gridDim.x * blockDim.x;
    const int delay_ticks = (int)llrint(%(DELAY_MS)r / dt);
    const int refractory_ticks = (int)llrint(%(REFRACTORY_MS)r / dt);
    const double ag = exp(-dt / %(TAU_SYN_MS)r);
    const long long warp_id = tid / 32;
    const int lane = (int)(tid %% 32);
    const long long nwarps = nthreads / 32;
    const double inv_scale = 1.0 / %(FIXED_SCALE)r;
    float* g_exc = g;
    float* g_inh = g + n;
    unsigned long long* acc_exc = acc;
    unsigned long long* acc_inh = acc + n;

    for (int step = 0; step < steps; ++step) {
        const long long cursor = cursor0 + step;
        const int slot = (int)(cursor %% delay_slots);
        const int future = (int)((cursor + delay_ticks) %% delay_slots);

        // Phase 1: integrate, threshold, enqueue.
        for (long long i = tid; i < n; i += nthreads) {
            if (active_flag[i] == 0) {
                if (drive[i] != 0.0f) active_flag[i] = 1; else continue;
            }
            if (refractory[i] > 0) refractory[i] -= 1;
            if (refractory[i] == 0) {
                const double ge = g_exc[i], gi = g_inh[i];
                const double gtot = 1.0 + ge + gi;
                const double vinf = (%(V_REST_MV)r + ge * %(E_EXC_MV)r + gi * e_inh + (double)drive[i]) / gtot;
                double vi = vinf + ((double)v[i] - vinf) * exp(-dt * gtot / %(TAU_M_MS)r);
                if (vi < e_inh) vi = e_inh; else if (vi > %(E_EXC_MV)r) vi = %(E_EXC_MV)r;
                v[i] = (float)vi;
                g_exc[i] = (float)(ge * ag);
                g_inh[i] = (float)(gi * ag);
                if (vi > %(V_THRESHOLD_MV)r) {
                    counts[i] += 1;
                    spiked[i] = 1;
                    const int k = atomicAdd(&queue_count[future], 1);
                    queue[(long long)future * n + k] = (int)i;
                }
            }
        }
        grid.sync();

        // Phase 2: deliver the spikes due this tick, one warp per spiker.
        const int nspk = queue_count[slot];
        for (long long q = warp_id; q < nspk; q += nwarps) {
            const int pre = queue[(long long)slot * n + q];
            const long long end = ptr[pre + 1];
            for (long long e = ptr[pre] + lane; e < end; e += 32) {
                const int j = post[e];
                if (refractory[j] > 0) continue;
                const long long inc = edge_inc[e];
                if (inc > 0) atomicAdd(&acc_exc[j], (unsigned long long)inc);
                else atomicAdd(&acc_inh[j], (unsigned long long)(-inc));
                active_flag[j] = 1;
            }
        }
        grid.sync();

        // Phase 3: fold arrivals into g, reset this tick's spikers.
        if (tid == 0) queue_count[slot] = 0;
        for (long long i = tid; i < n; i += nthreads) {
            if (spiked[i]) {
                spiked[i] = 0;
                v[i] = (float)%(V_RESET_MV)r;
                g_exc[i] = 0.0f; g_inh[i] = 0.0f;
                refractory[i] = (short)refractory_ticks;
                acc_exc[i] = 0ULL; acc_inh[i] = 0ULL;
            } else {
                const unsigned long long a0 = acc_exc[i], a1 = acc_inh[i];
                if (a0) { g_exc[i] = (float)((double)g_exc[i] + (double)(long long)a0 * inv_scale); acc_exc[i] = 0ULL; }
                if (a1) { g_inh[i] = (float)((double)g_inh[i] + (double)(long long)a1 * inv_scale); acc_inh[i] = 0ULL; }
            }
        }
        grid.sync();
    }
}
''' % dict(DELAY_MS=float(DELAY_MS), REFRACTORY_MS=float(REFRACTORY_MS), TAU_SYN_MS=float(TAU_SYN_MS),
           TAU_M_MS=float(TAU_M_MS), FIXED_SCALE=float(FIXED_SCALE), V_REST_MV=float(V_REST_MV),
           E_EXC_MV=float(E_EXC_MV), V_THRESHOLD_MV=float(V_THRESHOLD_MV),
           V_RESET_MV=float(V_RESET_MV))


def cupy_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class CupyV3State:
    """Device-resident graph and v3 state for one ``Brain`` (CuPy build)."""

    def __init__(self, ptr, post, weight, *, n, e_inh, g_unit_exc, g_unit_inh, dt,
                 delay_slots, blocks=None):
        import cupy as cp
        self.cp = cp
        self.n = int(n)
        self.dt = float(dt)
        self.e_inh = float(e_inh)
        self.delay_slots = int(delay_slots)
        self.kernel = cp.RawKernel(_SOURCE, 'advance_v3', options=('-std=c++17',),
                                   enable_cooperative_groups=True)
        self.d_ptr = cp.asarray(ptr)
        self.d_post = cp.asarray(post)
        self.d_edge_inc = cp.asarray(edge_increments(weight, g_unit_exc, g_unit_inh))
        self.d_acc = cp.zeros((2, self.n), dtype=cp.uint64)
        self.d_spiked = cp.zeros(self.n, dtype=cp.uint8)
        self.d_drive = cp.zeros(self.n, dtype=cp.float32)
        self._blocks = blocks
        self._last_drive = None

    def upload_state(self, v, g, refractory, queue, queue_count, counts, active_flag):
        cp = self.cp
        self.d_v = cp.asarray(v)
        self.d_g = cp.ascontiguousarray(cp.asarray(g))
        self.d_refractory = cp.asarray(refractory)
        self.d_queue = cp.ascontiguousarray(cp.asarray(queue))
        self.d_queue_count = cp.asarray(queue_count)
        self.d_counts = cp.asarray(counts)
        self.d_active_flag = cp.asarray(active_flag)

    def download_state(self, v, g, refractory, queue, queue_count, active_flag):
        v[...] = self.d_v.get()
        g[...] = self.d_g.get()
        refractory[...] = self.d_refractory.get()
        queue[...] = self.d_queue.get()
        queue_count[...] = self.d_queue_count.get()
        active_flag[...] = self.d_active_flag.get()

    def _grid_blocks(self):
        if self._blocks is None:
            cp = self.cp
            device = cp.cuda.Device()
            sms = device.attributes['MultiProcessorCount']
            per_sm = 1
            try:
                per_sm = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                    self.kernel.kernel.ptr, THREADS_PER_BLOCK, 0)
            except Exception:
                per_sm = 1
            needed = (self.n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            self._blocks = int(max(1, min(sms * max(1, per_sm), needed)))
        return self._blocks

    def advance(self, drive: np.ndarray, cursor: int, steps: int, counts_out: np.ndarray) -> int:
        cp = self.cp
        if self._last_drive is None or not np.array_equal(self._last_drive, drive):
            self.d_drive.set(drive)
            self._last_drive = drive.copy()
        self.d_counts.fill(0)
        args = (self.d_ptr, self.d_post, self.d_edge_inc, self.d_v, self.d_g, self.d_acc,
                self.d_refractory, self.d_drive, self.d_queue, self.d_queue_count,
                self.d_counts, self.d_active_flag, self.d_spiked,
                np.int64(self.n), np.int32(self.delay_slots), np.int64(cursor), np.int32(steps),
                np.float64(self.dt), np.float64(self.e_inh))
        self.kernel((self._grid_blocks(),), (THREADS_PER_BLOCK,), args)
        counts_out[...] = self.d_counts.get()
        return int(cursor) + int(steps)
