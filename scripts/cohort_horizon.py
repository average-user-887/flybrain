"""Long-horizon CPU/GPU characterisation on the REAL graph (rc2 Lane 3).

Compares the CPU reference (``CpuLoopCohortEngine`` = ``engine.advance_v3``) against a candidate that
sees exactly the same inputs, in one of two modes that are NEVER mixed in a report:

* ``emulation``  - a CPU emulation of the GPU kernel's arrival order (ascending (pre, edge), per-arrival
  float32 via double), after the probe approach of the F2 adversarial review.  CPU only; no CUDA.
* ``cuda``       - the actual ``GpuCohortEngine``.  Needs the GPU; run only when told the GPU is free.

The inputs are the real optomotor loop of the cohort runner (``FlyWorld``: Arena + OptomotorEncoder +
DNa02YawDecoder, one arena step = 20 ms = 200 ticks of 0.1 ms).  The CPU reference drives the loop;
the candidate is an open-loop replay of the SAME per-step drive, so a candidate divergence never
changes the stimulus.  Preregistration: docs/cohort_horizon_prereg.json (committed before any run).

Observation resolution is ``--chunk`` ticks per engine call: counts and state are compared after each
call.  With chunk=1 (emulation default) a first divergence is an exact tick; with chunk>1 it is only
"somewhere in ticks [a, b]" and is reported that way.  Nothing is tuned and no gate is relaxed: bounds are
the cohort contract's (max|dV| 1e-3 mV, g relative 1e-6, g==0 -> |g|<=1e-12).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

TICKS_PER_STEP = 200            # 20 ms arena step / 0.1 ms tick
DV_BOUND_MV, G_REL_BOUND, G_ZERO_ABS = 1e-3, 1e-6, 1e-12


def start_rss_guard(limit_gb: float):
    limit = int(limit_gb * 1024 ** 3)
    page = os.sysconf('SC_PAGE_SIZE')
    peak = {'rss': 0}

    def watch():
        while True:
            with open('/proc/self/statm') as fh:
                rss = int(fh.read().split()[1]) * page
            peak['rss'] = max(peak['rss'], rss)
            if rss > limit:
                sys.stderr.write(f'RSS {rss/1e9:.2f} GB exceeds the {limit_gb} GB bound; aborting\n')
                os._exit(97)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()
    return peak


def build_emulation_kernel():
    import numba
    from numba import njit
    from brainlab.engine import V_REST_MV, V_RESET_MV, V_THRESHOLD_MV, E_EXC_MV, TAU_M_MS, TAU_SYN_MS

    @njit(cache=False)
    def gpu_like(ptr, post, in_ptr, in_pre, in_w, v, g, refractory, drive, queue, queue_count, cursor, steps,
                 dt, counts, active_flag, spiked, due, recv, e_inh, gue, gui):
        ag = math.exp(-dt / TAU_SYN_MS); slots = queue.shape[0]; n = v.shape[0]
        delay = int(round(1.8 / dt)); refr = int(round(2.2 / dt))
        for step in range(steps):
            for i in range(n):                                   # phase 1
                if active_flag[i] == 0:
                    if drive[i] != 0.0: active_flag[i] = 1
                    else: continue
                if refractory[i] > 0: refractory[i] -= 1
                if refractory[i] == 0:
                    ge = g[0, i]; gi = g[1, i]
                    gtot = 1.0 + ge + gi
                    vinf = (V_REST_MV + ge * E_EXC_MV + gi * e_inh + drive[i]) / gtot
                    vi = vinf + (v[i] - vinf) * math.exp(-dt * gtot / TAU_M_MS)
                    if vi < e_inh: vi = e_inh
                    elif vi > E_EXC_MV: vi = E_EXC_MV
                    v[i] = vi; g[0, i] = ge * ag; g[1, i] = gi * ag
                    if vi > V_THRESHOLD_MV:
                        counts[i] += 1; spiked[i] = 1
                        fut = (cursor + delay) % slots
                        queue[fut, queue_count[fut]] = i; queue_count[fut] += 1
            stamp = cursor + 1; slot = cursor % slots            # phase 2a
            for q in range(queue_count[slot]):
                pre = queue[slot, q]; due[pre] = stamp
                for e in range(ptr[pre], ptr[pre + 1]):
                    j = post[e]
                    if refractory[j] > 0: continue
                    recv[j] = stamp
            queue_count[slot] = 0
            for i in range(n):                                   # phase 2b + 3
                if recv[i] == stamp:
                    ge = g[0, i]; gi = g[1, i]
                    for e in range(in_ptr[i], in_ptr[i + 1]):
                        if due[in_pre[e]] != stamp: continue
                        w = in_w[e]
                        if w > 0.0: ge = np.float32(np.float64(ge) + np.float64(w) * gue)
                        else: gi = np.float32(np.float64(gi) - np.float64(w) * gui)
                    g[0, i] = ge; g[1, i] = gi; active_flag[i] = 1
                if spiked[i]:
                    spiked[i] = 0; v[i] = V_RESET_MV; g[0, i] = 0.0; g[1, i] = 0.0; refractory[i] = refr
            cursor += 1
        return cursor
    return gpu_like


class EmulatedOrderEngine:
    """CPU emulation of the GPU kernel's per-arrival float32 ascending (pre, edge) order.  EMULATION."""

    backend_id = 'cpu-emulation-of-gpu-order'

    def __init__(self, arrays, ref_engine):
        self.kernel = build_emulation_kernel()
        n = ref_engine.n
        self.n, self.n_flies = n, ref_engine.n_flies
        self.ptr, self.post = arrays['ptr'], arrays['post']
        order = np.argsort(self.post, kind='stable')
        src = np.repeat(np.arange(n, dtype=np.int32), np.diff(self.ptr))
        self.in_ptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(self.post, minlength=n), out=self.in_ptr[1:])
        self.in_pre = src[order]
        self.in_w = np.ascontiguousarray(np.asarray(arrays['weight'])[order], dtype=np.float32)
        del order, src
        self.flies = []
        for b in ref_engine.brains:                       # identical fresh start (snapshot of tick 0)
            s = b.snapshot_state()
            self.flies.append(dict(
                v=s['v'].copy(), g=s['g'].copy(), refractory=s['refractory'].copy(), queue=s['queue'].copy(),
                queue_count=s['queue_count'].copy(), active_flag=s['active_flag'].copy(),
                spiked=np.zeros(n, np.uint8), due=np.zeros(n, np.int64), recv=np.zeros(n, np.int64), cursor=0,
                dt=b.dt, e_inh=b.e_inh_mV, gue=b.g_unit_exc, gui=b.g_unit_inh))

    def step(self, drive, ticks):
        out = np.zeros((self.n_flies, self.n), np.int32)
        for k, f in enumerate(self.flies):
            f['cursor'] = self.kernel(self.ptr, self.post, self.in_ptr, self.in_pre, self.in_w, f['v'], f['g'],
                                      f['refractory'], drive[k], f['queue'], f['queue_count'], f['cursor'],
                                      int(ticks), f['dt'], out[k], f['active_flag'], f['spiked'], f['due'],
                                      f['recv'], f['e_inh'], f['gue'], f['gui'])
        return out

    def view(self, k):
        f = self.flies[k]
        return f['v'], f['g'], f['refractory']


class CudaEngineView:
    def __init__(self, engine):
        self.engine = engine
        self.backend_id = engine.backend_id

    def step(self, drive, ticks):
        return self.engine.step(drive, ticks)

    def view(self, k):
        s = self.engine.read_state(k)
        return s['v'], s['g'], s['refractory']


def compare_state(ref_brain, cv, cg, cr):
    rv, rg = ref_brain.v, ref_brain.g
    dv = float(np.abs(cv.astype(np.float64) - rv.astype(np.float64)).max())
    # Reference subnormals count as zero: CuPy compiles with -ftz=true, so the GPU flushes
    # subnormal float32 g to 0 where the CPU keeps it (see docs/COHORT_HORIZON.md).
    nz = np.abs(rg) >= np.finfo(np.float32).tiny
    rel = 0.0
    if nz.any():
        rel = float((np.abs(cg[nz].astype(np.float64) - rg[nz].astype(np.float64))
                     / np.abs(rg[nz].astype(np.float64))).max())
    zabs = float(np.abs(cg[~nz]).max()) if (~nz).any() else 0.0
    return dv, rel, zabs, bool(np.array_equal(cr, ref_brain.refractory))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mode', choices=('emulation', 'cuda'), required=True)
    ap.add_argument('--graph-dir', default=None, help='directory holding the real MaleCNS v1 graph')
    ap.add_argument('--out', required=True, help='evidence directory (created)')
    ap.add_argument('--prereg', default=str(ROOT / 'docs' / 'cohort_horizon_prereg.json'))
    ap.add_argument('--ticks', type=int, required=True)
    ap.add_argument('--seeds', default='1,2,3,4,5,6,7,8', help='fly seeds; fly k of the run has this seed')
    ap.add_argument('--chunk', type=int, default=None, help='ticks per engine call (default 1 emulation, 200 cuda)')
    ap.add_argument('--curve-every', type=int, default=200)
    ap.add_argument('--mem-limit-gb', type=float, default=3.0)
    ap.add_argument('--label', default='run')
    ap.add_argument('--test-synthetic-graph', action='store_true', help='SMOKE ONLY: tiny synthetic graph, never a result')
    a = ap.parse_args(argv)
    chunk = a.chunk or (1 if a.mode == 'emulation' else TICKS_PER_STEP)
    if TICKS_PER_STEP % chunk or a.ticks % TICKS_PER_STEP:
        sys.exit('chunk must divide 200 and ticks must be a multiple of 200 (whole 20 ms arena steps)')
    if a.mode == 'emulation' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        sys.exit('emulation mode is CPU only: set CUDA_VISIBLE_DEVICES="" explicitly')
    peak = start_rss_guard(a.mem_limit_gb)
    seeds = [int(s) for s in a.seeds.split(',')]
    B = len(seeds)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    prereg_sha = hashlib.sha256(Path(a.prereg).read_bytes()).hexdigest()

    from brainlab.cohort.runner import FlyWorld, load_real_graph
    from brainlab.cohort.api import CpuLoopCohortEngine
    t0 = time.time()
    if a.test_synthetic_graph:
        from brainlab.cohort.runner import synthetic_graph
        graph = synthetic_graph(n=400, k_out=12, seed=3)
    else:
        if not a.graph_dir:
            sys.exit('--graph-dir is required for a real-graph run')
        graph = load_real_graph(a.graph_dir)
    ref = CpuLoopCohortEngine(graph.arrays, B)
    if a.mode == 'emulation':
        cand = EmulatedOrderEngine(graph.arrays, ref)
    else:
        from brainlab.cohort.gpu import GpuCohortEngine
        cand = CudaEngineView(GpuCohortEngine(graph.arrays, B))
    worlds = [FlyWorld(k, s, graph, 'optomotor', 20.0) for k, s in enumerate(seeds)]
    n = graph.n
    setup_s = time.time() - t0
    print(f'[{a.label}] mode={a.mode} flies={B} seeds={seeds} ticks={a.ticks} chunk={chunk} n={n} '
          f'setup {setup_s:.0f}s rss {peak["rss"]/1e9:.2f} GB', flush=True)

    first_spike = [None] * B       # dict(tick_end, tick_start, neurons, ...)
    first_dv = [None] * B          # first observation with |dV| > bound
    first_g = [None] * B
    first_gzero = [None] * B
    first_refr = [None] * B
    spike_mism_total = [0] * B
    max_dv = [0.0] * B; max_g = [0.0] * B; max_gzero = [0.0] * B           # whole run
    pre_dv = [0.0] * B; pre_g = [0.0] * B                                  # up to first spike divergence
    ref_spikes = [0] * B
    curve = []                     # rows per curve boundary
    drive = np.zeros((B, n), np.float32)
    tick = 0
    t_run = time.time()
    last_ckpt = 0
    for step_i in range(a.ticks // TICKS_PER_STEP):
        drive.fill(0.0)
        inputs = []
        for k, w in enumerate(worlds):
            slip, contrast = w.next_input()
            totals = w.encoder.encode(drive[k], w.cursor * w.step_ms, slip, contrast)
            inputs.append((slip, contrast, totals))
        ref_counts = np.zeros((B, n), np.int64)
        for _ in range(TICKS_PER_STEP // chunk):
            rc = ref.step(drive, chunk)
            cc = cand.step(drive, chunk)
            tick += chunk
            ref_counts += rc
            for k in range(B):
                ref_spikes[k] += int(rc[k].sum())
                diff = np.flatnonzero(rc[k] != cc[k])
                if len(diff):
                    spike_mism_total[k] += int(len(diff))
                    if first_spike[k] is None:
                        first_spike[k] = dict(tick_end=tick, tick_start=tick - chunk + 1, exact=(chunk == 1),
                                              n_neurons=int(len(diff)), neurons=[int(x) for x in diff[:10]],
                                              ref_counts=[int(rc[k][x]) for x in diff[:10]],
                                              cand_counts=[int(cc[k][x]) for x in diff[:10]])
                cv, cg, cr = cand.view(k)
                dv, rel, zabs, req = compare_state(ref.brains[k], cv, cg, cr)
                max_dv[k] = max(max_dv[k], dv); max_g[k] = max(max_g[k], rel); max_gzero[k] = max(max_gzero[k], zabs)
                if first_spike[k] is None:
                    pre_dv[k] = max(pre_dv[k], dv); pre_g[k] = max(pre_g[k], rel)
                if dv > DV_BOUND_MV and first_dv[k] is None: first_dv[k] = tick
                if rel > G_REL_BOUND and first_g[k] is None: first_g[k] = tick
                if zabs > G_ZERO_ABS and first_gzero[k] is None: first_gzero[k] = tick
                if not req and first_refr[k] is None: first_refr[k] = tick
            if tick % a.curve_every == 0:
                curve.append(dict(tick=tick, max_g_rel=list(max_g), max_dv_mV=list(max_dv),
                                  spike_mismatch_neuron_obs=list(spike_mism_total),
                                  frac_spike_diverged=sum(x is not None for x in first_spike) / B,
                                  frac_g_over_bound=sum(x is not None for x in first_g) / B,
                                  frac_dv_over_bound=sum(x is not None for x in first_dv) / B,
                                  ref_spikes=list(ref_spikes)))
        for k, w in enumerate(worlds):                       # decode + advance the Arena, as Cohort.step
            slip, contrast, totals = inputs[k]
            motor = w.decoder.decode(ref_counts[k], w.step_ms)
            w.controller.pending = (slip, contrast, motor)
            w.arena.step(w.step_ms / 1000.0)
            w.cursor += 1
        if tick - last_ckpt >= 2000 or tick == a.ticks:
            last_ckpt = tick
            print(f'[{a.label}] tick {tick} wall {time.time()-t_run:.0f}s rss {peak["rss"]/1e9:.2f} GB '
                  f'first_spike={[(x or {}).get("tick_end") for x in first_spike]} '
                  f'maxg={max(max_g):.2e} maxdv={max(max_dv):.2e}', flush=True)

    result = dict(
        schema='neurofly.cohort-horizon-result.v1', label=a.label, mode=a.mode.upper(),
        prereg_sha256=prereg_sha, code_commit_note='see git log of scripts/cohort_horizon.py',
        graph_sha256=graph.graph_sha256, io_map_sha256=graph.io_map_sha256, n_neurons=n,
        seeds=seeds, ticks=a.ticks, chunk_ticks=chunk, tick_ms=0.1, arena_step_ms=20.0,
        ticks_per_arena_step=TICKS_PER_STEP,
        resolution=('exact tick (per-tick comparison)' if chunk == 1 else
                    f'chunk of {chunk} ticks: first divergence is only known to lie in [tick_start, tick_end]'),
        bounds=dict(max_abs_dV_mV=DV_BOUND_MV, g_rel=G_REL_BOUND, g_zero_abs=G_ZERO_ABS),
        per_fly=[dict(fly=k, seed=seeds[k], first_spike_divergence=first_spike[k],
                      spike_mismatch_neuron_observations=spike_mism_total[k], ref_total_spikes=ref_spikes[k],
                      max_abs_dV_mV=max_dv[k], max_g_rel=max_g[k], max_abs_g_where_ref_zero=max_gzero[k],
                      max_abs_dV_mV_before_first_spike_divergence=pre_dv[k],
                      max_g_rel_before_first_spike_divergence=pre_g[k],
                      first_obs_tick_dV_over_bound=first_dv[k], first_obs_tick_g_rel_over_bound=first_g[k],
                      first_obs_tick_g_zero_violation=first_gzero[k], first_obs_tick_refractory_unequal=first_refr[k])
                 for k in range(B)],
        divergence_fraction_curve=curve,
        wall_s=time.time() - t_run, setup_s=setup_s, peak_rss_bytes=peak['rss'])
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    import csv
    with open(out / 'curve.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['tick', 'frac_spike_diverged', 'frac_g_over_bound', 'frac_dv_over_bound',
                    'max_g_rel_any_fly', 'max_dv_mV_any_fly', 'spike_mismatch_neuron_obs_total'])
        for r in curve:
            w.writerow([r['tick'], r['frac_spike_diverged'], r['frac_g_over_bound'], r['frac_dv_over_bound'],
                        max(r['max_g_rel']), max(r['max_dv_mV']), sum(r['spike_mismatch_neuron_obs'])])
    print(f'[{a.label}] done; wrote {out}', flush=True)


if __name__ == '__main__':
    main()
