"""Minimal, deterministic reproductions of the two WP5 dynamics defects.

Defect 1 (RUNAWAY / UNBOUNDED VOLTAGE): the current-based LIF proxy has no
reversal potentials, so synaptic input is a voltage-equivalent current that can
drive the membrane arbitrarily far from rest, and recurrent excitation has no
shunting brake.

Defect 2 (DEAD RIGHT SIDE): DNa02_R never fires, and its membrane sits at -120
to -200 mV.  Hypothesis under test: this is a consequence of defect 1 (the
neuron is driven below any physiological inhibitory reversal by the
self-sustained network state), not of the wiring, whose two-hop products are
near symmetric.

Probes A and B need no connectome.  Probe C needs the pinned MaleCNS graph
(NEUROFLY_GRAPH_DIR) and costs about 30 s of wall time per dynamics version
per direction.

    PYTHONPATH=. .venv/bin/python scripts/lif_dynamics_diagnosis.py \
        --dynamics v1 --out docs/receipts/lif_dynamics_diagnosis.json

Nothing here contacts a running service.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_brain(arrays, dynamics):
    from brainlab.brain import Brain
    return Brain(arrays=arrays, dynamics=dynamics)


def _graph(ptr, post, weight):
    n = len(ptr) - 1
    return dict(ptr=np.asarray(ptr, np.int64), post=np.asarray(post, np.int32),
                weight=np.asarray(weight, np.float32), ids=np.arange(1, n + 1, dtype=np.int64))


# ---------------------------------------------------------------------------
# Probe A: is the membrane potential bounded by a reversal potential?
# ---------------------------------------------------------------------------
def probe_a(dynamics, ms=2000.0, drive=20.0, weights=(-40.0, -400.0, -4000.0)):
    """Two neurons.  0 is driven above threshold; 0 -> 1 is inhibitory.

    Swept over three decades of inhibitory weight.  A conductance-based model
    cannot take neuron 1 below the inhibitory reversal potential however large
    the inhibitory weight; a current-based one has no such bound, so v1's
    minimum scales with the weight without limit.
    """
    rows = []
    for w_inh in weights:
        brain = make_brain(_graph([0, 1, 1], [1], [w_inh]), dynamics)
        currents = np.zeros(2, np.float32)
        currents[0] = drive
        v_min, v_last, spikes = 0.0, 0.0, 0
        for _ in range(int(ms / 10)):
            counts, _ = brain.step(currents, 10.0)
            spikes += int(counts[0])
            v_min = min(v_min, float(brain.v[1]))
            v_last = float(brain.v[1])
        rows.append(dict(inhibitory_weight=w_inh, presynaptic_spikes=spikes,
                         v_post_min_mV=round(v_min, 3), v_post_final_mV=round(v_last, 3)))
    # Unitary EPSP check: one presynaptic spike, excitatory weight 0.275.
    epsp = make_brain(_graph([0, 1, 1], [1], [0.275]), dynamics)
    pulse = np.zeros(2, np.float32)
    pulse[0] = 20.0
    epsp.step(pulse, 10.0)         # drive neuron 0 over threshold exactly once
    peak = -52.0
    for _ in range(400):
        epsp.step(np.zeros(2, np.float32), 0.1)
        peak = max(peak, float(epsp.v[1]))
    return dict(dynamics=dynamics, drive=drive, ms=ms, sweep=rows,
                unitary_epsp_mV=round(peak + 52.0, 5))


# ---------------------------------------------------------------------------
# Probe B: does a transient input leave a self-sustained state?
# ---------------------------------------------------------------------------
def probe_b(dynamics, n=2000, k=40, seed=11, stim_ms=200.0, free_ms=800.0, drive=20.0,
            gains=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Sparse random recurrent net, 80 % excitatory by edge count, brief input.

    The recurrent weight is swept over a gain factor.  The rate during the
    free (no-input) window is the runaway measure: the gain at which a
    transient input leaves a self-sustained state is the quantity compared
    between the two dynamics versions.
    """
    rows = []
    for gain in gains:
        rng = np.random.default_rng(seed)
        post = rng.integers(0, n, size=n * k).astype(np.int32)
        ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
        sign = np.where(rng.random(n * k) < 0.8, 1.0, -1.0)
        weight = (sign * rng.gamma(2.0, 0.6, n * k) * 0.275 * gain).astype(np.float32)
        brain = make_brain(_graph(ptr, post, weight), dynamics)
        stim = np.zeros(n, np.float32)
        stim[: n // 10] = drive
        zero = np.zeros(n, np.float32)
        stim_spikes = free_spikes = 0
        for _ in range(int(stim_ms / 10)):
            counts, _ = brain.step(stim, 10.0)
            stim_spikes += int(counts.sum())
        v_min_free, v_max_free = 1e9, -1e9
        for _ in range(int(free_ms / 10)):
            counts, _ = brain.step(zero, 10.0)
            free_spikes += int(counts.sum())
            v_min_free = min(v_min_free, float(brain.v.min()))
            v_max_free = max(v_max_free, float(brain.v.max()))
        rows.append(dict(weight_gain=gain,
                         mean_net_in_weight=round(float(np.bincount(post.astype(np.int64),
                                                                    weights=weight.astype(np.float64),
                                                                    minlength=n).mean()), 2),
                         stim_rate_hz_per_neuron=round(stim_spikes / (stim_ms / 1000) / n, 3),
                         free_rate_hz_per_neuron=round(free_spikes / (free_ms / 1000) / n, 3),
                         free_total_spikes=free_spikes,
                         v_min_during_free_mV=round(v_min_free, 2),
                         v_max_during_free_mV=round(v_max_free, 2),
                         self_sustained=bool(free_spikes > 0)))
    return dict(dynamics=dynamics, neurons=n, edges_per_neuron=k, excitatory_fraction=0.8,
                stim_ms=stim_ms, free_ms=free_ms, sweep=rows)


# ---------------------------------------------------------------------------
# Probe C: the real graph, two stimulus blocks, DNa02 input budget
# ---------------------------------------------------------------------------
def incoming(shared, node):
    """(edge indices, presynaptic node indices, weights) of edges onto ``node``."""
    edges = np.flatnonzero(shared.arrays['post'] == node)
    pre = np.searchsorted(shared.arrays['ptr'], edges, side='right') - 1
    return edges, pre, shared.arrays['weight'][edges]


def probe_c(dynamics, shared, io, prereg, direction, out_dir, incoming_cache, e_inh=None):
    """500 ms gray + 1 s rotation in one direction + 500 ms gray, DNa02 instrumented."""
    from brainlab.brain import Brain
    from brainlab.io_map import DNa02YawDecoder, OptomotorEncoder

    stim, dec = prereg['stimulus'], prereg['decoder']
    brain = Brain(arrays=shared.arrays, validate=False, dynamics=dynamics, e_inh_mV=e_inh)
    v_net_min, v_net_max = 1e9, -1e9
    rng = np.random.default_rng(0)
    encoder = OptomotorEncoder(io, rng, i_max=stim['i_max'],
                               spatial_period_deg=stim['spatial_period_deg'], noise_sd=stim['noise_sd'])
    DNa02YawDecoder(io, gain_rad_s_per_hz=dec['gain_rad_s_per_hz'], tau_ms=dec['tau_ms'])
    left, right = int(io.populations['DNa02_L'][0]), int(io.populations['DNa02_R'][0])
    budget = {}
    for name, node in (('DNa02_L', left), ('DNa02_R', right)):
        _, pre, w = incoming_cache[name]
        budget[name] = dict(in_edges=int(len(pre)),
                            exc_weight=float(w[w > 0].sum()), inh_weight=float(-w[w < 0].sum()))
    step_ms = stim['step_ms']
    schedule = ([(0.0, 0.0)] * int(round(stim['initial_gray_ms'] / step_ms))
                + [(direction * stim['angular_velocity_rad_s'], 1.0)] * int(round(stim['block_ms'] / step_ms))
                + [(0.0, 0.0)] * int(round(stim['gray_between_ms'] / step_ms)))
    currents = np.zeros(brain.n, np.float32)
    rows = []
    clock = time.perf_counter()
    for i, (slip, contrast) in enumerate(schedule):
        currents.fill(0.0)
        encoder.encode(currents, i * step_ms, slip, contrast)
        counts, _ = brain.step(currents, step_ms)
        arriving = {}
        for name, node in (('DNa02_L', left), ('DNa02_R', right)):
            _, pre, w = incoming_cache[name]
            fired = counts[pre] > 0
            arriving[f'{name}_exc'] = float(w[fired & (w > 0)].sum())
            arriving[f'{name}_inh'] = float(-w[fired & (w < 0)].sum())
        v_net_min = min(v_net_min, float(brain.v.min()))
        v_net_max = max(v_net_max, float(brain.v.max()))
        rows.append(dict(t_ms=(i + 1) * step_ms, slip=slip,
                         v_l=float(brain.v[left]), v_r=float(brain.v[right]),
                         spk_l=int(counts[left]), spk_r=int(counts[right]),
                         total=int(counts.sum()), **arriving))
    wall = time.perf_counter() - clock
    n_gray0 = int(round(stim['initial_gray_ms'] / step_ms))
    n_block = int(round(stim['block_ms'] / step_ms))
    mask_gray0 = np.zeros(len(rows), bool); mask_gray0[:n_gray0] = True
    mask_block = np.zeros(len(rows), bool); mask_block[n_gray0:n_gray0 + n_block] = True
    mask_after = np.zeros(len(rows), bool); mask_after[n_gray0 + n_block:] = True

    def agg(mask):
        sel = [r for r, m in zip(rows, mask) if m]
        dur = len(sel) * step_ms / 1000.0
        return dict(steps=len(sel),
                    v_l_mean=round(float(np.mean([r['v_l'] for r in sel])), 2),
                    v_l_min=round(float(np.min([r['v_l'] for r in sel])), 2),
                    v_r_mean=round(float(np.mean([r['v_r'] for r in sel])), 2),
                    v_r_min=round(float(np.min([r['v_r'] for r in sel])), 2),
                    rate_l_hz=round(sum(r['spk_l'] for r in sel) / dur, 2),
                    rate_r_hz=round(sum(r['spk_r'] for r in sel) / dur, 2),
                    network_rate_hz=round(sum(r['total'] for r in sel) / dur, 1),
                    arriving_exc_weight_L=round(float(np.mean([r['DNa02_L_exc'] for r in sel])), 1),
                    arriving_inh_weight_L=round(float(np.mean([r['DNa02_L_inh'] for r in sel])), 1),
                    arriving_exc_weight_R=round(float(np.mean([r['DNa02_R_exc'] for r in sel])), 1),
                    arriving_inh_weight_R=round(float(np.mean([r['DNa02_R_inh'] for r in sel])), 1))

    tag = f'{dynamics}-dir{direction:+d}' + ('' if e_inh is None else f'-einh{e_inh:g}')
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f'probe-c-{tag}.npz',
                            **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    return dict(dynamics=dynamics, e_inh_mV=brain.e_inh_mV if dynamics != 'v1' else None,
                direction=direction, wall_s=round(wall, 1),
                sim_s_per_wall_s=round(len(schedule) * step_ms / 1000 / wall, 4),
                static_input_budget=budget,
                network_membrane_range_mV=[round(v_net_min, 3), round(v_net_max, 3)],
                initial_gray=agg(mask_gray0), stimulus=agg(mask_block),
                gray_after_stimulus=agg(mask_after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynamics', default='v1', help='comma-separated: v1,v2,v3')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--no-graph', action='store_true', help='probes A and B only')
    parser.add_argument('--e-inh-sensitivity', action='store_true',
                        help='also run probe C at the predeclared E_inh = -56 mV arm (S2)')
    parser.add_argument('--unclear-sensitivity', action='store_true',
                        help='also run probe C with the predeclared unclear_mode=zero arm (S1)')
    args = parser.parse_args()
    versions = [v.strip() for v in args.dynamics.split(',') if v.strip()]

    from brainlab.graph_identity import DYNAMICS_VERSIONS, dynamics_pin
    result = dict(generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host=platform.node(),
                  dynamics_declared={v: DYNAMICS_VERSIONS[v] for v in versions},
                  dynamics_pins={v: dynamics_pin(v) for v in versions},
                  probe_a=[probe_a(v) for v in versions],
                  probe_b=[probe_b(v) for v in versions])
    print(json.dumps(result['probe_a'], indent=1))
    print(json.dumps(result['probe_b'], indent=1))
    if not args.no_graph:
        import hashlib
        from brainlab.io_map import resolve_optomotor_io
        from experiment_registry import SharedGraph
        raw = (ROOT / 'docs/wp5_optomotor_prereg.json').read_bytes()
        prereg = json.loads(raw)
        prereg['_sha256'] = hashlib.sha256(raw).hexdigest()
        from brainlab import transmitter_policy as tp
        shared = SharedGraph.load()
        io = resolve_optomotor_io()
        result['graph'] = shared.identity.to_dict()
        result['prereg_sha256'] = prereg['_sha256']
        result['io_map_sha256'] = io.sha256

        # Declared static fixed-point calculation (docs/LIF_DYNAMICS_SPEC.md
        # §6.5).  No simulation: total excitatory / inhibitory weight under each
        # transmitter policy, and the resulting high-conductance fixed point
        # under each conductance calibration.
        labels = tp.load_transmitters()
        fixed_points = {}
        graphs = {'legacy': (shared, None)}
        for name, (policy, mode) in (('legacy', (tp.POLICY_LEGACY, 'excitatory')),
                                     ('v3-primary', (tp.POLICY_V3, 'excitatory')),
                                     ('v3-unclear-zero', (tp.POLICY_V3, 'zero')),
                                     ('v3-unclear-exclude', (tp.POLICY_V3, 'exclude'))):
            _, report = tp.apply_policy(shared.arrays['ptr'], shared.arrays['post'],
                                        shared.arrays['weight'], labels,
                                        policy=policy, unclear_mode=mode)
            fixed_points[name] = report
        result['fixed_point'] = fixed_points
        print(json.dumps({k: {kk: v[kk] for kk in ('edges_zeroed', 'v2_quanta', 'v3_quanta')}
                          for k, v in fixed_points.items()}, indent=1), flush=True)

        arms = [(v, None, 'excitatory') for v in versions]
        if 'v2' in versions and args.e_inh_sensitivity:
            # Predeclared sensitivity arm (docs/LIF_DYNAMICS_SPEC.md §3.1): the
            # measured Drosophila larval chloride reversal, diagnostic only.
            arms.append(('v2', -56.0, 'excitatory'))
        if 'v3' in versions and args.e_inh_sensitivity:
            arms.append(('v3', -56.0, 'excitatory'))        # declared arm S2
        if 'v3' in versions and args.unclear_sensitivity:
            arms.append(('v3', None, 'zero'))               # declared arm S1
        result['probe_c'] = []
        result['probe_c_policies'] = {}
        for v, e_inh, mode in arms:
            key = ('v3', mode) if v == 'v3' else ('legacy', 'excitatory')
            if key not in graphs:
                g, report = tp.apply_to_shared(shared, policy=tp.POLICY_V3, unclear_mode=mode)
                graphs[key] = (g, report)
                result['probe_c_policies'][f'v3:{mode}'] = report
            graph, _ = graphs[key]
            cache = {name: incoming(graph, int(io.populations[name][0]))
                     for name in ('DNa02_L', 'DNa02_R')}
            for d in (+1, -1):
                row = probe_c(v, graph, io, prereg, d, args.traces, cache, e_inh=e_inh)
                row['transmitter_policy'] = tp.describe(
                    tp.POLICY_V3 if v == 'v3' else tp.POLICY_LEGACY, mode)
                row['graph_sha256'] = graph.identity.graph_sha256
                result['probe_c'].append(row)
                print(json.dumps({k: row[k] for k in ('dynamics', 'e_inh_mV', 'direction', 'wall_s',
                                                      'stimulus', 'gray_after_stimulus')}, indent=1),
                      flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
