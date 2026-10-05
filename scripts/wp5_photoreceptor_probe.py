"""Photoreceptor-encoder probe: does the GRAPH compute direction selectivity?

Pre-registration: ``docs/PHOTORECEPTOR_ENCODER.md`` (locked copy
``docs/receipts/photoreceptor_encoder_declaration.locked.md``).  This script runs
the cheap measurement of §5 and produces the §4 diagnostic:

* schedule identical to probe C of ``docs/LIF_DYNAMICS_SPEC.md`` §6.8 — 500 ms
  gray, 1 s rotation, 500 ms gray, contrast 1.0, seed 0, one run per direction,
* drive delivered to ``R1-R6`` photoreceptors only (``brainlab/io_map_photoreceptor``),
* the whole pathway recorded per cell type and annotated side,
* the Q1/R1/R2 gate evaluated by code copied **verbatim** from
  ``outputs/wp5/v3-verify-20260926/gate.py``,
* a direction-selectivity index per population, and the stage-by-stage table with
  membrane and first-spike evidence that §4 requires if the signal dies.

    NEUROFLY_GRAPH_DIR=<graph dir> PYTHONPATH=. .venv/bin/python \
        scripts/wp5_photoreceptor_probe.py --out outputs/wp5/<dir> [--arm P1,P2] \
        [--e-inh -60] [--dynamics v3]

Nothing here contacts a running service, and no existing module is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The declared dynamics must be selected BEFORE brainlab is imported (same reason
# as scripts/wp5_optomotor.py: the run identity is resolved at import time).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--dynamics', default='v3')
_pre.add_argument('--e-inh', default=None)
_pre_args = _pre.parse_known_args()[0]
os.environ['NEUROFLY_LIF_DYNAMICS'] = _pre_args.dynamics
if _pre_args.e_inh is not None:
    os.environ['NEUROFLY_LIF_E_INH_MV'] = str(float(_pre_args.e_inh))

from brainlab.graph_identity import (DYNAMICS_VERSIONS, active_dynamics_version,  # noqa: E402
                                     active_e_inh_mV, dynamics_pin, dynamics_variant,
                                     dynamics_variant_pin)
from brainlab.io_map import DNa02YawDecoder  # noqa: E402
from brainlab.io_map_photoreceptor import (DELTA_PHI_DEG, LAYOUT_RULE, PATHWAY_MONITORS,  # noqa: E402
                                           PATHWAY_STAGES, PHOTORECEPTOR_IO_PIN,
                                           PhotoreceptorGratingEncoder,
                                           direction_selectivity_index, resolve_photoreceptor_io)
from experiment_registry import SharedGraph  # noqa: E402
from neurofly.privacy import host_description, redact_local  # noqa: E402

PREREG = ROOT / 'docs/wp5_optomotor_prereg.json'
DECLARATION = ROOT / 'docs/PHOTORECEPTOR_ENCODER.md'
DECLARATION_LOCK = ROOT / 'docs/receipts/photoreceptor_encoder_declaration.locked.md'


# ---------------------------------------------------------------------------
# Gate: copied VERBATIM from outputs/wp5/v3-verify-20260926/gate.py (which
# implements docs/LIF_DYNAMICS_SPEC.md §6.7).  Not edited, not relaxed.
# ---------------------------------------------------------------------------
def gate(rows):
    """Both directions of one arm -> the three criteria."""
    by_dir = {r['direction']: r for r in rows}
    out = {}
    q1 = True
    for d, r in by_dir.items():
        for w in ('initial_gray', 'gray_after_stimulus'):
            g = r[w]
            q1 &= (g['network_rate_hz'] < 1.0e5 and g['rate_l_hz'] < 20 and g['rate_r_hz'] < 20)
    diffs = {}
    r1_sign = True
    r1_mag = False
    for d, r in by_dir.items():
        s = r['stimulus']
        diff = s['rate_l_hz'] - s['rate_r_hz']
        diffs[d] = diff
        r1_sign &= (s['network_rate_hz'] > 0 and (diff * d) > 0)
        r1_mag |= abs(diff) >= 1.0
    def vmin(r):
        rng = r.get('network_membrane_range_mV')
        if rng is not None:
            return min(rng)
        return min(r[w][k] for w in ('initial_gray', 'stimulus', 'gray_after_stimulus')
                   for k in ('v_l_min', 'v_r_min'))
    r2 = all(vmin(r) >= (r.get('e_inh_mV') or -70.0) - 1e-6 for r in by_dir.values())
    out['membrane_min_mV'] = {str(d): vmin(r) for d, r in by_dir.items()}
    out.update(Q1=bool(q1), R1=bool(r1_sign and r1_mag), R2=bool(r2),
               dna02_LminusR_per_direction={str(k): v for k, v in diffs.items()},
               gray=[{**{'direction': d, 'window': w}, **{k: r[w][k] for k in
                        ('network_rate_hz', 'rate_l_hz', 'rate_r_hz', 'v_l_min', 'v_r_min')}}
                     for d, r in by_dir.items() for w in ('initial_gray', 'gray_after_stimulus')],
               stimulus=[{**{'direction': d}, **{k: r['stimulus'][k] for k in
                            ('network_rate_hz', 'rate_l_hz', 'rate_r_hz')},
                          'membrane_range_mV': r.get('network_membrane_range_mV'),
                          'sim_s_per_wall_s': r['sim_s_per_wall_s']}
                         for d, r in by_dir.items()])
    return out


# ---------------------------------------------------------------------------


def in_weight_budget(arrays, nodes, driven_set=None):
    """(Sigma exc, Sigma inh, Sigma from driven) of edges onto ``nodes``."""
    post, weight, ptr = arrays['post'], arrays['weight'], arrays['ptr']
    mask = np.isin(post, nodes)
    edges = np.flatnonzero(mask)
    if not len(edges):
        return dict(exc_weight=0.0, inh_weight=0.0, from_driven_weight=0.0, in_edges=0)
    pre = np.searchsorted(ptr, edges, side='right') - 1
    w = weight[edges]
    out = dict(in_edges=int(len(edges)), exc_weight=float(w[w > 0].sum()),
               inh_weight=float(-w[w < 0].sum()))
    if driven_set is not None:
        sel = np.isin(pre, driven_set)
        out['from_driven_weight'] = float(w[sel].sum())
        out['from_driven_edges'] = int(sel.sum())
    return out


def run_direction(brain, shared, io, prereg, direction, arm, seed=0):
    """500 ms gray + 1 s rotation + 500 ms gray, whole pathway instrumented."""
    stim, dec = prereg['stimulus'], prereg['decoder']
    step_ms = stim['step_ms']
    rng = np.random.default_rng(seed)
    encoder = PhotoreceptorGratingEncoder(io, rng, i_max=stim['i_max'],
                                          spatial_period_deg=stim['spatial_period_deg'],
                                          noise_sd=stim['noise_sd'], arm=arm)
    decoder = DNa02YawDecoder(io, gain_rad_s_per_hz=dec['gain_rad_s_per_hz'], tau_ms=dec['tau_ms'])
    left, right = int(io.populations['DNa02_L'][0]), int(io.populations['DNa02_R'][0])
    schedule = ([(0.0, 0.0)] * int(round(stim['initial_gray_ms'] / step_ms))
                + [(direction * stim['angular_velocity_rad_s'], 1.0)] * int(round(stim['block_ms'] / step_ms))
                + [(0.0, 0.0)] * int(round(stim['gray_between_ms'] / step_ms)))
    n_gray0 = int(round(stim['initial_gray_ms'] / step_ms))
    n_block = int(round(stim['block_ms'] / step_ms))
    windows = dict(initial_gray=(0, n_gray0), stimulus=(n_gray0, n_gray0 + n_block),
                   gray_after_stimulus=(n_gray0 + n_block, len(schedule)))
    names = sorted(io.monitors)
    currents = np.zeros(brain.n, np.float32)
    per_step = {k: np.zeros(len(schedule), np.int64) for k in names}
    trace = {k: np.zeros(len(schedule)) for k in ('total', 'spk_l', 'spk_r', 'v_l', 'v_r', 'yaw')}
    v_net_min, v_net_max = 1e9, -1e9
    # per-window accumulators: spikes per neuron, and max V reached per neuron
    acc = {w: np.zeros(brain.n, np.int64) for w in windows}
    vmax = {w: np.full(brain.n, -1e9, np.float32) for w in windows}
    vmin = {w: np.full(brain.n, +1e9, np.float32) for w in windows}
    clock = time.perf_counter()
    for i, (slip, contrast) in enumerate(schedule):
        currents.fill(0.0)
        encoder.encode(currents, i * step_ms, slip, contrast)
        counts, _ = brain.step(currents, step_ms)
        w = ('initial_gray' if i < n_gray0 else
             'stimulus' if i < n_gray0 + n_block else 'gray_after_stimulus')
        acc[w] += counts
        np.maximum(vmax[w], brain.v, out=vmax[w])
        np.minimum(vmin[w], brain.v, out=vmin[w])
        for k in names:
            per_step[k][i] = counts[io.monitors[k]].sum()
        motor = decoder.decode(counts, step_ms)
        trace['total'][i] = int(counts.sum())
        trace['spk_l'][i] = int(counts[left]); trace['spk_r'][i] = int(counts[right])
        trace['v_l'][i] = float(brain.v[left]); trace['v_r'][i] = float(brain.v[right])
        trace['yaw'][i] = motor['yaw_rad_s']
        v_net_min = min(v_net_min, float(brain.v.min()))
        v_net_max = max(v_net_max, float(brain.v.max()))
    wall = time.perf_counter() - clock

    def agg(window):
        a, b = windows[window]
        dur = (b - a) * step_ms / 1000.0
        pops = {}
        for k in names:
            idx = io.monitors[k]
            if not len(idx):
                continue
            fired = acc[window][idx]
            pops[k] = dict(n=int(len(idx)),
                           rate_hz=round(float(fired.sum() / dur / len(idx)), 4),
                           frac_fired=round(float((fired > 0).mean()), 4),
                           v_max_mV=round(float(vmax[window][idx].max()), 3),
                           v_max_median_mV=round(float(np.median(vmax[window][idx])), 3),
                           v_min_mV=round(float(vmin[window][idx].min()), 3),
                           v_min_median_mV=round(float(np.median(vmin[window][idx])), 3),
                           frac_above_rest=round(float((vmax[window][idx] > -52.0).mean()), 4),
                           frac_below_rest=round(float((vmin[window][idx] < -52.0).mean()), 4))
            first = np.flatnonzero(per_step[k][a:b] > 0)
            pops[k]['first_spike_ms'] = (round(float((first[0] + 1) * step_ms), 1)
                                         if len(first) else None)
        return dict(steps=b - a,
                    network_rate_hz=round(float(trace['total'][a:b].sum() / dur), 1),
                    rate_l_hz=round(float(trace['spk_l'][a:b].sum() / dur), 2),
                    rate_r_hz=round(float(trace['spk_r'][a:b].sum() / dur), 2),
                    v_l_mean=round(float(trace['v_l'][a:b].mean()), 2),
                    v_l_min=round(float(trace['v_l'][a:b].min()), 2),
                    v_r_mean=round(float(trace['v_r'][a:b].mean()), 2),
                    v_r_min=round(float(trace['v_r'][a:b].min()), 2),
                    mean_yaw_rad_s=round(float(trace['yaw'][a:b].mean()), 6),
                    populations=pops)

    return dict(dynamics=brain.dynamics, e_inh_mV=brain.e_inh_mV if brain.dynamics != 'v1' else None,
                arm=arm, direction=direction, seed=seed, wall_s=round(wall, 1),
                sim_s_per_wall_s=round(len(schedule) * step_ms / 1000 / wall, 4),
                network_membrane_range_mV=[round(v_net_min, 3), round(v_net_max, 3)],
                encoder=encoder.describe(),
                **{w: agg(w) for w in windows})


def stage_table(rows_by_dir, e_inh):
    """§4 item 1-3: per-stage rates, DSI, first silent stage and attribution."""
    pos, neg = rows_by_dir[+1]['stimulus']['populations'], rows_by_dir[-1]['stimulus']['populations']
    table, dsi = [], {}
    for stage, groups in PATHWAY_STAGES:
        for group in groups:
            for side in ('L', 'R'):
                key = f'{group}_{side}'
                if key not in pos:
                    continue
                p, q = pos[key], neg[key]
                d = direction_selectivity_index(p['rate_hz'], q['rate_hz'])
                dsi[key] = d
                table.append(dict(stage=stage, population=key, n=p['n'],
                                  rate_sPos_hz=p['rate_hz'], rate_sNeg_hz=q['rate_hz'],
                                  frac_fired_sPos=p['frac_fired'], frac_fired_sNeg=q['frac_fired'],
                                  DSI=(None if d is None else round(d, 4)),
                                  v_max_sPos_mV=p['v_max_mV'], v_max_median_sPos_mV=p['v_max_median_mV'],
                                  v_min_sPos_mV=p['v_min_mV'], v_min_median_sPos_mV=p['v_min_median_mV'],
                                  frac_above_rest_sPos=p['frac_above_rest'],
                                  frac_below_rest_sPos=p['frac_below_rest'],
                                  first_spike_sPos_ms=p['first_spike_ms'],
                                  first_spike_sNeg_ms=q['first_spike_ms']))
    first_silent = None
    for stage, groups in PATHWAY_STAGES:
        live = [r for r in table if r['stage'] == stage]
        if live and all(r['rate_sPos_hz'] == 0 and r['rate_sNeg_hz'] == 0 for r in live):
            first_silent = stage
            break
    t45 = [r for r in table if r['stage'] == (first_silent or '')]
    attribution = None
    if first_silent is not None:
        # Rule (a) of §4 requires the stage to BE responding, subthreshold: its
        # membrane must rise above V_rest without reaching -45 mV.  A stage whose
        # max V never leaves V_rest is not responding at all, which is rule (b).
        subthreshold = [r for r in t45 if -52.0 < r['v_max_median_sPos_mV'] < -45.0]
        hyperpol = [r for r in t45 if r['v_max_median_sPos_mV'] <= -52.0]
        if len(subthreshold) >= max(1, len(t45) // 2):
            attribution = ('(a) graded cells never reach threshold: the first silent stage responds '
                           'subthreshold and the engine discards it')
        elif len(hyperpol) >= max(1, len(t45) // 2):
            attribution = ('(b) inhibition into silence: the first silent stage sits at or below '
                           'V_rest, so the arriving weight is inhibitory and there is no baseline')
        else:
            attribution = '(d) something else: see the per-population membrane numbers'
    else:
        t45 = [r for r in table if r['stage'] == 'T4/T5']
        if t45 and all(r['DSI'] is not None and abs(r['DSI']) < 0.05 for r in t45):
            attribution = ('(c) no tens-of-milliseconds delay mechanism: the pathway conducts to '
                           'T4/T5 but |DSI| < 0.05 in every subtype')
    return dict(first_silent_stage=first_silent, attribution=attribution, dsi=dsi, table=table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', default='P1,P2', help='declared encoder arms to run (P1 primary)')
    parser.add_argument('--dynamics', default='v3', choices=('v1', 'v2', 'v3'))
    parser.add_argument('--e-inh', default=None)
    parser.add_argument('--unclear-mode', default='excitatory', choices=('excitatory', 'zero', 'exclude'))
    args = parser.parse_args()
    assert args.dynamics == active_dynamics_version()
    assert (None if args.e_inh is None else float(args.e_inh)) == active_e_inh_mV()

    raw = PREREG.read_bytes()
    prereg = json.loads(raw)
    prereg_sha = hashlib.sha256(raw).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)

    from brainlab.brain import Brain
    from brainlab import transmitter_policy as tp

    shared = SharedGraph.load()
    io = resolve_photoreceptor_io(arrays=shared.arrays)     # pinned, released graph
    assert io.sha256 == PHOTORECEPTOR_IO_PIN
    driven = np.concatenate([io.photoreceptors[k] for k in sorted(io.photoreceptors)])
    policy_graph, policy_report = tp.apply_to_shared(shared, policy=tp.POLICY_V3,
                                                     unclear_mode=args.unclear_mode)
    e_inh = -70.0 if args.e_inh is None else float(args.e_inh)

    # Structural evidence, computed from the graph and independent of any response.
    budgets = {}
    for group in ('L1', 'L2', 'L3', 'Mi1', 'Tm3', 'Tm1', 'Tm2', 'Tm9', 'Mi9', 'Mi4',
                  'T4a', 'T4b', 'T5a', 'T5b', 'HS', 'H2', 'DNa02'):
        for side in ('L', 'R'):
            key = f'{group}_{side}'
            if len(io.monitors.get(key, ())):
                budgets[key] = in_weight_budget(policy_graph.arrays, io.monitors[key], driven)

    header = dict(
        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), host=host_description(),
        declaration=dict(document='docs/PHOTORECEPTOR_ENCODER.md',
                         locked_copy='docs/receipts/photoreceptor_encoder_declaration.locked.md',
                         sha256=hashlib.sha256(DECLARATION_LOCK.read_bytes()).hexdigest()),
        prereg_sha256=prereg_sha, io_map=io.describe(), io_map_pin=PHOTORECEPTOR_IO_PIN,
        layout_rule=LAYOUT_RULE, delta_phi_deg=DELTA_PHI_DEG,
        graph=shared.identity.to_dict(), policy_graph_sha256=policy_graph.identity.graph_sha256,
        transmitter_policy={k: policy_report[k] for k in
                            ('policy', 'unclear_mode', 'edges_zeroed', 'neurons_modulatory', 'graph_sha256')},
        lif_dynamics_version=active_dynamics_version(),
        lif_dynamics_pin=dynamics_pin(active_dynamics_version()),
        lif_dynamics=dict(DYNAMICS_VERSIONS[active_dynamics_version()]),
        lif_e_inh_mV=active_e_inh_mV(),
        lif_dynamics_variant=dynamics_variant(active_dynamics_version(), active_e_inh_mV())['dynamics_version'],
        lif_dynamics_variant_pin=dynamics_variant_pin(active_dynamics_version(), active_e_inh_mV()),
        structural_in_weight_budget=budgets)

    arms = [a.strip() for a in args.arm.split(',') if a.strip()]
    result = dict(header, arms={})
    for arm in arms:
        rows = {}
        for direction in (+1, -1):
            brain = Brain(arrays=policy_graph.arrays, validate=False,
                          dynamics=args.dynamics, e_inh_mV=args.e_inh and float(args.e_inh))
            row = run_direction(brain, policy_graph, io, prereg, direction, arm)
            rows[direction] = row
            print(f"arm {arm} dir {direction:+d}: net {row['stimulus']['network_rate_hz']:.4g} spikes/s, "
                  f"DNa02 L {row['stimulus']['rate_l_hz']} R {row['stimulus']['rate_r_hz']} Hz, "
                  f"{row['sim_s_per_wall_s']:.3f} sim s/wall s", flush=True)
            del brain
        g = gate([rows[+1], rows[-1]])
        diag = stage_table(rows, e_inh)
        result['arms'][arm] = dict(gate=g, diagnostic=diag,
                                   directions={str(d): rows[d] for d in rows})
        print(f"arm {arm}: Q1 {g['Q1']} R1 {g['R1']} R2 {g['R2']} "
              f"DNa02 L-R {g['dna02_LminusR_per_direction']}; "
              f"first silent stage {diag['first_silent_stage']}", flush=True)
    result['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (args.out / 'photoreceptor_probe.json').write_text(json.dumps(redact_local(result), indent=2) + '\n')
    print('wrote', args.out / 'photoreceptor_probe.json')


if __name__ == '__main__':
    main()
