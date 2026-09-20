"""WP5: run the preregistered optomotor causal test on the full MaleCNS graph.

All parameters come from docs/wp5_optomotor_prereg.json; its sha256 is
recorded in every output.  Conditions run sequentially with one graph in
memory at a time; each (condition, seed) gets its own ExperimentRegistry root
and a fresh connectome-fixed instance, checkpointed at the end.

    NEUROFLY_GRAPH_DIR=<graph dir> PYTHONPATH=. .venv/bin/python scripts/wp5_optomotor.py \
        --out outputs/wp5/<stamp> [--pilot] [--dynamics v2] \
        [--receipt docs/receipts/wp5_optomotor.json]

``--dynamics`` selects the declared LIF version (docs/LIF_DYNAMICS_SPEC.md).
It defaults to ``v1``, the dynamics the 19 September 2026 confirmatory run was
produced under, so the original result stays reproducible byte for byte.  The
preregistration, the schedule and the primary outcome are identical in both
cases; only the engine differs, and every manifest records which one.

Nothing here contacts running services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The declared dynamics must be selected BEFORE brainlab is imported: the run
# manifest's dynamics block and controller version are resolved at import time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--dynamics', default='v1')
os.environ['NEUROFLY_LIF_DYNAMICS'] = _pre.parse_known_args()[0].dynamics

from brainlab.graph_identity import (DYNAMICS_VERSIONS, GraphIdentity,  # noqa: E402
                                     active_dynamics_version, dynamics_pin, sha256_json)
from brainlab.io_map import (DNa02YawDecoder, OptomotorEncoder, OptomotorLoop,  # noqa: E402
                             resolve_optomotor_io)
from experiment_registry import ExperimentRegistry, SharedGraph  # noqa: E402

PREREG = ROOT / 'docs/wp5_optomotor_prereg.json'
GRAPH_CONDITIONS = ('intact', 'dna02_silenced', 'sham_no_input', 'shuffled_graph')
REPORTED_MONITORS = ('HS', 'H2', 'VS', 'LLPC1', 'PFL3', 'DNa01', 'DNa02', 'DNa03', 'DNb01', 'DNp09', 'MDN')


def rss_mib():
    with open('/proc/self/status') as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return None


def timeline(stim: dict, blocks=None):
    """Per-step (slip, contrast, block index or -1)."""
    step = stim['step_ms']
    blocks = stim['blocks'] if blocks is None else blocks
    rows = []
    def add(ms, slip, contrast, block):
        rows.extend([(slip, contrast, block)] * int(round(ms / step)))
    add(stim['initial_gray_ms'], 0.0, 0.0, -1)
    for b, (direction, contrast) in enumerate(blocks):
        add(stim['block_ms'], direction * stim['angular_velocity_rad_s'], contrast, b)
        add(stim['gray_between_ms'], 0.0, 0.0, -1)
    return rows


def run_graph_condition(shared, io, condition, seed, prereg, root, blocks=None, test_mode=False,
                        closed_loop=False, keep_trace=True):
    stim, dec = prereg['stimulus'], prereg['decoder']
    registry = ExperimentRegistry(shared, root, test_mode=test_mode)
    instance = registry.activate('optomotor', 'connectome-fixed')
    rng = np.random.default_rng(seed)
    encoder = OptomotorEncoder(io, rng, i_max=stim['i_max'], spatial_period_deg=stim['spatial_period_deg'],
                               noise_sd=stim['noise_sd'])
    decoder = DNa02YawDecoder(io, gain_rad_s_per_hz=dec['gain_rad_s_per_hz'], tau_ms=dec['tau_ms'])
    loop = OptomotorLoop(instance, io, encoder, decoder,
                         deliver_sensory=(condition != 'sham_no_input'),
                         silence=('DNa02_L', 'DNa02_R') if condition == 'dna02_silenced' else (),
                         step_ms=stim['step_ms'])
    schedule = timeline(stim, blocks)
    instance.manifest.intervention_schedule = [dict(
        wp5_condition=condition, stimulus_seed=seed, prereg_sha256=prereg['_sha256'], loop=loop.describe(),
        encoder=encoder.describe(), decoder=decoder.describe(), closed_loop=closed_loop,
        lif_dynamics_version=active_dynamics_version(),
        lif_dynamics_pin=dynamics_pin(active_dynamics_version()),
        schedule_sha256=sha256_json(schedule))]
    instance.manifest.record_event('wp5-start', step=instance.step_index, condition=condition, seed=seed)
    trace = {k: [] for k in ('slip', 'contrast', 'block', 'yaw', 'spk_l', 'spk_r', 'v_l', 'v_r', 'total')}
    mon_trace = {k: [] for k in io.monitors}
    enc_trace = {k: [] for k in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R')}
    yaw = 0.0
    clock = time.perf_counter()
    for slip, contrast, block in schedule:
        applied = slip - yaw if (closed_loop and slip != 0) else slip
        rec = loop.step(applied, contrast)
        yaw = rec['yaw_rad_s']
        trace['slip'].append(applied); trace['contrast'].append(contrast); trace['block'].append(block)
        trace['yaw'].append(yaw); trace['spk_l'].append(rec['spikes_l']); trace['spk_r'].append(rec['spikes_r'])
        trace['v_l'].append(rec['v_dna02_l']); trace['v_r'].append(rec['v_dna02_r']); trace['total'].append(rec['total_spikes'])
        for k, v in rec['monitors'].items():
            mon_trace[k].append(v)
        for k, v in rec['encoder_spikes'].items():
            enc_trace[k].append(v)
    wall = time.perf_counter() - clock
    sim_ms = len(schedule) * stim['step_ms']
    instance.manifest.record_event('wp5-end', step=instance.step_index, wall_s=wall, sim_ms=sim_ms)
    ckpt = registry.checkpoint(world_state=dict(condition=condition, seed=seed, t_ms=loop.t_ms, last_yaw=yaw))
    instance.manifest.write(registry.instance_dir(instance.instance_id) / 'manifest.json')
    run = dict(condition=condition, seed=seed, instance_id=instance.instance_id, run_id=instance.manifest.run_id,
               checkpoint=str(ckpt), wall_s=wall, sim_ms=sim_ms, rss_mib=rss_mib(),
               manifest_identity=instance.manifest.identity())
    instance.release()
    registry.active = None
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays.update({f'mon.{k}': np.asarray(v, dtype=np.int32) for k, v in mon_trace.items()})
    arrays.update({f'enc.{k}': np.asarray(v, dtype=np.int32) for k, v in enc_trace.items()})
    return run, arrays


def block_summary(arrays, io, stim, blocks):
    """Per-block s-aligned measures; rates in Hz per cell."""
    step_s = stim['step_ms'] / 1000.0
    out = []
    for b, (direction, contrast) in enumerate(blocks):
        m = arrays['block'] == b
        dur = m.sum() * step_s
        row = dict(block=b, s=direction, contrast=contrast,
                   yaw=float(arrays['yaw'][m].mean()),
                   rate_l=float(arrays['spk_l'][m].sum() / dur), rate_r=float(arrays['spk_r'][m].sum() / dur),
                   v_l=float(arrays['v_l'][m].mean()), v_r=float(arrays['v_r'][m].mean()),
                   total_rate=float(arrays['total'][m].sum() / dur))
        for name in REPORTED_MONITORS:
            for eye in ('L', 'R'):
                n = max(1, len(io.monitors[f'{name}_{eye}']))
                row[f'{name}_{eye}'] = float(arrays[f'mon.{name}_{eye}'][m].sum() / dur / n)
        for k in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R'):
            row[f'enc_{k}'] = float(arrays[f'enc.{k}'][m].sum() / dur / len(io.populations[k]))
        out.append(row)
    gray = arrays['block'] == -1
    baseline = dict(yaw=float(arrays['yaw'][gray].mean()),
                    rate_l=float(arrays['spk_l'][gray].sum() / (gray.sum() * step_s)),
                    rate_r=float(arrays['spk_r'][gray].sum() / (gray.sum() * step_s)),
                    total_rate=float(arrays['total'][gray].sum() / (gray.sum() * step_s)))
    return out, baseline


def seed_metrics(blocks_rows):
    s = np.array([r['s'] for r in blocks_rows], float)
    c = np.array([r['contrast'] for r in blocks_rows], float)
    get = lambda key: np.array([r[key] for r in blocks_rows], float)
    metrics = dict(TI=float(np.mean(s * get('yaw'))),
                   A_DNa02=float(np.mean(s * (get('rate_l') - get('rate_r')))),
                   dV_DNa02=float(np.mean(s * (get('v_l') - get('v_r')))),
                   TI_c1=float(np.mean((s * get('yaw'))[c == 1.0])),
                   TI_c05=float(np.mean((s * get('yaw'))[c == 0.5])))
    for name in REPORTED_MONITORS:
        metrics[f'asym_{name}'] = float(np.mean(s * (get(f'{name}_L') - get(f'{name}_R'))))
        metrics[f'{name}_L_sPos'] = float(get(f'{name}_L')[s > 0].mean())
        metrics[f'{name}_R_sPos'] = float(get(f'{name}_R')[s > 0].mean())
        metrics[f'{name}_L_sNeg'] = float(get(f'{name}_L')[s < 0].mean())
        metrics[f'{name}_R_sNeg'] = float(get(f'{name}_R')[s < 0].mean())
    for k in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R'):
        metrics[f'enc_{k}_sPos'] = float(get(f'enc_{k}')[s > 0].mean())
        metrics[f'enc_{k}_sNeg'] = float(get(f'enc_{k}')[s < 0].mean())
    metrics['total_rate'] = float(get('total_rate').mean())
    return metrics


def bootstrap(values, rng_seed=20260919, n=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(rng_seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return dict(mean=float(values.mean()), sd=sd, ci95=[float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
                n=int(len(values)), dz=(float(values.mean() / sd) if sd > 0 else None),
                n_positive=int((values > 0).sum()), n_negative=int((values < 0).sum()), n_zero=int((values == 0).sum()))


def modular_baseline(prereg):
    from vision import CompoundEyeVision
    stim = prereg['stimulus']
    dt = 0.02
    vision = CompoundEyeVision()
    rows = []
    def run(ms, slip, contrast):
        ys = []
        for _ in range(int(round(ms / 1000 / dt))):
            vision.step(np.zeros(2), 0.0, 0.0, 0.0, dt=dt, external_yaw_rad_s=slip, contrast=contrast)
            ys.append(vision.get_optomotor_yaw_bias())
        return float(np.mean(ys))
    run(stim['initial_gray_ms'], 0.0, 0.0)
    for b, (direction, contrast) in enumerate(stim['blocks']):
        yaw = run(stim['block_ms'], direction * stim['angular_velocity_rad_s'], contrast)
        rows.append(dict(block=b, s=direction, contrast=contrast, yaw=yaw))
        run(stim['gray_between_ms'], 0.0, 0.0)
    s = np.array([r['s'] for r in rows]); y = np.array([r['yaw'] for r in rows]); c = np.array([r['contrast'] for r in rows])
    return dict(blocks=rows, TI=float(np.mean(s * y)), TI_c1=float(np.mean((s * y)[c == 1.0])),
                TI_c05=float(np.mean((s * y)[c == 0.5])),
                note='modular CompoundEyeVision.get_optomotor_yaw_bias: hand-built HS difference x 0.8, clipped +-0.3')


def shuffled_shared(real: SharedGraph, seed=777) -> SharedGraph:
    rng = np.random.default_rng(seed)
    post = real.arrays['post'][rng.permutation(len(real.arrays['post']))]
    arrays = dict(ptr=real.arrays['ptr'], post=np.ascontiguousarray(post), weight=real.arrays['weight'],
                  ids=real.arrays['ids'])
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(arrays[key].tobytes())
    base = real.identity
    identity = GraphIdentity(
        dataset='malecns_v1-degree-preserving-shuffle', synthetic=True, graph_path=None,
        graph_path_source=f'permute post of {base.graph_sha256} with default_rng({seed})',
        graph_sha256=digest.hexdigest(), neuron_map_path=base.neuron_map_path,
        neuron_map_sha256=base.neuron_map_sha256, io_map_sha256=base.io_map_sha256, neurons=base.neurons,
        edges=base.edges, ids_sha256=base.ids_sha256,
        label='CONTROL GRAPH: degree-preserving shuffle of MaleCNS v1.0 targets; not the released wiring')
    return SharedGraph(arrays, identity, real.io_map)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--closed-loop', action='store_true', help='conditional secondary (prereg)')
    parser.add_argument('--exploratory-mirror', action='store_true',
                        help='EXPLORATORY: direction-negated schedule, seeds 0-2, not part of the verdict')
    parser.add_argument('--dynamics', default='v1', choices=('v1', 'v2'),
                        help='declared LIF dynamics version (docs/LIF_DYNAMICS_SPEC.md); default v1')
    args = parser.parse_args()
    assert args.dynamics == active_dynamics_version()   # set before the brainlab import
    raw = PREREG.read_bytes()
    prereg = json.loads(raw)
    prereg['_sha256'] = hashlib.sha256(raw).hexdigest()
    stim = prereg['stimulus']
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    clock = time.perf_counter()
    shared = SharedGraph.load()
    load_s = time.perf_counter() - clock
    io = resolve_optomotor_io()
    header = dict(started_at=started, host=platform.node(), prereg_sha256=prereg['_sha256'],
                  graph=shared.identity.to_dict(), io_map=io.describe(), graph_load_s=load_s,
                  rss_mib_after_load=rss_mib(),
                  lif_dynamics_version=active_dynamics_version(),
                  lif_dynamics_pin=dynamics_pin(active_dynamics_version()),
                  lif_dynamics=dict(DYNAMICS_VERSIONS[active_dynamics_version()]))

    if args.pilot:
        blocks = [[1, 1.0], [-1, 1.0]]
        run, arrays = run_graph_condition(shared, io, 'intact', 999, prereg, args.out / 'registry/pilot', blocks=blocks)
        rows, _ = block_summary(arrays, io, stim, blocks)
        # Pilot inspects runtime and encoder firing only (prereg): DN fields are dropped.
        enc = [{k: v for k, v in r.items() if k.startswith('enc_')} for r in rows]
        pilot = dict(header, run={k: run[k] for k in ('wall_s', 'sim_ms', 'rss_mib')},
                     sim_s_per_wall_s=run['sim_ms'] / 1000 / run['wall_s'], encoder_rates_hz=enc,
                     projected_confirmatory_wall_min=(len(timeline(stim)) * stim['step_ms'] / run['sim_ms'])
                     * run['wall_s'] * len(prereg['seeds']) * len(GRAPH_CONDITIONS) / 60)
        (args.out / 'pilot.json').write_text(json.dumps(pilot, indent=2) + '\n')
        print(json.dumps({k: pilot[k] for k in ('run', 'sim_s_per_wall_s', 'encoder_rates_hz',
                                                  'projected_confirmatory_wall_min')}, indent=1))
        return

    if args.exploratory_mirror:
        # EXPLORATORY (declared after the confirmatory run, not part of the verdict):
        # the same blocks with every direction negated, to test whether the
        # observed leftward bias depends on stimulus history (first block s=+1).
        blocks = [[-d, c] for d, c in stim['blocks']]
        out = []
        for seed in prereg['seeds'][:3]:
            run, arrays = run_graph_condition(shared, io, 'intact', seed, prereg,
                                              args.out / f'registry/mirror/seed-{seed}', blocks=blocks)
            rows, baseline = block_summary(arrays, io, stim, blocks)
            np.savez_compressed(args.out / f'trace-mirror-seed{seed}.npz', **arrays)
            out.append(dict(seed=seed, metrics=seed_metrics(rows), blocks=rows, gray_baseline=baseline,
                            dna02_spikes=dict(L=int(arrays['spk_l'].sum()), R=int(arrays['spk_r'].sum())),
                            wall_s=run['wall_s']))
            print(f'mirror seed {seed}: TI {out[-1]["metrics"]["TI"]:+.3f}', flush=True)
        summary = {k: bootstrap([s['metrics'][k] for s in out]) for k in ('TI', 'A_DNa02', 'dV_DNa02')}
        (args.out / 'exploratory_mirror.json').write_text(json.dumps(dict(
            header, label='EXPLORATORY - declared after confirmatory results; not part of the verdict',
            blocks=blocks, summary=summary, seeds=out), indent=2) + '\n')
        print(json.dumps(summary, indent=1))
        return

    if args.closed_loop:
        seeds = prereg['seeds'][:5]
        results = []
        for seed in seeds:
            for mode in (False, True):
                run, arrays = run_graph_condition(shared, io, 'intact', seed, prereg,
                                                  args.out / f'registry/closed-{mode}/seed-{seed}', closed_loop=mode)
                m = arrays['block'] >= 0
                results.append(dict(seed=seed, closed_loop=mode, mean_abs_slip=float(np.abs(arrays['slip'][m]).mean())))
        (args.out / 'closed_loop.json').write_text(json.dumps(dict(header, results=results), indent=2) + '\n')
        print(json.dumps(results, indent=1))
        return

    per_condition = {}
    runs = []
    for condition in GRAPH_CONDITIONS:
        graph = shared
        if condition == 'shuffled_graph':
            clock = time.perf_counter()
            graph = shuffled_shared(shared)
            header['shuffle_build_s'] = time.perf_counter() - clock
            header['shuffled_identity'] = graph.identity.to_dict()
        seeds_out = []
        for seed in prereg['seeds']:
            run, arrays = run_graph_condition(graph, io, condition, seed, prereg,
                                              args.out / f'registry/{condition}/seed-{seed}',
                                              test_mode=(condition == 'shuffled_graph'))
            rows, baseline = block_summary(arrays, io, stim, stim['blocks'])
            metrics = seed_metrics(rows)
            np.savez_compressed(args.out / f'trace-{condition}-seed{seed}.npz', **arrays)
            seeds_out.append(dict(seed=seed, metrics=metrics, blocks=rows, gray_baseline=baseline,
                                  yaw_all_zero=bool(np.all(arrays['yaw'] == 0)),
                                  dna02_spikes=dict(L=int(arrays['spk_l'].sum()), R=int(arrays['spk_r'].sum()))))
            runs.append(run)
            print(f"{condition} seed {seed}: wall {run['wall_s']:.1f}s sim {run['sim_ms']/1000:.1f}s", flush=True)
        per_condition[condition] = seeds_out
        if graph is not shared:
            del graph

    summary = {}
    keys = ['TI', 'TI_c1', 'TI_c05', 'A_DNa02', 'dV_DNa02', 'total_rate'] + [f'asym_{n}' for n in REPORTED_MONITORS]
    for condition, seeds_out in per_condition.items():
        summary[condition] = {k: bootstrap([s['metrics'][k] for s in seeds_out]) for k in keys}
        summary[condition]['yaw_identically_zero_all_seeds'] = all(s['yaw_all_zero'] for s in seeds_out)
        summary[condition]['dna02_spikes_total'] = dict(
            L=sum(s['dna02_spikes']['L'] for s in seeds_out), R=sum(s['dna02_spikes']['R'] for s in seeds_out))
        summary[condition]['side_rates_mean_hz'] = {
            k: float(np.mean([s['metrics'][k] for s in seeds_out]))
            for k in seeds_out[0]['metrics'] if k.endswith(('_sPos', '_sNeg'))}
    paired = {}
    for other in ('dna02_silenced', 'sham_no_input', 'shuffled_graph'):
        paired[f'intact-{other}'] = {k: bootstrap([a['metrics'][k] - b['metrics'][k] for a, b in
                                                   zip(per_condition['intact'], per_condition[other])])
                                     for k in ('TI', 'A_DNa02', 'dV_DNa02', 'asym_HS')}
    # Motor dependence on the neural response, across intact blocks (pooled seeds).
    rows = [r for s in per_condition['intact'] for r in s['blocks']]
    sv = np.array([r['s'] for r in rows], float)
    hs = sv * (np.array([r['HS_L'] for r in rows]) - np.array([r['HS_R'] for r in rows]))
    a02 = sv * (np.array([r['rate_l'] for r in rows]) - np.array([r['rate_r'] for r in rows]))
    yaw = sv * np.array([r['yaw'] for r in rows])
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    dependence = dict(n_blocks=len(rows), r_HSasym_vs_A_DNa02=corr(hs, a02), r_A_DNa02_vs_aligned_yaw=corr(a02, yaw))

    ti = summary['intact']['TI']['ci95']
    d_sham = paired['intact-sham_no_input']['TI']['ci95']
    d_shuf = paired['intact-shuffled_graph']['TI']['ci95']
    if ti[0] > 0 and d_sham[0] > 0 and d_shuf[0] > 0 and summary['dna02_silenced']['yaw_identically_zero_all_seeds']:
        verdict = 'POSITIVE'
    elif ti[1] < 0 and d_sham[1] < 0:
        verdict = 'NEGATIVE'
    else:
        verdict = 'NULL'

    intact_runs = [r for r in runs if r['condition'] == 'intact']
    sim_s = sum(r['sim_ms'] for r in intact_runs) / 1000
    wall_s = sum(r['wall_s'] for r in intact_runs)
    compute = dict(intact_sim_s=sim_s, intact_wall_s=wall_s, sim_s_per_wall_s=sim_s / wall_s,
                   per_condition_sim_s_per_wall_s={c: sum(r['sim_ms'] for r in runs if r['condition'] == c) / 1000 /
                                                   sum(r['wall_s'] for r in runs if r['condition'] == c)
                                                   for c in GRAPH_CONDITIONS},
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   step_ms=stim['step_ms'], lif_dt_ms=0.1)
    result = dict(header, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), verdict=verdict,
                  summary=summary, paired=paired, motor_dependence=dependence, modular_baseline=modular_baseline(prereg),
                  compute=compute, runs=runs, per_condition=per_condition)
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('VERDICT', verdict)
    print(json.dumps(dict(intact_TI=summary['intact']['TI'], paired_TI={k: v['TI'] for k, v in paired.items()},
                          A_DNa02=summary['intact']['A_DNa02'], dV=summary['intact']['dV_DNa02'],
                          dependence=dependence, compute=compute), indent=1))


if __name__ == '__main__':
    main()
