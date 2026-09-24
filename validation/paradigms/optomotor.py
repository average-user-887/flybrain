"""Optomotor yaw, open loop (tethered): the WP5 loop run under a validation spec.

Reuses the WP5 encoder and decoder unchanged (``brainlab.io_map``): drive on
eye-specific T4/T5 subtypes in, DNa02 L-R rate difference out.  The schedule
format is the WP5 preregistration's.  What is new is only what the spec asks
for: per-direction and per-contrast responses, the controls as paired
differences, and population rates for the physiology check.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from brainlab.io_map import (BACK_TO_FRONT, FRONT_TO_BACK, MONITOR_TYPES, SILENCE_DRIVE,
                             DNa02YawDecoder, OptomotorEncoder, OptomotorIOMap, resolve_optomotor_io)
from brainlab.graph_identity import GraphIdentity, sha256_json

from ..runner import BrainRunner, SpikeLedger

ENCODER_POPS = ('ftb_L', 'btf_L', 'ftb_R', 'btf_R')


def io_from_cells(cells) -> OptomotorIOMap:
    """The WP5 resolution rule applied to any cell table (used for the synthetic graph).

    The real graph goes through :func:`brainlab.io_map.resolve_optomotor_io`
    instead, which also checks the pinned digest.
    """
    available, per_type = {}, {}
    for t in FRONT_TO_BACK + BACK_TO_FRONT:
        for eye in ('L', 'R'):
            per_type[(t, eye)] = cells.select((t,), eye)
            available[f'{t}_{eye}'] = int(len(per_type[(t, eye)]))
    matched = {'T4': min(v for k, v in available.items() if k.startswith('T4')),
               'T5': min(v for k, v in available.items() if k.startswith('T5'))}
    populations, source_ids = {}, {}
    for eye in ('L', 'R'):
        for label, types in (('ftb', FRONT_TO_BACK), ('btf', BACK_TO_FRONT)):
            frames = [per_type[(t, eye)].iloc[:matched[t[:2]]] for t in types]
            populations[f'{label}_{eye}'] = np.concatenate([f.node_index.to_numpy(np.int64) for f in frames])
            source_ids[f'{label}_{eye}'] = [int(s) for f in frames for s in f.source_id]
        rows = cells.select(('DNa02',), eye)
        if len(rows) != 1:
            raise ValueError(f'expected one DNa02_{eye}, found {len(rows)}')
        populations[f'DNa02_{eye}'] = rows.node_index.to_numpy(np.int64)
        source_ids[f'DNa02_{eye}'] = [int(rows.source_id.iat[0])]
    monitors = {f'{name}_{eye}': np.sort(cells.select(types, eye).node_index.to_numpy(np.int64))
                for name, types in MONITOR_TYPES.items() for eye in ('L', 'R')}
    digest = sha256_json(dict(source_ids=source_ids, rule='T4/T5 a|b by somaSide, matched lowest source_id; '
                                                          'DNa02 by instance+somaSide'))
    return OptomotorIOMap(populations=populations, source_ids=source_ids, matched_counts=matched,
                          available_counts=available, monitors=monitors, sha256=digest)


def timeline(stim):
    """Per-step (slip rad/s, contrast, block index or -1): the WP5 schedule."""
    rows = []

    def add(ms, slip, contrast, block):
        rows.extend([(slip, contrast, block)] * int(round(ms / stim['step_ms'])))
    add(stim['initial_gray_ms'], 0.0, 0.0, -1)
    for b, (direction, contrast) in enumerate(stim['blocks']):
        add(stim['block_ms'], direction * stim['angular_velocity_rad_s'], contrast, b)
        add(stim['gray_between_ms'], 0.0, 0.0, -1)
    return rows


def shuffled_shared(shared, seed: int):
    """Degree-preserving control graph: the WP5 shuffle (permute ``post`` once)."""
    from experiment_registry import SharedGraph
    rng = np.random.default_rng(seed)
    post = np.ascontiguousarray(shared.arrays['post'][rng.permutation(len(shared.arrays['post']))])
    arrays = dict(ptr=shared.arrays['ptr'], post=post, weight=shared.arrays['weight'], ids=shared.arrays['ids'])
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(arrays[key].tobytes())
    base = shared.identity
    identity = GraphIdentity(
        dataset=f'{base.dataset}-degree-preserving-shuffle', synthetic=True, graph_path=None,
        graph_path_source=f'permute post of {base.graph_sha256} with default_rng({seed})',
        graph_sha256=digest.hexdigest(), neuron_map_path=base.neuron_map_path,
        neuron_map_sha256=base.neuron_map_sha256, io_map_sha256=base.io_map_sha256, neurons=base.neurons,
        edges=base.edges, ids_sha256=base.ids_sha256,
        label='CONTROL GRAPH: degree-preserving shuffle of the run graph; not the released wiring')
    return SharedGraph(arrays, identity, getattr(shared, 'io_map', {}))


def run_trial(runner: BrainRunner, io: OptomotorIOMap, spec: dict, seed: int, condition: str) -> dict:
    stim, enc, dec = spec['stimulus'], spec['encoder'], spec['decoder']
    runner.reset()
    rng = np.random.default_rng(seed)
    encoder = OptomotorEncoder(io, rng, i_max=enc['i_max'], spatial_period_deg=stim['spatial_period_deg'],
                               noise_sd=enc['noise_sd'])
    decoder = DNa02YawDecoder(io, gain_rad_s_per_hz=dec['gain_rad_s_per_hz'], tau_ms=dec['tau_ms'])
    deliver = condition != 'sham_no_input'
    silence = (np.concatenate([io.populations['DNa02_L'], io.populations['DNa02_R']])
               if condition == 'dna02_silenced' else np.zeros(0, np.int64))
    pops = {**{k: io.populations[k] for k in ENCODER_POPS + ('DNa02_L', 'DNa02_R')},
            **{k: v for k, v in io.monitors.items() if len(v)}}
    ledger = SpikeLedger(runner.n, pops)
    schedule = timeline(stim)
    step = float(stim['step_ms'])
    scratch = np.zeros(runner.n, np.float32)
    currents = np.zeros(runner.n, np.float32)
    yaw = np.zeros(len(schedule))
    block = np.array([b for _, _, b in schedule])
    t_ms = 0.0
    for i, (slip, contrast, b) in enumerate(schedule):
        scratch.fill(0.0)
        encoder.encode(scratch, t_ms, slip, contrast)      # draws the RNG whether or not delivered
        currents.fill(0.0)
        if deliver:
            currents += scratch
        if len(silence):
            currents[silence] = SILENCE_DRIVE
        counts = runner.step(currents, step)
        yaw[i] = decoder.decode(counts, step)['yaw_rad_s']
        ledger.add(counts, step, 'stimulus' if b >= 0 else ('baseline' if i * step < stim['initial_gray_ms'] else None))
        t_ms += step
    blocks = []
    for b, (direction, contrast) in enumerate(stim['blocks']):
        m = block == b
        blocks.append(dict(block=b, s=direction, contrast=contrast, yaw=float(yaw[m].mean())))
    return dict(seed=seed, condition=condition, blocks=blocks, yaw_abs_max=float(np.abs(yaw).max()),
                sim_ms=len(schedule) * step, rates=ledger.summary())


def seed_metrics(blocks) -> dict:
    s = np.array([b['s'] for b in blocks], float)
    c = np.array([b['contrast'] for b in blocks], float)
    y = np.array([b['yaw'] for b in blocks], float)
    aligned = s * y
    return dict(TI=float(aligned.mean()), R_pos=float(aligned[s > 0].mean()), R_neg=float(aligned[s < 0].mean()),
                R_c1=float(aligned[c == 1.0].mean()), R_c05=float(aligned[c == 0.5].mean()))


def wp5_rule(samples: dict) -> dict:
    """The WP5 preregistered decision rule (docs/wp5_optomotor_prereg.json), for continuity.

    POSITIVE: TI CI > 0, intact-sham CI > 0, intact-shuffled CI > 0 and DNa02-silenced
    yaw identically 0.  NEGATIVE: TI CI < 0 and intact-sham CI < 0.  Otherwise NULL.
    Same bootstrap as WP5 (10000 resamples, default_rng(20260919)).
    """
    from ..stats import bootstrap_mean
    needed = ('TI', 'TI_minus_sham_no_input', 'TI_minus_shuffled_graph', 'silenced_yaw_abs_max')
    if any(k not in samples for k in needed):
        return dict(verdict=None, reason='not every WP5 condition was run')
    ci = {k: bootstrap_mean(samples[k], seed=20260919)['ci'] for k in needed[:3]}
    silenced_zero = all(v == 0 for v in samples['silenced_yaw_abs_max'])
    if ci['TI'][0] > 0 and ci['TI_minus_sham_no_input'][0] > 0 and ci['TI_minus_shuffled_graph'][0] > 0 and silenced_zero:
        verdict = 'POSITIVE'
    elif ci['TI'][1] < 0 and ci['TI_minus_sham_no_input'][1] < 0:
        verdict = 'NEGATIVE'
    else:
        verdict = 'NULL'
    return dict(verdict=verdict, ci=ci, dna02_silenced_yaw_identically_zero=silenced_zero,
                note='WP5 rule for comparison with docs/receipts/wp5_optomotor.json (v1); the harness verdict uses the spec gates')


def run(spec: dict, ctx) -> dict:
    io = io_from_cells(ctx.cells) if ctx.synthetic else resolve_optomotor_io(ctx.connectome_dir)
    trials = {}
    for condition in spec['conditions']:
        shared = ctx.shared
        if condition == 'shuffled_graph':
            shared = shuffled_shared(ctx.shared, spec['controls']['shuffle_seed'])
        runner = ctx.make_runner(shared)
        trials[condition] = []
        for seed in ctx.seeds:
            trial = run_trial(runner, io, spec, seed, condition)
            trials[condition].append(trial)
            ctx.log(f'{condition} seed {seed}: TI {seed_metrics(trial["blocks"])["TI"]:+.4f} rad/s')
        del runner
    metrics = {c: [seed_metrics(t['blocks']) for t in ts] for c, ts in trials.items()}
    intact = metrics['intact']
    samples = {k: [m[k] for m in intact] for k in intact[0]}
    for control in ('sham_no_input', 'shuffled_graph', 'dna02_silenced'):
        if control in metrics:
            samples[f'TI_minus_{control}'] = [a['TI'] - b['TI'] for a, b in zip(intact, metrics[control])]
    if 'dna02_silenced' in trials:
        samples['silenced_yaw_abs_max'] = [t['yaw_abs_max'] for t in trials['dna02_silenced']]
    return dict(samples=samples, trials=trials, io=io.describe(), extra=dict(wp5_rule=wp5_rule(samples)),
                rates={c: [t['rates'] for t in ts] for c, ts in trials.items()},
                sim_ms=sum(t['sim_ms'] for ts in trials.values() for t in ts))
