"""T-maze odour choice, naive (pre-learning): ORN drive in, DNa02 steering out.

One trial is one fly at the choice point: odour A in the left arm, odour B in
the right arm (or air), presented for ``decision_ms`` after a clean-air
``pre_ms``.  Reciprocal design as in the T-maze PI: every seed runs both side
assignments, so a side bias cancels.

Encoder (declared, spec ``encoder``):
* ORN target rate per glomerulus = spontaneous + DoOR 2.0 consensus response
  (spontaneous-subtracted, 0-1 per receptor) x ``r_max_hz``, floored at 0.
  The table is copied into the spec with its source commit.
* The antenna ipsilateral to an arm receives that arm's odour; the
  contralateral antenna receives ``contralateral_fraction`` of it.
* A target rate becomes a per-neuron drive by inverting the resting LIF
  firing rate (the declared constants of ``brainlab.engine``), with
  per-cell multiplicative noise.  The achieved ORN rates are recorded and
  checked against the physiology bounds, not assumed.

Decoder: the choice is the side of the larger DNa02 spike count over the
decision window (DNa02 steers ipsilaterally; the WP5 decoder's convention).
Equal counts, including none, is no choice; no-choice trials are excluded
from the PI and their fraction is reported and gated.
"""
from __future__ import annotations

import math

import numpy as np

from brainlab.engine import REFRACTORY_MS, TAU_M_MS, V_REST_MV, V_THRESHOLD_MV

from ..runner import SpikeLedger


def drive_for_rate(rate_hz: np.ndarray) -> np.ndarray:
    """Constant drive (mV-equivalent) that makes a resting LIF cell fire at ``rate_hz``.

    Period = t_ref + tau_m * ln(I / (I - dV)), dV = V_th - V_rest, solved for I.
    Rates at or above the refractory limit are refused rather than clipped.
    """
    rate_hz = np.asarray(rate_hz, float)
    dv = V_THRESHOLD_MV - V_REST_MV
    out = np.zeros_like(rate_hz)
    on = rate_hz > 0
    free_ms = 1000.0 / rate_hz[on] - REFRACTORY_MS
    if np.any(free_ms <= 0):
        raise ValueError('target rate at or above the refractory limit')
    out[on] = dv / (1.0 - np.exp(-free_ms / TAU_M_MS))
    return out


def orn_rates(spec: dict, odour_left: str, odour_right: str) -> dict:
    """Target Hz per ORN population name ('ORN_<glom>_L' / '_R')."""
    enc = spec['encoder']
    table = enc['glomeruli']
    contra = enc['contralateral_fraction']
    out = {}
    for glom, row in table.items():
        spont = row['spontaneous_hz'] if row['spontaneous_hz'] is not None else enc['default_spontaneous_hz']
        resp = {name: (row['response'].get(name) or 0.0) for name in (odour_left, odour_right) if name != 'air'}
        left = resp.get(odour_left, 0.0) + contra * resp.get(odour_right, 0.0)
        right = resp.get(odour_right, 0.0) + contra * resp.get(odour_left, 0.0)
        out[f'ORN_{glom}_L'] = max(0.0, spont + enc['r_max_hz'] * left)
        out[f'ORN_{glom}_R'] = max(0.0, spont + enc['r_max_hz'] * right)
    return out


def run_trial(runner, pops, spec: dict, seed: int, odour_left: str, odour_right: str) -> dict:
    stim, enc = spec['stimulus'], spec['encoder']
    step = float(stim['step_ms'])
    runner.reset()
    rng = np.random.default_rng(seed)
    orn_names = [k for k in pops.nodes if k.startswith('ORN_') and len(pops.nodes[k])]
    air = orn_rates(spec, 'air', 'air')
    odour = orn_rates(spec, odour_left, odour_right)
    idx = np.concatenate([pops.nodes[k] for k in orn_names])
    base_drive = np.concatenate([np.full(len(pops.nodes[k]), drive_for_rate(np.array([air[k]]))[0]) for k in orn_names])
    odour_drive = np.concatenate([np.full(len(pops.nodes[k]), drive_for_rate(np.array([odour[k]]))[0])
                                  for k in orn_names])
    ledger = SpikeLedger(runner.n, pops.nodes)
    currents = np.zeros(runner.n, np.float32)
    n_pre, n_dec = int(round(stim['pre_ms'] / step)), int(round(stim['decision_ms'] / step))
    left_id, right_id = pops.nodes['DNa02_L'], pops.nodes['DNa02_R']
    spikes_l = spikes_r = 0
    for i in range(n_pre + n_dec):
        noise = rng.standard_normal(len(idx))
        target = base_drive if i < n_pre else odour_drive
        currents.fill(0.0)
        currents[idx] = np.maximum(0.0, target * (1 + enc['noise_sd'] * noise))
        counts = runner.step(currents, step)
        window = 'baseline' if i < n_pre else 'stimulus'
        ledger.add(counts, step, window)
        if window == 'stimulus':
            spikes_l += int(counts[left_id].sum())
            spikes_r += int(counts[right_id].sum())
    choice = 'left' if spikes_l > spikes_r else ('right' if spikes_r > spikes_l else None)
    chosen = {'left': odour_left, 'right': odour_right}.get(choice)
    return dict(seed=seed, left=odour_left, right=odour_right, dna02_spikes=[spikes_l, spikes_r],
                choice=choice, chosen=chosen, sim_ms=(n_pre + n_dec) * step, rates=ledger.summary(),
                target_orn_hz=dict(air=air, odour=odour))


def run(spec: dict, ctx) -> dict:
    declared = dict(spec['populations'])
    for glom in spec['encoder']['glomeruli']:
        for side in ('L', 'R'):
            declared[f'ORN_{glom}_{side}'] = dict(types=[f'ORN_{glom}'], side=side)
    optional = {k for k in declared if k.startswith('ORN_')} | set(spec.get('optional_populations', []))
    pops = ctx.cells.resolve(declared, allow_empty=optional)
    missing = sorted({k[4:-2] for k in declared if k.startswith('ORN_') and not len(pops.nodes[k])})
    runner = ctx.make_runner(ctx.shared)
    trials, samples = {}, {}
    for comparison in spec['comparisons']:
        a, b = comparison['a'], comparison['b']
        key = f'{a}_vs_{b}'
        rows, score_sum, n_chose = [], [], []
        for seed in ctx.seeds:
            pair = [run_trial(runner, pops, spec, seed, a, b), run_trial(runner, pops, spec, seed, b, a)]
            rows.extend(pair)
            # +1 chose a, -1 chose b.  No-choice trials are excluded from the PI
            # (flies that stay at the choice point are not counted in a T-maze),
            # never scored as 0, which would pull the PI toward 0.
            score_sum.append(float(sum(1 if t['chosen'] == a else -1 for t in pair if t['chosen'] is not None)))
            n_chose.append(float(sum(t['chosen'] is not None for t in pair)))
        trials[key] = rows
        # PI = sum(scores)/choosers, resampled by seed (see stats.bootstrap_ratio_of_means).
        samples[f'PI_{key}'] = dict(num=score_sum, den=n_chose)
        samples[f'choice_rate_{key}'] = dict(k=int(sum(n_chose)), n=len(rows))
        chose = [t for t in rows if t['chosen'] is not None]
        ctx.log(f'{key}: choices {len(chose)}/{len(rows)}, '
                f'PI(a) {np.mean([1 if t["chosen"] == a else -1 for t in chose]) if chose else float("nan"):+.3f}')
    return dict(samples=samples, trials=trials, io=dict(pops.describe(), orn_glomeruli_missing=missing),
                rates={k: [t['rates'] for t in ts] for k, ts in trials.items()},
                sim_ms=sum(t['sim_ms'] for ts in trials.values() for t in ts))
