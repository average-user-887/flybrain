"""Looming escape: LC4/LPLC2 drive in, Giant Fiber (DNp01) spikes out.

Stimulus: a dark disc of half-size r approaching at constant speed v, so
theta(t) = 2*atan((r/v) / (t_c - t)) (full angular size, t_c = collision).
Each loom starts at ``theta_start_deg`` and expands to ``theta_end_deg``
(collision when absent), then holds the final frame for ``post_ms``.

Encoder (declared engineering assumption, spec ``encoder``): LPLC2 is driven
by angular size and LC4 by angular velocity, the division of labour reported
for the GF's two main visual inputs (von Reyn et al. 2017; Ache et al. 2019).
Each is a clipped linear ramp times ``i_max`` with per-cell multiplicative
noise, the same form and amplitude as the WP5 optomotor encoder.  Nothing
reaches the GF except through the graph; there is no direct DNp01 injection.

Decoder: any DNp01 spike between loom onset and the end of the post window is
a GF-mediated (short-mode) takeoff (von Reyn et al. 2014: a GF spike forces a
short-mode takeoff).  Long-mode takeoffs are carried by parallel descending
pathways and are NOT decoded here.
"""
from __future__ import annotations

import math

import numpy as np

from ..runner import SpikeLedger


def theta_series(rv_ms: float, stim: dict, receding: bool = False) -> np.ndarray:
    """Angular size (deg) at the START of each control step, onset to the end size.

    The end size is ``theta_end_deg`` (the last step is clamped to it) or, when
    the spec has none, collision.  A receding disc runs the same series backwards.
    """
    step = stim['step_ms']
    t_total = rv_ms / math.tan(math.radians(stim['theta_start_deg']) / 2)   # ms from onset to collision
    n = int(math.floor(t_total / step))
    ttc = t_total - np.arange(n) * step
    theta = np.degrees(2 * np.arctan(rv_ms / ttc))
    end = stim.get('theta_end_deg')
    if end is not None:
        theta = np.append(theta[theta < end], float(end))
    return theta[::-1].copy() if receding else theta


def drive(theta_deg: float, dtheta_deg_s: float, enc: dict) -> tuple:
    lp = enc['lplc2']
    size = (theta_deg - lp['theta_on_deg']) / (lp['theta_sat_deg'] - lp['theta_on_deg'])
    lc = enc['lc4']
    vel = dtheta_deg_s / lc['omega_sat_deg_s']
    return (enc['i_max'] * min(1.0, max(0.0, size)), enc['i_max'] * min(1.0, max(0.0, vel)))


def run_trial(runner, pops, spec: dict, seed: int, rv_ms: float, condition: str) -> dict:
    stim, enc = spec['stimulus'], spec['encoder']
    step = float(stim['step_ms'])
    runner.reset()
    rng = np.random.default_rng(seed)
    lplc2 = np.concatenate([pops.nodes['LPLC2_L'], pops.nodes['LPLC2_R']])
    lc4 = np.concatenate([pops.nodes['LC4_L'], pops.nodes['LC4_R']])
    gf = pops.nodes['GF']
    ledger = SpikeLedger(runner.n, pops.nodes)
    theta = theta_series(rv_ms, stim, receding=(condition == 'receding'))
    last = theta[-1]
    frames = ([(None, 'baseline')] * int(round(stim['pre_ms'] / step))
              + [(th, 'stimulus') for th in theta]
              + [(last, 'stimulus')] * int(round(stim['post_ms'] / step)))
    currents = np.zeros(runner.n, np.float32)
    prev = None
    gf_first = None
    for i, (th, window) in enumerate(frames):
        currents.fill(0.0)
        n_lp, n_lc = rng.standard_normal(len(lplc2)), rng.standard_normal(len(lc4))
        if th is not None and condition != 'sham_no_input':
            dth = 0.0 if prev is None else (th - prev) / (step / 1000.0)
            d_size, d_vel = drive(th, dth, enc)
            currents[lplc2] = np.maximum(0.0, d_size * (1 + enc['noise_sd'] * n_lp))
            currents[lc4] = np.maximum(0.0, d_vel * (1 + enc['noise_sd'] * n_lc))
        prev = th
        counts = runner.step(currents, step)
        ledger.add(counts, step, window)
        if window == 'stimulus' and gf_first is None and counts[gf].sum() > 0:
            gf_first = dict(step=i, theta_deg=float(th) if th is not None else None,
                            ms_after_onset=(i - int(round(stim['pre_ms'] / step))) * step)
    return dict(seed=seed, rv_ms=rv_ms, condition=condition, gf_spike=gf_first is not None, gf_first=gf_first,
                sim_ms=len(frames) * step, rates=ledger.summary())


def run(spec: dict, ctx) -> dict:
    pops = ctx.cells.resolve(spec['populations'])
    runner = ctx.make_runner(ctx.shared)
    trials = {}
    for rv in spec['stimulus']['r_over_v_ms']:
        key = f'loom_rv{rv:g}'
        trials[key] = [run_trial(runner, pops, spec, seed, rv, 'loom') for seed in ctx.seeds]
        ctx.log(f'{key}: GF spike in {sum(t["gf_spike"] for t in trials[key])}/{len(ctx.seeds)}')
    for control, rv in spec['controls'].items():
        trials[control] = [run_trial(runner, pops, spec, seed, rv, control) for seed in ctx.seeds]
        ctx.log(f'{control}: GF spike in {sum(t["gf_spike"] for t in trials[control])}/{len(ctx.seeds)}')
    samples = dict(
        gf_by_log10_rv={math.log10(rv): [float(t['gf_spike']) for t in trials[f'loom_rv{rv:g}']]
                        for rv in spec['stimulus']['r_over_v_ms']})
    for control in spec['controls']:
        samples[f'{control}_gf'] = [float(t['gf_spike']) for t in trials[control]]
    pooled = [t['gf_spike'] for rv in spec['reported']['pooled_r_over_v_ms'] for t in trials[f'loom_rv{rv:g}']]
    samples['pooled_gf_fraction'] = dict(k=int(sum(pooled)), n=len(pooled))
    samples['theta_at_gf_deg'] = {k: [t['gf_first']['theta_deg'] for t in ts if t['gf_first']]
                                  for k, ts in trials.items()}
    samples['theta_at_first_gf_all_looms_deg'] = [t['gf_first']['theta_deg'] for rv in spec['stimulus']['r_over_v_ms']
                                                  for t in trials[f'loom_rv{rv:g}'] if t['gf_first']]
    return dict(samples=samples, trials=trials, io=pops.describe(),
                rates={k: [t['rates'] for t in ts] for k, ts in trials.items()},
                sim_ms=sum(t['sim_ms'] for ts in trials.values() for t in ts))
