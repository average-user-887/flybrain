"""v8 input timing: training targets, protocols, measurements, objective and gates (pure logic plus
a small protocol runner over a BrainV6).  Targets are Drosophila NON-MOTION intracellular voltage
timing of R1-R6 and LMCs (qualification/v8/V8_data_pin.json).  The held-out medulla data
are NOT used here (heldout_v8.py).
"""
from __future__ import annotations

import copy
import math

import numpy as np

# --- targets (V8_data_pin.json; prereg v2) ---------------------------------------------------
# Mansour et al. 2026 Supp. Fig. 1b: 10-ms bright flash after brief dark adaptation, 25 C.
# R1-R6: 25.5 +- 2.4 ms, n=4, printed range 24.0-29.0 -> the +-2.4 is consistent with a sample s.d.
# (feasible s.d. given n, mean and range: 2.35-2.38 ms; feasible_sd()).
# LMC: 12.2 +- 1.1 ms, n=5, range 11.0-15.0 -> +-1.1 is neither s.d. (feasible 1.59-1.79) nor
# s.e.m. (0.71-0.80): it is NOT used.  v2 (audit 1 item 2/5a, audit 2 D1): windows are built from
# the PRINTED RANGE widened by 1 ms (model sampling resolution), so every recorded cell, including
# the 15.0 ms LMC, is inside; objective sigmas are the largest feasible sample s.d.
T_FLASH_R = dict(mean=25.5, n=4, lo=24.0, hi=29.0, sigma=2.4)
T_FLASH_LMC = dict(mean=12.2, n=5, lo=11.0, hi=15.0, sigma=1.79)
WINDOW_PAD_MS = 1.0
T_FLASH_LATENCY = (6.5, 1.5)        # '~6-7 ms', both cell types; sigma = declared reading tolerance
# Juusola & Hardie 2001 (v2, audit 1 item 2b): the 'impulse response' k_V(t) is the LINEAR KERNEL
# estimated from Gaussian contrast noise (s.d. ~0.32, white to 150 Hz); the dead time D is a PURE
# DELAY derived from the measured minus the minimum phase (10-90 Hz).  v2 compares like with like:
# the model's noise kernel time to peak (G4) and the model's pure-delay PARAMETER dead_time_ms (G5).
T_KERNEL_TP = {'BG-4': 40.0, 'BG0': 20.0}
T_DEAD_BG0 = 10.0
SIG_TP_FRAC = 0.20

# --- protocols ------------------------------------------------------------------------------
DARK_ADAPT_MS = 10_000              # Mansour: 5-15 s dark adaptation
FLASH_MS, FLASH_LEVEL = 10, 330.0   # encoder 330 = 3.3 x BG0 (JH2001's brightest impulse); ASSUMPTION
FLASH_RECORD_MS = 100
BACKGROUND = {'BG-4': 0.01, 'BG0': 100.0}   # encoder level (x 0.01 = BG0 units)
BG_SETTLE_MS = 3000
NOISE_MS, NOISE_SD, NOISE_CUTOFF_HZ, NOISE_SEED = 4000, 0.32, 150.0, 8002
KERNEL_SNIPPET_MS, KERNEL_HOP_MS, KERNEL_LAGS = 1000, 100, 150
ONSET_FRAC = 0.10                   # flash onset/latency = first |dV| >= 10 % of the peak deflection
SAMPLE_OFFSET_MS = 1.0              # v2 (audit 1 item 5c): V is read at the END of each 1 ms step
# BG-4 decidability (v2, audit 2 D2): the deterministic model resolves any finite deflection, but a
# kernel smaller than this is treated as unmeasurable (pre-written NOT EVALUABLE rule in the prereg)
K_MIN_MV_PER_CONTRAST_MS = 1e-3
SD_MIN_MV = 0.01

LMC_TYPES = ('L1', 'L2')


def feasible_sd(n, mean, lo, hi, step=0.01):
    """(min, max) sample s.d. of n values in [lo, hi] with the given mean whose min is lo and max is
    hi (the printed range): brute force over the n-2 free values on a grid."""
    import itertools
    rest = n * mean - lo - hi
    grid = np.arange(lo, hi + 1e-9, step)
    best = [np.inf, -np.inf]
    if n - 2 == 1:
        cands = [(rest,)] if lo <= rest <= hi else []
    else:
        cands = []
        for c in itertools.combinations_with_replacement(grid, n - 3):
            last = rest - sum(c)
            if lo <= last <= hi:
                cands.append(c + (last,))
    for c in cands:
        sd = float(np.std(np.r_[lo, hi, c], ddof=1))
        best = [min(best[0], sd), max(best[1], sd)]
    return tuple(best)


def window(t):
    return t['lo'] - WINDOW_PAD_MS, t['hi'] + WINDOW_PAD_MS


def jh_noise(seed=NOISE_SEED, n=NOISE_MS):
    """Gaussian contrast noise at 1 ms: white noise through a first-order 150 Hz low-pass, rescaled
    to s.d. 0.32 and clipped to [-1, 1] (JH2001: Gaussian, white to 150 Hz, s.d. ~0.32)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n + 200)
    a = math.exp(-2 * math.pi * NOISE_CUTOFF_HZ * 1e-3)
    y = np.empty_like(w); y[0] = w[0]
    for i in range(1, len(w)):
        y[i] = a * y[i - 1] + (1 - a) * w[i]
    y = y[200:]
    return np.clip(y / y.std() * NOISE_SD, -1.0, 1.0)


def onset_tp(x, frac=ONSET_FRAC):
    """(onset_ms, time_to_peak_ms, peak) of a deflection trace whose sample k was read at the end of
    the 1 ms step that starts at k ms after stimulus onset (v2: times include the +1 ms offset)."""
    x = np.asarray(x, float)
    k = int(np.argmax(np.abs(x)))
    above = np.flatnonzero(np.abs(x[:k + 1]) >= frac * abs(x[k]))
    on = float(above[0]) + SAMPLE_OFFSET_MS if len(above) else math.nan
    return on, float(k) + SAMPLE_OFFSET_MS, float(x[k])


def apply_input_params(b, idx, light, pt_base, fitted):
    """Write the v8 free parameters into a BrainV6 (in memory).  Phototransduction keys are global
    to R1-R6; L1/L2 keys per type.  Synapses (release) and the medulla are never touched."""
    from brainlab.v6_calibrated import Phototransduction
    pt = copy.deepcopy(pt_base)
    for k, v in fitted.get('R1-R6', {}).items():
        if k not in ('tau_p0_ms', 'Kt', 'dead_time_ms'):
            raise ValueError(f'R1-R6 key {k} is not a v8 free parameter')
        pt[k] = int(round(v)) if k == 'dead_time_ms' else float(v)
    b.v6_params = dict(b.v6_params, phototransduction=pt)
    b.pt = Phototransduction(pt, len(light))
    for t in LMC_TYPES:
        i = idx[t][b.graded[idx[t]] != 0]          # v2 (audit 1 item 3): the cells BrainV6 loads
        for k, v in fitted.get(t, {}).items():
            if k == 'tau_m_ms':
                b.tau_m[i] = v
            elif k == 'ih_g':
                b.gh[i] = v
            elif k == 'ih_tau_ms':
                b.h_tau[i] = v
            elif k == 'ih_vh_mV':
                b.h_vh[i] = v
            else:
                raise ValueError(f'{t} key {k} is not a v8 free parameter')
    for t in fitted:
        if t not in ('R1-R6',) + LMC_TYPES:
            raise ValueError(f'type {t} is not free in v8 (medulla and synapses are fixed)')


def run_levels(b, light, cols, levels):
    drive = np.zeros(b.n, np.float32)
    out = np.zeros((len(levels), len(cols)))
    for k, lv in enumerate(levels):
        drive[light] = lv
        b.step(drive, 1.0)
        out[k] = b.v[cols]
    return out


def kernel(s, v):
    """Linear kernel of v (mV) on contrast s (per unit contrast per ms), the same FFT estimator as
    v7 (1 s zero-padded snippets every 100 ms, eps 1 %), after removing the means."""
    s = np.asarray(s, float) - np.mean(s); r = np.asarray(v, float) - np.mean(v)
    nfft = 2 * KERNEL_SNIPPET_MS
    crs = np.zeros(nfft // 2 + 1, complex); css = np.zeros(nfft // 2 + 1); n = 0
    for k in range(0, len(s) - KERNEL_SNIPPET_MS + 1, KERNEL_HOP_MS):
        S = np.fft.rfft(s[k:k + KERNEL_SNIPPET_MS], nfft); R = np.fft.rfft(r[k:k + KERNEL_SNIPPET_MS], nfft)
        crs += R * np.conj(S); css += (S * np.conj(S)).real; n += 1
    K = (crs / n) / (css / n + 0.01 * (css / n).mean())
    return np.fft.irfft(K, nfft)[:KERNEL_LAGS]


def measure(b, light, cols):
    """cols: dict name -> node index for 'R', 'L1', 'L2'.  Returns all protocol measurements.
    Every time is in ms from stimulus onset including SAMPLE_OFFSET_MS."""
    names = list(cols)
    c = [cols[n] for n in names]
    res = {}
    b.reset_state()
    pre = run_levels(b, light, c, np.zeros(DARK_ADAPT_MS))
    V = run_levels(b, light, c, np.r_[np.full(FLASH_MS, FLASH_LEVEL), np.zeros(FLASH_RECORD_MS - FLASH_MS)])
    res['flash'] = {n: dict(zip(('onset_ms', 'tp_ms', 'peak_mV'), onset_tp(V[:, j] - pre[-1, j])))
                    for j, n in enumerate(names)}
    s = jh_noise()
    jr = names.index('R')
    for bg, lv in BACKGROUND.items():
        b.reset_state()
        run_levels(b, light, c, np.full(BG_SETTLE_MS, lv))
        V = run_levels(b, light, c, lv * (1.0 + s))
        K = kernel(s, V[:, jr])
        k = int(np.argmax(np.abs(K)))
        res[bg] = {'R': dict(kernel_tp_ms=float(k) + SAMPLE_OFFSET_MS, kernel_peak=float(K[k]),
                             response_sd_mV=float(np.std(V[:, jr])))}
    return res


def bg_evaluable(m, bg):
    r = m[bg]['R']
    return bool(np.isfinite(r['kernel_peak']) and r['kernel_peak'] >= K_MIN_MV_PER_CONTRAST_MS
                and r['response_sd_mV'] >= SD_MIN_MV)


def _sq(x, mu, sd):
    return ((x - mu) / sd) ** 2 if np.isfinite(x) else 1e6


def objective(m, backgrounds=tuple(BACKGROUND)):
    """Sum of squared standardised errors over the training targets (finite-safe).  The dead time
    is a grid-fixed parameter (gate G5 only).  backgrounds: those declared evaluable at step 0."""
    c = _sq(m['flash']['R']['tp_ms'], T_FLASH_R['mean'], T_FLASH_R['sigma'])
    c += _sq(m['flash']['R']['onset_ms'], *T_FLASH_LATENCY)
    for t in LMC_TYPES:
        c += _sq(m['flash'][t]['tp_ms'], T_FLASH_LMC['mean'], T_FLASH_LMC['sigma'])
        c += _sq(m['flash'][t]['onset_ms'], *T_FLASH_LATENCY)
    for bg in backgrounds:
        c += _sq(m[bg]['R']['kernel_tp_ms'], T_KERNEL_TP[bg], SIG_TP_FRAC * T_KERNEL_TP[bg])
        if not bg_evaluable(m, bg):
            c += 1e6
    return float(c)


# --- training gates (frozen in V8_input_prereg_v2.json) ----------------------------------------
G_LATENCY = (4.0, 10.0)
G_LATENCY_DIFF = 3.0
G_TP_FRAC, G_DEAD_FRAC = 0.25, 0.30


def train_gates(m, dead_time_ms, backgrounds=tuple(BACKGROUND)):
    """G1-G5 on one run (prereg v3).
    R stage:   G1 (flash R tp), G3_R (flash R onset 4-10 ms), G4 (kernel tp), G5 (pure delay D).
    LMC stage: G2 (flash L1/L2 tp, earlier than R), G3_L (L1/L2 onsets 4-10 ms, |L - R| <= 3 ms).
    v3 (audits of v2): the R onset is an R-stage gate, so an R-only failure can never be read as
    F1_LMC.  Engine bounds (best_case_onsets): R onset >= D + 1 ms (ring delay D + end-of-step read),
    L onset >= D + 2 ms (1.8 ms graded-release delay); G3's 10 ms ceiling therefore admits only
    D <= 8, and G5 (7-13 ms) only D >= 7: grid {7, 8}.  Over that grid G5 is NON-BINDING (it acts
    only by excluding D <= 6 from the grid)."""
    f = m['flash']
    r_lo, r_hi = window(T_FLASH_R); l_lo, l_hi = window(T_FLASH_LMC)
    g1 = bool(r_lo <= f['R']['tp_ms'] <= r_hi)
    g2 = {t: bool((l_lo <= f[t]['tp_ms'] <= l_hi) and f[t]['tp_ms'] < f['R']['tp_ms']) for t in LMC_TYPES}
    g3_r = bool(G_LATENCY[0] <= f['R']['onset_ms'] <= G_LATENCY[1])
    g3_l = {t: bool(G_LATENCY[0] <= f[t]['onset_ms'] <= G_LATENCY[1]) for t in LMC_TYPES}
    g3_l['no_synaptic_delay'] = bool(all(abs(f[t]['onset_ms'] - f['R']['onset_ms']) <= G_LATENCY_DIFF for t in LMC_TYPES))
    g4 = {bg: bool(bg_evaluable(m, bg) and abs(m[bg]['R']['kernel_tp_ms'] - T_KERNEL_TP[bg]) <= G_TP_FRAC * T_KERNEL_TP[bg])
          for bg in backgrounds}
    if len(backgrounds) == 2:
        g4['adapts'] = bool(m['BG0']['R']['kernel_tp_ms'] < m['BG-4']['R']['kernel_tp_ms'])
    g5 = bool(abs(dead_time_ms - T_DEAD_BG0) <= G_DEAD_FRAC * T_DEAD_BG0)
    r_ok = g1 and g3_r and all(g4.values()) and g5
    l_ok = all(g2.values()) and all(g3_l.values())
    return dict(G1_flash_R_tp=g1, G2_flash_LMC_tp=g2, G3_R_onset=g3_r, G3_L_onset=g3_l, G4_kernel_tp=g4,
                G5_dead_time=g5, R_STAGE_PASS=bool(r_ok), LMC_STAGE_PASS=bool(l_ok), TRAIN_PASS=bool(r_ok and l_ok))


def best_case_onsets(D, base_params, dark_ms=2000, weight=-50.0):
    """Engine lower bounds on the flash onsets for pure delay D (prereg v3 grid derivation): a
    minimal BrainV6 (one R1-R6 node driving one L1 and one L2 node through an extreme weight) with
    the fastest free values (tau_p0 0.5 ms, L tau_m 1 ms, no Ih) and the base R release; returns
    {'R': onset, 'L1': onset, 'L2': onset} in ms with the end-of-step offset."""
    from brainlab.v6_calibrated import BrainV6
    pt = dict(base_params['phototransduction'], dead_time_ms=int(D), tau_p0_ms=0.5)
    types = {'R1-R6': base_params['types']['R1-R6'], 'L1': {'tau_m_ms': 1.0, 'e_leak_mV': -40.0},
             'L2': {'tau_m_ms': 1.0, 'e_leak_mV': -40.0}}
    arrays = dict(ptr=np.array([0, 2, 2, 2], np.int64), post=np.array([1, 2], np.int32),
                  weight=np.array([weight, weight], np.float32), ids=np.arange(1, 4, dtype=np.int64))
    b = BrainV6(None, dict(phototransduction=pt, types=types), 'e' * 64, light_nodes=np.array([0]),
                cell_type=np.array(['R1-R6', 'L1', 'L2']), arrays=arrays, graded_policy=np.ones(3, np.uint8))
    b.reset_state()
    pre = run_levels(b, np.array([0]), [0, 1, 2], np.zeros(dark_ms))
    X = run_levels(b, np.array([0]), [0, 1, 2], np.r_[np.full(FLASH_MS, FLASH_LEVEL), np.zeros(FLASH_RECORD_MS - FLASH_MS)])
    return {n: onset_tp(X[:, j] - pre[-1, j])[0] for j, n in enumerate(('R', 'L1', 'L2'))}


def g3_feasible(onsets):
    return bool(all(G_LATENCY[0] <= onsets[n] <= G_LATENCY[1] for n in ('R',) + LMC_TYPES)
                and all(abs(onsets[t] - onsets['R']) <= G_LATENCY_DIFF for t in LMC_TYPES))


MIN_COMPLETED_FRACTION = 0.5


def outcome(reports, n_runs):
    """v2 (audit 1 item 5b / audit 2 D3):
      fewer than half of the runs completed            -> NOT_EVALUABLE (never F1)
      some completed run has TRAIN_PASS                 -> TRAIN_PASS, lowest cost among them (tie: run id)
      no completed run passes the R-stage gates (G1, G3_R, G4, G5) -> F1_R (photoreceptor stage not reproduced)
      else (R stage passes somewhere, LMC never jointly) -> F1_LMC (R->LMC transfer fails on timing)
    Both F1 readings are CONDITIONAL on the flash level, noise contrast, single fixed dead time and
    the rejected S1-A3 gain."""
    done = [r for r in reports if r.get('status') == 'completed']
    if len(done) < MIN_COMPLETED_FRACTION * n_runs:
        return 'NOT_EVALUABLE', None
    passing = [r for r in done if r['gates']['TRAIN_PASS']]
    if passing:
        return 'TRAIN_PASS', min(passing, key=lambda r: (r['cost'], r['run']))
    if not any(r['gates']['R_STAGE_PASS'] for r in done):
        return 'F1_R', None
    return 'F1_LMC', None
