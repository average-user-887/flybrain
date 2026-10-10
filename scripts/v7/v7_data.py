"""v7 medulla timing: TRAINING data loading, the Behnia 2014 stimulus, the filter estimator and the
mV observation model.  Pure functions; no simulation here.

Owner rules (10 Oct 2026): per-type parameters only, fitted only to NON-MOTION recordings of
medulla cells upstream of T4/T5; no edge changes; held-out data kept separate; calcium is not an
mV target.  The training set is Behnia et al. 2014 Fig. 3b/3e (in vivo whole-cell, mV); the
held-out set lives in ``heldout_v7.py`` and is never imported here.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

RECORDED = ('Mi1', 'Tm3', 'Tm1', 'Tm2')
POLARITY = {'Mi1': 1.0, 'Tm3': 1.0, 'Tm1': -1.0, 'Tm2': -1.0}     # sign of the main filter lobe
TEXT_PEAK_MS = {'Mi1': (71.0, 3.8, 7), 'Tm3': (53.0, 5.2, 11), 'Tm1': (56.0, 3.8, 15), 'Tm2': (43.0, 2.7, 14)}

# Stimulus (Behnia 2014 Methods): q(t) = m (1 + s(t)), s Gaussian, s.d. 0.5, exponential
# autocovariance with correlation time 10 ms, frames at 240 Hz, q bounded to [0, 2m].
NOISE_SIGMA = 0.5
NOISE_TAU_MS = 10.0
FRAME_HZ = 240.0
NOISE_MS = 10_000
# Filter estimation (Baccus & Meister 2002 as in Behnia 2014 Methods).
SNIPPET_MS = 5000
HOP_MS = 100
EPS_FRAC = 0.01
HIGHPASS_HZ = 0.008
N_LAGS = 220                          # 0..219 ms, inside every digitised trace


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_training(directory, pins: dict):
    """Digitised Fig. 3 filters, one CSV per type, each checked against its pinned sha256.

    Refuses anything under a ``heldout`` path and any type outside ``RECORDED``."""
    directory = Path(directory)
    if 'heldout' in directory.resolve().parts:
        raise ValueError('training loader refuses a held-out directory')
    out = {}
    for cell, pin in pins.items():
        if cell not in RECORDED:
            raise ValueError(f'{cell} is not a recorded training type')
        path = directory / pin['file']
        if 'heldout' in path.name:
            raise ValueError('training loader refuses a held-out file')
        got = sha256_file(path)
        if got != pin['sha256']:
            raise ValueError(f'{path.name}: sha256 {got} != pinned {pin["sha256"]}; refusing')
        rows = list(csv.DictReader(path.open()))
        t = np.array([float(r['t_ms']) for r in rows])
        if not np.array_equal(t, np.arange(len(t), dtype=float)):
            raise ValueError(f'{path.name}: expected a 1 ms grid from 0')
        out[cell] = dict(t_ms=t, mean=np.array([float(r['filter_mV_per_contrast_per_ms']) for r in rows]),
                         sem=np.array([float(r['sem']) for r in rows]),
                         interpolated=np.array([int(r['interpolated']) for r in rows], bool))
    return out


def behnia_noise(duration_ms: int, seed: int, sigma=NOISE_SIGMA, tau_ms=NOISE_TAU_MS, frame_hz=FRAME_HZ):
    """Contrast s(t) at 1 ms resolution: an AR(1) Gaussian process sampled per 240 Hz frame
    (autocovariance sigma^2 exp(-|t|/tau)), held for the frame, clipped to [-1, 1] (q in [0, 2m])."""
    frame_ms = 1000.0 / frame_hz
    n_frames = int(math.ceil(duration_ms / frame_ms)) + 1
    a = math.exp(-frame_ms / tau_ms)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_frames)
    f = np.empty(n_frames)
    f[0] = sigma * z[0]
    for k in range(1, n_frames):
        f[k] = a * f[k - 1] + math.sqrt(1.0 - a * a) * sigma * z[k]
    idx = np.floor(np.arange(duration_ms) / frame_ms).astype(int)
    return np.clip(f[idx], -1.0, 1.0)


def contrast_to_encoder(s, mean_level=10.0):
    """Encoder level on R1-R6 (v6: level x 0.01 = light in BG0 units).  Mean = BG-1 (level 10),
    the gray level of the S2 fit and of the frozen S3 protocol."""
    return mean_level * (1.0 + np.asarray(s, float))


def highpass(r, fc_hz=HIGHPASS_HZ, dt_ms=1.0):
    """First-order high-pass (Behnia's 0.008 Hz pre-filter; the 60 Hz notch is a no-op on a
    noise-free model and is not applied)."""
    r = np.asarray(r, float)
    rc = 1.0 / (2 * math.pi * fc_hz)
    dt = dt_ms * 1e-3
    a = rc / (rc + dt)
    y = np.empty_like(r)
    y[0] = 0.0
    for i in range(1, len(r)):
        y[i] = a * (y[i - 1] + r[i] - r[i - 1])
    return y


def estimate_filter(s, r, n_lags=N_LAGS, snippet_ms=SNIPPET_MS, hop_ms=HOP_MS, eps_frac=EPS_FRAC):
    """K(w) = <R S*> / (<S S*> + eps) over zero-padded snippets (5 s, every 0.1 s);
    eps = eps_frac x mean <S S*>.  Returns K(t) for lags 0..n_lags-1 ms, in units of r per
    contrast per ms (1 ms bins)."""
    s = np.asarray(s, float) - np.mean(s)
    r = np.asarray(r, float) - np.mean(r)
    if s.shape != r.shape:
        raise ValueError('stimulus and response must have equal length')
    nfft = 2 * snippet_ms
    starts = range(0, len(s) - snippet_ms + 1, hop_ms)
    crs = np.zeros(nfft // 2 + 1, complex)
    css = np.zeros(nfft // 2 + 1)
    n = 0
    for k in starts:
        S = np.fft.rfft(s[k:k + snippet_ms], nfft)
        R = np.fft.rfft(r[k:k + snippet_ms], nfft)
        crs += R * np.conj(S); css += (S * np.conj(S)).real; n += 1
    if n == 0:
        raise ValueError('response shorter than one snippet')
    crs /= n; css /= n
    K = crs / (css + eps_frac * css.mean())
    return np.fft.irfft(K, nfft)[:n_lags]


def observe_filter(s, v_mV, settle_ms=0):
    """mV observation model: somatic V of the recorded cell (single compartment = soma),
    high-passed at 0.008 Hz, then the same filter estimator as the data."""
    v = np.asarray(v_mV, float)[settle_ms:]
    return estimate_filter(np.asarray(s, float)[settle_ms:], highpass(v))


SIGMA_DIG = 0.01      # digitisation s.d. (mV contrast^-1 ms^-1): ~2 px in y incl. time misregistration
SIGMA_FLOOR = 0.02    # no lag is weighted more than this


def objective(model_K: dict, data: dict, n_lags=N_LAGS):
    """Training objective: mean over the four types of mean_t ((K_model - K_data) / sigma_t)^2,
    sigma_t = max(sqrt(sem_t^2 + SIGMA_DIG^2), SIGMA_FLOOR).  Also returns per-type NRMSE."""
    per, nrmse = {}, {}
    for cell in RECORDED:
        d = data[cell]
        kd = d['mean'][:n_lags]; km = np.asarray(model_K[cell])[:n_lags]
        sig = np.maximum(np.sqrt(d['sem'][:n_lags] ** 2 + SIGMA_DIG ** 2), SIGMA_FLOOR)
        per[cell] = float(np.mean(((km - kd) / sig) ** 2))
        nrmse[cell] = float(np.sqrt(np.sum((km - kd) ** 2) / np.sum(kd ** 2)))
    return float(np.mean(list(per.values()))), per, nrmse


def peak_ms(K, cell):
    return float(np.argmax(POLARITY[cell] * np.asarray(K)))


# Training gates (frozen in qualification/v7/V7_medulla_prereg.json before any fit).
GATE_NRMSE = 0.35
GATE_PEAK_SLACK_MS = 5.0
GATE_ORDER_MS = 5.0


def train_gates(model_K: dict, data: dict):
    _, _, nrmse = objective(model_K, data)
    pk_m = {c: peak_ms(model_K[c], c) for c in RECORDED}
    pk_d = {c: peak_ms(data[c]['mean'], c) for c in RECORDED}
    a1 = {c: nrmse[c] <= GATE_NRMSE for c in RECORDED}
    a2 = {c: abs(pk_m[c] - pk_d[c]) <= 2 * TEXT_PEAK_MS[c][1] + GATE_PEAK_SLACK_MS for c in RECORDED}
    a3 = dict(ON=pk_m['Mi1'] - pk_m['Tm3'] >= GATE_ORDER_MS, OFF=pk_m['Tm1'] - pk_m['Tm2'] >= GATE_ORDER_MS)
    a4 = {c: float(POLARITY[c] * np.asarray(model_K[c])[int(pk_d[c])]) > 0 for c in RECORDED}
    ok = all(a1.values()) and all(a2.values()) and all(a3.values()) and all(a4.values())
    return dict(A1_nrmse=a1, A2_peak=a2, A3_order=a3, A4_polarity=a4, nrmse=nrmse, peak_model_ms=pk_m,
                peak_data_ms=pk_d, TRAIN_PASS=bool(ok))
