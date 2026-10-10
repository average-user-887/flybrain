"""v7 medulla timing: HELD-OUT evaluation, run once on the frozen fitted parameter file.

Held-out data (never read by fit_v7.py):
  * Behnia 2014 Fig. 2 (1 s full-field flash from dark): digitised onset/offset peak deflections
    (mV) of Mi1, Tm3, Tm1, Tm2  -> heldout/behnia2014_fig2_flash1s.json (sha256 pinned);
  * Behnia 2014 Extended Data Fig. 1c, released as numbers in the text: the weak-polarity
    response as a fraction of the strong one (mean, s.e.m., n).
Different stimulus class from the training flicker (flashes from darkness vs. Gaussian flicker
about a mean), same lab and preparation; not guaranteed to be different animals.

  python scripts/v7/heldout_v7.py --prereg P --prereg-sha256 SHA --params F --params-sha256 SHA \\
      --heldout FILE --phase1 DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

RECORDED = ('Mi1', 'Tm3', 'Tm1', 'Tm2')
ON_CELLS = ('Mi1', 'Tm3')
# Extended Data Fig. 1c via the text: OFF/ON (Mi1, Tm3) or ON/OFF (Tm1, Tm2) in % (mean, sem, n).
TEXT_RATIO_PCT = {'Mi1': (11.0, 3.5, 7), 'Tm3': (36.6, 7.1, 10), 'Tm1': (26.1, 3.8, 10), 'Tm2': (17.7, 2.3, 11)}
DARK_SETTLE_MS, PRE_MS, FLASH_MS, POST_MS = 3000, 1000, 1000, 1500
OFFSET_WINDOW_MS = 727                 # = the digitised window (24 px at 33 px/s)
FLASH_LEVEL = 20.0                     # intensity 1 = 2 x the flicker mean (encoder 2 x 10)
GATE_AMP_FACTOR = 2.0
GATE_RATIO_SD = 2.0


def load_heldout(path, sha):
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        raise ValueError(f'held-out sha256 {got} != {sha}; refusing')
    return json.loads(raw)


def flash_metrics(v_mV):
    """v_mV: (PRE+FLASH+POST) samples at 1 ms of one cell.  Onset/offset peak deflections
    relative to the pre-flash mean, with the same polarity convention as the digitiser."""
    if len(v_mV) != PRE_MS + FLASH_MS + POST_MS:
        raise ValueError('wrong trace length')
    v = np.asarray(v_mV, float)
    base = v[:PRE_MS].mean()
    on = v[PRE_MS:PRE_MS + FLASH_MS] - base
    off = v[PRE_MS + FLASH_MS:PRE_MS + FLASH_MS + OFFSET_WINDOW_MS] - base
    return on, off


def cell_values(cell, v_mV):
    on, off = flash_metrics(v_mV)
    if cell in ON_CELLS:
        return dict(onset_peak_mV=float(on.max()), offset_peak_mV=float(off.min()))
    return dict(onset_peak_mV=float(on.min()), offset_peak_mV=float(off.max()))


def heldout_gates(model: dict, heldout: dict):
    """H1 polarity (all four), H2 weak/strong ratio within mean +/- 2 population s.d. of the
    recorded cells (>= 3 of 4), H3 strong-polarity peak within a factor 2 of the digitised
    value (>= 3 of 4).  HELDOUT_PASS = H1 and H2 and H3."""
    h1, h2, h3, ratio = {}, {}, {}, {}
    for c in RECORDED:
        m = model[c]
        if c in ON_CELLS:
            strong, weak = m['onset_peak_mV'], m['offset_peak_mV']
            d_strong = heldout[c]['onset_peak_mV']
            h1[c] = strong > 0 and weak < 0
        else:
            strong, weak = m['offset_peak_mV'], m['onset_peak_mV']
            d_strong = heldout[c]['offset_peak_mV']
            h1[c] = strong > 0 and weak < 0
        r = 100.0 * abs(weak) / abs(strong) if strong != 0 else math.inf
        mu, sem, n = TEXT_RATIO_PCT[c]
        sd = sem * math.sqrt(n)
        ratio[c] = r
        h2[c] = max(0.0, mu - GATE_RATIO_SD * sd) <= r <= mu + GATE_RATIO_SD * sd
        h3[c] = strong > 0 and (1.0 / GATE_AMP_FACTOR) <= strong / d_strong <= GATE_AMP_FACTOR
    ok = all(h1.values()) and sum(h2.values()) >= 3 and sum(h3.values()) >= 3
    return dict(H1_polarity=h1, H2_ratio=h2, H3_amplitude=h3, model_ratio_pct=ratio, HELDOUT_PASS=bool(ok))


def run_flash(b, light, cols):
    drive = np.zeros(b.n, np.float32)
    b.reset_state()
    for _ in range(DARK_SETTLE_MS):
        b.step(drive, 1.0)
    V = np.zeros((PRE_MS + FLASH_MS + POST_MS, len(cols)))
    for k in range(len(V)):
        drive[light] = FLASH_LEVEL if PRE_MS <= k < PRE_MS + FLASH_MS else 0.0
        b.step(drive, 1.0)
        V[k] = b.v[cols]
    return V


def main():
    ap = argparse.ArgumentParser()
    for k in ('--prereg', '--params', '--heldout', '--phase1', '--out'):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument('--prereg-sha256', required=True); ap.add_argument('--params-sha256', required=True)
    a = ap.parse_args()
    from fit_v7 import load_prereg
    from brainlab.v6_calibrated import load_params
    sys.path.insert(0, str(REPO / 'scripts' / 'v6'))
    import fit_s2
    fit_s2.PHASE1 = a.phase1
    pre = load_prereg(a.prereg, a.prereg_sha256)
    held = load_heldout(a.heldout, pre['data']['heldout']['fig2_file']['sha256'])
    params, psha = load_params(a.params, a.params_sha256)
    b, ct, light, _cl, cc, info = fit_s2.build_cutout(params, psha)
    cols = [int(cc[t][0]) for t in RECORDED]
    V = run_flash(b, light, cols)
    model = {t: cell_values(t, V[:, j]) for j, t in enumerate(RECORDED)}
    rep = dict(prereg_sha256=a.prereg_sha256, params_sha256=psha, model=model, heldout=held,
               gates=heldout_gates(model, held))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'heldout_report.json').write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps(rep['gates'], indent=1))


if __name__ == '__main__':
    main()
