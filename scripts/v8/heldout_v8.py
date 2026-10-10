"""v8 HELD-OUT: Behnia 2014 Mi1/Tm3/Tm1/Tm2 voltage-filter timing PREDICTED by the frozen v8 input
parameters with NO medulla fitting (medulla and synapses at v6 leak40), on the frozen S2 cutout,
with the v7 v2 noise protocol and observation model (filter against u = c/2).  Run once, after
params_v8.json is frozen and committed.  Independent data (other lab, cells, method) but NOT
analyst-blind (values seen, v7 F1 known).

  python scripts/v8/heldout_v8.py --prereg P --prereg-sha256 SHA --params F --params-sha256 SHA \\
      --training-dir <v7 Behnia digitised training dir> --v7-prereg P7 --v7-prereg-sha256 SHA7 --phase1 DIR --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, HERE, REPO / 'scripts' / 'v7', REPO / 'scripts' / 'v6'):
    sys.path.insert(0, str(p))

GATE_PEAK_SLACK_MS = 5.0
GATE_ORDER_MS = 5.0
MIN_PEAKS_OK = 3
SAMPLE_OFFSET_MS = 1.0


def heldout_gates(peak_model, peak_data, text_sem):
    """H1: |model - data peak| <= 2 x text s.e.m. + 5 ms for >= 3 of 4; H2: Mi1 later than Tm3 and
    Tm1 later than Tm2 by >= 5 ms.  HELDOUT_PASS = H1 and H2."""
    h1 = {c: abs(peak_model[c] - peak_data[c]) <= 2 * text_sem[c] + GATE_PEAK_SLACK_MS for c in peak_data}
    h2 = dict(ON=peak_model['Mi1'] - peak_model['Tm3'] >= GATE_ORDER_MS,
              OFF=peak_model['Tm1'] - peak_model['Tm2'] >= GATE_ORDER_MS)
    return dict(H1_peaks=h1, H2_order=h2, HELDOUT_PASS=bool(sum(h1.values()) >= MIN_PEAKS_OK and all(h2.values())),
                label='independent data, not analyst-blind')


def main():
    ap = argparse.ArgumentParser()
    for k in ('--prereg', '--params', '--training-dir', '--v7-prereg', '--phase1', '--out'):
        ap.add_argument(k, type=Path, required=True)
    for k in ('--prereg-sha256', '--params-sha256', '--v7-prereg-sha256'):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    import fit_s2
    import fit_v7
    import v7_data as D
    from fit_v8 import load_prereg
    from brainlab.v6_calibrated import load_params
    load_prereg(a.prereg, a.prereg_sha256)
    pre7 = fit_v7.load_prereg(a.v7_prereg, a.v7_prereg_sha256)
    fit_s2.PHASE1 = a.phase1
    params, psha = load_params(a.params, a.params_sha256)
    data = D.load_training(a.training_dir, pre7['data']['training']['files'])
    b, ct, light, _cl, cc, info = fit_s2.build_cutout(params, psha)
    m = fit_v7.NoiseModel(b, ct, light, cc, int(pre7['stimulus']['training_seed']))
    K, sd = m.filters_and_sd({})
    pk_m = {c: D.peak_ms(K[c], c) + SAMPLE_OFFSET_MS for c in D.RECORDED}   # v2: V read at the end of each step
    pk_d = {c: D.peak_ms(data[c]['mean'], c) for c in D.RECORDED}
    sem = {c: D.TEXT_PEAK_MS[c][1] for c in D.RECORDED}
    cost, per, nrmse = D.objective(K, data)
    rep = dict(params_sha256=psha, peak_model_ms=pk_m, peak_data_ms=pk_d, gates=heldout_gates(pk_m, pk_d, sem),
               descriptive=dict(nrmse=nrmse, response_sd_mV=sd, fig3a_response_sd_mV=D.FIG3A_RESPONSE_SD_MV),
               filters={c: K[c].tolist() for c in D.RECORDED})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps(rep['gates'], indent=1))


if __name__ == '__main__':
    main()
