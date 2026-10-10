"""v7 full-network filter diagnostic (prereg v2 'network_filter_diagnostic'; DESCRIPTIVE ONLY:
no gate, no fitting, cannot change any parameter).

The S3 BrainV6 (full MaleCNS graph, both eyes' R1-R6 as light nodes, exactly as
scripts/v6/measure_tuning_v6.py builds it) with the FROZEN params_v7.json, driven on every R1-R6
by the same full-field Behnia noise as the fit (seed 1; 3 s gray at encoder 10, then 10 s),
1 ms steps.  Readout: the right-eye central-column (17, 27) Mi1/Tm3/Tm1/Tm2 of the Path B column
map; the same observation model (v7_data.observe_filter, response_sd).  Run from the checkout
root (the graph path is relative, as in the runner).

  python scripts/v7/network_filters_v7.py --params F --params-sha256 SHA --prereg-v2 P \\
      --prereg-v2-sha256 SHA --training-dir DIR --phase1 DIR --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import v7_data as D  # noqa: E402

CENTER = (17, 27)
GRAPH_REL = 'outputs/brainlab/malecns_v1/graph.npz'
SETTLE_MS = 3000


def central_cells(phase1):
    pc = np.load(Path(phase1) / 'cell_columns.npz')
    out = {}
    for t in D.RECORDED:
        m = (pc['cell_type'] == t) & (pc['eye'] == 'R') & (pc['hex1'].astype(int) == CENTER[0]) & \
            (pc['hex2'].astype(int) == CENTER[1])
        nodes = pc['node'][m].astype(int)
        if len(nodes) != 1:
            raise SystemExit(f'expected one central {t}, got {len(nodes)}')
        out[t] = int(nodes[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('--params', '--prereg-v2', '--training-dir', '--phase1', '--out'):
        ap.add_argument(k, type=Path, required=True)
    ap.add_argument('--params-sha256', required=True); ap.add_argument('--prereg-v2-sha256', required=True)
    a = ap.parse_args()
    raw = a.prereg_v2.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.prereg_v2_sha256:
        raise SystemExit('prereg v2 sha256 mismatch')
    pre = json.loads(raw)
    if Path.cwd().resolve() != REPO.resolve():
        raise SystemExit('run from the checkout root')
    g = hashlib.sha256(Path(GRAPH_REL).resolve().read_bytes()).hexdigest()
    if g != pre['downstream_s3']['environment']['graph_sha256']:
        raise SystemExit('graph sha256 differs from the pin')
    from brainlab.photoreceptor_io import _tables, resolve_photoreceptor_io
    from brainlab.v6_calibrated import BrainV6, load_params
    params, psha = load_params(a.params, a.params_sha256)
    data = D.load_training(a.training_dir, pre['data']['training']['files'])
    io = resolve_photoreceptor_io()
    tab = _tables(None)
    light = np.concatenate([io.r_nodes['L'], io.r_nodes['R']])
    t0 = time.time()
    b = BrainV6(GRAPH_REL, params, psha, light_nodes=light, cell_type=tab.cell_type.fillna('').to_numpy())
    cells = central_cells(a.phase1)
    cols = np.array([cells[t] for t in D.RECORDED])
    s = D.behnia_noise(D.NOISE_MS, int(pre['stimulus']['training_seed']))
    enc = D.contrast_to_encoder(s).astype(np.float32)
    drive = np.zeros(b.n, np.float32); drive[light] = 10.0
    for _ in range(SETTLE_MS):
        b.step(drive, 1.0)
    V = np.zeros((D.NOISE_MS, len(cols)))
    for k in range(D.NOISE_MS):
        drive[light] = enc[k]
        b.step(drive, 1.0)
        V[k] = b.v[cols]
    K = {t: D.observe_filter(s, V[:, j]) for j, t in enumerate(D.RECORDED)}
    sd = {t: D.response_sd(V[:, j]) for j, t in enumerate(D.RECORDED)}
    cost, per, nrmse = D.objective(K, data)
    rep = dict(label='DESCRIPTIVE full-network diagnostic; no gate', params_sha256=psha, graph_sha256=g,
               cells=cells, wall_s=round(time.time() - t0, 1), cost=cost, nrmse=nrmse,
               peak_model_ms={t: D.peak_ms(K[t], t) for t in D.RECORDED},
               peak_data_ms={t: D.peak_ms(data[t]['mean'], t) for t in D.RECORDED},
               response_sd_mV=sd, fig3a_response_sd_mV=D.FIG3A_RESPONSE_SD_MV,
               gates_for_reference_only=D.train_gates(K, data, sd),
               filters={t: K[t].tolist() for t in D.RECORDED})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ('cells', 'wall_s', 'nrmse', 'peak_model_ms', 'response_sd_mV')}, indent=1))


if __name__ == '__main__':
    main()
