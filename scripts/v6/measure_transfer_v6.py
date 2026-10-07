"""v6: the Path B phase-3 transfer protocol, unchanged, run on BrainV6.

A copy of scripts/pathB/measure_transfer.py (Path B commit 89bc3d9) in which the
only changes are: the contract path is passed in; the arm is ``v6`` built as
``BrainV6`` from a frozen parameter file whose sha256 is checked; the encoder
levels on R1-R6 become light through the frozen phototransduction.  Stimulus
levels, timing, windows, columns, seeds and readouts are Path B's.

    python scripts/v6/measure_transfer_v6.py --contract C --contract-sha256 SHA \
        --params P --params-sha256 SHA --phase1 DIR --out DIR [--smoke]

Writes ``<out>/<arm>/<condition>.npz`` (per-cell window readouts and group
time courses) and ``<out>/<arm>/meta.json``.  Analysis is a separate script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from brainlab.photoreceptor_io import _tables  # noqa: E402
from brainlab.v6_calibrated import BrainV6, load_params  # noqa: E402
from brainlab.graph_identity import resolve_graph_dir  # noqa: E402
from brainlab.photoreceptor_io import resolve_photoreceptor_io  # noqa: E402

BASE, ON, OFF = 10.0, 20.0, 0.0
SETTLE_MS, PRE_MS, STEP_MS, POST_MS = 1000, 100, 200, 200
WINDOWS = {'B': (-100, 0), 'E': (0, 50), 'S': (50, 200), 'O': (200, 300)}
K_COLUMNS, MIN_SEP = 8, 10
SEEDS = (0, 1, 2)
GROUP_OF = {'T4a': 'T4', 'T4b': 'T4', 'T4c': 'T4', 'T4d': 'T4',
            'T5a': 'T5', 'T5b': 'T5', 'T5c': 'T5', 'T5d': 'T5'}
NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))


def hexdist(a, b):
    dq, dr = a[0] - b[0], a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq - dr))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick_columns(table_csv, seed):
    rows = list(csv.DictReader(open(table_csv)))
    have = {(r['eye'], int(r['hex1']), int(r['hex2'])) for r in rows}
    cand = sorted((int(r['hex1']), int(r['hex2'])) for r in rows
                  if r['eye'] == 'R' and r['overall'] == 'COMPLETE'
                  and all(('R', int(r['hex1']) + a, int(r['hex2']) + b) in have for a, b in NEIGH))
    rng = np.random.default_rng(seed)
    chosen = []
    for k in rng.permutation(len(cand)):
        c = cand[k]
        if all(hexdist(c, x) >= MIN_SEP for x in chosen):
            chosen.append(c)
        if len(chosen) == K_COLUMNS:
            break
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--params', type=Path, required=True)
    ap.add_argument('--params-sha256', required=True)
    ap.add_argument('--arm', default='v6')
    ap.add_argument('--phase1', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--contract-sha256', required=True)
    ap.add_argument('--smoke', action='store_true', help='5 ms crash smoke; readouts discarded')
    args = ap.parse_args(argv)
    got = sha256(args.contract)
    if got != args.contract_sha256:
        raise SystemExit(f'contract sha256 {got} != {args.contract_sha256}; refusing')

    settle, pre, stepl, postl = (SETTLE_MS, PRE_MS, STEP_MS, POST_MS) if not args.smoke else (2, 1, 2, 2)
    gdir, _ = resolve_graph_dir(None)
    t0 = time.time()
    params, psha = load_params(args.params, args.params_sha256)
    io = resolve_photoreceptor_io()
    tab = _tables(None)
    assert (tab.node_index.to_numpy() == np.arange(len(tab))).all()
    all_ct = tab.cell_type.fillna('').to_numpy()
    r_light = np.concatenate([io.r_nodes['L'], io.r_nodes['R']])
    brain = BrainV6(str(gdir / 'graph.npz'), params, psha, light_nodes=r_light, cell_type=all_ct)
    cc = np.load(args.phase1 / 'cell_columns.npz')
    rec = cc['node']
    ctype = cc['cell_type']
    group = np.array([GROUP_OF.get(t, t) for t in ctype])
    eye, state = cc['eye'], cc['column_state']
    hexes = np.stack([cc['hex1'], cc['hex2']], 1)
    keys = sorted({(g, e, s) for g, e, s in zip(group, eye, state)})
    key_idx = {k: np.flatnonzero((group == k[0]) & (eye == k[1]) & (state == k[2])) for k in keys}

    r_all = np.concatenate([io.r_nodes['L'], io.r_nodes['R']])
    base = np.zeros(brain.n, np.float32)
    base[r_all] = BASE

    conditions = {'C0_sham': None, 'C1_fullfield_ON': (r_all, ON), 'C2_fullfield_OFF': (r_all, OFF)}
    column_sets = {}
    for seed in SEEDS:
        cols = pick_columns(args.phase1 / 'column_table_1.csv', seed)
        cart = [tuple(c) for c in io.r_cartridge['R'].tolist()]
        sel = np.array([io.r_nodes['R'][i] for i, c in enumerate(cart) if c in set(cols)], np.int64)
        conditions[f'C3_column_ON_seed{seed}'] = (sel, ON)
        column_sets[seed] = dict(columns=cols, photoreceptors=int(len(sel)))

    # settle at baseline, then snapshot
    brain.reset_state()
    for _ in range(settle // 10 if settle >= 10 else 1):
        brain.step(base, 10.0 if settle >= 10 else float(settle))
    snap = brain.snapshot_state()
    t_settle = time.time() - t0
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    total = pre + stepl + postl
    timing = {}
    for name, cond in conditions.items():
        tc = time.time()
        brain.restore_state(snap)
        V = np.zeros((total, len(rec)), np.float32)
        S = np.zeros((total, len(rec)), np.int16)
        G = np.zeros(len(rec), np.float64)
        ng = 0
        for t in range(total):
            drive = base
            if cond is not None and pre <= t < pre + stepl:
                drive = base.copy()
                drive[cond[0]] = cond[1]
            counts, _ = brain.step(drive, 1.0)
            V[t] = brain.v[rec]
            S[t] = counts[rec]
            if name == 'C0_sham' and t < pre:
                gg = brain.g  # CPU backend: host arrays are authoritative
                G += 1.0 + gg[:, rec].sum(0)
                ng += 1
        timing[name] = round(time.time() - tc, 1)
        if args.smoke:
            continue
        tt = np.arange(total) - pre      # ms relative to onset; V[t] is the state at the end of ms t
        win = {w: (tt >= a) & (tt < b) for w, (a, b) in WINDOWS.items()}
        mV = {w: V[m].mean(0) for w, m in win.items()}
        rate = {w: S[m].sum(0) * 1000.0 / m.sum() for w, m in win.items()}
        payload = dict(node=rec, dV_E=mV['E'] - mV['B'], dV_S=mV['S'] - mV['B'], dV_O=mV['O'] - mV['B'],
                       V_B=mV['B'], rate_B=rate['B'], rate_E=rate['E'], rate_S=rate['S'], rate_O=rate['O'],
                       group_keys=np.array(['|'.join(k) for k in keys]),
                       group_V=np.stack([V[:, key_idx[k]].mean(1) for k in keys], 1),
                       group_rate=np.stack([S[:, key_idx[k]].sum(1) * 1000.0 / max(1, len(key_idx[k]))
                                            for k in keys], 1),
                       t_ms=tt)
        if name == 'C0_sham':
            payload['gtot_B'] = G / max(1, ng)
        if name.startswith('C3'):
            seed = int(name[-1])
            cols = column_sets[seed]['columns']
            d = np.array([min(hexdist(tuple(h), c) for c in cols) if e == 'R' else -1
                          for h, e in zip(hexes.tolist(), eye)])
            payload['hexdist'] = d
            payload['stim_columns'] = np.array(cols)
        np.savez_compressed(out / f'{name}.npz', **payload)
        print(name, 'done', timing[name], 's', flush=True)
    meta = dict(arm=args.arm, contract_sha256=got, smoke=args.smoke, params_sha256=psha,
                v6_applied=brain.v6_applied,
                graph_sha256=sha256(gdir / 'graph.npz'), photoreceptor_io_sha256=io.sha256,
                graded_report=brain.graded_report, kinetics_report=brain.kinetics_report,
                backend=brain.backend, recorded_cells=int(len(rec)), driven_photoreceptors=int(len(r_all)),
                column_sets=column_sets, settle_wall_s=round(t_settle, 1), condition_wall_s=timing,
                total_spikes_at_snapshot=int(snap['total_spikes']))
    (out / ('meta_smoke.json' if args.smoke else 'meta.json')).write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ('arm', 'settle_wall_s', 'condition_wall_s', 'column_sets')}, default=str))


if __name__ == '__main__':
    main()
