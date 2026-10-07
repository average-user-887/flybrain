"""Path B, phase 3 analysis: the per-stage transfer table and failure location.

Applies the frozen rules of ``qualification/pathB/transfer_contract.json`` to the
npz files written by ``measure_transfer.py``.  No parameter is chosen here that
the contract does not name.

    python scripts/pathB/analyse_transfer.py --phase1 DIR --phase3 DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRESENT_MV, FUNCTIONAL_MV, RATE_HZ, LAT_RUN, TIMING_MS = 0.01, 0.5, 1.0, 5, 25
STAGES = ['R1-R6', 'L1', 'L2', 'L3', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'T4', 'T5']
UPSTREAM = {'L1': ['R1-R6'], 'L2': ['R1-R6'], 'L3': ['R1-R6'], 'L5': ['L1', 'L2'],
            'Mi1': ['L1'], 'Tm3': ['L1'], 'Mi4': ['L1', 'L5'], 'Mi9': ['L3'],
            'Tm1': ['L2'], 'Tm2': ['L2'], 'Tm4': ['L2'], 'Tm9': ['L3'],
            'T4': ['Mi1', 'Tm3', 'Mi4', 'Mi9'], 'T5': ['Tm1', 'Tm2', 'Tm4', 'Tm9']}
EXPECT_ON = {'R1-R6': +1, 'L1': -1, 'L2': -1, 'L3': -1, 'Mi1': +1, 'Tm3': +1, 'Mi4': +1,
             'Mi9': -1, 'Tm1': -1, 'Tm2': -1, 'Tm4': -1, 'Tm9': -1, 'T4': +1}
# sign the transmitter chain predicts for an ON step (R histamine -, L1 glu -, L2/L3 ACh +,
# Mi4 GABA -, Mi9 glu -, others ACh +): product of synapse signs along the main path
MODEL_ON = {'R1-R6': +1, 'L1': -1, 'L2': -1, 'L3': -1, 'L5': -1, 'Mi1': +1, 'Tm3': +1,
            'Mi4': -1, 'Mi9': -1, 'Tm1': -1, 'Tm2': -1, 'Tm4': -1, 'Tm9': -1}
GROUP_OF = {f'T4{s}': 'T4' for s in 'abcd'} | {f'T5{s}': 'T5' for s in 'abcd'}


def load(p):
    return dict(np.load(p, allow_pickle=False))


def latency(stim, sham, t):
    d = np.abs(stim - sham)
    ok = d >= PRESENT_MV
    for i in np.flatnonzero((t >= 0) & (t < 200)):
        if ok[i:i + LAT_RUN].all() and i + LAT_RUN <= len(ok):
            return int(t[i])
    return None


def analyse_arm(armdir, cc, graded_types):
    sham = load(armdir / 'C0_sham.npz')
    ctype = np.array([GROUP_OF.get(t, t) for t in cc['cell_type']])
    eye, state = cc['eye'], cc['column_state']
    t = sham['t_ms']
    keys = list(sham['group_keys'])
    out = {}
    for cond in ('C1_fullfield_ON', 'C2_fullfield_OFF'):
        d = load(armdir / f'{cond}.npz')
        rows = {}
        for st in STAGES:
            for e in ('L', 'R'):
                for s in ('COMPLETE', 'PARTIAL', 'BLIND', 'ALL'):
                    m = (ctype == st) & (eye == e) & ((state == s) if s != 'ALL' else True)
                    if not m.any():
                        continue
                    aE = float((d['dV_E'][m] - sham['dV_E'][m]).mean())
                    aS = float((d['dV_S'][m] - sham['dV_S'][m]).mean())
                    aO = float((d['dV_O'][m] - sham['dV_O'][m]).mean())
                    win, A = ('E', aE) if abs(aE) >= abs(aS) else ('S', aS)
                    drE = float((d['rate_E'][m] - sham['rate_E'][m]).mean())
                    drS = float((d['rate_S'][m] - sham['rate_S'][m]).mean())
                    dr = drE if abs(drE) >= abs(drS) else drS
                    lat = None
                    if s != 'ALL':
                        k = f'{st}|{e}|{s}'
                        if k in keys:
                            j = keys.index(k)
                            lat = latency(d['group_V'][:, j], sham['group_V'][:, j], t)
                    present = abs(A) >= PRESENT_MV or abs(dr) >= RATE_HZ
                    rows[f'{st}|{e}|{s}'] = dict(
                        n=int(m.sum()), A_mV=round(A, 5), window=win, A_offset_mV=round(aO, 5),
                        V_base_mV=round(float(sham['V_B'][m].mean()), 3),
                        rate_base_Hz=round(float(sham['rate_B'][m].mean()), 3),
                        d_rate_Hz=round(dr, 3), latency_ms=lat, present=present,
                        functional=abs(A) >= FUNCTIONAL_MV,
                        # PREREGISTERED: signed by the net membrane response A.
                        sign=int(np.sign(A)) if present else 0,
                        # POST-HOC (addendum 1, written after the v3 numbers were read): a
                        # spiking cell whose rate changes is signed by its rate instead.
                        sign_posthoc=(int(np.sign(dr)) if (st not in graded_types and abs(dr) >= RATE_HZ)
                                      else int(np.sign(A)) if present else 0),
                        graded=st in graded_types,
                        gtot_base=round(float(sham['gtot_B'][m].mean()), 3))
        out[cond] = rows
    # spatial spread, C3
    spread = {}
    for seed in (0, 1, 2):
        p = armdir / f'C3_column_ON_seed{seed}.npz'
        if not p.is_file():
            continue
        d = load(p)
        hd = d['hexdist']
        for st in STAGES:
            for dist in range(5):
                m = (ctype == st) & (eye == 'R') & (hd == dist)
                if m.any():
                    spread.setdefault(st, {}).setdefault(dist, []).append(
                        float((d['dV_S'][m] - sham['dV_S'][m]).mean()))
    out['C3_spread_mV_by_hexdist'] = {st: {dist: round(float(np.mean(v)), 5) for dist, v in dd.items()}
                                     for st, dd in spread.items()}
    return out


def transfer_and_verdict(res, comp, cond='C1_fullfield_ON', eye='R', sign_key='sign'):
    rows = res[cond]
    table, causes = [], []
    first_absent = None
    for st in STAGES:
        r = rows.get(f'{st}|{eye}|COMPLETE')
        if r is None:
            continue
        ups = UPSTREAM.get(st, [])
        upA = [abs(rows[f'{u}|{eye}|COMPLETE']['A_mV']) for u in ups if f'{u}|{eye}|COMPLETE' in rows]
        up_max = max(upA) if upA else None
        ratio = abs(r['A_mV']) / up_max if up_max else None
        blind = rows.get(f'{st}|{eye}|BLIND')
        rec = dict(stage=st, A_complete_mV=r['A_mV'], A_partial_mV=rows.get(f'{st}|{eye}|PARTIAL', {}).get('A_mV'),
                   A_blind_mV=blind['A_mV'] if blind else None, window=r['window'],
                   d_rate_Hz=r['d_rate_Hz'], rate_base_Hz=r['rate_base_Hz'], V_base_mV=r['V_base_mV'],
                   latency_ms=r['latency_ms'], present=r['present'], functional=r['functional'],
                   sign=r[sign_key], sign_basis=sign_key, expected_sign=EXPECT_ON.get(st) if cond.endswith('ON') else
                   (-EXPECT_ON[st] if st in EXPECT_ON else None),
                   transfer_ratio_vs_upstream=round(ratio, 4) if ratio is not None else None,
                   graded=r['graded'], gtot_base=r['gtot_base'])
        cls = []
        # missing anatomy
        if blind and r['present'] and not blind['present']:
            cls.append('missing_anatomy(BLIND columns silent)')
        # sign
        if r['present'] and rec['expected_sign'] and r[sign_key] != rec['expected_sign']:
            model = MODEL_ON.get(st)
            if cond.endswith('OFF') and model:
                model = -model
            cls.append('sign:' + ('transmitter-chain predicts this sign' if model == r[sign_key]
                                  else 'dynamics, not the transmitter chain'))
        # thresholding: upstream present, this stage present in V, spiking, silent, downstream absent
        if (not r['graded']) and r['present'] and abs(r['d_rate_Hz']) < RATE_HZ and r['rate_base_Hz'] < RATE_HZ:
            downs = [k for k, v in UPSTREAM.items() if st in v]
            if downs and all(not rows.get(f'{dn}|{eye}|COMPLETE', {}).get('present', False) for dn in downs):
                cls.append('thresholding(spiking LIF, subthreshold, silent)')
        # dilution
        if ratio is not None and r['present'] and up_max and up_max >= PRESENT_MV and ratio < 0.1 and r['graded']:
            share = comp.get(st)
            cls.append(f'dilution/fan-in(ratio {ratio:.3f}; upstream share of input synapses {share}; '
                       f'baseline g_tot {r["gtot_base"]} leak units)')
        # timing
        if r['latency_ms'] is not None and ups:
            ul = [rows[f'{u}|{eye}|COMPLETE']['latency_ms'] for u in ups if f'{u}|{eye}|COMPLETE' in rows]
            ul = [x for x in ul if x is not None]
            if ul and r['latency_ms'] - min(ul) > TIMING_MS:
                cls.append(f'timing(+{r["latency_ms"] - min(ul)} ms over upstream)')
        if r['present'] and abs(r['A_offset_mV']) >= PRESENT_MV and abs(r['A_mV']) < PRESENT_MV:
            cls.append('timing(offset window only)')
        if not r['present'] and up_max is not None and up_max >= PRESENT_MV and not cls:
            ups_spiking = [u for u in ups if not rows[f'{u}|{eye}|COMPLETE']['graded']]
            if ups_spiking and all(abs(rows[f'{u}|{eye}|COMPLETE']['d_rate_Hz']) < RATE_HZ for u in ups_spiking):
                cls.append('thresholding(upstream spiking cells changed V but not rate)')
            else:
                cls.append('unclassified loss')
        if not r['present'] and first_absent is None and st != 'R1-R6':
            first_absent = st
        rec['cause'] = cls
        table.append(rec)
        if cls:
            causes.append((st, cls))
    return dict(table=table, first_absent_stage=first_absent, causes=causes)


def reset_evidence(armdir, cc):
    """From recorded data only: R1-R6 mean V, rate and the per-cell relation between
    rate change and V change, sham vs ON vs OFF (window E and S), right eye."""
    sham = load(armdir / 'C0_sham.npz')
    m = (cc['cell_type'] == 'R1-R6') & (cc['eye'] == 'R')
    out = {}
    for cond in ('C1_fullfield_ON', 'C2_fullfield_OFF'):
        d = load(armdir / f'{cond}.npz')
        for w in ('E', 'S'):
            dv = d[f'dV_{w}'][m] - sham[f'dV_{w}'][m]
            dr = d[f'rate_{w}'][m] - sham[f'rate_{w}'][m]
            vB = d['V_B'][m]
            out[f'{cond}|{w}'] = dict(
                n=int(m.sum()), V_base_mV=round(float(vB.mean()), 3),
                V_window_mV=round(float((vB + d[f'dV_{w}'][m]).mean()), 3),
                rate_base_Hz=round(float(d['rate_B'][m].mean()), 2),
                rate_window_Hz=round(float(d[f'rate_{w}'][m].mean()), 2),
                mean_dV_net_mV=round(float(dv.mean()), 4), mean_drate_net_Hz=round(float(dr.mean()), 2),
                corr_dV_drate_across_cells=(round(float(np.corrcoef(dv, dr)[0, 1]), 3)
                                            if dv.std() > 0 and dr.std() > 0 else None),
                frac_cells_rate_up_and_V_down=round(float(((dr > 0) & (dv < 0)).mean()), 3))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase1', type=Path, required=True)
    ap.add_argument('--phase3', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    cc = dict(np.load(args.phase1 / 'cell_columns.npz'))
    cov = json.loads((args.phase1 / 'column_coverage.json').read_text())
    comp = {}
    for x, lst in cov['medulla_input_composition_within_study_types'].items():
        comp[x] = {t: f for t, _, f in lst if t in UPSTREAM.get(x, [])}
    result = {}
    for arm in ('v3', 'v4', 'v5'):
        ad = args.phase3 / arm
        if not (ad / 'meta.json').is_file():
            continue
        meta = json.loads((ad / 'meta.json').read_text())
        graded_types = set()
        if meta.get('graded_report'):
            graded_types = set(STAGES)  # every studied type is tier-1 graded in the declared policy
        res = analyse_arm(ad, cc, graded_types)
        # PREREGISTERED verdicts (primary) and POST-HOC verdicts (addendum 1) side by side.
        res['verdict_R_ON'] = transfer_and_verdict(res, comp, 'C1_fullfield_ON', 'R')
        res['verdict_R_OFF'] = transfer_and_verdict(res, comp, 'C2_fullfield_OFF', 'R')
        res['verdict_L_ON'] = transfer_and_verdict(res, comp, 'C1_fullfield_ON', 'L')
        res['POSTHOC_verdict_R_ON'] = transfer_and_verdict(res, comp, 'C1_fullfield_ON', 'R', 'sign_posthoc')
        res['POSTHOC_verdict_R_OFF'] = transfer_and_verdict(res, comp, 'C2_fullfield_OFF', 'R', 'sign_posthoc')
        res['reset_evidence_R1R6'] = reset_evidence(ad, cc)
        res['meta'] = {k: meta[k] for k in ('arm', 'contract_sha256', 'graph_sha256', 'condition_wall_s', 'settle_wall_s')}
        result[arm] = res
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'transfer_analysis.json').write_text(json.dumps(result, indent=1, default=str))
    lines = []
    for arm, res in result.items():
        for vk in ('verdict_R_ON', 'verdict_R_OFF', 'POSTHOC_verdict_R_ON', 'POSTHOC_verdict_R_OFF'):
            v = res[vk]
            lab = 'POST-HOC (addendum 1, rate-signed spiking cells)' if vk.startswith('POSTHOC') else 'PREREGISTERED'
            lines.append(f'\n## {arm} {vk} [{lab}] (COMPLETE columns; partial/blind alongside)\n')
            lines.append('| stage | A complete (mV) | A partial | A blind | win | d rate (Hz) | base rate | V base | latency (ms) | ratio vs upstream | sign / expected | graded | cause |')
            lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            for r in v['table']:
                lines.append(f"| {r['stage']} | {r['A_complete_mV']} | {r['A_partial_mV']} | {r['A_blind_mV']} | {r['window']} | "
                             f"{r['d_rate_Hz']} | {r['rate_base_Hz']} | {r['V_base_mV']} | {r['latency_ms']} | "
                             f"{r['transfer_ratio_vs_upstream']} | {r['sign']} / {r['expected_sign']} | {r['graded']} | {'; '.join(r['cause'])} |")
            lines.append(f"\nfirst absent stage: {v['first_absent_stage']}")
        lines.append(f"\nR1-R6 reset evidence (recorded): {json.dumps(res['reset_evidence_R1R6'])}")
        lines.append(f"\nC3 spread (mV by hex distance 0..4, right eye): {json.dumps(res['C3_spread_mV_by_hexdist'])}")
    (args.out / 'transfer_tables.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
