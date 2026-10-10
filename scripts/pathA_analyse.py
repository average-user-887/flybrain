#!/usr/bin/env python3
"""Path A: apply the frozen contract's criteria to runs.jsonl.  Committed with the contract.

Writes ``results.json`` and ``results.md`` (raw dose-response tables, controls,
per-criterion outcome, verdict per test).  Missing conditions make the
dependent criterion NOT EVALUATED, never a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pathA_provenance as prov  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--runs', required=True, help='directory with runs.jsonl (one or more, merged)', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    csha = hashlib.sha256(Path(args.contract).read_bytes()).hexdigest()
    seeds = contract['protocol']['seeds']
    # Provenance gate: every row must be bound to THIS contract, its graph and
    # dynamics, and all rows to one engine/source identity; otherwise refuse.
    base = dict(contract_sha256=csha, graph_npz_sha256=contract['data']['graph_npz_sha256'],
                dynamics=contract['protocol']['dynamics'], backend=contract['protocol']['backend'])
    known = {c['id'] for c in contract['conditions']}
    ident = None
    data = {}
    for d in args.runs:
        for line in (Path(d) / 'runs.jsonl').read_text().splitlines():
            r = json.loads(line)
            p = r.get('provenance') or {}
            if ident is None:
                try:  # the identity is taken only from a complete, well-formed record
                    prov.check_complete(r.get('provenance'), f"row {r.get('condition')!r} seed {r.get('seed')!r}")
                except prov.ProvenanceError as exc:
                    raise SystemExit(f'refusing to analyse: {exc}')
                ident = dict(engine_sha256=p.get('engine_sha256'), code_sha=p.get('code_sha'))
            try:
                prov.check_row(r, dict(base, **ident))
            except prov.ProvenanceError as exc:
                raise SystemExit(f'refusing to analyse: {exc}')
            if r['condition'] not in known or r['seed'] not in seeds:
                raise SystemExit(f"refusing to analyse: row {r['condition']} seed {r['seed']} is not in the contract")
            if r['seed'] in data.get(r['condition'], {}):
                raise SystemExit(f"refusing to analyse: duplicate row {r['condition']} seed {r['seed']}")
            data.setdefault(r['condition'], {})[r['seed']] = r

    # Rate 0 means "no input"; every series at level 0 shares its test's single no-input condition.
    baseline = {'A1': 'A1_JOCE_0', 'A2': 'A2_LC4_0', 'CAL': 'CAL_SUGAR_0'}

    def series(cid, readout):
        if cid.endswith('_0') and cid not in data:
            cid = baseline[cid.split('_')[0]]
        runs = data.get(cid, {})
        if any(s not in runs for s in seeds):
            return None
        return np.array([runs[s]['readouts'][readout]['mean_rate_hz'] for s in seeds])

    def m(cid, readout):
        v = series(cid, readout)
        return None if v is None else float(v.mean())

    def ratio(a, b):
        if a is None or b is None:
            return None
        if b == 0:
            return 0.0 if a == 0 else float('inf')
        return a / b

    def monotone(vals):
        if any(v is None for v in vals):
            return None
        return all(vals[i] >= vals[i - 1] - max(0.5, 0.10 * vals[i - 1]) for i in range(1, len(vals)))

    def all_true(xs):
        if any(x is None for x in xs):
            return None
        return all(xs)

    res = dict(contract_sha256=csha, run_identity=ident, tests={})
    tables = {}
    for tname, test in contract['tests'].items():
        crit = test['criteria']
        out = {}
        if tname == 'A1':
            c = crit['P1']; L = c['levels']
            vals = [m(c['series'].format(r=r), 'aBN1') for r in L]
            tables['A1 JO-CE -> aBN1'] = vals
            out['P1'] = dict(values=vals, monotone=monotone(vals),
                             robust={r: m(f'A1_JOCE_{r}', 'aBN1') for r in c['at']},
                             holds=all_true([monotone(vals)] + [None if m(f'A1_JOCE_{r}', 'aBN1') is None else m(f'A1_JOCE_{r}', 'aBN1') >= c['min_rate_hz'] for r in c['at']]))
            c = crit['P2']
            rr = {r: ratio(m(c['numerator'].format(r=r), 'aBN1'), m(c['denominator'].format(r=r), 'aBN1')) for r in c['at']}
            out['P2'] = dict(ratios=rr, holds=all_true([None if v is None else v <= c['max_ratio'] for v in rr.values()]))
            for ro in ['aBN1', 'aDN1', 'aDN2', 'ProLN-MN']:
                tables[f'A1 JO-CE -> {ro}'] = [m(f'A1_JOCE_{r}', ro) for r in L]
                tables[f'A1 JO-F -> {ro}'] = [m(f'A1_JOF_{r}', ro) for r in L]
            c = crit['S1']
            mo = {ro: monotone([m(f'A1_JOCE_{r}', ro) for r in L]) for ro in c['readouts']}
            out['S1'] = dict(monotone=mo, holds=all_true(list(mo.values())))
            c = crit['S2']
            rr = {f'{ro}@{r}': ratio(m(c['silenced'].format(r=r), ro), m(c['intact'].format(r=r), ro)) for ro in c['readouts'] for r in c['at']}
            out['S2'] = dict(ratios=rr, holds=all_true([None if v is None else v <= c['max_ratio'] for v in rr.values()]))
            c = crit['S3']
            mono_mn = monotone([m(f'A1_JOCE_{r}', 'ProLN-MN') for r in L])
            pos = {cid: m(cid, 'ProLN-MN') for cid in c['positive_at']}
            out['S3'] = dict(monotone=mono_mn, positive=pos,
                             holds=all_true([mono_mn] + [None if v is None else v > 0 for v in pos.values()]))
            ctrl = crit['control_irrelevant']
        elif tname == 'A2':
            c = crit['P1']; L = c['levels']
            sub = {}
            for s in c['series']:
                vals = [m(s.format(r=r), 'GF') for r in L]
                tables[f"A2 {s.split('_')[1]} -> GF"] = vals
                tables[f"A2 {s.split('_')[1]} -> TTMn"] = [m(s.format(r=r), 'TTMn') for r in L]
                sub[s] = dict(values=vals, monotone=monotone(vals), at200=vals[-1],
                              holds=all_true([monotone(vals), None if vals[-1] is None else vals[-1] >= c['min_rate_hz']]))
            out['P1'] = dict(series=sub, holds=all_true([v['holds'] for v in sub.values()]))
            c = crit['P2']
            rr = {cid: ratio(m(cid, c['numerator_readout']), m(cid, c['denominator_readout'])) for cid in c['conditions']}
            out['P2'] = dict(ttmn_over_gf=rr, holds=all_true([None if v is None else v >= c['min_ratio'] for v in rr.values()]))
            c = crit['S1']
            rr = {f'{a.format(r=r)}': ratio(m(a.format(r=r), 'TTMn'), m(b.format(r=r), 'TTMn')) for a, b in c['pairs'] for r in c['at']}
            out['S1'] = dict(ratios=rr, holds=all_true([None if v is None else v <= c['max_ratio'] for v in rr.values()]))
            ctrl = crit['control_irrelevant']
        else:
            c = crit['C1']; L = c['levels']
            vals = [m(c['series'].format(r=r), 'MN9') for r in L]
            tables['CAL sugar -> MN9'] = vals
            frac = None
            if all(v is not None for v in vals) and max(vals) > 0:
                frac = vals[L.index(c['at'])] / max(vals)
            out['C1'] = dict(values=vals, monotone=monotone(vals), fraction_of_max_at_100=frac,
                             holds=all_true([monotone(vals), None if frac is None else c['fraction_of_max'][0] <= frac <= c['fraction_of_max'][1]]))
            out['verdict'] = {True: 'CONSISTENT', False: 'INCONSISTENT', None: 'NOT EVALUATED'}[out['C1']['holds']]
            res['tests'][tname] = out
            continue
        rr = {f'{ro}@{r}': ratio(m(ctrl['control'].format(r=r), ro), m(ctrl['reference'].format(r=r), ro))
              for ro in ctrl['readouts'] for r in ctrl['at']}
        out['control_irrelevant'] = dict(ratios=rr, valid=all_true([None if v is None else v <= ctrl['max_ratio'] for v in rr.values()]))
        cs = crit['counterfactual_shuffled']
        pairs = cs.get('pairs') or [[cs['control'], cs['reference']]]
        out['counterfactual_shuffled'] = dict(label='LABELLED OFFLINE COUNTERFACTUAL', ratios={a: ratio(m(a, cs['readout']), m(b, cs['readout'])) for a, b in pairs})
        v = all_true([out['P1']['holds'], out['P2']['holds'], out['control_irrelevant']['valid']])
        out['verdict'] = {True: 'PASS', False: 'FAIL', None: 'NOT EVALUATED'}[v]
        res['tests'][tname] = out

    # raw per-seed table of every condition and readout
    raw = {}
    for cid, runs in sorted(data.items()):
        raw[cid] = {ro: dict(mean=float(np.mean([runs[s]['readouts'][ro]['mean_rate_hz'] for s in runs])),
                             sd=float(np.std([runs[s]['readouts'][ro]['mean_rate_hz'] for s in runs], ddof=1)) if len(runs) > 1 else 0.0,
                             n_seeds=len(runs))
                    for ro in next(iter(runs.values()))['readouts']}
        raw[cid]['_activated_mean_rate_hz'] = float(np.mean([r['activated_mean_rate_hz'] for r in runs.values()]))
        raw[cid]['_firing_neurons_mean'] = float(np.mean([r['firing_neurons'] for r in runs.values()]))
    res['dose_response_tables'] = tables
    res['raw'] = raw
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    lines = [f'# Path A results (contract sha256 {csha})', '']
    for t, o in res['tests'].items():
        lines.append(f"## {t}: {o['verdict']}")
        for k, v in o.items():
            if k != 'verdict':
                lines.append(f'- {k}: `{json.dumps(v, default=str)}`')
        lines.append('')
    lines.append('## Dose-response (seed-mean Hz; levels per contract)')
    for k, v in tables.items():
        lines.append(f"- {k}: {[None if x is None else round(x, 2) for x in v]}")
    lines.append('')
    lines.append('## All conditions (mean +- SD over seeds, Hz)')
    for cid, ro in raw.items():
        parts = [f"{k} {v['mean']:.2f}+-{v['sd']:.2f}" for k, v in ro.items() if not k.startswith('_')]
        lines.append(f"- {cid} (n={next(v for k, v in ro.items() if not k.startswith('_'))['n_seeds']}, "
                     f"activated {ro['_activated_mean_rate_hz']:.1f} Hz, firing neurons {ro['_firing_neurons_mean']:.0f}): " + '; '.join(parts))
    (out / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
