#!/usr/bin/env python3
"""Path A: apply the PVLP151 counterfactual prereg's gates and decision rule.  Committed with it.

LABELLED OFFLINE COUNTERFACTUAL.  Pairs every CF row with the frozen intact A2 row
of the same base condition and seed, reads all readouts from the stored per-run
all-neuron count files of both, and never re-grades A2.
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


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rows(dirs):
    rows = {}
    for d in dirs:
        for line in (Path(d) / 'runs.jsonl').read_text().splitlines():
            r = json.loads(line)
            key = (r['condition'], r['seed'])
            if key in rows:
                raise SystemExit(f'refusing to analyse: duplicate row {key}')
            rows[key] = (r, Path(d) / 'counts' / f"{r['condition']}_s{r['seed']}.npz")
    return rows


def sparse(path):
    with np.load(path) as z:
        return z['node_index'].astype(np.int64), z['counts'].astype(np.int64)


def set_rate(sp, idx, duration_s=1.0):
    """Mean rate over ALL cells of ``idx`` (silent cells included) from a sparse count file."""
    node, cnt = sp
    return float(cnt[np.isin(node, idx)].sum() / len(idx) / duration_s)


def decide(m100, m200, lower_all_100, lower_all_200):
    """Prereg decision rule (pure function, unit tested)."""
    if None in (m100, m200):
        return 'NOT EVALUATED'
    if m100 <= 0.5 and m200 <= 0.5 and lower_all_100 and lower_all_200:
        return 'SUPPORTED'
    if m100 >= 0.8 and m200 >= 0.8:
        return 'REJECTED'
    if lower_all_100 and lower_all_200:
        return 'PARTIAL'
    return 'INCONCLUSIVE'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--cf-runs', required=True, nargs='+')
    ap.add_argument('--frozen-runs', required=True, nargs='+', help='frozen A2 run directories (runs.jsonl + counts/)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    prereg = json.loads(Path(args.prereg).read_text())
    csha, psha = sha(args.contract), sha(args.prereg)
    if prereg['frozen_contract_sha256'] != csha:
        raise SystemExit('prereg was written for a different contract')
    base = dict(contract_sha256=csha, graph_npz_sha256=contract['data']['graph_npz_sha256'],
                dynamics=contract['protocol']['dynamics'], backend=contract['protocol']['backend'])
    frozen, cf = load_rows(args.frozen_runs), load_rows(args.cf_runs)
    engine = {r['provenance'].get('engine_sha256') for r, _ in frozen.values()}
    if len(engine) != 1:
        raise SystemExit(f'frozen rows carry {len(engine)} engine identities')
    engine = engine.pop()
    fields = ('contract_sha256', 'graph_npz_sha256', 'dynamics', 'backend', 'engine_sha256', 'seed')
    for r, _ in frozen.values():
        prov.check_row(r, dict(base, engine_sha256=engine), fields=fields)
    for r, _ in cf.values():
        prov.check_row(r, dict(base, engine_sha256=engine), fields=fields)
        if r['provenance'].get('cf_prereg_sha256') != psha:
            raise SystemExit(f"refusing to analyse: row {r['condition']} not bound to prereg {psha}")
    sets = {k: np.array(v['node_index']) for k, v in dict(contract['sets'], **prereg['sets']).items()}
    dur = contract['protocol']['duration_ms'] / 1000.0

    gates = {}
    want = [(c['id'], s) for c in prereg['conditions'] if c['counterfactual'] for s in c['seeds']]
    gates['V3_complete'] = all(k in cf for k in want) and len(want) == 96
    clamp_ok, repro_ok = True, True
    rates = {}
    for c in prereg['conditions']:
        for s in c['seeds']:
            if (c['id'], s) not in cf or (c['base_condition'], s) not in frozen:
                continue
            a = sparse(cf[(c['id'], s)][1])
            b = sparse(frozen[(c['base_condition'], s)][1])
            if c['counterfactual']:
                clamp_ok &= set_rate(a, sets['PVLP151']) == 0.0
                for k in prereg['readouts']:
                    rates.setdefault(c['id'], {}).setdefault(k, {})[s] = (set_rate(b, sets[k], dur), set_rate(a, sets[k], dur))
            else:
                repro_ok &= all(np.array_equal(x, y) for x, y in zip(a, b))
    gates['V1_clamp_holds'] = bool(clamp_ok)
    gates['V2_reproduction'] = bool(repro_ok) and all((c['id'], 0) in cf for c in prereg['conditions'] if not c['counterfactual'])
    valid = all(gates.values())

    def summary(cid, k):
        v = rates.get(cid, {}).get(k, {})
        if len(v) < 8:
            return None
        i = np.array([v[s][0] for s in sorted(v)]); f = np.array([v[s][1] for s in sorted(v)])
        return dict(intact_mean=float(i.mean()), intact_sd=float(i.std(ddof=1)), cf_mean=float(f.mean()),
                    cf_sd=float(f.std(ddof=1)), ratio=(float(f.mean() / i.mean()) if i.mean() else None),
                    lower_in_all_seeds=bool((f < i).all()))

    table = {c['id']: {k: summary(c['id'], k) for k in prereg['readouts']} for c in prereg['conditions'] if c['counterfactual']}
    g = {r: table.get(f'A2_IRR_LC15_{r}_CF_silence_PVLP151', {}).get('GF') for r in (100, 200)}
    verdict = 'INVALID' if not valid else decide(*(g[r] and g[r]['ratio'] for r in (100, 200)),
                                                 *(bool(g[r] and g[r]['lower_in_all_seeds']) for r in (100, 200)))
    ctrl = {}
    for r in (100, 200):
        for k in ('GF', 'TTMn'):
            x, y = table.get(f'A2_IRR_LC15_{r}_CF_silence_PVLP151', {}).get(k), table.get(f'A2_LC4_{r}_CF_silence_PVLP151', {}).get(k)
            ctrl[f'{k}@{r}'] = (x['cf_mean'] / y['cf_mean']) if x and y and y['cf_mean'] else None
    res = dict(label=prereg['label'], prereg_sha256=psha, contract_sha256=csha, gates=gates, verdict=verdict,
               a2_unchanged='A2 FAIL unchanged; this is not a re-grade', control_ratio_under_clamp=ctrl,
               control_limit_reused=0.1, table=table)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'cf_results.json').write_text(json.dumps(res, indent=1) + '\n')
    md = [f"# PVLP151 counterfactual ({prereg['label']})", f'prereg {psha}', f'gates {gates}',
          f'**verdict: {verdict}** (A2 FAIL unchanged)', f'control ratio under clamp (limit 0.1, reported only): {ctrl}', '']
    for cid, t in table.items():
        md.append(f'- {cid}: ' + '; '.join(f"{k} {v['intact_mean']:.2f}+-{v['intact_sd']:.2f} -> {v['cf_mean']:.2f}+-{v['cf_sd']:.2f}"
                                          if v else f'{k} n/a' for k, v in t.items()))
    (out / 'cf_results.md').write_text('\n'.join(md) + '\n')
    print(verdict, gates)


if __name__ == '__main__':
    main()
