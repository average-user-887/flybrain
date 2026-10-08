#!/usr/bin/env python3
"""Path A: gates and DESCRIPTIVE paired report for the PVLP151 counterfactual prereg.

LABELLED OFFLINE COUNTERFACTUAL.  No pass threshold, no verdict on biology, no
re-grade of A2.  Pairs each clamp row with the intact replay row of the same base
condition and seed; reads every readout from the stored per-run all-neuron count
files.  Differences are total network consequences, not pathway fractions.
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

FIELDS = ('contract_sha256', 'graph_npz_sha256', 'dynamics', 'backend', 'engine_sha256', 'seed')


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


def without(sp, idx):
    node, cnt = sp
    keep = ~np.isin(node, idx)
    return node[keep], cnt[keep]


def same_input_hashes(hashes):
    """G3: True only if every hash is a 64-char lowercase hex string and all are equal.
    A missing, null or malformed hash rejects (a set of identical Nones must not pass)."""
    hashes = list(hashes)
    ok = len(hashes) > 0 and all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                                 for h in hashes)
    return ok and len(set(hashes)) == 1


def paired(intact, cf):
    """Descriptive paired summary (unit tested).  Percent only with a nonzero intact mean."""
    i, c = np.asarray(intact, float), np.asarray(cf, float)
    d = c - i
    im = float(i.mean())
    sd = lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0  # noqa: E731
    return dict(intact_hz=i.tolist(), cf_hz=c.tolist(), diff_hz=d.tolist(), intact_mean=im, intact_sd=sd(i),
                cf_mean=float(c.mean()), cf_sd=sd(c), diff_mean=float(d.mean()), diff_sd=sd(d),
                seeds_lower=int((d < 0).sum()), seeds_higher=int((d > 0).sum()), seeds_equal=int((d == 0).sum()),
                percent_change=(100.0 * float(d.mean()) / im) if im != 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--cf-runs', required=True, nargs='+')
    ap.add_argument('--frozen-runs', required=True, nargs='+', help='frozen A2 run directories (runs.jsonl + counts/)')
    ap.add_argument('--graph', required=True, help='graph.npz, re-hashed against the contract')
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
    for r, _ in list(frozen.values()) + list(cf.values()):
        prov.check_row(r, dict(base, engine_sha256=engine), fields=FIELDS)
    code = {r['provenance']['code_sha'] for r, _ in cf.values()}
    for r, _ in cf.values():
        if r['provenance'].get('cf_prereg_sha256') != psha:
            raise SystemExit(f"refusing to analyse: row {r['condition']} not bound to prereg {psha}")
    sets = {k: np.array(v['node_index']) for k, v in dict(contract['sets'], **prereg['sets']).items()}
    pv = sets['PVLP151']
    dur = contract['protocol']['duration_ms'] / 1000.0
    want = [(c['id'], s) for c in prereg['conditions'] for s in c['seeds']]
    by = {}
    for c in prereg['conditions']:
        by.setdefault(c['base_condition'], {})[c['counterfactual'] or 'replay'] = c

    gates = dict(G1_clamp_zero=True, G2_clean_initial_state=True, G3_identical_inputs=True,
                 G4_zero_outgoing_semantics=True, G5_replay_reproduces_frozen=True,
                 G6_complete_and_bound=all(k in cf for k in want) and len(code) == 1
                 and sha(args.graph) == contract['data']['graph_npz_sha256'])
    for r, _ in cf.values():
        gates['G2_clean_initial_state'] &= (r.get('audit') or {}).get('initial_state_clean') is True
    report = {}
    for b, roles in by.items():
        rep, cl, zo = roles.get('replay'), roles.get('silence_PVLP151'), roles.get('zero_outgoing_PVLP151')
        series = {k: ([], []) for k in prereg['readouts']}
        for s in cl['seeds']:
            if (rep['id'], s) not in cf or (cl['id'], s) not in cf:
                continue
            (rr, rp), (cr, cp) = cf[(rep['id'], s)], cf[(cl['id'], s)]
            a, c = sparse(rp), sparse(cp)
            gates['G1_clamp_zero'] &= set_rate(c, pv) == 0.0
            ins = [(rr.get('audit') or {}).get('input_sha256'), (cr.get('audit') or {}).get('input_sha256')]
            if (b, s) in frozen:
                f = sparse(frozen[(b, s)][1])
                gates['G5_replay_reproduces_frozen'] &= all(np.array_equal(x, y) for x, y in zip(a, f))
            else:
                gates['G5_replay_reproduces_frozen'] = False
            if zo and (zo['id'], s) in cf:
                zr, zp = cf[(zo['id'], s)]
                ins.append((zr.get('audit') or {}).get('input_sha256'))
                gates['G4_zero_outgoing_semantics'] &= all(
                    np.array_equal(x, y) for x, y in zip(without(sparse(zp), pv), without(c, pv)))
            gates['G3_identical_inputs'] &= same_input_hashes(ins)
            for k in prereg['readouts']:
                series[k][0].append(set_rate(a, sets[k], dur))
                series[k][1].append(set_rate(c, sets[k], dur))
        if zo and (zo['id'], 0) not in cf:
            gates['G4_zero_outgoing_semantics'] = False
        report[b] = {k: paired(*v) if v[0] else None for k, v in series.items()}
    hard = all(gates[g] for g in gates if g != 'G5_replay_reproduces_frozen')
    status = 'INVALID' if not hard else ('VALID' if gates['G5_replay_reproduces_frozen']
                                         else 'VALID vs same-code replay only; frozen rows NOT reproduced')
    res = dict(label=prereg['label'], prereg_sha256=psha, contract_sha256=csha, code_sha=sorted(code),
               engine_sha256=engine, gates=gates, status=status, kind='DESCRIPTIVE; no threshold, no verdict',
               interpretation_limit=prereg['report']['interpretation_limit'],
               a2_unchanged='A2 FAIL unchanged; not a re-grade', paired=report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cf_results.json').write_text(json.dumps(res, indent=1) + '\n')
    md = [f"# PVLP151 counterfactual ({prereg['label']})", f'prereg {psha}; code {sorted(code)}',
          f'gates {gates}', f'**status: {status}** (descriptive only; A2 FAIL unchanged)',
          prereg['report']['interpretation_limit'], '']
    for b, t in report.items():
        md.append(f'## {b}: intact replay -> PVLP151 clamped (Hz, mean+-SD over seeds; paired diff; seeds lower/higher/equal)')
        for k, v in t.items():
            if v:
                pc = f"{v['percent_change']:+.1f} %" if v['percent_change'] is not None else 'n/a (intact 0)'
                md.append(f"- {k}: {v['intact_mean']:.2f}+-{v['intact_sd']:.2f} -> {v['cf_mean']:.2f}+-{v['cf_sd']:.2f}; "
                          f"diff {v['diff_mean']:+.2f}+-{v['diff_sd']:.2f} ({pc}); "
                          f"{v['seeds_lower']}/{v['seeds_higher']}/{v['seeds_equal']}; per seed {v['diff_hz']}")
    (out / 'cf_results.md').write_text('\n'.join(md) + '\n')
    print(status, gates)
    if status == 'INVALID':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
