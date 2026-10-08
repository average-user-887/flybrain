#!/usr/bin/env python3
"""Path A: gates and DESCRIPTIVE report for the PVLP151->GF direct-edge-deletion prereg.

LABELLED OFFLINE COUNTERFACTUAL.  Every identity is checked against EXTERNAL pins in
the prereg (code sha given on the command line), never learned from the rows.  The
reset fixture is checked independently and its canonical digest recomputed here.
No threshold, no verdict, no pathway fraction.  Exit 2 if INVALID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathA_cf_analyse import paired, same_input_hashes, set_rate, sparse  # noqa: E402

HEX64 = set('0123456789abcdef')


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture_digest(path, order, dynamics='v3', n=None, v_rest=-52.0):
    """Independent check of the reset fixture: v at rest, every other array and every scalar zero,
    trailing dimension n; then the canonical digest (same spec as the runner, own code)."""
    problems = []
    with np.load(path, allow_pickle=False) as z:
        names = [k[len('state__'):] for k in z.files if k.startswith('state__')]
        if names != list(order):
            problems.append(f'state arrays {names} != pinned order {list(order)}')
        if str(z['dynamics']) != dynamics:
            problems.append('fixture dynamics differs')
        h = hashlib.sha256(b'neurofly.state/1\0' + dynamics.encode() + b'\0')
        for name in order:
            a = np.ascontiguousarray(z[f'state__{name}'])
            if n is not None and (a.ndim == 0 or (a.shape[-1] != n and name not in ('queue_count', 'nactive'))):
                problems.append(f'{name}: shape {a.shape} does not end in n={n}')
            if name == 'v':
                if a.dtype != np.float32 or not np.all(a == v_rest):
                    problems.append('v is not float32 at rest everywhere')
            elif np.any(a):
                problems.append(f'{name} is not all zero')
            h.update(f'{name}|{a.dtype.str}|{a.shape}|'.encode() + a.tobytes() + b'\0')
        for name in ('cursor', 'total_spikes', 'sim_ms'):
            v = z[f'scalar__{name}'].item()
            if v != 0:
                problems.append(f'scalar {name} = {v!r} is not 0')
            h.update(f'{name}={v!r}\0'.encode())
    return h.hexdigest(), problems


def rng_digest(seed_seq) -> str:
    st = np.random.default_rng(seed_seq).bit_generator.state
    return hashlib.sha256(json.dumps(st, sort_keys=True, default=int).encode()).hexdigest()


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


def manifest_files(manifest_path, root, pinned_sha, problems, label):
    """{(condition, seed): path} from a count manifest, after verifying the manifest pin and every file hash."""
    if sha(manifest_path) != pinned_sha:
        problems.append(f'{label} manifest sha256 differs from the prereg pin')
        return {}
    out = {}
    for f in json.loads(Path(manifest_path).read_text())['files']:
        p = Path(root) / f['original_path']
        if not p.exists() or sha(p) != f['sha256']:
            problems.append(f'{label} file {f["original_path"]} missing or hash differs')
        out[(f['condition'], f['seed'])] = p
    return out


def evaluate(contract, prereg, psha, rows, exits, frozen, clamp, fixture, code_sha, graph_sha_now, n):
    """All gates; returns (gates, problems, report).  Pure given loaded inputs (unit tested)."""
    pins, problems = prereg['pins'], []
    g = {k: True for k in ('G1_clamp_zero', 'G2_clean_state', 'G3_identical_inputs', 'G4_deletion_exact',
                           'G5a_intact_reproduces_frozen', 'G5b_clamp_reproduces_accepted',
                           'G5c_sham_equals_intact', 'G6_complete_and_bound')}

    def fail(gate, msg):
        g[gate] = False
        problems.append(f'{gate}: {msg}')

    fdig, fprob = fixture_digest(fixture, pins['state_arrays_order'], contract['protocol']['dynamics'], n)
    if sha(fixture) != pins['reset_fixture_sha256']:
        fail('G2_clean_state', 'reset fixture sha256 differs from the pin')
    for p in fprob:
        fail('G2_clean_state', f'fixture: {p}')
    if fdig != pins['reset_state_sha256']:
        fail('G2_clean_state', 'fixture digest differs from the pinned reset_state_sha256')
    want_prov = dict(contract_sha256=prereg['frozen_contract_sha256'], graph_npz_sha256=pins['graph_npz_sha256'],
                     dynamics=contract['protocol']['dynamics'], backend=contract['protocol']['backend'],
                     engine_sha256=pins['engine_sha256'], code_sha=code_sha, cf_prereg_sha256=psha)
    expected = {(c['id'], s): c for c in prereg['conditions'] for s in c['seeds']}
    if set(rows) != set(expected):
        fail('G6_complete_and_bound', f'rows missing {sorted(set(expected) - set(rows))[:5]} or extra '
                                      f'{sorted(set(rows) - set(expected))[:5]}')
    if not exits or any(e.get('exit_code') != 0 for e in exits):
        fail('G6_complete_and_bound', f'shard exit codes {[e.get("exit_code") for e in exits]} (need all 0, >=1 shard)')
    if graph_sha_now != pins['graph_npz_sha256']:
        fail('G6_complete_and_bound', 'graph.npz sha256 changed')
    by_id = {c['id']: c for c in contract['conditions']}
    sets = {k: np.array(v['node_index']) for k, v in dict(contract['sets'], **prereg['sets']).items()}
    pv = sets['PVLP151']
    listed = sorted(int(e['edge_index']) for e in prereg['direct_edges'])
    sp = {}
    for key, (r, path) in rows.items():
        c = expected.get(key)
        if c is None:
            continue
        p = r.get('provenance') or {}
        for k, v in want_prov.items():
            if p.get(k) != v:
                fail('G6_complete_and_bound', f'{key} provenance {k}={p.get(k)!r}')
        a = r.get('audit') or {}
        base = by_id[c['base_condition']]
        n_act = sum(len(contract['sets'][s]['node_index']) for s in base['activate'])
        seq = [key[1], int(round(base['rate_hz'] * 10)), n_act]
        if a.get('initial_state_clean') is not True or a.get('state_sha256') != pins['reset_state_sha256']:
            fail('G2_clean_state', f'{key} reset state not the pinned fixture state')
        if a.get('rng_seed_seq') != seq or a.get('rng_state_sha256') != rng_digest(seq):
            fail('G2_clean_state', f'{key} RNG seed/state differs from the frozen input definition')
        want_w = pins['deleted_weight_sha256'] if c['arm'] == 'direct_edge_deletion' else pins['base_weight_sha256']
        if a.get('weight_sha256') != want_w or a.get('parent_weight_unchanged') is not True:
            fail('G4_deletion_exact', f'{key} weight array {a.get("weight_sha256")!r} != pinned for arm {c["arm"]}')
        if sorted(r.get('zero_edges', [])) != (listed if c['arm'] == 'direct_edge_deletion' else []):
            fail('G4_deletion_exact', f'{key} recorded zero_edges differ from the prereg')
        sp[key] = sparse(path, n)
        if c['arm'] == 'all_output_clamp' and set_rate(sp[key], pv) != 0.0:
            fail('G1_clamp_zero', f'{key} PVLP151 fired')
    arm_of = {(c['base_condition'], c['arm']): c['id'] for c in prereg['conditions']}
    report = {}
    eq = lambda x, y: all(np.array_equal(u, v) for u, v in zip(x, y))  # noqa: E731
    for b in dict.fromkeys(c['base_condition'] for c in prereg['conditions']):
        ids = {arm: arm_of.get((b, arm)) for arm in ('intact', 'direct_edge_deletion', 'all_output_clamp', 'sham_empty_deletion')}
        seeds = prereg['conditions'][0]['seeds']
        for s in seeds:
            ks = [(ids[arm], s) for arm in ('intact', 'direct_edge_deletion', 'all_output_clamp')]
            if ids['sham_empty_deletion'] and s == 0:
                ks.append((ids['sham_empty_deletion'], 0))
            if not all(k in rows for k in ks):
                continue
            if not same_input_hashes([(rows[k][0].get('audit') or {}).get('input_sha256') for k in ks]):
                fail('G3_identical_inputs', f'{b} seed {s}')
            fz = frozen.get((b, s))
            if fz is None or not eq(sp[ks[0]], sparse(fz, n)):
                fail('G5a_intact_reproduces_frozen', f'{b} seed {s}')
            cz = clamp.get((b, s))
            if cz is None or not eq(sp[ks[2]], sparse(cz, n)):
                fail('G5b_clamp_reproduces_accepted', f'{b} seed {s}')
            if len(ks) == 4 and not eq(sp[ks[3]], sp[ks[0]]):
                fail('G5c_sham_equals_intact', f'{b} seed 0')
        if not all((ids[a], s) in sp for a in ('intact', 'direct_edge_deletion', 'all_output_clamp') for s in seeds):
            continue
        dur = contract['protocol']['duration_ms'] / 1000.0
        excl = np.concatenate([pv, sets['GF']] + [sets[x] for x in by_id[b]['activate']])
        rate = {a: {k: [set_rate(sp[(ids[a], s)], sets[k], dur) for s in seeds] for k in prereg['readouts']}
                for a in ('intact', 'direct_edge_deletion', 'all_output_clamp')}

        def all_neuron(x, y):
            out = []
            for s in seeds:
                (na, ca), (nb, cb) = sp[(ids[x], s)], sp[(ids[y], s)]
                u = np.union1d(na, nb)
                da = np.zeros(len(u), np.int64); da[np.searchsorted(u, na)] = ca
                db = np.zeros(len(u), np.int64); db[np.searchsorted(u, nb)] = cb
                d = da - db
                keep = ~np.isin(u, excl)
                out.append(dict(seed=s, neurons_changed=int((d != 0).sum()), sum_abs_diff=int(np.abs(d).sum()),
                                neurons_changed_outside_stim_pvlp151_gf=int((d[keep] != 0).sum()),
                                sum_abs_diff_outside=int(np.abs(d[keep]).sum())))
            return out
        report[b] = dict(
            deletion_minus_intact={k: paired(rate['intact'][k], rate['direct_edge_deletion'][k]) for k in prereg['readouts']},
            clamp_minus_intact={k: paired(rate['intact'][k], rate['all_output_clamp'][k]) for k in prereg['readouts']},
            deletion_minus_clamp={k: paired(rate['all_output_clamp'][k], rate['direct_edge_deletion'][k]) for k in prereg['readouts']},
            all_neuron={'deletion_vs_intact': all_neuron('direct_edge_deletion', 'intact'),
                        'clamp_vs_intact': all_neuron('all_output_clamp', 'intact'),
                        'deletion_vs_clamp': all_neuron('direct_edge_deletion', 'all_output_clamp')})
    return g, problems, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--prereg-sha256', required=True)
    ap.add_argument('--expected-code-sha', required=True, help='the frozen candidate commit the run must come from')
    ap.add_argument('--runs', required=True, nargs='+')
    ap.add_argument('--exit-files', required=True, nargs='+')
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--frozen-manifest', required=True)
    ap.add_argument('--frozen-root', required=True)
    ap.add_argument('--clamp-manifest', required=True)
    ap.add_argument('--clamp-root', required=True)
    ap.add_argument('--graph', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if sha(a.prereg) != a.prereg_sha256:
        raise SystemExit('prereg sha256 mismatch')
    contract, prereg = json.loads(Path(a.contract).read_text()), json.loads(Path(a.prereg).read_text())
    if sha(a.contract) != prereg['frozen_contract_sha256']:
        raise SystemExit('contract sha256 mismatch')
    pins, fprob, cprob = prereg['pins'], [], []
    frozen = manifest_files(a.frozen_manifest, a.frozen_root, pins['frozen_counts_manifest_sha256'], fprob, 'frozen')
    clamp = manifest_files(a.clamp_manifest, a.clamp_root, pins['clamp_counts_manifest_sha256'], cprob, 'clamp')
    exits = [json.loads(Path(e).read_text()) for e in a.exit_files]
    with np.load(a.graph, allow_pickle=False) as z:
        n = len(z['ptr']) - 1
    g, p2, report = evaluate(contract, prereg, a.prereg_sha256, load_rows(a.runs), exits, frozen, clamp, a.fixture,
                             a.expected_code_sha, sha(a.graph), n)
    if fprob:
        g['G5a_intact_reproduces_frozen'] = False
    if cprob:
        g['G5b_clamp_reproduces_accepted'] = False
    problems = fprob + cprob + p2
    hard = [k for k in g if k not in ('G5a_intact_reproduces_frozen', 'G5b_clamp_reproduces_accepted')]
    status = 'INVALID' if not all(g[k] for k in hard) else (
        'VALID' if all(g.values()) else 'VALID vs same-code arms only; G5a/G5b failed (disclosed first)')
    res = dict(label=prereg['label'], prereg_sha256=a.prereg_sha256, code_sha=a.expected_code_sha, gates=g,
               status=status, problems=problems, kind=prereg['report']['kind'],
               interpretation_limit=prereg['report']['interpretation_limit'], report=report)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gf_edge_results.json').write_text(json.dumps(res, indent=1) + '\n')
    print(status, json.dumps(g))
    for m in problems[:50]:
        print(' -', m)
    if status == 'INVALID':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
