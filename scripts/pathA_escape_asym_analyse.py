#!/usr/bin/env python3
"""Independent static audit/report. Never imports the escape runner or steps a brain."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')


from pathA_escape_analyse import sha,integer,parse,independent_input,check_receipts,sparse


def expected_rows(contract, prereg):
    if contract['conditions'] != prereg['conditions']:
        raise ValueError('external contract/prereg conditions differ')
    expected = {}
    for c in contract['conditions']:
        if not isinstance(c['id'], str) or not c['id'] or any(not (s.isascii() and (s.isalnum() or s == '_')) for s in c['id']):
            raise ValueError('unsafe condition identifier')
        want_seeds = [0] if c['arm'] == 'empty_deletion_sham' else list(range(8))
        if c['seeds'] != want_seeds or not all(integer(s) for s in c['seeds']):
            raise ValueError('seed identity is not fixed typed integer sequence')
        for s in c['seeds']:
            if (c['id'], s) in expected:
                raise ValueError('duplicate expected row')
            expected[c['id'], s] = c
    if len(expected) != 75:
        raise ValueError('expected row count differs from 75')
    return expected


def audit_rows(contract, prereg, prereg_sha, code, out, n):
    expected = expected_rows(contract, prereg)
    lines = (out / 'runs.jsonl').read_text().splitlines()
    parsed, seen, counts = [], set(), {}
    want_prov = dict(contract_sha256=prereg['pins']['derived_contract_sha256'], prereg_sha256=prereg_sha,
                     code_sha=code, graph_npz_sha256=prereg['pins']['graph_npz_sha256'],
                     engine_sha256=prereg['pins']['engine_sha256'], dynamics='v3', backend='cpu')
    for line in lines:
        r = parse(line)
        if not integer(r.get('seed')):
            raise ValueError('recorded seed not a typed integer')
        key = r['condition'], r['seed']
        if key in seen or key not in expected:
            raise ValueError('duplicate or foreign row')
        seen.add(key)
        c = expected[key]
        if r['provenance'] != dict(want_prov, seed=key[1]) or not integer(r['provenance']['seed']):
            raise ValueError('row external provenance/seed mismatch')
        for field in ('arm', 'rate_hz', 'activate_nodes', 'clamp_nodes','zero_edges','weight_copy'):
            if r[field] != c[field] or (field == 'rate_hz' and type(r[field]) not in (int, float)):
                raise ValueError('row condition identity mismatch')
        if type(r['weight_copy']) is not bool: raise ValueError('weight copy flag must be bool')
        if not all(integer(x) for x in r['activate_nodes'] + r['clamp_nodes']+r['zero_edges']):
            raise ValueError('recorded node type mismatch')
        basename = key[0] + '_s' + str(key[1]) + '.npz'
        count_path, input_path = out / 'counts' / basename, out / 'inputs' / basename
        if sha(count_path) != r['counts_file_sha256'] or sha(input_path) != r['input_file_sha256']:
            raise ValueError('raw count/input artifact digest mismatch')
        exact, input_record = independent_input(c, key[1], contract['protocol'])
        if input_record != prereg['expected_inputs'][key[0] + '_s' + str(key[1])]:
            raise ValueError('input differs from external prereg manifest')
        manifest=prereg['expected_inputs'][key[0] + '_s' + str(key[1])]
        if not integer(manifest['event_ticks']) or not all(integer(v) for v in manifest['rng_seed_seq']+manifest['events_per_cell']):
            raise ValueError('external input manifest seed/count types malformed')
        with np.load(input_path, allow_pickle=False) as z:
            if set(z.files) != set(exact) or any(z[k].dtype != v.dtype or z[k].shape != v.shape or not np.array_equal(z[k], v)
                                                for k, v in exact.items()):
                raise ValueError('retained activation/event bytes differ from independent regeneration')
        a = r['audit']
        for field in ('input_sha256', 'rng_seed_seq', 'rng_state_sha256'):
            if a[field] != input_record[field]:
                raise ValueError('recorded runtime input/RNG mismatch')
        if not all(integer(s) for s in a['rng_seed_seq']):
            raise ValueError('runtime RNG sequence not integer')
        if a['initial_state_clean'] is not True or a['state_sha256'] != prereg['pins']['reset_state_sha256'] \
                or a['weight_sha256'] != prereg['pins']['deleted_weight_sha256' if c['zero_edges'] else 'v3_weight_sha256'] or a['parent_weight_unchanged'] is not True:
            raise ValueError('runtime reset/weight evidence mismatch')
        if not integer(r['event_ticks']) or r['event_ticks'] != input_record['event_ticks'] \
                or r['events_per_cell'] != input_record['events_per_cell'] or not all(integer(v) for v in r['events_per_cell']):
            raise ValueError('runtime forced-event count mismatch')
        sp = sparse(count_path, n)
        counts[key] = sp
        if not integer(r['total_spikes']) or r['total_spikes'] != sum(sp.values()) \
                or not integer(r['firing_neurons']) or r['firing_neurons'] != len(sp):
            raise ValueError('row network counts disagree with raw artifact')
        if c['rate_hz'] == 0 and sp:
            raise ValueError('no-input network fired')
        if any(sp.get(i, 0) != 0 for i in c['clamp_nodes']):
            raise ValueError('clamped GFC2 fired')
        if any(sp.get(i, 0) <= 0 for i in c['activate_nodes']):
            raise ValueError('direct activation control cell did not fire')
        duration = contract['protocol']['duration_ms'] / 1000
        if set(r['readouts']) != set(contract['readouts']):
            raise ValueError('readout coverage mismatch')
        for name in contract['readouts']:
            cc = [sp.get(i, 0) for i in contract['sets'][name]['node_index']]
            rr = r['readouts'][name]
            if rr['counts'] != cc or not all(integer(v) for v in rr['counts']) \
                    or type(rr['mean_rate_hz']) not in (int, float) or not math.isfinite(rr['mean_rate_hz']) \
                    or abs(rr['mean_rate_hz'] - float(np.mean(cc)) / duration) > 1e-12:
                raise ValueError('per-cell/population readout mismatch')
        stim_rate = float(np.mean([sp.get(i, 0) for i in c['activate_nodes']])) / duration if c['activate_nodes'] else 0.0
        if type(r['activated_mean_rate_hz']) not in (int, float) or not math.isfinite(r['activated_mean_rate_hz']) \
                or abs(r['activated_mean_rate_hz'] - stim_rate) > 1e-12:
            raise ValueError('actual activated-cell rate mismatch')
        if type(r['wall_s']) not in (int, float) or not math.isfinite(r['wall_s']) or r['wall_s'] < 0:
            raise ValueError('row elapsed time malformed')
        parsed.append((line, r))
    if seen != set(expected) or [(r['condition'], r['seed']) for _, r in parsed] != list(expected):
        raise ValueError('missing/extra rows or unexpected execution order')
    filenames = {c + '_s' + str(s) + '.npz' for c, s in expected}
    for folder in ('counts', 'inputs'):
        if {p.name for p in (out / folder).iterdir()} != filenames:
            raise ValueError('missing/foreign artifact files')
    for c in contract['conditions']:
        if c['arm'] == 'empty_deletion_sham':
            intact = c['id'].replace('empty_deletion_sham', 'intact')
            if counts[c['id'], 0] != counts[intact, 0]:
                raise ValueError('empty-deletion sham differs from intact')
    return parsed, counts


def check_deleted_weights(weight,deleted,pins):
    """Independent array-level check, separate from runner copy construction/checker."""
    if weight.dtype!=np.float32 or deleted.dtype!=np.float32 or deleted.shape!=weight.shape or np.flatnonzero(weight!=deleted).tolist()!=[0,14539] or np.any(deleted[[0,14539]]!=0) or hashlib.sha256(deleted.tobytes()).hexdigest()!=pins['deleted_weight_sha256']:
        raise ValueError('independent deleted weight reconstruction mismatch')


def check_accepted_controls(contract,counts,references):
    for condition in contract['conditions']:
        if condition['arm'] in ('intact','direct_activation_control'):
            for seed in condition['seeds']:
                if counts[condition['id'],seed]!=references[condition['id']+'_s'+str(seed)]:
                    raise ValueError('accepted control raw counts do not reproduce')


def describe(contract, counts):
    def stats(values):
        a = np.asarray(values, dtype=float)
        return dict(values=a.tolist(), mean=float(a.mean()), sample_sd=float(a.std(ddof=1)))
    def rates(condition, name):
        return [[counts[condition, seed].get(i, 0) / (contract['protocol']['duration_ms'] / 1000)
                 for i in contract['sets'][name]['node_index']] for seed in range(8)]
    report = {'absolute_rows': [], 'paired': {}}
    for c in contract['conditions']:
        for seed in c['seeds']:
            sp = counts[c['id'], seed]
            report['absolute_rows'].append(dict(condition=c['id'], seed=seed,
                readouts={name: dict(body_ids=contract['sets'][name]['body_ids'],
                    per_cell_counts=[sp.get(i,0) for i in contract['sets'][name]['node_index']],
                    population_mean_rate_hz=float(np.mean([sp.get(i,0) for i in contract['sets'][name]['node_index']]))
                                            / (contract['protocol']['duration_ms']/1000),
                    per_cell_rate_hz=[sp.get(i, 0) / (contract['protocol']['duration_ms'] / 1000)
                                      for i in contract['sets'][name]['node_index']]) for name in contract['readouts']}))
    for input_name in ('GF10001', 'GF10010', 'GF_BOTH'):
        intact, clamp = input_name + '_200_intact', input_name + '_200_delete_direct'
        pair = {}
        for name in contract['readouts']:
            a, b = np.array(rates(intact, name)), np.array(rates(clamp, name))
            d = b-a
            pair[name] = dict(intact_population=stats(a.mean(axis=1)), deletion_population=stats(b.mean(axis=1)),
                             deletion_minus_intact_population=dict(**stats(d.mean(axis=1)),
                                 lower=int((d.mean(axis=1)<0).sum()),equal=int((d.mean(axis=1)==0).sum()),
                                 higher=int((d.mean(axis=1)>0).sum())),
                             per_cell_change=[dict(body_id=body, **stats(d[:,j]), lower=int((d[:,j]<0).sum()),
                                equal=int((d[:,j]==0).sum()), higher=int((d[:,j]>0).sum()))
                                for j, body in enumerate(contract['sets'][name]['body_ids'])])
            if name in ('TTMn', 'GFC2'):
                positives = (a - np.array(rates('NO_INPUT_intact', name)) > 0).sum(axis=0)
                pair[name]['intact_presence'] = [dict(body_id=body, positive_seeds=int(n),
                    label='consistently observed' if n == 8 else ('absent at fixed input' if n == 0 else 'seed-dependent'))
                    for body, n in zip(contract['sets'][name]['body_ids'], positives)]
        active = next(c['activate_nodes'] for c in contract['conditions'] if c['id'] == intact)
        excluded = set(active) | set(contract['sets']['GFC2']['node_index'])
        pair['network_controls'] = []
        for seed in range(8):
            a, b = counts[intact, seed], counts[clamp, seed]
            changed = {i: b.get(i, 0)-a.get(i, 0) for i in set(a)|set(b) if b.get(i, 0)!=a.get(i, 0)}
            outside = {i:d for i,d in changed.items() if i not in excluded}
            pair['network_controls'].append(dict(seed=seed, neurons_changed=len(changed),
                sum_abs_difference=sum(abs(d) for d in changed.values()), neurons_changed_outside=len(outside),
                sum_abs_difference_outside=sum(abs(d) for d in outside.values())))
        report['paired'][input_name] = pair
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('contract', 'contract-sha256', 'prereg', 'prereg-sha256', 'expected-code-sha', 'expected-python',
                 'runs','accepted-runs', 'graph-dir', 'connectome-dir', 'out'):
        ap.add_argument('--'+flag, required=True)
    for flag in ('observed-child-pid', 'observed-wrapper-pid'):
        ap.add_argument('--'+flag, type=int, required=True)
    a = ap.parse_args()
    result = dict(status='INVALID', report={}, problems=[])
    try:
        if any(os.environ.get(k)!=v for k,v in ENV.items()): raise ValueError('CPU-only single-thread audit environment required')
        if sha(a.contract) != a.contract_sha256 or sha(a.prereg) != a.prereg_sha256:
            raise ValueError('external document pin mismatch')
        c, p = parse(Path(a.contract).read_text()), parse(Path(a.prereg).read_text())
        pins = p['pins']
        import importlib.metadata
        import platform
        if p['runtime_versions']!=dict(python=platform.python_version(),numpy=np.__version__,numba=importlib.metadata.version('numba')):
            raise ValueError('auditor runtime library versions differ from frozen pins')
        if pins['derived_contract_sha256'] != a.contract_sha256:
            raise ValueError('prereg contract mismatch')
        from pathA_escape_asym_common import validate_manifest,check_reference_coverage,accepted_references,check_edges
        validate_manifest(c,p); check_reference_coverage(c,p)
        references=accepted_references(p,a.accepted_runs)
        expected_rows(c,p)
        head = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        dirty = subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True)
        if head != a.expected_code_sha or dirty.strip() or len(a.expected_code_sha)!=40:
            raise ValueError('clean external source identity mismatch')
        for name, digest in pins['source_files_sha256'].items():
            if sha(ROOT/name)!=digest:
                raise ValueError('source pin mismatch: '+name)
        engine=hashlib.sha256()
        for name in ('brainlab/engine.py','brainlab/brain.py','brainlab/transmitter_policy.py'):
            engine.update(name.encode()+b'\0'+(ROOT/name).read_bytes()+b'\0')
        if engine.hexdigest()!=pins['engine_sha256']:
            raise ValueError('canonical engine identity mismatch')
        data = dict(graph_npz=Path(a.graph_dir)/'graph.npz',
                    neurons_feather=Path(a.connectome_dir)/'normalized/neurons.feather',
                    annotations_feather=Path(a.connectome_dir)/'annotations.feather',
                    edges_arrow=Path(a.connectome_dir)/'normalized/edges.arrow')
        for name, path in data.items():
            if sha(path)!=pins[name+'_sha256']:
                raise ValueError('external data pin mismatch: '+name)
        with np.load(data['graph_npz'],allow_pickle=False) as z:
            ids, ptr, post, weight = z['ids'],z['ptr'],z['post'],z['weight'].copy()
        import pyarrow as pa
        import pyarrow.feather as feather
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        nt = feather.read_table(data['neurons_feather']).to_pandas().sort_values('node_index')
        if not np.array_equal(nt.node_index,np.arange(len(ids))) or not np.array_equal(nt.source_id,ids):
            raise ValueError('neuron/body ordering mismatch')
        for name,s in c['sets'].items():
            if ids[s['node_index']].tolist()!=s['body_ids']:
                raise ValueError('readout/input body identity mismatch')
        # Independent default-v3 reconstruction: prepared signs remain; modulatory outputs are zero.
        labels = [str(x).strip().lower() for x in nt.neurotransmitter]
        for i, label in enumerate(labels):
            if label in ('dopamine','octopamine','serotonin'):
                weight[ptr[i]:ptr[i+1]]=0
        weight=np.ascontiguousarray(weight,dtype=np.float32)
        out=Path(a.runs).resolve()
        if hashlib.sha256(weight.tobytes()).hexdigest()!=pins['v3_weight_sha256'] \
                or sha(out/'weights.npy')!=pins['weights_npy_sha256']:
            raise ValueError('full shared weight identity mismatch')
        with (out/'weights.npy').open('rb') as f:
            saved=np.load(f,allow_pickle=False)
        if saved.dtype!=np.float32 or saved.shape!=weight.shape or saved.tobytes()!=weight.tobytes():
            raise ValueError('shared full weights differ from independent reconstruction')
        check_edges(ptr,post,ids,weight,p['direct_edges'])
        if sha(out/'deleted_weights.npy')!=pins['deleted_weights_npy_sha256']: raise ValueError('saved deleted weight pin mismatch')
        deleted=np.load(out/'deleted_weights.npy',allow_pickle=False)
        check_deleted_weights(weight,deleted,pins)
        sys.path.insert(0,str(ROOT/'scripts'))
        from pathA_gf_edge_analyse import fixture_digest
        fixture=out/'reset_fixture.npz'
        digest, problems=fixture_digest(fixture,pins['state_arrays_order'],'v3',len(ids))
        if sha(fixture)!=pins['reset_fixture_sha256'] or problems or digest!=pins['reset_state_sha256']:
            raise ValueError('shared full reset fixture mismatch')
        rows, counts=audit_rows(c,p,a.prereg_sha256,a.expected_code_sha,out,len(ids))
        check_accepted_controls(c,counts,references)
        argv=['timeout','--signal=TERM','3500',a.expected_python,'-B',str(ROOT/'scripts/pathA_escape_asym_run.py'),
              '--contract',str(Path(a.contract).resolve()),'--contract-sha256',a.contract_sha256,
              '--prereg',str(Path(a.prereg).resolve()),'--prereg-sha256',a.prereg_sha256,
              '--expected-code-sha',a.expected_code_sha,'--out',str(out),'--accepted-runs',str(Path(a.accepted_runs).resolve())]
        check_receipts(out,argv,a.expected_code_sha,a.observed_child_pid,a.observed_wrapper_pid,rows)
        for name,path in data.items():
            if sha(path)!=pins[name+'_sha256']:
                raise ValueError('data changed during audit')
        accepted_references(p,a.accepted_runs)
        result=dict(status='VALID',label=p['label'],code_sha=a.expected_code_sha,prereg_sha256=a.prereg_sha256,
                    problems=[],interpretation_limit=p['analysis']['interpretation'],report=describe(c,counts))
    except (ValueError,KeyError,TypeError,OSError,IndexError) as exc:
        result['problems']=[str(exc)]
    with Path(a.out).open('x') as f:
        f.write(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(result['status'])
    return 0 if result['status']=='VALID' else 2


if __name__=='__main__':
    sys.exit(main())
