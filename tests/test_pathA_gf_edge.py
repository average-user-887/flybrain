"""PVLP151->GF direct-edge deletion: runner flag, reset fixture, launcher and analyser gates,
with negative regressions.  Toy graphs only, except the pinned-graph checks (skipped without it)."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pathA_gf_edge_analyse as ga  # noqa: E402
import pathA_run as run  # noqa: E402
import pathA_shard as shard  # noqa: E402

Q = ROOT / 'qualification/pathA'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------- toy graph: PV = {1, 2}, GF = {0}; edges 1->0, 2->0 are the direct edges ----------
def toy():
    adj = {0: [3, 4], 1: [0, 3, 5], 2: [4, 0], 3: [1], 4: [2], 5: [1, 2]}
    n = 8
    ptr = np.zeros(n + 1, np.int64)
    post = []
    for i in range(n):
        post += adj.get(i, [])
        ptr[i + 1] = len(post)
    w = np.full(len(post), 0.275 * 300, np.float32)
    return dict(ptr=ptr, post=np.array(post, np.int32), weight=w, ids=np.arange(100, 100 + n, dtype=np.int64))


def toy_prereg(arrays, drop=None, extra=None, body=None, count=None):
    ptr, post, ids = arrays['ptr'], arrays['post'], arrays['ids']
    rows = []
    for s in (1, 2):
        for e in range(ptr[s], ptr[s + 1]):
            if post[e] == 0:
                rows.append(dict(edge_index=int(e), pre_body=int(ids[s]), post_body=int(ids[0])))
    if drop is not None:
        rows.pop(drop)
    if extra is not None:
        rows.append(extra)
    if body is not None:
        rows[0]['pre_body'] = body
    return dict(direct_edges=rows, direct_edges_from='PV', direct_edges_to='GF',
                direct_edge_count=count if count is not None else len(rows))


SETS = dict(PV=np.array([1, 2]), GF=np.array([0]))


def test_direct_edges_must_be_exactly_the_from_to_edges():
    a = toy()
    assert run.check_direct_edges(a, SETS, toy_prereg(a)).tolist() == [2, 6]
    for bad in (dict(drop=0), dict(extra=dict(edge_index=4, pre_body=101, post_body=103)),  # missing / extra
                dict(body=999), dict(count=3)):                                            # wrong body / count
        with pytest.raises(SystemExit):
            run.check_direct_edges(a, SETS, toy_prereg(a, **bad))
    z = toy()
    z['weight'][2] = 0.0  # a listed edge with zero weight cannot be "deleted"
    with pytest.raises(SystemExit):
        run.check_direct_edges(z, SETS, toy_prereg(z))


def test_deleted_weight_changes_only_the_listed_edges_and_empty_is_byte_identical():
    a = toy()
    before = a['weight'].tobytes()
    w = run.deleted_weight(a, [7, 3])
    assert np.flatnonzero(w != a['weight']).tolist() == [3, 7] and np.all(w[[3, 7]] == 0)
    e = run.deleted_weight(a, [])
    assert e is not a['weight'] and e.tobytes() == before and a['weight'].tobytes() == before
    a['weight'][4] = 0.0
    with pytest.raises(SystemExit):  # an already-zero edge would not change: refuse, never silently
        run.deleted_weight(a, [4])


def _brain(arrays):
    from brainlab.brain import Brain
    return Brain(arrays=arrays, validate=True, dynamics='v3', backend='cpu')


def test_state_digest_covers_bytes_dtype_shape_scalars_and_detects_contamination():
    b = _brain(toy())
    b.reset_state()
    d0 = run.state_digest(b)
    proto = json.loads((Q / 'contract.json').read_text())['protocol']
    run.run_one(b, np.array([1, 2, 5]), 200.0, np.zeros(0, np.int64), {}, dict(protocol=dict(proto, duration_ms=30.0)), 0)
    assert run.state_digest(b) != d0           # parent state after a run is contaminated...
    b.reset_state()
    assert run.state_digest(b) == d0           # ...and reset restores the canonical digest
    v = b.v
    b.v = v.astype(np.float64)
    assert run.state_digest(b) != d0           # dtype
    b.v = v.reshape(1, -1)
    assert run.state_digest(b) != d0           # shape
    b.v = v
    b.sim_ms = 0
    assert run.state_digest(b) != d0           # scalar repr (0 vs 0.0)
    b.sim_ms = 0.0
    b.g[0, 3] = 1e-9
    assert run.state_digest(b) != d0           # bytes
    assert run.rng_digest([0, 2000, 3]) == ga.rng_digest([0, 2000, 3]) != ga.rng_digest([1, 2000, 3])


def test_reset_fixture_is_checked_independently(tmp_path):
    b = _brain(toy())
    order = list(b._state_arrays())
    f = tmp_path / 'fx.npz'
    d = run.save_reset_fixture(b, f)
    got, problems = ga.fixture_digest(f, order, 'v3', b.n)
    assert got == d and problems == []
    with np.load(f) as z:
        good = {k: z[k] for k in z.files}
    for name, mutate in (('v', lambda x: x.__setitem__('state__v', x['state__v'] + 1)),
                         ('queue', lambda x: x['state__queue'].__setitem__((0, 0), 5)),
                         ('cursor', lambda x: x.__setitem__('scalar__cursor', np.array(3)))):
        bad = copy.deepcopy(good)
        mutate(bad)
        p = tmp_path / f'bad_{name}.npz'
        np.savez(p, **bad)
        got_bad, probs = ga.fixture_digest(p, order, 'v3', b.n)
        assert probs and got_bad != d, name
    _, probs = ga.fixture_digest(f, order[:-1], 'v3', b.n)  # a missing/renamed array is refused
    assert probs


def test_toy_deletion_arm_inputs_identical_parent_untouched_and_empty_sham_equal():
    a = toy()
    proto = json.loads((Q / 'contract.json').read_text())['protocol']
    c = dict(protocol=dict(proto, duration_ms=40.0))
    parent = hashlib.sha256(a['weight'].tobytes()).hexdigest()
    stim = np.array([5])
    out = {}
    for arm, w in (('intact', None), ('del', run.deleted_weight(a, [2, 6])), ('sham', run.deleted_weight(a, []))):
        b = _brain(a if w is None else dict(a, weight=w))
        au = {}
        _, cnt, *_ = run.run_one(b, stim, 200.0, np.zeros(0, np.int64), {}, c, 0, au)
        out[arm] = (cnt.copy(), au)
    assert hashlib.sha256(a['weight'].tobytes()).hexdigest() == parent
    assert len({out[k][1]['input_sha256'] for k in out}) == 1 and len({out[k][1]['state_sha256'] for k in out}) == 1
    assert np.array_equal(out['sham'][0], out['intact'][0])
    assert not np.array_equal(out['del'][0], out['intact'][0])  # the deletion matters in this toy


def test_shard_wrapper_captures_exit_code_and_refuses_overwrite(tmp_path):
    ef, log = tmp_path / 'x.exit.json', tmp_path / 'x.log'
    rc = shard.main(['--exit-file', str(ef), '--log', str(log), '--', sys.executable, '-c',
                     'import sys; print("diag"); sys.exit(3)'])
    rec = json.loads(ef.read_text())
    assert rc == 3 and rec['exit_code'] == 3 and 'diag' in log.read_text() and rec['log_sha256'] == sha(log)
    with pytest.raises(SystemExit):
        shard.main(['--exit-file', str(ef), '--log', str(log), '--', sys.executable, '-c', 'pass'])


# ---------- analyser gates on a consistent synthetic dataset, then one defect at a time ----------
def synthetic(tmp_path):
    n = 20
    b = _brain(dict(ptr=np.zeros(n + 1, np.int64), post=np.zeros(0, np.int32), weight=np.zeros(0, np.float32),
                    ids=np.arange(n, dtype=np.int64)))
    fx = tmp_path / 'fx.npz'
    sdig = run.save_reset_fixture(b, fx)
    contract = dict(protocol=dict(dynamics='v3', backend='cpu', duration_ms=1000.0),
                    sets=dict(GF=dict(node_index=[0]), S=dict(node_index=[5, 6])),
                    conditions=[dict(id='B', activate=['S'], rate_hz=200)])
    arms = ['intact', 'direct_edge_deletion', 'all_output_clamp']
    prereg = dict(frozen_contract_sha256='c' * 64, label='L', report=dict(kind='D', interpretation_limit='I'),
                  readouts=['GF', 'PVLP151'], sets=dict(PVLP151=dict(node_index=[1, 2])),
                  direct_edges=[dict(edge_index=3), dict(edge_index=7)],
                  conditions=[dict(id=f'B_{a}', base_condition='B', arm=a, seeds=[0, 1]) for a in arms]
                  + [dict(id='B_sham', base_condition='B', arm='sham_empty_deletion', seeds=[0])],
                  pins=dict(state_arrays_order=list(b._state_arrays()), reset_fixture_sha256=sha(fx),
                            reset_state_sha256=sdig, graph_npz_sha256='a' * 64, engine_sha256='e' * 64,
                            base_weight_sha256='1' * 64, deleted_weight_sha256='2' * 64))
    rows, frozen, clamp = {}, {}, {}
    shard_of = lambda c: 'sA' if c['arm'] in ('intact', 'sham_empty_deletion') else 'sB'  # noqa: E731
    prereg['run_procedure'] = dict(timeout_s=3500, shards=[
        dict(name=nm, only=[c['id'] for c in prereg['conditions'] if shard_of(c) == nm]) for nm in ('sA', 'sB')])
    for nm in ('sA', 'sB'):
        (tmp_path / nm / 'counts').mkdir(parents=True)
    logs = {'sA': [], 'sB': []}
    for c in prereg['conditions']:
        for s in c['seeds']:
            node = np.array([0, 5], np.int32) if c['arm'] != 'all_output_clamp' else np.array([0], np.int32)
            cnt = np.array([10 + s, 3], np.int32) if c['arm'] != 'all_output_clamp' else np.array([8], np.int32)
            if c['arm'] == 'direct_edge_deletion':
                cnt = cnt - np.array([1, 0], np.int32)
            p = tmp_path / shard_of(c) / 'counts' / f"{c['id']}_s{s}.npz"
            logs[shard_of(c)].append(f"{c['id']} {s} 1.0s {{}}")
            np.savez(p, node_index=node, counts=cnt)
            seq = [s, 2000, 2]
            r = dict(condition=c['id'], seed=s, zero_edges=[3, 7] if c['arm'] == 'direct_edge_deletion' else [],
                     provenance=dict(contract_sha256='c' * 64, graph_npz_sha256='a' * 64, dynamics='v3', backend='cpu',
                                     engine_sha256='e' * 64, code_sha='f' * 40, cf_prereg_sha256='p' * 64, seed=s),
                     audit=dict(initial_state_clean=True, state_sha256=sdig, rng_seed_seq=seq,
                                rng_state_sha256=ga.rng_digest(seq), input_sha256=f'{s:064x}',
                                weight_sha256='2' * 64 if c['arm'] == 'direct_edge_deletion' else '1' * 64,
                                parent_weight_unchanged=True))
            rows[(c['id'], s)] = (r, p)
            if c['arm'] == 'intact':
                frozen[('B', s)] = p
            if c['arm'] == 'all_output_clamp':
                clamp[('B', s)] = p
    exits = []
    for sh in prereg['run_procedure']['shards']:
        log = tmp_path / f"{sh['name']}.log"
        log.write_text('\n'.join(logs[sh['name']]) + '\n')
        exits.append(dict(exit_code=0, log=str(log), log_sha256=sha(log),
                          argv=['timeout', '--signal=TERM', '3500', sys.executable, '/w/scripts/pathA_run.py',
                                '--contract-sha256', 'c' * 64, '--cf-silence-prereg', '/w/p.json',
                                '--cf-silence-prereg-sha256', 'p' * 64, '--out', str(tmp_path / sh['name']),
                                '--only', ','.join(sh['only'])]))
    args = dict(contract=contract, prereg=prereg, psha='p' * 64, rows=rows, exits=exits, frozen=frozen,
                clamp=clamp, fixture=fx, code_sha='f' * 40, graph_sha_now='a' * 64, n=n)
    return args


def test_analyser_valid_synthetic_passes_every_gate(tmp_path):
    g, problems, report = ga.evaluate(**synthetic(tmp_path))
    assert all(g.values()), problems
    assert report['B']['deletion_minus_clamp']['GF']['diff_hz'] == [1.0, 2.0]


@pytest.mark.parametrize('defect,gate', [
    ('exit_missing', 'G6_complete_and_bound'), ('exit_nonzero', 'G6_complete_and_bound'),
    ('seed_missing', 'G6_complete_and_bound'), ('seed_999', 'G6_complete_and_bound'),
    ('unrelated_exit', 'G6_complete_and_bound'), ('missing_shard', 'G6_complete_and_bound'),
    ('duplicate_shard', 'G6_complete_and_bound'), ('wrong_log_hash', 'G6_complete_and_bound'),
    ('log_missing', 'G6_complete_and_bound'), ('log_lacks_row', 'G6_complete_and_bound'),
    ('no_timeout_wrapper', 'G6_complete_and_bound'), ('wrong_prereg_arg', 'G6_complete_and_bound'),
    ('wrong_out_dir', 'G6_complete_and_bound'), ('timeout_exit_124', 'G6_complete_and_bound'),
    ('code_sha', 'G6_complete_and_bound'), ('row_missing', 'G6_complete_and_bound'),
    ('graph_changed', 'G6_complete_and_bound'),
    ('state_hash', 'G2_clean_state'), ('rng_state', 'G2_clean_state'), ('fixture_tampered', 'G2_clean_state'),
    ('input_null', 'G3_identical_inputs'), ('input_differs', 'G3_identical_inputs'),
    ('deleted_weight', 'G4_deletion_exact'), ('parent_contaminated', 'G4_deletion_exact'),
    ('zero_edges', 'G4_deletion_exact'), ('intact_weight_copy', 'G4_deletion_exact'),
    ('clamp_fires', 'G1_clamp_zero'), ('frozen_mismatch', 'G5a_intact_reproduces_frozen'),
    ('clamp_mismatch', 'G5b_clamp_reproduces_accepted'), ('sham_differs', 'G5c_sham_equals_intact'),
])
def test_analyser_rejects_each_defect(tmp_path, defect, gate):
    a = synthetic(tmp_path)
    rows = a['rows']
    au = lambda k: rows[k][0]['audit']  # noqa: E731
    if defect == 'exit_missing':
        a['exits'] = []
    elif defect == 'exit_nonzero':
        a['exits'][1]['exit_code'] = -15
    elif defect == 'seed_missing':  # reviewer reproduction 1
        del rows[('B_intact', 1)][0]['provenance']['seed']
    elif defect == 'seed_999':
        rows[('B_direct_edge_deletion', 0)][0]['provenance']['seed'] = 999
    elif defect == 'unrelated_exit':  # reviewer reproduction 2: one unrelated record only
        a['exits'] = [dict(argv=['/bin/true'], exit_code=0, log='/nonexistent', log_sha256='0' * 64)]
    elif defect == 'missing_shard':
        a['exits'] = a['exits'][:1]
    elif defect == 'duplicate_shard':
        a['exits'] = [a['exits'][0], dict(a['exits'][0]), a['exits'][1]]
    elif defect == 'wrong_log_hash':
        a['exits'][0]['log_sha256'] = '0' * 64
    elif defect == 'log_missing':
        Path(a['exits'][1]['log']).unlink()
    elif defect == 'log_lacks_row':
        lg = Path(a['exits'][1]['log'])
        lg.write_text('\n'.join(lg.read_text().splitlines()[1:]) + '\n')
        a['exits'][1]['log_sha256'] = sha(lg)
    elif defect == 'no_timeout_wrapper':
        a['exits'][0]['argv'] = a['exits'][0]['argv'][3:]
    elif defect == 'wrong_prereg_arg':
        av = a['exits'][0]['argv']
        av[av.index('--cf-silence-prereg-sha256') + 1] = 'q' * 64
    elif defect == 'wrong_out_dir':
        av = a['exits'][0]['argv']
        av[av.index('--out') + 1] = str(Path(a['fixture']).parent / 'sB')
    elif defect == 'timeout_exit_124':
        a['exits'][1]['exit_code'] = 124
    elif defect == 'code_sha':
        rows[('B_intact', 1)][0]['provenance']['code_sha'] = 'd' * 40
    elif defect == 'row_missing':
        del rows[('B_direct_edge_deletion', 1)]
    elif defect == 'graph_changed':
        a['graph_sha_now'] = 'b' * 64
    elif defect == 'state_hash':
        au(('B_direct_edge_deletion', 0))['state_sha256'] = '0' * 64
    elif defect == 'rng_state':
        au(('B_intact', 1))['rng_seed_seq'] = [1, 2000, 3]
    elif defect == 'fixture_tampered':
        with np.load(a['fixture']) as z:
            d = {k: z[k] for k in z.files}
        d['state__v'] = d['state__v'] + 1
        np.savez(a['fixture'], **d)
    elif defect == 'input_null':
        for k in rows:
            au(k)['input_sha256'] = None
    elif defect == 'input_differs':
        au(('B_all_output_clamp', 1))['input_sha256'] = '9' * 64
    elif defect == 'deleted_weight':
        au(('B_direct_edge_deletion', 0))['weight_sha256'] = '3' * 64
    elif defect == 'parent_contaminated':
        au(('B_intact', 1))['parent_weight_unchanged'] = False
    elif defect == 'zero_edges':
        rows[('B_direct_edge_deletion', 0)][0]['zero_edges'] = [3]
    elif defect == 'intact_weight_copy':
        au(('B_intact', 0))['weight_sha256'] = '2' * 64
    elif defect == 'clamp_fires':
        p = rows[('B_all_output_clamp', 0)][1]
        np.savez(p, node_index=np.array([0, 1], np.int32), counts=np.array([8, 2], np.int32))
        a['clamp'][('B', 0)] = p
    elif defect == 'frozen_mismatch':
        q = Path(a['fixture']).parent / 'other.npz'
        np.savez(q, node_index=np.array([0], np.int32), counts=np.array([1], np.int32))
        a['frozen'][('B', 1)] = q
    elif defect == 'clamp_mismatch':
        q = Path(a['fixture']).parent / 'other.npz'
        np.savez(q, node_index=np.array([0], np.int32), counts=np.array([1], np.int32))
        a['clamp'][('B', 0)] = q
    elif defect == 'sham_differs':
        np.savez(rows[('B_sham', 0)][1], node_index=np.array([0], np.int32), counts=np.array([1], np.int32))
    g, problems, _ = ga.evaluate(**a)
    assert g[gate] is False, (defect, problems)


# ---------- pinned graph and frozen prereg (skipped without the data) ----------
GDIR, CDIR = os.environ.get('NEUROFLY_GRAPH_DIR'), os.environ.get('NEUROFLY_CONNECTOME_DIR')
HAVE = bool(GDIR and CDIR and (Path(GDIR) / 'graph.npz').exists())


@pytest.mark.skipif(not HAVE or not (Q / 'pvlp151_gf_edge_prereg.json').exists(), reason='needs the pinned graph')
def test_pinned_graph_direct_edges_weights_and_fixture_match_the_prereg():
    contract = json.loads((Q / 'contract.json').read_text())
    pre = json.loads((Q / 'pvlp151_gf_edge_prereg.json').read_text())
    pins = pre['pins']
    assert sha(Path(GDIR) / 'graph.npz') == pins['graph_npz_sha256'] == contract['data']['graph_npz_sha256']
    arrays = run.load_v3_arrays(Path(GDIR), Path(CDIR), contract['data']['neurons_feather_sha256'])
    sets, conds = run.cf_plan(contract, pre)
    S = {k: np.array(v['node_index']) for k, v in dict(contract['sets'], **sets).items()}
    listed = run.check_direct_edges(arrays, S, pre)
    assert len(listed) == 8 == pre['direct_edge_count']
    assert sorted({e['pre_body'] for e in pre['direct_edges']}) == [10173, 11677, 11826, 12275]
    assert sorted({e['post_body'] for e in pre['direct_edges']}) == [10001, 10010]
    assert hashlib.sha256(arrays['weight'].tobytes()).hexdigest() == pins['base_weight_sha256']
    w = run.deleted_weight(arrays, listed)
    assert hashlib.sha256(w.tobytes()).hexdigest() == pins['deleted_weight_sha256']
    assert np.flatnonzero(w != arrays['weight']).tolist() == listed.tolist()
    assert run.deleted_weight(arrays, []).tobytes() == arrays['weight'].tobytes()
    fx = Q / pins['reset_fixture_file']
    assert sha(fx) == pins['reset_fixture_sha256']
    d, probs = ga.fixture_digest(fx, pins['state_arrays_order'], 'v3', len(arrays['ids']))
    assert probs == [] and d == pins['reset_state_sha256']
    for c in conds:
        assert c['zero_edges'] in ([], listed.tolist()) and c.get('counterfactual') != 'shuffled_input'
