"""PVLP151 counterfactual: the silencing flag clamps only the prereg's cells, which in
v3 removes exactly their outgoing existing edges and touches nothing else."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pathA_cf_analyse as cfa  # noqa: E402
import pathA_run as run  # noqa: E402

CONTRACT = ROOT / 'qualification/pathA/contract.json'
PREREG = ROOT / 'qualification/pathA/pvlp151_cf_prereg.json'
FROZEN_CONTRACT_SHA = 'cd69cb6db2f2ba825dc804e762ea78c5651f3d61ec962955126c3fec02aa1c97'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    return json.loads(CONTRACT.read_text()), json.loads(PREREG.read_text())


def test_prereg_bound_to_frozen_contract():
    contract, prereg = load()
    assert sha(CONTRACT) == FROZEN_CONTRACT_SHA == prereg['frozen_contract_sha256']
    assert prereg['graph_npz_sha256'] == contract['data']['graph_npz_sha256']
    assert prereg['sets']['PVLP151']['body_ids'] == [10173, 11677, 11826, 12275]
    assert prereg['readouts'] == ['GF', 'TTMn', 'PVLP151', 'PSI', 'DLMn', 'GFC2']
    assert 'decision_rule' not in prereg and prereg['report']['kind'].startswith('DESCRIPTIVE')
    assert {c['base_condition'] for c in prereg['conditions']} == {'A2_IRR_LC15_200', 'A2_LC4_200', 'A2_LPLC2_200',
                                                                  'A2_LC4_0'}


def test_plan_changes_only_the_pvlp151_clamp():
    contract, prereg = load()
    sets, conds = run.cf_plan(contract, prereg)
    by_id = {c['id']: c for c in contract['conditions']}
    pv = set(prereg['sets']['PVLP151']['node_index'])
    assert len(pv) == 4
    kinds = [c['counterfactual'] for c in conds]
    assert kinds.count('silence_PVLP151') == 4 and kinds.count(None) == 4 and kinds.count('zero_outgoing_PVLP151') == 3
    for c, pc in zip(conds, prereg['conditions']):
        base = by_id[pc['base_condition']]
        for k in ('test', 'activate', 'rate_hz'):
            assert c[k] == base[k]
        assert c['rate_hz'] in (0, 200)
        clamp = c['counterfactual'] == 'silence_PVLP151'
        assert c['silence'] == base.get('silence', []) + (['PVLP151'] if clamp else [])  # never shuffled rewiring
        assert c['zero_outgoing'] == (['PVLP151'] if c['counterfactual'] == 'zero_outgoing_PVLP151' else [])
        assert c['seeds'] == ([0] if c['zero_outgoing'] else contract['protocol']['seeds'])
        stim = set().union(*(contract['sets'][s]['node_index'] for s in c['activate']))
        assert not stim & pv  # clamp never hits an activated cell, so the input trains are A2's
    assert sorted(by_id) == sorted(c['id'] for c in contract['conditions'])  # frozen list untouched


def _toy(n=150, k=14, seed=5):
    rng = np.random.default_rng(seed)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(rng.poisson(k, size=n), out=ptr[1:])
    post = rng.integers(0, n, size=int(ptr[-1]), dtype=np.int32)
    weight = (rng.geometric(0.3, size=len(post)) * np.where(rng.random(len(post)) < 0.25, -1, 1)
              * 0.275 * 8).astype(np.float32)
    return dict(ptr=ptr, post=post, weight=np.ascontiguousarray(weight), ids=np.arange(n, dtype=np.int64))


def _digest(arrays):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()}


def test_clamp_equals_zeroing_only_outgoing_edges_and_leaves_graph_untouched():
    from brainlab.brain import Brain
    contract, _ = load()
    proto = dict(contract['protocol'], duration_ms=60.0)
    arrays = _toy()
    before = _digest(arrays)
    stim, hub = np.arange(0, 20), np.array([40, 41, 42, 43])
    rest = np.setdiff1d(np.arange(len(arrays['ids'])), hub)
    clamp = Brain(arrays=arrays, validate=True, dynamics='v3', backend='cpu')
    a_clamp, a_zero, a_intact = {}, {}, {}
    _, c_clamp, *_ = run.run_one(clamp, stim, 200.0, hub, {}, dict(protocol=proto), 0, a_clamp)
    # Reference: no clamp, but the hub's outgoing edges given weight 0 (the runner's in-memory copy).
    w0 = run.zeroed_weight(arrays, hub)
    out_edges = np.concatenate([np.arange(arrays['ptr'][h], arrays['ptr'][h + 1]) for h in hub])
    assert np.all(w0[out_edges] == 0) and np.array_equal(np.delete(w0, out_edges), np.delete(arrays['weight'], out_edges))
    zeroed = Brain(arrays=dict(arrays, weight=w0), validate=True, dynamics='v3', backend='cpu')
    _, c_zero, *_ = run.run_one(zeroed, stim, 200.0, np.zeros(0, np.int64), {}, dict(protocol=proto), 0, a_zero)
    _, c_intact, *_ = run.run_one(Brain(arrays=arrays, dynamics='v3', backend='cpu'), stim, 200.0,
                                  np.zeros(0, np.int64), {}, dict(protocol=proto), 0, a_intact)
    for a in (a_clamp, a_zero, a_intact):  # in-run audit: clean reset, identical input train
        assert a['initial_state_clean'] is True and a['input_sha256'] == a_clamp['input_sha256']
    assert not run.state_at_rest(clamp)  # the audit can see a non-rest state (after a run)
    clamp.reset_state()
    assert run.state_at_rest(clamp)
    assert c_clamp[hub].sum() == 0 and c_zero[hub].sum() > 0  # the clamp holds; the hub did fire
    assert np.array_equal(c_clamp[rest], c_zero[rest])         # clamp == zeroing exactly its out-edges
    assert not np.array_equal(c_clamp[rest], c_intact[rest])   # and that removal matters in this toy
    assert _digest(arrays) == before                           # no graph array was modified


GDIR = os.environ.get('NEUROFLY_GRAPH_DIR')


@pytest.mark.skipif(not GDIR or not (Path(GDIR) / 'graph.npz').exists(), reason='needs the pinned MaleCNS graph')
def test_real_graph_silenced_edges_are_pvlp151_outgoing_and_sha_unchanged():
    contract, prereg = load()
    g = Path(GDIR) / 'graph.npz'
    assert sha(g) == contract['data']['graph_npz_sha256']
    with np.load(g, allow_pickle=False) as z:
        ptr, post, w = z['ptr'], z['post'], z['weight']
    _, conds = run.cf_plan(contract, prereg)
    sets = {k: np.array(v['node_index']) for k, v in dict(contract['sets'], **prereg['sets']).items()}
    pre = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    pv = sets['PVLP151']
    by_id = {c['id']: c for c in contract['conditions']}
    cells = lambda names: np.concatenate([sets[s] for s in names] or [np.zeros(0, np.int64)])  # noqa: E731
    out_edges = np.flatnonzero(np.isin(pre, pv))
    assert len(out_edges) == sum(int(ptr[p + 1] - ptr[p]) for p in pv) == 1752
    for c, pc in zip(conds, prereg['conditions']):
        newly = np.setdiff1d(cells(c['silence']), cells(by_id[pc['base_condition']].get('silence', [])))
        if c['counterfactual'] == 'silence_PVLP151':
            # the clamped sources are exactly PVLP151, so exactly its out-edges stop transmitting
            assert set(newly.tolist()) == set(pv.tolist())
        else:
            assert len(newly) == 0
        if c['zero_outgoing']:
            w0 = run.zeroed_weight(dict(ptr=ptr, weight=w), cells(c['zero_outgoing']))
            changed = np.flatnonzero(w0 != w)
            assert np.array_equal(changed, out_edges[w[out_edges] != 0]) and np.all(w0[out_edges] == 0)
    gf = np.isin(pre, pv) & np.isin(post, sets['GF'])
    assert round(float(w[gf].sum()) / 0.275) == 603  # PVLP151 -> GF contacts, as traced
    assert sha(g) == contract['data']['graph_npz_sha256']


def test_paired_summary_is_descriptive():
    p = cfa.paired([40.0, 44.0, 46.0], [10.0, 44.0, 50.0])
    assert p['diff_hz'] == [-30.0, 0.0, 4.0] and (p['seeds_lower'], p['seeds_higher'], p['seeds_equal']) == (1, 1, 1)
    assert abs(p['percent_change'] - 100 * (-26 / 3) / (130 / 3)) < 1e-9
    assert cfa.paired([0.0, 0.0], [1.0, 0.0])['percent_change'] is None  # no percent over a zero denominator
    assert not any(k in p for k in ('verdict', 'pass', 'supported'))
