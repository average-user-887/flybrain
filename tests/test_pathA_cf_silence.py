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
    assert prereg['readouts'] == ['GF', 'TTMn', 'PVLP151', 'DLMn', 'PSI']


def test_plan_changes_only_the_pvlp151_clamp():
    contract, prereg = load()
    sets, conds = run.cf_plan(contract, prereg)
    by_id = {c['id']: c for c in contract['conditions']}
    pv = set(prereg['sets']['PVLP151']['node_index'])
    assert len(pv) == 4
    assert len([c for c in conds if c['counterfactual']]) == 12
    for c, pc in zip(conds, prereg['conditions']):
        base = by_id[pc['base_condition']]
        for k in ('test', 'activate', 'rate_hz'):
            assert c[k] == base[k]
        assert c['counterfactual'] in ('silence_PVLP151', None)  # never the shuffled-input rewiring
        assert c['silence'] == base.get('silence', []) + (['PVLP151'] if c['counterfactual'] else [])
        assert c['seeds'] == (contract['protocol']['seeds'] if c['counterfactual'] else [0])
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
    _, c_clamp, *_ = run.run_one(clamp, stim, 200.0, hub, {}, dict(protocol=proto), 0)
    # Reference: no clamp, but the hub's outgoing edges given weight 0 (in a copy).
    w0 = arrays['weight'].copy()
    out_edges = np.concatenate([np.arange(arrays['ptr'][h], arrays['ptr'][h + 1]) for h in hub])
    w0[out_edges] = 0.0
    zeroed = Brain(arrays=dict(arrays, weight=w0), validate=True, dynamics='v3', backend='cpu')
    _, c_zero, *_ = run.run_one(zeroed, stim, 200.0, np.zeros(0, np.int64), {}, dict(protocol=proto), 0)
    _, c_intact, *_ = run.run_one(Brain(arrays=arrays, dynamics='v3', backend='cpu'), stim, 200.0,
                                  np.zeros(0, np.int64), {}, dict(protocol=proto), 0)
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
    for c, pc in zip(conds, prereg['conditions']):
        if not c['counterfactual']:
            continue
        cells = lambda names: np.concatenate([sets[s] for s in names] or [np.zeros(0, np.int64)])
        newly = np.setdiff1d(cells(c['silence']), cells(by_id[pc['base_condition']].get('silence', [])))
        assert set(newly.tolist()) == set(pv.tolist())
        mask = np.isin(pre, newly)  # edges that no longer transmit under the clamp
        assert int(mask.sum()) == sum(int(ptr[p + 1] - ptr[p]) for p in pv) == 1752
        assert np.all(np.isin(pre[mask], pv))
    gf = np.isin(pre, pv) & np.isin(post, sets['GF'])
    assert round(float(w[gf].sum()) / 0.275) == 603  # PVLP151 -> GF contacts, as traced
    assert sha(g) == contract['data']['graph_npz_sha256']


def test_decision_rule():
    assert cfa.decide(0.3, 0.4, True, True) == 'SUPPORTED'
    assert cfa.decide(0.3, 0.4, True, False) == 'INCONCLUSIVE'
    assert cfa.decide(0.9, 0.85, False, True) == 'REJECTED'
    assert cfa.decide(0.6, 0.4, True, True) == 'PARTIAL'
    assert cfa.decide(0.7, 0.9, False, False) == 'INCONCLUSIVE'
    assert cfa.decide(None, 0.4, True, True) == 'NOT EVALUATED'
