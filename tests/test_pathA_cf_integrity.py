"""Negative tests for the independent PVLP151 preparation review (probes of 51cc2c9):
an unpinned transmitter table and physically impossible sparse counts must be refused."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pathA_cf_analyse as cfa  # noqa: E402
import pathA_run as run  # noqa: E402


def _tiny(tmp_path, transmitter):
    pa = pytest.importorskip('pyarrow')
    feather = pytest.importorskip('pyarrow.feather')
    g, c = tmp_path / 'g', tmp_path / transmitter
    g.mkdir(exist_ok=True)
    (c / 'normalized').mkdir(parents=True)
    np.savez(g / 'graph.npz', ptr=np.array([0, 1, 1], np.int64), post=np.array([1], np.int32),
             weight=np.array([0.275], np.float32), ids=np.array([10, 11], np.int64))
    feather.write_feather(pa.table({'node_index': [0, 1], 'neurotransmitter': [transmitter, 'acetylcholine']}),
                          str(c / 'normalized/neurons.feather'))
    return g, c, hashlib.sha256((c / 'normalized/neurons.feather').read_bytes()).hexdigest()


def test_transmitter_table_must_match_contract_sha_before_labels_apply(tmp_path):
    g, ach, ach_sha = _tiny(tmp_path, 'acetylcholine')
    assert run.load_v3_arrays(g, ach, ach_sha)['weight'].tolist() == pytest.approx([0.275])
    # Probe: same graph, ACh relabelled dopamine, makes the 0.275 edge 0 under v3 with the graph sha unchanged.
    _, da, _ = _tiny(tmp_path, 'dopamine')
    with pytest.raises(SystemExit, match='neurons.feather sha256'):
        run.load_v3_arrays(g, da, ach_sha)


@pytest.mark.parametrize('node,counts,why', [
    ([156, 1565, 0], [1, -1, -10], 'negative'),                 # the probe: +1/-1 cancel at PVLP151
    ([0, 1], np.array([1.5, 2.0]), 'integer'),                   # non-integer counts
    ([0, 0], [3, 4], 'duplicated'),                              # duplicate node
    ([0, 166700], [1, 1], 'out of range'),                       # index >= n
    ([-1, 2], [1, 1], 'out of range'),                           # negative index
])
def test_impossible_sparse_counts_are_refused(tmp_path, node, counts, why):
    f = tmp_path / 'c.npz'
    np.savez(f, node_index=np.asarray(node, np.int32),
             counts=np.asarray(counts) if why == 'integer' else np.asarray(counts, np.int32))
    with pytest.raises(SystemExit, match=why):
        cfa.sparse(f, 166700)


def test_valid_sparse_counts_accepted(tmp_path):
    f = tmp_path / 'ok.npz'
    np.savez(f, node_index=np.array([0, 5], np.int32), counts=np.array([0, 7], np.int32))
    node, cnt = cfa.sparse(f, 166700)
    assert node.tolist() == [0, 5] and cnt.tolist() == [0, 7]
