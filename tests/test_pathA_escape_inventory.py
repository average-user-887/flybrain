"""Static selection/contact audit only; never instantiate or step a brain."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pathA_escape_inventory as inv


def toy():
    # GF0 -> TTM1 (direct), GFC2=2, inhibitory X=3; endpoint cycles excluded.
    adj = [[1, 2, 3], [0], [0, 1], [1], [], []]
    ptr = np.array([0] + list(np.cumsum([len(x) for x in adj])), dtype=np.int64)
    post = np.array(sum(adj, []), dtype=np.int32)
    raw = np.full(len(post), np.float32(0.275 * 4))
    raw[-1] *= -1
    arrays = dict(ptr=ptr, post=post, ids=np.arange(100, 106), weight=raw)
    sets = dict(GF=np.array([0]), TTMn=np.array([1]), GFC2=np.array([2]), PSI=np.array([4]), DLMn=np.array([5]))
    labels = [dict(cell_type=str(i), neurotransmitter='unknown', quality='toy') for i in range(6)]
    pre = np.repeat(np.arange(6), np.diff(ptr))
    counts = {(int(a), int(b)): 4 for a, b in zip(pre, post)}
    return arrays, sets, labels, counts


def test_routes_exclude_endpoints_keep_signed_legs_and_empty_pairs():
    a, sets, labels, counts = toy()
    before = a['weight'].tobytes()
    got = inv.inventory(a, a['weight'], sets, labels, counts, 0.275)
    assert a['weight'].tobytes() == before
    assert got['direct_and_feedback']['GF->TTMn']['contacts'] == 4
    assert got['direct_and_feedback']['GF->PSI']['rows'] == []
    routes = got['two_hop_gf_x_ttmn']['rows']
    assert [r['node_index'] for r in routes] == [2, 3]
    assert routes[1]['x_to_ttmn']['negative_contacts'] == 4
    assert routes[1]['gf_to_x']['positive_contacts'] == 4


@pytest.mark.parametrize('bad', ['missing', 'mismatch', 'zero'])
def test_arrow_evidence_cannot_be_missing_or_disagree(bad):
    a, sets, labels, counts = toy()
    if bad == 'missing':
        del counts[0, 1]
    else:
        counts[0, 1] = 0 if bad == 'zero' else 5
    with pytest.raises(ValueError):
        inv.inventory(a, a['weight'], sets, labels, counts, 0.275)


def test_pin_changes_are_refused(tmp_path):
    p = tmp_path / 'input'
    p.write_bytes(b'frozen')
    digest = inv.sha(p)
    assert inv.verify_pin(p, digest) == digest
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity mismatch'):
        inv.verify_pin(p, digest)
