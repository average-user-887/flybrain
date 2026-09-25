"""WP7 graph options D3 (KC->KC fast weight) and D5 (DPM transmitter).

docs/WP7_MB_LEARNING_SPEC.md §2.  Small synthetic graphs only; no connectome
download.  The defaults must reproduce the plain v3 policy byte for byte.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from brainlab import transmitter_policy as tp

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / 'validation/specs'
WP7_SPECS = ('mb_e0_kc_regime_v3.json', 'mb_e1_hige2015_v3.json')

# A toy mushroom body: 4 KCs, 2 MBONs, DPM, PPL101, APL, one unclear neuron.
TYPES = ['KCg-m', 'KCg-m', 'KCab-s', "KCa'b'-ap1", 'MBON11', 'MBON12', 'DPM', 'PPL101', 'APL', 'SMP001']
LABELS = ['acetylcholine'] * 4 + ['gaba', 'acetylcholine', 'dopamine', 'dopamine', 'gaba', 'unclear']
SIGN = {'acetylcholine': 1, 'gaba': -1, 'dopamine': 1, 'unclear': 1}   # the prepared graph's proxy
KC, MBON11, MBON12, DPM, PPL1, APL, UNCLEAR = [0, 1, 2, 3], 4, 5, 6, 7, 8, 9
EDGES = [  # (pre, post, synapse count)
    (0, 1, 7), (0, 2, 3), (0, 4, 12), (1, 0, 5), (1, 3, 9), (1, 5, 4), (2, 0, 2), (2, 4, 6), (3, 5, 8),
    (3, 8, 11),                              # KC -> APL
    (4, 5, 3), (5, 9, 2),                    # MBON -> MBON, MBON -> unclear
    (6, 0, 4), (6, 1, 6), (6, 3, 2),         # DPM -> KC
    (7, 0, 5), (7, 4, 9),                    # PPL101 -> KC, -> MBON11
    (8, 0, 10), (8, 2, 7),                   # APL -> KC
    (9, 1, 3), (9, 6, 2),                    # unclear -> KC, -> DPM
]


def toy_graph():
    """CSR arrays with prepared-graph weights: count(float32) * sign(int8) * 0.275."""
    n = len(TYPES)
    edges = sorted(EDGES)
    pre = np.array([e[0] for e in edges])
    post = np.array([e[1] for e in edges], dtype=np.int32)
    count = np.array([e[2] for e in edges])
    signs = np.array([SIGN[label] for label in LABELS], dtype=np.int8)
    weight = (count.astype(np.float32) * signs[pre] * .275).astype(np.float32)
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, pre + 1, 1)
    ptr = np.cumsum(ptr)
    return ptr, post, weight, count, pre


def edge_index(pre, post, a, b):
    hit = np.flatnonzero((pre == a) & (post == b))
    assert len(hit) == 1
    return int(hit[0])


def plain_v3_reference(ptr, post, weight, labels):
    """The v3 policy as it stood before WP7 (unclear_mode='excitatory')."""
    labels = tp.normalise(labels)
    new = weight.copy()
    modulatory = np.isin(labels.astype(str), tp.MODULATORY_TRANSMITTERS)
    for index in np.flatnonzero(modulatory):
        new[ptr[index]:ptr[index + 1]] = 0.0
    return new


# -- defaults are unchanged ---------------------------------------------------
def test_defaults_reproduce_plain_v3_byte_for_byte():
    ptr, post, weight, _, _ = toy_graph()
    reference = plain_v3_reference(ptr, post, weight, LABELS)
    new_default, report_default = tp.apply_policy(ptr, post, weight, LABELS)
    new_explicit, report_explicit = tp.apply_policy(ptr, post, weight, LABELS, kc_kc='released',
                                                    dpm='released', cell_types=TYPES)
    assert new_default.tobytes() == reference.tobytes()
    assert new_explicit.tobytes() == reference.tobytes()
    assert report_default == report_explicit
    assert report_default['weight_sha256'] == hashlib.sha256(reference.tobytes()).hexdigest()
    # No WP7 keys leak into a default report or declaration.
    for key in ('kc_kc', 'dpm', 'kc_kc_edges_zeroed'):
        assert key not in report_default
    assert tp.describe() == tp.describe(kc_kc='released', dpm='released')
    assert 'kc_kc' not in tp.describe()
    assert tp.graph_variant_tag() == ''


def test_defaults_unchanged_on_a_random_graph_for_every_unclear_mode():
    rng = np.random.default_rng(7)
    n, k = 300, 12
    ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    post = rng.integers(0, n, n * k).astype(np.int32)
    weight = (rng.integers(1, 40, n * k).astype(np.float32) * rng.choice([-1, 1], n * k).astype(np.int8)
              * .275).astype(np.float32)
    labels = rng.choice(['acetylcholine', 'gaba', 'glutamate', 'dopamine', 'serotonin', 'unclear', 'nan'], n)
    types = rng.choice(['KCg-m', 'KCab-c', 'DPM', 'MBON01', 'PAM01', ''], n)
    for mode in tp.UNCLEAR_MODES:
        a, ra = tp.apply_policy(ptr, post, weight, labels, unclear_mode=mode)
        b, rb = tp.apply_policy(ptr, post, weight, labels, unclear_mode=mode, cell_types=types)
        assert a.tobytes() == b.tobytes() and ra == rb
    np.testing.assert_array_equal(tp.apply_policy(ptr, post, weight, labels)[0],
                                  plain_v3_reference(ptr, post, weight, labels))


# -- D3: KC->KC ------------------------------------------------------------------
def test_kc_kc_modulatory_only_zeroes_exactly_the_kc_to_kc_edges():
    ptr, post, weight, _, pre = toy_graph()
    base, _ = tp.apply_policy(ptr, post, weight, LABELS)
    new, report = tp.apply_policy(ptr, post, weight, LABELS, kc_kc='modulatory-only', cell_types=TYPES)
    kc = np.isin(np.arange(len(TYPES)), KC)
    kc_kc = kc[pre] & kc[post]
    assert kc_kc.sum() == 5
    assert (new[kc_kc] == 0).all()
    np.testing.assert_array_equal(new[~kc_kc], base[~kc_kc])     # KC->MBON, KC->APL, APL->KC untouched
    assert new[edge_index(pre, post, 0, MBON11)] == pytest.approx(12 * .275)
    assert report['kc_kc_edges'] == 5 and report['kc_kc_edges_zeroed'] == 5 and report['kc_neurons'] == 4
    assert report['kc_kc'] == 'modulatory-only' and report['dpm'] == 'released'


def test_kc_kc_except_calyx_keeps_declared_calyx_synapses_with_prepare_arithmetic():
    ptr, post, weight, count, pre = toy_graph()
    e01, e13, e02 = edge_index(pre, post, 0, 1), edge_index(pre, post, 1, 3), edge_index(pre, post, 0, 2)
    calyx = (np.array([e01, e13]), np.array([2, 9]))              # e13: all 9 synapses in the calyx
    new, report = tp.apply_policy(ptr, post, weight, LABELS, kc_kc='modulatory-only-except-calyx',
                                  cell_types=TYPES, kc_kc_calyx=calyx)
    assert new[e01] == np.float32(np.float32(2) * np.int8(1) * .275)
    assert new[e13] == weight[e13]                                # full count reproduces the released weight
    assert new[e02] == 0.0                                        # KC->KC edge with no calyx synapse
    assert report['kc_kc_calyx_edges_kept'] == 2 and report['kc_kc_calyx_synapses_kept'] == 11
    assert report['kc_kc_edges_zeroed'] == 3


@pytest.mark.parametrize('calyx, message', [
    ((np.array([0]), np.array([1, 2])), 'each edge once'),
    ((np.array([0, 0]), np.array([1, 1])), 'each edge once'),
    (None, 'kc_kc_calyx is required'),
])
def test_kc_kc_calyx_input_is_checked(calyx, message):
    ptr, post, weight, _, _ = toy_graph()
    with pytest.raises(ValueError, match=message):
        tp.apply_policy(ptr, post, weight, LABELS, kc_kc='modulatory-only-except-calyx',
                        cell_types=TYPES, kc_kc_calyx=calyx)


def test_kc_kc_calyx_refuses_non_kc_edges_and_counts_above_released():
    ptr, post, weight, _, pre = toy_graph()
    kc_mbon = edge_index(pre, post, 0, MBON11)
    with pytest.raises(ValueError, match='not KC->KC'):
        tp.apply_policy(ptr, post, weight, LABELS, kc_kc='modulatory-only-except-calyx', cell_types=TYPES,
                        kc_kc_calyx=(np.array([kc_mbon]), np.array([1])))
    e01 = edge_index(pre, post, 0, 1)
    with pytest.raises(ValueError, match='exceeds the released'):
        tp.apply_policy(ptr, post, weight, LABELS, kc_kc='modulatory-only-except-calyx', cell_types=TYPES,
                        kc_kc_calyx=(np.array([e01]), np.array([8])))
    with pytest.raises(ValueError, match='only by'):
        tp.apply_policy(ptr, post, weight, LABELS, kc_kc='modulatory-only', cell_types=TYPES,
                        kc_kc_calyx=(np.array([e01]), np.array([1])))


def test_calyx_pairs_map_to_edges_and_missing_pairs_fail(tmp_path):
    ptr, post, weight, _, pre = toy_graph()
    edges, counts = tp.calyx_edges_from_pairs(ptr, post, [1, 0], [3, 1], [4, 2])
    assert list(edges) == [edge_index(pre, post, 0, 1), edge_index(pre, post, 1, 3)]
    assert list(counts) == [2, 4]
    with pytest.raises(ValueError, match='not edges'):
        tp.calyx_edges_from_pairs(ptr, post, [0], [3], [1])
    path = tmp_path / 'calyx.npz'
    np.savez(path, pre_index=np.array([0, 1]), post_index=np.array([1, 3]), calyx_synapses=np.array([2, 4]))
    e, c, sha = tp.load_kc_kc_calyx(path, ptr, post)
    assert list(c) == [2, 4] and sha == hashlib.sha256(path.read_bytes()).hexdigest()


# -- D5: DPM ---------------------------------------------------------------------
def test_dpm_released_is_silent_and_gaba_is_inhibitory_with_released_counts():
    ptr, post, weight, count, pre = toy_graph()
    rows = pre == DPM
    silent, _ = tp.apply_policy(ptr, post, weight, LABELS)
    assert (silent[rows] == 0).all()
    gaba, report = tp.apply_policy(ptr, post, weight, LABELS, dpm='gaba', cell_types=TYPES)
    np.testing.assert_array_equal(gaba[rows], (count[rows].astype(np.float32) * np.int8(-1) * .275)
                                  .astype(np.float32))
    np.testing.assert_array_equal(gaba[~rows], silent[~rows])   # PPL101 stays modulatory-only
    assert (gaba[pre == PPL1] == 0).all()
    assert report['dpm_neurons'] == 1 and report['dpm_out_edges_relabelled'] == 3
    assert report['dpm_released_labels'] == ['dopamine']
    assert report['neurons_modulatory'] == 1                     # only PPL101 now


def test_unclear_exclude_still_wins_over_the_wp7_options():
    ptr, post, weight, _, pre = toy_graph()
    new, _ = tp.apply_policy(ptr, post, weight, LABELS, unclear_mode='exclude', dpm='gaba', cell_types=TYPES)
    assert new[edge_index(pre, post, UNCLEAR, DPM)] == 0.0       # out of the unclear neuron
    assert new[edge_index(pre, post, MBON12, UNCLEAR)] == 0.0    # onto the unclear neuron
    assert new[edge_index(pre, post, DPM, 0)] < 0


def test_options_are_declared_and_v3_only():
    ptr, post, weight, _, _ = toy_graph()
    with pytest.raises(ValueError, match='kc_kc mode'):
        tp.apply_policy(ptr, post, weight, LABELS, kc_kc='zero', cell_types=TYPES)
    with pytest.raises(ValueError, match='dpm mode'):
        tp.apply_policy(ptr, post, weight, LABELS, dpm='serotonin', cell_types=TYPES)
    with pytest.raises(ValueError, match='only for the v3 policy'):
        tp.apply_policy(ptr, post, weight, LABELS, policy=tp.POLICY_LEGACY, dpm='gaba', cell_types=TYPES)
    with pytest.raises(ValueError, match='cell type per neuron'):
        tp.apply_policy(ptr, post, weight, LABELS, dpm='gaba')
    with pytest.raises(ValueError, match='cell type per neuron'):
        tp.apply_policy(ptr, post, weight, LABELS, dpm='gaba', cell_types=TYPES[:-1])


# -- graph identity ----------------------------------------------------------------
def _shared():
    from brainlab.graph_identity import synthetic_test_graph
    from experiment_registry import SharedGraph
    ptr, post, weight, _, _ = toy_graph()
    _, identity, io_map = synthetic_test_graph(n=len(TYPES), k_out=1)
    arrays = dict(ptr=ptr, post=post, weight=weight, ids=np.arange(1, len(TYPES) + 1, dtype=np.int64))
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    identity = dataclasses.replace(identity, dataset='toy-mb', graph_sha256=digest.hexdigest(),
                                   neurons=len(TYPES), edges=len(weight))
    return SharedGraph(arrays, identity, io_map)


def test_variants_get_distinct_graph_identities_and_defaults_keep_theirs(tmp_path):
    shared = _shared()
    default, _ = tp.apply_to_shared(shared, transmitters=LABELS)
    explicit, _ = tp.apply_to_shared(shared, transmitters=LABELS, cell_types=TYPES)
    assert default.identity == explicit.identity
    assert default.identity.dataset == 'toy-mb+v3-modulatory-only(excitatory)'
    assert 'WP7' not in default.identity.label
    path = tmp_path / 'calyx.npz'
    np.savez(path, pre_index=np.array([0]), post_index=np.array([1]), calyx_synapses=np.array([2]))
    variants = {
        'kc0': dict(kc_kc='modulatory-only'),
        'kc0_dpm_gaba': dict(kc_kc='modulatory-only', dpm='gaba'),
        'dpm_gaba': dict(dpm='gaba'),
        'calyx_dpm_gaba': dict(kc_kc='modulatory-only-except-calyx', dpm='gaba', kc_kc_calyx_file=path),
    }
    hashes = {default.identity.graph_sha256}
    for name, kwargs in variants.items():
        graph, report = tp.apply_to_shared(shared, transmitters=LABELS, cell_types=TYPES, **kwargs)
        ident = graph.identity
        assert 'v3-modulatory-only' in ident.dataset          # GraphInstance's v3 check still passes
        assert f"kc_kc={kwargs['kc_kc'] if 'kc_kc' in kwargs else 'released'}" in ident.dataset
        assert report['graph_sha256'] == ident.graph_sha256
        hashes.add(ident.graph_sha256)
        assert np.array_equal(graph.arrays['ptr'], shared.arrays['ptr'])
        assert np.array_equal(graph.arrays['post'], shared.arrays['post'])
    assert len(hashes) == 1 + len(variants)
    graph, report = tp.apply_to_shared(shared, transmitters=LABELS, cell_types=TYPES, **variants['calyx_dpm_gaba'])
    assert report['kc_kc_calyx_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['kc_kc_calyx_sha256'][:12] in graph.identity.dataset


# -- the WP7 validation specs ------------------------------------------------------
@pytest.mark.parametrize('name', WP7_SPECS)
def test_wp7_specs_load_and_declare_only_valid_graph_variants(name):
    from validation import harness
    spec = harness.load_spec(SPECS / name)
    bounds = harness.load_bounds(spec)
    assert spec['status'] == 'draft' and spec['freeze']['state'] == 'frozen-pending-owner-gate'
    for check in spec['physiology']['checks'] + spec['physiology'].get('reported', []):
        assert check['bound'] in bounds['bounds'], check
    dyn = spec['dynamics']
    assert (dyn['kc_kc'], dyn['dpm']) == ('modulatory-only', 'gaba')      # the P4 primary graph
    for variant in spec.get('graph_variants', {}):
        d = harness.apply_graph_variant(spec, variant)['dynamics']
        assert d.get('kc_kc', 'released') in tp.KC_KC_MODES and d.get('dpm', 'released') in tp.DPM_MODES
    with pytest.raises(harness.SpecError):
        harness.apply_graph_variant(spec, 'undeclared')
    with pytest.raises(harness.SpecError, match='not implemented'):
        harness.paradigm_module(spec['paradigm'])


def test_wp7_seed_sets_are_disjoint():
    e0 = json.loads((SPECS / WP7_SPECS[0]).read_text())
    e1 = json.loads((SPECS / WP7_SPECS[1]).read_text())
    calib, confirm = set(e1['calibration']['seeds']), set(e1['seeds'])
    parity = set(e1['calibration']['cpu_gpu_parity_seeds'])
    assert len(calib) == 5 and len(confirm) == 10
    assert not calib & confirm and not calib & set(e0['seeds']) and not confirm & set(e0['seeds'])
    assert parity <= confirm


def _frozen_content_sha256(spec):
    """The spec's freeze rule: everything except status, status_reason, freeze and calibration.result."""
    body = {k: v for k, v in spec.items() if k not in ('status', 'status_reason', 'freeze')}
    if 'calibration' in body:
        body['calibration'] = {k: v for k, v in body['calibration'].items() if k != 'result'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


@pytest.mark.parametrize('name', WP7_SPECS)
def test_wp7_specs_are_frozen(name):
    """Editing a frozen field fails here; a change needs a new spec id (freeze.rule)."""
    spec = json.loads((SPECS / name).read_text())
    assert _frozen_content_sha256(spec) == spec['freeze']['frozen_content_sha256']
    encoder = spec['encoder']
    assert hashlib.sha256((ROOT / encoder['from_spec']).read_bytes()).hexdigest() == encoder['from_spec_sha256']
