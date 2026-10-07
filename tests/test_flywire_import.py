"""FlyWire 783 (female) import (card F2): a separate, labelled dataset.

Everything here runs on small in-test fixtures shaped like the public files;
no FlyWire data is committed.  The real-data check runs only when
NEUROFLY_FLYWIRE_GRAPH_DIR and NEUROFLY_FLYWIRE_CONNECTOME_DIR are set.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from brainlab import flywire
from brainlab.brain import Brain
from brainlab.graph_identity import GraphUnavailable, verify_graph
from experiment_registry import ExperimentRegistry, IncompatibleCheckpoint, SharedGraph

ROOT = Path(__file__).resolve().parents[1]
BASE = 720575940600000000          # FlyWire-sized root IDs: far above 2**53
COLUMNS = ['supervoxel_id', 'root_id', 'flow', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type',
           'side', 'top_nt', 'top_nt_conf', 'known_nt', 'known_nt_source']


# ---------------------------------------------------------------------------
# known_nt parser and sign policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('raw, label, status', [
    ('histamine', 'histamine', 'classical_positive'),
    ('acetylcholine; acetylcholine', 'acetylcholine', 'classical_positive'),
    ('gaba-negative', None, 'negative_only'),
    ('acetylcholine-negative, glutamate-negative, gaba-negative, dopamine-negative', None, 'negative_only'),
    ('acetylcholine; gaba-negative', 'acetylcholine', 'classical_positive'),
    ('glutamate, gaba; glutamate', 'gaba,glutamate', 'classical_positive'),
    ('histamine; acetylcholine, histamine', 'acetylcholine,histamine', 'classical_positive'),
    ('acetylcholine; sNPF; acetylcholine, sNPF', 'acetylcholine', 'classical_positive'),
    ('allatostatin-a', None, 'non_classical_only'),
    ('octopamine; glutamate, octopamine-negative', 'glutamate', 'classical_positive'),
    ('octopamine; octopamine-negative', None, 'conflicted_only'),
    ('', None, 'empty'),
    (None, None, 'empty'),
])
def test_known_nt_parser(raw, label, status):
    parsed = flywire.parse_known_nt(raw)
    assert parsed.label == label and parsed.status == status


def test_negative_label_is_never_read_as_inhibitory():
    parsed = flywire.parse_known_nt('gaba-negative, glutamate-negative')
    assert parsed.negative == {'gaba', 'glutamate'} and not parsed.positive
    # The sign comes from the top_nt fallback, whatever it is, never from "negative".
    assert flywire.resolve_transmitter('gaba-negative', 'acetylcholine') == (
        'acetylcholine', 'top_nt_fallback:negative_only')
    assert flywire.resolve_transmitter('gaba-negative', '') == (None, 'unknown:negative_only+top_nt_missing')
    assert flywire.parse_known_nt('negative').unspecified_negative
    assert flywire.parse_known_nt('octopamine; octopamine-negative').conflicted == {'octopamine'}


def test_known_first_top_fallback_and_existing_sign_rule():
    # Photoreceptors: known histamine wins over a predicted acetylcholine.
    assert flywire.resolve_transmitter('histamine', 'acetylcholine') == ('histamine', 'known_nt')
    assert flywire._sign_of('histamine') == (-1, False)
    assert flywire.resolve_transmitter('', 'gaba') == ('gaba', 'top_nt_fallback:empty')
    assert flywire._sign_of('gaba,glutamate') == (-1, False)
    assert flywire._sign_of('acetylcholine,histamine') == (1, True)      # conflicting: declared +1
    assert flywire._sign_of('dopamine') == (1, True)                     # modulator-only: declared +1
    assert flywire._sign_of(None) == (1, True)


def test_method_identity_records_policy_and_scale():
    assert flywire.METHOD['sign_policy'] == flywire.SIGN_POLICY
    assert flywire.METHOD['synaptic_scale'] == 0.275
    assert 'NOT mean the datasets are calibrated' in flywire.METHOD['calibration_note']
    assert flywire.METHOD['fitting'] is None


# ---------------------------------------------------------------------------
# Fixture: a tiny FlyWire-shaped source set
# ---------------------------------------------------------------------------
def _rows():
    """Six proofread roots; one isolated (id 6), one connected but unannotated (id 5),
    one annotation row outside the proofread universe (id 9)."""
    ann = [
        (1, 'R1-6', 'right', 'sensory', 'acetylcholine', 'histamine'),
        (2, 'L1', 'right', 'optic', 'gaba', 'glutamate, gaba; glutamate'),
        (3, 'Mi1', 'right', 'optic', 'acetylcholine', 'acetylcholine; gaba-negative'),
        (4, 'X', 'left', 'central', 'dopamine', 'acetylcholine-negative, gaba-negative'),
        (6, 'Iso', 'left', 'sensory', 'acetylcholine', ''),
        (9, 'Ghost', 'left', 'central', 'gaba', 'gaba'),
    ]
    connections = [  # (pre, post, neuropil, syn_count): pairs repeat across neuropils
        (1, 2, 'LA_R', 30), (1, 2, 'ME_R', 5), (1, 3, 'LA_R', 4),
        (2, 3, 'ME_R', 7), (3, 4, 'ME_R', 2), (3, 4, 'LO_R', 1), (3, 4, 'LOP_R', 9),
        (5, 1, 'LA_R', 3), (4, 5, 'SMP_L', 6),
    ]
    return ann, connections


def make_sources(directory: Path) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    directory.mkdir(parents=True, exist_ok=True)
    ann, connections = _rows()
    roots = np.array([BASE + k for k in (6, 3, 1, 5, 2, 4)], dtype=np.uint64)
    np.save(directory / flywire.ROOT_IDS, roots)
    frame = pd.DataFrame([dict(supervoxel_id='0', root_id=str(BASE + k), flow='intrinsic', super_class=sc,
                               cell_class='', cell_sub_class='', cell_type=t, side=side, top_nt=top,
                               top_nt_conf='0.5', known_nt=known, known_nt_source='')
                          for k, t, side, sc, top, known in ann], columns=COLUMNS)
    frame.to_csv(directory / flywire.ANNOTATIONS, sep='\t', index=False)
    table = pa.table({
        'pre_pt_root_id': pa.array([BASE + c[0] for c in connections], pa.int64()),
        'post_pt_root_id': pa.array([BASE + c[1] for c in connections], pa.int64()),
        'neuropil': pa.array([c[2] for c in connections]),
        'syn_count': pa.array([c[3] for c in connections], pa.int64()),
    })
    feather.write_feather(table, directory / flywire.CONNECTIONS, chunksize=4)
    files = {}
    for name in (flywire.ROOT_IDS, flywire.ANNOTATIONS, flywire.CONNECTIONS):
        data = (directory / name).read_bytes()
        files[name] = dict(url='fixture', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    real = flywire.load_lock()
    return dict(files=files, annotations=real['annotations'], licence_gate=real['licence_gate'])


@pytest.fixture
def built(tmp_path):
    src, cdir, gdir = tmp_path / 'src', tmp_path / 'connectome', tmp_path / 'graph'
    lock = make_sources(src)
    report = flywire.import_flywire(src, cdir, gdir, lock=lock)
    pins = flywire.pins_from_build(gdir, cdir, lock=lock)
    return dict(src=src, cdir=cdir, gdir=gdir, lock=lock, report=report, pins=pins, tmp=tmp_path)


def test_sources_are_byte_verified(tmp_path):
    lock = make_sources(tmp_path / 'src')
    flywire.verify_sources(tmp_path / 'src', lock)
    with (tmp_path / 'src' / flywire.ANNOTATIONS).open('a') as stream:
        stream.write('x')
    with pytest.raises(flywire.FlyWireUnavailable):
        flywire.verify_sources(tmp_path / 'src', lock)


def test_counts_are_conserved_across_neuropil_rows(built):
    edges = built['report']['edges']
    total = sum(c[3] for c in _rows()[1])
    assert edges['source_rows'] == 9 and edges['source_synapses'] == total
    assert edges['aggregated_synapses'] == edges['retained_synapses'] == total
    assert edges['excluded_rows'] == edges['excluded_synapses'] == 0
    assert edges['aggregated_pairs'] == 6 and edges['rows_merged_into_existing_pairs'] == 3
    assert edges['pairs_spanning_multiple_neuropils'] == 2 and edges['max_neuropil_rows_per_pair'] == 3
    assert edges['incoming_equals_outgoing_total']


def test_rows_outside_the_universe_are_excluded_and_counted():
    ids = np.array([BASE + 1, BASE + 2], dtype=np.uint64)
    batch = (np.array([BASE + 1, BASE + 1, BASE + 7], np.int64), np.array([BASE + 2, BASE + 2, BASE + 1], np.int64),
             np.array([3, 4, 5]), np.array(['A', 'B', 'A'], dtype=object))
    pre, post, count, stats = flywire.aggregate_edges(ids, [batch])
    assert stats['excluded_rows'] == 1 and stats['excluded_synapses'] == 5
    assert list(count) == [7] and stats['source_synapses'] == 12


def test_whole_universe_kept_with_explicit_unknowns(built):
    import pyarrow.feather as feather
    nodes = feather.read_table(built['cdir'] / 'normalized/neurons.feather').to_pandas()
    rep = built['report']['nodes']
    assert len(nodes) == rep['nodes'] == 6
    assert rep['isolated_nodes'] == 1 and rep['unannotated_connected_nodes'] == 1
    assert rep['annotation_rows_outside_universe'] == 1
    by_id = nodes.set_index('source_id')
    iso = by_id.loc[np.uint64(BASE + 6)]
    assert not iso.connected and iso.annotation_status == 'annotated'
    unk = by_id.loc[np.uint64(BASE + 5)]
    assert unk.connected and unk.annotation_status == 'no_annotation_row'
    assert unk.neurotransmitter_source == 'unknown:no_annotation_row' and bool(unk.sign_ambiguous)
    assert np.uint64(BASE + 9) not in by_id.index


def test_ids_are_lossless_64_bit(built):
    with np.load(built['gdir'] / 'graph.npz') as data:
        ids = data['ids']
    expected = np.array(sorted(BASE + k for k in (1, 2, 3, 4, 5, 6)), dtype=np.int64)
    assert ids.dtype == np.int64 and np.array_equal(ids, expected)
    assert len(np.unique(ids.astype(np.float64))) < len(ids)   # float64 WOULD have collided


def test_representative_signs_and_weights(built):
    import pyarrow.feather as feather
    nodes = feather.read_table(built['cdir'] / 'normalized/neurons.feather').to_pandas().set_index('cell_type')
    assert (nodes.loc['R1-6', 'neurotransmitter'], nodes.loc['R1-6', 'sign']) == ('histamine', -1)
    assert (nodes.loc['L1', 'neurotransmitter'], nodes.loc['L1', 'sign']) == ('gaba,glutamate', -1)
    assert (nodes.loc['Mi1', 'neurotransmitter'], nodes.loc['Mi1', 'sign']) == ('acetylcholine', 1)
    # negative-only known label: label from the top_nt fallback (dopamine), not from "negative"
    assert nodes.loc['X', 'neurotransmitter_source'] == 'top_nt_fallback:negative_only'
    assert (nodes.loc['X', 'neurotransmitter'], nodes.loc['X', 'sign']) == ('dopamine', 1)
    with np.load(built['gdir'] / 'graph.npz') as data:
        ptr, post, weight, ids = data['ptr'], data['post'], data['weight'], data['ids']
    index = {int(v) - BASE: i for i, v in enumerate(ids)}
    edge = {(index_of, int(post[e])): float(weight[e])
            for index_of in range(len(ids)) for e in range(ptr[index_of], ptr[index_of + 1])}
    assert edge[(index[1], index[2])] == pytest.approx(35 * -1 * 0.275)       # LA_R 30 + ME_R 5, histamine
    assert edge[(index[3], index[4])] == pytest.approx(12 * 1 * 0.275)        # three neuropil rows summed
    Brain(arrays={k: np.ascontiguousarray(v) for k, v in
                  dict(ptr=ptr, post=post, weight=weight, ids=ids).items()}, dynamics='v1', backend='cpu')
    sign = built['report']['sign']
    assert sign['negative_only_cells'] == 1 and sign['unknown_transmitter'] == 1
    assert sign['known_vs_top_sign_disagreements'] >= 1                    # R1-6 histamine vs top_nt ACh


def test_verify_against_own_pins_and_labels(built):
    identity = flywire.verify_flywire_graph(built['gdir'], built['cdir'], pins=built['pins'])
    assert identity.dataset == 'flywire_783_female' and not identity.synthetic
    assert 'FlyWire 783 (female)' in identity.label and 'no VNC' in identity.label
    assert '0.275' in identity.label and 'NOT an equivalent calibration' in identity.label
    manifest = json.loads((built['gdir'] / 'manifest.json').read_text())
    assert manifest['dataset'] == 'FlyWire 783 (female)' and manifest['io'] == 'unavailable'
    bad = dict(built['pins'], graph_content_sha256='0' * 64)
    with pytest.raises(GraphUnavailable, match='content hash mismatch'):
        flywire.verify_flywire_graph(built['gdir'], built['cdir'], pins=bad)
    with pytest.raises(GraphUnavailable, match='method identity'):
        flywire.verify_flywire_graph(built['gdir'], built['cdir'], pins=dict(built['pins'], method_sha256='1' * 64))


def test_malecns_runtime_refuses_the_flywire_graph_by_name(built):
    with pytest.raises(GraphUnavailable, match="'flywire_783_female' graph, not MaleCNS"):
        verify_graph(built['gdir'], built['cdir'])


def test_io_mappings_are_unavailable_never_substituted():
    assert flywire.IO_DECLARATION['status'] == 'unavailable'
    assert 'column' in flywire.IO_DECLARATION['mappings']['photoreceptor_column_encoder']
    with pytest.raises(GraphUnavailable, match='never substituted'):
        flywire.io_unavailable('descending_neuron_channels')


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*'))}


def _drive(instance, steps, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        currents = rng.uniform(20.0, 60.0, size=instance.shared.n).astype(np.float32)
        out.append(instance.step(currents, 5.0).counts.copy())
    return out


def test_same_dataset_resume_and_cross_dataset_refusal(built, monkeypatch):
    monkeypatch.setenv('NEUROFLY_LIF_DYNAMICS', 'v3')
    shared = flywire.load_shared(built['gdir'], built['cdir'], dynamics='v3', pins=built['pins'])
    assert shared.identity.dataset.startswith('flywire_783_female+v3')
    root = built['tmp'] / 'registry-flywire_783_female'
    kwargs = dict(brain_backend='cpu', learning_enabled=False)
    ref = ExperimentRegistry(shared, built['tmp'] / 'ref', **kwargs).activate('optomotor', 'connectome-fixed')
    expected = _drive(ref, 3, 1) + _drive(ref, 3, 2)
    registry = ExperimentRegistry(shared, root, **kwargs)
    first = registry.activate('optomotor', 'connectome-fixed')
    trace = _drive(first, 3, 1)
    registry.checkpoint()
    manifest = json.loads((registry.instance_dir(first.instance_id) / 'manifest.json').read_text())
    assert 'FlyWire 783 (female)' in manifest['graph']['label']
    resumed = ExperimentRegistry(shared, root, **kwargs).activate('optomotor', 'connectome-fixed')
    assert resumed.restored_from['step_index'] == 3
    trace += _drive(resumed, 3, 2)
    assert all(np.array_equal(a, b) for a, b in zip(expected, trace))
    # Another dataset's graph on this store: refused before any write.
    other = SharedGraph.synthetic(allow_synthetic=True, n=6, k_out=2, seed=3)
    before = _tree(root)
    with pytest.raises(IncompatibleCheckpoint, match='different graph'):
        ExperimentRegistry(other, root, test_mode=True, **kwargs)
    assert _tree(root) == before
    # And this dataset on another dataset's store.
    other_root = built['tmp'] / 'registry-other'
    ExperimentRegistry(other, other_root, test_mode=True, **kwargs).activate('optomotor', 'connectome-fixed')
    before = _tree(other_root)
    with pytest.raises(IncompatibleCheckpoint, match='different graph'):
        ExperimentRegistry(shared, other_root, **kwargs)
    assert _tree(other_root) == before


def test_separate_namespace_and_malecns_stays_default():
    registry = json.loads((ROOT / 'brainlab/datasets.json').read_text())['datasets']
    assert list(registry)[0] == 'malecns_v1'
    assert registry['flywire_783_female']['default'] is False
    assert registry['flywire_783_female']['coverage'] == 'brain_only_no_vnc'
    malecns_pins = json.loads((ROOT / 'brainlab/graph_pins.json').read_text())
    assert malecns_pins['dataset'] == 'MaleCNS v1.0'
    assert flywire.PINS_PATH.name != 'graph_pins.json'
    assert flywire.DEFAULT_GRAPH_DIR.name == flywire.DEFAULT_CONNECTOME_DIR.name == 'flywire_783_female'
    assert flywire.DEFAULT_REGISTRY_ROOT.name == 'registry-flywire_783_female'


def test_lock_pins_sources_and_keeps_the_licence_gate_open():
    lock = flywire.load_lock()
    assert lock['annotations']['release'] == 'v3.2.0'
    assert lock['annotations']['commit'] == 'a83b2776d60d5764cef36b927f5f9679c16c47a2'
    assert lock['files'][flywire.ANNOTATIONS]['git_blob_sha1'] == lock['annotations']['git_blob_sha1']
    assert lock['files'][flywire.CONNECTIONS]['md5'] == 'f48f972d262323a102aed49af1396b8a'
    assert lock['licence_gate']['annotation_table'] == 'UNKNOWN'
    assert 'No clearance is claimed' in lock['licence_gate']['status']
    pins = flywire.load_pins()
    assert {k: v['sha256'] for k, v in pins['source_sha256'].items()} == {
        k: v['sha256'] for k, v in lock['files'].items()}
    assert pins['method_sha256'] == flywire._sha256_json(flywire.METHOD)


def test_no_annotation_data_is_tracked():
    import subprocess
    try:
        tracked = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('not a git checkout')
    names = tracked.splitlines()
    assert not [n for n in names if n.endswith('.tsv') and 'flywire' in n.lower()]
    assert not [n for n in names if 'Supplemental_file1' in n or n.startswith('connectome_data/')]


@pytest.mark.skipif(not (os.environ.get(flywire.GRAPH_ENV) and os.environ.get(flywire.CONNECTOME_ENV)),
                    reason='real FlyWire build not configured')
def test_real_flywire_build_matches_pins():
    import pyarrow.feather as feather
    identity = flywire.verify_flywire_graph()
    assert (identity.neurons, identity.edges) == (139255, 15091983)
    nodes = feather.read_table(Path(os.environ[flywire.CONNECTOME_ENV]) / 'normalized/neurons.feather',
                               columns=['cell_type', 'sign', 'connected', 'annotation_status']).to_pandas()
    assert set(nodes[nodes.cell_type == 'R1-6'].sign) == {-1}
    assert set(nodes[nodes.cell_type == 'Mi4'].sign) == {-1}
    assert set(nodes[nodes.cell_type == 'T4a'].sign) == {1}
    assert int((~nodes.connected).sum()) == 616
    assert int((nodes.annotation_status == 'no_annotation_row').sum()) == 14
