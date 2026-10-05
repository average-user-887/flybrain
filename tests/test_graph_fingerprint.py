"""Content fingerprints of the locally generated graph files.

neurons.feather and graph.npz are written on the user's machine, so their bytes
carry the pandas/pyarrow versions and the host OS.  verify_graph pins their
CONTENT instead.  These tests hold both halves of that contract: identical data
written differently is accepted, and any genuine content change (one value, one
row, column order, dtype, name, nullness) is refused.
"""
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from brainlab.graph_identity import (GraphUnavailable, arrays_content_sha256, feather_content_sha256,
                                     load_pins, npz_content_sha256, resolve_connectome_dir,
                                     resolve_graph_dir, table_content_sha256, verify_graph)


def _table():
    return pa.table({
        'node_index': pa.array([0, 1, 2, 3], pa.uint32()),
        'source_id': pa.array([10, 11, 12, 2**63 + 5], pa.uint64()),
        'retained': pa.array([True, True, False, True]),
        'weight': pa.array([0.5, float('nan'), -0.0, 2.75], pa.float32()),
        'cell_type': pa.array(['DNa02', None, '', 'MDN'], pa.large_string()),
        'neurotransmitter': pa.array(['acetylcholine', 'gaba', None, 'ünïcode'], pa.large_string()),
    })


def _write(table, path, **kwargs):
    feather.write_feather(table, path, **kwargs)
    return path


# ---------------------------------------------------------------- table scheme

def test_table_fingerprint_ignores_how_the_file_was_written(tmp_path):
    table = _table()
    reference = table_content_sha256(table)
    variants = {
        'plain': table,
        'pandas-3.0.6-metadata': table.replace_schema_metadata(
            {b'pandas': b'{"pandas_version": "3.0.6", "creator": {"library": "pyarrow", "version": "99"}}'}),
        'no-metadata': table.replace_schema_metadata(None),
        'pandas-2-string-offsets': table.cast(pa.schema(
            [pa.field(f.name, pa.string() if f.type == pa.large_string() else f.type) for f in table.schema])),
        'rechunked': pa.Table.from_batches(table.to_batches(max_chunksize=1)),
    }
    for name, variant in variants.items():
        for compression in ('uncompressed', 'lz4', 'zstd'):
            path = _write(variant, tmp_path / f'{name}-{compression}.feather', compression=compression)
            assert feather_content_sha256(path) == reference, (name, compression)
    # Bytes do differ, which is the whole reason for the content pin.
    hashes = {Path(p).read_bytes() for p in tmp_path.iterdir()}
    assert len(hashes) > 1


def test_table_fingerprint_matches_a_pandas_round_trip(tmp_path):
    # pandas turns a float NaN into an Arrow null on write (a real content
    # change, which the fingerprint must see), so the round trip uses no floats.
    table = _table().drop_columns(['weight'])
    table.to_pandas().to_feather(tmp_path / 'pandas.feather')
    back = feather.read_table(tmp_path / 'pandas.feather')
    assert table_content_sha256(back) == table_content_sha256(table)


def _mutations(table):
    cols = table.column_names
    idx = cols.index('cell_type')
    yield 'one value', table.set_column(idx, 'cell_type', pa.array(['DNa02', None, '', 'MDX'], pa.large_string()))
    yield 'empty string to null', table.set_column(idx, 'cell_type', pa.array(['DNa02', None, None, 'MDN'], pa.large_string()))
    yield 'one numeric value', table.set_column(1, 'source_id', pa.array([10, 11, 13, 2**63 + 5], pa.uint64()))
    yield 'nan to number', table.set_column(3, 'weight', pa.array([0.5, 0.0, -0.0, 2.75], pa.float32()))
    yield 'negative zero', table.set_column(3, 'weight', pa.array([0.5, float('nan'), 0.0, 2.75], pa.float32()))
    yield 'one row dropped', table.slice(0, 3)
    yield 'one row added', pa.concat_tables([table, table.slice(3, 1)])
    yield 'rows reordered', table.take([1, 0, 2, 3])
    yield 'column order', table.select([cols[1], cols[0]] + cols[2:])
    yield 'column renamed', table.rename_columns(['node_idx'] + cols[1:])
    yield 'dtype uint32->int64', table.set_column(0, 'node_index', table['node_index'].cast(pa.int64()))
    yield 'dtype float32->float64', table.set_column(3, 'weight', table['weight'].cast(pa.float64()))
    yield 'column dropped', table.drop_columns(['retained'])
    yield 'string boundary moved', table.set_column(
        cols.index('neurotransmitter'), 'neurotransmitter',
        pa.array(['acetylcholin', 'egaba', None, 'ünïcode'], pa.large_string()))


@pytest.mark.parametrize('label', [label for label, _ in _mutations(_table())])
def test_table_fingerprint_refuses_every_content_change(label):
    mutated = dict(_mutations(_table()))[label]
    assert table_content_sha256(mutated) != table_content_sha256(_table())


def test_table_fingerprint_refuses_undefined_types():
    table = pa.table({'x': pa.array(['a', 'b', 'a']).dictionary_encode()})
    with pytest.raises(ValueError, match='not defined'):
        table_content_sha256(table)


# --------------------------------------------------------------- arrays scheme

def _arrays():
    rng = np.random.default_rng(0)
    return dict(ptr=np.array([0, 2, 3, 5], np.int64), post=np.array([1, 2, 0, 0, 1], np.int32),
                weight=rng.standard_normal(5).astype(np.float32), ids=np.array([7, 8, 9], np.int64))


def test_npz_fingerprint_ignores_zip_and_npy_container(tmp_path, monkeypatch):
    arrays = _arrays()
    reference = arrays_content_sha256(arrays)
    np.savez(tmp_path / 'plain.npz', **arrays)
    np.savez_compressed(tmp_path / 'compressed.npz', **arrays)
    np.savez(tmp_path / 'reordered.npz', **dict(reversed(list(arrays.items()))))
    np.savez(tmp_path / 'fortran.npz', **{k: np.asfortranarray(v) for k, v in arrays.items()})
    np.savez(tmp_path / 'bigendian.npz', **{k: v.astype(v.dtype.newbyteorder('>')) for k, v in arrays.items()})
    init = zipfile.ZipInfo.__init__

    def windows_stamp(self, *a, **k):  # native Windows CPython writes create_system 0
        init(self, *a, **k)
        self.create_system = 0
    monkeypatch.setattr(zipfile.ZipInfo, '__init__', windows_stamp)
    np.savez(tmp_path / 'windows.npz', **arrays)
    monkeypatch.undo()
    assert (tmp_path / 'windows.npz').read_bytes() != (tmp_path / 'plain.npz').read_bytes()
    for path in tmp_path.iterdir():
        assert npz_content_sha256(path) == reference, path.name


@pytest.mark.parametrize('label', ['one value', 'one row', 'dtype', 'extra member', 'missing member',
                                   'renamed member', 'shape'])
def test_npz_fingerprint_refuses_every_content_change(label):
    arrays = _arrays()
    reference = arrays_content_sha256(arrays)
    if label == 'one value':
        arrays['weight'] = arrays['weight'].copy()
        arrays['weight'][2] = np.nextafter(arrays['weight'][2], np.float32(np.inf))
    elif label == 'one row':
        arrays['post'] = arrays['post'][:-1]
    elif label == 'dtype':
        arrays['post'] = arrays['post'].astype(np.int64)
    elif label == 'extra member':
        arrays['extra'] = np.zeros(1, np.int8)
    elif label == 'missing member':
        del arrays['ids']
    elif label == 'renamed member':
        arrays['idz'] = arrays.pop('ids')
    elif label == 'shape':
        arrays['post'] = arrays['post'].reshape(5, 1)
    assert arrays_content_sha256(arrays) != reference


def test_npz_fingerprint_refuses_object_members(tmp_path):
    with pytest.raises(ValueError, match='not defined'):
        arrays_content_sha256({'x': np.array(['a'], dtype=object)})


# ------------------------------------------------------- against the real pins

def _real_dirs():
    gdir, _ = resolve_graph_dir()
    cdir, _ = resolve_connectome_dir()
    if not (gdir / 'graph.npz').is_file() or not (cdir / 'normalized/neurons.feather').is_file():
        pytest.skip('real MaleCNS graph not available')
    return gdir, cdir


def _scratch_connectome(tmp_path, cdir, table, **write_kwargs):
    scratch = tmp_path / 'connectome'
    (scratch / 'normalized').mkdir(parents=True)
    for name in ('annotations.feather', 'neurotransmitters.feather'):
        os.symlink((cdir / name).resolve(), scratch / name)
    feather.write_feather(table, scratch / 'normalized/neurons.feather', **write_kwargs)
    return scratch


def test_pins_match_the_real_files():
    gdir, cdir = _real_dirs()
    pins = load_pins()
    assert npz_content_sha256(gdir / 'graph.npz') == pins['graph_content_sha256']
    assert feather_content_sha256(cdir / 'normalized/neurons.feather') == pins['neuron_map_content_sha256']


def test_real_neuron_map_rewritten_with_other_metadata_is_accepted(tmp_path):
    gdir, cdir = _real_dirs()
    pins = load_pins()
    table = feather.read_table(cdir / 'normalized/neurons.feather')
    meta = dict(table.schema.metadata)
    meta[b'pandas'] = meta[b'pandas'].replace(b'"pandas_version": "3.0.5"', b'"pandas_version": "3.0.6"')
    rewritten = table.replace_schema_metadata(meta)
    scratch = _scratch_connectome(tmp_path, cdir, rewritten)
    path = scratch / 'normalized/neurons.feather'
    from brainlab.graph_identity import sha256_file
    assert sha256_file(path) != pins['neuron_map_sha256'], 'the rewrite must change the bytes'
    identity = verify_graph(gdir, scratch)
    assert identity.neuron_map_content_sha256 == pins['neuron_map_content_sha256']
    assert identity.neuron_map_sha256 == pins['neuron_map_sha256']  # reference identifier
    assert identity.neuron_map_file_sha256 == sha256_file(path)
    # pandas 2.x writes `string` instead of `large_string`; still the same content.
    small = table.cast(pa.schema([pa.field(f.name, pa.string() if f.type == pa.large_string() else f.type)
                                  for f in table.schema]))
    shutil.rmtree(scratch)
    scratch = _scratch_connectome(tmp_path, cdir, small, compression='uncompressed')
    assert verify_graph(gdir, scratch).neurons == pins['neurons']


def _real_mutations(table):
    cols = table.column_names
    cell = cols.index('cell_type')
    values = table['cell_type'].to_pylist()
    values[332] = 'DNa01'
    yield 'one value', table.set_column(cell, 'cell_type', pa.array(values, table['cell_type'].type))
    nt = cols.index('neurotransmitter')
    values = table['neurotransmitter'].to_pylist()
    values[100] = 'gaba' if values[100] != 'gaba' else 'glutamate'
    yield 'one transmitter', table.set_column(nt, 'neurotransmitter', pa.array(values, table['neurotransmitter'].type))
    yield 'one row dropped', table.slice(0, table.num_rows - 1)
    yield 'column order', table.select([cols[1], cols[0]] + cols[2:])
    yield 'dtype', table.set_column(0, 'node_index', table['node_index'].cast(pa.int64()))


@pytest.mark.parametrize('label', ['one value', 'one transmitter', 'one row dropped', 'column order', 'dtype'])
def test_real_neuron_map_content_change_is_refused(tmp_path, label):
    gdir, cdir = _real_dirs()
    table = feather.read_table(cdir / 'normalized/neurons.feather')
    mutated = dict(_real_mutations(table))[label]
    scratch = _scratch_connectome(tmp_path, cdir, mutated)
    with pytest.raises(GraphUnavailable, match='Neuron map content hash mismatch'):
        verify_graph(gdir, scratch)


def test_real_graph_content_change_is_refused_and_container_change_accepted(tmp_path, monkeypatch):
    gdir, cdir = _real_dirs()
    with np.load(gdir / 'graph.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    init = zipfile.ZipInfo.__init__

    def windows_stamp(self, *a, **k):
        init(self, *a, **k)
        self.create_system = 0
    try:  # each copy is ~200 MB; never leave one in pytest's retained basetemp
        monkeypatch.setattr(zipfile.ZipInfo, '__init__', windows_stamp)
        (tmp_path / 'win').mkdir()
        np.savez(tmp_path / 'win/graph.npz', **arrays)
        monkeypatch.undo()
        identity = verify_graph(tmp_path / 'win', cdir)
        assert identity.graph_file_sha256 != load_pins()['graph_sha256']
        assert identity.graph_sha256 == load_pins()['graph_sha256']
        shutil.rmtree(tmp_path / 'win')
        arrays['weight'] = arrays['weight'].copy()
        arrays['weight'][12345] = -arrays['weight'][12345]
        (tmp_path / 'bad').mkdir()
        np.savez(tmp_path / 'bad/graph.npz', **arrays)
        with pytest.raises(GraphUnavailable, match='Graph content hash mismatch'):
            verify_graph(tmp_path / 'bad', cdir)
    finally:
        monkeypatch.undo()
        shutil.rmtree(tmp_path / 'win', ignore_errors=True)
        shutil.rmtree(tmp_path / 'bad', ignore_errors=True)
