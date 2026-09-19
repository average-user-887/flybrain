"""Resolve and verify the prepared MaleCNS graph by path, hash and neuron IDs.

The prepared graph is large and generated (ignored by git), so its location is
configuration, never an implicit guess:

1. an explicit ``graph_dir`` argument (CLI ``--graph-dir``),
2. the ``NEUROFLY_GRAPH_DIR`` environment variable,
3. ``<checkout>/outputs/brainlab/malecns_v1`` (the historical default).

The neuron table follows the same order with ``NEUROFLY_CONNECTOME_DIR`` and
``<checkout>/connectome_data/malecns_v1``.  Nothing searches other checkouts.
A missing or mismatching graph raises :class:`GraphUnavailable`; there is no
silent substitution.  Synthetic graphs are built only by
:func:`synthetic_test_graph`, which labels them conspicuously.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = Path(__file__).resolve().with_name('graph_pins.json')
GRAPH_ENV = 'NEUROFLY_GRAPH_DIR'
CONNECTOME_ENV = 'NEUROFLY_CONNECTOME_DIR'
DEFAULT_GRAPH_DIR = ROOT / 'outputs/brainlab/malecns_v1'
DEFAULT_CONNECTOME_DIR = ROOT / 'connectome_data/malecns_v1'
SYNTHETIC_LABEL = 'SYNTHETIC TEST GRAPH - not MaleCNS; no scientific claim'

# Descending-neuron channels used by brainlab.cosim_server (node indices into
# the pinned graph).  They are verified here against cell types; side labels
# are recorded from annotations, not inferred from row order.
DN_CHANNELS: Dict[str, List[int]] = {
    'dnp01': [0, 6], 'dna02_l': [332], 'dna02_r': [131957], 'dna01': [406, 704],
    'dnp09': [725, 1087], 'dnb01': [608, 703], 'mdn': [706, 1196, 1240, 2194],
}
DN_EXPECTED_TYPES = {'dnp01': 'DNp01', 'dna02_l': 'DNa02', 'dna02_r': 'DNa02',
                     'dna01': 'DNa01', 'dnp09': 'DNp09', 'dnb01': 'DNb01', 'mdn': 'MDN'}

# Constants of brainlab/engine.py, with units, so manifests never guess them.
LIF_DYNAMICS = {
    'model': 'fixed-weight leaky integrate-and-fire (brainlab.engine.advance)',
    'dt_ms': 0.1, 'tau_membrane_ms': 20.0, 'tau_synapse_ms': 5.0,
    'v_rest_mV': -52.0, 'v_reset_mV': -52.0, 'v_threshold_mV': -45.0,
    'refractory_ms': 2.2, 'transmission_delay_ms': 1.8,
    'synaptic_scale': 0.275,
    'weight_units': 'synapse_count * transmitter_sign * 0.275 (upstream mV-equivalent)',
    'input_units': 'per-neuron drive current (upstream mV-equivalent)',
    'output_units': 'spike counts per step window',
    'biological_validation': 'none; engineering proxy',
}


class GraphUnavailable(RuntimeError):
    """The requested graph or neuron map is missing or fails verification."""


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_pins() -> dict:
    return json.loads(PINS_PATH.read_text())


def resolve_graph_dir(graph_dir: Optional[os.PathLike] = None) -> tuple[Path, str]:
    """Return (directory, how it was chosen).  Does not check existence."""
    if graph_dir is not None:
        return Path(graph_dir).expanduser(), 'argument'
    if os.environ.get(GRAPH_ENV):
        return Path(os.environ[GRAPH_ENV]).expanduser(), f'env:{GRAPH_ENV}'
    return DEFAULT_GRAPH_DIR, 'default'


def resolve_connectome_dir(connectome_dir: Optional[os.PathLike] = None) -> tuple[Path, str]:
    if connectome_dir is not None:
        return Path(connectome_dir).expanduser(), 'argument'
    if os.environ.get(CONNECTOME_ENV):
        return Path(os.environ[CONNECTOME_ENV]).expanduser(), f'env:{CONNECTOME_ENV}'
    return DEFAULT_CONNECTOME_DIR, 'default'


@dataclass
class GraphIdentity:
    """Immutable identity of the graph a controller runs on."""
    dataset: str
    synthetic: bool
    graph_path: Optional[str]
    graph_path_source: str
    graph_sha256: str
    neuron_map_path: Optional[str]
    neuron_map_sha256: Optional[str]
    io_map_sha256: str
    neurons: int
    edges: int
    ids_sha256: str
    data_sha256: Dict[str, str] = field(default_factory=dict)
    label: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


def _ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype='<i8').tobytes()).hexdigest()


def verify_graph(graph_dir: Optional[os.PathLike] = None,
                 connectome_dir: Optional[os.PathLike] = None,
                 *, pins: Optional[dict] = None, check_hashes: bool = True) -> GraphIdentity:
    """Verify the real graph against pinned hashes, counts, IDs and DN cell types.

    Reads only the small ``ids``/``ptr`` arrays of the graph (``np.load`` is
    lazy per member) plus the neuron table; the full edge arrays are hashed as
    bytes, never loaded here.
    """
    pins = pins or load_pins()
    gdir, gsource = resolve_graph_dir(graph_dir)
    cdir, _ = resolve_connectome_dir(connectome_dir)
    graph = gdir / 'graph.npz'
    nodes_path = cdir / 'normalized/neurons.feather'
    if not graph.is_file():
        raise GraphUnavailable(
            f'Prepared graph not found at {graph} (chosen by {gsource}). Set {GRAPH_ENV} '
            'to the directory holding graph.npz, or pass --graph-dir. Synthetic '
            'substitution requires an explicit test option.')
    if not nodes_path.is_file():
        raise GraphUnavailable(f'Neuron map not found at {nodes_path}. Set {CONNECTOME_ENV}.')
    graph_sha = sha256_file(graph) if check_hashes else pins['graph_sha256']
    nodes_sha = sha256_file(nodes_path) if check_hashes else pins['neuron_map_sha256']
    if graph_sha != pins['graph_sha256']:
        raise GraphUnavailable(f'Graph hash mismatch at {graph}: {graph_sha} != pinned {pins["graph_sha256"]}')
    if nodes_sha != pins['neuron_map_sha256']:
        raise GraphUnavailable(f'Neuron map hash mismatch at {nodes_path}: {nodes_sha}')
    import pyarrow.feather as feather
    nodes = feather.read_table(nodes_path, columns=['node_index', 'source_id', 'cell_type']).to_pandas()
    with np.load(graph, allow_pickle=False) as data:
        ids = data['ids']
        ptr = data['ptr']
    if len(ids) != pins['neurons'] or int(ptr[-1]) != pins['edges']:
        raise GraphUnavailable(f'Graph counts {len(ids)}/{int(ptr[-1])} differ from pins')
    if not np.array_equal(ids, nodes.source_id.to_numpy(dtype=np.int64)):
        raise GraphUnavailable('Graph neuron IDs do not match neurons.feather source_id order')
    if not np.array_equal(nodes.node_index.to_numpy(), np.arange(len(nodes))):
        raise GraphUnavailable('neurons.feather node_index is not 0..n-1')
    ids_sha = _ids_digest(ids)
    if ids_sha != pins['ids_sha256']:
        raise GraphUnavailable(f'Graph ID digest {ids_sha} differs from pin')
    dn = []
    for channel, indices in DN_CHANNELS.items():
        for index in indices:
            ctype = nodes.cell_type.iat[index]
            if ctype != DN_EXPECTED_TYPES[channel]:
                raise GraphUnavailable(f'DN channel {channel} node {index} is {ctype}')
            dn.append(dict(channel=channel, node_index=index, source_id=int(ids[index]), cell_type=ctype))
    # Small source tables are rehashed; the 1 GB raw edge table is recorded from
    # source.lock.json (the prepared graph hash already pins what was derived from it).
    data_sha = {}
    for name, expected in pins.get('source_sha256', {}).items():
        path = cdir / name
        if check_hashes and name != 'edges.feather' and path.is_file():
            actual = sha256_file(path)
            if actual != expected:
                raise GraphUnavailable(f'{name} hash mismatch: {actual}')
            data_sha[name] = actual
        else:
            data_sha[name + ' (pinned, not rehashed)'] = expected
    return GraphIdentity(
        dataset='malecns_v1', synthetic=False, graph_path=str(graph.resolve()),
        graph_path_source=gsource, graph_sha256=graph_sha,
        neuron_map_path=str(nodes_path.resolve()), neuron_map_sha256=nodes_sha,
        io_map_sha256=sha256_json(dn), neurons=len(ids), edges=int(ptr[-1]),
        ids_sha256=ids_sha, data_sha256=data_sha, label='MaleCNS v1.0 prepared graph')


def synthetic_test_graph(n: int = 64, k_out: int = 6, seed: int = 0):
    """Build a small random CSR graph in memory, conspicuously labelled.

    Returns ``(arrays, identity)``.  Never writes a file; never presented as MaleCNS.
    """
    rng = np.random.default_rng(seed)
    post = rng.integers(0, n, size=n * k_out).astype(np.int32)
    ptr = np.arange(0, (n + 1) * k_out, k_out, dtype=np.int64)
    weight = (rng.standard_normal(n * k_out) * 4.0).astype(np.float32)
    ids = np.arange(1, n + 1, dtype=np.int64)
    arrays = dict(ptr=ptr, post=post, weight=weight, ids=ids)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(arrays[key].tobytes())
    io_map = {name: [i % n for i in idx] for name, idx in
              {'dnp01': [0, 1], 'dna02_l': [10], 'dna02_r': [11], 'dna01': [12, 13],
               'dnp09': [14, 15], 'dnb01': [16, 17], 'mdn': [18, 19]}.items()}
    identity = GraphIdentity(
        dataset='synthetic-test', synthetic=True, graph_path=None,
        graph_path_source=f'synthetic_test_graph(n={n},k_out={k_out},seed={seed})',
        graph_sha256=digest.hexdigest(), neuron_map_path=None, neuron_map_sha256=None,
        io_map_sha256=sha256_json(io_map), neurons=n, edges=n * k_out,
        ids_sha256=_ids_digest(ids), label=SYNTHETIC_LABEL)
    return arrays, identity, io_map


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify the prepared graph and write an identity receipt.')
    parser.add_argument('--graph-dir', type=Path)
    parser.add_argument('--connectome-dir', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    identity = verify_graph(args.graph_dir, args.connectome_dir)
    text = json.dumps(identity.to_dict(), indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
