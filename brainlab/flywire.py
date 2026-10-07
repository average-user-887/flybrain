"""FlyWire 783 (female) whole brain as a SEPARATE, labelled dataset.

This module imports the public FlyWire materialization 783 female adult brain
(FAFB) into its own dataset id, store namespace and hash pins.  It never
touches the MaleCNS files, IDs or pins, never splices the two graphs and
never maps an ID of one dataset onto the other.  MaleCNS stays the default
everywhere; nothing here is reached unless a caller asks for this dataset.

Scope (owner decision, 7 Oct 2026; Astra card F2 items 1-5):

* Node universe: every proofread root ID of Zenodo 10676866
  (``proofread_root_ids_783.npy``).  Isolated cells (no proofread
  connection) and connected cells without an annotation row stay in the
  graph, with an explicit ``annotation_status`` and an ``unknown``
  transmitter source; nothing is dropped silently.
* Edges: ``proofread_connections_783.feather`` has one row per neuron pair
  PER NEUROPIL.  Rows of one pair are summed, so every synapse is kept; the
  report reconciles the summed graph against the source totals.
* Sign: declared policy ``known_nt`` first, ``top_nt`` fallback
  (:func:`parse_known_nt`, :func:`resolve_transmitter`), then the existing
  repository rule :func:`brainlab.transmitters.transmitter_signs` unchanged
  (ACh +; GABA, glutamate, histamine -; anything else the declared
  ambiguous sign +1).  A ``<x>-negative`` label is evidence that the cell
  does NOT use ``x``; it is never read as "inhibitory".
* Weight: ``synapse_count x sign x 0.275``, the MaleCNS arithmetic.  The same
  0.275 does NOT make the two datasets equivalently calibrated: different
  animal, sex, EM volume, synapse detector and transmitter predictor.
* Brain only.  FlyWire 783 has no ventral nerve cord.
* I/O: no encoder or decoder mapping is declared for this dataset (see
  :data:`IO_DECLARATION`).  Anything that needs one fails with
  :class:`GraphUnavailable`; MaleCNS IDs and node indices are never used.

Licence (card item 5): the Zenodo tables are CC BY 4.0.  The annotation
table (flyconnectome/flywire_annotations, pinned at v3.2.0) has NO stated
licence.  The user downloads it from GitHub; NeuroFly does not ship it or any
table derived from it, and claims no clearance.  Everything this module
writes goes under the git-ignored ``connectome_data/`` and ``outputs/``.

Usage::

    python -m brainlab.flywire download   # fetch or re-verify the pinned sources
    python -m brainlab.flywire import     # build nodes, edges, graph.npz, reports
    python -m brainlab.flywire verify     # check the build against the pins
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from .connectome import ROOT, exact_ids, file_digest, index_edges
from .transmitters import transmitter_signs

DATASET_ID = 'flywire_783_female'
DATASET_LABEL = 'FlyWire 783 (female)'
IDENTITY_LABEL = ('FlyWire 783 (female) whole-brain graph; brain only, no VNC; separate dataset, '
                  'never spliced with MaleCNS')
COVERAGE = 'brain_only_no_vnc'
SIGN_POLICY = 'flywire-known_nt-first-top_nt-fallback-v1'
SYNAPTIC_SCALE = 0.275
AMBIGUOUS_SIGN = 1

LOCK_PATH = ROOT / 'data-provenance' / DATASET_ID / 'source.lock.json'
PINS_PATH = Path(__file__).resolve().with_name(f'graph_pins_{DATASET_ID}.json')
DEFAULT_CONNECTOME_DIR = ROOT / 'connectome_data' / DATASET_ID
DEFAULT_GRAPH_DIR = ROOT / 'outputs' / 'brainlab' / DATASET_ID
# Checkpoint store namespace.  A sibling of the MaleCNS ``outputs/registry``,
# never inside it, so no MaleCNS registry scan ever sees FlyWire instances.
DEFAULT_REGISTRY_ROOT = ROOT / 'outputs' / f'registry-{DATASET_ID}'
CONNECTOME_ENV = 'NEUROFLY_FLYWIRE_CONNECTOME_DIR'
GRAPH_ENV = 'NEUROFLY_FLYWIRE_GRAPH_DIR'

CONNECTIONS = 'proofread_connections_783.feather'
ROOT_IDS = 'proofread_root_ids_783.npy'
NEUROPIL_PRE = 'per_neuron_neuropil_count_pre_783.feather'
NEUROPIL_POST = 'per_neuron_neuropil_count_post_783.feather'
ANNOTATIONS = 'Supplemental_file1_neuron_annotations.tsv'

# The method identity: what turns the sources into weights.  Hashed into the
# graph identity, so a change of any field is a different graph.
METHOD = dict(
    dataset_id=DATASET_ID, label=DATASET_LABEL, coverage=COVERAGE,
    node_universe='all proofread root IDs of Zenodo 10676866 (proofread_root_ids_783.npy), '
                  'isolated and unannotated cells included',
    edge_rule='sum of syn_count over all neuropil rows of a (pre, post) pair; no threshold; '
              'autapses kept if present',
    sign_policy=SIGN_POLICY,
    sign_rule='known_nt first (explicit parser; "<x>-negative" = not x, never inhibitory), '
              'top_nt fallback; then brainlab.transmitters.transmitter_signs unchanged '
              '(ACh +; GABA, glutamate, histamine -; other, conflicting or missing: +1)',
    ambiguous_sign=AMBIGUOUS_SIGN,
    weight='synapse_count x sign x synaptic_scale',
    synaptic_scale=SYNAPTIC_SCALE,
    calibration_note='The same 0.275 as MaleCNS does NOT mean the datasets are calibrated '
                     'equivalently: different animal, sex, EM volume, synapse detection and '
                     'transmitter predictor.  Photoreceptor synapse counts are known to be '
                     'under-detected in FlyWire (Matsliah et al. 2024).',
    fitting=None,
)

# No encoder/decoder is declared for this dataset.  Each entry says why.
IO_DECLARATION = dict(
    dataset_id=DATASET_ID,
    status='unavailable',
    mappings={
        'descending_neuron_channels': 'unavailable: the MaleCNS DN channels are MaleCNS node indices '
                                      'and are never reused; no FlyWire DN mapping is declared',
        'optomotor_t4_t5_encoder': 'unavailable: the encoder reads MaleCNS annotation columns; no '
                                   'FlyWire adapter is declared',
        'photoreceptor_column_encoder': 'unavailable: the public FlyWire annotations carry no '
                                        'column/ommatidium index and no sourced mapping was found; '
                                        'columns are never fabricated',
        'embodied_body_io': 'unavailable: brain only, no VNC, no motor mapping',
    },
)

FAST_TRANSMITTERS = ('acetylcholine', 'gaba', 'glutamate', 'histamine')
AMINERGIC_TRANSMITTERS = ('dopamine', 'serotonin', 'octopamine', 'tyramine')
CLASSICAL_TRANSMITTERS = FAST_TRANSMITTERS + AMINERGIC_TRANSMITTERS


class FlyWireUnavailable(RuntimeError):
    """A FlyWire source, build or mapping is missing or fails verification."""


def _graph_unavailable():
    from .graph_identity import GraphUnavailable
    return GraphUnavailable


def io_unavailable(what: str) -> None:
    """Refuse any request for an I/O mapping on this dataset."""
    raise _graph_unavailable()(
        f'{DATASET_LABEL}: {what} is unavailable for this dataset '
        f'({IO_DECLARATION["mappings"].get(what, "no mapping declared")}). '
        'MaleCNS IDs and node indices are never substituted.')


# ---------------------------------------------------------------------------
# known_nt parser and the declared sign policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnownNT:
    """Parsed ``known_nt`` field.

    The field joins evidence groups with ``;`` and tokens with ``,``.  A token
    ``x-negative`` states that the cell was found NOT to use ``x``; a bare
    ``negative`` names no substance.  A substance listed both positive and
    negative (different sources disagreeing) is ``conflicted`` and is not used.
    """
    raw: str
    positive: frozenset = field(default_factory=frozenset)      # classical, not conflicted
    other_positive: frozenset = field(default_factory=frozenset)  # peptides, NO, glycine, ...
    negative: frozenset = field(default_factory=frozenset)
    conflicted: frozenset = field(default_factory=frozenset)
    unspecified_negative: bool = False

    @property
    def status(self) -> str:
        if not self.raw:
            return 'empty'
        if self.positive:
            return 'classical_positive'
        if self.conflicted:
            return 'conflicted_only'
        if self.other_positive:
            return 'non_classical_only'
        return 'negative_only'

    @property
    def label(self) -> Optional[str]:
        return ','.join(sorted(self.positive)) if self.positive else None


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ''
    return str(value).strip()


def parse_known_nt(value) -> KnownNT:
    raw = _clean(value)
    positive, negative, unspecified = set(), set(), False
    for group in raw.split(';'):
        for token in group.split(','):
            token = ' '.join(token.strip().lower().split())
            if not token:
                continue
            if token == 'negative':
                unspecified = True
            elif token.endswith('-negative'):
                negative.add(token[:-len('-negative')].strip())
            else:
                positive.add(token)
    conflicted = positive & negative
    usable = positive - conflicted
    return KnownNT(raw=raw, positive=frozenset(usable & set(CLASSICAL_TRANSMITTERS)),
                   other_positive=frozenset(usable - set(CLASSICAL_TRANSMITTERS)),
                   negative=frozenset(negative), conflicted=frozenset(conflicted),
                   unspecified_negative=unspecified)


def resolve_transmitter(known_nt, top_nt) -> Tuple[Optional[str], str]:
    """Declared policy: ``(label, source)`` for one annotated cell.

    ``label`` is a lower-case comma-joined set of classical transmitters (the
    vocabulary of :func:`brainlab.transmitters.transmitter_signs`) or None.
    """
    known = parse_known_nt(known_nt)
    if known.positive:
        return known.label, 'known_nt'
    top = _clean(top_nt).lower()
    if top in CLASSICAL_TRANSMITTERS:
        return top, f'top_nt_fallback:{known.status}'
    return None, f'unknown:{known.status}+top_nt_missing'


def _sign_of(label: Optional[str]) -> Tuple[int, bool]:
    signs, ambiguous = transmitter_signs([label], ambiguous_sign=AMBIGUOUS_SIGN)
    return int(signs[0]), bool(ambiguous[0])


# ---------------------------------------------------------------------------
# Sources: pins, attribution, download
# ---------------------------------------------------------------------------
ATTRIBUTION = """\
FlyWire 783 (female) -- FlyWire Consortium, Princeton University.
Connectivity, root IDs and neuropil counts: Zenodo record 10676866 (version 783.0),
  https://doi.org/10.5281/zenodo.10676866, licence CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/).
  Credits in the record: synapse detection Buhmann et al. 2021; cleft segmentation
  Heinrich et al. 2018; neurotransmitter prediction Eckstein, Bates et al. 2024 (Cell);
  segmentation and proofreading Dorkenwald et al. 2024 (Nature).
Annotations: github.com/flyconnectome/flywire_annotations, tag v3.2.0 (commit
  a83b2776d60d5764cef36b927f5f9679c16c47a2).  Its README asks you to cite Berg et al.
  2026 (Cell), Schlegel et al. 2024, Tastekin et al. 2026, Matsliah et al. 2024 and
  Dorkenwald et al. 2024.  The repository states NO licence: you download it yourself
  for local use; NeuroFly does not redistribute it or any table derived from it.
This is a female brain without VNC; it is a separate dataset and is never combined
with MaleCNS."""


def load_lock(path: Path = LOCK_PATH) -> dict:
    return json.loads(Path(path).read_text())


def resolve_connectome_dir(connectome_dir=None) -> Path:
    if connectome_dir is not None:
        return Path(connectome_dir).expanduser()
    if os.environ.get(CONNECTOME_ENV):
        return Path(os.environ[CONNECTOME_ENV]).expanduser()
    return DEFAULT_CONNECTOME_DIR


def resolve_graph_dir(graph_dir=None) -> Path:
    if graph_dir is not None:
        return Path(graph_dir).expanduser()
    if os.environ.get(GRAPH_ENV):
        return Path(os.environ[GRAPH_ENV]).expanduser()
    return DEFAULT_GRAPH_DIR


def verify_sources(source_dir: Path, lock: Optional[dict] = None, *, md5: bool = False) -> Dict[str, dict]:
    """Byte-verify every pinned source file; raise on any mismatch or absence."""
    lock = lock or load_lock()
    checked = {}
    for name, entry in lock['files'].items():
        path = Path(source_dir) / name
        if not path.is_file():
            raise FlyWireUnavailable(f'{path} missing; run python -m brainlab.flywire download')
        size = path.stat().st_size
        if size != entry['bytes']:
            raise FlyWireUnavailable(f'{name}: {size} bytes, pinned {entry["bytes"]}')
        digest = file_digest(path)
        if digest != entry['sha256']:
            raise FlyWireUnavailable(f'{name}: sha256 {digest} != pinned {entry["sha256"]}')
        result = dict(bytes=size, sha256=digest)
        if md5 and entry.get('md5'):
            h = hashlib.md5()
            with path.open('rb') as stream:
                while block := stream.read(8 << 20):
                    h.update(block)
            if h.hexdigest() != entry['md5']:
                raise FlyWireUnavailable(f'{name}: md5 {h.hexdigest()} != Zenodo {entry["md5"]}')
            result['md5'] = h.hexdigest()
        checked[name] = result
    return checked


def download(dest=None, lock: Optional[dict] = None) -> Dict[str, dict]:
    """Fetch each missing pinned file; re-verify files already present."""
    lock = lock or load_lock()
    out = resolve_connectome_dir(dest)
    out.mkdir(parents=True, exist_ok=True)
    print(ATTRIBUTION + '\n', flush=True)
    for name, entry in lock['files'].items():
        target = out / name
        if not target.exists():
            partial = target.with_name(target.name + '.partial')
            print(f'Downloading {name} ({entry["bytes"]:,} bytes)', flush=True)
            urllib.request.urlretrieve(entry['url'], partial)
            if file_digest(partial) != entry['sha256']:
                raise FlyWireUnavailable(f'Checksum mismatch: {partial}')
            partial.replace(target)
    checked = verify_sources(out, lock)
    for name in checked:
        print(f'Verified {name}', flush=True)
    return checked


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
ANNOTATION_COLUMNS = ('root_id', 'flow', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type',
                      'side', 'top_nt', 'top_nt_conf', 'known_nt', 'known_nt_source')


def read_annotations(path: Path):
    """Read the annotation TSV as text (root IDs never pass through float)."""
    import pandas as pd
    frame = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_values=[],
                        usecols=list(ANNOTATION_COLUMNS))
    frame['root_id'] = exact_ids(frame['root_id'].to_numpy())
    if not frame['root_id'].is_unique:
        raise FlyWireUnavailable('Duplicate root_id rows in the annotation table')
    return frame


def build_nodes(root_ids: np.ndarray, annotations, connected_ids: np.ndarray):
    """Node table over the whole proofread universe, plus a node report.

    ``root_ids``: the proofread universe (any order, unique, exact integers).
    ``connected_ids``: IDs that occur in at least one source connection row.
    """
    import pandas as pd
    ids = np.sort(exact_ids(root_ids))
    if len(np.unique(ids)) != len(ids):
        raise FlyWireUnavailable('Duplicate proofread root IDs')
    ann = annotations.set_index('root_id')
    in_universe = np.isin(ann.index.to_numpy(dtype=np.uint64), ids)
    outside = ann.index.to_numpy(dtype=np.uint64)[~in_universe]
    ann = ann.loc[in_universe]
    annotated = np.isin(ids, ann.index.to_numpy(dtype=np.uint64))
    connected = np.isin(ids, exact_ids(connected_ids))
    row = ann.reindex(ids)

    def text(column):
        values = row[column].to_numpy(dtype=object)
        return np.array([None if (v is None or (isinstance(v, float) and v != v) or v == '') else v
                         for v in values], dtype=object)

    labels, sources, statuses = [], [], []
    for is_ann, known, top in zip(annotated, row['known_nt'].to_numpy(dtype=object),
                                  row['top_nt'].to_numpy(dtype=object)):
        if not is_ann:
            labels.append(None)
            sources.append('unknown:no_annotation_row')
            statuses.append('no_annotation_row')
            continue
        label, source = resolve_transmitter(known, top)
        labels.append(label)
        sources.append(source)
        statuses.append(parse_known_nt(known).status)
    signs, ambiguous = transmitter_signs(labels, ambiguous_sign=AMBIGUOUS_SIGN)
    conf = pd.to_numeric(row['top_nt_conf'].replace('', np.nan), errors='coerce').to_numpy(dtype=np.float32)
    nodes = pd.DataFrame({
        'node_index': np.arange(len(ids), dtype=np.uint32),
        'source_id': ids,
        'annotation_status': np.where(annotated, 'annotated', 'no_annotation_row'),
        'connected': connected,
        'flow': text('flow'), 'super_class': text('super_class'), 'cell_class': text('cell_class'),
        'cell_sub_class': text('cell_sub_class'), 'cell_type': text('cell_type'), 'side': text('side'),
        'top_nt': text('top_nt'), 'top_nt_conf': conf,
        'known_nt': text('known_nt'), 'known_nt_source': text('known_nt_source'),
        'known_nt_status': np.asarray(statuses, dtype=object),
        'neurotransmitter': np.asarray(labels, dtype=object),
        'neurotransmitter_source': np.asarray(sources, dtype=object),
        'sign': signs.astype(np.int8),
        'sign_ambiguous': ambiguous,
    })
    report = dict(
        proofread_root_ids=int(len(ids)),
        annotation_rows=int(len(annotations)),
        annotation_rows_outside_universe=int(len(outside)),
        annotation_rows_outside_universe_ids=[str(int(v)) for v in outside],
        nodes=int(len(ids)),
        annotated_nodes=int(annotated.sum()),
        unannotated_nodes=int((~annotated).sum()),
        connected_nodes=int(connected.sum()),
        isolated_nodes=int((~connected).sum()),
        unannotated_connected_nodes=int((~annotated & connected).sum()),
        unannotated_isolated_nodes=int((~annotated & ~connected).sum()),
        connected_ids_outside_universe=int(len(np.setdiff1d(exact_ids(connected_ids), ids))),
    )
    return nodes, report


NODE_SCHEMA = (('node_index', 'uint32'), ('source_id', 'uint64'), ('annotation_status', 'string'),
               ('connected', 'bool_'), ('flow', 'string'), ('super_class', 'string'), ('cell_class', 'string'),
               ('cell_sub_class', 'string'), ('cell_type', 'string'), ('side', 'string'), ('top_nt', 'string'),
               ('top_nt_conf', 'float32'), ('known_nt', 'string'), ('known_nt_source', 'string'),
               ('known_nt_status', 'string'), ('neurotransmitter', 'string'),
               ('neurotransmitter_source', 'string'), ('sign', 'int8'), ('sign_ambiguous', 'bool_'))


def nodes_table(nodes):
    """The node table with a fixed Arrow schema (an all-null text column stays text)."""
    import pyarrow as pa
    schema = pa.schema([(name, getattr(pa, kind)()) for name, kind in NODE_SCHEMA])
    if list(nodes.columns) != schema.names:
        raise FlyWireUnavailable('Node table columns differ from NODE_SCHEMA')
    return pa.Table.from_pandas(nodes, schema=schema, preserve_index=False)


def sign_report(nodes) -> dict:
    """Semantic handling, disagreements, fallbacks and unknowns of the sign policy."""
    from collections import Counter
    parsed = [parse_known_nt(v) for v in nodes['known_nt'].to_numpy(dtype=object)]
    annotated = nodes['annotation_status'].to_numpy() == 'annotated'
    tops = [_clean(v).lower() for v in nodes['top_nt'].to_numpy(dtype=object)]
    negative_cells = sum(1 for p, a in zip(parsed, annotated) if a and (p.negative or p.unspecified_negative))
    negative_only = [i for i, (p, a) in enumerate(zip(parsed, annotated)) if a and p.status == 'negative_only']
    signs = nodes['sign'].to_numpy()
    amb = nodes['sign_ambiguous'].to_numpy()
    neg_only_signs = Counter('ambiguous(+1)' if amb[i] else ('+1' if signs[i] > 0 else '-1') for i in negative_only)
    conflicted = Counter(s for p in parsed for s in p.conflicted)
    label_disagree, sign_disagree, both = Counter(), 0, 0
    for p, top in zip(parsed, tops):
        if p.positive and top in CLASSICAL_TRANSMITTERS:
            both += 1
            if top not in p.positive:
                label_disagree[f'{p.label} vs top_nt {top}'] += 1
            ks, ka = _sign_of(p.label)
            ts, ta = _sign_of(top)
            if (ks, ka) != (ts, ta):
                sign_disagree += 1
    by_source = Counter(nodes['neurotransmitter_source'])
    resolved = Counter('missing' if not isinstance(v, str) else v for v in nodes['neurotransmitter'])
    return dict(
        policy=SIGN_POLICY,
        negative_label_semantics=('"<x>-negative" = evidence the cell does NOT use x; it carries no sign '
                                  'and is never read as inhibitory. A substance listed both positive and '
                                  'negative is "conflicted" and not used. A bare "negative" names no '
                                  'substance and is ignored (counted).'),
        known_nt_status_counts=dict(Counter(p.status for p, a in zip(parsed, annotated) if a)),
        cells_with_any_negative_token=negative_cells,
        cells_with_bare_negative_token=sum(1 for p in parsed if p.unspecified_negative),
        negative_only_cells=len(negative_only),
        negative_only_resolved_signs=dict(neg_only_signs),
        conflicted_substance_counts=dict(conflicted),
        cells_with_conflicted_substance=sum(1 for p in parsed if p.conflicted),
        cells_with_non_classical_positive=sum(1 for p in parsed if p.other_positive),
        known_vs_top_both_present=both,
        known_vs_top_label_disagreements=sum(label_disagree.values()),
        known_vs_top_label_disagreements_top=dict(label_disagree.most_common(12)),
        known_vs_top_sign_disagreements=sign_disagree,
        source_counts=dict(by_source),
        fallback_to_top_nt=sum(v for k, v in by_source.items() if k.startswith('top_nt_fallback')),
        unknown_transmitter=sum(v for k, v in by_source.items() if k.startswith('unknown')),
        resolved_label_counts=dict(resolved.most_common()),
        sign_counts={'+1': int(((signs > 0) & ~amb).sum()), '-1': int((signs < 0).sum()),
                     'ambiguous(+1)': int(amb.sum())},
    )


# ---------------------------------------------------------------------------
# Edges: neuropil rows -> neuron pairs, chunked
# ---------------------------------------------------------------------------
def iter_connection_batches(path: Path) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (pre, post, syn_count, neuropil) per record batch, memory-mapped."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), 'r') as source:
        reader = ipc.open_file(source)
        names = reader.schema.names
        for column in ('pre_pt_root_id', 'post_pt_root_id', 'syn_count', 'neuropil'):
            if column not in names:
                raise FlyWireUnavailable(f'{path.name} lacks column {column}')
        for number in range(reader.num_record_batches):
            batch = reader.get_batch(number)
            get = lambda c: batch.column(names.index(c))
            pre = get('pre_pt_root_id').to_numpy(zero_copy_only=False)
            post = get('post_pt_root_id').to_numpy(zero_copy_only=False)
            count = get('syn_count').to_numpy(zero_copy_only=False)
            neuropil = np.asarray(get('neuropil').to_pylist(), dtype=object)
            yield pre, post, count, neuropil


def aggregate_edges(ids: np.ndarray, batches: Iterable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sum per-neuropil rows into one edge per (pre, post) pair, losslessly.

    ``ids`` is the sorted unique uint64 node universe.  Returns ``(pre_index,
    post_index, synapse_count, stats)`` sorted by (pre, post).  Rows whose
    endpoints are outside the universe are excluded and counted, never dropped
    silently.
    """
    n = len(ids)
    keys, counts, triple_keys = [], [], []
    neuropil_codes: Dict[str, int] = {}
    stats = dict(source_rows=0, source_synapses=0, retained_rows=0, retained_synapses=0,
                 excluded_rows=0, excluded_synapses=0)
    for pre, post, count, neuropil in batches:
        count = np.asarray(count)
        i, j, kept, keep = index_edges(ids, pre, post, count)
        stats['source_rows'] += len(count)
        stats['source_synapses'] += int(count.sum(dtype=np.uint64))
        stats['retained_rows'] += len(kept)
        stats['retained_synapses'] += int(kept.sum(dtype=np.uint64))
        key = i.astype(np.uint64) * np.uint64(n) + j.astype(np.uint64)
        codes = np.fromiter((neuropil_codes.setdefault(str(v), len(neuropil_codes)) for v in neuropil[keep]),
                            dtype=np.uint64, count=int(keep.sum()))
        keys.append(key)
        counts.append(kept)
        triple_keys.append(codes)
    stats['excluded_rows'] = stats['source_rows'] - stats['retained_rows']
    stats['excluded_synapses'] = stats['source_synapses'] - stats['retained_synapses']
    key = np.concatenate(keys) if keys else np.zeros(0, np.uint64)
    count = np.concatenate(counts).astype(np.uint64) if counts else np.zeros(0, np.uint64)
    code = np.concatenate(triple_keys) if triple_keys else np.zeros(0, np.uint64)
    del keys, counts, triple_keys
    order = np.lexsort((code, key))
    key, count, code = key[order], count[order], code[order]
    del order
    duplicate_triples = int(np.count_nonzero((key[1:] == key[:-1]) & (code[1:] == code[:-1]))) if len(key) else 0
    del code
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.zeros(0, np.int64)
    pair_key = key[starts]
    summed = np.add.reduceat(count, starts) if len(key) else np.zeros(0, np.uint64)
    rows_per_pair = np.diff(np.r_[starts, len(key)])
    if len(summed) and int(summed.max()) > 2**32 - 1:
        raise FlyWireUnavailable('A pair synapse count exceeds uint32')
    pre_index = (pair_key // np.uint64(n)).astype(np.uint32)
    post_index = (pair_key % np.uint64(n)).astype(np.uint32)
    stats.update(
        aggregated_pairs=int(len(pair_key)),
        aggregated_synapses=int(summed.sum(dtype=np.uint64)),
        rows_merged_into_existing_pairs=int(len(key) - len(pair_key)),
        pairs_spanning_multiple_neuropils=int(np.count_nonzero(rows_per_pair > 1)),
        max_neuropil_rows_per_pair=int(rows_per_pair.max()) if len(rows_per_pair) else 0,
        duplicate_pair_neuropil_rows=duplicate_triples,
        distinct_neuropils=len(neuropil_codes),
        self_pairs=int(np.count_nonzero(pre_index == post_index)),
        pairs_below_5_synapses=int(np.count_nonzero(summed < 5)),
    )
    if stats['aggregated_synapses'] != stats['retained_synapses']:
        raise FlyWireUnavailable('Aggregation lost synapses: '
                                 f'{stats["aggregated_synapses"]} != {stats["retained_synapses"]}')
    if stats['retained_synapses'] + stats['excluded_synapses'] != stats['source_synapses']:
        raise FlyWireUnavailable('Source synapse totals do not reconcile')
    return pre_index, post_index, summed.astype(np.uint32), stats


def csr_arrays(n: int, pre_index, post_index, synapse_count, signs) -> Dict[str, np.ndarray]:
    """The brainlab.prepare CSR layout and arithmetic, verbatim."""
    pre, post, count = (np.asarray(pre_index), np.asarray(post_index), np.asarray(synapse_count))
    order = np.argsort(pre, kind='stable')
    return dict(
        ptr=np.r_[0, np.cumsum(np.bincount(pre, minlength=n))].astype(np.int64),
        post=post[order].astype(np.int32),
        weight=(count[order].astype(np.float32) * signs[pre[order]] * .275).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def import_flywire(source_dir=None, connectome_dir=None, graph_dir=None, *, lock: Optional[dict] = None,
                   verify: bool = True) -> dict:
    """Build the FlyWire dataset under its own namespace and return the report."""
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    lock = lock or load_lock()
    src = Path(source_dir) if source_dir is not None else resolve_connectome_dir(connectome_dir)
    cdir = resolve_connectome_dir(connectome_dir)
    gdir = resolve_graph_dir(graph_dir)
    checked = verify_sources(src, lock) if verify else {}
    roots = np.load(src / ROOT_IDS, allow_pickle=False)
    if roots.dtype.kind not in 'iu':
        raise FlyWireUnavailable(f'{ROOT_IDS} is {roots.dtype}, not an integer array')
    ids = np.sort(exact_ids(roots))
    pre_index, post_index, synapse_count, edge_stats = aggregate_edges(
        ids, iter_connection_batches(src / CONNECTIONS))
    connected = np.zeros(len(ids), dtype=bool)
    connected[pre_index] = True
    connected[post_index] = True
    annotations = read_annotations(src / ANNOTATIONS)
    nodes, node_report = build_nodes(ids, annotations, ids[connected])
    signs = nodes['sign'].to_numpy(dtype=np.int8)
    arrays = csr_arrays(len(ids), pre_index, post_index, synapse_count, signs)
    arrays['ids'] = ids.astype(np.int64)
    if not np.array_equal(arrays['ids'].astype(np.uint64), ids):
        raise FlyWireUnavailable('Root IDs did not survive int64 storage losslessly')
    incoming = np.bincount(post_index, weights=synapse_count, minlength=len(ids)).astype(np.int64)
    outgoing = np.bincount(pre_index, weights=synapse_count, minlength=len(ids)).astype(np.int64)
    edge_stats['incoming_equals_outgoing_total'] = bool(incoming.sum() == outgoing.sum()
                                                        == edge_stats['aggregated_synapses'])
    neuropil_crosscheck = {}
    for name, column in ((NEUROPIL_PRE, 'pre'), (NEUROPIL_POST, 'post')):
        if (src / name).is_file():
            table = feather.read_table(src / name, columns=['count'])
            neuropil_crosscheck[f'{column}_table_total'] = int(table.column('count').to_numpy().sum())
    neuropil_crosscheck['note'] = ('per-neuron neuropil tables count every detected synapse of each '
                                   'neuron, including partners outside the proofread set; informational, '
                                   'not a conservation target')
    out = cdir / 'normalized'
    out.mkdir(parents=True, exist_ok=True)
    gdir.mkdir(parents=True, exist_ok=True)
    feather.write_feather(nodes_table(nodes), out / 'neurons.feather')
    np.save(out / 'neuron_ids.npy', ids)
    schema = pa.schema([('pre_index', pa.uint32()), ('post_index', pa.uint32()), ('synapse_count', pa.uint32())])
    temporary = out / 'edges.arrow.partial'
    with pa.OSFile(str(temporary), 'wb') as sink, ipc.new_file(sink, schema) as writer:
        step = 1 << 22
        for start in range(0, len(pre_index), step):
            sl = slice(start, start + step)
            writer.write_batch(pa.record_batch([pa.array(pre_index[sl]), pa.array(post_index[sl]),
                                                pa.array(synapse_count[sl])], schema=schema))
    temporary.replace(out / 'edges.arrow')
    np.savez(gdir / 'graph.npz', **{k: arrays[k] for k in ('ptr', 'post', 'weight', 'ids')})
    signs_report = sign_report(nodes)
    report = dict(
        dataset_id=DATASET_ID, label=DATASET_LABEL, coverage=COVERAGE, sex='female',
        release='FlyWire materialization 783 (Zenodo 10676866 v783.0)',
        annotations=lock['annotations']['release'],
        method=METHOD, method_sha256=_sha256_json(METHOD), io=IO_DECLARATION,
        source_files_verified=checked, nodes=node_report, edges=edge_stats,
        neuropil_crosscheck=neuropil_crosscheck, sign=signs_report,
        graph=dict(neurons=int(len(ids)), edges=int(len(pre_index))),
        licence=lock['licence_gate'],
        neural_dynamics_validated=False, embodied_behavior_implemented=False,
    )
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = dict(dataset_id=DATASET_ID, dataset=DATASET_LABEL, label=IDENTITY_LABEL, coverage=COVERAGE,
                    neurons=int(len(ids)), edges=int(len(pre_index)),
                    uncertain_sign_neurons=int(nodes['sign_ambiguous'].sum()),
                    synaptic_scale=SYNAPTIC_SCALE, sign_policy=SIGN_POLICY,
                    method_sha256=report['method_sha256'], calibration_note=METHOD['calibration_note'],
                    io=IO_DECLARATION['status'], learning=False, game_interface=False)
    (gdir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return report


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Verification and identity
# ---------------------------------------------------------------------------
def load_pins(path: Path = PINS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def verify_flywire_graph(graph_dir=None, connectome_dir=None, *, pins: Optional[dict] = None):
    """Verify a FlyWire build against its own pins; return its GraphIdentity."""
    from .graph_identity import (GraphIdentity, _ids_digest, feather_content_sha256,
                                 npz_content_sha256, sha256_file, sha256_json)
    Unavailable = _graph_unavailable()
    pins = pins if pins is not None else load_pins()
    gdir, cdir = resolve_graph_dir(graph_dir), resolve_connectome_dir(connectome_dir)
    graph, nodes_path, manifest_path = gdir / 'graph.npz', cdir / 'normalized/neurons.feather', gdir / 'manifest.json'
    for path in (graph, nodes_path, manifest_path):
        if not path.is_file():
            raise Unavailable(f'{DATASET_LABEL}: {path} not found; run python -m brainlab.flywire import')
    if pins.get('dataset_id') != DATASET_ID:
        raise Unavailable(f'Pins are for {pins.get("dataset_id")!r}, not {DATASET_ID}')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('dataset_id') != DATASET_ID:
        raise Unavailable(f'{manifest_path} declares {manifest.get("dataset_id")!r}, not {DATASET_ID}')
    for key in ('graph_content_sha256', 'neuron_map_content_sha256', 'ids_sha256', 'method_sha256'):
        if not pins.get(key):
            raise Unavailable(f'{DATASET_LABEL} pins lack {key}; the build cannot be verified')
    if pins['method_sha256'] != _sha256_json(METHOD) or manifest.get('method_sha256') != pins['method_sha256']:
        raise Unavailable(f'{DATASET_LABEL}: method identity differs from the pinned method')
    graph_content = npz_content_sha256(graph)
    if graph_content != pins['graph_content_sha256']:
        raise Unavailable(f'{DATASET_LABEL} graph content hash mismatch: {graph_content}')
    nodes_content = feather_content_sha256(nodes_path)
    if nodes_content != pins['neuron_map_content_sha256']:
        raise Unavailable(f'{DATASET_LABEL} neuron map content hash mismatch: {nodes_content}')
    import pyarrow.feather as feather
    table = feather.read_table(nodes_path, columns=['node_index', 'source_id']).to_pandas()
    with np.load(graph, allow_pickle=False) as data:
        ids, ptr, post, weight = data['ids'], data['ptr'], data['post'], data['weight']
        if (len(ids) != pins['neurons'] or int(ptr[-1]) != pins['edges'] or ptr[0] != 0
                or np.any(np.diff(ptr) < 0) or len(post) != ptr[-1] or len(weight) != len(post)
                or np.any(post < 0) or np.any(post >= len(ids)) or not np.isfinite(weight).all()):
            raise Unavailable(f'{DATASET_LABEL}: CSR arrays fail validation or differ from the pinned counts')
    if not np.array_equal(ids.astype(np.uint64), table.source_id.to_numpy(dtype=np.uint64)):
        raise Unavailable(f'{DATASET_LABEL}: graph IDs do not match neurons.feather source_id order')
    if not np.array_equal(table.node_index.to_numpy(), np.arange(len(table))):
        raise Unavailable(f'{DATASET_LABEL}: node_index is not 0..n-1')
    ids_sha = _ids_digest(ids)
    if ids_sha != pins['ids_sha256']:
        raise Unavailable(f'{DATASET_LABEL}: ID digest {ids_sha} differs from pin')
    data_sha = {f'{name} (pinned source)': entry['sha256'] for name, entry in pins['source_sha256'].items()}
    data_sha['method'] = pins['method_sha256']
    return GraphIdentity(
        dataset=DATASET_ID, synthetic=False, graph_path=str(graph.resolve()),
        graph_path_source='flywire', graph_sha256=graph_content,
        neuron_map_path=str(nodes_path.resolve()), neuron_map_sha256=nodes_content,
        io_map_sha256=sha256_json(IO_DECLARATION), neurons=len(ids), edges=int(ptr[-1]),
        ids_sha256=ids_sha, data_sha256=data_sha,
        label=(f'{IDENTITY_LABEL}; sign policy {SIGN_POLICY}; weight = count x sign x {SYNAPTIC_SCALE} '
               '(same scale as MaleCNS, NOT an equivalent calibration)'),
        graph_content_sha256=graph_content, neuron_map_content_sha256=nodes_content,
        graph_file_sha256=sha256_file(graph), neuron_map_file_sha256=sha256_file(nodes_path))


def load_shared(graph_dir=None, connectome_dir=None, *, dynamics: Optional[str] = None, pins=None):
    """Verified FlyWire SharedGraph, with the v3 transmitter policy applied in
    memory when the dynamics is v3 (the same rule as MaleCNS, on FlyWire labels)."""
    from experiment_registry import SharedGraph
    from .graph_identity import active_dynamics_version
    identity = verify_flywire_graph(graph_dir, connectome_dir, pins=pins)
    with np.load(identity.graph_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ('ptr', 'post', 'weight', 'ids')}
    shared = SharedGraph(arrays, identity, {})
    if (dynamics or active_dynamics_version()) == 'v3':
        from .transmitter_policy import apply_to_shared
        shared, _ = apply_to_shared(shared, connectome_dir=resolve_connectome_dir(connectome_dir))
    return shared


def pins_from_build(graph_dir=None, connectome_dir=None, lock: Optional[dict] = None) -> dict:
    """Pin values of a fresh build (used once, by the maintainer, to write the pins file)."""
    from .graph_identity import _ids_digest, feather_content_sha256, npz_content_sha256, sha256_file
    lock = lock or load_lock()
    gdir, cdir = resolve_graph_dir(graph_dir), resolve_connectome_dir(connectome_dir)
    with np.load(gdir / 'graph.npz', allow_pickle=False) as data:
        ids, ptr = data['ids'], data['ptr']
    return dict(
        dataset_id=DATASET_ID, dataset=DATASET_LABEL, coverage=COVERAGE,
        prepared_by=f'brainlab.flywire (sign policy {SIGN_POLICY}, synaptic_scale {SYNAPTIC_SCALE})',
        graph_content_sha256=npz_content_sha256(gdir / 'graph.npz'),
        neuron_map_content_sha256=feather_content_sha256(cdir / 'normalized/neurons.feather'),
        graph_sha256_reference_bytes=sha256_file(gdir / 'graph.npz'),
        ids_sha256=_ids_digest(ids), neurons=int(len(ids)), edges=int(ptr[-1]),
        method_sha256=_sha256_json(METHOD),
        source_sha256={name: dict(sha256=e['sha256'], bytes=e['bytes']) for name, e in lock['files'].items()},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m brainlab.flywire', description=__doc__.split('\n\n')[0],
                                     epilog=ATTRIBUTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['download', 'import', 'verify', 'pins'])
    parser.add_argument('--source-dir', type=Path, help='directory holding the pinned source files')
    parser.add_argument('--connectome-dir', type=Path)
    parser.add_argument('--graph-dir', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'download':
        download(args.source_dir or args.connectome_dir)
    elif args.command == 'import':
        print(ATTRIBUTION + '\n', flush=True)
        report = import_flywire(args.source_dir, args.connectome_dir, args.graph_dir)
        print(json.dumps(dict(graph=report['graph'], edges=report['edges'], nodes={
            k: v for k, v in report['nodes'].items() if not k.endswith('_ids')}), indent=2))
    elif args.command == 'verify':
        print(json.dumps(verify_flywire_graph(args.graph_dir, args.connectome_dir).to_dict(), indent=2))
    else:
        print(json.dumps(pins_from_build(args.graph_dir, args.connectome_dir), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
