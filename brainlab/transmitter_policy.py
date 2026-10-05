"""Declared transmitter-class policy for the fast synaptic weights (LIF v3).

``brainlab/transmitters.py`` assigns one coarse fast sign per neuron when the
graph is prepared: acetylcholine ``+``; GABA, glutamate, histamine ``-``;
everything else (co-transmitter conflicts, missing labels, ``unclear``, and the
aminergic classes dopamine / octopamine / serotonin) the explicit
``ambiguous_sign``, which the pinned MaleCNS graph was built with as ``+1``.

That leaves every dopaminergic, octopaminergic and serotonergic neuron acting
as a **fast ionotropic excitatory** cell in the LIF proxy, which is not what
those transmitters do: their *Drosophila* receptors are G-protein coupled and
act on a timescale the 5 ms synaptic kernel of this engine does not represent
(``docs/LIF_DYNAMICS_SPEC.md`` §4.3, and the confound recorded in
``docs/WP6_PLASTICITY_SPEC.md`` §3.5).

This module applies a **declared, switchable** policy to the prepared CSR
weight array, in memory.  The pinned ``graph.npz`` is never rewritten: v1 and
v2 keep loading exactly the bytes they always loaded, and a policy-transformed
graph is given its own ``graph_sha256`` and its own conspicuous label, so a
checkpoint written under one policy is refused by the other at the registry
level as well as at the Brain level.

Nothing here is receptor physiology.  It is a coarser statement than the one it
replaces, and it is declared as such.

WP7 graph options (``docs/WP7_MB_LEARNING_SPEC.md`` §2, design decisions D3 and
D5 of ``docs/design/mb_learning.md``).  Both are opt-in, exist only under the
v3 policy, and leave every default weight byte-identical:

* ``kc_kc`` — ``'released'`` (default) keeps the Kenyon-cell -> Kenyon-cell
  fast weights; ``'modulatory-only'`` gives every KC->KC edge zero fast weight
  (D3: these synapses are axo-axonal and act through mAChR-B, not as fast
  somatic excitation); ``'modulatory-only-except-calyx'`` zeroes them except
  for the declared number of calyx synapses per edge, which must be supplied
  from the synapse-level ROI tables (the runtime graph does not carry ROIs).
* ``dpm`` — ``'released'`` (default) leaves the DPM neurons under their
  released prediction (dopamine, hence no fast weight under v3);
  ``'gaba'`` relabels them GABAergic (D5), so their out-edges carry the
  released contact count with an inhibitory sign.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# Declared aminergic / modulatory classes.  These get NO fast synaptic weight
# under the v3 policy; they are available to a plasticity rule as a modulatory
# signal m(t) and in no other way.
MODULATORY_TRANSMITTERS = ('dopamine', 'octopamine', 'serotonin')

# Labels that state nothing about the transmitter.  ``unclear`` is the released
# classifier's own abstention (Eckstein et al. 2024); a missing/NaN entry is an
# absent row.  These are handled by ``unclear_mode``, SEPARATELY from the
# identified aminergic populations above.
UNRESOLVED_TRANSMITTERS = ('unclear', 'unassigned', 'nan', 'none', '')

UNCLEAR_MODES = ('excitatory', 'zero', 'exclude')

POLICY_LEGACY = 'v1v2-as-prepared'
POLICY_V3 = 'v3-modulatory-only'
POLICIES = (POLICY_LEGACY, POLICY_V3)

# WP7 graph options (D3, D5).  The first entry of each tuple is the default and
# reproduces the plain v3 weights exactly.
KC_KC_RELEASED = 'released'
KC_KC_MODULATORY_ONLY = 'modulatory-only'
KC_KC_EXCEPT_CALYX = 'modulatory-only-except-calyx'
KC_KC_MODES = (KC_KC_RELEASED, KC_KC_MODULATORY_ONLY, KC_KC_EXCEPT_CALYX)
DPM_RELEASED = 'released'
DPM_GABA = 'gaba'
DPM_MODES = (DPM_RELEASED, DPM_GABA)
# Population queries on the released ``type`` column, the same regular
# expressions as scripts/mb_circuit_census.py (docs/design/mb_learning.md §1.1).
KC_TYPE_PATTERN = r'^KC'
DPM_TYPE_PATTERN = r'^DPM$'
# Prepared-graph weight per synaptic contact (brainlab/prepare.py).
SYNAPTIC_SCALE = 0.275


def normalise(transmitters: Iterable) -> np.ndarray:
    """Lower-cased transmitter label per neuron, NaN mapped to ``'nan'``."""
    return np.array([str(value).strip().lower() for value in transmitters], dtype=object)


def _row_mask(ptr: np.ndarray, selected: np.ndarray, n_edges: int) -> np.ndarray:
    """Edge mask for every out-edge of the selected presynaptic neurons."""
    mask = np.zeros(n_edges, dtype=bool)
    for index in np.flatnonzero(selected):
        mask[ptr[index]:ptr[index + 1]] = True
    return mask


def mb_options_are_default(kc_kc: str = KC_KC_RELEASED, dpm: str = DPM_RELEASED) -> bool:
    return kc_kc == KC_KC_RELEASED and dpm == DPM_RELEASED


def _type_mask(cell_types: Sequence, pattern: str) -> np.ndarray:
    rx = re.compile(pattern)
    return np.array([bool(rx.search('' if t is None or (isinstance(t, float) and t != t) else str(t)))
                     for t in cell_types], dtype=bool)


def edge_sources(ptr: np.ndarray) -> np.ndarray:
    """Presynaptic node index of every CSR edge."""
    ptr = np.asarray(ptr, dtype=np.int64)
    return np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))


def calyx_edges_from_pairs(ptr: np.ndarray, post: np.ndarray, pre_nodes, post_nodes,
                           counts) -> Tuple[np.ndarray, np.ndarray]:
    """Map (pre node, post node, calyx synapse count) rows onto CSR edge indices.

    Every pair must be an edge of the graph; a missing pair is an error, never
    silently dropped.  Returns ``(edges, counts)`` sorted by edge index.
    """
    pre_nodes = np.asarray(pre_nodes, dtype=np.int64)
    post_nodes = np.asarray(post_nodes, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if not (len(pre_nodes) == len(post_nodes) == len(counts)):
        raise ValueError('calyx table columns differ in length')
    n = len(ptr) - 1
    src = edge_sources(ptr)
    candidates = np.flatnonzero(np.isin(src, np.unique(pre_nodes)))
    keys = src[candidates] * n + np.asarray(post, dtype=np.int64)[candidates]
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    wanted = pre_nodes * n + post_nodes
    pos = np.searchsorted(keys, wanted)
    found = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == wanted)
    if not found.all():
        missing = int((~found).sum())
        raise ValueError(f'{missing} calyx (pre, post) pairs are not edges of this graph')
    edges = candidates[order][pos]
    if len(np.unique(edges)) != len(edges):
        raise ValueError('calyx table lists an edge more than once')
    sort = np.argsort(edges)
    return edges[sort], counts[sort]


def load_kc_kc_calyx(path, ptr: np.ndarray, post: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Read a KC->KC calyx sidecar (``.npz`` with ``pre_index``, ``post_index``,
    ``calyx_synapses``; spec §2.3) and return ``(edges, counts, sha256)``."""
    from pathlib import Path
    raw = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as data:
        edges, counts = calyx_edges_from_pairs(ptr, post, data['pre_index'], data['post_index'],
                                               data['calyx_synapses'])
    return edges, counts, hashlib.sha256(raw).hexdigest()


def apply_policy(ptr: np.ndarray, post: np.ndarray, weight: np.ndarray,
                 transmitters: Sequence, *, policy: str = POLICY_V3,
                 unclear_mode: str = 'excitatory', kc_kc: str = KC_KC_RELEASED,
                 dpm: str = DPM_RELEASED, cell_types: Optional[Sequence] = None,
                 kc_kc_calyx: Optional[Tuple[np.ndarray, np.ndarray]] = None
                 ) -> tuple[np.ndarray, dict]:
    """Return ``(new_weight, report)`` for the declared transmitter policy.

    ``policy=POLICY_LEGACY`` returns the weights unchanged (v1 / v2).

    ``policy=POLICY_V3``:

    * every out-edge of a dopaminergic, octopaminergic or serotonergic neuron
      gets weight 0 — the neuron is modulatory-only;
    * the ``unclear`` / unlabelled neurons are handled by ``unclear_mode``:
      ``'excitatory'`` leaves them exactly as v1 and v2 had them (the declared
      ambiguous sign ``+1``), ``'zero'`` removes their fast output weight, and
      ``'exclude'`` removes them from the simulated network altogether by
      zeroing both their out-edges and every edge onto them, without changing
      the node index space (no anatomy is deleted from the graph file).

    WP7 options (v3 only; defaults change nothing, see the module docstring):
    ``kc_kc`` and ``dpm`` need ``cell_types`` (released ``type`` per neuron,
    prepared-graph node order).  ``kc_kc='modulatory-only-except-calyx'`` needs
    ``kc_kc_calyx = (edge_indices, calyx_synapse_counts)``; each listed edge
    must be KC->KC and keeps ``count * 0.275`` with its released sign, exactly
    as ``brainlab/prepare.py`` computes a weight.  Precedence: the unclear-mode
    drop wins over both options.

    No edge is ever removed from the arrays and no neuron index changes, so the
    CSR structure, ``ptr``, ``post`` and ``ids`` are identical to the pinned
    graph's.  Only ``weight`` differs, and by how much is reported.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown transmitter policy {policy!r}; declared: {list(POLICIES)}')
    if unclear_mode not in UNCLEAR_MODES:
        raise ValueError(f'Unknown unclear_mode {unclear_mode!r}; declared: {list(UNCLEAR_MODES)}')
    if kc_kc not in KC_KC_MODES:
        raise ValueError(f'Unknown kc_kc mode {kc_kc!r}; declared: {list(KC_KC_MODES)}')
    if dpm not in DPM_MODES:
        raise ValueError(f'Unknown dpm mode {dpm!r}; declared: {list(DPM_MODES)}')
    mb_default = mb_options_are_default(kc_kc, dpm)
    if not mb_default and policy != POLICY_V3:
        raise ValueError('kc_kc / dpm options are declared only for the v3 policy')
    if (kc_kc == KC_KC_EXCEPT_CALYX) != (kc_kc_calyx is not None):
        raise ValueError("kc_kc_calyx is required by, and only by, kc_kc='modulatory-only-except-calyx'")
    labels = normalise(transmitters)
    n = len(labels)
    if len(ptr) != n + 1:
        raise ValueError(f'Transmitter table has {n} rows but ptr implies {len(ptr) - 1} neurons')
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    report = dict(policy=policy, unclear_mode=unclear_mode,
                  modulatory_transmitters=list(MODULATORY_TRANSMITTERS),
                  unresolved_transmitters=list(UNRESOLVED_TRANSMITTERS),
                  neurons=n, edges=int(len(weight)))
    if policy == POLICY_LEGACY:
        report.update(edges_zeroed=0, neurons_modulatory=0, neurons_unresolved=0, changed=False)
        report.update(_balance(weight))
        return weight.copy(), report

    new = weight.copy()
    mb_report = {}
    kc_drop = np.zeros(len(new), dtype=bool)
    if not mb_default:
        if cell_types is None or len(cell_types) != n:
            raise ValueError('kc_kc / dpm options need one cell type per neuron (cell_types)')
        mb_report = dict(kc_kc=kc_kc, dpm=dpm, kc_type_pattern=KC_TYPE_PATTERN,
                         dpm_type_pattern=DPM_TYPE_PATTERN)
        if dpm == DPM_GABA:
            dpm_mask = _type_mask(cell_types, DPM_TYPE_PATTERN)
            released = sorted(set(labels[dpm_mask].astype(str)))
            labels = labels.copy()
            labels[dpm_mask] = 'gaba'
            rows = _row_mask(ptr, dpm_mask, len(new))
            new[rows] = -np.abs(weight[rows])
            mb_report.update(dpm_neurons=int(dpm_mask.sum()), dpm_released_labels=released,
                             dpm_out_edges_relabelled=int(rows.sum()))
        if kc_kc != KC_KC_RELEASED:
            kc = _type_mask(cell_types, KC_TYPE_PATTERN)
            src = edge_sources(ptr)
            kc_edges = kc[src] & kc[np.asarray(post)]
            kc_drop = kc_edges.copy()
            mb_report.update(kc_neurons=int(kc.sum()), kc_kc_edges=int(kc_edges.sum()))
            if kc_kc == KC_KC_EXCEPT_CALYX:
                edges = np.asarray(kc_kc_calyx[0], dtype=np.int64)
                counts = np.asarray(kc_kc_calyx[1], dtype=np.int64)
                if len(edges) != len(counts) or len(np.unique(edges)) != len(edges):
                    raise ValueError('kc_kc_calyx must list each edge once, with one count')
                if len(edges) and (edges.min() < 0 or edges.max() >= len(new)):
                    raise ValueError('kc_kc_calyx edge index out of range')
                if not kc_edges[edges].all():
                    raise ValueError('kc_kc_calyx lists an edge that is not KC->KC')
                if (counts < 0).any():
                    raise ValueError('kc_kc_calyx counts must be >= 0')
                released_counts = np.rint(np.abs(weight[edges].astype(np.float64)) / SYNAPTIC_SCALE)
                if (counts > released_counts).any():
                    raise ValueError('kc_kc_calyx count exceeds the released contact count of its edge')
                sign = np.sign(weight[edges]).astype(np.int8)
                # Same arithmetic as brainlab/prepare.py: count(float32) * sign(int8) * 0.275.
                new[edges] = (counts.astype(np.float32) * sign * SYNAPTIC_SCALE).astype(np.float32)
                kc_drop[edges] = False
                mb_report.update(kc_kc_calyx_edges_kept=int((counts > 0).sum()),
                                 kc_kc_calyx_synapses_kept=int(counts.sum()))
            mb_report['kc_kc_edges_zeroed'] = int(kc_drop.sum())
    modulatory = np.isin(labels.astype(str), MODULATORY_TRANSMITTERS)
    unresolved = np.isin(labels.astype(str), UNRESOLVED_TRANSMITTERS)
    drop = _row_mask(ptr, modulatory, len(new)) | kc_drop
    if unclear_mode in ('zero', 'exclude'):
        drop |= _row_mask(ptr, unresolved, len(new))
    if unclear_mode == 'exclude':
        drop |= unresolved[post]
    new[drop] = 0.0
    changed = bool(drop.any()) or not np.array_equal(new, weight)
    report.update(mb_report)
    report.update(edges_zeroed=int(drop.sum()),
                  neurons_modulatory=int(modulatory.sum()),
                  neurons_unresolved=int(unresolved.sum()),
                  modulatory_out_edges=int(_row_mask(ptr, modulatory, len(new)).sum()),
                  unresolved_out_edges=int(_row_mask(ptr, unresolved, len(new)).sum()),
                  changed=changed)
    report.update(_balance(new))
    report['weight_sha256'] = hashlib.sha256(np.ascontiguousarray(new).tobytes()).hexdigest()
    return new, report


def _balance(weight: np.ndarray) -> dict:
    """Total excitatory / inhibitory weight and the resulting fixed points."""
    from .engine import E_INH_MV, G_UNIT_EXC_V3, G_UNIT_INH_V3, G_UNIT_PER_WEIGHT, V_THRESHOLD_MV
    w = np.asarray(weight, dtype=np.float64)
    w_exc = float(w[w > 0].sum())
    w_inh = float(-w[w < 0].sum())
    out = dict(total_excitatory_weight=w_exc, total_inhibitory_weight=w_inh)
    for name, (ge, gi) in (('v2_quanta', (G_UNIT_PER_WEIGHT, G_UNIT_PER_WEIGHT)),
                           ('v3_quanta', (G_UNIT_EXC_V3, G_UNIT_INH_V3))):
        r = (w_inh * gi) / (w_exc * ge) if w_exc > 0 else float('inf')
        fixed_point = E_INH_MV * r / (1.0 + r)
        out[name] = dict(conductance_ratio_g_inh_over_g_exc=r,
                         high_conductance_fixed_point_mV=fixed_point,
                         subthreshold=bool(fixed_point <= V_THRESHOLD_MV),
                         margin_below_threshold_mV=float(V_THRESHOLD_MV - fixed_point))
    return out


def describe(policy: str = POLICY_V3, unclear_mode: str = 'excitatory', *,
             kc_kc: str = KC_KC_RELEASED, dpm: str = DPM_RELEASED) -> dict:
    """The declaration that goes into a manifest, without touching a graph.

    The WP7 keys appear only when an option differs from its default, so every
    existing manifest declaration is unchanged."""
    out = dict(policy=policy, unclear_mode=unclear_mode,
               modulatory_transmitters=list(MODULATORY_TRANSMITTERS),
               unresolved_transmitters=list(UNRESOLVED_TRANSMITTERS),
               spec='docs/LIF_DYNAMICS_SPEC.md#4',
               note='coarse transmitter-class policy; not receptor physiology')
    if not mb_options_are_default(kc_kc, dpm):
        out.update(kc_kc=kc_kc, dpm=dpm, wp7_spec='docs/WP7_MB_LEARNING_SPEC.md#2')
    return out


def graph_variant_tag(kc_kc: str = KC_KC_RELEASED, dpm: str = DPM_RELEASED,
                      calyx_sha256: Optional[str] = None) -> str:
    """Suffix for the dataset name of a WP7 graph variant ('' for the defaults)."""
    if mb_options_are_default(kc_kc, dpm):
        return ''
    tag = f'+mb(kc_kc={kc_kc},dpm={dpm}'
    if calyx_sha256:
        tag += f',calyx={calyx_sha256[:12]}'
    return tag + ')'


def load_cell_types(connectome_dir=None) -> np.ndarray:
    """Released cell type per neuron, in prepared-graph node order ('' if none)."""
    import pyarrow.feather as feather
    from .graph_identity import resolve_connectome_dir
    cdir, _ = resolve_connectome_dir(connectome_dir)
    table = feather.read_table(cdir / 'normalized/neurons.feather',
                               columns=['node_index', 'cell_type']).to_pandas()
    if not np.array_equal(table.node_index.to_numpy(), np.arange(len(table))):
        raise ValueError('neurons.feather node_index is not 0..n-1; cannot align cell types')
    return np.array(['' if v is None or (isinstance(v, float) and v != v) else str(v)
                     for v in table.cell_type], dtype=object)


def load_transmitters(connectome_dir=None) -> np.ndarray:
    """Per-neuron transmitter labels, in prepared-graph node order."""
    import pyarrow.feather as feather
    from .graph_identity import resolve_connectome_dir
    cdir, _ = resolve_connectome_dir(connectome_dir)
    table = feather.read_table(cdir / 'normalized/neurons.feather',
                               columns=['node_index', 'neurotransmitter']).to_pandas()
    if not np.array_equal(table.node_index.to_numpy(), np.arange(len(table))):
        raise ValueError('neurons.feather node_index is not 0..n-1; cannot align transmitters')
    return normalise(table.neurotransmitter)


def apply_to_shared(shared, *, policy: str = POLICY_V3, unclear_mode: str = 'excitatory',
                    connectome_dir=None, kc_kc: str = KC_KC_RELEASED, dpm: str = DPM_RELEASED,
                    kc_kc_calyx_file=None, cell_types: Optional[Sequence] = None,
                    transmitters: Optional[Sequence] = None):
    """Return ``(new_shared, report)``: the same graph with v3 fast weights.

    The returned :class:`experiment_registry.SharedGraph` carries a NEW
    ``graph_sha256`` and a label naming the policy, so the registry refuses a
    checkpoint written under a different policy and no manifest can confuse the
    two.  The pinned graph file is not touched.

    ``kc_kc``, ``dpm`` and ``kc_kc_calyx_file`` select a WP7 graph variant
    (spec §2).  A non-default variant has different weights, hence its own
    ``graph_sha256``, and its dataset name and label say which variant it is.
    ``cell_types`` / ``transmitters`` default to the connectome tables; they
    exist so tests can pass small synthetic tables.
    """
    from experiment_registry import SharedGraph
    from .graph_identity import GraphIdentity
    labels = load_transmitters(connectome_dir) if transmitters is None else normalise(transmitters)
    arrays = shared.arrays
    calyx, calyx_sha = None, None
    if kc_kc_calyx_file is not None:
        edges, counts, calyx_sha = load_kc_kc_calyx(kc_kc_calyx_file, arrays['ptr'], arrays['post'])
        calyx = (edges, counts)
    if not mb_options_are_default(kc_kc, dpm) and cell_types is None:
        cell_types = load_cell_types(connectome_dir)
    new_weight, report = apply_policy(arrays['ptr'], arrays['post'], arrays['weight'], labels,
                                      policy=policy, unclear_mode=unclear_mode, kc_kc=kc_kc, dpm=dpm,
                                      cell_types=cell_types, kc_kc_calyx=calyx)
    if calyx_sha is not None:
        report['kc_kc_calyx_file'] = str(kc_kc_calyx_file)
        report['kc_kc_calyx_sha256'] = calyx_sha
    variant = graph_variant_tag(kc_kc, dpm, calyx_sha)
    if not report['changed']:
        return shared, report
    new_arrays = dict(ptr=arrays['ptr'], post=arrays['post'],
                      weight=np.ascontiguousarray(new_weight), ids=arrays['ids'])
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(np.ascontiguousarray(new_arrays[key]).tobytes())
    base = shared.identity
    variant_label = ''
    if variant:
        variant_label = (f'; WP7 graph variant kc_kc={kc_kc}, dpm={dpm}'
                         + (f', calyx sidecar {calyx_sha}' if calyx_sha else ''))
    identity = GraphIdentity(
        dataset=f'{base.dataset}+{policy}({unclear_mode}){variant}', synthetic=base.synthetic,
        graph_path=base.graph_path,
        graph_path_source=f'{policy}(unclear={unclear_mode}){variant} applied in memory to {base.graph_sha256}',
        graph_sha256=digest.hexdigest(), neuron_map_path=base.neuron_map_path,
        neuron_map_sha256=base.neuron_map_sha256, io_map_sha256=base.io_map_sha256,
        neurons=base.neurons, edges=base.edges, ids_sha256=base.ids_sha256,
        data_sha256=dict(base.data_sha256),
        label=f'{base.label} with DECLARED transmitter policy {policy} '
              f'(unclear={unclear_mode}); aminergic neurons carry no fast weight{variant_label}')
    report['graph_sha256'] = identity.graph_sha256
    report['base_graph_sha256'] = base.graph_sha256
    return SharedGraph(new_arrays, identity, shared.io_map), report
