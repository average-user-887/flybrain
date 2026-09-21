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
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

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


def normalise(transmitters: Iterable) -> np.ndarray:
    """Lower-cased transmitter label per neuron, NaN mapped to ``'nan'``."""
    return np.array([str(value).strip().lower() for value in transmitters], dtype=object)


def _row_mask(ptr: np.ndarray, selected: np.ndarray, n_edges: int) -> np.ndarray:
    """Edge mask for every out-edge of the selected presynaptic neurons."""
    mask = np.zeros(n_edges, dtype=bool)
    for index in np.flatnonzero(selected):
        mask[ptr[index]:ptr[index + 1]] = True
    return mask


def apply_policy(ptr: np.ndarray, post: np.ndarray, weight: np.ndarray,
                 transmitters: Sequence, *, policy: str = POLICY_V3,
                 unclear_mode: str = 'excitatory') -> tuple[np.ndarray, dict]:
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

    No edge is ever removed from the arrays and no neuron index changes, so the
    CSR structure, ``ptr``, ``post`` and ``ids`` are identical to the pinned
    graph's.  Only ``weight`` differs, and by how much is reported.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown transmitter policy {policy!r}; declared: {list(POLICIES)}')
    if unclear_mode not in UNCLEAR_MODES:
        raise ValueError(f'Unknown unclear_mode {unclear_mode!r}; declared: {list(UNCLEAR_MODES)}')
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

    modulatory = np.isin(labels.astype(str), MODULATORY_TRANSMITTERS)
    unresolved = np.isin(labels.astype(str), UNRESOLVED_TRANSMITTERS)
    new = weight.copy()
    drop = _row_mask(ptr, modulatory, len(new))
    if unclear_mode in ('zero', 'exclude'):
        drop |= _row_mask(ptr, unresolved, len(new))
    if unclear_mode == 'exclude':
        drop |= unresolved[post]
    new[drop] = 0.0
    report.update(edges_zeroed=int(drop.sum()),
                  neurons_modulatory=int(modulatory.sum()),
                  neurons_unresolved=int(unresolved.sum()),
                  modulatory_out_edges=int(_row_mask(ptr, modulatory, len(new)).sum()),
                  unresolved_out_edges=int(_row_mask(ptr, unresolved, len(new)).sum()),
                  changed=bool(drop.any()))
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


def describe(policy: str = POLICY_V3, unclear_mode: str = 'excitatory') -> dict:
    """The declaration that goes into a manifest, without touching a graph."""
    return dict(policy=policy, unclear_mode=unclear_mode,
                modulatory_transmitters=list(MODULATORY_TRANSMITTERS),
                unresolved_transmitters=list(UNRESOLVED_TRANSMITTERS),
                spec='docs/LIF_DYNAMICS_SPEC.md#4',
                note='coarse transmitter-class policy; not receptor physiology')


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
                    connectome_dir=None):
    """Return ``(new_shared, report)``: the same graph with v3 fast weights.

    The returned :class:`experiment_registry.SharedGraph` carries a NEW
    ``graph_sha256`` and a label naming the policy, so the registry refuses a
    checkpoint written under a different policy and no manifest can confuse the
    two.  The pinned graph file is not touched.
    """
    from experiment_registry import SharedGraph
    from .graph_identity import GraphIdentity
    labels = load_transmitters(connectome_dir)
    arrays = shared.arrays
    new_weight, report = apply_policy(arrays['ptr'], arrays['post'], arrays['weight'], labels,
                                      policy=policy, unclear_mode=unclear_mode)
    if not report['changed']:
        return shared, report
    new_arrays = dict(ptr=arrays['ptr'], post=arrays['post'],
                      weight=np.ascontiguousarray(new_weight), ids=arrays['ids'])
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(np.ascontiguousarray(new_arrays[key]).tobytes())
    base = shared.identity
    identity = GraphIdentity(
        dataset=f'{base.dataset}+{policy}({unclear_mode})', synthetic=base.synthetic,
        graph_path=base.graph_path,
        graph_path_source=f'{policy}(unclear={unclear_mode}) applied in memory to {base.graph_sha256}',
        graph_sha256=digest.hexdigest(), neuron_map_path=base.neuron_map_path,
        neuron_map_sha256=base.neuron_map_sha256, io_map_sha256=base.io_map_sha256,
        neurons=base.neurons, edges=base.edges, ids_sha256=base.ids_sha256,
        data_sha256=dict(base.data_sha256),
        label=f'{base.label} with DECLARED transmitter policy {policy} '
              f'(unclear={unclear_mode}); aminergic neurons carry no fast weight')
    report['graph_sha256'] = identity.graph_sha256
    report['base_graph_sha256'] = base.graph_sha256
    return SharedGraph(new_arrays, identity, shared.io_map), report
