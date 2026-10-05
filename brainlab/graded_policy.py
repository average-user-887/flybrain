"""Declared non-spiking (graded) cell classes for LIF dynamics v4.

``docs/LIF_DYNAMICS_SPEC.md`` §7.5 and the locked declaration
``docs/receipts/graded_transmission_v4_declaration.locked.md`` (sha256
``2634c824476c9b799bd2c0c656040255360cd10f8837e2d874a97af5f939872e``) name the
classes.  This module resolves them against the released tables and nothing
else.

Resolution is by ``cell_type`` and ``superclass`` strings from
``normalized/neurons.feather``, never by dataframe row order.  The resolved
node set is hashed (:func:`graded_set_sha256`) and that digest goes into every
manifest and receipt, so a checkpoint written under one graded declaration is
refused by a run under another.

Two declared policies:

* ``v4-optic-lobe-graded`` (PRIMARY) -- tier 1 (named cell types with direct
  evidence of graded signalling) plus tier 2 (the declared extension to every
  remaining ``ol_intrinsic`` / ``ol_sensory`` neuron, an ENGINEERING
  ASSUMPTION), plus the lobula plate tangential cells, which are annotated
  ``visual_projection`` but are not columnar.
* ``v4-tier1-only`` -- the declared sensitivity arm S1: the named cell types
  only.

``none`` selects no graded class at all, which makes v4 bit-identical to v3
(spec §7.7 P2).  Lobula *columnar* visual projection neurons are deliberately
excluded from both policies because they spike (von Reyn et al. 2017).

Nothing here is cell physiology measured by this project.  It is a declared
class list with its evidence written down, and it is switchable.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

# --- Tier 1: named cell types with direct evidence (spec §7.5) --------------
PHOTORECEPTORS = ('R1-R6', 'R7d', 'R7p', 'R7y', 'R8d', 'R8p', 'R8y',
                  'R7_unclear', 'R8_unclear', 'R7R8_unclear')
LAMINA = ('L1', 'L2', 'L3', 'L4', 'L5', 'Lai')
MEDULLA_RECORDED = ('Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9')
AMACRINE = ('CT1',)
T4T5 = ('T4a', 'T4b', 'T4c', 'T4d', 'T4_unclear',
        'T5a', 'T5b', 'T5c', 'T5d', 'T5a_unclear')
# Lobula plate tangential cells.  Graded with at most small irregular spikelets
# (Joesch 2008; Schnell 2010; Fujiwara 2017).  H1 is annotated ol_intrinsic;
# the rest are annotated visual_projection but are not columnar, so they are
# named explicitly rather than swept in by superclass.
LPTC = ('HSN', 'HSE', 'HSS', 'HST', 'H1', 'H2', 'VS', 'VST1', 'VST2', 'VSm')

TIER1_TYPES = PHOTORECEPTORS + LAMINA + MEDULLA_RECORDED + AMACRINE + T4T5 + LPTC

# --- Tier 2: declared extension by anatomical class (ENGINEERING ASSUMPTION)
TIER2_SUPERCLASSES = ('ol_intrinsic', 'ol_sensory')

POLICY_NONE = 'none'
POLICY_PRIMARY = 'v4-optic-lobe-graded'
POLICY_TIER1 = 'v4-tier1-only'
POLICIES = (POLICY_NONE, POLICY_PRIMARY, POLICY_TIER1)


def describe(policy: str = POLICY_PRIMARY) -> dict:
    """The declaration that goes into a manifest, without touching a graph."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown graded policy {policy!r}; declared: {list(POLICIES)}')
    out = dict(
        policy=policy,
        spec='docs/LIF_DYNAMICS_SPEC.md#7',
        declaration_lock='docs/receipts/graded_transmission_v4_declaration.locked.md',
        declaration_sha256='2634c824476c9b799bd2c0c656040255360cd10f8837e2d874a97af5f939872e',
        tier1_cell_types=list(TIER1_TYPES),
        note='declared non-spiking classes; a coarse class list with its evidence, '
             'not physiology measured here',
    )
    if policy == POLICY_PRIMARY:
        out['tier2_superclasses'] = list(TIER2_SUPERCLASSES)
        out['tier2_status'] = 'ENGINEERING ASSUMPTION (spec §7.5)'
        out['excluded'] = 'lobula columnar visual_projection neurons (they spike; von Reyn 2017)'
    elif policy == POLICY_TIER1:
        out['tier2_superclasses'] = []
        out['arm'] = 'S1, declared sensitivity arm; cannot change the primary verdict'
    else:
        out['effect'] = 'no class declared graded; v4 is then bit-identical to v3'
    return out


def graded_mask(cell_types: Sequence, superclasses: Sequence,
                *, policy: str = POLICY_PRIMARY) -> np.ndarray:
    """Per-neuron uint8 mask of the declared graded classes."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown graded policy {policy!r}; declared: {list(POLICIES)}')
    ct = np.asarray([str(c) for c in cell_types], dtype=object)
    sc = np.asarray([str(s) for s in superclasses], dtype=object)
    if len(ct) != len(sc):
        raise ValueError('cell_types and superclasses must have one entry per neuron')
    mask = np.zeros(len(ct), dtype=bool)
    if policy == POLICY_NONE:
        return mask.astype(np.uint8)
    mask |= np.isin(ct.astype(str), TIER1_TYPES)
    if policy == POLICY_PRIMARY:
        mask |= np.isin(sc.astype(str), TIER2_SUPERCLASSES)
    return mask.astype(np.uint8)


def graded_set_sha256(mask: np.ndarray) -> str:
    """Digest of the resolved graded node set: the re-pin for a class change."""
    idx = np.flatnonzero(np.asarray(mask))
    return hashlib.sha256(np.ascontiguousarray(idx, dtype='<i8').tobytes()).hexdigest()


def load_cell_classes(connectome_dir=None):
    """``(cell_type, superclass)`` per neuron, in prepared-graph node order."""
    import pyarrow.feather as feather

    from .graph_identity import resolve_connectome_dir
    cdir, _ = resolve_connectome_dir(connectome_dir)
    table = feather.read_table(cdir / 'normalized/neurons.feather',
                               columns=['node_index', 'cell_type', 'superclass']).to_pandas()
    if not np.array_equal(table.node_index.to_numpy(), np.arange(len(table))):
        raise ValueError('neurons.feather node_index is not 0..n-1; cannot align cell classes')
    return table.cell_type.fillna('').to_numpy(), table.superclass.fillna('').to_numpy()


def resolve(n: int, *, policy: str = POLICY_PRIMARY, connectome_dir=None,
            required: bool = True) -> tuple[np.ndarray, dict]:
    """``(mask, report)`` for ``policy`` on a graph of ``n`` neurons.

    ``required=False`` returns an empty mask when the tables are missing, which
    is what synthetic test graphs get; the real graph without its tables is
    refused rather than silently run with no graded class.
    """
    report = describe(policy)
    report['neurons'] = int(n)
    if policy == POLICY_NONE:
        mask = np.zeros(n, dtype=np.uint8)
        report.update(graded_neurons=0, graded_set_sha256=graded_set_sha256(mask))
        return mask, report
    from .graph_identity import GraphUnavailable
    try:
        ct, sc = load_cell_classes(connectome_dir)
    except (FileNotFoundError, OSError, GraphUnavailable, ImportError, ValueError) as exc:
        if required:
            raise GraphUnavailable(
                'LIF v4 needs the cell-type / superclass table '
                '(connectome_data/.../normalized/neurons.feather) to resolve its declared '
                f'graded classes: {exc}') from exc
        mask = np.zeros(n, dtype=np.uint8)
        report.update(graded_neurons=0, graded_set_sha256=graded_set_sha256(mask),
                      resolved='UNRESOLVED (no cell-class table); no class declared graded')
        return mask, report
    if len(ct) != n:
        if required:
            raise GraphUnavailable(
                f'cell-class table has {len(ct)} rows but the graph has {n} neurons')
        mask = np.zeros(n, dtype=np.uint8)
        report.update(graded_neurons=0, graded_set_sha256=graded_set_sha256(mask),
                      resolved='UNRESOLVED (table does not match the graph)')
        return mask, report
    mask = graded_mask(ct, sc, policy=policy)
    report.update(graded_neurons=int(mask.sum()),
                  graded_fraction=float(mask.mean()),
                  graded_set_sha256=graded_set_sha256(mask),
                  tier1_neurons=int(np.isin(np.asarray(ct, dtype=str), TIER1_TYPES).sum()),
                  resolved='cell_type + superclass from neurons.feather (never row order)')
    return mask, report
