"""Declared per-receptor-class synaptic kinetics for LIF dynamics v5.

``docs/LIF_DYNAMICS_SPEC.md`` §9 and the locked declaration
``docs/receipts/receptor_kinetics_v5_declaration.locked.md`` fix everything
here.  Nothing in this module was chosen after a v5 measurement.

**Granularity: per receptor class.**  Every synapse is assigned the receptor
class implied by its PRESYNAPTIC neuron's released transmitter label (the same
``neurotransmitter`` column the v3 sign policy already reads):

=============  ============================  ==================================
transmitter    receptor class (fast, ionotropic)  reversal used (unchanged from v3)
=============  ============================  ==================================
acetylcholine  ``nicotinic``  (nAChR)         E_exc = 0 mV
GABA           ``gaba_a``     (Rdl)           E_inh = -70 mV
glutamate      ``glucl``      (GluClalpha)    E_inh = -70 mV
histamine      ``hiscl``      (ort / HisCl1)  E_inh = -70 mV
unclear / nan  ``nicotinic``  DECLARED: their v3 sign is the declared excitatory
                              proxy, and the only excitatory fast class is nicotinic
aminergic      ``nicotinic``  irrelevant: their fast weight is zero under v3
=============  ============================  ==================================

The class is a property of the synapse type and is applied brain-wide, to every
spiking and graded cell alike.  No cell type is singled out, so the motion
pathway gets exactly the kinetics every other circuit gets.

**What a class changes: only the decay time constant of the conductance.**
Each class ``c`` has a single-exponential conductance with decay ``tau_c``.
Classes that share a ``tau`` share one *kinetic channel*.  Every arrival in
channel ``k`` is scaled by the **charge-preserving quantum factor**

    q_k = (1 - exp(-dt/tau_k)) / (1 - exp(-dt/TAU_REF_MS)),   TAU_REF_MS = 5 ms

so that, in the engine's own discrete time, the time-averaged conductance a
presynaptic release rate produces is exactly v4's: the operating point (the
maintained baseline v4 derived, every tonic set point) is unchanged and only the
time course of each synaptic event changes.  This is a derivation, not a fit: it
adds no parameter beyond the declared ``tau_c``.  With every ``tau_c = 5 ms``
there is one channel, ``q = 1.0`` exactly, and v5 is bit-identical to v4.

Rise times, per-class transmission delays, desensitisation and metabotropic
components (GABA_B, mGluR) are NOT modelled; the declaration says why.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Optional, Sequence

import numpy as np

CLASSES = ('nicotinic', 'gaba_a', 'glucl', 'hiscl')

TRANSMITTER_TO_CLASS = {
    'acetylcholine': 'nicotinic',
    'gaba': 'gaba_a',
    'glutamate': 'glucl',
    'histamine': 'hiscl',
}
#: Everything else (unclear, nan, unassigned, aminergic) -> this class.
DEFAULT_CLASS = 'nicotinic'

#: The single time constant every synapse had in v1-v4 (Shiu et al. 2024 ``tau``).
TAU_REF_MS = 5.0

KINETICS_PRIMARY = 'v5-receptor-class'
KINETICS_UPPER = 'v5-receptor-class-upper'
KINETICS_V4_EQUIVALENT = 'v4-equivalent'

#: Declared decay time constants in ms.  See the locked declaration (§9.3) for
#: the source of every number.  Fixed before the lock; never changed after it.
KINETICS = {
    # PRIMARY.  nicotinic / gaba_a: Su & O'Dowd 2003 J Neurosci 23:9246, Table 1,
    # Kenyon-cell mEPSC 1.4 +/- 0.1 ms and mIPSC 3.7 +/- 0.9 ms -- the only
    # same-preparation comparison of the two classes in Drosophila.  glucl: NO
    # Drosophila value exists; DECLARED ASSUMPTION = gaba_a (both Cys-loop
    # chloride channels).  hiscl: Pantazis et al. 2008 J Neurosci 28:7250,
    # Table 3, slow Lorentzian component of the ort (HCLA) homomer, 1.0 ms --
    # channel-noise kinetics, not a PSC decay (none is published).
    KINETICS_PRIMARY: dict(nicotinic=1.4, gaba_a=3.7, glucl=3.7, hiscl=1.0),
    # Declared sensitivity arm S2: the slowest defensible values.  nicotinic:
    # Gu & O'Dowd 2006 J Neurosci 26:265, adult in situ OK107 KC mEPSC 5.23 ms
    # (somatic, cable-slowed); inhibitory classes scaled by the Su & O'Dowd
    # GABA/nicotinic ratio 3.7/1.4 (DECLARED EXTRAPOLATION) = 13.8 ms; hiscl:
    # slowest histamine-receptor Lorentzian in Pantazis 2008 (HCLB) 6.1 ms.
    KINETICS_UPPER: dict(nicotinic=5.23, gaba_a=13.8, glucl=13.8, hiscl=6.1),
    # Declared arm S0: every class at v4's 5 ms; must be bit-identical to v4.
    KINETICS_V4_EQUIVALENT: dict(nicotinic=5.0, gaba_a=5.0, glucl=5.0, hiscl=5.0),
}


def describe(kinetics: str = KINETICS_PRIMARY) -> dict:
    if kinetics not in KINETICS:
        raise ValueError(f'Unknown receptor-kinetics table {kinetics!r}; declared: {sorted(KINETICS)}')
    return dict(kinetics=kinetics, tau_decay_ms=dict(KINETICS[kinetics]),
                tau_ref_ms=TAU_REF_MS,
                class_from='presynaptic neuron transmitter label (normalized/neurons.feather)',
                transmitter_to_class=dict(TRANSMITTER_TO_CLASS), default_class=DEFAULT_CLASS,
                quantum='q_k = (1 - exp(-dt/tau_k)) / (1 - exp(-dt/tau_ref)): charge-preserving',
                spec='docs/LIF_DYNAMICS_SPEC.md#9',
                declaration_lock='docs/receipts/receptor_kinetics_v5_declaration.locked.md',
                declaration_sha256='3f5996b5c132bd5d0ebb099446761344da11add9dc7340933d105a8269756b6b')


def classes_from_transmitters(labels: Sequence) -> np.ndarray:
    """Receptor class name per neuron from transmitter labels."""
    out = np.empty(len(labels), dtype=object)
    for i, lab in enumerate(labels):
        out[i] = TRANSMITTER_TO_CLASS.get(str(lab).strip().lower(), DEFAULT_CLASS)
    return out


def channels(tau_by_class: dict) -> list:
    """Distinct decay constants, in CLASSES order of first appearance."""
    taus = []
    for c in CLASSES:
        t = float(tau_by_class[c])
        if t not in taus:
            taus.append(t)
    return taus


def quantum_factor(tau_ms: float, dt: float) -> float:
    if float(tau_ms) == TAU_REF_MS:
        return 1.0
    return (1.0 - math.exp(-dt / float(tau_ms))) / (1.0 - math.exp(-dt / TAU_REF_MS))


def kinetics_sha256(kinetics: str, class_per_neuron: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(describe(kinetics)['tau_decay_ms'], sort_keys=True).encode())
    codes = np.array([CLASSES.index(str(c)) for c in class_per_neuron], dtype=np.int8)
    h.update(codes.tobytes())
    return h.hexdigest()


def resolve(n: int, *, kinetics: Optional[str] = None, receptor_classes=None, dt: float = 0.1,
            connectome_dir=None, required: bool = True) -> dict:
    """Per-neuron kinetic channel and per-channel decay / quantum for a graph of ``n``.

    ``receptor_classes`` (one class name per neuron) is for synthetic test
    graphs.  Otherwise the classes come from the released transmitter table;
    the real graph without it is refused.  A synthetic graph with neither gets
    every neuron in ``DEFAULT_CLASS`` and says so in the report.
    """
    kinetics = kinetics or KINETICS_PRIMARY
    report = describe(kinetics)
    tau = KINETICS[kinetics]
    if receptor_classes is not None:
        cls = np.asarray([str(c) for c in receptor_classes], dtype=object)
        if len(cls) != n or not set(cls) <= set(CLASSES):
            raise ValueError(f'receptor_classes needs one of {CLASSES} per neuron')
        report['resolved'] = 'explicit per-neuron classes (synthetic graph)'
    else:
        from .graph_identity import GraphUnavailable
        from .transmitter_policy import load_transmitters
        try:
            labels = load_transmitters(connectome_dir)
        except (FileNotFoundError, OSError, GraphUnavailable, ImportError, ValueError) as exc:
            if required:
                raise GraphUnavailable(f'LIF v5 needs the transmitter table to assign receptor classes: {exc}')
            labels = None
        if labels is not None and len(labels) != n:
            if required:
                raise GraphUnavailable(f'transmitter table has {len(labels)} rows, graph has {n}')
            labels = None
        if labels is None:
            cls = np.full(n, DEFAULT_CLASS, dtype=object)
            report['resolved'] = f'UNRESOLVED (no transmitter table); every neuron {DEFAULT_CLASS}'
        else:
            cls = classes_from_transmitters(labels)
            report['resolved'] = 'presynaptic transmitter label from normalized/neurons.feather'
    taus = channels(tau)
    chan_of_class = {c: taus.index(float(tau[c])) for c in CLASSES}
    pre_chan = np.array([chan_of_class[str(c)] for c in cls], dtype=np.int64)
    chan_decay = np.array([math.exp(-dt / t) for t in taus], dtype=np.float64)
    chan_q = np.array([quantum_factor(t, dt) for t in taus], dtype=np.float64)
    report.update(neurons=int(n),
                  class_counts={c: int(np.sum(cls == c)) for c in CLASSES},
                  channels=[dict(tau_ms=t, classes=[c for c in CLASSES if chan_of_class[c] == k],
                                 decay_per_dt=float(chan_decay[k]), quantum_factor=float(chan_q[k]))
                            for k, t in enumerate(taus)],
                  kinetics_sha256=kinetics_sha256(kinetics, cls))
    return dict(pre_chan=pre_chan, chan_decay=chan_decay, chan_q=chan_q, classes=cls, report=report)
