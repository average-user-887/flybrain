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
# the pinned graph).  They are verified here against cell types and, for
# lateralised channels, against the released somaSide annotation.
# WP5 correction: until f1395b1, dna02_l held node 332 (DNa02_R, soma R) and
# dna02_r held node 131957 (DNa02_L, soma L).  DNa02 steers ipsilaterally
# (Rayshubskiy et al. 2020) and each DNa02's MaleCNS output synapses are >90 %
# on its soma side, so the channels are now named by soma side.
DN_CHANNELS: Dict[str, List[int]] = {
    'dnp01': [0, 6], 'dna02_l': [131957], 'dna02_r': [332], 'dna01': [406, 704],
    'dnp09': [725, 1087], 'dnb01': [608, 703], 'mdn': [706, 1196, 1240, 2194],
}
DN_EXPECTED_TYPES = {'dnp01': 'DNp01', 'dna02_l': 'DNa02', 'dna02_r': 'DNa02',
                     'dna01': 'DNa01', 'dnp09': 'DNp09', 'dnb01': 'DNb01', 'mdn': 'MDN'}
DN_EXPECTED_SIDES = {'dna02_l': 'L', 'dna02_r': 'R'}

# Constants of brainlab/engine.py, with units, so manifests never guess them.
# Three explicitly versioned dynamics; all declared in docs/LIF_DYNAMICS_SPEC.md.
# The engine constants are imported, never restated, so the declaration and the
# compiled code cannot drift apart.
from . import engine as _engine

DYNAMICS_SPEC_DOC = 'docs/LIF_DYNAMICS_SPEC.md'

LIF_DYNAMICS_V1 = {
    'dynamics_version': 'v1',
    'model': 'fixed-weight leaky integrate-and-fire, CURRENT-based (brainlab.engine.advance)',
    'dt_ms': 0.1, 'tau_membrane_ms': _engine.TAU_M_MS, 'tau_synapse_ms': _engine.TAU_SYN_MS,
    'v_rest_mV': _engine.V_REST_MV, 'v_reset_mV': _engine.V_RESET_MV,
    'v_threshold_mV': _engine.V_THRESHOLD_MV,
    'refractory_ms': _engine.REFRACTORY_MS, 'transmission_delay_ms': _engine.DELAY_MS,
    'synaptic_scale': 0.275,
    'weight_units': 'synapse_count * transmitter_sign * 0.275 (upstream mV-equivalent)',
    'input_units': 'per-neuron drive current (upstream mV-equivalent)',
    'output_units': 'spike counts per step window',
    'membrane_bounds_mV': None,
    'reversal_potentials_mV': None,
    'integration': 'exact two-exponential subthreshold kernel',
    'known_defects': ['membrane potential unbounded (observed -200 mV in WP5)',
                      'inhibition is purely subtractive; no shunting / gain control',
                      'self-sustained ~1e6 spikes/s on the MaleCNS graph after any stimulus'],
    'parameter_source': 'Shiu et al. 2024 Nature 634:210-219 whole-brain LIF model',
    'spec': DYNAMICS_SPEC_DOC,
    'biological_validation': 'none; engineering proxy',
}

LIF_DYNAMICS_V2 = {
    'dynamics_version': 'v2',
    'model': 'fixed-weight leaky integrate-and-fire, CONDUCTANCE-based with reversal '
             'potentials (brainlab.engine.advance_v2)',
    'dt_ms': 0.1, 'tau_membrane_ms': _engine.TAU_M_MS, 'tau_synapse_ms': _engine.TAU_SYN_MS,
    'v_rest_mV': _engine.V_REST_MV, 'v_reset_mV': _engine.V_RESET_MV,
    'v_threshold_mV': _engine.V_THRESHOLD_MV,
    'refractory_ms': _engine.REFRACTORY_MS, 'transmission_delay_ms': _engine.DELAY_MS,
    'synaptic_scale': 0.275,
    'e_excitatory_mV': _engine.E_EXC_MV,
    'e_inhibitory_mV': _engine.E_INH_MV,
    'g_unit_per_weight': _engine.G_UNIT_PER_WEIGHT,
    'weight_units': 'synapse_count * transmitter_sign * 0.275, converted to a synaptic '
                    'conductance of |weight| * g_unit_per_weight in leak-conductance units',
    'input_units': 'per-neuron drive current (upstream mV-equivalent), shunted by 1/(1+ge+gi)',
    'output_units': 'spike counts per step window',
    'membrane_bounds_mV': [_engine.E_INH_MV, _engine.E_EXC_MV],
    'reversal_potentials_mV': {
        'excitatory': dict(value=_engine.E_EXC_MV, status='measured (Drosophila)',
                           source='Lee & O\'Dowd 1999 J Neurosci 19:5311 (nAChR mEPSC reverses near 0 mV); '
                                  'Su & O\'Dowd 2003 J Neurosci 23:9246 (+8.9 mV in Kenyon cells)'),
        'inhibitory': dict(value=_engine.E_INH_MV, status='ENGINEERING ASSUMPTION',
                           source='chloride reversal for Rdl/GluCl/HisCl; reported Drosophila values are '
                                  'internal-solution dependent (-56 mV Rohrbough & Broadie 2002; -37 mV '
                                  'Su & O\'Dowd 2003). -70 mV is the conventional low-[Cl-]i value, not a '
                                  'measured adult number. Declared sensitivity arm at -56 mV.')},
    'g_unit_derivation': 'g_unit = 1/(e_excitatory - v_rest) = 1/52; fixed by matching the v1 / '
                         'Shiu et al. unitary EPSP of 0.275 mV at rest. Not fitted.',
    'integration': 'exponential Euler, conductances frozen within dt (first order in the synaptic term)',
    'changes_from_v1': [
        'membrane potential bounded to [e_inhibitory, e_excitatory]',
        'inhibition shunts as well as hyperpolarises (tau_eff = tau_m/(1+ge+gi))',
        'excitatory driving force saturates as v depolarises',
        'unitary IPSP at rest is 18/52 = 0.346x the v1 IPSP for the same weight',
        'external drive current is divided by the total conductance',
        'synaptic state array g has shape (2, n); v1 checkpoints are refused, not reinterpreted'],
    'unchanged_engineering_properties': [
        'synaptic conductance zeroed on every spike (Brian2 unless-refractory schedule)',
        'synaptic input arriving at a refractory neuron is discarded',
        'no adaptation, short-term plasticity or after-hyperpolarisation',
        'coarse transmitter-sign proxy (brainlab/transmitters.py), not receptor physiology'],
    'parameter_source': 'Shiu et al. 2024 for the shared LIF constants; see spec for reversals',
    'spec': DYNAMICS_SPEC_DOC,
    'biological_validation': 'none; engineering proxy with declared reversal potentials',
}

LIF_DYNAMICS_V3 = {
    'dynamics_version': 'v3',
    'model': 'fixed-weight leaky integrate-and-fire, CONDUCTANCE-based with reversal '
             'potentials and a PER-SIGN PSP-preserving calibration '
             '(brainlab.engine.advance_v3)',
    'dt_ms': 0.1, 'tau_membrane_ms': _engine.TAU_M_MS, 'tau_synapse_ms': _engine.TAU_SYN_MS,
    'v_rest_mV': _engine.V_REST_MV, 'v_reset_mV': _engine.V_RESET_MV,
    'v_threshold_mV': _engine.V_THRESHOLD_MV,
    'refractory_ms': _engine.REFRACTORY_MS, 'transmission_delay_ms': _engine.DELAY_MS,
    'synaptic_scale': 0.275,
    'e_excitatory_mV': _engine.E_EXC_MV,
    'e_inhibitory_mV': _engine.E_INH_MV,
    'g_unit_excitatory_per_weight': _engine.G_UNIT_EXC_V3,
    'g_unit_inhibitory_per_weight': _engine.G_UNIT_INH_V3,
    'g_unit_ratio_inh_over_exc': _engine.G_UNIT_INH_V3 / _engine.G_UNIT_EXC_V3,
    'weight_units': 'synapse_count * transmitter_sign * 0.275, converted to a synaptic '
                    'conductance of |weight| * g_unit_<sign> in leak-conductance units',
    'input_units': 'per-neuron drive current (upstream mV-equivalent), shunted by 1/(1+ge+gi)',
    'output_units': 'spike counts per step window',
    'membrane_bounds_mV': [_engine.E_INH_MV, _engine.E_EXC_MV],
    'reversal_potentials_mV': dict(LIF_DYNAMICS_V2['reversal_potentials_mV']),
    'g_unit_derivation': 'g_unit_exc = 1/(e_excitatory - v_rest) = 1/52 and '
                         'g_unit_inh = 1/(v_rest - e_inhibitory) = 1/18. Each quantum is fixed '
                         'by requiring the unitary PSP at rest to equal the one the upstream '
                         'current-based model (Shiu et al. 2024, w_syn = 0.275 mV) specifies '
                         'for that sign. Derived, not fitted; no free parameter. v2 applied '
                         'the excitatory quantum to both signs, which is an extra assumption '
                         'the upstream model never made and which shrank the unitary IPSP to '
                         '18/52 = 0.346x its declared value.',
    'transmitter_policy': {
        'policy': 'v3-modulatory-only',
        'modulatory_only_transmitters': ['dopamine', 'octopamine', 'serotonin'],
        'effect': 'every out-edge of a dopaminergic, octopaminergic or serotonergic neuron '
                  'carries zero fast synaptic weight; those neurons are available to a '
                  'plasticity rule as a modulatory signal and in no other way',
        'unresolved_neurons': "the 2,999 'unclear' (plus 178 unlabelled) neurons are handled "
                              "SEPARATELY by the declared switch unclear_mode "
                              "{excitatory|zero|exclude}; the primary is 'excitatory', i.e. "
                              "unchanged from v1 and v2",
        'applied_by': 'brainlab.transmitter_policy.apply_to_shared (in memory; the pinned '
                      'graph.npz is never rewritten, and the transformed graph gets its own '
                      'graph_sha256)',
        'rationale': 'Drosophila dopamine, octopamine and serotonin receptors are '
                     'G-protein-coupled (Blenau & Baumann 2001 Arch Insect Biochem Physiol '
                     '48:13-38; Evans & Maqueira 2005 Invert Neurosci 5:111-118), so these '
                     'neurons do not make the fast ionotropic excitatory synapse the coarse '
                     'sign proxy gave them. See docs/LIF_DYNAMICS_SPEC.md §4.3.',
    },
    'integration': 'exponential Euler, conductances frozen within dt (first order in the synaptic term)',
    'changes_from_v2': [
        'inhibitory conductance quantum is 52/18 = 2.889x the excitatory one, so the unitary '
        'IPSP at rest equals the v1 IPSP instead of 0.346x it',
        'dopaminergic / octopaminergic / serotonergic neurons carry no fast synaptic weight',
        'the fast-weight array therefore differs from the pinned graph and carries its own '
        'graph_sha256; checkpoints do not cross between policies',
        'every snapshot records its dynamics version, so v2 and v3 checkpoints (which share '
        'the (2, n) synaptic state shape) refuse each other explicitly'],
    'unchanged_from_v2': [
        'E_exc = 0 mV, E_inh = -70 mV, membrane bounded to [E_inh, E_exc]',
        'shunting inhibition, saturating excitation, shunted external drive',
        'spike / delay / refractory / reset schedule, identical to v1',
        'synaptic conductance zeroed on every spike; arrivals at a refractory neuron dropped',
        'no adaptation, short-term plasticity or after-hyperpolarisation',
        'coarse transmitter-sign proxy for the fast classes, not receptor physiology'],
    'parameter_source': 'Shiu et al. 2024 for the shared LIF constants; see spec for reversals '
                        'and for the aminergic sources',
    'spec': DYNAMICS_SPEC_DOC,
    'biological_validation': 'none; engineering proxy with declared reversal potentials and a '
                             'declared transmitter-class policy',
}

LIF_DYNAMICS_V4 = {
    'dynamics_version': 'v4',
    'model': 'HYBRID leaky integrate-and-fire: declared non-spiking cell classes are '
             'integrated as passive membranes (no threshold, no reset, no refractory '
             'period) and their SUBTHRESHOLD membrane potential drives downstream '
             'synaptic conductance through a derived release function; every other '
             'neuron behaves exactly as v3 (brainlab.engine.advance_v4)',
    'dt_ms': 0.1, 'tau_membrane_ms': _engine.TAU_M_MS, 'tau_synapse_ms': _engine.TAU_SYN_MS,
    'v_rest_mV': _engine.V_REST_MV, 'v_reset_mV': _engine.V_RESET_MV,
    'v_threshold_mV': _engine.V_THRESHOLD_MV,
    'refractory_ms': _engine.REFRACTORY_MS, 'transmission_delay_ms': _engine.DELAY_MS,
    'synaptic_scale': 0.275,
    'e_excitatory_mV': _engine.E_EXC_MV,
    'e_inhibitory_mV': _engine.E_INH_MV,
    'g_unit_excitatory_per_weight': _engine.G_UNIT_EXC_V3,
    'g_unit_inhibitory_per_weight': _engine.G_UNIT_INH_V3,
    'g_unit_ratio_inh_over_exc': _engine.G_UNIT_INH_V3 / _engine.G_UNIT_EXC_V3,
    'weight_units': 'unchanged from v3; the same array means the same thing for a spiking '
                    'and for a graded presynaptic cell because graded release is expressed '
                    'as an event RATE',
    'input_units': 'per-neuron drive current (upstream mV-equivalent), shunted by 1/(1+ge+gi)',
    'output_units': 'spike counts per step window for spiking cells; membrane potential in mV '
                    'and release rate in s^-1 for graded cells, which never spike',
    'membrane_bounds_mV': [_engine.E_INH_MV, _engine.E_EXC_MV],
    'reversal_potentials_mV': dict(LIF_DYNAMICS_V2['reversal_potentials_mV']),
    'graded_release': {
        'function': 'r(V) = r_max * (V - e_inhibitory) / (e_excitatory - e_inhibitory)',
        'units': 'release events per second',
        'r_max_hz': _engine.R_MAX_HZ,
        'r_at_rest_hz': _engine.R_MAX_HZ * (_engine.V_REST_MV - _engine.E_INH_MV)
                        / (_engine.E_EXC_MV - _engine.E_INH_MV),
        'delivery': 'over each out-edge, every dt, after the same 1.8 ms transmission delay '
                    'the spiking path uses: dg = |w| * g_unit_<sign> * r(V) * dt/1000. A '
                    'graded cell at release rate r therefore delivers the mean conductance a '
                    'v3 spiking cell firing at r Hz delivers, which is what lets the two '
                    'populations interoperate on one weight array.',
        'derivation': 'No free parameter. The two anchors are the already-declared membrane '
                      'bounds (r=0 at e_inhibitory, r=r_max at e_excitatory, spec §3.1) and '
                      'the ceiling is the already-declared refractory period '
                      '(r_max = 1/t_rfc, spec §1.1). r(V_rest) is a consequence.',
        'maintained_baseline': 'Because the zero of r(V) is at e_inhibitory and not at '
                              'V_rest, a graded cell at rest releases tonically, so graded '
                              'transmission is BIDIRECTIONAL about a set point. This breaks '
                              'the zero-spontaneous-activity property v1-v3 inherit from '
                              'Shiu et al. 2024, deliberately: the histaminergic '
                              'photoreceptor -> lamina synapse is inhibitory and can only '
                              'work by disinhibition. A v4 no-stimulus window is a '
                              'tonic-release steady state, NOT silence, and is not '
                              'comparable with a v1/v2/v3 gray window.',
        'engineering_assumption': 'the LINEARITY between the two anchors. No graded '
                                  'input-output curve measured in Drosophila is expressed in '
                                  "this engine's units, so there is no shape to fit. Real "
                                  'graded synapses are sigmoid, saturate and depress; v4 has '
                                  'none of that.',
    },
    'graded_policy': {
        'primary': 'v4-optic-lobe-graded',
        'resolved_by': 'cell_type + superclass from normalized/neurons.feather, never row order',
        'applied_by': 'brainlab.graded_policy.resolve; the resolved node set is hashed as '
                      'graded_set_sha256 and recorded in every manifest and receipt',
        'arms': ['none (must be bit-identical to v3)', 'v4-tier1-only (declared arm S1)'],
        'spec': 'docs/LIF_DYNAMICS_SPEC.md#7',
    },
    'transmitter_policy': dict(LIF_DYNAMICS_V3['transmitter_policy']),
    'integration': 'exponential Euler, conductances frozen within dt (first order in the '
                   'synaptic term); identical for graded and spiking cells',
    'changes_from_v3': [
        'declared non-spiking cell classes have no threshold, no reset and no refractory '
        'period, and are integrated as passive membranes',
        'their subthreshold membrane potential drives downstream conductance directly '
        'through r(V); under v3 a subthreshold cell transmitted nothing at all',
        'graded cells release tonically at rest, so the network has a maintained baseline '
        'and a v4 no-stimulus window is not silence',
        'spiking and graded populations interoperate in both directions with no adapter',
        'the state adds a delayed-release ring buffer, so a v4 snapshot is not '
        'interchangeable with a v3 one even by shape',
    ],
    'unchanged_from_v3': [
        'every membrane and synaptic constant, both reversal potentials, both conductance '
        'quanta and the membrane bounds',
        'the v3 transmitter policy (v3-modulatory-only, unclear_mode=excitatory) and the '
        'weight array it produces; the pinned graph.npz is still never rewritten',
        'the spike / delay / refractory / reset schedule for every spiking cell',
        'arrivals at a refractory target are dropped; conductance zeroed on a spike',
        'no adaptation, short-term plasticity or after-hyperpolarisation',
        'no gap junctions, no internal state, no neuropeptides, no receptor kinetics, no '
        'cell-type-specific synaptic time constants',
    ],
    'known_limitations': [
        'one global tau_syn = 5 ms and one global 1.8 ms hop delay for the whole brain, so a '
        '20-50 ms delay line cannot be expressed; this is the declared reason a working '
        'graded pathway may still fail to compute motion (spec §7.7 P4)',
        'the release function is linear, unsaturating and undepressing',
        'no photoreceptor light adaptation, contrast gain control or spectral channels',
        'graded cells still have one compartment, so CT1 and HS compartmentalisation is absent',
    ],
    'parameter_source': 'Shiu et al. 2024 for every shared constant; see spec §7 for the '
                        'graded class evidence and §3.1 for the reversals',
    'spec': DYNAMICS_SPEC_DOC,
    'declaration_lock': 'docs/receipts/graded_transmission_v4_declaration.locked.md',
    'declaration_sha256': '2634c824476c9b799bd2c0c656040255360cd10f8837e2d874a97af5f939872e',
    'biological_validation': 'none; engineering proxy with declared reversal potentials, a '
                             'declared transmitter-class policy and a declared non-spiking '
                             'cell-class list',
}

def _v5_declaration():
    from . import receptor_kinetics as _rk
    d = {k: v for k, v in LIF_DYNAMICS_V4.items()
         if k not in ('dynamics_version', 'model', 'tau_synapse_ms', 'changes_from_v3',
                      'unchanged_from_v3', 'known_limitations', 'declaration_lock',
                      'declaration_sha256', 'biological_validation')}
    d.update({
        'dynamics_version': 'v5',
        'model': 'v4 (hybrid graded/spiking LIF) plus DECLARED per-receptor-class synaptic '
                 'kinetics: each synapse takes the receptor class of its presynaptic '
                 "neuron's transmitter (nicotinic, GABA_A/Rdl, GluCl, HisCl) and that class's "
                 'single-exponential decay time constant, with a charge-preserving conductance '
                 'quantum (brainlab.engine.advance_v5, brainlab.cupy_v5)',
        'tau_synapse_ms': 'per receptor class; see receptor_kinetics',
        'tau_reference_ms': _rk.TAU_REF_MS,
        'receptor_kinetics': {name: dict(tau) for name, tau in _rk.KINETICS.items()},
        'receptor_kinetics_primary': _rk.KINETICS_PRIMARY,
        'receptor_kinetics_arms': [n for n in _rk.KINETICS if n != _rk.KINETICS_PRIMARY],
        'receptor_class_from': 'presynaptic transmitter label; ' + str(_rk.TRANSMITTER_TO_CLASS)
                               + f'; anything else -> {_rk.DEFAULT_CLASS}',
        'quantum': 'q_k = (1 - exp(-dt/tau_k)) / (1 - exp(-dt/tau_ref)): the time-averaged '
                   'conductance per release rate is v4\'s exactly; only the time course changes',
        'changes_from_v4': [
            'each synapse decays with its receptor class\'s declared time constant instead of '
            'one global 5 ms',
            'arrivals are scaled by the charge-preserving quantum factor of their class',
            'the synaptic state is (2K, n) for K kinetic channels, so a v5 snapshot is not '
            'interchangeable with a v4 one; snapshots also carry kinetics_sha256',
        ],
        'unchanged_from_v4': [
            'every membrane constant, both reversal potentials, both conductance quanta, the '
            'membrane bounds, the graded class list, the release function and its baseline',
            'the single 1.8 ms transmission delay for every synapse',
            'the v3 transmitter policy and the weights it produces',
            'with every class at 5 ms, v5 is bit-identical to v4',
        ],
        'known_limitations': [
            'single-exponential conductances: no rise time, no desensitisation, no '
            'metabotropic (GABA_B, mGluR) component, no short-term plasticity',
            'one class per presynaptic neuron (the released table has one transmitter per '
            'neuron); postsynaptic receptor subunit differences within a class are not '
            'represented',
            'cell-type-specific intrinsic filtering (e.g. Mi4/Mi9 sustained, Tm3/Mi1 transient) '
            'is NOT imposed; it can arise only from the graph and the class kinetics',
            'no gap junctions, one compartment per cell',
        ],
        'spec': DYNAMICS_SPEC_DOC,
        'declaration_lock': 'docs/receipts/receptor_kinetics_v5_declaration.locked.md',
        'declaration_sha256': '3f5996b5c132bd5d0ebb099446761344da11add9dc7340933d105a8269756b6b',
        'biological_validation': 'none; engineering proxy with declared receptor-class '
                                 'kinetics taken from published Drosophila recordings where they '
                                 'exist and declared assumptions where they do not',
    })
    return d


LIF_DYNAMICS_V5 = _v5_declaration()

DYNAMICS_VERSIONS = {'v1': LIF_DYNAMICS_V1, 'v2': LIF_DYNAMICS_V2, 'v3': LIF_DYNAMICS_V3,
                     'v4': LIF_DYNAMICS_V4, 'v5': LIF_DYNAMICS_V5}
DYNAMICS_ENV = 'NEUROFLY_LIF_DYNAMICS'
DEFAULT_DYNAMICS = 'v3'
E_INH_ENV = 'NEUROFLY_LIF_E_INH_MV'
# Only the v2/v3 conductance dynamics have declared E_inh sensitivity variants
# (docs/EINH_SENSITIVITY.md).  v4/v5 derive their graded release function from
# e_inhibitory as well, and no variant of them has been declared.
E_INH_VARIANT_VERSIONS = ('v2', 'v3')


def active_e_inh_mV() -> Optional[float]:
    """Process-wide inhibitory-reversal override (``NEUROFLY_LIF_E_INH_MV``).

    ``None`` when unset, which is the declared v2/v3 value
    (``brainlab.engine.E_INH_MV`` = -70 mV).  A value here does NOT redefine
    v2 or v3: it selects a declared *variant*, whose declaration and pin come
    from :func:`dynamics_variant` / :func:`dynamics_variant_pin`, so the v3
    pin and every number already published under it stay exactly as they are.
    """
    raw = os.environ.get(E_INH_ENV)
    if raw is None or raw == '':
        return None
    return float(raw)


def dynamics_variant(version: str, e_inh_mV=None) -> dict:
    """The declared dict for an ``E_inh`` variant of a conductance dynamics.

    Declared for ``docs/EINH_SENSITIVITY.md``: every swept value of the
    inhibitory reversal potential is a NEW labelled variant carrying its own
    pin, never a silent redefinition of v2 or v3.  ``e_inh_mV=None`` returns
    the base declaration unchanged, so ``dynamics_variant('v3')`` is
    byte-identical to ``LIF_DYNAMICS_V3`` and pins to the same sha256.

    Under v3 the inhibitory conductance quantum is DERIVED from the driving
    force at rest, so it moves with ``E_inh``; that is the calibration rule of
    spec §6.2 and is not a second free parameter.
    """
    if version not in DYNAMICS_VERSIONS:
        raise ValueError(f'Unknown dynamics version {version!r}')
    base = dict(DYNAMICS_VERSIONS[version])
    if e_inh_mV is None:
        return base
    if version not in E_INH_VARIANT_VERSIONS:
        raise ValueError('e_inh_mV variants are declared only for the conductance-based '
                         f'dynamics {E_INH_VARIANT_VERSIONS}, not {version!r}')
    e_inh = float(e_inh_mV)
    if e_inh >= _engine.V_REST_MV:
        raise ValueError('e_inh_mV must be below v_rest for the v3 calibration to be finite')
    base['dynamics_version'] = f'{version}-einh{e_inh:g}'
    base['base_dynamics_version'] = version
    base['base_dynamics_pin'] = dynamics_pin(version)
    base['e_inhibitory_mV'] = e_inh
    base['membrane_bounds_mV'] = [e_inh, _engine.E_EXC_MV]
    reversals = dict(base.get('reversal_potentials_mV') or {})
    for key in list(reversals):
        if reversals[key] == _engine.E_INH_MV:
            reversals[key] = e_inh
    base['reversal_potentials_mV'] = reversals
    if version == 'v3':
        g_inh = 1.0 / (_engine.V_REST_MV - e_inh)
        base['g_unit_inhibitory_per_weight'] = g_inh
        base['g_unit_ratio_inh_over_exc'] = g_inh / _engine.G_UNIT_EXC_V3
    base['e_inh_variant'] = {
        'declared_by': 'docs/EINH_SENSITIVITY.md',
        'e_inhibitory_mV': e_inh,
        'note': 'DECLARED SENSITIVITY VARIANT of ' + version + '. The base version, its pin '
                'and every result published under it are unchanged. Under v3 the inhibitory '
                'conductance quantum follows E_inh by the spec §6.2 calibration rule '
                '(g_unit_inh = 1/(v_rest - e_inhibitory)); it is not tuned independently.',
    }
    return base


def dynamics_variant_pin(version: str, e_inh_mV=None) -> str:
    """sha256 of the declared variant dict: the re-pin for an ``E_inh`` arm."""
    return sha256_json(dynamics_variant(version, e_inh_mV))


def active_dynamics_version() -> str:
    """The process-wide default dynamics version (``NEUROFLY_LIF_DYNAMICS``).

    Defaults to ``v3`` (owner decision, 2026-09-24).  ``v1`` and ``v2`` stay
    selectable here or per instance; their checkpoints are refused by v3 brains.
    """
    version = os.environ.get(DYNAMICS_ENV) or DEFAULT_DYNAMICS
    if version not in DYNAMICS_VERSIONS:
        raise ValueError(f'{DYNAMICS_ENV}={version!r} is not a declared dynamics version '
                         f'({sorted(DYNAMICS_VERSIONS)}); see {DYNAMICS_SPEC_DOC}')
    return version


def active_dynamics() -> dict:
    return dict(DYNAMICS_VERSIONS[active_dynamics_version()])


def dynamics_pin(version: str) -> str:
    """sha256 of the declared dynamics dict: the re-pin for a dynamics change."""
    if version not in DYNAMICS_VERSIONS:
        raise ValueError(f'Unknown dynamics version {version!r}')
    return sha256_json(DYNAMICS_VERSIONS[version])


# Backwards-compatible name: the ACTIVE declaration, resolved once at import.
# experiment_registry copies this into every run manifest.
LIF_DYNAMICS = active_dynamics()


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


# ---------------------------------------------------------------------------
# Content fingerprints of the files a user GENERATES locally.
#
# ``normalized/neurons.feather`` and ``graph.npz`` are written on the user's own
# machine by ``brainlab.connectome`` and ``brainlab.prepare``.  Their raw bytes
# carry more than the data: pandas stores its own version and pyarrow's in the
# feather schema metadata, pandas 2 writes ``string`` where pandas 3 writes
# ``large_string``, and zipfile stamps the host OS into every graph.npz member
# header (``create_system`` 3 on Linux/macOS, 0 on Windows).  A byte pin of such
# a file refuses identical data written by a different library version or OS
# (pandas 3.0.6 vs 3.0.5 gave 416b4a36... vs the pinned 2103b520... for the
# same table).  These files are therefore pinned by WHAT THEY CONTAIN.
#
# Files a user DOWNLOADS (the three MaleCNS source tables) are served as fixed
# bytes, so they stay pinned by sha256 of the file.
#
# Both definitions below are sha256 over a byte stream in which every field has
# a length that is either fixed or written in front of it, so the stream is
# unambiguous.  All integers in the framing are unsigned 64-bit little-endian
# ("u64").  ``bytes(s)`` of a text field is ``u64 len(utf8) || utf8``.
#
# TABLE scheme ``neurofly-table-content-v1`` (used for neurons.feather):
#   b'neurofly-table-content-v1\0' || u64 n_rows || u64 n_columns
#   then for every column IN FILE ORDER:
#     bytes(column name) || bytes(dtype tag)
#     || null mask: n_rows bytes, 1 where the value is null, else 0
#     || values, null slots replaced by the zero value of the type:
#        bool                 -> n_rows bytes 0/1          (tag 'bool')
#        (u)int8..64          -> n_rows fixed-width little-endian two's
#                                complement               (tag 'int32', 'uint64', ...)
#        float16/32/64        -> n_rows fixed-width little-endian IEEE-754,
#                                every NaN written as the default quiet NaN
#                                                         (tag 'float32', ...)
#        string / large_string / string_view
#                             -> n_rows u64 UTF-8 byte lengths, then the
#                                concatenated UTF-8 bytes  (tag 'utf8')
#   Any other Arrow type (dictionary, temporal, binary, nested, null) is refused.
#   Schema metadata, the pandas index metadata, record-batch chunking, buffer
#   layout and the string offset width are NOT part of the content; column
#   names, column order, row order, dtype, nullness and every value ARE.
#
# ARRAYS scheme ``neurofly-npz-content-v1`` (used for graph.npz):
#   b'neurofly-npz-content-v1\0' || u64 n_members
#   then for every member, sorted by name (the '.npy' suffix removed):
#     bytes(name) || bytes(dtype tag) || u64 ndim || u64 each dimension
#     || the values in C order, fixed-width little-endian, floats with every NaN
#        written as the default quiet NaN.
#   The dtype tag is numpy's little-endian type string ('<i8', '<i4', '<f4',
#   '|b1', ...).  Object, string and structured dtypes are refused, and the
#   archive is read with allow_pickle=False.  Zip headers (timestamps, host OS,
#   zip64 extras), the .npy header text and memory order are NOT content.
#
# Neither scheme re-serialises through Arrow IPC or .npy, whose bytes may change
# between library versions; each hashes values through numpy's fixed-width
# little-endian representation, which is defined by the format, not by a library.
# ---------------------------------------------------------------------------

TABLE_CONTENT_SCHEME = 'neurofly-table-content-v1'
ARRAYS_CONTENT_SCHEME = 'neurofly-npz-content-v1'


def _u64(value: int) -> bytes:
    return int(value).to_bytes(8, 'little', signed=False)


def _text(value: str) -> bytes:
    raw = value.encode('utf-8')
    return _u64(len(raw)) + raw


def _canonical_values(values: np.ndarray) -> bytes:
    """Fixed-width little-endian bytes of a numeric array, NaNs canonicalised."""
    values = np.ascontiguousarray(values)
    if values.dtype.kind == 'f':
        nan = np.isnan(values)
        if nan.any():
            values = values.copy()
            values[nan] = np.nan  # the default quiet NaN of this width
    if values.dtype.kind == 'b':
        return values.astype(np.uint8).tobytes()
    return values.astype(values.dtype.newbyteorder('<'), copy=False).tobytes()


def table_content_sha256(table) -> str:
    """Content fingerprint of a pyarrow Table (scheme ``neurofly-table-content-v1``).

    See the block comment above for the exact definition.  Raises ValueError for
    a column type the scheme does not define.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    digest = hashlib.sha256()
    digest.update(TABLE_CONTENT_SCHEME.encode() + b'\0')
    digest.update(_u64(table.num_rows) + _u64(table.num_columns))
    for name, column in zip(table.column_names, table.columns):
        kind = column.type
        if pa.types.is_dictionary(kind) or pa.types.is_null(kind):
            raise ValueError(f'column {name!r}: type {kind} is not defined by {TABLE_CONTENT_SCHEME}')
        column = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
        nulls = column.is_null()
        null_mask = np.asarray(nulls.to_numpy(zero_copy_only=False), dtype=np.uint8)
        if pa.types.is_string(kind) or pa.types.is_large_string(kind) or getattr(pa.types, 'is_string_view', lambda _: False)(kind):
            tag = 'utf8'
            text = pc.fill_null(column, '').cast(pa.large_string())
            offsets = np.frombuffer(text.buffers()[1], dtype='<i8')[text.offset:text.offset + len(text) + 1]
            data = text.buffers()[2]
            lengths = np.diff(offsets).astype('<u8')
            payload = [lengths.tobytes(),
                       (data.to_pybytes()[offsets[0]:offsets[-1]] if data is not None and len(text) else b'')]
        elif pa.types.is_boolean(kind):
            tag = 'bool'
            values = np.asarray(pc.fill_null(column, False).to_numpy(zero_copy_only=False), dtype=bool)
            payload = [_canonical_values(values)]
        elif pa.types.is_integer(kind) or pa.types.is_floating(kind):
            tag = np.dtype(kind.to_pandas_dtype()).name  # 'int32', 'uint64', 'float32', ...
            zero = pa.scalar(0, type=kind) if pa.types.is_integer(kind) else pa.scalar(0.0, type=kind)
            values = pc.fill_null(column, zero).to_numpy(zero_copy_only=False)
            payload = [_canonical_values(values)]
        else:
            raise ValueError(f'column {name!r}: type {kind} is not defined by {TABLE_CONTENT_SCHEME}')
        digest.update(_text(name) + _text(tag) + null_mask.tobytes())
        for part in payload:
            digest.update(part)
    return digest.hexdigest()


def feather_content_sha256(path: os.PathLike) -> str:
    """Content fingerprint of a feather / Arrow IPC file (all columns, file order)."""
    import pyarrow.feather as feather
    return table_content_sha256(feather.read_table(path, memory_map=False))


def arrays_content_sha256(arrays: Dict[str, np.ndarray]) -> str:
    """Content fingerprint of named numpy arrays (scheme ``neurofly-npz-content-v1``)."""
    digest = hashlib.sha256()
    digest.update(ARRAYS_CONTENT_SCHEME.encode() + b'\0')
    digest.update(_u64(len(arrays)))
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        if values.dtype.kind not in 'biuf':
            raise ValueError(f'array {name!r}: dtype {values.dtype} is not defined by {ARRAYS_CONTENT_SCHEME}')
        tag = values.dtype.newbyteorder('<').str if values.dtype.byteorder != '|' else values.dtype.str
        digest.update(_text(name) + _text(tag) + _u64(values.ndim))
        for dim in values.shape:
            digest.update(_u64(dim))
        digest.update(_canonical_values(values))
    return digest.hexdigest()


def npz_content_sha256(path: os.PathLike) -> str:
    """Content fingerprint of every member of an .npz archive."""
    with np.load(path, allow_pickle=False) as data:
        return arrays_content_sha256({name: data[name] for name in data.files})


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
    """Immutable identity of the graph a controller runs on.

    For the verified MaleCNS graph, ``graph_sha256`` and ``neuron_map_sha256``
    are the REFERENCE identifiers recorded in ``graph_pins.json`` (the sha256 of
    the reference graph.npz and neurons.feather, as carried by every receipt and
    checkpoint).  They are reported only after the local files' CONTENT
    fingerprints (``graph_content_sha256``, ``neuron_map_content_sha256``) have
    matched the pins, so runs on identical data share one identity on every
    machine.  The local files' own byte hashes are in ``graph_file_sha256`` and
    ``neuron_map_file_sha256``; they are informational and may legitimately
    differ from the reference with the pandas/pyarrow version or OS that wrote
    the file.  For a synthetic or policy-transformed graph, ``graph_sha256`` is
    the digest of the effective arrays.
    """
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
    graph_content_sha256: Optional[str] = None
    neuron_map_content_sha256: Optional[str] = None
    graph_file_sha256: Optional[str] = None
    neuron_map_file_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype='<i8').tobytes()).hexdigest()


def verify_graph(graph_dir: Optional[os.PathLike] = None,
                 connectome_dir: Optional[os.PathLike] = None,
                 *, pins: Optional[dict] = None, check_hashes: bool = True) -> GraphIdentity:
    """Verify the real graph against pinned fingerprints, counts, IDs and DN cell types.

    The two files the user generates locally (graph.npz, neurons.feather) must
    match their pinned CONTENT fingerprints (scheme definitions above); their
    byte hashes are recorded but not required, because pandas, pyarrow and
    zipfile write version- and OS-dependent metadata into identical data.  The
    downloaded source tables are still verified by byte hash.
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
    for key in ('graph_content_sha256', 'neuron_map_content_sha256'):
        if not pins.get(key):
            raise GraphUnavailable(f'Pins lack {key}; a content fingerprint is required to verify the graph')
    graph_file_sha = nodes_file_sha = None
    if check_hashes:
        try:
            graph_content = npz_content_sha256(graph)
        except Exception as exc:  # corrupt zip, pickled or non-numeric member
            raise GraphUnavailable(f'Graph content hash mismatch at {graph}: unreadable ({exc})') from exc
        if graph_content != pins['graph_content_sha256']:
            raise GraphUnavailable(f'Graph content hash mismatch at {graph}: {graph_content} != pinned '
                                   f'{pins["graph_content_sha256"]} ({ARRAYS_CONTENT_SCHEME})')
        try:
            nodes_content = feather_content_sha256(nodes_path)
        except Exception as exc:  # pyarrow raises its own error types for a corrupt file
            raise GraphUnavailable(f'Neuron map content hash mismatch at {nodes_path}: unreadable ({exc})') from exc
        if nodes_content != pins['neuron_map_content_sha256']:
            raise GraphUnavailable(f'Neuron map content hash mismatch at {nodes_path}: {nodes_content} != pinned '
                                   f'{pins["neuron_map_content_sha256"]} ({TABLE_CONTENT_SCHEME})')
        graph_file_sha = sha256_file(graph)
        nodes_file_sha = sha256_file(nodes_path)
    else:
        graph_content = pins['graph_content_sha256']
        nodes_content = pins['neuron_map_content_sha256']
    # Reference identifiers (see GraphIdentity), reported only once the content
    # fingerprints above have matched.
    graph_sha = pins['graph_sha256']
    nodes_sha = pins['neuron_map_sha256']
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
    annotations_path = cdir / 'annotations.feather'
    if not annotations_path.is_file():
        raise GraphUnavailable(f'{annotations_path} missing: DN soma sides cannot be verified')
    ann = feather.read_table(annotations_path, columns=['bodyId', 'somaSide']).to_pandas()
    soma_side = dict(zip(ann.bodyId.astype('int64'), ann.somaSide))
    for channel, expected in DN_EXPECTED_SIDES.items():
        for index in DN_CHANNELS[channel]:
            actual = soma_side.get(int(ids[index]))
            if actual != expected:
                raise GraphUnavailable(f'DN channel {channel} node {index} has somaSide {actual}, expected {expected}')
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
        ids_sha256=ids_sha, data_sha256=data_sha, label='MaleCNS v1.0 prepared graph',
        graph_content_sha256=graph_content, neuron_map_content_sha256=nodes_content,
        graph_file_sha256=graph_file_sha, neuron_map_file_sha256=nodes_file_sha)


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
