"""Declared per-class graded transfer gain for LIF dynamics v6a.

``docs/LIF_DYNAMICS_SPEC.md`` §10 and the locked declaration
``docs/receipts/graded_gain_v6a_declaration.locked.md`` fix everything here.
Nothing in this module was chosen after a v6a measurement.

**What v6a changes: only the slope of the graded release function.**  v4/v5
release ``r(V) = R_MAX_HZ * (V - E_inh) / (E_exc - E_inh)`` is one straight
line across the whole 70 mV membrane range, so its slope (6.49 s^-1 per mV) is
the same for every graded cell.  v6a keeps, for each declared presynaptic
CLASS ``c``, v4's release VALUE at a declared anchor potential ``V_A,c`` and
multiplies the SLOPE there by a declared class gain ``G_c``:

    r_c(V) = clip( r_v4(V_A,c) + G_c * R_MAX_HZ / (E_exc - E_inh) * (V - V_A,c),
                   0, R_MAX_HZ )

The clip bounds are v4's own (release cannot be negative and cannot exceed the
refractory ceiling).  ``G_c = 1`` is v4/v5 exactly, for any anchor.

**Classes** are resolved by the PRESYNAPTIC neuron's ``cell_type`` and are a
property of the release site, applied to every out-edge of every neuron of the
class, brain-wide.  The only class with a measured transfer is the
photoreceptor -> lamina monopolar cell histaminergic synapse (``R1-R6``); every
other declared graded neuron belongs to ``default``.  The graded set itself is
v4's declared policy, unchanged.

**How ``G`` is derived from a measured voltage gain** (:func:`derive_gain`):
for a reference postsynaptic cell receiving total weight ``W_ref`` from
presynaptic cells that all carry the same signal, at the class anchor, with
only its leak and the class's own tonic conductance, the engine's small-signal
DC voltage gain is

    dV_post/dV_pre = G * (R_MAX_HZ/(E_exc - E_inh)) * W_ref * g_unit * tau_ref/1000
                     * |E_syn - V_post| / g_tot

(the charge-preserving quantum of v5 makes the time-averaged conductance per
release rate independent of the kinetics table, so ``tau_ref = 5 ms`` holds for
every arm).  Setting it equal to the measured gain and solving gives ``G``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from .engine import E_EXC_MV, E_INH_MV, G_UNIT_INH_V3, R_MAX_HZ, V_REST_MV

TAU_REF_MS = 5.0

CLASS_PHOTORECEPTOR = 'photoreceptor_R1-R6'
CLASS_DEFAULT = 'default'
CLASSES = (CLASS_PHOTORECEPTOR, CLASS_DEFAULT)
#: presynaptic cell types of each non-default class
CLASS_CELL_TYPES = {CLASS_PHOTORECEPTOR: ('R1-R6',)}

RELEASE_V5_LINEAR = 'v5-linear'
RELEASE_PRIMARY = 'v6a-calibrated'

# ---------------------------------------------------------------------------
# Declared derivation inputs (spec §10.3).  Fixed before the lock.
# ---------------------------------------------------------------------------
#: Measured photoreceptor -> LMC small-signal voltage gain magnitude (see the
#: declaration for the source; blowfly, light-adapted -- a cross-species transfer).
MEASURED_GAIN_PHOTORECEPTOR = 6.0   # Laughlin, Howard & Blakeslee 1987 (blowfly), "approximately 6"
#: Anchor of the photoreceptor class: V_rest + the photoreceptor encoder's mean
#: drive (i_max/2 = 10 mV-equivalent), i.e. the potential of a driven R1-R6 under
#: the declared adapting background, from declared constants only.
ANCHOR_PHOTORECEPTOR_MV = V_REST_MV + 10.0
#: Mean total |weight| from R1-R6 onto an L1 or L2 cell that receives any, in
#: the pinned graph under the v3 policy (1,623 cells; median 34.38).  Re-derived
#: from the graph by tests/test_release_v6a.py.
W_REF_PHOTORECEPTOR = 35.7603
ANCHOR_DEFAULT_MV = V_REST_MV


def derive_gain(measured_gain: float, w_ref: float, anchor_mV: float, *,
                e_inh: float = E_INH_MV, g_unit: float = G_UNIT_INH_V3,
                tau_ref_ms: float = TAU_REF_MS) -> dict:
    """Class gain ``G`` that gives a reference inhibitory target the measured
    small-signal DC voltage gain.  Returns every intermediate for the receipt."""
    r_a = R_MAX_HZ * (anchor_mV - e_inh) / (E_EXC_MV - e_inh)
    g0 = w_ref * g_unit * tau_ref_ms / 1000.0 * r_a
    v_post = (V_REST_MV + g0 * e_inh) / (1.0 + g0)
    dv_dr = (v_post - e_inh) / (1.0 + g0) * w_ref * g_unit * tau_ref_ms / 1000.0
    slope_v4 = R_MAX_HZ / (E_EXC_MV - e_inh)
    gain_v4 = dv_dr * slope_v4
    return dict(measured_gain=measured_gain, w_ref=w_ref, anchor_mV=anchor_mV,
                release_at_anchor_hz=r_a, tonic_conductance_leak_units=g0,
                reference_post_v_mV=v_post, reference_post_g_tot=1.0 + g0,
                dV_post_per_release_hz_mV=dv_dr, v4_slope_hz_per_mV=slope_v4,
                v4_small_signal_gain=gain_v4,
                G=(measured_gain / gain_v4) if measured_gain is not None else None)


def _tables() -> dict:
    d = derive_gain(MEASURED_GAIN_PHOTORECEPTOR, W_REF_PHOTORECEPTOR, ANCHOR_PHOTORECEPTOR_MV)
    return {
        RELEASE_V5_LINEAR: {
            CLASS_PHOTORECEPTOR: dict(gain=1.0, anchor_mV=ANCHOR_PHOTORECEPTOR_MV),
            CLASS_DEFAULT: dict(gain=1.0, anchor_mV=ANCHOR_DEFAULT_MV),
        },
        RELEASE_PRIMARY: {
            CLASS_PHOTORECEPTOR: dict(gain=d['G'], anchor_mV=ANCHOR_PHOTORECEPTOR_MV),
            CLASS_DEFAULT: dict(gain=DEFAULT_GAIN, anchor_mV=ANCHOR_DEFAULT_MV),
        },
    }


#: Gain of every graded class without a measured transfer.  See the declaration.
DEFAULT_GAIN = 1.0   # DECLARED DEFAULT: no measured transfer; v4/v5 line kept (spec §10.3)
RELEASE = None        # built at import below, once the constants exist


def describe(release: str = RELEASE_PRIMARY) -> dict:
    if release not in RELEASE:
        raise ValueError(f'Unknown graded release table {release!r}; declared: {sorted(RELEASE)}')
    return dict(release=release,
                classes={c: dict(v) for c, v in RELEASE[release].items()},
                class_cell_types={c: list(t) for c, t in CLASS_CELL_TYPES.items()},
                default_class=CLASS_DEFAULT,
                function='r_c(V) = clip(r_v4(V_A,c) + G_c*R_MAX/(E_exc-E_inh)*(V-V_A,c), 0, R_MAX)',
                photoreceptor_derivation=derive_gain(MEASURED_GAIN_PHOTORECEPTOR,
                                                     W_REF_PHOTORECEPTOR, ANCHOR_PHOTORECEPTOR_MV),
                spec='docs/LIF_DYNAMICS_SPEC.md#10',
                declaration_lock='docs/receipts/graded_gain_v6a_declaration.locked.md')


def classes_from_cell_types(cell_types) -> np.ndarray:
    ct = np.asarray([str(c) for c in cell_types], dtype=object)
    out = np.full(len(ct), CLASS_DEFAULT, dtype=object)
    for c, types in CLASS_CELL_TYPES.items():
        out[np.isin(ct.astype(str), types)] = c
    return out


def release_sha256(release: str, class_per_neuron: np.ndarray, graded: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(RELEASE[release], sort_keys=True).encode())
    codes = np.array([CLASSES.index(str(c)) for c in class_per_neuron], dtype=np.int8)
    codes = np.where(np.asarray(graded, dtype=bool), codes, -1).astype(np.int8)
    h.update(codes.tobytes())
    return h.hexdigest()


def resolve(n: int, graded: np.ndarray, *, release: Optional[str] = None, release_classes=None,
            e_inh: float = E_INH_MV, dt: float = 0.1, connectome_dir=None,
            required: bool = True) -> dict:
    """Per-neuron release zero ``rel_v0`` and slope ``rel_k`` for a graph of ``n``.

    ``release_classes`` (one class name per neuron) is for synthetic graphs;
    otherwise classes come from ``cell_type`` in the released neuron table, and
    the real graph without it is refused.  Spiking neurons get v5's values
    (they never emit through this path).
    """
    release = release or RELEASE_PRIMARY
    report = describe(release)
    table = RELEASE[release]
    if release_classes is not None:
        cls = np.asarray([str(c) for c in release_classes], dtype=object)
        if len(cls) != n or not set(cls) <= set(CLASSES):
            raise ValueError(f'release_classes needs one of {CLASSES} per neuron')
        report['resolved'] = 'explicit per-neuron classes (synthetic graph)'
    else:
        from .graded_policy import load_cell_classes
        from .graph_identity import GraphUnavailable
        try:
            ct, _ = load_cell_classes(connectome_dir)
        except (FileNotFoundError, OSError, GraphUnavailable, ImportError, ValueError) as exc:
            if required:
                raise GraphUnavailable(f'LIF v6a needs the cell-type table to assign release classes: {exc}')
            ct = None
        if ct is not None and len(ct) != n:
            if required:
                raise GraphUnavailable(f'cell-type table has {len(ct)} rows, graph has {n}')
            ct = None
        if ct is None:
            cls = np.full(n, CLASS_DEFAULT, dtype=object)
            report['resolved'] = f'UNRESOLVED (no cell-type table); every neuron {CLASS_DEFAULT}'
        else:
            cls = classes_from_cell_types(ct)
            report['resolved'] = 'presynaptic cell_type from normalized/neurons.feather'
    rel_scale = R_MAX_HZ * dt * 1e-3 / (E_EXC_MV - e_inh)
    rel_v0 = np.full(n, e_inh, dtype=np.float64)
    rel_k = np.full(n, rel_scale, dtype=np.float64)
    graded_b = np.asarray(graded, dtype=bool)
    for c in CLASSES:
        gain = float(table[c]['gain'])
        if gain == 1.0:
            continue   # v4/v5's line exactly, whatever the anchor
        anchor = float(table[c]['anchor_mV'])
        sel = graded_b & (cls == c)
        rel_v0[sel] = anchor - (anchor - e_inh) / gain
        rel_k[sel] = gain * rel_scale
    report.update(neurons=int(n),
                  graded_class_counts={c: int(np.sum(graded_b & (cls == c))) for c in CLASSES},
                  release_sha256=release_sha256(release, cls, graded_b))
    return dict(rel_v0=rel_v0, rel_k=rel_k, classes=cls, report=report)


def release_hz(v: np.ndarray, rel_v0: np.ndarray, rel_k: np.ndarray, *, e_inh: float = E_INH_MV,
               dt: float = 0.1) -> np.ndarray:
    """Release rate in s^-1 a graded cell at ``v`` has (host-side, for probes)."""
    rel_scale = R_MAX_HZ * dt * 1e-3 / (E_EXC_MV - e_inh)
    x = np.clip((np.asarray(v, np.float64) - rel_v0) * rel_k, 0.0, (E_EXC_MV - e_inh) * rel_scale)
    return x * 1000.0 / dt


RELEASE = _tables()
