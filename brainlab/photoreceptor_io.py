"""Retinotopic photoreceptor stimulus for LIF v4 (docs/LIF_DYNAMICS_SPEC.md §7.10c).

This is the encoder that removes the WP5 assumption. ``brainlab.io_map``
delivers direction-selective drive straight onto T4/T5 subtypes, so direction
selectivity is supplied by the encoder; this module delivers a drifting
grating to **R1-R6 only** and leaves every stage from the lamina onward to the
graph.

Every neuron is resolved by ``cell_type`` and by annotated side, never by
dataframe row order:

* R1-R6 carry no ``somaSide`` (3,364 of 3,377 are unannotated) but do carry
  ``rootSide`` / ``instance`` (``R1-R6_L``, ``R1-R6_R``), which is what the eye
  assignment uses.
* R1-R6 also carry no ``assignedOlHex*``, so a photoreceptor's **retinotopic
  position is taken from the graph's own wiring**: its cartridge is the
  ``(assignedOlHex1, assignedOlHex2)`` of the L1-L5 cells it synapses onto in
  the pinned graph.  In neural superposition every R1-R6 of one cartridge
  shares one optical axis, so the cartridge *is* the receptive-field position.
  A photoreceptor that reaches no hex-annotated lamina cell is left undriven
  rather than given a guessed position.

The resolved map is hashed and pinned (:data:`PHOTORECEPTOR_IO_PIN`); a
different map raises :class:`~brainlab.graph_identity.GraphUnavailable`.

Declared engineering assumptions (spec §7.10c), not biological claims:

* Interommatidial angle 5 deg, used only to turn lattice steps into degrees.
* Which lattice axis is horizontal is **not** assumed.  Directions are defined
  in the lattice's own plane and the T4/T5 subtypes' preferred directions are
  the measurement, not an input.
* The drive is a current in upstream mV-equivalent units with ``i_max = 20.0``,
  taken unchanged from the WP5 encoder so the comparison is like-for-like.
  There is no photoreceptor light-response model, no adaptation and no noise.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .graph_identity import (GraphUnavailable, resolve_connectome_dir,
                             resolve_graph_dir, sha256_json)

PHOTORECEPTOR_TYPE = 'R1-R6'
LAMINA_TYPES = ('L1', 'L2', 'L3', 'L4', 'L5')
EYES = ('L', 'R')

#: Interommatidial angle in degrees.  DECLARED ENGINEERING ASSUMPTION; only the
#: ratio of the grating's spatial period to the lattice spacing matters.
INTEROMMATIDIAL_DEG = 5.0

#: Hex-lattice -> plane conventions for :class:`PhotoreceptorGratingEncoder`.
#: ``'axial-v1'`` is the original v4/v5 encoder frame, ``u = h1 + h2/2``,
#: ``w = (sqrt3/2) h2``: it treats (1,-1)/(-1,1) as nearest neighbours.  The
#: released MaleCNS ``assignedOlHex1/2`` lattice has (1,1)/(-1,-1) as nearest
#: neighbours instead (cross-column synapses between hex-annotated columnar
#: cells: +-(1,1) carries ~10x the weight of +-(1,-1), which sits with the
#: second ring), so ``'axial-v1'`` is a shear of the real lattice
#: (``S = [[1, 2/sqrt3], [0, 1]]``), not a rotated/reflected equivalent.
#: ``'malecns-hex-v2'`` (``u = h1 - h2/2``) is the post-finding correction
#: (2026-10-08).  ``'axial-v1'`` stays the default so every published v4/v5
#: number is reproducible bit for bit; v2 must be requested explicitly.
LATTICE_VERSIONS = {
    'axial-v1': ((1.0, 0.5), (0.0, math.sqrt(3.0) / 2.0)),
    'malecns-hex-v2': ((1.0, -0.5), (0.0, math.sqrt(3.0) / 2.0)),
}
DEFAULT_LATTICE = 'axial-v1'


def lattice_matrix(lattice: str = DEFAULT_LATTICE) -> np.ndarray:
    """2x2 matrix taking (hex1, hex2) to the planar frame, in lattice steps."""
    if lattice not in LATTICE_VERSIONS:
        raise ValueError(f'unknown lattice {lattice!r}; choose one of {sorted(LATTICE_VERSIONS)}')
    return np.array(LATTICE_VERSIONS[lattice], dtype=np.float64)


def hex_to_plane_deg(cart: np.ndarray, lattice: str = DEFAULT_LATTICE) -> np.ndarray:
    """(n, 2) integer hex coordinates -> (n, 2) planar position in degrees."""
    cart = np.asarray(cart, dtype=np.float64).reshape(-1, 2)
    return (cart @ lattice_matrix(lattice).T) * INTEROMMATIDIAL_DEG


#: Populations recorded (never driven) along the pathway the spec names.
TRACE_TYPES: Dict[str, tuple] = {
    'R1-R6': ('R1-R6',),
    'L1': ('L1',), 'L2': ('L2',), 'L3': ('L3',), 'L4': ('L4',), 'L5': ('L5',),
    'Mi1': ('Mi1',), 'Mi4': ('Mi4',), 'Mi9': ('Mi9',),
    'Tm1': ('Tm1',), 'Tm2': ('Tm2',), 'Tm3': ('Tm3',), 'Tm4': ('Tm4',), 'Tm9': ('Tm9',),
    'CT1': ('CT1',),
    'T4a': ('T4a',), 'T4b': ('T4b',), 'T4c': ('T4c',), 'T4d': ('T4d',),
    'T5a': ('T5a',), 'T5b': ('T5b',), 'T5c': ('T5c',), 'T5d': ('T5d',),
    'HS': ('HSN', 'HSE', 'HSS', 'HST'),
    'H2': ('H2',),
    'VS': ('VS', 'VST1', 'VST2', 'VSm'),
    'DNa02': ('DNa02',),
    'DNa01': ('DNa01',),
    'PFL3': ('PFL3',),
}

#: sha256 of the resolved map, set after the first verified resolution on the
#: pinned MaleCNS graph: 1,107 driven photoreceptors on the left eye over 300
#: cartridges and 2,228 on the right over 524, plus 42 (5 L, 37 R) that reach no
#: hex-annotated lamina cell and are left undriven.
PHOTORECEPTOR_IO_PIN = '0446223b82a485e5de77fc8a4b99d1cabe2c3a668392c97f46b265963f46e033'


@dataclass
class PhotoreceptorIOMap:
    r_nodes: Dict[str, np.ndarray]        # 'L' / 'R' -> node indices, eye from rootSide/instance
    r_position_deg: Dict[str, np.ndarray]  # 'L' / 'R' -> (n, 2) retinotopic position in degrees
    r_cartridge: Dict[str, np.ndarray]    # 'L' / 'R' -> (n, 2) integer hex coordinates
    undriven: Dict[str, int]              # photoreceptors with no hex-annotated lamina target
    majority_assigned: Dict[str, int]
    trace: Dict[str, np.ndarray]          # 'L1_L' -> node indices
    sha256: str = ''

    def describe(self) -> dict:
        return dict(sha256=self.sha256,
                    photoreceptors={k: int(len(v)) for k, v in self.r_nodes.items()},
                    cartridges={k: int(len({tuple(c) for c in self.r_cartridge[k]}))
                                for k in self.r_nodes},
                    undriven=dict(self.undriven),
                    majority_assigned=dict(self.majority_assigned),
                    trace={k: int(len(v)) for k, v in self.trace.items()},
                    interommatidial_deg=INTEROMMATIDIAL_DEG)


def _tables(connectome_dir: Optional[Path]):
    import pyarrow.feather as feather
    cdir, _ = resolve_connectome_dir(connectome_dir)
    nodes_path = cdir / 'normalized/neurons.feather'
    ann_path = cdir / 'annotations.feather'
    for path in (nodes_path, ann_path):
        if not path.is_file():
            raise GraphUnavailable(f'{path} not found; set NEUROFLY_CONNECTOME_DIR')
    nodes = feather.read_table(nodes_path,
                              columns=['node_index', 'source_id', 'cell_type']).to_pandas()
    ann = feather.read_table(ann_path, columns=['bodyId', 'somaSide', 'rootSide', 'instance',
                                                'assignedOlHex1', 'assignedOlHex2']).to_pandas()
    return nodes.join(ann.drop_duplicates('bodyId').set_index('bodyId'), on='source_id')


def resolve_photoreceptor_io(connectome_dir: Optional[Path] = None,
                             graph_dir: Optional[Path] = None,
                             *, pin: Optional[str] = 'default') -> PhotoreceptorIOMap:
    """Resolve the driven photoreceptors, their retinotopy and the trace populations."""
    nodes = _tables(connectome_dir)
    ctype = nodes.cell_type.fillna('').to_numpy()
    soma = nodes.somaSide.fillna('').to_numpy()
    root = nodes.rootSide.fillna('').to_numpy()
    inst = nodes.instance.fillna('').to_numpy()
    hex1 = nodes.assignedOlHex1.to_numpy(dtype=float)
    hex2 = nodes.assignedOlHex2.to_numpy(dtype=float)
    source_id = nodes.source_id.to_numpy(dtype=np.int64)

    gdir, _ = resolve_graph_dir(graph_dir)
    graph_path = gdir / 'graph.npz'
    if not graph_path.is_file():
        raise GraphUnavailable(f'{graph_path} not found; set NEUROFLY_GRAPH_DIR')
    with np.load(graph_path, allow_pickle=False) as graph:
        ptr, post = graph['ptr'], graph['post']

    lamina = np.isin(ctype, LAMINA_TYPES) & ~np.isnan(hex1) & ~np.isnan(hex2)
    r_all = np.flatnonzero(ctype == PHOTORECEPTOR_TYPE)
    r_nodes: Dict[str, list] = {'L': [], 'R': []}
    r_cart: Dict[str, list] = {'L': [], 'R': []}
    undriven = {'L': 0, 'R': 0}
    majority = {'L': 0, 'R': 0}
    for i in r_all:
        eye = f'{inst[i]}'.rsplit('_', 1)[-1] or f'{root[i]}'
        if eye not in EYES:
            continue
        targets = post[ptr[i]:ptr[i + 1]]
        targets = targets[lamina[targets]]
        if not len(targets):
            undriven[eye] += 1
            continue
        counts = collections.Counter(zip(hex1[targets].astype(int).tolist(),
                                         hex2[targets].astype(int).tolist()))
        best, _ = counts.most_common(1)[0]
        if len(counts) > 1:
            majority[eye] += 1
        r_nodes[eye].append(int(i))
        r_cart[eye].append(best)

    r_node_arrays, r_cart_arrays, r_pos = {}, {}, {}
    for eye in EYES:
        order = np.argsort([source_id[i] for i in r_nodes[eye]])
        idx = np.asarray(r_nodes[eye], dtype=np.int64)[order]
        cart = np.asarray(r_cart[eye], dtype=np.int64)[order]
        if not len(idx):
            raise GraphUnavailable(f'No {PHOTORECEPTOR_TYPE} resolved for eye {eye}')
        r_node_arrays[eye] = idx
        r_cart_arrays[eye] = cart
        # Axial hex -> cartesian, scaled by the interommatidial angle.
        u = (cart[:, 0] + cart[:, 1] / 2.0) * INTEROMMATIDIAL_DEG
        w = (math.sqrt(3.0) / 2.0 * cart[:, 1]) * INTEROMMATIDIAL_DEG
        r_pos[eye] = np.stack([u, w], axis=1).astype(np.float64)

    trace = {}
    for name, types in TRACE_TYPES.items():
        member = np.isin(ctype, types)
        for eye in EYES:
            side = np.where(soma != '', soma, root)
            sel = np.flatnonzero(member & (side == eye))
            if name == PHOTORECEPTOR_TYPE:
                sel = r_node_arrays[eye]
            if len(sel):
                trace[f'{name}_{eye}'] = np.sort(sel).astype(np.int64)

    digest = sha256_json(dict(
        rule='R1-R6 by cell_type, eye from instance/rootSide; retinotopy from the hex '
             'annotation of the L1-L5 cells each photoreceptor synapses onto in the pinned '
             'graph (majority); trace populations by cell_type + somaSide else rootSide',
        interommatidial_deg=INTEROMMATIDIAL_DEG,
        photoreceptors={eye: [int(source_id[i]) for i in r_node_arrays[eye]] for eye in EYES},
        cartridges={eye: r_cart_arrays[eye].tolist() for eye in EYES},
        trace={k: [int(source_id[i]) for i in v] for k, v in sorted(trace.items())}))
    expected = PHOTORECEPTOR_IO_PIN if pin == 'default' else pin
    if expected is not None and digest != expected:
        raise GraphUnavailable(f'Photoreceptor IO map digest {digest} differs from pin {expected}')
    return PhotoreceptorIOMap(r_nodes=r_node_arrays, r_position_deg=r_pos,
                              r_cartridge=r_cart_arrays, undriven=undriven,
                              majority_assigned=majority, trace=trace, sha256=digest)


class PhotoreceptorGratingEncoder:
    """A drifting sinusoidal grating delivered to R1-R6 only.

    ``direction_deg`` is the drift direction in the retinotopic lattice plane.
    Which lattice direction corresponds to front-to-back in the animal is
    deliberately not assumed; it is what the T4/T5 measurement reports.
    """

    def __init__(self, io: PhotoreceptorIOMap, *, i_max: float = 20.0,
                 spatial_period_deg: float = 30.0, temporal_frequency_hz: float = 1.5,
                 lattice: str = DEFAULT_LATTICE):
        self.io = io
        self.i_max = float(i_max)
        self.spatial_period_deg = float(spatial_period_deg)
        self.temporal_frequency_hz = float(temporal_frequency_hz)
        lattice_matrix(lattice)  # validates the name
        self.lattice = lattice
        # axial-v1 keeps the map's own positions object, so it stays bit-identical.
        self.position_deg = (io.r_position_deg if lattice == 'axial-v1' else
                             {eye: hex_to_plane_deg(io.r_cartridge[eye], lattice) for eye in EYES})

    def describe(self) -> dict:
        out = dict(model='drifting sinusoidal grating on R1-R6 only, retinotopic by cartridge',
                   i_max=self.i_max, spatial_period_deg=self.spatial_period_deg,
                   temporal_frequency_hz=self.temporal_frequency_hz,
                   noise=None, direction_selectivity='NOT supplied by this encoder',
                   units='per-neuron drive current (upstream mV-equivalent)',
                   io_map_sha256=self.io.sha256,
                   interommatidial_deg=INTEROMMATIDIAL_DEG)
        if self.lattice != 'axial-v1':  # the v1 description stays exactly as published
            out.update(lattice=self.lattice, lattice_matrix=lattice_matrix(self.lattice).tolist(),
                       correction='post-finding correction 2026-10-08 (hex-lattice shear)')
        return out

    def encode(self, currents: np.ndarray, t_ms: float, direction_deg: float,
               contrast: float) -> dict:
        contrast = min(1.0, max(0.0, float(contrast)))
        theta = math.radians(float(direction_deg))
        k = 2 * math.pi / self.spatial_period_deg
        phase_t = 2 * math.pi * self.temporal_frequency_hz * t_ms / 1000.0
        totals = {}
        for eye in EYES:
            pos = self.position_deg[eye]
            projection = pos[:, 0] * math.cos(theta) + pos[:, 1] * math.sin(theta)
            drive = self.i_max * 0.5 * (1.0 + contrast * np.cos(k * projection - phase_t))
            drive = drive.astype(np.float32)
            currents[self.io.r_nodes[eye]] += drive
            totals[f'R1-R6_{eye}'] = float(drive.sum())
        return totals

    # -- added for LIF v5 (docs/LIF_DYNAMICS_SPEC.md §9.9); ``encode`` above is
    # unchanged, and with equal directions ``encode_per_eye`` is bit-identical to it.
    def encode_per_eye(self, currents: np.ndarray, t_ms: float, direction_by_eye: dict,
                       contrast: float) -> dict:
        """The same grating with its own drift direction on each eye.

        A yaw rotation of a drum moves the pattern front-to-back on one eye and
        back-to-front on the other, which no single lattice-plane direction
        applied to both eyes can express.  Same arithmetic as :meth:`encode`.
        """
        contrast = min(1.0, max(0.0, float(contrast)))
        k = 2 * math.pi / self.spatial_period_deg
        phase_t = 2 * math.pi * self.temporal_frequency_hz * t_ms / 1000.0
        totals = {}
        for eye in EYES:
            theta = math.radians(float(direction_by_eye[eye]))
            pos = self.position_deg[eye]
            projection = pos[:, 0] * math.cos(theta) + pos[:, 1] * math.sin(theta)
            drive = self.i_max * 0.5 * (1.0 + contrast * np.cos(k * projection - phase_t))
            drive = drive.astype(np.float32)
            currents[self.io.r_nodes[eye]] += drive
            totals[f'R1-R6_{eye}'] = float(drive.sum())
        return totals

    def encode_flicker(self, currents: np.ndarray, t_ms: float, contrast: float) -> dict:
        """Spatially uniform sinusoidal flicker at the grating's temporal frequency
        and mean: the grating with zero spatial frequency.  Every photoreceptor is
        driven in phase, so a population's mean response phase is its temporal
        filtering, free of retinotopy."""
        contrast = min(1.0, max(0.0, float(contrast)))
        phase_t = 2 * math.pi * self.temporal_frequency_hz * t_ms / 1000.0
        totals = {}
        for eye in EYES:
            n = len(self.io.r_nodes[eye])
            drive = self.i_max * 0.5 * (1.0 + contrast * np.cos(np.zeros(n) - phase_t))
            drive = drive.astype(np.float32)
            currents[self.io.r_nodes[eye]] += drive
            totals[f'R1-R6_{eye}'] = float(drive.sum())
        return totals
