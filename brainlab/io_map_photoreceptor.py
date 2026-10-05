"""Photoreceptor-level optomotor encoder: a NEW, separately pinned io-map variant.

Pre-registration: ``docs/PHOTORECEPTOR_ENCODER.md`` (locked verbatim copy at
``docs/receipts/photoreceptor_encoder_declaration.locked.md``).

Why this module exists and why it is separate:

* ``brainlab/io_map.py`` drives direction-selective ``T4a/T5a`` and ``T4b/T5b``
  populations per eye according to the stimulus direction, so the direction
  selectivity is *imposed by the encoder*.  That module, its
  ``OPTOMOTOR_IO_PIN`` and every number published under it
  (``docs/WP5_OPTOMOTOR.md`` §12, §13) are **not touched** by this file.
* This module delivers a moving luminance grating to ``R1-R6``
  photoreceptors **only**, resolved by cell type and the released ``rootSide``
  annotation, and lets the graph's own R1-R6 -> lamina -> medulla -> T4/T5
  circuitry decide whether anything is directional.  It has its own resolved-map
  digest (:data:`PHOTORECEPTOR_IO_PIN`) and its own name.

Declared, not biological claims (all in §1-§2 of the pre-registration):

* Retinal position comes from the connectome: each photoreceptor inherits the
  optic-lobe column (``assignedOlHex1/2``) of its dominant ``L1``/``L2``/``L3``
  target, the azimuth lattice direction is ``hex1 - hex2`` (fixed by regressing
  ``L1`` ``somaLocation`` on the hex indices and comparing with the anatomical
  anterior-posterior axis), and one ommatidial step is 5.0 deg.
* Drive is ``i_max * luminance`` with ``i_max = 20.0`` and 10 % multiplicative
  noise, both inherited unchanged from ``docs/wp5_optomotor_prereg.json``.  Arm
  ``P1`` (primary) keeps the mean-luminance DC term, so gray is a lit gray
  screen; arm ``P2`` removes it, so gray is zero drive as in §12/§13.
* The decoder is ``brainlab.io_map.DNa02YawDecoder``, imported unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .graph_identity import GraphUnavailable, resolve_connectome_dir, resolve_graph_dir, sha256_json
from .io_map import SILENCE_DRIVE, DNa02YawDecoder  # noqa: F401  (re-exported unchanged)

PHOTORECEPTOR_TYPE = 'R1-R6'
LAMINA_COLUMN_TYPES = ('L1', 'L2', 'L3')
EYES = ('L', 'R')

# Interommatidial angle, deg per column step.  Literature constant (Land 1997,
# Annu Rev Entomol 42:147-177; reported Drosophila range 4.6-5.5 deg).  Not fitted.
DELTA_PHI_DEG = 5.0

# Recorded (never driven), by cell type and annotated side.  The whole pathway,
# so the pre-registered §4 diagnostic can be produced whatever the outcome.
PATHWAY_MONITORS: Dict[str, tuple] = {
    'R1-R6': ('R1-R6',),
    'L1': ('L1',), 'L2': ('L2',), 'L3': ('L3',), 'L4': ('L4',), 'L5': ('L5',),
    'Lai': ('Lai',), 'T1': ('T1',), 'C2': ('C2',), 'C3': ('C3',),
    'Mi1': ('Mi1',), 'Mi4': ('Mi4',), 'Mi9': ('Mi9',),
    'Tm1': ('Tm1',), 'Tm2': ('Tm2',), 'Tm3': ('Tm3',), 'Tm4': ('Tm4',), 'Tm9': ('Tm9',),
    'CT1': ('CT1',), 'Dm9': ('Dm9',), 'TmY15': ('TmY15',),
    'T4a': ('T4a',), 'T4b': ('T4b',), 'T4c': ('T4c',), 'T4d': ('T4d',),
    'T5a': ('T5a',), 'T5b': ('T5b',), 'T5c': ('T5c',), 'T5d': ('T5d',),
    'HS': ('HSN', 'HSE', 'HSS'), 'H2': ('H2',), 'VS': ('VS',),
    'LLPC1': ('LLPC1',), 'LPi34': ('LPi34',), 'PFL3': ('PFL3',),
    'DNa01': ('DNa01',), 'DNa02': ('DNa02',), 'DNa03': ('DNa03',),
    'DNb01': ('DNb01',), 'DNp09': ('DNp09',), 'MDN': ('MDN',),
}
# Stage order for the §4 "where does the signal die" table.
PATHWAY_STAGES = (
    ('retina', ('R1-R6',)),
    ('lamina', ('L1', 'L2', 'L3', 'L4', 'L5', 'Lai', 'T1')),
    ('medulla', ('Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'C2', 'C3', 'CT1', 'Dm9')),
    ('T4/T5', ('T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d')),
    ('LPTC', ('HS', 'H2', 'VS', 'LPi34', 'LLPC1', 'TmY15')),
    ('DN', ('DNa02', 'DNa01', 'DNa03', 'DNb01', 'DNp09', 'MDN', 'PFL3')),
)
STEERING_TYPE = 'DNa02'

# sha256 of the resolved map (photoreceptor source IDs and their integer azimuth
# assignment per eye, the DNa02 pair, and the layout rule).  Set after the first
# verified resolution and checked on every load.
PHOTORECEPTOR_IO_PIN = '0931ec2fdc9c812af0477e3bc3e6b2077b07eedf61db8edbf4afeab1da943222'


@dataclass
class PhotoreceptorIOMap:
    photoreceptors: Dict[str, np.ndarray]        # 'R_L' / 'R_R' -> node indices
    azimuth_deg: Dict[str, np.ndarray]          # same keys -> counter-clockwise azimuth psi
    populations: Dict[str, np.ndarray]           # 'DNa02_L' / 'DNa02_R'
    monitors: Dict[str, np.ndarray]              # '<group>_<side>' -> node indices
    source_ids: Dict[str, List[int]] = field(default_factory=dict)
    layout: dict = field(default_factory=dict)
    unresolved: dict = field(default_factory=dict)
    sha256: str = ''

    def describe(self) -> dict:
        return dict(sha256=self.sha256,
                    driven={k: int(len(v)) for k, v in self.photoreceptors.items()},
                    azimuth_deg_range={k: [round(float(v.min()), 2), round(float(v.max()), 2)]
                                       for k, v in self.azimuth_deg.items() if len(v)},
                    layout=self.layout, unresolved=self.unresolved,
                    monitors={k: int(len(v)) for k, v in self.monitors.items()},
                    steering=dict(DNa02_L=[int(i) for i in self.populations['DNa02_L']],
                                  DNa02_R=[int(i) for i in self.populations['DNa02_R']]))


def _load_tables(connectome_dir: Optional[Path]):
    import pyarrow.feather as feather
    cdir, _ = resolve_connectome_dir(connectome_dir)
    nodes_path = cdir / 'normalized/neurons.feather'
    ann_path = cdir / 'annotations.feather'
    for path in (nodes_path, ann_path):
        if not path.is_file():
            raise GraphUnavailable(f'{path} not found; set NEUROFLY_CONNECTOME_DIR')
    nodes = feather.read_table(nodes_path, columns=['node_index', 'source_id', 'cell_type']).to_pandas()
    ann = feather.read_table(ann_path, columns=['bodyId', 'type', 'somaSide', 'rootSide', 'instance',
                                                'assignedOlHex1', 'assignedOlHex2', 'somaLocation']
                             ).to_pandas().drop_duplicates('bodyId').set_index('bodyId')
    nodes = nodes.join(ann, on='source_id')
    return nodes


def _lattice(nodes, side: str):
    """Azimuth coefficients (c1, c2) for one eye, from L1 soma positions and hex.

    Least squares of ``somaLocation`` on (hex1, hex2) gives the two 3-D lattice
    vectors.  Their difference is the horizontal (azimuth) direction and their
    sum the vertical one; the sign is chosen so azimuth points POSTERIORLY
    (+z, established anatomically from the VNC leg neuromeres).  See §2.2.
    """
    sel = nodes[(nodes.cell_type == 'L1') & (nodes.somaSide == side)
                & nodes.assignedOlHex1.notna() & nodes.somaLocation.notna()]
    pos = np.array([list(p) for p in sel.somaLocation], dtype=float)
    keep = np.array([p is not None and len(p) == 3 for p in sel.somaLocation])
    if keep.sum() < 20:
        raise GraphUnavailable(f'Too few L1 soma positions on side {side} to derive the retinal layout')
    A = np.c_[sel.assignedOlHex1.to_numpy()[keep], sel.assignedOlHex2.to_numpy()[keep], np.ones(int(keep.sum()))]
    coef, *_ = np.linalg.lstsq(A, pos[keep], rcond=None)
    v1, v2 = coef[0], coef[1]
    d_col = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))
    axis = v1 - v2
    axis = np.array([axis[0], 0.0, axis[2]])          # project out dorsoventral
    if axis[2] < 0:
        axis = -axis                                   # + = posterior
    unit = axis / np.linalg.norm(axis)
    c1 = float(v1 @ unit / d_col)
    c2 = float(v2 @ unit / d_col)
    return dict(c1=c1, c2=c2, d_col_nm=float(d_col),
                lattice_vector_hex1_nm=[float(x) for x in v1],
                lattice_vector_hex2_nm=[float(x) for x in v2],
                azimuth_unit_vector=[float(x) for x in unit],
                vertical_vector_nm=[float(x) for x in (v1 + v2)])


LAYOUT_RULE = (
    'R1-R6 driven only, eye = annotated rootSide. Retinal column = the (assignedOlHex1, '
    'assignedOlHex2) of the dominant L1/L2/L3 target by sum |weight|. Posteriority '
    'p = DELTA_PHI_DEG * (c1*hex1 + c2*hex2), median-centred per eye, with (c1, c2) the '
    'projections of the two hex lattice vectors (from least squares of L1 somaLocation on '
    'hex) onto the horizontal lattice direction hex1-hex2 signed posteriorly, divided by the '
    'column pitch. Counter-clockwise azimuth psi = 90 + p (left eye), 270 - p (right eye). '
    'Elevation ignored (vertical grating). docs/PHOTORECEPTOR_ENCODER.md 2.2'
)


def resolve_photoreceptor_io(connectome_dir: Optional[Path] = None, *, arrays=None,
                             graph_dir: Optional[Path] = None,
                             pin: Optional[str] = 'default') -> PhotoreceptorIOMap:
    """Resolve photoreceptors, their retinal azimuth, the DNa02 pair and monitors."""
    nodes = _load_tables(connectome_dir)
    if arrays is None:
        gdir, _ = resolve_graph_dir(graph_dir)
        with np.load(gdir / 'graph.npz', allow_pickle=False) as data:
            arrays = dict(ptr=data['ptr'], post=data['post'], weight=data['weight'])
    ptr, post, weight = arrays['ptr'], arrays['post'], arrays['weight']

    ctype = nodes.cell_type.fillna('')
    soma = nodes.somaSide.fillna('?')
    root = nodes.rootSide.fillna('?')
    # Cell type in the prepared map must agree with the released annotation for
    # every type this map selects (same guard as brainlab/io_map.py).
    selected = set([PHOTORECEPTOR_TYPE, STEERING_TYPE]) | {t for ts in PATHWAY_MONITORS.values() for t in ts}
    typed = nodes[ctype != '']
    bad = typed[(typed.type.notna()) & (typed.type != typed.cell_type) & typed.cell_type.isin(selected)]
    if len(bad):
        raise GraphUnavailable(f'{len(bad)} selected neurons disagree between cell_type and annotation type')

    hex1 = nodes.assignedOlHex1.to_numpy()
    hex2 = nodes.assignedOlHex2.to_numpy()
    node_type = ctype.to_numpy()
    lamina_ok = np.isin(node_type, LAMINA_COLUMN_TYPES) & ~np.isnan(hex1) & ~np.isnan(hex2)

    layout = {side: _lattice(nodes, side) for side in EYES}
    photoreceptors, azimuth, source_ids = {}, {}, {}
    unresolved = {}
    for side in EYES:
        rows = nodes[ctype.eq(PHOTORECEPTOR_TYPE) & root.eq(side)].sort_values('source_id')
        idx, cols, sids = [], [], []
        no_out = no_column = 0
        for node, sid in zip(rows.node_index.to_numpy(), rows.source_id.to_numpy()):
            lo, hi = int(ptr[node]), int(ptr[node + 1])
            if hi <= lo:
                no_out += 1
                continue
            best: Dict[tuple, float] = {}
            for e in range(lo, hi):
                j = int(post[e])
                if lamina_ok[j]:
                    key = (float(hex1[j]), float(hex2[j]))
                    best[key] = best.get(key, 0.0) + abs(float(weight[e]))
            if not best:
                no_column += 1
                continue
            h1, h2 = max(best, key=best.get)
            idx.append(int(node)); cols.append((h1, h2)); sids.append(int(sid))
        if not idx:
            raise GraphUnavailable(f'No photoreceptor resolved a retinal column on side {side}')
        lay = layout[side]
        p = DELTA_PHI_DEG * np.array([lay['c1'] * h1 + lay['c2'] * h2 for h1, h2 in cols])
        p = p - np.median(p)
        psi = (90.0 + p) if side == 'L' else (270.0 - p)
        photoreceptors[f'R_{side}'] = np.asarray(idx, dtype=np.int64)
        azimuth[f'R_{side}'] = psi.astype(np.float64)
        source_ids[f'R_{side}'] = sids
        unresolved[side] = dict(no_out_edges=no_out, no_hex_annotated_lamina_target=no_column)
        layout[side]['n_columns'] = int(len({c for c in cols}))
        layout[side]['posteriority_deg_range'] = [round(float(p.min()), 2), round(float(p.max()), 2)]

    populations = {}
    for side in EYES:
        rows = nodes[ctype.eq(STEERING_TYPE) & soma.eq(side)]
        if len(rows) != 1 or rows.instance.iat[0] != f'{STEERING_TYPE}_{side}':
            raise GraphUnavailable(f'Expected exactly one {STEERING_TYPE}_{side}, found {rows.instance.tolist()}')
        populations[f'DNa02_{side}'] = rows.node_index.to_numpy(dtype=np.int64)
        source_ids[f'DNa02_{side}'] = [int(rows.source_id.iat[0])]

    monitors = {}
    for name, types in PATHWAY_MONITORS.items():
        side_field = root if name == 'R1-R6' else soma
        for side in EYES:
            monitors[f'{name}_{side}'] = np.sort(
                nodes[ctype.isin(types) & side_field.eq(side)].node_index.to_numpy(dtype=np.int64))

    digest = sha256_json(dict(
        rule=LAYOUT_RULE, delta_phi_deg=DELTA_PHI_DEG,
        source_ids=source_ids,
        azimuth_deg={k: [round(float(x), 3) for x in v] for k, v in azimuth.items()},
        layout={s: {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in layout[s].items()} for s in EYES}))
    io = PhotoreceptorIOMap(photoreceptors=photoreceptors, azimuth_deg=azimuth, populations=populations,
                            monitors=monitors, source_ids=source_ids, layout=layout,
                            unresolved=unresolved, sha256=digest)
    expected = PHOTORECEPTOR_IO_PIN if pin == 'default' else pin
    if expected is not None and digest != expected:
        raise GraphUnavailable(f'Photoreceptor IO map digest {digest} differs from pin {expected}')
    return io


class PhotoreceptorGratingEncoder:
    """Moving luminance grating -> drive on R1-R6 only.

    ``slip_rad_s`` > 0 means the pattern rotates counter-clockwise seen from
    above (leftward), i.e. toward increasing counter-clockwise azimuth: the same
    physical stimulus the WP5 preregistration's ``s = +1`` denotes.  Nothing
    downstream of the photoreceptors is touched, so any direction selectivity
    must be computed by the graph.

    ``arm='P1'`` (primary, luminance-faithful) keeps the mean-luminance DC term,
    so a gray screen delivers ``0.5 * i_max``; ``arm='P2'`` (declared sensitivity
    arm) removes it, so a gray screen delivers zero.
    """

    ARMS = ('P1', 'P2')

    def __init__(self, io: PhotoreceptorIOMap, rng: np.random.Generator, *, i_max: float = 20.0,
                 spatial_period_deg: float = 30.0, noise_sd: float = 0.1, arm: str = 'P1'):
        if arm not in self.ARMS:
            raise ValueError(f'arm must be one of {self.ARMS}')
        self.io = io
        self.rng = rng
        self.i_max = float(i_max)
        self.spatial_period_deg = float(spatial_period_deg)
        self.noise_sd = float(noise_sd)
        self.arm = arm
        self.keys = tuple(sorted(io.photoreceptors))
        self.nodes = {k: io.photoreceptors[k] for k in self.keys}
        self.psi = {k: io.azimuth_deg[k] for k in self.keys}

    def describe(self) -> dict:
        return dict(model='moving sinusoidal luminance grating on R1-R6 photoreceptors only; '
                          'direction selectivity NOT imposed (docs/PHOTORECEPTOR_ENCODER.md)',
                    arm=self.arm,
                    arm_meaning=('P1 primary: drive = i_max * luminance, gray = 0.5*i_max'
                                 if self.arm == 'P1' else
                                 'P2 declared arm: DC removed, gray = 0 drive'),
                    i_max=self.i_max, spatial_period_deg=self.spatial_period_deg,
                    noise_sd=self.noise_sd, delta_phi_deg=DELTA_PHI_DEG,
                    driven={k: int(len(v)) for k, v in self.nodes.items()},
                    units='per-neuron drive, upstream mV-equivalent (brainlab LIF); luminance in [0,1]',
                    io_map_sha256=self.io.sha256, layout_rule=LAYOUT_RULE)

    def luminance(self, key: str, t_ms: float, slip_rad_s: float, contrast: float) -> np.ndarray:
        tf_hz = abs(math.degrees(slip_rad_s)) / self.spatial_period_deg
        s = 1.0 if slip_rad_s > 0 else (-1.0 if slip_rad_s < 0 else 0.0)
        phase = (2 * math.pi * self.psi[key] / self.spatial_period_deg
                 - 2 * math.pi * tf_hz * s * t_ms / 1000.0)
        return 0.5 * (1.0 + contrast * np.cos(phase))

    def encode(self, currents: np.ndarray, t_ms: float, slip_rad_s: float, contrast: float) -> dict:
        """Add drive into ``currents`` in place; return per-eye drive totals.

        Noise is drawn for every photoreceptor on every call, whatever the
        stimulus, so the random stream never depends on the stimulus or on
        whether the drive is delivered (the sham condition).
        """
        contrast = min(1.0, max(0.0, float(contrast)))
        totals = {}
        for key in self.keys:
            noise = self.rng.standard_normal(len(self.nodes[key]))
            lum = self.luminance(key, t_ms, slip_rad_s, contrast)
            if self.arm == 'P2':
                lum = lum * contrast
            drive = np.maximum(0.0, self.i_max * lum * (1.0 + self.noise_sd * noise)).astype(np.float32)
            currents[self.nodes[key]] += drive
            totals[key] = float(drive.sum())
        return totals


class PhotoreceptorOptomotorLoop:
    """One step: grating -> photoreceptors -> graph -> DNa02 decoder.  No other inputs.

    Mirrors ``brainlab.io_map.OptomotorLoop`` (same sham and silencing semantics,
    same step size, same decoder) but drives photoreceptors and records the whole
    pathway.  ``instance`` is an ``experiment_registry.GraphInstance`` or anything
    with ``step(currents, ms)`` returning ``.counts`` and a ``brain`` with ``v``.
    """

    def __init__(self, instance, io: PhotoreceptorIOMap, encoder: PhotoreceptorGratingEncoder,
                 decoder: DNa02YawDecoder, *, deliver_sensory: bool = True, silence: tuple = (),
                 step_ms: float = 2.0):
        self.instance = instance
        self.io = io
        self.encoder = encoder
        self.decoder = decoder
        self.deliver_sensory = bool(deliver_sensory)
        self.silence = tuple(silence)
        self.silence_nodes = (np.concatenate([io.populations[name] for name in self.silence])
                              if self.silence else np.zeros(0, np.int64))
        self.step_ms = float(step_ms)
        self.t_ms = 0.0
        n = instance.brain.n
        self._currents = np.zeros(n, dtype=np.float32)
        self._scratch = np.zeros(n, dtype=np.float32)

    def describe(self) -> dict:
        return dict(deliver_sensory=self.deliver_sensory, silence=list(self.silence),
                    silence_drive=SILENCE_DRIVE if self.silence else None, step_ms=self.step_ms,
                    engineered_assistance=[], other_inputs='none',
                    driven_stage='R1-R6 photoreceptors only')

    def step(self, slip_rad_s: float, contrast: float) -> dict:
        self._scratch.fill(0.0)
        totals = self.encoder.encode(self._scratch, self.t_ms, slip_rad_s, contrast)
        self._currents.fill(0.0)
        if self.deliver_sensory:
            self._currents += self._scratch
        if len(self.silence_nodes):
            self._currents[self.silence_nodes] = SILENCE_DRIVE
        result = self.instance.step(self._currents, self.step_ms)
        counts = result.counts
        motor = self.decoder.decode(counts, self.step_ms)
        v = self.instance.brain.v
        monitors = {k: int(counts[idx].sum()) for k, idx in self.io.monitors.items()}
        self.t_ms += self.step_ms
        return dict(t_ms=self.t_ms, slip_rad_s=slip_rad_s, contrast=contrast,
                    delivered=self.deliver_sensory, encoder_totals=totals,
                    monitors=monitors, total_spikes=int(counts.sum()),
                    v_dna02_l=float(v[self.io.populations['DNa02_L']].mean()),
                    v_dna02_r=float(v[self.io.populations['DNa02_R']].mean()), **motor)


def direction_selectivity_index(rate_pos: float, rate_neg: float):
    """DSI = (r+ - r-)/(r+ + r-); ``None`` when both rates are zero (§5)."""
    total = rate_pos + rate_neg
    if total <= 0:
        return None
    return (rate_pos - rate_neg) / total
