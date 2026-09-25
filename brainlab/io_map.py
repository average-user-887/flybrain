"""Eye-specific optomotor sensor encoder and DNa02 yaw decoder (WP5).

Every neuron here is resolved by stable body ID, cell type and the released
``somaSide`` annotation, never by dataframe row order.  The resolved map is
hashed and pinned (:data:`OPTOMOTOR_IO_PIN`); a different map raises
:class:`~brainlab.graph_identity.GraphUnavailable`.

Separation of concerns (docs/WP5_OPTOMOTOR.md):

* :class:`OptomotorEncoder` turns a retinal-slip velocity (rad/s, + = pattern
  rotating counter-clockwise seen from above) and a contrast into per-neuron
  drive on direction-selective T4/T5 cells of each eye.  Nothing else.
* The graph (``Brain``/``GraphInstance``) integrates it.  No stimulus value
  reaches any other neuron.
* :class:`DNa02YawDecoder` turns DNa02 spikes into a yaw rate.  Zero spikes
  give zero yaw; there is no clipping, floor, positive-only rectification or
  fallback command.
* Body dynamics (heading integration) belong to the caller/arena.

Engineering assumptions (declared, not biological claims):

* Direction selectivity is imposed by the encoder choosing T4/T5 subtypes
  (a = front-to-back, b = back-to-front; Maisak et al. 2013, Nature 500:212).
  The LIF proxy cannot compute Hassenstein-Reichardt/Barlow-Levick direction
  selectivity from photoreceptors, so the medulla stage is bypassed.
* Drive is a rectified sinusoid at the stimulus temporal frequency with a
  per-cell phase (column positions are not used) and multiplicative noise.
  Amplitude is linear in contrast; T4/T5 temporal tuning is not modelled.
* The yaw gain is a fixed unit conversion, not fitted to any behaviour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .graph_identity import GraphUnavailable, resolve_connectome_dir, sha256_json

# T4/T5 subtype preferred directions in the lobula plate (Maisak et al. 2013).
FRONT_TO_BACK = ('T4a', 'T5a')   # layer 1 (HS dendrites)
BACK_TO_FRONT = ('T4b', 'T5b')   # layer 2 (H2 / LPi2c)
EYES = ('L', 'R')

# Neurons recorded (never driven) to show where the response goes.
MONITOR_TYPES: Dict[str, tuple] = {
    'HS': ('HSN', 'HSE', 'HSS'), 'H2': ('H2',), 'VS': ('VS',), 'LLPC1': ('LLPC1',),
    'PFL3': ('PFL3',), 'DNa01': ('DNa01',), 'DNa02': ('DNa02',), 'DNa03': ('DNa03',),
    'DNb01': ('DNb01',), 'DNp09': ('DNp09',), 'MDN': ('MDN',),
}

# sha256 of the resolved populations (source IDs per population, sorted); set
# after the first verified resolution and checked on every load.
OPTOMOTOR_IO_PIN = '228c1b69251e73a3ec34d551202b98f8b6afddf723a77c89e3203053d2366c7b'
VISUAL_HEADING_IO_PIN = '7516d2da93cb466e2200f02998a6388f12debfc2e01ef6b302118d5a50d34948'

# Steering descending neuron.  Rayshubskiy et al. 2020 (bioRxiv 2020.04.04.024703,
# "Neural control of steering in walking Drosophila"): DNa02 activity predicts and
# unilateral activation evokes IPSILATERAL turning.  In MaleCNS v1.0 each DNa02's
# output synapses are >90 % on the same side as its soma (docs/WP5_OPTOMOTOR.md),
# so the soma side is the side it steers toward.
STEERING_TYPE = 'DNa02'


@dataclass
class OptomotorIOMap:
    populations: Dict[str, np.ndarray]          # name -> node indices (int64)
    source_ids: Dict[str, List[int]]
    matched_counts: Dict[str, int]
    available_counts: Dict[str, int]
    monitors: Dict[str, np.ndarray] = field(default_factory=dict)   # 'HS_L' -> nodes
    sha256: str = ''

    def describe(self) -> dict:
        return dict(sha256=self.sha256, matched_counts=self.matched_counts,
                    available_counts=self.available_counts,
                    populations={k: len(v) for k, v in self.populations.items()},
                    monitors={k: [int(i) for i in v] if len(v) <= 4 else len(v)
                              for k, v in self.monitors.items()},
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
    ann = feather.read_table(ann_path, columns=['bodyId', 'type', 'somaSide', 'instance']).to_pandas()
    ann = ann.drop_duplicates('bodyId').set_index('bodyId')
    nodes = nodes.join(ann, on='source_id')
    return nodes


def resolve_optomotor_io(connectome_dir: Optional[Path] = None, *, pin: Optional[str] = 'default') -> OptomotorIOMap:
    """Resolve encoder, decoder and monitor neurons by type + annotated soma side."""
    nodes = _load_tables(connectome_dir)
    ctype = nodes.cell_type.fillna('')
    side = nodes.somaSide.fillna('?')
    # Cell type in the prepared map must agree with the released annotation.
    typed = nodes[ctype != '']
    disagree = typed[(typed.type.notna()) & (typed.type != typed.cell_type)]
    selected_types = set(FRONT_TO_BACK + BACK_TO_FRONT + (STEERING_TYPE,))
    bad = disagree[disagree.cell_type.isin(selected_types)]
    if len(bad):
        raise GraphUnavailable(f'{len(bad)} selected neurons disagree between cell_type and annotation type')

    def members(types, eye):
        rows = nodes[ctype.isin(types) & side.eq(eye)].sort_values('source_id')
        return rows

    available, per_type = {}, {}
    for t in FRONT_TO_BACK + BACK_TO_FRONT:
        for eye in EYES:
            rows = members((t,), eye)
            per_type[(t, eye)] = rows
            available[f'{t}_{eye}'] = int(len(rows))
    # Matched contrast: every (eye, direction) population gets the same number
    # of T4 and of T5 cells, chosen as the lowest source IDs (stable, not row order).
    n_t4 = min(v for k, v in available.items() if k.startswith('T4'))
    n_t5 = min(v for k, v in available.items() if k.startswith('T5'))
    matched = {'T4': n_t4, 'T5': n_t5}
    populations, source_ids = {}, {}
    for eye in EYES:
        for label, types in (('ftb', FRONT_TO_BACK), ('btf', BACK_TO_FRONT)):
            frames = [per_type[(t, eye)].iloc[:matched[t[:2]]] for t in types]
            idx = np.concatenate([f.node_index.to_numpy(dtype=np.int64) for f in frames])
            populations[f'{label}_{eye}'] = idx
            source_ids[f'{label}_{eye}'] = [int(s) for f in frames for s in f.source_id]
    for eye in EYES:
        rows = nodes[ctype.eq(STEERING_TYPE) & side.eq(eye)]
        if len(rows) != 1 or rows.instance.iat[0] != f'{STEERING_TYPE}_{eye}':
            raise GraphUnavailable(f'Expected exactly one {STEERING_TYPE}_{eye}, found {rows.instance.tolist()}')
        populations[f'DNa02_{eye}'] = rows.node_index.to_numpy(dtype=np.int64)
        source_ids[f'DNa02_{eye}'] = [int(rows.source_id.iat[0])]
    monitors = {}
    for name, types in MONITOR_TYPES.items():
        for eye in EYES:
            monitors[f'{name}_{eye}'] = np.sort(nodes[ctype.isin(types) & side.eq(eye)].node_index.to_numpy(dtype=np.int64))
    for name in list(populations):
        if not len(populations[name]):
            raise GraphUnavailable(f'Population {name} resolved empty')
    digest = sha256_json(dict(source_ids=source_ids, rule='T4/T5 a|b by somaSide, matched lowest source_id; '
                                                          'DNa02 by instance+somaSide'))
    io = OptomotorIOMap(populations=populations, source_ids=source_ids, matched_counts=matched,
                        available_counts=available, monitors=monitors, sha256=digest)
    expected = OPTOMOTOR_IO_PIN if pin == 'default' else pin
    if expected is not None and digest != expected:
        raise GraphUnavailable(f'Optomotor IO map digest {digest} differs from pin {expected}')
    return io


class OptomotorEncoder:
    """Retinal slip -> drive on direction-selective T4/T5 of each eye.

    ``slip_rad_s`` > 0 means the pattern rotates counter-clockwise seen from
    above (leftward).  Then the LEFT eye sees front-to-back motion (T4a/T5a L)
    and the RIGHT eye back-to-front motion (T4b/T5b R); < 0 mirrors this.
    """

    def __init__(self, io: OptomotorIOMap, rng: np.random.Generator, *, i_max: float = 20.0,
                 spatial_period_deg: float = 30.0, noise_sd: float = 0.1):
        self.io = io
        self.rng = rng
        self.i_max = float(i_max)
        self.spatial_period_deg = float(spatial_period_deg)
        self.noise_sd = float(noise_sd)
        self.phase = {k: rng.uniform(0, 2 * math.pi, len(io.populations[k]))
                      for k in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R')}

    def describe(self) -> dict:
        return dict(model='rectified-sinusoid drive on T4/T5 subtypes by eye', i_max=self.i_max,
                    spatial_period_deg=self.spatial_period_deg, noise_sd=self.noise_sd,
                    units='per-neuron drive, upstream mV-equivalent (brainlab LIF)', io_map_sha256=self.io.sha256)

    @staticmethod
    def driven_populations(slip_rad_s: float) -> tuple:
        if slip_rad_s > 0:
            return ('ftb_L', 'btf_R')
        if slip_rad_s < 0:
            return ('ftb_R', 'btf_L')
        return ()

    def encode(self, currents: np.ndarray, t_ms: float, slip_rad_s: float, contrast: float) -> dict:
        """Add drive into ``currents`` in place; return per-population totals.

        Noise is drawn for all four populations every call so the random stream
        does not depend on the stimulus or on whether drive is delivered.
        """
        tf_hz = abs(math.degrees(slip_rad_s)) / self.spatial_period_deg
        contrast = min(1.0, max(0.0, float(contrast)))
        driven = self.driven_populations(slip_rad_s)
        totals = {}
        for name in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R'):
            noise = self.rng.standard_normal(len(self.phase[name]))
            if name not in driven or contrast == 0:
                totals[name] = 0.0
                continue
            m = 0.5 * (1.0 + np.cos(2 * math.pi * tf_hz * t_ms / 1000.0 + self.phase[name]))
            drive = np.maximum(0.0, self.i_max * contrast * m * (1.0 + self.noise_sd * noise)).astype(np.float32)
            currents[self.io.populations[name]] += drive
            totals[name] = float(drive.sum())
        return totals


class DNa02YawDecoder:
    """DNa02 spikes -> yaw rate (rad/s, + = counter-clockwise / leftward).

    yaw = gain * (r_L - r_R), r = exponentially filtered spike rate (Hz) of the
    left/right DNa02 (ipsilateral steering).  Linear, unclipped, symmetric:
    silence gives exactly zero yaw.
    """

    def __init__(self, io: OptomotorIOMap, *, gain_rad_s_per_hz: float = 0.02, tau_ms: float = 50.0):
        self.left = io.populations['DNa02_L']
        self.right = io.populations['DNa02_R']
        self.gain = float(gain_rad_s_per_hz)
        self.tau_ms = float(tau_ms)
        self.rate_l = 0.0
        self.rate_r = 0.0

    def describe(self) -> dict:
        return dict(model='yaw = gain*(rate_DNa02_L - rate_DNa02_R)', gain_rad_s_per_hz=self.gain,
                    tau_ms=self.tau_ms, sign='+ = counter-clockwise (leftward) seen from above',
                    clipping=None, rectification=None, fallback=None)

    def reset(self):
        self.rate_l = self.rate_r = 0.0

    def decode(self, counts: np.ndarray, dt_ms: float) -> dict:
        spikes_l = int(counts[self.left].sum())
        spikes_r = int(counts[self.right].sum())
        alpha = 1.0 - math.exp(-dt_ms / self.tau_ms)
        inst_l = spikes_l / (dt_ms / 1000.0) / len(self.left)
        inst_r = spikes_r / (dt_ms / 1000.0) / len(self.right)
        self.rate_l += alpha * (inst_l - self.rate_l)
        self.rate_r += alpha * (inst_r - self.rate_r)
        contrib_l = self.gain * self.rate_l
        contrib_r = -self.gain * self.rate_r
        return dict(spikes_l=spikes_l, spikes_r=spikes_r, rate_l=self.rate_l, rate_r=self.rate_r,
                    yaw_rad_s=contrib_l + contrib_r,
                    contributions={'DNa02_L': contrib_l, 'DNa02_R': contrib_r})


# Hyperpolarising clamp (Kir2.1-like), upstream mV-equivalent.  Under the v1
# current-based dynamics this drags the membrane to about -200 mV, which no
# neuron can do; under the v2 conductance-based dynamics the same value pins the
# membrane at the inhibitory (chloride) reversal, -70 mV, which is what a Kir2.1
# experiment approximates.  Caveat (audit 2026-09-24): under v2/v3 the drive is
# divided by the total conductance, so the clamp holds only while
# (V_rest + g_e*E_exc + g_i*E_inh + drive) / (1 + g_e + g_i) stays below
# threshold, i.e. roughly g_e < 4.6 with no inhibition.  Under v2 it did not:
# the "silenced" DNa02 fired about 240 Hz (docs/receipts/lif_dynamics_v2.json,
# optomotor_rerun.v2.summary.dna02_silenced.side_rates_mean_hz).  The validation
# harness checks it on every run: gate O7 requires the silenced yaw to be exactly
# zero, which in practice means no DNa02 spikes.  See docs/LIF_DYNAMICS_SPEC.md §3.3.
SILENCE_DRIVE = -200.0


class OptomotorLoop:
    """One closed/open-loop step: encode -> graph -> decode.  No other inputs.

    ``instance`` is an ``experiment_registry.GraphInstance`` (or anything with
    ``step(currents, ms)`` returning ``.counts`` and a ``brain`` with ``v``).
    ``deliver_sensory=False`` computes but withholds the encoder output (sham).
    ``silence`` lists population names clamped with :data:`SILENCE_DRIVE`.
    """

    def __init__(self, instance, io: OptomotorIOMap, encoder: OptomotorEncoder, decoder: DNa02YawDecoder, *,
                 deliver_sensory: bool = True, silence: tuple = (), step_ms: float = 2.0):
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
                    engineered_assistance=[], other_inputs='none')

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
        self.last_counts = counts            # read by run recordings (neurofly/recording.py)
        motor = self.decoder.decode(counts, self.step_ms)
        v = self.instance.brain.v
        monitors = {k: int(counts[idx].sum()) for k, idx in self.io.monitors.items()}
        drive_pops = {k: int(counts[self.io.populations[k]].sum()) for k in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R')}
        self.t_ms += self.step_ms
        return dict(t_ms=self.t_ms, slip_rad_s=slip_rad_s, contrast=contrast,
                    delivered=self.deliver_sensory, encoder_totals=totals, encoder_spikes=drive_pops,
                    monitors=monitors, total_spikes=int(counts.sum()),
                    v_dna02_l=float(v[self.io.populations['DNa02_L']].mean()),
                    v_dna02_r=float(v[self.io.populations['DNa02_R']].mean()), **motor)


@dataclass
class VisualHeadingIOMap:
    er_nodes: np.ndarray       # ER4d + ER2 (67 neurons)
    epg_nodes: np.ndarray      # EPG + EPGt (50 neurons)
    el_nodes: np.ndarray       # EL octopaminergic modulators (18 neurons)
    pfl3_nodes: np.ndarray     # PFL3 compass-to-motor steering (24 neurons)
    dna02_nodes: Dict[str, np.ndarray]  # 'DNa02_L', 'DNa02_R'
    plastic_edges: np.ndarray  # exactly 3,081 edges
    sha256: str = ''

    def describe(self) -> dict:
        return dict(
            er_neurons=int(len(self.er_nodes)),
            epg_neurons=int(len(self.epg_nodes)),
            el_neurons=int(len(self.el_nodes)),
            pfl3_neurons=int(len(self.pfl3_nodes)),
            plastic_edges=int(len(self.plastic_edges)),
            sha256=self.sha256,
        )


def resolve_visual_heading_io(connectome_dir: Optional[Path] = None, graph_dir: Optional[Path] = None,
                              *, pin: Optional[str] = 'default') -> VisualHeadingIOMap:
    """Resolve visual heading compass circuit (ER -> EPG -> PFL3 -> DNa02 + EL modulators)."""
    from .graph_identity import resolve_graph_dir
    gdir, _ = resolve_graph_dir(graph_dir)
    nodes = _load_tables(connectome_dir)
    ctype = nodes.cell_type.fillna('')
    side = nodes.somaSide.fillna('?')

    graph_path = gdir / 'graph.npz'
    if not graph_path.is_file():
        raise GraphUnavailable(f'{graph_path} not found; set NEUROFLY_GRAPH_DIR')
    graph = np.load(graph_path)
    ptr, post = graph['ptr'], graph['post']

    er_types = ('ER4d', 'ER2_a', 'ER2_b', 'ER2_c', 'ER2_d')
    epg_types = ('EPG', 'EPGt')

    er_nodes = np.sort(nodes[ctype.isin(er_types)].node_index.to_numpy(dtype=np.int64))
    epg_nodes = np.sort(nodes[ctype.isin(epg_types)].node_index.to_numpy(dtype=np.int64))
    el_nodes = np.sort(nodes[ctype.isin(('EL',))].node_index.to_numpy(dtype=np.int64))
    pfl3_nodes = np.sort(nodes[ctype.isin(('PFL3',))].node_index.to_numpy(dtype=np.int64))
    dna02_l = np.sort(nodes[ctype.eq('DNa02') & side.eq('L')].node_index.to_numpy(dtype=np.int64))
    dna02_r = np.sort(nodes[ctype.eq('DNa02') & side.eq('R')].node_index.to_numpy(dtype=np.int64))

    epg_set = set(epg_nodes)
    edges = []
    for pre in er_nodes:
        for e in range(ptr[pre], ptr[pre + 1]):
            if post[e] in epg_set:
                edges.append(e)

    edges_arr = np.array(edges, dtype=np.int64)
    digest = sha256_json(dict(er=er_nodes.tolist(), epg=epg_nodes.tolist(), el=el_nodes.tolist(),
                              pfl3=pfl3_nodes.tolist(), edges=edges_arr.tolist()))
    expected = VISUAL_HEADING_IO_PIN if pin == 'default' else pin
    if expected is not None and digest != expected:
        raise GraphUnavailable(f'Visual heading IO map digest {digest} differs from pin {expected}')

    return VisualHeadingIOMap(
        er_nodes=er_nodes,
        epg_nodes=epg_nodes,
        el_nodes=el_nodes,
        pfl3_nodes=pfl3_nodes,
        dna02_nodes={'DNa02_L': dna02_l, 'DNa02_R': dna02_r},
        plastic_edges=edges_arr,
        sha256=digest,
    )
