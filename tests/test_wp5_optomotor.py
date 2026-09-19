"""WP5: eye-specific optomotor encoder, DNa02 yaw decoder and loop semantics.

Map-resolution tests read the released annotations shipped in
connectome_data/malecns_v1 (skipped if absent).  Loop tests run on a
SYNTHETIC TEST GRAPH in registry test mode; they check wiring semantics
(sham delivers nothing, silencing clamps, zero spikes give zero yaw), never
a behavioural claim.
"""
import math
import os

import numpy as np
import pytest

from brainlab.cosim_server import ConnectomeServer
from brainlab.graph_identity import DN_CHANNELS, DN_EXPECTED_SIDES, DEFAULT_CONNECTOME_DIR, GraphUnavailable
from brainlab.io_map import (OPTOMOTOR_IO_PIN, SILENCE_DRIVE, DNa02YawDecoder, OptomotorEncoder,
                             OptomotorIOMap, OptomotorLoop, resolve_optomotor_io)
from experiment_registry import ExperimentRegistry, SharedGraph

HAVE_ANNOTATIONS = (DEFAULT_CONNECTOME_DIR / 'annotations.feather').is_file() or bool(
    os.environ.get('NEUROFLY_CONNECTOME_DIR'))
needs_data = pytest.mark.skipif(not HAVE_ANNOTATIONS, reason='released annotations not present')


def small_io(n_per=5):
    pops = {}
    base = 20
    for k, name in enumerate(('ftb_L', 'btf_L', 'ftb_R', 'btf_R')):
        pops[name] = np.arange(base + k * n_per, base + (k + 1) * n_per, dtype=np.int64)
    pops['DNa02_L'] = np.array([10], np.int64)
    pops['DNa02_R'] = np.array([11], np.int64)
    return OptomotorIOMap(populations=pops, source_ids={k: [int(i) for i in v] for k, v in pops.items()},
                          matched_counts={'T4': n_per, 'T5': 0}, available_counts={},
                          monitors={'HS_L': np.array([1]), 'HS_R': np.array([2])}, sha256='test')


# ---------------------------------------------------------------------------
# Map resolution on the released annotations
# ---------------------------------------------------------------------------
@needs_data
def test_io_map_matches_pin_and_annotated_sides():
    import pyarrow.feather as feather
    io = resolve_optomotor_io()
    assert io.sha256 == OPTOMOTOR_IO_PIN
    cdir = DEFAULT_CONNECTOME_DIR
    ann = feather.read_table(cdir / 'annotations.feather', columns=['bodyId', 'type', 'somaSide']).to_pandas()
    ann = ann.drop_duplicates('bodyId').set_index('bodyId')
    expected_types = {'ftb': {'T4a', 'T5a'}, 'btf': {'T4b', 'T5b'}}
    sizes = set()
    for name in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R'):
        direction, eye = name.split('_')
        rows = ann.loc[io.source_ids[name]]
        assert set(rows.somaSide) == {eye}
        assert set(rows.type) == expected_types[direction]
        sizes.add(len(io.populations[name]))
    assert len(sizes) == 1, 'matched contrast needs equal population sizes'
    assert ann.loc[io.source_ids['DNa02_L'][0]].somaSide == 'L'
    assert ann.loc[io.source_ids['DNa02_R'][0]].somaSide == 'R'


@needs_data
def test_io_map_rejects_wrong_pin():
    with pytest.raises(GraphUnavailable, match='differs from pin'):
        resolve_optomotor_io(pin='0' * 64)


@needs_data
def test_dn_channels_are_named_by_verified_soma_side():
    import pyarrow.feather as feather
    nodes = feather.read_table(DEFAULT_CONNECTOME_DIR / 'normalized/neurons.feather',
                               columns=['source_id']).to_pandas()
    ann = feather.read_table(DEFAULT_CONNECTOME_DIR / 'annotations.feather',
                             columns=['bodyId', 'somaSide']).to_pandas().drop_duplicates('bodyId').set_index('bodyId')
    for channel, side in DN_EXPECTED_SIDES.items():
        for index in DN_CHANNELS[channel]:
            assert ann.somaSide[int(nodes.source_id.iat[index])] == side
    io = resolve_optomotor_io()
    assert list(io.populations['DNa02_L']) == DN_CHANNELS['dna02_l']
    assert list(io.populations['DNa02_R']) == DN_CHANNELS['dna02_r']


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
def drive_by_population(io, slip, contrast=1.0, seed=0, t_ms=0.0):
    currents = np.zeros(int(max(v.max() for v in io.populations.values())) + 1, np.float32)
    OptomotorEncoder(io, np.random.default_rng(seed)).encode(currents, t_ms, slip, contrast)
    return {k: float(currents[io.populations[k]].sum()) for k in ('ftb_L', 'btf_L', 'ftb_R', 'btf_R')}, currents


def test_encoder_drives_eye_specific_direction_selective_populations():
    io = small_io()
    ccw, _ = drive_by_population(io, +0.8)
    assert ccw['ftb_L'] > 0 and ccw['btf_R'] > 0 and ccw['btf_L'] == 0 and ccw['ftb_R'] == 0
    cw, _ = drive_by_population(io, -0.8)
    assert cw['ftb_R'] > 0 and cw['btf_L'] > 0 and cw['ftb_L'] == 0 and cw['btf_R'] == 0
    still, currents = drive_by_population(io, 0.0)
    assert not currents.any()
    _, dark = drive_by_population(io, 0.8, contrast=0.0)
    assert not dark.any()


def test_encoder_touches_only_encoder_cells_and_rng_stream_is_stimulus_independent():
    io = small_io()
    _, currents = drive_by_population(io, 0.8)
    touched = set(np.flatnonzero(currents))
    allowed = set(io.populations['ftb_L']) | set(io.populations['btf_R'])
    assert touched <= allowed
    states = []
    for slip in (0.8, -0.8, 0.0):
        rng = np.random.default_rng(3)
        OptomotorEncoder(io, rng).encode(np.zeros(100, np.float32), 0.0, slip, 1.0)
        states.append(rng.bit_generator.state['state']['state'])
    assert len(set(states)) == 1


def test_encoder_left_right_drive_is_matched_in_expectation():
    io = small_io(n_per=400)
    totals = {'ccw': 0.0, 'cw': 0.0}
    for seed in range(20):
        a, _ = drive_by_population(io, 0.8, seed=seed)
        b, _ = drive_by_population(io, -0.8, seed=seed + 100)
        totals['ccw'] += a['ftb_L'] + a['btf_R']
        totals['cw'] += b['ftb_R'] + b['btf_L']
    assert math.isclose(totals['ccw'], totals['cw'], rel_tol=0.05)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def test_decoder_zero_spikes_give_exactly_zero_yaw_and_sign_is_symmetric():
    io = small_io()
    dec = DNa02YawDecoder(io)
    counts = np.zeros(100, np.int32)
    for _ in range(10):
        assert dec.decode(counts, 2.0)['yaw_rad_s'] == 0.0
    counts[10] = 1   # DNa02_L spike -> leftward (+)
    left = DNa02YawDecoder(io).decode(counts, 2.0)
    counts[10], counts[11] = 0, 1
    right = DNa02YawDecoder(io).decode(counts, 2.0)
    assert left['yaw_rad_s'] > 0 and right['yaw_rad_s'] < 0
    assert math.isclose(left['yaw_rad_s'], -right['yaw_rad_s'])
    assert math.isclose(sum(left['contributions'].values()), left['yaw_rad_s'])


def test_decoder_has_no_floor_or_clip():
    io = small_io()
    dec = DNa02YawDecoder(io, tau_ms=2.0)
    counts = np.zeros(100, np.int32)
    counts[11] = 20
    yaw = dec.decode(counts, 2.0)['yaw_rad_s']
    assert yaw < -1.0      # large unclipped rightward command
    counts[11] = 0
    for _ in range(200):
        yaw = dec.decode(counts, 2.0)['yaw_rad_s']
    assert abs(yaw) < 1e-9  # decays back to zero, no tonic residue


# ---------------------------------------------------------------------------
# Loop on a synthetic graph (semantics only)
# ---------------------------------------------------------------------------
def make_instance(tmp_path, name):
    shared = SharedGraph.synthetic(allow_synthetic=True, n=100, k_out=5, seed=1)
    registry = ExperimentRegistry(shared, tmp_path / name, test_mode=True)
    return registry.activate('optomotor', 'connectome-fixed')


def test_sham_loop_delivers_nothing(tmp_path):
    io = small_io()
    instance = make_instance(tmp_path, 'sham')
    loop = OptomotorLoop(instance, io, OptomotorEncoder(io, np.random.default_rng(0)), DNa02YawDecoder(io),
                         deliver_sensory=False)
    for _ in range(50):
        rec = loop.step(0.8, 1.0)
        assert rec['total_spikes'] == 0 and rec['yaw_rad_s'] == 0.0
    assert loop.describe()['engineered_assistance'] == []


def test_intact_loop_delivers_and_silencing_clamps_dna02(tmp_path):
    io = small_io()
    instance = make_instance(tmp_path, 'intact')
    loop = OptomotorLoop(instance, io, OptomotorEncoder(io, np.random.default_rng(0)), DNa02YawDecoder(io))
    spikes = sum(loop.step(0.8, 1.0)['encoder_spikes']['ftb_L'] for _ in range(50))
    assert spikes > 0
    silenced = make_instance(tmp_path, 'silenced')
    loop = OptomotorLoop(silenced, io, OptomotorEncoder(io, np.random.default_rng(0)), DNa02YawDecoder(io),
                         silence=('DNa02_L', 'DNa02_R'))
    for _ in range(50):
        rec = loop.step(0.8, 1.0)
        assert rec['spikes_l'] == 0 and rec['spikes_r'] == 0 and rec['yaw_rad_s'] == 0.0
    assert silenced.brain.v[10] < -52 and silenced.brain.v[11] < -52
    assert loop.describe()['silence_drive'] == SILENCE_DRIVE


# ---------------------------------------------------------------------------
# Server gating of engineered assistance
# ---------------------------------------------------------------------------
def test_server_reports_and_gates_engineered_assistance(tmp_path, monkeypatch):
    monkeypatch.delenv('NEUROFLY_GRAPH_DIR', raising=False)
    on = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    reply = on.step({'looming_trigger': True})
    assert len(reply['engineered_assistance_applied']) == 2
    off = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True, engineered_assistance=False)
    reply = off.step({'looming_trigger': True, 'energy_reserve': 0.1})
    assert reply['engineered_assistance_applied'] == []
    assert reply['total_step_spikes'] == 0, 'nothing else may drive the graph'
    assert off.get_status()['engineered_assistance_enabled'] is False
    assert reply['optomotor'] is None   # the optomotor map is never faked on a synthetic graph
