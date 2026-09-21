"""The declared LIF dynamics versions (docs/LIF_DYNAMICS_SPEC.md).

v1 (current-based) must keep behaving exactly as before; v2
(conductance-based) must honour the declared reversal potentials; v3 must
honour the declared per-sign PSP-preserving calibration and the declared
transmitter policy.  No two of them may ever be silently interchanged.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brainlab import engine
from brainlab.brain import Brain
from brainlab.graph_identity import (DYNAMICS_ENV, DYNAMICS_VERSIONS, LIF_DYNAMICS,
                                     active_dynamics_version, dynamics_pin)
from provenance import BACKENDS, controller_version_for

ROOT = Path(__file__).resolve().parents[1]


def graph(weights, n=2, post=(1,), ptr=(0, 1, 1)):
    return dict(ptr=np.asarray(ptr, np.int64), post=np.asarray(post, np.int32),
                weight=np.asarray(weights, np.float32), ids=np.arange(1, n + 1, dtype=np.int64))


def run(brain, drive0, ms, step=10.0):
    currents = np.zeros(brain.n, np.float32)
    currents[0] = drive0
    for _ in range(int(ms / step)):
        brain.step(currents, step)
    return brain


# -- declaration ------------------------------------------------------------
def test_declared_versions_match_the_compiled_engine():
    v2 = DYNAMICS_VERSIONS['v2']
    assert v2['e_excitatory_mV'] == engine.E_EXC_MV == 0.0
    assert v2['e_inhibitory_mV'] == engine.E_INH_MV == -70.0
    assert v2['g_unit_per_weight'] == pytest.approx(1.0 / 52.0)
    v3 = DYNAMICS_VERSIONS['v3']
    assert v3['g_unit_excitatory_per_weight'] == engine.G_UNIT_EXC_V3 == pytest.approx(1 / 52)
    assert v3['g_unit_inhibitory_per_weight'] == engine.G_UNIT_INH_V3 == pytest.approx(1 / 18)
    assert v3['g_unit_ratio_inh_over_exc'] == pytest.approx(52 / 18)
    assert v3['transmitter_policy']['modulatory_only_transmitters'] == [
        'dopamine', 'octopamine', 'serotonin']
    for version in ('v1', 'v2', 'v3'):
        declared = DYNAMICS_VERSIONS[version]
        assert declared['dynamics_version'] == version
        assert declared['tau_membrane_ms'] == engine.TAU_M_MS
        assert declared['v_threshold_mV'] == engine.V_THRESHOLD_MV
        assert declared['refractory_ms'] == engine.REFRACTORY_MS
        assert declared['spec'] == 'docs/LIF_DYNAMICS_SPEC.md'


def test_the_spec_document_exists_and_names_both_versions():
    text = (ROOT / 'docs/LIF_DYNAMICS_SPEC.md').read_text()
    for token in ('v1', 'v2', 'v3', 'E_exc', 'E_inh', 'Shiu', "O'Dowd",
                  'ENGINEERING ASSUMPTION'.title(), 'Blenau', 'Evans', 'Croset', 'Eckstein',
                  'unclear', 'falsif'):
        assert token.lower() in text.lower()


def test_dynamics_pins_differ_and_are_stable():
    pins = {v: dynamics_pin(v) for v in ('v1', 'v2', 'v3')}
    assert len(set(pins.values())) == 3
    assert dynamics_pin('v2') == pins['v2']
    with pytest.raises(ValueError):
        dynamics_pin('v4')


def test_default_is_v1_so_existing_results_keep_their_meaning(monkeypatch):
    monkeypatch.delenv(DYNAMICS_ENV, raising=False)
    assert active_dynamics_version() == 'v1'
    assert LIF_DYNAMICS['dynamics_version'] == 'v1'
    assert Brain(arrays=graph([1.0])).dynamics == 'v1'


def test_unknown_dynamics_is_refused(monkeypatch):
    monkeypatch.setenv(DYNAMICS_ENV, 'v99')
    with pytest.raises(ValueError):
        active_dynamics_version()
    monkeypatch.delenv(DYNAMICS_ENV)
    with pytest.raises(ValueError):
        Brain(arrays=graph([1.0]), dynamics='conductance')


def test_env_var_selects_v2_in_a_fresh_process():
    code = ('import json;from brainlab.graph_identity import LIF_DYNAMICS;'
            'from brainlab.brain import Brain;import numpy as np;'
            'print(json.dumps([LIF_DYNAMICS["dynamics_version"],'
            'Brain(arrays=dict(ptr=np.array([0,1,1],"i8"),post=np.array([1],"i4"),'
            'weight=np.array([1.],"f4"),ids=np.array([1,2],"i8"))).dynamics]))')
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                         env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(ROOT),
                             DYNAMICS_ENV: 'v2'})
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ['v2', 'v2']


# -- v2 physiology ----------------------------------------------------------
@pytest.mark.parametrize('weight', [-40.0, -400.0, -4000.0])
def test_v2_membrane_never_falls_below_the_inhibitory_reversal(weight):
    brain = run(Brain(arrays=graph([weight]), dynamics='v2'), 20.0, 2000.0)
    assert brain.v.min() >= engine.E_INH_MV - 1e-4
    # ...while v1, for the same input, is unbounded and scales with the weight.
    v1 = run(Brain(arrays=graph([weight]), dynamics='v1'), 20.0, 2000.0)
    if weight <= -400.0:
        assert v1.v.min() < -200.0


def test_v2_membrane_never_rises_above_the_excitatory_reversal():
    brain = run(Brain(arrays=graph([4000.0]), dynamics='v2'), 20.0, 500.0)
    assert brain.v.max() <= engine.E_EXC_MV + 1e-4


def test_huge_negative_drive_is_floored_not_followed():
    """The WP5 Kir2.1-like clamp (-200 units) pins v at the chloride reversal."""
    brain = Brain(arrays=graph([0.0]), dynamics='v2')
    currents = np.full(2, -200.0, np.float32)
    for _ in range(100):
        counts, _ = brain.step(currents, 10.0)
        assert counts.sum() == 0
    assert brain.v.min() == pytest.approx(engine.E_INH_MV, abs=1e-4)
    v1 = Brain(arrays=graph([0.0]), dynamics='v1')
    for _ in range(100):
        v1.step(currents, 10.0)
    assert v1.v.min() < -190.0        # v1 simply follows the drive


def test_unitary_epsp_at_rest_matches_v1():
    """The conductance quantum is calibrated, not fitted: same unitary EPSP."""
    peaks = {}
    for dynamics in ('v1', 'v2'):
        brain = Brain(arrays=graph([0.275]), dynamics=dynamics)
        brain.step(np.array([20.0, 0.0], np.float32), 10.0)
        peak = engine.V_REST_MV
        for _ in range(400):
            brain.step(np.zeros(2, np.float32), 0.1)
            peak = max(peak, float(brain.v[1]))
        peaks[dynamics] = peak - engine.V_REST_MV
    assert peaks['v1'] == pytest.approx(0.0433, abs=2e-4)
    assert peaks['v2'] == pytest.approx(peaks['v1'], rel=0.02)


def test_v2_inhibition_shunts_the_external_drive():
    """Large inhibitory conductance divides the response to injected current."""
    strong = run(Brain(arrays=graph([-400.0, 0.0], n=3, post=(2, 2), ptr=(0, 1, 2, 2)),
                       dynamics='v2'), 20.0, 400.0)
    # neuron 2 receives only inhibition; it must sit between rest and E_inh
    assert engine.E_INH_MV <= strong.v[2] < engine.V_REST_MV


def test_v2_is_deterministic():
    a = run(Brain(arrays=graph([-40.0]), dynamics='v2'), 20.0, 300.0)
    b = run(Brain(arrays=graph([-40.0]), dynamics='v2'), 20.0, 300.0)
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.g, b.g)


def test_e_inh_sensitivity_arm_is_explicit_and_v2_only():
    brain = run(Brain(arrays=graph([-4000.0]), dynamics='v2', e_inh_mV=-56.0), 20.0, 1000.0)
    assert brain.v.min() >= -56.0 - 1e-4
    with pytest.raises(ValueError):
        Brain(arrays=graph([1.0]), dynamics='v1', e_inh_mV=-56.0)


# -- checkpoint separation --------------------------------------------------
def test_state_shapes_differ_so_checkpoints_cannot_be_confused():
    assert Brain(arrays=graph([1.0]), dynamics='v1').g.shape == (2,)
    assert Brain(arrays=graph([1.0]), dynamics='v2').g.shape == (2, 2)


def test_v1_snapshot_is_refused_by_a_v2_brain_and_the_reverse():
    one = run(Brain(arrays=graph([-40.0]), dynamics='v1'), 20.0, 100.0)
    two = run(Brain(arrays=graph([-40.0]), dynamics='v2'), 20.0, 100.0)
    with pytest.raises(ValueError, match='LIF dynamics'):
        two.restore_state(one.snapshot_state())
    with pytest.raises(ValueError, match='LIF dynamics'):
        one.restore_state(two.snapshot_state())
    # ...but a same-version round trip still works.
    three = Brain(arrays=graph([-40.0]), dynamics='v2')
    three.restore_state(two.snapshot_state())
    np.testing.assert_array_equal(three.v, two.v)
    np.testing.assert_array_equal(three.g, two.g)


# -- provenance -------------------------------------------------------------
def test_controller_version_is_repinned_for_a_dynamics_change():
    fixed = BACKENDS['connectome-fixed']
    assert controller_version_for(fixed, DYNAMICS_VERSIONS['v1']) == 'brainlab-lif-v1'
    assert controller_version_for(fixed, DYNAMICS_VERSIONS['v2']) == 'brainlab-lif-v2'
    assert controller_version_for(fixed, None) == 'brainlab-lif-v1'
    plastic = BACKENDS['connectome-plastic']
    assert controller_version_for(plastic, DYNAMICS_VERSIONS['v2']) == 'brainlab-lif-plastic-v2'
    # A modular (non-graph) backend never inherits a graph dynamics version.
    assert controller_version_for(BACKENDS['modular'], DYNAMICS_VERSIONS['v2']) == 'modular-mb-v1'


def test_cosim_server_identity_names_the_dynamics_version(tmp_path):
    from brainlab.cosim_server import ConnectomeServer
    server = ConnectomeServer(graph_dir=tmp_path, allow_synthetic=True)
    fields = server.identity_fields()
    assert fields['lif_dynamics_version'] == server.brain.dynamics == 'v1'
    assert fields['lif_dynamics_pin'] == dynamics_pin('v1')
    server.reset()                      # must work for either g shape
    assert not server.brain.g.any()


def test_run_manifest_records_the_dynamics_version():
    from provenance import RunManifest
    manifest = RunManifest.create(
        backend='connectome-fixed', assay='optomotor', instance_id='x', seed=0,
        graph=dict(synthetic=False, graph_sha256='a' * 64), dynamics=DYNAMICS_VERSIONS['v2'],
        learned_parameter_locations={})
    assert manifest.controller_version == 'brainlab-lif-v2'
    assert manifest.dynamics['dynamics_version'] == 'v2'
    assert manifest.identity()['controller_version'] == 'brainlab-lif-v2'


# -- v3 calibration and transmitter policy ----------------------------------
def test_v3_unitary_ipsp_matches_v1_while_v2_undershoots():
    """The declared per-sign calibration: the v3 IPSP is the v1 IPSP again."""
    troughs = {}
    for dynamics in ('v1', 'v2', 'v3'):
        brain = Brain(arrays=graph([-0.275]), dynamics=dynamics)
        brain.step(np.array([20.0, 0.0], np.float32), 10.0)
        trough = engine.V_REST_MV
        for _ in range(400):
            brain.step(np.zeros(2, np.float32), 0.1)
            trough = min(trough, float(brain.v[1]))
        troughs[dynamics] = engine.V_REST_MV - trough
    assert troughs['v3'] == pytest.approx(troughs['v1'], rel=0.02)
    # v2 gave both signs one quantum, so its IPSP is 18/52 of the declared one.
    assert troughs['v2'] == pytest.approx(troughs['v1'] * 18 / 52, rel=0.05)


def test_v3_unitary_epsp_is_unchanged_from_v2():
    peaks = {}
    for dynamics in ('v2', 'v3'):
        brain = Brain(arrays=graph([0.275]), dynamics=dynamics)
        brain.step(np.array([20.0, 0.0], np.float32), 10.0)
        peak = engine.V_REST_MV
        for _ in range(400):
            brain.step(np.zeros(2, np.float32), 0.1)
            peak = max(peak, float(brain.v[1]))
        peaks[dynamics] = peak - engine.V_REST_MV
    assert peaks['v3'] == pytest.approx(peaks['v2'], rel=1e-6)


@pytest.mark.parametrize('weight', [-40.0, -400.0, -4000.0])
def test_v3_membrane_stays_within_the_reversal_potentials(weight):
    brain = run(Brain(arrays=graph([weight]), dynamics='v3'), 20.0, 2000.0)
    assert brain.v.min() >= engine.E_INH_MV - 1e-4
    assert brain.v.max() <= engine.E_EXC_MV + 1e-4


def test_v3_inhibitory_quantum_follows_the_declared_e_inh():
    """The calibration is a statement about driving forces, not a number."""
    brain = Brain(arrays=graph([1.0]), dynamics='v3', e_inh_mV=-56.0)
    assert brain.g_unit_inh == pytest.approx(1.0 / (engine.V_REST_MV + 56.0))
    assert brain.g_unit_exc == pytest.approx(engine.G_UNIT_EXC_V3)
    default = Brain(arrays=graph([1.0]), dynamics='v3')
    assert default.g_unit_inh == pytest.approx(engine.G_UNIT_INH_V3)


def test_v3_is_deterministic():
    a = run(Brain(arrays=graph([-40.0]), dynamics='v3'), 20.0, 300.0)
    b = run(Brain(arrays=graph([-40.0]), dynamics='v3'), 20.0, 300.0)
    np.testing.assert_array_equal(a.v, b.v)
    np.testing.assert_array_equal(a.g, b.g)


def test_v2_and_v3_differ_on_the_same_graph():
    two = run(Brain(arrays=graph([-40.0]), dynamics='v2'), 20.0, 300.0)
    three = run(Brain(arrays=graph([-40.0]), dynamics='v3'), 20.0, 300.0)
    assert not np.array_equal(two.v, three.v)


def test_v2_and_v3_checkpoints_refuse_each_other_despite_equal_shapes():
    two = run(Brain(arrays=graph([-40.0]), dynamics='v2'), 20.0, 100.0)
    three = run(Brain(arrays=graph([-40.0]), dynamics='v3'), 20.0, 100.0)
    assert two.g.shape == three.g.shape        # the shape check cannot separate them
    with pytest.raises(ValueError, match='LIF dynamics'):
        three.restore_state(two.snapshot_state())
    with pytest.raises(ValueError, match='LIF dynamics'):
        two.restore_state(three.snapshot_state())
    again = Brain(arrays=graph([-40.0]), dynamics='v3')
    again.restore_state(three.snapshot_state())
    np.testing.assert_array_equal(again.v, three.v)


def test_a_snapshot_without_a_dynamics_tag_still_loads_by_shape():
    """Checkpoints written before the tag existed keep their meaning."""
    three = run(Brain(arrays=graph([-40.0]), dynamics='v3'), 20.0, 100.0)
    state = three.snapshot_state()
    state.pop('dynamics')
    Brain(arrays=graph([-40.0]), dynamics='v3').restore_state(state)
    with pytest.raises(ValueError, match='LIF dynamics'):
        Brain(arrays=graph([-40.0]), dynamics='v1').restore_state(state)


def test_controller_version_is_repinned_for_v3():
    assert controller_version_for(BACKENDS['connectome-fixed'],
                                  DYNAMICS_VERSIONS['v3']) == 'brainlab-lif-v3'
    assert controller_version_for(BACKENDS['connectome-plastic'],
                                  DYNAMICS_VERSIONS['v3']) == 'brainlab-lif-plastic-v3'


# -- the declared transmitter policy ----------------------------------------
def _policy_graph():
    """Four neurons, one out-edge each, so ptr rows are trivial to check."""
    ptr = np.array([0, 1, 2, 3, 4], np.int64)
    post = np.array([1, 2, 3, 0], np.int32)
    weight = np.array([1.0, -2.0, 3.0, 4.0], np.float32)
    return ptr, post, weight


def test_policy_zeroes_only_the_aminergic_rows():
    from brainlab import transmitter_policy as tp
    ptr, post, weight = _policy_graph()
    labels = ['acetylcholine', 'gaba', 'dopamine', 'octopamine']
    new, report = tp.apply_policy(ptr, post, weight, labels)
    np.testing.assert_array_equal(new, np.array([1.0, -2.0, 0.0, 0.0], np.float32))
    assert report['neurons_modulatory'] == 2
    assert report['edges_zeroed'] == 2
    assert report['unclear_mode'] == 'excitatory'


def test_policy_leaves_unclear_neurons_alone_by_default():
    from brainlab import transmitter_policy as tp
    ptr, post, weight = _policy_graph()
    labels = ['unclear', 'gaba', 'serotonin', 'acetylcholine']
    new, _ = tp.apply_policy(ptr, post, weight, labels)
    assert new[0] == pytest.approx(1.0)          # the unclear neuron keeps its weight
    assert new[2] == 0.0                         # the serotonergic one does not


def test_policy_unclear_modes_are_declared_and_distinct():
    from brainlab import transmitter_policy as tp
    ptr, post, weight = _policy_graph()
    labels = ['unclear', 'gaba', 'acetylcholine', 'acetylcholine']
    keep, _ = tp.apply_policy(ptr, post, weight, labels, unclear_mode='excitatory')
    zero, _ = tp.apply_policy(ptr, post, weight, labels, unclear_mode='zero')
    excl, _ = tp.apply_policy(ptr, post, weight, labels, unclear_mode='exclude')
    assert keep[0] != 0.0
    assert zero[0] == 0.0 and zero[3] != 0.0     # only the unclear row is dropped
    assert excl[0] == 0.0 and excl[3] == 0.0     # ...and every edge ONTO it too
    with pytest.raises(ValueError):
        tp.apply_policy(ptr, post, weight, labels, unclear_mode='guess')
    with pytest.raises(ValueError):
        tp.apply_policy(ptr, post, weight, labels, policy='whatever')


def test_legacy_policy_is_a_no_op_so_v1_and_v2_are_untouched():
    from brainlab import transmitter_policy as tp
    ptr, post, weight = _policy_graph()
    labels = ['dopamine', 'unclear', 'gaba', 'acetylcholine']
    new, report = tp.apply_policy(ptr, post, weight, labels, policy=tp.POLICY_LEGACY)
    np.testing.assert_array_equal(new, weight)
    assert report['changed'] is False


def test_policy_reports_the_fixed_point_both_ways():
    """The static prediction of spec §6.5, on a toy graph."""
    from brainlab import transmitter_policy as tp
    ptr = np.array([0, 1, 2], np.int64)
    post = np.array([1, 0], np.int32)
    weight = np.array([1.0, -2.0], np.float32)
    _, report = tp.apply_policy(ptr, post, weight, ['acetylcholine', 'gaba'])
    assert report['v2_quanta']['conductance_ratio_g_inh_over_g_exc'] == pytest.approx(2.0)
    assert report['v3_quanta']['conductance_ratio_g_inh_over_g_exc'] == pytest.approx(2.0 * 52 / 18)
    assert report['v3_quanta']['subthreshold'] is True
