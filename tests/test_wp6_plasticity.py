"""Unit and integration tests for WP6 Visual-Heading Plasticity.

Specification: docs/WP6_PLASTICITY_SPEC.md
Machine-readable companion: docs/wp6_plastic_subset.json
"""
import os
from pathlib import Path
import numpy as np
import pytest

from brainlab.graph_identity import DEFAULT_CONNECTOME_DIR, DEFAULT_GRAPH_DIR, GraphUnavailable
from brainlab.io_map import VISUAL_HEADING_IO_PIN, resolve_visual_heading_io
from brainlab.wp6_plasticity import VisualHeadingPlasticityRule
from experiment_registry import ExperimentRegistry, SharedGraph


def test_visual_heading_io_resolution():
    """Verify exact counts and pinned SHA-256 for the visual-heading circuit."""
    if not (DEFAULT_GRAPH_DIR / 'graph.npz').is_file():
        pytest.skip('MaleCNS graph.npz not available in standard path')

    io = resolve_visual_heading_io()
    assert io.sha256 == VISUAL_HEADING_IO_PIN
    assert len(io.er_nodes) == 67
    assert len(io.epg_nodes) == 50
    assert len(io.el_nodes) == 18
    assert len(io.pfl3_nodes) == 24
    assert len(io.plastic_edges) == 3081
    assert len(io.dna02_nodes['DNa02_L']) == 1
    assert len(io.dna02_nodes['DNa02_R']) == 1


def test_plastic_subset_anatomy_and_sign():
    """Verify that all 3,081 edges are GABAergic negative weights and match metadata."""
    if not (DEFAULT_GRAPH_DIR / 'graph.npz').is_file():
        pytest.skip('MaleCNS graph.npz not available in standard path')

    rule = VisualHeadingPlasticityRule.from_connectome()
    desc = rule.describe()
    assert desc['name'] == 'visual-heading-plasticity-v1'
    assert desc['id'] == 'ER_to_EPG_visual_subset'
    assert desc['n_edges'] == 3081
    assert desc['pre_neurons'] == 67
    assert desc['post_neurons'] == 50
    assert desc['modulator_neurons'] == 18

    # All initial weights must be negative (GABAergic ring neurons)
    assert (rule.initial_weights < 0).all()
    # Synaptic contacts = sum(|w| / 0.275) == 40919
    contacts = float(np.sum(np.abs(rule.initial_weights) / 0.275))
    assert abs(contacts - 40919.0) < 1.0


def test_depression_dynamics_and_sign_preservation():
    """Verify two-factor depression, bounds, and that sign never flips."""
    # Synthetic small rule fixture
    edges = np.array([10, 20, 30], dtype=np.int64)
    w0 = np.array([-2.75, -5.50, -0.275], dtype=np.float32)
    rule = VisualHeadingPlasticityRule(edges, w0, dt=0.002, eta=0.05, tau_rec=60.0)

    delta = np.zeros(len(edges), dtype=np.float32)
    pre_counts = np.array([5, 5, 5], dtype=np.int32)
    post_counts = np.zeros(len(edges), dtype=np.int32)

    # 1. Active depression step
    rule.update(delta, pre_counts, post_counts, modulator_rate=30.0)
    assert (delta > 0).all(), "Depression must increase delta (reducing magnitude toward 0)"
    w_eff = w0 + delta
    assert (w_eff <= 0.0).all(), "Sign flip is forbidden: weights must remain negative"
    assert (np.abs(w_eff) < np.abs(w0)).all(), "Magnitude must decrease under depression"

    # 2. Extreme depression cannot invert sign (clipping at 0 mV)
    for _ in range(500):
        rule.update(delta, pre_counts, post_counts, modulator_rate=50.0)
    w_extreme = w0 + delta
    assert (w_extreme <= 0.0).all(), "Even extreme depression must never exceed 0 mV"
    assert (delta <= np.abs(w0)).all(), "Delta must not exceed initial conductance magnitude"

    # 3. Recovery toward initial anatomy when modulation is zero
    rule.reset()
    delta_before = delta.copy()
    rule.update(delta, pre_counts=np.zeros(len(edges), dtype=np.int32),
                post_counts=np.zeros(len(edges), dtype=np.int32), modulator_rate=0.0)
    assert (delta < delta_before).all(), "With no drive/modulator, delta must decay toward 0"


def test_experiment_registry_plastic_backend_integration(tmp_path):
    """Verify that connectome-plastic works in ExperimentRegistry with VisualHeadingPlasticityRule."""
    if not (DEFAULT_GRAPH_DIR / 'graph.npz').is_file():
        pytest.skip('MaleCNS graph.npz not available in standard path')

    shared = SharedGraph.load()
    registry = ExperimentRegistry(shared, tmp_path / 'reg')

    # Activation should automatically attach VisualHeadingPlasticityRule
    instance = registry.activate('buridan', 'connectome-plastic')
    assert instance.rule is not None
    assert instance.rule.name == 'visual-heading-plasticity-v1'
    assert len(instance.plastic_edges) == 3081
    assert len(instance.plastic_delta) == 3081

    # Checkpoint and manifest
    ckpt_path = registry.checkpoint()
    assert ckpt_path.is_file()
    meta, arrays = registry.read_checkpoint(instance.instance_id)
    assert 'plastic.edges' in arrays
    assert 'plastic.delta' in arrays
    assert meta['rule']['name'] == 'visual-heading-plasticity-v1'
    assert meta['rule']['n_edges'] == 3081

    # Release and restore
    instance.release()
    restored = registry.activate('buridan', 'connectome-plastic')
    np.testing.assert_array_equal(restored.plastic_edges, instance.plastic_edges)
    np.testing.assert_array_equal(restored.plastic_delta, instance.plastic_delta)


def test_rule_substeps_split_a_control_step_into_rule_steps():
    """The rule is declared per 2 ms; a 20 ms control step is 10 rule steps."""
    rule = VisualHeadingPlasticityRule(np.array([0]), np.array([-1.0], np.float32), dt=0.002)
    assert rule.substeps(2.0) == 1
    assert rule.substeps(20.0) == 10
    with pytest.raises(ValueError):
        rule.substeps(3.0)


def test_registry_updates_the_rule_once_per_rule_dt(tmp_path):
    """A 20 ms registry step runs ten 2 ms brain steps, each followed by one update."""
    from experiment_registry import TestOnlyCoactivityRule

    class CountingRule(TestOnlyCoactivityRule):
        __test__ = False
        dt = 0.002

        def __init__(self, edges):
            super().__init__(edges=edges, eta=0.05)
            self.calls = 0

        substeps = VisualHeadingPlasticityRule.substeps

        def update(self, delta, pre_counts, post_counts):
            self.calls += 1
            super().update(delta, pre_counts, post_counts)

    shared = SharedGraph.synthetic(allow_synthetic=True, n=64, k_out=6, seed=5)
    rule = CountingRule(np.arange(0, 384, 7))
    registry = ExperimentRegistry(shared, tmp_path / 'reg', test_mode=True, plasticity_rule=rule)
    instance = registry.activate('buridan', 'connectome-plastic')
    currents = np.full(shared.n, 20.0, np.float32)
    counts = instance.step(currents, 20.0).counts
    assert rule.calls == 10
    assert counts.shape == (shared.n,)
    registry.learning_enabled = False
    instance.step(currents, 20.0)
    assert rule.calls == 10
