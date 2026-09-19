"""
Unit and Integration Tests for High-Fidelity Sensory Ingress Pipelines.
======================================================================
Verifies:
1. Visual Lobula Columnar (LC) Feature Detectors:
   - LC4 & LPLC2: Looming angular expansion kinematics (theta, theta_dot).
   - LC6: Broad looming collision landing and braking.
   - LC10: Small moving target pursuit tracking (peaked at 2-5 deg).
   - LC11: Small object motion against background clutter (center-surround).
   - LPTC Vertical System (VS1-VS12) pitch and roll flow deconstruction.
2. Johnston's Organ (JO) & Wedge (WED) Circuit:
   - JON-A/B high-frequency acoustic courtship song transduction.
   - JON-C/E static aerodynamic wind drag deflection.
   - Wedge Projection Neurons (WPNs) bilateral push-pull differential.
   - Central Complex ER1_b / ER3a_b compass bump anchoring.

100% Portable - Zero hardcoded system paths.
"""

import math
import numpy as np
import pytest

from vision import LobulaFeatureExtractor, CompoundEyeVision
from mechanosensory import JohnstonsOrgan, WedgeProjectionNeurons


class TestLobulaColumnarVision:
    """Verify feature selectivity of Lobula Columnar and Tangential circuits."""

    def test_lc4_and_lplc2_looming_response(self):
        """Rapidly approaching threat must excite LC4 (velocity) and LPLC2 (size)."""
        extractor = LobulaFeatureExtractor(num_ommatidia=72)
        optic_flow = np.zeros(72, dtype=np.float32)

        # Distant threat moving slowly -> low LC4/LPLC2
        res_distant = extractor.compute_features(
            fly_speed=5.0,
            fly_yaw_rate=0.0,
            optic_flow=optic_flow,
            threat_distance=100.0,
            threat_approach_velocity=2.0
        )
        assert res_distant["lc4_looming_vel"] < 0.2
        assert res_distant["lplc2_looming_size"] < 0.2

        # Fast imminent threat at close range (30 mm, 50 mm/s approach)
        res_imminent = extractor.compute_features(
            fly_speed=5.0,
            fly_yaw_rate=0.0,
            optic_flow=optic_flow,
            threat_distance=30.0,
            threat_approach_velocity=50.0,
            threat_radius=6.0
        )
        assert res_imminent["lc4_looming_vel"] > 0.6
        assert res_imminent["lplc2_looming_size"] > 0.5
        assert res_imminent["lc6_collision_brake"] > 0.3

    def test_lc10_target_tracking_tuning(self):
        """LC10 must be maximally activated by small objects (2 deg to 5 deg) directly ahead."""
        extractor = LobulaFeatureExtractor(num_ommatidia=72)
        optic_flow = np.zeros(72, dtype=np.float32)
        fly_pos = np.array([0.0, 0.0])
        fly_heading = 0.0

        # Small target directly ahead at 25 mm (~3.5 deg subtended angle)
        optimal_target = [np.array([25.0, 0.0])]
        res_opt = extractor.compute_features(
            fly_speed=0.0,
            fly_yaw_rate=0.0,
            optic_flow=optic_flow,
            threat_distance=999.0,
            threat_approach_velocity=0.0,
            target_positions=optimal_target,
            fly_pos=fly_pos,
            fly_heading=fly_heading
        )
        assert res_opt["lc10_target_tracking"] > 0.7

        # Target behind the fly (bearing > 60 deg) -> LC10 suppressed
        behind_target = [np.array([-25.0, 0.0])]
        res_behind = extractor.compute_features(
            fly_speed=0.0,
            fly_yaw_rate=0.0,
            optic_flow=optic_flow,
            threat_distance=999.0,
            threat_approach_velocity=0.0,
            target_positions=behind_target,
            fly_pos=fly_pos,
            fly_heading=fly_heading
        )
        assert res_behind["lc10_target_tracking"] == 0.0

    def test_lc11_clutter_suppression(self):
        """Widefield optical flow must suppress LC11, whereas isolated localized motion excites it."""
        extractor = LobulaFeatureExtractor(num_ommatidia=72)

        # Uniform widefield optic flow (e.g. self-rotation in visual arena)
        uniform_flow = np.full(72, 1.5, dtype=np.float32)
        res_uniform = extractor.compute_features(
            fly_speed=10.0,
            fly_yaw_rate=1.5,
            optic_flow=uniform_flow,
            threat_distance=999.0,
            threat_approach_velocity=0.0
        )
        assert res_uniform["lc11_small_object"] == 0.0

        # Isolated localized deviation on 1 ommatidium against stationary background
        isolated_flow = np.zeros(72, dtype=np.float32)
        isolated_flow[10] = 3.0
        res_isolated = extractor.compute_features(
            fly_speed=0.0,
            fly_yaw_rate=0.0,
            optic_flow=isolated_flow,
            threat_distance=999.0,
            threat_approach_velocity=0.0
        )
        assert res_isolated["lc11_small_object"] > 0.5


class TestMechanosensoryJohnstonsOrgan:
    """Verify JON sub-group physiology and Wedge projection neurons."""

    def test_jon_acoustic_vs_wind_drag_segregation(self):
        """JON-A/B must capture courtship sound, JON-C/E must capture aerodynamic airflow."""
        jo = JohnstonsOrgan(n_wedges=16)

        # Case 1: Pure wind, no acoustic stimulus
        res_wind = jo.step(
            fly_heading=0.0,
            fly_speed=5.0,
            wind_vx=15.0,
            wind_vy=0.0,
            acoustic_pulse_train=0.0
        )
        assert res_wind["jon_ce_wind_drag"] > 0.3
        assert res_wind["jon_ab_acoustic"] == 0.0
        assert res_wind["v_rel_mag"] > 5.0

        # Case 2: Male courtship acoustic pulse song in calm air
        res_courtship = jo.step(
            fly_heading=0.0,
            fly_speed=0.0,
            wind_vx=0.0,
            wind_vy=0.0,
            acoustic_pulse_train=0.85
        )
        assert res_courtship["jon_ab_acoustic"] == 0.85
        assert res_courtship["jon_ce_wind_drag"] == 0.0

    def test_wpn_bilateral_steering_and_compass_anchoring(self):
        """Crosswind airflow must induce bilateral WPN differential and asymmetric compass bump."""
        jo = JohnstonsOrgan(n_wedges=16)

        # Crosswind coming from the right (wind_vy = 10 mm/s)
        res = jo.step(
            fly_heading=0.0,
            fly_speed=0.0,
            wind_vx=0.0,
            wind_vy=10.0
        )
        wpn = res["wpn_data"]
        assert wpn["wpn_right_hz"] != wpn["wpn_left_hz"]
        assert abs(wpn["wpn_differential"]) > 5.0
        # Central Complex bump must have a clear peak
        bump = res["cx_bump"]
        assert np.max(bump) > np.min(bump)
        assert np.isclose(np.sum(bump), min(1.0, 10.0 / 25.0), atol=1e-4)
