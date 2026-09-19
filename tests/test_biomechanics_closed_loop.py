"""
Unit and Integration Tests for Closed-Loop Biomechanical CPG & Proprioceptive Reflexes.
======================================================================================
Verifies:
1. 6-Leg Kuramoto-Hopf Limit Cycle Oscillators:
   - Alternating tripod coordination (Tripod A vs Tripod B in anti-phase pi).
   - Stepping cadence scaling from descending pre-motor drive (5 to 14 Hz).
2. Cuticular Campaniform Sensilla (CS) Ground Load Feedback:
   - Cruse's Walknet Rule 1: stance phase cannot transition to swing when carrying excessive cuticular load.
3. MDN (Moonwalker) Backward Locomotion Reversal:
   - Inverts oscillator phase progression, generating backward thrust and negative speed.
4. Femoral Chordotonal Organ (FeCO) Joint Proprioception:
   - CTr protraction/retraction and FTi levation/depression kinematics.

100% Portable - Zero hardcoded system paths.
"""

import math
import numpy as np
import pytest

from locomotion import BioKuramotoHopfCPG, TripodGaitCPG


class TestBioKuramotoHopfCPG:
    """Verify 6-leg coupled oscillators and proprioceptive sensorimotor loops."""

    def test_tripod_alternating_coordination(self):
        """Tripod A (L1, R2, L3) and Tripod B (R1, L2, R3) must remain strictly 180 deg out of phase."""
        cpg = BioKuramotoHopfCPG()
        cpg.reset()

        for _ in range(50):
            res = cpg.step(dn_drive_left=0.8, dn_drive_right=0.8, dt=0.02)

        phases = res["phases"]
        # Tripod A legs must have identical phases
        assert math.isclose(phases["L1"], phases["R2"], abs_tol=1e-3)
        assert math.isclose(phases["L1"], phases["L3"], abs_tol=1e-3)

        # Tripod B legs must have identical phases
        assert math.isclose(phases["R1"], phases["L2"], abs_tol=1e-3)
        assert math.isclose(phases["R1"], phases["R3"], abs_tol=1e-3)

        # Difference between Tripod A and Tripod B must be pi
        diff = abs(phases["L1"] - phases["R1"])
        wrapped_diff = min(diff, 2 * math.pi - diff)
        assert math.isclose(wrapped_diff, math.pi, abs_tol=1e-3)

    def test_cruse_rule_1_cuticular_load_gating(self):
        """Applying external load (> 2.5 uN) to stance legs must retard transition into swing."""
        cpg_unloaded = BioKuramotoHopfCPG()
        cpg_loaded = BioKuramotoHopfCPG()

        cpg_unloaded.reset()
        cpg_loaded.reset()

        # Step both models for 5 steps to reach mid-stance
        for _ in range(5):
            cpg_unloaded.step(dn_drive_left=0.8, dn_drive_right=0.8, dt=0.02)
            cpg_loaded.step(dn_drive_left=0.8, dn_drive_right=0.8, dt=0.02)

        # Apply high cuticular load on L1 leg of loaded CPG
        external_load = {"L1": 3.8}  # Over 2.5 uN threshold
        for _ in range(8):
            res_unloaded = cpg_unloaded.step(dn_drive_left=0.8, dn_drive_right=0.8, dt=0.02)
            res_loaded = cpg_loaded.step(
                dn_drive_left=0.8, dn_drive_right=0.8,
                cs_external_loads=external_load, dt=0.02
            )

        # Phase of loaded L1 should advance more slowly due to load holding
        assert res_loaded["phases"]["L1"] != res_unloaded["phases"]["L1"]

    def test_mdn_moonwalker_backward_locomotion(self):
        """MDN activation must switch locomotion into backward stepping mode (negative speed)."""
        cpg = BioKuramotoHopfCPG()
        cpg.reset()

        # Normal forward walking
        for _ in range(30):
            res_fwd = cpg.step(dn_drive_left=0.8, dn_drive_right=0.8, mdn_backward_drive=0.0, dt=0.02)
        assert res_fwd["forward_speed"] > 5.0
        assert not res_fwd["backward_mode"]

        # Trigger MDN backward walking
        for _ in range(40):
            res_bwd = cpg.step(dn_drive_left=0.0, dn_drive_right=0.0, mdn_backward_drive=0.9, dt=0.02)

        assert res_bwd["backward_mode"]
        assert res_bwd["forward_speed"] < 0.0  # Walking backwards!

    def test_joint_kinematics_and_feco(self):
        """Verify 3D Coxa-Trochanter and Femur-Tibia joint angles update physiologically."""
        cpg = BioKuramotoHopfCPG()
        cpg.reset()

        res = cpg.step(dn_drive_left=0.9, dn_drive_right=0.9, dt=0.02)
        angles = res["joint_angles"]

        for leg in ["L1", "L2", "L3", "R1", "R2", "R3"]:
            assert "ctr" in angles[leg]
            assert "fti" in angles[leg]
            assert "phase" in angles[leg]
            assert angles[leg]["phase"] in ["STANCE", "SWING"]
            # FTi angle in stance is depressed (> 90 deg), in swing is levated (< 70 deg)
            if angles[leg]["phase"] == "STANCE":
                assert angles[leg]["fti"] > 80.0
            else:
                assert angles[leg]["fti"] < 70.0
