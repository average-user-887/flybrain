"""Tests for Step 3.1: Embodied body state telemetry in NeuroFly daemon."""

import math
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurofly_daemon import ContinuousExperimentRunner, NeuroflyHTTPHandler


class TestEmbodiedTelemetry(unittest.TestCase):
    """Verifies that the daemon emits complete body telemetry in compliant format."""

    def setUp(self):
        self.tmp_output = PROJECT_ROOT / "outputs" / "test_embodied_telemetry"
        self.runner = ContinuousExperimentRunner(
            initial_paradigm="multisensory-sandbox",
            sim_speed=50.0,
            checkpoint_interval=10.0,
            output_dir=self.tmp_output,
        )

    def tearDown(self):
        if self.runner.running:
            self.runner.stop()

    def test_assembled_telemetry_contains_all_body_fields(self):
        step_res = self.runner.arena.step(0.02)
        self.runner.total_steps += 1
        telem = self.runner._assemble_telemetry(step_res)

        # 1. 18-DOF joint angles array
        self.assertIn("joint_angles_rad", telem)
        self.assertIsInstance(telem["joint_angles_rad"], list)
        self.assertEqual(len(telem["joint_angles_rad"]), 18)
        self.assertTrue(all(isinstance(v, (float, int)) and math.isfinite(v) for v in telem["joint_angles_rad"]))

        # 2. 6-element leg contacts boolean array
        self.assertIn("leg_contacts", telem)
        self.assertIsInstance(telem["leg_contacts"], list)
        self.assertEqual(len(telem["leg_contacts"]), 6)
        self.assertTrue(all(isinstance(v, bool) for v in telem["leg_contacts"]))

        # 3. 3D Body position in mm
        self.assertIn("body_position_mm", telem)
        self.assertIsInstance(telem["body_position_mm"], list)
        self.assertEqual(len(telem["body_position_mm"]), 3)
        self.assertTrue(all(isinstance(v, (float, int)) and math.isfinite(v) for v in telem["body_position_mm"]))

        # 4. Body quaternion wxyz
        self.assertIn("body_quaternion_wxyz", telem)
        self.assertIsInstance(telem["body_quaternion_wxyz"], list)
        self.assertEqual(len(telem["body_quaternion_wxyz"]), 4)
        quat = telem["body_quaternion_wxyz"]
        self.assertTrue(all(isinstance(v, (float, int)) and math.isfinite(v) for v in quat))
        norm_sq = sum(v * v for v in quat)
        self.assertAlmostEqual(norm_sq, 1.0, places=3)

        # 5. Descending rates
        self.assertIn("dn_rates", telem)
        dn = telem["dn_rates"]
        for key in ("dna02_l", "dna02_r", "dnp09", "mdn", "gf"):
            self.assertIn(key, dn)
            self.assertGreaterEqual(dn[key], 0.0)

        # 6. Controller identity
        self.assertIn("controller_id", telem)
        self.assertEqual(telem["controller_id"], "modular")

    def test_status_payload_includes_body_fields(self):
        step_res = self.runner.arena.step(0.02)
        self.runner.total_steps += 1
        self.runner.latest_telemetry = self.runner._assemble_telemetry(step_res)

        # Simulate handler status payload generation
        class DummyHandler:
            def __init__(self, runner):
                self.runner = runner
                self.gateway = type("Gateway", (), {"describe": lambda self: {}})()

        handler = DummyHandler(self.runner)
        payload = NeuroflyHTTPHandler._status_payload(handler)

        self.assertIn("joint_angles_rad", payload)
        self.assertEqual(len(payload["joint_angles_rad"]), 18)
        self.assertIn("leg_contacts", payload)
        self.assertEqual(len(payload["leg_contacts"]), 6)
        self.assertIn("body_position_mm", payload)
        self.assertEqual(len(payload["body_position_mm"]), 3)
        self.assertIn("body_quaternion_wxyz", payload)
        self.assertEqual(len(payload["body_quaternion_wxyz"]), 4)
        self.assertIn("dn_rates", payload)
        self.assertIn("controller_id", payload)
        self.assertIn("body", payload)
        self.assertEqual(payload["body"]["controller_id"], "modular")

    def test_paradigm_switching_preserves_weights_and_updates_body_state(self):
        import numpy as np

        # 1. Switch to t-maze
        res1 = self.runner.dispatch_command({"action": "switch_paradigm", "paradigm": "t-maze"})
        self.assertEqual(res1["status"], "ok")
        self.assertEqual(self.runner.active_paradigm_id, "t-maze")
        w_tmaze_initial = self.runner.arena.fly.circuit.get_effective_weights().copy()

        # Step t-maze
        self.runner.arena.step(0.02)
        telem1 = self.runner.latest_telemetry
        self.assertIn("body_position_mm", telem1)
        self.assertEqual(len(telem1["joint_angles_rad"]), 18)

        # 2. Switch to buridan
        res2 = self.runner.dispatch_command({"action": "switch_paradigm", "paradigm": "buridan"})
        self.assertEqual(res2["status"], "ok")
        self.assertEqual(self.runner.active_paradigm_id, "buridan")
        telem2 = self.runner.latest_telemetry
        self.assertIn("body_position_mm", telem2)

        # 3. Switch to optomotor
        res3 = self.runner.dispatch_command({"action": "switch_paradigm", "paradigm": "optomotor"})
        self.assertEqual(res3["status"], "ok")
        self.assertEqual(self.runner.active_paradigm_id, "optomotor")
        telem3 = self.runner.latest_telemetry
        self.assertIn("body_position_mm", telem3)

        # 4. Switch back to t-maze and verify weights were preserved
        res4 = self.runner.dispatch_command({"action": "switch_paradigm", "paradigm": "t-maze"})
        self.assertEqual(res4["status"], "ok")
        w_tmaze_restored = self.runner.arena.fly.circuit.get_effective_weights()
        np.testing.assert_allclose(w_tmaze_restored, w_tmaze_initial, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
