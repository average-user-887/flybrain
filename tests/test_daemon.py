"""Unit tests for NeuroFly 24/7 Continuous Background Learning Daemon."""

import json
import time
import unittest
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from neurofly_daemon import ContinuousExperimentRunner, NeuroflyHTTPHandler


class TestNeuroflyDaemonEngine(unittest.TestCase):
    """Verifies continuous simulation runner, plasticity, and command dispatch."""

    def setUp(self):
        self.tmp_output = PACKAGE_ROOT / "outputs" / "test_daemon"
        self.runner = ContinuousExperimentRunner(
            initial_paradigm="multisensory-sandbox",
            sim_speed=50.0,
            checkpoint_interval=10.0,
            output_dir=self.tmp_output
        )

    def tearDown(self):
        if self.runner.running:
            self.runner.stop()

    def test_runner_initialization(self):
        """Runner must correctly initialize arena, trial counters, and telemetry."""
        self.assertEqual(self.runner.active_paradigm_id, "multisensory-sandbox")
        self.assertEqual(self.runner.current_trial, 1)
        self.assertEqual(self.runner.total_steps, 0)
        self.assertFalse(self.runner.running)

    def test_step_and_assemble_telemetry(self):
        """Stepping arena and assembling telemetry packet must produce compliant schema."""
        step_res = self.runner.arena.step(0.02)
        self.runner.total_steps += 1
        telem = self.runner._assemble_telemetry(step_res)

        self.assertEqual(telem["type"], "telemetry")
        self.assertIn("fly", telem)
        self.assertIn("x", telem["fly"])
        self.assertIn("y", telem["fly"])
        self.assertIn("sensory", telem)
        self.assertIn("descending", telem)
        self.assertIn("biomechanics", telem)
        self.assertIn("joint_angles", telem["biomechanics"])
        self.assertIn("plasticity", telem)
        self.assertIn("mb_weights_mean", telem["plasticity"])

    def test_command_dispatch_set_speed(self):
        """Dispatching set_speed command updates sim_speed within valid bounds."""
        res = self.runner.dispatch_command({"action": "set_speed", "speed": 25.0})
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self.runner.sim_speed, 25.0)

        # Bounds check
        self.runner.dispatch_command({"action": "set_speed", "speed": 999.0})
        self.assertEqual(self.runner.sim_speed, 100.0)

    def test_command_dispatch_switch_paradigm(self):
        """Dispatching switch_paradigm re-initializes arena cleanly."""
        res = self.runner.dispatch_command({"action": "switch_paradigm", "paradigm": "t-maze"})
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self.runner.active_paradigm_id, "t-maze")

    def test_command_dispatch_inject_stimulus(self):
        """Optogenetic and sensory flares inject proper force/velocity overrides."""
        res = self.runner.dispatch_command({
            "action": "inject_stimulus",
            "type": "optogenetic_dna02",
            "value": 0.5
        })
        self.assertEqual(res["status"], "ok")

        res_gf = self.runner.dispatch_command({
            "action": "inject_stimulus",
            "type": "gf_looming"
        })
        self.assertEqual(res_gf["status"], "ok")
        self.assertEqual(self.runner.arena.fly.behavioral_state, "ESCAPE")

    def test_checkpoint_generation_and_rotation(self):
        """Periodic checkpointing creates JSON ledger without exceeding max files."""
        ckpt_path = self.runner.save_checkpoint("test_unit")
        self.assertTrue(ckpt_path.exists())
        with open(ckpt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["tag"], "test_unit")
        self.assertIn("weights_mean", data)
        self.assertIn("uptime_sec", data)
        ckpt_path.unlink()


if __name__ == "__main__":
    unittest.main()
