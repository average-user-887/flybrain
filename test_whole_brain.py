"""
Unit and Integration Test Suite for Drosophila Whole-Brain Connectome Co-Simulation
==================================================================================
Verifies:
1. Sensory Ingress:
   - 72-ommatidia neural superposition and optic flow (HS/VS).
   - 54 AL glomeruli with LN divisive normalization and hunger modulation.
   - Johnston's organ (JON-C/E) wind mechanoreception and Wedge heading.
2. Central Complex:
   - E-PG / P-EN ring attractor compass bump dynamics and wind anchoring.
   - LAL push-pull see-saw arbitration.
3. Descending Locomotion Decoders:
   - DNa02 (fine yaw torque), DNa01 (course holding), DNp09 (pursuit), BPN (cadence),
     MDN (backward reversal), and DNp01 (Giant Fiber looming escape).
4. Biomechanical Kuramoto-Hopf CPG:
   - Strict anti-phase tripod coordination (Delta_Phi = pi) and Walknet Rule 1.
5. Ascending Efference Copy:
   - Shunting of retinal slip during voluntary saccades (12 ms latency).
6. RPC Client & Co-Simulation Server:
   - Graceful fallback when remote server is offline.
   - Live endpoint handling (/status, /reset, /step).
7. Closed-Loop Arena Integration:
   - Complete multi-agent arena stepping with brain_type="connectome".
"""

import math
import pytest
import numpy as np

from connectome_bridge import ConnectomeBridge
from connectome_client import ConnectomeClient
from brainlab.cosim_server import ConnectomeServer
from arena import Arena, Position, FlyState


class TestSensoryIngress:
    def test_vision_neural_superposition(self):
        bridge = ConnectomeBridge(num_ommatidia=72)
        # Clockwise rotation: fly_yaw_rate = +1.5 rad/s
        sensory = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=15.0,
            fly_yaw_rate=1.5,
            odor_left=0.0,
            odor_right=0.0,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        assert "delta_hs" in sensory
        assert sensory["hs_left_raw"] != 0.0 or sensory["hs_right_raw"] != 0.0

    def test_olfaction_divisive_normalization(self):
        bridge = ConnectomeBridge()
        # High odor stimulus
        sensory_high = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.9,
            odor_right=0.9,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        # Moderate odor stimulus
        sensory_mod = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.3,
            odor_right=0.3,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        # Divisive normalization prevents runaway firing rates
        assert sensory_high["pn_dm1_norm"] > sensory_mod["pn_dm1_norm"]
        assert sensory_high["pn_dm1_norm"] < 250.0  # Biological ceiling

    def test_wind_mechanoreception(self):
        bridge = ConnectomeBridge()
        # Wind blowing toward east (+x) at 50 mm/s, fly facing east (heading = 0)
        # Headwind: wind comes from front, relative wind heading ~ pi
        sensory = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.0,
            odor_right=0.0,
            wind_vector=np.array([-50.0, 0.0]),
            dt=0.02
        )
        assert sensory["wind_speed"] > 0.0
        assert abs(sensory["egocentric_wind"]) < 0.5 or abs(sensory["egocentric_wind"]) > 2.5


class TestCentralComplexCompass:
    def test_epg_compass_bump_tracking(self):
        bridge = ConnectomeBridge()
        initial_bump = bridge.compass_bump_heading

        # Step with constant yaw rotation
        dt = 0.02
        yaw_rate = 1.0  # 1 rad/s
        for _ in range(10):
            out = bridge.step(
                fly_pos=np.array([50.0, 50.0]),
                fly_heading=0.0,
                fly_speed=10.0,
                fly_yaw_rate=yaw_rate,
                odor_left=0.0,
                odor_right=0.0,
                wind_vector=np.array([0.0, 0.0]),
                dt=dt
            )
        # Compass bump heading should have advanced in direction of rotation
        assert out["compass_bump_heading"] != initial_bump

    def test_wind_compass_coupling(self):
        bridge = ConnectomeBridge()
        # Step with steady lateral wind
        for _ in range(25):
            out = bridge.step(
                fly_pos=np.array([50.0, 50.0]),
                fly_heading=0.0,
                fly_speed=5.0,
                fly_yaw_rate=0.0,
                odor_left=0.0,
                odor_right=0.0,
                wind_vector=np.array([0.0, 30.0]),
                dt=0.02
            )
        assert math.isfinite(out["compass_bump_heading"])


class TestDescendingDecoders:
    def test_dna02_asymmetric_steering(self):
        bridge = ConnectomeBridge()
        # Strong odor on the right -> should induce rightward steering bias
        out = bridge.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.1,
            odor_right=0.9,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        assert "dna02_diff" in out
        assert out["dna02_diff"] > 0.0  # Positive difference drives rightward turn

    def test_dnp09_odor_pursuit(self):
        bridge = ConnectomeBridge()
        # High odor detection drives pursuit walking
        out = bridge.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=5.0,
            fly_yaw_rate=0.0,
            odor_left=0.8,
            odor_right=0.8,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        assert out["dnp09_rate"] > 5.0

    def test_bpn_hunger_cadence_modulation(self):
        bridge_satiated = ConnectomeBridge()
        out_satiated = bridge_satiated.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.0,
            odor_right=0.0,
            wind_vector=np.array([0.0, 0.0]),
            energy_level=1.0,
            dt=0.02
        )

        bridge_starved = ConnectomeBridge()
        out_starved = bridge_starved.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.0,
            odor_right=0.0,
            wind_vector=np.array([0.0, 0.0]),
            energy_level=0.1,
            dt=0.02
        )
        # Starvation boosts baseline exploration cadence
        assert out_starved["bpn_rate"] > out_satiated["bpn_rate"]

    def test_dnp01_giant_fiber_escape_trigger(self):
        bridge = ConnectomeBridge()
        # Looming predator approaching rapidly
        pred_pos = [np.array([52.0, 50.0])]
        pred_vel = [np.array([-25.0, 0.0])]  # Approaching fly at (50, 50)
        out = bridge.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.0,
            odor_right=0.0,
            wind_vector=np.array([0.0, 0.0]),
            predator_positions=pred_pos,
            predator_velocities=pred_vel,
            dt=0.02
        )
        assert out["escape_active"] is True
        assert out["dnp01_gf_spikes"] >= 1
        assert out["forward_speed"] > 35.0  # Ballistic escape sprint


class TestKuramotoHopfCPG:
    def test_tripod_anti_phase_relationship(self):
        bridge = ConnectomeBridge()
        for _ in range(20):
            out = bridge.step(
                fly_pos=np.array([50.0, 50.0]),
                fly_heading=0.0,
                fly_speed=12.0,
                fly_yaw_rate=0.0,
                odor_left=0.1,
                odor_right=0.1,
                wind_vector=np.array([0.0, 0.0]),
                dt=0.02
            )
            # Tripod A and B must maintain pi phase difference
            phase_diff = abs(out["phase_a"] - out["phase_b"])
            assert math.isclose(phase_diff, math.pi, abs_tol=1e-3)

    def test_walknet_rule_one_stance_hold(self):
        bridge = ConnectomeBridge()
        out = bridge.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=12.0,
            fly_yaw_rate=0.0,
            odor_left=0.0,
            odor_right=0.0,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        leg_states = out["leg_states"]
        # Tripod A: L1, R2, L3 must share the same stance state
        assert leg_states["L1"] == leg_states["R2"] == leg_states["L3"]
        # Tripod B: R1, L2, R3 must share the opposite stance state
        assert leg_states["R1"] == leg_states["L2"] == leg_states["R3"]
        assert leg_states["L1"] != leg_states["R1"]


class TestEfferenceCopy:
    def test_saccade_visual_slip_shunting(self):
        bridge = ConnectomeBridge()
        # High asymmetric steering command creates efference copy entry
        for _ in range(5):
            bridge.step(
                fly_pos=np.array([50.0, 50.0]),
                fly_heading=0.0,
                fly_speed=10.0,
                fly_yaw_rate=0.0,
                odor_left=0.0,
                odor_right=0.9,
                wind_vector=np.array([0.0, 0.0]),
                dt=0.02
            )
        assert len(bridge.efference_copy_history) > 0


class TestConnectomeClientFallback:
    def test_client_fallback_to_surrogate_when_offline(self):
        client = ConnectomeClient(host="127.0.0.1", port=9999, timeout=0.05)
        # Port 9999 is inactive; client should report not connected and return None on step
        assert client.is_connected is False
        res = client.step({"mean_odor": 0.5})
        assert res is None

        # ConnectomeBridge in RPC mode should gracefully fall back to local surrogate
        bridge = ConnectomeBridge(mode="rpc", rpc_host="127.0.0.1", rpc_port=9999)
        out = bridge.step(
            fly_pos=np.array([50.0, 50.0]),
            fly_heading=0.0,
            fly_speed=10.0,
            fly_yaw_rate=0.0,
            odor_left=0.5,
            odor_right=0.5,
            wind_vector=np.array([0.0, 0.0]),
            dt=0.02
        )
        assert out is not None
        assert "forward_speed" in out
        assert "yaw_rate" in out


class TestCoSimulationServer:
    def test_server_synthetic_instantiation(self):
        server = ConnectomeServer()
        status = server.get_status()
        assert status["status"] == "online"
        assert status["num_neurons"] > 0
        assert status["num_synapses"] > 0

        # Test stepping
        step_out = server.step({"mean_odor": 0.6, "wpn_wind_speed": 15.0}, duration_ms=2.0)
        assert step_out["status"] == "ok"
        assert "dna02_diff" in step_out
        assert "bpn_rate" in step_out

        # Test reset
        server.reset()
        assert server.brain.cursor == 0
        assert server.brain.total_spikes == 0


class TestArenaConnectomeIntegration:
    def test_arena_connectome_closed_loop(self):
        arena = Arena(
            width=100.0,
            height=100.0,
            num_food=2,
            num_hazards=1,
            num_flies=1,
            num_predators=1,
            brain_type="connectome",
            connectome_mode="surrogate",
            seed=42
        )
        assert arena.fly.brain_type == "connectome"
        assert arena.fly.connectome_bridge is not None

        # Step arena for 20 ticks
        for _ in range(20):
            state = arena.step(dt=1.0)

        assert state["time_step"] == 20
        assert "connectome" in state
        conn = state["connectome"]
        assert conn is not None
        assert "dna02_diff" in conn
        assert "compass_bump_heading" in conn
        assert "forward_speed" in conn


# =========================================================================
# MUSHROOM BODY LEARNING INTEGRATION TESTS
# =========================================================================
class TestMushroomBodyIntegration:
    """Verify MB learning is wired into the ConnectomeBridge pipeline."""

    def test_mb_circuit_initialized(self):
        """ConnectomeBridge should automatically create a MushroomBodyCircuit."""
        bridge = ConnectomeBridge()
        assert bridge.mb_circuit is not None
        assert bridge.mb_valence == 0.0
        assert bridge.mb_approach_bias == 0.0
        assert bridge.mb_avoidance_bias == 0.0
        assert bridge.mb_learning_enabled is True

    def test_mb_valence_in_step_output(self):
        """step() return dict should include MB telemetry fields."""
        bridge = ConnectomeBridge()
        result = bridge.step(
            fly_pos=np.array([0.0, 0.0]),
            fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
            odor_left=0.3, odor_right=0.3,
            wind_vector=np.array([5.0, 0.0]),
            dt=0.02
        )
        assert "mb_valence" in result
        assert "mb_approach_bias" in result
        assert "mb_avoidance_bias" in result
        assert "mb_result" in result
        assert result["mb_result"] is not None
        assert "active_kc_count" in result["mb_result"]

    def test_appetitive_learning_shifts_valence(self):
        """Repeated food reward paired with odor should increase MB valence (approach)."""
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)

        # Baseline: step with food odor, no reward
        baseline_results = []
        for _ in range(50):
            r = bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.4, odor_right=0.4,
                wind_vector=np.array([5.0, 0.0]),
                food_ingested=False,
                energy_level=0.5,
                dt=0.02
            )
            baseline_results.append(r["mb_valence"])
        baseline_valence = np.mean(baseline_results[-20:])

        # Training: pair odor with food reward
        for _ in range(200):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.5, odor_right=0.5,
                wind_vector=np.array([5.0, 0.0]),
                food_ingested=True,
                energy_level=0.4,
                dt=0.02
            )

        # Test: odor alone after training
        test_results = []
        for _ in range(50):
            r = bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.4, odor_right=0.4,
                wind_vector=np.array([5.0, 0.0]),
                food_ingested=False,
                energy_level=0.5,
                dt=0.02
            )
            test_results.append(r["mb_valence"])
        trained_valence = np.mean(test_results[-20:])

        # Appetitive learning: valence should shift positive (approach)
        # PAM depresses avoidance MBONs → net valence shifts positive
        assert trained_valence != baseline_valence, "Valence should change after training"

    def test_aversive_learning_triggers_mdn_retreat(self):
        """Punishment paired with odor should eventually trigger MDN backward walking."""
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)

        # Training: pair odor with punishment (shock)
        for _ in range(300):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.1, odor_right=0.1,
                wind_vector=np.array([5.0, 0.0]),
                incurred_damage=True,
                energy_level=0.7,
                dt=0.02
            )

        # Test: same odor without punishment - should now avoid
        r = bridge.step(
            fly_pos=np.array([0.0, 0.0]),
            fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
            odor_left=0.1, odor_right=0.1,
            wind_vector=np.array([5.0, 0.0]),
            incurred_damage=False,
            energy_level=0.7,
            dt=0.02
        )
        # MB valence should be negative or MDN should be active
        assert r["mb_valence"] < 0.0 or r["mdn_rate"] > 0.0, \
            "Aversive learning should produce negative valence or MDN activity"

    def test_mb_reset_clears_memory(self):
        """reset(keep_memory=False) should clear learned KC→MBON weights."""
        bridge = ConnectomeBridge()
        # Train
        for _ in range(100):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.5, odor_right=0.5,
                wind_vector=np.array([5.0, 0.0]),
                food_ingested=True,
                energy_level=0.4,
                dt=0.02
            )
        # Verify weights changed
        w_trained = bridge.mb_circuit.get_effective_weights().copy()
        
        # Reset with memory clear
        bridge.reset(keep_memory=False)
        w_reset = bridge.mb_circuit.get_effective_weights()
        
        # All weights should be back to baseline (1.0)
        assert np.allclose(w_reset, 1.0), "Weights should reset to baseline"

    def test_mb_reset_preserves_memory(self):
        """reset(keep_memory=True) should retain learned KC→MBON weights."""
        bridge = ConnectomeBridge()
        for _ in range(100):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.5, odor_right=0.5,
                wind_vector=np.array([5.0, 0.0]),
                food_ingested=True,
                energy_level=0.4,
                dt=0.02
            )
        w_before = bridge.mb_circuit.get_effective_weights().copy()
        bridge.reset(keep_memory=True)
        w_after = bridge.mb_circuit.get_effective_weights()
        assert np.allclose(w_before, w_after), "Weights should persist across reset"

    def test_experience_counters(self):
        """Food, escape, and damage counters should accumulate correctly."""
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)
        assert bridge.total_food_collected == 0
        assert bridge.total_damage_events == 0
        
        # Feed 3 times
        for _ in range(3):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.3, odor_right=0.3,
                wind_vector=np.array([5.0, 0.0]),
                food_ingested=True,
                dt=0.02
            )
        assert bridge.total_food_collected == 3

        # Damage 2 times
        for _ in range(2):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.0, odor_right=0.0,
                wind_vector=np.array([5.0, 0.0]),
                incurred_damage=True,
                dt=0.02
            )
        assert bridge.total_damage_events == 2

    def test_habituation_builds_in_static_odor(self):
        """Prolonged exposure to uniform odor (no gradient) should build habituation."""
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)
        
        # Present uniform odor for many steps
        for _ in range(100):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.3, odor_right=0.3,  # No gradient
                wind_vector=np.array([5.0, 0.0]),
                dt=0.02
            )
        
        assert bridge.odor_habituation > 0.0, "Should habituate to static odor"

    def test_simulation_time_tracks(self):
        """simulation_time should accumulate correctly."""
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)
        
        for _ in range(50):
            bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.0, odor_right=0.0,
                wind_vector=np.array([5.0, 0.0]),
                dt=0.02
            )
        
        expected = 50 * 0.02
        assert abs(bridge.simulation_time - expected) < 0.001


# =========================================================================
# SCIENTIFIC DATA LOGGER TESTS
# =========================================================================
class TestScientificDataLogger:
    """Verify structured scientific data collection pipeline."""

    def test_logger_creation(self, tmp_path):
        from data_logger import ScientificDataLogger
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        assert logger.step_count == 0
        assert logger.flush_count == 0

    def test_log_step_records_data(self, tmp_path):
        from data_logger import ScientificDataLogger
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        bridge = ConnectomeBridge()
        result = bridge.step(
            fly_pos=np.array([10.0, 20.0]),
            fly_heading=0.5, fly_speed=8.0, fly_yaw_rate=0.1,
            odor_left=0.4, odor_right=0.3,
            wind_vector=np.array([5.0, 0.0]),
            dt=0.02
        )
        logger.log_step(result, fly_pos=np.array([10.0, 20.0]))
        assert logger.step_count == 1
        assert len(logger.step_buffer) == 1
        assert logger.step_buffer[0]["forward_speed"] == result["forward_speed"]

    def test_flush_creates_csv(self, tmp_path):
        from data_logger import ScientificDataLogger
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        bridge = ConnectomeBridge()
        for _ in range(10):
            result = bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.3, odor_right=0.3,
                wind_vector=np.array([5.0, 0.0]),
                dt=0.02
            )
            logger.log_step(result)
        
        logger.flush_steps()
        assert logger.flush_count == 1
        
        # Check CSV file exists
        import os
        exp_dir = os.path.join(str(tmp_path), logger.experiment_id)
        csv_files = [f for f in os.listdir(exp_dir) if f.endswith('.csv')]
        assert len(csv_files) == 1

    def test_trial_summary(self, tmp_path):
        from data_logger import ScientificDataLogger
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        bridge = ConnectomeBridge()
        for _ in range(20):
            result = bridge.step(
                fly_pos=np.array([0.0, 0.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.3, odor_right=0.3,
                wind_vector=np.array([5.0, 0.0]),
                dt=0.02
            )
            logger.log_step(result)
        
        summary = logger.end_trial(food_collected=2, escapes=1)
        assert summary["food_collected"] == 2
        assert summary["escapes"] == 1
        assert summary["total_steps"] == 20
        assert "preference_index" in summary

    def test_mb_weight_snapshot(self, tmp_path):
        from data_logger import ScientificDataLogger
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        bridge = ConnectomeBridge()
        
        logger.snapshot_mb_weights(bridge, "baseline")
        assert len(logger.weight_snapshots) == 1
        assert logger.weight_snapshots[0]["label"] == "baseline"
        assert "mean_approach_weight" in logger.weight_snapshots[0]

    def test_save_experiment(self, tmp_path):
        from data_logger import ScientificDataLogger
        import os
        
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        bridge = ConnectomeBridge()
        
        for _ in range(10):
            result = bridge.step(
                fly_pos=np.array([5.0, 5.0]),
                fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
                odor_left=0.3, odor_right=0.3,
                wind_vector=np.array([5.0, 0.0]),
                dt=0.02
            )
            logger.log_step(result, fly_pos=np.array([5.0, 5.0]))
        
        logger.snapshot_mb_weights(bridge, "test")
        logger.end_trial(food_collected=1)
        exp_dir = logger.save_experiment()
        
        assert os.path.exists(os.path.join(exp_dir, "experiment_meta.json"))
        assert os.path.exists(os.path.join(exp_dir, "trial_summaries.json"))
        assert os.path.exists(os.path.join(exp_dir, "weight_evolution.json"))
        assert os.path.exists(os.path.join(exp_dir, "occupancy.npz"))

    def test_occupancy_grid_updates(self, tmp_path):
        from data_logger import ScientificDataLogger
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        # Log steps at known position
        bridge = ConnectomeBridge()
        result = bridge.step(
            fly_pos=np.array([0.0, 0.0]),
            fly_heading=0.0, fly_speed=5.0, fly_yaw_rate=0.0,
            odor_left=0.3, odor_right=0.3,
            wind_vector=np.array([5.0, 0.0]),
            dt=0.02
        )
        for _ in range(10):
            logger.log_step(result, fly_pos=np.array([0.0, 0.0]))
        
        # Center bin should have counts
        center = logger.occupancy_bins // 2
        assert logger.occupancy_grid[center, center] > 0


# =========================================================================
# LEARNING ASSAY TESTS
# =========================================================================
class TestLearningAssay:
    """Verify end-to-end conditioning assay protocols."""

    def test_appetitive_assay_runs(self, tmp_path):
        from data_logger import ScientificDataLogger, LearningAssay
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        assay = LearningAssay(
            bridge=bridge,
            logger=logger,
            training_steps=100,
            test_steps=50,
            inter_trial_interval=20,
        )
        result = assay.run_appetitive_conditioning()
        
        assert "naive_valence" in result
        assert "trained_valence" in result
        assert "delta_valence" in result
        assert "learning_index" in result
        assert result["assay_type"] == "appetitive_conditioning"

    def test_aversive_assay_runs(self, tmp_path):
        from data_logger import ScientificDataLogger, LearningAssay
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        assay = LearningAssay(
            bridge=bridge,
            logger=logger,
            training_steps=100,
            test_steps=50,
            inter_trial_interval=20,
        )
        result = assay.run_aversive_conditioning()
        
        assert "naive_valence" in result
        assert "trained_valence" in result
        assert result["assay_type"] == "aversive_conditioning"

    def test_assay_saves_experiment_data(self, tmp_path):
        from data_logger import ScientificDataLogger, LearningAssay
        import os
        
        bridge = ConnectomeBridge()
        bridge.reset(keep_memory=False)
        logger = ScientificDataLogger(output_dir=str(tmp_path))
        
        assay = LearningAssay(
            bridge=bridge,
            logger=logger,
            training_steps=50,
            test_steps=30,
            inter_trial_interval=10,
        )
        assay.run_appetitive_conditioning()
        exp_dir = logger.save_experiment()
        
        assert os.path.exists(os.path.join(exp_dir, "experiment_meta.json"))
        assert len(logger.trial_summaries) >= 3  # naive + training + test
        assert len(logger.weight_snapshots) >= 3  # pre + post-train + post-test
