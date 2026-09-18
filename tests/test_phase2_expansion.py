"""
Comprehensive Unit and Integration Test Suite for Phase 2 Drosophila Neuroethological Expansion.
=============================================================================================
Verifies:
1. Thermosensory Channels:
   - Warm cells (dTrpA1/Gr28b): r_warm = max(0, (T_floor - 26.0) * 8.0) Hz.
   - Cold cells (BReP): r_cold = max(0, (25.0 - T_floor) * 8.0) Hz.
   - Thermal stress (T_floor > 35 C) driving PPL1 nociceptive dopamine and MDN backward walking.
   - Pain relief reward (Delta T < -3.0 C) triggering transient PAM dopaminergic burst (+1.0).
2. Visual Landmark Ingress:
   - Ring neurons (ER2/ER4d) mapping landmark azimuths to E-PG compass attractor bump.
   - Heading anchoring to external allocentric landmark cues.
3. Saccadic Efference Copy:
   - Corollary discharge shunt from voluntary saccade steering (DNa02) suppressing retinal slip by >= 80% (factor <= 0.20).
4. Courtship & Pheromone Sensilla:
   - cVA (Or67d -> DA1) and bitter mated female pheromones (Gr32a).
   - P1 courtship command neuron modulation and wing extension command.
   - MB gamma-lobe dopamine modulation.
5. Arena Experiment Paradigm Integration:
   - Arena(paradigm=...) instantiation via string and instance.
   - Geometry, dimensions, and spawn coordinate adaptation.
   - Sliding collisions, multimodal stimuli sampling, active zone reward/punishment.
   - Full backward compatibility with paradigm=None.
   - Closed-loop simulation with both modular and connectome brains.
"""

import math
import pytest
import numpy as np

from connectome_bridge import ConnectomeBridge
from arena import Arena, Position, FlyState
from maze import (
    ExperimentRegistry,
    TMazeParadigm,
    HeatMazeParadigm,
    BuridanParadigm,
    CourtshipParadigm,
    WindTunnelParadigm
)


# =========================================================================
# 1. THERMOSENSORY CHANNEL TESTS
# =========================================================================
class TestThermosensoryChannels:
    """Test warm/cold receptor kinetics, thermal stress, and pain relief burst."""

    def test_warm_receptor_kinetics(self):
        """Warm cells: r_warm = max(0, (T - 26.0) * 8.0) Hz."""
        bridge = ConnectomeBridge()
        
        # Below threshold (24 C)
        s1 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=24.0
        )
        assert s1["r_warm"] == 0.0
        
        # At threshold (26 C)
        s2 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=26.0
        )
        assert s2["r_warm"] == 0.0
        
        # Moderate warm (30 C): (30 - 26) * 8.0 = 32.0 Hz
        s3 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=30.0
        )
        assert abs(s3["r_warm"] - 32.0) < 1e-4

        # High warm (34 C): (34 - 26) * 8.0 = 64.0 Hz
        s4 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=34.0
        )
        assert abs(s4["r_warm"] - 64.0) < 1e-4

    def test_cold_receptor_kinetics(self):
        """Cold cells: r_cold = max(0, (25.0 - T) * 8.0) Hz."""
        bridge = ConnectomeBridge()
        
        # Above threshold (27 C)
        s1 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=27.0
        )
        assert s1["r_cold"] == 0.0
        
        # At threshold (25 C)
        s2 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=25.0
        )
        assert s2["r_cold"] == 0.0
        
        # Moderate cold (20 C): (25 - 20) * 8.0 = 40.0 Hz
        s3 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=20.0
        )
        assert abs(s3["r_cold"] - 40.0) < 1e-4

        # Strong cold (15 C): (25 - 15) * 8.0 = 80.0 Hz
        s4 = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=15.0
        )
        assert abs(s4["r_cold"] - 80.0) < 1e-4

    def test_thermal_stress_nociception_and_mdn_reversal(self):
        """T > 35 C triggers thermal stress, PPL1 dopamine, and MDN backward walking."""
        bridge = ConnectomeBridge()
        
        # Step at harmless temperature (25 C)
        out_normal = bridge.step(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=25.0
        )
        assert not out_normal["thermal_stress"]
        assert out_normal["mdn_rate"] < 5.0
        assert out_normal["forward_speed"] >= 0.0

        # Step into extreme heat (38 C)
        for _ in range(10):
            out_hot = bridge.step(
                fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
                odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=38.0, dt=0.05
            )
        assert out_hot["thermal_stress"] is True
        assert out_hot["ppl1_dopamine"] >= 1.0
        assert out_hot["mdn_rate"] >= 40.0  # MDN moonwalker activation
        # MDN command drives reverse propulsion
        assert out_hot["forward_speed"] < 0.0

    def test_pain_relief_reward_burst(self):
        """Delta T < -3.0 C triggers transient PAM burst (+1.0)."""
        bridge = ConnectomeBridge()
        bridge.reset()
        bridge.pam_dopamine = 0.0

        # Fly starts at 36 C
        bridge.step(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=36.0
        )
        bridge.pam_dopamine = 0.0  # Reset reward dopamine baseline

        # Small drop: 36 -> 34 C (Delta T = -2.0 C, not reaching -3.0 threshold)
        out_small_drop = bridge.step(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=34.0
        )
        assert not out_small_drop["pain_relief_burst"]
        assert out_small_drop["delta_t"] == -2.0

        # Substantial cooling relief: 34 -> 29 C (Delta T = -5.0 C < -3.0 C)
        out_relief = bridge.step(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2), temperature=29.0
        )
        assert out_relief["pain_relief_burst"] is True
        assert out_relief["delta_t"] == -5.0
        assert out_relief["pam_dopamine"] > 0.8  # Strong reward dopamine burst


# =========================================================================
# 2. VISUAL LANDMARK INGRESS TESTS
# =========================================================================
class TestVisualLandmarkIngress:
    """Test ER2/ER4d ring neuron mapping and compass bump anchoring to landmarks."""

    def test_ring_neuron_profile_orientation(self):
        """ER2/ER4d profile peak should correspond to landmark bearing."""
        bridge = ConnectomeBridge()
        
        # Landmark directly in front (bearing 0.0)
        s = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            landmarks=[0.0]
        )
        er_profile = s["er2_er4d_profile"]
        assert len(er_profile) == 16
        peak_idx = int(np.argmax(er_profile))
        peak_angle = bridge.wedge_angles[peak_idx]
        assert abs(peak_angle) < (2.0 * math.pi / 16)

    def test_compass_anchoring_to_landmark(self):
        """Heading bump should be pulled toward the external landmark bearing."""
        bridge = ConnectomeBridge()
        bridge.compass_bump_heading = 0.0
        bridge._update_compass_bump(0.0)

        # External landmark at +0.6 rad to the right
        target_bearing = 0.6
        for _ in range(15):
            out = bridge.step(
                fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=0.0, fly_yaw_rate=0.0,
                odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
                landmarks=[target_bearing], dt=0.05
            )

        # Bump should shift toward positive angles (closer to landmark)
        assert out["compass_bump_heading"] > 0.0
        assert out["visual_landmark_bearing"] == target_bearing


# =========================================================================
# 3. SACCADIC EFFERENCE COPY TESTS
# =========================================================================
class TestSaccadicEfferenceCopy:
    """Test ascending corollary discharge shunt suppressing retinal slip by >= 80%."""

    def test_voluntary_saccade_shunting(self):
        """During active saccade, optic slip should be shunted by >= 80% (factor <= 0.20)."""
        bridge = ConnectomeBridge()

        # Passive yaw (e.g. external rotation, not voluntary saccade)
        s_passive = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=1.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            is_saccade=False
        )
        assert s_passive["shunt_factor"] == 1.0
        assert not s_passive["efference_copy_active"]

        # Active voluntary saccade (is_saccade=True)
        s_active = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=1.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            is_saccade=True
        )
        assert s_active["efference_copy_active"] is True
        assert s_active["shunt_factor"] <= 0.20  # >= 80% suppression
        # Verify LPTC shunted signals are attenuated accordingly
        assert abs(s_active["hs_left_shunted"]) <= 0.20 * abs(s_passive["hs_left_raw"]) + 1e-6
        assert abs(s_active["hs_right_shunted"]) <= 0.20 * abs(s_passive["hs_right_raw"]) + 1e-6

    def test_dna02_high_rate_auto_triggers_efference_copy(self):
        """High DNa02 steering command automatically activates efference copy shunt."""
        bridge = ConnectomeBridge()
        # Set DNa02 differential steering to high value (> 15.0 Hz)
        bridge.dna02_rate_r = 25.0
        bridge.dna02_rate_l = 0.0

        s = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.5,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2)
        )
        assert s["efference_copy_active"] is True
        assert s["shunt_factor"] <= 0.20


# =========================================================================
# 4. COURTSHIP & PHEROMONE SENSILLA TESTS
# =========================================================================
class TestCourtshipPheromones:
    """Test cVA, bitter Gr32a sensilla, P1 command neurons, and MB gamma dopamine."""

    def test_cva_activation_of_da1(self):
        """cVA activates DA1 glomerulus in antennal lobe."""
        bridge = ConnectomeBridge()
        s = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            cva_odor=0.8
        )
        assert s["cva_rate"] > 80.0
        assert bridge.glom_rates[bridge.GLOM_DA1] == s["cva_rate"]

    def test_gr32a_bitter_pheromone_activation(self):
        """Bitter mated female pheromone stimulates Gr32a sensilla."""
        bridge = ConnectomeBridge()
        s = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            bitter_pheromone=0.75
        )
        assert s["gr32a_rate"] > 70.0

    def test_p1_command_activation_and_inhibition(self):
        """P1 command is stimulated by aphrodisiac and suppressed by cVA and bitter pheromones."""
        bridge = ConnectomeBridge()
        
        # Virgin female aphrodisiac only -> high P1 rate and wing extension
        s_virgin = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            female_aphrodisiac=0.9, cva_odor=0.0, bitter_pheromone=0.0
        )
        assert s_virgin["p1_courtship_rate"] > 50.0
        assert s_virgin["wing_extension_command"] > 70.0

        # Mated female (high cVA + bitter pheromones) -> suppresses P1
        s_mated = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            female_aphrodisiac=0.9, cva_odor=0.8, bitter_pheromone=0.8
        )
        assert s_mated["p1_courtship_rate"] < s_virgin["p1_courtship_rate"]
        assert s_mated["p1_courtship_rate"] == 0.0 or s_mated["p1_courtship_rate"] < 10.0
        assert s_mated["wing_extension_command"] < s_virgin["wing_extension_command"]

    def test_mb_gamma_dopamine_modulation(self):
        """Bitter pheromones drive MB gamma-lobe dopamine."""
        bridge = ConnectomeBridge()
        s_clean = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            bitter_pheromone=0.0
        )
        s_bitter = bridge.encode_sensory(
            fly_pos=np.array([50.0, 50.0]), fly_heading=0.0, fly_speed=10.0, fly_yaw_rate=0.0,
            odor_left=0.0, odor_right=0.0, wind_vector=np.zeros(2),
            bitter_pheromone=0.9
        )
        assert s_bitter["mb_gamma_dopamine"] > s_clean["mb_gamma_dopamine"]
        assert s_bitter["mb_gamma_dopamine"] >= 0.40


# =========================================================================
# 5. ARENA EXPERIMENT PARADIGM INTEGRATION TESTS
# =========================================================================
class TestArenaParadigmIntegration:
    """Test Arena integration with ExperimentParadigm catalog."""

    def test_arena_with_paradigm_by_name(self):
        """Arena should accept paradigm by string name from ExperimentRegistry."""
        arena = Arena(paradigm="t-maze")
        assert arena.paradigm is not None
        assert isinstance(arena.paradigm, TMazeParadigm)
        assert arena.width == arena.paradigm.dimensions[0]
        assert arena.height == arena.paradigm.dimensions[1]
        assert arena.fly.pos.x == 70.0
        assert arena.fly.pos.y == 18.0

    def test_arena_with_paradigm_by_instance(self):
        """Arena should accept paradigm as an instantiated object."""
        paradigm = HeatMazeParadigm()
        arena = Arena(paradigm=paradigm)
        assert arena.paradigm is paradigm
        assert arena.width == paradigm.dimensions[0]
        assert arena.height == paradigm.dimensions[1]

    def test_arena_backward_compatibility_when_paradigm_is_none(self):
        """Arena without paradigm must preserve all standard properties and behaviors."""
        arena = Arena(paradigm=None, num_food=3, num_hazards=2, num_predators=1)
        assert arena.paradigm is None
        assert hasattr(arena, "fly")
        assert hasattr(arena, "circuit")
        assert hasattr(arena, "surge_cast")
        assert hasattr(arena, "cx")
        assert hasattr(arena, "total_escapes")
        assert len(arena.food_positions) == 3
        assert len(arena.hazard_positions) == 2
        assert len(arena.predators) == 1

        # Run 5 steps
        for _ in range(5):
            res = arena.step(dt=1.0)
            assert "time_step" in res
            assert "fly_x" in res
            assert "fly_y" in res
            assert "satiety" in res
            assert "food_collected" in res

    def test_arena_paradigm_step_closed_loop_modular(self):
        """Verify Arena step query paradigm, collision check, and telemetry in modular mode."""
        arena = Arena(paradigm="heat-maze", brain_type="modular")
        step_out = arena.step(dt=0.1)

        assert "paradigm" in step_out
        assert step_out["paradigm"] == "heat_maze"
        assert "paradigm_telemetry" in step_out
        assert "paradigm_metrics" in step_out
        assert "stimuli" in step_out
        assert "temperature" in step_out["stimuli"]
        assert "active_zones" in step_out
        assert isinstance(step_out["active_zones"], list)

    def test_arena_paradigm_step_closed_loop_connectome(self):
        """Verify Arena step query paradigm with whole-brain connectome mode."""
        arena = Arena(paradigm="t-maze", brain_type="connectome", connectome_mode="surrogate")
        
        for _ in range(10):
            step_out = arena.step(dt=0.1)
            assert "connectome" in step_out
            if step_out["connectome"] is not None:
                assert "r_warm" in step_out["connectome"]
                assert "r_cold" in step_out["connectome"]
                assert "compass_bump_heading" in step_out["connectome"]

    def test_arena_paradigm_sliding_collision_containment(self):
        """Fly should slide along maze walls without penetrating or tunneling."""
        arena = Arena(paradigm="t-maze")
        fly = arena.fly

        # Aim fly directly into the left stem wall
        fly.pos.x = 66.0
        fly.pos.y = 25.0
        fly.heading = math.pi  # facing West into wall at x=63.0
        fly.speed = 3.0

        for _ in range(20):
            arena.step(dt=0.1)
            # Wall is at x=63.0, fly radius is 1.5 -> fly x must not penetrate < 64.5
            assert fly.pos.x >= 64.4

    def test_arena_courtship_paradigm_integration(self):
        """Verify Courtship paradigm stimulates pheromone sensilla in Arena."""
        arena = Arena(paradigm="courtship", brain_type="connectome", connectome_mode="surrogate")
        step_out = arena.step(dt=0.1)
        
        assert step_out["paradigm"] == "courtship"
        assert "aphrodisiac_concentration" in step_out["stimuli"]
        assert "cva_concentration" in step_out["stimuli"]
        c_telemetry = step_out["connectome"]
        assert c_telemetry is not None
        assert "p1_courtship_rate" in c_telemetry
        assert "wing_extension_command" in c_telemetry
