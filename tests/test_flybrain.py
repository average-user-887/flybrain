
from metabolic import MetabolicState
from mechanosensory import JohnstonsOrgan
from vision import CompoundEyeVision
from locomotion import TripodGaitCPG
from arena import Predator
"""Comprehensive test suite for FlyBrain Drosophila learning and navigation simulation.

Verifies:
1. Sparse combinatorial representation in Kenyon Cells.
2. Baseline neutral behavioral choices.
3. PAM dopamine-gated depression of avoidance synapses (appetitive learning).
4. PPL1 dopamine-gated depression of approach synapses (aversive learning).
5. Differential conditioning (Odor A appetitive vs Odor B aversive).
6. Strict adherence to biological efficacy bounds from Huang, Luo et al. Nature 2024.
7. Bilateral antenna gradient sniffing geometry.
8. Closed-loop chemotaxis navigation in the 2D arena.
9. Biological Surge-and-Cast state machine (Alvarez-Salvado 2018 / Demir 2020).
10. Central Complex (CX) E-PG heading compass and FB vector working memory (Hulse 2021 / Lyu 2022).
"""

import math
import unittest
import numpy as np

from circuit import MushroomBodyCircuit, MushroomBodySimulator
from arena import Arena, Position, FlyState, ContinuousOdorField
from surge_cast import SurgeCastEngine
from central_complex import CentralComplexEngine
from env_adapter import FlyBrainEnvAdapter


class TestMushroomBodyCircuit(unittest.TestCase):
    def test_kc_sparse_coding(self):
        """Kenyon cell population must show sparse (~5-15%) coding for odors."""
        circuit = MushroomBodyCircuit(seed=42)
        _, kc_hz_a = circuit.encode_odor(0.8, 0.0)
        active_a = np.sum(kc_hz_a > 0)
        fraction_a = active_a / circuit.n_kc

        self.assertTrue(0.03 <= fraction_a <= 0.20, f"KC active fraction {fraction_a:.1%} out of expected range")

        # Odor B should activate a distinct sparse subset
        _, kc_hz_b = circuit.encode_odor(0.0, 0.8)
        active_b = np.sum(kc_hz_b > 0)
        fraction_b = active_b / circuit.n_kc
        self.assertTrue(0.03 <= fraction_b <= 0.20)

        # Subsets should not be identical
        overlap = np.sum((kc_hz_a > 0) & (kc_hz_b > 0))
        self.assertTrue(overlap < active_a, "Odor A and Odor B should activate distinct sparse representations")

    def test_baseline_neutrality(self):
        """Untrained circuit should produce near-neutral behavioral preference."""
        sim = MushroomBodySimulator(seed=42)
        assay_a = sim.run_behavioral_assay((0.8, 0.0), num_trials=100)
        assay_b = sim.run_behavioral_assay((0.0, 0.8), num_trials=100)

        self.assertLess(abs(assay_a['mean_net_valence']), 0.05)
        self.assertLess(abs(assay_b['mean_net_valence']), 0.05)
        self.assertTrue(0.40 <= assay_a['approach_rate'] <= 0.60)
        self.assertTrue(0.40 <= assay_b['approach_rate'] <= 0.60)

    def test_pam_reward_plasticity(self):
        """Pairing Odor A with PAM reward should depress avoidance synapses and increase approach."""
        sim = MushroomBodySimulator(seed=42)
        naive_rate = sim.run_behavioral_assay((0.8, 0.0), num_trials=100)['approach_rate']

        for _ in range(4):
            sim.run_learning_trial(0.8, 0.0, reward=True, punishment=False, steps=80)

        eff_weights = sim.circuit.get_effective_weights()
        _, kc_hz = sim.circuit.encode_odor(0.8, 0.0)
        active_kcs = kc_hz > 0
        self.assertTrue(np.all(eff_weights[active_kcs, 1] < 0.95), "Avoidance synapses on active KCs must depress")
        self.assertTrue(np.allclose(eff_weights[active_kcs, 0], 1.0, atol=0.01))

        trained_assay = sim.run_behavioral_assay((0.8, 0.0), num_trials=100)
        self.assertGreater(trained_assay['mean_net_valence'], 0.25)
        self.assertGreater(trained_assay['approach_rate'], 0.75)
        self.assertGreater(trained_assay['approach_rate'], naive_rate)

    def test_ppl1_punishment_plasticity(self):
        """Pairing Odor B with PPL1 punishment should depress approach synapses and increase avoidance."""
        sim = MushroomBodySimulator(seed=42)

        for _ in range(4):
            sim.run_learning_trial(0.0, 0.8, reward=False, punishment=True, steps=80)

        eff_weights = sim.circuit.get_effective_weights()
        _, kc_hz = sim.circuit.encode_odor(0.0, 0.8)
        active_kcs = kc_hz > 0

        self.assertTrue(np.all(eff_weights[active_kcs, 0] < 0.95), "Approach synapses on active KCs must depress")
        self.assertTrue(np.allclose(eff_weights[active_kcs, 1], 1.0, atol=0.01))

        trained_assay = sim.run_behavioral_assay((0.0, 0.8), num_trials=100)
        self.assertLess(trained_assay['mean_net_valence'], -0.25)
        self.assertGreater(trained_assay['avoidance_rate'], 0.75)

    def test_differential_conditioning(self):
        """Simultaneous appetitive conditioning for Odor A and aversive for Odor B."""
        sim = MushroomBodySimulator(seed=42)

        for _ in range(3):
            sim.run_learning_trial(0.8, 0.0, reward=True, punishment=False, steps=80)
            sim.run_learning_trial(0.0, 0.8, reward=False, punishment=True, steps=80)

        assay_a = sim.run_behavioral_assay((0.8, 0.0), num_trials=100)
        assay_b = sim.run_behavioral_assay((0.0, 0.8), num_trials=100)

        self.assertGreater(assay_a['approach_rate'], 0.75)
        self.assertGreater(assay_b['avoidance_rate'], 0.75)

    def test_biological_weight_bounds(self):
        """Weights must remain strictly within Nature 2024 bounds [0.1, 2.0]."""
        sim = MushroomBodySimulator(seed=42)
        for _ in range(15):
            sim.run_learning_trial(0.8, 0.0, reward=True, punishment=False, steps=100)

        eff_weights = sim.circuit.get_effective_weights()
        self.assertTrue(np.all(eff_weights >= 0.099))
        self.assertTrue(np.all(eff_weights <= 2.001))


class TestSurgeCastAndCentralComplex(unittest.TestCase):
    def test_surge_cast_state_machine(self):
        """SurgeCastEngine must transition between WANDER, SURGE, CAST, and FEED."""
        sc = SurgeCastEngine(dt=0.01)
        self.assertEqual(sc.behavioral_state, 'WANDER')

        # Plume encounter: Odor concentration high -> enter SURGE
        for _ in range(25):
            v, omega, state = sc.step(c_left=0.8, c_right=0.75, wind_angle_rad=0.0)
        self.assertEqual(state, 'SURGE')
        self.assertGreater(v, sc.v0, "Surge speed should exceed baseline cruise speed")

        # Plume loss: Odor drops to zero -> enter CAST
        for _ in range(30):
            v, omega, state = sc.step(c_left=0.0, c_right=0.0, wind_angle_rad=0.0)
        self.assertEqual(state, 'CAST')
        self.assertLessEqual(v, sc.v0, "Cast speed should decrease for search")

        # Feeding contact -> enter FEED
        v_feed, omega_feed, state_feed = sc.step(c_left=0.9, c_right=0.9, is_feeding=True)
        self.assertEqual(state_feed, 'FEED')
        self.assertEqual(v_feed, 0.2)

    def test_central_complex_compass(self):
        """E-PG compass ring attractor tracks physical heading."""
        cx = CentralComplexEngine(n_wedges=16, seed=42)
        target_heading = math.pi / 3.0  # 60 degrees

        for _ in range(40):
            _, decoded_h, _ = cx.step(
                angular_vel=0.0,
                v_forward=1.2,
                mbon_valence=0.0,
                current_fly_heading=target_heading,
                dt=0.01
            )

        err = abs((decoded_h - target_heading + math.pi) % (2 * math.pi) - math.pi)
        self.assertLess(err, 0.35, "E-PG compass must align with head direction")

    def test_central_complex_vector_memory(self):
        """Fan-Shaped Body stores goal vector when positive valence is experienced."""
        cx = CentralComplexEngine(n_wedges=16, seed=42)
        food_heading = math.pi / 2.0 # 90 degrees North

        # Fly heads North while sensing reward (mbon_valence = +0.8)
        for _ in range(20):
            cx.step(
                angular_vel=0.0,
                v_forward=1.5,
                mbon_valence=0.8,
                current_fly_heading=food_heading,
                dt=0.01
            )

        self.assertTrue(cx.has_goal, "Goal vector must be anchored upon high reward")
        goal_angle = math.atan2(cx.goal_vector[1], cx.goal_vector[0])
        err = abs((goal_angle - food_heading + math.pi) % (2 * math.pi) - math.pi)
        self.assertLess(err, 0.30, "Memorized goal vector angle must point toward food direction")


class TestArenaNavigation(unittest.TestCase):
    def test_bilateral_sniffing(self):
        """Antenna closer to odor source must detect higher concentration."""
        arena = Arena(seed=42)
        arena.food_positions = [Position(70.0, 50.0)]
        arena.odor_a.clear()
        arena.odor_a.add_source(70.0, 50.0, 1.0)
        arena.initialize_fly(50.0, 50.0, heading=0.0)

        sensory = arena.sample_antennae()
        self.assertGreater(sensory['mean_a'], 0.0)
        self.assertAlmostEqual(sensory['diff_a'], 0.0, delta=0.05)

        # Fly heading North with source to East (Right)
        arena.fly.heading = math.pi / 2.0
        sensory = arena.sample_antennae()
        self.assertGreater(sensory['right_a'], sensory['left_a'])
        self.assertLess(sensory['diff_a'], 0.0)

    def test_closed_loop_foraging(self):
        """Trained fly uses chemotaxis, surge-cast, and CX memory to reach food closer than naive wanderer."""
        np.random.seed(42)
        food_pos = Position(70.0, 50.0)

        # Naive Arena
        arena_naive = Arena(seed=42)
        arena_naive.food_positions = [Position(70.0, 50.0)]
        arena_naive.hazard_positions = []
        arena_naive.odor_a.clear()
        arena_naive.odor_a.add_source(70.0, 50.0, 1.0)
        arena_naive.initialize_fly(50.0, 50.0, heading=math.pi / 2.0)

        # Trained Arena
        arena_trained = Arena(seed=42)
        arena_trained.food_positions = [Position(70.0, 50.0)]
        arena_trained.hazard_positions = []
        arena_trained.odor_a.clear()
        arena_trained.odor_a.add_source(70.0, 50.0, 1.0)
        arena_trained.initialize_fly(50.0, 50.0, heading=math.pi / 2.0)

        # Train arena_trained's Mushroom Body circuit
        sim = MushroomBodySimulator(seed=42)
        sim.circuit = arena_trained.circuit
        for _ in range(4):
            sim.run_learning_trial(0.8, 0.0, reward=True, steps=80)

        for _ in range(30):
            arena_naive.step()
            arena_trained.step()

        dist_naive = arena_naive.fly.pos.distance_to(food_pos)
        dist_trained = arena_trained.fly.pos.distance_to(food_pos)

        self.assertLess(dist_trained, dist_naive, f"Trained distance ({dist_trained:.1f}) must be closer than naive ({dist_naive:.1f})")



class TestFlyBrainEnvAdapter(unittest.TestCase):
    def test_env_reset_and_dimensions(self):
        """Env reset must return 12-dim observation and valid info state."""
        env = FlyBrainEnvAdapter(seed=42)
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (12,))
        self.assertIn("state", info)
        self.assertIn("dist_to_food", info)
        self.assertIn("in_food_zone", info)
        self.assertFalse(info["in_food_zone"])

    def test_env_biological_step(self):
        """Pure biological step (action=None) executes valid kinematics and state transitions."""
        env = FlyBrainEnvAdapter(seed=42)
        obs, _ = env.reset(seed=42)

        next_obs, reward, terminated, truncated, info = env.step(action=None)
        self.assertEqual(next_obs.shape, (12,))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn(info["state"], ["WANDER", "SURGE", "CAST", "FEED"])

    def test_env_residual_rl_step(self):
        """Residual RL blending applies forward acceleration and steering torque."""
        env = FlyBrainEnvAdapter(seed=42, residual_weight=0.5)
        env.reset(seed=42)

        # Apply strong forward and rightward torque action
        action = np.array([1.0, -0.5], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action=action)

        self.assertEqual(next_obs.shape, (12,))
        self.assertGreater(info["speed"], 0.0)

    def test_env_food_encounter_reward(self):
        """Reaching food triggers positive goal arrival bonus (+10) and termination."""
        env = FlyBrainEnvAdapter(seed=42)
        env.reset(seed=42)
        # Manually teleport fly to the edge of the food zone
        env.pos = env.food_pos - np.array([5.0, 0.0])

        next_obs, reward, terminated, truncated, info = env.step(action=None)
        self.assertTrue(terminated)
        self.assertTrue(info["in_food_zone"])
        self.assertGreater(reward, 5.0, "Reward must include substantial goal arrival bonus")



class TestAdvancedEcosystemAndBiophysics(unittest.TestCase):
    def test_metabolic_hunger_modulation(self):
        """Metabolic state decays over time, feeding restores it, and hunger amplifies dopamine."""
        met = MetabolicState(initial_satiety=0.8, decay_rate=0.01)
        self.assertAlmostEqual(met.satiety, 0.8)
        self.assertFalse(met.is_starving)

        # Step 50 seconds equivalent
        for _ in range(50):
            met.step(dt=1.0, speed=10.0)

        self.assertLess(met.satiety, 0.8)
        # Gain should be higher than baseline
        self.assertGreater(met.get_dopamine_gain(), 1.0)
        self.assertLess(met.get_stop_suppression(), 1.0)

        # Feed
        prev_satiety = met.satiety
        met.feed(0.3)
        self.assertGreater(met.satiety, prev_satiety)

    def test_johnstons_organ_wind_deflection(self):
        """Johnston's organ deflects aristae based on relative aerodynamic wind."""
        jo = JohnstonsOrgan(n_wedges=16)

        # 1. Fly flying directly into headwind (wind blowing West (-x), fly heading East (+x))
        # Wind vx = -10.0, fly heading = 0.0 (East)
        data = jo.step(fly_heading=0.0, fly_speed=5.0, wind_vx=-10.0, wind_vy=0.0)
        self.assertGreater(data["v_rel_mag"], 0.0)
        # 16-wedge bump must have non-zero activity
        self.assertEqual(len(data["cx_bump"]), 16)
        self.assertGreater(np.sum(data["cx_bump"]), 0.0)

        # 2. Crosswind / oblique airflow produces distinct left vs right arista deflection
        cross_data = jo.step(fly_heading=0.0, fly_speed=0.0, wind_vx=5.0, wind_vy=10.0)
        self.assertNotEqual(cross_data["deflect_left_deg"], cross_data["deflect_right_deg"])

    def test_optic_flow_and_looming_escape(self):
        """Approaching visual predator triggers LC4 looming detector and ballistic escape."""
        vision = CompoundEyeVision(looming_threshold=0.05, escape_duration=0.3)
        fly_pos = np.array([50.0, 50.0])
        fly_heading = 0.0

        # Stationary predator far away -> no escape
        far_pred = [np.array([180.0, 50.0])]
        data_far = vision.step(fly_pos, fly_heading, fly_speed=1.0, fly_yaw_rate=0.0,
                               predator_positions=far_pred, predator_velocities=[np.zeros(2)], dt=0.02)
        self.assertFalse(data_far["escape_active"])

        # Rapidly approaching predator closing in from behind
        close_pred = [np.array([58.0, 50.0])]
        vel_approaching = [np.array([-15.0, 0.0])] # Charging at fly at 15 mm/s
        data_threat = vision.step(fly_pos, fly_heading, fly_speed=1.0, fly_yaw_rate=0.0,
                                  predator_positions=close_pred, predator_velocities=vel_approaching, dt=0.02)

        self.assertTrue(data_threat["escape_active"])
        self.assertGreater(data_threat["looming_intensity"], 0.05)

    def test_tripod_cpg_gait(self):
        """Kuramoto CPG maintains anti-phase tripod coupling and modulates stepping frequency."""
        cpg = TripodGaitCPG(base_freq_hz=8.0)

        # Step forward with symmetric descending drive
        res = cpg.step(dn_drive_left=1.0, dn_drive_right=1.0, dt=0.02)
        self.assertGreater(res["forward_speed"], 0.0)
        self.assertAlmostEqual(abs(res["phase_b"] - res["phase_a"]), math.pi, delta=0.05)

        # Verify 6 legs have stance states
        self.assertEqual(len(res["leg_states"]), 6)
        # In tripod gait, L1 and R1 must have opposite stance/swing states
        self.assertNotEqual(res["leg_states"]["L1"], res["leg_states"]["R1"])

    def test_multi_agent_predator_foraging_ecosystem(self):
        """Multi-fly arena with predators runs stably with foraging, stalking, and escapes."""
        arena = Arena(
            width=120.0,
            height=120.0,
            num_food=3,
            num_hazards=2,
            wind=(-0.4, 0.1),
            seed=42,
            num_flies=5,
            num_predators=2
        )

        self.assertEqual(len(arena.flies), 5)
        self.assertEqual(len(arena.predators), 2)

        # Run 40 ticks
        for _ in range(40):
            st = arena.step(dt=1.0)
            self.assertIn("satiety", st)

        # Check all flies are within arena boundaries and have finite coordinates
        for fly in arena.flies:
            self.assertTrue(0.0 <= fly.pos.x <= arena.width)
            self.assertTrue(0.0 <= fly.pos.y <= arena.height)
            self.assertFalse(math.isnan(fly.pos.x))
            self.assertFalse(math.isnan(fly.pos.y))


if __name__ == '__main__':
    unittest.main(verbosity=2)
