"""Unit and Integration Tests for Multisensory Limb & Full Body Integration Benchmark.

Tests:
1. Paradigm initialization, 64-wall circular + pillar geometry, and zone containment.
2. Multi-sensory stimulus sampling (odors A/B/cVA, thermal terrain, vector wind, visual landmarks).
3. 6-leg biomechanical joint angle kinematics and Campaniform Sensilla (CS) load forces.
4. Direct manual neuro-stimulation overrides (thrust, steering, limb stance/swing).
5. Quantitative benchmark scoring (coordination, sensory integration, smoothness, composite).
6. Full closed-loop integration with Arena.
"""

import math
import pytest
try:
    from flybrain.maze import ExperimentRegistry, MultisensoryLimbBenchmark
    from flybrain.arena import Arena
except ImportError:
    from maze import ExperimentRegistry, MultisensoryLimbBenchmark
    from arena import Arena


class TestMultisensoryBenchmarkInitialization:
    def test_registry_lookup(self):
        p1 = ExperimentRegistry.get("multisensory_benchmark")
        p2 = ExperimentRegistry.get("multisensory_sandbox")
        p3 = ExperimentRegistry.get("multisensory")
        assert isinstance(p1, MultisensoryLimbBenchmark)
        assert isinstance(p2, MultisensoryLimbBenchmark)
        assert isinstance(p3, MultisensoryLimbBenchmark)

    def test_dimensions_and_geometry(self):
        bm = MultisensoryLimbBenchmark()
        assert bm.dimensions == (160.0, 160.0)
        assert bm.arena_radius == 75.0
        # 32 boundary walls + 4 pillars * 8 walls = 64
        assert len(bm.walls) == 64
        assert len(bm.zones) == 4
        assert len(bm.landmarks) == 4

    def test_zone_containment(self):
        bm = MultisensoryLimbBenchmark()
        active_food = [z.name for z in bm.get_active_zones(45.0, 45.0)]
        assert "food_refuge" in active_food
        assert "open_arena" in active_food

        active_hot = [z.name for z in bm.get_active_zones(-40.0, 40.0)]
        assert "thermal_hotspot" in active_hot

        active_phero = [z.name for z in bm.get_active_zones(45.0, -45.0)]
        assert "pheromone_zone" in active_phero


class TestMultiSensoryStimulusIngress:
    def test_odor_gradients(self):
        bm = MultisensoryLimbBenchmark()
        stim_food = bm.sample_stimuli(45.0, 45.0, heading=0.0)
        assert stim_food["odor_a"] > 0.95
        assert stim_food["odor_b"] < 0.05

        stim_repel = bm.sample_stimuli(-45.0, -45.0, heading=0.0)
        assert stim_repel["odor_b"] > 0.95
        assert stim_repel["odor_a"] < 0.05

    def test_thermal_terrain(self):
        bm = MultisensoryLimbBenchmark()
        stim_hot = bm.sample_stimuli(-40.0, 40.0, heading=0.0)
        assert stim_hot["temperature"] > 37.0

        stim_cool = bm.sample_stimuli(45.0, 45.0, heading=0.0)
        assert stim_cool["temperature"] < 23.0

        stim_ambient = bm.sample_stimuli(0.0, 0.0, heading=0.0)
        assert 23.5 <= stim_ambient["temperature"] <= 24.5

    def test_mechanosensory_wind(self):
        bm = MultisensoryLimbBenchmark()
        stim = bm.sample_stimuli(0.0, 0.0, heading=0.0)
        assert stim["wind_magnitude"] == pytest.approx(15.0, abs=0.1)
        assert stim["jo_antenna_deflect_un"] > 1.5

    def test_visual_landmarks(self):
        bm = MultisensoryLimbBenchmark()
        stim = bm.sample_stimuli(0.0, 0.0, heading=0.0)
        lms = stim["landmarks"]
        assert len(lms) == 4
        # Pillar NE at (35, 35) from origin should have bearing ~ +pi/4
        ne = next(l for l in lms if l["id"] == "pillar_ne")
        assert ne["bearing_rad"] == pytest.approx(math.pi / 4, abs=0.05)


class TestBiomechanicsAndLimbControl:
    def test_tripod_phase_progression(self):
        bm = MultisensoryLimbBenchmark()
        fly = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 12.0}

        # Step 20 times
        for _ in range(20):
            res = bm.step(fly, dt=0.02)

        bio = res["biomechanics"]
        assert bio["cpg_freq_hz"] >= 6.0
        # Leg states exist for all 6 legs
        assert len(bio["leg_states"]) == 6
        assert len(bio["joint_angles"]) == 6
        assert len(bio["cuticular_loads"]) == 6

        # Check anti-phase between L1 and R1
        l1_phi = bm.leg_phases["L1"]
        r1_phi = bm.leg_phases["R1"]
        diff = abs((l1_phi - r1_phi + math.pi) % (2 * math.pi) - math.pi)
        assert diff == pytest.approx(math.pi, abs=0.2)

    def test_manual_override_deck(self):
        bm = MultisensoryLimbBenchmark()
        fly = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 12.0}

        # Step with override_thrust
        res_fast = bm.step(fly, dt=0.02, override_thrust=1.0)
        assert res_fast["biomechanics"]["manual_override_active"] is True
        assert res_fast["biomechanics"]["cpg_freq_hz"] > 10.0

        # Step with manual leg stance hold
        res_leg = bm.step(fly, dt=0.02, override_legs={"L1": True, "R1": False})
        assert res_leg["biomechanics"]["leg_states"]["L1"] is True
        assert res_leg["biomechanics"]["leg_states"]["R1"] is False


class TestBenchmarkScoringEngine:
    def test_benchmark_metrics_computation(self):
        bm = MultisensoryLimbBenchmark()
        fly = {"x": 0.0, "y": 0.0, "heading": math.pi / 4, "speed": 14.0}

        for _ in range(50):
            res = bm.step(fly, dt=0.02)

        m = res["metrics"]
        assert 0.0 <= m["locomotor_coordination_index"] <= 1.0
        assert 0.0 <= m["multisensory_integration_score"] <= 1.0
        assert 0.0 <= m["biomechanical_efficiency"] <= 1.0
        assert 0.0 <= m["kinematic_smoothness"] <= 1.0
        assert 0.0 <= m["composite_benchmark_score"] <= 100.0

    def test_reset_trial_clears_metrics(self):
        bm = MultisensoryLimbBenchmark()
        fly = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 10.0}
        for _ in range(20):
            bm.step(fly, dt=0.02)
        assert bm.total_distance > 0.0

        bm.reset_trial()
        assert bm.total_distance == 0.0
        assert bm.wall_collision_count == 0
        assert len(bm.path_points) == 0


class TestClosedLoopArenaIntegration:
    def test_arena_runs_multisensory_benchmark(self):
        arena = Arena(paradigm="multisensory_benchmark")
        assert arena.paradigm.name == "Multisensory Limb & Body Benchmark"

        for _ in range(30):
            obs = arena.step()

        assert obs["paradigm_metrics"]["composite_benchmark_score"] > 0.0
        assert "food_refuge" in obs["paradigm_metrics"] or "locomotor_coordination_index" in obs["paradigm_metrics"]
        assert obs["food_collected"] >= 0
