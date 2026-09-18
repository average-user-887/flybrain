"""Comprehensive Test Suite for Drosophila Neuroethological Experiment Catalog.

Verifies:
1. Geometric and physics primitives (WallSegment, CircularMoat, PeltierGrid, VisualLandmark, MazeZone, CollisionEngine).
2. Zero-tunneling continuous sliding collision resolution over 20,000 simulation steps in narrow corridors.
3. All 12 canonical neuroethological experiment paradigms:
   - T-Maze olfactory Pavlovian conditioning and Performance Index (PI).
   - Y-Maze spontaneous alternation (SAR) and turn handedness bias.
   - Thermal Heat-Maze place learning and Ofstad et al. 2011 double dissociation (CX vs MB).
   - Buridan's visual landmark fixation and centrophobism.
   - Visual Operant flight simulator yaw conditioning and laser heat punishment.
   - Wind Tunnel surge-and-cast anemotactic odor plume navigation.
   - Looming predator escape, Giant Fiber (GF) spike threshold, and ballistic jump timing.
   - Optomotor gaze stabilization and saccadic efference copy shunting (>80%).
   - Gap Crossing chasm reachability decision threshold (3.8mm).
   - Circadian DAM locomotor rhythm, beam breaks, and 5-min sleep bout detection.
   - Courtship conditioning, male wing extension, female rejection kicks, and cVA suppression.
   - Corridor Labyrinth 12-segment multi-fork navigation and sliding collisions.
4. ExperimentRegistry factory registration and paradigm retrieval.
"""

import math
import os
import sys
import unittest
import numpy as np

# Adjust module search path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maze import (
    WallSegment,
    CircularMoat,
    PeltierGrid,
    VisualLandmark,
    MazeZone,
    CollisionEngine,
    ExperimentParadigm,
    ExperimentRegistry,
    TMazeParadigm,
    YMazeParadigm,
    HeatMazeParadigm,
    BuridanParadigm,
    VisualOperantParadigm,
    WindTunnelParadigm,
    LoomingEscapeParadigm,
    OptomotorParadigm,
    GapCrossingParadigm,
    CircadianDAMParadigm,
    CourtshipParadigm,
    LabyrinthParadigm,
)


class TestMazeGeometricPrimitives(unittest.TestCase):
    """Unit tests for geometric and physics primitives."""

    def test_wall_segment_sliding_physics(self):
        """Test WallSegment projection, distance, normal, and sliding collision response."""
        # Horizontal wall from (0, 10) to (20, 10)
        wall = WallSegment((0.0, 10.0), (20.0, 10.0), friction=0.5, restitution=0.1)

        # 1. Projection and distance
        cx, cy, t = wall.project_point(10.0, 15.0)
        self.assertAlmostEqual(cx, 10.0)
        self.assertAlmostEqual(cy, 10.0)
        self.assertAlmostEqual(t, 0.5)
        self.assertAlmostEqual(wall.distance_to_point(10.0, 15.0), 5.0)

        # Clamping at endpoints
        cx_left, cy_left, t_left = wall.project_point(-5.0, 10.0)
        self.assertAlmostEqual(cx_left, 0.0)
        self.assertAlmostEqual(t_left, 0.0)

        cx_right, cy_right, t_right = wall.project_point(25.0, 10.0)
        self.assertAlmostEqual(cx_right, 20.0)
        self.assertAlmostEqual(t_right, 1.0)

        # 2. Normal vector
        nx, ny = wall.get_normal()
        self.assertAlmostEqual(nx * nx + ny * ny, 1.0)

        # 3. Direct perpendicular collision
        # Circle at (10, 9.0) with radius 1.5 penetrating wall at y=10
        # Incoming velocity (0, 2) moving towards wall
        rx, ry, rvx, rvy, collided, norm = wall.resolve_circle_collision(
            x=10.0, y=9.0, vx=0.0, vy=2.0, radius=1.5
        )
        self.assertTrue(collided)
        # Position should be pushed out to exactly radius distance (y = 10 - 1.5 = 8.5)
        self.assertAlmostEqual(ry, 8.5, places=5)
        self.assertAlmostEqual(rx, 10.0, places=5)
        # Normal velocity reflected with restitution 0.1: incoming 2.0 -> rebound -0.2
        self.assertAlmostEqual(rvy, -0.2, places=5)
        self.assertAlmostEqual(rvx, 0.0, places=5)

        # 4. Oblique collision with sliding friction (45 deg angle)
        rx, ry, rvx, rvy, collided, norm = wall.resolve_circle_collision(
            x=10.0, y=9.0, vx=2.0, vy=2.0, radius=1.5
        )
        self.assertTrue(collided)
        self.assertAlmostEqual(ry, 8.5, places=5)
        # Tangential velocity vx=2.0 attenuated by friction 0.5 -> 1.0
        self.assertAlmostEqual(rvx, 1.0, places=5)
        self.assertAlmostEqual(rvy, -0.2, places=5)

        # 5. Non-colliding circle
        rx, ry, rvx, rvy, collided, norm = wall.resolve_circle_collision(
            x=10.0, y=5.0, vx=0.0, vy=1.0, radius=1.5
        )
        self.assertFalse(collided)
        self.assertEqual(rx, 10.0)
        self.assertEqual(ry, 5.0)

    def test_zero_collision_tunneling_20000_steps(self):
        """Stress-test CollisionEngine over 20,000 steps in a narrow 12mm corridor.

        Acceptance criterion: Exactly 0 tunneling or clipping events outside corridor walls.
        """
        # Corridor with walls at y = 0 and y = 12, length 100mm
        corridor_walls = [
            WallSegment((0.0, 0.0), (100.0, 0.0)),
            WallSegment((0.0, 12.0), (100.0, 12.0)),
            WallSegment((0.0, 0.0), (0.0, 12.0)),
            WallSegment((100.0, 0.0), (100.0, 12.0)),
        ]
        engine = CollisionEngine(corridor_walls)

        radius = 1.5
        x, y = 50.0, 6.0
        rng = np.random.default_rng(42)

        tunneling_violations = 0
        total_steps = 20000

        for _ in range(total_steps):
            # Aggressive random velocity
            speed = rng.uniform(0.5, 4.0)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            # Move and resolve
            x_cand = x + vx * 0.1
            y_cand = y + vy * 0.1
            x, y, vx, vy, collided, _ = engine.resolve(x_cand, y_cand, vx, vy, radius=radius)

            # Assert circle center is strictly confined within [radius, 12 - radius]
            if y < (radius - 1e-5) or y > (12.0 - radius + 1e-5):
                tunneling_violations += 1
            if x < (radius - 1e-5) or x > (100.0 - radius + 1e-5):
                tunneling_violations += 1

        self.assertEqual(tunneling_violations, 0, f"Found {tunneling_violations} tunneling violations!")

    def test_circular_moat_containment(self):
        """Test CircularMoat boundary detection and containment resolution."""
        moat = CircularMoat(center=(50.0, 50.0), radius=40.0)

        # Inside platform
        self.assertTrue(moat.contains(50.0, 50.0))
        self.assertTrue(moat.contains(70.0, 50.0))
        self.assertFalse(moat.is_in_water(50.0, 50.0))
        self.assertGreater(moat.distance_to_boundary(50.0, 50.0), 0.0)

        # In water moat
        self.assertFalse(moat.contains(95.0, 50.0))
        self.assertTrue(moat.is_in_water(95.0, 50.0))
        self.assertLess(moat.distance_to_boundary(95.0, 50.0), 0.0)

        # Containment resolution
        # Circle center at (92, 50) with radius 1.5 should be clamped inside radius 40
        cx, cy, cvx, cvy, contained = moat.resolve_containment(92.0, 50.0, vx=2.0, vy=0.0, radius=1.5)
        self.assertTrue(contained)
        self.assertAlmostEqual(cx, 50.0 + (40.0 - 1.5), places=5)
        self.assertLessEqual(cvx, 0.0)  # Rebounded inward

    def test_peltier_grid_temperature_field(self):
        """Test PeltierGrid temperature field smooth radial gradient."""
        grid = PeltierGrid(
            baseline_temp=36.5,
            cool_spot=(22.0, 18.0),
            cool_radius=9.0,
            cool_temp=24.0,
            gradient_sigma=8.0
        )

        # Inside cool refuge: must be exactly cool_temp (24.0 C)
        self.assertAlmostEqual(grid.get_temperature(22.0, 18.0), 24.0, places=3)
        self.assertAlmostEqual(grid.get_temperature(22.0 + 8.5, 18.0), 24.0, places=3)

        # Boundary of cool refuge
        self.assertAlmostEqual(grid.get_temperature(22.0 + 9.0, 18.0), 24.0, places=3)

        # Intermediate distance: smooth gradient rising towards 36.5 C
        t_mid = grid.get_temperature(22.0 + 17.0, 18.0)
        self.assertGreater(t_mid, 24.0)
        self.assertLess(t_mid, 36.5)

        # Distant floor: asymptotically approaches baseline_temp (36.5 C)
        t_far = grid.get_temperature(22.0 + 50.0, 18.0)
        self.assertAlmostEqual(t_far, 36.5, delta=0.1)

    def test_visual_landmark_bearings(self):
        """Test VisualLandmark apparent bearing calculation relative to fly heading."""
        # Distal landmark at 0 rad (East)
        lm_distal = VisualLandmark("lm_east", azimuth_rad=0.0)

        # Fly at origin facing East (heading = 0) -> bearing = 0
        b1 = lm_distal.get_apparent_bearing(0.0, 0.0, fly_heading=0.0)
        self.assertAlmostEqual(b1, 0.0)

        # Fly facing North (heading = pi/2) -> landmark is 90 deg to right (-pi/2)
        b2 = lm_distal.get_apparent_bearing(0.0, 0.0, fly_heading=math.pi / 2.0)
        self.assertAlmostEqual(b2, -math.pi / 2.0)

        # Proximal landmark with explicit coordinates (10, 10)
        lm_prox = VisualLandmark("lm_fixed", azimuth_rad=0.0, pos=(10.0, 10.0))
        # Fly at (10, 0) facing North (pi/2) -> looking directly at landmark
        b3 = lm_prox.get_apparent_bearing(10.0, 0.0, fly_heading=math.pi / 2.0)
        self.assertAlmostEqual(b3, 0.0)

    def test_maze_zone_containment(self):
        """Test MazeZone containment for circle, rectangle, and polygon bounds."""
        # Rectangular zone
        rect_zone = MazeZone("rect", "corridor", (10.0, 20.0, 30.0, 40.0), reward=1.0)
        self.assertTrue(rect_zone.contains(20.0, 30.0))
        self.assertFalse(rect_zone.contains(5.0, 30.0))
        self.assertEqual(rect_zone.reward, 1.0)

        # Circular zone
        circ_zone = MazeZone("circle", "refuge", (50.0, 50.0, 10.0), punishment=0.5)
        self.assertTrue(circ_zone.contains(55.0, 50.0))
        self.assertFalse(circ_zone.contains(65.0, 50.0))
        self.assertEqual(circ_zone.punishment, 0.5)

        # Polygon zone (triangle)
        poly = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        poly_zone = MazeZone("poly", "custom", poly)
        self.assertTrue(poly_zone.contains(5.0, 3.0))
        self.assertFalse(poly_zone.contains(8.0, 8.0))


class TestTwelveParadigms(unittest.TestCase):
    """Comprehensive test suite for all 12 concrete neuroethological paradigms."""

    def test_t_maze_conditioning_and_choice(self):
        """Paradigm 1: T-Maze corridor confinement, odor sampling, and Performance Index."""
        t_maze = TMazeParadigm(cs_plus_arm='arm_a')
        self.assertEqual(t_maze.name, "t_maze")

        # Stem base: vacuum airflow should point down (0, -15 mm/s)
        stim_stem = t_maze.sample_stimuli(70.0, 20.0, heading=math.pi / 2.0)
        self.assertEqual(stim_stem['wind'], (0.0, -15.0))

        # Left arm: CS+ odor should be stronger than CS-
        stim_left = t_maze.sample_stimuli(25.0, 50.0, heading=0.0)
        self.assertGreater(stim_left['odor_cs_plus'], stim_left['odor_cs_minus'])

        # Right arm: CS- odor should be stronger than CS+
        stim_right = t_maze.sample_stimuli(115.0, 50.0, heading=0.0)
        self.assertGreater(stim_right['odor_cs_minus'], stim_right['odor_cs_plus'])

        # Step fly choosing Arm A
        res_a = t_maze.step({'x': 30.0, 'y': 50.0, 'heading': math.pi}, dt=0.05)
        self.assertIn("arm_a", res_a['active_zones'])
        self.assertEqual(res_a['reward'], 1.0)
        self.assertEqual(res_a['first_choice'], "arm_a")

        metrics = t_maze.get_metrics()
        self.assertEqual(metrics['first_choice'], "arm_a")
        self.assertEqual(metrics['performance_index'], 1.0)

        # Reset trial
        reset_info = t_maze.reset_trial()
        self.assertEqual(reset_info['trial_number'], 2)
        self.assertIsNone(t_maze.first_choice)

    def test_y_maze_spontaneous_alternation(self):
        """Paradigm 2: Y-Maze 3-arm geometry and Spontaneous Alternation Rate (SAR)."""
        y_maze = YMazeParadigm()
        self.assertEqual(y_maze.name, "y_maze")
        self.assertEqual(len(y_maze.arm_angles), 3)

        # Simulate alternating arm sequence A -> B -> C -> A -> B -> C
        # Arm 0 (tip at 60, 100), Arm 1 (tip at 25, 40), Arm 2 (tip at 94, 40)
        arm_coords = [(60.0, 95.0), (28.0, 42.0), (92.0, 42.0)]
        for i in range(6):
            arm_idx = i % 3
            coord = arm_coords[arm_idx]
            # Visit hub then arm
            y_maze.step({'x': 60.0, 'y': 60.0, 'heading': 0.0})
            y_maze.step({'x': coord[0], 'y': coord[1], 'heading': 0.0})

        metrics = y_maze.get_metrics()
        # 6 arm visits form 4 triads: (0,1,2), (1,2,0), (2,0,1), (0,1,2). All alternating!
        self.assertEqual(metrics['total_triads'], 4)
        self.assertEqual(metrics['alternating_triads'], 4)
        self.assertAlmostEqual(metrics['spontaneous_alternation_rate'], 1.0)
        self.assertIn('handedness_index', metrics)

    def test_heat_maze_thermal_field_and_pain_relief(self):
        """Paradigm 3: Heat-Maze temperature gradient, thermosensory rates, and cool refuge PAM burst."""
        heat_maze = HeatMazeParadigm()
        self.assertEqual(heat_maze.name, "heat_maze")

        # Nociceptive heated floor
        stim_floor = heat_maze.sample_stimuli(20.0, 20.0, heading=0.0)
        self.assertAlmostEqual(stim_floor['temperature'], 36.5, delta=0.5)
        self.assertGreater(stim_floor['r_warm_thermosensory'], 50.0)
        self.assertEqual(stim_floor['r_cold_thermosensory'], 0.0)

        # Cool refuge entry
        rx, ry = heat_maze.refuge_pos
        step_res = heat_maze.step({'x': rx, 'y': ry, 'heading': 0.0}, dt=0.05)
        self.assertTrue(step_res['refuge_reached'])
        self.assertTrue(step_res['pam_burst_active'])
        self.assertEqual(step_res['reward'], 1.0)
        self.assertIsNotNone(step_res['escape_latency_ms'])

    def test_heat_maze_cx_vs_mb_lesion_dissociation(self):
        """Paradigm 3: Double dissociation of spatial place learning (Ofstad et al., Nature 2011).

        - WT & MB_lesion: Learn refuge location via Central Complex, achieving low escape latency.
        - CX_lesion: Central complex lesion prevents allocentric place navigation, causing high latency.
        """
        heat_maze = HeatMazeParadigm()

        # Trained WT agent
        res_wt = heat_maze.simulate_agent_trial(agent_type='WT', trained=True, seed=42)
        self.assertTrue(res_wt['refuge_reached'])
        self.assertLess(res_wt['steps'], 100)

        # Trained MB-lesioned agent (place learning spared)
        res_mb = heat_maze.simulate_agent_trial(agent_type='MB_lesion', trained=True, seed=42)
        self.assertTrue(res_mb['refuge_reached'])
        self.assertLess(res_mb['steps'], 100)

        # CX-lesioned agent (place learning impaired, random wandering)
        res_cx = heat_maze.simulate_agent_trial(agent_type='CX_lesion', trained=True, seed=42, max_steps=200)
        self.assertGreater(res_cx['steps'], res_wt['steps'] * 2)

    def test_buridan_stripe_fixation_and_centrophobism(self):
        """Paradigm 4: Buridan's stripe fixation and centrophobism index."""
        buridan = BuridanParadigm()
        self.assertEqual(buridan.name, "buridan")

        # Fly walking along perimeter facing stripe at 0 rad (East)
        buridan.step({'x': 90.0, 'y': 60.0, 'heading': 0.0}, dt=0.1)
        buridan.step({'x': 95.0, 'y': 60.0, 'heading': 0.0}, dt=0.1)

        metrics = buridan.get_metrics()
        self.assertGreater(metrics['centrophobism_index'], 0.9)
        self.assertGreater(metrics['mean_stripe_fixation'], 0.9)

    def test_visual_operant_torque_conditioning(self):
        """Paradigm 5: Visual Operant flight simulator closed-loop yaw torque and learning index."""
        operant = VisualOperantParadigm(coupling_gain=120.0)
        self.assertEqual(operant.name, "visual_operant")

        # Start drum in safe quadrant (angle = 45 deg)
        operant.drum_angle_deg = 45.0
        res_safe = operant.step({'x': 40.0, 'y': 40.0, 'angular_velocity': 0.0}, dt=0.1)
        self.assertFalse(res_safe['laser_active'])
        self.assertEqual(res_safe['punishment'], 0.0)

        # Rotate drum into punished quadrant (angle = 135 deg)
        operant.drum_angle_deg = 135.0
        res_punished = operant.step({'x': 40.0, 'y': 40.0, 'angular_velocity': 0.0}, dt=0.1)
        self.assertTrue(res_punished['laser_active'])
        self.assertEqual(res_punished['punishment'], 1.0)

        metrics = operant.get_metrics()
        self.assertIn('operant_learning_index', metrics)

    def test_wind_tunnel_surge_cast_plume_tracking(self):
        """Paradigm 6: Wind tunnel laminar wind vector, plume encounter, and surge/cast transitions."""
        tunnel = WindTunnelParadigm()
        self.assertEqual(tunnel.name, "wind_tunnel")

        # In plume centerline (y = 30) -> SURGE
        res_surge = tunnel.step({'x': 100.0, 'y': 30.0, 'heading': 0.0}, dt=0.1)
        self.assertEqual(res_surge['behavioral_state'], "SURGE")

        # Outside plume (y = 55) -> CAST
        res_cast = tunnel.step({'x': 100.0, 'y': 55.0, 'heading': 0.0}, dt=0.1)
        self.assertEqual(res_cast['behavioral_state'], "CAST")

        # Reaching upstream source nozzle (180, 30)
        res_goal = tunnel.step({'x': 180.0, 'y': 30.0, 'heading': 0.0}, dt=0.1)
        self.assertTrue(res_goal['source_reached'])
        self.assertIsNotNone(res_goal['time_to_source_ms'])

    def test_looming_escape_gf_vs_non_gf_pathway(self):
        """Paradigm 7: Visual looming dark disk expansion and Giant Fiber takeoff threshold."""
        looming = LoomingEscapeParadigm(t_collision_s=0.400, r_over_v_s=0.020)
        self.assertEqual(looming.name, "looming_escape")

        # Initial steps: disk is small, GF should not spike
        for _ in range(25):
            res = looming.step({'x': 40.0, 'y': 40.0, 'heading': 0.0}, dt=0.01)

        # Advance until threshold theta >= 65 deg is crossed
        spike_occurred = False
        for _ in range(25):
            res = looming.step({'x': 40.0, 'y': 40.0, 'heading': 0.0}, dt=0.01)
            if res['gf_spike']:
                spike_occurred = True
                break

        self.assertTrue(spike_occurred)
        metrics = looming.get_metrics()
        self.assertTrue(metrics['escape_initiated'])
        # Time to collision at jump should be in [10, 45] ms
        self.assertGreaterEqual(metrics['time_to_collision_at_jump_ms'], 10.0)
        self.assertLessEqual(metrics['time_to_collision_at_jump_ms'], 45.0)

    def test_optomotor_gaze_stabilization_and_efference_copy(self):
        """Paradigm 8: Optomotor gaze stabilization and >80% efference copy shunting during saccades."""
        optomotor = OptomotorParadigm(drum_velocity_deg_s=30.0)
        self.assertEqual(optomotor.name, "optomotor")

        # Involuntary gaze stabilization (normal walking, no saccade)
        # Fly yaw is slow (5 deg/s) -> positive retinal slip -> elevated HS firing
        res_normal = optomotor.step({'x': 45.0, 'y': 45.0, 'angular_velocity': math.radians(5.0), 'is_saccade': False})
        self.assertFalse(res_normal['efference_copy_active'])
        hs_normal = res_normal['hs_firing_rate']
        self.assertGreater(hs_normal, 40.0)

        # Voluntary rapid saccade (is_saccade=True)
        # Efference copy shunt must suppress >= 80% of self-generated retinal slip
        res_saccade = optomotor.step({'x': 45.0, 'y': 45.0, 'angular_velocity': math.radians(150.0), 'is_saccade': True})
        self.assertTrue(res_saccade['efference_copy_active'])
        self.assertLessEqual(abs(res_saccade['effective_slip']), abs(res_saccade['retinal_slip']) * 0.20)

    def test_gap_crossing_reachability_decision(self):
        """Paradigm 9: Gap crossing reachability threshold (3.8mm)."""
        # Surmountable gap (3.0mm < 3.8mm threshold) -> decision CROSS
        gap_small = GapCrossingParadigm(gap_width_mm=3.0)
        res_cross = gap_small.step({'x': 43.5, 'y': 10.0, 'heading': 0.0})
        self.assertEqual(res_cross['decision_outcome'], "CROSS")

        # Step fly to landing track
        gap_small.step({'x': 50.0, 'y': 10.0, 'heading': 0.0})
        self.assertTrue(gap_small.crossing_success)

        # Insurmountable gap (4.5mm > 4.2mm threshold) -> decision ABORT
        gap_large = GapCrossingParadigm(gap_width_mm=4.5)
        res_abort = gap_large.step({'x': 43.5, 'y': 10.0, 'heading': 0.0})
        self.assertEqual(res_abort['decision_outcome'], "ABORT")
        self.assertFalse(gap_large.crossing_success)

    def test_circadian_dam_sleep_wake_metrics(self):
        """Paradigm 10: Circadian DAM locomotor assay and 5-min sleep bout detection."""
        dam = CircadianDAMParadigm(num_tubes=16, photoperiod='LD')
        self.assertEqual(dam.name, "circadian_dam")

        # Beam crossing at x = 32.5mm
        dam.step({'x': 30.0, 'y': 5.0, 'speed': 1.0})
        res_cross = dam.step({'x': 35.0, 'y': 5.0, 'speed': 1.0})
        self.assertTrue(res_cross['beam_crossed'])
        self.assertEqual(res_cross['total_beam_crossings'], 1)

        # Immobility for 6 consecutive steps (1 step ~ 1 min)
        for _ in range(6):
            res_sleep = dam.step({'x': 35.0, 'y': 5.0, 'speed': 0.0})

        # At minute 5 and 6, fly should enter sleep bout
        self.assertTrue(res_sleep['is_sleeping'])
        self.assertGreaterEqual(res_sleep['total_sleep_minutes'], 2.0)
        self.assertEqual(res_sleep['sleep_bouts'], 1)

    def test_courtship_conditioning_suppression(self):
        """Paradigm 11: Courtship conditioning, male wing extension song, and female rejection kicks."""
        # Receptive virgin female: no rejection kicks, male courts close
        court_virgin = CourtshipParadigm(female_type='virgin')
        res_virgin = court_virgin.step({'x': 10.5, 'y': 10.5, 'heading': 0.0})
        self.assertTrue(res_virgin['courtship_active'])
        self.assertGreater(res_virgin['wing_extension_angle_deg'], 20.0)
        self.assertEqual(res_virgin['rejection_kicks'], 0)

        # Mated female: cVA presence and rejection kicks when male approaches very close
        court_mated = CourtshipParadigm(female_type='mated')
        female_x, female_y = court_mated.female_pos
        res_mated = court_mated.step({'x': female_x + 0.5, 'y': female_y + 0.5, 'heading': 0.0})
        self.assertTrue(res_mated['courtship_active'])
        self.assertGreater(res_mated['rejection_kicks'], 0)
        self.assertEqual(res_mated['punishment'], 1.0)
        self.assertGreater(res_mated['stimuli']['cva_concentration'], 0.5)

    def test_corridor_labyrinth_sliding_and_goal(self):
        """Paradigm 12: 12-segment multi-fork labyrinth, sliding collisions, and food chamber goal."""
        labyrinth = LabyrinthParadigm()
        self.assertEqual(labyrinth.name, "labyrinth")
        # 4 outer walls + 12 internal walls = 16 total wall segments
        self.assertEqual(len(labyrinth.walls), 16)

        # Food goal chamber entry
        res_goal = labyrinth.step({'x': 130.0, 'y': 85.0, 'heading': 0.0, 'speed': 1.0})
        self.assertTrue(res_goal['goal_reached'])
        self.assertEqual(res_goal['reward'], 1.0)
        self.assertIsNotNone(res_goal['time_to_goal_ms'])

        metrics = labyrinth.get_metrics()
        self.assertTrue(metrics['goal_reached'])
        self.assertIn('path_tortuosity', metrics)

    def test_experiment_registry_lookup_and_coverage(self):
        """Test ExperimentRegistry factory: registration, catalog listing, and retrieval for all 12 paradigms."""
        catalog = ExperimentRegistry.list_paradigms()
        self.assertEqual(len(catalog), 12)

        expected_paradigms = [
            "t_maze",
            "y_maze",
            "heat_maze",
            "buridan",
            "visual_operant",
            "wind_tunnel",
            "looming_escape",
            "optomotor",
            "gap_crossing",
            "circadian_dam",
            "courtship",
            "labyrinth",
        ]

        for p_name in expected_paradigms:
            self.assertIn(p_name, catalog)
            instance = ExperimentRegistry.get(p_name)
            self.assertIsInstance(instance, ExperimentParadigm)
            self.assertIsNotNone(instance.dimensions)
            self.assertIsNotNone(instance.trial_manager)

            # Test duck-typed step
            step_out = instance.step({'x': 10.0, 'y': 10.0, 'heading': 0.0, 'speed': 1.0})
            self.assertIsInstance(step_out, dict)

            # Test stimuli sampling
            stim_out = instance.sample_stimuli(10.0, 10.0, 0.0)
            self.assertIsInstance(stim_out, dict)

            # Test metrics
            metrics_out = instance.get_metrics()
            self.assertIsInstance(metrics_out, dict)


if __name__ == '__main__':
    unittest.main()
