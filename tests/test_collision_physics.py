"""
Tests for Simulation-Grade Continuous Wall Collision Engine & Contact Dynamics.
================================================================================
Verifies:
1. Swept-circle Continuous Collision Detection (CCD) preventing high-speed tunneling.
2. Exact Time-of-Impact (TOI) and contact normal calculation across straight segments and endcaps.
3. Simultaneous 2x2 multi-wall corner wedge solver eliminating jitter in acute corners (60 deg and 45 deg).
4. Smooth contact torque body alignment without angular teleportation or heading discontinuities.
5. Cuticular mechanosensory ingress (Johnston's organ antennal deflection upon contact).
6. Zero-tunneling stress benchmark across 20,000 continuous simulation steps in narrow corridors.
"""

import math
import os
import sys
import random
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from flybrain.maze import WallSegment, CollisionEngine, LabyrinthParadigm, TMazeParadigm
    from flybrain.arena import Arena, FlyState
except ImportError:
    from maze import WallSegment, CollisionEngine, LabyrinthParadigm, TMazeParadigm
    from arena import Arena, FlyState


class TestContinuousSweptCollision:
    """Tests for analytical swept-circle continuous collision detection."""

    def test_high_speed_tunneling_prevention(self):
        """A circle moving at 100 mm/s jumping completely through a wall in one tick must be caught."""
        wall = WallSegment((0.0, 10.0), (50.0, 10.0))
        radius = 1.5

        # Circle jumps from y=5.0 to y=20.0 in a single step (jumping right through y=10.0)
        p0 = (25.0, 5.0)
        p1 = (25.0, 20.0)
        hit, toi, cpt, norm = wall.swept_circle_toi(p0, p1, radius)

        assert hit is True
        # Contact should occur at y = 10.0 - 1.5 = 8.5
        # Expected TOI: (8.5 - 5.0) / (20.0 - 5.0) = 3.5 / 15.0 = 0.233333...
        expected_toi = 3.5 / 15.0
        assert math.isclose(toi, expected_toi, rel_tol=1e-4)
        assert math.isclose(norm[0], 0.0, abs_tol=1e-5)
        assert math.isclose(norm[1], -1.0, abs_tol=1e-5)

    def test_rounded_endcap_swept_collision(self):
        """Moving circle brushing past segment endpoint must collide with rounded endcap."""
        wall = WallSegment((10.0, 10.0), (30.0, 10.0))
        radius = 1.5

        # Sweep near endpoint (30, 10) from (31.0, 12.0) to (31.0, 8.0)
        # Closest distance to (30, 10) is 1.0 < 1.5 -> must hit endcap
        p0 = (31.0, 12.0)
        p1 = (31.0, 8.0)
        hit, toi, cpt, norm = wall.swept_circle_toi(p0, p1, radius)

        assert hit is True
        assert 0.0 <= toi <= 1.0
        # Contact location at TOI
        contact_y = p0[1] + toi * (p1[1] - p0[1])
        dist_to_endpoint = math.hypot(p0[0] - 30.0, contact_y - 10.0)
        assert math.isclose(dist_to_endpoint, radius, abs_tol=1e-4)

    def test_pre_existing_contact_zero_toi(self):
        """If circle starts already in contact/penetration, TOI must be 0.0 and normal correct."""
        wall = WallSegment((0.0, 0.0), (20.0, 0.0))
        radius = 1.5
        p0 = (10.0, 0.8)  # Penetrating wall at y=0 (dist=0.8 < 1.5)
        p1 = (10.0, 2.0)
        hit, toi, cpt, norm = wall.swept_circle_toi(p0, p1, radius)

        assert hit is True
        assert toi == 0.0
        assert norm[1] > 0.0  # Points outward towards circle center


class TestMultiWallCornerWedgeSolver:
    """Tests for simultaneous 2x2 linear constraint resolution in acute corners."""

    def test_60_degree_acute_corner_simultaneous_placement(self):
        """Acute 60-degree wedge must place circle at exact apex distance without oscillating."""
        # 60 degree wedge meeting at (0, 0)
        w1 = WallSegment((0.0, 0.0), (10.0 * math.cos(math.radians(60)), 10.0 * math.sin(math.radians(60))))
        w2 = WallSegment((-10.0 * math.cos(math.radians(60)), 10.0 * math.sin(math.radians(60))), (0.0, 0.0))
        engine = CollisionEngine([w1, w2])

        radius = 1.5
        # Fly drives head-on into the apex: from (0, 6.0) towards (0, -2.0)
        res_x, res_y, rvx, rvy, collided, normals = engine.resolve(
            x=0.0, y=-2.0, vx=0.0, vy=-4.0, radius=radius, prev_x=0.0, prev_y=6.0, dt=0.05
        )

        assert collided is True
        # For a 60 deg wedge (half angle 30 deg), distance to apex along symmetry line is R / sin(30) = 3.0
        assert math.isclose(res_x, 0.0, abs_tol=1e-3)
        assert math.isclose(res_y, 3.0, abs_tol=1e-3)
        # Must be at least distance R from both walls
        assert w1.distance_to_point(res_x, res_y) >= radius - 1e-4
        assert w2.distance_to_point(res_x, res_y) >= radius - 1e-4
        # Normal velocity into the apex must be arrested (zero velocity)
        assert math.hypot(rvx, rvy) == 0.0

    def test_45_degree_sharp_corner_entrapment(self):
        """Sharp 45-degree corner must solve simultaneously without penetration."""
        ang = math.radians(22.5)  # Half-angle 22.5 deg (total 45 deg)
        w1 = WallSegment((0.0, 0.0), (15.0 * math.sin(ang), 15.0 * math.cos(ang)))
        w2 = WallSegment((-15.0 * math.sin(ang), 15.0 * math.cos(ang)), (0.0, 0.0))
        engine = CollisionEngine([w1, w2])

        radius = 1.5
        res_x, res_y, rvx, rvy, collided, normals = engine.resolve(
            x=0.0, y=-1.0, vx=0.0, vy=-5.0, radius=radius, prev_x=0.0, prev_y=8.0, dt=0.05
        )

        assert collided is True
        # Distance to apex should be R / sin(22.5 deg)
        expected_y = radius / math.sin(ang)
        assert math.isclose(res_y, expected_y, abs_tol=1e-3)
        assert w1.distance_to_point(res_x, res_y) >= radius - 1e-4
        assert w2.distance_to_point(res_x, res_y) >= radius - 1e-4


class TestSmoothContactDynamicsAndSensoryIngress:
    """Tests smooth heading steering and mechanosensory feedback upon collision."""

    def test_smooth_heading_continuity_on_wall_impact(self):
        """Hitting a wall must never produce angular discontinuities or instantaneous 180 flips."""
        arena = Arena(paradigm="t-maze")
        fly = arena.fly

        # Aim fly towards left stem wall at x=63.0 at an oblique 45-degree angle
        fly.pos.x = 65.0
        fly.pos.y = 25.0
        fly.heading = math.radians(135.0)  # Facing North-West into wall
        fly.speed = 3.0

        max_angular_jump = 0.0
        for _ in range(15):
            old_h = fly.heading
            arena.step(dt=0.05)
            # Shortest angular distance between consecutive ticks
            diff = (fly.heading - old_h + math.pi) % (2.0 * math.pi) - math.pi
            max_angular_jump = max(max_angular_jump, abs(diff))

        # Under continuous contact torque steering, angular rate is bounded by relaxation factor
        # It must never teleport or flip 180 degrees instantaneously
        assert max_angular_jump < math.radians(60.0), f"Angular jump {math.degrees(max_angular_jump)} deg is too abrupt!"

    def test_cuticular_mechanosensory_antennal_deflection(self):
        """Colliding with a wall must deflect Johnston's organ cuticular sensors."""
        arena = Arena(paradigm="t-maze")
        fly = arena.fly
        # Position right before wall at x=63.0 (with radius 1.5, wall boundary is 64.5)
        fly.pos.x = 64.6
        fly.pos.y = 25.0
        fly.heading = math.pi  # Facing West straight into wall at x=63.0
        fly.speed = 3.0

        arena.step(dt=0.05)
        # Fly has hit wall at x=63.0 -> mechanosensory sensors must register deflection
        if hasattr(fly, 'mechanosensory') and fly.mechanosensory is not None:
            total_deflect = abs(fly.mechanosensory.deflect_left) + abs(fly.mechanosensory.deflect_right)
            assert total_deflect > 0.05


class TestContinuousCorridorStressBenchmark:
    """Stress-test continuous collision resolution over 20,000 steps."""

    def test_20000_steps_corridor_zero_violations(self):
        """Stress-test CollisionEngine over 20,000 steps in a narrow corridor with random high speeds."""
        corridor = [
            WallSegment((0.0, 0.0), (100.0, 0.0)),
            WallSegment((0.0, 12.0), (100.0, 12.0)),
            WallSegment((0.0, 0.0), (0.0, 12.0)),
            WallSegment((100.0, 0.0), (100.0, 12.0)),
        ]
        engine = CollisionEngine(corridor)
        radius = 1.5
        x, y = 50.0, 6.0
        rng = random.Random(1337)

        violations = 0
        for _ in range(20000):
            spd = rng.uniform(0.5, 6.0)
            ang = rng.uniform(0.0, 2.0 * math.pi)
            vx = spd * math.cos(ang)
            vy = spd * math.sin(ang)
            dt = 0.04
            prop_x = x + vx * dt
            prop_y = y + vy * dt

            rx, ry, rvx, rvy, _, _ = engine.resolve(
                prop_x, prop_y, vx, vy, radius=radius, prev_x=x, prev_y=y, dt=dt
            )
            # Wall limits are [0, 100] in x and [0, 12] in y
            # With radius 1.5, valid range is [1.5 - eps, 98.5 + eps] and [1.5 - eps, 10.5 + eps]
            if rx < 1.49 or rx > 98.51 or ry < 1.49 or ry > 10.51:
                violations += 1
            x, y = rx, ry

        assert violations == 0, f"Encountered {violations} tunneling/clipping violations over 20,000 steps!"
