"""Drosophila Neuroethological Experiment Paradigms and Maze Biophysics.

Implements biophysically grounded, empirical neuroethological experiment paradigms:
1. Geometric primitives: WallSegment, CircularMoat, PeltierGrid, VisualLandmark, MazeZone, CollisionEngine.
2. Abstract base class: ExperimentParadigm with continuous sliding collisions, multi-modal stimuli,
   active zone tracking, and trial metrics.
3. 12 Canonical empirical paradigms:
   - TMazeParadigm: Olfactory Pavlovian conditioning with CS+/CS-, vacuum airflow, and performance index.
   - YMazeParadigm: Spontaneous alternation, 3-arm hexagonal hub, and handedness tracking.
   - HeatMazeParadigm: Drosophila Morris water maze with 55mm circular arena, heated floor, cool refuge, and distal landmarks.
   - BuridanParadigm: 50mm circular platform with water moat, opposing dark stripes, stripe fixation and centrophobism.
   - VisualOperantParadigm: 360-deg drum, operant yaw torque closed-loop rotation, and laser heat conditioning.
   - WindTunnelParadigm: 200x60 laminar wind tunnel, surge-and-cast olfactory plume tracking.
   - LoomingEscapeParadigm: Approaching dark disk, Giant Fiber (GF) spike threshold, and emergency escape jump.
   - OptomotorParadigm: Rotating sinusoidal grating drum, HS cell optic flow, and saccadic efference copy shunting.
   - GapCrossingParadigm: Elevated chasm runway, visual/antennal parallax reachability threshold (3.8mm).
   - CircadianDAMParadigm: Drosophila Activity Monitor, 16 tubes, mid-tube IR beam break, 5-min sleep bouts, 12:12 LD cycle.
   - CourtshipParadigm: Circular chamber, male courtship song/wing extension, female rejection kicks, cVA pheromone suppression.
   - LabyrinthParadigm: 140x100 multi-junction corridor maze with 12 wall segments, 4 junctions, dead ends, sliding physics, and food goal.
4. ExperimentRegistry: Central factory for paradigm lookup, instantiation, and catalog listing.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any, Type, Union, Set
import numpy as np
from collections import deque
import functools
import inspect
from online_metrics import ScalarHistory, PathHistory
from online_metrics import (EVIDENCE_EVENT_CAP, ContactCounter, MetricFault, ObservationLifecycle, _fly_field,
                            event, evidence, pose_data, ratio, record, s_to_us, unsupported, us_to_s)


def _finite_stimulus(value: Any, path: str):
    """Reject a nonfinite sensory measurement before arithmetic can conceal it."""
    try:
        finite = math.isfinite(value)
    except (TypeError, ValueError):
        raise MetricFault(path, f'invalid numeric stimulus at {path}') from None
    if not finite:
        raise MetricFault(path)
    return value


# ==============================================================================
# 1. GEOMETRIC AND BIOPHYSICAL PRIMITIVES
# ==============================================================================

class WallSegment:
    """2D line segment wall with continuous sliding collision resolution.

    Implements:
    - Closest point projection and clamped scalar t in [0, 1].
    - Orthogonal distance to point.
    - Normal vector calculation.
    - Continuous circle-segment collision resolution with Coulomb sliding friction
      (mu = 0.5) and restitution (epsilon = 0.1).
    """

    def __init__(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        friction: float = 0.5,
        restitution: float = 0.1
    ):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.friction = float(friction)
        self.restitution = float(restitution)

        self.dx = self.p2[0] - self.p1[0]
        self.dy = self.p2[1] - self.p1[1]
        self.length_sq = self.dx * self.dx + self.dy * self.dy
        self.length = math.sqrt(self.length_sq)

        if self.length > 1e-12:
            self.ux = self.dx / self.length
            self.uy = self.dy / self.length
            # Intrinsic normal pointing left relative to segment direction
            self.nx = -self.uy
            self.ny = self.ux
        else:
            self.ux, self.uy = 1.0, 0.0
            self.nx, self.ny = 0.0, 1.0

    def get_normal(self) -> Tuple[float, float]:
        """Return unit normal vector of the wall segment."""
        return (self.nx, self.ny)

    def project_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """Project (x, y) onto segment.

        Returns:
            (closest_x, closest_y, t) where t is clamped to [0.0, 1.0].
        """
        if self.length_sq < 1e-12:
            return self.p1[0], self.p1[1], 0.0

        vx = x - self.p1[0]
        vy = y - self.p1[1]
        t = (vx * self.dx + vy * self.dy) / self.length_sq
        t_clamped = max(0.0, min(1.0, t))
        cx = self.p1[0] + t_clamped * self.dx
        cy = self.p1[1] + t_clamped * self.dy
        return cx, cy, t_clamped

    def distance_to_point(self, x: float, y: float) -> float:
        """Compute shortest Euclidean distance from (x, y) to the segment."""
        cx, cy, _ = self.project_point(x, y)
        dx = x - cx
        dy = y - cy
        return math.sqrt(dx * dx + dy * dy)

    def swept_circle_toi(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        radius: float
    ) -> Tuple[bool, float, Tuple[float, float], Tuple[float, float]]:
        """Compute exact Time-of-Impact (TOI in [0, 1]) for a circle swept along p0 -> p1.

        Continuous collision detection tests the swept capsule volume against this segment.
        Returns:
            (hit, toi, contact_point_on_wall, contact_normal_pointing_to_circle)
        """
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        l_sq = vx * vx + vy * vy

        # Check if p0 is already intersecting or in contact
        cx0, cy0, _ = self.project_point(p0[0], p0[1])
        d0 = math.hypot(p0[0] - cx0, p0[1] - cy0)
        if d0 <= radius:
            nx = (p0[0] - cx0) / d0 if d0 > 1e-8 else self.nx
            ny = (p0[1] - cy0) / d0 if d0 > 1e-8 else self.ny
            return True, 0.0, (cx0, cy0), (nx, ny)

        if l_sq < 1e-12:
            return False, 1.0, (0.0, 0.0), (0.0, 0.0)

        candidates = []

        # 1. Straight segment offset boundaries (+radius and -radius)
        s0 = (p0[0] - self.p1[0]) * self.nx + (p0[1] - self.p1[1]) * self.ny
        s1 = (p1[0] - self.p1[0]) * self.nx + (p1[1] - self.p1[1]) * self.ny
        denom = s0 - s1
        if abs(denom) > 1e-12:
            if s0 >= radius and s1 < radius:
                s_cand = (s0 - radius) / denom
                if 0.0 <= s_cand <= 1.0:
                    qx = p0[0] + s_cand * vx
                    qy = p0[1] + s_cand * vy
                    t = ((qx - self.p1[0]) * self.dx + (qy - self.p1[1]) * self.dy) / self.length_sq
                    if 0.0 <= t <= 1.0:
                        candidates.append((s_cand, (self.p1[0] + t * self.dx, self.p1[1] + t * self.dy), (self.nx, self.ny)))
            elif s0 <= -radius and s1 > -radius:
                s_cand = (s0 + radius) / denom
                if 0.0 <= s_cand <= 1.0:
                    qx = p0[0] + s_cand * vx
                    qy = p0[1] + s_cand * vy
                    t = ((qx - self.p1[0]) * self.dx + (qy - self.p1[1]) * self.dy) / self.length_sq
                    if 0.0 <= t <= 1.0:
                        candidates.append((s_cand, (self.p1[0] + t * self.dx, self.p1[1] + t * self.dy), (-self.nx, -self.ny)))

        # 2. Rounded endcaps at p1 and p2
        for W in (self.p1, self.p2):
            rx = p0[0] - W[0]
            ry = p0[1] - W[1]
            A = l_sq
            B = 2.0 * (rx * vx + ry * vy)
            C = rx * rx + ry * ry - radius * radius
            disc = B * B - 4 * A * C
            if disc >= 0 and A > 1e-12:
                s_cand = (-B - math.sqrt(disc)) / (2.0 * A)
                if 0.0 <= s_cand <= 1.0:
                    qx = p0[0] + s_cand * vx
                    qy = p0[1] + s_cand * vy
                    dist = math.hypot(qx - W[0], qy - W[1])
                    nx = (qx - W[0]) / dist if dist > 1e-8 else self.nx
                    ny = (qy - W[1]) / dist if dist > 1e-8 else self.ny
                    candidates.append((s_cand, W, (nx, ny)))

        if candidates:
            candidates.sort(key=lambda c: c[0])
            return True, candidates[0][0], candidates[0][1], candidates[0][2]

        return False, 1.0, (0.0, 0.0), (0.0, 0.0)

    def resolve_circle_collision(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None
    ) -> Tuple[float, float, float, float, bool, Tuple[float, float]]:
        """Resolve continuous circle collision against this wall segment.

        Args:
            x, y: Center coordinates of circle (proposed).
            vx, vy: Incoming velocity vector.
            radius: Radius of colliding circle.
            prev_x, prev_y: Prior valid coordinates before step.

        Returns:
            (new_x, new_y, new_vx, new_vy, collided, (normal_x, normal_y))
        """
        cx, cy, t = self.project_point(x, y)
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)

        p0 = (prev_x if prev_x is not None else (x - vx * 0.02), prev_y if prev_y is not None else (y - vy * 0.02))
        p1 = (x, y)
        hit, toi, cpt, norm = self.swept_circle_toi(p0, p1, radius)

        if not hit and dist >= radius:
            return x, y, vx, vy, False, (0.0, 0.0)

        # Contact normal calculation
        if hit:
            cn_x, cn_y = norm
        elif dist > 1e-8:
            cn_x, cn_y = dx / dist, dy / dist
        else:
            cn_x, cn_y = self.nx, self.ny

        resolved_x = cx + cn_x * radius
        resolved_y = cy + cn_y * radius

        # Velocity resolution: decompose into normal and tangential components
        v_dot_n = vx * cn_x + vy * cn_y

        if v_dot_n < 0.0:
            # Moving towards wall: rebound normal velocity with restitution
            v_normal_mag = -self.restitution * v_dot_n

            # Tangential velocity component
            vt_x = vx - v_dot_n * cn_x
            vt_y = vy - v_dot_n * cn_y

            # Coulomb sliding friction reduces tangential velocity
            friction_factor = max(0.0, 1.0 - self.friction)
            vt_x_new = vt_x * friction_factor
            vt_y_new = vt_y * friction_factor

            # Combine reflected normal and friction-damped tangential velocity
            new_vx = v_normal_mag * cn_x + vt_x_new
            new_vy = v_normal_mag * cn_y + vt_y_new
        else:
            # Moving away from wall: keep velocity unchanged
            new_vx = vx
            new_vy = vy

        return resolved_x, resolved_y, new_vx, new_vy, True, (cn_x, cn_y)


class CircularMoat:
    """Circular boundary detection, platform containment, and water moat deterrence."""

    def __init__(self, center: Tuple[float, float], radius: float):
        self.center = (float(center[0]), float(center[1]))
        self.radius = float(radius)

    def contains(self, x: float, y: float) -> bool:
        """Return True if point (x, y) is on the platform inside the moat."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        return (dx * dx + dy * dy) <= (self.radius * self.radius)

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Signed distance to moat boundary: positive inside platform, negative in water."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return self.radius - dist

    def is_in_water(self, x: float, y: float) -> bool:
        """Return True if fly slipped into the surrounding water moat."""
        return not self.contains(x, y)

    def resolve_containment(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float = 1.5
    ) -> Tuple[float, float, float, float, bool]:
        """Contain circle within circular platform perimeter."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        max_dist = max(0.0, self.radius - radius)

        if dist > max_dist and dist > 1e-8:
            nx = dx / dist
            ny = dy / dist
            clamped_x = self.center[0] + nx * max_dist
            clamped_y = self.center[1] + ny * max_dist

            v_dot_n = vx * nx + vy * ny
            if v_dot_n > 0.0:  # Moving outward
                # Reflect inward with restitution 0.1 and friction 0.5
                vx_new = (vx - 1.1 * v_dot_n * nx) * 0.5
                vy_new = (vy - 1.1 * v_dot_n * ny) * 0.5
                return clamped_x, clamped_y, vx_new, vy_new, True
            return clamped_x, clamped_y, vx, vy, True

        return x, y, vx, vy, False


class PeltierGrid:
    """Continuous 2D temperature field with heated floor and cool target refuge."""

    def __init__(
        self,
        baseline_temp: float = 36.5,
        cool_spot: Tuple[float, float] = (22.0, 18.0),
        cool_radius: float = 9.0,
        cool_temp: float = 24.0,
        gradient_sigma: float = 8.0
    ):
        self.baseline_temp = float(baseline_temp)
        self.cool_spot = (float(cool_spot[0]), float(cool_spot[1]))
        self.cool_radius = float(cool_radius)
        self.cool_temp = float(cool_temp)
        self.gradient_sigma = float(gradient_sigma)

    def get_temperature(self, x: float, y: float) -> float:
        """Return temperature in Celsius at (x, y).

        Returns cool_temp within cool_radius, transitioning smoothly via a Gaussian
        radial gradient to baseline_temp across the arena floor.
        """
        dx = x - self.cool_spot[0]
        dy = y - self.cool_spot[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= self.cool_radius:
            return self.cool_temp

        excess = dist - self.cool_radius
        delta_t = self.baseline_temp - self.cool_temp
        gaussian_factor = math.exp(-(excess * excess) / (2.0 * self.gradient_sigma * self.gradient_sigma))
        t = self.baseline_temp - delta_t * gaussian_factor
        return float(np.clip(t, self.cool_temp, self.baseline_temp))


class VisualLandmark:
    """Distal or proximal high-contrast visual landmark cue."""

    def __init__(
        self,
        landmark_id: str,
        azimuth_rad: float,
        angular_width_rad: float = 0.21,
        glyph: str = 'stripe',
        pos: Optional[Tuple[float, float]] = None
    ):
        self.landmark_id = landmark_id
        self.azimuth_rad = float(azimuth_rad)
        self.angular_width_rad = float(angular_width_rad)
        self.glyph = glyph
        self.pos = (float(pos[0]), float(pos[1])) if pos is not None else None

    def get_apparent_bearing(self, fly_x: float, fly_y: float, fly_heading: float) -> float:
        """Calculate apparent egocentric bearing to landmark relative to fly heading.

        Returns bearing in radians normalized to [-pi, +pi].
        """
        if self.pos is not None:
            dx = self.pos[0] - fly_x
            dy = self.pos[1] - fly_y
            target_angle = math.atan2(dy, dx)
        else:
            target_angle = self.azimuth_rad

        rel_angle = (target_angle - fly_heading + math.pi) % (2.0 * math.pi) - math.pi
        return float(rel_angle)


class MazeZone:
    """Spatial region of interest triggering rewards, punishments, or state transitions."""

    def __init__(
        self,
        name: str,
        zone_type: str,
        bounds: Any,
        reward: float = 0.0,
        punishment: float = 0.0
    ):
        self.name = name
        self.zone_type = zone_type
        self.bounds = bounds
        self.reward = float(reward)
        self.punishment = float(punishment)

    def contains(self, x: float, y: float) -> bool:
        """Check if point (x, y) lies inside the zone boundary."""
        if isinstance(self.bounds, (tuple, list)):
            # Check if elements are tuples/lists (polygon vertices)
            if len(self.bounds) > 0 and isinstance(self.bounds[0], (tuple, list)):
                poly = self.bounds
                inside = False
                n = len(poly)
                p1x, p1y = poly[0]
                for i in range(1, n + 1):
                    p2x, p2y = poly[i % n]
                    if y > min(p1y, p2y) and y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                    p1x, p1y = p2x, p2y
                return inside
            elif len(self.bounds) == 3:
                # Circular zone: (center_x, center_y, radius)
                cx, cy, r = self.bounds
                dx = x - cx
                dy = y - cy
                return (dx * dx + dy * dy) <= (r * r)
            elif len(self.bounds) == 4:
                # Rectangular zone: (xmin, ymin, xmax, ymax)
                xmin, ymin, xmax, ymax = self.bounds
                return (xmin <= x <= xmax) and (ymin <= y <= ymax)
        elif isinstance(self.bounds, dict):
            if 'radius' in self.bounds:
                cx = self.bounds.get('x', 0.0)
                cy = self.bounds.get('y', 0.0)
                r = self.bounds['radius']
                dx = x - cx
                dy = y - cy
                return (dx * dx + dy * dy) <= (r * r)
            elif 'x_min' in self.bounds:
                return (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                        self.bounds['y_min'] <= y <= self.bounds['y_max'])
        return False


class CollisionEngine:
    """Continuous collision detection and sliding physics engine for multi-segment mazes."""

    def __init__(
        self,
        walls: Optional[List[WallSegment]] = None,
        restitution: float = 0.0,
        friction: float = 0.4
    ):
        self.walls: List[WallSegment] = list(walls) if walls is not None else []
        self.restitution = float(restitution)
        self.friction = float(friction)

    def add_wall(self, wall: WallSegment):
        self.walls.append(wall)

    def add_segment(self, p1: Tuple[float, float], p2: Tuple[float, float], friction: float = 0.5, restitution: float = 0.1):
        self.walls.append(WallSegment(p1, p2, friction, restitution))

    def clear(self):
        self.walls.clear()

    def resolve(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float = 1.5,
        max_iterations: int = 6,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None,
        dt: float = 0.02
    ) -> Tuple[float, float, float, float, bool, List[Tuple[float, float]]]:
        """Continuously resolve circle-wall collisions with zero tunneling across multiple walls.

        Features:
        1. Swept-circle Continuous Collision Detection (CCD) preventing high-speed tunneling.
        2. Simultaneous 2x2 multi-wall corner wedge solver eliminating jitter and ping-ponging.
        3. Biomechanical sliding physics with Coulomb friction and 0-restitution contact.
        4. Residual timestep sliding integration.

        Returns:
            (resolved_x, resolved_y, resolved_vx, resolved_vy, collided_any, contact_normals)
        """
        p0 = (prev_x if prev_x is not None else (x - vx * dt), prev_y if prev_y is not None else (y - vy * dt))
        p1 = (x, y)

        earliest_toi = 1.0
        hit_any = False
        collided_walls = []
        for wall in self.walls:
            hit, toi, cpt, norm = wall.swept_circle_toi(p0, p1, radius)
            if hit:
                hit_any = True
                if toi < earliest_toi:
                    earliest_toi = toi

        # Check static penetration at p1 if no swept hit
        if not hit_any:
            for wall in self.walls:
                if wall.distance_to_point(p1[0], p1[1]) < radius:
                    hit_any = True
                    earliest_toi = 0.0
                    break

        if not hit_any:
            return x, y, vx, vy, False, []

        # Advance to earliest TOI contact location
        s = max(0.0, min(1.0, earliest_toi))
        cx = p0[0] + s * (p1[0] - p0[0])
        cy = p0[1] + s * (p1[1] - p0[1])

        # Gather all active contacts at (cx, cy)
        contacts = []
        for wall in self.walls:
            proj_x, proj_y, _ = wall.project_point(cx, cy)
            d = math.hypot(cx - proj_x, cy - proj_y)
            if d <= radius + 0.05:
                if d > 1e-8:
                    nx = (cx - proj_x) / d
                    ny = (cy - proj_y) / d
                else:
                    nx, ny = wall.nx, wall.ny
                contacts.append((wall, (proj_x, proj_y), (nx, ny), d))

        if not contacts:
            # The overlap was found at p1 but nothing touches the start pose: never
            # return the overlapping pose itself, push it out of the walls instead.
            fx, fy = self._project_out(x, y, radius, self.walls)
            return fx, fy, vx, vy, True, []

        # Position resolution: push the body out of only those walls it actually
        # overlaps, by iterated projection. The former 2x2 wedge solve placed the body
        # on the intersection of both offset lines whenever a second wall was merely
        # inside the 0.05 mm contact band, snapping it up to ~0.2 mm along a wall
        # (further at shallow polygon vertices) in a single step: an unexplained jump.
        res_x, res_y = self._project_out(cx, cy, radius, [c[0] for c in contacts])

        # Velocity constraints come from walls actually touched at the resolved pose;
        # a wall still 0.05 mm away must not zero the motion toward it.
        touching = []
        for wall, _cp, _n, _d in contacts:
            proj_x, proj_y, _ = wall.project_point(res_x, res_y)
            d = math.hypot(res_x - proj_x, res_y - proj_y)
            if d <= radius + self.CONTACT_SKIN:
                if d > 1e-8:
                    n = ((res_x - proj_x) / d, (res_y - proj_y) / d)
                else:
                    n = (wall.nx, wall.ny)
                touching.append((wall, (proj_x, proj_y), n, d))
        if touching:
            contacts = touching
        best_pair = (contacts[0], contacts[0])
        if len(contacts) > 1:
            best_pair = (contacts[0], contacts[1])
            max_cross = abs(contacts[0][2][0] * contacts[1][2][1] - contacts[0][2][1] * contacts[1][2][0])
            for i in range(len(contacts)):
                for j in range(i + 1, len(contacts)):
                    cross = abs(contacts[i][2][0] * contacts[j][2][1] - contacts[i][2][1] * contacts[j][2][0])
                    if cross > max_cross:
                        max_cross = cross
                        best_pair = (contacts[i], contacts[j])

        # Velocity resolution
        if len(contacts) == 1:
            n1 = contacts[0][2]
            vd = vx * n1[0] + vy * n1[1]
            if vd < 0.0:
                vt_x = vx - vd * n1[0]
                vt_y = vy - vd * n1[1]
                fric = max(0.0, 1.0 - self.friction)
                res_vx = -self.restitution * vd * n1[0] + vt_x * fric
                res_vy = -self.restitution * vd * n1[1] + vt_y * fric
            else:
                res_vx, res_vy = vx, vy
        else:
            n1, n2 = best_pair[0][2], best_pair[1][2]
            vd1 = vx * n1[0] + vy * n1[1]
            vd2 = vx * n2[0] + vy * n2[1]
            if vd1 <= 0.0 and vd2 <= 0.0:
                res_vx, res_vy = 0.0, 0.0
            elif vd1 < 0.0:
                vt_x = (vx - vd1 * n1[0]) * max(0.0, 1.0 - self.friction)
                vt_y = (vy - vd1 * n1[1]) * max(0.0, 1.0 - self.friction)
                res_vx = vt_x if (vt_x * n2[0] + vt_y * n2[1] >= -1e-5) else 0.0
                res_vy = vt_y if (vt_x * n2[0] + vt_y * n2[1] >= -1e-5) else 0.0
            elif vd2 < 0.0:
                vt_x = (vx - vd2 * n2[0]) * max(0.0, 1.0 - self.friction)
                vt_y = (vy - vd2 * n2[1]) * max(0.0, 1.0 - self.friction)
                res_vx = vt_x if (vt_x * n1[0] + vt_y * n1[1] >= -1e-5) else 0.0
                res_vy = vt_y if (vt_x * n1[0] + vt_y * n1[1] >= -1e-5) else 0.0
            else:
                res_vx, res_vy = vx, vy

        # Residual timestep sliding integration
        rem_dt = (1.0 - s) * dt
        if rem_dt > 1e-5 and math.hypot(res_vx, res_vy) > 1e-5:
            slide_x = res_x + res_vx * rem_dt
            slide_y = res_y + res_vy * rem_dt
            # Keep the slide out of every wall, including one first reached while sliding
            res_x, res_y = self._project_out(slide_x, slide_y, radius, self.walls)

        return res_x, res_y, res_vx, res_vy, True, [c[2] for c in contacts]

    CONTACT_SKIN = 1e-3   # mm: a wall this close to the body edge is being touched

    @staticmethod
    def _project_out(x: float, y: float, radius: float, walls, iterations: int = 8) -> Tuple[float, float]:
        """Move (x, y) the minimum distance out of every overlapped wall (iterated projection)."""
        target = radius + 1e-4
        for _ in range(iterations):
            moved = False
            for wall in walls:
                cp_x, cp_y, _ = wall.project_point(x, y)
                dx, dy = x - cp_x, y - cp_y
                d = math.hypot(dx, dy)
                if d < target - 1e-12:
                    nx, ny = (dx / d, dy / d) if d > 1e-8 else (wall.nx, wall.ny)
                    x, y = cp_x + nx * target, cp_y + ny * target
                    moved = True
            if not moved:
                break
        return x, y


class TrialManager:
    """Manages experiment trial lifecycle, timing, and step progression."""

    def __init__(self, max_duration_steps: int = 1000):
        self.trial_number: int = 1
        self.current_step: int = 0
        self.max_duration_steps: int = max_duration_steps
        self.trial_active: bool = True
        self.history: List[Dict[str, Any]] = []

    def step(self) -> bool:
        """Advance one trial step. Returns True if trial continues, False if completed."""
        self.current_step += 1
        if self.current_step >= self.max_duration_steps:
            self.trial_active = False
            return False
        return True

    def reset(self):
        """Reset for the next trial."""
        self.trial_number += 1
        self.current_step = 0
        self.trial_active = True


# ==============================================================================
# 2. ABSTRACT BASE CLASS: EXPERIMENT PARADIGM
# ==============================================================================

class ExperimentParadigm(ObservationLifecycle, ABC):
    """Abstract Base Class for all Drosophila Neuroethological Experiment Paradigms.

    Metric contract v1: every concrete ``step`` also feeds the pre-motor sample to the
    paradigm's v1 observer, and every ``reset_trial`` starts a new v1 segment. The legacy
    step bodies and ``get_metrics()`` are untouched (contract §8).
    """

    _v1_legacy_out = None
    _v1_last_pose = None
    _v1_fault = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        legacy_step = cls.__dict__.get('step')
        if legacy_step is not None and not getattr(legacy_step, '_v1_wrapped', False):
            signature = inspect.signature(legacy_step)

            @functools.wraps(legacy_step)
            def step(self, *args, **kw):
                if self.OBSERVATION_SPEC is None:
                    # A paradigm with no v1 spec (a test fixture or an unknown assay): the legacy
                    # step runs unchanged and nothing is observed. The daemon must refuse to
                    # activate it as an assay (v1 §5.6); observation_spec() raises.
                    return legacy_step(self, *args, **kw)
                bound = signature.bind(self, *args, **kw)
                bound.apply_defaults()
                fly, dt = bound.arguments['fly'], bound.arguments['dt']
                pose = self._extract_fly_pose(fly)
                out = legacy_step(self, *args, **kw)
                self._v1_legacy_out = out
                try:
                    self._v1_sample(fly, dt, pose)
                except MetricFault as fault:
                    # Surfaced when the records are read (the daemon's F policy); the
                    # legacy step result is still returned unchanged.
                    self._v1_fault = fault
                return out

            step._v1_wrapped = True
            cls.step = step
        legacy_reset = cls.__dict__.get('reset_trial')
        if legacy_reset is not None and not getattr(legacy_reset, '_v1_wrapped', False):
            @functools.wraps(legacy_reset)
            def reset_trial(self, *args, config=None, **kw):
                out = legacy_reset(self, *args, **kw)
                if self.OBSERVATION_SPEC is not None:
                    self._v1_reset_segment(config)  # I-5 with the v1.1 ObservationConfig
                return out

            reset_trial._v1_wrapped = True
            cls.reset_trial = reset_trial

    def _v1_reset_segment(self, config=None):
        self._v1_fault = None
        super()._v1_reset_segment(config)

    def _v1_on_new_observation_segment(self):
        self._v1_fault = None

    def _v1_ready(self):
        if self.OBSERVATION_SPEC is None:
            raise MissingObservationSpec(f'assay {getattr(self, "name", type(self).__name__)} has no observation spec')
        super()._v1_ready()

    def observation_spec(self):
        self._v1_ready()
        return super().observation_spec()

    def get_metric_records(self):
        if self._v1_fault is not None:
            raise self._v1_fault
        return super().get_metric_records()

    def __init__(
        self,
        name: str,
        description: str,
        dimensions: Tuple[float, float],
        walls: Optional[List[WallSegment]] = None,
        zones: Optional[List[MazeZone]] = None,
        landmarks: Optional[List[VisualLandmark]] = None,
        max_duration_steps: int = 1000
    ):
        self.name = name
        self.description = description
        self.dimensions = (float(dimensions[0]), float(dimensions[1]))
        self.walls = walls or []
        self.zones = zones or []
        self.landmarks = landmarks or []
        self.collision_engine = CollisionEngine(self.walls)
        self.trial_manager = TrialManager(max_duration_steps=max_duration_steps)
        self.time_elapsed_ms: float = 0.0

    @abstractmethod
    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        """Execute one simulation tick of the experiment paradigm.

        Args:
            fly: FlyState object or duck-typed fly state dict.
            dt: Simulation time delta in seconds or steps.

        Returns:
            Dictionary containing sensory stimuli, active zones, rewards, punishments,
            and paradigm-specific telemetry.
        """
        pass

    def check_collisions(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        radius: float = 1.5,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None,
        dt: float = 0.02
    ) -> Tuple[float, float, float, float, bool]:
        """Check and resolve circle collisions against paradigm walls with sliding physics."""
        new_x, new_y, new_vx, new_vy, collided, _ = self.collision_engine.resolve(
            x, y, vx, vy, radius=radius, prev_x=prev_x, prev_y=prev_y, dt=dt
        )
        return new_x, new_y, new_vx, new_vy, collided

    def check_collisions_advanced(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        radius: float = 1.5,
        prev_x: Optional[float] = None,
        prev_y: Optional[float] = None,
        dt: float = 0.02
    ) -> Tuple[float, float, float, float, bool, List[Tuple[float, float]]]:
        """Simulation-grade collision check returning contact normals for smooth torque steering."""
        return self.collision_engine.resolve(
            x, y, vx, vy, radius=radius, prev_x=prev_x, prev_y=prev_y, dt=dt
        )

    def _normalize_stimuli_args(
        self,
        x: Any,
        y: Optional[float] = None,
        heading: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """Normalize stimuli query args to support both (pos_tuple, heading) and (x, y, heading)."""
        if isinstance(x, (tuple, list)):
            return float(x[0]), float(x[1]), float(y if y is not None else 0.0)
        elif hasattr(x, 'x') and hasattr(x, 'y'):
            return float(x.x), float(x.y), float(y if y is not None else 0.0)
        return float(x), float(y if y is not None else 0.0), float(heading if heading is not None else 0.0)

    @abstractmethod
    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        """Sample local environmental and multi-modal stimuli at fly's pose (x, y, heading)."""
        pass

    def get_active_zones(self, x: float, y: float) -> List[MazeZone]:
        """Return all zones containing (x, y)."""
        return [z for z in self.zones if z.contains(x, y)]

    @abstractmethod
    def reset_trial(self) -> Dict[str, Any]:
        """Reset trial state for a new experimental run."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Return quantitative neuroethological behavioral metrics."""
        pass

    def _extract_fly_pose(self, fly: Any) -> Tuple[float, float, float, float, float]:
        """Helper to extract (x, y, heading, speed, angular_vel) from fly object or dict."""
        if fly is None:
            return 0.0, 0.0, 0.0, 1.2, 0.0

        if hasattr(fly, 'pos'):
            x, y = float(fly.pos.x), float(fly.pos.y)
        elif isinstance(fly, dict):
            x = float(fly.get('x', fly.get('fly_x', 0.0)))
            y = float(fly.get('y', fly.get('fly_y', 0.0)))
        else:
            x = float(getattr(fly, 'x', 0.0))
            y = float(getattr(fly, 'y', 0.0))

        heading = float(getattr(fly, 'heading', fly.get('heading', 0.0) if isinstance(fly, dict) else 0.0))
        speed = float(getattr(fly, 'speed', fly.get('speed', 1.2) if isinstance(fly, dict) else 1.2))
        angular_vel = float(getattr(fly, 'angular_velocity', fly.get('angular_velocity', 0.0) if isinstance(fly, dict) else 0.0))

        return x, y, heading, speed, angular_vel


# ==============================================================================
# 2b. METRIC CONTRACT v1.2 PRODUCERS (metric-contract/1.2; docs/ASSAY_SEMANTICS.md)
#
# Each concrete paradigm inherits one observer below. The observer receives the same
# pre-motor sample as the legacy step and keeps its own presentation-scoped
# accumulators, so the legacy get_metrics() view is unchanged (contract §8, C0).
# ==============================================================================

WALKING_SPEED_MM_S = 0.5            # §9.2: walking |speed| > 0.5 mm/s; immobile <= 0.5 mm/s
YMAZE_REARM_RADIUS_MM = 12.0        # §9.3 y_maze: hub radius 10 mm + 2 mm hysteresis
LABYRINTH_REARM_MM = 1.0            # §9.3 labyrinth: re-arm >= 1.0 mm outside the dead end
GAP_TURN_COMPLETE_RAD = 0.4         # mirrors the gap ABORT controller's `abs(error) < .4` (assay_response.py)
DAM_BOUT_US = 300 * 1_000_000       # 300 s immobility criterion, never compressed
BURIDAN_WALK_MIN_US = 1_000_000     # 1 s walking eligibility (engineering threshold)


def _spec(assay, mode, window_s, *, headline, companions, hold_s=0.0, terminal_events=(),
          triggers=(), curves=(), required_states=None, **extra):
    spec = {
        'assay': assay, 'spec_version': f'{assay}/1.2', 'mode': mode, 'window_s': window_s,
        'hold_s': float(hold_s), 'terminal_events': list(terminal_events),
        're_presentation_triggers': list(triggers), 'headline_metric': headline,
        'companion_metrics': list(companions),
        'curve_metrics': [{'name': n, 'curve_label': label} for n, label in curves],
        'learning_claim': 'none', 'required_states': dict(required_states or {}),
    }
    spec.update(extra)
    return spec


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class MissingObservationSpec(LookupError):
    """A paradigm without a metric-contract spec; v1 §5.6: the daemon must refuse it visibly."""


def _require_finite_yaw(yaw):
    if not math.isfinite(yaw):
        raise MetricFault('yaw_torque')


def _cap(seq, item):
    if len(seq) < EVIDENCE_EVENT_CAP:
        seq.append(item)


class _V1Paths:
    """Path length and net displacement from successive pre-motor samples."""

    def _v1_path_clear(self):
        self._p_first = None
        self._p_last = None
        self._p_len = 0.0

    def _v1_path(self, s):
        xy = (s.x, s.y)
        if self._p_first is None:
            self._p_first = xy
        else:
            self._p_len += math.dist(self._p_last, xy)
        self._p_last = xy

    def _v1_path_records(self, iv, *, path_name='path_length_mm', net_name=None, net_note=None):
        if self._p_first is None:
            out = {path_name: record(kind='kinematic', unit='mm', reason='not_observed', interval_rel_s=iv)}
            if net_name:
                out[net_name] = record(kind='kinematic', unit='mm', reason='not_observed',
                                       note=net_note, interval_rel_s=iv)
            return out
        out = {path_name: record(self._p_len, kind='kinematic', unit='mm', interval_rel_s=iv)}
        if net_name:
            out[net_name] = record(math.dist(self._p_first, self._p_last), kind='kinematic', unit='mm',
                                   note=net_note, interval_rel_s=iv)
        return out

    def _v1_duration(self, us, iv, note=None, samples=None):
        observed = self._v1_observed_samples() if samples is None else samples
        if observed == 0:
            return record(kind='duration', unit='s', reason='not_observed', note=note, interval_rel_s=iv)
        return record(us_to_s(us), kind='duration', unit='s', note=note, interval_rel_s=iv)

    def _v1_gated_duration(self, name, us, iv):
        missing = self._v1_state_gate(self.OBSERVATION_SPEC['required_states'][name])
        if missing:
            return unsupported('duration', 's', f'controller does not emit {missing}', iv)
        return self._v1_duration(us, iv)


class _TMazeV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        't_maze', 'fixed', 120.0, triggers=['reverse_arms'], headline='first_choice',
        companions=['first_choice_latency_s', 'cs_plus_entries', 'cs_minus_entries', 'arm_entry_preference_index'],
        curves=[('arm_entry_preference_index', 'measured arm-entry preference index per completed observation')])
    PI_NOTE = 'arm-entry preference index over repeated visits; not a conditioning PI or evidence of learning'

    def _v1_clear(self):
        self._tm = {'cs_plus': 0, 'cs_minus': 0, 'last_arm': None, 'first': None, 'first_arm': None,
                    'first_us': None, 'first_pose': None, 'entries': []}

    def _v1_carry(self):
        # A fly still inside an arm at a reversal has not made a new entry (physical hysteresis).
        return self._tm['last_arm']

    def _v1_restore(self, carry):
        self._tm['last_arm'] = carry

    def _v1_segment_carry(self):
        return self._tm['last_arm']

    def _v1_segment_restore(self, carry):
        self._tm['last_arm'] = carry

    def _v1_observe(self, s):
        tm = self._tm
        zones = [z.name for z in self.get_active_zones(s.x, s.y)]
        # Entry and re-arm rule of the legacy counter (maze.py TMazeParadigm.step).
        arm = 'arm_a' if 'arm_a' in zones and s.x < 45 else ('arm_b' if 'arm_b' in zones and s.x > 95 else None)
        if arm and arm != tm['last_arm']:
            identity = 'cs_plus' if arm == self.cs_plus_arm else 'cs_minus'
            tm[identity] += 1
            if tm['first'] is None:
                tm['first'], tm['first_arm'], tm['first_us'] = identity, arm, s.t_us
                tm['first_pose'] = pose_data(s.x, s.y, s.heading, s.t_rel_s, s.step)
            _cap(tm['entries'], event('arm_entry', s.t_rel_s, s.step, arm=arm, identity=identity))
            tm['last_arm'] = arm
        elif 63 <= s.x <= 77:
            tm['last_arm'] = None

    def _v1_records(self, cut):
        tm, iv = self._tm, self._v1_interval()
        counts = {'cs_plus': tm['cs_plus'], 'cs_minus': tm['cs_minus']}
        if self._v1_observed_samples() == 0:
            index = record(kind='index', unit='index', reason='not_observed', counts=counts,
                           numerator=0, denominator=0, note=self.PI_NOTE, interval_rel_s=iv)
            plus = record(kind='count', unit='count', reason='not_observed', interval_rel_s=iv)
            minus = record(kind='count', unit='count', reason='not_observed', interval_rel_s=iv)
        else:
            index = ratio(tm['cs_plus'] - tm['cs_minus'], tm['cs_plus'] + tm['cs_minus'], kind='index',
                          unit='index', counts=counts, note=self.PI_NOTE, interval_rel_s=iv)
            plus = record(tm['cs_plus'], kind='count', unit='count', interval_rel_s=iv, evidence_ref='entries')
            minus = record(tm['cs_minus'], kind='count', unit='count', interval_rel_s=iv, evidence_ref='entries')
        if tm['first'] is not None:
            first = record(tm['first'], kind='label', unit='label', counts={'arm': tm['first_arm']},
                           interval_rel_s=iv, evidence_ref='first_choice_pose')
        else:
            first = self._v1_pending_value('label', 'label')
        return {'arm_entry_preference_index': index, 'cs_plus_entries': plus, 'cs_minus_entries': minus,
                'first_choice': first, 'first_choice_latency_s': self._v1_latency(self._tm['first_us'])}

    def _v1_evidence(self):
        ev = {'entries': evidence('event_sequence', self._tm['entries'])}
        if self._tm['first_pose']:
            ev['first_choice_pose'] = evidence('pose', self._tm['first_pose'])
        return ev


class _YMazeV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'y_maze', 'fixed', 300.0, headline='spontaneous_alternation_rate',
        companions=['physical_entries', 'collapsed_length', 'handedness_index'],
        curves=[('spontaneous_alternation_rate', 'measured alternation rate of the collapsed arm sequence per completed observation')])
    SYMBOLS = ('A', 'B', 'C')

    def _v1_clear_segment(self):
        self._ym = {'armed': True, 'physical': [], 'collapsed': []}

    def _v1_clear(self):
        pass

    def _v1_carry(self):
        return self._ym['armed']

    def _v1_restore(self, carry):
        self._ym['armed'] = carry

    def _v1_segment_carry(self):
        return self._ym['armed']

    def _v1_segment_restore(self, carry):
        self._ym['armed'] = carry

    def _v1_observe(self, s):
        ym = self._ym
        cx, cy = self.center
        if math.hypot(s.x - cx, s.y - cy) <= YMAZE_REARM_RADIUS_MM:
            ym['armed'] = True
            return
        if not ym['armed']:
            return
        zones = {z.name for z in self.get_active_zones(s.x, s.y)}
        for idx in range(3):
            if f'arm_{idx}' in zones:
                ym['physical'].append(idx)
                if not ym['collapsed'] or ym['collapsed'][-1] != idx:
                    ym['collapsed'].append(idx)
                ym['armed'] = False
                return

    def _v1_records(self, cut):
        ym, iv = self._ym, self._v1_seg_interval()
        seq = ym['collapsed']
        triads = max(0, len(seq) - 2)
        alternating = sum(1 for i in range(triads) if len(set(seq[i:i + 3])) == 3)
        left = sum(1 for a, b in zip(seq, seq[1:]) if (b - a) % 3 == 1)
        right = sum(1 for a, b in zip(seq, seq[1:]) if (b - a) % 3 == 2)
        counts = {'collapsed_length': len(seq), 'physical_entries': len(ym['physical'])}
        if self._v1_segment_samples() == 0:
            na = lambda kind, unit: record(kind=kind, unit=unit, reason='not_observed', counts=counts, interval_rel_s=iv)
            return {'physical_entries': na('count', 'count'), 'spontaneous_alternation_rate': na('ratio', 'ratio'),
                    'handedness_index': na('index', 'index'), 'left_turns': na('count', 'count'),
                    'right_turns': na('count', 'count'), 'total_triads': na('count', 'count'),
                    'alternating_triads': na('count', 'count')}
        sar = ratio(alternating, triads, counts=counts, interval_rel_s=iv,
                    reason_if_zero='insufficient_events',
                    note='alternation over the consecutive-repeat-collapsed arm sequence')
        cnt = lambda v: record(v, kind='count', unit='count', interval_rel_s=iv)
        return {
            'physical_entries': record(len(ym['physical']), kind='count', unit='count', counts=counts,
                                       interval_rel_s=iv, evidence_ref='physical_visit_sequence'),
            'spontaneous_alternation_rate': sar,
            'handedness_index': ratio(right - left, right + left, kind='index', unit='index',
                                      counts={'left_turns': left, 'right_turns': right}, interval_rel_s=iv),
            'left_turns': cnt(left), 'right_turns': cnt(right),
            'total_triads': cnt(triads), 'alternating_triads': cnt(alternating),
        }

    def _v1_evidence(self):
        sym = self.SYMBOLS
        return {
            'physical_visit_sequence': evidence('sequence', {'symbols': [sym[i] for i in self._ym['physical']],
                                                             'collapse': 'none'}),
            'alternation_sequence': evidence('sequence', {'symbols': [sym[i] for i in self._ym['collapsed']],
                                                          'collapse': 'consecutive_repeats'}),
        }


class _HeatMazeV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'heat_maze', 'until_terminal', 300.0, hold_s=2.0, terminal_events=['refuge_entry'],
        triggers=['floorTemp'], headline='escape_latency_s',
        companions=['thermal_dose_degC_s', 'path_length_mm', 'refuge_reached'])

    def _v1_window_params(self):
        return {'floor_temp_degC': float(self.peltier.baseline_temp)}

    def _v1_clear(self):
        self._v1_path_clear()
        self._hm = {'obs_us': 0, 'dose': 0.0, 'target_us': 0, 'refuge_us': None, 'refuge_pose': None}

    def _v1_observe(self, s):
        hm = self._hm
        self._v1_path(s)
        zones = {z.name for z in self.get_active_zones(s.x, s.y)}
        if 'refuge' in zones and hm['refuge_us'] is None:
            hm['refuge_us'] = s.t_us
            hm['refuge_pose'] = pose_data(s.x, s.y, s.heading, s.t_rel_s, s.step)
            if self._v1_terminal('refuge_entry', s):
                return
        temp = _finite_stimulus(self.peltier.get_temperature(s.x, s.y),
                                'heat_maze.stimuli.temperature')
        hm['dose'] += max(0.0, temp - 25.0) * s.observation_dt
        hm['obs_us'] += s.observation_dt_us
        if 'target_quadrant' in zones:
            hm['target_us'] += s.observation_dt_us

    def _v1_records(self, cut):
        hm, iv = self._hm, self._v1_interval()
        out = {
            'escape_latency_s': self._v1_latency(hm['refuge_us']),
            'refuge_reached': self._v1_outcome(hm['refuge_us'] is not None),
            'target_quadrant_fraction': ratio(hm['target_us'], hm['obs_us'], kind='fraction', unit='fraction',
                                              reason_if_zero='not_observed', interval_rel_s=iv),
            'place_learning': unsupported('index', 'index', 'no place-memory mechanism is implemented', iv),
        }
        if self._v1_observed_samples() == 0:
            out['thermal_dose_degC_s'] = record(kind='integral', unit='degC*s', reason='not_observed', interval_rel_s=iv)
        else:
            out['thermal_dose_degC_s'] = record(hm['dose'], kind='integral', unit='degC*s', interval_rel_s=iv,
                                                note='sum of max(0, T - 25 degC) * dt at pre-motor samples')
        out.update(self._v1_path_records(iv))
        return out

    def _v1_evidence(self):
        return {'refuge_entry_pose': evidence('pose', self._hm['refuge_pose'])} if self._hm['refuge_pose'] else {}


class _BuridanV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'buridan', 'fixed', 300.0, triggers=['rotate_stripes', 'contrast'], headline='walking_stripe_alignment',
        companions=['walking_s', 'displacement_mm', 'heading_alignment', 'centre_fraction'],
        curves=[('walking_stripe_alignment', 'measured walking stripe alignment per completed observation')])
    ALIGN_NOTE = 'alignment, not navigation success'

    def _v1_clear(self):
        self._v1_path_clear()
        self._bu = {'obs_us': 0, 'walk_us': 0, 'align': 0.0, 'walk_align': 0.0, 'centre_us': 0}

    def _v1_observe(self, s):
        bu = self._bu
        self._v1_path(s)
        fixation = self.sample_stimuli(s.x, s.y, s.heading)['stripe_fixation']
        bu['obs_us'] += s.observation_dt_us
        bu['align'] += fixation * s.observation_dt_us
        if abs(s.speed) > WALKING_SPEED_MM_S:
            bu['walk_us'] += s.observation_dt_us
            bu['walk_align'] += fixation * s.observation_dt_us
        if math.hypot(s.x - self.center[0], s.y - self.center[1]) < 25.0:
            bu['centre_us'] += s.observation_dt_us

    def _v1_records(self, cut):
        bu, iv = self._bu, self._v1_interval()
        observed = bu['obs_us'] > 0
        na = lambda kind, unit, note=None: record(kind=kind, unit=unit, reason='not_observed', note=note, interval_rel_s=iv)
        if not observed:
            align = na('mean', 'index', self.ALIGN_NOTE)
            walk = na('mean', 'index', self.ALIGN_NOTE)
            centre = na('fraction', 'fraction')
            phobism = na('fraction', 'fraction')
        else:
            align = record(bu['align'] / bu['obs_us'], kind='mean', unit='index', note=self.ALIGN_NOTE, interval_rel_s=iv)
            if bu['walk_us'] < BURIDAN_WALK_MIN_US:
                walk = record(kind='mean', unit='index', reason='insufficient_events', note=self.ALIGN_NOTE,
                              counts={'walking_s': us_to_s(bu['walk_us']), 'min_walking_s': 1.0}, interval_rel_s=iv)
            else:
                walk = record(bu['walk_align'] / bu['walk_us'], kind='mean', unit='index', note=self.ALIGN_NOTE,
                              counts={'walking_s': us_to_s(bu['walk_us'])}, interval_rel_s=iv)
            frac = bu['centre_us'] / bu['obs_us']
            centre = record(frac, kind='fraction', unit='fraction', numerator=bu['centre_us'],
                            denominator=bu['obs_us'], interval_rel_s=iv)
            phobism = record(1.0 - frac, kind='fraction', unit='fraction', numerator=bu['obs_us'] - bu['centre_us'],
                             denominator=bu['obs_us'], interval_rel_s=iv)
        out = {
            'observation_s': self._v1_duration(bu['obs_us'], iv),
            'walking_s': self._v1_duration(bu['walk_us'], iv),
            'heading_alignment': align, 'walking_stripe_alignment': walk,
            'centre_fraction': centre, 'centrophobism_index': phobism,
            'stripe_traversals': unsupported('count', 'count', 'heading-side flips are not traversals; '
                                             'a spatial traversal metric is not defined in spec buridan/1', iv),
        }
        out.update(self._v1_path_records(iv, net_name='displacement_mm'))
        return out

    def _v1_evidence(self):
        return {}


class _VisualOperantV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'visual_operant', 'presentation', 120.0, triggers=['reverse_heat'], headline='safe_occupancy_fraction',
        companions=['occupancy_index', 'mean_yaw_command_safe_rad_s', 'mean_yaw_command_punished_rad_s',
                    'mean_abs_yaw_command_safe_rad_s', 'mean_abs_yaw_command_punished_rad_s',
                    'yaw_samples_safe', 'yaw_samples_punished'])
    YAW_NOTE = 'controller yaw command used as torque proxy'
    SIGNED_YAW_NOTE = 'signed controller yaw command used as torque proxy'

    def _v1_window_params(self):
        return {'invert_sectors': bool(self.invert_sectors), 'coupling_gain': float(self.coupling_gain)}

    def _v1_clear(self):
        self._vo = {'safe_us': 0, 'pun_us': 0, 'safe_signed': 0.0, 'pun_signed': 0.0, 'safe_abs': 0.0,
                    'pun_abs': 0.0, 'safe_n': 0, 'pun_n': 0}

    def _v1_observe(self, s):
        # The sector and the yaw command are those of the legacy step for this same sample.
        # Sign convention: positive = counter-clockwise (left), as fly.angular_velocity / yaw_torque.
        vo, out = self._vo, self._v1_legacy_out or {}
        punished = bool(out.get('stimuli', {}).get('is_punished'))
        yaw = float(out.get('yaw_torque', s.angular_velocity))
        _require_finite_yaw(yaw)
        key = 'pun' if punished else 'safe'
        vo[f'{key}_us'] += s.observation_dt_us
        vo[f'{key}_signed'] += yaw
        vo[f'{key}_abs'] += abs(yaw)
        vo[f'{key}_n'] += 1

    def _v1_records(self, cut):
        vo, iv = self._vo, self._v1_interval()
        total = vo['safe_us'] + vo['pun_us']

        def sector_counts(key):
            return {'n': vo[f'{key}_n'], 'sum_signed_rad_s': vo[f'{key}_signed'], 'sum_abs_rad_s': vo[f'{key}_abs']}

        def mean(key, signed=False):
            n = vo[f'{key}_n']
            note = self.SIGNED_YAW_NOTE if signed else self.YAW_NOTE
            if n == 0:
                return record(kind='mean', unit='rad/s', reason='not_observed', counts=sector_counts(key),
                              note=note, interval_rel_s=iv)
            total_yaw = vo[f'{key}_signed'] if signed else vo[f'{key}_abs']
            return record(total_yaw / n, kind='mean', unit='rad/s', counts=sector_counts(key),
                          note=note, interval_rel_s=iv)

        return {
            'safe_occupancy_fraction': ratio(vo['safe_us'], total, kind='fraction', unit='fraction',
                                             reason_if_zero='not_observed', interval_rel_s=iv,
                                             note='occupancy, not learned avoidance'),
            'occupancy_index': ratio(vo['safe_us'] - vo['pun_us'], total, kind='index', unit='index',
                                     reason_if_zero='not_observed', interval_rel_s=iv,
                                     note='occupancy, not learned avoidance'),
            'mean_abs_yaw_command_safe_rad_s': mean('safe'),
            'mean_abs_yaw_command_punished_rad_s': mean('pun'),
            'mean_yaw_command_safe_rad_s': mean('safe', signed=True),
            'mean_yaw_command_punished_rad_s': mean('pun', signed=True),
            'yaw_samples_safe': record(vo['safe_n'], kind='count', unit='count', interval_rel_s=iv),
            'yaw_samples_punished': record(vo['pun_n'], kind='count', unit='count', interval_rel_s=iv),
            'operant_learning': unsupported('index', 'index', 'no pattern-specific operant memory is implemented', iv),
        }

    def _v1_evidence(self):
        return {}


class _WindTunnelV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'wind_tunnel', 'until_terminal', 120.0, hold_s=1.0, terminal_events=['source_entry'],
        triggers=['windVelocity', 'plumeWidth', 'shift_plume'], headline='time_to_source_s',
        companions=['upwind_displacement_mm', 'odor_contact_s', 'surge_s', 'cast_s', 'rest_s'],
        required_states={'surge_s': ['SURGE'], 'cast_s': ['CAST'], 'rest_s': ['REST'],
                         'other_state_s': ['SURGE', 'CAST', 'REST'], 'surge_cast_ratio': ['SURGE', 'CAST']})

    def _v1_window_params(self):
        return {'wind_velocity_mm_s': float(-self.wind_flow[0]), 'plume_sigma_mm': float(self.filament_sigma),
                'nozzle_y_mm': float(self.nozzle_pos[1])}

    def _v1_clear(self):
        self._v1_path_clear()
        self._wt = {'obs_us': 0, 'odor_us': 0, 'SURGE': 0, 'CAST': 0, 'REST': 0, 'other': 0,
                    'x0': None, 'x': None, 'source_us': None}

    def _v1_observe(self, s):
        wt = self._wt
        self._v1_path(s)
        if wt['x0'] is None:
            wt['x0'] = s.x
        wt['x'] = s.x
        if wt['source_us'] is None and any(z.name == 'source' for z in self.get_active_zones(s.x, s.y)):
            wt['source_us'] = s.t_us
            if self._v1_terminal('source_entry', s):
                return
        wt['obs_us'] += s.observation_dt_us
        odor_conc = self.sample_stimuli(s.x, s.y, s.heading)['odor_conc']
        if _finite_stimulus(odor_conc, 'wind_tunnel.stimuli.odor_conc') > 0.05:
            wt['odor_us'] += s.observation_dt_us
        state = str(_fly_field(s.fly, 'behavioral_state', '') or '')
        wt[state if state in ('SURGE', 'CAST', 'REST') else 'other'] += s.observation_dt_us

    def _v1_records(self, cut):
        wt, iv = self._wt, self._v1_interval()
        out = {
            'observation_s': self._v1_duration(wt['obs_us'], iv),
            'odor_contact_s': self._v1_duration(wt['odor_us'], iv, note='samples with odour concentration > 0.05'),
            'source_reached': self._v1_outcome(wt['source_us'] is not None),
            'time_to_source_s': self._v1_latency(wt['source_us']),
            'surge_s': self._v1_gated_duration('surge_s', wt['SURGE'], iv),
            'cast_s': self._v1_gated_duration('cast_s', wt['CAST'], iv),
            'rest_s': self._v1_gated_duration('rest_s', wt['REST'], iv),
            'other_state_s': self._v1_gated_duration('other_state_s', wt['other'], iv),
        }
        missing = self._v1_state_gate(self.OBSERVATION_SPEC['required_states']['surge_cast_ratio'])
        if missing:
            out['surge_cast_ratio'] = unsupported('ratio', 'ratio', f'controller does not emit {missing}', iv)
        elif self._v1_observed_samples() == 0:
            out['surge_cast_ratio'] = record(kind='ratio', unit='ratio', reason='not_observed', interval_rel_s=iv)
        else:
            out['surge_cast_ratio'] = ratio(us_to_s(wt['SURGE']), us_to_s(wt['CAST']), interval_rel_s=iv,
                                            counts={'surge_s': us_to_s(wt['SURGE']), 'cast_s': us_to_s(wt['CAST']),
                                                    'rest_s': us_to_s(wt['REST'])})
        if wt['x0'] is None:
            out['upwind_displacement_mm'] = record(kind='kinematic', unit='mm', reason='not_observed', interval_rel_s=iv,
                                                   note='displacement, not evidence of wind sensing')
        else:
            out['upwind_displacement_mm'] = record(wt['x'] - wt['x0'], kind='kinematic', unit='mm', interval_rel_s=iv,
                                                   note='displacement, not evidence of wind sensing')
        out.update(self._v1_path_records(iv))
        return out

    def _v1_evidence(self):
        return {}


class _LoomingV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'looming_escape', 'presentation', None, triggers=['loom'], headline='escape_initiated',
        companions=['ttc_at_initiation_ms', 'theta_at_initiation_deg', 'escape_completed'])

    def _v1_window_s(self):
        return self.t_collision_s + 1.6

    def _v1_window_params(self):
        return {'t_collision_s': float(self.t_collision_s), 'r_over_v_s': float(self.r_over_v_s),
                'gf_threshold_deg': math.degrees(self.gf_threshold_rad)}

    def _v1_clear(self):
        self._le = {'onset': False, 'collision': False, 'init': None, 'motor_started': False,
                    'completed_us': None, 'displacement': None, 'events': []}

    def _v1_theta(self, t_us):
        """Disc angle at presentation time t (the looming equation of sample_stimuli)."""
        t = us_to_s(t_us - self._v1_pres_start_us)
        ttc = max(0.001, self.t_collision_s - t)
        return 2.0 * math.atan(self.r_over_v_s / ttc), self.t_collision_s - t

    def _v1_gf_source(self):
        prov = self.provenance if isinstance(self.provenance, dict) else {}
        return prov.get('gf_source') or self.gf_source

    def _v1_initiate(self, t_us, step, x, y, heading, source, stimuli):
        theta_deg = float(stimuli['theta_deg'])
        ttc_ms = float(stimuli['time_to_collision_s']) * 1000.0
        le = self._le
        le['init'] = {'t_us': t_us, 'theta_deg': theta_deg, 'ttc_ms': ttc_ms,
                      'pose': (x, y, heading), 'step': step}
        _cap(le['events'], event('gf_event', us_to_s(t_us), step, theta_deg=theta_deg,
                                 ttc_ms=ttc_ms, source=source))

    def _v1_record_gf_event(self, stimuli):
        """record_gf_spike(): the connectome DNp01 event, stamped at the latest pre-motor sample."""
        self._v1_ready()
        if self._le['init'] is None and self._v1_end is None and self._v1_last_t_us is not None:
            pose = self._v1_last_pose or (0.0, 0.0, 0.0)
            self._v1_initiate(self._v1_last_t_us, self._v1_last_step, *pose,
                              source='connectome', stimuli=stimuli)

    def _v1_observe(self, s):
        le = self._le
        self._v1_last_pose = (s.x, s.y, s.heading)
        if not le['onset']:
            le['onset'] = True
            _cap(le['events'], event('stimulus_onset', us_to_s(self._v1_pres_start_us), s.step,
                                     offset_s=0.0))
        collision_us = self._v1_pres_start_us + s_to_us(self.t_collision_s)
        if not le['collision'] and s.t_us >= collision_us:
            # A scheduled stimulus event, stamped at P + t_collision_s (segment-relative), never an escape.
            le['collision'] = True
            _cap(le['events'], event('stimulus_collision', us_to_s(collision_us), s.step,
                                     offset_s=float(self.t_collision_s), observed_at_rel_s=s.t_rel_s))
        actual = self._v1_legacy_out if isinstance(self._v1_legacy_out, dict) else {}
        stimuli = actual.get('stimuli')
        if (le['init'] is None and self._v1_gf_source() == 'geometric' and actual.get('gf_spike')
                and isinstance(stimuli, dict)):
            self._v1_initiate(s.t_us, s.step, s.x, s.y, s.heading, 'geometric', stimuli)
        if le['init'] is not None and le['completed_us'] is None:
            remaining = float(_fly_field(s.fly, 'assay_escape_remaining', 0.0) or 0.0)
            if remaining > 0 and not le['motor_started']:
                le['motor_started'] = True
                _cap(le['events'], event('escape_motor_start', s.t_rel_s, s.step))
            elif remaining <= 0 and le['motor_started']:
                le['completed_us'] = s.t_us
                x0, y0, _ = le['init']['pose']
                le['displacement'] = math.hypot(s.x - x0, s.y - y0)
                _cap(le['events'], event('escape_motor_end', s.t_rel_s, s.step))

    def _v1_records(self, cut):
        le, iv = self._le, self._v1_interval()
        init = le['init']
        out = {'escape_initiated': self._v1_outcome(init is not None,
                                                    note='initiation from an explicit GF event only'),
               'escape_completed': self._v1_outcome(le['completed_us'] is not None)}
        if init:
            out['initiation_latency_s'] = self._v1_latency(init['t_us'])
            out['ttc_at_initiation_ms'] = self._v1_faulted_value(
                init['ttc_ms'], kind='latency', unit='ms', evidence_ref='presentation_events')
            out['theta_at_initiation_deg'] = record(init['theta_deg'], kind='kinematic', unit='deg',
                                                    interval_rel_s=iv, evidence_ref='presentation_events')
        else:
            out['initiation_latency_s'] = self._v1_latency(None)
            out['ttc_at_initiation_ms'] = self._v1_pending_value('latency', 'ms')
            out['theta_at_initiation_deg'] = self._v1_pending_value('kinematic', 'deg')
        if le['displacement'] is not None:
            out['escape_displacement_mm'] = record(le['displacement'], kind='kinematic', unit='mm', interval_rel_s=iv)
        else:
            out['escape_displacement_mm'] = self._v1_pending_value('kinematic', 'mm')
        out['gf_source'] = record(str(self._v1_gf_source()), kind='label', unit='label', interval_rel_s=iv)
        return out

    def _v1_evidence(self):
        return {'presentation_events': evidence('event_sequence', self._le['events'])}

    def _v1_envelope_extra(self):
        return {'stimulus_collision_rel_s': us_to_s(self._v1_pres_start_us + s_to_us(self.t_collision_s))}

    def _v1_on_new_presentation(self, reason):
        # A new presentation is a new loom: restart the stimulus clock in place (as act('loom')).
        self.stimulus_started_ms = self.time_elapsed_ms
        self.escape_initiated = False
        self.gf_spike = False


class _OptomotorV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'optomotor', 'presentation', 60.0, triggers=['patternSpeed', 'reverse_grating', 'contrast'],
        headline='gain', companions=['mean_fly_yaw_deg_s', 'mean_retinal_slip_deg_s'])

    def _v1_window_params(self):
        v = float(self.drum_velocity_deg_s)
        return {'drum_velocity_deg_s': v, 'contrast': float(self.contrast),
                'direction': 'positive' if v > 0 else ('negative' if v < 0 else 'static')}

    def _v1_clear(self):
        self._om = {'obs_us': 0, 'yaw': 0.0, 'drum': 0.0, 'slip': 0.0}

    def _v1_observe(self, s):
        om = self._om
        yaw = math.degrees(s.angular_velocity)
        drum = float(self.drum_velocity_deg_s)
        om['obs_us'] += s.observation_dt_us
        om['yaw'] += yaw * s.observation_dt
        om['drum'] += drum * s.observation_dt
        om['slip'] += (drum - yaw) * s.observation_dt

    def _v1_records(self, cut):
        om, iv = self._om, self._v1_interval()
        prov = self.provenance if isinstance(self.provenance, dict) else {}
        stage = prov.get('stimulus_entry_stage')
        out = {
            'mean_hs_firing_rate': unsupported('mean', 'label', 'phenomenological formula, not a measured neuron', iv),
            'stimulus_entry_stage': (record(str(stage), kind='label', unit='label', interval_rel_s=iv) if stage
                                     else record(kind='label', unit='label', reason='not_observed',
                                                 note='no stimulus-entry provenance supplied', interval_rel_s=iv)),
        }
        if om['obs_us'] == 0:
            out['gain'] = record(kind='ratio', unit='gain', reason='not_observed', interval_rel_s=iv)
            out['mean_fly_yaw_deg_s'] = record(kind='mean', unit='deg/s', reason='not_observed', interval_rel_s=iv)
            out['mean_retinal_slip_deg_s'] = record(kind='mean', unit='deg/s', reason='not_observed', interval_rel_s=iv)
            return out
        t = us_to_s(om['obs_us'])
        out['gain'] = ratio(om['yaw'], om['drum'], unit='gain', interval_rel_s=iv,
                            note='ratio of integrals: sum(yaw*dt) / sum(drum velocity*dt)')
        out['mean_fly_yaw_deg_s'] = record(om['yaw'] / t, kind='mean', unit='deg/s', interval_rel_s=iv)
        out['mean_retinal_slip_deg_s'] = record(om['slip'] / t, kind='mean', unit='deg/s', interval_rel_s=iv)
        return out

    def _v1_evidence(self):
        return {}

    def _v1_on_new_presentation(self, reason):
        # A legacy world has no phase origin. Only a new presentation declares one;
        # speed/reversal/contrast changes preserve any already known phase.
        if getattr(self, 'drum_angle_deg', None) is None:
            self.drum_angle_deg = 0.0


class _GapCrossingV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'gap_crossing', 'until_terminal', 120.0, hold_s=1.0, terminal_events=['landed', 'turn_complete'],
        triggers=['gapWidth'], headline='crossing_success',
        companions=['decision_outcome', 'time_to_cross_s', 'probing_duration_s', 'decision_latency_s'])
    DECISION_NOTE = 'paradigm geometric threshold decision, not tactile planning'
    SUCCESS_NOTE = 'crossed by the declared deadline'

    def _v1_window_params(self):
        return {'gap_width_mm': float(self.gap_width_mm),
                'reachability_threshold_mm': float(self.reachability_threshold_mm)}

    def _v1_clear(self):
        self._gc = {'probe_us': 0, 'obs_us': 0, 'decision': None, 'decision_us': None, 'landed': False,
                    'landed_us': None, 'turned': False, 'events': []}

    def _v1_observe(self, s):
        gc = self._gc
        stim = self.sample_stimuli(s.x, s.y, s.heading)
        decided_now = False
        if stim['is_probing'] and gc['decision'] is None:
            gc['decision'] = 'CROSS' if stim['p_cross'] >= 0.5 else 'ABORT'
            gc['decision_us'] = s.t_us
            _cap(gc['events'], event('probe', s.t_rel_s, s.step))
            _cap(gc['events'], event(gc['decision'].lower(), s.t_rel_s, s.step))
            decided_now = True
        if gc['landed_us'] is None and any(z.name == 'landing_track' for z in self.get_active_zones(s.x, s.y)):
            gc['landed'], gc['landed_us'] = True, s.t_us
            _cap(gc['events'], event('landed', s.t_rel_s, s.step))
            if self._v1_terminal('landed', s):
                return
        if (gc['decision'] == 'ABORT' and not decided_now and not gc['turned']
                and abs(_wrap(math.pi - s.heading)) < GAP_TURN_COMPLETE_RAD):
            gc['turned'] = True
            _cap(gc['events'], event('turn_complete', s.t_rel_s, s.step))
            if self._v1_terminal('turn_complete', s):
                return
        gc['obs_us'] += s.observation_dt_us
        if stim['is_probing']:
            gc['probe_us'] += s.observation_dt_us

    def _v1_records(self, cut):
        gc, iv = self._gc, self._v1_interval()
        if gc['decision']:
            decision = record(gc['decision'], kind='label', unit='label', note=self.DECISION_NOTE, interval_rel_s=iv,
                              evidence_ref='decision_sequence')
        else:
            decision = self._v1_pending_value('label', 'label', self.DECISION_NOTE)
        success = self._v1_outcome(gc['landed'], note=self.SUCCESS_NOTE)
        turn = self._v1_outcome(gc['turned'], note='after abort: first sample with |wrap(pi - heading)| < 0.4 rad')
        if self._v1_observed_samples() == 0:
            probing = record(kind='duration', unit='s', reason='not_observed', interval_rel_s=iv)
        else:
            probing = record(us_to_s(gc['probe_us']), kind='duration', unit='s', interval_rel_s=iv)
        return {'decision_outcome': decision, 'crossing_success': success, 'turn_complete': turn,
                'time_to_cross_s': self._v1_latency(gc['landed_us'], note='latency from the presentation start to landed'),
                'probing_duration_s': probing, 'decision_latency_s': self._v1_latency(gc['decision_us'])}

    def _v1_evidence(self):
        return {'decision_sequence': evidence('event_sequence', self._gc['events'])}


class _CircadianDAMV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'circadian_dam', 'continuous', None, headline='bout_immobility_min',
        companions=['beam_crossings', 'current_immobile_s', 'immobility_bouts_300s', 'light_phase'])
    IMMOBILITY_NOTE = 'operational immobility proxy (speed <= 0.5 mm/s, no beam crossing); not validated sleep'

    def _v1_light_schedule(self):
        sched = getattr(self, '_light_schedule', None)
        if sched:
            return dict(sched)
        if self.photoperiod == 'DD':
            return {'mode': 'DD', 'light_s': 0.0, 'dark_s': 86400.0}
        return {'mode': 'LD', 'light_s': 43200.0, 'dark_s': 43200.0}

    def set_light_schedule(self, light_s, dark_s):
        """A compressed light cycle: a labelled software test, never an entrainment claim."""
        light_s, dark_s = float(light_s), float(dark_s)
        if not (math.isfinite(light_s) and math.isfinite(dark_s)) or light_s <= 0 or dark_s <= 0:
            raise ValueError('light_s and dark_s must be positive and finite')
        self._light_schedule = {'mode': 'compressed', 'light_s': light_s, 'dark_s': dark_s,
                                'note': 'software light-cycle test; no entrainment claim'}

    def observation_spec(self):
        spec = super().observation_spec()
        spec['light_schedule'] = self._v1_light_schedule()
        return spec

    def _v1_phase(self, t_us):
        sched = self._v1_light_schedule()
        if sched['mode'] == 'DD':
            return 'dark'
        light, period = s_to_us(sched['light_s']), s_to_us(sched['light_s'] + sched['dark_s'])
        return 'light' if t_us % period < light else 'dark'

    def _v1_clear_segment(self):
        self._dam = {'beam': 0, 'last_x': None, 'cur_us': 0, 'bout_us': 0, 'bouts': 0, 'in_bout': False,
                     'bins': [], 'phase': None}

    def _v1_clear(self):
        pass

    def _v1_observe(self, s):
        dam = self._dam
        crossed = dam['last_x'] is not None and ((dam['last_x'] < 32.5 <= s.x) or (dam['last_x'] > 32.5 >= s.x))
        dam['last_x'] = s.x
        dam['phase'] = self._v1_phase(s.t_us)
        if self._v1_light_schedule()['mode'] == 'compressed':
            actual = self._v1_legacy_out if isinstance(self._v1_legacy_out, dict) else {}
            stimuli = actual.get('stimuli')
            lights_on = stimuli.get('is_lights_on') if isinstance(stimuli, dict) else None
            if isinstance(lights_on, bool):
                dam['phase'] = 'light' if lights_on else 'dark'
        b = s.t_us // (60 * 1_000_000)
        while len(dam['bins']) <= b:
            dam['bins'].append(0)
        if crossed:
            dam['beam'] += 1
            dam['bins'][b] += 1
        if crossed or abs(s.speed) > WALKING_SPEED_MM_S:
            dam['cur_us'] = 0
            dam['in_bout'] = False
            return
        dam['cur_us'] += s.observation_dt_us
        if dam['in_bout']:
            dam['bout_us'] += s.observation_dt_us
        elif dam['cur_us'] >= DAM_BOUT_US:
            dam['in_bout'] = True
            dam['bouts'] += 1
            dam['bout_us'] += dam['cur_us']  # the full qualifying interval, counted once

    def _v1_records(self, cut):
        dam, iv = self._dam, self._v1_seg_interval()
        n = self._v1_segment_samples()
        na = lambda kind, unit, note=None: record(kind=kind, unit=unit, reason='not_observed', note=note, interval_rel_s=iv)
        if n == 0:
            out = {'beam_crossings': na('count', 'count'), 'immobility_bouts_300s': na('count', 'count', self.IMMOBILITY_NOTE),
                   'bout_immobility_min': na('duration', 'min', self.IMMOBILITY_NOTE),
                   'mean_bout_min': na('ratio', 'min'), 'current_immobile_s': na('duration', 's'),
                   'light_phase': na('label', 'label')}
        else:
            bout_min = dam['bout_us'] / 60_000_000
            out = {
                'beam_crossings': record(dam['beam'], kind='count', unit='count', interval_rel_s=iv,
                                         evidence_ref='activity_bins'),
                'immobility_bouts_300s': record(dam['bouts'], kind='count', unit='count', note=self.IMMOBILITY_NOTE,
                                                interval_rel_s=iv),
                'bout_immobility_min': record(bout_min, kind='duration', unit='min', note=self.IMMOBILITY_NOTE,
                                              interval_rel_s=iv),
                'mean_bout_min': ratio(bout_min, dam['bouts'], kind='ratio', unit='min', interval_rel_s=iv),
                'current_immobile_s': record(us_to_s(dam['cur_us']), kind='duration', unit='s', interval_rel_s=iv),
                'light_phase': record(dam['phase'], kind='label', unit='label', interval_rel_s=iv,
                                      evidence_ref='light_phases'),
            }
        out['circadian_rhythm'] = unsupported('label', 'label', 'activity and immobility monitor; no endogenous oscillator', iv)
        return out

    def _v1_evidence(self):
        sched = self._v1_light_schedule()
        end = self._v1_seg_us
        phases = []
        if sched['mode'] == 'DD':
            phases.append({'phase': 'dark', 'start_rel_s': 0.0, 'end_rel_s': us_to_s(end)})
        else:
            light, dark = s_to_us(sched['light_s']), s_to_us(sched['dark_s'])
            t = 0
            while t <= end and len(phases) < EVIDENCE_EVENT_CAP:
                phases.append({'phase': 'light', 'start_rel_s': us_to_s(t), 'end_rel_s': us_to_s(t + light)})
                if t + light <= end:
                    phases.append({'phase': 'dark', 'start_rel_s': us_to_s(t + light),
                                   'end_rel_s': us_to_s(t + light + dark)})
                t += light + dark
        return {'activity_bins': evidence('bins', {'bin_s': 60.0, 'start_rel_s': 0.0,
                                                   'quantity': 'beam_crossings', 'counts': list(self._dam['bins'])}),
                'light_phases': evidence('phase_boundaries', phases)}


class _CourtshipV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'courtship', 'fixed', 600.0, triggers=['receptivity'], headline='proximity_fraction',
        companions=['min_distance_mm', 'approach_s', 'avoid_s', 'courtship_state_fraction'],
        required_states={'approach_s': ['SOCIAL_APPROACH'], 'avoid_s': ['SOCIAL_AVOID'],
                         'courtship_state_s': ['COURTSHIP'], 'courtship_state_fraction': ['COURTSHIP']})

    def _v1_window_params(self):
        return {'female_type': self.female_type, 'proximity_radius_mm': 3.5}

    def _v1_clear(self):
        self._co = {'obs_us': 0, 'near_us': 0, 'min_d': None, 'SOCIAL_APPROACH': 0, 'SOCIAL_AVOID': 0,
                    'COURTSHIP': 0}

    def _v1_observe(self, s):
        co = self._co
        d = math.hypot(self.female_pos[0] - s.x, self.female_pos[1] - s.y)
        co['obs_us'] += s.observation_dt_us
        co['min_d'] = d if co['min_d'] is None else min(co['min_d'], d)
        if d < 3.5:
            co['near_us'] += s.observation_dt_us
        state = str(_fly_field(s.fly, 'behavioral_state', '') or '')
        if state in ('SOCIAL_APPROACH', 'SOCIAL_AVOID', 'COURTSHIP'):
            co[state] += s.observation_dt_us

    def _v1_records(self, cut):
        co, iv = self._co, self._v1_interval()
        out = {'proximity_fraction': ratio(co['near_us'], co['obs_us'], kind='fraction', unit='fraction',
                                           reason_if_zero='not_observed', interval_rel_s=iv,
                                           note='time within 3.5 mm of the female position'),
               'min_distance_mm': (record(co['min_d'], kind='kinematic', unit='mm', interval_rel_s=iv)
                                   if co['min_d'] is not None else
                                   record(kind='kinematic', unit='mm', reason='not_observed', interval_rel_s=iv)),
               'approach_s': self._v1_gated_duration('approach_s', co['SOCIAL_APPROACH'], iv),
               'avoid_s': self._v1_gated_duration('avoid_s', co['SOCIAL_AVOID'], iv),
               'courtship_state_s': self._v1_gated_duration('courtship_state_s', co['COURTSHIP'], iv)}
        missing = self._v1_state_gate(['COURTSHIP'])
        out['courtship_state_fraction'] = (
            unsupported('fraction', 'fraction', f'controller does not emit {missing}', iv) if missing else
            ratio(co['COURTSHIP'], co['obs_us'], kind='fraction', unit='fraction', reason_if_zero='not_observed',
                  interval_rel_s=iv, note="time in the controller's COURTSHIP state"))
        for name, note in (('courtship_conditioning', 'no courtship memory is implemented'),
                           ('learned_suppression', 'no courtship memory is implemented'),
                           ('wing_song', 'no wing-song model'),
                           ('rejection_kicks', 'no female rejection behaviour is modelled')):
            out[name] = unsupported('count' if name == 'rejection_kicks' else 'label',
                                    'count' if name == 'rejection_kicks' else 'label', note, iv)
        return out

    def _v1_evidence(self):
        return {}


class _LabyrinthV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'labyrinth', 'until_terminal', 300.0, hold_s=2.0, terminal_events=['goal_entry'], headline='time_to_goal_s',
        companions=['path_length_mm', 'wall_contact_onsets', 'dead_end_entries', 'tortuosity'])

    def _v1_clear_segment(self):
        self._v1_path_clear()
        self._lb = {'contacts': ContactCounter(), 'armed': {}, 'entries': 0, 'dead_us': 0, 'goal_us': None,
                    'obs_us': 0}

    def _v1_clear(self):
        pass

    def _v1_segment_carry(self):
        return {'armed': dict(self._lb['armed']), 'in_contact': self._lb['contacts'].prev}

    def _v1_segment_restore(self, carry):
        self._lb['armed'] = carry['armed']
        self._lb['contacts'].prev = carry['in_contact']

    @staticmethod
    def _outside_by(bounds, x, y):
        x0, y0, x1, y1 = bounds
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        return math.hypot(dx, dy)

    def _v1_observe(self, s):
        lb = self._lb
        self._v1_path(s)
        if lb['goal_us'] is None and any(z.name == 'goal' and z.contains(s.x, s.y) for z in self.zones):
            lb['goal_us'] = s.t_us
            if self._v1_terminal('goal_entry', s):
                return
        lb['obs_us'] += s.observation_dt_us
        inside_any = False
        for z in self.zones:
            if not z.name.startswith('dead_end'):
                continue
            armed = lb['armed'].get(z.name, True)
            if z.contains(s.x, s.y):
                inside_any = True
                if armed:
                    lb['entries'] += 1
                    lb['armed'][z.name] = False
            elif self._outside_by(z.bounds, s.x, s.y) >= LABYRINTH_REARM_MM:
                lb['armed'][z.name] = True
        if inside_any:
            lb['dead_us'] += s.observation_dt_us

    def _v1_contact(self, step, t_us, dt_us, in_contact, source):
        self._lb['contacts'].observe(step, t_us, dt_us, in_contact, source)

    def _v1_contact_prefix(self, dt_us):
        self._lb['contacts'].integrate_prefix(dt_us)

    def _v1_records(self, cut):
        lb, iv = self._lb, self._v1_seg_interval()
        out = lb['contacts'].records(iv)
        segment_samples = self._v1_segment_samples()
        observed = segment_samples > 0
        out['dead_end_entries'] = (record(lb['entries'], kind='count', unit='count', interval_rel_s=iv) if observed
                                   else record(kind='count', unit='count', reason='not_observed', interval_rel_s=iv))
        out['dead_end_s'] = self._v1_duration(lb['dead_us'], iv, samples=segment_samples)
        out.update(self._v1_path_records(iv, net_name='net_displacement_mm'))
        if not observed:
            out['tortuosity'] = record(kind='ratio', unit='ratio', reason='not_observed', interval_rel_s=iv)
        else:
            net = math.dist(self._p_first, self._p_last)
            out['tortuosity'] = (record(kind='ratio', unit='ratio', reason='zero_denominator', numerator=self._p_len,
                                        denominator=0.0, interval_rel_s=iv) if net < 1e-9 else
                                 record(self._p_len / net, kind='ratio', unit='ratio', numerator=self._p_len,
                                        denominator=net, interval_rel_s=iv))
        out['goal_reached'] = self._v1_segment_outcome(lb['goal_us'] is not None)
        out['time_to_goal_s'] = self._v1_segment_latency(lb['goal_us'])
        return out

    def _v1_evidence(self):
        return {'contact_events': evidence('event_sequence', self._lb['contacts'].events)}


class _MultisensoryV1(_V1Paths):
    OBSERVATION_SPEC = _spec(
        'multisensory', 'fixed', 120.0, headline='distance_mm',
        companions=['odor_a_exposure_s', 'heat_exposure_s', 'wall_contact_onsets'])
    SCRIPTED_NOTE = 'scripted leg phases are illustrative; no neural coordination'

    def _v1_clear_segment(self):
        self._v1_path_clear()
        self._ms = {'contacts': ContactCounter(), 'obs_us': 0, 'odor_us': 0, 'heat_us': 0, 'v': None, 'a': None,
                    'jerk_sum': 0.0, 'jerk_n': 0}

    def _v1_clear(self):
        pass

    def _v1_segment_carry(self):
        return self._ms['contacts'].prev

    def _v1_segment_restore(self, carry):
        self._ms['contacts'].prev = carry

    def _v1_observe(self, s):
        ms = self._ms
        self._v1_path(s)
        stim = self.sample_stimuli(s.x, s.y, s.heading)
        _finite_stimulus(stim['odor_a'], 'multisensory.stimuli.odor_a')
        _finite_stimulus(stim['temperature'], 'multisensory.stimuli.temperature')
        ms['obs_us'] += s.observation_dt_us
        if stim['odor_a'] >= 0.5:
            ms['odor_us'] += s.observation_dt_us
        if stim['temperature'] > 35.0:
            ms['heat_us'] += s.observation_dt_us
        if ms['v'] is not None:
            accel = (s.speed - ms['v']) / s.dt
            if ms['a'] is not None:
                ms['jerk_sum'] += abs(accel - ms['a']) / s.dt
                ms['jerk_n'] += 1
            ms['a'] = accel
        ms['v'] = s.speed

    def _v1_contact(self, step, t_us, dt_us, in_contact, source):
        self._ms['contacts'].observe(step, t_us, dt_us, in_contact, source)

    def _v1_contact_prefix(self, dt_us):
        self._ms['contacts'].integrate_prefix(dt_us)

    def _v1_records(self, cut):
        ms, iv = self._ms, self._v1_seg_interval()
        segment_samples = self._v1_segment_samples()
        out = ms['contacts'].records(iv)
        out.update(self._v1_path_records(iv, path_name='distance_mm'))
        out['odor_a_exposure_s'] = self._v1_duration(
            ms['odor_us'], iv, note='time with odour A >= 0.5', samples=segment_samples)
        out['heat_exposure_s'] = self._v1_duration(
            ms['heat_us'], iv, note='time with temperature > 35 degC', samples=segment_samples)
        out['mean_abs_speed_jerk_mm_s3'] = (
            record(ms['jerk_sum'] / ms['jerk_n'], kind='mean', unit='mm/s^3', counts={'samples': ms['jerk_n']},
                   note='proxy: mean |delta acceleration| / dt from the sampled speed', interval_rel_s=iv)
            if ms['jerk_n'] else
            record(kind='mean', unit='mm/s^3', reason='not_observed', counts={'samples': 0}, interval_rel_s=iv))
        for name in ('composite_benchmark_score', 'locomotor_coordination_index', 'multisensory_integration_score',
                     'biomechanical_efficiency', 'kinematic_smoothness', 'energy_proxy'):
            out[name] = unsupported('index', 'index', self.SCRIPTED_NOTE, iv)
        return out

    def _v1_evidence(self):
        return {'contact_events': evidence('event_sequence', self._ms['contacts'].events)}


# ==============================================================================
# 3. 12 CONCRETE EXPERIMENT PARADIGMS
# ==============================================================================

class TMazeParadigm(_TMazeV1, ExperimentParadigm):
    """Paradigm 1: T-Maze Olfactory Associative Conditioning (Tully & Quinn 1985).

    Stem (60x14mm), Left Arm (CS+, sucrose reward), Right Arm (CS-, shock grid 60V).
    Includes vacuum airflow (15mm/s down the stem) and terminal odor emitters.
    Calculates Classical Performance Index (PI in [-1.0, +1.0]).
    """

    def __init__(self, cs_plus_arm: str = 'arm_a', max_duration_steps: int = 1000):
        dimensions = (140.0, 80.0)
        self.cs_plus_arm = cs_plus_arm  # 'arm_a' (left) or 'arm_b' (right)

        # Build T-Maze corridor walls (stem: x in [63, 77], y in [10, 50]; arms: y in [43, 57], x in [10, 130])
        walls = [
            # Stem base and sides
            WallSegment((63.0, 10.0), (77.0, 10.0)),
            WallSegment((63.0, 10.0), (63.0, 43.0)),
            WallSegment((77.0, 10.0), (77.0, 43.0)),
            # Left Arm (Arm A)
            WallSegment((10.0, 43.0), (63.0, 43.0)),
            WallSegment((10.0, 43.0), (10.0, 57.0)),
            WallSegment((10.0, 57.0), (130.0, 57.0)),  # Continuous top wall
            # Right Arm (Arm B)
            WallSegment((130.0, 43.0), (130.0, 57.0)),
            WallSegment((77.0, 43.0), (130.0, 43.0)),
        ]

        # Zones
        reward_a = 1.0 if cs_plus_arm == 'arm_a' else 0.0
        punish_a = 1.0 if cs_plus_arm != 'arm_a' else 0.0
        reward_b = 1.0 if cs_plus_arm == 'arm_b' else 0.0
        punish_b = 1.0 if cs_plus_arm != 'arm_b' else 0.0

        zones = [
            MazeZone("stem", "corridor", (63.0, 10.0, 77.0, 43.0)),
            MazeZone("choice_hub", "hub", (63.0, 43.0, 77.0, 57.0)),
            MazeZone("arm_a", "reward" if reward_a > 0 else "punishment", (10.0, 43.0, 63.0, 57.0), reward=reward_a, punishment=punish_a),
            MazeZone("arm_b", "reward" if reward_b > 0 else "punishment", (77.0, 43.0, 130.0, 57.0), reward=reward_b, punishment=punish_b),
        ]

        super().__init__(
            name="t_maze",
            description="T-Maze Pavlovian Olfactory Conditioning with CS+/CS- odors and electric shock reinforcement",
            dimensions=dimensions,
            walls=walls,
            zones=zones,
            max_duration_steps=max_duration_steps
        )

        self.choice_counts = {'arm_a': 0, 'arm_b': 0}
        self.last_choice_arm = None
        self.first_choice: Optional[str] = None
        self.latency_to_choice_ms: Optional[float] = None
        self.step_history = deque(maxlen=2048)

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        # Odor emitters at arm terminals: CS+ at (15, 50), CS- at (125, 50)
        source_a = (15.0, 50.0)
        source_b = (125.0, 50.0)

        dist_a = math.sqrt((x - source_a[0]) ** 2 + (y - source_a[1]) ** 2)
        dist_b = math.sqrt((x - source_b[0]) ** 2 + (y - source_b[1]) ** 2)

        sigma = 22.0
        conc_a = float(np.clip(math.exp(-(dist_a * dist_a) / (2.0 * sigma * sigma)), 0.0, 1.0))
        conc_b = float(np.clip(math.exp(-(dist_b * dist_b) / (2.0 * sigma * sigma)), 0.0, 1.0))

        # Vacuum airflow: in stem (y < 45), suction draws air down towards base (0, -15 mm/s)
        if y < 45.0:
            wind = (0.0, -15.0)
        elif x < 70.0:
            wind = (15.0, 0.0)  # Air drawn from left arm towards center
        else:
            wind = (-15.0, 0.0)  # Air drawn from right arm towards center

        return {
            'odor_a': conc_a,
            'odor_b': conc_b,
            'odor_cs_plus': conc_a if self.cs_plus_arm == 'arm_a' else conc_b,
            'odor_cs_minus': conc_b if self.cs_plus_arm == 'arm_a' else conc_a,
            'wind': wind,
            'temperature': 24.0,
            'food_contact': (dist_a < 4 if self.cs_plus_arm == 'arm_a' else dist_b < 4)
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)
        active_zones = self.get_active_zones(x, y)

        reward = sum(z.reward for z in active_zones)
        punishment = sum(z.punishment for z in active_zones)

        # Count an arm entry once, not every integration tick spent in that arm.
        active_zone_names = [z.name for z in active_zones]
        arm = 'arm_a' if 'arm_a' in active_zone_names and x < 45 else ('arm_b' if 'arm_b' in active_zone_names and x > 95 else None)
        if arm and arm != self.last_choice_arm:
            if self.first_choice is None:
                self.first_choice = arm
                self.latency_to_choice_ms = self.time_elapsed_ms
            self.choice_counts[arm] += 1
            self.step_history.append(arm)
            self.last_choice_arm = arm
        elif 63 <= x <= 77:
            self.last_choice_arm = None

        return {
            'stimuli': stimuli,
            'active_zones': active_zone_names,
            'reward': reward,
            'punishment': punishment,
            'first_choice': self.first_choice,
            'latency_to_choice_ms': self.latency_to_choice_ms,
            'performance_index': self.get_metrics()['performance_index']
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.first_choice = None
        self.latency_to_choice_ms = None
        self.choice_counts = {'arm_a': 0, 'arm_b': 0}
        self.last_choice_arm = None
        self.step_history.clear()
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        n_cs_plus = self.choice_counts['arm_a'] if self.cs_plus_arm == 'arm_a' else self.choice_counts['arm_b']
        n_cs_minus = self.choice_counts['arm_b'] if self.cs_plus_arm == 'arm_a' else self.choice_counts['arm_a']
        total = n_cs_plus + n_cs_minus
        pi = float((n_cs_plus - n_cs_minus) / total) if total > 0 else 0.0
        return {
            'performance_index': pi,
            'cs_plus_choices': n_cs_plus,
            'cs_minus_choices': n_cs_minus,
            'first_choice': self.first_choice,
            'latency_to_choice_ms': self.latency_to_choice_ms
        }


class YMazeParadigm(_YMazeV1, ExperimentParadigm):
    """Paradigm 2: Y-Maze Spontaneous Alternation & Handedness (Buchanan et al. 2015).

    3 symmetric arms oriented at 0, 120, 240 degrees (arm length 40mm, width 12mm),
    surrounding a central hexagonal decision hub (radius 10mm).
    Tracks spontaneous alternation rate (SAR) and turn handedness bias.
    """

    def __init__(self, max_duration_steps: int = 1000):
        dimensions = (120.0, 120.0)
        cx, cy = 60.0, 60.0
        self.center = (cx, cy)
        arm_length = 40.0
        half_w = 6.0
        hub_r = 10.0

        # 3 arm orientations: 90 deg (North), 210 deg (South-West), 330 deg (South-East)
        angles = [math.pi / 2.0, 7.0 * math.pi / 6.0, 11.0 * math.pi / 6.0]
        self.arm_angles = angles

        # Construct corridor walls enclosing 3 arms
        walls: List[WallSegment] = []
        arm_endpoints: List[Tuple[float, float]] = []

        for i in range(3):
            j = (i + 1) % 3
            ang_i = angles[i]
            ang_j = angles[j]
            arm_endpoints.append((cx + arm_length * math.cos(ang_i), cy + arm_length * math.sin(ang_i)))

            tip_i_left = (
                cx + arm_length * math.cos(ang_i) - math.sin(ang_i) * half_w,
                cy + arm_length * math.sin(ang_i) + math.cos(ang_i) * half_w
            )
            tip_i_right = (
                cx + arm_length * math.cos(ang_i) + math.sin(ang_i) * half_w,
                cy + arm_length * math.sin(ang_i) - math.cos(ang_i) * half_w
            )
            tip_j_right = (
                cx + arm_length * math.cos(ang_j) + math.sin(ang_j) * half_w,
                cy + arm_length * math.sin(ang_j) - math.cos(ang_j) * half_w
            )

            mid_ang = (ang_i + ang_j) / 2.0
            if ang_j < ang_i:
                mid_ang += math.pi
            corner_x = cx + hub_r * math.cos(mid_ang)
            corner_y = cy + hub_r * math.sin(mid_ang)

            walls.append(WallSegment(tip_i_right, tip_i_left))
            walls.append(WallSegment(tip_i_left, (corner_x, corner_y)))
            walls.append(WallSegment((corner_x, corner_y), tip_j_right))

        zones = [
            MazeZone("hub", "decision", (cx, cy, hub_r)),
            MazeZone("arm_0", "arm", (arm_endpoints[0][0], arm_endpoints[0][1], 15.0)),
            MazeZone("arm_1", "arm", (arm_endpoints[1][0], arm_endpoints[1][1], 15.0)),
            MazeZone("arm_2", "arm", (arm_endpoints[2][0], arm_endpoints[2][1], 15.0)),
        ]

        super().__init__(
            name="y_maze",
            description="Y-Maze Spontaneous Alternation and Turn Handedness Assay",
            dimensions=dimensions,
            walls=walls,
            zones=zones,
            max_duration_steps=max_duration_steps
        )

        self.arm_sequence: List[int] = []
        self.turn_directions: List[str] = []  # 'L' or 'R'
        self.last_zone: str = "hub"

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        return {
            'wind': (0.0, 0.0),
            'temperature': 24.0,
            'odor_a': 0.0,
            'odor_b': 0.0
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)
        active_zones = self.get_active_zones(x, y)
        zone_names = [z.name for z in active_zones]

        current_zone = "hub"
        for name in ["arm_0", "arm_1", "arm_2"]:
            if name in zone_names:
                current_zone = name
                break

        if current_zone != self.last_zone and current_zone.startswith("arm_"):
            arm_idx = int(current_zone.split("_")[1])
            if not self.arm_sequence or self.arm_sequence[-1] != arm_idx:
                if self.arm_sequence:
                    prev_arm = self.arm_sequence[-1]
                    # Turn direction: (arm_idx - prev_arm) mod 3
                    diff = (arm_idx - prev_arm) % 3
                    if diff == 1:
                        self.turn_directions.append('L')
                    elif diff == 2:
                        self.turn_directions.append('R')
                self.arm_sequence.append(arm_idx)

        self.last_zone = current_zone

        return {
            'stimuli': stimuli,
            'active_zones': zone_names,
            'current_zone': current_zone,
            'arm_sequence': list(self.arm_sequence),
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.arm_sequence.clear()
        self.turn_directions.clear()
        self.last_zone = "hub"
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        seq = self.arm_sequence
        total_triads = max(0, len(seq) - 2)
        alternating_triads = 0
        for i in range(total_triads):
            triad = seq[i:i + 3]
            if len(set(triad)) == 3:
                alternating_triads += 1

        sar = float(alternating_triads / total_triads) if total_triads > 0 else 0.0

        n_l = self.turn_directions.count('L')
        n_r = self.turn_directions.count('R')
        total_turns = n_l + n_r
        handedness = float((n_r - n_l) / total_turns) if total_turns > 0 else 0.0

        return {
            'spontaneous_alternation_rate': sar,
            'total_triads': total_triads,
            'alternating_triads': alternating_triads,
            'handedness_index': handedness,
            'left_turns': n_l,
            'right_turns': n_r,
            'choice_sequence': "".join(["A", "B", "C"][idx] for idx in seq)
        }


class HeatMazeParadigm(_HeatMazeV1, ExperimentParadigm):
    """Paradigm 3: Thermal Heat-Maze Place Learning (Ofstad, Zuker & Reiser, Nature 2011).

    Circular arena (R=55mm), 36.5 deg C heated floor, 24.0 deg C cool refuge at (22, 18),
    surrounded by 4 distal visual landmark stripes at 0, 90, 180, 270 deg.
    Models central complex vs mushroom body double dissociation of spatial place memory.
    """

    def __init__(self, max_duration_steps: int = 1000):
        dimensions = (120.0, 120.0)
        cx, cy = 60.0, 60.0
        self.arena_center = (cx, cy)
        self.arena_radius = 55.0

        # Construct 32-segment circular perimeter wall
        num_segments = 32
        walls: List[WallSegment] = []
        for i in range(num_segments):
            a1 = 2.0 * math.pi * i / num_segments
            a2 = 2.0 * math.pi * (i + 1) / num_segments
            p1 = (cx + self.arena_radius * math.cos(a1), cy + self.arena_radius * math.sin(a1))
            p2 = (cx + self.arena_radius * math.cos(a2), cy + self.arena_radius * math.sin(a2))
            walls.append(WallSegment(p1, p2))

        # Cool spot refuge at (cx + 22.0, cy + 18.0)
        refuge_pos = (cx + 22.0, cy + 18.0)
        self.refuge_pos = refuge_pos
        self.refuge_radius = 9.0

        self.peltier = PeltierGrid(
            baseline_temp=36.5,
            cool_spot=refuge_pos,
            cool_radius=self.refuge_radius,
            cool_temp=24.0,
            gradient_sigma=8.0
        )

        # 4 Distal Visual Landmarks on perimeter
        landmarks = [
            VisualLandmark("stripe_single", azimuth_rad=0.0, glyph='stripe', pos=(cx + 55.0, cy)),
            VisualLandmark("stripe_double", azimuth_rad=math.pi / 2.0, glyph='double_stripe', pos=(cx, cy + 55.0)),
            VisualLandmark("rectangle_wide", azimuth_rad=math.pi, glyph='rectangle', pos=(cx - 55.0, cy)),
            VisualLandmark("cross_inverted", azimuth_rad=3.0 * math.pi / 2.0, glyph='cross', pos=(cx, cy - 55.0)),
        ]

        zones = [
            MazeZone("refuge", "refuge", (refuge_pos[0], refuge_pos[1], self.refuge_radius), reward=1.0),
            MazeZone("target_quadrant", "quadrant", (cx, cy, cx + self.arena_radius, cy + self.arena_radius)),
        ]

        super().__init__(
            name="heat_maze",
            description="Thermal Heat-Maze Allocentric Place Learning with Distal Landmarks",
            dimensions=dimensions,
            walls=walls,
            zones=zones,
            landmarks=landmarks,
            max_duration_steps=max_duration_steps
        )

        self.cumulative_thermal_dose: float = 0.0
        self.escape_latency_ms: Optional[float] = None
        self.refuge_reached: bool = False
        self.pam_burst_active: bool = False
        self.time_in_target_quadrant_ms: float = 0.0
        self.path_points = PathHistory()

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        temp = _finite_stimulus(self.peltier.get_temperature(x, y),
                                'heat_maze.stimuli.temperature')
        r_warm = max(0.0, (temp - 26.0) * 8.0)
        r_cold = max(0.0, (25.0 - temp) * 8.0)

        landmark_bearings = {
            lm.landmark_id: lm.get_apparent_bearing(x, y, heading)
            for lm in self.landmarks
        }

        return {
            'temperature': temp,
            'r_warm_thermosensory': r_warm,
            'r_cold_thermosensory': r_cold,
            'landmark_bearings': landmark_bearings,
            'wind': (0.0, 0.0)
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        self.path_points.append((x, y))

        stimuli = self.sample_stimuli(x, y, heading)
        temp = _finite_stimulus(stimuli['temperature'], 'heat_maze.stimuli.temperature')
        self.cumulative_thermal_dose += max(0.0, temp - 25.0) * dt

        # Check refuge
        active_zones = self.get_active_zones(x, y)
        zone_names = [z.name for z in active_zones]

        if "refuge" in zone_names:
            if not self.refuge_reached:
                self.refuge_reached = True
                self.escape_latency_ms = self.time_elapsed_ms
                self.pam_burst_active = True
            reward = 1.0
        else:
            reward = 0.0
            self.pam_burst_active = False

        if "target_quadrant" in zone_names:
            self.time_in_target_quadrant_ms += dt * 1000.0

        return {
            'stimuli': stimuli,
            'active_zones': zone_names,
            'reward': reward,
            'refuge_reached': self.refuge_reached,
            'escape_latency_ms': self.escape_latency_ms,
            'pam_burst_active': self.pam_burst_active,
            'cumulative_thermal_dose': self.cumulative_thermal_dose
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.cumulative_thermal_dose = 0.0
        self.escape_latency_ms = None
        self.refuge_reached = False
        self.pam_burst_active = False
        self.time_in_target_quadrant_ms = 0.0
        self.path_points.clear()
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        quadrant_pct = (
            (self.time_in_target_quadrant_ms / max(1.0, self.time_elapsed_ms)) * 100.0
        )
        return {
            'escape_latency_ms': self.escape_latency_ms,
            'refuge_reached': self.refuge_reached,
            'cumulative_thermal_dose': self.cumulative_thermal_dose,
            'time_in_target_quadrant_pct': quadrant_pct,
            'path_length': self.path_points.distance
        }

    def simulate_agent_trial(
        self,
        agent_type: str = 'WT',
        trained: bool = False,
        seed: int = 42,
        max_steps: int = 300,
        dt: float = 1.0
    ) -> Dict[str, Any]:
        """Simulate a trial with WT, MB_lesion, or CX_lesion agent.

        Ofstad, Zuker & Reiser (Nature 2011) double dissociation:
        - WT & MB_lesion: Intact Central Complex place learning allows navigation to refuge.
        - CX_lesion: Impaired place learning; wanders randomly regardless of training.
        """
        rng = random.Random(seed)
        self.reset_trial()

        # Spawn near center but outside refuge
        fx = self.arena_center[0] + rng.uniform(-15.0, 15.0)
        fy = self.arena_center[1] + rng.uniform(-15.0, 15.0)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        speed = 1.5

        cx_intact = (agent_type.upper() != 'CX_LESION' and agent_type.upper() != 'ABLATE_CX')

        for step_idx in range(max_steps):
            # Target direction towards cool refuge
            dx_target = self.refuge_pos[0] - fx
            dy_target = self.refuge_pos[1] - fy
            target_heading = math.atan2(dy_target, dx_target)

            if cx_intact and trained:
                # Guided heading towards refuge using allocentric landmarks
                heading_err = (target_heading - heading + math.pi) % (2.0 * math.pi) - math.pi
                dheading = np.clip(heading_err * 0.45, -0.4, 0.4)
            else:
                # Random wander / undirected search
                dheading = rng.gauss(0.0, 0.3)

            heading = (heading + dheading) % (2.0 * math.pi)
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)

            # Move and resolve boundary collision
            fx, fy, vx, vy, _ = self.check_collisions(fx + vx * dt, fy + vy * dt, vx, vy, radius=1.5)

            step_res = self.step({'x': fx, 'y': fy, 'heading': heading, 'speed': speed}, dt=dt)
            if step_res['refuge_reached']:
                break

        metrics = self.get_metrics()
        metrics['steps'] = self.trial_manager.current_step
        metrics['agent_type'] = agent_type
        metrics['trained'] = trained
        return metrics


class BuridanParadigm(_BuridanV1, ExperimentParadigm):
    """Paradigm 4: Buridan's Visual Landmark Paradigm (Götz 1980; Colomb & Brembs 2012).

    Circular platform (R=50mm) surrounded by water moat, with 2 opposing black stripes
    at 0 and 180 deg. Evaluates stripe fixation and centrophobism (avoidance of center).
    """

    def __init__(self, max_duration_steps: int = 1000):
        dimensions = (120.0, 120.0)
        cx, cy = 60.0, 60.0
        self.center = (cx, cy)
        self.platform_radius = 50.0
        self.moat = CircularMoat(center=(cx, cy), radius=self.platform_radius)

        landmarks = [
            VisualLandmark("stripe_0", azimuth_rad=0.0, angular_width_rad=0.21, glyph='stripe', pos=(cx + 50.0, cy)),
            VisualLandmark("stripe_180", azimuth_rad=math.pi, angular_width_rad=0.21, glyph='stripe', pos=(cx - 50.0, cy)),
        ]

        zones = [
            MazeZone("center", "open_field", (cx, cy, 25.0)),
            MazeZone("perimeter", "thigmotaxis", (cx, cy, 50.0)),
        ]

        super().__init__(
            name="buridan",
            description="Buridan Visual Landmark Fixation and Open-Field Centrophobism Assay",
            dimensions=dimensions,
            zones=zones,
            landmarks=landmarks,
            max_duration_steps=max_duration_steps
        )

        self.time_center_ms: float = 0.0
        self.time_perimeter_ms: float = 0.0
        self.fixation_scores = ScalarHistory()
        self.stripe_contrast = 1.0
        self.stripe_crossings: int = 0
        self.last_heading_side: Optional[int] = None

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        bearings = [lm.get_apparent_bearing(x, y, heading) for lm in self.landmarks]
        nearest_dev = min(abs(b) for b in bearings)
        stripe_fixation = math.cos(nearest_dev)

        return {
            'stripe_bearings': bearings,
            'stripe_contrast': self.stripe_contrast,
            'nearest_bearing': min(bearings, key=abs),
            'stripe_fixation': stripe_fixation,
            'is_in_moat': self.moat.is_in_water(x, y),
            'distance_to_center': math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2)
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)

        # Record stripe fixation metric
        self.fixation_scores.append(stimuli['stripe_fixation'])

        # Center vs perimeter dwell
        dist_c = stimuli['distance_to_center']
        if dist_c < 25.0:
            self.time_center_ms += dt * 1000.0
        else:
            self.time_perimeter_ms += dt * 1000.0

        # Track crossings between orienting to stripe 0 vs stripe 180
        bearing_0 = stimuli['stripe_bearings'][0]
        side = 0 if abs(bearing_0) < (math.pi / 2.0) else 1
        if self.last_heading_side is not None and self.last_heading_side != side:
            self.stripe_crossings += 1
        self.last_heading_side = side

        return {
            'stimuli': stimuli,
            'centrophobism_index': self.get_metrics()['centrophobism_index'],
            'mean_stripe_fixation': self.get_metrics()['mean_stripe_fixation'],
            'stripe_crossings': self.stripe_crossings
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.time_center_ms = 0.0
        self.time_perimeter_ms = 0.0
        self.fixation_scores.clear()
        self.stripe_crossings = 0
        self.last_heading_side = None
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        total_time = max(1.0, self.time_center_ms + self.time_perimeter_ms)
        centrophobism = 1.0 - (self.time_center_ms / total_time)
        mean_fix = self.fixation_scores.mean if self.fixation_scores else 0.0

        return {
            'centrophobism_index': centrophobism,
            'mean_stripe_fixation': mean_fix,
            'stripe_crossings': self.stripe_crossings,
            'time_center_pct': (self.time_center_ms / total_time) * 100.0,
            'time_perimeter_pct': (self.time_perimeter_ms / total_time) * 100.0
        }


class VisualOperantParadigm(_VisualOperantV1, ExperimentParadigm):
    """Paradigm 5: Visual Operant Flight Simulator / Yaw Conditioning (Wolf & Heisenberg 1991).

    Tethered fly in 360-deg drum with alternating upright 'T' (safe) and inverted 'T' (punished).
    Closed-loop yaw torque rotation coupled to drum, laser heat punishment in punished quadrants.
    """

    def __init__(self, coupling_gain: float = 120.0, max_duration_steps: int = 1000):
        dimensions = (80.0, 80.0)
        super().__init__(
            name="visual_operant",
            description="Visual Operant Flight Simulator with Closed-Loop Yaw Torque and Laser Punishment",
            dimensions=dimensions,
            max_duration_steps=max_duration_steps
        )
        self.coupling_gain = float(coupling_gain)
        self.drum_angle_deg: float = 0.0  # [0, 360)
        self.invert_sectors = False
        self.time_safe_ms: float = 0.0
        self.time_punished_ms: float = 0.0
        self.laser_heat_active: bool = False
        self.torque_history_safe = ScalarHistory()
        self.torque_history_punished = ScalarHistory()

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        # Quadrants: [0, 90) Safe ('T'), [90, 180) Punished ('_|_'),
        #            [180, 270) Safe ('T'), [270, 360) Punished ('_|_')
        angle = self.drum_angle_deg % 360.0
        quadrant = int(angle // 90.0)
        pattern = 'T' if quadrant in [0, 2] else 'inverted_T'
        is_punished = (pattern == 'inverted_T') != self.invert_sectors

        temp = 41.0 if is_punished else 24.0

        return {
            'drum_angle_deg': angle,
            'quadrant': quadrant,
            'pattern_in_view': pattern,
            'is_punished': is_punished,
            'temperature': temp,
            'laser_active': is_punished
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        # Angular velocity acts as yaw torque
        yaw_torque = float(getattr(fly, 'yaw_torque', angular_vel))

        # Update drum closed-loop rotation
        omega_drum = -self.coupling_gain * yaw_torque
        self.drum_angle_deg = (self.drum_angle_deg + omega_drum * dt) % 360.0

        stimuli = self.sample_stimuli(x, y, heading)
        self.laser_heat_active = stimuli['laser_active']

        if stimuli['is_punished']:
            self.time_punished_ms += dt * 1000.0
            self.torque_history_punished.append(yaw_torque)
            punishment = 1.0
        else:
            self.time_safe_ms += dt * 1000.0
            self.torque_history_safe.append(yaw_torque)
            punishment = 0.0

        return {
            'stimuli': stimuli,
            'yaw_torque': yaw_torque,
            'laser_active': self.laser_heat_active,
            'punishment': punishment,
            'operant_learning_index': self.get_metrics()['operant_learning_index']
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.drum_angle_deg = 0.0
        self.time_safe_ms = 0.0
        self.time_punished_ms = 0.0
        self.laser_heat_active = False
        self.torque_history_safe.clear()
        self.torque_history_punished.clear()
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        total = self.time_safe_ms + self.time_punished_ms
        li = float((self.time_safe_ms - self.time_punished_ms) / total) if total > 0 else 0.0
        mean_t_safe = self.torque_history_safe.mean if self.torque_history_safe else 0.0
        mean_t_punished = self.torque_history_punished.mean if self.torque_history_punished else 0.0

        return {
            'operant_learning_index': li,
            'time_safe_pct': (self.time_safe_ms / max(1.0, total)) * 100.0,
            'time_punished_pct': (self.time_punished_ms / max(1.0, total)) * 100.0,
            'mean_torque_safe': mean_t_safe,
            'mean_torque_punished': mean_t_punished
        }


class WindTunnelParadigm(_WindTunnelV1, ExperimentParadigm):
    """Paradigm 6: Wind Tunnel Odor Plume Tracking (Alvarez-Salvado 2018; Demir 2020).

    200x60mm laminar wind tunnel with downwind flow (-25, 0) mm/s, upstream odor nozzle
    at (180, 30) dispensing Gaussian odor filaments. Evaluates surge-and-cast olfactory navigation.
    """

    def __init__(self, max_duration_steps: int = 1000):
        dimensions = (200.0, 60.0)
        self.wind_flow = (-25.0, 0.0)
        self.nozzle_pos = (180.0, 30.0)
        self.filament_sigma = 3.5

        walls = [
            WallSegment((0.0, 0.0), (200.0, 0.0)),
            WallSegment((0.0, 60.0), (200.0, 60.0)),
            WallSegment((0.0, 0.0), (0.0, 60.0)),
            WallSegment((200.0, 0.0), (200.0, 60.0)),
        ]

        zones = [
            MazeZone("source", "goal", (180.0, 30.0, 6.0), reward=1.0),
        ]

        super().__init__(
            name="wind_tunnel",
            description="Wind Tunnel Surge-and-Cast Anemotactic Plume Navigation",
            dimensions=dimensions,
            walls=walls,
            zones=zones,
            max_duration_steps=max_duration_steps
        )

        self.surge_steps: int = 0
        self.cast_steps: int = 0
        self.source_reached: bool = False
        self.time_to_source_ms: Optional[float] = None
        self.initial_x: Optional[float] = None
        self.last_x: Optional[float] = None

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        # Odor plume extends downwind from nozzle (x <= 180)
        if x <= self.nozzle_pos[0] + 5.0:
            dy = y - self.nozzle_pos[1]
            conc = math.exp(-(dy * dy) / (2.0 * self.filament_sigma * self.filament_sigma))
            # Attenuate slightly downwind
            downwind_dist = max(0.0, self.nozzle_pos[0] - x)
            conc *= math.exp(-downwind_dist / 250.0)
        else:
            conc = 0.0

        conc = _finite_stimulus(conc, 'wind_tunnel.stimuli.odor_conc')
        wind_x = _finite_stimulus(self.wind_flow[0], 'wind_tunnel.stimuli.wind[0]')
        wind_y = _finite_stimulus(self.wind_flow[1], 'wind_tunnel.stimuli.wind[1]')
        return {
            'odor_conc': float(np.clip(conc, 0.0, 1.0)),
            'wind': self.wind_flow,
            'wind_speed': math.hypot(wind_x, wind_y),
            'wind_direction_rad': math.atan2(wind_y, wind_x)  # Wind blows in -x direction
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        if self.initial_x is None:
            self.initial_x = x
        self.last_x = x

        stimuli = self.sample_stimuli(x, y, heading)
        odor_on = (_finite_stimulus(stimuli['odor_conc'],
                                    'wind_tunnel.stimuli.odor_conc') > 0.05)

        if odor_on:
            self.surge_steps += 1
            behavioral_state = "SURGE"
        else:
            self.cast_steps += 1
            behavioral_state = "CAST"

        active_zones = self.get_active_zones(x, y)
        if "source" in [z.name for z in active_zones] and not self.source_reached:
            self.source_reached = True
            self.time_to_source_ms = self.time_elapsed_ms

        return {
            'stimuli': stimuli,
            'behavioral_state': behavioral_state,
            'source_reached': self.source_reached,
            'time_to_source_ms': self.time_to_source_ms,
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.surge_steps = 0
        self.cast_steps = 0
        self.source_reached = False
        self.time_to_source_ms = None
        self.initial_x = None
        self.last_x = None
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        total = self.surge_steps + self.cast_steps
        ratio = float(self.surge_steps / max(1, self.cast_steps))
        upwind_progress = (self.last_x - self.initial_x) if (self.last_x is not None and self.initial_x is not None) else 0.0

        return {
            'surge_to_cast_ratio': ratio,
            'surge_steps': self.surge_steps,
            'cast_steps': self.cast_steps,
            'source_reached': self.source_reached,
            'time_to_source_ms': self.time_to_source_ms,
            'upwind_progress_mm': upwind_progress
        }


class LoomingEscapeParadigm(_LoomingV1, ExperimentParadigm):
    """Paradigm 7: Visual Looming Predator Escape & Takeoff Assay (Card & Dickinson 2008).

    Circular stage with an approaching visual dark disk expanding as theta(t) = 2 arctan(r / (v (t_coll - t))).
    Giant Fiber (GF) membrane potential fires when disk exceeds 65 deg threshold, triggering ballistic escape.
    """

    def __init__(self, t_collision_s: float = 0.400, r_over_v_s: float = 0.020, max_duration_steps: int = 500):
        dimensions = (80.0, 80.0)
        super().__init__(
            name="looming_escape",
            description="Visual Looming Predator Escape and Giant Fiber Ballistic Takeoff",
            dimensions=dimensions,
            max_duration_steps=max_duration_steps
        )
        self.stimulus_started_ms = 0.0
        self.t_collision_s = float(t_collision_s)
        self.r_over_v_s = float(r_over_v_s)
        self.gf_threshold_rad = math.radians(65.0)  # ~1.134 rad
        self.escape_initiated: bool = False
        self.time_to_collision_at_jump_ms: Optional[float] = None
        self.looming_size_at_jump_deg: Optional[float] = None
        self.gf_spike: bool = False
        # 'geometric': the GF fires when the disc passes gf_threshold_rad (modular
        # baseline).  'connectome': the arena reports DNp01 spikes of the brain via
        # record_gf_spike() and the size threshold is not used.
        self.gf_source: str = 'geometric'

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        t = (self.time_elapsed_ms - self.stimulus_started_ms) / 1000.0
        time_to_coll = max(0.001, self.t_collision_s - t)

        # Looming equation: theta(t) = 2 * arctan((r/v) / (t_coll - t))
        theta_rad = 2.0 * math.atan(self.r_over_v_s / time_to_coll)
        expansion_rate = 2.0 * self.r_over_v_s / (self.r_over_v_s * self.r_over_v_s + time_to_coll * time_to_coll)

        # GF membrane potential: resting -70mV, depolarizes towards threshold -20mV
        vm = -70.0 + 55.0 * (theta_rad / self.gf_threshold_rad)

        return {
            'theta_rad': theta_rad,
            'theta_deg': math.degrees(theta_rad),
            'expansion_rate_rad_s': expansion_rate,
            'gf_membrane_potential_mv': float(np.clip(vm, -70.0, 30.0)),
            'time_to_collision_s': time_to_coll
        }

    def step(self, fly: Any, dt: float = 0.01) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)

        if (self.gf_source == 'geometric' and stimuli['theta_rad'] >= self.gf_threshold_rad
                and not self.escape_initiated):
            self._mark_escape(stimuli)
        else:
            self.gf_spike = False

        return {
            'stimuli': stimuli,
            'gf_spike': self.gf_spike,
            'escape_initiated': self.escape_initiated,
            'time_to_collision_at_jump_ms': self.time_to_collision_at_jump_ms,
            'metrics': self.get_metrics()
        }

    def _mark_escape(self, stimuli: Dict[str, Any]) -> None:
        self.escape_initiated = True
        self.gf_spike = True
        self.time_to_collision_at_jump_ms = stimuli['time_to_collision_s'] * 1000.0
        self.looming_size_at_jump_deg = stimuli['theta_deg']

    def record_gf_spike(self) -> None:
        """A DNp01 spike of the connectome brain: the first one of a loom is the jump."""
        actual = self._v1_legacy_out if isinstance(self._v1_legacy_out, dict) else {}
        actual_stimuli = actual.get('stimuli')
        stimuli = (actual_stimuli if isinstance(actual_stimuli, dict)
                   else self.sample_stimuli(0.0, 0.0, 0.0))
        if not self.escape_initiated:
            self._mark_escape(stimuli)
        if isinstance(actual_stimuli, dict):
            self._v1_record_gf_event(actual_stimuli)

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.escape_initiated = False
        self.time_to_collision_at_jump_ms = None
        self.looming_size_at_jump_deg = None
        self.gf_spike = False
        self.time_elapsed_ms = 0.0
        self.stimulus_started_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'escape_initiated': self.escape_initiated,
            'time_to_collision_at_jump_ms': self.time_to_collision_at_jump_ms,
            'looming_size_at_jump_deg': self.looming_size_at_jump_deg,
            'gf_threshold_deg': math.degrees(self.gf_threshold_rad)
        }


class OptomotorParadigm(_OptomotorV1, ExperimentParadigm):
    """Paradigm 8: Optomotor Gaze Stabilization & Saccadic Efference Copy (Götz 1964; Kim 2017).

    Rotating vertical sinusoidal grating drum (R=45mm, omega=30 deg/s).
    Models Horizontal System (HS) wide-field optic flow motion integration
    and efference copy collateral suppression (>80%) during voluntary saccades.
    """

    def __init__(self, drum_velocity_deg_s: float = 30.0, max_duration_steps: int = 1000):
        dimensions = (90.0, 90.0)
        super().__init__(
            name="optomotor",
            description="Optomotor Gaze Stabilization and Saccadic Efference Copy Shunting",
            dimensions=dimensions,
            max_duration_steps=max_duration_steps
        )
        self.drum_velocity_deg_s = float(drum_velocity_deg_s)
        # Declared external stimulus phase; presentation only, never an encoder input.
        self.drum_angle_deg = 0.0
        self.contrast = 0.9
        self.hs_firing_history = ScalarHistory()
        self.retinal_slip_history = ScalarHistory()
        self.gain_history = ScalarHistory()

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        return {
            'drum_velocity_deg_s': self.drum_velocity_deg_s,
            'spatial_wavelength_deg': 30.0,
            'contrast': self.contrast
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()
        # Integrate actual completed stimulus time, preserving reversal continuity.
        # Missing historical phase stays unknown until an explicit presentation reset.
        if self.drum_angle_deg is not None:
            self.drum_angle_deg = (self.drum_angle_deg + self.drum_velocity_deg_s * dt) % 360.0

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        fly_yaw_deg_s = math.degrees(angular_vel)
        is_saccade = bool(getattr(fly, 'is_saccade', abs(fly_yaw_deg_s) > 100.0))

        # Retinal slip = drum velocity - fly yaw velocity
        retinal_slip = self.drum_velocity_deg_s - fly_yaw_deg_s

        # Efference copy shunts >= 80% (85%) of retinal slip during voluntary saccade
        if is_saccade:
            effective_slip = retinal_slip * (1.0 - 0.85)
            efference_copy_active = True
        else:
            effective_slip = retinal_slip
            efference_copy_active = False

        # Baseline HS cell firing rate ~ 40 Hz
        hs_firing = float(np.clip(40.0 + 1.2 * effective_slip * self.contrast, 0.0, 150.0))
        hs_raw = float(np.clip(40.0 + 1.2 * retinal_slip * self.contrast, 0.0, 150.0))

        self.hs_firing_history.append(hs_firing)
        self.retinal_slip_history.append(effective_slip)

        gain = float(fly_yaw_deg_s / self.drum_velocity_deg_s) if abs(self.drum_velocity_deg_s) > 1e-5 else 0.0
        self.gain_history.append(gain)

        return {
            'drum_velocity_deg_s': self.drum_velocity_deg_s,
            'fly_yaw_deg_s': fly_yaw_deg_s,
            'retinal_slip': retinal_slip,
            'effective_slip': effective_slip,
            'hs_firing_rate': hs_firing,
            'hs_firing_unshunted': hs_raw,
            'is_saccade': is_saccade,
            'efference_copy_active': efference_copy_active,
            'optomotor_gain': gain
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.hs_firing_history.clear()
        self.retinal_slip_history.clear()
        self.gain_history.clear()
        self.time_elapsed_ms = 0.0
        self.drum_angle_deg = 0.0  # Explicit trial reset declares a fresh stimulus origin.
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        mean_gain = self.gain_history.mean if self.gain_history else 0.0
        mean_hs = self.hs_firing_history.mean if self.hs_firing_history else 0.0
        mean_slip = self.retinal_slip_history.mean if self.retinal_slip_history else 0.0

        return {
            'optomotor_gain': mean_gain,
            'mean_hs_firing_rate': mean_hs,
            'mean_retinal_slip': mean_slip,
            'efference_copy_shunt_pct': 85.0
        }


class GapCrossingParadigm(_GapCrossingV1, ExperimentParadigm):
    """Paradigm 9: Gap Crossing & Spatial Motor Planning (Pick & Strauss 2005; Triphan 2010).

    Elevated linear track (100x5mm) with adjustable chasm (2.0 to 5.5mm).
    Fly probes chasm with antennae/front legs. Gaps <= 3.8mm trigger step-over;
    gaps > 4.2mm trigger 180-deg abort turn.
    """

    def __init__(self, gap_width_mm: float = 3.5, max_duration_steps: int = 1000):
        dimensions = (100.0, 20.0)
        self.gap_width_mm = float(gap_width_mm)
        self.reachability_threshold_mm = 3.8

        zones = [
            MazeZone("start_track", "track", (0.0, 7.5, 45.0, 12.5)),
            MazeZone("chasm", "hazard", (45.0, 0.0, 45.0 + self.gap_width_mm, 20.0), punishment=1.0),
            MazeZone("landing_track", "goal", (45.0 + self.gap_width_mm, 7.5, 100.0, 12.5), reward=1.0),
        ]

        super().__init__(
            name="gap_crossing",
            description="Gap Crossing Chasm Estimation and Motor Reach Planning",
            dimensions=dimensions,
            zones=zones,
            max_duration_steps=max_duration_steps
        )

        self.decision_outcome: Optional[str] = None  # 'CROSS' or 'ABORT'
        self.crossing_success: bool = False
        self.probing_duration_ms: float = 0.0

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        dist_to_gap = 45.0 - x
        probing = (abs(dist_to_gap) < 3.0)

        # Crossable probability based on threshold 3.8mm
        if self.gap_width_mm <= self.reachability_threshold_mm:
            p_cross = 1.0
        elif self.gap_width_mm >= 4.2:
            p_cross = 0.0
        else:
            p_cross = 1.0 - (self.gap_width_mm - self.reachability_threshold_mm) / 0.4

        return {
            'gap_width_mm': self.gap_width_mm,
            'dist_to_gap_mm': dist_to_gap,
            'is_probing': probing,
            'p_cross': float(p_cross)
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)

        if stimuli['is_probing']:
            self.probing_duration_ms += dt * 1000.0
            if self.decision_outcome is None:
                self.decision_outcome = "CROSS" if stimuli['p_cross'] >= 0.5 else "ABORT"

        active_zones = self.get_active_zones(x, y)
        if "landing_track" in [z.name for z in active_zones]:
            self.crossing_success = True

        return {
            'stimuli': stimuli,
            'decision_outcome': self.decision_outcome,
            'crossing_success': self.crossing_success,
            'probing_duration_ms': self.probing_duration_ms,
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.decision_outcome = None
        self.crossing_success = False
        self.probing_duration_ms = 0.0
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'gap_width_mm': self.gap_width_mm,
            'reachability_threshold_mm': self.reachability_threshold_mm,
            'decision_outcome': self.decision_outcome,
            'crossing_success': self.crossing_success,
            'probing_duration_ms': self.probing_duration_ms
        }


class CircadianDAMParadigm(_CircadianDAMV1, ExperimentParadigm):
    """Paradigm 10: Circadian Locomotor Rhythm & Sleep Deprivation Assay (Konopka 1971; Allada 2010).

    Array of 16 cylindrical activity tubes (5x65mm) with mid-tube infrared beam break (x=32.5mm).
    12:12 Light:Dark cycle. Defines sleep as >= 5 consecutive minutes of immobility.
    """

    def __init__(self, num_tubes: int = 16, photoperiod: str = 'LD', max_duration_steps: int = 1440):
        dimensions = (65.0, 160.0)
        self.num_tubes = num_tubes
        self.photoperiod = photoperiod  # 'LD' or 'DD'

        zones = [
            MazeZone(f"tube_{i}", "tube", (0.0, i * 10.0, 65.0, (i + 1) * 10.0))
            for i in range(num_tubes)
        ]

        super().__init__(
            name="circadian_dam",
            description="Circadian Locomotor Rhythm and 5-min Sleep Bout DAM Assay",
            dimensions=dimensions,
            zones=zones,
            max_duration_steps=max_duration_steps
        )

        self.beam_crossings: int = 0
        self.consecutive_immobile_minutes: float = 0.0
        self.total_sleep_minutes: float = 0.0
        self.sleep_bouts: int = 0
        self.in_sleep_bout: bool = False
        self.last_x: Optional[float] = None

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        # Minute of the 24-hour day (1440 min)
        current_minute = (self.time_elapsed_ms / 60000.0) % 1440
        hour_of_day = current_minute / 60.0

        if self._v1_light_schedule()['mode'] == 'compressed':
            elapsed_us = s_to_us(self.time_elapsed_ms / 1000.0)
            is_lights_on = self._v1_phase(elapsed_us) == 'light'
        elif self.photoperiod == 'LD':
            is_lights_on = (hour_of_day < 12.0)
        else:
            is_lights_on = False

        return {
            'minute_of_day': current_minute,
            'hour_of_day': hour_of_day,
            'is_lights_on': is_lights_on,
            'photoperiod': self.photoperiod
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)

        # Check mid-tube beam crossing at x = 32.5mm
        beam_crossed = False
        if self.last_x is not None:
            if (self.last_x < 32.5 <= x) or (self.last_x > 32.5 >= x):
                beam_crossed = True
                self.beam_crossings += 1
        self.last_x = x

        # Time is in seconds throughout the arena; count real simulated minutes.
        if beam_crossed or abs(speed) > 0.5:
            self.consecutive_immobile_minutes = 0.0
            self.in_sleep_bout = False
        else:
            self.consecutive_immobile_minutes += dt / 60.0
            if self.consecutive_immobile_minutes >= 5.0 - 1e-9:
                if not self.in_sleep_bout:
                    # Include the qualifying immobility interval once it reaches 5 min.
                    self.total_sleep_minutes += self.consecutive_immobile_minutes
                    self.in_sleep_bout = True
                    self.sleep_bouts += 1
                else:
                    self.total_sleep_minutes += dt / 60.0

        return {
            'stimuli': stimuli,
            'beam_crossed': beam_crossed,
            'total_beam_crossings': self.beam_crossings,
            'is_sleeping': self.in_sleep_bout,
            'total_sleep_minutes': self.total_sleep_minutes,
            'sleep_bouts': self.sleep_bouts,
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.beam_crossings = 0
        self.consecutive_immobile_minutes = 0.0
        self.total_sleep_minutes = 0.0
        self.sleep_bouts = 0
        self.in_sleep_bout = False
        self.last_x = None
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        mean_bout = (self.total_sleep_minutes / self.sleep_bouts) if self.sleep_bouts > 0 else 0.0
        return {
            'total_beam_crossings': self.beam_crossings,
            'total_sleep_minutes': self.total_sleep_minutes,
            'sleep_bouts_count': self.sleep_bouts,
            'mean_sleep_bout_length_min': mean_bout
        }


class CourtshipParadigm(_CourtshipV1, ExperimentParadigm):
    """Paradigm 11: Courtship Conditioning & Pheromone Memory (Siegel & Hall 1979; Keleman 2007).

    Circular courtship chamber (R=8.5mm, as drawn by the dashboard), male and female fly (virgin or mated).
    Mated female emits cVA anti-aphrodisiac and delivers rejection kicks,
    inducing dopaminergic suppression of male courtship song and wing extension.
    """

    def __init__(self, female_type: str = 'mated', max_duration_steps: int = 1000):
        dimensions = (20.0, 20.0)
        cx, cy = 10.0, 10.0
        self.chamber_center = (cx, cy)
        self.chamber_radius = 8.5
        self.female_type = female_type  # 'virgin' or 'mated'
        self.female_pos = (cx + 1.5, cy + 1.0)

        self.moat = CircularMoat(center=(cx, cy), radius=self.chamber_radius)

        super().__init__(
            name="courtship",
            description="Courtship Conditioning, Male Wing Extension Song, and cVA Suppression Assay",
            dimensions=dimensions,
            max_duration_steps=max_duration_steps
        )

        self.courtship_active_steps: int = 0
        self.total_steps: int = 0
        self.rejection_kicks: int = 0
        self.wing_extension_angle_deg: float = 0.0

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        dx = self.female_pos[0] - x
        dy = self.female_pos[1] - y
        dist = math.sqrt(dx * dx + dy * dy)

        # Receptive female emits aphrodisiac; mated female emits cVA
        cva_conc = math.exp(-dist / 3.0) if self.female_type == 'mated' else 0.0
        aphrodisiac_conc = math.exp(-dist / 3.0) if self.female_type == 'virgin' else 0.0

        return {
            'inter_fly_distance_mm': dist,
            'social_bearing_rad': math.atan2(math.sin(math.atan2(dy, dx)-heading), math.cos(math.atan2(dy, dx)-heading)),
            'cva_concentration': float(np.clip(cva_conc, 0.0, 1.0)),
            'aphrodisiac_concentration': float(np.clip(aphrodisiac_conc, 0.0, 1.0)),
            'female_type': self.female_type
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()
        self.total_steps += 1

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)
        dist = stimuli['inter_fly_distance_mm']

        # Male courts if close (dist < 3.5mm)
        courtship_active = (getattr(fly, 'behavioral_state', '') == 'COURTSHIP') if not isinstance(fly, dict) else (dist < 3.5)
        punishment = 0.0

        if courtship_active:
            self.courtship_active_steps += 1
            self.wing_extension_angle_deg = min(90.0, 30.0 + (3.5 - dist) * 20.0)

            # Mated female delivers rejection kicks when male approaches very close
            if self.female_type == 'mated' and dist < 2.0:
                self.rejection_kicks += 1
                punishment = 1.0
        else:
            self.wing_extension_angle_deg = 0.0

        return {
            'stimuli': stimuli,
            'courtship_active': courtship_active,
            'wing_extension_angle_deg': self.wing_extension_angle_deg,
            'rejection_kicks': self.rejection_kicks,
            'punishment': punishment,
            'courtship_index': self.get_metrics()['courtship_index']
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.courtship_active_steps = 0
        self.total_steps = 0
        self.rejection_kicks = 0
        self.wing_extension_angle_deg = 0.0
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        ci = float(self.courtship_active_steps / max(1, self.total_steps))
        return {
            'courtship_index': ci,
            'courtship_index_pct': ci * 100.0,
            'rejection_kicks_count': self.rejection_kicks,
            'female_type': self.female_type
        }


class LabyrinthParadigm(_LabyrinthV1, ExperimentParadigm):
    """Paradigm 12: Corridor Obstacle Labyrinth (Sliding Collision Physics).

    140x100mm multi-junction labyrinth with 12 internal wall segments forming
    4 decision junctions, 3 dead ends, and a food goal chamber at (130, 85).
    Continuous sliding collisions with friction mu=0.5 and restitution eps=0.1.
    """

    def __init__(self, max_duration_steps: int = 1500):
        dimensions = (140.0, 100.0)

        # 4 Outer bounding walls
        walls = [
            WallSegment((0.0, 0.0), (140.0, 0.0)),
            WallSegment((140.0, 0.0), (140.0, 100.0)),
            WallSegment((140.0, 100.0), (0.0, 100.0)),
            WallSegment((0.0, 100.0), (0.0, 0.0)),
        ]

        # 12 Internal wall segments forming 4 junctions, 3 dead ends, and corridor width ~ 12mm
        internal_walls = [
            WallSegment((25.0, 0.0), (25.0, 70.0)),      # Wall 1
            WallSegment((25.0, 70.0), (50.0, 70.0)),     # Wall 2
            WallSegment((50.0, 30.0), (50.0, 70.0)),     # Wall 3 (Junction 1 fork)
            WallSegment((50.0, 30.0), (75.0, 30.0)),     # Wall 4
            WallSegment((75.0, 0.0), (75.0, 55.0)),      # Wall 5 (Junction 2 fork)
            WallSegment((75.0, 55.0), (100.0, 55.0)),    # Wall 6
            WallSegment((100.0, 20.0), (100.0, 55.0)),   # Wall 7 (Junction 3 fork)
            WallSegment((100.0, 20.0), (125.0, 20.0)),   # Wall 8
            WallSegment((125.0, 20.0), (125.0, 70.0)),   # Wall 9
            WallSegment((100.0, 80.0), (140.0, 80.0)),   # Wall 10 (Junction 4 / goal partition)
            WallSegment((50.0, 85.0), (100.0, 85.0)),    # Wall 11
            WallSegment((100.0, 70.0), (100.0, 80.0)),   # Wall 12 (Dead end 3)
        ]
        walls.extend(internal_walls)

        # Zones
        goal_pos = (130.0, 85.0)
        zones = [
            MazeZone("goal", "food", (120.0, 75.0, 140.0, 95.0), reward=1.0),
            MazeZone("dead_end_1", "dead_end", (25.0, 70.0, 50.0, 85.0)),
            MazeZone("dead_end_2", "dead_end", (75.0, 55.0, 100.0, 70.0)),
            MazeZone("dead_end_3", "dead_end", (100.0, 0.0, 125.0, 20.0)),
        ]

        super().__init__(
            name="labyrinth",
            description="12-Wall Multi-Junction Corridor Labyrinth with Zero-Tunneling Sliding Physics",
            dimensions=dimensions,
            walls=walls,
            zones=zones,
            max_duration_steps=max_duration_steps
        )

        self.goal_pos = goal_pos
        self.goal_reached: bool = False
        self.time_to_goal_ms: Optional[float] = None
        self.wall_collision_count: int = 0
        self.dead_end_entries: int = 0
        self.path_points = PathHistory()

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        # Continuous odor diffusion from food goal chamber
        dx = self.goal_pos[0] - x
        dy = self.goal_pos[1] - y
        dist = math.sqrt(dx * dx + dy * dy)
        odor_conc = float(np.clip(math.exp(-dist / 35.0), 0.0, 1.0))

        return {
            'odor_conc': odor_conc,
            'temperature': 24.0,
            'wind': (0.0, 0.0),
            'goal_distance_mm': dist,
            'food_contact': dist < 4.0
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        self.path_points.append((x, y))

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        resolved_x, resolved_y, new_vx, new_vy, collided = self.check_collisions(x, y, vx, vy, radius=1.5)

        if collided:
            self.wall_collision_count += 1

        stimuli = self.sample_stimuli(resolved_x, resolved_y, heading)
        active_zones = self.get_active_zones(resolved_x, resolved_y)
        zone_names = [z.name for z in active_zones]

        if "goal" in zone_names and not self.goal_reached:
            self.goal_reached = True
            self.time_to_goal_ms = self.time_elapsed_ms
            reward = 1.0
        else:
            reward = 0.0

        for zn in zone_names:
            if zn.startswith("dead_end"):
                self.dead_end_entries += 1
                break

        return {
            'stimuli': stimuli,
            'active_zones': zone_names,
            'resolved_position': (resolved_x, resolved_y),
            'collided': collided,
            'goal_reached': self.goal_reached,
            'time_to_goal_ms': self.time_to_goal_ms,
            'reward': reward,
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.goal_reached = False
        self.time_to_goal_ms = None
        self.wall_collision_count = 0
        self.dead_end_entries = 0
        self.path_points.clear()
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        # Path tortuosity = actual path length / straight line distance
        if isinstance(self.path_points, PathHistory):
            total_dist = self.path_points.distance
            tortuosity = self.path_points.tortuosity
        elif len(self.path_points) > 1:
            total_dist = sum(
                math.sqrt((self.path_points[i][0] - self.path_points[i - 1][0]) ** 2 +
                          (self.path_points[i][1] - self.path_points[i - 1][1]) ** 2)
                for i in range(1, len(self.path_points))
            )
            net_disp = math.sqrt(
                (self.path_points[-1][0] - self.path_points[0][0]) ** 2 +
                (self.path_points[-1][1] - self.path_points[0][1]) ** 2
            )
            tortuosity = float(total_dist / net_disp) if net_disp > 1e-9 else None
        else:
            total_dist = 0.0
            tortuosity = None

        return {
            'goal_reached': self.goal_reached,
            'time_to_goal_ms': self.time_to_goal_ms,
            'wall_collision_count': self.wall_collision_count,
            'dead_end_entries': self.dead_end_entries,
            'total_distance_mm': total_dist,
            'path_tortuosity': tortuosity
        }



class MultisensoryLimbBenchmark(_MultisensoryV1, ExperimentParadigm):
    """Assay 13: Multisensory Ingress & Full-Body Limb Biomechanics Benchmark.

    A comprehensive closed-loop benchmark integrating:
    1. Multi-Sensory Ingress:
       - 3 olfactory plumes: Food Odor A (+45, +45), Repellent Odor B (-45, -45), cVA pheromone (+45, -45).
       - 2D continuous thermal terrain: Hotspot (38.5C) at (-40, +40), Cool Refuge (22.0C) at (+45, +45), ambient 24.0C.
       - Vector anemotaxis wind field [-15.0, 0.0] mm/s with Johnston's organ deflection.
       - 4 contrasting chromatic visual pillars at (+-35, +-35) with parallax landmark tracking.
       - Dynamic looming threat disc expanding optically towards fly.
    2. Full-Body 6-Leg Biomechanics & Proprioception:
       - 6 articulated legs (L1, L2, L3, R1, R2, R3) with Coxa-Trochanter, Femur-Tibia, and Tibia-Tarsus joint angles.
       - Femoral Chordotonal Organ (FeCO) angular velocity & Campaniform Sensilla (CS) cuticular load feedback.
       - Coupled Kuramoto-Hopf tripod gait with adjustable frequency (3-14 Hz) and stance/swing ratio.
    3. Direct Manual Neuro-Stimulation & Override Deck:
       - DNa02 asymmetric steering yaw torque override.
       - DNp09 forward pursuit thrust override.
       - MDN moonwalker backward walking drive override.
       - DNp01 / Giant Fiber (GF) emergency escape jump trigger.
       - Direct 6-leg stance/swing phase offsets and wing extension angles.
    4. Quantitative Sensorimotor Benchmark Suite:
       - Locomotor Coordination Index (tripod anti-phase coherence).
       - Multi-Sensory Integration Score (sensory gradient alignment).
       - Biomechanical Efficiency (speed per unit metabolic power).
       - Kinematic Smoothness (jerk minimization).
       - Compound Benchmark Score (composite 0 - 100).
    """

    def __init__(
        self,
        dimensions: Tuple[float, float] = (160.0, 160.0),
        arena_radius: float = 75.0,
        pillar_radius: float = 6.0,
        max_duration_steps: int = 1500
    ):
        self.arena_radius = float(arena_radius)
        self.pillar_radius = float(pillar_radius)

        walls: List[WallSegment] = []
        # 32-segment circular boundary wall
        n_outer = 32
        for i in range(n_outer):
            a1 = (2 * math.pi * i) / n_outer
            a2 = (2 * math.pi * (i + 1)) / n_outer
            p1 = (self.arena_radius * math.cos(a1), self.arena_radius * math.sin(a1))
            p2 = (self.arena_radius * math.cos(a2), self.arena_radius * math.sin(a2))
            walls.append(WallSegment(p1, p2, friction=0.5, restitution=0.1))

        # 4 internal visual pillar obstacles at (+-35, +-35)
        self.pillar_centers = [
            (35.0, 35.0), (-35.0, 35.0), (-35.0, -35.0), (35.0, -35.0)
        ]
        n_p = 8
        for pc in self.pillar_centers:
            for i in range(n_p):
                a1 = (2 * math.pi * i) / n_p
                a2 = (2 * math.pi * (i + 1)) / n_p
                p1 = (pc[0] + self.pillar_radius * math.cos(a1), pc[1] + self.pillar_radius * math.sin(a1))
                p2 = (pc[0] + self.pillar_radius * math.cos(a2), pc[1] + self.pillar_radius * math.sin(a2))
                walls.append(WallSegment(p1, p2, friction=0.5, restitution=0.1))

        # Interactive sensory zones
        zones = [
            MazeZone("food_refuge", "circle", (45.0, 45.0, 12.0), reward=1.0),
            MazeZone("thermal_hotspot", "circle", (-40.0, 40.0, 15.0), punishment=1.0),
            MazeZone("pheromone_zone", "circle", (45.0, -45.0, 12.0), reward=0.5),
            MazeZone("open_arena", "rectangle", (-75.0, -75.0, 75.0, 75.0))
        ]

        # Visual landmarks corresponding to pillars
        landmarks = [
            VisualLandmark("pillar_ne", azimuth_rad=math.pi / 4, glyph="cylinder", pos=(35.0, 35.0)),
            VisualLandmark("pillar_nw", azimuth_rad=3 * math.pi / 4, glyph="cylinder", pos=(-35.0, 35.0)),
            VisualLandmark("pillar_sw", azimuth_rad=-3 * math.pi / 4, glyph="cylinder", pos=(-35.0, -35.0)),
            VisualLandmark("pillar_se", azimuth_rad=-math.pi / 4, glyph="cylinder", pos=(35.0, -35.0)),
        ]

        super().__init__(
            name="Multisensory Limb & Body Benchmark",
            description="Comprehensive closed-loop benchmark integrating multi-sensory ingress with direct 6-limb biomechanics.",
            dimensions=dimensions,
            walls=walls,
            zones=zones,
            landmarks=landmarks,
            max_duration_steps=max_duration_steps
        )

        # Multi-sensory positions
        self.food_pos = (45.0, 45.0)
        self.repellent_pos = (-45.0, -45.0)
        self.pheromone_pos = (45.0, -45.0)
        self.hotspot_pos = (-40.0, 40.0)
        self.cool_pos = (45.0, 45.0)
        self.wind_vector = [-15.0, 0.0]

        # 6-leg biomechanical state
        self.leg_names = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
        self.leg_phases = {
            'L1': 0.0, 'R2': 0.0, 'L3': 0.0,
            'R1': math.pi, 'L2': math.pi, 'R3': math.pi
        }
        self.cpg_freq_hz = 8.5
        self.leg_states = {name: math.sin(phi) > 0 for name, phi in self.leg_phases.items()}
        self.joint_angles = {
            name: {'ctr': 0.0, 'fti': 85.0, 'tita': 35.0} for name in self.leg_names
        }
        self.cuticular_loads = {name: 1.85 if self.leg_states[name] else 0.0 for name in self.leg_names}

        # Benchmark metrics
        self.path_points: List[Tuple[float, float]] = []
        self.coordination_history: List[float] = []
        self.sensory_alignment_history: List[float] = []
        self.jerk_history: List[float] = []
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.wall_collision_count = 0
        self.prev_speed = 0.0
        self.prev_accel = 0.0

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        fx, fy, fheading = self._normalize_stimuli_args(x, y, heading)

        # 1. Olfactory plumes (Gaussian diffusion)
        da = math.hypot(fx - self.food_pos[0], fy - self.food_pos[1])
        db = math.hypot(fx - self.repellent_pos[0], fy - self.repellent_pos[1])
        dc = math.hypot(fx - self.pheromone_pos[0], fy - self.pheromone_pos[1])

        odor_a = _finite_stimulus(float(math.exp(-(da * da) / (2 * 20.0 * 20.0))),
                                  'multisensory.stimuli.odor_a')
        odor_b = _finite_stimulus(float(math.exp(-(db * db) / (2 * 20.0 * 20.0))),
                                  'multisensory.stimuli.odor_b')
        odor_cva = _finite_stimulus(float(0.8 * math.exp(-(dc * dc) / (2 * 18.0 * 18.0))),
                                    'multisensory.stimuli.odor_cva')

        # 2. Thermal terrain (ambient 24.0, hotspot 38.5, cool refuge 22.0)
        dh = math.hypot(fx - self.hotspot_pos[0], fy - self.hotspot_pos[1])
        d_cool = math.hypot(fx - self.cool_pos[0], fy - self.cool_pos[1])
        temp_hot = 14.5 * math.exp(-(dh * dh) / (2 * 18.0 * 18.0))
        temp_cool = -2.0 * math.exp(-(d_cool * d_cool) / (2 * 12.0 * 12.0))
        raw_temperature = _finite_stimulus(24.0 + temp_hot + temp_cool,
                                           'multisensory.stimuli.temperature')
        temperature = float(max(20.0, min(42.0, raw_temperature)))

        # 3. Mechanosensory wind
        wind_x = _finite_stimulus(self.wind_vector[0], 'multisensory.stimuli.wind[0]')
        wind_y = _finite_stimulus(self.wind_vector[1], 'multisensory.stimuli.wind[1]')
        wind_mag = math.hypot(wind_x, wind_y)
        wind_angle = math.atan2(wind_y, wind_x)
        upwind_angle = math.atan2(-wind_y, -wind_x)
        egocentric_wind = ((upwind_angle - fheading + math.pi) % (2 * math.pi)) - math.pi
        jo_antenna_deflect_un = float(wind_mag * 0.12)

        # 4. Visual landmarks
        landmarks_data = []
        for lm in self.landmarks:
            bearing = lm.get_apparent_bearing(fx, fy, fheading)
            d_lm = math.hypot(lm.pos[0] - fx, lm.pos[1] - fy) if lm.pos else 50.0
            landmarks_data.append({
                'id': lm.landmark_id,
                'distance': float(d_lm),
                'bearing_rad': float(bearing),
                'glyph': lm.glyph
            })

        return {
            'odor_a': odor_a,
            'odor_b': odor_b,
            'odor_cva': odor_cva,
            'cva_concentration': odor_cva,
            'wind': self.wind_vector,
            'food_contact': da < 4.0,
            'temperature': temperature,
            'wind_magnitude': float(wind_mag),
            'egocentric_wind': float(egocentric_wind),
            'jo_antenna_deflect_un': jo_antenna_deflect_un,
            'landmarks': landmarks_data
        }

    def step(self, fly: Any, dt: float = 0.02, **kwargs) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()
        fx, fy, fheading, fspeed, fang_vel = self._extract_fly_pose(fly)

        # Check for direct manual control overrides
        override_dna02 = kwargs.get('override_dna02', None)
        override_thrust = kwargs.get('override_thrust', None)
        override_mdn = kwargs.get('override_mdn', None)
        override_gf = kwargs.get('override_gf', False)
        override_legs = kwargs.get('override_legs', None)
        override_wings = kwargs.get('override_wings', None)
        manual_active = any(k is not None for k in [override_dna02, override_thrust, override_mdn, override_legs])

        # Step CPG biomechanical limbs
        effective_drive = override_thrust * 50.0 if override_thrust is not None else max(10.0, fspeed * 3.0)
        self.cpg_freq_hz = min(14.0, max(3.0, 6.0 + effective_drive * 0.12))
        dphi = 2 * math.pi * self.cpg_freq_hz * dt

        for name in self.leg_names:
            if override_legs and name in override_legs:
                self.leg_states[name] = bool(override_legs[name])
            else:
                self.leg_phases[name] = (self.leg_phases[name] + dphi) % (2 * math.pi)
                self.leg_states[name] = math.sin(self.leg_phases[name]) > 0

            # Kinematic joint angles
            phi = self.leg_phases[name]
            self.joint_angles[name] = {
                'ctr': float(22.0 * math.sin(phi)),
                'fti': float(80.0 + 32.0 * math.cos(phi)),
                'tita': float(38.0 - 14.0 * math.sin(phi))
            }
            self.cuticular_loads[name] = 1.85 if self.leg_states[name] else 0.0

        # Collision resolution
        proposed_vx = fspeed * math.cos(fheading)
        proposed_vy = fspeed * math.sin(fheading)
        res_x, res_y, res_vx, res_vy, collided = self.check_collisions(
            fx + proposed_vx * dt, fy + proposed_vy * dt, proposed_vx, proposed_vy, radius=1.5
        )
        if collided:
            self.wall_collision_count += 1

        # Distance & Trajectory
        step_dist = math.hypot(res_x - fx, res_y - fy)
        self.total_distance += step_dist
        self.path_points.append((res_x, res_y))
        if len(self.path_points) > 300:
            self.path_points.pop(0)

        # Benchmark metrics:
        phase_diff = abs((self.leg_phases['L1'] - self.leg_phases['R1'] + math.pi) % (2 * math.pi) - math.pi)
        coordination = float(max(0.0, 1.0 - abs(phase_diff - math.pi) / math.pi))
        self.coordination_history.append(coordination)
        if len(self.coordination_history) > 100: self.coordination_history.pop(0)

        stim = self.sample_stimuli(res_x, res_y, fheading)
        desirable_dir = math.atan2(self.food_pos[1] - res_y, self.food_pos[0] - res_x)
        dir_error = abs(((desirable_dir - fheading + math.pi) % (2 * math.pi)) - math.pi)
        sensory_alignment = float(math.cos(dir_error * 0.5))
        self.sensory_alignment_history.append(sensory_alignment)
        if len(self.sensory_alignment_history) > 100: self.sensory_alignment_history.pop(0)

        accel = (fspeed - self.prev_speed) / max(1e-4, dt)
        jerk = abs(accel - self.prev_accel) / max(1e-4, dt)
        self.prev_speed = fspeed
        self.prev_accel = accel
        self.jerk_history.append(jerk)
        if len(self.jerk_history) > 100: self.jerk_history.pop(0)

        power = (self.cpg_freq_hz * 0.8 + fspeed * 0.5) * dt
        self.total_energy += power

        odor_a = _finite_stimulus(stim['odor_a'], 'multisensory.stimuli.odor_a')
        odor_b = _finite_stimulus(stim['odor_b'], 'multisensory.stimuli.odor_b')
        temperature = _finite_stimulus(stim['temperature'], 'multisensory.stimuli.temperature')
        reward = 1.0 if odor_a > 0.6 and temperature < 25.0 else 0.0
        punishment = 1.0 if temperature > 35.0 or odor_b > 0.5 else 0.0

        return {
            'stimuli': stim,
            'active_zones': [z.name for z in self.get_active_zones(res_x, res_y)],
            'resolved_position': (res_x, res_y),
            'collided': collided,
            'reward': reward,
            'punishment': punishment,
            'biomechanics': {
                'cpg_freq_hz': self.cpg_freq_hz,
                'leg_states': self.leg_states,
                'joint_angles': self.joint_angles,
                'cuticular_loads': self.cuticular_loads,
                'manual_override_active': manual_active
            },
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.time_elapsed_ms = 0.0
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.wall_collision_count = 0
        self.path_points.clear()
        self.coordination_history.clear()
        self.sensory_alignment_history.clear()
        self.jerk_history.clear()
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        mean_coord = float(sum(self.coordination_history) / len(self.coordination_history)) if self.coordination_history else 1.0
        mean_sensory = float(sum(self.sensory_alignment_history) / len(self.sensory_alignment_history)) if self.sensory_alignment_history else 0.85
        mean_jerk = float(sum(self.jerk_history) / len(self.jerk_history)) if self.jerk_history else 0.0
        smoothness = float(1.0 / (1.0 + mean_jerk * 0.005))
        efficiency = float(min(1.0, self.total_distance / max(1.0, self.total_energy * 10.0)))

        composite_score = float(max(0.0, min(100.0,
            (0.30 * mean_coord + 0.25 * mean_sensory + 0.25 * efficiency + 0.20 * smoothness) * 100.0
        )))

        return {
            'locomotor_coordination_index': mean_coord,
            'multisensory_integration_score': mean_sensory,
            'biomechanical_efficiency': efficiency,
            'kinematic_smoothness': smoothness,
            'composite_benchmark_score': composite_score,
            'wall_collisions': self.wall_collision_count,
            'total_distance_mm': self.total_distance,
            'total_energy_atp': self.total_energy
        }


# ==============================================================================
# 4. EXPERIMENT REGISTRY FACTORY
# ==============================================================================

class ExperimentRegistry:
    """Central registry and factory for all 12 neuroethological experiment paradigms."""

    _registry: Dict[str, Type[ExperimentParadigm]] = {}

    STANDARD_PARADIGMS: List[str] = [
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
        "multisensory_benchmark",
    ]

    @classmethod
    def register(cls, name: str, paradigm_cls: Type[ExperimentParadigm]):
        canonical = cls._canonical_name(name)
        cls._registry[canonical] = paradigm_cls

    @classmethod
    def get(cls, name: str, **kwargs) -> ExperimentParadigm:
        canonical = cls._canonical_name(name)
        if canonical not in cls._registry:
            raise KeyError(
                f"Paradigm '{name}' (canonical: '{canonical}') not found in registry. "
                f"Available paradigms: {cls.list_paradigms()}"
            )
        return cls._registry[canonical](**kwargs)

    @classmethod
    def list_paradigms(cls) -> List[str]:
        return list(cls.STANDARD_PARADIGMS)

    @staticmethod
    def _canonical_name(name: str) -> str:
        s = name.strip().lower().replace("paradigm", "").replace("_", "").replace("-", "").replace(" ", "")
        return s


# Auto-register all 12 canonical paradigms with both snake_case, ClassName, and canonical forms
for p_cls in [
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
    MultisensoryLimbBenchmark,
]:
    ExperimentRegistry.register(p_cls.__name__, p_cls)
    ExperimentRegistry.register(p_cls.__name__.lower().replace("paradigm", "").replace("benchmark", ""), p_cls)

ExperimentRegistry.register("multisensory_benchmark", MultisensoryLimbBenchmark)
ExperimentRegistry.register("multisensory_sandbox", MultisensoryLimbBenchmark)
ExperimentRegistry.register("multisensory", MultisensoryLimbBenchmark)
