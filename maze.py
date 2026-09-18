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

    def resolve_circle_collision(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        radius: float
    ) -> Tuple[float, float, float, float, bool, Tuple[float, float]]:
        """Resolve continuous circle collision against this wall segment.

        Args:
            x, y: Center coordinates of circle.
            vx, vy: Incoming velocity vector.
            radius: Radius of colliding circle.

        Returns:
            (new_x, new_y, new_vx, new_vy, collided, (normal_x, normal_y))
        """
        cx, cy, t = self.project_point(x, y)
        dx = x - cx
        dy = y - cy
        dist = math.sqrt(dx * dx + dy * dy)

        if dist >= radius:
            return x, y, vx, vy, False, (0.0, 0.0)

        # Collision detected!
        penetration = radius - dist

        # Contact normal pointing from closest point to circle center
        if dist > 1e-8:
            cn_x = dx / dist
            cn_y = dy / dist
        else:
            cn_x = self.nx
            cn_y = self.ny

        # Push circle center out along contact normal to eliminate penetration
        resolved_x = x + cn_x * penetration
        resolved_y = y + cn_y * penetration

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

    def __init__(self, walls: Optional[List[WallSegment]] = None):
        self.walls: List[WallSegment] = list(walls) if walls is not None else []

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
        max_iterations: int = 4
    ) -> Tuple[float, float, float, float, bool, List[Tuple[float, float]]]:
        """Continuously resolve circle-wall collisions with zero tunneling across multiple walls.

        Iteratively solves segment penetrations and rebounds until all collisions are resolved
        or max_iterations is reached.

        Returns:
            (resolved_x, resolved_y, resolved_vx, resolved_vy, collided_any, contact_normals)
        """
        cur_x, cur_y = x, y
        cur_vx, cur_vy = vx, vy
        collided_any = False
        normals: List[Tuple[float, float]] = []

        for _ in range(max_iterations):
            collision_in_pass = False
            for wall in self.walls:
                cx, cy, cvx, cvy, collided, norm = wall.resolve_circle_collision(
                    cur_x, cur_y, cur_vx, cur_vy, radius
                )
                if collided:
                    collision_in_pass = True
                    collided_any = True
                    cur_x, cur_y = cx, cy
                    cur_vx, cur_vy = cvx, cvy
                    normals.append(norm)

            if not collision_in_pass:
                break

        return cur_x, cur_y, cur_vx, cur_vy, collided_any, normals


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

class ExperimentParadigm(ABC):
    """Abstract Base Class for all Drosophila Neuroethological Experiment Paradigms."""

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
        radius: float = 1.5
    ) -> Tuple[float, float, float, float, bool]:
        """Check and resolve circle collisions against paradigm walls with sliding physics."""
        new_x, new_y, new_vx, new_vy, collided, _ = self.collision_engine.resolve(
            x, y, vx, vy, radius=radius
        )
        return new_x, new_y, new_vx, new_vy, collided

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
# 3. 12 CONCRETE EXPERIMENT PARADIGMS
# ==============================================================================

class TMazeParadigm(ExperimentParadigm):
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
        self.first_choice: Optional[str] = None
        self.latency_to_choice_ms: Optional[float] = None
        self.step_history: List[str] = []

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
            'temperature': 24.0
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        stimuli = self.sample_stimuli(x, y, heading)
        active_zones = self.get_active_zones(x, y)

        reward = sum(z.reward for z in active_zones)
        punishment = sum(z.punishment for z in active_zones)

        # Check arm choices
        active_zone_names = [z.name for z in active_zones]
        if "arm_a" in active_zone_names and x < 45.0:
            if self.first_choice is None:
                self.first_choice = "arm_a"
                self.latency_to_choice_ms = self.time_elapsed_ms
            self.choice_counts['arm_a'] += 1
            self.step_history.append("arm_a")
        elif "arm_b" in active_zone_names and x > 95.0:
            if self.first_choice is None:
                self.first_choice = "arm_b"
                self.latency_to_choice_ms = self.time_elapsed_ms
            self.choice_counts['arm_b'] += 1
            self.step_history.append("arm_b")

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


class YMazeParadigm(ExperimentParadigm):
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


class HeatMazeParadigm(ExperimentParadigm):
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
        self.path_points: List[Tuple[float, float]] = []

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        temp = self.peltier.get_temperature(x, y)
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
        temp = stimuli['temperature']
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
            'path_length': float(len(self.path_points))
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


class BuridanParadigm(ExperimentParadigm):
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
        self.fixation_scores: List[float] = []
        self.stripe_crossings: int = 0
        self.last_heading_side: Optional[int] = None

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        bearings = [lm.get_apparent_bearing(x, y, heading) for lm in self.landmarks]
        nearest_dev = min(abs(b) for b in bearings)
        stripe_fixation = math.cos(nearest_dev)

        return {
            'stripe_bearings': bearings,
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
        mean_fix = float(np.mean(self.fixation_scores)) if self.fixation_scores else 0.0

        return {
            'centrophobism_index': centrophobism,
            'mean_stripe_fixation': mean_fix,
            'stripe_crossings': self.stripe_crossings,
            'time_center_pct': (self.time_center_ms / total_time) * 100.0,
            'time_perimeter_pct': (self.time_perimeter_ms / total_time) * 100.0
        }


class VisualOperantParadigm(ExperimentParadigm):
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
        self.time_safe_ms: float = 0.0
        self.time_punished_ms: float = 0.0
        self.laser_heat_active: bool = False
        self.torque_history_safe: List[float] = []
        self.torque_history_punished: List[float] = []

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        # Quadrants: [0, 90) Safe ('T'), [90, 180) Punished ('_|_'),
        #            [180, 270) Safe ('T'), [270, 360) Punished ('_|_')
        angle = self.drum_angle_deg % 360.0
        quadrant = int(angle // 90.0)
        pattern = 'T' if quadrant in [0, 2] else 'inverted_T'
        is_punished = (pattern == 'inverted_T')

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
        mean_t_safe = float(np.mean(self.torque_history_safe)) if self.torque_history_safe else 0.0
        mean_t_punished = float(np.mean(self.torque_history_punished)) if self.torque_history_punished else 0.0

        return {
            'operant_learning_index': li,
            'time_safe_pct': (self.time_safe_ms / max(1.0, total)) * 100.0,
            'time_punished_pct': (self.time_punished_ms / max(1.0, total)) * 100.0,
            'mean_torque_safe': mean_t_safe,
            'mean_torque_punished': mean_t_punished
        }


class WindTunnelParadigm(ExperimentParadigm):
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

        return {
            'odor_conc': float(np.clip(conc, 0.0, 1.0)),
            'wind': self.wind_flow,
            'wind_speed': 25.0,
            'wind_direction_rad': math.pi  # Wind blows in -x direction
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

        x, y, heading, speed, angular_vel = self._extract_fly_pose(fly)
        if self.initial_x is None:
            self.initial_x = x
        self.last_x = x

        stimuli = self.sample_stimuli(x, y, heading)
        odor_on = (stimuli['odor_conc'] > 0.05)

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


class LoomingEscapeParadigm(ExperimentParadigm):
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
        self.t_collision_s = float(t_collision_s)
        self.r_over_v_s = float(r_over_v_s)
        self.gf_threshold_rad = math.radians(65.0)  # ~1.134 rad
        self.escape_initiated: bool = False
        self.time_to_collision_at_jump_ms: Optional[float] = None
        self.looming_size_at_jump_deg: Optional[float] = None
        self.gf_spike: bool = False

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        t = self.time_elapsed_ms / 1000.0
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

        if stimuli['theta_rad'] >= self.gf_threshold_rad and not self.escape_initiated:
            self.escape_initiated = True
            self.gf_spike = True
            self.time_to_collision_at_jump_ms = stimuli['time_to_collision_s'] * 1000.0
            self.looming_size_at_jump_deg = stimuli['theta_deg']
        else:
            self.gf_spike = False

        return {
            'stimuli': stimuli,
            'gf_spike': self.gf_spike,
            'escape_initiated': self.escape_initiated,
            'time_to_collision_at_jump_ms': self.time_to_collision_at_jump_ms,
            'metrics': self.get_metrics()
        }

    def reset_trial(self) -> Dict[str, Any]:
        self.trial_manager.reset()
        self.escape_initiated = False
        self.time_to_collision_at_jump_ms = None
        self.looming_size_at_jump_deg = None
        self.gf_spike = False
        self.time_elapsed_ms = 0.0
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'escape_initiated': self.escape_initiated,
            'time_to_collision_at_jump_ms': self.time_to_collision_at_jump_ms,
            'looming_size_at_jump_deg': self.looming_size_at_jump_deg,
            'gf_threshold_deg': math.degrees(self.gf_threshold_rad)
        }


class OptomotorParadigm(ExperimentParadigm):
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
        self.hs_firing_history: List[float] = []
        self.retinal_slip_history: List[float] = []
        self.gain_history: List[float] = []

    def sample_stimuli(self, x: Any, y: Optional[float] = None, heading: Optional[float] = None) -> Dict[str, Any]:
        x, y, heading = self._normalize_stimuli_args(x, y, heading)
        return {
            'drum_velocity_deg_s': self.drum_velocity_deg_s,
            'spatial_wavelength_deg': 30.0,
            'contrast': 0.9
        }

    def step(self, fly: Any, dt: float = 1.0) -> Dict[str, Any]:
        self.time_elapsed_ms += dt * 1000.0
        self.trial_manager.step()

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
        hs_firing = float(np.clip(40.0 + 1.2 * effective_slip, 0.0, 150.0))
        hs_raw = float(np.clip(40.0 + 1.2 * retinal_slip, 0.0, 150.0))

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
        return {'trial_number': self.trial_manager.trial_number}

    def get_metrics(self) -> Dict[str, Any]:
        mean_gain = float(np.mean(self.gain_history)) if self.gain_history else 0.0
        mean_hs = float(np.mean(self.hs_firing_history)) if self.hs_firing_history else 0.0
        mean_slip = float(np.mean(self.retinal_slip_history)) if self.retinal_slip_history else 0.0

        return {
            'optomotor_gain': mean_gain,
            'mean_hs_firing_rate': mean_hs,
            'mean_retinal_slip': mean_slip,
            'efference_copy_shunt_pct': 85.0
        }


class GapCrossingParadigm(ExperimentParadigm):
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


class CircadianDAMParadigm(ExperimentParadigm):
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
        current_minute = (self.trial_manager.current_step) % 1440
        hour_of_day = current_minute / 60.0

        if self.photoperiod == 'LD':
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

        # Immobility and Sleep Bout Tracking (1 step ~ 1 min of DAM recording)
        if beam_crossed or speed > 0.5:
            self.consecutive_immobile_minutes = 0.0
            self.in_sleep_bout = False
        else:
            self.consecutive_immobile_minutes += 1.0
            if self.consecutive_immobile_minutes >= 5.0:
                self.total_sleep_minutes += 1.0
                if not self.in_sleep_bout:
                    self.in_sleep_bout = True
                    self.sleep_bouts += 1

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


class CourtshipParadigm(ExperimentParadigm):
    """Paradigm 11: Courtship Conditioning & Pheromone Memory (Siegel & Hall 1979; Keleman 2007).

    Circular courtship chamber (R=5mm), male and female fly (virgin or mated).
    Mated female emits cVA anti-aphrodisiac and delivers rejection kicks,
    inducing dopaminergic suppression of male courtship song and wing extension.
    """

    def __init__(self, female_type: str = 'mated', max_duration_steps: int = 1000):
        dimensions = (20.0, 20.0)
        cx, cy = 10.0, 10.0
        self.chamber_center = (cx, cy)
        self.chamber_radius = 5.0
        self.female_type = female_type  # 'virgin' or 'mated'
        self.female_pos = (cx + 1.5, cy + 1.0)

        moat = CircularMoat(center=(cx, cy), radius=self.chamber_radius)

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
        courtship_active = (dist < 3.5)
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


class LabyrinthParadigm(ExperimentParadigm):
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
        self.path_points: List[Tuple[float, float]] = []

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
            'goal_distance_mm': dist
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
        if len(self.path_points) > 1:
            total_dist = sum(
                math.sqrt((self.path_points[i][0] - self.path_points[i - 1][0]) ** 2 +
                          (self.path_points[i][1] - self.path_points[i - 1][1]) ** 2)
                for i in range(1, len(self.path_points))
            )
            net_disp = math.sqrt(
                (self.path_points[-1][0] - self.path_points[0][0]) ** 2 +
                (self.path_points[-1][1] - self.path_points[0][1]) ** 2
            )
            tortuosity = float(total_dist / max(1.0, net_disp))
        else:
            total_dist = 0.0
            tortuosity = 1.0

        return {
            'goal_reached': self.goal_reached,
            'time_to_goal_ms': self.time_to_goal_ms,
            'wall_collision_count': self.wall_collision_count,
            'dead_end_entries': self.dead_end_entries,
            'total_distance_mm': total_dist,
            'path_tortuosity': tortuosity
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
]:
    ExperimentRegistry.register(p_cls.__name__, p_cls)
    ExperimentRegistry.register(p_cls.__name__.lower().replace("paradigm", ""), p_cls)
