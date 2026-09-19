"""2D Multi-Agent Insect Kinematics and Odor Diffusion Arena for Drosophila.

Simulates an embodied 2D multi-agent biological ecosystem:
1. Multi-fly population with individual Mushroom Body, Central Complex,
   Surge-Cast, Metabolic (hunger/satiety), Johnston's Organ, Optic Flow,
   and Tripod Gait CPG engines.
2. Stalking visual predators casting expanding looming shadows that trigger
   LC4 emergency ballistic escape saccades.
3. Alarm pheromones emitted at predator strike sites (Odor B) driving localized aversive learning.
4. Food competition, satiety depletion, and risk-sensitive foraging.
5. Backward compatible with single-fly unit tests and Gymnasium adapters.
"""

import base64
import collections
import enum
import functools
import importlib
import math
import random
import sys
import types
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from assay_response import respond as assay_response

try:
    from circuit import MushroomBodyCircuit
    from surge_cast import SurgeCastEngine
    from central_complex import CentralComplexEngine
    from metabolic import MetabolicState
    from mechanosensory import JohnstonsOrgan
    from vision import CompoundEyeVision
    from locomotion import TripodGaitCPG
    from connectome_bridge import ConnectomeBridge
    from maze import ExperimentParadigm, ExperimentRegistry
except ImportError:
    from .circuit import MushroomBodyCircuit
    from .surge_cast import SurgeCastEngine
    from .central_complex import CentralComplexEngine
    from .metabolic import MetabolicState
    from .mechanosensory import JohnstonsOrgan
    from .vision import CompoundEyeVision
    from .locomotion import TripodGaitCPG
    try:
        from .connectome_bridge import ConnectomeBridge
    except ImportError:
        ConnectomeBridge = None
    try:
        from .maze import ExperimentParadigm, ExperimentRegistry
    except ImportError:
        ExperimentParadigm = None
        ExperimentRegistry = None


class Position:
    """Represents a 2D position in the arena."""
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


class FlyState:
    """Kinematics and biophysical sub-circuits of an individual fruit fly."""
    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        speed: float = 1.2,
        antenna_length: float = 2.0,
        antenna_angle: float = math.pi / 4.0,
        seed: int = 42,
        fly_id: int = 0,
        color: str = '#4285F4',
        ablate_mb: bool = False,
        ablate_cx: bool = False,
        ablate_jo: bool = False,
        ablate_lc4: bool = False,
        ablate_off: bool = False,
        brain_type: str = 'modular',
        connectome_mode: str = 'surrogate',
        connectome_host: str = '192.168.194.227',
        connectome_port: int = 8768,
        connectome_on_fault: str = 'halt',
        vision_rng: Optional[np.random.Generator] = None
    ):
        self.id = fly_id
        self.color = color
        self.pos = Position(x, y)
        self.heading = float(heading)
        self.speed = float(speed)
        self.angular_velocity = 0.0
        self.antenna_length = antenna_length
        self.antenna_angle = antenna_angle
        self.alive = True
        self.brain_type = brain_type
        self.radius = 1.5
        # Hysteresis for the wall-avoidance reflex: which way the fly last turned away
        # from a boundary (+1 = counter-clockwise). Breaks head-on ties consistently.
        self.wall_turn_dir = 1.0

        # Ablation / Knockout switches
        self.ablate_mb = ablate_mb
        self.ablate_cx = ablate_cx
        self.ablate_jo = ablate_jo
        self.ablate_lc4 = ablate_lc4
        self.ablate_off = ablate_off

        # Biophysical sub-circuits
        self.circuit = MushroomBodyCircuit(seed=seed)
        self.surge_cast = SurgeCastEngine(dt=0.01, seed=seed, ablate_off=ablate_off)
        self.cx = CentralComplexEngine(n_wedges=16, seed=seed)
        self.metabolic = MetabolicState(initial_satiety=0.75 + 0.2 * (fly_id % 3 - 1))
        self.mechanosensory = JohnstonsOrgan(n_wedges=16)
        self.vision = CompoundEyeVision(num_ommatidia=72, rng=vision_rng)
        self.cpg = TripodGaitCPG()

        # Whole-Brain Connectome Bridge
        self.connectome_bridge = None
        self.last_connectome_telemetry = None
        if self.brain_type == 'connectome' and ConnectomeBridge is not None:
            self.connectome_bridge = ConnectomeBridge(
                mode=connectome_mode,
                rpc_host=connectome_host,
                rpc_port=connectome_port,
                on_rpc_fault=connectome_on_fault
            )
        # Which controller produced this step's motor command, and whether it is
        # faulted (see Arena.MOTOR_SOURCES_NORMAL). ``motor_halted`` means no motion.
        self.motor_source = 'modular' if self.brain_type != 'connectome' else (
            self.connectome_bridge.motor_source if self.connectome_bridge is not None else 'none')
        self.controller_fault: Optional[str] = None
        self.motor_halted = False

        # Telemetry
        self.behavioral_state: str = 'WANDER'
        self.compass_heading: float = float(heading)
        self.goal_angle: float = float(heading)
        self.wind_angle: float = 0.0
        self.food_collected: int = 0
        self.escapes_performed: int = 0

    def get_antennae_positions(self) -> Tuple[Position, Position]:
        left_angle = self.heading + self.antenna_angle
        right_angle = self.heading - self.antenna_angle

        left_x = self.pos.x + self.antenna_length * math.cos(left_angle)
        left_y = self.pos.y + self.antenna_length * math.sin(left_angle)

        right_x = self.pos.x + self.antenna_length * math.cos(right_angle)
        right_y = self.pos.y + self.antenna_length * math.sin(right_angle)

        return Position(left_x, left_y), Position(right_x, right_y)

    def update(self, dheading: float, dt: float = 1.0):
        self.angular_velocity = dheading
        self.heading = (self.heading + dheading * dt) % (2.0 * math.pi)
        self.pos.x += self.speed * math.cos(self.heading) * dt
        self.pos.y += self.speed * math.sin(self.heading) * dt


class Predator:
    """Stalking visual predator (e.g. Jumping Spider / Mantis)."""
    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        predator_id: int = 0,
        cruise_speed: float = 0.9,
        sprint_speed: float = 2.4,
        strike_radius: float = 8.0,
        vision_radius: float = 35.0
    ):
        self.id = predator_id
        self.pos = Position(x, y)
        self.heading = float(heading)
        self.cruise_speed = cruise_speed
        self.sprint_speed = sprint_speed
        self.speed = cruise_speed
        self.strike_radius = strike_radius
        self.vision_radius = vision_radius
        self.state = 'PATROL'  # 'PATROL', 'STALK', 'EAT'
        self.target_fly_id: Optional[int] = None
        self.eat_timer = 0.0
        self.stalk_timer = 0.0
        self.stalk_cooldown = 0.0
        self.total_kills = 0

    def get_velocity(self) -> np.ndarray:
        return np.array([
            self.speed * math.cos(self.heading),
            self.speed * math.sin(self.heading)
        ], dtype=np.float64)

    def step(self, flies: List[FlyState], width: float, height: float, dt: float = 1.0) -> Optional[int]:
        """
        Advance predator behavior. Returns caught fly ID if strike occurs.
        """
        if self.eat_timer > 0.0:
            self.eat_timer -= dt
            self.speed = 0.0
            self.state = 'EAT'
            self.stalk_timer = 0.0
            return None

        if self.stalk_cooldown > 0.0:
            self.stalk_cooldown -= dt

        # Find closest living fly
        closest_fly = None
        min_dist = 999.0
        for fly in flies:
            if not fly.alive:
                continue
            d = self.pos.distance_to(fly.pos)
            if d < min_dist:
                min_dist = d
                closest_fly = fly

        caught_fly_id = None
        # Ambush stalking: requires fly within vision radius, no active cooldown, and pursuit < 3.0s
        if closest_fly and min_dist <= self.vision_radius and self.stalk_cooldown <= 0.0 and self.stalk_timer < 3.0:
            # Stalking
            self.state = 'STALK'
            self.stalk_timer += dt
            self.target_fly_id = closest_fly.id
            dx = closest_fly.pos.x - self.pos.x
            dy = closest_fly.pos.y - self.pos.y
            target_heading = math.atan2(dy, dx)

            # Smooth turn toward fly
            diff = math.atan2(math.sin(target_heading - self.heading), math.cos(target_heading - self.heading))
            self.heading = (self.heading + np.clip(diff, -0.25, 0.25)) % (2.0 * math.pi)

            # Acceleration when closing in
            if min_dist < 35.0:
                self.speed = self.sprint_speed
            else:
                self.speed = self.cruise_speed

            # Strike check
            # Actively escaping fly (LC4 triggered ballistic takeoff) has low strike susceptibility
            fly_escaping = (closest_fly.behavioral_state == 'ESCAPE')
            effective_strike_radius = 3.5 if fly_escaping else self.strike_radius

            if min_dist <= effective_strike_radius:
                caught_fly_id = closest_fly.id
                self.total_kills += 1
                self.eat_timer = 3.0  # Pause to feed
                self.speed = 0.0
                self.state = 'EAT'
            elif fly_escaping and min_dist <= self.strike_radius:
                # Predator lunged and missed because fly executed an emergency escape saccade!
                self.eat_timer = 1.5  # Missed-strike refractory recovery
                self.speed = 0.0
                self.state = 'PATROL'
                self.stalk_cooldown = 4.0
        else:
            # Pursuit aborted or patrolling: set cooldown if aborting an active stalk
            if self.state == 'STALK':
                self.stalk_cooldown = 4.0
            self.state = 'PATROL'
            self.stalk_timer = 0.0
            self.target_fly_id = None
            self.speed = self.cruise_speed * 0.7
            self.heading = (self.heading + random.uniform(-0.15, 0.15)) % (2.0 * math.pi)

        # Update position
        self.pos.x += self.speed * math.cos(self.heading) * dt
        self.pos.y += self.speed * math.sin(self.heading) * dt

        # Arena boundaries
        margin = 3.0
        if self.pos.x < margin or self.pos.x > width - margin:
            self.heading = math.pi - self.heading
            self.pos.x = max(margin, min(width - margin, self.pos.x))
        if self.pos.y < margin or self.pos.y > height - margin:
            self.heading = -self.heading
            self.pos.y = max(margin, min(height - margin, self.pos.y))

        return caught_fly_id


class ContinuousOdorField:
    """Continuous Gaussian odor plume field."""
    def __init__(self, sigma: float = 15.0):
        self.sources: List[Tuple[float, float, float]] = []
        self.sigma = sigma

    def add_source(self, x: float, y: float, intensity: float = 1.0):
        self.sources.append((float(x), float(y), float(intensity)))

    def clear(self):
        self.sources.clear()

    def sample(self, x: float, y: float) -> float:
        conc = 0.0
        two_sigma_sq = 2.0 * (self.sigma ** 2)
        for sx, sy, intensity in self.sources:
            dist_sq = (x - sx) ** 2 + (y - sy) ** 2
            conc += intensity * math.exp(-dist_sq / two_sigma_sq)
        return float(np.clip(conc, 0.0, 1.0))


class ContainmentRegion:
    """Closed legal region for a fly body centre.

    ``signed_gap(x, y)`` is the distance from the point to the region boundary
    (positive inside, negative outside); a body of radius r is inside when
    ``signed_gap >= r``. ``inward_normal`` is the unit vector pointing from the nearest
    boundary into the region, i.e. the direction that moves the body away from the wall.
    ``clamp`` projects a point back inside with the given margin. These three queries
    drive the hard containment failsafe, the wall-avoidance reflex and the audit.
    """

    def signed_gap(self, x: float, y: float) -> float:
        raise NotImplementedError

    def inward_normal(self, x: float, y: float) -> Tuple[float, float]:
        raise NotImplementedError

    def clamp(self, x: float, y: float, margin: float) -> Tuple[float, float]:
        raise NotImplementedError

    def bbox(self) -> Tuple[float, float, float, float]:
        raise NotImplementedError


class RectRegion(ContainmentRegion):
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin, self.ymin, self.xmax, self.ymax = float(xmin), float(ymin), float(xmax), float(ymax)

    def signed_gap(self, x: float, y: float) -> float:
        return min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y)

    def inward_normal(self, x: float, y: float) -> Tuple[float, float]:
        sides = [
            (x - self.xmin, (1.0, 0.0)),
            (self.xmax - x, (-1.0, 0.0)),
            (y - self.ymin, (0.0, 1.0)),
            (self.ymax - y, (0.0, -1.0)),
        ]
        return min(sides, key=lambda s: s[0])[1]

    def clamp(self, x: float, y: float, margin: float) -> Tuple[float, float]:
        return (
            max(self.xmin + margin, min(self.xmax - margin, x)),
            max(self.ymin + margin, min(self.ymax - margin, y)),
        )

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class CircleRegion(ContainmentRegion):
    def __init__(self, cx: float, cy: float, radius: float):
        self.cx, self.cy, self.radius = float(cx), float(cy), float(radius)

    def signed_gap(self, x: float, y: float) -> float:
        return self.radius - math.hypot(x - self.cx, y - self.cy)

    def inward_normal(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = self.cx - x, self.cy - y
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)

    def clamp(self, x: float, y: float, margin: float) -> Tuple[float, float]:
        max_d = max(0.0, self.radius - margin)
        dx, dy = x - self.cx, y - self.cy
        d = math.hypot(dx, dy)
        if d <= max_d or d < 1e-9:
            return x, y
        return self.cx + dx * (max_d / d), self.cy + dy * (max_d / d)

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.radius, self.cy - self.radius, self.cx + self.radius, self.cy + self.radius)


class UnionRegion(ContainmentRegion):
    """Union of overlapping members (T-maze stem + cross-bar). The member with the
    largest signed gap is the one the point 'belongs' to."""

    def __init__(self, members: List[ContainmentRegion]):
        self.members = list(members)

    def _best(self, x: float, y: float) -> ContainmentRegion:
        return max(self.members, key=lambda m: m.signed_gap(x, y))

    def signed_gap(self, x: float, y: float) -> float:
        return max(m.signed_gap(x, y) for m in self.members)

    def inward_normal(self, x: float, y: float) -> Tuple[float, float]:
        return self._best(x, y).inward_normal(x, y)

    def clamp(self, x: float, y: float, margin: float) -> Tuple[float, float]:
        if self.signed_gap(x, y) >= margin:
            return x, y
        return self._best(x, y).clamp(x, y, margin)

    def bbox(self) -> Tuple[float, float, float, float]:
        boxes = [m.bbox() for m in self.members]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


class HoledRegion(ContainmentRegion):
    """An outer region minus circular obstacles (multisensory arena pillars)."""

    def __init__(self, outer: ContainmentRegion, holes: List[CircleRegion]):
        self.outer = outer
        self.holes = list(holes)

    def _terms(self, x: float, y: float):
        yield self.outer.signed_gap(x, y), None
        for h in self.holes:
            yield math.hypot(x - h.cx, y - h.cy) - h.radius, h

    def signed_gap(self, x: float, y: float) -> float:
        return min(g for g, _ in self._terms(x, y))

    def inward_normal(self, x: float, y: float) -> Tuple[float, float]:
        _, hole = min(self._terms(x, y), key=lambda t: t[0])
        if hole is None:
            return self.outer.inward_normal(x, y)
        dx, dy = x - hole.cx, y - hole.cy
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)

    def clamp(self, x: float, y: float, margin: float) -> Tuple[float, float]:
        x, y = self.outer.clamp(x, y, margin)
        for h in self.holes:
            dx, dy = x - h.cx, y - h.cy
            d = math.hypot(dx, dy)
            keep_out = h.radius + margin
            if d < keep_out:
                if d < 1e-9:
                    dx, dy, d = 1.0, 0.0, 1.0
                x = h.cx + dx * (keep_out / d)
                y = h.cy + dy * (keep_out / d)
        return x, y

    def bbox(self) -> Tuple[float, float, float, float]:
        return self.outer.bbox()


# ---------------------------------------------------------------------------
# World snapshots (Arena.snapshot_world / restore_world)
# ---------------------------------------------------------------------------
class WorldStateError(ValueError):
    """A world snapshot cannot be encoded or does not fit this arena."""


# Only these modules' classes may be (re)constructed from a snapshot: no pickle and
# no arbitrary imports.  Existing objects are otherwise restored in place.
_WORLD_MODULES = ('arena', 'maze', 'vision', 'circuit', 'surge_cast', 'central_complex', 'metabolic',
                  'mechanosensory', 'locomotion', 'assay_response', 'assay_controls', 'online_metrics')
# Controller plumbing and derived telemetry are not world state.
_WORLD_SKIP_ATTRS = frozenset(('connectome_bridge', 'last_connectome_telemetry', 'graph_controller'))
_WORLD_SKIP_TYPES = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.ModuleType,
                     type, functools.partial)
_MARKERS = frozenset(('__i__', '__nd__', '__np__', '__f__', '__t__', '__map__', '__set__', '__dq__', '__pyrng__',
                      '__npgen__', '__obj__', '__alias__', '__enum__'))


def _world_encode(value, memo, path):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, np.integer):
        # Beyond 2**53 a JSON reader in a browser loses precision (RNG states).
        return value if abs(value) < 2 ** 53 else {'__i__': str(value)}
    if isinstance(value, float) and not isinstance(value, np.floating):
        return value if math.isfinite(value) else {'__f__': repr(value)}
    if isinstance(value, np.generic):
        return {'__np__': value.dtype.str, 'b64': base64.b64encode(value.tobytes()).decode()}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise WorldStateError(f'{path}: object arrays are not world state')
        data = np.ascontiguousarray(value)
        return {'__nd__': data.dtype.str, 'shape': list(data.shape),
                'b64': base64.b64encode(data.tobytes()).decode()}
    if isinstance(value, list):
        return [_world_encode(v, memo, f'{path}[{i}]') for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return {'__t__': [_world_encode(v, memo, f'{path}[{i}]') for i, v in enumerate(value)]}
    if isinstance(value, (set, frozenset)):
        return {'__set__': sorted((_world_encode(v, memo, path) for v in value), key=repr)}
    if isinstance(value, collections.deque):
        out = {'__dq__': [_world_encode(v, memo, path) for v in value], 'maxlen': value.maxlen}
        if type(value) is not collections.deque:   # e.g. online_metrics.PathHistory
            cls = type(value)
            out['cls'] = f'{cls.__module__.rsplit(".", 1)[-1]}.{cls.__qualname__}'
            out['attrs'] = {k: _world_encode(v, memo, f'{path}.{k}') for k, v in vars(value).items()}
        return out
    if isinstance(value, enum.Enum):
        return {'__enum__': value.name}
    if isinstance(value, dict):
        if all(isinstance(k, str) and k not in _MARKERS for k in value):
            return {k: _world_encode(v, memo, f'{path}.{k}') for k, v in value.items()}
        return {'__map__': [[_world_encode(k, memo, path), _world_encode(v, memo, f'{path}[{k!r}]')]
                            for k, v in value.items()]}
    # A shared (aliased) object is stored once; in-place restore keeps the alias.
    if id(value) in memo:
        return {'__alias__': memo[id(value)]}
    if isinstance(value, random.Random):
        memo[id(value)] = path
        return {'__pyrng__': _world_encode(value.getstate(), memo, path)}
    if isinstance(value, np.random.Generator):
        memo[id(value)] = path
        return {'__npgen__': _world_encode(value.bit_generator.state, memo, path)}
    if hasattr(value, '__dict__') and not isinstance(value, _WORLD_SKIP_TYPES):
        memo[id(value)] = path
        cls = type(value)
        return {'__obj__': f'{cls.__module__.rsplit(".", 1)[-1]}.{cls.__qualname__}',
                'state': {k: _world_encode(v, memo, f'{path}.{k}') for k, v in vars(value).items()
                          if k not in _WORLD_SKIP_ATTRS and not isinstance(v, _WORLD_SKIP_TYPES)}}
    raise WorldStateError(f'{path}: cannot encode {type(value).__name__} as world state')


def _world_class(name: str):
    module_name, _, qual = name.partition('.')
    if module_name not in _WORLD_MODULES:
        raise WorldStateError(f'Refusing to construct {name}: module not allowed in world state')
    module = sys.modules.get(module_name)
    if module is None:
        module = next((m for key, m in list(sys.modules.items())
                       if m is not None and key.rsplit('.', 1)[-1] == module_name), None)
    if module is None:
        module = importlib.import_module(module_name)
    obj = module
    for part in qual.split('.'):
        obj = getattr(obj, part)
    return obj


def _world_decode(enc, current, path, refs):
    """Decode ``enc`` (encoded at ``path``).  Objects, arrays and generators are updated
    inside ``current`` when possible, and an alias resolves to the object restored at
    its canonical path (``refs``), so shared references stay shared."""
    if enc is None or isinstance(enc, (bool, int, float, str)):
        return enc
    if isinstance(enc, list):
        cur = current if isinstance(current, list) else []
        return [_world_decode(v, cur[i] if i < len(cur) else None, f'{path}[{i}]', refs)
                for i, v in enumerate(enc)]
    if '__f__' in enc:
        return float(enc['__f__'])
    if '__i__' in enc:
        return int(enc['__i__'])
    if '__np__' in enc:
        return np.frombuffer(base64.b64decode(enc['b64']), dtype=np.dtype(enc['__np__']))[0]
    if '__nd__' in enc:
        arr = np.frombuffer(base64.b64decode(enc['b64']), dtype=np.dtype(enc['__nd__'])).reshape(enc['shape'])
        if (isinstance(current, np.ndarray) and current.shape == arr.shape and current.dtype == arr.dtype
                and current.flags.writeable):
            current[...] = arr
            return current
        return arr.copy()
    if '__t__' in enc:
        cur = current if isinstance(current, tuple) else ()
        return tuple(_world_decode(v, cur[i] if i < len(cur) else None, f'{path}[{i}]', refs)
                     for i, v in enumerate(enc['__t__']))
    if '__set__' in enc:
        return set(_world_decode(v, None, path, refs) for v in enc['__set__'])
    if '__dq__' in enc:
        items = [_world_decode(v, None, path, refs) for v in enc['__dq__']]
        if 'cls' not in enc:
            return collections.deque(items, maxlen=enc['maxlen'])
        cls = _world_class(enc['cls'])
        if not issubclass(cls, collections.deque):
            raise WorldStateError(f'{path}: {enc["cls"]} is not a deque')
        target = cls.__new__(cls)
        collections.deque.__init__(target, items, enc['maxlen'])
        _world_restore_attrs(target, enc.get('attrs') or {}, path, refs)
        return target
    if '__enum__' in enc:
        if isinstance(current, enum.Enum):
            return type(current)[enc['__enum__']]
        raise WorldStateError(f'{path}: cannot restore enum member {enc["__enum__"]} without a current value')
    if '__map__' in enc:
        cur = current if isinstance(current, dict) else {}
        out = {}
        for k_enc, v_enc in enc['__map__']:
            key = _world_decode(k_enc, None, path, refs)
            out[key] = _world_decode(v_enc, cur.get(key), f'{path}[{key!r}]', refs)
        return out
    if '__alias__' in enc:
        if enc['__alias__'] not in refs:
            raise WorldStateError(f'{path}: alias to unknown {enc["__alias__"]}')
        return refs[enc['__alias__']]
    if '__pyrng__' in enc:
        gen = current if isinstance(current, random.Random) else random.Random()
        refs[path] = gen
        gen.setstate(_world_decode(enc['__pyrng__'], None, path, {}))
        return gen
    if '__npgen__' in enc:
        state = _world_decode(enc['__npgen__'], None, path, {})
        gen = current if isinstance(current, np.random.Generator) else \
            np.random.Generator(getattr(np.random, state['bit_generator'])())
        gen.bit_generator.state = state
        refs[path] = gen
        return gen
    if '__obj__' in enc:
        cls = _world_class(enc['__obj__'])
        target = current if type(current) is cls else cls.__new__(cls)
        refs[path] = target
        _world_restore_attrs(target, enc['state'], path, refs)
        return target
    cur = current if isinstance(current, dict) else {}
    return {k: _world_decode(v, cur.get(k), f'{path}.{k}', refs) for k, v in enc.items()}


def _world_restore_attrs(target, state: dict, path: str, refs: dict) -> None:
    current = vars(target)
    for key, enc in state.items():
        setattr(target, key, _world_decode(enc, current.get(key), f'{path}.{key}', refs))


class Arena:
    """Multi-Agent 2D Simulation Arena with Drosophila and Stalking Predators."""

    FLY_COLORS = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#AB47BC', '#00ACC1', '#FF7043']

    # Wall-avoidance reflex (mirrored in web/app.js as wallAvoidanceTurn):
    # boundaries closer than this to the body edge are perceived, and the reflex can
    # command up to this yaw rate away from them. Mechanosensory/visual proximity at
    # descending-neuron level; it does not touch the brain models.
    WALL_PERCEPTION_MM = 4.0
    WALL_AVOID_YAW_RAD_S = 4.0

    # Engineered motor assists (decision contract item 4). Neither is a physical
    # constraint: both rotate the fly away from boundaries on the controller's behalf,
    # so a wall escape they produce must not be credited to any brain. They stay on by
    # default to preserve the established modular behaviour; every use is logged per
    # step in ``fly.motor_record`` and cumulatively in ``fly.assist_totals``.
    #   wall_avoidance_reflex - pre-contact yaw added from perceived boundaries, which
    #       also discards the controller's own yaw when the body is at contact.
    #   contact_turn - heading rotation applied when the containment failsafe (or the
    #       open-arena edge clamp) resolves a contact. Without it contact only removes
    #       the into-wall speed, as a physical constraint should.
    # Default: ON only for the legacy hand-built controllers (MOTOR_ASSIST_DEFAULT_ON);
    # OFF for every graph backend and the RPC hybrid (plan decision item 4), so a
    # graph-driven fly that pushes into a wall is reported as such, not steered free.
    MOTOR_ASSISTS = ('wall_avoidance_reflex', 'contact_turn')
    MOTOR_ASSIST_DEFAULT_ON = ('modular', 'bridge-surrogate')
    # motor_source values that mean "the named controller drove this step". Anything
    # else (halted-rpc-fault, surrogate-fallback-TEST, graph-unmapped-io, none...) is
    # flagged in telemetry and the dashboard banner.
    MOTOR_SOURCES_NORMAL = ('modular', 'surrogate', 'graph-rpc', 'graph')
    WORLD_STATE_FORMAT = 'neurofly.world-state.v1'
    NEAR_WALL_MM = 1.0   # body-edge gap counted as "near a wall" in motor records

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        num_food: int = 2,
        num_hazards: int = 2,
        wind: Tuple[float, float] = (-0.3, 0.0),
        seed: int = 42,
        num_flies: int = 1,
        num_predators: int = 0,
        fly_ablations: Optional[List[Dict[str, bool]]] = None,
        brain_type: str = 'modular',
        connectome_mode: str = 'surrogate',
        connectome_host: str = '192.168.194.227',
        connectome_port: int = 8768,
        paradigm: Optional[Union[Any, str]] = None,
        motor_assists: Optional[Dict[str, bool]] = None,
        connectome_on_fault: str = 'halt',
        controller_backend: Optional[str] = None,
        graph_controller: Optional[Any] = None
    ):
        # Named controller backend (provenance.BACKENDS). Derived from brain_type and
        # connectome_mode when not given; graph backends pass it explicitly together
        # with ``graph_controller`` (see compute_steering).
        if controller_backend is None:
            if brain_type == 'connectome':
                controller_backend = 'hybrid-bridge-rpc-experimental' if connectome_mode == 'rpc' else 'bridge-surrogate'
            else:
                controller_backend = 'modular'
        self.controller_backend = controller_backend
        self.graph_controller = graph_controller
        assist_default = controller_backend in self.MOTOR_ASSIST_DEFAULT_ON
        self.motor_assists: Dict[str, bool] = {name: assist_default for name in self.MOTOR_ASSISTS}
        for name, enabled in (motor_assists or {}).items():
            if name not in self.motor_assists:
                raise ValueError(f'Unknown motor assist {name!r}; expected one of {self.MOTOR_ASSISTS}')
            self.motor_assists[name] = bool(enabled)
        self.last_failsafe = (0.0, 0.0, 0.0, 0.0)
        self.width = width
        self.height = height
        self.wind = wind
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.seed = seed
        self.brain_type = brain_type

        # Neuroethological Experiment Paradigm
        self.paradigm = None
        if paradigm is not None:
            if isinstance(paradigm, str):
                if ExperimentRegistry is not None:
                    self.paradigm = ExperimentRegistry.get(paradigm)
                else:
                    raise ImportError("ExperimentRegistry from maze is not available")
            else:
                self.paradigm = paradigm
            self.width = float(self.paradigm.dimensions[0])
            self.height = float(self.paradigm.dimensions[1])

        # Odor fields: Odor A = Food, Odor B = Hazard / Alarm pheromone
        self.odor_a = ContinuousOdorField(sigma=18.0)
        self.odor_b = ContinuousOdorField(sigma=14.0)

        # Entities
        self.food_positions: List[Position] = []
        self.hazard_positions: List[Position] = []
        self.num_food = num_food
        self.num_hazards = num_hazards

        if self.paradigm is None:
            self._spawn_entities()

        # Multi-Fly Population
        self.num_flies = max(1, num_flies)
        self.flies: List[FlyState] = []

        if self.paradigm is not None:
            sx, sy, sh = self._get_paradigm_spawn(self.paradigm)
        else:
            sx, sy, sh = None, None, None

        for i in range(self.num_flies):
            if self.paradigm is not None:
                if i == 0:
                    fx, fy, fh = sx, sy, sh
                else:
                    fx = sx + self.rng.uniform(-1.0, 1.0)
                    fy = sy + self.rng.uniform(-1.0, 1.0)
                    fh = sh
            else:
                fx = self.rng.uniform(25.0, self.width - 25.0)
                fy = self.rng.uniform(25.0, self.height - 25.0)
                fh = self.rng.uniform(0, 2 * math.pi)

            color = self.FLY_COLORS[i % len(self.FLY_COLORS)]
            abl = (fly_ablations[i] if fly_ablations and i < len(fly_ablations) else {})
            fly = FlyState(
                fx, fy, heading=fh, seed=seed + i * 10, fly_id=i, color=color,
                ablate_mb=abl.get('ablate_mb', False),
                ablate_cx=abl.get('ablate_cx', False),
                ablate_jo=abl.get('ablate_jo', False),
                ablate_lc4=abl.get('ablate_lc4', False),
                ablate_off=abl.get('ablate_off', False),
                brain_type=brain_type,
                connectome_mode=connectome_mode,
                connectome_host=connectome_host,
                connectome_port=connectome_port,
                connectome_on_fault=connectome_on_fault,
                vision_rng=self.np_rng
            )
            if graph_controller is not None:
                fly.motor_source = 'graph'
            fly.motor_record = {}
            fly.assist_totals = self._empty_assist_totals()
            self.flies.append(fly)

        # Predators
        self.num_predators = 0 if self.paradigm is not None else num_predators
        self.predators: List[Predator] = []
        for p_idx in range(self.num_predators):
            px = self.rng.uniform(15.0, self.width - 15.0)
            py = self.rng.uniform(15.0, self.height - 15.0)
            self.predators.append(Predator(px, py, heading=self.rng.uniform(0, 2 * math.pi), predator_id=p_idx))

        # Backward compatibility alias for single-fly callers
        self.fly: FlyState = self.flies[0]
        self.circuit = self.fly.circuit
        self.surge_cast = self.fly.surge_cast
        self.cx = self.fly.cx

        # Legal region for fly body centres: hard failsafe, wall perception and audits
        # all read this one description of the paradigm's enclosure.
        self.containment: ContainmentRegion = self._build_containment()
        self.world_bounds: Tuple[float, float, float, float] = self.containment.bbox()

        # Ecological Metrics
        self.time_step = 0
        self.food_collected = 0
        self.hazard_encounters = 0
        self.total_predator_kills = 0
        self.total_escapes = 0
        self.total_distance = 0.0
        self.time_to_food_history: List[int] = []
        self.last_food_step = 0

    @staticmethod
    def _empty_assist_totals() -> Dict[str, float]:
        return {
            'steps': 0, 'near_wall_steps': 0, 'contact_steps': 0,
            'wall_reflex_steps': 0, 'wall_reflex_abs_yaw_rad': 0.0,
            'controller_yaw_suppressed_steps': 0,
            'contact_turn_events': 0, 'contact_turn_abs_rad': 0.0,
            'solver_contact_steps': 0, 'solver_correction_mm': 0.0,
            'overlap_correction_steps': 0, 'overlap_correction_mm': 0.0,
            'failsafe_corrections': 0, 'failsafe_correction_mm': 0.0,
        }

    def motor_provenance(self, fly: Optional[FlyState] = None) -> Dict[str, Any]:
        """Which engineered assists are enabled and how much each acted on ``fly``."""
        f = fly or self.fly
        source = getattr(f, 'motor_source', 'modular')
        return {'controller_backend': self.controller_backend,
                'motor_assists': dict(self.motor_assists),
                'motor_assists_enabled': any(self.motor_assists.values()),
                'motor_source': source,
                'motor_source_normal': source in self.MOTOR_SOURCES_NORMAL,
                'motor_halted': bool(getattr(f, 'motor_halted', False)),
                'controller_fault': getattr(f, 'controller_fault', None),
                'assist_totals': dict(getattr(f, 'assist_totals', None) or self._empty_assist_totals())}

    # ------------------------------------------------------------------ world snapshots
    def snapshot_world(self) -> Dict[str, Any]:
        """JSON-safe snapshot of the whole world, for replay and per-assay checkpoints.

        ``summary`` is a readable digest (pose, velocities, paradigm, RNG states).
        ``state`` is the complete encoded arena: every fly's pose, velocities and
        body/sensor state (including the modular controller sub-circuits that live
        on the fly), the paradigm's mutable state, odor fields, entities, counters
        and both random generators (``rng`` and the ``np_rng`` bit generator, which
        also drives vision).  Controller plumbing (ConnectomeBridge, RPC clients and
        the graph controller) is excluded: graph brain state lives in the registry
        checkpoint next to this snapshot.  Arrays are base64 of their raw bytes, so
        a restore is bit-exact.
        """
        state = _world_encode({k: v for k, v in vars(self).items()
                               if k not in _WORLD_SKIP_ATTRS and not isinstance(v, _WORLD_SKIP_TYPES)},
                              {}, 'arena')
        flies = [{'id': f.id, 'x': f.pos.x, 'y': f.pos.y, 'heading': f.heading, 'speed': f.speed,
                  'angular_velocity': f.angular_velocity, 'alive': f.alive,
                  'behavioral_state': getattr(f, 'behavioral_state', None)} for f in self.flies]
        return {
            'format': self.WORLD_STATE_FORMAT,
            'paradigm': self.paradigm_key(self.paradigm) if self.paradigm is not None else None,
            'controller_backend': self.controller_backend,
            'seed': self.seed,
            'time_step': self.time_step,
            'summary': {'flies': flies,
                        'rng': {'python_random': _world_encode(self.rng.getstate(), {}, 'rng'),
                                'np_rng': _world_encode(self.np_rng.bit_generator.state, {}, 'np_rng')},
                        'motor_assists': dict(self.motor_assists)},
            'state': state,
        }

    def restore_world(self, snapshot: Dict[str, Any]) -> None:
        """Restore :meth:`snapshot_world` output into this arena, in place.

        The arena must host the same paradigm.  Continuing after a restore follows
        the same trajectory as continuing the snapshotted arena, step for step.
        Raises :class:`WorldStateError` on a foreign or mismatching snapshot and
        leaves the arena untouched in that case.
        """
        if not isinstance(snapshot, dict) or snapshot.get('format') != self.WORLD_STATE_FORMAT:
            raise WorldStateError(f'Not a {self.WORLD_STATE_FORMAT} snapshot')
        mine = self.paradigm_key(self.paradigm) if self.paradigm is not None else None
        if snapshot.get('paradigm') != mine:
            raise WorldStateError(f'Snapshot is for paradigm {snapshot.get("paradigm")!r}, arena hosts {mine!r}')
        state = snapshot.get('state')
        if not isinstance(state, dict):
            raise WorldStateError('Snapshot has no state')
        if len(state.get('flies') or []) != len(self.flies):
            raise WorldStateError('Snapshot fly count differs from this arena')
        # Keep the live controller plumbing and the configured assists/backend.
        keep = {k: getattr(self, k) for k in ('graph_controller', 'motor_assists', 'controller_backend')}
        bridges = [(f, getattr(f, 'connectome_bridge', None)) for f in self.flies]
        current = {k: v for k, v in vars(self).items() if k not in _WORLD_SKIP_ATTRS}
        refs: Dict[str, Any] = {}
        decoded = {k: _world_decode(enc, current.get(k), f'arena.{k}', refs) for k, enc in state.items()}
        for key, value in decoded.items():
            setattr(self, key, value)
        for key, value in keep.items():
            setattr(self, key, value)
        for fly, bridge in bridges:
            fly.connectome_bridge = bridge
        # Aliases of the first fly (single-fly API).
        self.fly = self.flies[0]
        self.circuit, self.surge_cast, self.cx = self.fly.circuit, self.fly.surge_cast, self.fly.cx
        for fly in self.flies:
            fly.vision.rng = self.np_rng

    def nearest_boundary_gap(self, x: float, y: float, radius: float) -> Tuple[float, float, float]:
        """(gap, nx, ny) of the closest boundary: body-edge clearance and away-normal."""
        g = self.containment.signed_gap(x, y) - radius
        nx, ny = self.containment.inward_normal(x, y)
        for wall in (getattr(self.paradigm, 'walls', None) or []):
            px, py, _ = wall.project_point(x, y)
            d = math.hypot(x - px, y - py)
            if d - radius < g:
                g = d - radius
                nx, ny = ((x - px) / d, (y - py) / d) if d > 1e-8 else (wall.nx, wall.ny)
        return g, nx, ny

    def _finish_motor_record(self, fly: FlyState, rec: Dict[str, Any], start: Tuple[float, float], dt: float) -> None:
        """Complete and store one step's motor record and add it to the totals."""
        gap, nx, ny = self.nearest_boundary_gap(fly.pos.x, fly.pos.y, getattr(fly, 'radius', 1.5))
        rec['realized_dx'] = fly.pos.x - start[0]
        rec['realized_dy'] = fly.pos.y - start[1]
        rec['realized_mm'] = math.hypot(rec['realized_dx'], rec['realized_dy'])
        rec['wall_gap_mm'] = gap
        rec['wall_normal'] = (nx, ny)
        rec['near_wall'] = gap <= self.NEAR_WALL_MM
        rec['in_contact'] = gap <= 0.05 or bool(rec.get('contact_normals'))
        fly.motor_record = rec
        t = fly.assist_totals
        t['steps'] += 1
        t['near_wall_steps'] += int(rec['near_wall'])
        t['contact_steps'] += int(rec['in_contact'])
        if rec.get('wall_reflex_yaw', 0.0) != 0.0:
            t['wall_reflex_steps'] += 1
            t['wall_reflex_abs_yaw_rad'] += abs(rec['wall_reflex_yaw']) * dt
        t['controller_yaw_suppressed_steps'] += int(rec.get('controller_yaw_suppressed', False))
        if rec.get('contact_turn_rad', 0.0) != 0.0:
            t['contact_turn_events'] += 1
            t['contact_turn_abs_rad'] += abs(rec['contact_turn_rad'])
        if rec.get('contact_normals'):
            t['solver_contact_steps'] += 1
        t['solver_correction_mm'] += rec.get('solver_correction_mm', 0.0)
        if rec.get('overlap_correction_mm', 0.0) > 0.0:
            t['overlap_correction_steps'] += 1
            t['overlap_correction_mm'] += rec['overlap_correction_mm']
        if rec.get('failsafe_correction_mm', 0.0) > 0.0:
            t['failsafe_corrections'] += 1
            t['failsafe_correction_mm'] += rec['failsafe_correction_mm']

    @staticmethod
    def paradigm_key(paradigm: Any) -> str:
        """Canonical snake_case key of a paradigm ('t_maze', 'heat_maze', 'multisensory', ...).

        Exact keys, never substring tests: 'heat_maze' contains 't_maze', and matching by
        substring once clamped the heat-maze fly into the T-maze corridor.
        """
        if paradigm is None:
            return ''
        name = str(getattr(paradigm, 'name', '')).strip().lower().replace('-', '_').replace(' ', '_')
        if name.startswith('multisensory'):
            return 'multisensory'
        return name

    def _get_paradigm_spawn(self, paradigm: Any) -> Tuple[float, float, float]:
        """Compute initial fly spawn coordinates (x, y, heading) based on paradigm geometry."""
        if hasattr(paradigm, 'spawn_pos') and paradigm.spawn_pos is not None:
            sp = paradigm.spawn_pos
            return float(sp[0]), float(sp[1]), float(getattr(paradigm, 'spawn_heading', 0.0))

        spawns = {
            # Stem base center (x=70, y=18), facing up stem (+y, pi/2)
            't_maze': (70.0, 18.0, math.pi / 2.0),
            # Central hub
            'y_maze': (60.0, 60.0, 0.0),
            # Circular platform center
            'heat_maze': (60.0, 60.0, 0.0),
            'buridan': (60.0, 60.0, 0.0),
            # Flight simulator center
            'visual_operant': (40.0, 40.0, 0.0),
            # Downwind release point facing upwind (East, heading 0.0)
            'wind_tunnel': (20.0, 30.0, 0.0),
            # Center of stage
            'looming_escape': (40.0, 40.0, 0.0),
            # Center of drum
            'optomotor': (45.0, 45.0, 0.0),
            # Takeoff track start
            'gap_crossing': (20.0, 10.0, 0.0),
            # Tube 0 start
            'circadian_dam': (15.0, 5.0, 0.0),
            # Male facing female at center (10, 10)
            'courtship': (10.0, 13.0, -math.pi / 2.0),
            # Maze entrance
            'labyrinth': (15.0, 15.0, 0.0),
            # Origin-centred circular arena (r=75): the dashboard spawns at (0, 0)
            'multisensory': (0.0, 0.0, 0.0),
        }
        return spawns.get(self.paradigm_key(paradigm), (self.width / 2.0, self.height / 2.0, 0.0))

    def _build_containment(self) -> ContainmentRegion:
        """Legal body-centre region of the active paradigm.

        Matches the enclosure web/app.js draws for the same paradigm id so that the
        daemon's coordinates land inside the dashboard's picture of the arena.
        """
        p = self.paradigm
        key = self.paradigm_key(p)
        w, h = float(self.width), float(self.height)
        if key == 't_maze':
            # Stem [63,77]x[10,57] joins the cross-bar [10,130]x[43,57]
            return UnionRegion([RectRegion(63.0, 10.0, 77.0, 57.0), RectRegion(10.0, 43.0, 130.0, 57.0)])
        if key == 'y_maze':
            cx, cy = getattr(p, 'center', (60.0, 60.0))
            return CircleRegion(cx, cy, 48.0)  # arms are enclosed by their own walls
        if key == 'heat_maze':
            cx, cy = getattr(p, 'arena_center', (60.0, 60.0))
            return CircleRegion(cx, cy, float(getattr(p, 'arena_radius', 55.0)))
        if key == 'buridan':
            cx, cy = getattr(p, 'center', (60.0, 60.0))
            return CircleRegion(cx, cy, float(getattr(p, 'platform_radius', 50.0)))
        if key == 'courtship':
            cx, cy = getattr(p, 'chamber_center', (10.0, 10.0))
            return CircleRegion(cx, cy, float(getattr(p, 'chamber_radius', 8.5)))
        if key == 'wind_tunnel':
            return RectRegion(0.0, 0.0, 200.0, 60.0)
        if key == 'gap_crossing':
            return RectRegion(5.0, 7.5, 95.0, 12.5)
        if key == 'circadian_dam':
            return RectRegion(5.0, 1.0, 60.0, 9.0)  # active tube 0
        if key == 'labyrinth':
            return RectRegion(0.0, 0.0, 140.0, 100.0)
        if key == 'multisensory':
            outer = CircleRegion(0.0, 0.0, float(getattr(p, 'arena_radius', 75.0)))
            pr = float(getattr(p, 'pillar_radius', 6.0))
            holes = [CircleRegion(pc[0], pc[1], pr) for pc in getattr(p, 'pillar_centers', [])]
            return HoledRegion(outer, holes)
        # visual_operant, looming_escape, optomotor and the open arena are plain rectangles
        return RectRegion(0.0, 0.0, w, h)

    def reset_fly_to_spawn(self, fly: Optional[FlyState] = None):
        """Return a fly to the paradigm's spawn pose for a new trial, keeping its memory."""
        f = fly or self.fly
        if self.paradigm is not None:
            sx, sy, sh = self._get_paradigm_spawn(self.paradigm)
        else:
            sx, sy, sh = self.width / 2.0, self.height / 2.0, 0.0
        f.pos.x, f.pos.y = float(sx), float(sy)
        f.heading = float(sh) % (2.0 * math.pi)
        f.speed = 1.2
        f.angular_velocity = 0.0
        f.assay_escape_remaining = 0.0
        f.feeding_remaining = 0.0
        f.food_contact_active = False
        f.alive = True

    def sense_boundaries(self, x: float, y: float, radius: float, perception: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """Boundaries within ``perception`` mm of the body edge as (gap, nx, ny).

        ``gap`` is the free space between body edge and boundary (negative when
        penetrating); (nx, ny) is the unit normal pointing away from that boundary.
        Wall segments of the paradigm and the containment region are both sensed.
        """
        reach = self.WALL_PERCEPTION_MM if perception is None else perception
        found: List[Tuple[float, float, float]] = []
        g = self.containment.signed_gap(x, y) - radius
        if g < reach:
            nx, ny = self.containment.inward_normal(x, y)
            found.append((g, nx, ny))
        for wall in (getattr(self.paradigm, 'walls', None) or []):
            px, py, _ = wall.project_point(x, y)
            dx, dy = x - px, y - py
            d = math.hypot(dx, dy)
            g = d - radius
            if g < reach:
                if d > 1e-8:
                    found.append((g, dx / d, dy / d))
                else:
                    found.append((g, wall.nx, wall.ny))
        return found

    def wall_avoidance_turn(self, fly: FlyState, dheading: float, dt: float = 0.02) -> float:
        """Descending-level reflex: steer away from boundaries the fly can perceive.

        Each perceived boundary contributes its away-normal weighted by proximity
        (1 at contact, 0 at the perception range). If the fly is heading into the net
        normal, an extra yaw of up to WALL_AVOID_YAW_RAD_S is added in the direction
        that rotates the heading away from the wall; the side is remembered in
        ``fly.wall_turn_dir`` so a head-on approach does not dither. Sliding parallel
        to a wall (thigmotaxis) is untouched because the approach term is zero.

        This is an engineered assist, not physics: it is skipped entirely when
        ``motor_assists['wall_avoidance_reflex']`` is False, and ``fly.wall_reflex_suppressed``
        records whether the controller's own yaw was discarded at contact.
        """
        fly.wall_reflex_suppressed = False
        if not self.motor_assists.get('wall_avoidance_reflex', True):
            return dheading
        radius = getattr(fly, 'radius', 1.5)
        sensed = self.sense_boundaries(fly.pos.x, fly.pos.y, radius)
        if not sensed:
            return dheading

        reach = self.WALL_PERCEPTION_MM
        net_x = net_y = 0.0
        proximity = 0.0
        for gap, nx, ny in sensed:
            wgt = 1.0 - max(0.0, gap) / reach
            net_x += wgt * nx
            net_y += wgt * ny
            proximity = max(proximity, wgt)
        n_mag = math.hypot(net_x, net_y)
        if n_mag < 1e-9 or proximity <= 0.0:
            return dheading
        net_x /= n_mag
        net_y /= n_mag

        # Direction of travel, not the heading: a fly walking backwards (MDN reverse
        # under heat or laser) can back into a wall while facing away from it.
        travel = fly.heading if fly.speed >= 0.0 else fly.heading + math.pi
        hx, hy = math.cos(travel), math.sin(travel)
        approach = -(hx * net_x + hy * net_y)  # > 0 when moving into the boundary
        if approach <= 0.0:
            return dheading

        cross = hx * net_y - hy * net_x  # > 0: counter-clockwise turn moves travel toward the normal
        if abs(cross) > 0.1:
            fly.wall_turn_dir = 1.0 if cross > 0.0 else -1.0
        avoid = fly.wall_turn_dir * self.WALL_AVOID_YAW_RAD_S * proximity * approach
        # At contact, sensory attraction must not cancel the avoidance torque
        # and hold the body against a wall. Resume taxis once facing away.
        if proximity > .9:
            fly.wall_reflex_suppressed = dheading != 0.0
            dheading = 0.0
        # Rate limit; with legacy whole-second ticks never rotate more than a quarter turn per step
        limit = min(self.WALL_AVOID_YAW_RAD_S, (math.pi / 2.0) / max(float(dt), 1e-6))
        return float(max(-limit, min(limit, dheading + avoid)))

    def _spawn_entities(self):
        self.food_positions.clear()
        self.hazard_positions.clear()
        self.odor_a.clear()
        self.odor_b.clear()

        for _ in range(self.num_food):
            fx = self.rng.uniform(15.0, self.width - 15.0)
            fy = self.rng.uniform(15.0, self.height - 15.0)
            self.food_positions.append(Position(fx, fy))
            self.odor_a.add_source(fx, fy, 1.0)

        for _ in range(self.num_hazards):
            hx = self.rng.uniform(15.0, self.width - 15.0)
            hy = self.rng.uniform(15.0, self.height - 15.0)
            self.hazard_positions.append(Position(hx, hy))
            self.odor_b.add_source(hx, hy, 1.0)

    def initialize_fly(self, x: float, y: float, heading: float = 0.0):
        """Backward-compatible reset of primary fly position."""
        self.fly.pos = Position(x, y)
        self.fly.heading = float(heading)
        self.fly.speed = 1.2
        self.fly.angular_velocity = 0.0
        self.fly.alive = True

    def _sample_paradigm_antennae(self, fly: FlyState, stimuli: Dict[str, Any]) -> Dict[str, float]:
        """Bilateral odour sample of a paradigm that publishes 'odor_a'/'odor_b' fields."""
        left_pos, right_pos = fly.get_antennae_positions()

        def at(pos: Position) -> Dict[str, Any]:
            try:
                return self.paradigm.sample_stimuli((pos.x, pos.y), fly.heading)
            except TypeError:
                return self.paradigm.sample_stimuli(pos.x, pos.y, fly.heading)

        sl, sr = at(left_pos), at(right_pos)
        left_a, right_a = float(sl.get('odor_a', sl.get('odor_conc', 0.0))), float(sr.get('odor_a', sr.get('odor_conc', 0.0)))
        left_b, right_b = float(sl.get('odor_b', 0.0)), float(sr.get('odor_b', 0.0))
        return {
            'left_a': left_a, 'right_a': right_a,
            'mean_a': 0.5 * (left_a + right_a), 'diff_a': left_a - right_a,
            'left_b': left_b, 'right_b': right_b,
            'mean_b': 0.5 * (left_b + right_b), 'diff_b': left_b - right_b,
            'temperature_left': float(sl.get('temperature', 25.0)),
            'temperature_right': float(sr.get('temperature', 25.0))
        }

    def sample_antennae(self, fly: FlyState = None) -> Dict[str, float]:
        f = fly or self.fly
        left_pos, right_pos = f.get_antennae_positions()

        left_a = self.odor_a.sample(left_pos.x, left_pos.y)
        right_a = self.odor_a.sample(right_pos.x, right_pos.y)
        left_b = self.odor_b.sample(left_pos.x, left_pos.y)
        right_b = self.odor_b.sample(right_pos.x, right_pos.y)

        return {
            'left_a': left_a,
            'right_a': right_a,
            'mean_a': 0.5 * (left_a + right_a),
            'diff_a': left_a - right_a,
            'left_b': left_b,
            'right_b': right_b,
            'mean_b': 0.5 * (left_b + right_b),
            'diff_b': left_b - right_b
        }

    def compute_steering(
        self,
        sensory: Dict[str, float],
        is_feeding: bool = False,
        fly: FlyState = None,
        temperature: float = 25.0,
        wind_vector: Optional[np.ndarray] = None,
        landmarks: Optional[Any] = None,
        cva_odor: float = 0.0,
        bitter_pheromone: float = 0.0,
        female_aphrodisiac: float = 0.0,
        incurred_damage: bool = False,
        is_saccade: Optional[bool] = None,
        dt: float = 0.01,
        **kwargs
    ) -> Tuple[float, float, str, float, float]:
        f = fly or self.fly
        pred_pos_list = [p.pos.to_array() for p in self.predators]
        pred_vel_list = [p.get_velocity() for p in self.predators]
        w_vec = wind_vector if wind_vector is not None else np.array(self.wind, dtype=np.float64)

        # Graph backend dispatch (connectome-fixed / -plastic / -with-trained-readout).
        # The controller returns body-frame commands in arena units: forward_speed in
        # mm/s and yaw_rate in rad/s (+ = counter-clockwise).  They are applied as
        # given: no floor, no clipping and no substitute command when the graph is silent.
        if self.graph_controller is not None:
            out = self.graph_controller(fly=f, sensory=sensory, dt=dt, temperature=temperature,
                                        wind_vector=w_vec, **kwargs)
            f.last_connectome_telemetry = out
            f.motor_source = str(out.get('motor_source', 'graph'))
            f.controller_fault = out.get('controller_fault')
            f.motor_halted = bool(out.get('halted', False))
            if f.motor_halted:
                return 0.0, 0.0, str(out.get('state', 'HALTED')), float(f.heading), float(f.heading)
            return (float(out.get('yaw_rate', 0.0)), float(out.get('forward_speed', 0.0)),
                    str(out.get('state', 'GRAPH')), float(f.heading), float(f.heading))

        # Whole-Brain Connectome Bridge Dispatch
        if getattr(f, 'brain_type', 'modular') == 'connectome' and getattr(f, 'connectome_bridge', None) is not None:
            bridge_out = f.connectome_bridge.step(
                fly_pos=f.pos.to_array(),
                fly_heading=f.heading,
                fly_speed=f.speed,
                fly_yaw_rate=f.angular_velocity,
                odor_left=sensory.get('left_a', 0.0),
                odor_right=sensory.get('right_a', 0.0),
                wind_vector=w_vec,
                predator_positions=pred_pos_list,
                predator_velocities=pred_vel_list,
                food_ingested=is_feeding,
                incurred_damage=incurred_damage,
                energy_level=f.metabolic.satiety if hasattr(f, 'metabolic') else 1.0,
                dt=dt,
                temperature=temperature,
                landmarks=landmarks,
                cva_odor=cva_odor,
                bitter_pheromone=bitter_pheromone,
                female_aphrodisiac=female_aphrodisiac,
                is_saccade=is_saccade,
                **kwargs
            )
            f.last_connectome_telemetry = bridge_out
            f.motor_source = str(bridge_out.get('motor_source', f.connectome_bridge.motor_source))
            f.controller_fault = bridge_out.get('controller_fault')
            f.motor_halted = f.motor_source == 'halted-rpc-fault'
            if f.motor_halted:
                # RPC fault under on_rpc_fault='halt': no controller, so no propulsion
                # and no steering.  The legacy 0.1 speed floor below must not apply.
                return 0.0, 0.0, 'HALTED', float(f.heading), float(f.heading)
            dheading = float(np.clip(bridge_out['yaw_rate'] * 0.01, -0.45, 0.45))
            new_speed = float(np.clip(bridge_out['forward_speed'] * 0.1, 0.1, 3.5))
            if bridge_out.get('escape_active'):
                state = 'ESCAPE'
                f.escapes_performed += 1
                self.total_escapes += 1
            elif bridge_out.get('mdn_rate', 0.0) > 30.0:
                state = 'REVERSE'
                new_speed = -0.5
            elif bridge_out.get('dnp09_rate', 0.0) > 12.0:
                state = 'SURGE'
            else:
                state = 'WANDER'
            compass_heading = float(bridge_out.get('compass_bump_heading', f.heading))
            goal_angle = float(bridge_out.get('wpn_wind_heading', f.heading))
            return dheading, new_speed, state, compass_heading, goal_angle

        circuit = f.circuit
        surge_cast = f.surge_cast
        cx = f.cx
        metabolic = f.metabolic
        jo = f.mechanosensory
        vision = f.vision
        cpg = f.cpg

        # 1. Mushroom Body learned valences
        if f.ablate_mb:
            valence_a = 0.0
            valence_b = 0.0
        else:
            _, _, valence_a = circuit.forward(circuit.encode_odor(1.0, 0.0)[1])
            _, _, valence_b = circuit.forward(circuit.encode_odor(0.0, 1.0)[1])

        # 2. Johnston's Organ mechanosensory wind deflection
        if f.ablate_jo:
            wind_data = {'rel_wind_angle': 0.0, 'deflection_amplitude': 0.0, 'left_deflection': 0.0, 'right_deflection': 0.0}
            wind_relative = 0.0
        else:
            wind_data = jo.step(
                fly_heading=f.heading,
                fly_speed=f.speed,
                wind_vx=w_vec[0],
                wind_vy=w_vec[1]
            )
            wind_relative = wind_data['rel_wind_angle']

        # 3. Visual Optic Flow & LC4 Looming Threat Detection
        pred_pos_list = [p.pos.to_array() for p in self.predators]
        pred_vel_list = [p.get_velocity() for p in self.predators]
        vis_data = vision.step(
            fly_pos=f.pos.to_array(),
            fly_heading=f.heading,
            fly_speed=f.speed,
            fly_yaw_rate=f.angular_velocity,
            predator_positions=pred_pos_list,
            predator_velocities=pred_vel_list,
            external_yaw_rad_s=math.radians(float(kwargs.get("drum_velocity_deg_s", 0.0))),
            contrast=float(kwargs.get("visual_contrast", 1.0)),
            dt=dt
        )
        if f.ablate_lc4:
            vis_data['escape_active'] = False

        # 4. Step Surge-Cast Engine
        v_surge, omega_surge, state = surge_cast.step(
            c_left=sensory['left_a'],
            c_right=sensory['right_a'],
            wind_angle_rad=wind_relative,
            is_feeding=is_feeding, dt=dt, stop_rate_scale=metabolic.get_stop_suppression()
        )

        # Apply metabolic hunger modulation
        v_surge *= metabolic.get_surge_multiplier()

        # In still air, reduce approach speed in concentrated attractive odor.
        # Without this local braking, proportional taxis creates a permanent orbit
        # around a point source as its bilateral gradient vanishes near the peak.
        if np.linalg.norm(w_vec) < 1 and sensory['mean_a'] > .5 and valence_a >= 0:
            v_surge *= max(.08, min(1.0, (1.0 - sensory['mean_a']) / .5))

        # 5. Step Central Complex Engine
        if f.ablate_cx:
            pfl3_torque = 0.0
            compass_heading = float(f.heading)
            goal_angle = float(f.heading)
        else:
            pfl3_torque, compass_heading, goal_angle = cx.step(
                angular_vel=f.angular_velocity,
                v_forward=v_surge,
                mbon_valence=valence_a,
                current_fly_heading=f.heading,
                dt=dt
            )

        # 6. Spatial Chemotaxis (Tropotaxis)
        steering_gain = 12.0
        # Normalize bilateral differences by local concentration; a dilute plume
        # must retain a directional signal. Innate attraction/aversion is explicit
        # and learned MB values can strengthen or reverse it (heuristic gains).
        chemotaxis = steering_gain * (
            (.25 + valence_a) * sensory['diff_a'] / (sensory['mean_a'] + .05)
            + (-.25 + valence_b) * sensory['diff_b'] / (sensory['mean_b'] + .05))

        # 7. Sensorimotor Integration & Escape Override
        if vis_data['escape_active']:
            # Emergency LC4 ballistic evasion overrides other drives!
            state = 'ESCAPE'
            f.escapes_performed += 1
            self.total_escapes += 1
            # Steer rapidly toward escape target heading
            target_diff = math.atan2(
                math.sin(vis_data['escape_heading_target'] - f.heading),
                math.cos(vis_data['escape_heading_target'] - f.heading)
            )
            total_turn = np.clip(target_diff * 4.0, -0.65, 0.65)
            new_speed = 3.5  # High-speed escape sprint
        elif state == 'SURGE':
            total_turn = 0.55 * chemotaxis + 0.35 * omega_surge + 0.15 * pfl3_torque
            new_speed = v_surge
        elif state == 'CAST':
            total_turn = 0.20 * chemotaxis + 0.65 * omega_surge + 0.15 * pfl3_torque
            new_speed = v_surge
        elif state == 'FEED':
            total_turn = 0.0
            new_speed = 0.2
        else:
            wandering_noise = self.rng.gauss(0.0, 0.12)
            total_turn = chemotaxis + wandering_noise + 0.25 * pfl3_torque
            new_speed = v_surge

        # Add optomotor yaw stabilization from vision
        total_turn += vision.get_optomotor_yaw_bias()
        dheading = float(np.clip(total_turn, -2.5, 2.5))

        dheading, new_speed, state = assay_response(
            f, kwargs.get("assay_stimuli", {"temperature": temperature}),
            dheading, new_speed, state, dt)

        # 8. CPG Tripod Gait Stepping Drive
        dn_left = max(0.0, 1.0 - dheading * 1.5)
        dn_right = max(0.0, 1.0 + dheading * 1.5)
        cpg_data = cpg.step(dn_left, dn_right, dt=dt)

        return dheading, new_speed, state, compass_heading, goal_angle

    def enforce_containment(self, fly: FlyState, dt: float = 0.02) -> bool:
        """Hard geometric failsafe: keep the body inside the paradigm's legal region.

        Runs after collision resolution and the wall reflex, so it only fires when
        those have already failed. When it does fire it behaves like a wall contact
        rather than a silent clamp: the position is projected back inside, the speed
        component driving into the boundary is removed, and the heading receives the
        same away-from-wall torque as a real collision, so the fly can never be left
        pushing into an invisible boundary step after step.
        That torque is the engineered ``contact_turn`` assist: with it disabled the
        failsafe only projects the body inside and removes the into-wall speed.
        ``self.last_failsafe`` holds (correction_mm, nx, ny, turn_rad) of the last call.
        Returns True when the position had to be corrected.
        """
        self.last_failsafe = (0.0, 0.0, 0.0, 0.0)
        if not fly or not fly.alive:
            return False

        r = getattr(fly, 'radius', 1.5)
        x, y = fly.pos.x, fly.pos.y
        if self.containment.signed_gap(x, y) >= r:
            return False

        nx, ny = self.containment.inward_normal(x, y)
        fly.pos.x, fly.pos.y = self.containment.clamp(x, y, r)
        turn_applied = 0.0

        travel = fly.heading if fly.speed >= 0.0 else fly.heading + math.pi
        hx, hy = math.cos(travel), math.sin(travel)
        into = -(hx * nx + hy * ny)
        if into > 0.0:
            # Keep only the tangential share of the commanded speed (no bounce)
            fly.speed = float(fly.speed) * math.sqrt(max(0.0, 1.0 - into * into))
            # Retain the direction selected from all nearby boundaries. The
            # nearest face alternates at a corner and otherwise reverses this
            # torque every tick, trapping the fly against the same corner.
            if self.motor_assists.get('contact_turn', True):
                turn_applied = fly.wall_turn_dir * min(self.WALL_AVOID_YAW_RAD_S * dt, math.pi / 2.0)
                fly.heading = (fly.heading + turn_applied) % (2.0 * math.pi)
        self.last_failsafe = (math.hypot(fly.pos.x - x, fly.pos.y - y), nx, ny, turn_applied)
        return True

    def step(self, dt: float = 1.0) -> Dict:
        """Execute one simulation tick for all flies and predators."""
        self.time_step += 1

        if self.paradigm is not None:
            # -------------------------------------------------------------
            # EXPERIMENT PARADIGM STEPPING
            # -------------------------------------------------------------
            paradigm_res: Dict[str, Any] = {}
            zone_names: List[str] = []
            stimuli: Dict[str, Any] = {}
            reward = 0.0
            punishment = 0.0

            for fly in self.flies:
                if not fly.alive:
                    continue

                radius = getattr(fly, 'radius', 1.5)
                vx = fly.speed * math.cos(fly.heading)
                vy = fly.speed * math.sin(fly.heading)
                step_start = (fly.pos.x, fly.pos.y)
                rec: Dict[str, Any] = {'step': self.time_step, 'dt': dt}

                # 1. Static overlap check against paradigm geometry. The position is
                # already resolved by the previous step, so this only acts when a pose
                # was set from outside (reset, test, restore). The swept start is the
                # current pose itself: inventing prev = pos - v*dt pushed a fly in
                # contact an extra v*dt away from the wall, and |v| flipped reverse speed.
                # dt=0: resolve the overlap only; the step's motion happens once, below.
                col_x, col_y, new_vx, new_vy, collided = self.paradigm.check_collisions(
                    fly.pos.x, fly.pos.y, vx, vy, radius=radius,
                    prev_x=fly.pos.x, prev_y=fly.pos.y, dt=0.0
                )
                rec['overlap_correction_mm'] = math.hypot(col_x - fly.pos.x, col_y - fly.pos.y)
                fly.pos.x = col_x
                fly.pos.y = col_y
                if collided:
                    fly.speed = math.copysign(math.hypot(new_vx, new_vy), fly.speed)

                # 2. Query paradigm step
                paradigm_res = self.paradigm.step(fly, dt)

                # 3. Sample multi-modal stimuli (temperature, wind, odor, landmarks, laser, grating)
                try:
                    stimuli = self.paradigm.sample_stimuli(fly.pos.to_tuple(), fly.heading)
                except TypeError:
                    stimuli = self.paradigm.sample_stimuli(fly.pos.x, fly.pos.y, fly.heading)

                # 4. Check active zones for reward/punishment triggers
                active_zones = self.paradigm.get_active_zones(fly.pos.x, fly.pos.y)
                zone_names = [z.name for z in active_zones]
                zone_reward = sum(z.reward for z in active_zones)
                zone_punishment = sum(z.punishment for z in active_zones)

                reward = max(zone_reward, float(paradigm_res.get('reward', 0.0)))
                punishment = max(zone_punishment, float(paradigm_res.get('punishment', 0.0)))

                # Parse multi-modal sensory cues
                temp = float(stimuli.get('temperature', 25.0))
                wind = stimuli.get('wind', self.wind)
                if isinstance(wind, (list, tuple)) and len(wind) == 2:
                    wind_vec = np.array(wind, dtype=np.float64)
                else:
                    wind_vec = np.array(self.wind, dtype=np.float64)

                if stimuli.get('laser_active', False):
                    punishment = max(punishment, 1.0)
                    temp = max(temp, 40.0)

                # Sample physical fields at both antennae, including odor_conc assays.
                sensory = self._sample_paradigm_antennae(fly, stimuli)
                stimuli.update({k:sensory[k] for k in ('temperature_left','temperature_right')})
                stimuli['odor_a'], stimuli['odor_b'] = sensory['mean_a'], sensory['mean_b']
                stimuli['wind'] = list(wind_vec)
                if self.paradigm_key(self.paradigm) == 'gap_crossing':
                    stimuli.update({k:paradigm_res.get(k) for k in ('decision_outcome','probing_duration_ms')})
                fly.sensory_input = dict(sensory)

                landmarks = stimuli.get('landmark_bearings', stimuli.get('stripe_bearings', getattr(self.paradigm, 'landmarks', None)))
                cva_odor = float(stimuli.get('cva_concentration', 0.0))
                aphrodisiac = float(stimuli.get('aphrodisiac_concentration', 0.0))
                bitter_phero = 1.0 if (stimuli.get('female_type') == 'mated' and stimuli.get('inter_fly_distance_mm', 999.0) < 2.5) else 0.0
                is_saccade = stimuli.get('is_saccade', None)

                # Reinforcement is not ingestion. Relief, social cues and successful
                # crossings must never masquerade as eating or produce infinite FEED.
                fly.metabolic.step(dt=dt, speed=abs(fly.speed))
                contact = bool(stimuli.get('food_contact', False))
                if contact and not getattr(fly, 'food_contact_active', False) and not fly.metabolic.is_satiated:
                    fly.feeding_remaining = 1.0
                    fly.food_collected += 1
                    self.food_collected += 1
                fly.food_contact_active = contact
                feeding = contact and getattr(fly, 'feeding_remaining', 0) > 0
                if feeding:
                    fly.metabolic.feed(.35 * dt)
                fly.feeding_remaining = max(0.0, getattr(fly, 'feeding_remaining', 0) - dt)

                if punishment > 0.0:
                    self.hazard_encounters += 1

                # Mushroom Body learning step (if modular)
                if not fly.ablate_mb and hasattr(fly, 'circuit') and fly.circuit is not None:
                    dopamine_gain = fly.metabolic.get_dopamine_gain() if hasattr(fly, 'metabolic') else 1.0
                    fly.circuit.advance(
                        odor_a=sensory['mean_a'],
                        odor_b=sensory['mean_b'],
                        reward=reward * dopamine_gain,
                        punishment=punishment,
                        dt_seconds=dt,
                        learning=getattr(fly, "learning_enabled", True)
                    )

                # 5. Feed sampled stimuli into fly brain / connectome bridge
                dheading, new_speed, state, compass_h, goal_a = self.compute_steering(
                    sensory=sensory,
                    is_feeding=feeding,
                    fly=fly,
                    temperature=temp,
                    wind_vector=wind_vec,
                    landmarks=landmarks,
                    cva_odor=cva_odor,
                    bitter_pheromone=bitter_phero,
                    female_aphrodisiac=aphrodisiac,
                    incurred_damage=(punishment > 0.0),
                    is_saccade=is_saccade, dt=dt,
                    drum_velocity_deg_s=stimuli.get("drum_velocity_deg_s", 0.0),
                    visual_contrast=stimuli.get("contrast", 1.0),
                    assay_stimuli=stimuli
                )

                # The assay's expanding disk is an actual visual input, not merely
                # a metric counter. A GF event triggers a bounded motor escape.
                if paradigm_res.get('gf_spike') and not fly.ablate_lc4:
                    fly.assay_escape_remaining = 0.2
                    fly.escapes_performed += 1
                    self.total_escapes += 1
                halted = bool(getattr(fly, 'motor_halted', False))
                if getattr(fly, 'assay_escape_remaining', 0.0) > 0:
                    fly.assay_escape_remaining = max(0.0, fly.assay_escape_remaining - dt)
                    if not halted:   # a halted controller commands no escape run either
                        new_speed, state = 3.5, 'ESCAPE'
                fly.speed = new_speed
                fly.behavioral_state = state
                fly.compass_heading = compass_h
                fly.goal_angle = goal_a
                rec['state'] = state
                rec['controller_yaw'] = float(dheading)
                rec['controller_speed'] = float(new_speed)
                rec['motor_source'] = getattr(fly, 'motor_source', 'modular')
                rec['controller_fault'] = getattr(fly, 'controller_fault', None)
                rec['halted'] = halted

                # Wall perception: turn away from boundaries before touching them
                # (engineered assist, logged separately from the controller's yaw).
                # A halted fly receives no assist yaw: zero motion means zero.
                if not halted:
                    dheading = self.wall_avoidance_turn(fly, dheading, dt)
                else:
                    fly.wall_reflex_suppressed = False
                rec['wall_reflex_yaw'] = float(dheading) - rec['controller_yaw']
                rec['controller_yaw_suppressed'] = bool(getattr(fly, 'wall_reflex_suppressed', False))

                # Save pre-update position for true continuous swept trajectory
                prev_x = fly.pos.x
                prev_y = fly.pos.y

                # Advance heading from brain yaw
                fly.angular_velocity = dheading
                fly.heading = (fly.heading + dheading * dt) % (2.0 * math.pi)

                # Proposed position step
                vx_step = fly.speed * math.cos(fly.heading)
                vy_step = fly.speed * math.sin(fly.heading)
                tethered = self.paradigm_key(self.paradigm) in ('visual_operant', 'optomotor')
                prop_x = prev_x if tethered else prev_x + vx_step * dt
                prop_y = prev_y if tethered else prev_y + vy_step * dt
                if tethered:
                    vx_step = vy_step = 0.0
                rec['tethered'] = tethered
                rec['attempted_dx'] = prop_x - prev_x
                rec['attempted_dy'] = prop_y - prev_y
                rec['attempted_mm'] = math.hypot(rec['attempted_dx'], rec['attempted_dy'])

                # Simulation-grade continuous swept collision resolution
                if hasattr(self.paradigm, 'check_collisions_advanced'):
                    res_x, res_y, res_vx, res_vy, collided, normals = self.paradigm.check_collisions_advanced(
                        prop_x, prop_y, vx_step, vy_step, radius=radius,
                        prev_x=prev_x, prev_y=prev_y, dt=dt
                    )
                else:
                    col_res = self.paradigm.check_collisions(
                        prop_x, prop_y, vx_step, vy_step, radius=radius,
                        prev_x=prev_x, prev_y=prev_y, dt=dt
                    )
                    res_x, res_y, res_vx, res_vy, collided = col_res
                    normals = []

                fly.pos.x = res_x
                fly.pos.y = res_y
                fly.speed = math.copysign(math.hypot(res_vx, res_vy), new_speed)
                rec['solver_correction_mm'] = math.hypot(res_x - prop_x, res_y - prop_y)
                rec['contact_normals'] = [(float(n[0]), float(n[1])) for n in normals] if collided else []
                self.enforce_containment(fly, dt)
                fs_mm, fs_nx, fs_ny, fs_turn = self.last_failsafe
                rec['failsafe_correction_mm'] = fs_mm
                rec['contact_turn_rad'] = fs_turn
                if fs_mm > 0.0:
                    rec['contact_normals'].append((fs_nx, fs_ny))
                self._finish_motor_record(fly, rec, step_start, dt)

                if collided and normals:
                    # Continuous physical contact torque steering (zero angular teleportation)
                    net_nx = sum(n[0] for n in normals)
                    net_ny = sum(n[1] for n in normals)
                    n_mag = math.hypot(net_nx, net_ny)
                    if n_mag > 1e-6:
                        net_nx /= n_mag
                        net_ny /= n_mag

                        # Cuticular mechanosensory ingress: antennal deflection
                        if hasattr(fly, 'mechanosensory') and fly.mechanosensory is not None:
                            h_cross_n = math.cos(fly.heading) * net_ny - math.sin(fly.heading) * net_nx
                            if h_cross_n > 0:
                                fly.mechanosensory.deflect_left = min(
                                    fly.mechanosensory.max_deflect,
                                    getattr(fly.mechanosensory, 'deflect_left', 0.0) + 0.35
                                )
                            else:
                                fly.mechanosensory.deflect_right = min(
                                    fly.mechanosensory.max_deflect,
                                    getattr(fly.mechanosensory, 'deflect_right', 0.0) + 0.35
                                )

            self.total_distance += abs(self.fly.speed) * dt
            metrics = self.paradigm.get_metrics()

            return {
                'time_step': self.time_step,
                'fly_x': self.fly.pos.x,
                'fly_y': self.fly.pos.y,
                'fly_heading': self.fly.heading,
                'fly_speed': self.fly.speed,
                'state': self.fly.behavioral_state,
                'satiety': self.fly.metabolic.satiety if hasattr(self.fly, 'metabolic') else 1.0,
                'food_collected': self.food_collected,
                'hazard_encounters': self.hazard_encounters,
                'predator_kills': self.total_predator_kills,
                'escapes': self.total_escapes,
                'connectome': getattr(self.fly, 'last_connectome_telemetry', None),
                'paradigm': self.paradigm.name,
                'paradigm_telemetry': paradigm_res,
                'paradigm_metrics': metrics,
                'active_zones': zone_names,
                'stimuli': stimuli,
                'reward': reward,
                'punishment': punishment,
                'motor': dict(getattr(self.fly, 'motor_record', None) or {}),
            }

        # -----------------------------------------------------------------
        # STANDARD / BACKWARD-COMPATIBLE SIMULATION (paradigm is None)
        # -----------------------------------------------------------------
        # 1. Update Predators
        for pred in self.predators:
            caught_fly_id = pred.step(self.flies, self.width, self.height, dt)
            if caught_fly_id is not None:
                self.total_predator_kills += 1
                for fly in self.flies:
                    if fly.id == caught_fly_id:
                        # Incapacitated fly emits an aversive alarm pheromone Odor B
                        self.odor_b.add_source(fly.pos.x, fly.pos.y, 1.8)
                        # Respawn fly in safe perimeter
                        fly.pos.x = self.rng.uniform(20.0, self.width - 20.0)
                        fly.pos.y = self.rng.uniform(20.0, self.height - 20.0)
                        fly.metabolic.reset(0.5)
                        fly.vision.reset()
                        break

        # 2. Update Flies
        for fly in self.flies:
            if not fly.alive:
                continue

            sensory = self.sample_antennae(fly)
            fly.sensory_input = dict(sensory)

            # Check interactions with Food and Hazards
            reward = 0.0
            punishment = 0.0
            food_eaten_this_step = False

            for i, food in enumerate(list(self.food_positions)):
                if fly.pos.distance_to(food) < 3.5:
                    reward = 1.0
                    food_eaten_this_step = True
                    fly.food_collected += 1
                    self.food_collected += 1
                    fly.metabolic.feed(0.35)
                    self.time_to_food_history.append(self.time_step - self.last_food_step)
                    self.last_food_step = self.time_step

                    # Respawn food at a new random location
                    new_fx = self.rng.uniform(15.0, self.width - 15.0)
                    new_fy = self.rng.uniform(15.0, self.height - 15.0)
                    self.food_positions[i] = Position(new_fx, new_fy)
                    self.odor_a.clear()
                    for f in self.food_positions:
                        self.odor_a.add_source(f.x, f.y, 1.0)
                    break

            for hazard in self.hazard_positions:
                if fly.pos.distance_to(hazard) < 4.0:
                    punishment = 1.0
                    self.hazard_encounters += 1
                    break

            # Advance metabolic hunger
            fly.metabolic.step(dt=dt, speed=abs(fly.speed))

            # Modulate dopamine learning rate by hunger
            dopamine_gain = fly.metabolic.get_dopamine_gain()

            # Step Mushroom Body circuit with modulated dopamine (unless ablated)
            if not fly.ablate_mb:
                fly.circuit.advance(
                    odor_a=sensory['mean_a'],
                    odor_b=sensory['mean_b'],
                    reward=reward * dopamine_gain,
                    punishment=punishment,
                    dt_seconds=dt,
                    learning=getattr(fly, "learning_enabled", True)
                )

            # Compute steering and advance kinematics
            dheading, new_speed, state, compass_h, goal_a = self.compute_steering(
                sensory, is_feeding=food_eaten_this_step, fly=fly, dt=dt
            )
            fly.speed = new_speed
            fly.behavioral_state = state
            fly.compass_heading = compass_h
            fly.goal_angle = goal_a
            step_start = (fly.pos.x, fly.pos.y)
            rec = {'step': self.time_step, 'dt': dt, 'state': state, 'controller_yaw': float(dheading),
                   'controller_speed': float(new_speed), 'wall_reflex_yaw': 0.0,
                   'motor_source': getattr(fly, 'motor_source', 'modular'),
                   'controller_fault': getattr(fly, 'controller_fault', None),
                   'halted': bool(getattr(fly, 'motor_halted', False)),
                   'controller_yaw_suppressed': False, 'tethered': False, 'overlap_correction_mm': 0.0}
            fly.update(dheading, dt)
            rec['attempted_dx'] = fly.pos.x - step_start[0]
            rec['attempted_dy'] = fly.pos.y - step_start[1]
            rec['attempted_mm'] = math.hypot(rec['attempted_dx'], rec['attempted_dy'])
            unclamped = (fly.pos.x, fly.pos.y)

            # Smooth physical boundary steering
            margin = 2.0
            wall_nx, wall_ny = 0.0, 0.0
            if fly.pos.x < margin:
                fly.pos.x = margin
                wall_nx += 1.0
            elif fly.pos.x > self.width - margin:
                fly.pos.x = self.width - margin
                wall_nx -= 1.0

            if fly.pos.y < margin:
                fly.pos.y = margin
                wall_ny += 1.0
            elif fly.pos.y > self.height - margin:
                fly.pos.y = self.height - margin
                wall_ny -= 1.0

            rec['solver_correction_mm'] = 0.0
            rec['contact_normals'] = []
            rec['contact_turn_rad'] = 0.0
            rec['failsafe_correction_mm'] = math.hypot(fly.pos.x - unclamped[0], fly.pos.y - unclamped[1])
            if wall_nx != 0.0 or wall_ny != 0.0:
                n_mag = math.hypot(wall_nx, wall_ny)
                wall_nx /= n_mag
                wall_ny /= n_mag
                rec['contact_normals'] = [(wall_nx, wall_ny)]
                # Engineered contact_turn assist: continuous torque away from the edge
                if self.motor_assists.get('contact_turn', True):
                    h_cross_n = math.cos(fly.heading) * wall_ny - math.sin(fly.heading) * wall_nx
                    turn_dir = 1.0 if h_cross_n >= 0.0 else -1.0
                    rec['contact_turn_rad'] = turn_dir * 4.0 * dt
                    fly.heading = (fly.heading + rec['contact_turn_rad']) % (2.0 * math.pi)
            self._finish_motor_record(fly, rec, step_start, dt)

        self.total_distance += self.fly.speed * dt

        return {
            'time_step': self.time_step,
            'fly_x': self.fly.pos.x,
            'fly_y': self.fly.pos.y,
            'fly_heading': self.fly.heading,
            'fly_speed': self.fly.speed,
            'state': self.fly.behavioral_state,
            'stimuli': {'odor_a': sensory['mean_a'], 'odor_b': sensory['mean_b'], 'wind': self.wind},
            'reward': reward, 'punishment': punishment,
            'satiety': self.fly.metabolic.satiety,
            'food_collected': self.food_collected,
            'hazard_encounters': self.hazard_encounters,
            'predator_kills': self.total_predator_kills,
            'escapes': self.total_escapes,
            'connectome': getattr(self.fly, 'last_connectome_telemetry', None),
            'motor': dict(getattr(self.fly, 'motor_record', None) or {}),
        }
