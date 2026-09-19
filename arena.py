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

import math
import random
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np

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
        connectome_port: int = 8768
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
        self.vision = CompoundEyeVision(num_ommatidia=72)
        self.cpg = TripodGaitCPG()

        # Whole-Brain Connectome Bridge
        self.connectome_bridge = None
        self.last_connectome_telemetry = None
        if self.brain_type == 'connectome' and ConnectomeBridge is not None:
            self.connectome_bridge = ConnectomeBridge(
                mode=connectome_mode,
                rpc_host=connectome_host,
                rpc_port=connectome_port
            )

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


class Arena:
    """Multi-Agent 2D Simulation Arena with Drosophila and Stalking Predators."""

    FLY_COLORS = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#AB47BC', '#00ACC1', '#FF7043']

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
        paradigm: Optional[Union[Any, str]] = None
    ):
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
                connectome_port=connectome_port
            )
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

        # Ecological Metrics
        self.time_step = 0
        self.food_collected = 0
        self.hazard_encounters = 0
        self.total_predator_kills = 0
        self.total_escapes = 0
        self.total_distance = 0.0
        self.time_to_food_history: List[int] = []
        self.last_food_step = 0

    def _get_paradigm_spawn(self, paradigm: Any) -> Tuple[float, float, float]:
        """Compute initial fly spawn coordinates (x, y, heading) based on paradigm geometry."""
        if hasattr(paradigm, 'spawn_pos') and paradigm.spawn_pos is not None:
            sp = paradigm.spawn_pos
            return float(sp[0]), float(sp[1]), float(getattr(paradigm, 'spawn_heading', 0.0))

        p_name = getattr(paradigm, 'name', '').lower().replace('-', '_')
        if 't_maze' in p_name:
            # Stem base center (x=70, y=18), facing up stem (+y, pi/2)
            return 70.0, 18.0, math.pi / 2.0
        elif 'y_maze' in p_name:
            # Central hub
            return 60.0, 60.0, 0.0
        elif 'heat_maze' in p_name:
            # Circular platform center
            return 60.0, 60.0, 0.0
        elif 'buridan' in p_name:
            # Circular platform center
            return 60.0, 60.0, 0.0
        elif 'visual_operant' in p_name:
            # Flight simulator center
            return 40.0, 40.0, 0.0
        elif 'wind_tunnel' in p_name:
            # Downwind release point facing upwind (East, heading 0.0)
            return 20.0, 30.0, 0.0
        elif 'looming_escape' in p_name:
            # Center of stage
            return 40.0, 40.0, 0.0
        elif 'optomotor' in p_name:
            # Center of drum
            return 45.0, 45.0, 0.0
        elif 'gap_crossing' in p_name:
            # Takeoff track start
            return 20.0, 10.0, 0.0
        elif 'circadian_dam' in p_name:
            # Tube 0 start
            return 15.0, 5.0, 0.0
        elif 'courtship' in p_name:
            # Male facing female at center (10, 10)
            return 10.0, 13.0, -math.pi / 2.0
        elif 'labyrinth' in p_name:
            # Maze entrance
            return 15.0, 15.0, 0.0
        else:
            return self.width / 2.0, self.height / 2.0, 0.0

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
        **kwargs
    ) -> Tuple[float, float, str, float, float]:
        f = fly or self.fly
        pred_pos_list = [p.pos.to_array() for p in self.predators]
        pred_vel_list = [p.get_velocity() for p in self.predators]
        w_vec = wind_vector if wind_vector is not None else np.array(self.wind, dtype=np.float64)

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
                dt=0.01,
                temperature=temperature,
                landmarks=landmarks,
                cva_odor=cva_odor,
                bitter_pheromone=bitter_pheromone,
                female_aphrodisiac=female_aphrodisiac,
                is_saccade=is_saccade,
                **kwargs
            )
            f.last_connectome_telemetry = bridge_out
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

        # Thermal stress in modular brain: MDN-like backward walking
        if temperature > 35.0:
            state = 'REVERSE'
            new_speed = -0.5
            dheading = float(np.clip(self.rng.uniform(-0.25, 0.25), -0.45, 0.45))
            return dheading, new_speed, state, float(f.heading), float(f.heading)

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
            dt=0.01
        )
        if f.ablate_lc4:
            vis_data['escape_active'] = False

        # 4. Step Surge-Cast Engine
        v_surge, omega_surge, state = surge_cast.step(
            c_left=sensory['left_a'],
            c_right=sensory['right_a'],
            wind_angle_rad=wind_relative,
            is_feeding=is_feeding
        )

        # Apply metabolic hunger modulation
        v_surge *= metabolic.get_surge_multiplier()

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
                dt=0.01
            )

        # 6. Spatial Chemotaxis (Tropotaxis)
        steering_gain = 3.5
        chemotaxis = steering_gain * (valence_a * sensory['diff_a'] + valence_b * sensory['diff_b'])

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
        dheading = float(np.clip(total_turn, -0.45, 0.45))

        # 8. CPG Tripod Gait Stepping Drive
        dn_left = max(0.0, 1.0 - dheading * 1.5)
        dn_right = max(0.0, 1.0 + dheading * 1.5)
        cpg_data = cpg.step(dn_left, dn_right, dt=0.01)

        return dheading, new_speed, state, compass_heading, goal_angle

    def enforce_containment(self, fly: FlyState) -> bool:
        """Absolute geometric bounding enforcement preventing any out-of-bounds clipping across all paradigms."""
        if not fly or not fly.alive:
            return False

        r = getattr(fly, 'radius', 1.5)
        clamped = False
        p_name = getattr(self.paradigm, 'name', '').lower().replace('-', '_') if self.paradigm else ''

        if 't_maze' in p_name:
            # Stem: x in [63+r, 77-r], y in [10+r, 43.0]
            # Arms: x in [10+r, 130-r], y in [43.0, 57-r]
            # Junction: x in [63+r, 77-r], y in [43.0, 57-r]
            in_stem = (63.0 + r <= fly.pos.x <= 77.0 - r) and (10.0 + r <= fly.pos.y <= 43.0)
            in_arms = (10.0 + r <= fly.pos.x <= 130.0 - r) and (43.0 <= fly.pos.y <= 57.0 - r)
            if not (in_stem or in_arms):
                clamped = True
                if fly.pos.y < 43.0:
                    fly.pos.x = max(63.0 + r, min(77.0 - r, fly.pos.x))
                    fly.pos.y = max(10.0 + r, min(43.0, fly.pos.y))
                else:
                    fly.pos.x = max(10.0 + r, min(130.0 - r, fly.pos.x))
                    fly.pos.y = max(43.0, min(57.0 - r, fly.pos.y))

        elif 'y_maze' in p_name:
            # Distance from hub (60, 60) max 50mm
            cx, cy, max_r = 60.0, 60.0, 48.0 - r
            d = math.hypot(fly.pos.x - cx, fly.pos.y - cy)
            if d > max_r:
                clamped = True
                fly.pos.x = cx + (fly.pos.x - cx) * (max_r / d)
                fly.pos.y = cy + (fly.pos.y - cy) * (max_r / d)

        elif any(k in p_name for k in ['heat_maze', 'buridan', 'courtship', 'optomotor', 'visual_operant']):
            # Circular platform
            cx = self.width / 2.0
            cy = self.height / 2.0
            max_r = min(self.width, self.height) * 0.48 - r
            d = math.hypot(fly.pos.x - cx, fly.pos.y - cy)
            if d > max_r:
                clamped = True
                fly.pos.x = cx + (fly.pos.x - cx) * (max_r / d)
                fly.pos.y = cy + (fly.pos.y - cy) * (max_r / d)

        elif 'wind_tunnel' in p_name:
            orig_x, orig_y = fly.pos.x, fly.pos.y
            fly.pos.x = max(r, min(200.0 - r, fly.pos.x))
            fly.pos.y = max(r, min(60.0 - r, fly.pos.y))
            if fly.pos.x != orig_x or fly.pos.y != orig_y:
                clamped = True

        elif 'gap_crossing' in p_name:
            orig_x, orig_y = fly.pos.x, fly.pos.y
            fly.pos.x = max(r, min(100.0 - r, fly.pos.x))
            fly.pos.y = max(7.5 + r, min(12.5 - r, fly.pos.y))
            if fly.pos.x != orig_x or fly.pos.y != orig_y:
                clamped = True

        elif 'circadian_dam' in p_name:
            # Active tube 0: y in [0, 10], x in [0, 65]
            orig_x, orig_y = fly.pos.x, fly.pos.y
            fly.pos.x = max(r, min(65.0 - r, fly.pos.x))
            fly.pos.y = max(r, min(10.0 - r, fly.pos.y))
            if fly.pos.x != orig_x or fly.pos.y != orig_y:
                clamped = True

        elif 'labyrinth' in p_name:
            orig_x, orig_y = fly.pos.x, fly.pos.y
            fly.pos.x = max(r, min(140.0 - r, fly.pos.x))
            fly.pos.y = max(r, min(100.0 - r, fly.pos.y))
            if fly.pos.x != orig_x or fly.pos.y != orig_y:
                clamped = True

        else:
            # Default rectangular arena containment
            orig_x, orig_y = fly.pos.x, fly.pos.y
            fly.pos.x = max(r, min(self.width - r, fly.pos.x))
            fly.pos.y = max(r, min(self.height - r, fly.pos.y))
            if fly.pos.x != orig_x or fly.pos.y != orig_y:
                clamped = True

        return clamped

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

                # 1. Collision check against paradigm geometry with sliding physics
                col_x, col_y, new_vx, new_vy, collided = self.paradigm.check_collisions(
                    fly.pos.x, fly.pos.y, vx, vy, radius=radius
                )
                fly.pos.x = col_x
                fly.pos.y = col_y
                if collided:
                    fly.speed = math.hypot(new_vx, new_vy)

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

                # Odor representation
                if 'odor_cs_plus' in stimuli and 'odor_cs_minus' in stimuli:
                    odor_l = stimuli['odor_cs_plus']
                    odor_r = stimuli['odor_cs_minus']
                    sensory = {
                        'left_a': odor_l, 'right_a': odor_r,
                        'mean_a': 0.5 * (odor_l + odor_r), 'diff_a': odor_l - odor_r,
                        'left_b': 0.0, 'right_b': 0.0, 'mean_b': 0.0, 'diff_b': 0.0
                    }
                elif 'odor_conc' in stimuli:
                    c = stimuli['odor_conc']
                    sensory = {
                        'left_a': c, 'right_a': c, 'mean_a': c, 'diff_a': 0.0,
                        'left_b': 0.0, 'right_b': 0.0, 'mean_b': 0.0, 'diff_b': 0.0
                    }
                else:
                    sensory = self.sample_antennae(fly)

                landmarks = stimuli.get('landmark_bearings', stimuli.get('stripe_bearings', getattr(self.paradigm, 'landmarks', None)))
                cva_odor = float(stimuli.get('cva_concentration', 0.0))
                aphrodisiac = float(stimuli.get('aphrodisiac_concentration', 0.0))
                bitter_phero = 1.0 if (stimuli.get('female_type') == 'mated' and stimuli.get('inter_fly_distance_mm', 999.0) < 2.5) else 0.0
                is_saccade = stimuli.get('is_saccade', None)

                # Food ingestion / metabolic feed
                if reward > 0.0:
                    fly.food_collected += 1
                    self.food_collected += 1
                    fly.metabolic.feed(0.35 * reward)

                if punishment > 0.0:
                    self.hazard_encounters += 1

                # Mushroom Body learning step (if modular)
                if not fly.ablate_mb and hasattr(fly, 'circuit') and fly.circuit is not None:
                    dopamine_gain = fly.metabolic.get_dopamine_gain() if hasattr(fly, 'metabolic') else 1.0
                    fly.circuit.step(
                        odor_a=sensory['mean_a'],
                        odor_b=sensory['mean_b'],
                        reward=reward * dopamine_gain,
                        punishment=punishment,
                        dt_seconds=0.01 * dt,
                        learning=True
                    )

                # 5. Feed sampled stimuli into fly brain / connectome bridge
                dheading, new_speed, state, compass_h, goal_a = self.compute_steering(
                    sensory=sensory,
                    is_feeding=(reward > 0.0),
                    fly=fly,
                    temperature=temp,
                    wind_vector=wind_vec,
                    landmarks=landmarks,
                    cva_odor=cva_odor,
                    bitter_pheromone=bitter_phero,
                    female_aphrodisiac=aphrodisiac,
                    incurred_damage=(punishment > 0.0),
                    is_saccade=is_saccade
                )

                fly.speed = new_speed
                fly.behavioral_state = state
                fly.compass_heading = compass_h
                fly.goal_angle = goal_a

                # Save pre-update position for true continuous swept trajectory
                prev_x = fly.pos.x
                prev_y = fly.pos.y

                # Advance heading from brain yaw
                fly.angular_velocity = dheading
                fly.heading = (fly.heading + dheading * dt) % (2.0 * math.pi)

                # Proposed position step
                vx_step = fly.speed * math.cos(fly.heading)
                vy_step = fly.speed * math.sin(fly.heading)
                prop_x = prev_x + vx_step * dt
                prop_y = prev_y + vy_step * dt

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
                fly.speed = math.hypot(res_vx, res_vy)
                self.enforce_containment(fly)

                if collided and normals:
                    # Continuous physical contact torque steering (zero angular teleportation)
                    net_nx = sum(n[0] for n in normals)
                    net_ny = sum(n[1] for n in normals)
                    n_mag = math.hypot(net_nx, net_ny)
                    if n_mag > 1e-6:
                        net_nx /= n_mag
                        net_ny /= n_mag

                        if fly.speed > 0.05:
                            # Align body smoothly with sliding velocity
                            slide_angle = math.atan2(res_vy, res_vx)
                            dtheta = (slide_angle - fly.heading + math.pi) % (2.0 * math.pi) - math.pi
                            fly.heading = (fly.heading + dtheta * min(1.0, 15.0 * dt)) % (2.0 * math.pi)
                        else:
                            # Head-on or wedged in corner: gentle repulsion torque away from wall
                            h_cross_n = math.cos(fly.heading) * net_ny - math.sin(fly.heading) * net_nx
                            turn_dir = 1.0 if h_cross_n >= 0.0 else -1.0
                            fly.heading = (fly.heading + turn_dir * 4.0 * dt) % (2.0 * math.pi)

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

            self.total_distance += self.fly.speed * dt
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
                'punishment': punishment
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
            fly.metabolic.step(dt=0.01 * dt, speed=fly.speed)

            # Modulate dopamine learning rate by hunger
            dopamine_gain = fly.metabolic.get_dopamine_gain()

            # Step Mushroom Body circuit with modulated dopamine (unless ablated)
            if not fly.ablate_mb:
                fly.circuit.step(
                    odor_a=sensory['mean_a'],
                    odor_b=sensory['mean_b'],
                    reward=reward * dopamine_gain,
                    punishment=punishment,
                    dt_seconds=0.01,
                    learning=True
                )

            # Compute steering and advance kinematics
            dheading, new_speed, state, compass_h, goal_a = self.compute_steering(
                sensory, is_feeding=food_eaten_this_step, fly=fly
            )
            fly.speed = new_speed
            fly.behavioral_state = state
            fly.compass_heading = compass_h
            fly.goal_angle = goal_a
            fly.update(dheading, dt)

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

            if wall_nx != 0.0 or wall_ny != 0.0:
                n_mag = math.hypot(wall_nx, wall_ny)
                wall_nx /= n_mag
                wall_ny /= n_mag
                # Smooth continuous contact torque steering away from boundary
                h_cross_n = math.cos(fly.heading) * wall_ny - math.sin(fly.heading) * wall_nx
                turn_dir = 1.0 if h_cross_n >= 0.0 else -1.0
                fly.heading = (fly.heading + turn_dir * 4.0 * dt) % (2.0 * math.pi)

        self.total_distance += self.fly.speed * dt

        return {
            'time_step': self.time_step,
            'fly_x': self.fly.pos.x,
            'fly_y': self.fly.pos.y,
            'fly_heading': self.fly.heading,
            'fly_speed': self.fly.speed,
            'state': self.fly.behavioral_state,
            'satiety': self.fly.metabolic.satiety,
            'food_collected': self.food_collected,
            'hazard_encounters': self.hazard_encounters,
            'predator_kills': self.total_predator_kills,
            'escapes': self.total_escapes,
            'connectome': getattr(self.fly, 'last_connectome_telemetry', None)
        }
