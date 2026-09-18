"""
FlyBrain Gymnasium Adapter
==========================
Bridges the biologically grounded Drosophila simulation engines:
- Mushroom Body Olfactory Plasticity (circuit.py)
- Surge-and-Cast Anemotaxis / Klinotaxis Engine (surge_cast.py)
- Central Complex Heading Compass & Vector Working Memory (central_complex.py)
- 2D Continuous Kinematics Arena (arena.py)

Compatible with standard RL harnesses (Gymnasium, Stable-Baselines3, CleanRL)
as well as the Doom learning v6 / Brainlab pipelines.
"""

import math
import numpy as np

# Optional Gymnasium dependency with lightweight fallback
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False

from circuit import MushroomBodyCircuit
from surge_cast import SurgeCastEngine
from central_complex import CentralComplexEngine
from metabolic import MetabolicState


class DummyBoxSpace:
    """Minimal Box space fallback when gymnasium/gym is not installed."""
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        if shape is None:
            self.shape = self.low.shape
        else:
            self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(
            np.where(np.isneginf(self.low), -1.0, self.low),
            np.where(np.isposinf(self.high), 1.0, self.high),
            size=self.shape
        ).astype(self.dtype)

    def contains(self, x):
        return isinstance(x, np.ndarray) and x.shape == self.shape


class FlyBrainEnv:
    """
    Continuous 2D Drosophila Navigation Environment with Neuromorphic Circuitry.
    
    Observation Space (12-dim vector):
    [0]: Forward ground speed (mm/s)
    [1]: Angular velocity (rad/s)
    [2]: Heading angle (rad, [-pi, pi])
    [3]: Left antenna odor concentration
    [4]: Right antenna odor concentration
    [5]: Filtered temporal odor signal (ON state)
    [6]: Relative wind heading (rad)
    [7]: MBON net valence (approach - avoidance)
    [8]: Central Complex decoded compass heading (rad)
    [9]: Distance to food source (mm, normalized / 1000)
    [10]: Vector memory goal X component
    [11]: Vector memory goal Y component
    
    Action Space (2-dim continuous vector):
    [0]: Forward speed command [0.0, 1.0] (mapped to 0 - 25 mm/s)
    [1]: Steering torque command [-1.0, 1.0] (mapped to -4 to +4 rad/s)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        dt=0.01,
        arena_size=(600.0, 400.0),
        food_pos=(450.0, 200.0),
        wind_speed=120.0,
        wind_direction=math.pi,
        residual_weight=0.3,
        bio_mode=True,
        max_steps=1000,
        seed=42
    ):
        self.dt = dt
        self.arena_size = arena_size
        self.food_pos = np.array(food_pos, dtype=np.float64)
        self.food_radius = 20.0
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.residual_weight = residual_weight
        self.bio_mode = bio_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.seed_val = seed

        # Sub-circuit engines
        self.circuit = MushroomBodyCircuit(seed=seed)
        self.plume_engine = SurgeCastEngine(dt=dt, seed=seed)
        self.cx_engine = CentralComplexEngine(n_wedges=16, seed=seed)
        self.metabolic = MetabolicState(initial_satiety=0.8)

        # Observation & Action spaces
        if HAS_GYMNASIUM:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )
        else:
            self.observation_space = DummyBoxSpace(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            )
            self.action_space = DummyBoxSpace(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            )

        # Kinematic state
        self.antenna_span = 0.35
        self.pos = np.array([100.0, 200.0], dtype=np.float64)
        self.heading = 0.0
        self.speed = 0.0
        self.omega = 0.0
        self.last_valence = 0.0
        self.last_app = 0.0
        self.last_avo = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)

        self.current_step = 0
        self.pos = np.array([
            np.random.uniform(80.0, 140.0),
            np.random.uniform(150.0, 250.0)
        ], dtype=np.float64)
        self.heading = float(np.random.uniform(-math.pi, math.pi))
        self.speed = 0.0
        self.omega = 0.0

        self.plume_engine.reset()
        self.cx_engine.reset()
        self.circuit.reset_transients()
        self.metabolic.reset(0.8)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_odor_concentration(self, point: np.ndarray) -> float:
        rel = point - self.food_pos
        cos_w = math.cos(self.wind_direction)
        sin_w = math.sin(self.wind_direction)
        
        downwind = -(rel[0] * cos_w + rel[1] * sin_w)
        crosswind = -rel[0] * sin_w + rel[1] * cos_w

        dist = float(np.linalg.norm(rel))
        if dist < self.food_radius:
            return 1.0

        if downwind < -10.0:
            return float(math.exp(-dist * dist / (2.0 * 25.0 * 25.0)))

        sigma_y = max(15.0, 15.0 + 0.15 * downwind)
        decay = 1.0 / (1.0 + 0.005 * downwind)
        conc = decay * math.exp(-(crosswind * crosswind) / (2.0 * sigma_y * sigma_y))
        return float(np.clip(conc, 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        half_span = self.antenna_span / 2.0
        left_ant = self.pos + np.array([
            -half_span * math.sin(self.heading),
            half_span * math.cos(self.heading)
        ])
        right_ant = self.pos + np.array([
            half_span * math.sin(self.heading),
            -half_span * math.cos(self.heading)
        ])

        c_L = self._get_odor_concentration(left_ant)
        c_R = self._get_odor_concentration(right_ant)

        # Odor processing through Mushroom Body circuit
        c_mean = 0.5 * (c_L + c_R)
        _, kc_hz = self.circuit.encode_odor(c_mean, 0.0)
        app, avo, val = self.circuit.forward(kc_hz)
        self.last_app = app
        self.last_avo = avo
        self.last_valence = val

        rel_wind = math.atan2(
            math.sin(self.wind_direction - self.heading),
            math.cos(self.wind_direction - self.heading)
        )

        cx_heading = self.cx_engine.get_decoded_heading()
        goal_x, goal_y = self.cx_engine.goal_vector
        dist_to_food = float(np.linalg.norm(self.pos - self.food_pos))

        obs = np.array([
            self.speed,
            self.omega,
            self.heading,
            c_L,
            c_R,
            self.plume_engine.ON_state,
            rel_wind,
            self.last_valence,
            cx_heading,
            dist_to_food / 1000.0,
            goal_x,
            goal_y
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        dist_to_food = float(np.linalg.norm(self.pos - self.food_pos))
        return {
            "step": self.current_step,
            "dist_to_food": dist_to_food,
            "pos": self.pos.copy(),
            "heading_deg": math.degrees(self.heading) % 360.0,
            "speed": self.speed,
            "state": self.plume_engine.behavioral_state,
            "satiety": self.metabolic.satiety,
            "mbon_approach": self.last_app,
            "mbon_avoidance": self.last_avo,
            "cx_heading_deg": math.degrees(self.cx_engine.get_decoded_heading()) % 360.0,
            "in_food_zone": bool(dist_to_food <= self.food_radius)
        }

    def step(self, action=None):
        self.current_step += 1
        obs = self._get_obs()
        c_L, c_R = obs[3], obs[4]
        rel_wind = obs[6]
        net_valence = obs[7]
        dist_prev = float(np.linalg.norm(self.pos - self.food_pos))
        in_food = dist_prev <= self.food_radius

        # 1. Biological plume tracking step
        v_bio, omega_bio, _ = self.plume_engine.step(c_L, c_R, rel_wind, is_feeding=in_food)
        torque_cx, _, _ = self.cx_engine.step(
            angular_vel=omega_bio,
            v_forward=v_bio,
            mbon_valence=net_valence,
            current_fly_heading=self.heading,
            dt=self.dt
        )

        # 2. Blend Biological Prior with RL Residual Action
        if action is not None and self.bio_mode:
            v_cmd = np.clip(action[0], 0.0, 1.0) * 25.0
            steer_cmd = np.clip(action[1], -1.0, 1.0) * 4.0
            
            alpha = self.residual_weight
            v_final = (1.0 - alpha) * v_bio + alpha * v_cmd
            omega_final = (1.0 - alpha) * (omega_bio + 0.5 * torque_cx) + alpha * steer_cmd
        elif action is not None and not self.bio_mode:
            v_final = np.clip(action[0], 0.0, 1.0) * 25.0
            omega_final = np.clip(action[1], -1.0, 1.0) * 4.0
        else:
            v_final = v_bio
            omega_final = omega_bio + 0.5 * torque_cx

        self.speed = float(v_final)
        self.omega = float(omega_final)

        # Kinematic update
        self.heading = (self.heading + self.omega * self.dt) % (2.0 * math.pi)
        self.pos[0] += self.speed * math.cos(self.heading) * self.dt
        self.pos[1] += self.speed * math.sin(self.heading) * self.dt

        self.pos[0] = np.clip(self.pos[0], 10.0, self.arena_size[0] - 10.0)
        self.pos[1] = np.clip(self.pos[1], 10.0, self.arena_size[1] - 10.0)

        # Advance metabolic state
        self.metabolic.step(dt=self.dt, speed=self.speed)

        dist_now = float(np.linalg.norm(self.pos - self.food_pos))
        if dist_now <= self.food_radius:
            self.metabolic.feed(0.35)
            self.circuit.step(1.0, 0.0, reward=1.0 * self.metabolic.get_dopamine_gain(), punishment=0.0, dt_seconds=self.dt, learning=True)
            self.plume_engine.behavioral_state = 'FEED'

        progress = (dist_prev - dist_now) * 0.05
        energy_penalty = -0.005 * (self.speed / 25.0)
        step_cost = -0.005

        reward = progress + energy_penalty + step_cost
        reached_goal = dist_now <= self.food_radius
        if reached_goal:
            reward += 10.0

        terminated = bool(reached_goal)
        truncated = bool(self.current_step >= self.max_steps)

        next_obs = self._get_obs()
        info = self._get_info()
        return next_obs, reward, terminated, truncated, info


FlyBrainEnvAdapter = FlyBrainEnv
