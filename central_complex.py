"""Central Complex (CX) Heading Compass & Vector Working Memory.

Implements the canonical Drosophila Central Complex architecture:
- E-PG (Ellipsoid Body) compass ring attractor tracking current head direction theta.
- P-EN (Protocerebral Bridge) angular velocity integration.
- Fan-Shaped Body (FB) vector working memory accumulating goal vectors (food locations).
- PFL3 asymmetric steering torque calculation comparing current heading to goal vector,
  gated by Mushroom Body Output Neuron (MBON) learned valence.
Calibrated to Hulse et al. (eLife 2021) and Lyu et al. (Nature 2022).
"""

import numpy as np


class CentralComplexEngine:
    def __init__(self, n_wedges: int = 16, seed: int = 42):
        self.N = n_wedges
        self.angles = np.linspace(-np.pi, np.pi, self.N, endpoint=False)
        self.rng = np.random.default_rng(seed)
        
        # E-PG ring attractor compass activity bump
        self.epg = np.exp(3.0 * np.cos(self.angles))
        self.epg /= np.sum(self.epg)
        
        # Fan-Shaped Body (FB) working memory vector in allocentric coordinates [x, y]
        self.goal_vector = np.zeros(2, dtype=np.float64)
        self.has_goal = False

    def reset(self):
        self.epg = np.exp(3.0 * np.cos(self.angles))
        self.epg /= np.sum(self.epg)
        self.goal_vector.fill(0.0)
        self.has_goal = False

    def get_decoded_heading(self) -> float:
        """Decode head direction theta from E-PG population vector."""
        sin_sum = float(np.sum(self.epg * np.sin(self.angles)))
        cos_sum = float(np.sum(self.epg * np.cos(self.angles)))
        return float(np.arctan2(sin_sum, cos_sum))

    def step(
        self,
        angular_vel: float,
        v_forward: float,
        mbon_valence: float,
        current_fly_heading: float,
        dt: float = 0.01
    ):
        """
        angular_vel: Angular velocity from steering (rad/s)
        v_forward: Forward ground speed (units/s)
        mbon_valence: Learned valence from MBON (+1 appetitive, -1 aversive)
        current_fly_heading: True physical fly heading (used for visual reafference)
        Returns: pfl3_steering_torque, decoded_compass_heading, goal_angle
        """
        # 1. P-EN Shift Mechanics (Angular Velocity Integration)
        shift_idx = 1 # 1 wedge = 22.5 deg
        pen_L = np.roll(self.epg, shift_idx) * max(0.0, angular_vel)
        pen_R = np.roll(self.epg, -shift_idx) * max(0.0, -angular_vel)
        
        # Attractor update with visual reafference anchoring
        visual_bump = np.exp(3.0 * np.cos(self.angles - current_fly_heading))
        visual_bump /= np.sum(visual_bump)
        
        d_epg = -self.epg + 1.2 * (pen_L + pen_R) + 0.3 * visual_bump
        self.epg = np.maximum(0.0, self.epg + d_epg * (dt / 0.02))
        self.epg /= (np.sum(self.epg) + 1e-6)
        
        theta_heading = self.get_decoded_heading()
        
        # 2. Allocentric displacement vector
        v_allocentric = np.array([
            v_forward * np.cos(theta_heading),
            v_forward * np.sin(theta_heading)
        ])
        
        # 3. Vector Working Memory (FB Layer 4/5)
        # If rewarding food is sensed or eaten (high positive valence), reinforce goal vector
        if mbon_valence > 0.3:
            # Anchor goal in direction of recent approach
            self.goal_vector = 0.85 * self.goal_vector + 0.15 * v_allocentric
            self.has_goal = True
        elif self.has_goal:
            # Leaky integration / persistence
            self.goal_vector += -v_allocentric * dt * 0.05
            if np.linalg.norm(self.goal_vector) < 0.05:
                self.has_goal = False
                
        # 4. PFL3 Asymmetric Steering Torque
        if self.has_goal and np.linalg.norm(self.goal_vector) > 1e-4:
            theta_goal = float(np.arctan2(self.goal_vector[1], self.goal_vector[0]))
            # Angle error between goal and current heading
            heading_err = (theta_goal - theta_heading + np.pi) % (2.0 * np.pi) - np.pi
            # PFL3 left/right asymmetric comparison onto LAL descending neurons
            pfl3_torque = float(np.sin(heading_err))
        else:
            theta_goal = theta_heading
            pfl3_torque = 0.0
            
        # Gating by Mushroom Body learned valence: appetitive valence amplifies goal pursuit
        valence_gain = max(0.0, 1.0 + mbon_valence)
        steering_torque = float(valence_gain * pfl3_torque)
        
        return steering_torque, theta_heading, theta_goal
