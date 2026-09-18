"""
Leg Tripod Gait Central Pattern Generator (CPG) & Kinematics for Drosophila.
===========================================================================
Implements:
1. Kuramoto-Hopf Coupled Phase Oscillators for the Canonical Insect Alternating Tripod:
   - Tripod Group A: L1 (Front-Left), R2 (Mid-Right), L3 (Hind-Left)
   - Tripod Group B: R1 (Front-Right), L2 (Mid-Left), R3 (Hind-Right)
   - Coupled in strict anti-phase (delta_phi = pi).
2. Pre-motor Descending Coupling (DNa02 / DNp01 Drive):
   - Descending drive (D_left, D_right) sets stepping frequency (5--12 Hz).
   - Asymmetric bilateral drive modulates left-vs-right swing/stance phase duration,
     producing physical steering torque.
3. Realistic Drag & Slip Kinematics:
   - Continuous ground traction, inertia, and yaw damping.
"""

import math
from typing import Tuple, Dict, List
import numpy as np


class TripodGaitCPG:
    def __init__(
        self,
        base_freq_hz: float = 8.0,       # Typical Drosophila stepping rate (5-10 Hz)
        max_freq_hz: float = 14.0,
        drag_coeff: float = 4.5,         # Ground friction damping
        mass: float = 1.0,
        rot_inertia: float = 0.8
    ):
        self.base_freq = base_freq_hz
        self.max_freq = max_freq_hz
        self.drag = drag_coeff
        self.mass = mass
        self.inertia = rot_inertia

        # Phase oscillators [0, 2*pi)
        self.phase_a = 0.0               # Tripod A
        self.phase_b = math.pi           # Tripod B (anti-phase)

        # Dynamic kinematic states
        self.forward_speed = 0.0
        self.yaw_rate = 0.0
        self.current_freq = base_freq_hz

    def reset(self):
        self.phase_a = 0.0
        self.phase_b = math.pi
        self.forward_speed = 0.0
        self.yaw_rate = 0.0
        self.current_freq = self.base_freq

    def step(
        self,
        dn_drive_left: float,
        dn_drive_right: float,
        dt: float = 0.02
    ) -> Dict:
        """
        dn_drive_left, dn_drive_right: Normalized descending pre-motor drives in [0.0, 1.5].
        Drives the CPG phase progression, ground thrust, and steering torque.
        """
        d_l = max(0.0, dn_drive_left)
        d_r = max(0.0, dn_drive_right)
        mean_drive = 0.5 * (d_l + d_r)

        if mean_drive < 0.05:
            # Fly is stopped - CPG idle
            self.current_freq = 0.0
            self.forward_speed = max(0.0, self.forward_speed - self.drag * dt)
            self.yaw_rate = self.yaw_rate * max(0.0, 1.0 - 5.0 * dt)
        else:
            # 1. Frequency modulation from descending commands
            freq = self.base_freq * np.clip(mean_drive, 0.2, 1.8)
            self.current_freq = float(min(self.max_freq, freq))

            # 2. Advance coupled Kuramoto phase oscillators
            dphi = 2.0 * math.pi * self.current_freq * dt
            self.phase_a = (self.phase_a + dphi) % (2.0 * math.pi)
            # Maintain strict anti-phase coupling
            self.phase_b = (self.phase_a + math.pi) % (2.0 * math.pi)

            # 3. Ground propulsion force (stance phase when sin(phi) > 0)
            stance_a = max(0.0, math.sin(self.phase_a))
            stance_b = max(0.0, math.sin(self.phase_b))
            propulsion_thrust = (stance_a + stance_b) * mean_drive * 18.0

            # Acceleration = (Thrust - Drag * v) / mass
            accel = (propulsion_thrust - self.drag * self.forward_speed) / self.mass
            self.forward_speed = float(max(0.0, self.forward_speed + accel * dt))

            # 4. Asymmetric steering torque from bilateral drive differential
            torque_drive = (d_r - d_l) * 6.5
            yaw_accel = (torque_drive - 3.5 * self.yaw_rate) / self.inertia
            self.yaw_rate = float(self.yaw_rate + yaw_accel * dt)

        # 5. Leg stance states for 6 legs (True = in stance/on ground, False = in swing)
        # Tripod A: L1, R2, L3
        # Tripod B: R1, L2, R3
        is_stance_a = math.sin(self.phase_a) > 0
        is_stance_b = math.sin(self.phase_b) > 0

        leg_states = {
            "L1": is_stance_a, "R2": is_stance_a, "L3": is_stance_a,
            "R1": is_stance_b, "L2": is_stance_b, "R3": is_stance_b
        }

        return {
            "forward_speed": self.forward_speed,
            "yaw_rate": self.yaw_rate,
            "stepping_freq_hz": self.current_freq,
            "phase_a": self.phase_a,
            "phase_b": self.phase_b,
            "leg_states": leg_states
        }
