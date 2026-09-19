"""Sensory Plume Tracking Engine for Drosophila.

Implements adaptive compression (Hill kinetics), ON/OFF linear filters,
bilateral tropotaxis, temporal klinotaxis, and Markovian walk/stop transitions.
Calibrated to Alvarez-Salvado et al. (eLife 2018) and Demir et al. (eLife 2020).
"""

import numpy as np


class SurgeCastEngine:
    def __init__(self, dt: float = 0.01, seed: int = 42, ablate_off: bool = False):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.ablate_off = ablate_off
        
        # Model Parameters (Alvarez-Salvado et al. 2018)
        self.tau_A = 9.80       # Adaptation time constant (s)
        self.tau_ON = 0.72      # ON filter time constant (s)
        self.scale_ON = 7.30
        self.tau_OFF1 = 0.62    # OFF differentiating filter fast timescale (s)
        self.tau_OFF2 = 4.84    # OFF differentiating filter slow timescale (s)
        self.scale_OFF = 0.60
        
        # Kinematics Coefficients
        self.v0 = 1.2           # Baseline cruise speed
        self.kappa1 = 0.35      # ON speed boost (surge acceleration)
        self.kappa2 = 0.40      # OFF speed reduction (casting deceleration)
        self.P0 = 0.15          # Baseline turn rate (1/s)
        self.kappa3 = 0.05      # ON turn suppression
        self.kappa4 = 0.60      # OFF turn boost
        self.kappa5 = 3.0       # Upwind drive strength
        self.kappa6 = 0.4       # Downwind drive baseline
        self.kappa7 = 1.2       # Bilateral tropotaxis gain
        self.sigma_theta = 20.0 # Saccade std dev (deg/s)
        
        # Internal Filter States
        self.K_A = 1.0
        self.ON_state = 0.0
        self.u1 = 0.0
        self.u2 = 0.0
        self.OFF_state = 0.0
        
        # Locomotion state: 0 = STOP, 1 = WALK (Demir et al. 2020)
        self.locomotion_state = 1
        self.time_since_last_encounter = 10.0
        self.accumulated_evidence = 0.0
        
        # High-level behavioral tag: 'WANDER', 'SURGE', 'CAST', 'FEED'
        self.behavioral_state = 'WANDER'
        # Cast alternating direction flag
        self.cast_direction = 1.0
        self.cast_step_counter = 0

    def reset(self):
        self.K_A = 1.0
        self.ON_state = 0.0
        self.u1 = 0.0
        self.u2 = 0.0
        self.OFF_state = 0.0
        self.locomotion_state = 1
        self.time_since_last_encounter = 10.0
        self.accumulated_evidence = 0.0
        self.behavioral_state = 'WANDER'
        self.cast_direction = 1.0
        self.cast_step_counter = 0

    def step(self, c_left: float, c_right: float, wind_angle_rad: float = 0.0, is_feeding: bool = False, dt: float = None):
        """
        c_left, c_right: Odor concentration at antennae
        wind_angle_rad: Wind heading relative to fly (0 = into wind/upwind, pi = downwind)
        is_feeding: True when fly is in contact with food
        Returns: forward speed, angular velocity, and behavioral tag
        """
        if dt is not None:
            self.dt = float(dt)
        if is_feeding:
            self.behavioral_state = 'FEED'
            return 0.2, 0.0, 'FEED'

        c_mean = 0.5 * (c_left + c_right)
        self.time_since_last_encounter += self.dt
        
        # Detect encounter onset (> threshold)
        if c_mean > 0.04:
            self.time_since_last_encounter = 0.0
            self.accumulated_evidence += 1.0
            
        # 1. Adaptive Compression (Alvarez-Salvado 2018)
        self.K_A += ((c_mean - self.K_A) / self.tau_A) * self.dt
        x = c_mean / (c_mean + max(1e-4, self.K_A))
        
        # 2. Linear ON Filter
        self.ON_state += ((self.scale_ON * x - self.ON_state) / self.tau_ON) * self.dt
        
        # 3. Differentiating OFF Filter
        self.u1 += ((x - self.u1) / self.tau_OFF1) * self.dt
        self.u2 += ((x - self.u2) / self.tau_OFF2) * self.dt
        if self.ablate_off:
            self.OFF_state = 0.0
        else:
            self.OFF_state = self.scale_OFF * max(0.0, self.u2 - self.u1)
        
        # 4. Determine behavioral state (SURGE vs CAST vs WANDER)
        if c_mean <= 0.04:
            if not self.ablate_off and (self.OFF_state > 0.05 or self.time_since_last_encounter < 3.0):
                self.behavioral_state = 'CAST'
            else:
                self.behavioral_state = 'WANDER'
        else:
            if self.ON_state > 0.15 or c_mean > 0.05:
                self.behavioral_state = 'SURGE'
            else:
                self.behavioral_state = 'WANDER'

        # 5. Markovian Walk/Stop Transitions (Demir et al. 2020)
        # Active surge suppresses stopping to ensure persistent plume entry
        if self.behavioral_state == 'SURGE':
            self.locomotion_state = 1
        else:
            self.accumulated_evidence *= np.exp(-self.dt / 1.8)
            if self.locomotion_state == 1: # WALKING
                r_S = 0.78 - (0.78 - 0.17) * np.exp(-self.time_since_last_encounter / 0.25)
                if self.rng.random() < (1.0 - np.exp(-r_S * self.dt)):
                    self.locomotion_state = 0
            else: # STOPPED
                r_W = 0.2 + 0.8 * (self.accumulated_evidence / (1.0 + self.accumulated_evidence))
                if self.rng.random() < (1.0 - np.exp(-r_W * self.dt)):
                    self.locomotion_state = 1

        # 6. Forward Speed
        if self.locomotion_state == 1:
            if self.behavioral_state == 'SURGE':
                v = self.v0 + self.kappa1 * min(3.0, self.ON_state)
            elif self.behavioral_state == 'CAST':
                v = max(0.5, self.v0 - self.kappa2 * min(2.0, self.OFF_state))
            else:
                v = self.v0
        else:
            v = 0.0
            
        # 7. Steering & Turning (Anemotaxis + Tropotaxis + Zigzag Casting)
        delta_c = (c_left - c_right) / (c_left + c_right + 1e-4)
        
        if self.behavioral_state == 'SURGE':
            d_upwind = -np.sin(wind_angle_rad)
            omega = 0.25 * d_upwind + 0.35 * delta_c + self.rng.normal(0, 0.04)
        elif self.behavioral_state == 'CAST':
            self.cast_step_counter += 1
            if self.cast_step_counter >= 12:
                self.cast_step_counter = 0
                self.cast_direction *= -1.0
            cast_turn = self.cast_direction * 0.28
            omega = cast_turn + 0.20 * delta_c + self.rng.normal(0, 0.08)
        else:
            omega = self.rng.normal(0, 0.15)

        return float(v), float(np.clip(omega, -0.45, 0.45)), ('REST' if self.locomotion_state == 0 else self.behavioral_state)
