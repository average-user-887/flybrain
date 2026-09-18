"""
Visual Optic Flow & Looming Collision/Predator Escape Reflex for Drosophila.
===========================================================================
Implements:
1. Panoramic Retinal Optic Flow (T4/T5 -> Lobula Plate Tangential Cells LPTCs):
   - 72-ommatidia cylindrical flow integration.
   - Progressive forward flow (Sum_HS) and differential yaw flow (Delta_HS).
   - Optomotor feedback stabilizing unintended rotational slip.
2. LC4 / LPLC2 Looming Detector & Emergency Escape Saccade:
   - Evaluates optical expansion theta_dot of approaching visual threats (predators).
   - When angular expansion exceeds biological threshold (tau_loom > 0.08 rad/s),
     triggers an all-or-none ballistic escape takeoff / steering saccade away from
     the threat with temporary refractory period.
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np


class CompoundEyeVision:
    def __init__(
        self,
        num_ommatidia: int = 72,
        arena_radius: float = 200.0,
        looming_threshold: float = 0.08,    # rad/s expansion rate trigger
        escape_duration: float = 0.30,      # Duration of ballistic escape flight (s)
        refractory_period: float = 0.50     # Cooldown before another escape can trigger (s)
    ):
        self.num_ommatidia = num_ommatidia
        self.arena_radius = arena_radius
        self.looming_threshold = looming_threshold
        self.escape_duration = escape_duration
        self.refractory_period = refractory_period

        self.azimuths = np.linspace(-np.pi, np.pi, num_ommatidia, endpoint=False)
        self.right_mask = self.azimuths < 0
        self.left_mask = self.azimuths >= 0

        # Dynamic states
        self.optic_flow = np.zeros(num_ommatidia, dtype=np.float64)
        self.hs_left = 0.0
        self.hs_right = 0.0
        self.delta_hs = 0.0  # Optomotor yaw stabilization torque
        self.sum_hs = 0.0    # Forward progressive flow

        # Looming & Escape states
        self.looming_intensity = 0.0
        self.escape_active = False
        self.escape_timer = 0.0
        self.cooldown_timer = 0.0
        self.escape_heading_target = 0.0
        self.threat_distance = 999.0
        self.threat_bearing = 0.0

    def reset(self):
        self.optic_flow.fill(0.0)
        self.hs_left = 0.0
        self.hs_right = 0.0
        self.delta_hs = 0.0
        self.sum_hs = 0.0
        self.looming_intensity = 0.0
        self.escape_active = False
        self.escape_timer = 0.0
        self.cooldown_timer = 0.0
        self.threat_distance = 999.0

    def step(
        self,
        fly_pos: np.ndarray,
        fly_heading: float,
        fly_speed: float,
        fly_yaw_rate: float,
        predator_positions: List[np.ndarray] = None,
        predator_velocities: List[np.ndarray] = None,
        dt: float = 0.02
    ) -> Dict:
        """
        Update optic flow and evaluate visual looming threats.
        """
        # 1. Update timers
        if self.escape_active:
            self.escape_timer -= dt
            if self.escape_timer <= 0.0:
                self.escape_active = False
                self.cooldown_timer = self.refractory_period
        elif self.cooldown_timer > 0.0:
            self.cooldown_timer -= dt

        # 2. Retinal Optic Flow Calculation
        # Translatory component (forward speed divided by visual scene depth)
        translatory = (fly_speed * np.sin(self.azimuths)) / self.arena_radius
        # Rotatory component opposes head turn (reafference)
        rotatory = -fly_yaw_rate
        self.optic_flow = translatory + rotatory

        # LPTC Horizontal System (HS) responses
        self.hs_right = float(np.mean(self.optic_flow[self.right_mask]))
        self.hs_left = float(-np.mean(self.optic_flow[self.left_mask]))
        self.delta_hs = float(self.hs_right - self.hs_left)
        self.sum_hs = float(self.hs_right + self.hs_left)

        # 3. LC4 Looming Threat Detection
        max_looming = 0.0
        detected_threat_bearing = 0.0
        min_dist = 999.0

        if predator_positions:
            for idx, pred_pos in enumerate(predator_positions):
                rel_pos = pred_pos - fly_pos
                dist = float(np.linalg.norm(rel_pos))
                if dist < min_dist:
                    min_dist = dist

                if dist < 120.0 and dist > 1.0: # Threat visual detection radius
                    # Relative bearing to predator
                    bearing_allocentric = math.atan2(rel_pos[1], rel_pos[0])
                    rel_bearing = math.atan2(
                        math.sin(bearing_allocentric - fly_heading),
                        math.cos(bearing_allocentric - fly_heading)
                    )

                    # Approach velocity
                    pred_v = predator_velocities[idx] if predator_velocities else np.zeros(2)
                    v_rel = (pred_v[0] - fly_speed * math.cos(fly_heading),
                             pred_v[1] - fly_speed * math.sin(fly_heading))
                    v_approach = -(rel_pos[0] * v_rel[0] + rel_pos[1] * v_rel[1]) / dist

                    # Looming angular expansion rate: theta_dot ~ (r_pred * v_app) / d^2
                    pred_radius = 12.0 # mm
                    if v_approach > 0.0:
                        expansion_rate = (pred_radius * v_approach) / (dist * dist)
                        if expansion_rate > max_looming:
                            max_looming = expansion_rate
                            detected_threat_bearing = rel_bearing

        self.looming_intensity = max_looming
        self.threat_distance = min_dist
        self.threat_bearing = detected_threat_bearing

        # 4. Trigger LC4 ballistic escape if threshold exceeded and not in cooldown
        if not self.escape_active and self.cooldown_timer <= 0.0:
            if self.looming_intensity >= self.looming_threshold or min_dist < 18.0:
                self.escape_active = True
                self.escape_timer = self.escape_duration
                # Ballistic evade target: point directly opposite to threat bearing
                escape_offset = math.pi + np.random.uniform(-0.3, 0.3)
                self.escape_heading_target = (fly_heading + detected_threat_bearing + escape_offset) % (2.0 * math.pi)

        return {
            "delta_hs": self.delta_hs,
            "sum_hs": self.sum_hs,
            "looming_intensity": self.looming_intensity,
            "threat_distance": self.threat_distance,
            "escape_active": self.escape_active,
            "escape_heading_target": self.escape_heading_target
        }

    def get_optomotor_yaw_bias(self) -> float:
        """Optomotor response counteracts unintended rotational slip."""
        gain = 0.8
        return float(np.clip(self.delta_hs * gain, -0.3, 0.3))
