"""
Mechanosensory Johnston's Organ (JO) for Drosophila.
===================================================
Models the mechanical deflection of the antennal flagellum and arista
by aerodynamic relative airflow (v_rel = v_wind - v_fly):
1. Antennal deflection angle and amplitude based on drag kinematics.
2. Differential strain on left vs right Johnston's Organ chordotonal sensilla.
3. Central Complex (CX) AMMC projection: converts mechanosensory wind cues
   into a 16-wedge ring attractor input vector for anemotactic heading alignment.
"""

import math
from typing import Tuple, Dict
import numpy as np


class JohnstonsOrgan:
    def __init__(
        self,
        n_wedges: int = 16,
        antennal_rest_angle: float = math.pi / 6.0,  # 30 deg offset from midline
        deflection_sensitivity: float = 0.05,       # rad per (mm/s) airflow
        max_deflection_rad: float = 0.45            # Mechanical cuticle limit (~25 deg)
    ):
        self.n_wedges = n_wedges
        self.ant_rest = antennal_rest_angle
        self.sensitivity = deflection_sensitivity
        self.max_deflect = max_deflection_rad
        self.wedge_angles = np.linspace(-np.pi, np.pi, n_wedges, endpoint=False)

        # Internal states
        self.v_rel_mag = 0.0
        self.rel_wind_angle = 0.0
        self.deflect_left = 0.0
        self.deflect_right = 0.0
        self.cx_mechanosensory_bump = np.zeros(n_wedges, dtype=np.float64)

    def reset(self):
        self.v_rel_mag = 0.0
        self.rel_wind_angle = 0.0
        self.deflect_left = 0.0
        self.deflect_right = 0.0
        self.cx_mechanosensory_bump.fill(0.0)

    def step(
        self,
        fly_heading: float,
        fly_speed: float,
        wind_vx: float,
        wind_vy: float
    ) -> Dict:
        """
        Compute antennal deflection and mechanosensory neural activation.
        
        fly_heading: Current physical head angle (rad)
        fly_speed: Current forward locomotion speed (mm/s)
        wind_vx, wind_vy: Global airflow velocity vector (mm/s)
        """
        # Fly velocity in world coordinates
        v_fly_x = fly_speed * math.cos(fly_heading)
        v_fly_y = fly_speed * math.sin(fly_heading)

        # Relative airflow experienced by the head (v_wind - v_fly)
        v_rel_x = wind_vx - v_fly_x
        v_rel_y = wind_vy - v_fly_y
        self.v_rel_mag = math.sqrt(v_rel_x * v_rel_x + v_rel_y * v_rel_y)

        # Allocentric and egocentric wind angle
        allocentric_wind = math.atan2(v_rel_y, v_rel_x)
        # Egocentric angle: 0 = airflow hitting fly head-on (from front)
        self.rel_wind_angle = math.atan2(
            math.sin(allocentric_wind - fly_heading),
            math.cos(allocentric_wind - fly_heading)
        )

        # Mechanical deflection on angled left/right aristae
        # Left antenna angled at +ant_rest; Right antenna at -ant_rest
        force_left = self.v_rel_mag * math.sin(self.rel_wind_angle - self.ant_rest)
        force_right = self.v_rel_mag * math.sin(self.rel_wind_angle + self.ant_rest)

        self.deflect_left = float(np.clip(force_left * self.sensitivity, -self.max_deflect, self.max_deflect))
        self.deflect_right = float(np.clip(force_right * self.sensitivity, -self.max_deflect, self.max_deflect))

        # Project into 16-wedge Central Complex mechanosensory ring bump
        # Peak of bump aligns with the direction wind is blowing toward (or from)
        peak_angle = self.rel_wind_angle
        diff = self.wedge_angles - peak_angle
        wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
        bump = np.exp(2.5 * np.cos(wrapped_diff))
        self.cx_mechanosensory_bump = (bump / np.sum(bump)) * min(1.0, self.v_rel_mag / 20.0)

        return {
            "v_rel_mag": self.v_rel_mag,
            "rel_wind_angle": self.rel_wind_angle,
            "deflect_left_deg": math.degrees(self.deflect_left),
            "deflect_right_deg": math.degrees(self.deflect_right),
            "cx_bump": self.cx_mechanosensory_bump
        }

    def get_upwind_drive(self) -> float:
        """
        Steering error to orient head-on into the oncoming relative wind.
        Negative sine drives heading toward rel_wind_angle = pi (into wind).
        """
        into_wind_target = math.pi
        diff = into_wind_target - abs(self.rel_wind_angle)
        sign = 1.0 if self.rel_wind_angle > 0 else -1.0
        return float(sign * diff)
