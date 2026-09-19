"""
Mechanosensory Johnston's Organ (JO) & Wedge Circuit for Drosophila.
===================================================================
Biological Drosophila Mechanosensory & Anemotaxis Transduction:
1. Johnston's Organ (JO) Sensillar Division:
   - JON-A/B: High-frequency (200-500 Hz) cuticular vibration and courtship acoustic song.
   - JON-C/E: Static and low-frequency aerodynamic drag deflection and gravitaxis.
2. AMMC & Wedge (WED) Neuropil Projections:
   - JON-A/B project to AMMC Zones A/B (auditory courtship processing).
   - JON-C/E project to AMMC Zones C/E and the Wedge (WED).
3. Wedge Projection Neurons (WPNs) & Central Complex Coupling:
   - Bilateral push-pull wind angle decoding: WPN_L vs WPN_R.
   - Routes into Central Complex ring neurons (ER1_b, ER3a_b) to align the E-PG compass
     bump to ambient wind vectors during flight and darkness.
   - Routes to Lateral Accessory Lobe (LAL010) for upwind surge and crosswind cast steering.

100% Portable - Zero hardcoded system paths.
"""

import math
from typing import Tuple, Dict, Optional, Any
import numpy as np


class WedgeProjectionNeurons:
    """Decodes bilateral antennal deflection into allocentric and egocentric wind bearings."""

    def __init__(self, n_wedges: int = 16):
        self.n_wedges = n_wedges
        self.wedge_angles = np.linspace(-np.pi, np.pi, n_wedges, endpoint=False)
        self.wpn_left_rate = 0.0
        self.wpn_right_rate = 0.0
        self.anemotactic_compass_bias = np.zeros(n_wedges, dtype=np.float32)

    def reset(self):
        self.wpn_left_rate = 0.0
        self.wpn_right_rate = 0.0
        self.anemotactic_compass_bias.fill(0.0)

    def update(self, deflect_l_rad: float, deflect_r_rad: float, rel_wind_angle: float, wind_speed: float) -> Dict[str, Any]:
        """
        Bilateral wedge projection neuron firing rates and ring attractor compass bias.
        """
        # Push-pull bilateral activation
        # Left WPN responds to leftward aerodynamic drag; Right WPN to rightward drag
        self.wpn_left_rate = float(np.clip(100.0 * max(0.0, deflect_l_rad) * (wind_speed / 15.0), 0.0, 200.0))
        self.wpn_right_rate = float(np.clip(100.0 * max(0.0, deflect_r_rad) * (wind_speed / 15.0), 0.0, 200.0))

        # Direct projection to Central Complex ER1_b / ER3a_b ring neurons
        diff = self.wedge_angles - rel_wind_angle
        wrapped = np.arctan2(np.sin(diff), np.cos(diff))
        bump = np.exp(3.0 * np.cos(wrapped))
        scale = min(1.0, wind_speed / 25.0)
        self.anemotactic_compass_bias = ((bump / np.sum(bump)) * scale).astype(np.float32)

        return {
            "wpn_left_hz": self.wpn_left_rate,
            "wpn_right_hz": self.wpn_right_rate,
            "wpn_differential": self.wpn_right_rate - self.wpn_left_rate,
            "compass_bias": self.anemotactic_compass_bias
        }


class JohnstonsOrgan:
    """
    Simulates Johnston's Organ chordotonal sensilla, arista mechanics, and Wedge wind compass.
    Preserves 100% backward compatibility with existing arena and benchmark interfaces.
    """

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

        # Internal states: Wind & Drag
        self.v_rel_mag = 0.0
        self.rel_wind_angle = 0.0
        self.deflect_left = 0.0
        self.deflect_right = 0.0
        self.cx_mechanosensory_bump = np.zeros(n_wedges, dtype=np.float64)

        # Specialized JON Subgroup States
        self.jon_ce_wind_drag = 0.0          # Subgroup C/E (static and low-freq drag)
        self.jon_ab_courtship_acoustic = 0.0 # Subgroup A/B (high-freq 200-500 Hz song)

        # Wedge projection neurons
        self.wedge = WedgeProjectionNeurons(n_wedges=n_wedges)

    def reset(self):
        self.v_rel_mag = 0.0
        self.rel_wind_angle = 0.0
        self.deflect_left = 0.0
        self.deflect_right = 0.0
        self.cx_mechanosensory_bump.fill(0.0)
        self.jon_ce_wind_drag = 0.0
        self.jon_ab_courtship_acoustic = 0.0
        self.wedge.reset()

    def step(
        self,
        fly_heading: float,
        fly_speed: float,
        wind_vx: float,
        wind_vy: float,
        acoustic_pulse_train: float = 0.0
    ) -> Dict:
        """
        Compute antennal deflection, JON subgroup activation, and mechanosensory neural activation.
        
        fly_heading: Current physical head angle (rad)
        fly_speed: Current forward locomotion speed (mm/s)
        wind_vx, wind_vy: Global airflow velocity vector (mm/s)
        acoustic_pulse_train: Courtship song sound pressure amplitude [0.0, 1.0]
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
            math.sin(allocentric_wind + math.pi - fly_heading),
            math.cos(allocentric_wind + math.pi - fly_heading)
        )

        # Mechanical deflection on angled left/right aristae (outward vs inward)
        # Left antenna angled at +ant_rest; Right antenna at -ant_rest
        force_left = self.v_rel_mag * (math.sin(self.rel_wind_angle) * math.cos(self.ant_rest) - math.cos(self.rel_wind_angle) * math.sin(self.ant_rest))
        force_right = self.v_rel_mag * (-math.sin(self.rel_wind_angle) * math.cos(self.ant_rest) - math.cos(self.rel_wind_angle) * math.sin(self.ant_rest))

        self.deflect_left = float(np.clip(force_left * self.sensitivity, -self.max_deflect, self.max_deflect))
        self.deflect_right = float(np.clip(force_right * self.sensitivity, -self.max_deflect, self.max_deflect))

        # JON Subgroup activations:
        # C/E: low-frequency airflow magnitude
        self.jon_ce_wind_drag = float(min(1.0, self.v_rel_mag / 30.0))
        # A/B: acoustic vibration (courtship pulse song / sine song)
        self.jon_ab_courtship_acoustic = float(np.clip(acoustic_pulse_train, 0.0, 1.0))

        # Wedge projection update
        wpn_data = self.wedge.update(
            deflect_l_rad=self.deflect_left,
            deflect_r_rad=self.deflect_right,
            rel_wind_angle=self.rel_wind_angle,
            wind_speed=self.v_rel_mag
        )

        # Project into 16-wedge Central Complex mechanosensory ring bump
        self.cx_mechanosensory_bump = wpn_data["compass_bias"]

        return {
            "v_rel_mag": self.v_rel_mag,
            "rel_wind_angle": self.rel_wind_angle,
            "deflect_left_deg": math.degrees(self.deflect_left),
            "deflect_right_deg": math.degrees(self.deflect_right),
            "cx_bump": self.cx_mechanosensory_bump,
            "jon_ce_wind_drag": self.jon_ce_wind_drag,
            "jon_ab_acoustic": self.jon_ab_courtship_acoustic,
            "wpn_data": wpn_data
        }

    def get_upwind_drive(self) -> float:
        """
        Returns an egocentric steering bias toward the upwind heading (rel_wind_angle = 0).
        Positive = turn right toward wind; Negative = turn left toward wind.
        """
        if self.v_rel_mag < 0.5:
            return 0.0
        # When relative wind is from right (rel_wind_angle > 0), turn right (+drive)
        # Desirability function D_u(psi) = sin(psi)
        return float(np.clip(math.sin(self.rel_wind_angle) * min(1.0, self.v_rel_mag / 15.0), -1.0, 1.0))
