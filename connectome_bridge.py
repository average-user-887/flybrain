"""
Drosophila Whole-Brain Connectome Sensorimotor Bridge (MaleCNS v1.0 & FlyWire)
=============================================================================
Connects continuous multi-agent arena physics with whole-brain connectomic circuits:
1. Sensory Ingress:
   - Vision: 72-ommatidia neural superposition (R1-R6 -> Lamina L1-L5) + R8 chromatic channels.
   - Olfaction: 54 AL glomeruli with adaptive Hill-equation compression & LN divisive normalization.
   - Wind Mechanoreception: Johnston's Organ (JON-C/E) arista drag deflection -> Wedge (WED) -> WPNs.
   - Proprioception: Femoral Chordotonal Organ (FeCO: claw angle, hook velocity) & Campaniform Sensilla (CS load).
   - Metabolic State: Sucrose ingestion -> PAM reward cluster; danger/collision -> PPL1 punishment cluster.
2. Central Complex (CX) Heading Compass & Steering:
   - E-PG / P-EN Ring Attractor maintaining continuous azimuthal heading bump, shifting at dot(mu) = omega_z.
   - PFL3 (contralateral asymmetric goal error) & PFL2 (bilateral gain & rear stalemate resolution).
   - LAL Push-Pull Arbitration: DNa03 <-> LAL013 winner-take-all & LAL010 interhemispheric cross-inhibition.
3. Descending Locomotion Decoders:
   - DNa02: Transient high-gain yaw steering torque.
   - DNa01: Low-gain course holding integration.
   - DNp09 (P9): Pursuit forward walking drive.
   - BPN: High-speed straight walking cadence.
   - MDN (Moonwalker): Backward walking reversal, inverting metachronal stepping sequence.
   - DNp01 (Giant Fiber): All-or-none looming escape takeoff.
   - Ascending Efference Copy: Shunts HS/VS LPTC retinal slip during voluntary saccades.
4. Compliant Biomechanics:
   - Kuramoto-Hopf coupled oscillators for 6 legs enforcing alternating tripod gait (Delta_Phi = pi).
   - Campaniform Sensilla ground force feedback enforcing Cruse's Walknet Rule 1 (stance maintenance under load).
5. Dual-Mode Co-Simulation:
   - In-process biophysical surrogate (default, < 1 ms / step).
   - Remote RPC client connecting to Ryzen workstation (166,700 neurons, 25.58M synapses).
"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    from circuit import MushroomBodyCircuit
except ImportError:
    try:
        from .circuit import MushroomBodyCircuit
    except ImportError:
        MushroomBodyCircuit = None

try:
    from vision import LobulaFeatureExtractor
    from mechanosensory import WedgeProjectionNeurons
    from locomotion import BioKuramotoHopfCPG
except ImportError:
    try:
        from .vision import LobulaFeatureExtractor
        from .mechanosensory import WedgeProjectionNeurons
        from .locomotion import BioKuramotoHopfCPG
    except ImportError:
        LobulaFeatureExtractor = None
        WedgeProjectionNeurons = None
        BioKuramotoHopfCPG = None


class ConnectomeBridge:
    def __init__(
        self,
        mode: str = "surrogate",          # "surrogate" or "rpc"
        rpc_host: str = "192.168.194.227",
        rpc_port: int = 8768,
        num_ommatidia: int = 72,
        arena_radius: float = 200.0,
        base_stepping_freq: float = 8.0,
        max_stepping_freq: float = 14.0,
        seed: Optional[int] = 42,
    ):
        self.mode = mode
        self.rpc_host = rpc_host
        self.rpc_port = rpc_port
        self.num_ommatidia = num_ommatidia
        self.arena_radius = arena_radius
        self.base_freq = base_stepping_freq
        self.max_freq = max_stepping_freq
        self.seed = seed

        # -------------------------------------------------------------
        # 1. Sensory Ingress State
        # -------------------------------------------------------------
        # Visual: Retinal ommatidia azimuths
        self.ommatidia_azimuths = np.linspace(-math.pi, math.pi, num_ommatidia, endpoint=False)
        self.left_eye_mask = self.ommatidia_azimuths < 0
        self.right_eye_mask = self.ommatidia_azimuths >= 0

        # Olfactory: 54 Glomeruli activation states and adaptation
        self.num_glomeruli = 54
        self.glom_adaptation = np.zeros(self.num_glomeruli, dtype=np.float32)
        self.tau_olfactory_adapt = 9.8    # seconds (Alvarez-Salvado et al., 2018)
        self.glom_rates = np.zeros(self.num_glomeruli, dtype=np.float32)

        # Key Glomerular Indices
        self.GLOM_DM1 = 15   # Ripe fruit / apple cider vinegar (Or42b) - Attraction
        self.GLOM_DM2 = 16   # Fruit esters / banana (Or22a) - Food search
        self.GLOM_DA2 = 1    # Geosmin / toxic mold (Or56a) - Strong innate aversion
        self.GLOM_DA1 = 0    # cVA sex pheromone (Or67d) - Courtship
        self.GLOM_DP1m = 21  # Extreme acid (Ir64a) - Avoidance
        self.GLOM_V = 24     # CO2 stress odorant (Gr21a/Gr63a)

        # Mechanosensory: Johnston's Organ & Wedge
        self.antenna_deflection_l = 0.0
        self.antenna_deflection_r = 0.0
        self.wpn_wind_heading = 0.0      # Egocentric wind azimuth [-pi, pi]
        self.wpn_wind_speed = 0.0        # mm/s

        # Proprioceptive Mechanosensors: 6 legs
        self.leg_names = ["L1", "L2", "L3", "R1", "R2", "R3"]
        self.feco_claw_angles = {leg: 0.0 for leg in self.leg_names}      # Tibia position
        self.feco_hook_velocities = {leg: 0.0 for leg in self.leg_names}  # Joint velocity
        self.cs_ground_forces = {leg: 0.0 for leg in self.leg_names}      # Normal force (uN)

        # Metabolic & Neuromodulation
        self.pam_dopamine = 0.0          # Reward / sucrose ingestion
        self.ppl1_dopamine = 0.0         # Punishment / hazard / collision
        self.energy_reserve = 1.0        # [0.0, 1.0]

        # -------------------------------------------------------------
        # 1b. Mushroom Body (MB) Olfactory Learning Circuit
        # -------------------------------------------------------------
        # Biologically: ~2000 KCs receive sparse random PN input; KC→MBON
        # synaptic weights are depressed by coincident DAN (PAM/PPL1)
        # dopamine, implementing the Huang, Luo et al. 2024 anti-Hebbian
        # plasticity rule. MBON valence readout modulates approach/avoidance.
        self.mb_circuit = None
        if MushroomBodyCircuit is not None:
            self.mb_circuit = MushroomBodyCircuit(
                n_pn=40,           # 40 projection neurons (reduced from ~150 biological)
                n_kc=120,          # 120 Kenyon cells (reduced from ~2000 biological)
                pn_per_kc=5,       # Sparse random connectivity (~12.5% claw convergence)
                kc_threshold=0.20, # Competitive inhibition via APL GABAergic feedback
                eta=0.05,          # Learning rate
                seed=self.seed
            )
        self.mb_valence = 0.0            # Net approach-avoidance valence from MBONs
        self.mb_approach_bias = 0.0      # MBON-α3 / MBON-γ2α'1 appetitive readout
        self.mb_avoidance_bias = 0.0     # MBON-γ1pedc / MBON-α'2 aversive readout
        self.mb_learning_enabled = True  # Can be toggled for experimental controls

        # Odor identity mapping for MB: Arena odor A → food, odor B → alarm pheromone
        # In the arena, odor_left/odor_right represent food gradient (Odor A)
        # Alarm pheromone (Odor B) is emitted at predator strike sites
        self.current_odor_a = 0.0        # Food / vinegar odor concentration
        self.current_odor_b = 0.0        # Alarm / hazard odor concentration

        # Habituation dynamics (sensory-specific adaptation)
        # Biologically mediated by SFA in PNs and lateral inhibition in AL
        self.visual_habituation = 0.0    # Habituates to repeated non-threatening visual stim
        self.odor_habituation = 0.0      # Habituates to prolonged static odor (no gradient)
        self.tau_habituation = 45.0      # Recovery time constant (seconds)

        # Experience counters for scientific data
        self.total_food_collected = 0
        self.total_escapes = 0
        self.total_damage_events = 0
        self.simulation_time = 0.0       # Cumulative simulated time (seconds)

        # -------------------------------------------------------------
        # 2. Central Complex (CX) Heading Compass & Steering
        # -------------------------------------------------------------
        # E-PG Ring Attractor Compass: 16 discrete wedges around the protocerebral bridge
        self.num_compass_wedges = 16
        self.wedge_angles = np.linspace(-math.pi, math.pi, self.num_compass_wedges, endpoint=False)
        self.compass_bump_heading = 0.0   # Current internal heading representation
        self.compass_bump_profile = np.zeros(self.num_compass_wedges, dtype=np.float32)
        self._update_compass_bump(0.0)

        # Central Complex Lesion Study Flags
        self.cx_lesioned = False         # If True, decouples E-PG compass from sensory/motor integration
        self.gf_lesioned = False         # If True, silences DNp01 Giant Fiber escape pathway

        # Lateral Accessory Lobe (LAL) Steering Cascade
        self.pfl3_error_l = 0.0
        self.pfl3_error_r = 0.0
        self.pfl2_gain = 0.0
        self.dna03_activity = 0.0
        self.lal010_cross_inhibition = 0.0

        # -------------------------------------------------------------
        # 3. Descending Command Neurons (DNs) Firing Rates (Hz)
        # -------------------------------------------------------------
        self.dna02_rate_l = 0.0          # Transient steering yaw (left)
        self.dna02_rate_r = 0.0          # Transient steering yaw (right)
        self.dna01_rate_l = 0.0          # Slow course holding (left)
        self.dna01_rate_r = 0.0          # Slow course holding (right)
        self.dna01_integral = 0.0        # Integrated heading bias
        self.dnp09_rate = 0.0            # Pursuit forward speed
        self.bpn_rate = 0.0              # Straight walking cadence
        self.mdn_rate = 0.0              # Moonwalker backward walking command
        self.dnp01_gf_spikes = 0         # Giant Fiber looming escape spikes

        # Ascending Efference Copy (shunting visual slip during saccades)
        self.efference_copy_history: List[Tuple[float, float]] = []  # (timestamp, amplitude)
        self.hs_left_shunted = 0.0
        self.hs_right_shunted = 0.0
        self.vs_shunted = 0.0
        self.shunt_factor = 1.0
        self.efference_copy_active = False

        # -------------------------------------------------------------
        # 3b. Neuroethological Expansion State (Phase 2)
        # -------------------------------------------------------------
        # Thermosensory Channels: Warm cells (dTrpA1/Gr28b) & Cold cells (BReP)
        self.temperature = 25.0
        self.prev_temperature = 25.0
        self.r_warm = 0.0
        self.r_cold = 0.0
        self.thermal_stress = False
        self.pain_relief_burst = False

        # Visual Landmark Ingress: Ring neurons (ER2 / ER4d) mapping to E-PG compass
        self.er2_er4d_activity = np.zeros(self.num_compass_wedges, dtype=np.float32)
        self.visual_landmark_bearing: Optional[float] = None

        # Courtship & Pheromone Sensilla:
        # cVA (Or67d -> DA1), Bitter mated female pheromones (Gr32a), P1 command neurons, MB gamma-lobe dopamine
        self.cva_rate = 0.0
        self.gr32a_rate = 0.0
        self.p1_courtship_rate = 0.0
        self.wing_extension_command = 0.0
        self.mb_gamma_dopamine = 0.0

        # -------------------------------------------------------------
        # 4. Biomechanical Kuramoto-Hopf Tripod CPG
        # -------------------------------------------------------------
        self.phase_a = 0.0               # Tripod A: L1, R2, L3
        self.phase_b = math.pi           # Tripod B: R1, L2, R3
        self.forward_speed = 0.0         # mm/s
        self.yaw_rate = 0.0              # rad/s
        self.stepping_freq = base_stepping_freq

        # Giant Fiber Escape State
        self.escape_active = False
        self.escape_timer = 0.0
        self.escape_cooldown = 0.0

        # Specialized Modular Feature Extractors
        self.lobula = LobulaFeatureExtractor(num_ommatidia=num_ommatidia) if LobulaFeatureExtractor else None
        self.wedge = WedgeProjectionNeurons(n_wedges=self.num_compass_wedges) if WedgeProjectionNeurons else None
        self.bio_cpg = BioKuramotoHopfCPG(base_freq_hz=base_stepping_freq, max_freq_hz=max_stepping_freq) if BioKuramotoHopfCPG else None
        self.lc_features: Dict[str, float] = {}
        self.joint_angles = {
            leg: {"ctr": 0.0, "fti": 60.0, "phase": "STANCE"} for leg in self.leg_names
        }

        # RPC Client handle (if enabled)
        self.rpc_client = None
        if self.mode == "rpc":
            try:
                from .connectome_client import ConnectomeClient
                self.rpc_client = ConnectomeClient(host=self.rpc_host, port=self.rpc_port)
            except Exception:
                self.mode = "surrogate"

    def reset(self, keep_memory: bool = True):
        """Reset internal biophysical states to resting baseline.
        
        Args:
            keep_memory: If True, retains learned KC→MBON synaptic weights
                         across resets (persistent memory). If False, clears all.
        """
        self.glom_adaptation.fill(0.0)
        self.glom_rates.fill(0.0)
        self.compass_bump_heading = 0.0
        self._update_compass_bump(0.0)
        self.dna01_integral = 0.0
        self.phase_a = 0.0
        self.phase_b = math.pi
        self.forward_speed = 0.0
        self.yaw_rate = 0.0
        self.stepping_freq = self.base_freq
        self.escape_active = False
        self.escape_timer = 0.0
        self.escape_cooldown = 0.0
        self.efference_copy_history.clear()
        self.hs_left_shunted = 0.0
        self.hs_right_shunted = 0.0
        self.vs_shunted = 0.0
        self.shunt_factor = 1.0
        self.efference_copy_active = False
        # Thermosensory reset
        self.temperature = 25.0
        self.prev_temperature = 25.0
        self.r_warm = 0.0
        self.r_cold = 0.0
        self.thermal_stress = False
        self.pain_relief_burst = False
        # Visual landmark reset
        self.er2_er4d_activity.fill(0.0)
        self.visual_landmark_bearing = None
        # Courtship & pheromone reset
        self.cva_rate = 0.0
        self.gr32a_rate = 0.0
        self.p1_courtship_rate = 0.0
        self.wing_extension_command = 0.0
        self.mb_gamma_dopamine = 0.0
        # Reset MB learning circuit
        if self.mb_circuit is not None:
            self.mb_circuit.reset_state(keep_memory=keep_memory)
        self.mb_valence = 0.0
        self.mb_approach_bias = 0.0
        self.mb_avoidance_bias = 0.0
        self.current_odor_a = 0.0
        self.current_odor_b = 0.0
        # Reset habituation
        self.visual_habituation = 0.0
        self.odor_habituation = 0.0
        # Reset experience counters
        self.total_food_collected = 0
        self.total_escapes = 0
        self.total_damage_events = 0
        self.simulation_time = 0.0
        if self.lobula:
            self.lobula.reset()
        if self.wedge:
            self.wedge.reset()
        if self.bio_cpg:
            self.bio_cpg.reset()
        self.lc_features.clear()
        for leg in self.leg_names:
            self.joint_angles[leg] = {"ctr": 0.0, "fti": 60.0, "phase": "STANCE"}
        if self.rpc_client:
            self.rpc_client.reset()

    # -------------------------------------------------------------------------
    # INTERNAL BIOPHYSICAL ROUTINES
    # -------------------------------------------------------------------------
    def _update_compass_bump(self, heading_angle: float):
        """Maintains the unimodal Gaussian bump in the E-PG ring attractor."""
        self.compass_bump_heading = (heading_angle + math.pi) % (2.0 * math.pi) - math.pi
        # von Mises / wrapped Gaussian activity bump (kappa = 3.5)
        self.compass_bump_profile = np.exp(3.5 * np.cos(self.wedge_angles - self.compass_bump_heading))
        self.compass_bump_profile /= np.max(self.compass_bump_profile) + 1e-9

    def encode_sensory(
        self,
        fly_pos: np.ndarray,
        fly_heading: float,
        fly_speed: float,
        fly_yaw_rate: float,
        odor_left: float,
        odor_right: float,
        wind_vector: np.ndarray,         # [v_x, v_y] in arena coordinates (mm/s)
        predator_positions: Optional[List[np.ndarray]] = None,
        predator_velocities: Optional[List[np.ndarray]] = None,
        food_ingested: bool = False,
        incurred_damage: bool = False,
        energy_level: float = 1.0,
        dt: float = 0.02,
        temperature: float = 25.0,
        landmarks: Optional[Any] = None,
        cva_odor: float = 0.0,
        bitter_pheromone: float = 0.0,
        female_aphrodisiac: float = 0.0,
        is_saccade: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Translates arena environment states into biologically grounded sensory inputs.
        """
        self.energy_reserve = float(np.clip(energy_level, 0.0, 1.0))

        # -------------------------------------------------------------
        # 1. Vision: Compound Eye, Optic Flow & Looming Detectors
        # -------------------------------------------------------------
        # Retinal flow across 72 ommatidia
        # Translatory component = v * sin(azimuth) / depth; Rotatory = -yaw_rate
        translatory_flow = (fly_speed * np.sin(self.ommatidia_azimuths)) / self.arena_radius
        rotatory_flow = -fly_yaw_rate
        optic_flow = translatory_flow + rotatory_flow

        # Neural superposition: integrate co-axial cartridges (L1/L2)
        hs_right_raw = float(np.mean(optic_flow[self.right_eye_mask]))
        hs_left_raw = float(-np.mean(optic_flow[self.left_eye_mask]))

        # Ascending Efference Copy Shunting (12 ms delayed DNa02 motor collateral)
        now = time.perf_counter()
        current_efference = 0.0
        while self.efference_copy_history and (now - self.efference_copy_history[0][0]) > 0.035:
            self.efference_copy_history.pop(0)

        for t_stamp, eff_amp in self.efference_copy_history:
            dt_eff = now - t_stamp
            if 0.008 <= dt_eff <= 0.020:  # 12 ms window
                current_efference += eff_amp

        # LPTCs (HS/VS) corollary discharge shunt from voluntary saccade commands (DNa02)
        # Suppresses self-generated retinal slip by >= 80% (85% suppression -> factor 0.15)
        saccade_active = False
        if is_saccade is not None:
            saccade_active = bool(is_saccade)
        else:
            if abs(fly_yaw_rate) > 1.75 or abs(self.dna02_rate_r - self.dna02_rate_l) > 15.0 or current_efference > 5.0:
                saccade_active = True

        if saccade_active:
            shunt_factor = 0.15
            self.efference_copy_active = True
        elif current_efference > 0.0:
            shunt_factor = min(0.20, max(0.05, 1.0 - 0.85 * min(1.0, current_efference / 10.0)))
            self.efference_copy_active = True
        else:
            shunt_factor = 1.0
            self.efference_copy_active = False

        self.shunt_factor = shunt_factor
        self.hs_left_shunted = hs_left_raw * shunt_factor
        self.hs_right_shunted = hs_right_raw * shunt_factor
        vs_raw = float(np.abs(fly_speed) * 0.05)
        self.vs_shunted = vs_raw * shunt_factor
        delta_hs = self.hs_right_shunted - self.hs_left_shunted

        # LC4 & LPLC2 Looming Collision Detection
        looming_trigger = False
        min_threat_dist = 999.0
        if predator_positions and len(predator_positions) > 0:
            for i, p_pos in enumerate(predator_positions):
                rel_pos = p_pos - fly_pos
                dist = float(np.linalg.norm(rel_pos))
                min_threat_dist = min(min_threat_dist, dist)

                p_vel = predator_velocities[i] if (predator_velocities and i < len(predator_velocities)) else np.zeros(2)
                v_approach = -float(np.dot(rel_pos / (dist + 1e-6), p_vel))

                if dist < 85.0 and v_approach > 10.0:
                    # Optical expansion: theta = 2 * arctan(r / dist), theta_dot = 2*r*v / dist^2
                    pred_radius = 6.0
                    theta = 2.0 * math.atan2(pred_radius, dist)
                    theta_dot = (2.0 * pred_radius * v_approach) / (dist**2 + pred_radius**2)

                    # LC4 (velocity) + LPLC2 (size) convergence onto Giant Fiber
                    if (theta_dot > 0.075 or (theta > 0.35 and theta_dot > 0.04)) and self.escape_cooldown <= 0.0:
                        looming_trigger = True

        # -------------------------------------------------------------
        # 1c. Visual Landmark Ingress: Ring neurons (ER2/ER4d)
        # -------------------------------------------------------------
        landmark_bearings_list: List[float] = []
        if isinstance(landmarks, dict):
            landmark_bearings_list = [float(v) for v in landmarks.values() if isinstance(v, (int, float))]
        elif isinstance(landmarks, (list, tuple)):
            for lm in landmarks:
                if isinstance(lm, (int, float)):
                    landmark_bearings_list.append(float(lm))
                elif hasattr(lm, 'get_apparent_bearing'):
                    landmark_bearings_list.append(float(lm.get_apparent_bearing(fly_pos[0], fly_pos[1], fly_heading)))
        elif isinstance(landmarks, (int, float)):
            landmark_bearings_list.append(float(landmarks))

        if landmark_bearings_list:
            er_prof = np.zeros(self.num_compass_wedges, dtype=np.float32)
            for b in landmark_bearings_list:
                er_prof += np.exp(2.5 * np.cos(self.wedge_angles - b))
            max_p = np.max(er_prof)
            if max_p > 1e-6:
                er_prof /= max_p
            self.er2_er4d_activity = er_prof
            self.visual_landmark_bearing = min(landmark_bearings_list, key=abs)
        else:
            self.er2_er4d_activity.fill(0.0)
            self.visual_landmark_bearing = None

        # -------------------------------------------------------------
        # 2. Olfaction: 54 AL Glomeruli with Adaptive Compression & LN Divisive Norm
        # -------------------------------------------------------------
        # Adaptive Hill equation kinetics (tau_A = 9.8 s)
        mean_odor = 0.5 * (odor_left + odor_right)
        for g in range(self.num_glomeruli):
            self.glom_adaptation[g] += dt * (mean_odor - self.glom_adaptation[g]) / self.tau_olfactory_adapt

        # Map arena odors into key glomeruli:
        # Food odors excite DM1 (vinegar), DM2 (fruit esters), DM4 (alcohol)
        # Starvation boosts food sensitivity (hunger state modulation)
        hunger_gain = 1.0 + 1.5 * (1.0 - self.energy_reserve)
        self.glom_rates[self.GLOM_DM1] = 120.0 * (mean_odor / (mean_odor + self.glom_adaptation[self.GLOM_DM1] + 0.15)) * hunger_gain
        self.glom_rates[self.GLOM_DM2] = 95.0 * (mean_odor / (mean_odor + self.glom_adaptation[self.GLOM_DM2] + 0.20)) * hunger_gain

        # Divisive normalization across glomeruli: r_PN = R_max * (I^n) / (sigma^n + (gamma * sum_I)^n)
        # Includes biological spontaneous baseline firing of non-stimulated glomeruli (~250 Hz total)
        network_pool = float(np.sum(self.glom_rates)) + 250.0
        divisive_denom = 25.0**1.5 + (0.28 * network_pool)**1.5
        pn_dm1_norm = float(180.0 * (self.glom_rates[self.GLOM_DM1]**1.5) / (divisive_denom + 1e-6))

        # Bilateral odor contrast for chemotactic steering
        odor_diff = odor_left - odor_right

        # -------------------------------------------------------------
        # 2b. Courtship & Pheromone Sensilla: cVA (Or67d) & Bitter (Gr32a)
        # -------------------------------------------------------------
        # cVA sex pheromone (Or67d -> DA1 glomerulus)
        cva_conc = float(np.clip(cva_odor, 0.0, 1.0))
        self.cva_rate = float(120.0 * (cva_conc / (cva_conc + 0.20)))
        self.glom_rates[self.GLOM_DA1] = self.cva_rate

        # Bitter mated female pheromones (Gr32a gustatory / contact chemosensory sensilla)
        bitter_conc = float(np.clip(bitter_pheromone, 0.0, 1.0))
        self.gr32a_rate = float(100.0 * (bitter_conc / (bitter_conc + 0.15)))

        # P1 Courtship Command Neurons in lateral protocerebrum
        aphro_conc = float(np.clip(female_aphrodisiac, 0.0, 1.0))
        base_p1 = float(80.0 * (aphro_conc / (aphro_conc + 0.25))) if aphro_conc > 0.0 else 0.0
        p1_inhibition = 0.65 * (self.cva_rate / 120.0 * 50.0) + 0.75 * self.gr32a_rate
        self.p1_courtship_rate = float(max(0.0, base_p1 - p1_inhibition))
        self.wing_extension_command = float(min(90.0, self.p1_courtship_rate * 1.5))

        # -------------------------------------------------------------
        # 3. Johnston's Organ Wind Drag & Wedge (WED) Heading
        # -------------------------------------------------------------
        # Relative wind vector v_rel = v_wind - v_fly
        fly_vel = fly_speed * np.array([math.cos(fly_heading), math.sin(fly_heading)])
        rel_wind = wind_vector - fly_vel
        wind_speed = float(np.linalg.norm(rel_wind))
        self.wpn_wind_speed = wind_speed

        # Upwind vector (oncoming airflow origin): points in the direction the wind is coming FROM
        upwind_vector = -rel_wind
        wind_angle_global = math.atan2(upwind_vector[1], upwind_vector[0]) if wind_speed > 1e-4 else fly_heading
        # Egocentric upwind azimuth [-pi, pi]: 0 rad means facing directly upwind into the oncoming flow
        egocentric_wind = (wind_angle_global - fly_heading + math.pi) % (2.0 * math.pi) - math.pi
        self.wpn_wind_heading = egocentric_wind

        # Aerodynamic drag torque deflecting antennae
        drag_force = 0.5 * 1.225e-3 * 0.85 * 0.12 * (wind_speed**2)  # micro-Newtons
        self.antenna_deflection_l = drag_force * math.cos(egocentric_wind + 0.4)
        self.antenna_deflection_r = drag_force * math.cos(egocentric_wind - 0.4)

        # -------------------------------------------------------------
        # 4. Thermosensory Channels: Warm (dTrpA1/Gr28b) & Cold (BReP)
        # -------------------------------------------------------------
        r_warm = max(0.0, (temperature - 26.0) * 8.0)
        r_cold = max(0.0, (25.0 - temperature) * 8.0)
        delta_t = temperature - self.prev_temperature

        # Thermal stress (T_floor > 35 C) drives PPL1 nociceptive dopamine
        if temperature > 35.0:
            self.thermal_stress = True
            self.ppl1_dopamine = max(self.ppl1_dopamine, 1.0)
        else:
            self.thermal_stress = False

        # Pain relief reward (Delta T < -3.0 C) triggers transient PAM dopaminergic burst (+1.0)
        if delta_t < -3.0:
            self.pain_relief_burst = True
            self.pam_dopamine = min(1.0, self.pam_dopamine + 1.0)
        else:
            self.pain_relief_burst = False

        self.temperature = float(temperature)
        self.r_warm = float(r_warm)
        self.r_cold = float(r_cold)

        # -------------------------------------------------------------
        # 5. Metabolic & Dopaminergic Reward / Punishment State
        # -------------------------------------------------------------
        if food_ingested:
            self.pam_dopamine = 1.0      # Burst of reward dopamine (PAM cluster)
        else:
            self.pam_dopamine = max(0.0, self.pam_dopamine - 3.5 * dt)

        if incurred_damage or (min_threat_dist < 20.0) or self.thermal_stress:
            self.ppl1_dopamine = 1.0     # Hazard / pain dopamine (PPL1 cluster)
        else:
            self.ppl1_dopamine = max(0.0, self.ppl1_dopamine - 2.5 * dt)

        # MB gamma-lobe dopamine modulation from bitter pheromone (Gr32a) and punishment
        self.mb_gamma_dopamine = float(np.clip(
            0.5 * (self.gr32a_rate / 100.0) + 0.3 * (self.cva_rate / 120.0) + 0.5 * self.ppl1_dopamine,
            0.0, 1.0
        ))

        return {
            "delta_hs": delta_hs,
            "hs_left_raw": hs_left_raw,
            "hs_right_raw": hs_right_raw,
            "hs_left_shunted": self.hs_left_shunted,
            "hs_right_shunted": self.hs_right_shunted,
            "vs_shunted": self.vs_shunted,
            "shunt_factor": self.shunt_factor,
            "efference_copy_active": self.efference_copy_active,
            "looming_trigger": looming_trigger,
            "min_threat_dist": min_threat_dist,
            "pn_dm1_norm": pn_dm1_norm,
            "odor_diff": odor_diff,
            "egocentric_wind": egocentric_wind,
            "wind_speed": wind_speed,
            "pam_dopamine": self.pam_dopamine,
            "ppl1_dopamine": self.ppl1_dopamine,
            "temperature": self.temperature,
            "r_warm": self.r_warm,
            "r_cold": self.r_cold,
            "delta_t": delta_t,
            "thermal_stress": self.thermal_stress,
            "pain_relief_burst": self.pain_relief_burst,
            "er2_er4d_profile": self.er2_er4d_activity,
            "visual_landmark_bearing": self.visual_landmark_bearing,
            "cva_rate": self.cva_rate,
            "gr32a_rate": self.gr32a_rate,
            "p1_courtship_rate": self.p1_courtship_rate,
            "wing_extension_command": self.wing_extension_command,
            "mb_gamma_dopamine": self.mb_gamma_dopamine
        }

    def step(
        self,
        fly_pos: np.ndarray,
        fly_heading: float,
        fly_speed: float,
        fly_yaw_rate: float,
        odor_left: float,
        odor_right: float,
        wind_vector: np.ndarray,
        predator_positions: Optional[List[np.ndarray]] = None,
        predator_velocities: Optional[List[np.ndarray]] = None,
        food_ingested: bool = False,
        incurred_damage: bool = False,
        energy_level: float = 1.0,
        dt: float = 0.02,
        temperature: float = 25.0,
        landmarks: Optional[Any] = None,
        cva_odor: float = 0.0,
        bitter_pheromone: float = 0.0,
        female_aphrodisiac: float = 0.0,
        is_saccade: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes one full closed-loop step: Sensory Ingress -> Central Compass & LAL -> Descending Motor Decoders -> CPG Biomechanics.
        """
        # 1. Update Escape Timers
        if self.escape_active:
            self.escape_timer -= dt
            if self.escape_timer <= 0.0:
                self.escape_active = False
                self.escape_cooldown = 0.45  # refractory period
        elif self.escape_cooldown > 0.0:
            self.escape_cooldown -= dt

        # 2. Sensory Ingress Encoding
        sensory = self.encode_sensory(
            fly_pos=fly_pos,
            fly_heading=fly_heading,
            fly_speed=fly_speed,
            fly_yaw_rate=fly_yaw_rate,
            odor_left=odor_left,
            odor_right=odor_right,
            wind_vector=wind_vector,
            predator_positions=predator_positions,
            predator_velocities=predator_velocities,
            food_ingested=food_ingested,
            incurred_damage=incurred_damage,
            energy_level=energy_level,
            dt=dt,
            temperature=temperature,
            landmarks=landmarks,
            cva_odor=cva_odor,
            bitter_pheromone=bitter_pheromone,
            female_aphrodisiac=female_aphrodisiac,
            is_saccade=is_saccade,
            **kwargs
        )

        # 2b. Mushroom Body Learning Step
        # ---------------------------------------------------------------
        # The MB receives odor identity (food / alarm) and dopamine
        # reinforcement signals, updates KC→MBON synaptic weights via
        # anti-Hebbian plasticity, and outputs net valence that biases
        # the LAL steering cascade toward approach or avoidance.
        mb_result = None
        if self.mb_circuit is not None:
            # Map arena odors to MB inputs:
            # Odor A (food) = mean of bilateral food odor sensors
            # Odor B (alarm pheromone) = emitted at predator strike / damage sites
            # Arena odors are typically 0.0-0.5; the MB circuit's KC threshold
            # requires PN-level input ~1.0+ for sparse activation. This gain
            # stage models the ORN→PN amplification (~10-50× convergent fan-out
            # through the antennal lobe calyx - Caron et al., Nature 2013).
            raw_odor_a = 0.5 * (odor_left + odor_right)
            self.current_odor_a = min(2.5, raw_odor_a * 4.0)  # PN amplification gain
            # Alarm pheromone is triggered by nearby damage events
            if incurred_damage:
                self.current_odor_b = min(2.5, self.current_odor_b + 1.5)
            else:
                self.current_odor_b = max(0.0, self.current_odor_b - 1.5 * dt)

            # Reward signal: PAM dopamine fires upon food ingestion (sucrose)
            # scaled by hunger (starving flies learn faster - Krashes et al. 2009)
            reward_signal = sensory["pam_dopamine"] * (1.0 + 0.8 * (1.0 - self.energy_reserve))

            # Punishment signal: PPL1 dopamine fires upon hazard/damage
            punishment_signal = sensory["ppl1_dopamine"]

            # Run one MB time-bin (10ms plasticity rule bins within dt)
            n_bins = max(1, int(dt / 0.01))
            bin_dt = dt / n_bins
            for _ in range(n_bins):
                mb_result = self.mb_circuit.step(
                    odor_a=self.current_odor_a,
                    odor_b=self.current_odor_b,
                    reward=reward_signal,
                    punishment=punishment_signal,
                    dt_seconds=bin_dt,
                    learning=self.mb_learning_enabled
                )

            # Extract valence readout from MBONs
            self.mb_valence = mb_result['net_behavior']
            self.mb_approach_bias = mb_result['approach_bias']
            self.mb_avoidance_bias = mb_result['avoidance_bias']

        # 2c. Habituation Dynamics
        # Exponential recovery toward 0 (no habituation)
        self.visual_habituation *= math.exp(-dt / self.tau_habituation)
        self.odor_habituation *= math.exp(-dt / self.tau_habituation)
        # Prolonged odor without gradient change induces habituation
        odor_gradient_strength = abs(odor_left - odor_right)
        if (odor_left + odor_right) > 0.1 and odor_gradient_strength < 0.01:
            self.odor_habituation = min(0.8, self.odor_habituation + 0.3 * dt)

        # Update experience counters
        self.simulation_time += dt
        if food_ingested:
            self.total_food_collected += 1
        if incurred_damage:
            self.total_damage_events += 1

        # 3. Central Complex Compass Update
        if self.cx_lesioned:
            # Lesioned Central Complex (e.g. EB/PB ablation): heading bump drifts randomly
            compass_drift = float(np.random.normal(0.0, 0.4))
            wind_bias = 0.0
            landmark_bias = 0.0
        else:
            # Angular velocity integration via P-EN clockwise/counter-clockwise shifting: dot(mu) = omega_z
            # Also anchor compass bump toward upwind direction via ER1_b / ER3a_b ring neurons
            compass_drift = fly_yaw_rate * dt
            # Wind coupling via ER1_b ring neurons (anemotactic bias)
            wind_bias = 0.15 * math.sin(sensory["egocentric_wind"]) * dt
            # Visual landmark coupling via ER2 / ER4d ring neurons (allocentric visual anchor)
            if self.visual_landmark_bearing is not None:
                landmark_bias = 0.35 * math.sin(self.visual_landmark_bearing) * dt
            else:
                landmark_bias = 0.0
        new_compass_heading = self.compass_bump_heading + compass_drift + wind_bias + landmark_bias
        self._update_compass_bump(new_compass_heading)

        # 4. Premotor LAL Steering Hierarchy
        # Target heading: In odor plume, turn upwind (surge-and-cast anemotaxis)
        # When detecting food, align with egocentric wind: theta_goal ~ 0 (directly upwind)
        # MB valence modulates approach/avoidance: positive = approach, negative = avoid
        has_odor = (odor_left + odor_right) > 0.04
        ambient_wind_speed = float(np.linalg.norm(wind_vector))

        # MB-mediated learned valence contribution to steering
        mb_steering_bias = 0.0
        if self.mb_circuit is not None and has_odor:
            # Learned appetitive valence enhances chemotactic approach gain
            # Learned aversive valence suppresses approach and drives avoidance turning
            mb_steering_bias = 0.35 * self.mb_valence * (1.0 - self.odor_habituation)

        if has_odor:
            # Odor detected: bilateral antenna difference + optional upwind anemotaxis in moving air
            # (odor_diff = odor_left - odor_right; when right > left, steer right)
            chemotaxis_drive = -0.55 * (sensory["odor_diff"] * 3.5)
            # MB valence scales the chemotaxis gain: positive = stronger approach, negative = weaker/reversed
            valence_gain = max(0.2, 1.0 + mb_steering_bias)
            chemotaxis_drive *= valence_gain
            if ambient_wind_speed > 5.0:
                if self.cx_lesioned:
                    # Loss of compass anchoring results in wandering, disoriented search
                    anemo_drive = float(np.random.uniform(-math.pi, math.pi))
                else:
                    anemo_drive = 0.45 * sensory["egocentric_wind"]
                goal_error = 0.60 * anemo_drive + 0.40 * chemotaxis_drive
            else:
                goal_error = chemotaxis_drive
        else:
            # Plume lost: cast perpendicular to wind or optomotor stabilize
            goal_error = -0.45 * sensory["delta_hs"]

        # PFL3 (contralateral asymmetric goal error, 90 deg phase shift)
        self.pfl3_error_l = max(0.0, -goal_error)
        self.pfl3_error_r = max(0.0, goal_error)

        # PFL2 (bilateral gain & rear stalemate resolution when error ~ 180 deg)
        abs_err = abs(goal_error)
        self.pfl2_gain = float(np.clip(abs_err / math.pi, 0.0, 1.5))
        if abs_err > 2.7:  # Rear stalemate: break symmetry with stochastic burst
            self.pfl3_error_r += 0.35

        # LAL010 Interhemispheric Cross-Inhibition (Push-Pull See-Saw)
        lal_drive_l = self.pfl3_error_l * (1.0 + self.pfl2_gain)
        lal_drive_r = self.pfl3_error_r * (1.0 + self.pfl2_gain)
        self.lal010_cross_inhibition = lal_drive_r - lal_drive_l

        # 5. Descending Neuron (DN) Decoding
        # DNa02 (Transient fine steering yaw rate)
        self.dna02_rate_l = float(max(0.0, lal_drive_l - 0.4 * lal_drive_r)) * 45.0
        self.dna02_rate_r = float(max(0.0, lal_drive_r - 0.4 * lal_drive_l)) * 45.0
        dna02_diff = self.dna02_rate_r - self.dna02_rate_l

        # Log efference copy for visual slip cancellation
        if abs(dna02_diff) > 5.0:
            self.efference_copy_history.append((time.perf_counter(), abs(dna02_diff)))

        # DNa01 (Slow course holding integration)
        self.dna01_rate_l = float(max(0.0, lal_drive_l * 12.0))
        self.dna01_rate_r = float(max(0.0, lal_drive_r * 12.0))
        self.dna01_integral = float(np.clip(self.dna01_integral + (self.dna01_rate_r - self.dna01_rate_l) * dt * 0.15, -15.0, 15.0))

        # DNp09 (P9: Pursuit forward walking velocity)
        if has_odor:
            self.dnp09_rate = float(min(65.0, sensory["pn_dm1_norm"] * 0.45))
        else:
            self.dnp09_rate = max(0.0, self.dnp09_rate - 25.0 * dt)

        # BPN (Brain-derived Peduncular Neurons: baseline walking cadence)
        # Starvation boosts baseline exploration cadence
        self.bpn_rate = 22.0 * (1.0 + 0.8 * (1.0 - self.energy_reserve))

        # MDN (Moonwalker backward walking): Triggered if facing obstacle, sudden head-on threat,
        # learned aversive MB valence, or thermal stress (T_floor > 35 C)
        if temperature > 35.0:
            self.mdn_rate = 45.0
        elif sensory["min_threat_dist"] < 12.0 and not sensory["looming_trigger"]:
            self.mdn_rate = 45.0
        elif self.mb_valence < -0.25 and has_odor:
            # Learned aversion: MB output drives backward retreat from dangerous odor
            self.mdn_rate = float(min(40.0, -self.mb_valence * 55.0))
        else:
            self.mdn_rate = max(0.0, self.mdn_rate - 50.0 * dt)

        # DNp01 (Giant Fiber Looming Escape Trigger)
        if not self.gf_lesioned and sensory["looming_trigger"] and not self.escape_active and self.escape_cooldown <= 0.0:
            self.dnp01_gf_spikes += 1
            self.escape_active = True
            self.escape_timer = 0.30  # 300 ms ballistic jump takeoff flight
            self.total_escapes += 1

        # Remote RPC connectome integration
        if self.mode == "rpc" and self.rpc_client:
            remote_res = self.rpc_client.step(sensory, duration_ms=dt * 1000.0)
            if remote_res and remote_res.get("status") == "ok":
                dna02_diff = remote_res.get("dna02_diff", dna02_diff)
                self.dna02_rate_l = remote_res.get("dna02_rate_l", self.dna02_rate_l)
                self.dna02_rate_r = remote_res.get("dna02_rate_r", self.dna02_rate_r)
                if "dnp09_rate" in remote_res and remote_res["dnp09_rate"] > 0.0:
                    self.dnp09_rate = remote_res["dnp09_rate"]
                if "bpn_rate" in remote_res and remote_res["bpn_rate"] > 0.0:
                    self.bpn_rate = remote_res["bpn_rate"]
                if "mdn_rate" in remote_res and remote_res["mdn_rate"] > 0.0:
                    self.mdn_rate = remote_res["mdn_rate"]
                if remote_res.get("dnp01_gf_spikes", 0) > 0:
                    self.dnp01_gf_spikes += remote_res["dnp01_gf_spikes"]
                    self.escape_active = True
                    self.escape_timer = 0.30

        # 6. Compliant Biomechanical Locomotion Actuation (Kuramoto-Hopf CPG)
        if self.escape_active:
            # Ballistic escape flight: High speed surge away from threat
            self.forward_speed = 42.0  # mm/s
            self.yaw_rate = float(np.sign(dna02_diff) if abs(dna02_diff) > 1.0 else 1.0) * 8.5
            self.stepping_freq = self.max_freq
        else:
            # Terrestrial stepping CPG: Frequency modulated by (DNp09 + BPN)
            total_fwd_drive = self.dnp09_rate + self.bpn_rate
            mean_drive = float(np.clip(total_fwd_drive / 50.0, 0.1, 1.8))
            self.stepping_freq = float(min(self.max_freq, self.base_freq * mean_drive))

            # Advance coupled Kuramoto-Hopf phase oscillators (strict anti-phase pi)
            # When MDN fires, reverse phase advancement direction!
            phase_sign = -1.0 if self.mdn_rate > 30.0 else 1.0
            dphi = phase_sign * 2.0 * math.pi * self.stepping_freq * dt
            self.phase_a = (self.phase_a + dphi) % (2.0 * math.pi)
            self.phase_b = (self.phase_a + math.pi) % (2.0 * math.pi)

            # Stance / Swing Ground Reaction Forces (Campaniform Sensilla load feedback)
            # Stance when sin(phi) > 0
            stance_a = max(0.0, math.sin(self.phase_a))
            stance_b = max(0.0, math.sin(self.phase_b))

            # Forward propulsion thrust vs backward retreat
            if self.mdn_rate > 30.0:
                thrust = -8.5 * (stance_a + stance_b)
            else:
                thrust = (stance_a + stance_b) * mean_drive * 16.5 - 0.15 * abs(dna02_diff)

            # Ground friction drag
            accel = (thrust - 4.5 * self.forward_speed) / 1.0
            self.forward_speed = float(max(-10.0, min(35.0, self.forward_speed + accel * dt)))

            # Steering yaw torque from DNa02 + DNa01 integration + Optomotor feedback
            steering_torque = 0.085 * dna02_diff + 0.025 * self.dna01_integral - 0.20 * sensory["delta_hs"]
            yaw_accel = (steering_torque - 3.8 * self.yaw_rate) / 0.8
            self.yaw_rate = float(self.yaw_rate + yaw_accel * dt)

        # 7. Leg Stance States (Walknet Rule 1: stance hold under load)
        is_stance_a = math.sin(self.phase_a) > 0
        is_stance_b = math.sin(self.phase_b) > 0
        leg_states = {
            "L1": is_stance_a, "R2": is_stance_a, "L3": is_stance_a,
            "R1": is_stance_b, "L2": is_stance_b, "R3": is_stance_b
        }

        # 8. Biomechanical 6-Leg Kuramoto-Hopf CPG & Proprioceptive Closed-Loop
        if self.bio_cpg:
            cpg_drive_l = lal_drive_l + (self.dnp09_rate + self.bpn_rate) / 50.0
            cpg_drive_r = lal_drive_r + (self.dnp09_rate + self.bpn_rate) / 50.0
            cpg_out = self.bio_cpg.step(
                dn_drive_left=cpg_drive_l,
                dn_drive_right=cpg_drive_r,
                mdn_backward_drive=(self.mdn_rate / 45.0),
                dt=dt
            )
            self.joint_angles = cpg_out["joint_angles"]
            self.cs_ground_forces = cpg_out["cs_loads"]

        # Update previous temperature for next step's delta T
        self.prev_temperature = self.temperature

        return {
            "forward_speed": self.forward_speed,
            "yaw_rate": self.yaw_rate,
            "stepping_freq_hz": self.stepping_freq,
            "phase_a": self.phase_a,
            "phase_b": self.phase_b,
            "leg_states": leg_states,
            "compass_bump_heading": self.compass_bump_heading,
            "dna02_diff": dna02_diff,
            "dnp09_rate": self.dnp09_rate,
            "bpn_rate": self.bpn_rate,
            "mdn_rate": self.mdn_rate,
            "dnp01_gf_spikes": self.dnp01_gf_spikes,
            "escape_active": self.escape_active,
            "wpn_wind_heading": self.wpn_wind_heading,
            "wpn_wind_speed": self.wpn_wind_speed,
            "pam_dopamine": self.pam_dopamine,
            "ppl1_dopamine": self.ppl1_dopamine,
            "energy_reserve": self.energy_reserve,
            "mode": self.mode,
            # Mushroom Body Learning Telemetry
            "mb_valence": self.mb_valence,
            "mb_approach_bias": self.mb_approach_bias,
            "mb_avoidance_bias": self.mb_avoidance_bias,
            "mb_result": mb_result,
            # Habituation State
            "visual_habituation": self.visual_habituation,
            "odor_habituation": self.odor_habituation,
            # Experience Counters (cumulative)
            "total_food_collected": self.total_food_collected,
            "total_escapes": self.total_escapes,
            "total_damage_events": self.total_damage_events,
            "simulation_time": self.simulation_time,
            # Phase 2 Multimodal Telemetry
            "temperature": self.temperature,
            "r_warm": self.r_warm,
            "r_cold": self.r_cold,
            "delta_t": sensory["delta_t"],
            "thermal_stress": self.thermal_stress,
            "pain_relief_burst": self.pain_relief_burst,
            "er2_er4d_profile": self.er2_er4d_activity,
            "visual_landmark_bearing": self.visual_landmark_bearing,
            "efference_copy_active": self.efference_copy_active,
            "joint_angles": self.joint_angles,
            "cuticular_loads": self.cs_ground_forces,
            "lc_features": self.lc_features,
            "wpn_differential": sensory.get("wpn_differential", 0.0),
            "shunt_factor": self.shunt_factor,
            "vs_shunted": self.vs_shunted,
            "cva_rate": self.cva_rate,
            "gr32a_rate": self.gr32a_rate,
            "p1_courtship_rate": self.p1_courtship_rate,
            "wing_extension_command": self.wing_extension_command,
            "mb_gamma_dopamine": self.mb_gamma_dopamine,
        }
