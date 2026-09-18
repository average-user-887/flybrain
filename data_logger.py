"""Scientific Data Logger for Drosophila Whole-Brain Connectome Simulation.

Captures per-timestep neural, behavioral, and environmental telemetry in
structured formats (CSV, JSON, NPZ) for offline analysis and publication.

Implements the standard Drosophila behavioral assay data protocols:
- T-maze preference index (PI) calculation
- Learning curve quantification (ΔPI = PI_trained - PI_naive)
- Occupancy heatmaps (spatial probability distributions)
- Descending neuron population activity vectors
- Mushroom Body KC→MBON weight evolution traces

References:
    Tully & Quinn (1985) J Comp Physiol A — classical conditioning assay
    Cognigni et al. (2018) Curr Biol — MB output → approach/avoidance
    Aso & Rubin (2016) eLife — DAN compartment-specific roles
"""

import json
import time
import os
import math
import csv
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np


class ScientificDataLogger:
    """Logs simulation telemetry for scientific analysis and reproducibility.
    
    Captures hierarchical data streams at multiple time scales:
    - Per-step (~20ms): Neural firing rates, motor commands, sensory inputs
    - Per-trial (~minutes): Learning curves, preference indices, survival
    - Per-experiment: Population statistics, parameter sweeps
    """
    
    def __init__(
        self,
        output_dir: str = "./experiment_data",
        experiment_id: Optional[str] = None,
        log_interval: int = 1,          # Log every N steps (1 = every step)
        max_buffer_size: int = 50000,   # Flush to disk after this many rows
        include_mb_weights: bool = False,  # Log full KC→MBON weight matrix (large)
    ):
        self.output_dir = output_dir
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.log_interval = log_interval
        self.max_buffer_size = max_buffer_size
        self.include_mb_weights = include_mb_weights
        
        # Create output directory
        os.makedirs(os.path.join(output_dir, self.experiment_id), exist_ok=True)
        
        # Per-step telemetry buffer
        self.step_buffer: List[Dict[str, Any]] = []
        self.step_count = 0
        self.flush_count = 0
        
        # Per-trial summaries
        self.trial_summaries: List[Dict[str, Any]] = []
        self.current_trial: Dict[str, Any] = {
            "trial_id": 0,
            "start_time": 0.0,
            "food_collected": 0,
            "escapes": 0,
            "damage_events": 0,
            "total_steps": 0,
            "mean_speed": 0.0,
            "mean_valence": 0.0,
            "preference_index": 0.0,
            "time_in_odor_zone": 0.0,
            "time_total": 0.0,
        }
        
        # Spatial occupancy histogram (for heatmap analysis)
        self.occupancy_bins = 50
        self.occupancy_range = (-250.0, 250.0)
        self.occupancy_grid = np.zeros(
            (self.occupancy_bins, self.occupancy_bins), dtype=np.float64
        )
        
        # MB weight evolution tracking
        self.weight_snapshots: List[Dict[str, Any]] = []
        
        # Experiment metadata
        self.metadata: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "log_interval": log_interval,
            "software": "FlyBrain Connectome Simulation v1.0",
            "description": "",
        }
    
    def set_metadata(self, **kwargs):
        """Set experiment metadata for reproducibility."""
        self.metadata.update(kwargs)
    
    def log_step(self, telemetry: Dict[str, Any], fly_pos: Optional[np.ndarray] = None):
        """Log one simulation step telemetry.
        
        Args:
            telemetry: Dict returned by ConnectomeBridge.step()
            fly_pos: Optional [x, y] position for occupancy tracking
        """
        self.step_count += 1
        
        if self.step_count % self.log_interval != 0:
            return
        
        # Extract scalar telemetry (skip nested dicts/arrays for CSV compat)
        row = {
            "step": self.step_count,
            "sim_time": telemetry.get("simulation_time", 0.0),
            # Motor output
            "forward_speed": telemetry.get("forward_speed", 0.0),
            "yaw_rate": telemetry.get("yaw_rate", 0.0),
            "stepping_freq_hz": telemetry.get("stepping_freq_hz", 0.0),
            # Descending neuron population
            "dna02_diff": telemetry.get("dna02_diff", 0.0),
            "dnp09_rate": telemetry.get("dnp09_rate", 0.0),
            "bpn_rate": telemetry.get("bpn_rate", 0.0),
            "mdn_rate": telemetry.get("mdn_rate", 0.0),
            "dnp01_gf_spikes": telemetry.get("dnp01_gf_spikes", 0),
            "escape_active": int(telemetry.get("escape_active", False)),
            # Central complex
            "compass_heading": telemetry.get("compass_bump_heading", 0.0),
            # Neuromodulation
            "pam_dopamine": telemetry.get("pam_dopamine", 0.0),
            "ppl1_dopamine": telemetry.get("ppl1_dopamine", 0.0),
            "energy_reserve": telemetry.get("energy_reserve", 1.0),
            # Mushroom Body Learning
            "mb_valence": telemetry.get("mb_valence", 0.0),
            "mb_approach_bias": telemetry.get("mb_approach_bias", 0.0),
            "mb_avoidance_bias": telemetry.get("mb_avoidance_bias", 0.0),
            # Habituation
            "visual_habituation": telemetry.get("visual_habituation", 0.0),
            "odor_habituation": telemetry.get("odor_habituation", 0.0),
            # Wind
            "wpn_wind_heading": telemetry.get("wpn_wind_heading", 0.0),
            "wpn_wind_speed": telemetry.get("wpn_wind_speed", 0.0),
            # Locomotion phases
            "phase_a": telemetry.get("phase_a", 0.0),
            "phase_b": telemetry.get("phase_b", 0.0),
        }
        
        # Add position if available
        if fly_pos is not None:
            row["pos_x"] = float(fly_pos[0])
            row["pos_y"] = float(fly_pos[1])
            # Update occupancy histogram
            self._update_occupancy(fly_pos)
        
        # Add MB learning detail if available
        mb_result = telemetry.get("mb_result")
        if mb_result and isinstance(mb_result, dict):
            row["active_kc_count"] = mb_result.get("active_kc_count", 0)
            row["active_kc_fraction"] = mb_result.get("active_kc_fraction", 0.0)
            row["mean_approach_weight"] = mb_result.get("mean_approach_weight", 1.0)
            row["mean_avoidance_weight"] = mb_result.get("mean_avoidance_weight", 1.0)
            row["pam_firing"] = mb_result.get("pam_firing", 0.0)
            row["ppl1_firing"] = mb_result.get("ppl1_firing", 0.0)
        
        self.step_buffer.append(row)
        
        # Update current trial stats
        self.current_trial["total_steps"] += 1
        self.current_trial["mean_speed"] += row["forward_speed"]
        self.current_trial["mean_valence"] += row["mb_valence"]
        if row.get("pos_x") is not None and (row["forward_speed"] > 0.5):
            # Check if in odor zone (simplified: positive odor concentration at position)
            if row["dnp09_rate"] > 5.0:
                self.current_trial["time_in_odor_zone"] += self.log_interval * 0.02
        self.current_trial["time_total"] += self.log_interval * 0.02
        
        # Auto-flush if buffer is full
        if len(self.step_buffer) >= self.max_buffer_size:
            self.flush_steps()
    
    def _update_occupancy(self, pos: np.ndarray):
        """Update spatial occupancy histogram."""
        lo, hi = self.occupancy_range
        bx = int((pos[0] - lo) / (hi - lo) * self.occupancy_bins)
        by = int((pos[1] - lo) / (hi - lo) * self.occupancy_bins)
        if 0 <= bx < self.occupancy_bins and 0 <= by < self.occupancy_bins:
            self.occupancy_grid[by, bx] += 1.0
    
    def snapshot_mb_weights(self, bridge, label: str = ""):
        """Snapshot current KC→MBON weight matrix for learning curve analysis.
        
        Args:
            bridge: ConnectomeBridge instance with mb_circuit
            label: Human-readable label (e.g. "pre-training", "post-CS+")
        """
        if bridge.mb_circuit is None:
            return
        eff_weights = bridge.mb_circuit.get_effective_weights()
        snapshot = {
            "step": self.step_count,
            "sim_time": bridge.simulation_time,
            "label": label,
            "mean_approach_weight": float(np.mean(eff_weights[:, 0])),
            "mean_avoidance_weight": float(np.mean(eff_weights[:, 1])),
            "std_approach_weight": float(np.std(eff_weights[:, 0])),
            "std_avoidance_weight": float(np.std(eff_weights[:, 1])),
            "mb_valence": bridge.mb_valence,
            "mb_approach": bridge.mb_approach_bias,
            "mb_avoidance": bridge.mb_avoidance_bias,
        }
        if self.include_mb_weights:
            snapshot["weights_approach"] = eff_weights[:, 0].tolist()
            snapshot["weights_avoidance"] = eff_weights[:, 1].tolist()
        self.weight_snapshots.append(snapshot)
    
    def end_trial(
        self,
        food_collected: int = 0,
        escapes: int = 0,
        damage_events: int = 0,
    ):
        """Finalize current trial summary and start a new trial.
        
        Returns:
            Dict with trial summary statistics
        """
        n = max(1, self.current_trial["total_steps"])
        self.current_trial["mean_speed"] /= n
        self.current_trial["mean_valence"] /= n
        self.current_trial["food_collected"] = food_collected
        self.current_trial["escapes"] = escapes
        self.current_trial["damage_events"] = damage_events
        
        # Calculate Preference Index (PI):
        # In Drosophila conditioning (Tully & Quinn 1985; Aso & Rubin 2016),
        # PI reflects the net approach-avoidance choice driven by MBON valence:
        if abs(self.current_trial["mean_valence"]) > 1e-4:
            self.current_trial["preference_index"] = float(np.clip(self.current_trial["mean_valence"], -1.0, 1.0))
        else:
            t_in = self.current_trial["time_in_odor_zone"]
            t_total = max(0.001, self.current_trial["time_total"])
            t_out = t_total - t_in
            self.current_trial["preference_index"] = (t_in - t_out) / t_total
        
        summary = dict(self.current_trial)
        self.trial_summaries.append(summary)
        
        # Reset for next trial
        trial_id = self.current_trial["trial_id"] + 1
        self.current_trial = {
            "trial_id": trial_id,
            "start_time": self.step_count * 0.02,
            "food_collected": 0,
            "escapes": 0,
            "damage_events": 0,
            "total_steps": 0,
            "mean_speed": 0.0,
            "mean_valence": 0.0,
            "preference_index": 0.0,
            "time_in_odor_zone": 0.0,
            "time_total": 0.0,
        }
        
        return summary
    
    def compute_learning_index(self) -> float:
        """Compute learning index (ΔPI) from trial summaries.
        
        Learning index = PI_trained_test - PI_naive_baseline
        following Tully & Quinn (1985) convention.
        
        Returns:
            float: Learning index [-2.0, +2.0]. Positive = appetitive learning,
                   negative = aversive learning.
        """
        if len(self.trial_summaries) < 2:
            return 0.0
        # First trial is naive baseline, last trial is post-training test
        pi_baseline = self.trial_summaries[0]["preference_index"]
        pi_trained = self.trial_summaries[-1]["preference_index"]
        return float(pi_trained - pi_baseline)
    
    def flush_steps(self):
        """Write buffered step telemetry to CSV file."""
        if not self.step_buffer:
            return
        
        exp_dir = os.path.join(self.output_dir, self.experiment_id)
        filename = f"steps_{self.flush_count:04d}.csv"
        filepath = os.path.join(exp_dir, filename)
        
        # Write CSV header + rows
        headers = list(self.step_buffer[0].keys())
        with open(filepath, 'w') as f:
            f.write(",".join(headers) + "\n")
            for row in self.step_buffer:
                values = [str(row.get(h, "")) for h in headers]
                f.write(",".join(values) + "\n")
        
        self.flush_count += 1
        self.step_buffer.clear()
    
    def save_experiment(self):
        """Save all experiment data to disk.
        
        Creates:
            - experiment_meta.json: Metadata and parameters
            - trial_summaries.json: Per-trial summary statistics
            - weight_evolution.json: MB weight snapshots
            - occupancy.npz: Spatial occupancy histogram
            - steps_NNNN.csv: Per-step telemetry (already flushed)
        """
        exp_dir = os.path.join(self.output_dir, self.experiment_id)
        
        # Flush remaining steps
        self.flush_steps()
        
        # Save metadata
        self.metadata["total_steps"] = self.step_count
        self.metadata["total_trials"] = len(self.trial_summaries)
        self.metadata["learning_index"] = self.compute_learning_index()
        self.metadata["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        with open(os.path.join(exp_dir, "experiment_meta.json"), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save trial summaries
        with open(os.path.join(exp_dir, "trial_summaries.json"), 'w') as f:
            json.dump(self.trial_summaries, f, indent=2)
        
        # Save weight evolution
        if self.weight_snapshots:
            with open(os.path.join(exp_dir, "weight_evolution.json"), 'w') as f:
                json.dump(self.weight_snapshots, f, indent=2)
        
        # Save occupancy grid
        np.savez_compressed(
            os.path.join(exp_dir, "occupancy.npz"),
            grid=self.occupancy_grid,
            bins=self.occupancy_bins,
            range_min=self.occupancy_range[0],
            range_max=self.occupancy_range[1],
        )
        
        return exp_dir
    
    def get_summary(self) -> Dict[str, Any]:
        """Return a quick summary of experiment state."""
        return {
            "experiment_id": self.experiment_id,
            "total_steps": self.step_count,
            "total_trials": len(self.trial_summaries),
            "learning_index": self.compute_learning_index(),
            "buffer_size": len(self.step_buffer),
            "flush_count": self.flush_count,
            "weight_snapshots": len(self.weight_snapshots),
        }


class LearningAssay:
    """Standard Drosophila olfactory learning assay runner.
    
    Implements the T-maze aversive/appetitive conditioning paradigm:
    1. Baseline test: Naive preference between odor A and odor B
    2. Training: Pair CS+ odor with US (reward or punishment) 
    3. Test: Re-measure preference after training
    4. Compute Performance Index (PI) and Learning Index (ΔPI)
    
    This directly maps to the Tully & Quinn (1985) protocol and
    can be compared against real behavioral data from Berlin-K, Canton-S,
    and mutant lines (rutabaga, dunce, amnesiac).
    
    References:
        Tully & Quinn (1985) J Comp Physiol A 157: 263-277
        Aso et al. (2014) eLife — complete MB output map
    """
    
    def __init__(
        self,
        bridge,  # ConnectomeBridge instance
        logger: Optional[ScientificDataLogger] = None,
        training_steps: int = 500,
        test_steps: int = 200,
        inter_trial_interval: int = 100,
    ):
        self.bridge = bridge
        self.logger = logger or ScientificDataLogger()
        self.training_steps = training_steps
        self.test_steps = test_steps
        self.iti_steps = inter_trial_interval
        self.dt = 0.02  # 20ms per step
        
    def _run_phase(
        self,
        steps: int,
        odor_a: float = 0.0,
        odor_b: float = 0.0,
        reward: bool = False,
        punishment: bool = False,
        label: str = "phase",
    ) -> Dict[str, Any]:
        """Run a single assay phase (test or training).
        
        Returns dict with phase summary metrics.
        """
        pos = np.array([0.0, 0.0])
        heading = 0.0
        speed = 0.0
        yaw = 0.0
        wind = np.array([5.0, 0.0])  # Constant gentle wind
        
        phase_data = {
            "label": label,
            "steps": steps,
            "valence_history": [],
            "speed_history": [],
        }
        
        for i in range(steps):
            # Bilateral odor: symmetric (no gradient in T-maze arms)
            result = self.bridge.step(
                fly_pos=pos,
                fly_heading=heading,
                fly_speed=speed,
                fly_yaw_rate=yaw,
                odor_left=odor_a * 0.5,
                odor_right=odor_a * 0.5,
                wind_vector=wind,
                predator_positions=None,
                predator_velocities=None,
                food_ingested=reward,
                incurred_damage=punishment,
                energy_level=0.6,  # Moderately hungry
                dt=self.dt
            )
            
            speed = result["forward_speed"]
            yaw = result["yaw_rate"]
            heading += yaw * self.dt
            pos = pos + speed * np.array([
                np.cos(heading), np.sin(heading)
            ]) * self.dt
            
            phase_data["valence_history"].append(result["mb_valence"])
            phase_data["speed_history"].append(speed)
            
            if self.logger:
                self.logger.log_step(result, fly_pos=pos)
        
        phase_data["mean_valence"] = float(np.mean(phase_data["valence_history"]))
        phase_data["final_valence"] = float(phase_data["valence_history"][-1]) if phase_data["valence_history"] else 0.0
        phase_data["mean_speed"] = float(np.mean(phase_data["speed_history"]))
        
        return phase_data
    
    def run_appetitive_conditioning(self) -> Dict[str, Any]:
        """Run a complete appetitive conditioning assay.
        
        Protocol:
        1. Naive test (Odor A, no reinforcement) → baseline PI
        2. Training (Odor A + sucrose reward via PAM dopamine)
        3. Test (Odor A, no reinforcement) → trained PI
        4. Compute ΔPI (learning index)
        
        Returns:
            Dict with assay results including naive_PI, trained_PI, delta_PI
        """
        if self.logger:
            self.logger.set_metadata(
                assay_type="appetitive_conditioning",
                protocol="Tully_Quinn_1985_appetitive",
                cs_plus="odor_A_vinegar",
                us="sucrose_reward_PAM",
            )
            self.logger.snapshot_mb_weights(self.bridge, "pre-naive-test")
        
        # Phase 1: Naive test
        naive = self._run_phase(
            self.test_steps, odor_a=0.3, odor_b=0.0,
            label="naive_test"
        )
        if self.logger:
            trial1 = self.logger.end_trial()
            self.logger.snapshot_mb_weights(self.bridge, "post-naive-test")
        
        # Inter-trial interval (no odor)
        self._run_phase(self.iti_steps, label="iti_1")
        
        # Phase 2: Training (CS+ = Odor A paired with reward)
        training = self._run_phase(
            self.training_steps, odor_a=0.5, odor_b=0.0,
            reward=True, label="training_CS_plus"
        )
        if self.logger:
            trial2 = self.logger.end_trial()
            self.logger.snapshot_mb_weights(self.bridge, "post-training")
        
        # Inter-trial interval
        self._run_phase(self.iti_steps, label="iti_2")
        
        # Phase 3: Test (CS+ alone, no reinforcement)
        test = self._run_phase(
            self.test_steps, odor_a=0.3, odor_b=0.0,
            label="trained_test"
        )
        if self.logger:
            trial3 = self.logger.end_trial()
            self.logger.snapshot_mb_weights(self.bridge, "post-test")
        
        # Compute learning index
        delta_pi = 0.0
        if self.logger:
            delta_pi = self.logger.compute_learning_index()
        
        return {
            "assay_type": "appetitive_conditioning",
            "naive_valence": naive["mean_valence"],
            "training_valence": training["mean_valence"],
            "trained_valence": test["mean_valence"],
            "delta_valence": test["mean_valence"] - naive["mean_valence"],
            "learning_index": delta_pi,
            "naive_speed": naive["mean_speed"],
            "trained_speed": test["mean_speed"],
        }
    
    def run_aversive_conditioning(self) -> Dict[str, Any]:
        """Run a complete aversive conditioning assay.
        
        Protocol:
        1. Naive test (Odor B alarm pheromone, no reinforcement) → baseline
        2. Training (Odor B + electric shock punishment via PPL1 dopamine)
        3. Test (Odor B, no reinforcement) → trained response
        4. Compute avoidance learning
        """
        if self.logger:
            self.logger.set_metadata(
                assay_type="aversive_conditioning",
                protocol="Tully_Quinn_1985_aversive",
                cs_plus="odor_B_alarm_pheromone",
                us="electric_shock_PPL1",
            )
            self.logger.snapshot_mb_weights(self.bridge, "pre-naive-test")
        
        # Phase 1: Naive test (Odor B)
        naive = self._run_phase(
            self.test_steps, odor_a=0.0, odor_b=0.4,
            label="naive_test_odorB"
        )
        if self.logger:
            self.logger.end_trial()
            self.logger.snapshot_mb_weights(self.bridge, "post-naive-test")
        
        # ITI
        self._run_phase(self.iti_steps, label="iti_1")
        
        # Phase 2: Training (Odor B + punishment)
        training = self._run_phase(
            self.training_steps, odor_a=0.0, odor_b=0.6,
            punishment=True, label="training_CS_plus_shock"
        )
        if self.logger:
            self.logger.end_trial()
            self.logger.snapshot_mb_weights(self.bridge, "post-training")
        
        # ITI
        self._run_phase(self.iti_steps, label="iti_2")
        
        # Phase 3: Test (Odor B alone)
        test = self._run_phase(
            self.test_steps, odor_a=0.0, odor_b=0.4,
            label="trained_test_odorB"
        )
        if self.logger:
            self.logger.end_trial()
            self.logger.snapshot_mb_weights(self.bridge, "post-test")
        
        delta_pi = 0.0
        if self.logger:
            delta_pi = self.logger.compute_learning_index()
        
        return {
            "assay_type": "aversive_conditioning",
            "naive_valence": naive["mean_valence"],
            "training_valence": training["mean_valence"],
            "trained_valence": test["mean_valence"],
            "delta_valence": test["mean_valence"] - naive["mean_valence"],
            "learning_index": delta_pi,
        }


# ==============================================================================
# Statistical Helper Functions for Cohort Analysis (Pure Python / NumPy)
# ==============================================================================

def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function."""
    maxit = 100
    eps = 3.0e-7
    fpmin = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < eps:
            break
    return h


def ibeta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    else:
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_distribution_p_value(t_stat: float, df: int) -> float:
    """Two-tailed p-value for Student's t-distribution with df degrees of freedom."""
    if df < 1 or math.isnan(t_stat):
        return 1.0
    t = abs(float(t_stat))
    if t == 0.0:
        return 1.0
    x = float(df) / (float(df) + t * t)
    p = ibeta(0.5 * df, 0.5, x)
    return float(max(0.0, min(1.0, p)))


def compute_cohort_statistics(values: List[Any]) -> Dict[str, float]:
    """Computes mean, std, SEM, t-statistic, two-tailed p-value, and Cohen's d."""
    clean_vals: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
            if not math.isnan(fv):
                clean_vals.append(fv)
        except (ValueError, TypeError):
            continue

    n = len(clean_vals)
    if n == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
        }
    mean = float(np.mean(clean_vals))
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "sem": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
        }

    std = float(np.std(clean_vals, ddof=1))
    sem = float(std / math.sqrt(n)) if n > 0 else 0.0
    t_stat = float(mean / sem) if sem > 1e-12 else 0.0
    cohens_d = float(mean / std) if std > 1e-12 else 0.0
    p_val = t_distribution_p_value(t_stat, n - 1)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "t_stat": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
    }


def _json_serialize_value(obj: Any) -> Any:
    """Helper to convert NumPy and custom objects into standard JSON-serializable primitives."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _json_serialize_value(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_json_serialize_value(item) for item in obj]
    return obj


class SummaryPath(str):
    """String subclass representing path to summary JSON with attached metadata."""
    def __new__(cls, file_path: str, trial_id: str, summary: Dict[str, Any]):
        s = super().__new__(cls, file_path)
        s.file_path = file_path
        s.trial_id = trial_id
        s.summary = summary
        return s


class ReportContent(str):
    """String subclass containing markdown report with attached file_path attribute."""
    def __new__(cls, content: str, file_path: str):
        s = super().__new__(cls, content)
        s.file_path = file_path
        s.path = file_path
        return s


# ==============================================================================
# ParadigmDataLogger
# ==============================================================================

class ParadigmDataLogger:
    """Specialized scientific telemetry logger for neuroethological assay paradigms.

    Supports:
    - Continuous CSV telemetry streaming (<trial_id>_telemetry.csv)
    - Clean paradigm-specific telemetry extraction (heat-maze, optomotor, looming, etc.)
    - 2D trajectory accumulation & spatial occupancy binning (<trial_id>_occupancy.npz)
    - Full trial summary compilation with canonical metrics (<trial_id>_summary.json)
    - Statistical cohort reporting with t-tests & Cohen's d (PARADIGM_REPORT.md)
    """

    PARADIGM_COLUMNS = {
        'heat_maze': ['floor_temp', 'epg_bump', 'pam_burst'],
        'optomotor': ['hs_rate', 'drum_vel', 'saccade_shunt'],
        'looming_escape': ['looming_size', 'gf_spike'],
        't_maze': ['odor_a', 'odor_b', 'active_arm'],
        'courtship': ['wing_angle', 'p1_rate', 'rejection'],
        'buridan': ['centrophobism', 'stripe_fixation', 'in_moat'],
        'y_maze': ['current_zone', 'alternation_rate'],
        'visual_operant': ['drum_angle', 'laser_active', 'yaw_torque'],
        'wind_tunnel': ['wind_speed', 'odor_conc', 'source_reached'],
        'gap_crossing': ['gap_width', 'probing', 'crossing_success'],
        'circadian_dam': ['is_lights_on', 'beam_crossed', 'is_sleeping'],
        'labyrinth': ['goal_distance', 'goal_reached', 'collided'],
    }

    BASE_COLUMNS = [
        'step', 'sim_time', 'pos_x', 'pos_y', 'heading', 'speed', 'reward', 'punishment'
    ]

    def __init__(
        self,
        paradigm_name: str,
        output_dir: str = "./experiment_data",
        trial_id: Optional[str] = None,
        occupancy_bins: int = 50,
        occupancy_range: Optional[Tuple[float, float]] = None,
        dt: float = 0.02,
        max_buffer_size: int = 500,
    ):
        self.paradigm_name = paradigm_name
        self.norm_paradigm = paradigm_name.lower().replace('-', '_')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.trial_counter = 1
        self.trial_id = trial_id or f"{self.norm_paradigm}_trial_{self.trial_counter:03d}"
        self.dt = dt
        self.max_buffer_size = max_buffer_size

        # Spatial tracking
        self.occupancy_bins = occupancy_bins
        self.occupancy_range = occupancy_range or (-150.0, 150.0)
        self.occupancy_grid = np.zeros(
            (self.occupancy_bins, self.occupancy_bins), dtype=np.float64
        )
        self.trajectory: List[Tuple[float, float]] = []

        # Buffering & state
        self.step_buffer: List[Dict[str, Any]] = []
        self.step_count = 0
        self.trial_start_time = time.time()
        self.trial_summaries: List[Dict[str, Any]] = []
        self.last_summary: Optional[Dict[str, Any]] = None
        self.last_report: Optional[str] = None
        self.last_report_path: Optional[str] = None

        # CSV streaming state
        self.csv_path = os.path.join(self.output_dir, f"{self.trial_id}_telemetry.csv")
        self.csv_headers: Optional[List[str]] = None
        self.csv_headers_written = False

    def start_trial(self, trial_id: Optional[str] = None):
        """Starts a new trial, resetting per-trial buffers and setting a new trial ID."""
        self.flush_steps()
        if trial_id is not None:
            self.trial_id = trial_id
        else:
            self.trial_counter += 1
            self.trial_id = f"{self.norm_paradigm}_trial_{self.trial_counter:03d}"

        self.step_count = 0
        self.trial_start_time = time.time()
        self.step_buffer.clear()
        self.trajectory.clear()
        self.occupancy_grid.fill(0.0)

        self.csv_path = os.path.join(self.output_dir, f"{self.trial_id}_telemetry.csv")
        self.csv_headers = None
        self.csv_headers_written = False

    def _update_occupancy(self, pos: Tuple[float, float]):
        """Bins 2D position into spatial occupancy grid."""
        x, y = float(pos[0]), float(pos[1])
        if len(self.occupancy_range) == 4:
            x_lo, x_hi, y_lo, y_hi = self.occupancy_range
        else:
            x_lo, x_hi = self.occupancy_range
            y_lo, y_hi = x_lo, x_hi

        if x_hi > x_lo and y_hi > y_lo:
            bx = int((x - x_lo) / (x_hi - x_lo) * self.occupancy_bins)
            by = int((y - y_lo) / (y_hi - y_lo) * self.occupancy_bins)
            if 0 <= bx < self.occupancy_bins and 0 <= by < self.occupancy_bins:
                self.occupancy_grid[by, bx] += 1.0

    def _extract_paradigm_telemetry(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts clean, normalized paradigm-specific telemetry columns."""
        stimuli = step_data.get('stimuli') or {}
        telem = step_data.get('paradigm_telemetry') or {}
        conn = step_data.get('connectome') or {}
        active_zones = step_data.get('active_zones') or []
        metrics = step_data.get('paradigm_metrics') or {}

        p_dict: Dict[str, Any] = {}

        # 1. Heat Maze
        if 'heat' in self.norm_paradigm:
            temp = step_data.get('floor_temp', stimuli.get('temperature', step_data.get('temperature', 25.0)))
            epg = step_data.get('epg_bump', conn.get('compass_bump_heading', conn.get('epg_bump', step_data.get('compass_bump_heading', 0.0))))
            pam = step_data.get('pam_burst', float(telem.get('pam_burst_active', False)))
            if pam == 0.0 and 'pam_dopamine' in conn:
                pam = float(conn['pam_dopamine'] > 0.5)
            p_dict['floor_temp'] = float(temp)
            p_dict['epg_bump'] = float(epg)
            p_dict['pam_burst'] = float(pam)

        # 2. Optomotor
        elif 'optomotor' in self.norm_paradigm:
            hs = step_data.get('hs_rate', telem.get('hs_firing_rate', conn.get('hs_left_shunted', step_data.get('hs_firing_rate', 0.0))))
            drum = step_data.get('drum_vel', telem.get('drum_velocity_deg_s', stimuli.get('drum_velocity_deg_s', step_data.get('drum_velocity_deg_s', 0.0))))
            saccade = step_data.get('saccade_shunt', float(telem.get('efference_copy_active', conn.get('efference_copy_active', False))))
            p_dict['hs_rate'] = float(hs)
            p_dict['drum_vel'] = float(drum)
            p_dict['saccade_shunt'] = float(saccade)

        # 3. Looming Escape
        elif 'looming' in self.norm_paradigm:
            l_size = step_data.get('looming_size', stimuli.get('theta_deg', stimuli.get('theta_rad', 0.0)))
            gf = step_data.get('gf_spike', float(telem.get('gf_spike', conn.get('dnp01_gf_spikes', 0) > 0)))
            p_dict['looming_size'] = float(l_size)
            p_dict['gf_spike'] = float(gf)

        # 4. T-Maze
        elif 't_maze' in self.norm_paradigm or 'tmaze' in self.norm_paradigm:
            oa = step_data.get('odor_a', stimuli.get('odor_a', stimuli.get('odor_cs_plus', 0.0)))
            ob = step_data.get('odor_b', stimuli.get('odor_b', stimuli.get('odor_cs_minus', 0.0)))
            active = step_data.get('active_arm')
            if active is None:
                active = active_zones[0] if active_zones else telem.get('first_choice', '')
            p_dict['odor_a'] = float(oa)
            p_dict['odor_b'] = float(ob)
            p_dict['active_arm'] = str(active)

        # 5. Courtship
        elif 'courtship' in self.norm_paradigm:
            w_angle = step_data.get('wing_angle', telem.get('wing_extension_angle_deg', conn.get('wing_extension_command', 0.0)))
            p1 = step_data.get('p1_rate', conn.get('p1_courtship_rate', 0.0))
            rej = step_data.get('rejection', telem.get('rejection_kicks', 0))
            p_dict['wing_angle'] = float(w_angle)
            p_dict['p1_rate'] = float(p1)
            p_dict['rejection'] = float(rej)

        # 6. Buridan
        elif 'buridan' in self.norm_paradigm:
            cp = step_data.get('centrophobism', telem.get('centrophobism_index', 0.0))
            sf = step_data.get('stripe_fixation', stimuli.get('stripe_fixation', telem.get('mean_stripe_fixation', 0.0)))
            moat = step_data.get('in_moat', float(stimuli.get('is_in_moat', False)))
            p_dict['centrophobism'] = float(cp)
            p_dict['stripe_fixation'] = float(sf)
            p_dict['in_moat'] = float(moat)

        # 7. Y-Maze
        elif 'y_maze' in self.norm_paradigm or 'ymaze' in self.norm_paradigm:
            cz = step_data.get('current_zone', telem.get('current_zone', ''))
            sar = step_data.get('alternation_rate', metrics.get('spontaneous_alternation_rate', 0.0))
            p_dict['current_zone'] = str(cz)
            p_dict['alternation_rate'] = float(sar)

        # 8. Visual Operant
        elif 'visual_operant' in self.norm_paradigm:
            da = step_data.get('drum_angle', stimuli.get('drum_angle_deg', 0.0))
            laser = step_data.get('laser_active', float(stimuli.get('laser_active', False)))
            torque = step_data.get('yaw_torque', telem.get('yaw_torque', 0.0))
            p_dict['drum_angle'] = float(da)
            p_dict['laser_active'] = float(laser)
            p_dict['yaw_torque'] = float(torque)

        # 9. Wind Tunnel
        elif 'wind_tunnel' in self.norm_paradigm:
            ws = step_data.get('wind_speed', stimuli.get('wind_speed', 0.0))
            oc = step_data.get('odor_conc', stimuli.get('odor_conc', 0.0))
            sr = step_data.get('source_reached', float(telem.get('source_reached', False)))
            p_dict['wind_speed'] = float(ws)
            p_dict['odor_conc'] = float(oc)
            p_dict['source_reached'] = float(sr)

        # 10. Gap Crossing
        elif 'gap_crossing' in self.norm_paradigm:
            gw = step_data.get('gap_width', stimuli.get('gap_width_mm', 0.0))
            pb = step_data.get('probing', float(stimuli.get('is_probing', False)))
            cs = step_data.get('crossing_success', float(telem.get('crossing_success', False)))
            p_dict['gap_width'] = float(gw)
            p_dict['probing'] = float(pb)
            p_dict['crossing_success'] = float(cs)

        # 11. Circadian DAM
        elif 'circadian' in self.norm_paradigm:
            lo = step_data.get('is_lights_on', float(stimuli.get('is_lights_on', False)))
            bc = step_data.get('beam_crossed', float(telem.get('beam_crossed', False)))
            sl = step_data.get('is_sleeping', float(telem.get('is_sleeping', False)))
            p_dict['is_lights_on'] = float(lo)
            p_dict['beam_crossed'] = float(bc)
            p_dict['is_sleeping'] = float(sl)

        # 12. Labyrinth
        elif 'labyrinth' in self.norm_paradigm:
            gd = step_data.get('goal_distance', stimuli.get('goal_distance_mm', 0.0))
            gr = step_data.get('goal_reached', float(telem.get('goal_reached', False)))
            col = step_data.get('collided', float(telem.get('collided', False)))
            p_dict['goal_distance'] = float(gd)
            p_dict['goal_reached'] = float(gr)
            p_dict['collided'] = float(col)

        return p_dict

    def log_step(
        self,
        step_data: Dict[str, Any],
        fly_pos: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Logs one timestep of simulation telemetry, streaming to CSV and buffering."""
        self.step_count += 1

        # Position extraction
        if fly_pos is not None:
            px, py = float(fly_pos[0]), float(fly_pos[1])
        else:
            raw_x = step_data.get('fly_x', step_data.get('pos_x'))
            raw_y = step_data.get('fly_y', step_data.get('pos_y'))
            if raw_x is not None and raw_y is not None:
                px, py = float(raw_x), float(raw_y)
            else:
                px, py = 0.0, 0.0

        pos = (px, py)
        self.trajectory.append(pos)
        self._update_occupancy(pos)

        # Common kinematics
        step_num = step_data.get('time_step', step_data.get('step', self.step_count))
        sim_time = step_data.get('sim_time', float(step_num) * self.dt)
        heading = float(step_data.get('fly_heading', step_data.get('heading', 0.0)))
        speed = float(step_data.get('fly_speed', step_data.get('forward_speed', step_data.get('speed', 0.0))))
        reward = float(step_data.get('reward', 0.0))
        punishment = float(step_data.get('punishment', 0.0))

        row: Dict[str, Any] = {
            'step': int(step_num),
            'sim_time': float(sim_time),
            'pos_x': px,
            'pos_y': py,
            'heading': heading,
            'speed': speed,
            'reward': reward,
            'punishment': punishment,
        }

        # Extract paradigm-specific columns
        paradigm_telemetry = self._extract_paradigm_telemetry(step_data)
        row.update(paradigm_telemetry)

        # Ingest any extra flat scalar columns present in step_data
        for k, v in step_data.items():
            if k not in row and isinstance(v, (int, float, str, bool)):
                row[k] = v

        self.step_buffer.append(row)
        self._write_csv_row(row)

        return row

    def _write_csv_row(self, row: Dict[str, Any]):
        """Appends a row to the CSV file, writing header if necessary."""
        write_header = not self.csv_headers_written and (
            not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        )

        if self.csv_headers is None:
            spec_cols = self.PARADIGM_COLUMNS.get(self.norm_paradigm, [])
            cols = list(self.BASE_COLUMNS)
            for c in spec_cols:
                if c not in cols:
                    cols.append(c)
            for k in row.keys():
                if k not in cols:
                    cols.append(k)
            self.csv_headers = cols

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers, extrasaction='ignore')
            if write_header:
                writer.writeheader()
                self.csv_headers_written = True
            writer.writerow(row)
            f.flush()

    def flush_steps(self):
        """Ensures all buffered steps are flushed to disk."""
        self.csv_headers_written = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

    def end_trial(
        self,
        trial_metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Finalizes current trial summary, exports JSON and NPZ files, and resets state.

        Args:
            trial_metrics: Paradigm metrics dictionary (from arena or assay).
            **kwargs: Extra metric fields to merge.

        Returns:
            str: Absolute/relative path to saved <trial_id>_summary.json.
        """
        self.flush_steps()

        metrics = dict(trial_metrics or {})
        metrics.update(kwargs)

        now = time.time()
        duration_s = max(self.dt * self.step_count, now - self.trial_start_time)
        total_steps = self.step_count

        # Trajectory & distance metrics
        traj_arr = np.array(self.trajectory, dtype=np.float64) if self.trajectory else np.empty((0, 2), dtype=np.float64)
        if len(traj_arr) > 1:
            diffs = np.diff(traj_arr, axis=0)
            step_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            total_dist = float(np.sum(step_dists))
            net_disp = float(np.sqrt(np.sum((traj_arr[-1] - traj_arr[0]) ** 2)))
            calculated_tortuosity = float(total_dist / max(1e-4, net_disp))
        else:
            total_dist = 0.0
            calculated_tortuosity = 1.0

        # Latencies
        latencies = {
            'escape_latency_ms': metrics.get('escape_latency_ms'),
            'latency_to_choice_ms': metrics.get('latency_to_choice_ms'),
            'time_to_source_ms': metrics.get('time_to_source_ms'),
            'time_to_goal_ms': metrics.get('time_to_goal_ms'),
            'probing_duration_ms': metrics.get('probing_duration_ms'),
            'time_to_collision_at_jump_ms': metrics.get('time_to_collision_at_jump_ms'),
        }
        valid_latencies = {k: float(v) for k, v in latencies.items() if v is not None}

        # Canonical paradigm metrics mapping
        pi_val = metrics.get('PI', metrics.get('performance_index', metrics.get('preference_index')))
        sar_val = metrics.get('SAR', metrics.get('spontaneous_alternation_rate'))
        cp_val = metrics.get('Centrophobism', metrics.get('centrophobism_index', metrics.get('centrophobism')))
        li_val = metrics.get('LI', metrics.get('operant_learning_index', metrics.get('learning_index')))
        opt_val = metrics.get('optomotor_gain')
        sleep_val = metrics.get('sleep_minutes', metrics.get('total_sleep_minutes'))
        court_val = metrics.get('courtship_index')
        tort_val = metrics.get('tortuosity', metrics.get('path_tortuosity', calculated_tortuosity))

        summary: Dict[str, Any] = {
            'trial_id': self.trial_id,
            'paradigm': self.paradigm_name,
            'timestamps': {
                'start_time': self.trial_start_time,
                'end_time': now,
                'start_iso': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.trial_start_time)),
                'end_iso': time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            },
            'duration': float(duration_s),
            'total_steps': int(total_steps),
            'total_distance': float(total_dist),
            'latencies': valid_latencies,
            'metrics': metrics,
            'PI': float(pi_val) if pi_val is not None else None,
            'SAR': float(sar_val) if sar_val is not None else None,
            'Centrophobism': float(cp_val) if cp_val is not None else None,
            'LI': float(li_val) if li_val is not None else None,
            'optomotor_gain': float(opt_val) if opt_val is not None else None,
            'sleep_minutes': float(sleep_val) if sleep_val is not None else None,
            'courtship_index': float(court_val) if court_val is not None else None,
            'tortuosity': float(tort_val) if tort_val is not None else None,
        }

        for k, v in metrics.items():
            if k not in summary:
                summary[k] = v

        # 1. Export JSON summary (<trial_id>_summary.json)
        json_filename = f"{self.trial_id}_summary.json"
        json_path = os.path.join(self.output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_json_serialize_value(summary), f, indent=2)

        # 2. Export spatial occupancy and trajectory arrays (<trial_id>_occupancy.npz)
        npz_filename = f"{self.trial_id}_occupancy.npz"
        npz_path = os.path.join(self.output_dir, npz_filename)
        np.savez_compressed(
            npz_path,
            occupancy=self.occupancy_grid,
            grid=self.occupancy_grid,
            trajectory=traj_arr,
            bins=self.occupancy_bins,
            range=np.array(self.occupancy_range),
        )

        self.trial_summaries.append(summary)
        self.last_summary = summary
        current_trial_id = self.trial_id

        # Prepare for next trial
        self.trial_counter += 1
        self.trial_id = f"{self.norm_paradigm}_trial_{self.trial_counter:03d}"
        self.step_count = 0
        self.trial_start_time = time.time()
        self.step_buffer.clear()
        self.trajectory.clear()
        self.occupancy_grid.fill(0.0)
        self.csv_path = os.path.join(self.output_dir, f"{self.trial_id}_telemetry.csv")
        self.csv_headers = None
        self.csv_headers_written = False

        return SummaryPath(json_path, current_trial_id, summary)

    def compute_cohort_metrics(
        self,
        trial_summaries: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Aggregates cohort statistics across trial summaries."""
        summaries = trial_summaries if trial_summaries is not None else self.trial_summaries
        if not summaries:
            return {}

        candidate_keys = set()
        for s in summaries:
            for k, v in s.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    candidate_keys.add(k)
            for k, v in s.get('metrics', {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    candidate_keys.add(k)
            for k, v in s.get('latencies', {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    candidate_keys.add(k)

        ignore_keys = {'trial_id', 'step', 'sim_time', 'start_time', 'end_time', 'total_steps'}
        cohort_metrics: Dict[str, Dict[str, float]] = {}

        for key in sorted(candidate_keys - ignore_keys):
            vals: List[float] = []
            for s in summaries:
                val = None
                if key in s and isinstance(s[key], (int, float)) and not isinstance(s[key], bool):
                    val = s[key]
                elif key in s.get('metrics', {}) and isinstance(s['metrics'][key], (int, float)) and not isinstance(s['metrics'][key], bool):
                    val = s['metrics'][key]
                elif key in s.get('latencies', {}) and isinstance(s['latencies'][key], (int, float)) and not isinstance(s['latencies'][key], bool):
                    val = s['latencies'][key]
                if val is not None:
                    vals.append(float(val))

            if vals:
                cohort_metrics[key] = compute_cohort_statistics(vals)

        return cohort_metrics

    def generate_cohort_report(
        self_or_cls,
        trial_summaries: Optional[List[Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """Aggregates cohort statistics and generates/writes Markdown summary report (PARADIGM_REPORT.md).

        Args:
            trial_summaries: Optional list of trial summaries (defaults to self.trial_summaries).
            output_dir: Optional directory to save PARADIGM_REPORT.md (defaults to self.output_dir).

        Returns:
            ReportContent: Markdown formatted summary string with .file_path attribute.
        """
        if isinstance(self_or_cls, type):
            p_name = trial_summaries[0].get('paradigm', 'paradigm') if trial_summaries else "paradigm"
            inst = self_or_cls(paradigm_name=p_name, output_dir=output_dir or "./experiment_data")
            return inst.generate_cohort_report(trial_summaries=trial_summaries, output_dir=output_dir)

        self = self_or_cls
        summaries = trial_summaries if trial_summaries is not None else self.trial_summaries
        target_dir = output_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)
        report_path = os.path.join(target_dir, "PARADIGM_REPORT.md")

        n_trials = len(summaries)
        stats = self.compute_cohort_metrics(summaries)
        date_str = time.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Drosophila Neuroethological Cohort Report: {self.paradigm_name.upper()}",
            "",
            f"- **Paradigm**: `{self.paradigm_name}`",
            f"- **Cohort Size (N)**: `{n_trials}` trials",
            f"- **Generated At**: `{date_str}`",
            f"- **Output Directory**: `{target_dir}`",
            "",
            "## 1. Aggregated Cohort Statistics",
            "",
            "| Metric | N | Mean | Std | SEM | t-statistic | p-value | Cohen's d |",
            "|:-------|:-:|:----:|:---:|:---:|:-----------:|:-------:|:---------:|",
        ]

        priority_keys = [
            'PI', 'SAR', 'Centrophobism', 'LI', 'optomotor_gain', 'sleep_minutes',
            'courtship_index', 'tortuosity', 'total_distance', 'duration'
        ]
        ordered_keys = [k for k in priority_keys if k in stats] + [k for k in sorted(stats.keys()) if k not in priority_keys]

        if not ordered_keys:
            lines.append("| *None* | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |")
        else:
            for k in ordered_keys:
                s = stats[k]
                lines.append(
                    f"| **{k}** | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['sem']:.4f} | {s['t_stat']:.4f} | {s['p_value']:.4f} | {s['cohens_d']:.4f} |"
                )

        lines.extend([
            "",
            "## 2. Individual Trial Breakdown",
            "",
            "| Trial ID | Duration (s) | Steps | Distance (mm) | Key Metric | Latency |",
            "|:---------|:------------:|:-----:|:-------------:|:----------:|:-------:|",
        ])

        if not summaries:
            lines.append("| *No trials logged* | 0.0 | 0 | 0.0 | N/A | N/A |")
        else:
            for s in summaries:
                t_id = s.get('trial_id', 'unknown')
                dur = s.get('duration', 0.0)
                st = s.get('total_steps', 0)
                dist = s.get('total_distance', 0.0)
                key_metric_str = "N/A"
                for cand in ['PI', 'SAR', 'Centrophobism', 'LI', 'optomotor_gain', 'courtship_index', 'sleep_minutes']:
                    if s.get(cand) is not None:
                        key_metric_str = f"{cand}={s[cand]:.3f}"
                        break
                lat_dict = s.get('latencies', {})
                lat_str = ", ".join(f"{k}={v:.1f}ms" for k, v in lat_dict.items()) if lat_dict else "N/A"

                lines.append(f"| `{t_id}` | {dur:.2f} | {st} | {dist:.2f} | {key_metric_str} | {lat_str} |")

        lines.extend([
            "",
            "## 3. Statistical Methodology",
            "",
            "- **Sample Standard Deviation ($s$)**: $s = \\sqrt{\\frac{1}{N-1}\\sum_{i=1}^N (x_i - \\bar{x})^2}$",
            "- **Standard Error of the Mean (SEM)**: $\\text{SEM} = \\frac{s}{\\sqrt{N}}$",
            "- **One-Sample Student's t-test**: $t = \\frac{\\bar{x} - \\mu_0}{\\text{SEM}}$ against $\\mu_0 = 0$",
            "- **Two-Tailed p-value**: Calculated via regularized incomplete beta function $I_x(a, b)$",
            "- **Cohen's d Effect Size**: $d = \\frac{\\bar{x} - \\mu_0}{s}$",
            "",
            "---",
            "*Report generated by FlyBrain Neuroethological Battery Engine.*",
            ""
        ])

        report_content = "\n".join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.last_report = report_content
        self.last_report_path = report_path

        return ReportContent(report_content, report_path)
