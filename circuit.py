"""Mushroom Body Olfactory Learning Circuit for Drosophila.

Implements the biological Drosophila mushroom body architecture:
- Antennal Lobe Projection Neurons (PNs) encoding odor spectra.
- Kenyon Cells (KCs) with sparse, random combinatorial input from PNs (~5-10% active).
- Mushroom Body Output Neurons (MBONs): Approach (appetitive) and Avoidance (aversive).
- Dopaminergic Neurons (DANs): PAM (reward / food) and PPL1 (punishment / hazard).
- Plastic synapses KC -> MBON governed by the baseline-centered anti-Hebbian rate rule
  from Huang, Luo et al. (Nature 2024, doi:10.1038/s41586-024-07819-w).
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from doom_learning_v6.rule import advance as rule_advance, PARAMETERS as RULE_PARAMETERS
except ImportError:
    # Standalone fallback if package structure differs
    RULE_PARAMETERS = {
        'trace_kc_seconds': 1.0,
        'trace_dan_seconds': 1.0,
        'memory_decay_seconds': 1800.0,
        'weight_filter_seconds': 0.05,
        'minimum_fraction': 0.1,
        'maximum_fraction': 2.0,
        'maximum_rate_bin_ms': 10.0,
    }

    def rule_advance(y_kc, y_dan, u, w, kc_hz, dan_hz, gain, dt_seconds, eta, learning=True, frozen=False):
        if not math.isfinite(dt_seconds) or dt_seconds <= 0 or dt_seconds > 0.0100001:
            raise ValueError('Rate bins must be 0--10 ms')
        h = dt_seconds
        p = RULE_PARAMETERS
        ak = math.exp(-h / p['trace_kc_seconds'])
        ad = math.exp(-h / p['trace_dan_seconds'])
        kmid = y_kc * math.sqrt(ak) + kc_hz * (1 - math.sqrt(ak))
        dmid = y_dan * math.sqrt(ad) + dan_hz * (1 - math.sqrt(ad))
        y_kc[:] = y_kc * ak + kc_hz * (1 - ak)
        y_dan[:] = y_dan * ad + dan_hz * (1 - ad)
        if frozen:
            return
        drive = eta * (kc_hz * (gain.T @ dmid) - (gain.T @ dan_hz) * kmid) if learning else np.zeros_like(u)
        tu = p['memory_decay_seconds']
        tw = p['weight_filter_seconds']
        eu = math.exp(-h / tu)
        ew = math.exp(-h / tw)
        c = tu / (tu - tw) * (eu - ew)
        old_u = u.copy()
        u[:] = old_u * eu + drive * tu * (-math.expm1(-h / tu))
        w[:] = w * ew + old_u * c + drive * tu * (-math.expm1(-h / tw) - c)
        lo = p['minimum_fraction'] - 1
        hi = p['maximum_fraction'] - 1
        np.clip(u, lo, hi, out=u)
        np.clip(w, lo, hi, out=w)


class MushroomBodyCircuit:
    """Canonical Drosophila Mushroom Body Circuit."""

    def __init__(
        self,
        n_pn: int = 40,
        n_kc: int = 120,
        pn_per_kc: int = 5,
        kc_threshold: float = 0.20,
        eta: float = 0.05,
        seed: Optional[int] = 42
    ):
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.pn_per_kc = pn_per_kc
        self.kc_threshold = kc_threshold
        self.eta = eta
        self.rng = np.random.default_rng(seed)

        # Odor tuning for Projection Neurons (2 primary odors: A = Food/Sugar, B = Hazard/Shock)
        # Odor A excites PNs 0..(n_pn//2 - 1), Odor B excites PNs (n_pn//2)..n_pn
        self.pn_tuning = np.zeros((2, self.n_pn), dtype=np.float64)
        half_pn = self.n_pn // 2
        for i in range(half_pn):
            self.pn_tuning[0, i] = np.exp(-((i - half_pn / 2) ** 2) / 8.0)
        for i in range(half_pn, self.n_pn):
            self.pn_tuning[1, i] = np.exp(-((i - 3 * half_pn / 2) ** 2) / 8.0)
        # Normalize
        self.pn_tuning[0] /= np.max(self.pn_tuning[0])
        self.pn_tuning[1] /= np.max(self.pn_tuning[1])

        # PN -> KC connectivity matrix: random sparse connections (each KC samples ~pn_per_kc PNs)
        self.w_pn_kc = np.zeros((self.n_pn, self.n_kc), dtype=np.float64)
        for k in range(self.n_kc):
            sampled_pns = self.rng.choice(self.n_pn, size=self.pn_per_kc, replace=False)
            self.w_pn_kc[sampled_pns, k] = 1.0 / self.pn_per_kc

        # Synapses KC -> MBONs
        # Column 0: Approach MBON (appetitive)
        # Column 1: Avoidance MBON (aversive)
        self.w_kc_mbon_baseline = np.ones((self.n_kc, 2), dtype=np.float64)
        self.u = np.zeros((self.n_kc, 2), dtype=np.float64)  # Long-term efficacy deviation
        self.w = np.zeros((self.n_kc, 2), dtype=np.float64)  # Short-term filtered weight deviation

        # Eligibility traces
        self.y_kc = np.zeros(self.n_kc, dtype=np.float64)
        # Traces for DANs: index 0 = PAM (reward), index 1 = PPL1 (punishment)
        self.y_dan_pam = np.zeros(1, dtype=np.float64)
        self.y_dan_ppl1 = np.zeros(1, dtype=np.float64)

        # Gain matrices for Huang, Luo et al. 2024 rule
        # PAM dopamine depresses avoidance MBON synapses (s_idx = 1)
        # PPL1 dopamine depresses approach MBON synapses (s_idx = 0)
        self.gain_pam = np.array([[1.0]])
        self.gain_ppl1 = np.array([[1.0]])

        # Track history
        self.step_count = 0

    def reset_transients(self):
        """Clear transient eligibility traces between distinct trials (ITI)."""
        self.y_kc.fill(0.0)
        self.y_dan_pam.fill(0.0)
        self.y_dan_ppl1.fill(0.0)

    def reset_state(self, keep_memory: bool = False):
        """Reset transient dynamics and optionally clear memory."""
        self.reset_transients()
        self.step_count = 0
        if not keep_memory:
            self.u.fill(0.0)
            self.w.fill(0.0)

    def encode_odor(self, odor_a: float, odor_b: float) -> Tuple[np.ndarray, np.ndarray]:
        """Convert odor concentrations to PN rates and sparse KC firing rates."""
        # Odor concentrations drive PNs
        pn_activity = odor_a * self.pn_tuning[0] + odor_b * self.pn_tuning[1]
        pn_hz = np.clip(pn_activity * 50.0, 0.0, 60.0)  # Max 60 Hz

        # Kenyon cells integrate PN inputs with non-linear thresholding
        kc_current = (pn_hz / 50.0) @ self.w_pn_kc  # range [0, 1]
        kc_active = kc_current > self.kc_threshold
        # Rectified firing: 0 if subthreshold, up to 35 Hz if active
        kc_hz = np.zeros(self.n_kc, dtype=np.float64)
        kc_hz[kc_active] = 10.0 + 25.0 * (kc_current[kc_active] - self.kc_threshold) / (1.0 - self.kc_threshold)
        return pn_hz, kc_hz

    def get_effective_weights(self) -> np.ndarray:
        """Effective synaptic weights KC -> MBON [Approach, Avoidance]."""
        return self.w_kc_mbon_baseline * (1.0 + self.w)

    def forward(self, kc_hz: np.ndarray) -> Tuple[float, float, float]:
        """Compute MBON approach, MBON avoidance, and net valence."""
        eff_weights = self.get_effective_weights()
        # Readout firing rates (Hz)
        mbon_approach = float(np.sum(kc_hz * eff_weights[:, 0]) / max(np.sum(kc_hz), 1.0))
        mbon_avoidance = float(np.sum(kc_hz * eff_weights[:, 1]) / max(np.sum(kc_hz), 1.0))
        # Net valence: positive indicates approach attraction, negative indicates avoidance
        net_valence = (mbon_approach - mbon_avoidance)
        return mbon_approach, mbon_avoidance, net_valence

    def step(
        self,
        odor_a: float,
        odor_b: float,
        reward: float = 0.0,
        punishment: float = 0.0,
        dt_seconds: float = 0.01,
        learning: bool = True
    ) -> Dict:
        """Run one simulation time-bin (e.g. 10 ms)."""
        dt = min(dt_seconds, 0.01)  # Max 10ms per bin as required by rule
        pn_hz, kc_hz = self.encode_odor(odor_a, odor_b)
        mbon_approach, mbon_avoidance, net_valence = self.forward(kc_hz)

        # Dopaminergic neuron firing: baseline ~2 Hz + evoked phasic burst up to 40 Hz
        dan_pam_hz = np.array([np.clip(reward, 0.0, 1.0) * 40.0])
        dan_ppl1_hz = np.array([np.clip(punishment, 0.0, 1.0) * 40.0])

        if learning:
            # PAM dopamine depresses avoidance synapses (column 1)
            rule_advance(
                self.y_kc,
                self.y_dan_pam,
                self.u[:, 1],
                self.w[:, 1],
                kc_hz,
                dan_pam_hz,
                self.gain_pam,
                dt,
                self.eta,
                learning=True
            )

            # PPL1 dopamine depresses approach synapses (column 0)
            rule_advance(
                self.y_kc,
                self.y_dan_ppl1,
                self.u[:, 0],
                self.w[:, 0],
                kc_hz,
                dan_ppl1_hz,
                self.gain_ppl1,
                dt,
                self.eta,
                learning=True
            )

        self.step_count += 1
        active_kcs = np.flatnonzero(kc_hz > 0)

        return {
            'step': self.step_count,
            'net_behavior': net_valence,
            'approach_bias': mbon_approach,
            'avoidance_bias': mbon_avoidance,
            'active_kc_count': len(active_kcs),
            'active_kc_fraction': float(len(active_kcs) / self.n_kc),
            'pam_firing': float(dan_pam_hz[0]),
            'ppl1_firing': float(dan_ppl1_hz[0]),
            'mean_avoidance_weight': float(np.mean(self.get_effective_weights()[:, 1])),
            'mean_approach_weight': float(np.mean(self.get_effective_weights()[:, 0]))
        }

    def get_connectivity_stats(self) -> Dict:
        """Return connectivity statistics."""
        return {
            'n_pn': self.n_pn,
            'n_kc': self.n_kc,
            'pn_per_kc': self.pn_per_kc,
            'sparsity': float(self.pn_per_kc / self.n_pn),
            'total_kc_mbon_synapses': self.n_kc * 2
        }


class MushroomBodySimulator:
    """High-level experiment runner and behavioral assay for MushroomBodyCircuit."""

    def __init__(self, seed: int = 42):
        self.circuit = MushroomBodyCircuit(seed=seed)
        self.seed = seed

    def reset(self):
        self.circuit.reset_state(keep_memory=False)

    def run_learning_trial(
        self,
        odor_a: float,
        odor_b: float,
        reward: bool = False,
        punishment: bool = False,
        steps: int = 20,
        dt: float = 0.01,
        clear_transients: bool = True
    ) -> List[Dict]:
        """Run a structured learning trial."""
        if clear_transients:
            self.circuit.reset_transients()
        states = []
        # Pre-exposure CS (odor alone for half the steps)
        cs_steps = steps // 2
        for _ in range(cs_steps):
            st = self.circuit.step(odor_a, odor_b, reward=0.0, punishment=0.0, dt_seconds=dt, learning=True)
            states.append(st)

        # US pairing (odor + reward / punishment for remaining steps)
        us_reward = 1.0 if reward else 0.0
        us_punishment = 1.0 if punishment else 0.0
        for _ in range(steps - cs_steps):
            st = self.circuit.step(odor_a, odor_b, reward=us_reward, punishment=us_punishment, dt_seconds=dt, learning=True)
            states.append(st)

        if clear_transients:
            self.circuit.reset_transients()
        return states

    def run_behavioral_assay(
        self,
        odor_stimulus: Tuple[float, float],
        num_trials: int = 50,
        temperature: float = 0.2
    ) -> Dict:
        """Evaluate behavioral choice preference without learning/reinforcement."""
        odor_a, odor_b = odor_stimulus
        choices = []
        for _ in range(num_trials):
            _, kc_hz = self.circuit.encode_odor(odor_a, odor_b)
            _, _, net_valence = self.circuit.forward(kc_hz)
            p_approach = 1.0 / (1.0 + math.exp(-np.clip(net_valence / temperature, -20.0, 20.0)))
            action = int(self.circuit.rng.random() < p_approach)  # 1 = approach, 0 = avoid
            choices.append(action)

        approach_rate = float(np.mean(choices))
        return {
            'trials': num_trials,
            'approach_rate': approach_rate,
            'avoidance_rate': 1.0 - approach_rate,
            'mean_net_valence': float(net_valence)
        }


def demonstrate_basic_learning():
    """Demonstrate basic olfactory associative learning and print report."""
    print("================================================================")
    print("   FlyBrain: Drosophila Mushroom Body Olfactory Learning Demo   ")
    print("   Governed by Huang, Luo et al. Nature 2024 Plasticity Rule   ")
    print("================================================================")

    simulator = MushroomBodySimulator(seed=42)
    stats = simulator.circuit.get_connectivity_stats()
    print(f"Circuit configuration: {stats['n_kc']} KCs, {stats['n_pn']} PNs, Sparsity: {stats['sparsity']:.1%}")

    # 1. Naive behavioral baseline for Odor A (Food)
    naive_assay = simulator.run_behavioral_assay((0.8, 0.0), num_trials=100)
    print(f"\n[Naive Baseline - Food Odor A]")
    print(f"  Net Valence:   {naive_assay['mean_net_valence']:+.4f}")
    print(f"  Approach Rate: {naive_assay['approach_rate']:.1%}")

    # 2. Conditioning: Pair Odor A with Food Reward (PAM Dopamine)
    print(f"\n[Training Phase] Conditioning Odor A with PAM Reward (4 trials)...")
    for t in range(4):
        trial_states = simulator.run_learning_trial(0.8, 0.0, reward=True, punishment=False, steps=80)
        print(f"  Trial {t+1}: end net valence = {trial_states[-1]['net_behavior']:+.4f}, "
              f"avoidance weight = {trial_states[-1]['mean_avoidance_weight']:.4f}")

    # 3. Post-training assay on Odor A
    trained_assay = simulator.run_behavioral_assay((0.8, 0.0), num_trials=100)
    print(f"\n[Trained Response - Food Odor A]")
    print(f"  Net Valence:   {trained_assay['mean_net_valence']:+.4f}")
    print(f"  Approach Rate: {trained_assay['approach_rate']:.1%}")

    # 4. Conditioning: Pair Odor B (Hazard) with PPL1 Punishment
    print(f"\n[Training Phase] Conditioning Odor B with PPL1 Punishment (4 trials)...")
    for t in range(4):
        trial_states = simulator.run_learning_trial(0.0, 0.8, reward=False, punishment=True, steps=80)
        print(f"  Trial {t+1}: end net valence = {trial_states[-1]['net_behavior']:+.4f}, "
              f"approach weight = {trial_states[-1]['mean_approach_weight']:.4f}")

    hazard_assay = simulator.run_behavioral_assay((0.0, 0.8), num_trials=100)
    print(f"\n[Trained Response - Hazard Odor B]")
    print(f"  Net Valence:   {hazard_assay['mean_net_valence']:+.4f}")
    print(f"  Approach Rate: {hazard_assay['approach_rate']:.1%}")
    print(f"  Avoidance Rate:{hazard_assay['avoidance_rate']:.1%}")

    return simulator


if __name__ == "__main__":
    demonstrate_basic_learning()
