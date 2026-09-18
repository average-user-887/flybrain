"""
Neuromodulatory Metabolic State for Drosophila.
==============================================
Simulates the internal metabolic hunger/satiety drive:
1. Satiety S(t) in [0.0, 1.0] decays continuously as energy is metabolized.
2. Feeding on food sources replenishes satiety.
3. Neuromodulation:
   - Starvation (H = 1 - S) upregulates octopamine / PAM dopamine sensitivity,
     amplifying appetitive learning rates.
   - Starvation suppresses Markovian stopping hazard (hunger-driven persistence).
   - Satiety modulates risk-taking vs predator avoidance trade-offs.
"""

import math
from typing import Dict


class MetabolicState:
    def __init__(
        self,
        initial_satiety: float = 0.75,
        decay_rate: float = 0.002,   # Depletion per second (~500s full starvation)
        feed_boost: float = 0.35,    # Satiety boost per full food encounter
        starvation_threshold: float = 0.25,
        satiated_threshold: float = 0.85
    ):
        self.satiety = float(min(1.0, max(0.0, initial_satiety)))
        self.decay_rate = decay_rate
        self.feed_boost = feed_boost
        self.starvation_threshold = starvation_threshold
        self.satiated_threshold = satiated_threshold
        self.total_food_consumed = 0
        self.starvation_time = 0.0

    def reset(self, satiety: float = 0.75):
        self.satiety = float(min(1.0, max(0.0, satiety)))
        self.total_food_consumed = 0
        self.starvation_time = 0.0

    def step(self, dt: float = 0.02, speed: float = 1.2) -> Dict:
        """
        Advance metabolic consumption. Higher flight speed consumes energy faster.
        """
        # Basal metabolic rate + locomotion expenditure
        speed_factor = 1.0 + 0.5 * (speed / 15.0)
        burn = self.decay_rate * speed_factor * dt
        self.satiety = max(0.0, self.satiety - burn)

        if self.is_starving:
            self.starvation_time += dt
        else:
            self.starvation_time = 0.0

        return {
            "satiety": self.satiety,
            "hunger": self.hunger,
            "is_starving": self.is_starving,
            "is_satiated": self.is_satiated
        }

    def feed(self, amount: float = None) -> float:
        """Ingest food and restore satiety."""
        boost = self.feed_boost if amount is None else amount
        self.satiety = min(1.0, self.satiety + boost)
        self.total_food_consumed += 1
        return self.satiety

    @property
    def hunger(self) -> float:
        """Hunger drive in [0.0, 1.0]."""
        return 1.0 - self.satiety

    @property
    def is_starving(self) -> bool:
        return self.satiety <= self.starvation_threshold

    @property
    def is_satiated(self) -> bool:
        return self.satiety >= self.satiated_threshold

    def get_dopamine_gain(self) -> float:
        """
        Hunger scales PAM dopamine learning rate.
        Starving flies learn appetitive odors up to 2.5x faster.
        """
        return 1.0 + 1.5 * self.hunger

    def get_surge_multiplier(self) -> float:
        """Hunger elevates forward upwind surging vigor."""
        return 1.0 + 0.35 * self.hunger

    def get_stop_suppression(self) -> float:
        """
        Hungry flies suppress spontaneous Markovian stops.
        Returns factor in [0.15, 1.0] scaling the stop transition rate r_S.
        """
        return max(0.15, self.satiety)

    def get_predator_fear_weight(self) -> float:
        """
        Satiated flies prioritize safety (fear weight ~1.0).
        Starving flies take riskier paths near food (fear weight down to 0.4).
        """
        return 0.4 + 0.6 * self.satiety
