"""Brain stepping and spike bookkeeping shared by every paradigm.

The harness only reads spike COUNTS from the brain.  Counts are identical in
layout on the CPU and CUDA backends, so a paradigm never reads the membrane
state (on the GPU the host copy of ``v`` is stale between snapshots) and the
receipt means the same thing on either backend.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from brainlab.brain import Brain


class BrainRunner:
    """One brain on the shared graph, reset to its rest state between trials."""

    def __init__(self, shared, *, dynamics: str, backend: str = 'auto', e_inh_mV: Optional[float] = None):
        kwargs = dict(arrays=shared.arrays, validate=False, dynamics=dynamics, e_inh_mV=e_inh_mV)
        try:
            # 'auto' defers to Brain's own default (NEUROFLY_BRAIN_BACKEND, else GPU when usable).
            self.brain = Brain(**kwargs, backend=None if backend == 'auto' else backend)
        except TypeError:
            # A brainlab without the backend switch (before the CUDA brain landed): CPU only.
            if backend not in ('auto', 'cpu'):
                raise
            self.brain = Brain(**kwargs)
        self.backend = getattr(self.brain, 'backend', 'cpu')
        self.n = self.brain.n
        self._rest = self.brain.snapshot_state()

    def reset(self):
        """Every trial starts from the identical all-at-rest state."""
        self.brain.restore_state(self._rest)

    def step(self, currents: np.ndarray, ms: float) -> np.ndarray:
        counts, _ = self.brain.step(currents, ms)
        return counts

    def describe(self) -> dict:
        return dict(backend=self.backend, dynamics=self.brain.dynamics, dt_ms=self.brain.dt,
                    e_inh_mV=getattr(self.brain, 'e_inh_mV', None), neurons=int(self.n))


class SpikeLedger:
    """Per-window spike totals for every neuron plus named populations.

    Windows are labels such as ``baseline`` and ``stimulus``; ``None`` steps
    are simulated but not counted (e.g. a settle period).
    """

    def __init__(self, n: int, populations: Dict[str, np.ndarray]):
        self.n = n
        self.populations = populations
        self.per_neuron: Dict[str, np.ndarray] = {}
        self.ms: Dict[str, float] = {}

    def add(self, counts: np.ndarray, ms: float, window: Optional[str]):
        if window is None:
            return
        if window not in self.per_neuron:
            self.per_neuron[window] = np.zeros(self.n, np.int64)
            self.ms[window] = 0.0
        self.per_neuron[window] += counts
        self.ms[window] += ms

    def summary(self) -> dict:
        """Hz per cell for each population and whole-brain measures, per window."""
        out = {}
        for window, total in self.per_neuron.items():
            seconds = self.ms[window] / 1000.0
            rates = total / seconds
            pops = {name: dict(mean_hz=float(rates[idx].mean()) if len(idx) else None,
                               max_hz=float(rates[idx].max()) if len(idx) else None, n=int(len(idx)))
                    for name, idx in self.populations.items()}
            out[window] = dict(seconds=seconds, populations=pops,
                               brain=dict(mean_hz=float(rates.mean()), max_hz=float(rates.max()),
                                          argmax_node=int(rates.argmax()),
                                          fraction_active=float((total > 0).mean()),
                                          n_above_300_hz=int((rates > 300).sum())))
        return out
