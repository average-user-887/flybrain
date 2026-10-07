"""Interface contract for cohort engines: many independent fixed-v3 brains
sharing one read-only graph.

Semantics every engine must honour (v0.5.0-rc1 contract):

* Dynamics are LIF **v3 only**, with fixed (non-plastic) weights. Any other
  dynamics or a plastic graph is refused at construction.
* The graph (ptr, post, weight, io map) is loaded once and shared, read-only,
  by all flies. Per-fly state is never shared.
* ``step`` advances every fly by the same number of 0.1 ms ticks. Fly ``k``
  receives only ``drive[k]``. Its result must not depend on the other flies,
  their number or their order.
* State dicts use the same keys, dtypes and shapes as
  ``Brain.snapshot_state()`` for v3, so the CPU reference and the GPU engine
  are directly comparable and checkpoints are interchangeable between engines.
"""
from __future__ import annotations

import abc
from typing import Dict, List, Sequence

import numpy as np

COHORT_SCHEMA = "neurofly.cohort.v1"
TICK_MS = 0.1


class CohortEngine(abc.ABC):
    """B independent fixed-v3 brains over one shared, read-only graph."""

    backend_id: str = "abstract"

    def __init__(self, arrays: dict, n_flies: int):
        if n_flies < 1:
            raise ValueError("n_flies must be >= 1")
        self.arrays = arrays
        self.n_flies = int(n_flies)
        self.n = int(len(arrays["ptr"]) - 1)

    @abc.abstractmethod
    def reset(self, fly_ids: Sequence[int]) -> None:
        """Put the listed flies at rest (v = -52 mV, empty conductances,
        refractory, queue and clocks), exactly like a freshly built Brain."""

    @abc.abstractmethod
    def step(self, drive: np.ndarray, ticks: int) -> np.ndarray:
        """Advance all flies by ``ticks`` 0.1 ms ticks.

        ``drive``: float32 array, shape (n_flies, n), finite. It is held
        constant over the ticks, exactly as ``Brain.step(currents, ticks*0.1)``.
        Returns spike counts, shape (n_flies, n), dtype matching
        ``Brain.counts``, for this call only.
        """

    @abc.abstractmethod
    def read_state(self, fly: int) -> Dict[str, object]:
        """Host copy of fly ``fly``'s state, in ``Brain.snapshot_state()`` format."""

    @abc.abstractmethod
    def write_state(self, fly: int, state: Dict[str, object]) -> None:
        """Install a state dict for one fly. Refuse anything that is not v3
        or has a wrong key, shape or dtype; never reinterpret."""

    def describe(self) -> dict:
        return {"schema": COHORT_SCHEMA, "backend": self.backend_id,
                "n_flies": self.n_flies, "n_neurons": self.n, "dynamics": "v3",
                "plastic": False, "tick_ms": TICK_MS}


class CpuLoopCohortEngine(CohortEngine):
    """Reference engine: one CPU ``Brain`` per fly, sharing the graph arrays.

    This is the numerical reference for the GPU engine and lets the runner
    (CLI, recording, checkpoints) work without a GPU.
    """

    backend_id = "cpu-loop-v3"

    def __init__(self, arrays: dict, n_flies: int):
        super().__init__(arrays, n_flies)
        from brainlab.brain import Brain
        self.brains: List = []
        for k in range(self.n_flies):
            self.brains.append(Brain(arrays=arrays, validate=(k == 0),
                                     dynamics="v3", backend="cpu"))

    def reset(self, fly_ids):
        for k in fly_ids:
            self.brains[k].reset_state()

    def step(self, drive, ticks):
        drive = np.ascontiguousarray(drive, dtype=np.float32)
        if drive.shape != (self.n_flies, self.n):
            raise ValueError(f"drive must have shape {(self.n_flies, self.n)}")
        if int(ticks) < 1:
            raise ValueError("ticks must be >= 1")
        out = None
        for k, brain in enumerate(self.brains):
            counts, _ = brain.step(drive[k], int(ticks) * TICK_MS)
            if out is None:
                out = np.zeros((self.n_flies, self.n), dtype=counts.dtype)
            out[k] = counts
        return out

    def read_state(self, fly):
        return self.brains[fly].snapshot_state()

    def write_state(self, fly, state):
        if state.get("dynamics") != "v3":
            raise ValueError("Cohort checkpoints are fixed v3 only; refused, not reinterpreted")
        self.brains[fly].restore_state(state)
