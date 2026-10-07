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
  are directly comparable. Checkpoints may be restored across engines ONLY
  once the numeric restore proof (cohort verify) passes for that pair.
* Rejection is atomic: ``step`` validates the whole (B, n) drive and an
  integer tick count before any fly advances; ``write_state`` validates every
  field (keys, shapes, dtypes, finiteness, clocks, queue bounds) before any
  live array changes. The shared graph arrays are made read-only.
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
        if isinstance(n_flies, bool) or int(n_flies) != n_flies or n_flies < 1:
            raise ValueError("n_flies must be an integer >= 1")
        # The graph is shared by every fly: freeze it so no fly can mutate it.
        for value in arrays.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        self.arrays = arrays
        self.n_flies = int(n_flies)
        self.n = int(len(arrays["ptr"]) - 1)

    # --- validation shared by every engine: check everything, then mutate ---
    def _check_step_args(self, drive, ticks) -> np.ndarray:
        if isinstance(ticks, bool) or not isinstance(ticks, (int, np.integer)) or int(ticks) < 1:
            raise ValueError("ticks must be a positive integer number of 0.1 ms ticks")
        drive = np.ascontiguousarray(drive, dtype=np.float32)
        if drive.shape != (self.n_flies, self.n):
            raise ValueError(f"drive must have shape {(self.n_flies, self.n)}")
        if not np.isfinite(drive).all():
            raise ValueError("drive must be finite for every fly and neuron; no fly was advanced")
        return drive

    def _check_state(self, fly: int, state: Dict[str, object], template: Dict[str, object]) -> Dict[str, object]:
        """Validate a whole state dict against ``template`` (a live snapshot)
        and return staged copies. Raises before anything live is touched."""
        if isinstance(fly, bool) or not (0 <= int(fly) < self.n_flies):
            raise ValueError(f"fly must be in [0, {self.n_flies})")
        if state.get("dynamics") != "v3":
            raise ValueError("Cohort states are fixed v3 only; refused, not reinterpreted")
        missing = [k for k in template if k not in state]
        if missing:
            raise ValueError(f"State is missing required fields {sorted(missing)}; refused")
        staged = {}
        for key, ref in template.items():
            value = state[key]
            if isinstance(ref, np.ndarray):
                arr = np.asarray(value)
                if arr.shape != ref.shape or arr.dtype != ref.dtype:
                    raise ValueError(f"State array {key} {arr.shape}/{arr.dtype} does not fit "
                                     f"{ref.shape}/{ref.dtype}; refused")
                if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                    raise ValueError(f"State array {key} has non-finite values; refused")
                staged[key] = arr.copy()
            else:
                staged[key] = value
        for key in ("cursor", "total_spikes"):
            v = staged.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
                raise ValueError(f"State field {key} must be a non-negative integer; refused")
        if not np.isfinite(float(staged.get("sim_ms", 0.0))) or float(staged["sim_ms"]) < 0:
            raise ValueError("State field sim_ms must be finite and >= 0; refused")
        if "queue_count" in staged and "queue" in staged:
            qc, q = np.asarray(staged["queue_count"]), np.asarray(staged["queue"])
            if qc.size and (qc.min() < 0 or qc.max() > q.shape[-1]):
                raise ValueError("State queue_count is outside the queue bounds; refused")
        return staged

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
        from brainlab import engine as _e
        return {"schema": COHORT_SCHEMA, "backend": self.backend_id,
                "constants": {"V_REST_MV": _e.V_REST_MV, "V_RESET_MV": _e.V_RESET_MV,
                              "V_THRESHOLD_MV": _e.V_THRESHOLD_MV, "TAU_M_MS": _e.TAU_M_MS,
                              "TAU_SYN_MS": _e.TAU_SYN_MS, "REFRACTORY_MS": _e.REFRACTORY_MS,
                              "DELAY_MS": _e.DELAY_MS, "E_EXC_MV": _e.E_EXC_MV,
                              "E_INH_MV": _e.E_INH_MV},
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
        drive = self._check_step_args(drive, ticks)
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
        staged = self._check_state(fly, state, self.brains[int(fly)].snapshot_state())
        self.brains[int(fly)].restore_state(staged)
