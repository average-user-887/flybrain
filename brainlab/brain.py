"""Direct current input and spike output; no sensory or motor policy."""
import math
import time
import numpy as np
from .engine import advance


class Brain:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as graph:
            for name, dtype in [('ptr', np.int64), ('post', np.int32),
                                ('weight', np.float32), ('ids', np.int64)]:
                value = graph[name]
                if value.ndim != 1 or value.dtype != dtype or not value.flags.c_contiguous:
                    raise ValueError(f'Invalid graph array: {name}')
                setattr(self, name, value)
        self.n = len(self.ids)
        if (not self.n or self.ptr.shape != (self.n + 1,) or self.ptr[0] != 0
                or self.ptr[-1] != len(self.post) or np.any(np.diff(self.ptr) < 0)
                or len(self.weight) != len(self.post) or not np.isfinite(self.weight).all()
                or np.any(self.post < 0) or np.any(self.post >= self.n)
                or len(np.unique(self.ids)) != self.n):
            raise ValueError('Invalid CSR graph')
        self.dt = .1
        self.cursor = 0
        self.v = np.full(self.n, -52, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.queue = np.zeros((19, self.n), dtype=np.int32)
        self.queue_count = np.zeros(19, dtype=np.int32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        self.total_spikes = 0
        self.sim_ms = 0.

    def step(self, currents, duration_ms):
        """Apply one finite current per neuron (upstream mV-equivalent units)."""
        drive = np.ascontiguousarray(currents, dtype=np.float32)
        if drive.shape != (self.n,) or not np.isfinite(drive).all():
            raise ValueError('Provide a finite current for every neuron')
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Duration must be finite and positive')
        steps = round(duration_ms / self.dt)
        if steps < 1 or not math.isclose(steps*self.dt, duration_ms, abs_tol=1e-9):
            raise ValueError('Duration must be a multiple of 0.1 ms')
        newly_active = np.flatnonzero((drive != 0) & (self.active_flag == 0))
        start = int(self.nactive[0])
        self.active[start:start+len(newly_active)] = newly_active
        self.active_flag[newly_active] = 1
        self.nactive[0] += len(newly_active)
        self.counts.fill(0)
        clock = time.perf_counter()
        self.cursor = advance(self.ptr, self.post, self.weight, self.v, self.g,
            self.refractory, drive, self.queue, self.queue_count, self.cursor,
            steps, self.dt, self.counts, self.active, self.active_flag, self.nactive)
        elapsed = time.perf_counter()-clock
        self.total_spikes += int(self.counts.sum())
        self.sim_ms += steps*self.dt
        return self.counts.copy(), elapsed
