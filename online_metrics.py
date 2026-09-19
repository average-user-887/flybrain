"""Bounded display histories with cumulative scientific statistics.

Continuous observation must not rescan an ever-growing trajectory every tick.
Evicting display samples does not discard the cumulative distance or mean.
"""
from collections import deque
import math


class ScalarHistory(deque):
    def __init__(self, maxlen=2048):
        super().__init__(maxlen=maxlen)
        self.count = 0
        self.total = 0.0

    def append(self, value):
        self.total += float(value)
        self.count += 1
        super().append(value)

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    def clear(self):
        super().clear()
        self.count = 0
        self.total = 0.0


class PathHistory(deque):
    def __init__(self, maxlen=2048):
        super().__init__(maxlen=maxlen)
        self.origin = None
        self.distance = 0.0
        self.count = 0

    def append(self, point):
        if self:
            self.distance += math.dist(self[-1], point)
        else:
            self.origin = point
        self.count += 1
        super().append(point)

    @property
    def tortuosity(self):
        net = math.dist(self.origin, self[-1]) if self else 0.0
        return self.distance / net if net > 1e-9 else None

    def clear(self):
        super().clear()
        self.origin = None
        self.distance = 0.0
        self.count = 0
