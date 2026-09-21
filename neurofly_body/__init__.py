"""Embodied MaleCNS-to-FlyGym co-simulation.

This package is deliberately independent of the dashboard and assay runtime.  Its
public seam is small enough to exercise with fake graph/body implementations while
the production CLI uses the verified MaleCNS graph and FlyGym 2.1 directly.
"""

from .decoder import DNa02CPGDecoder
from .runner import EmbodiedConfig, run_embodied

__all__ = ["DNa02CPGDecoder", "EmbodiedConfig", "run_embodied"]
__version__ = "0.1.0"
