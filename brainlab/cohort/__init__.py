"""Batched multi-fly ("cohort") execution of the fixed v3 connectome.

The interface contract lives in :mod:`brainlab.cohort.api`.
"""
from .api import CohortEngine, CpuLoopCohortEngine, COHORT_SCHEMA

__all__ = ["CohortEngine", "CpuLoopCohortEngine", "COHORT_SCHEMA"]
