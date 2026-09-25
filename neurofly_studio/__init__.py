"""Experiment studio (roadmap P5): build, queue, watch and compare embodied runs.

Every run the studio queues is exploratory.  Preregistered validation specs are
listed read-only and are never edited or re-run from here; see
docs/EXPERIMENT_STUDIO.md.
"""

__all__ = ["catalog", "experiment", "metrics", "server"]
