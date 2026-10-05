"""Shared locations for the v4 graded-transmission measurement scripts.

REPO is the repository root, resolved from this file, so the scripts run from
any checkout. The real MaleCNS graph is expected at the usual gitignored
locations under it (``outputs/brainlab/malecns_v1`` and
``connectome_data/malecns_v1``).

SCRATCH holds intermediate measurement JSON (measure_a.json, measure_bc.json,
timing.json, ...). It defaults to the gitignored ``outputs/v4_measurement/``
and can be moved with the ``NEUROFLY_V4_SCRATCH`` environment variable.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRATCH = pathlib.Path(os.environ.get('NEUROFLY_V4_SCRATCH')
                       or REPO / 'outputs' / 'v4_measurement')
SCRATCH.mkdir(parents=True, exist_ok=True)
