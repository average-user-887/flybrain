#!/usr/bin/env python3
"""TEST HARNESS: the daemon with a STUB WP5 optomotor IO map on the synthetic graph.

``scripts/browser_firefox_check.py`` uses this to exercise the dashboard's optomotor
provenance (assistance gate, IO-map hash, the 1x-unattainable and no-forward-drive
banner notes) WITHOUT loading the 600 MB prepared graph.

What it is not: the stub map is a handful of node indices inside the 64-neuron
synthetic test graph and its hash is :data:`STUB_IO_MAP_SHA256`, never the pinned
WP5 map (``brainlab.io_map.OPTOMOTOR_IO_PIN``).  A run started this way is a
synthetic test-mode run: it refuses to start on anything but a synthetic graph, so
its identity, telemetry and dashboard banner are labelled SYNTHETIC TEST GRAPH and
it can never be read as a scientific result.

    <venv>/bin/python scripts/browser_optomotor_stub_daemon.py \\
        --port 19600 --paradigm optomotor --backend connectome-fixed --test-synthetic-graph
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import neurofly_daemon
from brainlab.io_map import OptomotorIOMap   # read-only use of the WP5 public API

STUB_IO_MAP_SHA256 = 'stub-browser-check-optomotor-map-not-the-wp5-pin'


def stub_io_map() -> OptomotorIOMap:
    """A WP5-shaped map over nodes of the synthetic test graph. Never the real pin."""
    populations = {name: np.arange(20 + index * 5, 25 + index * 5, dtype=np.int64)
                   for index, name in enumerate(('ftb_L', 'btf_L', 'ftb_R', 'btf_R'))}
    populations['DNa02_L'] = np.array([10], np.int64)
    populations['DNa02_R'] = np.array([11], np.int64)
    return OptomotorIOMap(populations=populations,
                          source_ids={name: [int(i) for i in nodes] for name, nodes in populations.items()},
                          matched_counts={'T4': 5, 'T5': 0}, available_counts={},
                          monitors={'HS_L': np.array([1], np.int64), 'HS_R': np.array([2], np.int64)},
                          sha256=STUB_IO_MAP_SHA256)


_original_init = neurofly_daemon.GraphArenaController.__init__


def _init_with_stub_map(self, runner, step_ms):
    _original_init(self, runner, step_ms)
    identity = getattr(getattr(runner, 'shared_graph', None), 'identity', None)
    if not getattr(identity, 'synthetic', False):
        raise SystemExit('browser_optomotor_stub_daemon: the stub map is only ever installed on a '
                         'synthetic test graph; start with --test-synthetic-graph')
    self._optomotor_io = stub_io_map()


neurofly_daemon.GraphArenaController.__init__ = _init_with_stub_map


if __name__ == '__main__':
    print(f'[stub-daemon] STUB optomotor IO map {STUB_IO_MAP_SHA256} on the synthetic test graph; '
          'not a scientific result.', flush=True)
    neurofly_daemon.run_daemon()
