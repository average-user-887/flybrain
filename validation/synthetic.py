"""A small, conspicuously labelled synthetic graph WITH cell types, for tests.

It contains every cell type the three flagship specs resolve, a few planted
pathways so each paradigm's encoder -> graph -> decoder path carries spikes,
and a sparse random background.  It exercises the harness code path only.
Receipts from it are ``SYNTHETIC_PLUMBING_ONLY`` and make no claim.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from brainlab.graph_identity import SYNTHETIC_LABEL, GraphIdentity, _ids_digest, sha256_json

from .cells import CellTable

# (type, per-side count); side None = unlateralised.
TYPES = [
    ('T4a', 12), ('T5a', 12), ('T4b', 12), ('T5b', 12), ('HSE', 1), ('HSN', 1), ('HSS', 1), ('H2', 1),
    ('VS', 2), ('DNa02', 1), ('DNa01', 1), ('LC4', 10), ('LPLC2', 10), ('DNp01', 1),
    ('ORN_VA3', 8), ('ORN_VM5d', 8), ('ORN_D', 8), ('ORN_DM2', 8),
    ('VA3_lPN', 2), ('VM5d_adPN', 2), ('D_adPN', 2), ('DM2_lPN', 2), ('KCg-m', 20), ('MBON01', 1),
]
BACKGROUND = 200


def typed_synthetic_graph(seed: int = 0):
    from experiment_registry import SharedGraph
    rng = np.random.default_rng(seed)
    rows = []
    for ctype, count in TYPES:
        for side in ('L', 'R'):
            rows += [(ctype, side)] * count
    rows += [('background', None)] * BACKGROUND
    n = len(rows)
    frame = pd.DataFrame(dict(node_index=np.arange(n, dtype=np.int64),
                              source_id=np.arange(1000, 1000 + n, dtype=np.int64),
                              cell_type=[r[0] for r in rows], side=[r[1] for r in rows],
                              side_source=['synthetic' if r[1] else None for r in rows]))
    cells = CellTable(frame, source='synthetic', synthetic=True)

    def ids(ctype, side=None, pattern=None):
        return cells.select((ctype,) if ctype else (), side, pattern=pattern).node_index.to_numpy()

    edges = []

    def connect(pre, post, w):
        for i in pre:
            for j in post:
                edges.append((int(i), int(j), float(w)))
    for eye, other in (('L', 'R'), ('R', 'L')):
        # Front-to-back on eye X -> HS_X -> DNa02_X; back-to-front on eye X -> H2_X -> DNa02 of the other side.
        hs = np.concatenate([ids('HSE', eye), ids('HSN', eye), ids('HSS', eye)])
        connect(np.concatenate([ids('T4a', eye), ids('T5a', eye)]), hs, 2.0)
        connect(np.concatenate([ids('T4b', eye), ids('T5b', eye)]), ids('H2', eye), 2.0)
        connect(hs, ids('DNa02', eye), 12.0)
        connect(ids('H2', eye), ids('DNa02', other), 12.0)
        connect(np.concatenate([ids('LC4', eye), ids('LPLC2', eye)]), ids('DNp01', eye), 0.6)
        connect(np.concatenate([ids('LC4', eye), ids('LPLC2', eye)]), ids('DNp01', other), 0.6)
        connect(ids('ORN_VA3', eye), ids('VA3_lPN', eye), 3.0)
        connect(ids('VA3_lPN', eye), ids('DNa02', eye), 8.0)
        connect(ids('ORN_VM5d', eye), ids('VM5d_adPN', eye), 3.0)
        connect(ids('VM5d_adPN', eye), ids('DNa02', other), 8.0)
    # Sparse random background, mixed sign, weak.
    for i in range(n):
        for j in rng.integers(0, n, 4):
            edges.append((i, int(j), float(rng.normal(0, 0.5))))
    edges.sort(key=lambda e: (e[0], e[1]))
    pre = np.array([e[0] for e in edges], np.int64)
    ptr = np.searchsorted(pre, np.arange(n + 1), side='left').astype(np.int64)
    post = np.array([e[1] for e in edges], np.int32)
    weight = np.array([e[2] for e in edges], np.float32)
    arrays = dict(ptr=ptr, post=post, weight=weight, ids=frame.source_id.to_numpy(np.int64))
    digest = hashlib.sha256()
    for key in ('ptr', 'post', 'weight', 'ids'):
        digest.update(arrays[key].tobytes())
    identity = GraphIdentity(
        dataset='synthetic-typed-test', synthetic=True, graph_path=None,
        graph_path_source=f'validation.synthetic.typed_synthetic_graph(seed={seed})',
        graph_sha256=digest.hexdigest(), neuron_map_path=None, neuron_map_sha256=None,
        io_map_sha256=sha256_json(dict(types=TYPES, background=BACKGROUND)), neurons=n, edges=len(edges),
        ids_sha256=_ids_digest(arrays['ids']), label=SYNTHETIC_LABEL)
    return SharedGraph(arrays, identity, {}), cells
