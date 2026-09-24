"""Resolve named neuron populations by cell type and annotated side.

Populations are declared in a spec as ``{"types": [...], "side": "L"|"R"|null,
"limit": n}`` and resolved against a cell table with the columns
``node_index, source_id, cell_type, side``.  Never by row order: members are
sorted by ``source_id`` and ``limit`` keeps the lowest source IDs (the WP5
rule).  Every resolution is hashed so a receipt pins exactly which neurons
were driven and recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from brainlab.graph_identity import GraphUnavailable, resolve_connectome_dir, sha256_json

# Sides are read from the released annotation.  somaSide is the WP5 rule;
# sensory neurons whose somata lie outside the reconstructed volume carry
# their side in rootSide instead (docs/WP5_OPTOMOTOR.md §2).
SIDE_COLUMNS = ('somaSide', 'rootSide')


@dataclass
class CellTable:
    frame: pd.DataFrame          # node_index, source_id, cell_type, side, side_source
    source: str
    synthetic: bool

    @classmethod
    def from_connectome(cls, connectome_dir: Optional[Path] = None) -> 'CellTable':
        import pyarrow.feather as feather
        cdir, _ = resolve_connectome_dir(connectome_dir)
        nodes_path, ann_path = cdir / 'normalized/neurons.feather', cdir / 'annotations.feather'
        for path in (nodes_path, ann_path):
            if not path.is_file():
                raise GraphUnavailable(f'{path} not found; set NEUROFLY_CONNECTOME_DIR')
        nodes = feather.read_table(nodes_path, columns=['node_index', 'source_id', 'cell_type']).to_pandas()
        try:
            import pyarrow.ipc as ipc
            with ipc.open_file(ann_path) as reader:   # feather v2 = Arrow IPC file; schema only
                available = set(reader.schema.names)
        except Exception:   # noqa: BLE001 - feather v1: read the table to learn its columns
            available = set(feather.read_table(ann_path).column_names)
        cols = ['bodyId'] + [c for c in SIDE_COLUMNS if c in available]
        ann = feather.read_table(ann_path, columns=cols).to_pandas().drop_duplicates('bodyId').set_index('bodyId')
        nodes = nodes.join(ann, on='source_id')
        side = pd.Series(np.nan, index=nodes.index, dtype=object)
        side_source = pd.Series(None, index=nodes.index, dtype=object)
        for col in SIDE_COLUMNS:
            if col not in nodes:
                continue
            take = side.isna() & nodes[col].isin(['L', 'R'])
            side[take] = nodes.loc[take, col]
            side_source[take] = col
        frame = pd.DataFrame(dict(node_index=nodes.node_index.astype(np.int64),
                                  source_id=nodes.source_id.astype(np.int64),
                                  cell_type=nodes.cell_type.fillna(''), side=side, side_source=side_source))
        return cls(frame, source=str(cdir), synthetic=False)

    def select(self, types=(), side: Optional[str] = None, limit: Optional[int] = None,
               pattern: Optional[str] = None) -> pd.DataFrame:
        """Members by exact ``types`` and/or a regex ``pattern`` on the cell type."""
        f = self.frame
        keep = f.cell_type.isin(list(types))
        if pattern:
            keep |= f.cell_type.str.match(pattern)
        rows = f[keep]
        if side is not None:
            rows = rows[rows.side == side]
        rows = rows.sort_values('source_id')
        return rows.iloc[:limit] if limit is not None else rows

    def resolve(self, declared: Dict[str, dict], *, allow_empty=()) -> 'Populations':
        nodes, ids, meta = {}, {}, {}
        for name, d in declared.items():
            rows = self.select(d.get('types', ()), d.get('side'), d.get('limit'), d.get('pattern'))
            if not len(rows) and name not in allow_empty:
                raise GraphUnavailable(f'Population {name} ({d}) resolved empty in {self.source}')
            nodes[name] = rows.node_index.to_numpy(np.int64)
            ids[name] = [int(s) for s in rows.source_id]
            meta[name] = dict(declared=d, n=int(len(rows)),
                              side_source=sorted(set(rows.side_source.dropna())) if 'side_source' in rows else [])
        return Populations(nodes=nodes, source_ids=ids, meta=meta,
                           sha256=sha256_json(dict(source_ids=ids, rule='type+side, lowest source_id')))


@dataclass
class Populations:
    nodes: Dict[str, np.ndarray]
    source_ids: Dict[str, list]
    meta: Dict[str, dict]
    sha256: str

    def describe(self) -> dict:
        return dict(sha256=self.sha256, populations=self.meta)
