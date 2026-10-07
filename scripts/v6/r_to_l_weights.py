"""Summed R1-R6 -> post inhibitory weight per postsynaptic node, from the engine's
v4 policy weights (read-only).  python scripts/v6/r_to_l_weights.py OUT.npz"""
import hashlib, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from brainlab.brain import Brain
from brainlab.graph_identity import resolve_graph_dir
from brainlab.photoreceptor_io import resolve_photoreceptor_io

gdir, _ = resolve_graph_dir(None)
b = Brain(str(gdir / 'graph.npz'), dynamics='v4', backend='cpu')
io = resolve_photoreceptor_io()
acc = np.zeros(b.n)
for i in np.concatenate([io.r_nodes['L'], io.r_nodes['R']]):
    s, e = b.ptr[i], b.ptr[i + 1]
    w = b.weight[s:e]
    np.add.at(acc, b.post[s:e][w < 0], -w[w < 0])
nz = np.flatnonzero(acc)
np.savez(sys.argv[1], post=nz, w_from_R=acc[nz],
         graph_sha256=hashlib.sha256((gdir / 'graph.npz').read_bytes()).hexdigest(), io_sha256=io.sha256)
print(len(nz), float(np.median(acc[nz])))
