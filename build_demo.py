"""Build a dependency-free, offline viewer from the downloaded MaleCNS SWCs."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/'
neurons = []
for body, name, color in [('12781', 'DNge104_R', '#70e5c5'), ('556329', 'DNge104_L', '#ffb878')]:
    path = ROOT / 'data' / f'{body}.swc'
    nodes = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        n, _, x, y, z, radius, parent = line.split()
        nodes.append([int(n), float(x)*.008, float(y)*.008, float(z)*.008, int(parent)])
    lookup = {row[0]: i for i, row in enumerate(nodes)}
    assert len(lookup) == len(nodes), 'Duplicate node IDs'
    assert all(row[4] == -1 or row[4] in lookup for row in nodes), 'Missing parent'
    edges = [[i, lookup[row[4]]] for i, row in enumerate(nodes) if row[4] != -1]
    length = sum(math.dist(nodes[a][1:4], nodes[b][1:4]) for a, b in edges)
    neurons.append(dict(id=body, name=name, color=color, points=[n[1:4] for n in nodes], edges=edges,
                        length_um=round(length, 1), source=BASE+path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
template = (ROOT / 'viewer.html').read_text()
(ROOT / 'demo.html').write_text(template.replace('__NEURON_DATA__', json.dumps(neurons, separators=(',', ':'))))
(ROOT / 'data' / 'manifest.json').write_text(json.dumps([{k:v for k,v in n.items() if k not in ('points','edges')} for n in neurons], indent=2)+'\n')
for n in neurons:
    print(f"{n['name']}: {len(n['points']):,} nodes, {len(n['edges']):,} segments, {n['length_um']:,.1f} µm cable")
print(f"Built {ROOT / 'demo.html'}")
