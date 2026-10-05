"""PR #25 review: on the REAL MaleCNS graph, the WP7 options at their defaults give the
same weights and graph identity as master's transmitter policy, and the non-default
variants touch exactly the edge counts the spec states (no simulation is run)."""
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

from brainlab import transmitter_policy as tp
from experiment_registry import SharedGraph

spec = importlib.util.spec_from_file_location('brainlab._tp_master', '<scratch>/tp_master.py')
tpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpm)

t0 = time.time()
base = SharedGraph.load()
out = {'base_graph_sha256': base.identity.graph_sha256, 'neurons': int(len(base.arrays['ids'])),
       'edges': int(len(base.arrays['post']))}
master_shared, master_report = tpm.apply_to_shared(base)
pr_shared, pr_report = tp.apply_to_shared(base)
out['default'] = {
    'master_graph_sha256': master_shared.identity.graph_sha256,
    'pr_graph_sha256': pr_shared.identity.graph_sha256,
    'identity_equal': master_shared.identity.to_dict() == pr_shared.identity.to_dict(),
    'weights_byte_identical': bool(np.array_equal(master_shared.arrays['weight'], pr_shared.arrays['weight'])
                                   and master_shared.arrays['weight'].tobytes() == pr_shared.arrays['weight'].tobytes()),
    'report_equal': master_report == pr_report,
    'describe_equal': tpm.describe() == tp.describe(),
    'dataset': pr_shared.identity.dataset,
}
del master_shared
types = tp.load_cell_types()
labels = tp.load_transmitters()
dpm = tp._type_mask(types, tp.DPM_TYPE_PATTERN)
out['dpm_released_labels'] = sorted(set(labels[dpm].astype(str)))
variants = {'V2_primary': dict(kc_kc='modulatory-only', dpm='gaba'),
            'V1_kc0_dpm_silent': dict(kc_kc='modulatory-only', dpm='released'),
            'V3_kckc_released_dpm_gaba': dict(kc_kc='released', dpm='gaba')}
out['variants'] = {}
for name, opts in variants.items():
    g, r = tp.apply_to_shared(base, cell_types=types, **opts)
    keep = {k: r[k] for k in ('kc_neurons', 'kc_kc_edges', 'kc_kc_edges_zeroed', 'dpm_neurons',
                              'dpm_released_labels', 'dpm_out_edges_relabelled', 'edges_zeroed') if k in r}
    out['variants'][name] = dict(graph_sha256=g.identity.graph_sha256, dataset=g.identity.dataset,
                                 ptr_post_unchanged=bool(g.arrays['ptr'] is base.arrays['ptr'] or
                                                         np.array_equal(g.arrays['ptr'], base.arrays['ptr'])),
                                 **keep)
out['spec_expectations'] = {'kc_neurons': 4064, 'kc_kc_edges': 642933, 'dpm_neurons': 2}
out['wall_s'] = round(time.time() - t0, 1)
json.dump(out, sys.stdout, indent=1, default=str)
print()
