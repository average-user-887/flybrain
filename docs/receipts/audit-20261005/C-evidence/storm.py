"""Does the high-activity state outlive its stimulus, and is it stimulus-specific?"""
import sys, os, json
from pathlib import Path
import numpy as np
W = Path(os.environ["W"]); sys.path.insert(0, str(W)); os.chdir(W)
import neurofly_daemon as nd
OUT = Path("<SCRATCH>")
r = nd.ContinuousExperimentRunner(initial_paradigm="y-maze", backend="connectome-fixed",
                                  graph_dir=W / "outputs/brainlab/malecns_v1", output_dir=OUT / "storm",
                                  registry_root=OUT / "stormreg", continuous=True)
gc = r.graph_controller
inst = r.registry.active
n = inst.brain.n
out = {}
for label, chan, amp in [("orn_food", "orn_food", 35.0), ("thermo", "thermo_receptors", 30.0),
                         ("cva", "courtship_cva", 19.0), ("looming", "visual_looming", 55.0),
                         ("er_ring", "er_ring", 20.0), ("orn_food_weak", "orn_food", 6.0),
                         ("orn_food_8", "orn_food", 8.0)]:
    inst.brain.reset_state()
    trace = []
    dl = dr = 0
    for k in range(150):          # 1 s on, 2 s off at 20 ms
        c = np.zeros(n, dtype=np.float32)
        if k < 50:
            c[gc.sensory_indices[chan]] = amp
        res = inst.step(c, 20.0)
        cnt = np.asarray(res.counts)
        trace.append(int(cnt.sum()))
        dl += int(cnt[gc.dn_indices["dna02_l"]].sum()); dr += int(cnt[gc.dn_indices["dna02_r"]].sum())
    out[label] = {"on_mean": float(np.mean(trace[5:50])), "off_first_10": trace[50:60],
                  "off_last_mean": float(np.mean(trace[100:150])), "dna02_l": dl, "dna02_r": dr}
    print(label, json.dumps(out[label]), flush=True)
(OUT / "storm.json").write_text(json.dumps(out, indent=1))
