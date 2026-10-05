"""Optomotor direction check on the live daemon code path: +30, -30 and 0 deg/s drum."""
import sys, os, json
from pathlib import Path
import numpy as np
W = Path(os.environ["W"]); sys.path.insert(0, str(W)); os.chdir(W)
import neurofly_daemon as nd
OUT = Path("<SCRATCH>")
res = {}
for v in (30.0, -30.0, 0.0):
    r = nd.ContinuousExperimentRunner(initial_paradigm="optomotor", backend="connectome-fixed",
                                      graph_dir=W / "outputs/brainlab/malecns_v1", output_dir=OUT / f"opto{v}",
                                      registry_root=OUT / f"optoreg{v}", continuous=True)
    r.arena.paradigm.drum_velocity_deg_s = v
    yaws, dl, dr = [], [], []
    for i in range(400):
        with r.lock:
            r.step_once(publish=False)
        t = r.arena.fly.last_connectome_telemetry or {}
        o = t.get("optomotor") or {}
        if i >= 100:
            yaws.append(float(o.get("yaw_rad_s", 0.0))); dl.append(o.get("rate_l", 0)); dr.append(o.get("rate_r", 0))
    m = r.arena.paradigm.get_metrics()
    res[v] = {"mean_yaw_rad_s": float(np.mean(yaws)), "mean_yaw_deg_s": float(np.degrees(np.mean(yaws))),
              "rate_l": float(np.mean(dl)), "rate_r": float(np.mean(dr)), "gain_metric": m.get("optomotor_gain")}
    print(v, json.dumps(res[v]), flush=True)
(OUT / "opto.json").write_text(json.dumps(res, indent=1))
