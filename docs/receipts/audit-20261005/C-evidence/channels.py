"""Per-assay channel probe: which sensory channel is driven, how hard, and which cells answer."""
import sys, os, json, time, math
from pathlib import Path
import numpy as np
W = Path(os.environ["W"]); sys.path.insert(0, str(W)); os.chdir(W)
import neurofly_daemon as nd
OUT = Path("<SCRATCH>")
backend = os.environ.get("BK", "connectome-fixed")
steps = int(os.environ.get("STEPS", "500"))
ALL = ["open-arena", "t-maze", "y-maze", "heat-maze", "buridan", "visual-operant", "wind-tunnel", "looming-escape",
       "optomotor", "gap-crossing", "circadian-dam", "courtship", "labyrinth", "multisensory-sandbox"]
assays = os.environ["ASSAYS"].split(",") if os.environ.get("ASSAYS") else ALL
r = nd.ContinuousExperimentRunner(initial_paradigm=assays[0], backend=backend,
                                  graph_dir=W / "outputs/brainlab/malecns_v1", output_dir=OUT / f"chan-{backend}",
                                  registry_root=OUT / f"chanreg-{backend}", trial_length_s=1e6, continuous=True)
gc = r.graph_controller
chans = dict(gc.sensory_indices); chans["epg"] = gc.epg_indices
for k, v in gc.dn_indices.items():
    chans["DN_" + k] = v
import pyarrow.feather as feather
df = feather.read_table(r.shared_graph.identity.neuron_map_path).to_pandas()
ct = df.sort_values("node_index")["cell_type"].astype(str).values
res = {}
for a in assays:
    r._init_arena(a)
    inst = r.registry.active
    seen = {"inj": {}, "stim": {}}
    orig = inst.step

    def wrapped(currents, ms, _o=orig):
        c = np.asarray(currents)
        for name, idx in chans.items():
            if idx:
                v = float(np.max(c[idx]))
                if v != 0:
                    seen["inj"][name] = max(seen["inj"].get(name, 0.0), v)
        return _o(currents, ms)
    inst.step = wrapped
    spk = {k: 0 for k in chans}
    tot = np.zeros(len(ct), dtype=np.int64)
    stimmax = {}
    for i in range(steps):
        with r.lock:
            res_step = r.step_once(publish=False)
        cnt = gc.last_counts
        if cnt is not None:
            cnt = np.asarray(cnt)
            tot += cnt.astype(np.int64)
            for k, idx in chans.items():
                if idx:
                    spk[k] += int(cnt[idx].sum())
        for k, v in (res_step.get("stimuli") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stimmax[k] = max(stimmax.get(k, -1e9), float(v))
    inst.step = orig
    active = int((tot > 0).sum())
    top = {}
    for idx in np.argsort(-tot)[:400]:
        if tot[idx] == 0:
            break
        top[ct[idx]] = top.get(ct[idx], 0) + int(tot[idx])
    top = dict(sorted(top.items(), key=lambda kv: -kv[1])[:12])
    res[a] = {"injected_max_current": seen["inj"], "channel_spikes": {k: v for k, v in spk.items() if v},
              "neurons_ever_active": active, "total_spikes": int(tot.sum()), "top_types": top,
              "stim_max": {k: round(v, 4) for k, v in stimmax.items()}}
    print("==", a, json.dumps(res[a]), flush=True)
(OUT / f"channels-{backend}.json").write_text(json.dumps(res, indent=1))
