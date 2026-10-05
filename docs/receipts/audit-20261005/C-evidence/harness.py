import sys, os, json, time, math
from pathlib import Path
import numpy as np
W = Path(os.environ["W"]); sys.path.insert(0, str(W)); os.chdir(W)
import neurofly_daemon as nd
OUT = Path("<SCRATCH>")
backend = os.environ.get("BK", "connectome-fixed")
steps = int(os.environ.get("STEPS", "1500"))
ALL = ["open-arena", "t-maze", "y-maze", "heat-maze", "buridan", "visual-operant", "wind-tunnel", "looming-escape",
       "optomotor", "gap-crossing", "circadian-dam", "courtship", "labyrinth", "multisensory-sandbox"]
assays = os.environ["ASSAYS"].split(",") if os.environ.get("ASSAYS") else ALL
t0 = time.time()
r = nd.ContinuousExperimentRunner(initial_paradigm=assays[0], backend=backend,
                                  graph_dir=W / "outputs/brainlab/malecns_v1", output_dir=OUT / f"run-{backend}",
                                  registry_root=OUT / f"reg-{backend}", trial_length_s=float(os.environ.get("TRIAL_S", "10")))
print("init s", round(time.time() - t0, 1), "compute", r.compute_info(), flush=True)
gc = r.graph_controller
print("sensory idx sizes", {k: len(v) for k, v in gc.sensory_indices.items()}, "epg", len(gc.epg_indices),
      "dn", {k: len(v) for k, v in gc.dn_indices.items()}, "unavail", gc.sensory_unavailable, flush=True)
results = {}
for a in assays:
    r._init_arena(a)
    inst = r.registry.active
    cap = {"inj_n": [], "inj_sum": []}
    orig = inst.step

    def wrapped(currents, ms, _o=orig, cap=cap):
        c = np.asarray(currents)
        cap["inj_n"].append(int((c != 0).sum()))
        cap["inj_sum"].append(float(c.sum()))
        return _o(currents, ms)
    inst.step = wrapped
    rows = []
    t1 = time.time()
    trials0 = len(r.trial_history)
    for i in range(steps):
        with r.lock:
            r.step_once(publish=(i % 10 == 0))
        if i % 10 == 0:
            p = r.latest_telemetry
            c = p.get("connectome") or {}
            m = p["motor"]
            rec = m.get("record") or {}
            act = p.get("activity") or {}
            rates = act.get("rates")
            rows.append({"i": i, "x": p["fly"]["x"], "y": p["fly"]["y"], "h": p["fly"]["heading"], "v": p["fly"]["speed"],
                         "state": p["fly"]["state"], "spikes": c.get("total_spikes"), "dn": c.get("dn_rates"),
                         "src": m.get("motor_source"), "halted": m.get("motor_halted"),
                         "unsup": (c.get("unsupported") or c.get("optomotor_unsupported")),
                         "near_wall": rec.get("near_wall"), "contact": rec.get("in_contact"),
                         "attempted": rec.get("attempted_mm"), "realized": rec.get("realized_mm"),
                         "yaw": c.get("yaw_rate"), "fwd": c.get("forward_speed"),
                         "act_max": (max(rates) if rates else None),
                         "act_nz": (sum(1 for x in rates if x > 0) if rates else None),
                         "metrics": p.get("metrics"), "stim_keys": sorted((p.get("stimuli") or {}).keys()),
                         "err": p.get("error"), "epg_sum": sum(c.get("epg_wedges") or [0]),
                         "opto": c.get("optomotor")})
    inst.step = orig
    el = time.time() - t1
    trials = r.trial_history[trials0:]
    xs = [q["x"] for q in rows]; ys = [q["y"] for q in rows]
    sp = [q["spikes"] or 0 for q in rows]
    summ = {"wall_s": round(el, 1), "graph_steps_called": len(cap["inj_n"]),
            "inj_nonzero_mean": float(np.mean(cap["inj_n"])) if cap["inj_n"] else None,
            "inj_nonzero_max": max(cap["inj_n"]) if cap["inj_n"] else None,
            "spikes_mean": float(np.mean(sp)), "spikes_max": max(sp), "frac_steps_spiking": float(np.mean([s > 0 for s in sp])),
            "x_range": [min(xs), max(xs)], "y_range": [min(ys), max(ys)],
            "path_extent": math.hypot(max(xs) - min(xs), max(ys) - min(ys)),
            "states": {s: sum(1 for q in rows if q["state"] == s) for s in set(q["state"] for q in rows)},
            "srcs": sorted(set(str(q["src"]) for q in rows)),
            "fwd_set": sorted(set(round(q["fwd"], 2) for q in rows if q["fwd"] is not None))[:8],
            "yaw_absmax": max((abs(q["yaw"] or 0) for q in rows), default=0),
            "near_wall_frac": float(np.mean([bool(q["near_wall"]) for q in rows])),
            "contact_frac": float(np.mean([bool(q["contact"]) for q in rows])),
            "realized_mean": float(np.mean([q["realized"] or 0 for q in rows])),
            "attempted_mean": float(np.mean([q["attempted"] or 0 for q in rows])),
            "dn_max": {k: max(((q["dn"] or {}).get(k, 0) for q in rows), default=0) for k in ["dna02_l", "dna02_r", "dnp09", "mdn", "gf"]},
            "act_max": max((q["act_max"] or 0) for q in rows), "act_nz_max": max((q["act_nz"] or 0) for q in rows),
            "epg_max": max(q["epg_sum"] for q in rows),
            "unsup": sorted(set(str(q["unsup"]) for q in rows if q["unsup"]))[:2],
            "errors": sorted(set(str(q["err"]) for q in rows if q["err"]))[:2],
            "stim_keys": rows[-1]["stim_keys"],
            "trials": [{"reason": t.get("reason"), "metric": t.get("metric"),
                        "metrics": {k: v for k, v in (t.get("metrics") or {}).items() if isinstance(v, (int, float, str, bool))}}
                       for t in trials][:8],
            "last_metrics": {k: v for k, v in (rows[-1]["metrics"] or {}).items() if isinstance(v, (int, float, str, bool))}}
    results[a] = summ
    print("==", a, json.dumps(summ, default=str)[:3500], flush=True)
    (OUT / f"rows-{backend}-{a}.json").write_text(json.dumps(rows, default=str))
(OUT / f"summary-{backend}-{os.environ.get('TAG','x')}.json").write_text(json.dumps(results, indent=1, default=str))
