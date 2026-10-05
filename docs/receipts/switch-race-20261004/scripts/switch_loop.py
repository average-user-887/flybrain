"""API switching loop against a test daemon: counts freezes.

A freeze is: /api/status reports status=="error", or total_steps does not advance
within STEP_TIMEOUT after a switch acknowledged ok while the run is not paused.
Usage: switch_loop.py DAEMON_URL PASSES MODE(settled|rapid) OUT.json
"""
import json, sys, time, urllib.request

ASSAYS = ["open-arena", "t-maze", "y-maze", "heat-maze", "buridan", "visual-operant",
          "wind-tunnel", "looming-escape", "optomotor", "gap-crossing", "circadian-dam",
          "courtship", "labyrinth", "multisensory-sandbox"]
STEP_TIMEOUT = 30.0
url, passes, mode, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]


def call(path, body=None, timeout=60):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def status():
    return call("/api/status")


def switch(p):
    r = call("/api/command", {"action": "switch_paradigm", "paradigm": p})
    deadline = time.time() + 60
    while r.get("status") == "queued" and time.time() < deadline:
        time.sleep(0.1)
        if status().get("active_paradigm") == p:
            return {"status": "ok", "queued": True}
    return r


def advances(timeout=STEP_TIMEOUT):
    s0 = status()
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.25)
        s = status()
        if s.get("status") == "error":
            return False, s
        if s["total_steps"] > s0["total_steps"]:
            return True, s
    return False, status()


call("/api/command", {"action": "set_paused", "paused": False})
log = {"url": url, "passes": passes, "mode": mode, "switches": 0, "freezes": [], "started": time.time()}
for k in range(passes):
    for p in ASSAYS:
        r = switch(p)
        log["switches"] += 1
        if r.get("status") != "ok":
            log["freezes"].append({"pass": k, "assay": p, "kind": "switch-failed", "reply": r})
            continue
        if mode == "settled":
            ok, s = advances()
            if not ok:
                log["freezes"].append({"pass": k, "assay": p, "kind": "no-advance", "status": s.get("status"),
                                       "error": s.get("error"), "error_detail": s.get("error_detail"),
                                       "total_steps": s.get("total_steps"), "paused": s.get("paused")})
                # A latched halt is permanent on master; stop counting this daemon.
                if s.get("status") == "error":
                    break
    else:
        if mode == "rapid":
            ok, s = advances()
            if not ok:
                log["freezes"].append({"pass": k, "kind": "no-advance-after-rapid", "status": s.get("status"),
                                       "error": s.get("error"), "total_steps": s.get("total_steps")})
                break
        continue
    break
s = status()
log.update(final_status=s.get("status"), final_error=s.get("error"), final_steps=s.get("total_steps"),
           cleared_errors=s.get("cleared_errors"), elapsed_s=round(time.time() - log["started"], 1))
json.dump(log, open(out, "w"), indent=1, default=str)
print(json.dumps({k: log[k] for k in ("mode", "passes", "switches", "final_status", "final_error", "elapsed_s")}),
      "freezes:", len(log["freezes"]))
for f in log["freezes"][:3]:
    print("  ", {k: f.get(k) for k in ("pass", "assay", "kind", "error")})
