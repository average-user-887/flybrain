#!/usr/bin/env python3
"""HTTP responsiveness of the daemon while the simulation steps, saves and rebuilds.

Starts one daemon (``--process-mode single`` or ``split``), then for each phase
polls GET /api/status (20 Hz) and POSTs set_speed (2 Hz) while an SSE client
records frame arrival times:

  stepping    30 s of normal stepping
  checkpoint  5 x save_checkpoint, each window from the POST to its final ack
  rebuild     switch_backend connectome-fixed -> connectome-plastic -> fixed

Writes ``<out>/<mode>.json`` with per-phase latency percentiles, failures, SSE
frame gaps and simulation throughput.  Usage:

  bench_split_sim_web.py --mode split --port 8962 --out DIR -- <daemon args>
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))] * 1e3, 1)


def summary(values):
    return {"n": len(values), "p50_ms": pct(values, 50), "p95_ms": pct(values, 95),
            "p99_ms": pct(values, 99), "max_ms": pct(values, 100)}


class Bench:
    def __init__(self, port):
        self.port = port
        self.t0 = time.monotonic()
        self.samples = []          # (t_start, kind, latency_s or None, http_status, body_status)
        self.frames = []           # (t, seq, step, last_step_wall_s)
        self.acks = {}             # command_id -> arrival time
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def now(self):
        return time.monotonic() - self.t0

    def request(self, method, path, body=None, timeout=5.0):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        start = self.now()
        try:
            conn.request(method, path, body=json.dumps(body) if body is not None else None,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read()
            lat = self.now() - start
            try:
                parsed = json.loads(data)
            except ValueError:
                parsed = None
            return start, lat, resp.status, parsed
        except (OSError, http.client.HTTPException) as exc:
            return start, None, type(exc).__name__, None
        finally:
            conn.close()

    def poller(self, kind, period, method, path, body=None):
        while not self.stop.is_set():
            t = time.monotonic()
            start, lat, status, parsed = self.request(method, path, body)
            with self.lock:
                self.samples.append((start, kind, lat, status,
                                     parsed.get("status") if isinstance(parsed, dict) else None,
                                     parsed.get("command_id") if isinstance(parsed, dict) else None))
            self.stop.wait(max(0.0, period - (time.monotonic() - t)))

    def sse(self):
        while not self.stop.is_set():
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
                conn.request("GET", "/api/stream")
                resp = conn.getresponse()
                buf = b""
                while not self.stop.is_set():
                    line = resp.fp.readline()
                    if not line:
                        break
                    if line.startswith(b"data: ") and buf == b"":
                        t = self.now()
                        try:
                            d = json.loads(line[6:])
                        except ValueError:
                            continue
                        if d.get("type") == "telemetry":
                            timing = d.get("timing") or {}
                            self.frames.append((t, timing.get("snapshot_seq"), timing.get("step"),
                                                timing.get("last_step_wall_s")))
                            for ack in d.get("command_acks") or ():
                                cid = ack.get("command_id")
                                if cid and cid not in self.acks:
                                    self.acks[cid] = t
                    elif line.startswith(b"event: "):
                        buf = line
                    elif line.strip() == b"":
                        buf = b""
            except (OSError, http.client.HTTPException):
                time.sleep(0.2)

    def command_until_ack(self, body, limit=600.0):
        start, lat, status, parsed = self.request("POST", "/api/command", body, timeout=limit)
        if not isinstance(parsed, dict):
            return start, self.now(), {"http": status}
        if parsed.get("status") == "queued":
            cid = parsed.get("command_id")
            deadline = time.monotonic() + limit
            while cid not in self.acks and time.monotonic() < deadline:
                time.sleep(0.02)
            return start, self.acks.get(cid, self.now()), {"queued": True, "command_id": cid,
                                                           "reply_ms": round(lat * 1e3, 1)}
        return start, start + lat, {"queued": False, "status": parsed.get("status"),
                                    "message": parsed.get("message"), "reply_ms": round(lat * 1e3, 1)}


def tree_rss_mb(root):
    """Resident memory of a process and all its descendants (MB)."""
    kids = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
                kids.setdefault(int(fields[1]), []).append(int(d.name))
            except OSError:
                pass
    total, todo = 0, [root]
    while todo:
        pid = todo.pop()
        todo += kids.get(pid, [])
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            pass
    return total / 1024.0


def rss_sampler(pid, bench, out):
    while not bench.stop.is_set():
        out.append(tree_rss_mb(pid))
        bench.stop.wait(1.0)


def wait_online(bench, limit=600):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        _, _, status, parsed = bench.request("GET", "/api/status", timeout=3)
        if status == 200 and parsed and parsed.get("status") == "online" and parsed.get("total_steps", 0) > 50:
            return parsed
        time.sleep(1)
    raise SystemExit("daemon did not come online")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("single", "split"), required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stepping-s", type=float, default=30.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--checkpoints", type=int, default=5)
    ap.add_argument("--daemon-cpus", default=None, help="taskset CPU list for the daemon under test")
    ap.add_argument("--rebuilds", default="connectome-plastic,connectome-fixed,connectome-plastic,connectome-fixed")
    ap.add_argument("--rebuild-phase", default="rebuild")
    ap.add_argument("daemon_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    extra = [a for a in args.daemon_args if a != "--"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"{args.mode}-{args.rebuild_phase}_daemon.log", "w")
    pin = ["taskset", "-c", args.daemon_cpus] if args.daemon_cpus else []
    proc = subprocess.Popen([*pin, args.python, "-u", str(ROOT / "neurofly_daemon.py"), "--port", str(args.port),
                             "--process-mode", args.mode, *extra], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    bench = Bench(args.port)
    rss = []
    phases = []
    result = {"mode": args.mode, "daemon_args": extra}
    try:
        st = wait_online(bench)
        result["identity"] = {k: st.get("identity", {}).get(k) for k in ("backend", "graph_sha256", "synthetic")}
        result["compute"] = st.get("compute")
        threads = [threading.Thread(target=bench.sse, daemon=True),
                   threading.Thread(target=bench.poller, args=("status", 0.05, "GET", "/api/status"), daemon=True),
                   threading.Thread(target=bench.poller, args=("command", 0.5, "POST", "/api/command",
                                                               {"action": "set_speed", "params": {"speed": 100}}),
                                    daemon=True)]
        threads.append(threading.Thread(target=rss_sampler, args=(proc.pid, bench, rss), daemon=True))
        for t in threads:
            t.start()
        time.sleep(2)
        a = bench.now()
        time.sleep(args.stepping_s)
        phases.append({"phase": "stepping", "start": a, "end": bench.now()})
        for i in range(args.checkpoints):
            s, e, info = bench.command_until_ack({"action": "save_checkpoint", "label": f"bench{i}"})
            phases.append({"phase": "checkpoint", "start": s, "end": e + 0.25, "completed_ms": round((e - s) * 1e3, 1),
                           **info})
            time.sleep(3)
        for target in filter(None, args.rebuilds.split(",")):
            s, e, info = bench.command_until_ack({"action": "switch_backend", "params": {"backend": target}})
            phases.append({"phase": args.rebuild_phase, "target": target, "start": s, "end": e + 0.25,
                           "completed_ms": round((e - s) * 1e3, 1), **info})
            time.sleep(5)
        time.sleep(1)
    finally:
        bench.stop.set()
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        result["daemon_exit"] = proc.returncode

    def in_phase(t, name):
        return any(p["phase"] == name and p["start"] <= t <= p["end"] for p in phases)

    report = {}
    for name in ("stepping", "checkpoint", args.rebuild_phase):
        dur = sum(p["end"] - p["start"] for p in phases if p["phase"] == name)
        row = {"window_s": round(dur, 2)}
        for kind in ("status", "command"):
            rows = [s for s in bench.samples if s[1] == kind and in_phase(s[0], name)]
            lats = [s[2] for s in rows if s[2] is not None]
            row[kind] = summary(lats)
            row[kind]["failed"] = sum(1 for s in rows if s[2] is None or s[3] != 200)
            row[kind]["over_2s"] = sum(1 for s in rows if s[2] is None or s[2] > 2.0)
            row[kind]["body_status"] = sorted({str(s[4]) for s in rows})
            if kind == "command":
                # Completed = final acknowledgement (in the reply, or later in the stream).
                done = [s[2] if s[4] in ("ok", "error") else
                        (bench.acks[s[5]] - s[0] if s[5] in bench.acks else None) for s in rows if s[2] is not None]
                row["command_completed"] = summary([d for d in done if d is not None])
                row["command_completed"]["never_acked"] = sum(1 for d in done if d is None)
        fr = [f for f in bench.frames if in_phase(f[0], name)]
        # A gap belongs to a phase when it overlaps one of that phase's windows, so a
        # stall that spans a save is counted, and gaps between windows are not.
        gaps = [b[0] - a[0] for a, b in zip(bench.frames, bench.frames[1:])
                if any(p["phase"] == name and a[0] <= p["end"] and b[0] >= p["start"] for p in phases)]
        row["sse"] = {"frames": len(fr), "gap": summary(gaps)}
        steps = [f for f in fr if f[2] is not None]
        if name == "stepping" and len(steps) > 1:
            row["steps_per_s"] = round((steps[-1][2] - steps[0][2]) / (steps[-1][0] - steps[0][0]), 1)
            walls = [f[3] for f in steps if f[3]]
            row["last_step_wall"] = summary(walls)
        report[name] = row
    result["rss_mb_max"] = round(max(rss), 1) if rss else None
    result["phases"] = phases
    result["report"] = report
    result["raw"] = {"samples": bench.samples, "frames": bench.frames, "acks": bench.acks}
    (out / f"{args.mode}{'' if args.rebuild_phase == 'rebuild' else '-' + args.rebuild_phase}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
