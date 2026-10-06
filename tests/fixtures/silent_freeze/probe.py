"""Observe a daemon for N seconds: /api/status, SSE frames vs heartbeats, threads, CPU.

python probe.py <port> <seconds> <fault_dir> [pid]
Prints one JSON line summarising what a user (status probe, dashboard stream) would see.
"""
import json
import socket
import sys
import threading
import time
import urllib.request

port, seconds, fault_dir = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
pid = sys.argv[4] if len(sys.argv) > 4 else None
base = f"http://127.0.0.1:{port}"


def status():
    try:
        with urllib.request.urlopen(base + "/api/status", timeout=5) as r:
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def cpu_ticks():
    if not pid:
        return None
    fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


sse = {"data": 0, "heartbeat": 0, "stream": 0, "steps_seen": set()}


def read_sse(stop):
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall(f"GET /api/stream HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        buf, event = b"", None
        while not stop.is_set():
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                kind = "data"
                for line in block.split(b"\n"):
                    if line.startswith(b"event: "):
                        kind = line[7:].decode()
                    if line.startswith(b"data: {") and kind == "data":
                        try:
                            pkt = json.loads(line[6:])
                            sse["steps_seen"].add((pkt.get("timing") or {}).get("step"))
                        except Exception:  # noqa: BLE001
                            pass
                if kind in sse:
                    sse[kind] += 1
        s.close()
    except Exception as exc:  # noqa: BLE001
        sse["error"] = f"{type(exc).__name__}: {exc}"


stop = threading.Event()
th = threading.Thread(target=read_sse, args=(stop,), daemon=True)
before = status()
c0, t0 = cpu_ticks(), time.time()
th.start()
time.sleep(seconds)
stop.set()
after = status()
c1, t1 = cpu_ticks(), time.time()
try:
    threads = json.load(open(f"{fault_dir}/threads.json"))["threads"]
except Exception:  # noqa: BLE001
    threads = None


def pick(d):
    if "probe_error" in d:
        return d
    tm = d.get("timing") or {}
    return {k: d.get(k) for k in ("status", "halted", "paused", "error", "total_steps")} | {
        "achieved_speed": tm.get("achieved_speed"), "snapshot_seq": tm.get("snapshot_seq"),
        "overloaded": tm.get("overloaded"),
        # Added with the fix (absent on a daemon from before it).
        "liveness": (d.get("liveness") or {}).get("state"),
        "persistence": (d.get("persistence") or {}).get("state"),
        "persistence_failing": sorted((d.get("persistence") or {}).get("failing") or {}),
        "recording_error": bool(d.get("recording_error")),
        "mode": d.get("mode"),
        "failure_class": (d.get("error_detail") or {}).get("failure_class"),
        "result_validity": (d.get("result_validity") or {}).get("state"),
        "incidents": len((d.get("result_validity") or {}).get("incidents") or []),
        "other_runs_incomplete": len((d.get("result_validity") or {}).get("other_runs_incomplete") or [])}


print(json.dumps({
    "window_s": seconds,
    "before": pick(before), "after": pick(after),
    "steps_advanced": (after.get("total_steps") or 0) - (before.get("total_steps") or 0),
    "sse_data_frames": sse["data"], "sse_heartbeats": sse["heartbeat"],
    "sse_distinct_steps": len(sse["steps_seen"]), "sse_error": sse.get("error"),
    "simloop_alive": None if threads is None else "NeuroFly-SimLoop" in threads,
    "recorder_alive": None if threads is None else "NeuroFly-Recorder" in threads,
    "cpu_pct": None if c0 is None else round(100 * (c1 - c0) / 100 / (t1 - t0), 1),
}))
