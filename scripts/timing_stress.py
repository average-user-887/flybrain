#!/usr/bin/env python3
"""Isolated timing, lock-wait and responsiveness stress check for the NeuroFly daemon.

Starts its OWN daemon in a child process on a free port >= 19000 with every output
under ``--work-dir`` (never the default outputs/, never a live service port), attaches
one raw SSE client, sends commands at a fixed cadence, and writes a JSON receipt:

* achieved versus requested speed (from server-authored telemetry),
* SSE inter-arrival gaps and whether a stale-stream disconnect would have fired
  (``stale_threshold_s``, the dashboard watchdog's threshold for the code under test),
* command round-trip latency p50/p95/max against the proposed gate
  (p95 < 250 ms, max < 1 s),
* simulation-lock wait profile per thread role (``TimedLock``), sampled by the child.

``--module`` selects the daemon source: ``current`` (this checkout) or a path to
another copy (e.g. ``git show 1dfa7f7:neurofly_daemon.py > /tmp/x/neurofly_daemon.py``)
to profile the unchanged baseline with the same instrumentation.

Usage:
  python scripts/timing_stress.py --work-dir /tmp/job --seconds 60 --speed 100 \
      --paradigm wind-tunnel --receipt outputs/rethink-audit/stress_100x_current.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_P95_MS = 250.0
GATE_MAX_MS = 1000.0


# ----------------------------------------------------------------------------- child
def serve(args) -> None:
    sys.path.insert(0, str(ROOT))
    import neurofly_daemon as current
    from stream_gateway import StreamGateway, StreamPolicy
    if args.module == "current":
        mod = current
    else:
        spec = importlib.util.spec_from_file_location("neurofly_daemon_under_test", args.module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    work = Path(args.work_dir)
    runner = mod.ContinuousExperimentRunner(initial_paradigm=args.paradigm, sim_speed=args.speed,
                                            checkpoint_interval=3600, output_dir=work / "outputs")
    if args.yield_s is not None and hasattr(runner, "yield_wall_s"):
        runner.yield_wall_s = args.yield_s
    if not isinstance(runner.lock, current.TimedLock):
        runner.lock = current.TimedLock()   # same instrumentation for the baseline
    mod.NeuroflyHTTPHandler.runner = runner
    mod.NeuroflyHTTPHandler.gateway = StreamGateway(StreamPolicy())
    server = mod.ThreadingHTTPServer(("127.0.0.1", args.port), mod.NeuroflyHTTPHandler)
    server.daemon_threads = True
    runner.start()
    # The durable recorder polls the runner under its lock, as in run_daemon().
    from learning_recorder import LearningRecorder, RecorderThread
    recorder = RecorderThread(runner, LearningRecorder(work / "data", session={"stress": True}),
                              summary_interval=5.0)
    recorder.start()
    profile_path = work / "child_profile.json"

    def dump():
        while True:
            time.sleep(1.0)
            data = {"lock_profile": runner.lock.profile(), "total_steps": runner.total_steps,
                    "timing": runner.timing_snapshot() if hasattr(runner, "timing_snapshot") else None}
            tmp = profile_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, profile_path)

    threading.Thread(target=dump, daemon=True).start()

    def stop(*_):
        runner.running = False
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    (work / "ready").write_text(str(args.port))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        runner.running = False
        server.server_close()


# ----------------------------------------------------------------------------- parent
def free_port(start: int = 19000) -> int:
    for port in range(start, start + 500):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port >= 19000")


class SSEClient(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.frames = []        # (recv_wall, step, server_ts, achieved, requested, seq)
        self.events = []        # (recv_wall, kind)
        self.error = None
        self.stop = False

    def run(self):
        try:
            s = socket.create_connection(("127.0.0.1", self.port), timeout=30)
            s.sendall(b"GET /api/stream HTTP/1.1\r\nHost: x\r\nAccept: text/event-stream\r\n\r\n")
            buf = b""
            header_done = False
            while not self.stop:
                chunk = s.recv(65536)
                if not chunk:
                    self.error = "stream closed by server"
                    return
                buf += chunk
                if not header_done:
                    if b"\r\n\r\n" not in buf:
                        continue
                    buf = buf.split(b"\r\n\r\n", 1)[1]
                    header_done = True
                while b"\n\n" in buf:
                    raw, buf = buf.split(b"\n\n", 1)
                    now = time.time()
                    event, data = "message", None
                    for line in raw.split(b"\n"):
                        if line.startswith(b"event: "):
                            event = line[7:].decode()
                        elif line.startswith(b"data: "):
                            data = line[6:]
                    self.events.append((now, event))
                    if event == "message" and data:
                        pkt = json.loads(data)
                        t = pkt.get("timing") or {}
                        self.frames.append((now, pkt.get("step"), pkt.get("timestamp"),
                                            t.get("achieved_speed"), pkt.get("sim_speed"), t.get("snapshot_seq")))
        except Exception as exc:  # recorded, not swallowed
            self.error = f"{type(exc).__name__}: {exc}"


def post(port, payload, timeout=10.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/command", data=json.dumps(payload).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return time.perf_counter() - t0, body


def get_latency(port, path, timeout=10.0):
    t0 = time.perf_counter()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
        resp.read()
    return time.perf_counter() - t0


def pct(values, q):
    vals = sorted(values)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None


def run(args) -> dict:
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    assert port >= 19000, "isolated instances must use a port >= 19000"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    child = subprocess.Popen([sys.executable, __file__, "--serve", "--port", str(port), "--work-dir", str(work),
                              "--module", args.module, "--paradigm", args.paradigm, "--speed", str(args.speed)]
                             + ([] if args.yield_s is None else ["--yield-s", str(args.yield_s)]),
                             env=env, cwd=str(work), stdout=open(work / "child.log", "w"), stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 60
        while not (work / "ready").exists():
            if child.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"child failed to start; see {work / 'child.log'}")
            time.sleep(0.1)
        time.sleep(1.0)
        sse = SSEClient(port)
        sse.start()
        latencies, status_lat, acks, failures = [], [], [], []
        t_end = time.time() + args.seconds
        i = 0
        while time.time() < t_end:
            cmd = ({"action": "set_speed", "speed": args.speed} if i % 2 == 0 else
                   {"action": "set_param", "name": args.param, "value": args.param_values[(i // 2) % 2]})
            try:
                lat, body = post(port, cmd)
                latencies.append(lat)
                acks.append({"action": cmd["action"], "status": body.get("status"),
                             "applied_step": (body.get("ack") or {}).get("applied_step")})
            except Exception as exc:
                failures.append(f"{cmd['action']}: {type(exc).__name__}: {exc}")
            try:
                status_lat.append(get_latency(port, "/api/status"))
            except Exception as exc:
                failures.append(f"status: {type(exc).__name__}: {exc}")
            i += 1
            time.sleep(args.command_interval)
        sse.stop = True
        time.sleep(1.2)
        profile = json.loads((work / "child_profile.json").read_text())
    finally:
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    frames = sse.frames
    gaps = [b[0] - a[0] for a, b in zip(frames, frames[1:])]
    all_events = [t for t, _ in sse.events]
    any_gaps = [b - a for a, b in zip(all_events, all_events[1:])]
    # Achieved speed from server-authored timestamps and sim steps (first/last frame).
    achieved = None
    if len(frames) >= 2 and frames[-1][2] and frames[0][2] and frames[-1][2] > frames[0][2]:
        achieved = (frames[-1][1] - frames[0][1]) * 0.02 / (frames[-1][2] - frames[0][2])
    lat_ms = [v * 1e3 for v in latencies]
    p95 = pct(lat_ms, 0.95)
    mx = max(lat_ms) if lat_ms else None
    stale_threshold = args.stale_threshold_s
    return {
        "kind": "isolated daemon stress check",
        "module": args.module,
        "yield_s_override": args.yield_s,
        "paradigm": args.paradigm,
        "requested_speed": args.speed,
        "wall_seconds": args.seconds,
        "port": port,
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "cpu_count": os.cpu_count(), "machine": platform.machine()},
        "achieved_speed_server_timestamps": round(achieved, 3) if achieved else None,
        "achieved_speed_reported_last": frames[-1][3] if frames else None,
        "sse": {"frames": len(frames), "events": len(sse.events),
                "frame_gap_p50_ms": round(statistics.median(gaps) * 1e3, 1) if gaps else None,
                "frame_gap_p95_ms": round(pct(gaps, 0.95) * 1e3, 1) if gaps else None,
                "frame_gap_max_ms": round(max(gaps) * 1e3, 1) if gaps else None,
                "any_event_gap_max_ms": round(max(any_gaps) * 1e3, 1) if any_gaps else None,
                "client_error": sse.error,
                "stale_threshold_s": stale_threshold,
                "stale_disconnect_would_fire": bool(gaps and max(gaps) > stale_threshold) or bool(sse.error),
                "first_step": frames[0][1] if frames else None, "last_step": frames[-1][1] if frames else None},
        "commands": {"count": len(latencies), "failures": failures,
                     "p50_ms": round(pct(lat_ms, 0.5), 2) if lat_ms else None,
                     "p95_ms": round(p95, 2) if p95 is not None else None,
                     "max_ms": round(mx, 2) if mx is not None else None,
                     "acks_with_applied_step": sum(1 for a in acks if a["applied_step"] is not None),
                     "gate": {"p95_lt_ms": GATE_P95_MS, "max_lt_ms": GATE_MAX_MS,
                              "pass": bool(lat_ms) and p95 < GATE_P95_MS and mx < GATE_MAX_MS and not failures}},
        "status_get": {"p95_ms": round(pct(status_lat, 0.95) * 1e3, 2) if status_lat else None,
                       "max_ms": round(max(status_lat) * 1e3, 2) if status_lat else None},
        "child_profile": profile,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--module", default="current")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--paradigm", default="wind-tunnel")
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--command-interval", type=float, default=0.25)
    ap.add_argument("--param", default="windVelocity")
    ap.add_argument("--param-values", type=float, nargs=2, default=[20.0, 25.0])
    ap.add_argument("--stale-threshold-s", type=float, default=8.0)
    ap.add_argument("--yield-s", type=float, default=None, help="override the scheduler GIL hand-over (seconds)")
    ap.add_argument("--receipt", default="")
    args = ap.parse_args()
    if args.serve:
        serve(args)
        return
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.receipt:
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
