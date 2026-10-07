"""Simulation / web process split: HTTP stays responsive while the simulation is
blocked, and a dead simulation process is reported as down, never online."""
from __future__ import annotations

import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurofly import split  # noqa: E402

# A headless "simulation process" around a fake runner: a stepping thread publishes
# snapshots without the lock; the lock is held by a "long step" from the start, so any
# path that waited for it would block.
FAKE_SIM = r'''
import json, sys, threading, time
from collections import deque
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from neurofly.split import SimServer

class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.lock.acquire()                     # a step that never finishes
        self.start_time = time.time(); self.last_error = None; self.cleared_errors = []
        self.paused = False; self.continuous = True; self.running = True; self.total_steps = 0
        self.sim_speed = 1.0; self.active_paradigm_id = "t-maze"; self.active_paradigm_title = "T"
        self.current_trial = 1; self.trial_history = []; self.trial_sim_time = 0.0
        self.trial_length_s = None; self.arena = SimpleNamespace(world_bounds=[0, 1, 0, 1])
        self.backend = "connectome-fixed"; self.last_step_wall_s = 0.01
        self.recordings_dir = Path(sys.argv[3]); self.run_id = "fake"; self.published = None
        self.command_acks = deque(maxlen=8); self.seq = 0
    def health(self):
        return {"status": "online", "error": None, "error_detail": None, "halted": False,
                "liveness": {"state": "advancing", "step": self.total_steps, "step_in_progress_s": 0.0,
                             "last_advance_age_s": 0.0, "sim_thread_alive": True},
                "mode": "scientific", "result_validity": None, "observation_validity_update": None,
                "persistence": {"ok": True}, "recording_error": None, "loop_failure": None, "thread_failures": []}
    def recording_status(self): return None
    def identity(self): return {"run_id": "fake", "backend": self.backend}
    def compute_info(self): return {"device": "cpu"}
    def timing_snapshot(self): return {"step": self.total_steps, "achieved_speed": 1.0}
    def step_in_progress_s(self): return 0.0
    def _publish_snapshot(self):
        self.seq += 1
        data = json.dumps({"type": "telemetry", "timing": {"step": self.total_steps},
                           "command_acks": list(self.command_acks)}).encode()
        self.published = SimpleNamespace(seq=self.seq, step=self.total_steps, data=data, wall_time=time.time())
        return self.published
    def dispatch_command(self, cmd, on_queued=None):
        cid = "fake-%d" % (self.seq + 1000)
        if on_queued: on_queued(cid)
        self.lock.acquire(); self.lock.release()     # applied only when the step ends
        return {"status": "ok", "command_id": cid}
    def read_view(self, key, build, wait_s=0.2, max_wait_s=3.0):
        return None

runner = Runner()
server = SimServer(runner, sys.argv[1]); server.start()
while True:
    runner.total_steps += 1
    runner._publish_snapshot()
    time.sleep(0.02)
'''


def _get(port, path, timeout=5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t = time.monotonic()
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    return resp.status, json.loads(body), time.monotonic() - t


def _post(port, body, timeout=10.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t = time.monotonic()
    conn.request("POST", "/api/command", body=json.dumps(body), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read()), time.monotonic() - t


def _wait(pred, limit=20.0):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not reached")


@pytest.fixture
def fake_split(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "STALE_AFTER_S", 1.0)
    monkeypatch.setattr(split, "ACCEPT_WAIT_S", 0.5)
    sock_dir = tempfile.mkdtemp(prefix="nf-split-test-")
    path = os.path.join(sock_dir, "sim.sock")
    sim = subprocess.Popen([sys.executable, "-c", FAKE_SIM, path, str(ROOT), str(tmp_path)])
    server, link = split.build_web_server(path, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    _wait(lambda: link.connected and link.state)
    try:
        yield sim, port, link
    finally:
        server.shutdown()
        server.server_close()
        if sim.poll() is None:
            sim.send_signal(signal.SIGCONT)
            sim.kill()
        sim.wait(10)
        for name in os.listdir(sock_dir):
            os.unlink(os.path.join(sock_dir, name))
        os.rmdir(sock_dir)


def test_http_answers_while_the_simulation_holds_its_lock(fake_split):
    sim, port, link = fake_split
    for _ in range(20):
        status, body, took = _get(port, "/api/status")
        assert status == 200 and body["status"] == "online" and took < 0.5
    status, body, took = _post(port, {"action": "set_speed", "params": {"speed": 2},
                                      "client_command_id": "c-1"})
    # Answered at once with the id; the result follows when the step ends.
    assert status == 200 and body["status"] == "queued" and body["command_id"].startswith("fake-")
    assert took < split.REPLY_WAIT_S + 0.5
    status, _, took = _get(port, "/api/telemetry")
    assert status == 200 and took < 0.5


def test_a_frozen_simulation_process_is_reported_unresponsive(fake_split):
    sim, port, link = fake_split
    sim.send_signal(signal.SIGSTOP)                 # the whole simulation process blocked
    try:
        status, body, took = _get(port, "/api/status")
        assert status == 200 and took < 0.5         # the web process still answers at once
        _wait(lambda: _get(port, "/api/status")[1]["status"] == "error", limit=5)
        status, body, took = _get(port, "/api/status")
        assert body["halted"] is True and body["liveness"]["state"] == "sim_process_unresponsive"
        assert "not responding" in body["error"] and took < 0.5
        status, body, took = _post(port, {"action": "set_speed", "params": {"speed": 3}})
        assert body["status"] == "error" and body["applied"] is None and "unknown" in body["message"]
    finally:
        sim.send_signal(signal.SIGCONT)
    _wait(lambda: _get(port, "/api/status")[1]["status"] == "online", limit=5)


def test_a_dead_simulation_process_is_reported_down_never_online(fake_split):
    sim, port, link = fake_split
    sim.kill()
    sim.wait(10)
    _wait(lambda: not link.connected, limit=5)
    status, body, took = _get(port, "/api/status")
    assert status == 200 and took < 0.5
    assert body["status"] == "error" and body["halted"] is True
    assert body["liveness"]["state"] == "sim_process_down" and body["liveness"]["sim_thread_alive"] is False
    assert body["timing"]["achieved_speed"] == 0.0
    assert body["process_layout"]["sim"]["connected"] is False
    status, body, _ = _post(port, {"action": "set_speed", "params": {"speed": 3}})
    assert body["status"] == "error" and body["applied"] is False and body["sim_process_down"] is True


def test_launcher_flags_are_not_forwarded_twice():
    assert split._strip_launcher_flags(
        ["--port", "1", "--process-mode", "split", "--ipc-socket=/x", "--single-process", "--speed", "2"]
    ) == ["--port", "1", "--speed", "2"]


def test_split_daemon_survives_web_death_and_reports_sim_death(tmp_path):
    """End to end with the real runner (synthetic test graph, CPU)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.Popen([sys.executable, str(ROOT / "neurofly_daemon.py"), "--port", str(port),
                             "--backend", "connectome-fixed", "--test-synthetic-graph", "--brain-backend", "cpu",
                             "--paradigm", "t-maze", "--speed", "5", "--output-dir", str(tmp_path / "out"),
                             "--data-dir", str(tmp_path / "data"), "--pid-file", str(tmp_path / "pid")],
                            cwd=tmp_path, env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def status():
        try:
            return _get(port, "/api/status", timeout=2)[1]
        except OSError:
            return None
    try:
        first = _wait(lambda: (lambda b: b if b and b["total_steps"] > 20 else None)(status()), limit=120)
        sim_pid = first["process_layout"]["sim"]["sim_pid"]
        os.kill(first["process_layout"]["web_pid"], signal.SIGKILL)       # the web process dies
        again = _wait(lambda: (lambda b: b if b and b["process_layout"]["web_pid"] != first["process_layout"]["web_pid"]
                                else None)(status()), limit=30)
        assert again["process_layout"]["sim"]["sim_pid"] == sim_pid      # same simulation, not restarted
        assert again["status"] == "online" and again["total_steps"] > first["total_steps"]
        os.kill(sim_pid, signal.SIGKILL)                                  # the simulation process dies
        down = _wait(lambda: (lambda b: b if b and b["status"] == "error" else None)(status()), limit=10)
        assert down["halted"] is True and down["liveness"]["state"] == "sim_process_down"
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
