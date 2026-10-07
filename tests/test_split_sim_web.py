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
import neurofly.split as sp
from neurofly.split import SimServer
sp.MAX_PENDING_COMMANDS = int(sys.argv[4])

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
        self.command_acks = deque(maxlen=8); self.seq = 0; self._commands = []
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
        cid = "fake-%d" % (self.seq + 1000 + len(self._commands))
        self._commands.append(cid)
        if on_queued: on_queued(cid)
        self.lock.acquire(); self.lock.release()     # applied only when the step ends
        return {"status": "ok", "command_id": cid}
    def read_view(self, key, build, wait_s=0.2, max_wait_s=3.0):
        return None
    def command_ack_lookup(self, command_id=None, client_command_id=None):
        return {"state": "queued" if command_id in self._commands else "unknown", "command_id": command_id}

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


def _start_fake_sim(path, tmp_path, sim_limit=64):
    return subprocess.Popen([sys.executable, "-c", FAKE_SIM, path, str(ROOT), str(tmp_path), str(sim_limit)])


@pytest.fixture
def fake_split(tmp_path, monkeypatch, request):
    monkeypatch.setattr(split, "STALE_AFTER_S", 1.0)
    monkeypatch.setattr(split, "ACCEPT_WAIT_S", 0.5)
    sock_dir = split.private_socket_dir()
    path = os.path.join(sock_dir, "sim.sock")
    sims = [_start_fake_sim(path, tmp_path, getattr(request, "param", 64))]
    server, link = split.build_web_server(path, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    _wait(lambda: link.connected and link.state)
    link.path_for_test, link.sims = path, sims
    try:
        yield sims[0], port, link
    finally:
        server.shutdown()
        server.server_close()
        for sim in sims:
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


@pytest.mark.parametrize("fake_split", [3], indirect=True)
def test_a_full_simulation_queue_refuses_explicitly(fake_split):
    sim, port, link = fake_split
    ids = []
    for i in range(3):          # the simulation is mid-step: these wait in its queue
        _, body, _ = _post(port, {"action": "set_speed", "params": {"speed": 2}})
        assert body["status"] == "queued"
        ids.append(body["command_id"])
    _, body, took = _post(port, {"action": "set_speed", "params": {"speed": 2}, "client_command_id": "c-9"})
    assert body["status"] == "error" and body["applied"] is False and body["queue_full"] is True
    assert took < 1.0
    _, ack, _ = _get(port, f"/api/command_ack?command_id={ids[0]}")
    assert ack["state"] == "queued"


def test_the_web_process_bounds_its_own_pending_requests(fake_split, monkeypatch):
    sim, port, link = fake_split
    monkeypatch.setattr(split, "MAX_PENDING_COMMANDS", 2)
    for i in range(2):
        assert _post(port, {"action": "set_speed", "params": {"speed": 2}})[1]["status"] == "queued"
    _, body, _ = _post(port, {"action": "set_speed", "params": {"speed": 2}})
    assert body["queue_full"] is True and body["applied"] is False
    assert _get(port, "/api/status")[1]["process_layout"]["pending_commands"] == 2


def test_the_web_process_reconnects_to_a_restarted_simulation(fake_split, tmp_path):
    sim, port, link = fake_split
    first_pid = _get(port, "/api/status")[1]["process_layout"]["sim"]["sim_pid"]
    sim.kill()
    sim.wait(10)
    _wait(lambda: _get(port, "/api/status")[1]["status"] == "error", limit=5)
    _, ack, _ = _get(port, "/api/command_ack?command_id=x-1")
    assert ack["state"] == "unknown"                       # never success while down
    link.sims.append(_start_fake_sim(link.path_for_test, tmp_path))
    body = _wait(lambda: (lambda b: b if b["status"] == "online" else None)(_get(port, "/api/status")[1]))
    assert body["process_layout"]["sim"]["sim_pid"] not in (None, first_pid)
    assert body["process_layout"]["sim"]["connects"] == 2
    assert _post(port, {"action": "set_speed", "params": {"speed": 2}})[1]["status"] == "queued"


def test_a_child_exits_when_its_launcher_vanishes_without_pdeathsig(tmp_path):
    """The parent-liveness fallback alone (no prctl): reparenting means the launcher is gone."""
    child = ("import os,sys,time; sys.path.insert(0, sys.argv[1]); from neurofly import split; "
             "split.PARENT_POLL_S = 0.1; split.bind_to_launcher(use_prctl=False); time.sleep(60)")
    middle = ("import os,subprocess,sys,time; os.environ['NEUROFLY_LAUNCHER_PID'] = str(os.getpid()); "
              "p = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); print(p.pid, flush=True); "
              "time.sleep(60)")
    mid = subprocess.Popen([sys.executable, "-c", middle, child, str(ROOT)], stdout=subprocess.PIPE, text=True)
    child_pid = int(mid.stdout.readline())
    time.sleep(1.0)
    os.kill(child_pid, 0)                                   # alive while the launcher lives
    mid.kill()
    mid.wait(10)

    def gone():
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return True
        return False
    _wait(gone, limit=5)


def test_launcher_flags_are_not_forwarded_twice():
    assert split._strip_launcher_flags(
        ["--port", "1", "--process-mode", "split", "--ipc-socket=/x", "--single-process", "--split", "--speed", "2"]
    ) == ["--port", "1", "--speed", "2"]


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_split_daemon(tmp_path, port, *extra):
    """The real runner (synthetic test graph, CPU) behind the split launcher."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    log = open(tmp_path / f"daemon-{port}-{time.monotonic_ns()}.log", "w")
    return subprocess.Popen([sys.executable, str(ROOT / "neurofly_daemon.py"), "--split", "--port", str(port),
                             "--backend", "connectome-fixed", "--test-synthetic-graph", "--brain-backend", "cpu",
                             "--paradigm", "t-maze", "--speed", "5", "--output-dir", str(tmp_path / "out"),
                             "--data-dir", str(tmp_path / "data"), "--pid-file", str(tmp_path / "pid"), *extra],
                            cwd=tmp_path, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)


def _status_or_none(port):
    try:
        return _get(port, "/api/status", timeout=2)[1]
    except OSError:
        return None


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_split_daemon_survives_web_death_and_reports_sim_death(tmp_path):
    """End to end with the real runner (synthetic test graph, CPU)."""
    port = _free_port()
    proc = _start_split_daemon(tmp_path, port)

    def status():
        return _status_or_none(port)
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


def test_children_stop_cleanly_when_the_launcher_is_killed_and_state_restores(tmp_path):
    """SIGKILL the launcher: both children exit, the simulation with its final save and a
    finalized recording; a new split daemon then restores that saved state."""
    from neurofly.recording import recording_invalid_reason
    port = _free_port()
    proc = _start_split_daemon(tmp_path, port, "--record", "split-e2e")
    try:
        first = _wait(lambda: (lambda b: b if b and b["total_steps"] > 50 else None)(_status_or_none(port)),
                      limit=120)
        sim_pid = first["process_layout"]["sim"]["sim_pid"]
        web_pid = first["process_layout"]["web_pid"]
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(10)
        _wait(lambda: _gone(sim_pid) and _gone(web_pid), limit=60)
        assert not (tmp_path / "pid").exists()               # the simulation's own clean shutdown ran
        rec = tmp_path / "out" / "recordings" / "split-e2e.nfrec"
        assert rec.is_file() and recording_invalid_reason(rec) is None
    finally:
        if proc.poll() is None:
            proc.kill()
    proc = _start_split_daemon(tmp_path, port)
    try:
        again = _wait(lambda: (lambda b: b if b and b["status"] == "online" and b["total_steps"] > 0 else None)(
            _status_or_none(port)), limit=120)
        assert again["identity"]["instance_id"] == first["identity"]["instance_id"]   # same saved brain
        assert again["process_layout"]["sim"]["sim_pid"] != sim_pid
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)


# --------------------------------------------------------------------- kills during a save
# The simulation process, with a test-only pause inside the atomic checkpoint commit:
# while a trigger file exists, the os.replace() that publishes a file under a
# checkpoints/ directory first writes a marker, then waits.  Product code is unchanged.
SIM_WITH_SAVE_PAUSE = r'''
import os, sys, time
trigger, marker = sys.argv[1], sys.argv[2]
_replace = os.replace
def paused_replace(src, dst, *a, **k):
    if "/checkpoints/" in str(dst) and os.path.exists(trigger):
        open(marker, "w").write(str(dst))
        while os.path.exists(trigger):
            time.sleep(0.05)
    return _replace(src, dst, *a, **k)
os.replace = paused_replace
sys.path.insert(0, sys.argv[3])
import neurofly_daemon
sys.argv = ["neurofly_daemon.py"] + sys.argv[4:]
neurofly_daemon.run_daemon()
'''


def _store_files(out):
    """Every registry checkpoint file with its content hash."""
    import hashlib
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*") if p.is_file() and "/checkpoints/" in str(p) and "registry" in str(p)}


class _MidSave:
    def __init__(self, tmp_path):
        self.tmp, self.out = tmp_path, tmp_path / "out"
        self.trigger, self.marker = tmp_path / "pause-save", tmp_path / "in-save"
        self.sock = os.path.join(split.private_socket_dir(), "sim.sock")
        self.port = _free_port()
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
        self.procs = []

    def sim(self):
        args = ["--process-mode", "sim", "--ipc-socket", self.sock, "--port", str(self.port),
                "--backend", "connectome-fixed", "--test-synthetic-graph", "--brain-backend", "cpu",
                "--paradigm", "t-maze", "--speed", "5", "--output-dir", str(self.out),
                "--data-dir", str(self.tmp / "data"), "--pid-file", str(self.tmp / "pid")]
        p = subprocess.Popen([sys.executable, "-c", SIM_WITH_SAVE_PAUSE, str(self.trigger), str(self.marker),
                              str(ROOT), *args], cwd=self.tmp, env=self.env,
                             stdout=open(self.tmp / f"sim-{len(self.procs)}.log", "w"), stderr=subprocess.STDOUT)
        self.procs.append(p)
        return p

    def web(self):
        p = subprocess.Popen([sys.executable, str(ROOT / "neurofly_daemon.py"), "--process-mode", "web",
                              "--ipc-socket", self.sock, "--port", str(self.port)], cwd=self.tmp, env=self.env,
                             stdout=open(self.tmp / f"web-{len(self.procs)}.log", "w"), stderr=subprocess.STDOUT)
        self.procs.append(p)
        return p

    def status(self):
        return _status_or_none(self.port)

    def online(self):
        return _wait(lambda: (lambda b: b if b and b["status"] == "online" and b["total_steps"] > 20
                              else None)(self.status()), limit=120)

    def save(self):
        return _post(self.port, {"action": "save_checkpoint", "label": "midsave"})[1]

    def close(self):
        if self.trigger.exists():
            self.trigger.unlink()
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    p.kill()


def _web_store_handles(pid, roots):
    held = []
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if any(target.startswith(str(r)) for r in roots):
            held.append(target)
    return held


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc fd check")
def test_web_killed_mid_save_keeps_the_save_and_never_touches_the_stores(tmp_path):
    m = _MidSave(tmp_path)
    try:
        m.sim()
        web = m.web()
        m.online()
        assert m.save()["status"] == "ok"
        good = _store_files(m.out)
        assert good
        m.trigger.write_text("pause")
        reply = m.save()
        assert reply["status"] == "queued" and reply["command_id"]      # queued is not completion
        _wait(m.marker.exists, limit=30)                               # the sim is inside the commit
        assert _web_store_handles(web.pid, (m.out, tmp_path / "data")) == []
        web.kill()
        web.wait(10)
        m.trigger.unlink()                                             # the save finishes, no web alive
        m.web()
        def acknowledged():
            try:
                a = _get(m.port, f"/api/command_ack?command_id={reply['command_id']}", timeout=2)[1]
            except OSError:
                return None
            return a if a.get("state") == "acknowledged" else None
        ack = _wait(acknowledged, limit=60)
        assert ack["ack"]["status"] == "ok"                            # durable ack survives web death
        now = _store_files(m.out)
        assert all(now.get(k) == v for k, v in good.items())           # earlier checkpoints untouched
        assert len(now) > len(good)                                    # and the new one committed
        st = m.status()
        assert st["status"] == "online" and st["result_validity"]["state"] != "complete"
    finally:
        m.close()


def test_sim_killed_mid_save_keeps_the_last_good_checkpoint_and_marks_the_run_incomplete(tmp_path):
    m = _MidSave(tmp_path)
    try:
        sim = m.sim()
        m.web()
        first = m.online()
        assert m.save()["status"] == "ok"
        good = _store_files(m.out)
        m.trigger.write_text("pause")
        assert m.save()["status"] == "queued"
        _wait(m.marker.exists, limit=30)
        sim.kill()                                                     # dies inside the checkpoint commit
        sim.wait(10)
        down = _wait(lambda: (lambda b: b if b and b["status"] == "error" else None)(m.status()), limit=10)
        assert down["halted"] is True and down["liveness"]["state"] == "sim_process_down"
        m.trigger.unlink()
        assert all(_store_files(m.out).get(k) == v for k, v in good.items())   # last good checkpoint intact
        m.sim()                                                        # restart on the same stores
        again = m.online()
        assert again["process_layout"]["sim"]["sim_pid"] != first["process_layout"]["sim"]["sim_pid"]
        validity = again["result_validity"]
        assert validity["state"] != "complete"
        assert first["result_validity"]["run_id"] in validity["other_runs_incomplete"] \
            or validity["state"] == "incomplete"
    finally:
        m.close()


# ---- rc2 Lane 1: run-scoped snapshots, status and pending requests (A2 + F5) ----

def _snap(link, seq, step, run_id):
    link._on_message({"t": "snap", "seq": seq, "step": step}, json.dumps({"run_id": run_id}).encode())


def test_a_paused_restart_reusing_seq_never_shows_the_old_run():
    """A2: rc1 keyed telemetry by seq only, so a restarted (paused) simulation that published
    seq 1 again kept serving the previous run's telemetry and status."""
    import socket as _socket
    link = split.SimLink("unused")
    proxy = split.SimProxy(link)
    a, b = _socket.socketpair()
    link._on_connect(a)
    _snap(link, 1, 9, "old")
    link._on_message({"t": "state", "state": {"run_id": "old", "pid": 1, "paused": True}}, b"")
    assert proxy.latest_telemetry == {"run_id": "old"}
    # Astra's exact in-memory control: a replacement frame with the same seq on the same link.
    _snap(link, 1, 0, "new-same-link")
    assert proxy.latest_telemetry == {"run_id": "new-same-link"}
    # The simulation process dies: nothing of its run is shown as current.
    link._on_disconnect("test")
    assert link.published is None and link.state == {} and proxy.latest_telemetry is None
    assert proxy.command_ack_lookup(command_id="c-1")["daemon_run_id"] is None
    # It restarts paused and publishes seq 1 again.
    c, d = _socket.socketpair()
    link._on_connect(c)
    assert link.published is None and link.state == {}
    _snap(link, 1, 0, "restarted")
    assert proxy.latest_telemetry == {"run_id": "restarted"}
    assert link.published.epoch == link.epoch
    for s in (a, b, c, d):
        s.close()


def test_a_pending_command_across_a_disconnect_resolves_as_unknown():
    """F5: a command queued by a process that then dies is never reported as queued, and no
    request can be inserted after the disconnect sweep."""
    import socket as _socket
    link = split.SimLink("unused")
    proxy = split.SimProxy(link)
    a, b = _socket.socketpair()
    link._on_connect(a)
    link._on_message({"t": "state", "state": {"run_id": "run-A"}}, b"")
    result = {}

    def dispatch():
        result.update(proxy.dispatch_command({"action": "set_speed", "client_command_id": "k1"}))
    t = threading.Thread(target=dispatch)
    t.start()
    header, _ = split.recv_msg(b)
    link._on_message({"t": "queued", "req": header["req"], "command_id": "cmd-7"}, b"")
    link._on_disconnect("killed")
    t.join(10)
    assert result["status"] == "error" and result["applied"] is None
    assert "outcome is unknown" in result["message"]
    assert result["command_id"] == "cmd-7" and result["daemon_run_id"] == "run-A"
    assert link.pending_commands() == 0
    assert link.request({"t": "cmd", "cmd": {}}) is None          # down: refused, nothing stranded
    assert link.pending_commands() == 0
    for s in (a, b):
        s.close()
