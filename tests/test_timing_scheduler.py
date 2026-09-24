"""Scheduler, snapshot delivery and command acknowledgement (work package 2)."""
import json
import socket
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import neurofly_daemon as nd
from stream_gateway import StreamGateway, StreamPolicy


@pytest.fixture
def runner(tmp_path):
    r = nd.ContinuousExperimentRunner(initial_paradigm="wind-tunnel", sim_speed=20.0,
                                      checkpoint_interval=3600, output_dir=tmp_path)
    yield r
    r.running = False
    r._wake.set()
    if hasattr(r, "sim_thread"):
        r.sim_thread.join(timeout=5)


@pytest.fixture
def served(runner):
    """The real handler on an ephemeral 127.0.0.1 port, backed by ``runner``."""
    orig_runner, orig_gateway = nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway
    nd.NeuroflyHTTPHandler.runner = runner
    nd.NeuroflyHTTPHandler.gateway = StreamGateway(StreamPolicy())
    server = ThreadingHTTPServer(("127.0.0.1", 0), nd.NeuroflyHTTPHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", server.server_address[1]
    runner.running = False
    server.shutdown()
    server.server_close()
    nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway = orig_runner, orig_gateway


class _FakeClock:
    """Simulated wall clock for the scheduler: waits advance it, steps cost ~nothing.

    The scheduler's pacing is then independent of how fast (or how loaded) the test
    machine is, so these tests check the scheduling logic, not host speed.  Each read
    advances the clock by 1 us, as a real clock does between reads; a frozen clock
    lets the loop spin on a deadline that float rounding puts a hair in the future.
    """

    TICK_S = 1e-6

    def __init__(self):
        self.t = 1000.0
        self._lock = threading.Lock()

    def perf_counter(self):
        with self._lock:
            self.t += self.TICK_S
            return self.t

    monotonic = time = perf_counter

    def sleep(self, seconds):
        with self._lock:
            self.t += max(0.0, seconds)
        time.sleep(0)               # still hand the GIL over, like the real sleep


class _FakeWake:
    """``threading.Event`` whose timed wait advances the fake clock instead of blocking."""

    def __init__(self, clock):
        self._clock, self._event = clock, threading.Event()

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def is_set(self):
        return self._event.is_set()

    def wait(self, timeout=None):
        if not self._event.is_set() and timeout is not None:
            self._clock.sleep(timeout)
        return self._event.is_set()


@pytest.fixture
def fake_clock(runner, monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(nd, "time", clock)
    runner._wake = _FakeWake(clock)
    return clock


def _run_for(runner, clock, sim_wall_s):
    """Run the loop until ``sim_wall_s`` of fake wall time has passed, then stop it."""
    start = clock.t
    runner.start()
    assert _wait(lambda: clock.t - start >= sim_wall_s, timeout=120.0), "scheduler stalled"
    timing = runner.timing_snapshot()
    runner.running = False
    runner._wake.set()
    runner.sim_thread.join(timeout=5)
    return timing


def _wait(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_achieved_speed_tracks_requested_without_overshoot(runner, fake_clock):
    timing = _run_for(runner, fake_clock, 2.5)
    assert timing["requested_speed"] == 20.0
    assert timing["integration_dt_s"] == 0.02
    # The old threshold pacing overshot 20x to ~23.7x; deadlines must not.  On the fake
    # clock steps are free, so the scheduler alone decides the speed and must hit it.
    assert 19.5 <= timing["achieved_speed"] <= 20.6, timing
    assert timing["overloaded"] is False and timing["schedule_rebases"] == 0


def test_overload_is_visible_and_never_bursts(runner, fake_clock):
    original = runner.step_once

    def slow_step(publish=True):
        fake_clock.sleep(0.002)    # ~10x is the most this "machine" can do
        return original(publish)

    runner.step_once = slow_step
    runner.sim_speed = 100.0
    timing = _run_for(runner, fake_clock, 2.5)
    assert timing["requested_speed"] == 100.0
    assert timing["achieved_speed"] < 15.0, timing
    assert timing["overloaded"] is True
    assert timing["schedule_rebases"] > 0 and timing["forgiven_wall_s"] > 0
    # Batches stay short so readers and commands are served.
    assert timing["max_batch_hold_ms"] < 40.0


def test_command_ack_reports_applied_step_while_running(runner):
    runner.start()
    assert _wait(lambda: runner.total_steps > 20)
    before = runner.total_steps
    res = runner.dispatch_command({"action": "set_param", "name": "windVelocity", "value": 25})
    assert res["status"] == "ok"
    ack = res["ack"]
    assert ack["applied"] is True and ack["applied_step"] >= before
    assert ack["latency_ms"] < 250.0
    bad = runner.dispatch_command({"action": "set_speed", "speed": "fast"})
    assert bad["status"] == "error" and bad["ack"]["applied"] is False


def test_pause_freezes_clock_and_weights_but_keeps_metrics(runner):
    runner.start()
    assert _wait(lambda: runner.total_steps > 60 and runner.latest_telemetry.get("metrics"))
    res = runner.dispatch_command({"action": "set_paused", "paused": True})
    paused_step = res["ack"]["applied_step"]
    weights = runner.arena.fly.circuit.w.copy()
    time.sleep(0.5)
    assert runner.total_steps == paused_step
    assert np.array_equal(runner.arena.fly.circuit.w, weights)
    tel = json.loads(runner.published.data)
    assert tel["paused"] is True and tel["step"] == paused_step
    assert tel["metrics"], "measured assay metrics must survive a pause"
    assert tel["timing"]["achieved_speed"] == 0.0


def test_delivery_does_not_need_the_simulation_lock(runner, served):
    base, port = served
    runner.start()
    assert _wait(lambda: runner.published is not None and runner.total_steps > 10)
    with runner.lock:          # a stuck simulation step
        t0 = time.perf_counter()
        with urllib.request.urlopen(base + "/api/status", timeout=3) as r:
            status = json.loads(r.read())
        with urllib.request.urlopen(base + "/api/telemetry", timeout=3) as r:
            tel = json.loads(r.read())
        s = socket.create_connection(("127.0.0.1", port), timeout=3)
        s.sendall(b"GET /api/stream HTTP/1.1\r\nHost: x\r\n\r\n")
        buf = b""
        while b"data: {" not in buf:
            buf += s.recv(65536)
        s.close()
        elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert status["timing"]["requested_speed"] == 20.0 and "lock_profile" in status
    assert tel["type"] == "telemetry" and "timing" in tel and "path" in tel


def test_sse_decimates_latest_value_and_reports_counts(runner, served):
    base, port = served
    runner.publish_hz = 60.0
    runner.start()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"GET /api/stream HTTP/1.1\r\nHost: x\r\n\r\n")
    buf, end = b"", time.time() + 3.0
    while time.time() < end and b"event: stream" not in buf:
        buf += s.recv(65536)
    while time.time() < end:
        buf += s.recv(65536)
    s.close()
    seqs = [int(line[4:]) for line in buf.split(b"\n") if line.startswith(b"id: ")]
    assert len(seqs) > 10 and seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    stats = [json.loads(chunk.split(b"data: ", 1)[1].split(b"\n", 1)[0])
             for chunk in buf.split(b"event: stream\n")[1:]]
    assert stats and "decimated_snapshots" in stats[0]


def test_path_holds_measured_steps_of_current_segment(runner):
    for _ in range(30):
        with runner.lock:
            runner.step_once(publish=False)
    tel = runner._assemble_telemetry(runner._last_step_result)
    steps = [p[0] for p in tel["path"]]
    assert steps == list(range(steps[0], runner.total_steps + 1))
    assert tel["path"][-1][1] == round(float(runner.arena.fly.pos.x), 4)
    runner.dispatch_command({"action": "reset_trial"})
    assert runner._assemble_telemetry({})["path"] == []
