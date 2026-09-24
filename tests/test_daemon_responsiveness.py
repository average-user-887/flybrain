"""A slow computer must read as slow, not broken.

On a CPU-only laptop one 20 ms daemon step of the v3 connectome takes seconds of
wall time.  These tests slow every step with the real compiled LIF kernel (the
``SlowStepBallast``) and check that the HTTP API, commands and the SSE stream keep
answering promptly, and that slowing the steps does not change the simulation.
"""
import json
import random
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

import neurofly_daemon as nd
from brainlab.brain import Brain
from brainlab.graph_identity import synthetic_test_graph
from stream_gateway import StreamGateway, StreamPolicy
from tests.test_timing_determinism import CHECK_STEPS, PARADIGM, SCHEDULE, _capture, compare

SLOW_STEP_MS = 2500.0


def _get(url, timeout=5.0):
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as res:
        body = json.loads(res.read())
    return body, time.perf_counter() - start


def _post(url, payload, timeout=5.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as res:
        body = json.loads(res.read())
    return body, time.perf_counter() - start


def _wait(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        value = pred()
        if value:
            return value
        time.sleep(0.02)
    return None


def test_lif_kernel_releases_the_gil():
    """While the compiled kernel runs, other Python threads keep running."""
    arrays, _, _ = synthetic_test_graph(n=60000, k_out=40, seed=1)
    brain = Brain(arrays=arrays, dynamics="v3", backend="cpu")
    drive = np.full(brain.n, 30.0, dtype=np.float32)
    brain.step(drive, 0.1)                     # compile outside the measurement
    start = time.perf_counter()
    brain.step(drive, 1.0)
    per_ms = time.perf_counter() - start
    duration_ms = float(max(1, min(200, round(0.4 / max(per_ms, 1e-4)))))  # ~0.4 s of kernel

    worker = threading.Thread(target=brain.step, args=(drive, duration_ms))
    gaps, last = [], time.perf_counter()
    start = time.perf_counter()
    worker.start()
    while worker.is_alive():
        time.sleep(0.005)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    kernel_s = time.perf_counter() - start
    assert kernel_s > 0.2, kernel_s
    # With the GIL held the main thread would be frozen for the whole call.
    assert max(gaps) < 0.1, (max(gaps), kernel_s)


@pytest.fixture
def slow_served(tmp_path):
    runner = nd.ContinuousExperimentRunner(initial_paradigm="wind-tunnel", sim_speed=1.0,
                                           checkpoint_interval=3600, output_dir=tmp_path)
    runner.step_hook = nd.SlowStepBallast(SLOW_STEP_MS)
    # Well below one slow step, whatever this machine's exact step time.
    runner.command_reply_wait_s = 0.3
    orig = nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway
    nd.NeuroflyHTTPHandler.runner = runner
    nd.NeuroflyHTTPHandler.gateway = StreamGateway(StreamPolicy())
    server = ThreadingHTTPServer(("127.0.0.1", 0), nd.NeuroflyHTTPHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    runner.start()
    yield runner, f"http://127.0.0.1:{server.server_address[1]}"
    runner.running = False
    runner._wake.set()
    runner.sim_thread.join(timeout=10)
    server.shutdown()
    server.server_close()
    nd.NeuroflyHTTPHandler.runner, nd.NeuroflyHTTPHandler.gateway = orig


def test_slow_step_does_not_delay_api_commands_or_stream(slow_served):
    runner, base = slow_served
    # Be inside a long step: the simulation lock is held for ~1.5 s.
    assert _wait(lambda: runner.step_in_progress_s() > 0.1, 10.0)
    assert runner.lock.locked()

    status, took = _get(base + "/api/status")
    assert took < 0.5, took
    assert status["timing"]["step_in_progress_s"] > 0

    # Views that need a consistent state: the first read may wait for a step
    # boundary; once requested, later reads never wait behind a step.
    _get(base + "/api/manifest")
    _get(base + "/api/observatory")
    assert _wait(lambda: runner.step_in_progress_s() > 0.1, 10.0)
    for path in ("/api/manifest", "/api/observatory"):
        _, took = _get(base + path)
        assert took < 0.5, (path, took)

    # A command answers within the reply budget even though the step is not done.
    assert _wait(lambda: 0.05 < runner.step_in_progress_s() < 0.2, 10.0)
    reply, took = _post(base + "/api/command", {"action": "set_param", "name": "windVelocity", "value": 25})
    assert took < runner.command_reply_wait_s + 0.5, took
    assert reply["status"] == "queued" and reply["applied"] is False, reply
    command_id = reply["command_id"]

    # It is applied at the next step boundary, never dropped, and the
    # acknowledgement arrives in the published stream frames.
    def acked():
        snap = runner.published
        if snap is None:
            return None
        frame = json.loads(snap.data)
        return next((a for a in frame.get("command_acks", []) if a.get("command_id") == command_id), None)
    ack = _wait(acked, 3 * SLOW_STEP_MS / 1e3 + 2.0)
    assert ack and ack["status"] == "ok", ack
    assert ack["ack"]["applied"] is True and ack["ack"]["latency_ms"] > 300

    # The SSE heartbeat says a step is in progress instead of going silent.
    with urllib.request.urlopen(base + "/api/stream", timeout=5.0) as stream:
        start, beat = time.perf_counter(), None
        while time.perf_counter() - start < 3 * SLOW_STEP_MS / 1e3:
            line = stream.readline().decode()
            if line.startswith("event: heartbeat"):
                beat = json.loads(stream.readline().decode()[len("data: "):])
                if beat.get("step_in_progress_s", 0) > 0:
                    break
        assert beat and beat["step_in_progress_s"] > 0, beat
    timing = runner.timing_snapshot()
    assert timing["last_step_wall_s"] > 0.5, timing
    assert 0 < timing["achieved_speed"] < 0.1, timing


def _run(directory, ballast_ms):
    random.seed(20260919)
    np.random.seed(20260919)
    runner = nd.ContinuousExperimentRunner(initial_paradigm=PARADIGM, sim_speed=100.0,
                                           checkpoint_interval=3600, output_dir=Path(directory))
    ballast = nd.SlowStepBallast(ballast_ms) if ballast_ms else None
    captured = {}

    def hook(r):
        if ballast is not None:
            ballast(r)
        if r.total_steps in CHECK_STEPS:
            captured[r.total_steps] = _capture(r)

    runner.step_hook = hook
    runner.stop_at_step = CHECK_STEPS[-1]
    for step, cmd in SCHEDULE:
        runner.schedule_command(step, cmd)
    runner.start()
    try:
        _wait(lambda: len(captured) == len(CHECK_STEPS), 120.0)
    finally:
        runner.running = False
        runner._wake.set()
        runner.sim_thread.join(timeout=10)
    return captured, runner.sched_stats["rebases"]


def test_slow_steps_do_not_change_the_simulation(tmp_path):
    """Same seed and interventions: bit-identical state with and without slow steps."""
    reference, _ = _run(tmp_path / "fast", 0)
    slowed, rebases = _run(tmp_path / "slow", 25.0)
    assert set(reference) == set(CHECK_STEPS)
    assert rebases > 0          # the slowed run really fell behind its schedule
    assert compare(reference, slowed) == []
