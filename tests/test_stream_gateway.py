"""Tests for the public-stream gateway (stream_gateway.py) and its daemon wiring.

Covers:
1. StreamPolicy defaults reproduce the historical private behaviour.
2. Environment and CLI layering, token validation.
3. Command authorisation (constant-time bearer check) and SSE slot capping.
4. End-to-end HTTP behaviour of neurofly_daemon.NeuroflyHTTPHandler in
   private and public mode: 403 without token, 200 with token, 503 past the
   client cap, CORS origin header, status exposure, oversized bodies.
"""

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stream_gateway import (  # noqa: E402
    MAX_COMMAND_BYTES,
    MIN_ADMIN_TOKEN_LENGTH,
    SSESlot,
    StreamGateway,
    StreamPolicy,
    env_flag,
)

TOKEN = "unit-test-admin-token-0123456789"


# =============================================================================
# 1. Policy construction
# =============================================================================

class TestStreamPolicy:
    def test_defaults_are_private_and_unchanged(self):
        p = StreamPolicy()
        assert p.public is False
        assert p.read_only is False
        assert p.admin_token is None
        assert p.max_stream_clients == 0
        assert p.stream_hz == 30.0
        assert p.allowed_origin == "*"
        assert abs(p.stream_interval - 1 / 30) < 1e-9

    def test_from_env_empty_environment_is_private(self):
        p = StreamPolicy.from_env(environ={})
        assert p == StreamPolicy()

    def test_from_env_public_defaults(self):
        p = StreamPolicy.from_env(environ={"NEUROFLY_PUBLIC": "1"})
        assert p.public and p.read_only
        assert p.max_stream_clients == 50
        assert p.stream_hz == 10.0

    def test_from_env_token_and_overrides(self):
        env = {
            "NEUROFLY_PUBLIC": "true",
            "NEUROFLY_ADMIN_TOKEN": TOKEN,
            "NEUROFLY_MAX_STREAM_CLIENTS": "7",
            "NEUROFLY_STREAM_HZ": "4",
            "NEUROFLY_ALLOWED_ORIGIN": "https://example.org",
        }
        p = StreamPolicy.from_env(environ=env)
        assert p.public and not p.read_only
        assert p.admin_token == TOKEN
        assert p.max_stream_clients == 7
        assert p.stream_hz == 4.0
        assert p.allowed_origin == "https://example.org"

    def test_cli_overrides_beat_environment(self):
        env = {"NEUROFLY_PUBLIC": "1", "NEUROFLY_MAX_STREAM_CLIENTS": "7", "NEUROFLY_STREAM_HZ": "4"}
        p = StreamPolicy.from_env(public=False, max_stream_clients=3, stream_hz=20, environ=env)
        assert p.public is False
        assert p.max_stream_clients == 3
        assert p.stream_hz == 20.0

    def test_short_token_rejected(self):
        with pytest.raises(ValueError):
            StreamPolicy(public=True, admin_token="x" * (MIN_ADMIN_TOKEN_LENGTH - 1))

    def test_blank_token_means_no_token(self):
        p = StreamPolicy(public=True, admin_token="   ")
        assert p.admin_token is None and p.read_only

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            StreamPolicy(max_stream_clients=-1)
        with pytest.raises(ValueError):
            StreamPolicy(stream_hz=0.0)

    def test_describe_never_leaks_token(self):
        p = StreamPolicy(public=True, admin_token=TOKEN)
        d = p.describe()
        assert TOKEN not in json.dumps(d)
        assert d["commands_require_token"] is True
        assert d["read_only"] is False

    def test_env_flag_parsing(self):
        assert env_flag("X", environ={"X": "yes"})
        assert env_flag("X", environ={"X": "ON"})
        assert not env_flag("X", environ={"X": "0"})
        assert not env_flag("X", environ={})
        assert env_flag("X", default=True, environ={"X": ""})


# =============================================================================
# 2. Gateway runtime behaviour
# =============================================================================

class TestStreamGateway:
    def test_private_mode_authorises_everything(self):
        g = StreamGateway(StreamPolicy())
        assert g.authorize_command({}) is True
        assert g.authorize_command({"Authorization": "Bearer nonsense"}) is True

    def test_public_read_only_rejects_all_commands(self):
        g = StreamGateway(StreamPolicy(public=True))
        assert g.authorize_command({}) is False
        assert g.authorize_command({"Authorization": "Bearer " + TOKEN}) is False
        assert g.command_rejection()["error"] == "read_only"
        assert g.stats()["rejected_commands"] == 2

    def test_public_token_gate(self):
        g = StreamGateway(StreamPolicy(public=True, admin_token=TOKEN))
        assert g.authorize_command({"Authorization": "Bearer " + TOKEN}) is True
        assert g.authorize_command({"authorization": "bearer " + TOKEN}) is True
        assert g.authorize_command({"Authorization": "Bearer " + TOKEN[:-1]}) is False
        assert g.authorize_command({"Authorization": "Basic " + TOKEN}) is False
        assert g.authorize_command({"Authorization": TOKEN}) is False
        assert g.authorize_command({}) is False
        assert g.command_rejection()["error"] == "unauthorized"

    def test_slot_cap_and_release(self):
        g = StreamGateway(StreamPolicy(public=True, max_stream_clients=2))
        a = g.acquire_stream_slot()
        b = g.acquire_stream_slot()
        c = g.acquire_stream_slot()
        assert bool(a) and bool(b) and not bool(c)
        assert g.active_streams == 2
        assert g.stats()["rejected_stream_clients"] == 1
        a.release()
        a.release()  # idempotent
        assert g.active_streams == 1
        with g.acquire_stream_slot() as d:
            assert bool(d)
            assert g.active_streams == 2
        assert g.active_streams == 1
        c.release()  # releasing a denied slot is a no-op
        assert g.active_streams == 1
        b.release()
        assert g.active_streams == 0

    def test_unlimited_slots_by_default(self):
        g = StreamGateway()
        slots = [g.acquire_stream_slot() for _ in range(200)]
        assert all(bool(s) for s in slots)
        for s in slots:
            s.release()
        assert g.active_streams == 0

    def test_pace_sleeps_to_interval(self):
        g = StreamGateway(StreamPolicy(stream_hz=50))
        t0 = time.monotonic()
        last = g.pace(0.0)          # first call: nothing to wait for
        assert last - t0 < 0.01
        t1 = g.pace(last)           # second call: waits ~20 ms
        assert t1 - last >= 0.015

    def test_sse_slot_type(self):
        g = StreamGateway()
        assert isinstance(g.acquire_stream_slot(), SSESlot)


# =============================================================================
# 3. Daemon HTTP integration
# =============================================================================

class _StubRunner:
    """Minimal stand-in for ContinuousExperimentRunner (no Arena needed)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.start_time = time.time()
        self.total_steps = 5
        self.sim_speed = 10.0
        self.active_paradigm_id = "open-arena"
        self.active_paradigm_title = "Open Arena Assay"
        self.current_trial = 1
        self.trial_history = []
        self.learning_curve = []
        self.latest_telemetry = {"type": "telemetry", "step": 5, "fly": {"x": 1.0, "y": 2.0}}
        self.commands = []

    def dispatch_command(self, cmd):
        self.commands.append(cmd)
        return {"status": "ok", "echo": cmd.get("action")}


@pytest.fixture
def daemon_server():
    """Start NeuroflyHTTPHandler on an ephemeral port with a given policy."""
    import neurofly_daemon as nd

    created = []
    original_gateway = nd.NeuroflyHTTPHandler.gateway
    original_runner = nd.NeuroflyHTTPHandler.runner

    def _start(policy: StreamPolicy):
        runner = _StubRunner()
        nd.NeuroflyHTTPHandler.runner = runner
        nd.NeuroflyHTTPHandler.gateway = StreamGateway(policy)
        server = ThreadingHTTPServer(("127.0.0.1", 0), nd.NeuroflyHTTPHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        created.append((server, runner))
        return f"http://127.0.0.1:{server.server_address[1]}", runner

    yield _start

    for server, runner in created:
        runner.running = False
        server.shutdown()
        server.server_close()
    nd.NeuroflyHTTPHandler.gateway = original_gateway
    nd.NeuroflyHTTPHandler.runner = original_runner


def _post(url, payload, headers=None, timeout=3.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8")), dict(err.headers)


def _get_json(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers)


def _open_sse(host, port, timeout=3.0):
    """Open a raw SSE connection and return (socket, status_line)."""
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(b"GET /api/stream HTTP/1.1\r\nHost: x\r\nAccept: text/event-stream\r\n\r\n")
    head = s.recv(4096)
    return s, head.split(b"\r\n", 1)[0].decode()


class TestDaemonPrivateModeUnchanged:
    def test_command_allowed_and_cors_star(self, daemon_server):
        base, runner = daemon_server(StreamPolicy())
        status, body, headers = _post(base + "/api/command", {"action": "set_speed", "speed": 5})
        assert status == 200 and body["status"] == "ok"
        assert runner.commands and runner.commands[0]["action"] == "set_speed"
        assert headers.get("Access-Control-Allow-Origin") == "*"

    def test_status_exposes_stream_policy(self, daemon_server):
        base, _ = daemon_server(StreamPolicy())
        status, body, _ = _get_json(base + "/api/status")
        assert status == 200
        assert body["stream"]["public"] is False
        assert body["stream"]["read_only"] is False
        assert body["stream"]["max_stream_clients"] == 0

    def test_sse_delivers_telemetry(self, daemon_server):
        base, _ = daemon_server(StreamPolicy())
        host, port = base.replace("http://", "").split(":")
        s, status_line = _open_sse(host, int(port))
        try:
            assert status_line.endswith("200 OK")
            buf = b""
            deadline = time.time() + 3
            while b"data: " not in buf and time.time() < deadline:
                buf += s.recv(4096)
            assert b'"type": "telemetry"' in buf
        finally:
            s.close()


class TestDaemonPublicMode:
    def test_read_only_rejects_command(self, daemon_server):
        base, runner = daemon_server(StreamPolicy(public=True))
        status, body, _ = _post(base + "/api/command", {"action": "set_speed", "speed": 5})
        assert status == 403 and body["error"] == "read_only"
        assert runner.commands == []
        status, body, _ = _post(base + "/api/command", {"action": "set_speed"},
                                headers={"Authorization": "Bearer " + TOKEN})
        assert status == 403  # no token configured: nothing unlocks commands

    def test_token_gate(self, daemon_server):
        base, runner = daemon_server(StreamPolicy(public=True, admin_token=TOKEN))
        status, body, _ = _post(base + "/api/command", {"action": "reset_trial"})
        assert status == 403 and body["error"] == "unauthorized"
        status, body, _ = _post(base + "/api/command", {"action": "reset_trial"},
                                headers={"Authorization": "Bearer wrong-token-wrong-token"})
        assert status == 403
        status, body, _ = _post(base + "/api/command", {"action": "reset_trial"},
                                headers={"Authorization": "Bearer " + TOKEN})
        assert status == 200 and body["status"] == "ok"
        assert len(runner.commands) == 1

    def test_reads_still_work_and_status_flags_public(self, daemon_server):
        base, _ = daemon_server(StreamPolicy(public=True, allowed_origin="https://viewer.example"))
        status, body, headers = _get_json(base + "/api/status")
        assert status == 200 and body["stream"]["public"] is True and body["stream"]["read_only"] is True
        assert headers.get("Access-Control-Allow-Origin") == "https://viewer.example"
        status, body, _ = _get_json(base + "/api/telemetry")
        assert status == 200 and body["type"] == "telemetry"
        status, body, _ = _get_json(base + "/api/paradigms")
        assert status == 200 and len(body["paradigms"]) >= 14

    def test_sse_client_cap_returns_503_then_recovers(self, daemon_server):
        base, _ = daemon_server(StreamPolicy(public=True, max_stream_clients=2, stream_hz=50))
        host, port = base.replace("http://", "").split(":")
        port = int(port)
        a, la = _open_sse(host, port)
        b, lb = _open_sse(host, port)
        try:
            assert la.endswith("200 OK") and lb.endswith("200 OK")
            c, lc = _open_sse(host, port)
            c.close()
            assert lc.endswith("503 Service Unavailable")
        finally:
            a.close()
            b.close()
        # Slots are reclaimed once the server notices the closed sockets.
        deadline = time.time() + 5
        ok = False
        while time.time() < deadline:
            d, ld = _open_sse(host, port)
            d.close()
            if ld.endswith("200 OK"):
                ok = True
                break
            time.sleep(0.1)
        assert ok, "slot was not released after clients disconnected"

    def test_oversized_command_body_rejected(self, daemon_server):
        base, runner = daemon_server(StreamPolicy())
        big = {"action": "set_param", "name": "x", "value": "y" * (MAX_COMMAND_BYTES + 10)}
        status, body, _ = _post(base + "/api/command", big)
        assert status == 413
        assert runner.commands == []
