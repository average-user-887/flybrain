"""Simulation and web server as separate processes (standard library only).

The simulation process (``--process-mode sim``) owns the runner: brain, arena,
recorder, checkpoints and the command queue.  It is the single writer of all
scientific state and serves nothing over HTTP.  It publishes, over a local Unix
socket, every telemetry snapshot it already builds plus a small status record a
few times a second, and accepts commands and view requests.

The web process (``--process-mode web``) runs the unchanged
``NeuroflyHTTPHandler`` against :class:`SimProxy`, a read-only stand-in for the
runner built from the latest records it received.  It never holds the
simulation lock, never touches brain state and answers a command with
``queued`` plus the command id as soon as the simulation process has queued it;
the final acknowledgement follows in the stream (``command_acks``) as before.

If the simulation process dies or stops publishing, the web process reports
``status: error`` / ``halted: true`` with the reason, never ``online``.  If the
web process dies, the simulation process keeps running and recording, and the
launcher (``--process-mode split``, the default) starts a new web process.

Wire format: ``>II`` (header length, blob length), a JSON header, an optional
binary blob (the serialized snapshot, sent as is).
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

_HDR = struct.Struct(">II")
MAX_HEADER = 64 << 20
MAX_BLOB = 512 << 20

STATE_INTERVAL_S = 0.2      # status record cadence while no frame is published (sim -> web)
STATE_MAX_HZ = 30.0         # status record at most this often, following new frames
STALE_AFTER_S = 10.0        # connected but silent this long: reported as unresponsive
REPLY_WAIT_S = 0.25         # web: wait this long for a final command result, then "queued"
ACCEPT_WAIT_S = 5.0         # web: no command id from the sim within this: outcome unknown
VIEW_WAIT_S = 3.5           # web: runner.read_view waits at most ~3.2 s in the sim
SIM_STOP_TIMEOUT_S = 900.0  # launcher: time allowed for the final save on shutdown
WEB_BIND_FAILED = 3         # web exit code: the HTTP port cannot be bound (fatal for the launcher)
MAX_PENDING_COMMANDS = 64   # commands accepted but not yet applied (web and sim side each)
MAX_PENDING_VIEWS = 16      # concurrent view requests in the simulation process
LAUNCHER_ENV = "NEUROFLY_LAUNCHER_PID"
PARENT_POLL_S = 1.0


def bind_to_launcher(use_prctl: bool = True) -> Optional[int]:
    """Exit with the split launcher, even when it is SIGKILLed.

    Linux: PR_SET_PDEATHSIG delivers SIGTERM (a graceful stop: the simulation's
    final save) when the launcher dies.  Everywhere: a thread checks that the
    parent is still the launcher and sends the same SIGTERM if it is not.
    Returns the launcher pid, or None when not started by a launcher.
    """
    raw = os.environ.get(LAUNCHER_ENV)
    if not raw:
        return None
    launcher = int(raw)
    if use_prctl and sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(1, signal.SIGTERM, 0, 0, 0)          # PR_SET_PDEATHSIG = 1
        except (OSError, AttributeError):
            pass
    if os.getppid() != launcher:                           # died before prctl took effect
        os.kill(os.getpid(), signal.SIGTERM)

    def watch():
        while os.getppid() == launcher:
            time.sleep(PARENT_POLL_S)
        print(f"[{os.getpid()}] The split launcher {launcher} is gone; stopping.", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch, daemon=True, name="NeuroFly-launcher-watch").start()
    return launcher


def _short_runtime_dir() -> str:
    """A short directory for sockets: AF_UNIX paths are limited to ~107 bytes, so a long
    TMPDIR (e.g. on a data disk) must not decide where the socket lives."""
    for base in (os.environ.get("XDG_RUNTIME_DIR"), "/tmp", tempfile.gettempdir()):
        if base and os.path.isdir(base) and len(base) < 60:
            return base
    return tempfile.gettempdir()


def private_socket_dir() -> str:
    return tempfile.mkdtemp(prefix="neurofly-split-", dir=_short_runtime_dir())     # mode 0700


def default_socket_path(port: int) -> str:
    return os.path.join(_short_runtime_dir(), f"neurofly-sim-{int(port)}.sock")


def send_msg(sock: socket.socket, header: Dict[str, Any], blob: bytes = b"") -> None:
    head = json.dumps(header, default=str).encode("utf-8")
    sock.sendall(_HDR.pack(len(head), len(blob)) + head)
    if blob:
        sock.sendall(blob)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket):
    """(header, blob), or None at end of stream."""
    raw = _recv_exact(sock, _HDR.size)
    if raw is None:
        return None
    hlen, blen = _HDR.unpack(raw)
    if hlen > MAX_HEADER or blen > MAX_BLOB:
        raise ValueError(f"oversized IPC message ({hlen}, {blen})")
    head = _recv_exact(sock, hlen)
    blob = _recv_exact(sock, blen) if blen else b""
    if head is None or blob is None:
        return None
    return json.loads(head.decode("utf-8")), blob


# ---------------------------------------------------------------------------------------- sim side

def sim_state(runner) -> Dict[str, Any]:
    """Everything the status, heartbeat and timing views read, read the way the
    one-process HTTP handler reads it (plain attribute reads, never the sim lock)."""
    try:
        compute = runner.compute_info()
    except Exception as exc:  # noqa: BLE001 -- never let a device query stop publication
        compute = {"device": None, "error": f"{type(exc).__name__}: {exc}"}
    lock = getattr(runner, "lock", None)
    arena = getattr(runner, "arena", None)
    return {
        "pid": os.getpid(),
        "sent_wall": time.time(),
        "start_time": runner.start_time,
        "health": runner.health(),
        "last_error": runner.last_error,
        "error_detail": getattr(runner, "error_detail", None),
        "cleared_errors": list(getattr(runner, "cleared_errors", ())),
        "paused": runner.paused,
        "continuous": runner.continuous,
        "running": runner.running,
        "total_steps": runner.total_steps,
        "sim_speed": runner.sim_speed,
        "active_paradigm_id": runner.active_paradigm_id,
        "active_paradigm_title": runner.active_paradigm_title,
        "current_trial": copy.deepcopy(runner.current_trial),
        "trials_completed": len(runner.trial_history),
        "recording": runner.recording_status(),
        "trial_sim_time": float(getattr(runner, "trial_sim_time", 0.0)),
        "trial_length_s": getattr(runner, "trial_length_s", None),
        "world_bounds": list(getattr(arena, "world_bounds", ()) or ()),
        "identity": runner.identity(),
        "backend": getattr(runner, "backend", "modular"),
        "launch_backend": getattr(runner, "launch_backend", None),
        "compute": compute,
        "timing": runner.timing_snapshot(),
        "lock_profile": lock.profile() if hasattr(lock, "profile") else None,
        "step_in_progress_s": runner.step_in_progress_s(),
        "last_step_wall_s": runner.last_step_wall_s,
        "recordings_dir": str(runner.recordings_dir),
        "run_id": getattr(runner, "run_id", None),
    }


def view_builder(runner, key: str):
    """The same consistent views the one-process handler builds (do_GET)."""
    if key == "observatory":
        return lambda: {"brain": runner.active_brain.summary(details=True),
                        "telemetry": runner.latest_telemetry,
                        "brains": runner.brains.catalog(runner.active_paradigm_id)}
    if key == "manifest":
        return lambda: {"identity": runner.identity(),
                        "manifest": runner.manifest.to_dict() if getattr(runner, "manifest", None) else None}
    if key == "/api/brain":
        return lambda: runner.active_brain.summary(details=True)
    if key == "/api/brains":
        return lambda: {"active": runner.active_paradigm_id,
                        "brains": runner.brains.catalog(runner.active_paradigm_id)}
    raise KeyError(key)


class _Conn:
    """One web client.  The publisher never blocks on it: snapshots and status are
    latest-value-wins slots, replies are queued; a writer thread drains them."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.cv = threading.Condition()
        self.control: deque = deque()
        self.state = None
        self.snap = None
        self.closed = False
        threading.Thread(target=self._writer, daemon=True, name="NeuroFly-IPC-writer").start()

    def put(self, slot: str, msg) -> None:
        with self.cv:
            if slot == "control":
                self.control.append(msg)
            else:
                setattr(self, slot, msg)
            self.cv.notify()

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify()
        try:
            self.sock.close()
        except OSError:
            pass

    def _writer(self) -> None:
        while True:
            with self.cv:
                while not (self.closed or self.control or self.state or self.snap):
                    self.cv.wait()
                if self.closed:
                    return
                if self.control:
                    msg = self.control.popleft()
                elif self.state is not None:
                    msg, self.state = self.state, None
                else:
                    msg, self.snap = self.snap, None
            try:
                send_msg(self.sock, *msg)
            except OSError:
                self.close()
                return


class SimServer:
    """Publishes the runner over a Unix socket; applies commands through the runner's
    own queue (``dispatch_command``), so step-boundary semantics are unchanged."""

    def __init__(self, runner, path: str):
        self.runner = runner
        self.path = path
        self.conns: list = []
        self._conns_lock = threading.Lock()
        self._snap_event = threading.Event()
        self._stop = threading.Event()
        self.sock: Optional[socket.socket] = None
        self.state_errors = 0
        self._pending = {"cmd": 0, "view": 0}
        self._pending_lock = threading.Lock()
        self.refused = {"cmd": 0, "view": 0}

    def start(self) -> None:
        if os.path.exists(self.path):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(self.path)
            except OSError:
                os.unlink(self.path)        # stale socket of a dead simulation process
            else:
                probe.close()
                raise OSError(f"another simulation process already serves {self.path}")
            finally:
                probe.close()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.path)
        os.chmod(self.path, 0o600)
        self.sock.listen(8)
        # Wake the publisher the moment the simulation thread publishes a frame.
        original = self.runner._publish_snapshot

        def publish_and_notify():
            snap = original()
            self._snap_event.set()
            return snap
        self.runner._publish_snapshot = publish_and_notify
        threading.Thread(target=self._accept_loop, daemon=True, name="NeuroFly-IPC-accept").start()
        threading.Thread(target=self._publish_loop, daemon=True, name="NeuroFly-IPC-publish").start()

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass
        with self._conns_lock:
            conns, self.conns = list(self.conns), []
        for conn in conns:
            conn.close()

    def _state_msg(self):
        try:
            return ({"t": "state", "state": sim_state(self.runner)}, b"")
        except Exception as exc:  # noqa: BLE001 -- e.g. a dict resized mid-read; retry next tick
            self.state_errors += 1
            print(f"[Sim] status record failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return None

    @staticmethod
    def _snap_msg(snap):
        return ({"t": "snap", "seq": snap.seq, "step": snap.step, "wall_time": snap.wall_time}, snap.data)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _ = self.sock.accept()
            except OSError:
                return
            conn = _Conn(sock)
            state = self._state_msg()
            if state is not None:
                conn.put("state", state)
            snap = getattr(self.runner, "published", None)
            if snap is not None:
                conn.put("snap", self._snap_msg(snap))
            with self._conns_lock:
                self.conns.append(conn)
            threading.Thread(target=self._reader, args=(conn,), daemon=True, name="NeuroFly-IPC-reader").start()

    def _broadcast(self, slot: str, msg) -> None:
        with self._conns_lock:
            self.conns = [c for c in self.conns if not c.closed]
            conns = list(self.conns)
        for conn in conns:
            conn.put(slot, msg)

    def _publish_loop(self) -> None:
        last_seq = None
        last_state = 0.0
        while not self._stop.is_set():
            self._snap_event.wait(STATE_INTERVAL_S)
            self._snap_event.clear()
            snap = getattr(self.runner, "published", None)
            fresh = snap is not None and snap.seq != last_seq
            now = time.monotonic()
            # Status follows each frame (at most STATE_MAX_HZ) and is queued before it,
            # so /api/status is never older than the frame the page is showing.
            if now - last_state >= (1.0 / STATE_MAX_HZ if fresh else STATE_INTERVAL_S):
                last_state = now
                state = self._state_msg()
                if state is not None:
                    self._broadcast("state", state)
            if fresh:
                last_seq = snap.seq
                self._broadcast("snap", self._snap_msg(snap))

    def _reader(self, conn: _Conn) -> None:
        try:
            while True:
                msg = recv_msg(conn.sock)
                if msg is None:
                    break
                header, _ = msg
                kind = header.get("t")
                if kind == "ack":       # read-only, never the simulation lock: answer inline
                    self._handle_ack(conn, header)
                    continue
                target = {"cmd": self._handle_cmd, "view": self._handle_view}.get(kind)
                if target is None:
                    continue
                limit = MAX_PENDING_COMMANDS if kind == "cmd" else MAX_PENDING_VIEWS
                with self._pending_lock:
                    # Commands: bound the runner's own queue (accepted, not yet applied)
                    # as well as the requests still being handed to it.
                    queued = len(getattr(self.runner, "_commands", ())) if kind == "cmd" else 0
                    full = max(self._pending[kind], queued) >= limit
                    if full:
                        self.refused[kind] += 1
                    else:
                        self._pending[kind] += 1
                if full:
                    conn.put("control", (self._queue_full_reply(kind, header, limit), b""))
                    continue
                threading.Thread(target=self._run_counted, args=(kind, target, conn, header), daemon=True,
                                 name=f"NeuroFly-IPC-{kind}").start()
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    @staticmethod
    def _queue_full_reply(kind, header, limit):
        message = (f"simulation command queue full ({limit} commands not yet applied); "
                   f"the command was not applied, retry later")
        if kind == "cmd":
            return {"t": "result", "req": header.get("req"),
                    "result": {"status": "error", "applied": False, "queue_full": True,
                               "action": (header.get("cmd") or {}).get("action", ""), "message": message}}
        return {"t": "view", "req": header.get("req"), "view": None, "error": "view queue full"}

    def _run_counted(self, kind, target, conn, header) -> None:
        try:
            target(conn, header)
        finally:
            with self._pending_lock:
                self._pending[kind] -= 1

    def _handle_ack(self, conn: _Conn, header) -> None:
        try:
            ack = self.runner.command_ack_lookup(command_id=header.get("command_id"),
                                                 client_command_id=header.get("client_command_id"))
        except Exception as exc:  # noqa: BLE001
            ack = {"state": "unknown", "message": f"{type(exc).__name__}: {exc}"}
        conn.put("control", ({"t": "ack", "req": header.get("req"), "ack": ack}, b""))

    def _handle_cmd(self, conn: _Conn, header) -> None:
        req = header.get("req")
        try:
            result = self.runner.dispatch_command(
                header.get("cmd"),
                on_queued=lambda cid: conn.put("control", ({"t": "queued", "req": req, "command_id": cid}, b"")))
        except Exception as exc:  # noqa: BLE001 -- answer, never drop the request
            result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
        conn.put("control", ({"t": "result", "req": req, "result": result}, b""))

    def _handle_view(self, conn: _Conn, header) -> None:
        req, key = header.get("req"), header.get("key")
        try:
            view = self.runner.read_view(key, view_builder(self.runner, key))
            reply = {"t": "view", "req": req, "view": view}
        except Exception as exc:  # noqa: BLE001
            reply = {"t": "view", "req": req, "view": None, "error": f"{type(exc).__name__}: {exc}"}
        conn.put("control", (reply, b""))


def serve_sim(runner, cleanup, args) -> int:
    """Main thread of the headless simulation process."""
    path = args.ipc_socket or default_socket_path(args.port)
    server = SimServer(runner, path)
    try:
        server.start()
    except OSError as exc:
        print(f"[Sim] Cannot open the IPC socket {path}: {exc}", file=sys.stderr, flush=True)
        cleanup()
        return 2
    stop = threading.Event()
    received = []

    def _on_signal(signum, frame):
        received.append(signum)
        stop.set()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    print(f"[Sim] Headless simulation process {os.getpid()} publishing on {path}", flush=True)
    while not stop.wait(0.5):
        pass
    print(f"\n[Sim] Received signal {received[0]}. Initiating graceful shutdown...", flush=True)
    server.close()
    ok = cleanup()
    print("[Sim] Clean shutdown complete." if ok else
          "[Sim] Shutdown complete, but not everything was saved (exit code 1).", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------- web side

class QueueFull(Exception):
    pass


class _Pending:
    __slots__ = ("cv", "command_id", "result", "view", "done", "error", "kind", "epoch", "run_id")

    def __init__(self):
        self.cv = threading.Condition()
        self.command_id = None
        self.result = None
        self.view = None
        self.done = False
        self.error = None
        self.kind = None
        self.epoch = None       # connection epoch the request was sent on
        self.run_id = None      # run id of the simulation process that queued it


class SimLink:
    """The web process's connection to the simulation process (reconnects forever)."""

    def __init__(self, path: str):
        self.path = path
        self.sock: Optional[socket.socket] = None
        self.send_lock = threading.Lock()
        self.connected = False
        self.ever_connected = False
        self.state: Dict[str, Any] = {}
        self.state_rx: Optional[float] = None
        self.published = None
        self.down_reason = "simulation process has not connected yet (starting up, or not running)"
        self.down_since = time.time()
        self.connects = 0
        self._reqs = itertools.count(1)
        self._pending: Dict[int, _Pending] = {}
        self._plock = threading.Lock()
        # Connection epoch: bumped on every connect and disconnect.  Snapshots, status and
        # pending requests belong to one epoch; nothing crosses into the next one.
        self.epoch = 0

    def start(self) -> "SimLink":
        threading.Thread(target=self._run, daemon=True, name="NeuroFly-SimLink").start()
        return self

    def _run(self) -> None:
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.path)
            except OSError:
                sock.close()
                time.sleep(0.25)
                continue
            self._on_connect(sock)
            reason = "connection closed by the simulation process"
            try:
                while True:
                    msg = recv_msg(sock)
                    if msg is None:
                        break
                    self._on_message(*msg)
            except (OSError, ValueError) as exc:
                reason = f"connection lost: {type(exc).__name__}: {exc}"
            self._on_disconnect(reason)
            try:
                sock.close()
            except OSError:
                pass
            time.sleep(0.25)

    def _clear_run_state(self) -> None:
        # Never show the previous simulation run's snapshot or status as the current one.
        self.published, self.state, self.state_rx = None, {}, None

    def _on_connect(self, sock) -> None:
        with self._plock:
            self.epoch += 1
            self._clear_run_state()
            self.sock, self.connected, self.ever_connected = sock, True, True
            self.connects += 1

    def _on_disconnect(self, reason: str) -> None:
        with self._plock:
            self.connected = False
            self.epoch += 1
            self.down_reason, self.down_since = f"simulation process is down ({reason})", time.time()
            self._clear_run_state()
            pending, self._pending = list(self._pending.values()), {}
        for p in pending:
            with p.cv:
                p.error, p.done = self.down_reason, True
                p.cv.notify_all()

    def _on_message(self, header, blob) -> None:
        kind = header.get("t")
        if kind == "snap":
            self.published = SimpleNamespace(seq=header["seq"], step=header["step"], data=blob,
                                             wall_time=header.get("wall_time"), epoch=self.epoch)
        elif kind == "state":
            self.state, self.state_rx = header["state"], time.monotonic()
        else:
            with self._plock:
                p = self._pending.get(header.get("req"))
                if p is None:
                    return
                if kind != "queued":
                    self._pending.pop(header.get("req"), None)
            with p.cv:
                if kind == "queued":
                    p.command_id = header.get("command_id")
                    p.run_id = self.state.get("run_id")
                elif kind == "result":
                    p.result, p.done = header.get("result"), True
                elif kind == "view":
                    p.view, p.error, p.done = header.get("view"), header.get("error"), True
                elif kind == "ack":
                    p.view, p.done = header.get("ack"), True
                p.cv.notify_all()

    def state_age_s(self) -> Optional[float]:
        return None if self.state_rx is None else time.monotonic() - self.state_rx

    def problem(self) -> Optional[str]:
        """None while the simulation process is connected and publishing."""
        if not self.connected:
            return self.down_reason
        age = self.state_age_s()
        if age is None:
            return "simulation process connected but has not published its status yet"
        if age > STALE_AFTER_S:
            return f"simulation process is not responding: no status for {age:.0f} s"
        return None

    def pending_commands(self) -> int:
        with self._plock:
            return sum(1 for p in self._pending.values() if p.kind == "cmd")

    def request(self, header: Dict[str, Any], limit: Optional[int] = None) -> Optional[_Pending]:
        """Send one request; None when not connected.  With ``limit``, raises
        :class:`QueueFull` instead of exceeding that many pending requests of this kind."""
        p = _Pending()
        p.kind = header.get("t")
        req = next(self._reqs)
        # The connected check and the insert are one step under _plock, so a request can
        # never be inserted after a disconnect's sweep and stranded (F5).
        with self._plock:
            if not self.connected:
                return None
            if limit is not None and sum(1 for q in self._pending.values() if q.kind == p.kind) >= limit:
                raise QueueFull(limit)
            self._pending[req] = p
            p.epoch, sock = self.epoch, self.sock
        try:
            with self.send_lock:
                send_msg(sock, dict(header, req=req))
        except OSError:
            with self._plock:
                self._pending.pop(req, None)
            return None
        return p

    def describe(self) -> Dict[str, Any]:
        age = self.state_age_s()
        return {"connected": self.connected, "problem": self.problem(), "socket": self.path,
                "sim_pid": self.state.get("pid"), "connects": self.connects,
                "state_age_s": None if age is None else round(age, 3),
                "down_since": None if self.connected else round(self.down_since, 3)}


class _NoLock:
    """The web process never holds the simulation lock; this only satisfies
    ``with runner.lock`` fallbacks for runners without published snapshots."""

    def __init__(self, proxy):
        self._proxy = proxy

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def profile(self):
        return self._proxy.state.get("lock_profile")


class SimProxy:
    """Read-only runner stand-in for ``NeuroflyHTTPHandler`` in the web process."""

    running = True

    def __init__(self, link: SimLink):
        self.link = link
        self.lock = _NoLock(self)
        self._web_start = time.time()
        self._telemetry_cache = (None, None)

    # -- plain fields ----------------------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        return self.link.state

    def _get(self, key, default=None):
        return self.link.state.get(key, default)

    start_time = property(lambda self: self._get("start_time", self._web_start))
    last_error = property(lambda self: self.link.problem() or self._get("last_error"))
    error_detail = property(lambda self: self._get("error_detail"))
    cleared_errors = property(lambda self: self._get("cleared_errors", []))
    paused = property(lambda self: self._get("paused", False))
    continuous = property(lambda self: self._get("continuous", False))
    @property
    def total_steps(self):
        snap = self.link.published
        return max(int(self._get("total_steps", 0) or 0), int(snap.step) if snap is not None else 0)
    sim_speed = property(lambda self: self._get("sim_speed", 0.0))
    active_paradigm_id = property(lambda self: self._get("active_paradigm_id"))
    active_paradigm_title = property(lambda self: self._get("active_paradigm_title"))
    current_trial = property(lambda self: self._get("current_trial"))
    trial_history = property(lambda self: range(int(self._get("trials_completed", 0))))
    trial_sim_time = property(lambda self: self._get("trial_sim_time", 0.0))
    trial_length_s = property(lambda self: self._get("trial_length_s"))
    backend = property(lambda self: self._get("backend", "modular"))
    launch_backend = property(lambda self: self._get("launch_backend"))
    last_step_wall_s = property(lambda self: self._get("last_step_wall_s", 0.0))
    arena = property(lambda self: SimpleNamespace(world_bounds=self._get("world_bounds", [])))
    recordings_dir = property(lambda self: Path(self._get("recordings_dir") or "."))
    published = property(lambda self: self.link.published)

    @property
    def latest_telemetry(self):
        snap = self.link.published
        if snap is None:
            return None
        # Keyed by the snapshot itself, not its seq: a restarted simulation reuses seqs.
        source, cached = self._telemetry_cache
        if source is not snap:
            cached = json.loads(snap.data)
            self._telemetry_cache = (snap, cached)
        return cached

    # -- methods the handler calls -----------------------------------------------------
    def recording_status(self):
        return self._get("recording")

    def identity(self):
        return self._get("identity") or {}

    def compute_info(self):
        return self._get("compute") or {"device": None, "error": "simulation process status not received"}

    def observation_publication_status(self):  # presence only; the handler reads the snapshot
        return None

    def step_in_progress_s(self) -> float:
        base = float(self._get("step_in_progress_s", 0.0) or 0.0)
        age = self.link.state_age_s()
        return round(base + age, 3) if base > 0 and age is not None else base

    def timing_snapshot(self):
        timing = dict(self._get("timing") or {})
        timing["step_in_progress_s"] = self.step_in_progress_s()
        if self.link.problem():
            timing["achieved_speed"] = 0.0
        return timing

    def health(self) -> Dict[str, Any]:
        health = copy.deepcopy(self._get("health")) or {
            "status": "error", "error": None, "error_detail": None, "halted": True, "liveness": {},
            "mode": None, "result_validity": None, "observation_validity_update": None,
            "persistence": None, "recording_error": None, "loop_failure": None, "thread_failures": []}
        live = dict(health.get("liveness") or {})
        problem = self.link.problem()
        age = self.link.state_age_s()
        if problem:
            down = not self.link.connected
            health.update(status="error", error=problem, halted=True,
                          error_detail={"kind": "simulation_process", "message": problem,
                                        "last_status_age_s": None if age is None else round(age, 1),
                                        "last_step": self._get("total_steps")})
            live.update(state="sim_process_down" if down else "sim_process_unresponsive",
                        sim_process_connected=self.link.connected)
            if down:
                live["sim_thread_alive"] = False
        elif age is not None:
            if live.get("step_in_progress_s"):
                live["step_in_progress_s"] = round(live["step_in_progress_s"] + age, 3)
            if live.get("last_advance_age_s") is not None and live.get("state") != "advancing":
                live["last_advance_age_s"] = round(live["last_advance_age_s"] + age, 2)
        health["liveness"] = live
        return health

    def read_view(self, key, build=None, wait_s=None, max_wait_s=None):
        p = self.link.request({"t": "view", "key": key})
        if p is None:
            return None
        with p.cv:
            p.cv.wait_for(lambda: p.done, VIEW_WAIT_S)
            return p.view if p.done else None

    def command_ack_lookup(self, *, command_id=None, client_command_id=None) -> Dict[str, Any]:
        base = {"daemon_run_id": self._get("run_id"), "command_id": command_id,
                "client_command_id": client_command_id}
        p = self.link.request({"t": "ack", "command_id": command_id, "client_command_id": client_command_id})
        if p is not None:
            with p.cv:
                p.cv.wait_for(lambda: p.done, VIEW_WAIT_S)
                if p.done and isinstance(p.view, dict):
                    return p.view
        # Never success: the simulation process could not be asked.
        return dict(base, state="unknown",
                    message=f"{self.link.problem() or 'no reply from the simulation process'}; outcome unknown")

    def dispatch_command(self, cmd) -> Dict[str, Any]:
        if not isinstance(cmd, dict):
            return {"status": "error", "message": "Command must be a JSON object"}
        action = cmd.get("action", "")
        client_id = cmd.get("client_command_id")
        try:
            p = self.link.request({"t": "cmd", "cmd": cmd}, limit=MAX_PENDING_COMMANDS)
        except QueueFull:
            return {"status": "error", "applied": False, "queue_full": True, "action": action,
                    "client_command_id": client_id,
                    "message": f"simulation command queue full ({MAX_PENDING_COMMANDS} commands not yet "
                               f"applied); the command was not applied, retry later"}
        if p is None:
            return {"status": "error", "applied": False, "sim_process_down": True, "action": action,
                    "message": f"{self.link.problem() or 'simulation process unavailable'}; "
                               f"the command was not applied."}
        with p.cv:
            p.cv.wait_for(lambda: p.done, REPLY_WAIT_S)
            if not p.done and p.command_id is None:
                p.cv.wait_for(lambda: p.done or p.command_id is not None, ACCEPT_WAIT_S)
            if p.done and p.result is not None:
                return p.result
            if p.error is not None or p.epoch != self.link.epoch:
                # The connection dropped before a result: the command may or may not have
                # been applied by the process that queued it.  Never reported as queued.
                return {"status": "error", "applied": None, "action": action,
                        "client_command_id": client_id, "command_id": p.command_id,
                        "daemon_run_id": p.run_id,
                        "message": f"{p.error or 'the simulation process connection was replaced'}; "
                                   f"the command outcome is unknown, check /api/status."}
            if p.command_id is not None:
                return {"status": "queued", "applied": False, "command_id": p.command_id, "action": action,
                        "client_command_id": client_id, "daemon_run_id": p.run_id,
                        "step_in_progress_s": self.step_in_progress_s(),
                        "message": "Queued: applied by the simulation process at the next step boundary"}
            reason = p.error or f"no reply from the simulation process within {ACCEPT_WAIT_S:g} s"
        return {"status": "error", "applied": None, "action": action,
                "message": f"{reason}; the command outcome is unknown, check /api/status."}


def build_web_server(path: str, host: str, port: int, policy=None):
    """(server, link): the existing handler served from a :class:`SimProxy`."""
    from http.server import ThreadingHTTPServer
    import neurofly_daemon as nd
    from stream_gateway import StreamGateway, StreamPolicy

    link = SimLink(path).start()
    proxy = SimProxy(link)

    class SplitWebHandler(nd.NeuroflyHTTPHandler):
        runner = proxy
        gateway = StreamGateway(policy or StreamPolicy())

        def _status_payload(self):
            payload = super()._status_payload()
            payload["process_layout"] = {"mode": "split", "web_pid": os.getpid(), "sim": link.describe(),
                                         "pending_commands": link.pending_commands(),
                                         "max_pending_commands": MAX_PENDING_COMMANDS}
            return payload

    return ThreadingHTTPServer((host, port), SplitWebHandler), link


def run_web(args) -> int:
    """Main thread of the web process."""
    from stream_gateway import StreamPolicy

    path = args.ipc_socket or default_socket_path(args.port)
    policy = StreamPolicy.from_env(public=args.public, max_stream_clients=args.max_stream_clients,
                                   stream_hz=args.stream_hz, allowed_origin=args.allowed_origin)
    try:
        server, _ = build_web_server(path, args.host, args.port, policy)
    except OSError as exc:
        print(f"[Web] Cannot serve http://{args.host}:{args.port}/: {exc}", file=sys.stderr, flush=True)
        return WEB_BIND_FAILED

    def _on_signal(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    print(f"[Web] Web process {os.getpid()} at http://{args.host}:{server.server_address[1]}/ "
          f"(simulation via {path})", flush=True)
    try:
        server.serve_forever()
    except SystemExit:
        pass
    finally:
        server.server_close()
    return 0


# ---------------------------------------------------------------------------------------- launcher

_LAUNCHER_FLAGS = {"--process-mode": True, "--ipc-socket": True, "--single-process": False, "--split": False}


def _strip_launcher_flags(argv):
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        name = arg.split("=", 1)[0]
        if name in _LAUNCHER_FLAGS:
            skip = _LAUNCHER_FLAGS[name] and "=" not in arg
            continue
        out.append(arg)
    return out


def _stop_child(proc: subprocess.Popen, timeout: float) -> Optional[int]:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)
    return proc.returncode


def run_split(args, argv) -> int:
    """Spawn the simulation process and the web process; restart the web process if it
    dies; on SIGTERM/SIGINT stop the simulation first (its final save) and then the web."""
    import neurofly_daemon as nd

    private_dir = None
    path = args.ipc_socket
    if not path:
        private_dir = private_socket_dir()
        path = os.path.join(private_dir, "sim.sock")
    script = str(Path(nd.__file__).resolve())
    rest = _strip_launcher_flags(argv)
    sim_cmd = [sys.executable, "-u", script, *rest, "--process-mode", "sim", "--ipc-socket", path]
    web_cmd = [sys.executable, "-u", script, *rest, "--process-mode", "web", "--ipc-socket", path]
    # Own sessions: a terminal Ctrl-C reaches only this launcher, which then stops the
    # simulation exactly once (its final save) instead of every process at once.
    child_env = dict(os.environ, **{LAUNCHER_ENV: str(os.getpid())})
    sim = subprocess.Popen(sim_cmd, start_new_session=True, env=child_env)
    web = subprocess.Popen(web_cmd, start_new_session=True, env=child_env)
    print(f"[Launcher] simulation pid {sim.pid}, web pid {web.pid}, IPC {path}", flush=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda s, f: stop.append(s))
    signal.signal(signal.SIGINT, lambda s, f: stop.append(s))
    web_restarts: deque = deque(maxlen=5)
    sim_reported = False
    socket_seen = False
    code = 0
    try:
        while not stop:
            time.sleep(0.2)
            sim_code = sim.poll()
            socket_seen = socket_seen or os.path.exists(path)
            if sim_code is not None and not sim_reported:
                sim_reported = True
                if sim_code == 0 or not socket_seen:
                    # A clean stop (stop_daemon.sh signals the PID file's simulation
                    # process) or a startup refusal: nothing is left to serve.
                    code = sim_code
                    break
                print(f"[Launcher] The simulation process exited with code {sim_code}; the web process "
                      f"keeps reporting it as down. Restart the daemon to resume.", file=sys.stderr, flush=True)
                code = sim_code if sim_code > 0 else 1
            if web.poll() is not None:
                if stop:
                    # A stop signal reached the whole group (e.g. systemd, pkill): no restart.
                    break
                if web.returncode == WEB_BIND_FAILED:
                    print("[Launcher] The web process cannot bind its port; stopping the simulation.",
                          file=sys.stderr, flush=True)
                    code = 2
                    break
                now = time.monotonic()
                if len(web_restarts) == web_restarts.maxlen and now - web_restarts[0] < 60:
                    print("[Launcher] The web process keeps failing; giving up on it. The simulation "
                          "process keeps running.", file=sys.stderr, flush=True)
                    while not stop and sim.poll() is None:
                        time.sleep(0.5)
                    code = sim.returncode if sim.returncode is not None else code
                    break
                web_restarts.append(now)
                print(f"[Launcher] The web process exited ({web.returncode}); starting a new one. "
                      f"The simulation was not interrupted.", file=sys.stderr, flush=True)
                web = subprocess.Popen(web_cmd, start_new_session=True, env=child_env)
    finally:
        sim_code = _stop_child(sim, SIM_STOP_TIMEOUT_S)
        _stop_child(web, 10)
        if private_dir:
            try:
                os.unlink(path)
            except OSError:
                pass
            try:
                os.rmdir(private_dir)
            except OSError:
                pass
    if stop:
        code = sim_code if sim_code and sim_code > 0 else 0
    return code
