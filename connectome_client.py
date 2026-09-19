"""
Drosophila Connectome RPC Co-Simulation Client
==============================================
HTTP/JSON client for the MaleCNS v1.0 spiking server (brainlab/cosim_server.py).

The client never substitutes a surrogate.  ``step`` returns ``None`` when the
server is unreachable, errors, or reports a graph identity other than the one
expected; the reason is kept in ``last_error`` and every transition is kept in
``events`` so the caller can report it.  What the caller does next (halt,
raise, or an explicitly labelled test fallback) is the bridge's decision.
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PINS_PATH = Path(__file__).resolve().parent / "brainlab/graph_pins.json"


def _json_safe(value: Any):
    """Encode numpy arrays and scalars in a sensory packet as plain JSON data.

    The bridge's packet carries numpy state (e.g. the E-PG ring profile) next to the
    scalar channels.  Without this the whole request fails to serialize, so no
    sensory key reaches the server at all -- including the WP5 optomotor keys.
    """
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable in a sensory packet")


def pinned_graph_sha256() -> Optional[str]:
    try:
        return json.loads(PINS_PATH.read_text())["graph_sha256"]
    except (OSError, ValueError, KeyError):
        return None


class ConnectomeClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 8768,
        timeout: float = 0.20,           # 200 ms timeout per step
        fallback_enabled: bool = False,  # retained for signature compatibility; the client never falls back
        expected_graph_sha256: Optional[str] = "pinned",
        allow_synthetic: bool = False,
    ):
        self.host = host or os.environ.get("NEUROFLY_CONNECTOME_HOST", "127.0.0.1")
        self.port = port
        self.base_url = f"http://{self.host}:{port}"
        self.timeout = timeout
        self.fallback_enabled = fallback_enabled
        self.expected_graph_sha256 = pinned_graph_sha256() if expected_graph_sha256 == "pinned" else expected_graph_sha256
        self.allow_synthetic = allow_synthetic
        self.is_connected = False
        self.server_identity: Optional[Dict[str, Any]] = None
        self.last_latency_ms = 0.0
        self.consecutive_errors = 0
        self.last_error: Optional[str] = None
        self.events: List[Dict[str, Any]] = []

        # Check initial connection
        self.check_health()

    def _event(self, kind: str, **fields):
        self.events.append(dict(kind=kind, wall_time=time.time(), **fields))

    def _identity_problem(self, payload: Dict[str, Any]) -> Optional[str]:
        if payload.get("synthetic") and not self.allow_synthetic:
            return "server is running a synthetic test graph"
        sha = payload.get("graph_sha256")
        if self.expected_graph_sha256 and not payload.get("synthetic") and sha != self.expected_graph_sha256:
            return f"server graph {sha} differs from expected {self.expected_graph_sha256}"
        if self.server_identity and sha != self.server_identity.get("graph_sha256"):
            return f"server graph changed from {self.server_identity.get('graph_sha256')} to {sha}"
        return None

    def _mark_down(self, reason: str):
        self.last_error = reason
        if self.is_connected:
            self._event("disconnect", reason=reason)
        self.is_connected = False

    def check_health(self) -> bool:
        """Ping the server and verify the graph identity it reports."""
        url = f"{self.base_url}/status"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as error:
            self._mark_down(f"unreachable: {error}")
            return False
        problem = self._identity_problem(data)
        if problem:
            self._event("identity_mismatch", reason=problem)
            self._mark_down(problem)
            return False
        if not self.is_connected:
            self._event("connect", graph_sha256=data.get("graph_sha256"), backend=data.get("backend"))
        # Identity plus the WP5 provenance the dashboard must show: whether the
        # server's engineered assistance is gated on, and which optomotor IO map
        # (brainlab/io_map.py) it resolved.  ``None`` means the server has none.
        self.server_identity = {k: data.get(k) for k in
                                ("backend", "graph_sha256", "neuron_map_sha256", "io_map_sha256",
                                 "sensory_map_sha256", "synthetic", "label",
                                 "engineered_assistance_enabled", "optomotor_io_map_sha256")}
        self.is_connected = True
        self.consecutive_errors = 0
        self.last_error = None
        return True

    def reset(self) -> bool:
        """Reset remote connectome membrane potentials to resting state."""
        if not self.is_connected and not self.check_health():
            return False
        url = f"{self.base_url}/reset"
        try:
            req = urllib.request.Request(url, method="POST", data=b"{}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                ok = resp.status == 200
        except Exception as error:
            self._mark_down(f"reset failed: {error}")
            return False
        self._event("remote_reset", ok=ok)
        return ok

    def step(
        self,
        sensory_packet: Dict[str, Any],
        duration_ms: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """Send sensory drive; return descending-neuron readouts or ``None``.

        ``None`` always comes with ``last_error`` set.  The caller must report it.

        ``sensory_packet`` is forwarded verbatim, including the WP5 keys
        ``optomotor_slip_rad_s`` and ``optomotor_contrast``.  So is the reply,
        including its ``optomotor`` block, ``engineered_assistance_applied`` and
        ``engineered_assistance_enabled``.  A reply without an ``optomotor``
        block is passed on as it came: the caller decides what that means and
        must not read the absence as zero yaw.
        """
        if not self.is_connected and not self.check_health():
            return None

        url = f"{self.base_url}/step"
        payload = json.dumps({
            "sensory": sensory_packet,
            "duration_ms": duration_ms
        }, default=_json_safe).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        clock = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as error:
            self.consecutive_errors += 1
            self.last_error = f"step failed: {error}"
            if self.consecutive_errors >= 3:
                self._mark_down(self.last_error)
            return None
        if result.get("status") != "ok":
            self.consecutive_errors += 1
            self.last_error = f"server replied {result.get('status')!r}: {result.get('error')}"
            return None
        problem = self._identity_problem(result)
        if problem:
            self._event("identity_mismatch", reason=problem)
            self._mark_down(problem)
            return None
        self.last_latency_ms = (time.perf_counter() - clock) * 1000.0
        self.consecutive_errors = 0
        self.last_error = None
        return result
