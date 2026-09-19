"""Public-stream gateway policy for the NeuroFly daemon.

The daemon's HTTP surface was written for a trusted LAN: ``/api/stream`` is an
unbounded 30 Hz Server-Sent-Events feed and ``POST /api/command`` lets any
client switch paradigms, inject stimuli and reset trials.  Publishing that
surface unchanged would let anyone on the internet drive the fly.

This module holds everything that changes when the daemon is exposed publicly,
so that ``neurofly_daemon.py`` only needs a few call sites:

* :class:`StreamPolicy` -- the immutable configuration (public flag, admin
  token, client cap, stream rate, allowed origin), built from CLI flags and the
  ``NEUROFLY_*`` environment variables.
* :class:`StreamGateway` -- runtime state: authorises commands, hands out and
  reclaims SSE client slots, and paces the stream.

Defaults reproduce the historical behaviour exactly (private mode, no token,
unlimited clients, 30 Hz, ``Access-Control-Allow-Origin: *``) so that existing
deployments and tests are unaffected unless ``--public`` or
``NEUROFLY_PUBLIC=1`` is set.

Environment variables
---------------------
``NEUROFLY_PUBLIC``          ``1``/``true``/``yes`` enables public (read-only) mode.
``NEUROFLY_ADMIN_TOKEN``     Bearer token that re-enables ``POST /api/command``
                             in public mode.  Must be at least 16 characters.
``NEUROFLY_MAX_STREAM_CLIENTS``  Cap on concurrent SSE clients (public default 50).
``NEUROFLY_STREAM_HZ``       SSE broadcast rate in Hz (public default 10).
``NEUROFLY_ALLOWED_ORIGIN``  Value for ``Access-Control-Allow-Origin`` (default ``*``).
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "StreamPolicy",
    "StreamGateway",
    "SSESlot",
    "MIN_ADMIN_TOKEN_LENGTH",
    "MAX_COMMAND_BYTES",
    "env_flag",
]

MIN_ADMIN_TOKEN_LENGTH = 16
MAX_COMMAND_BYTES = 64 * 1024

# Historical defaults of the daemon before this module existed.
PRIVATE_STREAM_HZ = 30.0
PUBLIC_STREAM_HZ = 10.0
PUBLIC_MAX_STREAM_CLIENTS = 50

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False,
             environ: Optional[Mapping[str, str]] = None) -> bool:
    """Read a boolean environment variable (``1``/``true``/``yes``/``on``)."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class StreamPolicy:
    """Configuration for one daemon process.

    ``public=False`` is the historical private-LAN behaviour.  ``public=True``
    disables ``POST /api/command`` unless a matching bearer token is presented,
    caps concurrent SSE clients and throttles the broadcast rate.
    """

    public: bool = False
    admin_token: Optional[str] = None
    max_stream_clients: int = 0          # 0 = unlimited
    stream_hz: float = PRIVATE_STREAM_HZ
    allowed_origin: str = "*"

    def __post_init__(self) -> None:
        token = self.admin_token
        if token is not None:
            token = token.strip()
            if token == "":
                token = None
            elif len(token) < MIN_ADMIN_TOKEN_LENGTH:
                raise ValueError(
                    f"admin token must be at least {MIN_ADMIN_TOKEN_LENGTH} characters"
                )
        object.__setattr__(self, "admin_token", token)
        if self.max_stream_clients < 0:
            raise ValueError("max_stream_clients must be >= 0 (0 = unlimited)")
        if not (0.1 <= float(self.stream_hz) <= 120.0):
            raise ValueError("stream_hz must be between 0.1 and 120")
        origin = (self.allowed_origin or "*").strip() or "*"
        object.__setattr__(self, "allowed_origin", origin)

    @property
    def read_only(self) -> bool:
        """True when no client can issue commands at all."""
        return self.public and self.admin_token is None

    @property
    def stream_interval(self) -> float:
        return 1.0 / float(self.stream_hz)

    @classmethod
    def from_env(
        cls,
        *,
        public: Optional[bool] = None,
        admin_token: Optional[str] = None,
        max_stream_clients: Optional[int] = None,
        stream_hz: Optional[float] = None,
        allowed_origin: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "StreamPolicy":
        """Build a policy from explicit overrides layered over the environment.

        Explicit keyword arguments (typically CLI flags) win over environment
        variables, which win over the public/private defaults.
        """
        env = os.environ if environ is None else environ
        is_public = env_flag("NEUROFLY_PUBLIC", False, env) if public is None else bool(public)
        token = admin_token if admin_token is not None else env.get("NEUROFLY_ADMIN_TOKEN")
        default_clients = PUBLIC_MAX_STREAM_CLIENTS if is_public else 0
        clients = (
            _env_int(env, "NEUROFLY_MAX_STREAM_CLIENTS", default_clients)
            if max_stream_clients is None
            else int(max_stream_clients)
        )
        default_hz = PUBLIC_STREAM_HZ if is_public else PRIVATE_STREAM_HZ
        hz = _env_float(env, "NEUROFLY_STREAM_HZ", default_hz) if stream_hz is None else float(stream_hz)
        origin = allowed_origin if allowed_origin is not None else env.get("NEUROFLY_ALLOWED_ORIGIN", "*")
        return cls(
            public=is_public,
            admin_token=token,
            max_stream_clients=clients,
            stream_hz=hz,
            allowed_origin=origin or "*",
        )

    def describe(self) -> dict:
        """JSON-safe summary for ``/api/status`` (never includes the token)."""
        return {
            "public": self.public,
            "read_only": self.read_only,
            "commands_require_token": self.public and self.admin_token is not None,
            "max_stream_clients": self.max_stream_clients,
            "stream_hz": self.stream_hz,
        }


class SSESlot:
    """Context manager returned by :meth:`StreamGateway.acquire_stream_slot`.

    ``bool(slot)`` is False when the cap was reached.  Always use it in a
    ``with`` block (or call :meth:`release`) so the slot is reclaimed.
    """

    __slots__ = ("_gateway", "_granted", "_released")

    def __init__(self, gateway: "StreamGateway", granted: bool):
        self._gateway = gateway
        self._granted = granted
        self._released = False

    def __bool__(self) -> bool:
        return self._granted

    def release(self) -> None:
        if self._granted and not self._released:
            self._released = True
            self._gateway._release_slot()

    def __enter__(self) -> "SSESlot":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class StreamGateway:
    """Runtime enforcement of a :class:`StreamPolicy`.

    Thread-safe; one instance is shared by every HTTP handler thread.
    """

    def __init__(self, policy: Optional[StreamPolicy] = None):
        self.policy = policy or StreamPolicy()
        self._lock = threading.Lock()
        self._active_streams = 0
        self._rejected_streams = 0
        self._rejected_commands = 0
        self._delivered_snapshots = 0
        self._decimated_snapshots = 0

    # ---- commands ---------------------------------------------------------
    @staticmethod
    def _bearer_token(headers: Mapping[str, str]) -> Optional[str]:
        raw = headers.get("Authorization") or headers.get("authorization")
        if not raw:
            return None
        parts = raw.strip().split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return parts[1].strip() or None

    def authorize_command(self, headers: Mapping[str, str]) -> bool:
        """Return True when a ``POST /api/command`` may be dispatched.

        Private mode: always True (historical behaviour).
        Public mode: only with ``Authorization: Bearer <NEUROFLY_ADMIN_TOKEN>``;
        the comparison is constant-time.
        """
        if not self.policy.public:
            return True
        expected = self.policy.admin_token
        if expected is None:
            self._count_rejected_command()
            return False
        presented = self._bearer_token(headers)
        if presented is None:
            self._count_rejected_command()
            return False
        ok = hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
        if not ok:
            self._count_rejected_command()
        return ok

    def _count_rejected_command(self) -> None:
        with self._lock:
            self._rejected_commands += 1

    def command_rejection(self) -> dict:
        """Body for the 403 response."""
        if self.policy.read_only:
            return {"status": "error", "error": "read_only",
                    "message": "This NeuroFly stream is read-only; commands are disabled."}
        return {"status": "error", "error": "unauthorized",
                "message": "Commands require a valid bearer token."}

    # ---- SSE client cap ---------------------------------------------------
    def acquire_stream_slot(self) -> SSESlot:
        cap = self.policy.max_stream_clients
        with self._lock:
            if cap and self._active_streams >= cap:
                self._rejected_streams += 1
                return SSESlot(self, False)
            self._active_streams += 1
            return SSESlot(self, True)

    def _release_slot(self) -> None:
        with self._lock:
            if self._active_streams > 0:
                self._active_streams -= 1

    @property
    def active_streams(self) -> int:
        with self._lock:
            return self._active_streams

    # ---- pacing -----------------------------------------------------------
    def pace(self, last_send: float) -> float:
        """Sleep until the next broadcast is due; return the new timestamp."""
        interval = self.policy.stream_interval
        now = time.monotonic()
        remaining = (last_send + interval) - now
        if remaining > 0:
            time.sleep(remaining)
        return time.monotonic()

    # ---- delivery accounting ----------------------------------------------
    def record_delivery(self, sent: int, decimated: int) -> None:
        """Accumulate SSE delivery counters (latest-value-wins decimation policy)."""
        with self._lock:
            self._delivered_snapshots += int(sent)
            self._decimated_snapshots += int(decimated)

    # ---- reporting --------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {
                "active_stream_clients": self._active_streams,
                "rejected_stream_clients": self._rejected_streams,
                "rejected_commands": self._rejected_commands,
                "delivered_snapshots": self._delivered_snapshots,
                "decimated_snapshots": self._decimated_snapshots,
            }

    def describe(self) -> dict:
        out = self.policy.describe()
        out.update(self.stats())
        return out
