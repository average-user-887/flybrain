"""
Drosophila Connectome RPC Co-Simulation Client
==============================================
Provides high-throughput HTTP/JSON communication with the remote
166,700-neuron MaleCNS v1.0 spiking engine (brainlab/cosim_server.py).

Features:
- Automatic fallback to local biophysical surrogate if remote server is unreachable.
- Round-trip latency tracking & health diagnostics.
- Batch packet transmission (20 sub-steps per 2.0 ms physics tick).
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
import numpy as np


class ConnectomeClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 8768,
        timeout: float = 0.20,           # 200 ms timeout per step
        fallback_enabled: bool = True
    ):
        self.host = host or os.environ.get("NEUROFLY_CONNECTOME_HOST", "127.0.0.1")
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.fallback_enabled = fallback_enabled
        self.is_connected = False
        self.last_latency_ms = 0.0
        self.consecutive_errors = 0

        # Check initial connection
        self.check_health()

    def check_health(self) -> bool:
        """Ping the remote co-simulation server."""
        url = f"{self.base_url}/status"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.is_connected = True
                    self.consecutive_errors = 0
                    return True
        except Exception:
            self.is_connected = False
            return False
        return False

    def reset(self) -> bool:
        """Reset remote connectome membrane potentials to resting state."""
        if not self.is_connected and not self.check_health():
            return False
        url = f"{self.base_url}/reset"
        try:
            req = urllib.request.Request(url, method="POST", data=b"{}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def step(
        self,
        sensory_packet: Dict[str, Any],
        duration_ms: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """
        Transmits sensory drive vector to the remote connectome and returns descending neuron activity.
        Returns None if remote server fails, prompting client to use local surrogate.
        """
        if not self.is_connected and not self.check_health():
            return None

        url = f"{self.base_url}/step"
        payload = json.dumps({
            "sensory": sensory_packet,
            "duration_ms": duration_ms
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        clock = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    result = json.loads(resp.read().decode("utf-8"))
                    self.last_latency_ms = (time.perf_counter() - clock) * 1000.0
                    self.consecutive_errors = 0
                    return result
        except Exception:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                self.is_connected = False
            return None

        return None
