"""Strip local absolute paths from run files before they leave this computer.

A run's manifest.json, and the header of its body.nfbody, record where the graph,
the connectome data and the output directory were (``/home/<user>/...``).  That is
the right record on the machine that ran it, but a curated run is committed to a
public repository and a bundle is made to be handed to someone else, so both go
through :func:`redact_bytes` first:

* the repository root becomes ``<repo>`` and the home directory ``<home>``;
* any other absolute path under a per-user or mount root (``/home/``, ``/Users/``,
  ``/media/``, ``/run/media/``, ``/mnt/``, ``/tmp/``) becomes ``<local-path>``
  followed by its last component, e.g. ``<local-path>/malecns_v1``;
* this machine's host name becomes ``<host>`` and the account name ``<user>``.

Only these strings change.  The trajectory and recording-frame SHA-256 values cover
telemetry and frame records, which carry no paths, so they still verify.
"""

from __future__ import annotations

import getpass
import gzip
import re
import socket
from pathlib import Path

from .catalog import PROJECT_ROOT

_OTHER = re.compile(r'(?:/run/media|/home|/Users|/media|/mnt|/tmp)/[^"\s\\,;]*')
NOTE = ("local absolute paths replaced: repository root -> <repo>, home directory -> <home>, "
        "other per-user or mount paths -> <local-path>/<last component>, host name -> <host>, "
        "account name -> <user>")


def _last_component(match: re.Match) -> str:
    tail = match.group(0).rstrip("/").rsplit("/", 1)[-1]
    return "<local-path>/" + tail if tail else "<local-path>"


def redact_text(text: str) -> str:
    for root, label in sorted(((str(PROJECT_ROOT.resolve()), "<repo>"), (str(PROJECT_ROOT), "<repo>"),
                               (str(Path.home()), "<home>")), key=lambda item: -len(item[0])):
        if root and root != "/":
            text = text.replace(root, label)
    text = _OTHER.sub(_last_component, text)
    # The machine's name and the account name, wherever else they appear.
    for value, label in ((_safe(socket.gethostname), "<host>"), (_safe(getpass.getuser), "<user>")):
        if value and len(value) >= 3:
            text = re.sub(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", label, text)
    return text


def _safe(getter) -> str:
    try:
        return getter() or ""
    except Exception:  # noqa: BLE001 - no user or host name available
        return ""


def redact_bytes(name: str, data: bytes) -> bytes:
    """Redacted copy of one run file (``.nfbody`` is gzip: inflated, redacted, re-deflated
    with a zero timestamp so the same input always gives the same bytes)."""
    if name.endswith(".nfbody"):
        text = gzip.decompress(data).decode("utf-8")
        return gzip.compress(redact_text(text).encode("utf-8"), compresslevel=6, mtime=0)
    if name.endswith((".json", ".jsonl", ".txt")):
        return redact_text(data.decode("utf-8")).encode("utf-8")
    return data


def leaks(data: bytes, name: str = "") -> list[str]:
    """Absolute local paths still present (for tests and checks)."""
    if name.endswith(".nfbody"):
        data = gzip.decompress(data)
    return _OTHER.findall(data.decode("utf-8", errors="replace"))
