"""Keep personal data out of everything NeuroFly writes to disk.

Receipts, run manifests, recordings and bundles are made to be shared. They
must describe the *hardware and software* a result came from, never the
*machine or person*: no hostname, no account name, no absolute path that
contains either. Every writer in the repository goes through this module, so
the rule lives in one place and cannot drift.

* :func:`host_description` replaces the hostname. It records public hardware
  and software facts only: OS/architecture, CPU model, CPU count, GPU model,
  Python and key library versions.
* :func:`redact_text` / :func:`redact_local` / :func:`portable_path` replace
  absolute local paths with the placeholders already used by the committed
  receipts (``docs/receipts/REDACTIONS.md``):

  ===================  ======================================================
  ``<repo>/...``       the repository root of this checkout
  ``<scratch>/...``    the system temporary directory (``/tmp``, ``$TMPDIR``)
  ``<home>/...``       the current user's home directory
  ``<local-path>/x``   any other absolute path under a per-user or mount root
                       (``/home/``, ``/Users/``, ``/media/``, ``/run/media/``,
                       ``/mnt/``); only its last component ``x`` is kept
  ===================  ======================================================

Standard library only; importing it is cheap and has no side effects.
"""
from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePath
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Libraries whose versions are part of a host description (looked up from package
# metadata, never imported, so describing the host is fast and side-effect free).
LIBRARIES = ('numpy', 'numba', 'pandas', 'pyarrow', 'scipy', 'mujoco', 'flygym', 'cupy')

_OTHER_LOCAL = re.compile(r'(?:/run/media|/home|/Users|/media|/mnt)/[^"\'\s\\,;:]*')


# ---------------------------------------------------------------------------
# Host description (replaces platform.node() / socket.gethostname())
# ---------------------------------------------------------------------------

def _cpu_model() -> Optional[str]:
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as stream:
            for line in stream:
                if line.lower().startswith(('model name', 'hardware', 'cpu model')):
                    return line.split(':', 1)[1].strip() or None
    except OSError:
        pass
    if sys.platform == 'darwin':
        try:
            out = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or None


def _gpu_models() -> Optional[list]:
    """NVIDIA GPU model names, or None when no NVIDIA driver is visible."""
    smi = shutil.which('nvidia-smi')
    if not smi:
        return None
    try:
        out = subprocess.run([smi, '--query-gpu=name,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] or None


def _library_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
        except Exception:  # noqa: BLE001 - broken metadata must not break a receipt
            versions[name] = None
    return versions


@functools.lru_cache(maxsize=1)
def _host_description_cached() -> tuple:
    gpus = _gpu_models()
    description = {
        'schema': 'neurofly.host-description.v1',
        'platform': f'{platform.system()}-{platform.machine()}',
        'os_release': platform.release(),
        'cpu': _cpu_model(),
        'cpu_count': os.cpu_count(),
        'gpu': gpus if gpus else 'none',
        'python': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'libraries': _library_versions(),
    }
    return tuple(description.items())


def host_description() -> dict:
    """Non-identifying description of the machine: hardware model and software versions.

    Recorded wherever older code wrote the hostname. Contains no hostname, no
    account name and no path. CPU and GPU model names are public hardware facts.
    The result is computed once per process and copied on every call.
    """
    description = dict(_host_description_cached())
    description['libraries'] = dict(description['libraries'])
    if isinstance(description['gpu'], list):
        description['gpu'] = list(description['gpu'])
    return description


# ---------------------------------------------------------------------------
# Local-path redaction
# ---------------------------------------------------------------------------

def _roots() -> list:
    """(absolute prefix, placeholder) pairs, longest prefix first."""
    pairs = []
    for root in {str(REPO_ROOT), os.path.abspath(REPO_ROOT)}:
        pairs.append((root, '<repo>'))
    scratch = {tempfile.gettempdir(), '/tmp', '/var/tmp'}
    for root in list(scratch):
        try:
            scratch.add(os.path.realpath(root))
        except OSError:
            pass
    pairs += [(root, '<scratch>') for root in scratch]
    try:
        home = str(Path.home())
    except (KeyError, RuntimeError):
        home = ''
    if home:
        pairs.append((home, '<home>'))
        pairs.append((os.path.realpath(home), '<home>'))
    pairs = [(root.rstrip('/\\'), label) for root, label in pairs if root and root not in ('/', '\\')]
    return sorted(set(pairs), key=lambda item: -len(item[0]))


def _other_local(match: re.Match) -> str:
    tail = match.group(0).rstrip('/').rsplit('/', 1)[-1]
    return '<local-path>/' + tail if tail else '<local-path>'


def redact_text(text: str) -> str:
    """Replace absolute local paths inside ``text`` with placeholders (see module doc)."""
    if not isinstance(text, str) or not text:
        return text
    for root, label in _roots():
        if root in text:
            text = re.sub(re.escape(root) + r'(?=[/\\]|$|[^A-Za-z0-9._-])', label, text)
    text = _OTHER_LOCAL.sub(_other_local, text)
    if '/' in text or '\\' in text:
        # Path-like strings can still carry the account or machine name in a
        # component (e.g. <scratch>/pytest-of-<user>/...). Plain values are left
        # alone so that no scientific string is ever rewritten.
        for value, label in _identity_tokens():
            text = re.sub(r'(?<![A-Za-z0-9])' + re.escape(value) + r'(?![A-Za-z0-9])', label, text)
    return text


@functools.lru_cache(maxsize=1)
def _identity_tokens() -> tuple:
    tokens = []
    for getter, label in ((_account_name, '<user>'), (_machine_name, '<host>')):
        try:
            value = getter() or ''
        except Exception:  # noqa: BLE001 - no user or host name available
            value = ''
        if len(value) >= 3 and not value.startswith('<'):
            tokens.append((value, label))
    return tuple(tokens)


def _account_name() -> str:
    import getpass
    return getpass.getuser()


def _machine_name() -> str:
    # Read only to *remove* it from paths; never written anywhere.
    return platform.node()


def portable_path(path: Any) -> Optional[str]:
    """A path as it may be written into a shared artifact.

    Inside the repository it becomes ``<repo>/relative/path``; elsewhere the
    placeholder scheme of :func:`redact_text` applies. Relative paths are kept
    as given (they carry no machine identity). ``None`` stays ``None``.
    """
    if path is None:
        return None
    text = os.fspath(path) if isinstance(path, (str, PurePath)) or hasattr(path, '__fspath__') else str(path)
    return redact_text(text)


def redact_local(value: Any) -> Any:
    """Recursively redact absolute local paths in a JSON-like structure.

    Strings and :class:`pathlib.PurePath` values are redacted; dicts, lists and
    tuples are rebuilt; every other value (numbers, booleans, None, arrays) is
    returned unchanged. Dict keys are left as they are.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, PurePath):
        return redact_text(str(value))
    if isinstance(value, dict):
        return {key: redact_local(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_local(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_local(item) for item in value)
    return value


def restore_local(text: Optional[str]) -> Optional[str]:
    """Inverse of :func:`portable_path` on *this* machine, where it is possible.

    ``<repo>``, ``<home>`` and ``<scratch>`` are mapped back to this checkout,
    home and temporary directory. A ``<local-path>/...`` value cannot be mapped
    back (only its last component was kept) and gives ``None``, meaning "use the
    default location"; callers that need the file verify it by hash anyway.
    Values without a placeholder are returned unchanged.
    """
    if not isinstance(text, str) or '<' not in text:
        return text
    if text.startswith('<local-path>'):
        return None
    for label, root in (('<repo>', str(REPO_ROOT)), ('<home>', str(Path.home())),
                        ('<scratch>', tempfile.gettempdir())):
        if text == label or text.startswith(label + '/'):
            return root + text[len(label):]
    return text
