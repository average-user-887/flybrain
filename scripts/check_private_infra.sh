#!/usr/bin/env bash
# Release-hygiene guard for Project NeuroFly: fails on personal or private-infrastructure
# data (private IPs, home/share paths, agent scratch paths, e-mail addresses, machine
# hostnames, known private tokens). Works in a git checkout and in an exported tree.
# All logic lives in check_private_infra.py; arguments are passed through
# (e.g. --root DIR, --no-git). Allow-list: scripts/private_infra_allowlist.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
fi
exec "$PY" "$HERE/check_private_infra.py" "$@"
