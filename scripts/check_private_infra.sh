#!/usr/bin/env bash
# Release-hygiene guard for Project NeuroFly: fails on personal or private-infrastructure
# data (private IPs, home/share paths, agent scratch paths, e-mail addresses, machine
# hostnames, known private tokens). Works in a git checkout and in an exported tree.
# All logic lives in check_private_infra.py; arguments are passed through
# (e.g. --root DIR, --no-git). Allow-list: scripts/private_infra_allowlist.txt.
# Commit-metadata mode (messages, author, committer of a revision's whole ancestry):
#   scripts/check_private_infra.sh --commits HEAD
# Committed tree of a revision (what pushing it publishes):
#   scripts/check_private_infra.sh --tree-rev HEAD
# The pre-push hook scripts/hooks/pre-push runs both for every outgoing ref;
# install once per clone: git config core.hooksPath scripts/hooks
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
fi
exec "$PY" "$HERE/check_private_infra.py" "$@"
