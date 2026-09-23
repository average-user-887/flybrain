#!/usr/bin/env bash
set -euo pipefail

# Private infrastructure grep guard for Project NeuroFly
echo "[audit] Checking tracked files for private infrastructure leaks..."

ERRORS=0

# Grep for RFC 1918 private IPs (excluding RELEASE_AUDIT.md and .git)
if git grep -E -I -n "(192\.168\.[0-9]{1,3}\.[0-9]{1,3}|10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.[0-9]{1,3})" -- ":(exclude)docs/RELEASE_AUDIT.md" ":(exclude)scripts/check_private_infra.sh"; then
    echo "[audit] ERROR: Found private IP addresses in tracked files above!"
    ERRORS=$((ERRORS + 1))
fi

# Grep for UNC Windows share paths
if git grep -E -I -n "(//[a-zA-Z0-9_-]+/Storage|[A-Z]:[/\\]neurofly)" -- ":(exclude)docs/RELEASE_AUDIT.md" ":(exclude)scripts/check_private_infra.sh"; then
    echo "[audit] ERROR: Found UNC/Windows internal share paths in tracked files above!"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "[audit] FAILED: Private infrastructure identifiers detected."
    exit 1
fi

echo "[audit] PASSED: No private infrastructure leaks found in tracked files."
exit 0
