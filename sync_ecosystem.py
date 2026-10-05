#!/usr/bin/env python3
"""
RETIRED (2026-10-05): this tool no longer runs. See docs/RETIREMENT_INDEX.md.
Its SYNC_MANIFEST below omits current product and provenance modules (neurofly/,
neurofly_body/, neurofly_studio/, validation/, provenance.py, experiment_registry.py,
pyproject.toml, ...), so a synced target could not even import the daemon. Use git.
The original description is kept for the record:

Project NeuroFly — maintainer tool: push the working tree to test/compute targets
================================================================================
Copies the files in SYNC_MANIFEST to any of three optional targets and, unless
``--skip-tests`` is given, runs the test suite on the remote ones:

1. A remote workstation over SSH/SCP   (``--remote-host`` / ``NEUROFLY_REMOTE_HOST``)
2. A local Docker container            (``--docker-target`` / ``NEUROFLY_DOCKER_TARGET``)
3. A mounted archive directory         (``--archive-dir`` / ``NEUROFLY_ARCHIVE_DIR``)

Every target is opt-in: a stage whose target is not configured is skipped. No
hostnames, addresses, key names or container ids are baked into this file;
configure them per machine through the environment or flags. For day-to-day
development prefer ``git``; this script exists for hosts without a checkout.

Examples::

    NEUROFLY_REMOTE_HOST=user@workstation NEUROFLY_REMOTE_DIR=/srv/neurofly \\
        python sync_ecosystem.py
    python sync_ecosystem.py --docker-target <container>:/workspace --skip-tests
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent

# File / folder whitelist for synchronization
SYNC_MANIFEST = [
    "arena.py",
    "central_complex.py",
    "circuit.py",
    "connectome_bridge.py",
    "connectome_client.py",
    "data_logger.py",
    "env_adapter.py",
    "learning_recorder.py",
    "locomotion.py",
    "maze.py",
    "mechanosensory.py",
    "metabolic.py",
    "neurofly_daemon.py",
    "start_daemon.sh",
    "stop_daemon.sh",
    "stream_gateway.py",
    "surge_cast.py",
    "vision.py",
    "README.md",
    "brainlab",
    "experiments",
    "tests",
    "web",
    "flybrain_scientific_instrument.html",
    "sync_ecosystem.py",
]

SSH_OPTS = ["-o", "BatchMode=yes"]


def resolve_ssh_key(custom_key: Optional[str] = None) -> Optional[str]:
    """``--ssh-key`` > ``NEUROFLY_SSH_KEY`` > the SSH agent / default identity."""
    for candidate in (custom_key, os.environ.get("NEUROFLY_SSH_KEY")):
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser().resolve())
    return None


def resolve_archive_dir(custom_path: Optional[str] = None) -> Optional[Path]:
    """``--archive-dir`` > ``NEUROFLY_ARCHIVE_DIR``; None when neither is set."""
    raw = custom_path or os.environ.get("NEUROFLY_ARCHIVE_DIR")
    if not raw:
        return None
    p = Path(raw).expanduser()
    try:
        if p.exists() or p.parent.exists():
            return p
    except OSError:
        pass
    print(f"[Sync] Archive directory not reachable: {p}", flush=True)
    return None


def _ssh_base(ssh_key: Optional[str]) -> List[str]:
    cmd = ["ssh", *SSH_OPTS]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    return cmd


def sync_docker(target_container: str) -> None:
    print("==================================================================", flush=True)
    print(f"[Sync] SYNCING TO DOCKER CONTAINER: {target_container}", flush=True)
    print("==================================================================", flush=True)
    for item in SYNC_MANIFEST:
        src = PACKAGE_ROOT / item
        if not src.exists():
            continue
        dest = f"{target_container}/{item}"
        cmd = ["docker", "cp", f"{src}/." if src.is_dir() else str(src), dest]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] {item}: {res.stderr.strip()}", flush=True)
        else:
            print(f"  [OK] Copied {item} -> {dest}", flush=True)


def sync_remote(host: str, remote_dir: str, ssh_key: Optional[str]) -> None:
    print("\n==================================================================", flush=True)
    print(f"[Sync] SYNCING TO REMOTE HOST: {host}:{remote_dir}", flush=True)
    print("==================================================================", flush=True)

    pre_cmd = _ssh_base(ssh_key) + [host, f"mkdir -p {remote_dir} && chmod -R u+w {remote_dir} 2>/dev/null || true"]
    try:
        subprocess.run(pre_cmd, capture_output=True, timeout=15)
    except Exception as err:
        print(f"  [WARNING] Remote preparation skipped: {err}", flush=True)

    for item in SYNC_MANIFEST:
        src = PACKAGE_ROOT / item
        if not src.exists():
            continue
        cmd = ["scp", *SSH_OPTS]
        if ssh_key:
            cmd.extend(["-i", ssh_key])
        if src.is_dir():
            cmd.extend(["-r", str(src), f"{host}:{remote_dir}/"])
        else:
            cmd.extend([str(src), f"{host}:{remote_dir}/{item}"])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] scp {item}: {res.stderr.strip()}", flush=True)
        else:
            print(f"  [OK] scp {item} -> {host}", flush=True)

    chmod_cmd = _ssh_base(ssh_key) + [host, f"cd {remote_dir} && chmod +x *.sh 2>/dev/null || true"]
    try:
        subprocess.run(chmod_cmd, capture_output=True, timeout=15)
    except Exception as err:
        print(f"  [WARNING] Remote chmod note: {err}", flush=True)


def sync_archive(archive_dir: Path) -> None:
    print("\n==================================================================", flush=True)
    print(f"[Sync] SYNCING TO ARCHIVE DIRECTORY: {archive_dir}", flush=True)
    print("==================================================================", flush=True)
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for item in SYNC_MANIFEST:
            src = PACKAGE_ROOT / item
            if not src.exists():
                continue
            dst = archive_dir / item
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)
            print(f"  [OK] Copied {item} -> {dst}", flush=True)
        print(f"[Sync] Archive synchronized at {archive_dir}", flush=True)
    except Exception as err:
        print(f"  [WARNING] Could not sync archive ({archive_dir}): {err}", flush=True)


def run_remote_tests(host: Optional[str], remote_dir: Optional[str], ssh_key: Optional[str],
                     docker_target: Optional[str]) -> None:
    print("\n==================================================================", flush=True)
    print("VERIFYING TEST SUITES ON CONFIGURED TARGETS...", flush=True)
    print("==================================================================", flush=True)

    if docker_target:
        container_id = docker_target.split(":")[0]
        print(f"\n--- Pytest in Docker ({container_id}) ---", flush=True)
        cmd_doc = ["docker", "exec", "-w", "/workspace", container_id, "pytest", "-q", "tests/"]
        try:
            res = subprocess.run(cmd_doc, capture_output=True, text=True, timeout=600)
            print(res.stdout.strip(), flush=True)
            if res.returncode != 0:
                print(res.stderr.strip(), flush=True)
        except Exception as err:
            print(f"  [ERROR] Docker test execution error: {err}", flush=True)

    if host and remote_dir:
        print(f"\n--- Pytest on remote host ({host}) ---", flush=True)
        ssh_cmd = _ssh_base(ssh_key) + [
            host,
            f"cd {remote_dir} && PYTHONPATH=. .venv/bin/python -m pytest -q tests/",
        ]
        try:
            res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=600)
            print(res.stdout.strip(), flush=True)
            if res.returncode != 0:
                print(res.stderr.strip(), flush=True)
        except Exception as err:
            print(f"  [ERROR] Remote test execution error: {err}", flush=True)


RETIRED_NOTICE = """\
sync_ecosystem.py is RETIRED and does not run. Its file manifest (SYNC_MANIFEST)
omits current product and provenance modules: neurofly/, neurofly_body/,
neurofly_studio/, validation/, provenance.py, experiment_registry.py,
experiment_brains.py, assay_controls.py, assay_response.py, online_metrics.py and
pyproject.toml. A target synced with it cannot import the daemon.

Use instead: git (clone, fetch or push a branch) on the target host.

Last historical commit and details: docs/RETIREMENT_INDEX.md
"""


def main(argv: Optional[List[str]] = None) -> int:
    """Retired: print why (``--help`` shows it too) and copy nothing."""
    argparse.ArgumentParser(prog="sync_ecosystem.py", description=RETIRED_NOTICE,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_known_args(argv)
    print(RETIRED_NOTICE, file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    sys.exit(main())
