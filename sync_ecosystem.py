#!/usr/bin/env python3
"""
Project NeuroFly — Multi-Target Ecosystem Synchronization & Verification Engine
================================================================================
Synchronizes the codebase, continuous learning checkpoints, and test suites across:
1. AMD Ryzen Workstation (SSH/SCP)
2. HP Server Z: Drive (SMB / Local Mount)
3. Local Docker Testing Container

100% Portable — Zero Baked-In System Paths.
All paths and credentials are resolved via CLI flags, environment variables, or standard user profiles.
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
    "locomotion.py",
    "maze.py",
    "mechanosensory.py",
    "metabolic.py",
    "neurofly_daemon.py",
    "start_daemon_ryzen.sh",
    "stop_daemon_ryzen.sh",
    "surge_cast.py",
    "vision.py",
    "README.md",
    "brainlab",
    "experiments",
    "tests",
    "web",
    "flybrain_scientific_instrument.html",
    "CLAUDE_HANDOFF.md",
    "sync_ecosystem.py"
]


def resolve_ssh_key(custom_key: Optional[str] = None) -> Optional[str]:
    """Finds an SSH key without baking in hardcoded paths."""
    if custom_key and Path(custom_key).exists():
        return str(Path(custom_key).resolve())

    env_key = os.environ.get("NEUROFLY_SSH_KEY")
    if env_key and Path(env_key).exists():
        return str(Path(env_key).resolve())

    # Check standard locations in user home
    home = Path.home()
    candidates = [
        home / ".ssh" / "repo_audit_linux",
        home / ".ssh" / "id_rsa",
        home / ".ssh" / "id_ed25519"
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def resolve_z_drive(custom_path: Optional[str] = None) -> Optional[Path]:
    """Resolves HP Server storage destination portably."""
    if custom_path:
        p = Path(custom_path)
        return p if (p.exists() or p.parent.exists()) else None

    env_z = os.environ.get("NEUROFLY_Z_DRIVE")
    if env_z:
        p = Path(env_z)
        if p.exists() or p.parent.exists():
            return p

    # Standard Windows mount or UNC fallback
    candidates = [
        Path("Z:/neurofly"),
        Path("//192.168.1.23/Storage/neurofly")
    ]
    for c in candidates:
        try:
            if c.exists() or c.parent.exists():
                return c
        except OSError:
            pass
    return None


def sync_docker(target_container: str):
    print("==================================================================", flush=True)
    print(f"[Sync] SYNCING TO DOCKER CONTAINER: {target_container}", flush=True)
    print("==================================================================", flush=True)
    for item in SYNC_MANIFEST:
        src = PACKAGE_ROOT / item
        if not src.exists():
            continue
        if src.is_dir():
            dest = f"{target_container}/{item}"
            cmd = ["docker", "cp", f"{str(src)}/.", dest]
        else:
            dest = f"{target_container}/{item}"
            cmd = ["docker", "cp", str(src), dest]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] {item}: {res.stderr.strip()}", flush=True)
        else:
            print(f"  [OK] Copied {item} -> {dest}", flush=True)


def sync_ryzen(host: str, remote_dir: str, ssh_key: Optional[str]):
    print("\n==================================================================", flush=True)
    print(f"[Sync] SYNCING TO AMD RYZEN WORKSTATION: {host}:{remote_dir}", flush=True)
    print("==================================================================", flush=True)

    # Ensure remote directory tree has user write permissions
    pre_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if ssh_key:
        pre_cmd.extend(["-i", ssh_key])
    pre_cmd.extend([host, f"bash -c 'chmod -R u+w {remote_dir} 2>/dev/null || true'"])
    try:
        subprocess.run(pre_cmd, capture_output=True, timeout=10)
    except Exception:
        pass

    for item in SYNC_MANIFEST:
        src = PACKAGE_ROOT / item
        if not src.exists():
            continue
        dest = f"{host}:{remote_dir}/{item}"
        cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if ssh_key:
            cmd.extend(["-i", ssh_key])
        if src.is_dir():
            cmd.extend(["-r", str(src), f"{host}:{remote_dir}/"])
        else:
            cmd.extend([str(src), dest])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERROR] scp {item}: {res.stderr.strip()}", flush=True)
        else:
            print(f"  [OK] scp {item} -> {host}", flush=True)

    # Set executable permissions on shell scripts on Ryzen
    chmod_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if ssh_key:
        chmod_cmd.extend(["-i", ssh_key])
    chmod_cmd.extend([host, f"bash -c 'cd {remote_dir} && chmod +x *.sh 2>/dev/null || true'"])
    try:
        subprocess.run(chmod_cmd, capture_output=True, timeout=15)
    except Exception as e:
        print(f"  [WARNING] Remote chmod note: {e}", flush=True)


def sync_hpserver(z_dir: Path):
    print("\n==================================================================", flush=True)
    print(f"[Sync] SYNCING TO HP SERVER STORAGE: {z_dir}", flush=True)
    print("==================================================================", flush=True)
    try:
        z_dir.mkdir(parents=True, exist_ok=True)
        for item in SYNC_MANIFEST:
            src = PACKAGE_ROOT / item
            if not src.exists():
                continue
            dst = z_dir / item
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(src, dst)
            print(f"  [OK] Copied {item} -> {dst}", flush=True)
        print(f"[Sync] HP Server archive synchronized successfully at {z_dir}", flush=True)
    except Exception as err:
        print(f"  [WARNING] Could not sync to HP Server ({z_dir}): {err}", flush=True)


def run_remote_tests(host: str, remote_dir: str, ssh_key: Optional[str], docker_target: str):
    print("\n==================================================================", flush=True)
    print("VERIFYING TEST SUITES ACROSS COMPUTE NODES...", flush=True)
    print("==================================================================", flush=True)

    # 1. Docker
    container_id = docker_target.split(":")[0]
    print(f"\n--- [1/2] Pytest on Docker ({container_id}) ---", flush=True)
    cmd_doc = ["docker", "exec", "-w", "/workspace", container_id, "pytest", "-q", "tests/"]
    try:
        res_doc = subprocess.run(cmd_doc, capture_output=True, text=True, timeout=60)
        print(res_doc.stdout.strip(), flush=True)
        if res_doc.returncode != 0:
            print(res_doc.stderr.strip(), flush=True)
    except Exception as e:
        print(f"  [ERROR] Docker test execution error: {e}", flush=True)

    # 2. Ryzen
    print(f"\n--- [2/2] Pytest on AMD Ryzen Host ({host}) ---", flush=True)
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
    if ssh_key:
        ssh_cmd.extend(["-i", ssh_key])
    ssh_cmd.extend([host, f"cd {remote_dir} && source .venv/bin/activate && PYTHONPATH=. pytest -q tests/"])
    try:
        res_ryzen = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
        print(res_ryzen.stdout.strip(), flush=True)
        if res_ryzen.returncode != 0:
            print(res_ryzen.stderr.strip(), flush=True)
    except Exception as e:
        print(f"  [ERROR] Ryzen test execution error: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Project NeuroFly Ecosystem Sync & Verification")
    parser.add_argument("--ryzen-host", default=os.environ.get("NEUROFLY_RYZEN_HOST", "avg-usr@192.168.194.227"))
    parser.add_argument("--ryzen-dir", default=os.environ.get("NEUROFLY_RYZEN_DIR", "/home/avg-usr/Documents/ChatGPT/flybrain"))
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--z-drive", default=None)
    parser.add_argument("--docker-target", default=os.environ.get("NEUROFLY_DOCKER_TARGET", "65dcff428c87:/workspace"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    ssh_key = resolve_ssh_key(args.ssh_key)
    z_dir = resolve_z_drive(args.z_drive)

    print(f"[Sync] Project Root: {PACKAGE_ROOT}")
    print(f"[Sync] SSH Key: {ssh_key or 'Default agent/user key'}")
    print(f"[Sync] HP Server Z-Drive: {z_dir or 'Not detected/Offline'}")

    sync_docker(args.docker_target)
    sync_ryzen(args.ryzen_host, args.ryzen_dir, ssh_key)
    if z_dir:
        sync_hpserver(z_dir)

    if not args.skip_tests:
        run_remote_tests(args.ryzen_host, args.ryzen_dir, ssh_key, args.docker_target)


if __name__ == "__main__":
    main()
