"""Canonical CLI entry points for Project NeuroFly.

Usage:
  neurofly run [daemon options]
  neurofly download-data
  neurofly status
  neurofly capability
  neurofly record --paradigm P --seconds S --out FILE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_run(args: list[str]) -> int:
    """Run the continuous neurofly daemon / simulation server."""
    import neurofly_daemon
    sys.argv = [sys.argv[0]] + args
    neurofly_daemon.main()
    return 0


def cmd_download_data(args: list[str]) -> int:
    """Download and verify MaleCNS v1.0 connectome source tables."""
    from brainlab import download
    print("[neurofly] Downloading and verifying MaleCNS v1.0 dataset...", flush=True)
    download.main()
    print("[neurofly] Dataset download and verification complete.", flush=True)
    return 0


def cmd_status(args: list[str]) -> int:
    """Report system health, environment dependencies, graph, and body state."""
    from brainlab.graph_identity import DEFAULT_GRAPH_DIR, DEFAULT_CONNECTOME_DIR, verify_graph, GraphUnavailable
    print("=" * 60)
    print(" Project NeuroFly — System Status")
    print("=" * 60)

    # 1. Physics Engine
    try:
        import flygym
        import mujoco
        import importlib.metadata
        fg_v = getattr(flygym, "__version__", None) or importlib.metadata.version("flygym")
        mj_v = getattr(mujoco, "__version__", None) or importlib.metadata.version("mujoco")
        print(f" [Physics] FlyGym {fg_v} | MuJoCo {mj_v} (INSTALLED)")
    except Exception as e:
        print(f" [Physics] Embodied physics not available: {e}")

    # 2. Connectome Graph
    try:
        identity = verify_graph()
        print(f" [Connectome] MaleCNS v1.0: {identity.neurons:,} neurons, {identity.edges:,} synapses")
        print(f"              Graph SHA-256: {identity.graph_sha256[:16]}... (VERIFIED)")
    except GraphUnavailable as e:
        print(f" [Connectome] MaleCNS graph unavailable: {e}")

    # 3. Plasticity Circuit
    try:
        from brainlab.io_map import resolve_visual_heading_io
        vh = resolve_visual_heading_io()
        print(f" [Plasticity] WP6 Visual Heading: {vh.describe()['plastic_edges']} edges (VERIFIED)")
    except Exception as e:
        print(f" [Plasticity] Visual heading IO unavailable: {e}")

    print("=" * 60)
    return 0


def cmd_embodied(args: list[str]) -> int:
    """Run embodied graph-to-body co-simulation with FlyGym and MuJoCo."""
    from neurofly_body import cli as body_cli
    return body_cli.main(args)


def cmd_full_sim(args: list[str]) -> int:
    """Run unified full-connectome multi-task simulation across paradigms."""
    from experiments.full_connectome_simulation import main as sim_main
    sys.argv = [sys.argv[0]] + args
    return sim_main()


def cmd_record(args: list[str]) -> int:
    """Run a paradigm headless and write a deterministic .nfrec recording."""
    from neurofly import recording
    return recording.main(args)


def cmd_capability(args: list[str]) -> int:
    """Display the 14-paradigm capability matrix."""
    matrix_path = Path(__file__).resolve().parents[1] / "docs" / "CAPABILITY_MATRIX.md"
    if matrix_path.is_file():
        print(matrix_path.read_text(encoding="utf-8"))
        return 0
    else:
        print(f"Error: Capability matrix not found at {matrix_path}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="neurofly",
        description="Project NeuroFly: Whole-Brain Connectome Coupled to Embodied Biomechanics",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("run", help="Launch the neurofly daemon / simulation server")
    subparsers.add_parser("full-sim", help="Run unified life-long multi-task simulation across 14 paradigms")
    subparsers.add_parser("embodied", help="Run embodied physics co-simulation with FlyGym and MuJoCo")
    subparsers.add_parser("download-data", help="Download & verify MaleCNS connectome tables")
    subparsers.add_parser("status", help="Print system health, dependencies, and graph verification")
    subparsers.add_parser("capability", help="Print the 14-paradigm capability matrix")
    subparsers.add_parser("record", help="Record a paradigm run (.nfrec) for 1x replay in the dashboard")

    if not argv:
        parser.print_help()
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "run":
        return cmd_run(rest)
    elif cmd == "full-sim":
        return cmd_full_sim(rest)
    elif cmd == "embodied":
        return cmd_embodied(rest)
    elif cmd == "download-data":
        return cmd_download_data(rest)
    elif cmd == "status":
        return cmd_status(rest)
    elif cmd == "capability":
        return cmd_capability(rest)
    elif cmd == "record":
        return cmd_record(rest)
    elif cmd in ("-h", "--help"):
        parser.print_help()
        return 0
    else:
        # Default: if arguments look like daemon flags, forward to run
        return cmd_run(argv)


if __name__ == "__main__":
    sys.exit(main())
