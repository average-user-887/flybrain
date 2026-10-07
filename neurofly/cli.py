"""Canonical CLI entry points for Project NeuroFly.

Usage:
  neurofly run [daemon options]      (one process; --split runs the simulation and the
                                      web server as two processes, v0.5 prototype)
  neurofly sim-serve [daemon options]   headless simulation process only
  neurofly web-serve [daemon options]   web process only (dashboard, REST, SSE)
  neurofly full-sim              (retired: prints why and exits non-zero)
  neurofly embodied [options]
  neurofly download-data
  neurofly status
  neurofly capability
  neurofly validate <spec>
  neurofly record --paradigm P --seconds S --out FILE
  neurofly cohort run|resume|verify [options]
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path


def cmd_run(args: list[str]) -> int:
    """Run the continuous neurofly daemon / simulation server."""
    import neurofly_daemon
    from neurofly import storage
    try:
        # Parse only when a saved selection exists. Preserve all legacy forwarding.
        path = None if any(flag in args for flag in ("-h", "--help")) else storage.config_path()
        if path is not None and (path.exists() or path.is_symlink()):
            parsed = neurofly_daemon.build_arg_parser().parse_args(args)
            args = storage.run_arguments(args, parsed, path)
    except storage.StorageError as exc:
        print(f"neurofly: {exc}", file=sys.stderr)
        return 2
    sys.argv = ["neurofly run"] + args
    neurofly_daemon.run_daemon()
    return 0


def cmd_storage(args: list[str]) -> int:
    """Remember existing directories; never copy, move or assess model compatibility."""
    from neurofly import storage
    parser = argparse.ArgumentParser(prog="neurofly storage", description=cmd_storage.__doc__,
        epilog="Config: NEUROFLY_CONFIG_HOME/storage.json, otherwise XDG_CONFIG_HOME/neurofly/storage.json. Selected stores must survive checkout replacement.")
    commands = parser.add_subparsers(dest="action", required=True)
    use = commands.add_parser("use", help="Adopt two explicit existing directories")
    use.add_argument("--output-dir", required=True)
    use.add_argument("--data-dir", required=True)
    commands.add_parser("show", help="Read-only JSON selection and directory preflight")
    parsed = parser.parse_args(args)
    try:
        if parsed.action == "use":
            storage.adopt(parsed.output_dir, parsed.data_dir)
        print(json.dumps(storage.show(Path(__file__).resolve().parents[1]), sort_keys=True))
        return 0
    except storage.StorageError as exc:
        print(f"neurofly: {exc}", file=sys.stderr)
        return 2


def cmd_download_data(args: list[str]) -> int:
    """Download and verify MaleCNS v1.0 connectome source tables."""
    from brainlab import download
    # Parse first: `download-data --help` (or a bad flag) prints/fails without downloading.
    download.build_parser().parse_args(args)
    print("[neurofly] Downloading and verifying MaleCNS v1.0 dataset...", flush=True)
    download.main(args)
    print("[neurofly] Dataset download and verification complete.", flush=True)
    return 0


def cmd_status(args: list[str]) -> int:
    """Report system health, environment dependencies, graph, and body state."""
    argparse.ArgumentParser(prog="neurofly status", description=cmd_status.__doc__).parse_args(args)
    from brainlab.graph_identity import verify_graph, GraphUnavailable
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
        print(f"              Graph content: {identity.graph_content_sha256[:16]}... (VERIFIED)")
        print(f"              Neuron map content: {identity.neuron_map_content_sha256[:16]}... (VERIFIED)")
        print(f"              Graph identity: {identity.graph_sha256[:16]}...")
    except GraphUnavailable as e:
        print(f" [Connectome] MaleCNS graph unavailable: {e}")

    # 3. Where a connectome brain would compute (the daemon prints the same at startup)
    try:
        from brainlab.brain import gpu_name, resolve_backend
        from brainlab.graph_identity import active_dynamics_version
        dynamics = active_dynamics_version()
        device = resolve_backend(dynamics)
        if device == 'wgpu-amd':
            print(f" [Compute] Brain backend: wgpu-amd requested for fixed LIF {dynamics}; "
                  "actual Vulkan/device identity is reported by the running daemon; learning unsupported")
        elif device == "cuda":
            print(f" [Compute] Brain backend: CUDA ({gpu_name() or 'unknown GPU'}) for LIF {dynamics}")
        else:
            from brainlab.gpu_probe import explain
            reason = explain(dynamics)[1]
            print(f" [Compute] Brain backend: CPU for LIF {dynamics} (GPU not used: {reason})")
    except Exception as e:
        print(f" [Compute] Brain backend unknown: {type(e).__name__}: {e}")

    # 4. Plasticity Circuit
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


FULL_SIM_RETIRED = """\
neurofly full-sim is RETIRED and no longer runs. It was not a full connectome
simulation: the brain received empty sensory input (Arena.get_sensory_inputs does
not exist, so every step fed it {}), and its motor output was overwritten by the
modular controller inside arena.step.

Use instead:
  neurofly run --backend connectome-fixed --paradigm optomotor   dashboard on :8769
  neurofly record --backend connectome-fixed --paradigm optomotor --seconds 30 --out run
  neurofly validate run <spec> --out <new dir>    preregistered validation specs

Last historical commit and details: docs/RETIREMENT_INDEX.md
"""


def cmd_full_sim(args: list[str]) -> int:
    """Retired: explain why and exit non-zero (docs/RETIREMENT_INDEX.md)."""
    argparse.ArgumentParser(prog="neurofly full-sim", description=FULL_SIM_RETIRED,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_known_args(args)
    print(FULL_SIM_RETIRED, file=sys.stderr, end="")
    return 2


def cmd_validate(args: list[str]) -> int:
    """Run or check a preregistered validation spec (docs/VALIDATION_HARNESS.md)."""
    from validation.__main__ import main as validate_main
    return validate_main(args)


def cmd_record(args: list[str]) -> int:
    """Run a paradigm headless and write a deterministic .nfrec recording."""
    from neurofly import recording
    return recording.main(args)


def cmd_cohort(args: list[str]) -> int:
    """Many independent fixed-v3 brains: run, resume or verify a cohort."""
    from brainlab.cohort import runner
    return runner.main(args)


def cmd_capability(args: list[str]) -> int:
    """Display the 14-paradigm capability matrix."""
    argparse.ArgumentParser(prog="neurofly capability", description=cmd_capability.__doc__).parse_args(args)
    try:
        matrix_path = files("neurofly._docs").joinpath("CAPABILITY_MATRIX.md")
    except ModuleNotFoundError as exc:
        if exc.name != "neurofly._docs":
            raise
        # The resource package is mapped from docs by setuptools; a normal
        # checkout reads that same canonical source before it has been built.
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
    parser.add_argument("--version", action="store_true", help="Print the NeuroFly version and exit")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("storage", help="Adopt/show persistent existing training directories")
    subparsers.add_parser("run", help="Launch the neurofly daemon / simulation server")
    subparsers.add_parser("sim-serve", help="Run only the headless simulation process (run --process-mode sim)")
    subparsers.add_parser("web-serve", help="Run only the web process (run --process-mode web)")
    subparsers.add_parser("full-sim", help="RETIRED: not a full connectome simulation; prints why and exits")
    subparsers.add_parser("embodied", help="Run embodied physics co-simulation with FlyGym and MuJoCo")
    subparsers.add_parser("download-data", help="Download & verify MaleCNS connectome tables")
    subparsers.add_parser("status", help="Print system health, dependencies, and graph verification")
    subparsers.add_parser("capability", help="Print the 14-paradigm capability matrix")
    subparsers.add_parser("validate", help="Run or check a preregistered validation spec (run|check <spec>)")
    subparsers.add_parser("record", help="Record a paradigm run (.nfrec) for 1x replay in the dashboard")
    subparsers.add_parser("cohort", help="Run/resume/verify many independent fixed-v3 brains (run|resume|verify)")

    if not argv:
        parser.print_help()
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "storage":
        return cmd_storage(rest)
    elif cmd == "run":
        return cmd_run(rest)
    elif cmd in ("sim-serve", "web-serve"):
        return cmd_run(rest + ["--process-mode", cmd.split("-")[0]])
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
    elif cmd == "validate":
        return cmd_validate(rest)
    elif cmd == "record":
        return cmd_record(rest)
    elif cmd == "cohort":
        return cmd_cohort(rest)
    elif cmd == "--version":
        from neurofly import __version__
        print(f"neurofly {__version__}")
        return 0
    elif cmd in ("-h", "--help"):
        parser.print_help()
        return 0
    elif cmd.startswith("-"):
        # Bare daemon flags (e.g. `neurofly --port 8769`) are forwarded to run
        return cmd_run(argv)
    else:
        print(f"neurofly: unknown command {cmd!r}", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
