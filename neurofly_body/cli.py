"""Command-line entry point for the embodied co-simulation MVP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .decoder import DNa02CPGDecoder
from .runner import EmbodiedConfig, run_embodied


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neurofly_body",
        description="Couple the verified MaleCNS v3 graph to a FlyGym 2.1 articulated fly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run deterministic graph-body co-simulation")
    run.add_argument("--duration", type=float, required=True, metavar="SECONDS")
    run.add_argument("--output", type=Path, required=True, metavar="DIRECTORY")
    run.add_argument(
        "--mode", choices=("intact", "output-disconnected"), default="intact"
    )
    run.add_argument("--graph-dir", type=Path)
    run.add_argument("--connectome-dir", type=Path)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--neural-dt-ms", type=float, default=2.0)
    run.add_argument("--physics-dt-s", type=float, default=0.0001)
    run.add_argument("--warmup-s", type=float, default=0.05)
    run.add_argument("--world-angular-velocity-rad-s", type=float, default=4.0)
    run.add_argument("--contrast", type=float, default=1.0)
    run.add_argument("--decoder-tau-ms", type=float, default=50.0)
    run.add_argument("--cpg-gain-per-hz", type=float, default=0.04)
    run.add_argument("--max-cpg-drive", type=float, default=1.2)
    run.add_argument(
        "--video",
        action="store_true",
        help="render output/body.mp4 offscreen (set MUJOCO_GL as needed, e.g. egl)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "run":  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)

    # Import after argument validation so CLI help has no heavyweight dependency.
    try:
        from brainlab.cosim_server import ConnectomeServer
        from .flygym_body import FlyGymBody
    except ImportError as error:
        raise SystemExit(
            "The embodied runtime requires brainlab plus FlyGym 2.1.0 and MuJoCo 3.9. "
            f"Import failed: {error}"
        ) from error

    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        neural = ConnectomeServer(
            graph_dir=args.graph_dir,
            connectome_dir=args.connectome_dir,
            allow_synthetic=False,
            dynamics="v3",
            transmitter_policy="v3-modulatory-only",
            unclear_mode="excitatory",
            engineered_assistance=False,
            optomotor_seed=args.seed,
        )
    except TypeError as error:
        raise SystemExit(
            "ConnectomeServer lacks the required explicit v3 transmitter-policy API; "
            "deploy the matching brainlab backend before running the body MVP."
        ) from error

    video_path = args.output.resolve() / "body.mp4" if args.video else None
    body = None
    try:
        body = FlyGymBody(
            physics_dt_s=args.physics_dt_s,
            warmup_s=args.warmup_s,
            video_path=video_path,
        )
        config = EmbodiedConfig(
            duration_s=args.duration,
            output_dir=args.output,
            mode=args.mode,
            neural_dt_ms=args.neural_dt_ms,
            physics_dt_s=args.physics_dt_s,
            world_angular_velocity_rad_s=args.world_angular_velocity_rad_s,
            contrast=args.contrast,
            seed=args.seed,
        )
        decoder = DNa02CPGDecoder(
            gain_per_hz=args.cpg_gain_per_hz,
            tau_ms=args.decoder_tau_ms,
            max_drive=args.max_cpg_drive,
        )
        summary = run_embodied(config, neural, body, decoder=decoder)
    except BaseException:
        if body is not None:
            body.close()
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
