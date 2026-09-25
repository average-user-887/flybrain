"""Command-line entry point for the embodied co-simulation MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .decoder import DNa02CPGDecoder, DNCommandDecoder
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
    run.add_argument(
        "--controller", choices=("connectome", "modular"), default="connectome",
        help="connectome: MaleCNS v3 graph; modular: the arena's phenomenological "
             "optomotor model as a researcher baseline (neurofly_body/modular.py)",
    )
    run.add_argument("--modular-forward-drive", type=float, default=1.0,
                     help="modular only: tonic CPG amplitude (assumption)")
    run.add_argument("--modular-turn-gain", type=float, default=1.0,
                     help="modular only: yaw-bias-to-amplitude gain (assumption)")
    run.add_argument(
        "--decoder", choices=("dn-v2", "dna02-crossed-v1"), default="dn-v2",
        help="dn-v2: DNp09 forward, DNa02 ipsilateral stride, MDN reverse, GF event "
             "(docs/EMBODIED_MVP.md); dna02-crossed-v1: legacy MVP mapping",
    )
    run.add_argument("--decoder-tau-ms", type=float, default=50.0)
    run.add_argument("--max-cpg-drive", type=float, default=1.2)
    run.add_argument("--p9-gain-per-hz", type=float, default=0.02, help="dn-v2 (assumption)")
    run.add_argument("--dna02-stride-k-per-hz", type=float, default=0.01, help="dn-v2 (assumption)")
    run.add_argument("--mdn-gain-per-hz", type=float, default=0.02, help="dn-v2 (assumption)")
    run.add_argument("--cpg-gain-per-hz", type=float, default=0.04, help="dna02-crossed-v1 only")
    run.add_argument(
        "--video",
        action="store_true",
        help="render output/body.mp4 offscreen (set MUJOCO_GL as needed, e.g. egl)",
    )
    check = subparsers.add_parser(
        "replay-check",
        help="re-run a finished run from its manifest and require a bit-identical trajectory",
    )
    check.add_argument("run_dir", type=Path, metavar="RUN_DIR")
    check.add_argument("--output", type=Path, required=True, metavar="DIRECTORY",
                       help="new directory for the re-run (never overwritten)")
    return parser


# Arguments that change what is simulated.  ``output`` and ``video`` do not.
RUN_ARGUMENTS = (
    "duration", "mode", "graph_dir", "connectome_dir", "seed", "neural_dt_ms",
    "physics_dt_s", "warmup_s", "world_angular_velocity_rad_s", "contrast",
    "controller", "modular_forward_drive", "modular_turn_gain",
    "decoder", "decoder_tau_ms", "max_cpg_drive", "p9_gain_per_hz",
    "dna02_stride_k_per_hz", "mdn_gain_per_hz", "cpg_gain_per_hz",
)
# Runs recorded before an argument existed ran with this value.
# The dn-v2 gains did not exist then and do not affect the legacy decoder.
INVOCATION_BACKFILL = {"controller": "connectome", "modular_forward_drive": 1.0,
                       "modular_turn_gain": 1.0, "decoder": "dna02-crossed-v1", "p9_gain_per_hz": 0.02,
                       "dna02_stride_k_per_hz": 0.01, "mdn_gain_per_hz": 0.02}


def _invocation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in ((name, getattr(args, name)) for name in RUN_ARGUMENTS)
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_difference(a: Path, b: Path) -> int | None:
    """1-based telemetry line where two runs first differ, or None."""
    with a.open(encoding="utf-8") as left, b.open(encoding="utf-8") as right:
        for number, (line_a, line_b) in enumerate(zip(left, right), start=1):
            if line_a != line_b:
                return number
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay-check":
        return _replay_check(args.run_dir, args.output)
    if args.command != "run":  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    summary = _run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _replay_check(run_dir: Path, output: Path) -> int:
    """Re-run ``run_dir`` from its recorded invocation and compare trajectories."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit(f"{run_dir} did not complete (status {manifest.get('status')!r})")
    invocation = {**INVOCATION_BACKFILL, **(manifest.get("invocation") or {})}
    missing = [name for name in RUN_ARGUMENTS if name not in invocation]
    if missing:
        raise SystemExit(f"{run_dir}/manifest.json has no recorded invocation for {missing}")
    argv = ["run", "--output", str(output)]
    for name in RUN_ARGUMENTS:
        value = invocation[name]
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    args = _parser().parse_args(argv)
    summary = _run(args)

    replay_manifest = json.loads((Path(output) / "manifest.json").read_text(encoding="utf-8"))
    original_backend = manifest["neural_backend"].get("brain_backend")
    replay_backend = replay_manifest["neural_backend"].get("brain_backend")
    original_sha = _sha256_file(run_dir / "telemetry.jsonl")
    replay_sha = _sha256_file(Path(output) / "telemetry.jsonl")
    identical = original_sha == replay_sha
    receipt = {
        "schema": "neurofly-embodied-replay-check-v1",
        "verdict": "BIT_IDENTICAL" if identical else "DIVERGED",
        "original_run": str(run_dir.resolve()),
        "replay_run": str(Path(output).resolve()),
        "original_trajectory_sha256": original_sha,
        "replay_trajectory_sha256": replay_sha,
        "recorded_trajectory_sha256": manifest.get("trajectory_sha256"),
        "first_differing_record": None if identical else _first_difference(
            run_dir / "telemetry.jsonl", Path(output) / "telemetry.jsonl"),
        "records": summary["records"],
        "duration_s": summary["duration_s"],
        "brain_backend": {"original": original_backend, "replay": replay_backend},
        "graph_sha256": manifest["neural_backend"].get("graph_sha256"),
        "invocation": invocation,
        "replay_wall_time_s": summary["wall_time_s"],
        "replay_real_time_factor": summary["real_time_factor"],
    }
    if original_backend != replay_backend:
        receipt["note"] = "brain backends differ; CPU and GPU are not expected to match bit for bit"
    (Path(output) / "replay_check.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if identical else 1


def _run(args: argparse.Namespace) -> dict[str, Any]:
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
    if args.controller == "modular":
        from .modular import ModularCommandDecoder, ModularOptomotorBackend

        neural = ModularOptomotorBackend(forward_drive=args.modular_forward_drive,
                                         turn_gain=args.modular_turn_gain)
        return _run_with(args, neural, ModularCommandDecoder(max_drive=args.max_cpg_drive))
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
    if args.decoder == "dn-v2":
        decoder = DNCommandDecoder(
            gain_p9_per_hz=args.p9_gain_per_hz,
            k_dna02_per_hz=args.dna02_stride_k_per_hz,
            gain_mdn_per_hz=args.mdn_gain_per_hz,
            tau_ms=args.decoder_tau_ms,
            max_drive=args.max_cpg_drive,
        )
    else:
        decoder = DNa02CPGDecoder(
            gain_per_hz=args.cpg_gain_per_hz,
            tau_ms=args.decoder_tau_ms,
            max_drive=args.max_cpg_drive,
        )
    return _run_with(args, neural, decoder)


def _run_with(args: argparse.Namespace, neural: Any, decoder: Any) -> dict[str, Any]:
    from .flygym_body import FlyGymBody

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
        summary = run_embodied(config, neural, body, decoder=decoder,
                               invocation=_invocation(args))
    except BaseException:
        if body is not None:
            body.close()
        raise
    return summary
