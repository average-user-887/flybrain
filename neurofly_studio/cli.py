"""python -m neurofly_studio: serve the studio, or queue experiment files headlessly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .experiment import ExperimentError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m neurofly_studio",
                                     description="NeuroFly experiment studio (docs/EXPERIMENT_STUDIO.md)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        from .server import DEFAULT_CURATED, DEFAULT_QUEUE

        p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE, metavar="DIR",
                       help="neurofly_body run queue directory (default outputs/studio-queue)")
        p.add_argument("--curated", type=Path, default=DEFAULT_CURATED, metavar="DIR",
                       help="curated runs shown in the gallery (default experiment_data/curated)")
        p.add_argument("--graph-dir", type=Path, help="prepared MaleCNS graph for connectome runs")
        p.add_argument("--connectome-dir", type=Path, help="MaleCNS annotation data for connectome runs")

    serve = sub.add_parser("serve", help="serve the studio page and API")
    common(serve)
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind address (default 127.0.0.1; anyone who can reach another "
                            "address can queue runs)")
    serve.add_argument("--port", type=int, default=8782)
    serve.add_argument("--no-worker", action="store_true",
                       help="do not execute the queue here (run `python -m neurofly_body queue "
                            "run QUEUE --watch 5` separately)")

    submit = sub.add_parser("submit", help="validate an experiment file and queue its runs")
    common(submit)
    submit.add_argument("experiment", type=Path, metavar="EXPERIMENT.json")
    submit.add_argument("--dry-run", action="store_true", help="print the planned runs only")

    status = sub.add_parser("status", help="list the studio queue")
    common(status)
    return parser


def _graph_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.graph_dir:
        out += ["--graph-dir", str(args.graph_dir.resolve())]
    if args.connectome_dir:
        out += ["--connectome-dir", str(args.connectome_dir.resolve())]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from .server import Studio, is_loopback, make_server
    from . import experiment as experiment_mod

    studio = Studio(args.queue, args.curated, graph_args=_graph_args(args))
    if args.command == "submit":
        raw = json.loads(args.experiment.read_text(encoding="utf-8"))
        try:
            if args.dry_run:
                experiment = experiment_mod.validate(raw)
                planned = experiment_mod.plan(experiment, graph_args=studio.graph_args)
                print(json.dumps({"experiment": experiment,
                                  "runs": [vars(run) for run in planned]}, indent=2))
            else:
                print(json.dumps(studio.submit(raw), indent=2))
        except ExperimentError as error:
            raise SystemExit(f"{args.experiment}: {error}") from error
        return 0
    if args.command == "status":
        print(json.dumps(studio.runs(), indent=2, default=str))
        return 0

    if not studio.graph_args:
        print("[studio] no --graph-dir/--connectome-dir given: connectome runs use "
              "neurofly_body's default graph and connectome locations")
    if not is_loopback(args.host):
        print(f"[studio] WARNING: listening on {args.host}; anyone who can reach it can queue runs")
    worker = "off (--no-worker)" if args.no_worker else studio.start_worker()
    server = make_server(studio, args.host, args.port)
    print(f"[studio] http://{args.host}:{args.port}/  queue {studio.queue_dir}  worker: {worker}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
