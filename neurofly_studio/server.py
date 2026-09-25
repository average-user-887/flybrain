"""The studio's local web server: the page, a small JSON API and the queue worker.

    python -m neurofly_studio serve --graph-dir G --connectome-dir C

serves ``web/studio.html`` (and the rest of ``web/``, so the 3D replay page is
same-origin) on http://127.0.0.1:8782/.  Builds become jobs in a
``neurofly_body`` run queue (neurofly_body/run_queue.py); by default this
process also runs that queue's worker, one run at a time.

API (JSON unless noted):

    GET  /api/studio/catalog                      paradigms, badges, parameters
    GET  /api/studio/runs                         queue jobs and curated runs
    POST /api/studio/experiments                  queue an experiment file
    POST /api/studio/runs/<name>/cancel           drop a job that has not started
    GET  /api/studio/metrics/<source>/<name>      comparison numbers of a finished run
    GET  /api/studio/files/<source>/<name>/<file> summary.json, manifest.json or
                                                  body.nfbody (gzip, not inflated)

``<source>`` is ``queue`` or ``curated``.  POST needs ``Content-Type:
application/json`` and, when the browser sends one, a same-host ``Origin``, so
another web page cannot queue runs through a visitor's browser.
"""

from __future__ import annotations

import fcntl
import json
import mimetypes
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from neurofly_body import run_queue

from . import catalog as catalog_mod
from . import experiment as experiment_mod
from .metrics import run_metrics

PROJECT_ROOT = catalog_mod.PROJECT_ROOT
WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_QUEUE = PROJECT_ROOT / "outputs" / "studio-queue"
DEFAULT_CURATED = PROJECT_ROOT / "experiment_data" / "curated"
RUN_FILES = {"summary.json": "application/json", "manifest.json": "application/json",
             "body.nfbody": "application/gzip"}
MAX_BODY = 16 * 1024
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class Studio:
    """State behind the HTTP handler; usable directly from Python and tests."""

    def __init__(self, queue_dir: Path = DEFAULT_QUEUE, curated_dir: Path = DEFAULT_CURATED, *,
                 graph_args: list[str] | None = None,
                 runner: Callable[[list[str], Path], int] | None = None) -> None:
        self.queue_dir = run_queue.init(Path(queue_dir))
        self.meta_dir = self.queue_dir / "studio"
        self.meta_dir.mkdir(exist_ok=True)
        self.curated_dir = Path(curated_dir)
        self.graph_args = list(graph_args or [])
        self.runner = runner
        self.worker: threading.Thread | None = None
        self.worker_state = "off"
        self._lock_handle = None
        self._submit_lock = threading.Lock()

    # -- worker ---------------------------------------------------------------
    def start_worker(self, poll_s: float = 2.0,
                     log: Callable[[str], None] = lambda message: print(message, flush=True)) -> str:
        """Run the queue in a background thread unless another studio already does."""
        handle = (self.queue_dir / ".studio-worker.lock").open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            self.worker_state = "another studio process runs this queue"
            return self.worker_state
        self._lock_handle = handle
        kwargs: dict[str, Any] = {"watch_s": poll_s, "log": log}
        if self.runner is not None:
            kwargs["runner"] = self.runner
        self.worker = threading.Thread(target=run_queue.run, args=(self.queue_dir,), kwargs=kwargs,
                                       name="studio-queue-worker", daemon=True)
        self.worker.start()
        self.worker_state = "running"
        return self.worker_state

    # -- queries --------------------------------------------------------------
    def catalog(self) -> dict[str, Any]:
        return catalog_mod.catalog()

    def _job(self, state: str, path: Path) -> dict[str, Any] | None:
        job = _read_json(path)
        if job is None:
            return None
        name = job["name"]
        meta = _read_json(self.meta_dir / f"{name}.json") or {}
        run_dir = self.queue_dir / "runs" / name
        return {
            "source": "queue", "name": name, "state": state, "order": path.name.split("-", 1)[0],
            "title": meta.get("title", name), "paradigm": meta.get("paradigm"),
            "role": meta.get("role"), "pair": meta.get("pair"), "seed": meta.get("seed"),
            "parameters": meta.get("parameters"), "controller": meta.get("controller"),
            "label": meta.get("label", "exploratory"),
            "added_at": job.get("added_at"), "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"), "exit_status": job.get("exit_status"),
            "wall_time_s": job.get("wall_time_s"), "error": job.get("error"),
            "has_recording": state == "done" and (run_dir / "body.nfbody").is_file(),
        }

    def _curated(self) -> list[dict[str, Any]]:
        runs = []
        if not self.curated_dir.is_dir():
            return runs
        for run_dir in sorted(self.curated_dir.iterdir()):
            info = _read_json(run_dir / "curated.json")
            if info is None or not _NAME.match(run_dir.name):
                continue
            runs.append({"source": "curated", "name": run_dir.name, "state": "done",
                         "title": info.get("title", run_dir.name), "paradigm": info.get("paradigm"),
                         "explanation": info.get("explanation"), "role": info.get("role"),
                         "pair": info.get("pair"), "parameters": info.get("parameters"),
                         "label": info.get("label", "exploratory"),
                         "has_recording": (run_dir / "body.nfbody").is_file()})
        return runs

    def runs(self) -> dict[str, Any]:
        jobs = []
        for state in run_queue.STATES:
            for path in sorted((self.queue_dir / state).glob("*.json")):
                job = self._job(state, path)
                if job is not None:
                    jobs.append(job)
        return {"queue": jobs, "curated": self._curated(), "worker": self.worker_state,
                "queue_dir": str(self.queue_dir)}

    def run_dir(self, source: str, name: str) -> Path:
        if not _NAME.match(name or ""):
            raise FileNotFoundError(name)
        base = {"queue": self.queue_dir / "runs", "curated": self.curated_dir}.get(source)
        if base is None:
            raise FileNotFoundError(source)
        path = base / name
        if source == "queue" and not any(
                (_read_json(p) or {}).get("name") == name for p in (self.queue_dir / "done").glob("*.json")):
            raise FileNotFoundError(f"{name} has not finished")
        if not path.is_dir():
            raise FileNotFoundError(name)
        return path

    # -- actions --------------------------------------------------------------
    def submit(self, raw: Any) -> dict[str, Any]:
        """Validate an experiment file and append its runs to the queue."""
        from neurofly_body.cli import _parser

        experiment = experiment_mod.validate(raw)
        with self._submit_lock:  # names carry a one-second timestamp
            planned = experiment_mod.plan(experiment, graph_args=self.graph_args)
            taken = {p.stem.split("-", 1)[1] for s in run_queue.STATES
                     for p in (self.queue_dir / s).glob("*.json")}
            while any(run.name in taken for run in planned):
                time.sleep(1.0)
                planned = experiment_mod.plan(experiment, graph_args=self.graph_args)
            for run in planned:  # the runner's own parser rejects anything it would reject
                try:
                    _parser().parse_args(["run", *run.argv, "--output", "validate-only"])
                except SystemExit as error:
                    raise experiment_mod.ExperimentError(f"the runner rejected {run.argv}") from error
            pairs = {}
            for run in planned:
                if run.role != "experiment":
                    pairs[run.name] = run.name[: -len("-control")]
                    pairs[run.name[: -len("-control")]] = run.name
            for run in planned:
                meta = {"schema": "neurofly-studio-run-v1", "name": run.name, "role": run.role,
                        "seed": run.seed, "pair": pairs.get(run.name), "argv": run.argv,
                        "experiment": experiment, "title": experiment["title"],
                        "paradigm": experiment["paradigm"], "controller": experiment["controller"],
                        "parameters": {**experiment["parameters"], "seed": run.seed},
                        "label": experiment_mod.LABEL}
                (self.meta_dir / f"{run.name}.json").write_text(
                    json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                run_queue.add(self.queue_dir, run.name, run.argv)
        return {"experiment": experiment, "queued": [run.name for run in planned]}

    def cancel(self, name: str) -> dict[str, Any]:
        for path in (self.queue_dir / "pending").glob("*.json"):
            if (_read_json(path) or {}).get("name") == name:
                try:
                    path.unlink()
                except FileNotFoundError:
                    break  # the worker picked it up in the meantime
                (self.meta_dir / f"{name}.json").unlink(missing_ok=True)
                return {"cancelled": name}
        raise FileNotFoundError(f"{name} is not waiting in the queue (it may already be running)")


class StudioHandler(BaseHTTPRequestHandler):
    studio: Studio
    server_version = "NeuroFlyStudio/1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data, default=str).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib name
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        parts = [p for p in path.split("/") if p]
        try:
            if parts[:2] == ["api", "studio"]:
                return self._api_get(parts[2:])
            return self._static(path)
        except FileNotFoundError as error:
            return self._error(404, f"not found: {error}")

    def _api_get(self, parts: list[str]) -> None:
        if parts == ["catalog"]:
            return self._json(self.studio.catalog())
        if parts == ["runs"]:
            return self._json(self.studio.runs())
        if len(parts) == 3 and parts[0] == "metrics":
            return self._json(run_metrics(self.studio.run_dir(parts[1], parts[2])))
        if len(parts) == 4 and parts[0] == "files" and parts[3] in RUN_FILES:
            file = self.studio.run_dir(parts[1], parts[2]) / parts[3]
            if not file.is_file():
                raise FileNotFoundError(parts[3])
            return self._send(200, file.read_bytes(), RUN_FILES[parts[3]])
        raise FileNotFoundError("/".join(parts))

    def _static(self, path: str) -> None:
        relative = path.lstrip("/") or "studio.html"
        target = (WEB_DIR / relative).resolve()
        if WEB_DIR.resolve() not in target.parents or not target.is_file():
            raise FileNotFoundError(relative)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True  # not a browser, or a same-origin navigation
        return urlsplit(origin).netloc == self.headers.get("Host")

    def do_POST(self) -> None:  # noqa: N802
        parts = [p for p in urlsplit(self.path).path.split("/") if p]
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self._error(415, "send the experiment as application/json")
        if not self._same_origin():
            return self._error(403, "cross-origin requests cannot queue runs")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            return self._error(413, "request body too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._error(400, "invalid JSON")
        try:
            if parts == ["api", "studio", "experiments"]:
                return self._json(self.studio.submit(body), 201)
            if len(parts) == 5 and parts[:3] == ["api", "studio", "runs"] and parts[4] == "cancel":
                return self._json(self.studio.cancel(parts[3]))
        except experiment_mod.ExperimentError as error:
            return self._error(400, str(error))
        except FileNotFoundError as error:
            return self._error(404, str(error))
        return self._error(404, "endpoint not found")


def make_server(studio: Studio, host: str = "127.0.0.1", port: int = 8782) -> ThreadingHTTPServer:
    handler = type("BoundStudioHandler", (StudioHandler,), {"studio": studio})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


__all__ = ["Studio", "StudioHandler", "make_server", "is_loopback", "DEFAULT_QUEUE", "DEFAULT_CURATED"]
