"""Local run queue: embodied runs execute one after another, never in parallel.

The simulation runs slower than real time, so people queue experiments, walk
away, and watch the results later.  A queue is a directory:

    QUEUE/pending/0001-name.json   {"name": ..., "argv": [...run arguments...]}
    QUEUE/running/                 the job being executed (at most one)
    QUEUE/done/  QUEUE/failed/     finished job files, with their exit status
    QUEUE/runs/<name>/             each run's output directory
    QUEUE/logs/<name>.log          each run's stdout and stderr

Jobs run in file-name order, each in its own ``python -m neurofly_body run``
process, so one run's memory, GPU state or crash cannot leak into the next.
A job left in ``running/`` by a killed worker is moved to ``failed/`` on the
next start and never re-run silently: its output directory may be partial,
and runs refuse to overwrite an existing directory.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

STATES = ("pending", "running", "done", "failed")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
# Arguments the queue owns: every run's output goes to QUEUE/runs/<name>.
_RESERVED = ("--output",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def init(queue_dir: Path) -> Path:
    queue_dir = Path(queue_dir)
    for state in STATES + ("runs", "logs"):
        (queue_dir / state).mkdir(parents=True, exist_ok=True)
    return queue_dir


def add(queue_dir: Path, name: str, argv: Sequence[str]) -> Path:
    """Append a job.  ``argv`` are ``neurofly_body run`` arguments without --output."""
    queue_dir = init(queue_dir)
    if not _NAME.match(name):
        raise ValueError("job name: letters, digits, '.', '_' or '-', at most 100 characters")
    argv = [str(item) for item in argv]
    if any(item.split("=", 1)[0] in _RESERVED for item in argv):
        raise ValueError("the queue sets --output itself (QUEUE/runs/<name>)")
    taken = {json.loads(p.read_text(encoding="utf-8"))["name"]
             for state in STATES for p in (queue_dir / state).glob("*.json")}
    if name in taken or (queue_dir / "runs" / name).exists():
        raise FileExistsError(f"job {name!r} already exists in {queue_dir}")
    numbers = [int(p.name.split("-", 1)[0]) for state in STATES
               for p in (queue_dir / state).glob("*.json") if p.name.split("-", 1)[0].isdigit()]
    path = queue_dir / "pending" / f"{max(numbers, default=0) + 1:04d}-{name}.json"
    _write(path, {"name": name, "argv": argv, "added_at": _now()})
    return path


def status(queue_dir: Path) -> dict:
    queue_dir = init(queue_dir)
    return {state: sorted(json.loads(p.read_text(encoding="utf-8"))["name"]
                          for p in (queue_dir / state).glob("*.json"))
            for state in STATES}


def _subprocess_runner(argv: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.call([sys.executable, "-m", "neurofly_body", "run", *argv],
                               stdout=log, stderr=subprocess.STDOUT)


def run(queue_dir: Path, *, watch_s: float | None = None,
        runner: Callable[[list[str], Path], int] = _subprocess_runner,
        log: Callable[[str], None] = print) -> dict:
    """Execute pending jobs in order until none are left (or keep polling with watch_s)."""
    queue_dir = init(queue_dir)
    for stale in sorted((queue_dir / "running").glob("*.json")):
        job = json.loads(stale.read_text(encoding="utf-8"))
        job.update(finished_at=_now(), exit_status=None,
                   error="interrupted: the queue worker stopped while this job was running")
        _write(queue_dir / "failed" / stale.name, job)
        stale.unlink()
        log(f"[queue] {job['name']}: interrupted earlier, moved to failed/")
    while True:
        pending = sorted((queue_dir / "pending").glob("*.json"))
        if not pending:
            if watch_s is None:
                return status(queue_dir)
            time.sleep(watch_s)
            continue
        job_path = pending[0]
        running_path = queue_dir / "running" / job_path.name
        os.replace(job_path, running_path)
        job = json.loads(running_path.read_text(encoding="utf-8"))
        output = queue_dir / "runs" / job["name"]
        argv = [*job["argv"], "--output", str(output)]
        job["started_at"] = _now()
        _write(running_path, job)
        log(f"[queue] {job['name']}: running")
        started = time.perf_counter()
        try:
            code = int(runner(argv, queue_dir / "logs" / f"{job['name']}.log"))
            error = None
        except Exception as exc:  # the worker keeps going; the job records why it failed
            code, error = None, f"{type(exc).__name__}: {exc}"
        job.update(finished_at=_now(), exit_status=code, wall_time_s=time.perf_counter() - started,
                   output=str(output))
        summary = output / "summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text(encoding="utf-8"))
            job["trajectory_sha256"] = data.get("trajectory_sha256")
            job["real_time_factor"] = data.get("real_time_factor")
        if error:
            job["error"] = error
        state = "done" if code == 0 else "failed"
        _write(queue_dir / state / running_path.name, job)
        running_path.unlink()
        log(f"[queue] {job['name']}: {state} (exit {code})")
