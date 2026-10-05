"""Fault-injection launcher for the NeuroFly daemon (audit F, robustness).

Runs the unmodified ``neurofly_daemon.run_daemon()`` in-process after wrapping a set
of functions.  A wrapped function raises when a control file exists:

    $FAULT_DIR/<name>        contents: ENOSPC | CUDA | RUNTIME   (optionally "once:<kind>")

"once:" removes the control file after the first raise (a transient fault); without it
the fault persists until the file is deleted (a disk that stays full).  Nothing is
written to fill a disk: the wrapper raises OSError(ENOSPC) exactly as a full disk would.

A side thread writes the live thread list to $FAULT_DIR/threads.json every second so the
probe can see whether NeuroFly-SimLoop is still alive.

Usage (from the checkout root, PYTHONPATH=.):
    FAULT_DIR=/path/faults python faultd.py <normal neurofly_daemon.py arguments>
"""
import errno
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
FAULT_DIR = Path(os.environ["FAULT_DIR"])
FAULT_DIR.mkdir(parents=True, exist_ok=True)

import experiment_brains  # noqa: E402
import experiment_registry  # noqa: E402
import learning_recorder  # noqa: E402
import neurofly_daemon  # noqa: E402
import provenance  # noqa: E402
from arena import Arena  # noqa: E402
from neurofly import recording  # noqa: E402

FIRED = FAULT_DIR / "fired.log"


def _exception(kind: str) -> BaseException:
    if kind == "ENOSPC":
        return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
    if kind == "CUDA":
        return RuntimeError("cudaErrorLaunchFailure: unspecified launch failure (injected)")
    return RuntimeError(f"injected fault ({kind})")


def _armed(name: str):
    control = FAULT_DIR / name
    try:
        spec = control.read_text().strip() or "ENOSPC"
    except FileNotFoundError:
        return None
    if spec.startswith("once:"):
        spec = spec[5:]
        try:
            control.unlink()
        except FileNotFoundError:
            pass
    with FIRED.open("a") as fh:
        fh.write(f"{time.time():.3f} {name} {spec} thread={threading.current_thread().name}\n")
    return spec


def wrap(owner, attr: str, name: str):
    original = getattr(owner, attr)

    def wrapper(*args, **kwargs):
        kind = _armed(name)
        if kind:
            raise _exception(kind)
        return original(*args, **kwargs)

    wrapper.__wrapped__ = original
    setattr(owner, attr, wrapper)


# Checkpoint write used by ExperimentRegistry.checkpoint (the owner's 15:22 path).
wrap(experiment_registry, "atomic_write_bytes", "ckpt_write")
wrap(experiment_brains.ExperimentBrain, "save", "brain_save")          # save_all in save_checkpoint
wrap(experiment_brains.ExperimentBrain, "log", "brain_log")            # events.jsonl ledger
wrap(learning_recorder.JsonlWriter, "append", "recorder_append")       # trials / telemetry_summary
wrap(recording.RunRecorder, "capture", "run_capture")                  # --record replay file
wrap(neurofly_daemon.ContinuousExperimentRunner, "_assemble_telemetry", "telemetry")
wrap(neurofly_daemon.GraphArenaController, "__call__", "brain_step")   # connectome step (in arena.step)
wrap(Arena, "step", "arena_step")                                      # world + body step
wrap(neurofly_daemon.NeuroflyHTTPHandler, "_status_payload", "http_status")
wrap(neurofly_daemon.ContinuousExperimentRunner, "_apply_command", "apply_command")


def _thread_dump():
    while True:
        names = sorted(t.name for t in threading.enumerate())
        tmp = FAULT_DIR / "threads.json.tmp"
        tmp.write_text(json.dumps({"t": time.time(), "threads": names}))
        os.replace(tmp, FAULT_DIR / "threads.json")
        time.sleep(1.0)


threading.Thread(target=_thread_dump, daemon=True, name="audit-thread-dump").start()

if __name__ == "__main__":
    sys.argv = ["neurofly_daemon.py"] + sys.argv[1:]
    neurofly_daemon.run_daemon()
