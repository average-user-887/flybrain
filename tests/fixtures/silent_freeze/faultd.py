"""Fault-injection launcher for the NeuroFly daemon (audit F, robustness).

Copied from docs/receipts/audit-20261005/F-robustness-repro/faultd.py (unchanged
apart from this note); used by repro.py next to it and by the browser check
scripts/silent_freeze_browser_check.py.

Runs the daemon's own ``neurofly_daemon.run_daemon()`` in-process after wrapping a set
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
    if kind == "THREADEXIT":
        # Not an Exception: ends the calling thread without a traceback (the silent
        # thread death of audit F, forced deliberately to exercise the watchdog).
        return SystemExit("injected simulation thread death")
    if kind.startswith("HANG"):
        # A step that does not return for N seconds (kernel hang, stalled I/O).
        time.sleep(float(kind[4:] or 30))
        return None
    if kind in ("ENOSPC", "EIO", "EROFS"):
        code = getattr(errno, kind)
        return OSError(code, os.strerror(code))
    if kind == "CUPY":
        # The genuine CuPy runtime error type a failed kernel or device copy raises
        # (cudaErrorLaunchFailure), not a RuntimeError with a CUDA-like message.
        from cupy_backends.cuda.api.runtime import CUDARuntimeError
        return CUDARuntimeError(719)
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
            exc = _exception(kind)
            if exc is not None:
                raise exc
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
# Engine layer: the checkpoint's device-to-host copy of the brain state.
wrap(experiment_registry.GraphInstance, "state_arrays", "device_copy")


def _nan_state_arrays(original):
    """'state_nan' control file: the brain state read for a checkpoint holds a NaN."""
    def wrapper(self):
        arrays = original(self)
        if _armed("state_nan"):
            import numpy as np
            v = np.array(arrays["brain.v"], dtype=float, copy=True)
            v.flat[0] = np.nan
            arrays = dict(arrays, **{"brain.v": v})
        return arrays
    return wrapper


experiment_registry.GraphInstance.state_arrays = _nan_state_arrays(experiment_registry.GraphInstance.state_arrays)
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
