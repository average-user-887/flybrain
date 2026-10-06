"""Audit F failure scenarios against a private daemon (default port 8861).

    python tests/fixtures/silent_freeze/repro.py <scenario> [<scenario> ...]
    python tests/fixtures/silent_freeze/repro.py all

Adapted from docs/receipts/audit-20261005/F-robustness-repro/repro.py (audit F, which
ran the unmodified daemon).  For each scenario it launches the daemon through
faultd.py (fault wrappers only), waits for stepping, records a healthy window, arms
the fault, waits until it has fired, DISARMS it (the disk is freed again, the GPU is
back) -- except for the ``*_still_full`` variants, which keep the disk "full" through
the recovery -- then records what a user sees for 10 s, tries the documented recovery
(re-select the assay), records again, and stops the daemon with SIGTERM.

Each result carries a ``category``: "keeps stepping" (degraded or not), "honest halt",
"SILENT FREEZE" (no steps while status says online), or for shutdown the exit code.
Nothing fills a disk: the wrapper raises OSError(ENOSPC) as a full disk would.
Output: one JSON line per scenario on stdout and in $AUDIT_SCRATCH/neurofly-silent-freeze/.
Ports 8769/8780/8781 (a running observatory) are never touched.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# Checkout root (this file lives in tests/fixtures/silent_freeze/).
CHECKOUT = Path(os.environ.get("NEUROFLY_CHECKOUT", SCRIPTS.parents[2]))
HERE = Path(os.environ.get("AUDIT_SCRATCH", tempfile.gettempdir())) / "neurofly-silent-freeze"
PYTHON = os.environ.get("NEUROFLY_PYTHON", sys.executable)
PORT = int(os.environ.get("NEUROFLY_REPRO_PORT", "8861"))
assert PORT not in (8769, 8780, 8781), "never use the observatory's ports"
BASE = f"http://127.0.0.1:{PORT}"

# name: (control files, specs, extra daemon args, what it models, keep armed during recovery[, brain backend])
# Policy under test (Codex direction 5 Oct 2026, decision 3): scientific mode stops on a
# failed required save; --exploratory continues unsaved; compute errors always halt.
X = ["--exploratory"]
SCENARIOS = {
    "ckpt_enospc": ("ckpt_write", "ENOSPC", [], "disk full during periodic registry checkpoint (owner 15:22)", False),
    "ckpt_enospc_still_full": ("ckpt_write", "ENOSPC", [], "as ckpt_enospc, the disk stays full", True),
    "ckpt_eio": ("ckpt_write", "EIO", [], "I/O error during the periodic checkpoint", False),
    "ckpt_enospc_exploratory": ("ckpt_write", "ENOSPC", X, "disk full, explicit exploratory mode", False),
    "brain_save_enospc": ("brain_save", "ENOSPC", [], "disk full in brains.save_all at checkpoint start", False),
    "telemetry_error": ("telemetry", "RUNTIME", [], "exception assembling a telemetry frame", False),
    "trial_log_enospc": ("brain_log", "ENOSPC", ["--trial-seconds", "1", "--no-continuous"],
                         "disk full writing the events ledger at a trial end", False),
    "run_capture_enospc": ("run_capture", "ENOSPC", ["--record", "freeze"],
                           "disk full writing a --record replay frame (required capture)", False),
    "run_capture_enospc_exploratory": ("run_capture", "ENOSPC", ["--record", "freeze"] + X,
                                       "as run_capture_enospc, exploratory mode", False),
    "recorder_enospc": ("recorder_append", "ENOSPC", [], "disk full in trials/telemetry_summary recorder", False),
    "brain_step_cupy": ("brain_step", "CUPY", [], "genuine CuPy CUDARuntimeError from the connectome step (GPU)",
                        False, "cuda"),
    "device_copy_cupy": ("device_copy", "CUPY", [], "genuine CuPy CUDARuntimeError from the checkpoint device copy (GPU)",
                         False, "cuda"),
    "device_copy_cupy_exploratory": ("device_copy", "CUPY", X, "as device_copy_cupy, exploratory mode: still halts",
                                     False, "cuda"),
    "state_nan": ("state_nan", "NAN", [], "brain state holds NaN at checkpoint time (GPU)", False, "cuda"),
    "arena_step_error": ("arena_step", "RUNTIME", [], "exception in world/body step", False),
    "step_error_disk_full": ("arena_step+brain_log", "RUNTIME+ENOSPC", [],
                             "step error while the disk is full (halt bookkeeping write fails)", False),
    "step_error_disk_full_still_full": ("arena_step+brain_log", "once:RUNTIME+ENOSPC", [],
                                        "as step_error_disk_full; re-select assay while the disk is STILL full",
                                        True),
    "http_status_error": ("http_status", "RUNTIME", [], "exception inside an HTTP handler", False),
    "shutdown_enospc": ("brain_save", "ENOSPC", [], "disk full at final shutdown checkpoint", False),
}


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def probe(run_dir, seconds, pid):
    out = subprocess.run([PYTHON, str(SCRIPTS / "probe.py"), str(PORT), str(seconds),
                          str(run_dir / "faults"), str(pid)], capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"probe_failed": out.stdout[-400:] + out.stderr[-400:]}


def category(window):
    """What a user would be told, versus what happens."""
    if not isinstance(window, dict) or "after" not in window:
        return "unknown"
    after = window["after"]
    st = after.get("status")
    if window.get("steps_advanced", 0) > 0:
        return "keeps stepping (degraded, visible)" if st == "degraded" else "keeps stepping"
    if st == "error":
        if not after.get("halted"):
            return "honest error (not advancing)"
        cls = after.get("failure_class")
        incomplete = " , result incomplete" if after.get("result_validity") == "incomplete" else ""
        if cls == "persistence":
            return "honest stop (required save failed" + incomplete.replace(" ,", ",") + ")"
        if cls == "compute":
            return "honest halt (compute" + incomplete.replace(" ,", ",") + ")"
        return "honest halt" + (f" ({cls})" if cls else "")
    if after.get("paused"):
        return "paused"
    return "SILENT FREEZE"


def run(name):
    control, spec, extra, model, keep_armed, *rest = SCENARIOS[name]
    brain_backend = rest[0] if rest else "cpu"
    run_dir = HERE / "runs" / name
    faults = run_dir / "faults"
    faults.mkdir(parents=True, exist_ok=True)
    for f in faults.iterdir():
        f.unlink()
    continuous = "--no-continuous" not in extra
    extra = [a for a in extra if a != "--no-continuous"]
    args = [PYTHON, "-u", str(SCRIPTS / "faultd.py"), "--host", "127.0.0.1", "--port", str(PORT),
            "--backend", "connectome-fixed", "--paradigm", "t-maze", "--speed", "3",
            "--checkpoint-interval", "5",
            "--output-dir", str(run_dir / "out"), "--data-dir", str(run_dir / "out" / "learning"),
            "--summary-interval", "5", "--pid-file", str(run_dir / "daemon.pid")]
    if continuous:
        args += ["--continuous", "--trial-seconds", "120"]
    args += extra
    env = dict(os.environ, FAULT_DIR=str(faults), NEUROFLY_BRAIN_BACKEND=brain_backend, PYTHONPATH=".",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    log = open(run_dir / "daemon.log", "w")
    proc = subprocess.Popen(args, cwd=CHECKOUT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {"scenario": name, "models": model, "fault": f"{control}={spec}", "brain_backend": brain_backend,
              "mode": "exploratory" if "--exploratory" in extra else "scientific", "recovery_disk": (
        "still full" if keep_armed else "freed")}
    try:
        deadline = time.time() + 240
        while time.time() < deadline:
            try:
                if get("/api/status")["total_steps"] > 50:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        result["healthy"] = probe(run_dir, 5, proc.pid)
        for c, s in zip(control.split("+"), spec.split("+")):
            (faults / c).write_text(s)
        if name == "shutdown_enospc":
            # Armed just before SIGTERM so the final_shutdown checkpoint is what hits it.
            proc.send_signal(signal.SIGTERM)
            try:
                result["exit_code"] = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                result["exit_code"] = "did not exit in 30 s"
            return result
        if name == "http_status_error":
            time.sleep(0.5)
            try:
                get("/api/status")
                result["status_endpoint"] = "answered"
            except Exception as exc:  # noqa: BLE001
                result["status_endpoint"] = f"{type(exc).__name__}: {exc}"
        fired = faults / "fired.log"
        deadline = time.time() + 60
        while time.time() < deadline and not (fired.exists() and fired.read_text().strip()):
            time.sleep(0.2)
        time.sleep(0.5)
        if not keep_armed:
            for c in control.split("+"):
                (faults / c).unlink(missing_ok=True)                 # the disk is free again
        result["fired"] = fired.read_text().strip().splitlines()[:3] if fired.exists() else None
        result["after_fault"] = probe(run_dir, 10, proc.pid)
        result["category_after_fault"] = category(result["after_fault"])
        try:
            ack = post("/api/command", {"action": "switch_paradigm", "params": {"paradigm": "t-maze"}})
            result["recovery_command"] = {k: ack.get(k) for k in ("status", "message", "applied",
                                                                  "loop_restarted")} | {
                "halted_by_error": (ack.get("ack") or {}).get("halted_by_error")}
        except Exception as exc:  # noqa: BLE001
            result["recovery_command"] = f"{type(exc).__name__}: {exc}"
        result["after_recovery"] = probe(run_dir, 8, proc.pid)
        result["category_after_recovery"] = category(result["after_recovery"])
    finally:
        for f in faults.iterdir():
            if f.name not in ("fired.log", "threads.json"):
                f.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                result.setdefault("exit_code", proc.wait(timeout=30))
            except subprocess.TimeoutExpired:
                proc.kill()
                result.setdefault("exit_code", "killed after 30 s")
        log.close()
        lines = (run_dir / "daemon.log").read_text().splitlines()
        result["log_excerpt"] = [l for l in lines if any(k in l for k in (
            "HALTED", "NOT SAVING", "STOPPED", "Watchdog", "Final checkpoint", "Recording", "Restarting",
            "Halt lifted", "Shutdown complete", "Exception in thread", "Traceback"))][:14]
        (HERE / "results").mkdir(parents=True, exist_ok=True)
        (HERE / "results" / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    names = list(SCENARIOS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for n in names:
        print(json.dumps(run(n)), flush=True)
