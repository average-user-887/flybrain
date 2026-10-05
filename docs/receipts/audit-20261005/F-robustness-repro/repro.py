"""Audit F reproduction driver: one failure scenario against a private daemon on :8851.

    python repro.py <scenario> [<scenario> ...]
    python repro.py all

For each scenario it launches the unmodified daemon through faultd.py (fault wrappers
only), waits for stepping, records a healthy window, arms the fault, waits until it has
fired, DISARMS it (the disk is freed again, the GPU is back), then records what a user
sees for 10 s, tries the documented recovery (re-select the assay), records again, and
stops the daemon with SIGTERM.  Output: one JSON object per scenario on stdout and in
$AUDIT_SCRATCH/neurofly-auditF/results/<scenario>.json.  The owner's daemons (8769/8780/8781) are never touched.
"""
import json
import os
import signal
import subprocess
import tempfile
import sys
import time
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# Checkout root (this file lives in docs/receipts/audit-20261005/F-robustness-repro/).
CHECKOUT = Path(os.environ.get("NEUROFLY_CHECKOUT", SCRIPTS.parents[3]))
# Run directories, logs and results go here (never into the checkout).
HERE = Path(os.environ.get("AUDIT_SCRATCH", tempfile.gettempdir())) / "neurofly-auditF"
PYTHON = os.environ.get("NEUROFLY_PYTHON", sys.executable)
PORT = 8851
BASE = f"http://127.0.0.1:{PORT}"

# name: (control file, spec, extra daemon args, what it models)
SCENARIOS = {
    "ckpt_enospc": ("ckpt_write", "ENOSPC", [], "disk full during periodic registry checkpoint (owner 15:22)"),
    "ckpt_cuda": ("ckpt_write", "CUDA", [], "GPU error while a checkpoint copies device state"),
    "brain_save_enospc": ("brain_save", "ENOSPC", [], "disk full in brains.save_all at checkpoint start"),
    "telemetry_error": ("telemetry", "RUNTIME", [], "exception assembling a telemetry frame"),
    "trial_log_enospc": ("brain_log", "ENOSPC", ["--trial-seconds", "1", "--no-continuous"],
                         "disk full writing the events ledger at a trial end"),
    "run_capture_enospc": ("run_capture", "ENOSPC", ["--record", "auditf"],
                           "disk full writing a --record replay frame"),
    "recorder_enospc": ("recorder_append", "ENOSPC", [], "disk full in trials/telemetry_summary recorder"),
    "brain_step_cuda": ("brain_step", "CUDA", [], "CUDA error inside the connectome step"),
    "arena_step_error": ("arena_step", "RUNTIME", [], "exception in world/body step"),
    "step_error_disk_full": ("arena_step+brain_log", "RUNTIME+ENOSPC", [],
                             "step error while the disk is full (halt bookkeeping write fails)"),
    "http_status_error": ("http_status", "RUNTIME", [], "exception inside an HTTP handler"),
    "shutdown_enospc": ("brain_save", "ENOSPC", [], "disk full at final shutdown checkpoint"),
}


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, body, timeout=30):
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


def run(name):
    control, spec, extra, model = SCENARIOS[name]
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
    env = dict(os.environ, FAULT_DIR=str(faults), NEUROFLY_BRAIN_BACKEND="cpu", PYTHONPATH=".",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    log = open(run_dir / "daemon.log", "w")
    proc = subprocess.Popen(args, cwd=CHECKOUT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {"scenario": name, "models": model, "fault": f"{control}={spec}"}
    try:
        deadline = time.time() + 180
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
        for c in control.split("+"):
            (faults / c).unlink(missing_ok=True)                 # the disk is free again
        result["fired"] = fired.read_text().strip().splitlines()[:3] if fired.exists() else None
        result["after_fault"] = probe(run_dir, 10, proc.pid)
        try:
            ack = post("/api/command", {"action": "switch_paradigm", "params": {"paradigm": "t-maze"}})
            result["recovery_command"] = {k: ack.get(k) for k in ("status", "message", "applied")} | {
                "halted_by_error": (ack.get("ack") or {}).get("halted_by_error")}
        except Exception as exc:  # noqa: BLE001
            result["recovery_command"] = f"{type(exc).__name__}: {exc}"
        result["after_recovery"] = probe(run_dir, 8, proc.pid)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                result.setdefault("exit_code", proc.wait(timeout=30))
            except subprocess.TimeoutExpired:
                proc.kill()
                result.setdefault("exit_code", "killed after 30 s")
        log.close()
        tb = [l for l in (run_dir / "daemon.log").read_text().splitlines()
              if "Error" in l or "Exception" in l or "Traceback" in l or "File \"/" in l]
        result["log_excerpt"] = tb[:14]
        (HERE / "results").mkdir(exist_ok=True)
        (HERE / "results" / f"{name}.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    names = list(SCENARIOS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for n in names:
        print(json.dumps(run(n)), flush=True)
