"""Startup with a damaged run directory: what does the daemon do?

python tests/fixtures/silent_freeze/startup.py <source out dir>
Copies the source output dir per case, damages it, starts the daemon on a private
port (default 8861) and reports whether it serves (and whether it resumed or reset)
or exits, and how.  Adapted from audit F's F-robustness-repro/startup.py.
"""
import json
import os
import shutil
import subprocess
import tempfile
import sys
import time
import urllib.request
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# Checkout root (this file lives in tests/fixtures/silent_freeze/).
CHECKOUT = Path(os.environ.get("NEUROFLY_CHECKOUT", SCRIPTS.parents[2]))
# Run directories, logs and results go here (never into the checkout).
HERE = Path(os.environ.get("AUDIT_SCRATCH", tempfile.gettempdir())) / "neurofly-silent-freeze"
PORT = os.environ.get("NEUROFLY_REPRO_PORT", "8861")
assert PORT not in ("8769", "8780", "8781"), "never use the observatory's ports"
PYTHON = os.environ.get("NEUROFLY_PYTHON", sys.executable)
SRC = Path(sys.argv[1])


def instance_dir(out):
    return next((out / "registry-v3" / "t-maze" / "connectome-fixed").iterdir())


def ckpt_sha_mismatch(out):
    cur = json.loads((instance_dir(out) / "CURRENT.json").read_text())
    with open(instance_dir(out) / "checkpoints" / cur["file"], "ab") as fh:
        fh.write(b"junk")


def ckpt_missing(out):
    cur = json.loads((instance_dir(out) / "CURRENT.json").read_text())
    (instance_dir(out) / "checkpoints" / cur["file"]).unlink()


def current_empty(out):
    (instance_dir(out) / "CURRENT.json").write_text("")


def registry_truncated(out):
    p = out / "registry-v3" / "registry.json"
    p.write_text(p.read_text()[:40])


def brain_json_corrupt(out):
    p = out / "graph-bookkeeping" / "connectome-fixed" / "t-maze.json"
    p.write_text(p.read_text()[:100])


CASES = [ckpt_sha_mismatch, ckpt_missing, current_empty, registry_truncated, brain_json_corrupt]

for case in CASES:
    out = HERE / "startup" / case.__name__
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(SRC, out, symlinks=True)
    case(out)
    env = dict(os.environ, NEUROFLY_BRAIN_BACKEND="cpu", PYTHONPATH=".", PYTHONUNBUFFERED="1")
    log = out.parent / f"{case.__name__}.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen([PYTHON, "-u", "neurofly_daemon.py", "--port", PORT, "--paradigm", "t-maze",
                                 "--continuous", "--output-dir", str(out), "--data-dir", str(out / "learning"),
                                 "--pid-file", str(out / "daemon.pid")], cwd=CHECKOUT, env=env,
                                stdout=fh, stderr=subprocess.STDOUT)
        result, deadline = {"case": case.__name__}, time.time() + 150
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=3) as r:
                    st = json.loads(r.read())
                if st["total_steps"] > 0:
                    result.update(served=True, status=st["status"], steps=st["total_steps"],
                                  instance=(st.get("identity") or {}).get("instance_id"))
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        else:
            result.update(served=False, exit_code=proc.returncode)
    text = log.read_text()
    lines = [l for l in text.splitlines() if "Error" in l or "Cannot" in l or "rror:" in l or "[Registry]" in l]
    result["log"] = lines[-3:]
    result["traceback_printed"] = "Traceback" in text
    print(json.dumps(result), flush=True)
