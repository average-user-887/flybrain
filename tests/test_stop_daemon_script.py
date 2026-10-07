"""stop_daemon.sh exit status: nonzero after any forced kill, with cleanup still done.

Only disposable child processes started by this test are signalled; the script runs
from a copy inside tmp_path, so no real daemon or PID file is touched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHILD = ("import signal, sys, time\n"
         "if sys.argv[1] == 'stubborn': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
         "print('ready', flush=True)\n"
         "time.sleep(600)\n")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not Path("/proc").is_dir(),
                                reason="needs bash and /proc")


def _child(kind):
    # "neurofly" in the command line satisfies the script's recycled-PID guard.
    proc = subprocess.Popen([sys.executable, "-c", CHILD, kind, "neurofly-disposable-test-child"],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    # Reap at once, as init would for a real daemon: a zombie still answers kill -0.
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def _setup(tmp_path):
    shutil.copy(ROOT / "stop_daemon.sh", tmp_path / "stop_daemon.sh")
    (tmp_path / "outputs").mkdir()
    return tmp_path / "outputs"


def _stop(tmp_path):
    env = dict(os.environ, NEUROFLY_STOP_TIMEOUT="1")
    return subprocess.run(["bash", str(tmp_path / "stop_daemon.sh")], env=env, capture_output=True,
                          text=True, timeout=60)


@pytest.mark.parametrize("pid_name", ["neurofly_launcher.pid", "neurofly_daemon.pid"])
def test_forced_kill_exits_nonzero_and_still_cleans_up(tmp_path, pid_name):
    out = _setup(tmp_path)
    proc = _child("stubborn")
    try:
        (out / pid_name).write_text(str(proc.pid))
        ipc_dir = tmp_path / "neurofly-split-test"
        ipc_dir.mkdir()
        (ipc_dir / "sim.sock").write_text("")
        (out / "neurofly_daemon.log").write_text(f"[Launcher] simulation pid 1, web pid 2, IPC {ipc_dir}/sim.sock\n")
        result = _stop(tmp_path)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Force killing" in result.stdout and "cleanly" not in result.stdout
        assert proc.wait(10) is not None                         # the process is gone
        assert not (out / pid_name).exists()
        if pid_name == "neurofly_launcher.pid":
            assert not ipc_dir.exists()                           # socket of a killed launcher removed
    finally:
        if proc.poll() is None:
            proc.kill()


def test_sigterm_exit_returns_zero(tmp_path):
    out = _setup(tmp_path)
    proc = _child("polite")
    try:
        (out / "neurofly_launcher.pid").write_text(str(proc.pid))
        result = _stop(tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Force killing" not in result.stdout
        assert not (out / "neurofly_launcher.pid").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
