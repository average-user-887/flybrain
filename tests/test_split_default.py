"""S1: a fresh launch runs split by default; --single-process stays an explicit opt-out.

End to end with the real runner on the synthetic test graph (CPU).  The default launch
passes no process-layout flag at all.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import neurofly_daemon as nd  # noqa: E402
from tests.test_split_sim_web import _free_port, _gone, _status_or_none, _wait  # noqa: E402


def _launch(tmp_path, port, *extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    log = tmp_path / f"daemon-{port}-{time.monotonic_ns()}.log"
    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "neurofly_daemon.py"), "--port", str(port),
                             "--backend", "connectome-fixed", "--test-synthetic-graph", "--brain-backend", "cpu",
                             "--paradigm", "t-maze", "--speed", "5", "--output-dir", str(tmp_path / "out"),
                             "--data-dir", str(tmp_path / "data"), "--pid-file", str(tmp_path / "pid"), *extra],
                            cwd=tmp_path, env=env, start_new_session=True, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)
    return proc, log


def _checkpoints(out: Path) -> dict:
    return {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file() and "/checkpoints/" in str(p)}


def test_parser_defaults_to_split_and_keeps_explicit_layouts():
    parser = nd.build_arg_parser()
    assert parser.parse_args([]).process_mode == "split"
    assert parser.parse_args(["--single-process"]).process_mode == "single"
    assert parser.parse_args(["--split"]).process_mode == "split"
    for mode in ("sim", "web", "single", "split"):
        assert parser.parse_args(["--process-mode", mode]).process_mode == mode
    help_text = parser.format_help()
    assert "split (default)" in help_text and "--single-process" in help_text


def test_cli_help_and_wrappers_agree_on_the_split_default():
    from neurofly import cli
    assert "two processes by default" in (cli.__doc__ or "")
    start = (ROOT / "start_daemon.sh").read_text()
    # No layout forced by the wrapper (the daemon default, split, applies) and no
    # forced modular backend: the daemon's own fresh-launch default chooses.
    assert "--process-mode" not in start
    assert 'BACKEND="${NEUROFLY_BACKEND:-}"' in start and "NEUROFLY_BACKEND:-modular" not in start
    assert "neurofly_launcher.pid" in (ROOT / "stop_daemon.sh").read_text()


def test_launch_backend_info_marks_only_the_default_modular_fallback():
    assert nd.launch_backend_info("modular", "r", False)["fallback"] is True
    assert nd.launch_backend_info("modular", "r", True)["fallback"] is False
    assert nd.launch_backend_info("connectome-fixed", "r", False)["fallback"] is False


def test_default_launch_gives_two_healthy_roles_and_a_graceful_stop_leaves_nothing(tmp_path):
    port = _free_port()
    proc, log = _launch(tmp_path, port)
    try:
        st = _wait(lambda: (lambda b: b if b and b["status"] == "online" and b["total_steps"] > 20 else None)(
            _status_or_none(port)), limit=120)
        layout = st["process_layout"]
        assert layout["mode"] == "split"
        sim_pid, web_pid = layout["sim"]["sim_pid"], layout["web_pid"]
        assert len({proc.pid, sim_pid, web_pid}) == 3                 # launcher, simulation, web
        assert st["launch_backend"]["source"] == "requested" and st["launch_backend"]["fallback"] is False
        assert (tmp_path / "pid").read_text().strip() == str(sim_pid)  # the simulation owns the run state
        ipc = re.search(r"IPC (\S+)", log.read_text()).group(1)
        assert os.path.exists(ipc)
        before = _checkpoints(tmp_path / "out")
        stopped_at = time.time_ns()
        proc.send_signal(signal.SIGTERM)                               # stop_daemon.sh's signal
        assert proc.wait(120) == 0
        assert _gone(sim_pid) and _gone(web_pid)                       # no orphan child
        assert not os.path.exists(ipc) and not os.path.exists(os.path.dirname(ipc))   # no socket left
        assert not (tmp_path / "pid").exists()
        after = _checkpoints(tmp_path / "out")
        saved = [p for p, m in after.items() if m >= stopped_at or before.get(p) != m]
        assert saved, "the final save wrote no checkpoint"
        text = log.read_text()
        assert "[Sim] Clean shutdown complete." in text
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(10)


def test_explicit_single_process_still_starts_and_stops(tmp_path):
    port = _free_port()
    proc, log = _launch(tmp_path, port, "--single-process")
    try:
        st = _wait(lambda: (lambda b: b if b and b["status"] == "online" and b["total_steps"] > 5 else None)(
            _status_or_none(port)), limit=120)
        assert "process_layout" not in st                       # one process: no launcher, no IPC
        assert (tmp_path / "pid").read_text().strip() == str(proc.pid)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(120) == 0
        assert "[Launcher]" not in log.read_text()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(10)
