"""Real-browser recovery check (work packages 1 and 2), opt-in.

Runs ``scripts/browser_firefox_check.py`` -- headless Firefox through geckodriver
against an isolated daemon and web server it starts itself -- and requires every
scenario to pass: live load, 100x observation without stale/disconnect, pause,
the ``?inject=malformed|apply|render`` fault injections with their recovery, and a
disconnect/restart that keeps the last measured frame.

Selenium is kept out of the project environment, so this is skipped unless
``NEUROFLY_BROWSER_PYTHON`` names an interpreter that has it, e.g.::

    NEUROFLY_BROWSER_PYTHON=/path/to/bvenv/bin/python pytest tests/test_browser_recovery.py

A headless pass is not the final live-UI sign-off required by AGENTS.md.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BROWSER_PY = os.environ.get("NEUROFLY_BROWSER_PYTHON")


@pytest.mark.skipif(not BROWSER_PY, reason="set NEUROFLY_BROWSER_PYTHON to an interpreter with selenium")
def test_firefox_recovers_from_injected_faults_and_disconnect(tmp_path):
    receipt = tmp_path / "receipt.json"
    subprocess.run([BROWSER_PY, str(ROOT / "scripts" / "browser_firefox_check.py"), "--work-dir", str(tmp_path),
                    "--seconds", os.environ.get("NEUROFLY_BROWSER_SECONDS", "15"), "--receipt", str(receipt)],
                   check=True, timeout=600)
    data = json.loads(receipt.read_text())
    failed = [name for name, scenario in data["scenarios"].items() if not scenario.get("pass")]
    assert not failed, failed
