#!/usr/bin/env python3
"""Exclusive evidence launcher; timeout is inside the child, never outside the receipt writer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pathA_escape_common as common
from pathA_escape_run import parser, write_json


def main():
    a = parser().parse_args()
    common.require_environment()
    common.load_plan(a.contract, a.contract_sha256, a.prereg, a.prereg_sha256, a.expected_code_sha)
    out = Path(a.out).resolve()
    out.mkdir()  # Existing evidence is never resumed, overwritten or appended.
    runner = str(common.ROOT / 'scripts/pathA_escape_run.py')
    run_args = [runner, '--contract', str(Path(a.contract).resolve()), '--contract-sha256', a.contract_sha256,
                '--prereg', str(Path(a.prereg).resolve()), '--prereg-sha256', a.prereg_sha256,
                '--expected-code-sha', a.expected_code_sha, '--out', str(out)]
    argv = ['timeout', '--signal=TERM', '3500', sys.executable, '-B', *run_args]
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    write_json(out / 'launch.json', dict(argv=argv, runner_argv=run_args, source_sha=a.expected_code_sha,
                                        wrapper_pid=os.getpid(), start_utc=started,
                                        thread_environment={k: os.environ[k] for k in common.THREAD_ENV}))
    with (out / 'run.log').open('xb') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(child_pid=child.pid, wrapper_pid=os.getpid(), argv=argv)), flush=True)
        rc = child.wait()
    write_json(out / 'exit.json', dict(argv=argv, child_pid=child.pid, wrapper_pid=os.getpid(),
                                      start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(),
                                      wall_s=time.monotonic() - clock, exit_code=rc,
                                      log='run.log', log_sha256=common.sha(out / 'run.log')))
    return rc


if __name__ == '__main__':
    sys.exit(main())
