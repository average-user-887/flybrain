#!/usr/bin/env python3
"""Run one Path A shard command and capture its full diagnostics.

    pathA_shard.py --exit-file OUT.exit.json --log OUT.log -- <command ...>

stdout+stderr go to the log; the exit file records argv, child pid, UTC start/end,
wall seconds, the exit code (negative = killed by that signal) and the log sha256.
The exit file is written with exclusive create, so it is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--exit-file', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    if not cmd:
        raise SystemExit('no command')
    ef = Path(a.exit_file)
    if ef.exists():
        raise SystemExit(f'{ef} exists: refusing to overwrite a captured exit record')
    start, t0 = utc(), time.monotonic()
    with open(a.log, 'ab') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(pid=proc.pid, start_utc=start, cmd=cmd)), flush=True)
        rc = proc.wait()
    rec = dict(argv=cmd, pid=proc.pid, wrapper_pid=os.getpid(), start_utc=start, end_utc=utc(),
               wall_s=round(time.monotonic() - t0, 3), exit_code=rc,
               log=str(a.log), log_sha256=hashlib.sha256(Path(a.log).read_bytes()).hexdigest())
    with open(ef, 'x') as fh:
        fh.write(json.dumps(rec, indent=1) + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main())
