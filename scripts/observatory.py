#!/usr/bin/env python3
"""Start, inspect or stop the local NeuroFly lab as persistent user services.

Run from any directory. Uses only the standard library; the daemon interpreter
is the project's .venv. The services survive this terminal and restart on failure.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parents[1]
MARKER = '# Managed by NeuroFly scripts/observatory.py\n'
BRAIN = 'neurofly-observatory-brain.service'
WEB = 'neurofly-observatory-web.service'
RESEARCH = 'neurofly-research.service'
UNITS = (BRAIN, WEB, RESEARCH)
WEB_URL = 'http://127.0.0.1:8780/index.html'
API_URL = 'http://127.0.0.1:8781'


def unit_quote(value):
    """Quote one ExecStart argument, preserving spaces and specifier literals."""
    value = str(value)
    if '\n' in value or '\r' in value or '\0' in value:
        raise ValueError('A service path cannot contain control characters')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def service_text(project: Path, kind: str):
    py = project / '.venv' / 'bin' / 'python'
    output = project / 'outputs' / 'observatory-live'
    if kind == 'brain':
        title = 'NeuroFly learning daemon (local observatory)'
        args = [py, '-u', project / 'neurofly_daemon.py', '--host', '127.0.0.1',
                '--port', '8781', '--paradigm', 't-maze', '--speed', '3',
                '--continuous', '--trial-seconds', '120', '--checkpoint-interval', '30',
                '--output-dir', output, '--data-dir', output / 'learning',
                '--pid-file', output / 'daemon.pid']
        dependencies = ''
    elif kind == 'research':
        title = 'NeuroFly independent research cohorts and scientific data'
        args = [py, '-u', project / 'scripts' / 'research_worker.py']
        dependencies = ''
    elif kind == 'web':
        title = 'NeuroFly learning observatory (local web UI)'
        args = [py, '-u', '-m', 'http.server', '8780', '--bind', '127.0.0.1',
                '--directory', project / 'web']
        # The page remains available if the brain fails, so it can explain recovery.
        dependencies = f'Wants={BRAIN}\nAfter={BRAIN}\n'
    else:
        raise ValueError(f'Unknown service: {kind}')
    return MARKER + f'''[Unit]
Description={title}
{dependencies}StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={str(project).replace('%', '%%')}
# ':' disables environment-variable expansion in ExecStart arguments.
ExecStart=:{' '.join(unit_quote(a) for a in args)}
Restart=always
RestartSec=2
TimeoutStopSec=20
UMask=0077
Environment=PYTHONUNBUFFERED=1
Environment=OPENBLAS_NUM_THREADS=1
Nice=10

[Install]
WantedBy=default.target
'''


def install_units(project: Path, unit_dir: Path):
    if not (project / '.venv' / 'bin' / 'python').is_file():
        raise RuntimeError(f'Missing Python environment: {project / ".venv"}')
    # Check every target before writing any of them.
    expected = {BRAIN: service_text(project, 'brain'), WEB: service_text(project, 'web'),
                RESEARCH: service_text(project, 'research')}
    for name in expected:
        path = unit_dir / name
        if path.exists() and not path.read_text().startswith(MARKER):
            raise RuntimeError(f'Refusing to overwrite an unmanaged service: {path}')
    unit_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    for name, content in expected.items():
        path = unit_dir / name
        if path.exists() and path.read_text() == content:
            continue
        tmp = path.with_suffix('.service.tmp')
        tmp.write_text(content)
        os.replace(tmp, path)
        changed.append(name)
    return changed


def ctl(*args, check=True):
    return subprocess.run(['systemctl', '--user', *args], text=True,
                          capture_output=True, check=check)


def check_ports():
    """Never replace an unrelated process just because it owns a familiar port."""
    for port, unit in [(8780, WEB), (8781, BRAIN)]:
        if ctl('is-active', '--quiet', unit, check=False).returncode == 0:
            continue
        with socket.socket() as sock:
            sock.settimeout(0.3)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is occupied outside {unit}; no process was stopped.')


def health():
    with urlopen(WEB_URL, timeout=2) as response:
        if response.status != 200 or b'Modular Sensorimotor Instrument' not in response.read(4096):
            raise RuntimeError('The web port is not serving the NeuroFly observatory')
    with urlopen(API_URL + '/api/status', timeout=2) as response:
        status = json.load(response)
    with urlopen(API_URL + '/api/brain', timeout=2) as response:
        brain = json.load(response)
    if status.get('status') != 'online' or not brain.get('brain_id'):
        raise RuntimeError('The learning daemon is not ready')
    return status, brain


def wait_ready(timeout=15):
    deadline = time.monotonic() + timeout
    error = 'No response'
    while time.monotonic() < deadline:
        try:
            return health()
        except (OSError, URLError, ValueError, RuntimeError) as exc:
            error = str(exc)
            time.sleep(0.2)
    raise RuntimeError(f'Lab did not become healthy: {error}\nRun: python3 {Path(__file__)} logs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'restart', 'stop', 'disable', 'logs'])
    args = parser.parse_args(argv)
    try:
        # A functioning user manager is required; a degraded manager can still run this lab.
        ctl('show-environment')
        if args.action in ('start', 'restart'):
            check_ports()
            unit_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'systemd' / 'user'
            changed = install_units(PROJECT, unit_dir)
            ctl('daemon-reload')
            ctl('enable', *UNITS)
            if args.action == 'restart':
                ctl('restart', *UNITS)
            else:
                if changed:
                    ctl('restart', *changed)
                ctl('start', *UNITS)
            status, brain = wait_ready()
            print(f'NeuroFly is ready: {WEB_URL}')
            print(f'Brain: {brain["paradigm"]} / {brain["brain_id"]} ({"restored" if brain["restored"] else "new"})')
            print('Managed by systemd --user; survives terminal closure and starts at user login.')
        elif args.action == 'status':
            print(ctl('show', *UNITS, '--property=Id,ActiveState,SubState,MainPID,NRestarts', check=False).stdout.strip())
            status, brain = health()
            print(f'Healthy: {WEB_URL}\nBrain: {brain["paradigm"]} / {brain["brain_id"]} | steps: {status["total_steps"]}')
        elif args.action == 'logs':
            return subprocess.run(['journalctl', '--user', '-u', BRAIN, '-u', WEB, '-u', RESEARCH,
                                   '-n', '50', '--no-pager']).returncode
        elif args.action == 'disable':
            ctl('disable', '--now', *UNITS)
            print('Lab stopped; automatic startup disabled. Saved brain data is retained.')
        else:
            ctl('stop', *UNITS)
            print('Lab stopped. Saved brain data is retained.')
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f'NeuroFly: {detail}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
