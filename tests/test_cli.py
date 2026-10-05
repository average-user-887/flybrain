"""The `neurofly` CLI dispatches to the entry points that actually exist."""
import re
import sys
import types

import pytest

from neurofly import cli


_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def _plain(text):
    """Help text without ANSI colour: Python 3.14 argparse colours it under FORCE_COLOR."""
    return _ANSI.sub('', text)


def _fake_daemon(monkeypatch, seen):
    fake = types.ModuleType('neurofly_daemon')
    fake.run_daemon = lambda: seen.append(list(sys.argv[1:]))
    monkeypatch.setitem(sys.modules, 'neurofly_daemon', fake)


def test_run_calls_the_daemon_entry_point(monkeypatch):
    seen = []
    _fake_daemon(monkeypatch, seen)
    assert cli.main(['run', '--port', '8769']) == 0
    assert seen == [['--port', '8769']]


def test_bare_daemon_flags_forward_to_run(monkeypatch):
    seen = []
    _fake_daemon(monkeypatch, seen)
    assert cli.main(['--port', '8769']) == 0
    assert seen == [['--port', '8769']]


def test_unknown_command_is_an_error(monkeypatch, capsys):
    seen = []
    _fake_daemon(monkeypatch, seen)
    assert cli.main(['rnu']) == 2
    assert seen == []
    assert 'unknown command' in capsys.readouterr().err


def test_real_daemon_exposes_run_daemon():
    import ast
    from pathlib import Path
    tree = ast.parse((Path(cli.__file__).resolve().parents[1] / 'neurofly_daemon.py').read_text(encoding='utf-8'))
    assert 'run_daemon' in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


# --- help never has side effects (Deck clean-room report: `download-data --help`
# started the 1.1 GB download) ------------------------------------------------

def _forbid_download(monkeypatch, tmp_path):
    from brainlab import download
    monkeypatch.setattr(download, 'ROOT', tmp_path)

    def refuse(*a, **k):
        raise AssertionError('a download was started')
    monkeypatch.setattr(download.urllib.request, 'urlretrieve', refuse)
    return download


@pytest.mark.parametrize('flag', ['--help', '-h'])
def test_download_data_help_prints_usage_and_downloads_nothing(monkeypatch, tmp_path, capsys, flag):
    _forbid_download(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['download-data', flag])
    assert exit_info.value.code == 0
    out = _plain(capsys.readouterr().out)
    assert 'usage: neurofly download-data' in out and 'Downloading' not in out
    assert not list(tmp_path.iterdir()), 'help must not create connectome_data/'


def test_download_data_rejects_unknown_flags_without_downloading(monkeypatch, tmp_path):
    _forbid_download(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(['download-data', '--bogus'])
    assert exit_info.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('command', ['status', 'capability'])
def test_read_only_commands_answer_help_without_running(monkeypatch, capsys, command):
    import brainlab.graph_identity as gi

    def refuse(*a, **k):
        raise AssertionError('status ran instead of printing help')
    monkeypatch.setattr(gi, 'verify_graph', refuse)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, '--help'])
    assert exit_info.value.code == 0
    out = _plain(capsys.readouterr().out)
    assert f'usage: neurofly {command}' in out and 'System Status' not in out


@pytest.mark.parametrize('command', ['run', 'full-sim', 'embodied', 'download-data', 'status',
                                     'capability', 'validate', 'record'])
def test_every_subcommand_help_exits_cleanly_and_writes_nothing(tmp_path, command):
    import os
    import subprocess
    from pathlib import Path
    root = Path(cli.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='')
    proc = subprocess.run([sys.executable, '-m', 'neurofly.cli', command, '--help'],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert 'usage:' in _plain(proc.stdout)
    assert not list(tmp_path.iterdir()), f'{command} --help wrote files'


def test_download_data_help_end_to_end(tmp_path):
    """A fresh process from a neutral directory: help exits 0 and fetches nothing."""
    import os
    import subprocess
    from pathlib import Path
    root = Path(cli.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    proc = subprocess.run([sys.executable, '-m', 'neurofly.cli', 'download-data', '--help'],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = _plain(proc.stdout)
    assert 'usage: neurofly download-data' in out and 'Downloading' not in out
