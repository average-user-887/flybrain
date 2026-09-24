"""The `neurofly` CLI dispatches to the entry points that actually exist."""
import sys
import types

from neurofly import cli


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
