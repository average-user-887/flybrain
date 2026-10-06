"""Persistent selection is opt-in and never migrates or initializes saved brains."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
from neurofly import cli, storage
from tests.test_wheel_packaging import installed_wheel


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv('NEUROFLY_CONFIG_HOME', str(tmp_path / 'config'))
    monkeypatch.delenv('NEUROFLY_DATA_DIR', raising=False)
    output, data = tmp_path / 'old-checkout' / 'outputs', tmp_path / 'records'
    output.mkdir(parents=True)
    data.mkdir()
    (output / 'saved-brain').write_bytes(b'old learned values')
    (data / 'history.jsonl').write_bytes(b'{"old":true}\n')
    return output, data


def daemon(monkeypatch):
    seen = []
    fake = types.ModuleType('neurofly_daemon')
    def parser():
        p = argparse.ArgumentParser()
        p.add_argument('--output-dir')
        p.add_argument('--data-dir')
        p.add_argument('--port')
        return p
    fake.build_arg_parser = parser
    fake.run_daemon = lambda: seen.append(sys.argv[1:])
    monkeypatch.setitem(sys.modules, 'neurofly_daemon', fake)
    return seen


def test_adoption_and_show_preserve_existing_files(stores, capsys):
    output, data = stores
    before = {p: p.read_bytes() for root in stores for p in root.rglob('*') if p.is_file()}
    assert cli.main(['storage', 'use', '--output-dir', str(output), '--data-dir', str(data)]) == 0
    capsys.readouterr()
    assert cli.main(['storage', 'show']) == 0
    assert json.loads(capsys.readouterr().out)['configured']
    assert before == {p: p.read_bytes() for p in before}
    assert storage.show(output.parent)['paths']['output_dir']['inside_current_checkout']


@pytest.mark.parametrize('args,expected', [
    ([], ('config', 'config')),
    (['--output-dir', '/override'], ('/override', 'config')),
    (['--output-d=/first', '--output-dir=/last'], ('/last', 'config')),
    (['--data-d=/override'], ('config', '/override')),
])
def test_independent_explicit_precedence(stores, monkeypatch, args, expected):
    storage.adopt(*map(str, stores))
    seen = daemon(monkeypatch)
    assert cli.main(['run', *args]) == 0
    parsed = sys.modules['neurofly_daemon'].build_arg_parser().parse_args(seen[0])
    assert parsed.output_dir == (str(stores[0]) if expected[0] == 'config' else expected[0])
    assert parsed.data_dir == (str(stores[1]) if expected[1] == 'config' else expected[1])


def test_data_environment_retains_priority(stores, monkeypatch):
    storage.adopt(*map(str, stores))
    monkeypatch.setenv('NEUROFLY_DATA_DIR', '/environment')
    seen = daemon(monkeypatch)
    assert cli.main(['--port', '8769']) == 0
    assert '--data-dir' not in seen[0]


@pytest.mark.parametrize('damage', ['json', 'schema', 'relative', 'duplicate', 'missing', 'not-directory'])
def test_damaged_config_refuses_before_daemon(stores, monkeypatch, damage):
    storage.adopt(*map(str, stores))
    path = storage.config_path()
    doc = storage.read_selection()
    if damage == 'json': path.write_text('{bad')
    elif damage == 'schema': path.write_text(json.dumps({**doc, 'schema': 2}))
    elif damage == 'relative': path.write_text(json.dumps({**doc, 'data_dir': 'relative'}))
    elif damage == 'duplicate': path.write_text('{"schema":1,"schema":1}')
    elif damage == 'missing': stores[1].rename(stores[1].with_name('retained-elsewhere'))
    else: doc['data_dir'] = str(stores[1] / 'history.jsonl'); path.write_text(json.dumps(doc))
    seen = daemon(monkeypatch)
    assert cli.main(['run']) == 2
    assert seen == []


def test_both_explicit_flags_bypass_corrupt_config(stores, monkeypatch):
    storage.config_path().parent.mkdir()
    storage.config_path().write_text('broken')
    seen = daemon(monkeypatch)
    assert cli.main(['run', '--output-dir', str(stores[0]), '--data-dir', str(stores[1])]) == 0
    assert len(seen) == 1


def test_no_config_forwarding_and_help_no_writes(stores, monkeypatch):
    seen = daemon(monkeypatch)
    assert cli.main(['run', '--port', '8769']) == 0
    assert seen == [['--port', '8769']]
    for args in [['storage', '--help'], ['storage', 'use', '--help']]:
        with pytest.raises(SystemExit) as result: cli.main(args)
        assert result.value.code == 0
    assert not storage.config_path().parent.exists()
    assert cli.main(['storage', 'show']) == 0
    assert not storage.config_path().parent.exists()


def test_failed_replace_retains_selection(stores, monkeypatch):
    storage.adopt(*map(str, stores))
    before = storage.config_path().read_bytes()
    def refuse(*args): raise OSError('injected replacement failure')
    monkeypatch.setattr(storage.os, 'replace', refuse)
    with pytest.raises(storage.StorageError): storage.adopt(str(stores[1]), str(stores[0]))
    assert storage.config_path().read_bytes() == before
    assert not list(storage.config_path().parent.glob('.storage-*'))


def test_concurrent_update_refuses(stores):
    import fcntl
    storage.adopt(*map(str, stores))
    before = storage.config_path().read_bytes()
    with (storage.config_path().parent / '.storage.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(storage.StorageError, match='already in progress'):
            storage.adopt(str(stores[1]), str(stores[0]))
    assert storage.config_path().read_bytes() == before


def test_installed_cli_reuses_selection_outside_new_checkout(installed_wheel, stores):
    scratch, installed, deps, _, _ = installed_wheel
    storage.adopt(*map(str, stores))
    code = r'''
import sys,json,types,argparse,pathlib
sys.path[:] = [sys.argv[1]] + json.loads(sys.argv[2]) + [p for p in sys.path if p]
from neurofly import cli,storage
assert pathlib.Path(cli.__file__).is_relative_to(sys.argv[1])
assert pathlib.Path(storage.__file__).is_relative_to(sys.argv[1])
expected=tuple(json.loads(sys.argv[3]))
fake=types.ModuleType('neurofly_daemon')
def parser():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir'); p.add_argument('--data-dir'); return p
fake.build_arg_parser=parser
seen=[]
fake.run_daemon=lambda:seen.append(parser().parse_args(sys.argv[1:]))
sys.modules['neurofly_daemon']=fake
assert cli.main(['run'])==0
assert (seen[0].output_dir,seen[0].data_dir)==expected
print('INSTALLED_SELECTION_REUSED')
'''
    env = {'PATH': os.defpath, 'NEUROFLY_CONFIG_HOME': str(storage.config_path().parent)}
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(installed), json.dumps(deps),
                             json.dumps(list(map(str, stores)))], cwd=scratch, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'INSTALLED_SELECTION_REUSED' in result.stdout


@pytest.mark.parametrize('output,data', [('', ''), (' ', ' '), ('', '/override'), ('/override', '\t')])
def test_empty_explicit_selection_refuses_without_writes(stores, monkeypatch, output, data):
    storage.adopt(*map(str, stores))
    before = {storage.config_path(): storage.config_path().read_bytes(),
              **{p: p.read_bytes() for root in stores for p in root.rglob('*') if p.is_file()}}
    seen = daemon(monkeypatch)
    assert cli.main(['run', '--output-dir', output, '--data-dir', data]) == 2
    assert seen == []
    assert all(p.read_bytes() == value for p, value in before.items())


def test_config_location_symlink_loop_refuses(monkeypatch, tmp_path):
    loop = tmp_path / 'loop'
    loop.symlink_to(loop)
    monkeypatch.setenv('NEUROFLY_CONFIG_HOME', str(loop))
    seen = daemon(monkeypatch)
    assert cli.main(['run']) == 2
    assert seen == []
    assert cli.main(['storage', 'show']) == 2


def test_run_help_bypasses_unresolvable_config(monkeypatch, tmp_path):
    loop = tmp_path / 'loop'
    loop.symlink_to(loop)
    monkeypatch.setenv('NEUROFLY_CONFIG_HOME', str(loop))
    seen = daemon(monkeypatch)
    assert cli.main(['run', '--help']) == 0
    assert seen == [['--help']]
    assert list(tmp_path.iterdir()) == [loop]


@pytest.mark.parametrize('name', ['NEUROFLY_CONFIG_HOME', 'XDG_CONFIG_HOME'])
def test_relative_config_override_refuses_across_checkout_changes(stores, monkeypatch, tmp_path, name):
    storage.adopt(*map(str, stores))
    selected = storage.config_path()
    before = {selected: selected.read_bytes(),
              **{p: p.read_bytes() for root in stores for p in root.rglob('*') if p.is_file()}}
    monkeypatch.delenv('NEUROFLY_CONFIG_HOME', raising=False)
    monkeypatch.setenv(name, 'relative-config')
    seen = daemon(monkeypatch)
    for dirname in ('release-old', 'release-new'):
        checkout = tmp_path / dirname
        checkout.mkdir()
        monkeypatch.chdir(checkout)
        assert cli.main(['run']) == 2
        assert cli.main(['storage', 'use', '--output-dir', str(stores[0]), '--data-dir', str(stores[1])]) == 2
        assert not list(checkout.iterdir())
    assert seen == []
    assert all(p.read_bytes() == value for p, value in before.items())
