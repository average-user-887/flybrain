"""The release-hygiene guard (scripts/check_private_infra.py) must catch every class.

Each planted example is assembled from pieces so that this file itself never contains
a string the guard would flag.
"""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / 'scripts' / 'check_private_infra.py'
J = ''.join

PLANTED = {
    'private-ip': J(['ip = "192', '.168.4', '.20"']),
    'home-path': J(['data at /ho', 'me/alice/flybrain/outputs']),
    'mac-home-path': J(['see /Us', 'ers/alice/Documents']),
    'win-home-path': J(['C:', '\\', 'Us', 'ers', '\\', 'alice\\flybrain']),
    'share-path': J(['/m', 'nt/archive-share/neurofly/snapshots']),
    'unc-path': J(['//file', 'server/Stor', 'age/neurofly']),
    'agent-scratch': J(['out: ~/.cla', 'ude/jo', 'bs/abc123/tmp/run.json']),
    'agent-worktree': J(['cwd .cla', 'ude/work', 'trees/feature-x']),
    'email': J(['contact: alice.smith', '@', 'corp-mail.com']),
    'host-field': J(['{"host', 'name": "alice-', 'desktop"}']),
    'denied-token': J(['ssh build', '-box-', '7 uptime']),
}
EXPECTED_CLASS = {
    'private-ip': 'private-ip', 'home-path': 'home-path', 'mac-home-path': 'home-path',
    'win-home-path': 'home-path', 'share-path': 'share-path', 'unc-path': 'share-path',
    'agent-scratch': 'agent-scratch', 'agent-worktree': 'agent-scratch', 'email': 'email',
    'host-field': 'host-field', 'denied-token': 'denied-token',
}
CLEAN = '\n'.join([
    'Paths are repository-relative: <repo>/outputs, <home>/x, <scratch>/run.json',
    'placeholder /ho' + 'me/<user>/flybrain and {"host": "<reference-host>"} and "host": "127.0.0.1"',
    'Co-Authored-By: Claude <noreply' + '@anthropic.com>',
    'someone' + '@users.noreply.github.com and git' + '@github.com:owner/repo.git',
    'AMD Ryzen 5 5600X, GTX 1660 Ti, /tmp/x, ~/.config/systemd/user/',
])


def _load_guard():
    spec = importlib.util.spec_from_file_location('check_private_infra', GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def guard(monkeypatch):
    mod = _load_guard()
    # A synthetic denied token, registered by hash exactly as the real ones are.
    monkeypatch.setattr(mod, 'DENIED_TOKEN_SHA256',
                        mod.DENIED_TOKEN_SHA256 | {hashlib.sha256(b'build-box-7').hexdigest()})
    return mod


def _tree(tmp_path, files):
    root = tmp_path / 'tree'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(REPO / 'scripts' / 'private_infra_allowlist.txt', root / 'scripts')
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + '\n')
    return root


def test_clean_tree_passes(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'docs/clean.md': CLEAN})
    assert guard.main(['--root', str(root), '--no-git']) == 0, capsys.readouterr().out


@pytest.mark.parametrize('name', sorted(PLANTED))
def test_each_class_is_caught(guard, tmp_path, capsys, name):
    root = _tree(tmp_path, {'docs/clean.md': CLEAN, f'docs/{name}.txt': PLANTED[name]})
    assert guard.main(['--root', str(root), '--no-git']) == 1
    out = capsys.readouterr().out
    assert f'docs/{name}.txt:1: [{EXPECTED_CLASS[name]}]' in out, out
    assert 'docs/clean.md' not in out


def test_denied_token_is_not_echoed(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'a.json': J(['"user": "build', '-box-', '7"'])})
    assert guard.main(['--root', str(root), '--no-git']) == 1
    out = capsys.readouterr().out
    assert '[denied-token]' in out and J(['build', '-box-', '7']) not in out


def test_runtime_tokens_from_environment(tmp_path, capsys, monkeypatch):
    mod = _load_guard()
    monkeypatch.setenv('NEUROFLY_PRIVATE_TOKENS', 'my-laptop')
    root = _tree(tmp_path, {'notes.md': 'ran on my-laptop.home yesterday'})
    assert mod.main(['--root', str(root), '--no-git']) == 1
    assert '[denied-token]' in capsys.readouterr().out


def test_allowlist_exempts_a_specific_match(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'docs/email.txt': PLANTED['email']})
    with open(root / 'scripts' / 'private_infra_allowlist.txt', 'a') as fh:
        fh.write('email docs/email.txt ^alice\\.smith@\n')
    assert guard.main(['--root', str(root), '--no-git']) == 0, capsys.readouterr().out


def test_walk_mode_honours_exclusions(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'outputs/run/log.txt': PLANTED['home-path'],
                            '.venv/lib/x.py': PLANTED['email'],
                            'docs/clean.md': CLEAN})
    assert guard.main(['--root', str(root), '--no-git']) == 0, capsys.readouterr().out


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_git_mode_scans_tracked_and_skips_ignored(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'docs/clean.md': CLEAN, 'ignored/leak.txt': PLANTED['share-path'],
                            '.gitignore': 'ignored/'})
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    assert guard.main(['--root', str(root)]) == 0, capsys.readouterr().out
    (root / 'docs' / 'leak.md').write_text(PLANTED['share-path'] + '\n')
    assert guard.main(['--root', str(root)]) == 1
    out = capsys.readouterr().out
    assert '(git mode)' in out and 'docs/leak.md:1: [share-path]' in out


def test_shell_wrapper_runs(tmp_path):
    root = _tree(tmp_path, {'docs/clean.md': CLEAN})
    r = subprocess.run(['bash', str(REPO / 'scripts' / 'check_private_infra.sh'), '--root', str(root),
                        '--no-git'], capture_output=True, text=True,
                       env={'PATH': '/usr/bin:/bin', 'PYTHON': sys.executable})
    assert r.returncode == 0, r.stdout + r.stderr


# --- no diagnostic may amplify a leak ------------------------------------------------------

PRIVATE_VALUES = {
    'email': J(['alice.smith', '@', 'corp-mail.com']),
    'share': J(['/m', 'nt/archive-share']),
    'home': J(['/ho', 'me/alice/']),
    'ip': J(['192', '.168.4', '.20']),
    'host': J(['alice-', 'desktop']),
    'token': J(['build', '-box-', '7']),
    'session': J(['claude', '.ai/', 'code/', 'session', '_', 'Qx9' * 8]),
}
NEEDLES = [*PRIVATE_VALUES.values(), 'alice', 'corp-mail', 'archive-share', 'Qx9Qx9', J(['build', '-box'])]


def _leaky_tree(tmp_path):
    """A failing tree: private values in file content and in file and directory names."""
    v = PRIVATE_VALUES
    files = {
        'docs/content.md': '\n'.join([f'mail {v["email"]}', f'data at {v["share"]}/x', f'{v["home"]}flybrain',
                                      f'ip = "{v["ip"]}"', f'ssh {v["token"]} uptime', f'https://{v["session"]}']),
        'docs/host.json': J(['{"host', 'name": "', v['host'], '"}']),
        f'docs/{v["email"]}/notes.md': 'clean',                 # private value as a directory name
        f'docs/{v["token"]}-notes.txt': 'clean',                 # private value in a file name
    }
    return _tree(tmp_path, files)


def _guard_cli(args):
    r = subprocess.run([sys.executable, str(GUARD), *args], capture_output=True, text=True,
                       env={'PATH': '/usr/bin:/bin', 'NEUROFLY_PRIVATE_TOKENS': PRIVATE_VALUES['token'],
                            'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull})
    both = r.stdout + r.stderr
    for needle in NEEDLES:
        assert needle not in both, (needle, both)
    return r


def _commit_all(root):
    git = ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t' + '@example.com',
           '-c', 'commit.gpgsign=false']
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(git + ['add', '-A'], check=True)
    subprocess.run(git + ['commit', '-qm', 'x'], check=True)


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_tree_diagnostics_never_print_private_values(tmp_path):
    root = _leaky_tree(tmp_path)
    r = _guard_cli(['--root', str(root), '--no-git'])
    assert r.returncode == 1
    for cls in ('email', 'share-path', 'home-path', 'private-ip', 'denied-token', 'agent-session', 'host-field'):
        assert f'[{cls}] <redacted> (id ' in r.stdout, cls
    assert 'docs/<redacted>/notes.md:0: [email]' in r.stdout              # the path itself matched
    assert 'docs/<redacted>:0: [denied-token]' in r.stdout
    _commit_all(root)
    assert _guard_cli(['--root', str(root)]).returncode == 1              # git mode
    r = _guard_cli(['--root', str(root), '--tree-rev', 'HEAD'])          # pushed-revision mode
    assert r.returncode == 1 and 'docs/<redacted>/notes.md:0: [email]' in r.stdout
    ids = [ln.rsplit('(id ', 1)[1] for ln in r.stdout.splitlines() if '(id ' in ln]
    assert ids and len(ids) == len(set(ids))                              # distinct finding ids
    assert ids == [ln.rsplit('(id ', 1)[1] for ln in
                   _guard_cli(['--root', str(root), '--tree-rev', 'HEAD']).stdout.splitlines() if '(id ' in ln]


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_error_paths_never_print_private_values(tmp_path):
    root = _leaky_tree(tmp_path)
    _commit_all(root)
    v = PRIVATE_VALUES
    for args in (['--root', str(root), '--tree-rev', v['token']],
                 ['--root', str(root), '--tree-rev', v['home'] + 'x'],
                 ['--root', str(root), '--commits', f'{v["token"]}..{v["email"]}'],
                 ['--root', str(root), '--commits', f'{v["host"]}.local'],
                 ['--root', str(root / v['token'] / 'missing')]):
        r = _guard_cli(args)
        assert r.returncode == 2, (args, r.stdout, r.stderr)
    # An absent allow-list means "no exceptions": still a failing scan, still nothing echoed.
    r = _guard_cli(['--root', str(root), '--no-git', '--allowlist', str(tmp_path / f'{v["token"]}.txt')])
    assert r.returncode == 1
    bad_allow = tmp_path / 'allow.txt'
    bad_allow.write_text(f'email docs/** ({v["email"]}\n')
    r = _guard_cli(['--root', str(root), '--no-git', '--allowlist', str(bad_allow)])
    assert r.returncode == 2 and 'failing closed' in r.stderr
    bad_allow.write_text(f'broken {v["token"]}\n')
    r = _guard_cli(['--root', str(root), '--no-git', '--allowlist', str(bad_allow)])
    assert r.returncode == 2 and 'failing closed' in r.stderr
