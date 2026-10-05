"""Commit-metadata mode of the release-hygiene guard (scripts/check_private_infra.py --commits),
the CI range helper (scripts/ci_commit_range.sh) and the pre-push hook (scripts/hooks/pre-push).

Commits are planted in throw-away repositories under tmp_path. Every planted private
string is assembled from pieces so that this file itself passes the tree guard.
"""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / 'scripts' / 'check_private_infra.py'
J = ''.join

SESSION_URL = J(['https://', 'claude', '.ai/', 'code/', 'session', '_', 'Zq7' * 8])
SESSION_TRAILER = J(['Claude', '-Session', ': ', SESSION_URL])
CODEX_TASK = J(['see https://', 'chatgpt', '.com/', 'codex/tasks/', 'task_e_', 'abc123' * 4])
PERSONAL_EMAIL = J(['alice.smith', '@', 'corp-mail.com'])
LAN_HOST = J(['built on alice-', 'desktop', '.local overnight'])
DENIED = J(['build', '-box-', '7'])
NOREPLY = J(['someone', '@users.noreply.github.com'])
CLEAN_MSG = J(['docs: tidy the README\n\nCo-Authored-By: Claude Opus 5.5 <noreply', '@anthropic.com>\n'])

GIT_ENV = {
    'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_CONFIG_GLOBAL': os.devnull,
    'LC_ALL': 'C',
}


def _git(repo, *args, env=None, input=None, check=True):
    e = dict(GIT_ENV, HOME=str(repo), **(env or {}))
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True,
                          env=e, input=input, check=check)


def _init(path):
    path.mkdir(parents=True, exist_ok=True)
    _git(path, 'init', '-q', '-b', 'master')
    _git(path, 'config', 'commit.gpgsign', 'false')
    return path


def _commit(repo, message, author=('Some One', NOREPLY), committer=('Some One', NOREPLY)):
    env = {'GIT_AUTHOR_NAME': author[0], 'GIT_AUTHOR_EMAIL': author[1],
           'GIT_COMMITTER_NAME': committer[0], 'GIT_COMMITTER_EMAIL': committer[1]}
    _git(repo, 'commit', '-q', '--allow-empty', '-m', message, env=env)
    return _git(repo, 'rev-parse', 'HEAD').stdout.strip()


def _load_guard():
    spec = importlib.util.spec_from_file_location('check_private_infra', GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def guard(monkeypatch):
    mod = _load_guard()
    monkeypatch.setattr(mod, 'DENIED_TOKEN_SHA256',
                        mod.DENIED_TOKEN_SHA256 | {hashlib.sha256(DENIED.encode()).hexdigest()})
    return mod


@pytest.fixture
def planted(tmp_path):
    """A repository with one clean base commit, then one commit per kind of leak, then a clean one."""
    repo = _init(tmp_path / 'repo')
    shas = {'base': _commit(repo, CLEAN_MSG)}
    shas['trailer'] = _commit(repo, 'feat: add a thing\n\n' + SESSION_TRAILER + '\n')
    shas['codex'] = _commit(repo, 'fix: a bug\n\n' + CODEX_TASK + '\n')
    shas['author-email'] = _commit(repo, 'chore: bump', author=('Alice Smith', PERSONAL_EMAIL))
    shas['hostname'] = _commit(repo, 'perf: faster\n\n' + LAN_HOST + '\n')
    shas['denied'] = _commit(repo, 'test: more', committer=(DENIED, NOREPLY))
    shas['clean'] = _commit(repo, CLEAN_MSG)
    return repo, shas


def _run(guard, repo, spec, capsys):
    rc = guard.main(['--root', str(repo), '--commits', spec])
    return rc, capsys.readouterr().out


def test_each_leak_is_reported_by_sha_field_and_rule(guard, planted, capsys):
    repo, s = planted
    rc, out = _run(guard, repo, f"{s['base']}..HEAD", capsys)
    assert rc == 1, out
    expect = [
        (s['trailer'], 'message:3', 'session-trailer'),
        (s['trailer'], 'message:3', 'agent-session'),
        (s['codex'], 'message:3', 'agent-session'),
        (s['author-email'], 'author-email', 'identity-email'),
        (s['hostname'], 'message:3', 'hostname'),
        (s['denied'], 'committer-name', 'denied-token'),
    ]
    for sha, where, cls in expect:
        assert f'{sha[:12]} {where}: [{cls}] <redacted>' in out, (sha, where, cls, out)
    flagged = {line.split()[0] for line in out.splitlines() if not line.startswith('[audit]')}
    assert flagged == {s[k][:12] for k in ('trailer', 'codex', 'author-email', 'hostname', 'denied')}
    assert 'metadata of 6 commit(s)' in out and 'in 5 commit(s)' in out


def test_output_never_echoes_the_private_text(guard, planted, capsys):
    repo, s = planted
    _, out = _run(guard, repo, f"{s['base']}..HEAD", capsys)
    for secret in (SESSION_URL, 'Zq7Zq7', J(['task_e_', 'abc']), PERSONAL_EMAIL, 'alice.smith',
                   J(['desktop', '.local']), DENIED, J(['corp', '-mail'])):
        assert secret not in out, secret


def test_clean_range_passes(guard, planted, capsys):
    repo, s = planted
    rc, out = _run(guard, repo, f"{s['denied']}..HEAD", capsys)
    assert rc == 0, out
    assert 'metadata of 1 commit(s)' in out and 'PASSED' in out
    rc, out = _run(guard, repo, s['base'], capsys)          # a root commit on its own
    assert rc == 0, out


def test_empty_range_passes_and_bad_range_is_a_usage_error(guard, planted, capsys):
    repo, _ = planted
    rc, out = _run(guard, repo, 'HEAD..HEAD', capsys)
    assert rc == 0 and 'metadata of 0 commit(s)' in out
    assert guard.main(['--root', str(repo), '--commits', 'no-such-ref..HEAD']) == 2
    assert guard.main(['--root', str(repo), '--commits', '  ']) == 2


def test_multi_argument_range(guard, planted, capsys):
    repo, s = planted
    _git(repo, 'update-ref', 'refs/remotes/origin/master', s['hostname'])
    rc, out = _run(guard, repo, 'HEAD --not --remotes=origin', capsys)
    assert rc == 1 and 'metadata of 2 commit(s)' in out
    assert f"{s['denied'][:12]} committer-name: [denied-token]" in out


@pytest.mark.parametrize('email,ok', [
    (NOREPLY, True),
    (J(['noreply', '@anthropic.com']), True),
    (J(['noreply', '@github.com']), True),
    (J(['git', '@github.com']), False),          # fine in the tree, not as an identity
    (J(['dev', '@example.com']), False),
    (J(['someone', '@ryzen']), False),            # no TLD: still not an allowed identity
])
def test_identity_email_allow_list(guard, tmp_path, capsys, email, ok):
    repo = _init(tmp_path / 'r')
    _commit(repo, 'x', author=('A', email))
    rc, out = _run(guard, repo, 'HEAD', capsys)
    assert rc == (0 if ok else 1), out
    if not ok:
        assert 'author-email: [identity-email] <redacted>' in out


def test_message_email_uses_the_strict_list(guard, tmp_path, capsys):
    repo = _init(tmp_path / 'r')
    _commit(repo, 'x\n\nreported by ' + J(['dev', '@example.com']))
    rc, out = _run(guard, repo, 'HEAD', capsys)
    assert rc == 1 and 'message:3: [email] <redacted>' in out


def test_tree_mode_catches_and_redacts_session_links(guard, tmp_path, capsys):
    root = tmp_path / 'tree'
    (root / 'docs').mkdir(parents=True)
    (root / 'docs' / 'notes.md').write_text('handoff: ' + SESSION_URL + '\n' + SESSION_TRAILER + '\n')
    assert guard.main(['--root', str(root), '--no-git']) == 1
    out = capsys.readouterr().out
    assert 'docs/notes.md:1: [agent-session] <redacted: agent-session match>' in out
    assert 'docs/notes.md:2: [session-trailer]' in out
    assert 'Zq7Zq7' not in out


def test_shell_wrapper_passes_commits_through(planted):
    repo, s = planted
    r = subprocess.run(['bash', str(REPO / 'scripts' / 'check_private_infra.sh'), '--root', str(repo),
                        '--commits', f"{s['denied']}..HEAD"], capture_output=True, text=True,
                       env=dict(GIT_ENV, PYTHON=sys.executable))
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(['bash', str(REPO / 'scripts' / 'check_private_infra.sh'), '--root', str(repo),
                        '--commits', f"{s['base']}..{s['trailer']}"], capture_output=True, text=True,
                       env=dict(GIT_ENV, PYTHON=sys.executable))
    assert r.returncode == 1 and '[session-trailer]' in r.stdout


# --- CI range helper -------------------------------------------------------------------

def _ci_range(repo, **env):
    r = subprocess.run(['bash', str(REPO / 'scripts' / 'ci_commit_range.sh')], cwd=repo,
                       capture_output=True, text=True, env=dict(GIT_ENV, HOME=str(repo), **env))
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_ci_range_rules(guard, planted, capsys):
    repo, s = planted
    zeros = '0' * 40
    assert _ci_range(repo, EVENT='pull_request', PR_BASE=s['base'], PR_HEAD=s['clean']) \
        == f"{s['base']}..{s['clean']}"
    assert _ci_range(repo, EVENT='push', BEFORE=s['denied'], AFTER=s['clean'], REF_NAME='master') \
        == f"{s['denied']}..{s['clean']}"
    assert _ci_range(repo, EVENT='workflow_dispatch') == ''
    # First push of a branch: only commits no *other* remote branch has. The pushed
    # branch's own remote-tracking ref (which already points at AFTER) is excluded.
    _git(repo, 'update-ref', 'refs/remotes/origin/master', s['denied'])
    _git(repo, 'update-ref', 'refs/remotes/origin/feature', s['clean'])
    first = _ci_range(repo, EVENT='push', BEFORE=zeros, AFTER=s['clean'], REF_NAME='feature')
    rc, out = _run(guard, repo, first, capsys)
    assert rc == 0 and 'metadata of 1 commit(s)' in out, (first, out)
    # Force push whose old tip is not in the clone behaves like a first push.
    forced = _ci_range(repo, EVENT='push', BEFORE='1' * 40, AFTER=s['clean'], REF_NAME='feature')
    assert forced == first


# --- pre-push hook ------------------------------------------------------------------------

@pytest.fixture
def hooked(tmp_path):
    """A clone with the guard scripts committed and core.hooksPath set, plus a bare remote."""
    remote = tmp_path / 'remote.git'
    _git(tmp_path, 'init', '-q', '--bare', '-b', 'master', str(remote))
    repo = _init(tmp_path / 'clone')
    for rel in ('scripts/check_private_infra.py', 'scripts/check_private_infra.sh',
                'scripts/private_infra_allowlist.txt', 'scripts/hooks/pre-push'):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    _git(repo, 'add', '-A')
    _commit(repo, CLEAN_MSG)
    _git(repo, 'remote', 'add', 'origin', str(remote))
    _git(repo, 'config', 'core.hooksPath', 'scripts/hooks')
    return repo


def _push(repo, *refspec):
    return _git(repo, 'push', '-q', 'origin', *refspec, env={'PYTHON': sys.executable}, check=False)


def test_pre_push_hook_allows_clean_and_blocks_private_metadata(hooked):
    repo = hooked
    r = _push(repo, 'master')                                   # first push of a new branch
    assert r.returncode == 0, r.stdout + r.stderr
    _commit(repo, 'feat: x\n\n' + SESSION_TRAILER + '\n')
    r = _push(repo, 'master')                                   # update of an existing branch
    assert r.returncode != 0
    assert '[session-trailer]' in r.stdout + r.stderr and 'Zq7Zq7' not in r.stdout + r.stderr
    _git(repo, 'reset', '-q', '--hard', 'HEAD~1')
    _git(repo, 'checkout', '-q', '-b', 'topic')
    _commit(repo, 'chore: y', author=('Alice', PERSONAL_EMAIL))
    r = _push(repo, 'topic')                                    # new branch with a bad commit
    assert r.returncode != 0 and '[identity-email]' in r.stdout + r.stderr
    _git(repo, 'reset', '-q', '--hard', 'HEAD~1')
    _commit(repo, CLEAN_MSG)
    r = _push(repo, 'topic')
    assert r.returncode == 0, r.stdout + r.stderr
    r = _push(repo, ':topic')                                   # deletion sends no commits
    assert r.returncode == 0, r.stdout + r.stderr


def test_pre_push_hook_runs_the_tree_guard(hooked):
    repo = hooked
    (repo / 'leak.md').write_text(J(['/m', 'nt/archive-share/x']) + '\n')
    _git(repo, 'add', 'leak.md')
    _commit(repo, CLEAN_MSG)
    r = _push(repo, 'master')
    assert r.returncode != 0 and 'leak.md:1: [share-path]' in r.stdout + r.stderr
