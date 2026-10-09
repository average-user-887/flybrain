"""Agent session references must fail the release-hygiene guard, in the tree and in messages.

Covers claude.ai session and artifact links, session trailers, bare session_ ids, the
agent home directories, /tmp/claude scratch paths, Codex rollout names and the canonical
UUIDs used as Claude session and Codex thread ids. Every planted value is assembled from
pieces so that this file itself never contains a string the guard would flag.
"""
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / 'scripts' / 'check_private_infra.py'
J = ''.join

SYNTH = J(['Rq5', 'Lm2', 'Vx8', 'Hb4', 'Tk6', 'Pw3', 'Zn9'])            # synthetic session id
UUID = J(['5f0c2a9e', '-7d1b-', '4c3a-', '9e2f-', '0b1c2d3e4f5a'])        # synthetic thread id
PLANTED = {
    'claude-session-link': ('agent-session', J(['see https://', 'claude', '.ai/', 'code/', 'session', '_', SYNTH])),
    'claude-artifact-link': ('agent-session', J(['https://', 'claude', '.ai/', 'code/', 'artifact', '/', UUID])),
    'claude-chat-link': ('agent-session', J(['https://', 'claude', '.ai/', 'chat', '/', UUID])),
    'session-trailer': ('session-trailer', J(['Claude', '-Session', ': ', 'internal'])),
    'bare-session-id': ('agent-session', J(['resumed ', 'session', '_', SYNTH])),
    'claude-home': ('agent-scratch', J(['copied to ', '~/', '.cla', 'ude', '/settings.json'])),
    'claude-projects': ('agent-scratch', J(['log in .cla', 'ude/', 'projects/', '-x-y/', UUID[:8]])),
    'tmp-claude': ('agent-scratch', J(['scratch in /tmp/', 'claude', '/tasks/out.txt'])),
    'codex-home': ('agent-scratch', J(['read ', '$HOME', '/.co', 'dex', '/config.toml'])),
    'codex-rollout': ('agent-scratch', J(['rollout', '-2026-10-07T12-00-00-', UUID, '.jsonl'])),
    'codex-thread-id': ('agent-uuid', J(['thread ', UUID, ' resumed'])),
}
CLEAN = '\n'.join([
    'Co-Authored-By: Claude Opus 5.5 <noreply' + '@anthropic.com>',
    'The project ignores /.claude/ in .gitignore; see .claude settings docs.',
    'A session trailer is removed before publishing; the product site is claude' + '.ai/code.',
    'commit 4599280a and digest 27ef1b6f41088314ae3d32ba2418a3d05dc5a8804f77683a248394b77a65ca19',
    'version 0.5.1rc2, /tmp/neurofly-run/x, ~/.config/neurofly/',
])
GIT_ENV = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': os.devnull, 'LC_ALL': 'C'}
NOREPLY = J(['someone', '@users.noreply.github.com'])


def _load_guard():
    spec = importlib.util.spec_from_file_location('check_private_infra', GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def guard():
    return _load_guard()


def _tree(tmp_path, files):
    root = tmp_path / 'tree'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(REPO / 'scripts' / 'private_infra_allowlist.txt', root / 'scripts')
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + '\n')
    return root


def _no_payload(out):
    assert SYNTH not in out and UUID not in out and UUID[:8] not in out, out


def test_clean_text_passes_tree_check(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'docs/clean.md': CLEAN})
    assert guard.main(['--root', str(root), '--no-git']) == 0, capsys.readouterr().out


@pytest.mark.parametrize('name', sorted(PLANTED))
def test_tree_check_fails_on_agent_reference(guard, tmp_path, capsys, name):
    cls, text = PLANTED[name]
    root = _tree(tmp_path, {'docs/clean.md': CLEAN, f'docs/{name}.txt': text})
    assert guard.main(['--root', str(root), '--no-git']) == 1
    out = capsys.readouterr().out
    assert f'docs/{name}.txt:1: [{cls}] <redacted>' in out, out
    assert 'docs/clean.md' not in out
    _no_payload(out)


def _git(repo, *args, env=None):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True,
                          env=dict(GIT_ENV, HOME=str(repo), **(env or {})))


def _commit(repo, message):
    env = {'GIT_AUTHOR_NAME': 'Some One', 'GIT_AUTHOR_EMAIL': NOREPLY,
           'GIT_COMMITTER_NAME': 'Some One', 'GIT_COMMITTER_EMAIL': NOREPLY}
    _git(repo, 'commit', '-q', '--allow-empty', '-m', message, env=env)
    return _git(repo, 'rev-parse', 'HEAD').stdout.strip()


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_commit_message_check_fails_on_each_agent_reference(guard, tmp_path, capsys):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'master')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    base = _commit(repo, 'docs: start\n\n' + CLEAN)
    shas = {name: _commit(repo, f'fix: {name}\n\n{text}\n') for name, (_, text) in sorted(PLANTED.items())}
    clean = _commit(repo, 'docs: tidy\n\n' + CLEAN + '\n')

    assert guard.main(['--root', str(repo), f'--commits={base}']) == 0, capsys.readouterr().out
    assert guard.main(['--root', str(repo), f'--commits={shas[max(shas)]}..{clean}']) == 0
    capsys.readouterr()

    assert guard.main(['--root', str(repo), f'--commits={base}..HEAD']) == 1
    out = capsys.readouterr().out
    for name, (cls, _) in PLANTED.items():
        assert f'{shas[name][:12]} message:3: [{cls}] <redacted>' in out, (name, out)
    assert clean[:12] not in out
    _no_payload(out)
