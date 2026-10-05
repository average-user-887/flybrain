"""Commit-metadata and pushed-tree modes of the release-hygiene guard
(scripts/check_private_infra.py --commits / --tree-rev), the frozen historical exception
manifest, the CI tip selector (scripts/ci_privacy_tip.sh) and the pre-push hook
(scripts/hooks/pre-push).

Commits are planted in throw-away repositories under tmp_path. Every planted private
string is assembled from pieces so that this file itself passes the tree guard.
"""
import hashlib
import importlib.util
import os
import re
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
SHARE_LEAK = J(['/m', 'nt/archive-share/x'])
CLEAN_MSG = J(['docs: tidy the README\n\nCo-Authored-By: Claude Opus 5.5 <noreply', '@anthropic.com>\n'])
# Text that must never appear in any diagnostic.
SECRETS = (SESSION_URL, 'Zq7Zq7', J(['task_e_', 'abc']), PERSONAL_EMAIL, 'alice.smith',
           J(['desktop', '.local']), DENIED, J(['corp', '-mail']))

GIT_ENV = {
    'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
    'GIT_CONFIG_NOSYSTEM': '1',
    'GIT_CONFIG_GLOBAL': os.devnull,
    'LC_ALL': 'C',
}
SCRIPTS = ('scripts/check_private_infra.py', 'scripts/check_private_infra.sh',
           'scripts/private_infra_allowlist.txt', 'scripts/private_infra_commit_exceptions.txt',
           'scripts/hooks/pre-push')


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


def _no_secrets(text):
    for s in SECRETS:
        assert s not in text, s


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


def _run(guard, repo, spec, capsys, *extra):
    rc = guard.main(['--root', str(repo), f'--commits={spec}', *extra])
    cap = capsys.readouterr()
    return rc, cap.out + cap.err


# --- metadata detection and redaction ---------------------------------------------------

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
    assert 'metadata of 6 object(s)' in out and 'in 5 commit(s)' in out
    _no_secrets(out)


@pytest.mark.parametrize('field', ['author', 'committer', 'body'])
def test_bad_new_author_committer_or_body_is_rejected(guard, tmp_path, capsys, field):
    repo = _init(tmp_path / 'r')
    _commit(repo, CLEAN_MSG)
    bad = ('Alice', PERSONAL_EMAIL)
    if field == 'author':
        _commit(repo, 'x', author=bad)
    elif field == 'committer':
        _commit(repo, 'x', committer=bad)
    else:
        _commit(repo, 'x\n\nhanded off from ' + SESSION_URL + '\n')
    rc, out = _run(guard, repo, 'HEAD', capsys)
    assert rc == 1, out
    want = {'author': 'author-email: [identity-email]', 'committer': 'committer-email: [identity-email]',
            'body': 'message:3: [agent-session]'}[field]
    assert want in out
    _no_secrets(out)


def test_clean_ranges_pass(guard, planted, capsys):
    repo, s = planted
    rc, out = _run(guard, repo, f"{s['denied']}..HEAD", capsys)
    assert rc == 0 and 'metadata of 1 object(s)' in out and 'PASSED' in out, out
    rc, out = _run(guard, repo, s['base'], capsys)          # a root commit, full ancestry
    assert rc == 0, out
    rc, out = _run(guard, repo, 'HEAD..HEAD', capsys)
    assert rc == 0 and 'metadata of 0 object(s)' in out


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


def test_allowlist_file_never_exempts_metadata(guard, tmp_path, capsys):
    repo = _init(tmp_path / 'r')
    _commit(repo, 'x', author=('A', PERSONAL_EMAIL))
    allow = tmp_path / 'allow.txt'
    allow.write_text('* * .\nidentity-email * .\n')
    rc, out = _run(guard, repo, 'HEAD', capsys, '--allowlist', str(allow))
    assert rc == 1 and '[identity-email]' in out


def test_annotated_tag_metadata_is_scanned(guard, tmp_path, capsys):
    repo = _init(tmp_path / 'r')
    _commit(repo, CLEAN_MSG)
    _git(repo, 'tag', '-a', 'v1', '-m', 'release\n\n' + SESSION_TRAILER,
         env={'GIT_COMMITTER_NAME': 'T', 'GIT_COMMITTER_EMAIL': PERSONAL_EMAIL})
    rc, out = _run(guard, repo, 'v1', capsys)
    assert rc == 1 and 'tagger-email: [identity-email]' in out and '[session-trailer]' in out
    _no_secrets(out)


def test_nested_annotated_tags_are_all_scanned(guard, tmp_path, capsys):
    """An outer tag with clean metadata must not hide an inner tag it points at."""
    repo = _init(tmp_path / 'r')
    _commit(repo, CLEAN_MSG)
    clean_tagger = {'GIT_COMMITTER_NAME': 'T', 'GIT_COMMITTER_EMAIL': NOREPLY}
    _git(repo, 'tag', '-a', 'inner', '-m', 'inner\n\n' + SESSION_TRAILER, env=clean_tagger)
    _git(repo, 'tag', '-a', 'middle', 'inner', '-m', 'middle', env=clean_tagger)
    _git(repo, 'tag', '-a', 'outer', 'middle', '-m', 'outer', env=clean_tagger)
    inner = _git(repo, 'rev-parse', 'inner').stdout.strip()
    _git(repo, 'tag', '-d', 'inner')                        # only reachable through the chain now
    outer = _git(repo, 'rev-parse', 'outer').stdout.strip()
    for spec in ('outer', outer, f'HEAD..{outer}', '--tags', '--all'):
        rc, out = _run(guard, repo, spec, capsys)
        assert rc == 1, (spec, out)
        assert f'{inner[:12]} message:3: [session-trailer] <redacted>' in out, (spec, out)
        _no_secrets(out)
    rc, out = _run(guard, repo, '--glob=refs/tags/*', capsys)   # unsupported selector: fail closed
    assert rc == 2


def test_broken_tag_chain_fails_closed(guard, tmp_path, capsys):
    repo = _init(tmp_path / 'r')
    _commit(repo, CLEAN_MSG)
    _git(repo, 'tag', '-a', 'inner', '-m', 'inner', env={'GIT_COMMITTER_NAME': 'T', 'GIT_COMMITTER_EMAIL': NOREPLY})
    _git(repo, 'tag', '-a', 'outer', 'inner', '-m', 'outer', env={'GIT_COMMITTER_NAME': 'T', 'GIT_COMMITTER_EMAIL': NOREPLY})
    inner = _git(repo, 'rev-parse', 'inner').stdout.strip()
    _git(repo, 'tag', '-d', 'inner')
    (repo / '.git' / 'objects' / inner[:2] / inner[2:]).unlink()
    rc, out = _run(guard, repo, 'outer', capsys)
    assert rc == 2 and 'failing closed' in out


def test_tree_mode_catches_and_redacts_session_links(guard, tmp_path, capsys):
    root = tmp_path / 'tree'
    (root / 'docs').mkdir(parents=True)
    (root / 'docs' / 'notes.md').write_text('handoff: ' + SESSION_URL + '\n' + SESSION_TRAILER + '\n')
    assert guard.main(['--root', str(root), '--no-git']) == 1
    out = capsys.readouterr().out
    assert 'docs/notes.md:1: [agent-session] <redacted> (id ' in out
    assert 'docs/notes.md:2: [session-trailer]' in out
    assert 'Zq7Zq7' not in out


# --- fail closed ----------------------------------------------------------------------------

def test_invalid_missing_or_empty_ranges_fail_closed(guard, planted, capsys):
    repo, _ = planted
    for spec in ('no-such-ref..HEAD', '  ', 'f' * 40, '--bogus-option HEAD'):
        rc, out = _run(guard, repo, spec, capsys)
        assert rc == 2 and 'failing closed' in out, (spec, out)
    rc = guard.main(['--root', str(repo), '--tree-rev', 'e' * 40])
    assert rc == 2


def test_missing_object_in_history_fails_closed(guard, tmp_path, capsys):
    repo = _init(tmp_path / 'r')
    first = _commit(repo, CLEAN_MSG)
    (repo / 'a.md').write_text('hello\n')
    _git(repo, 'add', 'a.md')
    tip = _commit(repo, CLEAN_MSG)
    objects = repo / '.git' / 'objects'
    (objects / first[:2] / first[2:]).unlink()                          # a missing ancestor commit
    assert guard.main(['--root', str(repo), '--commits', tip]) == 2
    blob = _git(repo, 'rev-parse', 'HEAD:a.md').stdout.strip()
    (objects / blob[:2] / blob[2:]).unlink()                            # a missing file in the tree
    assert guard.main(['--root', str(repo), '--tree-rev', tip]) == 2
    assert 'failing closed' in capsys.readouterr().err


def test_manifest_must_match_its_pinned_digest(tmp_path, planted):
    repo, _ = planted
    scripts = tmp_path / 'copy' / 'scripts'
    scripts.mkdir(parents=True)
    for rel in ('check_private_infra.py', 'private_infra_commit_exceptions.txt'):
        shutil.copy2(REPO / 'scripts' / rel, scripts / rel)
    ok = subprocess.run([sys.executable, str(scripts / 'check_private_infra.py'), '--root', str(repo),
                         '--commits', 'HEAD~1..HEAD'], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    with open(scripts / 'private_infra_commit_exceptions.txt', 'a') as fh:
        fh.write('a' * 40 + ' identity-email\n')
    r = subprocess.run([sys.executable, str(scripts / 'check_private_infra.py'), '--root', str(repo),
                        '--commits', 'HEAD~1..HEAD'], capture_output=True, text=True)
    assert r.returncode == 2 and 'pinned' in r.stderr
    (scripts / 'private_infra_commit_exceptions.txt').unlink()
    r = subprocess.run([sys.executable, str(scripts / 'check_private_infra.py'), '--root', str(repo),
                        '--commits', 'HEAD~1..HEAD'], capture_output=True, text=True)
    assert r.returncode == 2


@pytest.mark.parametrize('line', ['abc identity-email', 'a' * 40, 'a' * 40 + ' no-such-class',
                                  'a' * 40 + ' identity-email extra'])
def test_malformed_manifest_fails_closed(guard, planted, tmp_path, capsys, line):
    repo, _ = planted
    m = tmp_path / 'm.txt'
    m.write_text('# header\n' + line + '\n')
    rc, out = _run(guard, repo, 'HEAD', capsys, '--commit-exceptions', str(m))
    assert rc == 2 and 'malformed' in out


# --- frozen historical exceptions -----------------------------------------------------------

def test_repository_manifest_is_sanitised_and_well_formed(guard):
    path = REPO / guard.COMMIT_EXCEPTIONS_FILE
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == guard.COMMIT_EXCEPTIONS_SHA256
    entries = guard.load_commit_exceptions(path, guard.COMMIT_EXCEPTIONS_SHA256)
    assert len(entries) > 0
    body = [ln for ln in data.decode().splitlines() if ln and not ln.startswith('#')]
    assert all(re.fullmatch(r'[0-9a-f]{40} [a-z-]+(,[a-z-]+)*', ln) for ln in body)
    assert '@' not in data.decode() and '://' not in data.decode()


def test_exception_covers_only_listed_commit_and_classes(guard, planted, tmp_path, capsys):
    repo, s = planted
    m = tmp_path / 'm.txt'
    m.write_text(f"{s['trailer']} agent-session,session-trailer\n"
                 f"{s['author-email']} hostname\n")       # wrong class: still fails
    rc, out = _run(guard, repo, f"{s['base']}..{s['author-email']}", capsys, '--commit-exceptions', str(m))
    assert rc == 1
    assert s['trailer'][:12] + ' ' not in out and f"{s['author-email'][:12]} author-email" in out
    assert 'not approval' in out
    rc, out = _run(guard, repo, f"{s['base']}..{s['codex']}", capsys, '--commit-exceptions', str(m))
    assert rc == 1 and s['codex'][:12] in out                   # unlisted commit: checked


def test_clean_forward_update_over_frozen_history_passes(guard, planted, tmp_path, capsys):
    repo, s = planted
    m = tmp_path / 'm.txt'
    rc = guard.main(['--root', str(repo), '--commits', 'HEAD', '--no-commit-exceptions', '--emit-exceptions'])
    m.write_text(capsys.readouterr().out)
    assert rc == 0 and len(m.read_text().splitlines()) == 5
    _no_secrets(m.read_text())
    _commit(repo, CLEAN_MSG)
    rc, out = _run(guard, repo, 'HEAD', capsys, '--commit-exceptions', str(m))        # full ancestry
    assert rc == 0, out
    rc, out = _run(guard, repo, f"{s['base']}..HEAD", capsys, '--commit-exceptions', str(m))
    assert rc == 0, out
    _commit(repo, 'oops\n\n' + SESSION_TRAILER)                                           # new leak
    rc, out = _run(guard, repo, 'HEAD', capsys, '--commit-exceptions', str(m))
    assert rc == 1 and 'in 1 commit(s)' in out


def test_exceptions_never_exempt_a_tree(guard, tmp_path, capsys):
    repo = _init(tmp_path / 'r')
    (repo / 'leak.md').write_text(SHARE_LEAK + '\n')
    _git(repo, 'add', 'leak.md')
    tip = _commit(repo, 'x\n\n' + SESSION_TRAILER)
    m = tmp_path / 'm.txt'
    m.write_text(f'{tip} agent-session,session-trailer\n')
    assert guard.main(['--root', str(repo), '--commits', tip, '--commit-exceptions', str(m)]) == 0
    capsys.readouterr()
    assert guard.main(['--root', str(repo), '--tree-rev', tip, '--commit-exceptions', str(m)]) == 1
    assert 'leak.md:1: [share-path]' in capsys.readouterr().out


# --- CI tip selector -------------------------------------------------------------------------

def _ci_tip(repo, **env):
    return subprocess.run(['bash', str(REPO / 'scripts' / 'ci_privacy_tip.sh')], cwd=repo,
                          capture_output=True, text=True, env=dict(GIT_ENV, HOME=str(repo), **env))


def test_ci_tip_selection_and_fail_closed(planted):
    repo, s = planted
    assert _ci_tip(repo, EVENT='pull_request', PR_HEAD=s['clean']).stdout.strip() == s['clean']
    assert _ci_tip(repo, EVENT='push', AFTER=s['denied']).stdout.strip() == s['denied']
    assert _ci_tip(repo, EVENT='workflow_dispatch', SHA=s['base']).stdout.strip() == s['base']
    for env in ({'EVENT': 'push', 'AFTER': '0' * 40}, {'EVENT': 'push'}, {'EVENT': 'pull_request'},
                {'EVENT': 'push', 'AFTER': 'HEAD'}, {'EVENT': 'push', 'AFTER': 'd' * 40}, {}):
        r = _ci_tip(repo, **env)
        assert r.returncode == 1 and r.stdout == '', env


def test_ci_does_not_exempt_commits_other_remote_refs_have(guard, planted, capsys):
    """Two new refs sharing a bad commit cannot hide it from each other."""
    repo, s = planted
    _git(repo, 'update-ref', 'refs/remotes/origin/a', s['clean'])
    _git(repo, 'update-ref', 'refs/remotes/origin/b', s['clean'])
    tip = _ci_tip(repo, EVENT='push', AFTER=s['clean']).stdout.strip()
    rc, out = _run(guard, repo, tip, capsys)
    assert rc == 1 and 'in 5 commit(s)' in out


# --- pre-push hook --------------------------------------------------------------------------

@pytest.fixture
def hooked(tmp_path):
    """A clone with the guard scripts committed and core.hooksPath set, plus a bare remote."""
    remote = tmp_path / 'remote.git'
    _git(tmp_path, 'init', '-q', '--bare', '-b', 'master', str(remote))
    repo = _init(tmp_path / 'clone')
    for rel in SCRIPTS:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    _git(repo, 'add', '-A')
    _commit(repo, CLEAN_MSG)
    _git(repo, 'remote', 'add', 'origin', str(remote))
    _git(repo, 'config', 'core.hooksPath', 'scripts/hooks')
    return repo


def _push(repo, *args):
    r = _git(repo, 'push', '-q', 'origin', *args, env={'PYTHON': sys.executable}, check=False)
    _no_secrets(r.stdout + r.stderr)
    return r


def _hook(repo, stdin):
    r = subprocess.run(['bash', str(repo / 'scripts' / 'hooks' / 'pre-push'), 'origin', 'url'], cwd=repo,
                       input=stdin, capture_output=True, text=True,
                       env=dict(GIT_ENV, HOME=str(repo), PYTHON=sys.executable))
    _no_secrets(r.stdout + r.stderr)
    return r


def test_hook_rejects_a_dirty_ref_that_is_not_checked_out(hooked):
    repo = hooked
    assert _push(repo, 'master').returncode == 0
    _git(repo, 'checkout', '-q', '-b', 'topic')
    (repo / 'leak.md').write_text(SHARE_LEAK + '\n')
    _git(repo, 'add', 'leak.md')
    _commit(repo, CLEAN_MSG)
    _git(repo, 'checkout', '-q', 'master')                     # clean checkout, dirty other ref
    assert not (repo / 'leak.md').exists()
    r = _push(repo, 'topic')
    assert r.returncode != 0 and 'leak.md:1: [share-path]' in r.stdout + r.stderr


def test_hook_rejects_two_new_refs_sharing_a_bad_commit(hooked):
    repo = hooked
    assert _push(repo, 'master').returncode == 0
    _git(repo, 'checkout', '-q', '-b', 'a')
    _commit(repo, 'feat\n\n' + SESSION_TRAILER)
    _git(repo, 'branch', 'b')
    r = _push(repo, 'a', 'b')
    assert r.returncode != 0 and (r.stdout + r.stderr).count('[session-trailer]') == 2
    # Even if one of them reached the remote some other way, the other is still checked.
    assert _push(repo, '--no-verify', 'a').returncode == 0
    _git(repo, 'fetch', '-q', 'origin')
    r = _push(repo, 'b')
    assert r.returncode != 0 and '[session-trailer]' in r.stdout + r.stderr


@pytest.mark.parametrize('field', ['author', 'committer', 'body'])
def test_hook_rejects_bad_new_metadata(hooked, field):
    repo = hooked
    bad = ('Alice', PERSONAL_EMAIL)
    if field == 'author':
        _commit(repo, 'x', author=bad)
    elif field == 'committer':
        _commit(repo, 'x', committer=bad)
    else:
        _commit(repo, 'x\n\n' + CODEX_TASK)
    r = _push(repo, 'master')
    assert r.returncode != 0
    assert ('[agent-session]' if field == 'body' else f'{field}-email: [identity-email]') in r.stdout + r.stderr


def test_hook_accepts_clean_forward_update_over_frozen_history(hooked):
    repo = hooked
    hist = _commit(repo, 'old\n\n' + SESSION_TRAILER, author=('Alice', PERSONAL_EMAIL))
    assert _push(repo, '--no-verify', 'master').returncode == 0          # published before the guard
    manifest = repo / 'scripts' / 'private_infra_commit_exceptions.txt'
    manifest.write_text('# test manifest\n' + f'{hist} agent-session,identity-email,session-trailer\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    guard_py = repo / 'scripts' / 'check_private_infra.py'
    guard_py.write_text(re.sub(r"COMMIT_EXCEPTIONS_SHA256 = '[0-9a-f]{64}'",
                               f"COMMIT_EXCEPTIONS_SHA256 = '{digest}'", guard_py.read_text()))
    _git(repo, 'add', '-A')
    _commit(repo, CLEAN_MSG)
    r = _push(repo, 'master')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'not approval' in r.stdout
    _commit(repo, CLEAN_MSG)
    assert _push(repo, 'master').returncode == 0
    _git(repo, 'checkout', '-q', '-b', 'fresh')                          # new ref over history
    _commit(repo, CLEAN_MSG)
    assert _push(repo, 'fresh').returncode == 0
    _commit(repo, 'new\n\n' + SESSION_TRAILER)
    assert _push(repo, 'fresh').returncode != 0


def test_hook_deletion_and_malformed_input(hooked):
    repo = hooked
    tip = _git(repo, 'rev-parse', 'HEAD').stdout.strip()
    zero = '0' * 40
    r = _hook(repo, f'(delete) {zero} refs/heads/gone {tip}\n')
    assert r.returncode == 0 and 'deletion' in r.stdout and 'checking' not in r.stdout
    r = _hook(repo, f'(delete) {zero} refs/heads/gone {"9" * 40}\n')    # unknown remote object: fine
    assert r.returncode == 0
    for bad in ('garbage\n', f'refs/heads/x {tip} refs/heads/x\n', f'refs/heads/x {tip[:12]} refs/heads/x {zero}\n',
                f'refs/heads/x {tip} refs/heads/x {zero} extra\n', f'refs/heads/x {"a" * 40} refs/heads/x {zero}\n'):
        r = _hook(repo, bad)
        assert r.returncode != 0, bad
    r = _hook(repo, '')                                                   # nothing pushed
    assert r.returncode == 0
    assert _push(repo, 'master:side').returncode == 0
    assert _push(repo, ':side').returncode == 0                           # real deletion push
