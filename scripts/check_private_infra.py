#!/usr/bin/env python3
"""Release-hygiene guard: fail if personal or internal-infrastructure data is in the tree.

Runs inside a git checkout (scans tracked files plus untracked files that are not
ignored) or on a plain directory such as an exported tarball (walks the tree with
the same exclusions). Standard library only.

    python3 scripts/check_private_infra.py [--root DIR] [--no-git] [--allowlist FILE]
    scripts/check_private_infra.sh            # same, used by CI
    scripts/check_private_infra.sh --commits origin/master..HEAD   # commit metadata

Tree mode (the default) scans file contents. Commit mode (--commits RANGE) scans the
metadata of every commit in a git revision range instead: the full message, author
name and e-mail, committer name and e-mail. RANGE is handed to `git rev-list` after
shell-style splitting, so "A..B" and "B --not --remotes=origin" both work. Only the
commits a push or pull request introduces are meant to be scanned: older published
commits are an accepted residual (docs/OWNER_DECISIONS.md, 2026-10-05, "no rewrite").

Classes of finding (each can be silenced for a specific path and match through
the allow-list, see scripts/private_infra_allowlist.txt):

  private-ip     RFC 1918 addresses (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
  home-path      /home/<user>/, /Users/<user>/, C:\\Users\\<user> (placeholders such as
                 /home/<user>/ are fine)
  share-path     /mnt/<share>, /media/<user>/, UNC //<host>/Storage, <drive>:\\neurofly
  agent-scratch  Claude/Codex/Gemini job, session and worktree paths, /tmp/claude-N
  email          any e-mail address except the noreply allow-list below
  host-field     a JSON "host"/"hostname" field holding a real machine name
  agent-session  links to private agent sessions or tasks (Claude Code sessions,
                 Claude chats/shares, ChatGPT/Codex tasks and chats, Gemini chats)
  session-trailer  a commit trailer such as "Claude-Session:" or "Codex-Task:"
  denied-token   known private usernames, hostnames and internal host names. They
                 are stored as SHA-256 hashes so that the guard does not republish
                 them. Add your own at run time with NEUROFLY_PRIVATE_TOKENS=a,b,c.

Commit mode adds two classes and is stricter about e-mail:

  identity-email an author or committer e-mail outside {*@users.noreply.github.com,
                 noreply@anthropic.com, noreply@github.com}; the same short list is
                 the only one accepted for e-mail addresses inside commit messages
  hostname       a LAN-style host name (<name>.local, .lan, .home, .internal, ...)

Findings in commit mode, and agent-session findings anywhere, are reported by
location and class only; the matched text is never printed.

Placeholders used by redacted receipts (<repo>/, <home>/, <scratch>/,
<reference-host>) never match. Exit status: 0 clean, 1 findings, 2 usage error.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Lower-cased SHA-256 of private tokens (username, workstation hostname, internal hosts,
# a personal mail local-part). Matched against every [-._]-separated span of every word.
DENIED_TOKEN_SHA256 = {
    '49a20d5bc033e9c98fffebede701227d9d87ee00e3a5be7c89914837f6f7b83f',
    '43921294b18644b1059a71ce971e47c97a8204262dead79044255969a4648b4c',
    'f1e44a3d20392dd9f27da94421742930cbed6398c8d7e4c52afa839b39bf21fa',
    'b5752f1c53ee0bfe37cd043454e4fe2ffc82687bc8b74243cff135833e0a12d0',
    '3b81d2dd823dac0f5e0edaf940137a672c40804e8a35e18cf947bde3f6fd6a3e',
    '94dba5ae712cb3c8b8a118d6a9362544bf27103a3ec8e92625587acd8fb294df',
    # carried over from the earlier CI grep: a private container name and container id
    '698ef57b2fb5dc994e8cedfc074e936202f4d10eead43b80a2e44fbf042b5555',
    '69d0b083f23041a4ee875e3d8af96e7b8d2929828c2fc42f5bbb66e64408d3b9',
}

EMAIL_ALLOW = [
    re.compile(r'^noreply@anthropic\.com$', re.I),
    re.compile(r'^[A-Za-z0-9._+-]+@users\.noreply\.github\.com$', re.I),
    re.compile(r'^noreply@github\.com$', re.I),
    re.compile(r'^git@github\.com$', re.I),          # SSH clone URL, not a mailbox
    re.compile(r'^[A-Za-z0-9._+-]+@example\.(com|org|net)$', re.I),
]

PATTERNS = [
    ('private-ip', re.compile(
        r'(?<![\d.])(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.]*\d)')),
    ('home-path', re.compile(r'/home/[A-Za-z0-9._-]*[A-Za-z0-9_-]/|/Users/[A-Za-z0-9._-]+/'
                             r'|\b[A-Za-z]:(?:\\\\?|/)Users(?:\\\\?|/)[A-Za-z0-9._ -]*[A-Za-z0-9]')),
    ('share-path', re.compile(r'/mnt/[A-Za-z0-9][A-Za-z0-9._-]*|/media/[A-Za-z0-9._-]+/'
                              r'|//[A-Za-z0-9_-]+/Storage\b|\\\\[A-Za-z0-9_-]+\\[A-Za-z]'
                              r'|\b[A-Z]:[/\\]neurofly')),
    ('agent-scratch', re.compile(r'\.(?:claude|codex|gemini)/(?:jobs|projects|sessions|tmp|todos|'
                                 r'shell-snapshots|worktrees)/|/tmp/claude-\d+')),
    ('email', re.compile(r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b')),
    ('host-field', re.compile(r'"(?:host|hostname)"\s*:\s*"([^"]*)"')),
    ('agent-session', re.compile(
        r'(?i)claude\.ai/(?:code/)?(?:sessions?|chat|share)[/_]|chatgpt\.com/(?:codex/tasks|c|share|g)/'
        r'|chat\.openai\.com/(?:c|share)/|gemini\.google\.com/(?:app|share)/'
        r'|\bsession_[A-Za-z0-9]{16,}')),
    ('session-trailer', re.compile(
        r'(?i)^\s*(?:claude|codex|chatgpt|openai|gemini|agent)[-_ ]?(?:session|task|chat|conversation)'
        r'(?:[-_ ]?(?:id|url|link))?\s*:')),
]
# Classes whose matched text is never printed, in either mode.
REDACT_CLASSES = {'denied-token', 'agent-session'}

# Commit-metadata mode: the only e-mail addresses a new commit may carry.
METADATA_EMAIL_ALLOW = [
    re.compile(r'^[A-Za-z0-9._+-]+@users\.noreply\.github\.com$', re.I),
    re.compile(r'^noreply@anthropic\.com$', re.I),
    re.compile(r'^noreply@github\.com$', re.I),
]
METADATA_PATTERNS = PATTERNS + [
    ('hostname', re.compile(r'(?i)\b[A-Za-z0-9][A-Za-z0-9-]*\.(?:local|lan|home|internal|intranet|'
                            r'localdomain|fritz\.box)\b')),
]
HOST_FIELD_OK = re.compile(r'^(?:|<[^>]+>|localhost|127\.0\.0\.1|0\.0\.0\.0|::1)$')

WORD = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
SEP = re.compile(r'([-._])')

# Directory names never scanned when walking a plain directory (mirrors .gitignore).
WALK_EXCLUDE_DIRS = {'.git', '.venv', 'venv', '.tox', 'node_modules', '__pycache__', '.pytest_cache',
                     'outputs', 'runs', 'connectome_data', 'upstream', '.claude', 'build',
                     'neurofly-site', '.eggs'}
WALK_EXCLUDE_PATHS = {'experiment_data/battery'}
SELF_FILES = {'scripts/private_infra_allowlist.txt'}


def _denied_spans(word: str, extra: set[str]):
    parts = SEP.split(word)            # [w, sep, w, sep, w ...]
    words = parts[0::2]
    if len(words) > 12:
        words = words[:12]
    n = len(words)
    for i in range(n):
        s = words[i]
        for j in range(i, n):
            if j > i:
                s = s + parts[2 * j - 1] + words[j]
            low = s.lower()
            if low in extra or hashlib.sha256(low.encode()).hexdigest() in DENIED_TOKEN_SHA256:
                yield s


def load_allowlist(path: Path):
    rules = []
    if not path.is_file():
        return rules
    for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        bits = line.split(None, 2)
        if len(bits) != 3:
            raise SystemExit(f'{path}:{n}: expected "<class> <path-glob> <regex>"')
        cls, glob, rx = bits
        rules.append((cls, glob, re.compile(rx), f'{path.name}:{n}'))
    return rules


def allowed(rules, cls, rel, text):
    for rcls, glob, rx, _ in rules:
        if (rcls == cls or rcls == '*') and fnmatch.fnmatch(rel, glob) and rx.search(text):
            return True
    return False


def list_files(root: Path, use_git: bool):
    if use_git:
        try:
            out = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others',
                                  '--exclude-standard'], capture_output=True, check=True).stdout
            files = sorted({f for f in out.decode('utf-8', 'surrogateescape').split('\0') if f})
            return files, 'git'
        except (OSError, subprocess.CalledProcessError):
            pass
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, '/')
        rel_dir = '' if rel_dir == '.' else rel_dir
        dirnames[:] = sorted(d for d in dirnames if d not in WALK_EXCLUDE_DIRS
                             and f'{rel_dir}/{d}'.lstrip('/') not in WALK_EXCLUDE_PATHS)
        for f in sorted(filenames):
            files.append(f'{rel_dir}/{f}'.lstrip('/'))
    return files, 'walk'


def scan_text(rel, text, rules, extra, patterns=None, email_allow=None):
    patterns = PATTERNS if patterns is None else patterns
    email_allow = EMAIL_ALLOW if email_allow is None else email_allow
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for cls, rx in patterns:
            for m in rx.finditer(line):
                hit = m.group(0)
                if cls == 'email' and any(a.match(hit) for a in email_allow):
                    continue
                if cls == 'host-field' and HOST_FIELD_OK.match(m.group(1)):
                    continue
                if allowed(rules, cls, rel, hit):
                    continue
                if cls in REDACT_CLASSES:
                    hit = f'<redacted: {cls} match>'
                findings.append((rel, lineno, cls, hit))
        for w in WORD.finditer(line):
            for span in _denied_spans(w.group(0), extra):
                if not allowed(rules, 'denied-token', rel, span):
                    findings.append((rel, lineno, 'denied-token', '<redacted: matches a denied token>'))
                break
    return findings


_FIELDS = ('author-name', 'author-email', 'committer-name', 'committer-email', 'message')


def read_commits(root: Path, rev_args):
    """Yield (sha, {field: value}) for every commit in the revision range, oldest first."""
    fmt = '%x1e%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B'
    r = subprocess.run(['git', '-C', str(root), '-c', 'log.showSignature=false', 'log', '--reverse',
                        '--no-color', '--no-mailmap', f'--format={fmt}', *rev_args, '--'],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('utf-8', 'replace').strip() or 'git log failed')
    for rec in r.stdout.decode('utf-8', 'replace').split('\x1e')[1:]:
        parts = rec.split('\x1f', 5)
        if len(parts) != 6:
            continue
        yield parts[0].strip(), dict(zip(_FIELDS, parts[1:]))


def scan_commits(root: Path, rev_args, rules, extra):
    """Scan commit metadata. Returns (number of commits, findings); every hit is redacted."""
    findings, n = [], 0
    for sha, fields in read_commits(root, rev_args):
        n += 1
        for field, value in fields.items():
            rel = f'commit:{sha}:{field}'
            if field.endswith('-email'):
                v = value.strip()
                if not any(a.match(v) for a in METADATA_EMAIL_ALLOW) and not allowed(rules, 'identity-email', rel, v):
                    findings.append((sha, field, 0, 'identity-email'))
                hits = scan_text(rel, v, rules, extra, METADATA_PATTERNS, METADATA_EMAIL_ALLOW)
                hits = [h for h in hits if h[2] != 'email']       # covered by identity-email
            else:
                hits = scan_text(rel, value, rules, extra, METADATA_PATTERNS, METADATA_EMAIL_ALLOW)
            findings += [(sha, field, lineno, cls) for _, lineno, cls, _ in hits]
    return n, findings


def main_commits(root: Path, spec: str, rules, extra):
    rev_args = shlex.split(spec)
    if not rev_args:
        print('[audit] --commits needs a revision range, e.g. origin/master..HEAD', file=sys.stderr)
        return 2
    try:
        n, findings = scan_commits(root, rev_args, rules, extra)
    except (OSError, RuntimeError) as exc:
        print(f'[audit] cannot list commits for {spec!r}: {exc}', file=sys.stderr)
        return 2
    print(f'[audit] scanning metadata of {n} commit(s) in {spec!r}')
    for sha, field, lineno, cls in findings:
        where = f'{field}:{lineno}' if field == 'message' else field
        print(f'{sha[:12]} {where}: [{cls}] <redacted>')
    if findings:
        bad = len({f[0] for f in findings})
        print(f'[audit] FAILED: {len(findings)} private-data finding(s) in {bad} commit(s). Reword the '
              f'message (git commit --amend / git rebase -i) and set user.email to a noreply address '
              f'before pushing. Published history is not rewritten; see docs/OWNER_DECISIONS.md.')
        return 1
    print('[audit] PASSED: no personal or private data in commit metadata.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=None, help='tree to scan (default: repository root)')
    ap.add_argument('--no-git', action='store_true', help='walk the directory even inside a git checkout')
    ap.add_argument('--allowlist', default=None, help='allow-list file (default: scripts/private_infra_allowlist.txt under the root)')
    ap.add_argument('--commits', metavar='RANGE', default=None,
                    help='scan commit metadata (message, author, committer) of a git revision range '
                         'instead of the tree, e.g. origin/master..HEAD')
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    if not root.is_dir():
        print(f'[audit] no such directory: {root}', file=sys.stderr)
        return 2
    allow_path = Path(args.allowlist) if args.allowlist else root / 'scripts' / 'private_infra_allowlist.txt'
    rules = load_allowlist(allow_path)
    extra = {t.strip().lower() for t in os.environ.get('NEUROFLY_PRIVATE_TOKENS', '').split(',') if t.strip()}
    if args.commits is not None:
        return main_commits(root, args.commits, rules, extra)
    files, mode = list_files(root, not args.no_git)
    print(f'[audit] scanning {len(files)} files under {root.name}/ ({mode} mode)')
    findings = []
    for rel in files:
        if rel in SELF_FILES:
            continue
        p = root / rel
        if p.is_symlink() or not p.is_file():
            continue
        data = p.read_bytes()
        if b'\0' in data[:8192]:
            continue                                   # binary
        findings += scan_text(rel, data.decode('utf-8', 'replace'), rules, extra)
    for rel, lineno, cls, hit in findings:
        print(f'{rel}:{lineno}: [{cls}] {hit}')
    if findings:
        print(f'[audit] FAILED: {len(findings)} private-data finding(s). Redact them, or add a narrow '
              f'entry to scripts/private_infra_allowlist.txt if the match is a genuine exception.')
        return 1
    print('[audit] PASSED: no personal or private-infrastructure data found.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
