#!/usr/bin/env python3
"""Release-hygiene guard: fail if personal or internal-infrastructure data is in the tree.

Runs inside a git checkout (scans tracked files plus untracked files that are not
ignored) or on a plain directory such as an exported tarball (walks the tree with
the same exclusions). Standard library only.

    python3 scripts/check_private_infra.py [--root DIR] [--no-git] [--allowlist FILE]
    scripts/check_private_infra.sh            # same, used by CI
    scripts/check_private_infra.sh --commits HEAD        # commit metadata, whole ancestry
    scripts/check_private_infra.sh --tree-rev <sha>      # committed tree of a revision

Tree mode (the default) scans the checkout. Revision mode (--tree-rev REV) scans what a
push of REV publishes: the tree of REV itself (always, even if REV is old) and the tree of
every commit in REV's history that lies outside the frozen publication boundary
scripts/private_infra_published_boundary.txt (all commits already public at the cutoff,
digest-pinned, never extended from remote refs). The pre-push hook and CI use it for every
outgoing tip. Commit mode (--commits RANGE) scans the metadata of every
commit in a git revision range and of every annotated tag object reachable from its
arguments (tag chains are followed to the end): the full message,
author and committer (and tagger), plus other non-signature headers. RANGE is handed to
`git rev-list` after shell-style splitting; a single revision means its whole ancestry.

Commits that were already published when this guard was introduced are covered only by
the frozen historical exception manifest scripts/private_infra_commit_exceptions.txt
(owner ruling docs/OWNER_DECISIONS.md, 2026-10-05, "no rewrite"): an exact list of SHAs,
each with the rule classes it fails, pinned by COMMIT_EXCEPTIONS_SHA256 below. It is an
acknowledgement of immutable existing exposure, not an approval, and it never applies to
a tree. The allow-list file never applies to commit metadata. Anything the guard cannot
read or parse (bad range, missing object, malformed manifest) exits 2: fail closed.

Classes of finding (each can be silenced for a specific path and match through
the allow-list, see scripts/private_infra_allowlist.txt):

  private-ip     RFC 1918 addresses (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
  home-path      /home/<user>/, /Users/<user>/, C:\\Users\\<user> (placeholders such as
                 /home/<user>/ are fine)
  share-path     /mnt/<share>, /media/<user>/, UNC //<host>/Storage, <drive>:\\neurofly
  agent-scratch  Claude/Codex/Gemini job, session, project and worktree paths, the
                 agent home directories (.claude and .codex under ~), /tmp/claude-N scratch
                 directories and Codex rollout transcript names
  email          any e-mail address except the noreply allow-list below
  host-field     a JSON "host"/"hostname" field holding a real machine name
  agent-session  links to private agent sessions or tasks (Claude Code sessions,
                 Claude chats/shares/projects/artifacts, ChatGPT/Codex tasks and chats,
                 Gemini chats) and bare session_<id> identifiers
  agent-uuid     a canonical UUID, the form of Claude session and Codex thread
                 identifiers (allow-list a genuine non-agent UUID by path and value)
  session-trailer  an agent session or task trailer line ("<agent>-Session: ..."), as
                 added by agent tooling to commit messages
  denied-token   known private usernames, hostnames and internal host names. They
                 are stored as SHA-256 hashes so that the guard does not republish
                 them. Add your own at run time with NEUROFLY_PRIVATE_TOKENS=a,b,c.

Commit mode adds two classes and is stricter about e-mail:

  identity-email an author or committer e-mail outside {*@users.noreply.github.com,
                 noreply@anthropic.com, noreply@github.com}; the same short list is
                 the only one accepted for e-mail addresses inside commit messages
  hostname       a LAN-style host name (<name>.local, .lan, .home, .internal, ...)

No mode ever prints a matched value. Tree findings are reported as
<path>:<line>: [<class>] <redacted> (id <finding id>), where any path segment that itself
matches a rule is replaced by <redacted> and line 0 means the path itself matched.
Commit findings are <sha12> <field>[:line]: [<class>] <redacted>. Error messages and
echoed arguments are redacted the same way. Inspect the source to see what matched.

Placeholders used by redacted receipts (<repo>/, <home>/, <scratch>/,
<reference-host>) never match. Exit status: 0 clean, 1 findings, 2 error (fail closed).
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
    ('agent-scratch', re.compile(r'\.(?:claude|codex|gemini)/(?:jobs|projects|sessions|archived_sessions|tmp|'
                                 r'todos|shell-snapshots|worktrees|plans|file-history)/[^\s/\'"<>()\[\]]*'
                                 r'|(?:~|\$HOME|\$\{HOME\})/\.(?:claude|codex)\b'
                                 r'|/tmp/claude(?:-\d+)?/[^\s\'"<>()\[\]]*|/tmp/claude-\d+'
                                 r'|\brollout-\d{4}-\d\d-\d\dT[0-9-]+-[0-9a-f-]{36}\.jsonl')),
    ('email', re.compile(r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b')),
    ('host-field', re.compile(r'"(?:host|hostname)"\s*:\s*"([^"]*)"')),
    # The whole link is matched, including the identifier after the prefix (up to the end
    # of that path segment), so redaction never leaves the payload behind.
    ('agent-session', re.compile(
        r'(?i)(?:claude\.ai/(?:code/)?(?:sessions?|chats?|share|projects?|artifacts?|conversations?)[/_]|chatgpt\.com/(?:codex/tasks|c|share|g)/'
        r'|chat\.openai\.com/(?:c|share)/|gemini\.google\.com/(?:app|share)/)[^\s/\'"<>()\[\]]*'
        r'|\bsession_[A-Za-z0-9]{16,}'
        r'|claude\.ai/[^\s\'"<>()\[\]]*[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')),
    # Claude session ids and Codex thread ids are canonical UUIDs (v4 and v7).
    ('agent-uuid', re.compile(r'(?i)(?<![0-9a-z-])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                              r'(?![0-9a-z-])')),
    # A session trailer is matched with its whole value (rest of the line).
    ('session-trailer', re.compile(
        r'(?i)^\s*(?:claude|codex|chatgpt|openai|gemini|agent)[-_ ]?(?:session|task|chat|conversation)'
        r'(?:[-_ ]?(?:id|url|link))?\s*:.*$')),
]
# Output policy: no diagnostic ever prints a matched value. Findings are reported as
# <safe location>:<line>: [<class>] <redacted> (id <finding id>); a path segment that itself
# matches a rule is replaced by <redacted>. Error messages and echoed arguments pass
# through redact_text() as well. Inspect the source itself to see what matched.
_RUNTIME_EXTRA: set = set()

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


def redact_text(text: str, extra=None) -> str:
    """Replace every span any rule would flag (and every word containing a denied token)."""
    extra = _RUNTIME_EXTRA if extra is None else extra
    for cls, rx in METADATA_PATTERNS:
        def _sub(m, cls=cls):
            if cls == 'email' and any(a.match(m.group(0)) for a in EMAIL_ALLOW):
                return m.group(0)
            if cls == 'host-field' and HOST_FIELD_OK.match(m.group(1)):
                return m.group(0)
            return '<redacted>'
        text = rx.sub(_sub, text)
    return WORD.sub(lambda m: '<redacted>' if next(_denied_spans(m.group(0), extra), None) else m.group(0),
                    text)


def safe_location(rel: str) -> str:
    """The path with any segment (or cross-segment span) that matches a rule redacted."""
    red = redact_text('/' + rel)
    return red[1:] if red.startswith('/') else red


def finding_id(rel: str, lineno: int, cls: str, k: int) -> str:
    """Stable identifier of a finding; derived from its location, never from the matched value."""
    return hashlib.sha256(f'{rel}\0{lineno}\0{cls}\0{k}'.encode('utf-8', 'surrogateescape')).hexdigest()[:10]


def scan_path(rel, rules, extra):
    """Findings in a file's path itself (reported at line 0)."""
    return [(r, 0, c, h) for r, _, c, h in scan_text(rel, '/' + rel, rules, extra)]


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
            raise GuardError(f'{path.name}:{n}: expected "<class> <path-glob> <regex>"')
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
                findings.append((rel, lineno, cls, hit))
        for w in WORD.finditer(line):
            for span in _denied_spans(w.group(0), extra):
                if not allowed(rules, 'denied-token', rel, span):
                    findings.append((rel, lineno, 'denied-token', span))
                break
    return findings


class GuardError(RuntimeError):
    """A condition under which the guard cannot vouch for anything: it fails closed (exit 2)."""


def _git(root: Path, *args, input: bytes | None = None) -> bytes:
    try:
        r = subprocess.run(['git', '-C', str(root), *args], capture_output=True, input=input)
    except OSError as exc:
        raise GuardError(f'cannot run git: {exc}') from exc
    if r.returncode != 0:
        msg = r.stderr.decode('utf-8', 'replace').strip().splitlines()
        raise GuardError(f'git {args[0]} failed: {msg[-1] if msg else "exit " + str(r.returncode)}')
    return r.stdout


def read_objects(root: Path, shas):
    """Return {sha: (type, bytes)} for the given object ids; a missing object is an error."""
    shas = list(dict.fromkeys(shas))
    if not shas:
        return {}
    out = _git(root, 'cat-file', '--batch', input=('\n'.join(shas) + '\n').encode())
    objs, pos = {}, 0
    for sha in shas:
        nl = out.index(b'\n', pos)
        header = out[pos:nl].decode('utf-8', 'replace').split()
        if len(header) != 3 or header[0] != sha:
            raise GuardError(f'object {sha[:12]} is missing from the repository')
        size = int(header[2])
        objs[sha] = (header[1], out[nl + 1:nl + 1 + size])
        pos = nl + 1 + size + 1
    return objs


# ---------------------------------------------------------------- commit metadata

IDENT = re.compile(r'^(?P<name>.*) <(?P<email>[^<>]*)> -?\d+ [+-]\d{4}$')
SKIP_HEADERS = {'tree', 'parent', 'object', 'type', 'encoding', 'gpgsig', 'gpgsig-sha256'}


def parse_object(sha: str, otype: str, raw: bytes):
    """Split a raw commit or tag object into named metadata fields.

    Identity headers (author, committer, tagger) become <role>-name / <role>-email;
    the message is 'message'; any other header text except signatures and object
    links (e.g. a mergetag, which embeds a whole tag with its tagger) is 'header'.
    Malformed objects are errors, never silently skipped."""
    text = raw.decode('utf-8', 'replace')
    head, _, message = text.partition('\n\n')
    if not head.startswith('tree ' if otype == 'commit' else 'object '):
        raise GuardError(f'{otype} {sha[:12]} cannot be parsed')
    fields, extra_headers, current = {}, [], None
    for line in head.split('\n'):
        if line.startswith(' '):                         # continuation of the previous header
            if current not in SKIP_HEADERS:
                extra_headers.append(line[1:])
            continue
        key, _, value = line.partition(' ')
        current = key
        if key in ('author', 'committer', 'tagger'):
            m = IDENT.match(value)
            if not m:
                raise GuardError(f'{otype} {sha[:12]} has a malformed {key} line')
            fields[f'{key}-name'] = m.group('name')
            fields[f'{key}-email'] = m.group('email')
        elif key not in SKIP_HEADERS:
            extra_headers.append(line)
    if otype == 'commit' and not ('author-email' in fields and 'committer-email' in fields):
        raise GuardError(f'commit {sha[:12]} lacks an author or committer')
    if extra_headers:
        fields['header'] = '\n'.join(extra_headers)
    fields['message'] = message
    return fields


def scan_metadata(sha: str, fields: dict, extra):
    """Findings (sha, field, line, class) for one object's metadata. No allow-list applies."""
    findings = []
    for field, value in fields.items():
        rel = f'commit:{sha}:{field}'
        if field.endswith('-email'):
            v = value.strip()
            if not any(a.match(v) for a in METADATA_EMAIL_ALLOW):
                findings.append((sha, field, 0, 'identity-email'))
            hits = [h for h in scan_text(rel, v, [], extra, METADATA_PATTERNS, METADATA_EMAIL_ALLOW)
                    if h[2] != 'email']                   # covered by identity-email
        else:
            hits = scan_text(rel, value, [], extra, METADATA_PATTERNS, METADATA_EMAIL_ALLOW)
        findings += [(sha, field, lineno, cls) for _, lineno, cls, _ in hits]
    return findings


# Frozen historical exception manifest. It names, by full SHA, the commits that were
# already published when this guard was introduced and that fail a metadata rule, with
# the rule classes each one fails. Owner ruling 2026-10-05 ("no rewrite"): this is
# acknowledged, immutable existing exposure, NOT approval. It exempts only the listed
# classes of the listed commits' metadata; it never exempts any tree. The file is pinned
# by this digest, so changing it means changing this line too, in review.
COMMIT_EXCEPTIONS_FILE = 'scripts/private_infra_commit_exceptions.txt'
COMMIT_EXCEPTIONS_SHA256 = '668fffd0c16e6c32ac20a259402f1d80ad3469f62c82709a3230b8b0f13d47d2'
_SHA_RX = re.compile(r'^[0-9a-f]{40}(?:[0-9a-f]{24})?$')
_CLASSES = {c for c, _ in METADATA_PATTERNS} | {'identity-email', 'denied-token'}


def load_commit_exceptions(path: Path, expected_sha256: str | None):
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GuardError(f'commit exception manifest unreadable: {path.name}: {exc.strerror}') from exc
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise GuardError(f'commit exception manifest {path.name} does not match the digest pinned '
                         f'in check_private_infra.py; it is frozen and changes need owner review')
    out = {}
    for n, line in enumerate(data.decode('utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        bits = line.split()
        classes = set(bits[1].split(',')) if len(bits) == 2 else set()
        if len(bits) != 2 or not _SHA_RX.match(bits[0]) or not classes or not classes <= _CLASSES \
                or bits[0] in out:
            raise GuardError(f'{path.name}:{n}: malformed entry (expected "<full sha> <class>[,<class>]")')
        out[bits[0]] = classes
    return out


_REF_SELECTORS = {'--all': [], '--tags': ['refs/tags'], '--branches': ['refs/heads'],
                  '--remotes': ['refs/remotes']}
_TAG_DEPTH_LIMIT = 64


def _start_points(root: Path, rev_args):
    """Every object a revision argument list can name directly (positive or negative side).

    Ref-selection options are expanded to the refs they select; any other selection option
    that could name refs is refused rather than left unscanned (fail closed)."""
    pts = []
    for tok in rev_args:
        if tok in _REF_SELECTORS:
            pts += _git(root, 'for-each-ref', '--format=%(objectname)', *_REF_SELECTORS[tok]).decode().split()
        elif tok.startswith(('--glob', '--tags=', '--branches=', '--remotes=', '--bisect', '--reflog',
                             '--stdin', '--alternate-refs', '--indexed-objects')):
            raise GuardError('revision option not supported by the tag scan; name the refs explicitly')
        elif tok.startswith('-'):
            continue                                   # --not, --exclude=..., ordering flags
        else:
            for part in re.split(r'\.\.\.?', tok.lstrip('^')):
                if part:
                    pts.append(_git(root, 'rev-parse', '--verify', '--end-of-options', part).decode().strip())
    return list(dict.fromkeys(pts))


def _tag_objects(root: Path, rev_args):
    """Every annotated tag object reachable from the arguments by following tag targets.

    A pushed outer tag also publishes any tag it points at, so tag chains are walked to the
    end. Missing or malformed tag objects, and chains deeper than a sane limit, fail closed."""
    tags, seen = [], set()
    for start in _start_points(root, rev_args):
        cur, depth = start, 0
        while cur not in seen:
            otype = _git(root, 'cat-file', '-t', cur).decode().strip()
            if otype != 'tag':
                break
            seen.add(cur)
            tags.append(cur)
            raw = read_objects(root, [cur])[cur][1].decode('utf-8', 'replace')
            first = raw.split('\n', 1)[0].split(' ')
            if len(first) != 2 or first[0] != 'object' or not _SHA_RX.match(first[1]):
                raise GuardError(f'tag {cur[:12]} has a malformed target')
            cur, depth = first[1], depth + 1
            if depth > _TAG_DEPTH_LIMIT:
                raise GuardError(f'tag chain from {start[:12]} is deeper than {_TAG_DEPTH_LIMIT}')
    return tags


def scan_commits(root: Path, rev_args, extra, exceptions):
    """Scan metadata of every commit in the range (and of named annotated tags).

    Returns (n_objects, findings, acknowledged) where acknowledged counts findings
    matched by the frozen historical exception manifest."""
    tags = _tag_objects(root, rev_args)                 # every reachable tag object first
    shas = _git(root, 'rev-list', *rev_args, '--').decode().split()
    objs = read_objects(root, tags + shas)
    findings, acknowledged = [], 0
    for sha in tags + shas:
        otype, raw = objs[sha]
        if otype not in ('commit', 'tag'):
            raise GuardError(f'object {sha[:12]} is a {otype}, not a commit or tag')
        for f in scan_metadata(sha, parse_object(sha, otype, raw), extra):
            if f[3] in exceptions.get(sha, ()):
                acknowledged += 1
            else:
                findings.append(f)
    return len(shas) + len(tags), findings, acknowledged


def main_commits(root: Path, spec: str, extra, exceptions, emit_exceptions=False):
    rev_args = shlex.split(spec)
    if not rev_args:
        raise GuardError('--commits needs a revision or range, e.g. HEAD or origin/master..HEAD')
    n, findings, acknowledged = scan_commits(root, rev_args, extra, exceptions)
    if emit_exceptions:                                   # manifest body for a reviewed rebuild
        by_sha = {}
        for sha, _, _, cls in findings:
            by_sha.setdefault(sha, set()).add(cls)
        for sha in sorted(by_sha):
            print(f'{sha} {",".join(sorted(by_sha[sha]))}')
        return 0
    print(f'[audit] scanning metadata of {n} object(s) in {redact_text(spec)!r}')
    for sha, field, lineno, cls in findings:
        where = f'{field}:{lineno}' if field in ('message', 'header') else field
        print(f'{sha[:12]} {where}: [{cls}] <redacted>')
    if acknowledged:
        print(f'[audit] {acknowledged} finding(s) belong to commits in the frozen historical exception '
              f'manifest ({COMMIT_EXCEPTIONS_FILE}): acknowledged published exposure, not approval.')
    if findings:
        bad = len({f[0] for f in findings})
        print(f'[audit] FAILED: {len(findings)} private-data finding(s) in {bad} commit(s). Reword the '
              f'message (git commit --amend / git rebase -i) and set user.email to a noreply address '
              f'before pushing. Published history is not rewritten; see docs/OWNER_DECISIONS.md.')
        return 1
    print('[audit] PASSED: no unexempted metadata findings.')
    return 0


# ---------------------------------------------------------------- tree of a revision

# Frozen publication boundary: every commit already published at the cutoff. Their trees
# are not rescanned when they occur in a pushed history; every other commit's tree is.
PUBLICATION_BOUNDARY_FILE = 'scripts/private_infra_published_boundary.txt'
PUBLICATION_BOUNDARY_SHA256 = '27ef1b6f41088314ae3d32ba2418a3d05dc5a8804f77683a248394b77a65ca19'


def load_publication_boundary(path: Path, expected_sha256: str | None):
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GuardError(f'publication boundary unreadable: {path.name}: {exc.strerror}') from exc
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise GuardError(f'publication boundary {path.name} does not match the digest pinned in '
                         f'check_private_infra.py; it is frozen and changes need owner review')
    out = set()
    for n, line in enumerate(data.decode('utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if not _SHA_RX.match(line) or line in out:
            raise GuardError(f'{path.name}:{n}: malformed entry (expected one full commit SHA)')
        out.add(line)
    return out


def _tree_entries(root: Path, commit: str):
    listing = _git(root, 'ls-tree', '-r', '-z', '--full-tree', commit).decode('utf-8', 'surrogateescape')
    entries, allow_blob = [], None
    for rec in listing.split('\0'):
        if not rec:
            continue
        meta, _, path = rec.partition('\t')
        mode, otype, sha = meta.split()
        if path == 'scripts/private_infra_allowlist.txt' and otype == 'blob':
            allow_blob = sha
        if otype == 'blob' and mode != '120000' and path not in SELF_FILES:
            entries.append((path, sha))
    return entries, allow_blob


def scan_revision_tree(root: Path, rev: str, rules_path: str | None, extra, boundary):
    """Scan what pushing REV publishes: the tip's tree, plus the tree of every commit in its
    ancestry that is outside the frozen publication boundary (new history), so a value added
    in one new commit and removed in the next is still caught.

    Each (path, blob, allow-list) is scanned once. Returns (tip, n_commits, n_files, findings)
    where findings are (commit, path, line, class, hit) with commit None for the tip."""
    tip = _git(root, 'rev-parse', '--verify', '--end-of-options', f'{rev}^{{commit}}').decode().strip()
    ancestry = _git(root, 'rev-list', tip, '--').decode().split()
    targets = [tip] + [c for c in ancestry if c not in boundary and c != tip]
    rules_cache, done, findings, files = {}, set(), [], 0
    for commit in targets:
        entries, allow_blob = _tree_entries(root, commit)
        if rules_path:
            rules = rules_cache.setdefault('<cli>', load_allowlist(Path(rules_path)))
        elif allow_blob:      # the allow-list committed in that same revision, as CI would see it
            if allow_blob not in rules_cache:
                text = read_objects(root, [allow_blob])[allow_blob][1].decode('utf-8')
                rules_cache[allow_blob] = _rules_from_text(text, 'private_infra_allowlist.txt')
            rules = rules_cache[allow_blob]
        else:
            rules = []
        todo = [(p, b) for p, b in entries if (p, b, allow_blob) not in done]
        objs = read_objects(root, [b for _, b in todo])
        for path, blob in todo:
            done.add((path, blob, allow_blob))
            files += 1
            hits = scan_path(path, rules, extra)
            data = objs[blob][1]
            if b'\0' not in data[:8192]:
                hits += scan_text(path, data.decode('utf-8', 'replace'), rules, extra)
            findings += [(None if commit == tip else commit, *h) for h in hits]
    return tip, len(targets), files, findings


def _rules_from_text(text: str, name: str):
    rules = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        bits = line.split(None, 2)
        if len(bits) != 3:
            raise GuardError(f'{name}:{n}: expected "<class> <path-glob> <regex>"')
        cls, glob, rx = bits
        rules.append((cls, glob, re.compile(rx), f'{name}:{n}'))
    return rules


def _report_tree(findings, label):
    seen = {}
    for f in findings:
        commit, (rel, lineno, cls, _hit) = (f[0], f[1:]) if len(f) == 5 else (None, f)
        k = seen[(commit, rel, lineno, cls)] = seen.get((commit, rel, lineno, cls), -1) + 1
        where = f'{safe_location(rel)}:{lineno}'
        if commit:
            where = f'history {commit[:12]} {where}'
        print(f'{where}: [{cls}] <redacted> (id {finding_id(rel, lineno, cls, k)})')
    if findings:
        print(f'[audit] FAILED: {len(findings)} private-data finding(s) in {label}. Redact them, or add a '
              f'narrow entry to scripts/private_infra_allowlist.txt if the match is a genuine exception.')
        return 1
    print('[audit] PASSED: no personal or private-infrastructure data found.')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=None, help='tree or repository to scan (default: repository root)')
    ap.add_argument('--no-git', action='store_true', help='walk the directory even inside a git checkout')
    ap.add_argument('--allowlist', default=None, help='allow-list file (default: scripts/private_infra_allowlist.txt under the root)')
    ap.add_argument('--commits', metavar='RANGE', default=None,
                    help='scan commit metadata (message, author, committer) of a revision or range '
                         'instead of the tree, e.g. HEAD (full ancestry) or origin/master..HEAD')
    ap.add_argument('--tree-rev', metavar='REV', default=None,
                    help='scan the committed tree of REV (what a push publishes) instead of the checkout')
    ap.add_argument('--commit-exceptions', metavar='FILE', default=None,
                    help=f'frozen historical exception manifest (default: {COMMIT_EXCEPTIONS_FILE} next to '
                         f'this script, digest-pinned)')
    ap.add_argument('--publication-boundary', metavar='FILE', default=None,
                    help=f'frozen list of already-published commits for --tree-rev (default: '
                         f'{PUBLICATION_BOUNDARY_FILE} next to this script, digest-pinned)')
    ap.add_argument('--no-commit-exceptions', action='store_true',
                    help='apply no historical exceptions (used to rebuild the manifest for review)')
    ap.add_argument('--emit-exceptions', action='store_true',
                    help='with --commits and --no-commit-exceptions: print manifest lines (sha classes)')
    args = ap.parse_args(argv)
    extra = {t.strip().lower() for t in os.environ.get('NEUROFLY_PRIVATE_TOKENS', '').split(',') if t.strip()}
    _RUNTIME_EXTRA.clear()
    _RUNTIME_EXTRA.update(extra)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    if not root.is_dir():
        print(f'[audit] no such directory: {redact_text(str(root))}', file=sys.stderr)
        return 2
    try:
        if args.commits is not None:
            if args.emit_exceptions and not args.no_commit_exceptions:
                raise GuardError('--emit-exceptions requires --no-commit-exceptions')
            if args.no_commit_exceptions:
                exceptions = {}
            elif args.commit_exceptions:
                exceptions = load_commit_exceptions(Path(args.commit_exceptions), None)
            else:
                exceptions = load_commit_exceptions(Path(__file__).resolve().parents[1] / COMMIT_EXCEPTIONS_FILE,
                                                    COMMIT_EXCEPTIONS_SHA256)
            return main_commits(root, args.commits, extra, exceptions, args.emit_exceptions)
        if args.tree_rev is not None:
            if args.publication_boundary:
                boundary = load_publication_boundary(Path(args.publication_boundary), None)
            else:
                boundary = load_publication_boundary(Path(__file__).resolve().parents[1] / PUBLICATION_BOUNDARY_FILE,
                                                     PUBLICATION_BOUNDARY_SHA256)
            commit, ncommits, n, findings = scan_revision_tree(root, args.tree_rev, args.allowlist, extra, boundary)
            print(f'[audit] scanning the tree of {commit[:12]} and of {ncommits - 1} other unpublished commit(s) '
                  f'in its history: {n} distinct file version(s) (revision mode)')
            return _report_tree(findings, f'the tree or new history of {commit[:12]}')
        allow_path = Path(args.allowlist) if args.allowlist else root / 'scripts' / 'private_infra_allowlist.txt'
        rules = load_allowlist(allow_path)
        files, mode = list_files(root, not args.no_git)
        print(f'[audit] scanning {len(files)} files under {redact_text(root.name)}/ ({mode} mode)')
        findings = []
        for rel in files:
            if rel in SELF_FILES:
                continue
            p = root / rel
            if p.is_symlink() or not p.is_file():
                continue
            findings += scan_path(rel, rules, extra)
            data = p.read_bytes()
            if b'\0' in data[:8192]:
                continue                               # binary
            findings += scan_text(rel, data.decode('utf-8', 'replace'), rules, extra)
        return _report_tree(findings, 'the tree')
    except (GuardError, ValueError, OSError, UnicodeDecodeError, re.error) as exc:
        print(f'[audit] ERROR (failing closed): {redact_text(str(exc))}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:                            # any scanner failure fails closed
        print(f'[audit] ERROR (failing closed): {type(exc).__name__}', file=sys.stderr)
        sys.exit(2)
