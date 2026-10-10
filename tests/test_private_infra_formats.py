"""The release-hygiene guard (scripts/check_private_infra.py) beyond plain text files:
image metadata, other binaries, archives, the secret / overlay-net / chat-id / transcript
classes and the read-only --github mode.

Every fixture is synthetic and generated under tmp_path. Every planted value is fake and
assembled from pieces so that this file itself passes the tree guard; denied tokens are
registered by hash exactly as the real ones are.
"""
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / 'scripts' / 'check_private_infra.py'
J = ''.join

HOME_LEAK = J(['/ho', 'me/alice/flybrain/run.json'])
EMAIL_LEAK = J(['alice.smith', '@', 'corp-mail.com'])
SESSION_LEAK = J(['https://', 'claude', '.ai/', 'code/', 'session', '_', 'Wq3' * 8])
DENIED = J(['build', '-box-', '7'])
NEEDLES = ('alice', 'corp-mail', 'Wq3Wq3', DENIED)

SECRETS = {
    'github-token': J(['gh', 'p_', 'Ab1' * 12]),
    'github-pat': J(['github', '_pat_', 'Fake0' * 6]),
    'openai-key': J(['s', 'k-', 'proj-', 'Fake9' * 5]),
    'aws-key': J(['AK', 'IA', 'FAKEFAKEFAKE1234']),
    'slack-token': J(['xo', 'xb-', '1234567890-', 'fakefake']),
    'pem-block': J(['-----BEGIN ', 'OPENSSH PRI', 'VATE KEY-----']),
    'telegram-bot': J(['123456789', ':', 'AAF', 'a1B2c3' * 5, 'xy']),
    'google-oauth': J(['1234567', '-', 'a1' * 16, '.apps.goo', 'gleusercontent.com']),
}
OVERLAY = {
    'tailscale-ip': J(['ping ', '100', '.101.7.9']),
    'tailscale-ip-edge': J(['ssh ', '100', '.127.255.1']),
    'ts-net-name': J(['https://my-box.tail1234', '.ts', '.net/']),
    'zerotier-join': J(['zerotier', '-cli join ', '8056c2e21c', '000001']),
    'zt-network': J(['zt', ' network: ', 'a0cbf4b62a', '1234ab']),
}
CHAT_IDS = {
    'snake': J(['chat', '_id = ', '-100', '1234567890']),
    'json': J(['{"chat', '_id": ', '"987654321"}']),
    'env': J(['TELEGRAM', '_CHAT=', '123456789012']),
    'words': J(['send to chat', ' id ', '-987654321']),
}
TRANSCRIPTS = {
    'claude-jsonl': J(['{"parent', 'Uuid": null, "type": "user"}']),
    'role-session': J(['{"ro', 'le": "assistant", "session', '_id": "x", "content": "hi"}']),
    'codex-rollout': J(['{"ty', 'pe": "session', '_meta", "payload": {}}']),
    'human-assistant': J(['intro\n', 'Hu', 'man: hello\n', 'Assis', 'tant: hi there']),
}
CLEAN = '\n'.join([
    J(['messages = [{"ro', 'le": "user", "content": "hi"}]']),        # an API example, no session
    'chat_id = 42 and id = 1234567890 and token = abc',
    J(['ping ', '100', '.63.1.2 and ', '100', '.128.0.1 (outside the CGNAT range)']),
    J(['Tailscale uses ', '100', '.64.0.0/10']),
    'Human: one line without an answer',
    'desk-123 and a sk-short key and AKIA-prefixed words',
])


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


def _tree(tmp_path, files):
    root = tmp_path / 'tree'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(REPO / 'scripts' / 'private_infra_allowlist.txt', root / 'scripts')
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else (data + '\n').encode())
    return root


def _run(guard, root, capsys):
    rc = guard.main(['--root', str(root), '--no-git'])
    out = capsys.readouterr()
    for needle in NEEDLES:
        assert needle not in out.out + out.err, needle
    return rc, out.out + out.err


# --- synthetic file formats ----------------------------------------------------------------

def _png(chunks):
    def chunk(ctype, body):
        return struct.pack('>I', len(body)) + ctype + body + struct.pack('>I', zlib.crc32(ctype + body))
    ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + b''.join(chunk(t, b) for t, b in chunks)
            + chunk(b'IDAT', zlib.compress(b'\0\0')) + chunk(b'IEND', b''))


def _png_text(kind, text):
    raw = text.encode()
    if kind == 'tEXt':
        return _png([(b'tEXt', b'Comment\0' + raw)])
    if kind == 'zTXt':
        return _png([(b'zTXt', b'Comment\0\0' + zlib.compress(raw))])
    return _png([(b'iTXt', b'Comment\0\1\0en\0\0' + zlib.compress(raw))])


def _jpeg(segments):
    out = b'\xff\xd8'
    for marker, body in segments:
        out += bytes([0xFF, marker]) + struct.pack('>H', len(body) + 2) + body
    return out + b'\xff\xda\x00\x02' + bytes(range(256)) * 4 + b'\xff\xd9'


def _exif(text):
    return b'Exif\0\0II*\0\x08\0\0\0\x01\0\x3b\x01\x02\0' + text.encode() + b'\0\0\0\0\0'


def _glb(doc):
    js = json.dumps(doc).encode()
    js += b' ' * (-len(js) % 4)
    body = struct.pack('<I4s', len(js), b'JSON') + js + struct.pack('<I4s', 4, b'BIN\0') + b'\0\0\0\0'
    return b'glTF' + struct.pack('<II', 2, 12 + len(body)) + body


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _targz(members, uname='builder'):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.uname, info.gname = len(data), uname, 'users'
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _binary(*strings):
    return b'\0\x01\x02\x03' + b'\0\xff\0'.join(s.encode() for s in strings) + b'\0\x07' * 64


@pytest.mark.parametrize('kind', ['tEXt', 'zTXt', 'iTXt'])
def test_png_text_chunks_are_scanned(guard, tmp_path, capsys, kind):
    root = _tree(tmp_path, {'docs/shot.png': _png_text(kind, f'saved from {HOME_LEAK}\nby {EMAIL_LEAK}')})
    rc, out = _run(guard, root, capsys)
    assert rc == 1
    assert 'docs/shot.png:1: [home-path] <redacted>' in out and 'docs/shot.png:2: [email] <redacted>' in out, out


@pytest.mark.parametrize('segment', ['exif', 'xmp', 'com'])
def test_jpeg_metadata_segments_are_scanned(guard, tmp_path, capsys, segment):
    seg = {'exif': (0xE1, _exif(f'Artist {EMAIL_LEAK}')),
           'xmp': (0xE1, b'http://ns.adobe.com/xap/1.0/\0<x:xmpmeta><dc:source>' + SESSION_LEAK.encode()
                   + b'</dc:source></x:xmpmeta>'),
           'com': (0xFE, f'edited in {HOME_LEAK}'.encode())}[segment]
    root = _tree(tmp_path, {'docs/photo.jpg': _jpeg([(0xE0, b'JFIF\0\1\1\0\0\1\0\1\0\0'), seg])})
    rc, out = _run(guard, root, capsys)
    assert rc == 1
    assert re.search(r'docs/photo\.jpg:\d+: \[(email|agent-session|home-path)\] <redacted>', out), out


def test_glb_json_chunk_is_scanned(guard, tmp_path, capsys):
    doc = {'asset': {'version': '2.0', 'generator': 'x', 'extras': {'source': HOME_LEAK}}}
    root = _tree(tmp_path, {'assets/fly.glb': _glb(doc)})
    rc, out = _run(guard, root, capsys)
    assert rc == 1 and 'assets/fly.glb:1: [home-path] <redacted>' in out, out


def test_other_binaries_are_scanned_as_printable_runs(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'data/blob.bin': _binary('header', f'cache={HOME_LEAK}', SECRETS['aws-key'],
                                                      f'by {EMAIL_LEAK}')})
    rc, out = _run(guard, root, capsys)
    assert rc == 1
    for cls in ('home-path', 'secret', 'email'):
        assert f'data/blob.bin:' in out and f'[{cls}] <redacted>' in out, (cls, out)


def test_binary_ip_rules_need_a_delimited_address(guard, tmp_path, capsys):
    ip = J(['10', '.2.0.1'])
    root = _tree(tmp_path, {'lib/noise.so': _binary(f'libfoo version {ip} build', J(['a', '@', 'b.cd noise']),
                                                    J(['100', '.64.3.7 rev'])),
                            'docs/clean.md': CLEAN})
    rc, out = _run(guard, root, capsys)
    assert rc == 0, out
    for delimited in (f'http://{ip}:8080/x', ip, f'host={ip}'):
        root = _tree(tmp_path / delimited.replace('/', '_'), {'lib/conf.so': _binary('header', delimited)})
        rc, out = _run(guard, root, capsys)
        assert rc == 1 and '[private-ip] <redacted>' in out, (delimited, out)


# --- archives -----------------------------------------------------------------------------

def test_zip_and_wheel_members_are_scanned(guard, tmp_path, capsys):
    root = _tree(tmp_path, {
        'dist/pkg.whl': _zip({'pkg/__init__.py': f'DATA = "{HOME_LEAK}"\n',
                              'pkg-1.0.dist-info/METADATA': 'Name: pkg\n'}),
        'dist/bundle.zip': _zip({f'notes/{DENIED}.txt': 'clean\n',
                                 'shot.png': _png_text('zTXt', f'mail {EMAIL_LEAK}')}),
    })
    rc, out = _run(guard, root, capsys)
    assert rc == 1
    assert 'dist/pkg.whl!pkg/__init__.py:1: [home-path] <redacted>' in out, out
    assert 'dist/bundle.zip!notes/<redacted>:0: [denied-token] <redacted>' in out, out
    assert 'dist/bundle.zip!shot.png:1: [email] <redacted>' in out, out


def test_tar_gz_members_names_and_owner_are_scanned(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'dist/src.tar.gz': _targz({'src/run.sh': f'cd {HOME_LEAK}\n'.encode()},
                                                      uname=DENIED)})
    rc, out = _run(guard, root, capsys)
    assert rc == 1
    assert 'dist/src.tar.gz!src/run.sh:1: [home-path] <redacted>' in out, out
    assert 'dist/src.tar.gz!src/run.sh:0: [denied-token] <redacted>' in out, out


def test_archive_nesting_is_followed_two_levels_and_fails_closed_beyond(guard, tmp_path, capsys):
    inner = _zip({'leak.txt': HOME_LEAK})
    root = _tree(tmp_path / 'two', {'dist/outer.zip': _zip({'inner.zip': inner})})
    rc, out = _run(guard, root, capsys)
    assert rc == 1 and 'dist/outer.zip!inner.zip!leak.txt:1: [home-path] <redacted>' in out, out
    root = _tree(tmp_path / 'three', {'dist/outer.zip': _zip({'mid.zip': _zip({'inner.zip': inner})})})
    rc, out = _run(guard, root, capsys)
    assert rc == 2 and 'failing closed' in out, out


@pytest.mark.parametrize('name,data', [
    ('broken.zip', b'PK\x03\x04' + b'\x00garbage' * 20),
    ('broken.png', _png_text('tEXt', 'ok')[:40]),
    ('badz.png', _png([(b'zTXt', b'Comment\0\0not-zlib-data')])),
    ('broken.tar.gz', b'\x1f\x8b\x08\x00' + b'\x00' * 30),
    ('broken.jpg', b'\xff\xd8\xff\xe1\x7f\xff' + b'Exif'),
])
def test_unreadable_binary_input_fails_closed(guard, tmp_path, capsys, name, data):
    root = _tree(tmp_path, {f'data/{name}': data})
    rc, out = _run(guard, root, capsys)
    assert rc == 2 and 'failing closed' in out, out


def test_decompression_caps_fail_closed(guard, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(guard, 'MAX_DECOMPRESSED', 1 << 12)
    root = _tree(tmp_path, {'dist/big.zip': _zip({'zeros.bin': b'\0' * (1 << 16)})})
    rc, out = _run(guard, root, capsys)
    assert rc == 2 and 'failing closed' in out, out


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_revision_mode_scans_binaries_and_archives(guard, tmp_path, capsys):
    root = _tree(tmp_path, {'docs/shot.png': _png_text('iTXt', SESSION_LEAK),
                            'dist/pkg.whl': _zip({'pkg/x.py': f'P = "{HOME_LEAK}"\n'})})
    git = ['git', '-C', str(root), '-c', 'user.name=t', '-c', 'user.email=t' + '@example.com',
           '-c', 'commit.gpgsign=false']
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(git + ['add', '-A'], check=True)
    subprocess.run(git + ['commit', '-qm', 'x'], check=True)
    empty = tmp_path / 'boundary.txt'
    empty.write_text('')
    rc = guard.main(['--root', str(root), '--tree-rev', 'HEAD', '--publication-boundary', str(empty)])
    out = capsys.readouterr().out
    assert rc == 1
    assert 'docs/shot.png:1: [agent-session] <redacted>' in out and 'dist/pkg.whl!pkg/x.py:1: [home-path]' in out, out


# --- new text classes ---------------------------------------------------------------------

def test_clean_text_passes(guard, tmp_path, capsys):
    rc, out = _run(guard, _tree(tmp_path, {'docs/clean.md': CLEAN}), capsys)
    assert rc == 0, out


@pytest.mark.parametrize('cls,cases', [('secret', SECRETS), ('overlay-net', OVERLAY), ('chat-id', CHAT_IDS)])
def test_new_text_classes_are_caught_and_redacted(tmp_path, cls, cases):
    files = {f'docs/{name}.txt': f'value: {text}' for name, text in cases.items()}
    root = _tree(tmp_path, {'docs/clean.md': CLEAN, **files})
    r = subprocess.run([sys.executable, str(GUARD), '--root', str(root), '--no-git'], capture_output=True,
                       text=True, env={'PATH': '/usr/bin:/bin'})
    assert r.returncode == 1
    for name, text in cases.items():
        assert f'docs/{name}.txt:1: [{cls}] <redacted> (id ' in r.stdout, (name, r.stdout)
        assert text not in r.stdout + r.stderr
    assert 'docs/clean.md' not in r.stdout


@pytest.mark.parametrize('name', sorted(TRANSCRIPTS))
def test_transcripts_are_flagged_once_per_file(guard, tmp_path, capsys, name):
    body = '\n'.join([TRANSCRIPTS[name]] * 3)
    rc, out = _run(guard, _tree(tmp_path, {'logs/t.jsonl': body, 'docs/clean.md': CLEAN}), capsys)
    assert rc == 1
    assert out.count('[transcript]') == 1 and 'logs/t.jsonl:' in out, out


def test_commit_metadata_catches_a_secret(guard, tmp_path, capsys):
    repo = tmp_path / 'repo'
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_AUTHOR_NAME='Some One', GIT_AUTHOR_EMAIL='someone' + '@users.noreply.github.com',
               GIT_COMMITTER_NAME='Some One', GIT_COMMITTER_EMAIL='someone' + '@users.noreply.github.com')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True, env=env)
    subprocess.run(['git', '-C', str(repo), 'commit', '-q', '--allow-empty', '--no-gpg-sign', '-m',
                    f'ci: rotate\n\ntoken {SECRETS["github-token"]}'], check=True, env=env)
    rc = guard.main(['--root', str(repo), '--commits', 'HEAD'])
    out = capsys.readouterr().out
    assert rc == 1 and 'message:3: [secret] <redacted>' in out and SECRETS['github-token'] not in out, out


# --- --github mode (gh mocked; no network) ------------------------------------------------

FAKE_GH = '''#!{python}
import os, re, sys
d = os.environ['FAKE_GH_DIR']
with open(os.path.join(d, 'calls.log'), 'a') as fh:
    fh.write(' '.join(sys.argv[1:]) + '\\n')
if os.environ.get('FAKE_GH_FAIL'):
    sys.stderr.write('HTTP 404: Not Found\\n')
    sys.exit(1)
p = os.path.join(d, re.sub(r'[^A-Za-z0-9]+', '_', sys.argv[-1]) + '.json')
sys.stdout.write(open(p).read() if os.path.exists(p) else '[]')
'''


@pytest.fixture
def fake_gh(tmp_path, monkeypatch):
    bindir, data = tmp_path / 'bin', tmp_path / 'gh-data'
    bindir.mkdir()
    data.mkdir()
    gh = bindir / 'gh'
    gh.write_text(FAKE_GH.format(python=sys.executable))
    gh.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ.get("PATH", "")}')
    monkeypatch.setenv('FAKE_GH_DIR', str(data))

    def put(endpoint, payload, pages=1):
        name = re.sub(r'[^A-Za-z0-9]+', '_', endpoint) + '.json'
        if pages == 1:
            text = json.dumps(payload)
        else:                                   # gh --paginate concatenates one array per page
            text = ''.join(json.dumps(payload[i::pages]) for i in range(pages))
        (data / name).write_text(text)
    return put, data


def _github_fixture(put, leaky=True):
    b = 'repos/acme/widgets'
    leak = (lambda s: s) if leaky else (lambda s: 'clean text')
    put(b, {'description': leak(f'mirror of {HOME_LEAK}'), 'homepage': 'https://example.org'})
    put(f'{b}/pulls?state=all&per_page=100', [
        {'number': 7, 'title': 'feat: x', 'body': leak(f'Session: {SESSION_LEAK}')},
        {'number': 8, 'title': 'fix: y', 'body': 'clean'}], pages=2)
    put(f'{b}/issues?state=all&per_page=100', [
        {'number': 7, 'title': 'feat: x', 'body': 'dup of the PR', 'pull_request': {}},
        {'number': 9, 'title': 'bug', 'body': leak(f'contact {EMAIL_LEAK}')}])
    put(f'{b}/issues/comments?per_page=100', [
        {'issue_url': 'https://api.github.com/repos/acme/widgets/issues/9', 'body': leak(CHAT_IDS['snake'])},
        {'issue_url': 'https://api.github.com/repos/acme/widgets/issues/8', 'body': leak(SECRETS['slack-token'])}])
    put(f'{b}/pulls/comments?per_page=100', [
        {'pull_request_url': 'https://api.github.com/repos/acme/widgets/pulls/8',
         'body': leak(OVERLAY['tailscale-ip'])}])
    put(f'{b}/pulls/8/reviews?per_page=100', [{'body': leak(TRANSCRIPTS['human-assistant'])}])
    put(f'{b}/releases?per_page=100', [{'id': 55, 'name': 'v1', 'tag_name': 'v1', 'body': 'notes',
                                        'assets': [{'name': leak(f'build-{DENIED}.tar.gz'), 'label': None}]}])


def test_github_mode_scans_every_kind_of_text(guard, fake_gh, capsys):
    put, data = fake_gh
    _github_fixture(put)
    rc = guard.main(['--github', 'acme/widgets'])
    out = capsys.readouterr().out
    assert rc == 1
    for line in ('github repo #0: [home-path] <redacted>', 'github pr #7: [agent-session] <redacted>',
                 'github issue #9: [email] <redacted>', 'github issue-comment #9: [chat-id] <redacted>',
                 'github pr-comment #8: [secret] <redacted>', 'github review-comment #8: [overlay-net] <redacted>',
                 'github review #8: [transcript] <redacted>', 'github release-asset #55: [denied-token] <redacted>'):
        assert line in out, (line, out)
    assert 'github issue #7' not in out                      # PRs are not re-scanned as issues
    for needle in NEEDLES:
        assert needle not in out
    calls = (data / 'calls.log').read_text().splitlines()
    assert calls and all(c.startswith('api --method GET ') for c in calls), calls


def test_github_mode_passes_clean_text(guard, fake_gh, capsys):
    put, _ = fake_gh
    _github_fixture(put, leaky=False)
    rc = guard.main(['--github', 'acme/widgets'])
    out = capsys.readouterr().out
    assert rc == 0 and 'PASSED' in out, out


def test_github_mode_fails_closed(guard, fake_gh, capsys, monkeypatch):
    put, _ = fake_gh
    _github_fixture(put, leaky=False)
    assert guard.main(['--github', 'not a repo']) == 2
    put('repos/acme/widgets/releases?per_page=100', '{not json')
    assert guard.main(['--github', 'acme/widgets']) == 2
    monkeypatch.setenv('FAKE_GH_FAIL', '1')
    assert guard.main(['--github', 'acme/widgets']) == 2
    assert 'failing closed' in capsys.readouterr().err
