"""Actual file version controls legacy compatibility; current proofs fail closed."""
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from neurofly import recording as r
from tests.test_recording_streaming import _writer


MISSING = object()


def refused(writer):
    reason = r.recording_invalid_reason(writer.path)
    assert reason is not None
    with pytest.raises(ValueError): r.read_recording(writer.path)
    assert not r.list_recordings(writer.path.parent)
    with pytest.raises(ValueError): r.public_recording_artifacts(writer.path)
    artifacts = writer.artifacts()
    assert not artifacts['certified'] and not artifacts['completed']
    return reason


@pytest.mark.parametrize('protocol', [MISSING, None, False, True, 2.0, '2', 0, 1, 3, [], {}])
def test_v2_never_uses_legacy_fallback_for_invalid_sidecar_protocol(tmp_path, protocol):
    _, _, writer = _writer(tmp_path); writer.close()
    meta = json.loads(writer._sidecar_path.read_text())
    if protocol is MISSING: meta.pop('completion_protocol')
    else: meta['completion_protocol'] = protocol
    meta['projection']['source_evidence'][0]['source']['receipt']['durable'] = False
    writer._sidecar_path.write_text(json.dumps(meta)); writer._done_path.unlink()
    assert 'completion protocol' in refused(writer)


@pytest.mark.parametrize('protocol', [MISSING, None, False, 2.0, '2', 1, 3])
def test_done_requires_exact_supported_protocol_not_only_matching_hashes(tmp_path, protocol):
    _, _, writer = _writer(tmp_path); writer.close()
    done = json.loads(writer._done_path.read_text())
    if protocol is MISSING: done.pop('protocol')
    else: done['protocol'] = protocol
    writer._done_path.write_text(json.dumps(done))
    refused(writer)


def test_current_protocol_rejects_changed_exported_proof_with_stale_done(tmp_path):
    _, _, writer = _writer(tmp_path); writer.close()
    meta = json.loads(writer._sidecar_path.read_text())
    meta['projection']['source_evidence'][0]['source']['receipt']['durable'] = False
    writer._sidecar_path.write_text(json.dumps(meta))
    assert 'does not match' in refused(writer)


def retag(writer, version):
    lines = gzip.decompress(writer.path.read_bytes()).splitlines(keepends=True)
    header = json.loads(lines[0]); header['version'] = version
    writer.path.write_bytes(gzip.compress(r._line(header) + b''.join(lines[1:]), mtime=0))


@pytest.mark.parametrize('version', [None, True, 2.0, '2', 3])
def test_actual_header_rejects_unsupported_or_aliased_versions(tmp_path, version):
    _, _, writer = _writer(tmp_path); writer.close(); retag(writer, version)
    assert 'recording header' in refused(writer)


def test_retagged_v1_cannot_use_legacy_fallback_with_modern_projection_sidecar(tmp_path):
    _, _, writer = _writer(tmp_path); writer.close(); retag(writer, 1)
    meta = json.loads(writer._sidecar_path.read_text()); meta.pop('completion_protocol')
    writer._sidecar_path.write_text(json.dumps(meta)); writer._done_path.unlink()
    assert 'completion protocol' in refused(writer)


@pytest.mark.parametrize('protocol', [MISSING, None, 2])
def test_genuine_historical_v1_is_readable_with_appropriate_completion_rules(tmp_path, protocol):
    path = tmp_path / 'legacy.nfrec'
    frame = r._line({'k': 'f', 'i': 0, 'step': 0})
    with gzip.open(path, 'wb') as handle:
        handle.write(r._line({'k': 'header', 'format': r.FORMAT, 'version': 1}))
        handle.write(frame)
        handle.write(r._line({'k': 'end', 'frames': 1, 'events': 0,
                             'frames_sha256': hashlib.sha256(frame).hexdigest()}))
    sha = r._file_digest(path)
    meta = {'recording': {'status': 'complete', 'sha256': sha}}
    if protocol is not MISSING: meta['completion_protocol'] = protocol
    sidecar = path.with_name(path.name + '.json'); sidecar.write_text(json.dumps(meta))
    if protocol == 2:
        path.with_name(path.name + r.DONE_SUFFIX).write_text(json.dumps({
            'name': path.name, 'protocol': 2, 'sha256': sha,
            'sidecar_sha256': r._file_digest(sidecar)}))
    assert r.recording_invalid_reason(path) is None
    assert r.read_recording(path)['header']['version'] == 1
    assert [x['name'] for x in r.list_recordings(tmp_path)] == [path.name]
    assert path in r.public_recording_artifacts(path)


@pytest.mark.parametrize('state', ['pending writer', 'revoked writer'])
def test_in_process_owner_hook_is_authoritative_for_all_public_verdicts(tmp_path, monkeypatch, state):
    _, _, writer = _writer(tmp_path); writer.close()
    calls = []
    def owner(path):
        calls.append(path); return state
    monkeypatch.setattr(r, 'recording_writer_invalid_reason', owner, raising=False)
    assert refused(writer) == state
    assert calls and all(path == writer.path for path in calls)


def test_certifier_does_not_read_entire_recording_bytes(tmp_path, monkeypatch):
    _, _, writer = _writer(tmp_path); writer.close()
    original = Path.read_bytes
    def read(path):
        assert path != writer.path, 'certifier buffered the whole NF recording'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    assert r.recording_invalid_reason(writer.path) is None
