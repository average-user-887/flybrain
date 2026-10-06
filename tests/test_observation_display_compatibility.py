"""Check browser transport against current Python output, not JS-made numbers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from observation_envelopes import validate_observation_envelope

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures' / 'observation_display_all14.json'


def test_committed_python_fixtures_have_valid_envelopes_and_actual_matching_digests():
    data = json.loads(FIXTURES.read_text())
    assert len(data['fixtures']) == 14
    for case in data['fixtures'] + [data['deadlineFixture']]:
        if case['observation'] is not None:
            assert validate_observation_envelope(case['observation']) == case['observation']
        terminal = case['observationPublication']['last_terminal']
        envelope = validate_observation_envelope(terminal['observation'])
        digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert terminal['payload_sha256'] == digest == terminal['receipt']['payload_sha256']
        assert terminal['receipt']['durable'] is True
        assert terminal['receipt']['observation_key'] == terminal['observation_key']


def test_fresh_all14_python_producers_and_real_recorder_receipts_pass_node(tmp_path):
    node = shutil.which('node')
    assert node, 'Node required for browser-helper compatibility verification'
    generator_path = ROOT / 'tests' / 'fixtures' / 'generate_observation_display.py'
    spec = importlib.util.spec_from_file_location('observation_display_fixture_generator', generator_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    fresh = tmp_path / 'fixtures.json'
    fresh.write_text(json.dumps(generator.generate(), allow_nan=False))
    env = dict(os.environ, NEUROFLY_DISPLAY_FIXTURES=str(fresh))
    result = subprocess.run([node, '--test', str(ROOT / 'tests' / 'test_observation_display.js'),
                             str(ROOT / 'tests' / 'test_observation_renderer.js')],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_daemon_wire_packets_render_all14_without_identity_reconstruction(tmp_path):
    from neurofly_daemon import ContinuousExperimentRunner, PARADIGMS

    node = shutil.which('node')
    assert node, 'Node required for actual daemon wire compatibility'
    packets = []
    for assay in PARADIGMS:
        runner = ContinuousExperimentRunner(initial_paradigm=assay,
            output_dir=tmp_path / assay, continuous=True, checkpoint_interval=1e9)
        with runner.lock:
            packets.append(runner._assemble_telemetry({}))
            runner.step_once(publish=False)
            packets.append(runner._assemble_telemetry(runner._last_step_result))
    source = tmp_path / 'daemon-packets.json'
    source.write_text(json.dumps(packets, allow_nan=False))
    script = '''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const renderer = require('./web/observation_renderer.js');
const packets = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
assert.equal(packets.length, 28);
for (const packet of packets) {
    const original = JSON.stringify(packet);
    const display = renderer.view(packet, {connected: true, assay: packet.paradigm});
    assert.equal(display.live.state, 'available', JSON.stringify({assay: packet.paradigm, display}));
    assert.equal(display.currentIdentity.brain_id, packet.brain_id);
    assert.equal(JSON.stringify(packet), original);
}
'''
    result = subprocess.run([node, '-e', script, str(source)], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
