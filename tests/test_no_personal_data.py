"""Artifacts NeuroFly writes must not carry the machine's or the user's identity.

Receipts, run manifests, recordings, checkpoints and bundles are shared. They may
describe hardware (CPU/GPU model) and software versions, but never the hostname,
the account name, or an absolute path under the home directory or the checkout.
The comparison values are taken from this machine at test time.
"""
import getpass
import json
import platform
import re
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurofly import privacy  # noqa: E402
from neurofly.privacy import (host_description, portable_path, redact_local, redact_text,  # noqa: E402
                              restore_local)


def _identity_values():
    values = set()
    for getter in (platform.node, socket.gethostname, getpass.getuser):
        try:
            value = getter() or ''
        except Exception:  # noqa: BLE001
            value = ''
        if value:
            values.add(value)
            values.add(value.split('.')[0])         # short hostname of an FQDN
    return {v for v in values if len(v) >= 3}


def leaks(text: str) -> list:
    """Hostname / account-name tokens and absolute home or checkout paths in ``text``."""
    found = []
    for value in _identity_values():
        if re.search(r'(?<![A-Za-z0-9])' + re.escape(value) + r'(?![A-Za-z0-9])', text):
            found.append(f'identity token {value[:1]}***')
    for path in {str(Path.home()), str(ROOT), str(ROOT.resolve())}:
        if path and path != '/' and path in text:
            found.append('absolute path ' + path.replace(str(Path.home()), '~'))
    return found


def assert_clean(text: str, what: str):
    problems = leaks(text)
    assert not problems, f'{what} leaks: {problems}'


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_host_description_is_hardware_and_software_only():
    description = host_description()
    assert_clean(json.dumps(description), 'host_description()')
    assert description['platform'] == f'{platform.system()}-{platform.machine()}'
    assert description['cpu_count'] and description['python'] == platform.python_version()
    assert 'gpu' in description and 'cpu' in description and 'numpy' in description['libraries']
    for key in ('host', 'hostname', 'node', 'user', 'username'):
        assert key not in description
    # A copy each time: callers may annotate it without changing the next caller's.
    description['libraries']['numpy'] = 'changed'
    assert host_description()['libraries']['numpy'] != 'changed'


def test_redaction_uses_the_receipt_placeholders(tmp_path):
    home = Path.home()
    assert redact_text(str(ROOT / 'outputs' / 'x.json')) == '<repo>/outputs/x.json'
    assert redact_text(str(home / 'data' / 'graph.npz')).startswith('<home>/')
    assert redact_text('/tmp/run-1/receipt.json') == '<scratch>/run-1/receipt.json'
    assert redact_text('/media/' + 'someone/Storage/malecns_v1') == '<local-path>/malecns_v1'
    assert redact_text('/mnt/' + 'share/graphs/graph.npz') == '<local-path>/graph.npz'
    assert_clean(redact_text(str(tmp_path / 'out.json')), 'pytest tmp_path')
    # Values that are not paths are never rewritten.
    assert redact_text('R1-R6 at 20 units') == 'R1-R6 at 20 units'
    assert redact_local({'a': [str(ROOT), 3, None, (str(home),)], 'n': 1.5}) == \
        {'a': ['<repo>', 3, None, ('<home>',)], 'n': 1.5}
    assert portable_path(None) is None and portable_path('relative/path') == 'relative/path'
    # Placeholders map back on this machine; <local-path> cannot and means "default".
    assert restore_local('<repo>/outputs') == str(privacy.REPO_ROOT / 'outputs')
    assert restore_local('<local-path>/malecns_v1') is None
    assert restore_local(None) is None


def test_no_writer_reads_the_hostname_or_account_name():
    """Static check: only neurofly/privacy.py may read them, and only to remove them."""
    pattern = re.compile(r'platform\.node\(|gethostname\(|getpass\.getuser\(|getlogin\(|uname\(\)\.node'
                         r'|environ(?:\.get)?\(?\[?[\'"](?:HOSTNAME|USER|LOGNAME|USERNAME)[\'"]')
    offenders = []
    for path in ROOT.rglob('*.py'):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(('tests/', 'docs/', '.', 'outputs/', 'connectome_data/', 'build/')) \
                or '/.' in rel or rel == 'neurofly/privacy.py':
            continue
        for n, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            if pattern.search(line):
                offenders.append(f'{rel}:{n}')
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Each writer
# ---------------------------------------------------------------------------

def test_run_manifest_and_graph_identity(tmp_path):
    from brainlab.graph_identity import GraphIdentity, synthetic_test_graph
    from provenance import RunManifest, source_revision

    _, synthetic, _ = synthetic_test_graph()
    local = dict(synthetic.to_dict(), graph_path=str(Path.home() / 'graphs' / 'graph.npz'),
                 neuron_map_path=str(ROOT / 'connectome_data' / 'nodes.feather'))
    identity = GraphIdentity(**local)
    written = identity.to_dict()
    assert identity.graph_path == local['graph_path']          # in memory: still loadable
    assert written['graph_path'] == '<home>/graphs/graph.npz'
    assert written['neuron_map_path'] == '<repo>/connectome_data/nodes.feather'
    manifest = RunManifest.create(
        backend='connectome-fixed', assay='optomotor', instance_id='x', seed=1, graph=dict(local),
        dynamics={'dt_ms': 0.1}, learned_parameter_locations={'mb_weights': str(tmp_path / 'brain.json')},
        rng=np.random.default_rng(1), test_mode=True, source=source_revision())
    path = tmp_path / 'manifest.json'
    manifest.write(path)
    text = path.read_text()
    assert_clean(text, 'manifest.json')
    data = json.loads(text)
    assert isinstance(data['host'], dict) and data['host']['cpu_count']
    assert data['source']['root'] == '<repo>'
    # Older manifests carried a hostname string in "host"; they still load.
    data['host'] = '<reference-host>'
    path.write_text(json.dumps(data))
    assert RunManifest.read(path).host == '<reference-host>'


def test_run_recording_header_and_sidecar(tmp_path):
    from neurofly.recording import record_run
    record_run(paradigm='t-maze', out=tmp_path / 'a', steps=20)
    import gzip
    assert_clean(gzip.decompress((tmp_path / 'a.nfrec').read_bytes()).decode(), 'a.nfrec')
    sidecar = (tmp_path / 'a.nfrec.json').read_text()
    assert_clean(sidecar, 'a.nfrec.json')
    assert isinstance(json.loads(sidecar)['host'], dict)


def test_validation_receipt(tmp_path):
    from validation import harness
    specs = ROOT / 'validation' / 'specs'
    harness.run(specs / 'looming_gf_v3.json', tmp_path / 'out', synthetic=True, backend='cpu', seeds=[0],
                log=lambda *_: None)
    for name in ('receipt.json', 'trials.json'):
        assert_clean((tmp_path / 'out' / name).read_text(), name)
    assert isinstance(json.loads((tmp_path / 'out' / 'receipt.json').read_text())['host'], dict)


def test_benchmark_receipt(tmp_path):
    from scripts.benchmark import main
    out = tmp_path / 'bench.json'
    assert main(['--quick', '--stages', 'brain-synthetic', '--out', str(out)]) == 0
    text = out.read_text()
    assert_clean(text, 'benchmark receipt')
    host = json.loads(text)['host']
    assert 'hostname' not in host and host['description']['cpu_count']


def test_brainlab_run_record(tmp_path):
    from brainlab.runs import Run
    graph = tmp_path / 'graph.npz'
    np.savez(graph, ptr=np.array([0, 0], dtype=np.int64), post=np.array([], dtype=np.int32),
             weight=np.array([], dtype=np.float32), ids=np.array([1], dtype=np.int64))
    with Run(graph, {'experiment': 'privacy'}, 0, tmp_path / 'runs') as run:
        pass
    assert_clean((run.path / 'run.json').read_text(), 'run.json')


def test_daemon_checkpoint_file(tmp_path):
    from neurofly_daemon import ContinuousExperimentRunner
    runner = ContinuousExperimentRunner(initial_paradigm='t-maze', sim_speed=1, checkpoint_interval=3600,
                                        output_dir=tmp_path)
    with runner.lock:
        runner.step_once()
        saved = runner.save_checkpoint('privacy')
    assert_clean(saved.read_text(), 'daemon checkpoint')


def test_embodied_run_manifest_recording_and_replay_receipt(tmp_path):
    pytest.importorskip('flygym')
    import gzip
    from neurofly_body import cli
    out = tmp_path / 'run'
    assert cli.main(['run', '--controller', 'modular', '--duration', '0.1', '--output', str(out),
                     '--seed', '1']) == 0
    for name in ('manifest.json', 'summary.json'):
        assert_clean((out / name).read_text(), name)
    assert_clean(gzip.decompress((out / 'body.nfbody').read_bytes()).decode(), 'body.nfbody')
    assert cli.main(['replay-check', str(out), '--output', str(tmp_path / 'replay')]) == 0
    assert_clean((tmp_path / 'replay' / 'replay_check.json').read_text(), 'replay_check.json')
