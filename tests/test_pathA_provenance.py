"""Path A run provenance: stale or foreign result rows are refused, never mixed in."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pathA_provenance as prov  # noqa: E402

CONTRACT = ROOT / 'qualification/pathA/contract.json'
BASE = prov.expected('c' * 64, 'g' * 64, 'v3', 'cpu', 'e' * 64, 'a' * 40)


def row(seed=0, **override):
    p = prov.stamp(BASE, seed)
    p.update(override)
    return dict(condition='A1_JOCE_0', seed=seed, provenance=p)


def test_matching_row_accepted():
    prov.check_row(row(3), BASE)


@pytest.mark.parametrize('field', ['contract_sha256', 'graph_npz_sha256', 'dynamics', 'backend',
                                   'engine_sha256', 'code_sha'])
def test_any_field_mismatch_rejected(field):
    with pytest.raises(prov.ProvenanceError, match=field):
        prov.check_row(row(1, **{field: 'other'}), BASE)


def test_missing_provenance_rejected():
    with pytest.raises(prov.ProvenanceError, match='no provenance'):
        prov.check_row(dict(condition='A1_JOCE_0', seed=0), BASE)


def test_seed_mismatch_rejected():
    r = row(2)
    r['seed'] = 5
    with pytest.raises(prov.ProvenanceError, match='seed'):
        prov.check_row(r, BASE)


def test_engine_hash_covers_engine_files():
    assert len(prov.engine_sha256()) == 64


def _fake_runs(path: Path, contract: dict, csha: str, mutate=None):
    base = prov.expected(csha, contract['data']['graph_npz_sha256'], contract['protocol']['dynamics'],
                         contract['protocol']['backend'], 'e' * 64, 'a' * 40)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'runs.jsonl', 'w') as fh:
        for i, cond in enumerate(contract['conditions']):
            for s in contract['protocol']['seeds']:
                ro = {r: dict(mean_rate_hz=float(cond['rate_hz']), counts=[])
                      for r in contract['tests'][cond['test']]['readouts']}
                rec = dict(provenance=prov.stamp(base, s), condition=cond['id'], test=cond['test'], seed=s,
                           readouts=ro, activated_mean_rate_hz=1.0, firing_neurons=1)
                if mutate and i == 1 and s == 0:
                    mutate(rec)
                fh.write(json.dumps(rec) + '\n')


def _analyse(tmp_path, runs):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/pathA_analyse.py'), '--contract', str(CONTRACT),
                           '--runs', str(runs), '--out', str(tmp_path / 'an')], capture_output=True, text=True)


@pytest.fixture
def contract():
    return json.loads(CONTRACT.read_text()), hashlib.sha256(CONTRACT.read_bytes()).hexdigest()


def test_analyser_accepts_bound_rows(tmp_path, contract):
    _fake_runs(tmp_path / 'ok', *contract)
    res = _analyse(tmp_path, tmp_path / 'ok')
    assert res.returncode == 0, res.stderr
    assert json.loads((tmp_path / 'an/results.json').read_text())['run_identity']['code_sha'] == 'a' * 40


@pytest.mark.parametrize('mutate', [
    lambda r: r.pop('provenance'),
    lambda r: r['provenance'].update(contract_sha256='0' * 64),
    lambda r: r['provenance'].update(code_sha='b' * 40),
    lambda r: r['provenance'].update(engine_sha256='f' * 64),
    lambda r: r['provenance'].update(dynamics='v4'),
    lambda r: r['provenance'].update(seed=99),
], ids=['missing', 'contract', 'code', 'engine', 'dynamics', 'seed'])
def test_analyser_refuses_foreign_row(tmp_path, contract, mutate):
    _fake_runs(tmp_path / 'bad', *contract, mutate=mutate)
    res = _analyse(tmp_path, tmp_path / 'bad')
    assert res.returncode != 0
    assert 'refusing to analyse' in res.stderr
    assert not (tmp_path / 'an/results.json').exists()
