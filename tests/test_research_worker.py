import json
import hashlib
import zipfile

from scripts.research_worker import cohort, bundle, PROTOCOL


def test_cohort_has_independent_matched_controls_and_retains_only_complete_memory(tmp_path):
    row = cohort(tmp_path, index=0, seeds=1, steps=10)
    assert row['status'] == 'complete'
    assert row['trained']['brain_id'] != row['frozen']['brain_id']
    assert row['trained']['evaluation_seed'] == row['frozen']['evaluation_seed']
    assert row['frozen']['weight_change_l2'] == 0
    for group in ('trained', 'frozen'):
        assert row[group]['training']['discontinuities'] == 0
        path = tmp_path / row['directory'] / f'{group}-evaluation.jsonl'
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        assert samples[0]['step'] == 0 and samples[-1]['step'] == 10
    # The next round restores identities/weights from this pair, in a fresh arena.
    next_row = cohort(tmp_path, index=14, seeds=1, steps=10, previous=row)
    assert next_row['trained']['brain_id'] == row['trained']['brain_id']
    assert next_row['frozen']['brain_id'] == row['frozen']['brain_id']
    assert next_row['trained']['before'] == row['trained']['after']


def test_shareable_bundle_has_verifiable_raw_data_and_protocol(tmp_path):
    row = cohort(tmp_path, index=1, seeds=1, steps=10)
    target = tmp_path / 'share.zip'
    bundle(tmp_path, [row], {'protocol':PROTOCOL}, target)
    with zipfile.ZipFile(target) as z:
        manifest = json.loads(z.read('manifest.json'))
        assert 'frozen' in manifest['protocol']
        assert len([name for name in z.namelist() if name.endswith('.jsonl')]) == 4
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
