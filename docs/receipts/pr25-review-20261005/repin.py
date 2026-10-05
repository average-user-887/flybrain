"""Re-pin the WP7 specs' encoder source hash to the current T-maze spec (text edit only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path('<repo>')
OLD_SRC = '7eb6f62bf2c201d323dce826373b4b20dd25c702e3d4eccc7b364addd568be2a'
new_src = hashlib.sha256((ROOT / 'validation/specs/tmaze_odour_naive_v3.json').read_bytes()).hexdigest()


def frozen(spec):
    body = {k: v for k, v in spec.items() if k not in ('status', 'status_reason', 'freeze')}
    if 'calibration' in body:
        body['calibration'] = {k: v for k, v in body['calibration'].items() if k != 'result'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


for name in ('mb_e0_kc_regime_v3.json', 'mb_e1_hige2015_v3.json'):
    path = ROOT / 'validation/specs' / name
    text = path.read_text(encoding='utf-8')
    old_frozen = json.loads(text)['freeze']['frozen_content_sha256']
    assert text.count(OLD_SRC) == 1, name
    text = text.replace(OLD_SRC, new_src)
    new_frozen = frozen(json.loads(text))
    assert text.count(old_frozen) == 1
    text = text.replace(old_frozen, new_frozen)
    path.write_text(text, encoding='utf-8')
    print(name, 'source', OLD_SRC[:8], '->', new_src[:8], '| frozen', old_frozen, '->', new_frozen)
