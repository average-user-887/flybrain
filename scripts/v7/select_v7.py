"""v7: apply the prereg v2 decision rule (fit_v7.outcome) to the per-start fit reports and, on
TRAIN_PASS, copy the selected start's parameter file to the frozen location.

  python scripts/v7/select_v7.py --fit-dir DIR --n-starts 4 --out qualification/v7/params_v7.json
A start without fit_start<k>.json (killed at the cap or crashed) is not completed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_v7 as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-dir', type=Path, required=True)
    ap.add_argument('--n-starts', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    reps = []
    for k in range(a.n_starts):
        f = a.fit_dir / f'fit_start{k}.json'
        reps.append(json.loads(f.read_text()) if f.exists() else dict(start=k, status='not completed'))
    status, best = F.outcome(reps, a.n_starts)
    sel = dict(outcome=status, selected_start=None if best is None else best['start'],
               starts={r['start']: dict(status=r.get('status'), cost=r.get('cost'),
                                        TRAIN_PASS=(r.get('gates') or {}).get('TRAIN_PASS')) for r in reps})
    if best is not None:
        src = a.fit_dir / f'params_v7_start{best["start"]}.json'
        shutil.copyfile(src, a.out)
        sel['params_file'] = str(a.out)
        sel['params_sha256'] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    (a.fit_dir / 'selection.json').write_text(json.dumps(sel, indent=1, sort_keys=True))
    print(json.dumps(sel, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
