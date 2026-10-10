"""v8: apply v8_input.outcome to the run reports; on TRAIN_PASS copy the selected run's params to
the frozen location.  A run without fit_run<k>.json (cap or crash) is not completed.
  python scripts/v8/select_v8.py --fit-dir DIR --n-runs N --out qualification/v8/params_v8.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8_input as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit-dir', type=Path, required=True); ap.add_argument('--n-runs', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    reps = []
    for k in range(a.n_runs):
        f = a.fit_dir / f'fit_run{k}.json'
        reps.append(json.loads(f.read_text()) if f.exists() else dict(run=k, status='not completed'))
    status, best = V.outcome(reps, a.n_runs)
    sel = dict(outcome=status, selected_run=None if best is None else best['run'],
               runs={r['run']: dict(status=r.get('status'), cost=r.get('cost'), dead_time_ms=r.get('dead_time_ms'),
                                    TRAIN_PASS=(r.get('gates') or {}).get('TRAIN_PASS')) for r in reps})
    if best is not None:
        shutil.copyfile(a.fit_dir / f'params_v8_run{best["run"]}.json', a.out)
        sel['params_sha256'] = hashlib.sha256(a.out.read_bytes()).hexdigest()
    (a.fit_dir / 'selection.json').write_text(json.dumps(sel, indent=1, sort_keys=True))
    print(json.dumps(sel, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
