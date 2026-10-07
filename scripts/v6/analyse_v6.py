"""Per-stage transfer readout for a v6 run (right eye, COMPLETE columns) and the preregistered verdict.
  python scripts/v6/analyse_v6.py --contract C --contract-sha256 SHA --phase1 DIR --run DIR"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

GROUP = {'T4a': 'T4', 'T4b': 'T4', 'T4c': 'T4', 'T4d': 'T4', 'T5a': 'T5', 'T5b': 'T5', 'T5c': 'T5', 'T5d': 'T5'}
ap = argparse.ArgumentParser()
ap.add_argument('--contract', type=Path, required=True); ap.add_argument('--contract-sha256', required=True)
ap.add_argument('--phase1', type=Path, required=True); ap.add_argument('--run', type=Path, required=True)
a = ap.parse_args()
if hashlib.sha256(a.contract.read_bytes()).hexdigest() != a.contract_sha256:
    raise SystemExit('contract sha mismatch')
C = json.loads(a.contract.read_text())
cc = np.load(a.phase1 / 'cell_columns.npz')
grp = np.array([GROUP.get(t, t) for t in cc['cell_type']])
sel_base = (cc['eye'] == 'R') & (cc['column_state'] == 'COMPLETE')
out = {}
for cond in ('C1_fullfield_ON', 'C2_fullfield_OFF'):
    d = np.load(a.run / f'{cond}.npz')
    assert (d['node'] == cc['node']).all()
    row = {}
    for g in sorted(set(grp[sel_base])):
        m = sel_base & (grp == g)
        e, s = float(d['dV_E'][m].mean()), float(d['dV_S'][m].mean())
        row[g] = dict(dV_E=round(e, 4), dV_S=round(s, 4), resp=round(e if abs(e) >= abs(s) else s, 4),
                      V_B=round(float(d['V_B'][m].mean()), 3), n=int(m.sum()))
    out[cond] = row
checks = {}
for name, rule in C['pass_criteria'].items():
    r = out['C1_fullfield_ON'][rule['type']]['resp']
    ok = (r <= -rule['min_abs_mV']) if rule['sign'] == '-' else (r >= rule['min_abs_mV'])
    checks[name] = dict(resp=r, ok=bool(ok))
fals = [t for t in C['falsified_if_below_0p5_mV'] if abs(out['C1_fullfield_ON'][t]['resp']) < 0.5]
verdict = 'PASS' if all(c['ok'] for c in checks.values()) else 'FAIL'
res = dict(contract_sha256=a.contract_sha256, readout=out, checks=checks, falsified_by=fals, verdict=verdict)
(a.run / 'analysis.json').write_text(json.dumps(res, indent=1))
for g, v in out['C1_fullfield_ON'].items():
    print(f"{g:6s} ON resp {v['resp']:+8.3f} (E {v['dV_E']:+.3f} S {v['dV_S']:+.3f}) V_B {v['V_B']:.2f} | OFF {out['C2_fullfield_OFF'][g]['resp']:+.3f}")
print(json.dumps(dict(checks=checks, falsified_by=fals, verdict=verdict)))
