"""Print the preregistered verdict and the four condition summaries for each
confirmatory set of the declared E_inh sweep, side by side with the v3 primary.

Reads only results.json files; recomputes nothing, so it cannot silently
redefine the metric.  Usage:

    python scripts/einh_summarise_confirmatory.py <results.json> [<results.json> ...]

Ported from outputs/wp5/einh-sweep-20260926/summarise_confirmatory.py; only the
repository root is now resolved from scripts/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'outputs/wp5/v3-20260920/results.json'

CONDITIONS = ('intact', 'dna02_silenced', 'sham_no_input', 'shuffled_graph')


def label(d):
    ver = d.get('lif_dynamics_variant') or d.get('lif_dynamics_version')
    mode = (d.get('transmitter_policy') or {}).get('unclear_mode', 'excitatory')
    arm = 'A-primary' if mode == 'excitatory' else f'A-S1(unclear={mode})'
    e = d.get('lif_e_inh_mV')
    e = -70.0 if e is None else e
    return f'{arm} E_inh={e:g} [{ver}]'


def row(d, condition):
    s = (d.get('summary') or {}).get(condition, {}).get('TI')
    if s is None:
        return None
    return s


paths = [BASELINE] + [Path(p) for p in sys.argv[1:]]
docs = []
for p in paths:
    if not p.exists():
        print(f'MISSING {p}')
        continue
    docs.append((p, json.load(open(p))))

print('%-46s %-9s %s' % ('configuration', 'verdict', 'variant pin'))
for p, d in docs:
    print('%-46s %-9s %s' % (label(d), d.get('verdict'),
                             (d.get('lif_dynamics_variant_pin') or d.get('lif_dynamics_pin'))[:16]))

for condition in CONDITIONS:
    print(f'\n--- {condition}: TI mean [95% CI], seeds positive ---')
    for p, d in docs:
        s = row(d, condition)
        if s is None:
            print('%-46s  (absent)' % label(d))
            continue
        print('%-46s  %+.4f [%+.4f, %+.4f]  %d/%d positive  dz %s'
              % (label(d), s['mean'], s['ci95'][0], s['ci95'][1], s['n_positive'], s['n'],
                 'n/a' if s.get('dz') is None else f"{s['dz']:.2f}"))

print('\n--- paired contrasts (the preregistered causal clause) ---')
for p, d in docs:
    paired = d.get('paired') or {}
    for key, block in paired.items():
        v = (block or {}).get('TI')
        if not isinstance(v, dict) or 'mean' not in v:
            continue
        print('%-46s %-28s %+.4f [%+.4f, %+.4f]  %d/%d positive  dz %s'
              % (label(d), key + ' TI', v['mean'], v['ci95'][0], v['ci95'][1],
                 v.get('n_positive', -1), v.get('n', -1),
                 'n/a' if v.get('dz') is None else f"{v['dz']:.2f}"))

print('\n--- compute ---')
for p, d in docs:
    c = d.get('compute') or {}
    print('%-46s %s' % (label(d), json.dumps({k: c[k] for k in sorted(c)})[:180]))
