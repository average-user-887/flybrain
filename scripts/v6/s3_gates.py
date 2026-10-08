"""S3 gates from an analyse_tuning.py analysis.json (qualification/v6/S3_prereg.json).
  python scripts/v6/s3_gates.py ANALYSIS_JSON"""
import json, sys
r = json.load(open(sys.argv[1]))
D1 = r['D1']


def ok(k):
    v = D1.get(k)
    return v is not None and v['dsi'] is not None and v['dsi'] >= 0.2 and v['magnitude'] >= 0.5


def circ(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


pairs = []
for fam in ('T4', 'T5'):
    for x, y in (('a', 'b'), ('c', 'd')):
        for eye in ('L', 'R'):
            a, b = f'{fam}{x}_{eye}', f'{fam}{y}_{eye}'
            if a in D1 and b in D1:
                sep = circ(D1[a]['preferred_deg'], D1[b]['preferred_deg'])
                pairs.append(dict(pair=f'{a}/{b}', mag=(D1[a]['magnitude'], D1[b]['magnitude']),
                                  dsi=(D1[a]['dsi'], D1[b]['dsi']),
                                  pref=(D1[a]['preferred_deg'], D1[b]['preferred_deg']), sep_deg=sep,
                                  passed=bool(ok(a) and ok(b) and sep >= 135)))
g = r['gate']
out = dict(G1_verbatim_t4t5_clause=g['t4t5_clause'], G1_subtypes=g['t4t5_subtypes_passing'],
           G2_opposite_pair=any(p['passed'] for p in pairs), G2_pairs=pairs,
           G3_dna02_yaw=bool(g['dna02_clause']), G3_detail=r['D6'].get('yaw_clause'),
           max_t4t5_magnitude_mV=max(v['magnitude'] for k, v in D1.items() if k[:2] in ('T4', 'T5')))
out['EXPLORATORY_PASS'] = bool(out['G1_verbatim_t4t5_clause'] and out['G2_opposite_pair'] and out['G3_dna02_yaw'])
json.dump(out, open(sys.argv[1].replace('.analysis.json', '.s3_gates.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'G2_pairs'}, indent=1))
