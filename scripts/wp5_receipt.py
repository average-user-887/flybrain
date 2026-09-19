"""Assemble docs/receipts/wp5_optomotor.json from the WP5 run outputs (no simulation)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(run_dir: Path, structure: Path, out: Path):
    results = json.loads((run_dir / 'results.json').read_text())
    mirror_path = run_dir / 'exploratory_mirror.json'
    closed_path = run_dir / 'closed_loop.json'
    prereg = ROOT / 'docs/wp5_optomotor_prereg.json'
    keep = ('TI', 'TI_c1', 'TI_c05', 'A_DNa02', 'dV_DNa02', 'asym_HS', 'asym_H2', 'asym_DNa01', 'total_rate')
    summary = {c: {k: v for k, v in s.items() if k in keep or not isinstance(v, dict)}
               for c, s in results['summary'].items()}
    for c, s in results['summary'].items():
        summary[c]['side_rates_mean_hz'] = {k: v for k, v in s['side_rates_mean_hz'].items()
                                            if k.split('_')[0] in ('HS', 'H2', 'DNa02', 'DNa01')}
    receipt = dict(
        schema='neurofly.wp5-optomotor-receipt.v1',
        generator='scripts/wp5_optomotor.py (+ --exploratory-mirror, --closed-loop); assembled by scripts/wp5_receipt.py',
        verdict_by_preregistered_rule=results['verdict'],
        verdict_qualifiers=[
            'Effect is carried by leftward (s=+1) rotation: DNa02_L 21.3 Hz vs DNa02_R 0.2 Hz. Rightward rotation '
            'gives no net rightward DNa02 bias (L 4.0 Hz, R 3.7 Hz). Exploratory mirrored block order reproduces '
            'this, so it is not stimulus-history dependent.',
            'Contrast 0.5 gives no reliable effect (TI_c05 CI includes 0); contrast 1.0 carries the result.',
            'After the first stimulus the LIF proxy enters a self-sustained state (~1e6 spikes/s) that persists '
            'through gray periods; gray is not rest. Subthreshold v reaches about -200 mV (no reversal potentials), '
            'so dV_DNa02 is an engineering proxy, not a physiological potential.',
            'The yaw decoder is a fixed linear map; r(A_DNa02, yaw) ~ 1 is by construction. The causal evidence '
            'for motor dependence is DNa02 silencing (yaw identically 0) with the upstream HS/H2 response unchanged.',
            'Direction selectivity is supplied by the encoder (T4/T5 subtype choice), not computed by the graph.',
        ],
        prereg=dict(path='docs/wp5_optomotor_prereg.json', sha256=hashlib.sha256(prereg.read_bytes()).hexdigest(),
                    sha256_used_by_run=results['prereg_sha256'],
                    original_sha256_before_pilot_amendment='b05f48c6e7288168625038b6be4c2166ce1181783f48c31b6626b97e2eaa49c6'),
        graph=results['graph'], shuffled_control_graph=results.get('shuffled_identity'),
        io_map=results['io_map'],
        summary=summary, paired=results['paired'], motor_dependence=results['motor_dependence'],
        modular_baseline={k: v for k, v in results['modular_baseline'].items() if k != 'blocks'},
        compute=results['compute'],
        structural_evidence=json.loads(structure.read_text()),
        runs=[{k: r[k] for k in ('condition', 'seed', 'instance_id', 'run_id', 'wall_s', 'sim_ms')} for r in results['runs']],
        raw_outputs=str(run_dir.relative_to(ROOT)) + ' (ignored by git: results.json, trace-*.npz, registry/)',
        started_at=results['started_at'], finished_at=results['finished_at'], host=results['host'])
    if mirror_path.exists():
        mirror = json.loads(mirror_path.read_text())
        receipt['exploratory_mirror'] = dict(label=mirror['label'], blocks=mirror['blocks'], summary=mirror['summary'],
                                             per_seed_dna02_spikes={s['seed']: s['dna02_spikes'] for s in mirror['seeds']})
    if closed_path.exists():
        closed = json.loads(closed_path.read_text())
        receipt['conditional_secondary_closed_loop'] = closed['results']
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]))
