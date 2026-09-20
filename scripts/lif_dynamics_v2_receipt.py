"""Assemble docs/receipts/lif_dynamics_v2.json (no simulation).

Puts the v1 and v2 runs of the SAME preregistered WP5 protocol side by side,
together with the minimal reproductions of the two defects.

    PYTHONPATH=. .venv/bin/python scripts/lif_dynamics_v2_receipt.py \
        outputs/wp5/confirm-20260919 outputs/wp5/confirm-v2-20260920 \
        docs/receipts/lif_dynamics_diagnosis.json docs/receipts/lif_dynamics_v2.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEEP = ('TI', 'TI_c1', 'TI_c05', 'A_DNa02', 'dV_DNa02', 'total_rate')
SIDES = ('DNa02_L_sPos', 'DNa02_R_sPos', 'DNa02_L_sNeg', 'DNa02_R_sNeg',
         'HS_L_sPos', 'HS_R_sPos', 'HS_L_sNeg', 'HS_R_sNeg',
         'H2_L_sPos', 'H2_R_sPos', 'H2_L_sNeg', 'H2_R_sNeg')


def condense(results):
    out = {}
    for condition, summary in results['summary'].items():
        row = {k: dict(mean=summary[k]['mean'], ci95=summary[k]['ci95'], n=summary[k]['n'],
                       n_positive=summary[k]['n_positive'], n_negative=summary[k]['n_negative'])
               for k in KEEP if k in summary}
        row['yaw_identically_zero_all_seeds'] = summary['yaw_identically_zero_all_seeds']
        row['dna02_spikes_total'] = summary['dna02_spikes_total']
        row['side_rates_mean_hz'] = {k: round(summary['side_rates_mean_hz'][k], 3)
                                     for k in SIDES if k in summary['side_rates_mean_hz']}
        out[condition] = row
    return out


def gray_baseline(results):
    """Mean network rate during the gray (no-stimulus) windows, intact condition."""
    rows = [s['gray_baseline'] for s in results['per_condition']['intact']]
    return dict(network_rate_hz=round(sum(r['total_rate'] for r in rows) / len(rows), 1),
                dna02_l_hz=round(sum(r['rate_l'] for r in rows) / len(rows), 3),
                dna02_r_hz=round(sum(r['rate_r'] for r in rows) / len(rows), 3),
                yaw_rad_s=round(sum(r['yaw'] for r in rows) / len(rows), 5))


def main(v1_dir: Path, v2_dir: Path, diagnosis: Path, out: Path):
    from brainlab.graph_identity import DYNAMICS_VERSIONS, dynamics_pin
    v1 = json.loads((v1_dir / 'results.json').read_text())
    v2 = json.loads((v2_dir / 'results.json').read_text())
    diag = json.loads(diagnosis.read_text())
    assert v1['prereg_sha256'] == v2['prereg_sha256'], 'the two runs must share one preregistration'
    assert v1['graph']['graph_sha256'] == v2['graph']['graph_sha256'], 'the two runs must share one graph'

    receipt = dict(
        schema='neurofly.lif-dynamics-v2-receipt.v1',
        spec='docs/LIF_DYNAMICS_SPEC.md',
        generator='scripts/lif_dynamics_diagnosis.py + scripts/wp5_optomotor.py --dynamics {v1,v2}; '
                  'assembled by scripts/lif_dynamics_v2_receipt.py',
        declared_dynamics={v: dict(DYNAMICS_VERSIONS[v], pin_sha256=dynamics_pin(v)) for v in ('v1', 'v2')},
        controller_versions=dict(v1='brainlab-lif-v1', v2='brainlab-lif-v2'),
        checkpoint_compatibility='v1 and v2 checkpoints are mutually refused: the synaptic state array g is '
                                 '(n,) under v1 and (2, n) under v2, and Brain.restore_state names both '
                                 'versions in the error. No silent reinterpretation is possible.',
        graph=v1['graph'],
        prereg=dict(path='docs/wp5_optomotor_prereg.json', sha256=v1['prereg_sha256'],
                    unchanged='schedule, seeds, conditions and the primary outcome TI are identical; '
                              'only the LIF engine differs'),
        diagnosis=dict(
            receipt='docs/receipts/lif_dynamics_diagnosis.json',
            probe_a_voltage_bound=diag['probe_a'],
            probe_b_self_sustained_state=diag['probe_b'],
            probe_c_real_graph=diag['probe_c'],
            dna02_static_input_budget=diag['probe_c'][0]['static_input_budget']),
        optomotor_rerun=dict(
            v1=dict(run=str(v1_dir.relative_to(ROOT)), verdict=v1['verdict'], summary=condense(v1),
                    gray_baseline_intact=gray_baseline(v1), paired=v1['paired'], compute=v1['compute'],
                    started_at=v1['started_at'], finished_at=v1['finished_at']),
            v2=dict(run=str(v2_dir.relative_to(ROOT)), verdict=v2['verdict'], summary=condense(v2),
                    gray_baseline_intact=gray_baseline(v2), paired=v2['paired'], compute=v2['compute'],
                    started_at=v2['started_at'], finished_at=v2['finished_at'])),
        host=v2['host'])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
