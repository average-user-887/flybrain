"""Assemble the PARTIAL docs/receipts/lif_dynamics_v2.json (no simulation).

Used while the v2 confirmatory set is still running.  Once
outputs/wp5/confirm-v2-20260920/results.json exists, use
scripts/lif_dynamics_v2_receipt.py instead, which writes the complete receipt.
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from brainlab.graph_identity import DYNAMICS_VERSIONS, dynamics_pin  # noqa: E402

prereg = json.loads((ROOT / 'docs/wp5_optomotor_prereg.json').read_text())
stim = prereg['stimulus']
blocks = stim['blocks']
step_s = stim['step_ms'] / 1000.
diag = json.loads((ROOT / 'docs/receipts/lif_dynamics_diagnosis.json').read_text())
v1res = json.loads((ROOT / 'outputs/wp5/confirm-20260919/results.json').read_text())


def seed_metrics(path):
    d = np.load(ROOT / path)
    rows = []
    for b, (s, c) in enumerate(blocks):
        m = d['block'] == b
        dur = m.sum() * step_s
        rows.append((s, c, float(d['yaw'][m].mean()), float(d['spk_l'][m].sum() / dur),
                     float(d['spk_r'][m].sum() / dur), float(d['total'][m].sum() / dur)))
    s = np.array([r[0] for r in rows], float)
    c = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows])
    rl = np.array([r[3] for r in rows])
    rr = np.array([r[4] for r in rows])
    g = d['block'] == -1
    gd = g.sum() * step_s
    return dict(TI=float((s * y).mean()), TI_c1=float((s * y)[c == 1].mean()),
                TI_c05=float((s * y)[c == .5].mean()), A_DNa02_hz=float((s * (rl - rr)).mean()),
                blocks_aligned_with_stimulus=int(((s * y) > 0).sum()), blocks=len(rows),
                DNa02_L_sPos=float(rl[s > 0].mean()), DNa02_R_sPos=float(rr[s > 0].mean()),
                DNa02_L_sNeg=float(rl[s < 0].mean()), DNa02_R_sNeg=float(rr[s < 0].mean()),
                network_rate_hz_stimulus=float(np.array([r[5] for r in rows]).mean()),
                network_rate_hz_gray=float(d['total'][g].sum() / gd),
                DNa02_L_hz_gray=float(d['spk_l'][g].sum() / gd),
                DNa02_R_hz_gray=float(d['spk_r'][g].sum() / gd),
                v_min_mV=float(min(d['v_l'].min(), d['v_r'].min())),
                v_max_mV=float(max(d['v_l'].max(), d['v_r'].max())))


receipt = dict(
    schema='neurofly.lif-dynamics-v2-receipt.v1',
    status='PARTIAL - diagnosis complete; the v2 confirmatory optomotor re-run was still in progress '
           '(1 of 24 condition-seed runs finished) when this receipt was written. Complete it with '
           'scripts/lif_dynamics_v2_receipt.py once outputs/wp5/confirm-v2-20260920/results.json exists.',
    spec='docs/LIF_DYNAMICS_SPEC.md',
    generator='scripts/lif_dynamics_diagnosis.py; scripts/wp5_optomotor.py --dynamics v2; '
              'assembled by scripts/_make_partial_receipt.py',
    generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    declared_dynamics={v: dict(DYNAMICS_VERSIONS[v], pin_sha256=dynamics_pin(v)) for v in ('v1', 'v2')},
    controller_versions=dict(v1='brainlab-lif-v1', v2='brainlab-lif-v2'),
    checkpoint_compatibility='v1 and v2 checkpoints are mutually refused: the synaptic state array g is (n,) '
                             'under v1 and (2, n) under v2, and Brain.restore_state names both versions in the '
                             'error. No silent reinterpretation is possible.',
    graph=v1res['graph'],
    prereg=dict(path='docs/wp5_optomotor_prereg.json', sha256=v1res['prereg_sha256'],
                unchanged='schedule, seeds, conditions and the primary outcome TI are identical; '
                          'only the LIF engine differs'),
    v1_reproducibility_check=dict(
        claim='the v1 code path is unchanged by this work',
        evidence='probe C under v1 reproduces outputs/wp5/confirm-20260919/trace-intact-seed0.npz block 0 '
                 'exactly: v_L -72.31 mV, v_R -184.24 mV, 23 DNa02_L spikes, 1144772 network spikes/s'),
    defect_1_runaway_and_unbounded_voltage=dict(
        probe_a_voltage_bound=diag['probe_a'],
        probe_b_self_sustained_state=diag['probe_b'],
        finding='v1 membrane potential scales without limit with inhibitory weight (-71 / -246 / -1992 mV for '
                'weights -40 / -400 / -4000). v2 asymptotes at the chloride reversal (-56.98 / -66.66 / '
                '-69.75 mV). The unbounded-voltage half of the defect is fixed. The runaway half is NOT: on '
                'the random recurrent probe v2 self-sustains at a LOWER recurrent gain than v1 (gain 8: v1 '
                '2.46 Hz/neuron, v2 67.9 Hz/neuron), which docs/LIF_DYNAMICS_SPEC.md section 3.3 item 4 '
                'predeclared as a possible outcome of the single-conductance-quantum calibration.'),
    defect_2_dead_right_side=dict(
        dna02_static_input_budget=diag['probe_c'][0]['static_input_budget'],
        anatomy_is_symmetric='DNa02_L 1138 in-edges, exc 4427.5 / inh 2160.7; DNa02_R 1201 in-edges, '
                             'exc 4389.8 / inh 2256.4 - within 2 percent',
        probe_c_real_graph=diag['probe_c'],
        finding='Confirmed dynamical, not anatomical. Under v1 DNa02_R is held at -184 mV (leftward) and -70 '
                'to -123 mV (rightward), far below any physiological chloride reversal, and emits 0 spikes '
                'in either direction. Under v2 the same neuron cannot be taken below -70 mV, sits at -51 mV '
                'and fires. The hypothesis in the task - that DNa02_R was being driven below a reversal '
                'floor a conductance-based model would prevent - is supported.'),
    high_conductance_fixed_point_analysis=dict(
        note='analysis of the declared model, not a parameter change',
        network_inhibitory_to_excitatory_weight_ratio=0.6185,
        realised_ratio_arriving_at_DNa02_under_v2=0.74,
        asymptotic_fixed_point_mV=-26.75, threshold_mV=-45.0,
        required_ratio_for_subthreshold_fixed_point=1.80, factor_short=2.91,
        consequence='With E_exc = 0 mV, E_inh = -70 mV and one conductance quantum per unit weight, the '
                    'MaleCNS graph under the coarse transmitter-sign proxy has a SUPRAthreshold '
                    'high-conductance fixed point (-26.75 mV against a -45 mV threshold), so any sustained '
                    'input drives the whole network to its refractory-limited rate. v1 masked this with '
                    'unbounded hyperpolarisation. The 0.275 mV/synapse scale was itself calibrated inside a '
                    'current-based model; carrying it into a conductance model without recalibrating the '
                    'overall synaptic gain is what leaves the graph suprathreshold.',
        not_adopted='The alternative calibration - matching the v1 IPSP instead of the v1 EPSP - would give '
                    'an inhibitory quantum 2.889x the excitatory one, within 1 percent of the 2.91x needed. '
                    'Recorded as an observation for a future declared v3; NOT adopted here, because choosing '
                    'it after seeing this result would be tuning.'),
    optomotor_rerun=dict(
        v1=dict(run='outputs/wp5/confirm-20260919', seeds=6, verdict=v1res['verdict'],
                TI=v1res['summary']['intact']['TI'], TI_c1=v1res['summary']['intact']['TI_c1'],
                TI_c05=v1res['summary']['intact']['TI_c05'],
                A_DNa02=v1res['summary']['intact']['A_DNa02'],
                total_rate=v1res['summary']['intact']['total_rate'],
                side_rates_mean_hz={k: v1res['summary']['intact']['side_rates_mean_hz'][k]
                                    for k in ('DNa02_L_sPos', 'DNa02_R_sPos', 'DNa02_L_sNeg', 'DNa02_R_sNeg')},
                compute=v1res['compute'],
                seed0=seed_metrics('outputs/wp5/confirm-20260919/trace-intact-seed0.npz')),
        v2=dict(run='outputs/wp5/confirm-v2-20260920', seeds_completed=1, seeds_planned=6,
                verdict='PENDING - confirmatory set incomplete',
                intact_wall_s_per_seed=427.4, sim_s_per_wall_s=0.0293,
                seed0=seed_metrics('outputs/wp5/confirm-v2-20260920/trace-intact-seed0.npz'),
                caveat='n=1 seed, no confidence interval, and no silenced / sham / shuffled controls yet. '
                       'This is NOT a verdict under the preregistered rule.')),
    host=v1res['host'])

out = ROOT / 'docs/receipts/lif_dynamics_v2.json'
out.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt['optomotor_rerun']['v2']['seed0'], indent=1))
print('written', out)
