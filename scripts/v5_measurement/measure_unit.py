"""Predeclared measurement (a) a1/a2 of docs/LIF_DYNAMICS_SPEC.md §9.9. CPU, synthetic.

a1: one spike from a presynaptic spiking cell of each receptor class onto a passive
    graded target: peak, time-to-peak, decay tau (log-linear fit from the peak's 20 %
    point onward ... restricted to the tail after the peak), PSP area.
a2: a graded presynaptic cell of each class stepped +4 mV (drive) for 50 ms: the
    target's synaptic conductance time course (sum of its channels), 10-90 % rise of
    the step-on response and the time to fall to 1/e after the step ends.
"""
import json

import numpy as np

from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain

DT = 0.1


def chain(w):
    return dict(ptr=np.array([0, 1, 1], np.int64), post=np.array([1], np.int32),
                weight=np.array([w], np.float32), ids=np.array([1, 2], np.int64))


def a1(cls, kin):
    w = 4.0 if cls == 'nicotinic' else -4.0
    b = Brain(arrays=chain(w), dynamics='v5', backend='cpu', graded_policy=np.array([0, 1], np.uint8),
              receptor_classes=np.array([cls, 'nicotinic'], object), kinetics=kin)
    d = np.zeros(2, np.float32)
    d[0] = 200.0
    b.step(d, 2.0)
    assert b.total_spikes == 1
    tr, gt = [], []
    for _ in range(3000):
        b.step(np.zeros(2, np.float32), DT)
        tr.append(float(b.v[1]) + 52.0)
        gt.append(float(b.g[:, 1].sum()))
    tr, gt = np.array(tr), np.array(gt)
    ip = int(np.argmax(np.abs(tr)))
    gp = int(np.argmax(gt))
    # conductance decay tau: log-linear fit on the conductance after its peak
    tail = gt[gp:gp + 400]
    good = tail > tail[0] * 0.05
    slope = np.polyfit(np.arange(good.sum()) * DT, np.log(tail[good]), 1)[0]
    return dict(psp_peak_mV=float(tr[ip]), psp_time_to_peak_ms=(ip + 1) * DT,
                psp_area_mV_ms=float(tr.sum() * DT), conductance_peak=float(gt[gp]),
                conductance_decay_tau_fit_ms=float(-1.0 / slope),
                conductance_area=float(gt.sum() * DT))


def a2(cls, kin):
    w = 4.0 if cls == 'nicotinic' else -4.0
    b = Brain(arrays=chain(w), dynamics='v5', backend='cpu', graded_policy=np.array([1, 1], np.uint8),
              receptor_classes=np.array([cls, 'nicotinic'], object), kinetics=kin)
    z = np.zeros(2, np.float32)
    for _ in range(2000):
        b.step(z, DT)
    g0 = float(b.g[:, 1].sum())
    d = np.zeros(2, np.float32)
    d[0] = 4.0
    on, off = [], []
    for _ in range(500):
        b.step(d, DT)
        on.append(float(b.g[:, 1].sum()) - g0)
    for _ in range(1000):
        b.step(z, DT)
        off.append(float(b.g[:, 1].sum()) - g0)
    on, off = np.array(on), np.array(off)
    top = on[-1]
    t10 = int(np.argmax(on >= 0.1 * top)) * DT
    t90 = int(np.argmax(on >= 0.9 * top)) * DT
    t_e = int(np.argmax(off <= off[0] / np.e)) * DT
    return dict(baseline_conductance=g0, step_on_10_90_ms=t90 - t10, step_on_final=float(top),
                step_off_time_to_1_over_e_ms=t_e)


out = dict(spec='docs/LIF_DYNAMICS_SPEC.md#9.9', tables={k: dict(v) for k, v in rk.KINETICS.items()},
           a1={}, a2={})
for kin in (rk.KINETICS_PRIMARY, rk.KINETICS_UPPER, rk.KINETICS_V4_EQUIVALENT):
    out['a1'][kin] = {c: a1(c, kin) for c in rk.CLASSES}
    out['a2'][kin] = {c: a2(c, kin) for c in rk.CLASSES}
# F2: area within 5 % of v4, fitted tau equals declared
chk = {}
for kin in (rk.KINETICS_PRIMARY, rk.KINETICS_UPPER):
    for c in rk.CLASSES:
        r, ref = out['a1'][kin][c], out['a1'][rk.KINETICS_V4_EQUIVALENT][c]
        chk[f'{kin}/{c}'] = dict(
            declared_tau_ms=rk.KINETICS[kin][c], fitted_tau_ms=r['conductance_decay_tau_fit_ms'],
            psp_area_ratio_to_v4=r['psp_area_mV_ms'] / ref['psp_area_mV_ms'],
            conductance_area_ratio_to_v4=r['conductance_area'] / ref['conductance_area'],
            psp_peak_ratio_to_v4=r['psp_peak_mV'] / ref['psp_peak_mV'])
out['F2_check'] = chk
out['F2_triggered'] = bool(any(abs(v['psp_area_ratio_to_v4'] - 1) > 0.05
                               or abs(v['fitted_tau_ms'] / v['declared_tau_ms'] - 1) > 0.02
                               for v in chk.values()))
json.dump(out, open('docs/receipts/v5_raw/unit_level.json', 'w'), indent=1)
print(json.dumps(chk, indent=1))
print('F2 triggered:', out['F2_triggered'])
for kin in out['a2']:
    print(kin, {c: (round(v['step_on_10_90_ms'], 1), round(v['step_off_time_to_1_over_e_ms'], 1))
                for c, v in out['a2'][kin].items()})
