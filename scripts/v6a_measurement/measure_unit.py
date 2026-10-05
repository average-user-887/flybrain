"""Predeclared measurement (a) a1/a2 of docs/LIF_DYNAMICS_SPEC.md §10.9. CPU, synthetic.

a1  small-signal DC transfer: a photoreceptor-class graded cell held at its anchor
    +- 0.05 mV by drive, one histaminergic (inhibitory) edge of the declared
    reference weight W_ref onto a passive default-class graded target with
    nothing else.  Gain = dV_post / dV_pre.  Declared: -measured gain under the
    calibrated table, -(v4 small-signal gain) under v5-linear; both kinetics arms.
a2  static transfer curve of the same pair: V_post at steady state for presynaptic
    potentials from -52 to -32 mV.
a3  dynamic transfer at the protocol frequency: the presynaptic cell driven by a
    1.5 Hz sinusoid of amplitude 0.1 mV (small signal) and 9.57 mV (the v5 R1-R6
    F1 amplitude under the grating), F1 of the target over 3 cycles after a 1 s
    settle.
a4  the same three probes for the DEFAULT class (a default-class presynaptic cell
    at V_rest onto an inhibitory and onto an excitatory target, weight W_ref).
"""
import json
import math

import numpy as np

from brainlab import graded_release as gr
from brainlab import receptor_kinetics as rk
from brainlab.brain import Brain

DT = 0.1


def pair(release, kinetics, pre_class, w, pre_receptor):
    arrays = dict(ptr=np.array([0, 1, 1], np.int64), post=np.array([1], np.int32),
                  weight=np.array([w], np.float32), ids=np.array([1, 2], np.int64))
    return Brain(arrays=arrays, dynamics='v6a', backend='cpu', graded_policy=np.array([1, 1], np.uint8),
                 kinetics=kinetics, receptor_classes=[pre_receptor, 'nicotinic'],
                 release=release, release_classes=[pre_class, gr.CLASS_DEFAULT])


def steady(release, kinetics, pre_class, w, pre_receptor, pre_drive):
    b = pair(release, kinetics, pre_class, w, pre_receptor)
    d = np.array([pre_drive, 0.0], np.float32)
    for _ in range(60):
        b.step(d, 10.0)
    return float(b.v[0]), float(b.v[1]), float(b.release_rate_hz()[0])


def small_gain(release, kinetics, pre_class, w, pre_receptor, base_drive, h=0.05):
    p0, q0, _ = steady(release, kinetics, pre_class, w, pre_receptor, base_drive - h)
    p1, q1, _ = steady(release, kinetics, pre_class, w, pre_receptor, base_drive + h)
    return (q1 - q0) / (p1 - p0)


def dynamic(release, kinetics, pre_class, w, pre_receptor, base_drive, amp, f=1.5):
    b = pair(release, kinetics, pre_class, w, pre_receptor)
    d = np.array([base_drive, 0.0], np.float32)
    t = 0.0
    step = 2.0
    pre, post, ts = [], [], []
    for k in range(int(3000 / step)):
        # the drive amplitude that gives a presynaptic sinusoid of ~amp mV: the
        # isolated presynaptic membrane is a 20 ms low-pass, so scale by its gain
        lp = 1.0 / math.sqrt(1.0 + (2 * math.pi * f * 0.020) ** 2)
        d[0] = base_drive + (amp / lp) * math.cos(2 * math.pi * f * t / 1000.0)
        b.step(d, step)
        t += step
        if t > 1000.0:
            pre.append(float(b.v[0]))
            post.append(float(b.v[1]))
            ts.append(t)
    ts = np.array(ts)
    e = np.exp(-2j * math.pi * f * ts / 1000.0)
    f_pre = 2 * np.mean(np.array(pre) * e)
    f_post = 2 * np.mean(np.array(post) * e)
    return dict(pre_f1_mV=float(abs(f_pre)), post_f1_mV=float(abs(f_post)),
                gain=float(abs(f_post) / abs(f_pre)),
                post_mean_mV=float(np.mean(post)), post_min_mV=float(np.min(post)),
                post_max_mV=float(np.max(post)))


W = gr.W_REF_PHOTORECEPTOR
PR_DRIVE = gr.ANCHOR_PHOTORECEPTOR_MV + 52.0
out = dict(spec='docs/LIF_DYNAMICS_SPEC.md#10.9 (a)', w_ref=W,
           derivation=gr.derive_gain(gr.MEASURED_GAIN_PHOTORECEPTOR, W, gr.ANCHOR_PHOTORECEPTOR_MV),
           tables=gr.RELEASE, a1={}, a2={}, a3={}, a4={})
for kin in (rk.KINETICS_PRIMARY, rk.KINETICS_V4_EQUIVALENT):
    for rel in (gr.RELEASE_PRIMARY, gr.RELEASE_V5_LINEAR):
        key = f'{rel}|{kin}'
        out['a1'][key] = small_gain(rel, kin, gr.CLASS_PHOTORECEPTOR, -W, 'hiscl', PR_DRIVE)
        out['a2'][key] = [dict(zip(('pre_mV', 'post_mV', 'release_hz'),
                                   steady(rel, kin, gr.CLASS_PHOTORECEPTOR, -W, 'hiscl', dr)))
                          for dr in (0.0, 5.0, 8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 15.0, 20.0)]
        out['a3'][key] = {f'amp_{a}': dynamic(rel, kin, gr.CLASS_PHOTORECEPTOR, -W, 'hiscl', PR_DRIVE, a)
                          for a in (0.1, 9.57)}
        out['a4'][key] = {
            sign: dict(small_signal_gain=small_gain(rel, kin, gr.CLASS_DEFAULT, s * W, rec, 0.0),
                       dynamic={f'amp_{a}': dynamic(rel, kin, gr.CLASS_DEFAULT, s * W, rec, 0.0, a)
                                for a in (0.1, 5.0)})
            for sign, s, rec in (('inhibitory', -1.0, 'glucl'), ('excitatory', 1.0, 'nicotinic'))}
        print(key, 'a1 gain', round(out['a1'][key], 3),
              'a3 large', round(out['a3'][key]['amp_9.57']['post_f1_mV'], 3), flush=True)
json.dump(out, open('docs/receipts/v6a_raw/unit_level.json', 'w'), indent=1)
