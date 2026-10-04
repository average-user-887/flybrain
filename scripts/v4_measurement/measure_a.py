"""Predeclared measurement (a): unit-level, spec §7.10(a). CPU, synthetic graphs."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '<redacted-path>/Documents/ChatGPT/flybrain/<redacted-path>/.wt-graded')
os.environ['NEUROFLY_BRAIN_BACKEND'] = 'cpu'
from brainlab.brain import Brain
from brainlab.engine import E_EXC_MV, E_INH_MV, R_MAX_HZ, V_REST_MV

out = {}


def chain(n, edges, weights):
    """CSR for an explicit edge list."""
    ptr = np.zeros(n + 1, dtype=np.int64)
    post, w = [], []
    for i in range(n):
        for (a, b), wt in zip(edges, weights):
            if a == i:
                post.append(b)
                w.append(wt)
        ptr[i + 1] = len(post)
    return dict(ptr=ptr, post=np.array(post, dtype=np.int32),
                weight=np.array(w, dtype=np.float32),
                ids=np.arange(1, n + 1, dtype=np.int64))


# ---- a1: subthreshold transmission through a graded cell -------------------
# 0 -> 1 -> 2.  Node 0 is driven; node 1 is declared graded and is held strictly
# below -45 mV; does node 2 move?
# Node 1 is driven subthreshold, in both directions, and node 2 is read out.
arrays = chain(3, [(0, 1), (1, 2)], [10.0, 10.0])
res = {}
for label, version, mask, drive1 in (
        ('v3_depolarising', 'v3', None, +4.0),
        ('v3_hyperpolarising', 'v3', None, -4.0),
        ('v4_baseline_only', 'v4', np.array([0, 1, 0], np.uint8), 0.0),
        ('v4_depolarising', 'v4', np.array([0, 1, 0], np.uint8), +4.0),
        ('v4_hyperpolarising', 'v4', np.array([0, 1, 0], np.uint8), -4.0)):
    b = Brain(arrays=dict(arrays), dynamics=version,
              **({'graded_policy': mask} if mask is not None else {}))
    drive = np.zeros(3, np.float32)
    drive[1] = drive1
    for _ in range(100):
        b.step(drive, 5.0)
    res[label] = dict(v1=float(b.v[1]), v2=float(b.v[2]),
                      g_exc_node2=float(b.g[0, 2]) if b.g.ndim == 2 else None,
                      spikes=int(b.total_spikes))
base = res['v4_baseline_only']['v2']
out['a1_subthreshold_transmission'] = res
out['a1_verdict'] = (
    'PASS: node 1 held subthreshold, node 2 responds to it under v4 in BOTH directions '
    'and not at all under v3'
    if res['v4_depolarising']['v1'] < -45.0
    and res['v4_depolarising']['v2'] > base + 1e-3
    and res['v4_hyperpolarising']['v2'] < base - 1e-3
    and abs(res['v3_depolarising']['v2'] - V_REST_MV) < 1e-9
    and abs(res['v3_hyperpolarising']['v2'] - V_REST_MV) < 1e-9
    else 'FAIL')

# ---- a2: a spiking cell downstream of a graded one, and vice versa ---------
# 0 (spiking, driven hard) -> 1 (graded) -> 2 (spiking).
arrays = chain(3, [(0, 1), (1, 2)], [400.0, 4000.0])
b = Brain(arrays=dict(arrays), dynamics='v4', graded_policy=np.array([0, 1, 0], np.uint8))
drive = np.zeros(3, np.float32)
drive[0] = 30.0
tot = np.zeros(3, np.int64)
for _ in range(40):
    c, _ = b.step(drive, 25.0)
    tot += c
out['a2_spiking_graded_spiking'] = dict(
    spikes_node0=int(tot[0]), spikes_node1=int(tot[1]), spikes_node2=int(tot[2]),
    v_graded=float(b.v[1]), v_downstream=float(b.v[2]),
    release_hz_graded=float(b.release_rate_hz()[1]))
out['a2_verdict'] = ('PASS: the graded cell never spikes, is driven by node 0, and drives node 2'
                     if tot[0] > 0 and tot[1] == 0 and tot[2] > 0 else 'FAIL')

# ---- a3: S0 bit-identity, v4 with no declared graded class == v3 -----------
def probe_a(version, mask=None):
    arrays = chain(2, [(0, 1)], [-40.0])
    kw = {'graded_policy': mask} if mask is not None else {}
    b = Brain(arrays=arrays, dynamics=version, **kw)
    drive = np.zeros(2, np.float32)
    drive[0] = 20.0
    vmin = 1e9
    for _ in range(100):
        b.step(drive, 20.0)
        vmin = min(vmin, float(b.v[1]))
    return b, vmin


def probe_b(version, mask=None, gain=8.0, seed=11):
    rng = np.random.default_rng(seed)
    n, k = 2000, 40
    post = rng.integers(0, n, size=n * k).astype(np.int32)
    ptr = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    sign = np.where(rng.random(n * k) < 0.8, 1.0, -1.0)
    weight = (sign * gain).astype(np.float32)
    arrays = dict(ptr=ptr, post=post, weight=weight, ids=np.arange(1, n + 1, dtype=np.int64))
    kw = {'graded_policy': mask} if mask is not None else {}
    b = Brain(arrays=arrays, dynamics=version, **kw)
    drive = np.zeros(n, np.float32)
    drive[rng.choice(n, n // 10, replace=False)] = 25.0
    b.step(drive, 200.0)
    free = np.zeros(n, np.int64)
    zero = np.zeros(n, np.float32)
    for _ in range(8):
        c, _ = b.step(zero, 100.0)
        free += c
    return b, float(free.sum()) / 0.8 / n


zero_mask2 = np.zeros(2, np.uint8)
b3, a3_v3 = probe_a('v3')
b4, a3_v4 = probe_a('v4', zero_mask2)
ident_a = all(np.array_equal(getattr(b3, k), getattr(b4, k)) for k in
              ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts', 'active',
               'active_flag', 'nactive'))
zero_mask = np.zeros(2000, np.uint8)
c3, rate3 = probe_b('v3')
c4, rate4 = probe_b('v4', zero_mask)
ident_b = all(np.array_equal(getattr(c3, k), getattr(c4, k)) for k in
              ('v', 'g', 'refractory', 'queue', 'queue_count', 'counts', 'active',
               'active_flag', 'nactive'))
out['a3_s0_bit_identity'] = dict(
    probe_a_min_v_v3=a3_v3, probe_a_min_v_v4=a3_v4, probe_a_arrays_identical=bool(ident_a),
    probe_b_free_rate_hz_per_neuron_v3=rate3, probe_b_free_rate_hz_per_neuron_v4=rate4,
    probe_b_arrays_identical=bool(ident_b),
    v3_total_spikes=int(c3.total_spikes), v4_total_spikes=int(c4.total_spikes))
out['a3_verdict'] = ('PASS: bit-identical' if ident_a and ident_b and a3_v3 == a3_v4
                     and rate3 == rate4 and c3.total_spikes == c4.total_spikes else 'FAIL')

# ---- a4: release calibration ----------------------------------------------
# A graded cell clamped at V must deliver the mean conductance a v3 spiking cell
# firing at r(V) Hz delivers: mean g = |w| * g_unit * r * tau_syn / 1000.
rows = []
arrays = chain(2, [(0, 1)], [1.0])
for v_target in (-70.0, -52.0, -45.0, -20.0, 0.0):
    b = Brain(arrays=dict(arrays), dynamics='v4', graded_policy=np.array([1, 0], np.uint8))
    # Clamp node 0 at v_target with a current: at rest g_tot = 1 so V_inf = V_rest + I.
    drive = np.zeros(2, np.float32)
    drive[0] = v_target - V_REST_MV
    b.step(drive, 500.0)
    r_expected = R_MAX_HZ * (v_target - E_INH_MV) / (E_EXC_MV - E_INH_MV)
    g_expected = 1.0 * (1.0 / 52.0) * r_expected * 5.0 / 1000.0
    rows.append(dict(v_clamp_mV=v_target, v_actual_mV=float(b.v[0]),
                     r_declared_hz=r_expected, r_measured_hz=float(b.release_rate_hz()[0]),
                     g_exc_target_expected=g_expected, g_exc_target_measured=float(b.g[0, 1]),
                     rel_error=(float(b.g[0, 1]) - g_expected) / g_expected if g_expected else 0.0))
out['a4_release_calibration'] = rows
out['a4_verdict'] = ('PASS: within 1%' if all(abs(r['rel_error']) < 0.01 for r in rows[1:])
                     else 'CHECK')

print(json.dumps(out, indent=2))
with open('<redacted-path>/tmp/graded/measure_a.json', 'w') as fh:
    json.dump(out, fh, indent=2)
