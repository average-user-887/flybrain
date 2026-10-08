#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Independent envelope check of exported clip GLBs (standard library only).

    python3 scripts/animations/verify_clip_envelope.py CLIPS.glb JOINTS.json [--substeps 20] [--json OUT]

Reads the glTF animations exactly as a browser would, decodes each rotation channel of a
rig-v2 DOF node to its angle about the contract axis (and reports how far the rotation
strays off that axis), samples every channel densely with LINEAR interpolation, including
the loop wrap, and checks the W2 joint contract plus its display-safe envelope.  Exit status
1 if any sample violates.  Joints that a clip does not key stay at rest (0), as a player
resets them when it stops the previous clip.
"""
import argparse
import json
import math
import struct
import sys

ENV_LIMITS = {'wing_sweep': (0.0, 3.0), 'wing_elevate': (None, 0.8), 'wing_pitch': (-0.07, 0.13),
              'antenna_abduct': (-0.4, 0.6), 'antenna_extend': (-0.2, 0.2), 'antenna_twist': (-0.1, 0.09),
              'funiculus_rotate': (-0.1, 0.1), 'haltere_beat': (-0.2, 0.2)}
MIN_ELEV = [(0, 0), (0.25, 0), (0.5, -0.1), (0.75, -0.9), (1.0, -0.8), (1.25, -0.6), (1.5, -0.3), (1.75, 0),
            (2.0, 0.2), (2.25, 0.4), (3.0, 0.4)]
MAX_EXT = [(-0.4, 0.40), (0.0, 0.40), (0.1, 0.35), (0.3, 0.35), (0.4, 0.30), (0.5, 0.25), (0.6, 0.20),
           (0.7, 0.15), (0.8, -0.05)]
LOOPING = ('wingbeat_loop', 'antenna_sweep')


def table(tab, x, default):
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return default


def read_glb(path):
    data = open(path, 'rb').read()
    clen = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + clen])
    return doc, data[20 + clen + 8:]


def accessor(doc, binc, i):
    a = doc['accessors'][i]
    v = doc['bufferViews'][a['bufferView']]
    n = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[a['type']]
    off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride', 4 * n)
    return [struct.unpack_from('<' + 'f' * n, binc, off + k * stride) for k in range(a['count'])]


def quat_angle(q, axis):
    x, y, z, w = q
    s = x * axis[0] + y * axis[1] + z * axis[2]
    off = math.sqrt(max(0.0, x * x + y * y + z * z - s * s))
    return 2 * math.atan2(s, w), off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb')
    ap.add_argument('joints')
    ap.add_argument('--substeps', type=int, default=20)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    doc, binc = read_glb(a.glb)
    contract = json.load(open(a.joints))
    axes = {j['node']: j['axis'] for j in contract['joints'] if 'axis' in j}
    ranges = {j['node']: j['range_rad'] for j in contract['joints'] if 'axis' in j}
    names = [n.get('name', '') for n in doc['nodes']]
    out, total = {}, 0
    for anim in doc.get('animations', []):
        curves = {}
        max_off = 0.0
        for ch in anim['channels']:
            node = names[ch['target']['node']]
            if node not in axes or ch['target']['path'] != 'rotation':
                continue
            smp = anim['samplers'][ch['sampler']]
            times = [t[0] for t in accessor(doc, binc, smp['input'])]
            vals = []
            for q in accessor(doc, binc, smp['output']):
                ang, off = quat_angle(q, axes[node])
                vals.append(ang)
                max_off = max(max_off, off)
            curves[node] = (times, vals)
        t_end = max((c[0][-1] for c in curves.values()), default=0.0)
        n_frames = max((len(c[0]) for c in curves.values()), default=1)
        steps = max(1, (n_frames - 1) * a.substeps)

        def value(node, t):
            if node not in curves:
                return 0.0
            ts, vs = curves[node]
            if t <= ts[0]:
                return vs[0]
            for k in range(len(ts) - 1):
                if ts[k] <= t <= ts[k + 1]:
                    u = (t - ts[k]) / (ts[k + 1] - ts[k])
                    return vs[k] + (vs[k + 1] - vs[k]) * u
            return vs[-1]

        stats = {n: [math.inf, -math.inf] for n in axes}
        viol = {}
        samples = []
        for k in range(steps + 1):
            samples.append({n: value(n, t_end * k / steps) for n in axes})
        if anim['name'] in LOOPING:                      # loop wrap: last frame back to the first
            first, last = samples[0], samples[-1]
            for k in range(1, a.substeps):
                u = k / a.substeps
                samples.append({n: last[n] + (first[n] - last[n]) * u for n in axes})
        for pose in samples:
            for n, v in pose.items():
                stats[n][0] = min(stats[n][0], v)
                stats[n][1] = max(stats[n][1], v)
                lo, hi = ranges[n]
                elo, ehi = ENV_LIMITS[n[2:]]
                lo = max(lo, elo) if elo is not None else lo
                hi = min(hi, ehi) if ehi is not None else hi
                if v < lo - 1e-5 or v > hi + 1e-5:
                    viol[f'{n} outside [{lo}, {hi}]'] = viol.get(f'{n} outside [{lo}, {hi}]', 0) + 1
            for s in ('l', 'r'):
                sw, el, pi = pose[f'{s}_wing_sweep'], pose[f'{s}_wing_elevate'], pose[f'{s}_wing_pitch']
                if el < table(MIN_ELEV, sw, 0.4) - 1e-5:
                    viol[f'{s}_wing_elevate below minimum for sweep'] = viol.get(f'{s}_wing_elevate below minimum for sweep', 0) + 1
                if sw < 0.5 and abs(pi) > 1e-5:
                    viol[f'{s}_wing_pitch nonzero while folded'] = viol.get(f'{s}_wing_pitch nonzero while folded', 0) + 1
                ab, ex = pose[f'{s}_antenna_abduct'], pose[f'{s}_antenna_extend']
                if ex > table(MAX_EXT, ab, -0.05) + 1e-5:
                    viol[f'{s}_antenna_extend above maximum for abduct'] = viol.get(f'{s}_antenna_extend above maximum for abduct', 0) + 1
        nv = sum(viol.values())
        total += nv
        out[anim['name']] = {
            'samples': len(samples), 'substeps_per_frame': a.substeps, 'violations': nv, 'violation_kinds': viol,
            'max_off_axis_quaternion_component': round(max_off, 6),
            'per_joint': {n: {'min': round(stats[n][0], 4), 'max': round(stats[n][1], 4),
                              'limit': [max(ranges[n][0], ENV_LIMITS[n[2:]][0]) if ENV_LIMITS[n[2:]][0] is not None
                                        else ranges[n][0],
                                        min(ranges[n][1], ENV_LIMITS[n[2:]][1])],
                              'keyed': n in curves} for n in sorted(axes)}}
        print(f'{anim["name"]}: samples={len(samples)} violations={nv} off_axis={max_off:.2e}')
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=2)
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
