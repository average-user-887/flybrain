#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Independent check of exported clip GLBs, as a browser would play them (standard library only).

    python3 scripts/animations/verify_clip_envelope.py CLIPS.glb JOINTS.json [--hz 240] [--json OUT]

For every glTF animation:

* samples the WHOLE duration at ``--hz`` (default 240) with LINEAR interpolation (rotations by
  normalised quaternion lerp, within 1e-3 rad of slerp at these key spacings), and for looping
  clips also the wrap seam from the last key back to the first over one frame;
* decodes each rig-v2 DOF rotation to its angle about the contract axis (reporting off-axis
  residue) and checks the W2 contract ranges plus the display-safe envelope;
* checks continuity: the largest rotation change of any animated node between consecutive keys,
  and separately across the loop seam, against ``--max-step`` (radians per frame);
* checks that every femur-tibia joint keeps flexing the way the display pose does (angle about
  its local X in (0, pi)), so no knee bends backwards;
* runs forward kinematics of the whole node tree from the GLB at every sample and checks that no
  vertex of any leg mesh goes below the foot plane (the lowest tarsus vertex of the bind pose).

Joints that a clip does not key stay at rest, as a player resets them when it stops a clip.
Exit status 1 on any violation.
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
LOOPING = ('wingbeat_loop', 'antenna_sweep', 'walk_tripod_loop', 'idle_antenna_twitch')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
LEG_SUFFIX = ('_coxa', '_trochanterfemur', '_tibia', '_tarsus')
FOOT_TOL = 1e-4


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
    if w < 0:                                       # q and -q are the same rotation; use w >= 0
        x, y, z, w = -x, -y, -z, -w
    s = x * axis[0] + y * axis[1] + z * axis[2]
    off = math.sqrt(max(0.0, x * x + y * y + z * z - s * s))
    return 2 * math.atan2(s, w), off


def qlerp(a, b, u):
    if sum(x * y for x, y in zip(a, b)) < 0:
        b = tuple(-x for x in b)
    q = tuple(x + (y - x) * u for x, y in zip(a, b))
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return tuple(x / n for x in q)


def qdist(a, b):
    return 2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b)))))


def trs_matrix(t, r, s):
    x, y, z, w = r
    m = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[m[i][0] * s[0], m[i][1] * s[1], m[i][2] * s[2], t[i]] for i in range(3)] + [[0, 0, 0, 1]]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb')
    ap.add_argument('joints')
    ap.add_argument('--hz', type=float, default=240.0)
    ap.add_argument('--max-step', type=float, default=0.5)
    ap.add_argument('--json', default='')
    ap.add_argument('--rig', default='', help='source rig GLB: node rest TRS of the clip GLB must match it')
    a = ap.parse_args()
    doc, binc = read_glb(a.glb)
    rest_mismatch = []
    if a.rig:
        rdoc, _ = read_glb(a.rig)
        rnodes = {n.get('name', ''): n for n in rdoc['nodes']}
        for n in doc['nodes']:
            r = rnodes.get(n.get('name', ''))
            if r is None:
                rest_mismatch.append(n.get('name', '') + ' missing in rig')
                continue
            for key, dflt in (('translation', (0, 0, 0)), ('scale', (1, 1, 1))):
                if max(abs(p - q) for p, q in zip(n.get(key, dflt), r.get(key, dflt))) > 1e-4:
                    rest_mismatch.append(f"{n.get('name')} {key}")
            qa, qb = n.get('rotation', (0, 0, 0, 1)), r.get('rotation', (0, 0, 0, 1))
            if qdist(qa, qb) > 1e-4:
                rest_mismatch.append(f"{n.get('name')} rotation")
        print(f'rest pose vs rig: {len(rest_mismatch)} mismatches {rest_mismatch[:6]}')
    contract = json.load(open(a.joints))
    axes = {j['node']: j['axis'] for j in contract['joints'] if 'axis' in j}
    ranges = {j['node']: j['range_rad'] for j in contract['joints'] if 'axis' in j}
    nodes = doc['nodes']
    names = [n.get('name', '') for n in nodes]
    roots = doc['scenes'][doc.get('scene', 0)]['nodes']
    rest = [(tuple(n.get('translation', (0, 0, 0))), tuple(n.get('rotation', (0, 0, 0, 1))),
             tuple(n.get('scale', (1, 1, 1)))) for n in nodes]
    leg_nodes = [i for i, nm in enumerate(names) if 'mesh' in nodes[i] and nm.endswith(LEG_SUFFIX) and nm[:2] in LEGS]
    verts = {}
    for i in leg_nodes:
        pts = []
        for prim in doc['meshes'][nodes[i]['mesh']]['primitives']:
            pts += accessor(doc, binc, prim['attributes']['POSITION'])
        verts[i] = pts
    ident = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]

    def world(rot_over):
        mats = {}

        def walk(i, m):
            t, r, s = rest[i]
            mi = mul(m, trs_matrix(t, rot_over.get(i, r), s))
            mats[i] = mi
            for c in nodes[i].get('children', []):
                walk(c, mi)
        for r0 in roots:
            walk(r0, ident)
        return mats

    def lowest(mats, only):
        low = math.inf
        for i in only:
            m = mats[i]
            r0, r1, r2, r3 = m[1]
            for (x, y, z) in verts[i]:
                v = r0 * x + r1 * y + r2 * z + r3
                if v < low:
                    low = v
        return low

    foot_plane = lowest(world({}), [i for i in leg_nodes if names[i].endswith('_tarsus')])
    idx = {n: names.index(n) for n in axes if n in names}
    tibia_nodes = [i for i, nm in enumerate(names) if nm.endswith('_tibia_joint')]
    out, total = {}, 0
    for anim in doc.get('animations', []):
        curves = {}
        for ch in anim['channels']:
            if ch['target']['path'] != 'rotation':
                continue
            smp = anim['samplers'][ch['sampler']]
            curves[ch['target']['node']] = ([t[0] for t in accessor(doc, binc, smp['input'])],
                                            accessor(doc, binc, smp['output']))
        t_end = max((c[0][-1] for c in curves.values()), default=0.0)
        n_keys = max((len(c[0]) for c in curves.values()), default=1)
        dt = t_end / max(1, n_keys - 1)
        loop = anim['name'] in LOOPING
        legs_animated = any(names[i][:2] in LEGS for i in curves)

        def rot_at(t):
            res = {}
            for i, (ts, qs) in curves.items():
                if t >= ts[-1]:
                    res[i] = qlerp(qs[-1], qs[0], min(1.0, (t - ts[-1]) / dt)) if loop else qs[-1]
                    continue
                if t <= ts[0]:
                    res[i] = qs[0]
                    continue
                k = max(0, min(len(ts) - 2, int((t - ts[0]) / dt)))
                while ts[k + 1] < t:
                    k += 1
                while ts[k] > t:
                    k -= 1
                res[i] = qlerp(qs[k], qs[k + 1], (t - ts[k]) / (ts[k + 1] - ts[k]))
            return res

        span = t_end + (dt if loop else 0.0)
        n_samples = int(math.ceil(span * a.hz)) + 1
        stats = {n: [math.inf, -math.inf] for n in axes}
        viol, max_off, min_low = {}, 0.0, math.inf

        def bump(k):
            viol[k] = viol.get(k, 0) + 1
        for s in range(n_samples):
            rots = rot_at(min(span, s / a.hz))
            pose = {}
            for n, i in idx.items():
                ang, off = quat_angle(rots.get(i, rest[i][1]), axes[n])
                pose[n] = ang
                max_off = max(max_off, off)
                stats[n][0] = min(stats[n][0], ang)
                stats[n][1] = max(stats[n][1], ang)
                lo, hi = ranges[n]
                elo, ehi = ENV_LIMITS[n[2:]]
                lo = max(lo, elo) if elo is not None else lo
                hi = min(hi, ehi) if ehi is not None else hi
                if ang < lo - 1e-5 or ang > hi + 1e-5:
                    bump(f'{n} outside [{lo}, {hi}]')
            for sd in ('l', 'r'):
                sw, el, pi = pose[f'{sd}_wing_sweep'], pose[f'{sd}_wing_elevate'], pose[f'{sd}_wing_pitch']
                if el < table(MIN_ELEV, sw, 0.4) - 1e-5:
                    bump(f'{sd}_wing_elevate below minimum for sweep')
                if sw < 0.5 and abs(pi) > 1e-5:
                    bump(f'{sd}_wing_pitch nonzero while folded')
                ab, ex = pose[f'{sd}_antenna_abduct'], pose[f'{sd}_antenna_extend']
                if ex > table(MAX_EXT, ab, -0.05) + 1e-5:
                    bump(f'{sd}_antenna_extend above maximum for abduct')
            for i in tibia_nodes:                       # femur-tibia joint flexes the anatomical way
                ang, _ = quat_angle(rots.get(i, rest[i][1]), (1.0, 0.0, 0.0))
                if not 0.0 < ang < math.pi:
                    bump(f'{names[i]} flexes the non-anatomical way')
            if legs_animated:
                low = lowest(world(rots), leg_nodes)
                min_low = min(min_low, low)
                if low < foot_plane - FOOT_TOL:
                    bump('leg vertex below foot plane')
        max_step = max((qdist(qs[k], qs[k + 1]) for ts, qs in curves.values() for k in range(len(qs) - 1)),
                       default=0.0)
        seam_step = max((qdist(qs[-1], qs[0]) for ts, qs in curves.values()), default=0.0) if loop else 0.0
        if max_step > a.max_step:
            bump(f'step between keys {max_step:.3f} rad > {a.max_step}')
        if seam_step > a.max_step:
            bump(f'loop seam step {seam_step:.3f} rad > {a.max_step}')
        nv = sum(viol.values())
        total += nv
        below = max(0.0, foot_plane - min_low) if min_low < math.inf else 0.0
        out[anim['name']] = {
            'samples': n_samples, 'sample_hz': a.hz, 'duration_s': round(t_end, 4), 'loop': loop,
            'loop_seam_checked': loop, 'violations': nv, 'violation_kinds': viol,
            'max_off_axis_quaternion_component': round(max_off, 6),
            'max_step_between_keys_rad': round(max_step, 4),
            'loop_seam_step_rad': round(seam_step, 4) if loop else None,
            'foot_plane_y': round(foot_plane, 5), 'foot_plane_checked': legs_animated,
            'max_below_foot_plane': round(below, 5),
            'per_joint': {n: {'min': round(stats[n][0], 4), 'max': round(stats[n][1], 4),
                              'limit': [max(ranges[n][0], ENV_LIMITS[n[2:]][0]) if ENV_LIMITS[n[2:]][0] is not None
                                        else ranges[n][0], min(ranges[n][1], ENV_LIMITS[n[2:]][1])],
                              'keyed': idx.get(n) in curves} for n in sorted(axes)}}
        print(f'{anim["name"]}: samples={n_samples}@{a.hz:g}Hz violations={nv} below_foot={below:.4f} '
              f'max_step={max_step:.3f} seam={seam_step:.3f} off_axis={max_off:.1e}')
    if a.rig:
        out['_rest_pose_vs_rig'] = {'mismatches': rest_mismatch}
        total += len(rest_mismatch)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=2)
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
