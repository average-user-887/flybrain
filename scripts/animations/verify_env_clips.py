#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Independent check of the spider/mantis clip GLBs as a browser plays them (standard library only).

    python3 scripts/animations/verify_env_clips.py CLIPS.glb RIG.glb JOINTS.json META.json [--hz 240] [--json OUT]

Checks, on the exported GLB only:

* rest reset: every node's rest TRS equals the rig GLB (so a stopped clip returns to the bind pose);
* joint limits over the WHOLE duration at --hz (LINEAR interpolation; loops include the seam from the
  last key back to the first), decoding each DOF rotation about its contract axis;
* continuity: largest rotation change between consecutive keys and across the loop seam (< 0.5 rad);
  for root-motion loops the motion node's translation/heading jump at the seam is the per-loop
  displacement and is reported, not counted as a pop;
* ground: forward kinematics of every leg mesh vertex at every sample, never below the foot plane
  (the lowest bind-pose leg vertex);
* stance-foot sliding (root-motion clips): during every stance interval listed in META (the legs the
  clip claims are planted) each tarsus tip in world space must not move horizontally more than
  --slide-tol between consecutive samples;
* heading and scale: walk_forward_rootmotion must move along +Z (the head direction) at the stated mm/s
  times the MEASURED scale (body-mesh Z extent / body length), and META's scale must equal it;
* W4's stance and seam predicates: cumulative stance drift per stance phase (stance = the lowest
  rest tarsus vertex within 0.06 viewport-mm of the floor) <= 2 % of body length, and loop seams no
  larger than 1.5x the first/last key step (min 0.02).

Body/leg intersection is checked in Blender during the build (ray-parity test), not here.
Exit status 1 on any violation.
"""
import argparse
import json
import math
import struct
import sys


W4_BAND, W4_DRIFT_BL = 0.06, 0.02     # W4's stance band (viewport-mm) and drift limit (fraction of body length)


def read_glb(path):
    data = open(path, 'rb').read()
    clen = struct.unpack_from('<I', data, 12)[0]
    return json.loads(data[20:20 + clen]), data[20 + clen + 8:]


def accessor(doc, binc, i):
    a = doc['accessors'][i]
    v = doc['bufferViews'][a['bufferView']]
    n = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[a['type']]
    off = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride', 4 * n)
    return [struct.unpack_from('<' + 'f' * n, binc, off + k * stride) for k in range(a['count'])]


def qnorm(q):
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return tuple(x / n for x in q)


def qlerp(a, b, u):
    if sum(x * y for x, y in zip(a, b)) < 0:
        b = tuple(-x for x in b)
    return qnorm(tuple(x + (y - x) * u for x, y in zip(a, b)))


def qdist(a, b):
    return 2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b)))))


def quat_angle(q, axis):
    x, y, z, w = q
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    s = x * axis[0] + y * axis[1] + z * axis[2]
    return 2 * math.atan2(s, w), math.sqrt(max(0.0, x * x + y * y + z * z - s * s))


def trs(t, r, s):
    x, y, z, w = r
    m = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[m[i][0] * s[0], m[i][1] * s[1], m[i][2] * s[2], t[i]] for i in range(3)] + [[0, 0, 0, 1]]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def apply(m, p):
    return tuple(m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3] for i in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb')
    ap.add_argument('rig')
    ap.add_argument('joints')
    ap.add_argument('meta')
    ap.add_argument('--hz', type=float, default=240.0)
    ap.add_argument('--max-step', type=float, default=0.5)
    ap.add_argument('--contact', type=float, default=0.03)
    ap.add_argument('--slide-tol', type=float, default=0.01)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    doc, binc = read_glb(a.glb)
    rig, _ = read_glb(a.rig)
    contract = json.load(open(a.joints))
    meta = json.load(open(a.meta))
    axes = {j['node']: j['axis'] for j in contract['joints'] if 'axis' in j}
    limits = {j['node']: j['range_rad'] for j in contract['joints'] if 'axis' in j}
    nodes = doc['nodes']
    names = [n.get('name', '') for n in nodes]
    idx = {n: i for i, n in enumerate(names)}
    roots = doc['scenes'][doc.get('scene', 0)]['nodes']
    rest = [(tuple(n.get('translation', (0, 0, 0))), tuple(n.get('rotation', (0, 0, 0, 1))),
             tuple(n.get('scale', (1, 1, 1)))) for n in nodes]
    out, total = {}, 0
    # rest reset
    rn = {n.get('name'): n for n in rig['nodes']}
    mism = []
    for n in nodes:
        r = rn.get(n.get('name'))
        if r is None:
            mism.append(n.get('name') + ' missing')
            continue
        for key, dflt in (('translation', (0, 0, 0)), ('scale', (1, 1, 1))):
            if max(abs(p - q) for p, q in zip(n.get(key, dflt), r.get(key, dflt))) > 1e-5:
                mism.append(f"{n['name']} {key}")
        if qdist(n.get('rotation', (0, 0, 0, 1)), r.get('rotation', (0, 0, 0, 1))) > 1e-5:
            mism.append(f"{n['name']} rotation")
    out['_rest_reset'] = {'mismatches': mism}
    total += len(mism)
    pre = 'sp' if 'spider' in contract['rig'] else 'mn'
    legs = contract['legs']
    leg_mesh = [i for i, nm in enumerate(names) if 'mesh' in nodes[i] and nm.startswith(pre + '_') and
                nm.split('_')[1] in legs]
    verts = {}
    for i in leg_mesh:
        pts = []
        for prim in doc['meshes'][nodes[i]['mesh']]['primitives']:
            pts += accessor(doc, binc, prim['attributes']['POSITION'])
        verts[i] = pts
    tars = {leg: idx[f'{pre}_{leg}_tarsus'] for leg in legs}
    ident = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]

    def world(over):
        mats = {}

        def walk(i, m):
            t, r, s = rest[i]
            t = over.get((i, 'translation'), t)
            r = over.get((i, 'rotation'), r)
            mi = mul(m, trs(t, r, s))
            mats[i] = mi
            for c in nodes[i].get('children', []):
                walk(c, mi)
        for r0 in roots:
            walk(r0, ident)
        return mats

    def lowest(mats):
        low = math.inf
        for i in leg_mesh:
            m = mats[i]
            r0, r1, r2, r3 = m[1]
            for (x, y, z) in verts[i]:
                v = r0 * x + r1 * y + r2 * z + r3
                low = v if v < low else low
        return low

    rest_m = world({})
    foot_plane = lowest(rest_m)
    tip_local = {leg: tuple(meta.get('contact_points', {}).get(leg, legs[leg]['F'])) for leg in legs}
    rest_tip = {leg: apply(rest_m[tars[leg]], tip_local[leg]) for leg in legs}
    # scale: Z extent of the body meshes (the same set W4 measures) / stated body length
    import re
    body_set = re.compile(r'^spider_(body|eyes|chelicera_[LR])$' if pre == 'sp' else r'^mantis_(body|head)$')
    body_z = []
    for i, nm in enumerate(names):
        if 'mesh' in nodes[i] and body_set.match(nm):
            for prim in doc['meshes'][nodes[i]['mesh']]['primitives']:
                body_z += [apply(rest_m[i], p)[2] for p in accessor(doc, binc, prim['attributes']['POSITION'])]
    body_len = max(body_z) - min(body_z)
    bl_mm = meta['parameters']['body_length_mm']
    scale = body_len / bl_mm
    out['_scale'] = {'body_mesh_set': body_set.pattern, 'body_extent_viewport_mm': round(body_len, 4),
                     'body_length_mm': bl_mm, 'measured_viewport_mm_per_mm': round(scale, 5),
                     'metadata_viewport_mm_per_mm': meta['viewport_mm_per_mm']}
    if abs(scale - meta['viewport_mm_per_mm']) / scale > 1e-3:
        out['_scale']['violation'] = 'metadata scale differs from the measured scale'
        total += 1
    # W4 stance definition: per leg the tarsus vertex lowest at rest; stance while within 0.06 of the floor
    tipv = {}
    for leg in legs:
        i = tars[leg]
        best = min(verts[i], key=lambda q: apply(rest_m[i], q)[1])
        tipv[leg] = (i, best)
    floor_tars = min(apply(rest_m[i], q)[1] for i, q in tipv.values())
    motion = idx[f'{pre}_motion']
    for anim in doc.get('animations', []):
        name = anim['name']
        curves = {}
        for ch in anim['channels']:
            smp = anim['samplers'][ch['sampler']]
            curves[(ch['target']['node'], ch['target']['path'])] = (
                [t[0] for t in accessor(doc, binc, smp['input'])], accessor(doc, binc, smp['output']))
        info = meta['clips'].get(name, {})
        loop = info.get('loop', False)
        t0 = min(c[0][0] for c in curves.values())          # glTF times of the first key (frame 1)
        t_end = max(c[0][-1] for c in curves.values())
        n_keys = max(len(c[0]) for c in curves.values())
        dt = (t_end - t0) / max(1, n_keys - 1)
        root_motion = name.endswith('_rootmotion')
        stance = info.get('stance_key_intervals', {})

        def val(key, t):
            ts, vs = curves[key]
            rot = key[1] == 'rotation'
            if t >= ts[-1]:
                if loop and t > ts[-1] + 1e-9 and not (root_motion and key[0] == motion):
                    u = min(1.0, (t - ts[-1]) / dt)
                    return qlerp(vs[-1], vs[0], u) if rot else tuple(p + (q - p) * u for p, q in zip(vs[-1], vs[0]))
                return vs[-1]
            if t <= ts[0]:
                return vs[0]
            k = max(0, min(len(ts) - 2, int((t - ts[0]) / dt)))
            while ts[k + 1] < t:
                k += 1
            while ts[k] > t:
                k -= 1
            u = (t - ts[k]) / (ts[k + 1] - ts[k])
            return qlerp(vs[k], vs[k + 1], u) if rot else tuple(p + (q - p) * u for p, q in zip(vs[k], vs[k + 1]))

        # Loop seams are sampled for in-place loops only.  A root-motion loop is played by accumulating
        # its displacement, so its seam is checked as key continuity (below), not by holding the root still.
        span = (t_end - t0) + (dt if loop and not root_motion else 0.0)
        ns = int(math.ceil(span * a.hz)) + 1
        sample_times = sorted({t0 + min(span, s / a.hz) for s in range(ns)}
                              | {t for ts, _ in curves.values() for t in ts})
        ns = len(sample_times)
        viol, stats = {}, {n: [math.inf, -math.inf] for n in axes}
        min_low, max_slide, off_axis = math.inf, 0.0, 0.0
        prev = None
        w4 = {}

        def bump(k):
            viol[k] = viol.get(k, 0) + 1
        for t in sample_times:
            over = {key: val(key, t) for key in curves}
            for n, ax in axes.items():
                q = over.get((idx[n], 'rotation'), rest[idx[n]][1])
                ang, off = quat_angle(q, ax)
                off_axis = max(off_axis, off)
                stats[n][0] = min(stats[n][0], ang)
                stats[n][1] = max(stats[n][1], ang)
                lo, hi = limits[n]
                if ang < lo - 1e-5 or ang > hi + 1e-5:
                    bump(f'{n} outside [{lo}, {hi}]')
            mats = world(over)
            low = lowest(mats)
            min_low = min(min_low, low)
            if low < foot_plane - 1e-4:
                bump('leg vertex below foot plane')
            if root_motion:
                for leg, (ti, tq) in tipv.items():
                    w = apply(mats[ti], tq)
                    st = w4.setdefault(leg, {'in': False, 'start': None, 'max': 0.0})
                    if w[1] <= floor_tars + W4_BAND:
                        if not st['in']:
                            st['in'], st['start'] = True, w
                        st['max'] = max(st['max'], math.hypot(w[0] - st['start'][0], w[2] - st['start'][2]))
                    else:
                        st['in'] = False
                # stance intervals come from the clip metadata (the legs that CLAIM to be planted)
                tips = {leg: apply(mats[tars[leg]], tip_local[leg]) for leg in legs}
                key = (t - t0) / dt
                if prev is not None:
                    for leg in legs:
                        planted = any(f0 <= key <= f1 and f0 <= prev[1] <= f1 for f0, f1 in stance.get(leg, []))
                        if planted:
                            d = math.hypot(tips[leg][0] - prev[0][leg][0], tips[leg][2] - prev[0][leg][2])
                            max_slide = max(max_slide, d)
                            if d > a.slide_tol:
                                bump('stance foot slides')
                prev = (tips, key)
        max_step = 0.0
        seam = 0.0
        for (i, path), (ts, vs) in curves.items():
            if path != 'rotation' or (root_motion and i == motion):
                continue
            for k in range(len(vs) - 1):
                max_step = max(max_step, qdist(vs[k], vs[k + 1]))
            if loop:
                seam = max(seam, qdist(vs[-1], vs[0]))
        w4_drift = max((st['max'] for st in w4.values()), default=0.0)
        if root_motion and w4_drift > W4_DRIFT_BL * body_len:
            bump('W4 cumulative stance drift > 2% BL')
        w4_seam = []
        if loop:
            for (i, path), (ts, vs) in curves.items():
                if i == motion:
                    continue
                dd = (lambda p, q: qdist(vs[p], vs[q])) if path == 'rotation' else (lambda p, q: math.dist(vs[p], vs[q]))
                k = len(vs)
                if k < 3:
                    continue
                seam_v, allowed = dd(k - 1, 0), max(0.02, 1.5 * max(dd(0, 1), dd(k - 2, k - 1)))
                if seam_v > allowed:
                    w4_seam.append(f'{names[i]}.{path} {seam_v:.3f} > {allowed:.3f}')
            if w4_seam:
                bump('W4 loop seam (1.5x neighbouring step)')
        if max_step > a.max_step:
            bump('step between keys')
        if seam > a.max_step:
            bump('loop seam step')
        rec = {'samples': ns, 'hz': a.hz, 'duration_s': round(t_end - t0, 4), 'loop': loop, 'root_motion': root_motion,
               'violations': sum(viol.values()), 'violation_kinds': viol,
               'lowest_leg_vertex_minus_foot_plane': round(min_low - foot_plane, 5),
               'max_stance_slide_per_sample_viewport_mm': round(max_slide, 5) if root_motion else None,
               'slide_tolerance_per_sample': a.slide_tol if root_motion else None,
               'max_step_between_keys_rad': round(max_step, 4), 'loop_seam_step_rad': round(seam, 4) if loop else None,
               'max_off_axis_quaternion_component': round(off_axis, 6),
               'w4_max_cumulative_stance_drift_viewport_mm': round(w4_drift, 4) if root_motion else None,
               'w4_max_cumulative_stance_drift_pct_bl': round(100 * w4_drift / body_len, 3) if root_motion else None,
               'w4_seam_excess': w4_seam,
               'per_joint': {n: [round(stats[n][0], 4), round(stats[n][1], 4), limits[n]] for n in sorted(axes)
                             if (idx[n], 'rotation') in curves}}
        if root_motion and (motion, 'translation') in curves:
            ts, vs = curves[(motion, 'translation')]
            dx, dz = vs[-1][0] - vs[0][0], vs[-1][2] - vs[0][2]
            rec['root_displacement_viewport_mm'] = [round(dx, 3), round(dz, 3)]
            if name.startswith('walk_forward'):
                head = math.degrees(math.atan2(dx, dz))
                speed = math.hypot(dx, dz) / (t_end - t0)
                want = meta['parameters']['speed_mm_s'] * scale        # stated mm/s at the MEASURED scale
                rec['heading_of_travel_deg'] = round(head, 3)
                rec['speed_viewport_mm_s'] = round(speed, 3)
                if abs(head) > 0.5:
                    bump('travel not along +Z (head)')
                if abs(speed - want) / want > 0.05:
                    bump('speed differs from metadata')
        if root_motion and name.startswith(('turn_left', 'turn_right')):
            ts, qs = curves[(motion, 'rotation')]
            yaw_steps = [quat_angle(q, (0.0, 1.0, 0.0))[0] - quat_angle(p, (0.0, 1.0, 0.0))[0]
                         for p, q in zip(qs, qs[1:])]
            yaw_total = sum(math.atan2(math.sin(d), math.cos(d)) for d in yaw_steps)
            mean_rate = yaw_total / info['seconds']
            rec['mean_yaw_rate_rad_s'] = round(mean_rate, 5)
            expected_sign = 1 if name.startswith('turn_left') else -1
            if mean_rate * expected_sign <= 0:
                bump('turn direction differs from anatomical label')
            if abs(abs(mean_rate) - meta['turn']['yaw_rate_rad_s']) / meta['turn']['yaw_rate_rad_s'] > 0.15:
                bump('mean turn rate differs from metadata')
        rec['violations'] = sum(viol.values())
        total += rec['violations']
        out[name] = rec
        print(f"{name}: samples={ns} violations={rec['violations']} low={rec['lowest_leg_vertex_minus_foot_plane']} "
              f"slide={rec['max_stance_slide_per_sample_viewport_mm']} step={max_step:.3f} seam={seam:.3f}")
    print(f"rest reset mismatches: {len(mism)}; body z extent {body_len:.2f}")
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=2)
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main())
