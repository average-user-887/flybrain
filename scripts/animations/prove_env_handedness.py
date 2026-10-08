#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Prove which side each lettered leg of the spider/mantis clip GLBs lies on (standard library only).

    python3 scripts/animations/prove_env_handedness.py OUTDIR [--json PROOF.json]

For every L*/R* walking leg (and the spider palps and mantis raptorial forelegs) it takes the tip /
distal point from the exported GLB by forward kinematics, at rest and at every key of every clip, and
tests the sign of dot(p - body_centre, left) with left = up x forward = (+Y) x (+Z) = +X in the GLB
frame (right-handed, +Y up, +Z forward; forward x up = -X is the animal's right).

It then applies the renderer 750246c viewport mounting at several headings h (arena frame): the
fly's root uses rotation.y = h + pi/2 with local X reflected (scale.x = -s, setDisplayScale).  A prop
mount uses the same rotation WITHOUT the reflection.  On screen, with the Top camera (arena x right, y
up), the anatomical left of a drawn animal is up x (its drawn forward); with no other rotation and heading h
that is the arena direction (-sin h, cos h) = world (-sin h, 0, -cos h).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_env_clips as V  # noqa: E402


def mount(h, reflect):
    c, s = math.cos(h + math.pi / 2), math.sin(h + math.pi / 2)
    sx = -1.0 if reflect else 1.0
    return lambda p: (c * sx * p[0] + s * p[2], p[1], -s * sx * p[0] + c * p[2])    # R_y(h+pi/2) @ S(sx,1,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    proof, ok = {}, True
    for stem, pre in (('env_jumping_spider_rig', 'sp'), ('env_mantis_nymph_rig', 'mn')):
        doc, binc = V.read_glb(os.path.join(a.out, stem + '_clips.glb'))
        joints = json.load(open(os.path.join(a.out, stem + '_joints.json')))
        nodes = doc['nodes']
        names = [n.get('name', '') for n in nodes]
        idx = {n: i for i, n in enumerate(names)}
        rest = [(tuple(n.get('translation', (0, 0, 0))), tuple(n.get('rotation', (0, 0, 0, 1))),
                 tuple(n.get('scale', (1, 1, 1)))) for n in nodes]
        roots = doc['scenes'][0]['nodes']
        ident = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]

        def world(over):
            mats = {}

            def walk(i, m):
                t, r, s = rest[i]
                mi = V.mul(m, V.trs(over.get((i, 'translation'), t), over.get((i, 'rotation'), r), s))
                mats[i] = mi
                for c in nodes[i].get('children', []):
                    walk(c, mi)
            for r0 in roots:
                walk(r0, ident)
            return mats
        parts = {leg: (f'{pre}_{leg}_tarsus', tuple(joints['legs'][leg]['F'])) for leg in joints['legs']}
        if pre == 'sp':
            for sd in 'LR':
                parts[f'palp_{sd}'] = (f'spider_palp_{sd}', None)
        else:
            for sd in 'LR':
                parts[f'{sd}1_raptorial'] = (f'mn_{sd}1_tibia_mesh', None)
        centre_node = f'{pre}_body_shift'

        def point(mats, node, local):
            if local is not None:
                return V.apply(mats[idx[node]], local)
            prim = doc['meshes'][nodes[idx[node]]['mesh']]['primitives'][0]
            pts = V.accessor(doc, binc, prim['attributes']['POSITION'])
            c = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
            return V.apply(mats[idx[node]], c)
        frames = [('rest', world({}))]
        for anim in doc['animations']:
            curves = {}
            for ch in anim['channels']:
                smp = anim['samplers'][ch['sampler']]
                curves[(ch['target']['node'], ch['target']['path'])] = V.accessor(doc, binc, smp['output'])
            n = max(len(v) for v in curves.values())
            for k in range(n):
                frames.append((f"{anim['name']}#{k}", world({key: vs[min(k, len(vs) - 1)] for key, vs in curves.items()})))
        res = {}
        for part, (node, local) in parts.items():
            side = part[0] if part[0] in 'LR' else part[-1]
            want = 1 if side == 'L' else -1
            glb_signs, mount_ok, mount_reflect = set(), True, True
            for label, mats in frames:
                cen = V.apply(mats[idx[centre_node]], (0.0, 0.0, 0.0))
                body_m = mats[idx[f'{pre}_motion']]
                p = point(mats, node, local)
                d = tuple(p[k] - cen[k] for k in range(3))
                # left of the animal in the GLB frame: motion-node rotation applied to +X
                lx = (body_m[0][0], body_m[1][0], body_m[2][0])
                glb_signs.add(1 if sum(d[k] * lx[k] for k in range(3)) > 0 else -1)
                fz = (body_m[0][2], body_m[1][2], body_m[2][2])        # animal's forward (+Z of the motion node)
                for hdeg in (0, 41, 90, -128, 123, 180):
                    h = math.radians(hdeg)
                    for reflect in (False, True):
                        m = mount(h, reflect)
                        w, f = m(d), m(fz)
                        left_w = (f[2], 0.0, -f[0])                       # up x forward on screen, drawn heading
                        s = 1 if sum(w[k] * left_w[k] for k in range(3)) > 0 else -1
                        if not reflect and s != want:
                            mount_ok = False
                        if reflect and s == want:
                            mount_reflect = False
            correct = glb_signs == {want}
            ok &= correct and mount_ok
            res[part] = {'expected_side': 'left' if want > 0 else 'right',
                         'glb_frame_side': {1: 'left', -1: 'right'}.get(next(iter(glb_signs)), 'both') if len(glb_signs) == 1
                         else 'BOTH (crosses the midline)',
                         'renderer_750246c_prop_mount_no_reflection_correct_at_all_headings': mount_ok,
                         'renderer_750246c_fly_mount_with_reflection_would_be_mirrored': mount_reflect,
                         'samples': len(frames)}
        # Left turn vs drawn L side, in the SAME world transform (prop mount, no reflection), 6 headings.
        turn = {}
        for clip in ('turn_left_rootmotion', 'turn_right_rootmotion'):
            anim = next(a_ for a_ in doc['animations'] if a_['name'] == clip)
            curves = {}
            for ch in anim['channels']:
                smp = anim['samplers'][ch['sampler']]
                curves[(ch['target']['node'], ch['target']['path'])] = V.accessor(doc, binc, smp['output'])
            n = max(len(v_) for v_ in curves.values())
            m0 = world({key: vs[0] for key, vs in curves.items()})
            m1 = world({key: vs[min(n // 4, len(vs) - 1)] for key, vs in curves.items()})   # early in the turn
            mo0, mo1 = m0[idx[f'{pre}_motion']], m1[idx[f'{pre}_motion']]
            f0 = (mo0[0][2], mo0[1][2], mo0[2][2])
            f1 = (mo1[0][2], mo1[1][2], mo1[2][2])
            cen = V.apply(m0[idx[centre_node]], (0.0, 0.0, 0.0))
            ltips = [V.apply(m0[idx[f'{pre}_{leg}_tarsus']], tuple(joints['legs'][leg]['F'])) for leg in joints['legs']
                     if leg.startswith('L')]
            lcen = tuple(sum(p_[k] for p_ in ltips) / len(ltips) - cen[k] for k in range(3))
            per_heading = {}
            for hdeg in (0, 41, 90, -128, 123, 180):
                mnt = mount(math.radians(hdeg), False)
                F0, F1, Ld = mnt(f0), mnt(f1), mnt(lcen)
                left_drawn = (F0[2], 0.0, -F0[0])
                turn_side = 'left' if sum((F1[k] - F0[k]) * left_drawn[k] for k in range(3)) > 0 else 'right'
                l_side = 'left' if sum(Ld[k] * left_drawn[k] for k in range(3)) > 0 else 'right'
                per_heading[hdeg] = {'drawn_turn': turn_side, 'drawn_L_legs': l_side,
                                     'turn_toward_L_legs': turn_side == l_side}
            want_toward_L = clip.startswith('turn_left')
            good = all(v_['turn_toward_L_legs'] == want_toward_L and v_['drawn_L_legs'] == 'left'
                       for v_ in per_heading.values())
            ok &= good
            turn[clip] = {'pass': good, 'headings_deg': per_heading}
        res['_turns_vs_L_side'] = turn
        proof[stem] = res
        print(stem, {k: (v['glb_frame_side'], v['renderer_750246c_prop_mount_no_reflection_correct_at_all_headings'])
                     for k, v in res.items() if not k.startswith('_')},
              {c: t['pass'] for c, t in res['_turns_vs_L_side'].items()})
    proof['_method'] = (__doc__.strip().splitlines()[2:])
    proof['_result'] = 'PASS' if ok else 'FAIL'
    if a.json:
        json.dump(proof, open(a.json, 'w'), indent=2)
    print('RESULT', proof['_result'])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
