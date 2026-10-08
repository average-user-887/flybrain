# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Articulated rigs for the illustrative jumping spider and mantis nymph (ENV-ANIM-01).

Headless:  blender --background --factory-startup --threads 2 \
               --python tools/assets/build_env_rigs.py -- --out OUTDIR --animal spider|mantis \
               [--static ENV_DIR]

Rebuilds the SAME geometry as build_env.py (identical vertices in the bind pose) and splits the
legs (and the mantis head and raptorial forelegs) into parts hung on joint chains.  Writes
OUTDIR/env_<animal>_rig.glb and .blend and env_<animal>_rig_joints.json, the machine-readable
joint/frame/unit contract (see tools/assets/env/ENV_RIG_CONTRACT.md).  With --static ENV_DIR it
checks that the bind pose reproduces env_jumping_spider.glb / env_mantis_nymph.glb exactly.

Chain convention (as the W2 fly rig): <joint>_root (translation to the pivot) -> one node per
rotational DOF -> <joint>_offset (translation back).  Mesh vertices stay in the animal's root
frame, so the bind pose is every DOF at 0 (identity).  Frame: three.js, +Y up, +Z forward,
1 unit = 1 viewport-mm (fly display scale).  Presentation only: ILLUSTRATIVE, no predator stimulus
or behaviour; must not change looming or retinal input.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import nf_geom as g  # noqa: E402

ARGS = g.parse_args({'out': '', 'animal': '', 'static': ''})
ANIMAL = ARGS['animal']
if ANIMAL not in ('spider', 'mantis'):
    raise SystemExit('pass -- --animal spider|mantis')
scene = g.reset_scene()


# ------------------------------------------------------------------ geometry helpers (copied from build_env.py)
def lathe(profile, seg, geo, matrix=None, mat_fn=None):
    mat_fn = mat_fn or (lambda j, k: 0)
    verts, faces, mats, rings = [], [], [], []
    for (y, r) in profile:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, y, 0.0))
            continue
        ids = []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            ids.append(len(verts))
            verts.append((r * math.cos(a), y, r * math.sin(a)))
        rings.append(ids)
    for j in range(len(rings) - 1):
        a, b = rings[j], rings[j + 1]
        for k in range(seg):
            k2 = (k + 1) % seg
            if len(a) == 1:
                faces.append((a[0], b[k2], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
            mats.append(mat_fn(j, k))
    for ids, flip, j in ((rings[0], False, 0), (rings[-1], True, len(rings) - 1)):
        if len(ids) > 1:
            c = len(verts)
            verts.append((0.0, verts[ids[0]][1], 0.0))
            for k in range(seg):
                f = (c, ids[(k + 1) % seg], ids[k])
                faces.append(f[::-1] if flip else f)
                mats.append(mat_fn(j, k))
    base = len(geo.verts)
    for v in verts:
        p = Vector(v)
        if matrix is not None:
            p = matrix @ p
        geo.verts.append(tuple(p))
    for f, m in zip(faces, mats):
        geo.faces.append(tuple(base + i for i in f))
        geo.mats.append(m)
    return geo


def ellipsoid_z(center, rx, ry, rz, seg, rings, geo, mat=0, pitch=0.0):
    m = g.trans(*center) @ g.rot_x(pitch) @ g.rot_x(math.pi / 2) @ g.scale3(rx, rz, ry)
    return lathe(g.sphere_profile(rings), seg, geo, matrix=m,
                 mat_fn=(lambda j, k: mat(j)) if callable(mat) else (lambda j, k: mat))


def sphere(center, r, geo, mat=0, seg=12, rings=8, sy=1.0):
    return g.ellipsoid(center, (r, r * sy, r), seg, rings, mat=mat, geo=geo)


def mat(name, color, **kw):
    return g.material('nf_env_' + name, color, **kw)


def bezier(p0, p1, p2, t):
    return tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c for a, b, c in zip(p0, p1, p2))


# ------------------------------------------------------------------ rig bookkeeping
JOINTS = []                     # contract rows
MESHES = []                     # (name, geo, mats, smooth, parent_key)
NODES = {}


def chain(name, parent_key, pivot, dofs, note=''):
    """<name>_root -> DOF nodes -> <name>_offset; returns the key of the offset node.

    dofs: [(dof_name, axis_three, (lo, hi), meaning)].  Limits are DISPLAY limits (artistic, labelled)."""
    piv = Vector(pivot)
    root = g.empty(name + '_root', g.trans(*piv), NODES[parent_key])
    NODES[name + '_root'] = root
    JOINTS.append({'node': name + '_root', 'parent': parent_key, 'translation': [round(c, 5) for c in piv],
                   'kind': 'pivot'})
    prev = name + '_root'
    for dof, axis, lim, meaning in dofs:
        ax = Vector(axis).normalized()
        node = g.empty(f'{name}_{dof}', None, NODES[prev])
        NODES[f'{name}_{dof}'] = node
        JOINTS.append({'node': f'{name}_{dof}', 'parent': prev, 'axis': [round(c, 5) for c in ax],
                       'range_rad': list(lim), 'rest_rad': 0.0, 'note': meaning + (('; ' + note) if note else '')})
        prev = f'{name}_{dof}'
    off = g.empty(name + '_offset', g.trans(*(-piv)), NODES[prev])
    NODES[name + '_offset'] = off
    JOINTS.append({'node': name + '_offset', 'parent': prev, 'translation': [round(-c, 5) for c in piv],
                   'kind': 'offset'})
    return name + '_offset'


def leg_chain(prefix, leg, A, K, B, side, out, lims):
    """Hip (yaw, lift) at A, knee at K, ankle at B.  + yaw = protraction (tip forward) on both sides;
    + lift / knee / ankle = rotation of the distal part upward in the leg plane."""
    u = Vector(out).normalized()
    n = u.cross(Vector((0, 1, 0))).normalized()          # +rotation about n turns u toward +Y
    yaw_axis = (0, 1, 0) if side < 0 else (0, -1, 0)
    hip = chain(f'{prefix}_{leg}_hip', f'{prefix}_body_offset', A,
                [('yaw', yaw_axis, lims['yaw'], '+ = protraction (tip forward)'),
                 ('lift', tuple(n), lims['lift'], '+ = levation (leg raised)')])
    knee = chain(f'{prefix}_{leg}_knee', hip, K, [('flex', tuple(n), lims['knee'], '+ = distal part up')])
    ankle = chain(f'{prefix}_{leg}_ankle', knee, B, [('flex', tuple(n), lims['ankle'], '+ = distal part up')])
    return hip, knee, ankle, tuple(n)


root = g.empty('env_jumping_spider' if ANIMAL == 'spider' else 'env_mantis_nymph')
NODES['root'] = root
P = 'sp' if ANIMAL == 'spider' else 'mn'
motion = g.empty(f'{P}_motion', None, root)
NODES[f'{P}_motion'] = motion
JOINTS.append({'node': f'{P}_motion', 'parent': 'root', 'kind': 'locomotion',
               'note': 'root-motion carrier: translation in the floor plane (x, z) and yaw about +Y; identity in '
                       'in-place clips'})
shift = g.empty(f'{P}_body_shift', None, motion)
NODES[f'{P}_body_shift'] = shift
JOINTS.append({'node': f'{P}_body_shift', 'parent': f'{P}_motion', 'kind': 'translation',
               'translation_axes': {'x': [-3.0, 3.0], 'y': [-0.6, 0.6]},
               'note': 'body translation relative to the planted feet (lateral peering sway x, bob y); '
                       'the only translating DOF besides <p>_motion; legs re-solve to keep feet planted'})
LEG_INFO = {}

if ANIMAL == 'spider':
    M = [mat('spider_black', (0.025, 0.022, 0.02), roughness=0.55),
         mat('spider_white', (0.80, 0.78, 0.72), roughness=0.8),
         mat('spider_eye', (0.01, 0.01, 0.012), roughness=0.04),
         mat('spider_brown', (0.16, 0.10, 0.06), roughness=0.6),
         mat('spider_lens', (0.30, 0.18, 0.06), roughness=0.03)]
    body_pivot = (0.0, 3.0, 0.0)
    bk = chain('sp_body', f'{P}_body_shift', body_pivot,
               [('pitch', (1, 0, 0), (-0.15, 0.15), '+ = nose down'),
                ('roll', (0, 0, 1), (-0.12, 0.12), '+ = left side up'),
                ('yaw', (0, 1, 0), (-0.2, 0.2), '+ = turn left (toward +X)')])
    body = g.Geo()
    ellipsoid_z((0, 3.2, 3.0), 2.9, 2.0, 4.1, 28, 16, body, mat=lambda j: 1 if j in (9, 10) else 0, pitch=-0.06)
    ellipsoid_z((0, 2.5, -0.9), 0.7, 0.6, 0.9, 10, 6, body, mat=0)
    ellipsoid_z((0, 3.0, -5.0), 2.9, 2.3, 4.0, 28, 18,
                body, mat=lambda j: 1 if j in (4, 5, 9, 10, 14) else 0, pitch=0.12)
    MESHES.append(('spider_body', body, M, True, bk))
    ey = g.Geo()
    for sx in (-1, 1):
        sphere((sx * 0.98, 3.75, 6.95), 0.88, ey, mat=4, seg=16, rings=10)
        sphere((sx * 2.05, 4.05, 6.55), 0.45, ey, mat=2, seg=12, rings=8)
        sphere((sx * 2.35, 4.75, 5.2), 0.17, ey, mat=2, seg=8, rings=5)
        sphere((sx * 2.4, 5.0, 3.6), 0.4, ey, mat=2, seg=12, rings=8)
    MESHES.append(('spider_eyes', ey, M, True, bk))
    for sx, side in ((-1, 'L'), (1, 'R')):
        mo = g.Geo()
        g.segment_between((sx * 0.6, 2.4, 6.7), (sx * 0.55, 1.4, 7.3), 0.45, 0.3, 10, mat=3, geo=mo)
        MESHES.append((f'spider_chelicera_{side}', mo, M, True, bk))
        palp = chain(f'sp_palp_{side}', bk, (sx * 1.2, 2.2, 6.6),
                     [('flick', (1, 0, 0), (-0.35, 0.25), '+ = palp tip down/back (palp signalling, idle only)')])
        pg = g.Geo()
        g.segment_between((sx * 1.2, 2.2, 6.6), (sx * 1.6, 2.0, 7.9), 0.28, 0.26, 10, mat=0, geo=pg)
        g.segment_between((sx * 1.6, 2.0, 7.9), (sx * 1.7, 1.0, 8.6), 0.26, 0.3, 10, mat=1, geo=pg)
        sphere((sx * 1.72, 0.85, 8.65), 0.32, pg, mat=1, seg=10, rings=6)
        MESHES.append((f'spider_palp_{side}', pg, M, True, palp))
    legs = [(5.0, 30, 7.8, 0.55), (3.8, 72, 6.6, 0.44), (2.5, 112, 6.5, 0.42), (1.1, 150, 8.0, 0.46)]
    lims = {'yaw': (-0.9, 0.9), 'lift': (-0.6, 0.9), 'knee': (-1.0, 1.0), 'ankle': (-1.2, 1.2)}
    for sx, side in ((-1, 'L'), (1, 'R')):
        for i, (z0, yawdeg, L, r) in enumerate(legs):
            leg = f'{side}{i + 1}'
            a = math.radians(yawdeg)
            dx, dz = sx * math.sin(a), math.cos(a)
            A = Vector((sx * 2.1, 2.5, z0))
            out = Vector((dx, 0, dz))
            K = A + out * (L * 0.36) + Vector((0, 3.0, 0))
            Pp = K + out * (L * 0.12) + Vector((0, -0.1, 0))
            B = A + out * (L * 0.82) + Vector((0, -1.3, 0))
            F = A + out * (L * 1.02) + Vector((0, -2.3, 0))
            hip, knee, ankle, n = leg_chain('sp', leg, A, K, B, sx, out, lims)
            fg, sg, tg = g.Geo(), g.Geo(), g.Geo()
            g.segment_between(A, K, r, r * 0.9, 10, mat=0, geo=fg)
            sphere(tuple(K), r * 0.9, sg, mat=1, seg=8, rings=5)
            g.segment_between(K, Pp, r * 0.9, r * 0.82, 10, mat=1, geo=sg)
            g.segment_between(Pp, B, r * 0.82, r * 0.6, 10, mat=0, geo=sg)
            sphere(tuple(B), r * 0.6, tg, mat=1, seg=8, rings=5)
            g.segment_between(B, F, r * 0.6, r * 0.42, 10, mat=3, geo=tg)
            MESHES += [(f'sp_{leg}_femur', fg, M, True, hip), (f'sp_{leg}_shank', sg, M, True, knee),
                       (f'sp_{leg}_tarsus', tg, M, True, ankle)]
            LEG_INFO[leg] = {'A': list(A), 'K': list(K), 'B': list(B), 'F': list(F), 'out': list(out),
                             'plane_normal': list(n), 'side': sx, 'radius_tip': r * 0.42}
    meta = {'depicts': 'zebra jumping spider (Salticidae; Salticus scenicus style)',
            'scale': 'fly display scale (~3.3x life); ~6 mm spider -> ~20 viewport-mm body length',
            'legs': 'eight walking legs L1..L4 / R1..R4 (spider legs I-IV); pedipalps are not legs'}
else:
    M = [mat('mantis_green', (0.22, 0.40, 0.10), roughness=0.5),
         mat('mantis_dark', (0.20, 0.28, 0.08), roughness=0.5),
         mat('mantis_eye', (0.45, 0.55, 0.22), roughness=0.2),
         mat('mantis_pupil', (0.02, 0.02, 0.02), roughness=0.3),
         mat('mantis_spine', (0.12, 0.10, 0.05), roughness=0.4)]
    bk = chain('mn_body', f'{P}_body_shift', (0.0, 4.6, 0.0),
               [('pitch', (1, 0, 0), (-0.12, 0.12), '+ = front down'),
                ('roll', (0, 0, 1), (-0.15, 0.15), '+ = left side up'),
                ('yaw', (0, 1, 0), (-0.2, 0.2), '+ = turn left (toward +X)')])
    bd = g.Geo()
    p0, p1, p2 = (0, 4.6, -1.6), (0, 3.9, -10.5), (0, 9.8, -15.5)
    n9 = 9
    for i in range(n9):
        a, b = bezier(p0, p1, p2, i / n9), bezier(p0, p1, p2, (i + 1) / n9)
        r0 = 1.5 - 0.9 * (i / n9) + 0.55 * math.sin(math.pi * i / n9)
        r1 = 1.5 - 0.9 * ((i + 1) / n9) + 0.55 * math.sin(math.pi * (i + 1) / n9)
        g.segment_between(a, b, r0 * 0.94, r1, 14, mat=1 if i % 2 else 0, geo=bd)
    g.segment_between((0, 4.6, -1.6), (0, 4.8, 3.2), 1.0, 0.85, 14, mat=0, geo=bd)
    g.segment_between((0, 4.8, 3.2), (0, 6.2, 5.0), 0.85, 0.9, 14, mat=0, geo=bd)
    g.segment_between((0, 6.2, 5.0), (0, 10.4, 10.6), 0.8, 0.62, 14, mat=1, geo=bd)
    for sx in (-1, 1):
        m = g.trans(sx * 0.75, 5.6, 0.2) @ g.rot_x(0.15) @ g.scale3(0.6, 0.18, 1.6)
        lathe(g.sphere_profile(6), 10, bd, matrix=m, mat_fn=lambda j, k: 1)
    MESHES.append(('mantis_body', bd, M, True, bk))
    neck = chain('mn_neck', bk, (0.0, 10.5, 10.8),
                 [('yaw', (0, 1, 0), (-0.6, 0.6), '+ = head turns left (toward +X)'),
                  ('pitch', (1, 0, 0), (-0.3, 0.3), '+ = head down')])
    hd = g.Geo()
    m = g.trans(0, 11.2, 11.6) @ g.rot_x(-0.35) @ g.scale3(2.3, 1.7, 0.95)
    lathe(g.sphere_profile(10), 18, hd, matrix=m, mat_fn=lambda j, k: 0)
    g.segment_between((0, 10.6, 11.9), (0, 9.4, 12.3), 0.55, 0.2, 10, mat=1, geo=hd)
    for sx in (-1, 1):
        sphere((sx * 2.25, 11.9, 11.6), 1.0, hd, mat=2, seg=14, rings=9)
        sphere((sx * 2.45, 11.85, 12.5), 0.25, hd, mat=3, seg=8, rings=5)
        pts = [(sx * 0.45, 12.4, 12.2), (sx * 1.8, 14.6, 15.6), (sx * 3.6, 15.0, 19.0), (sx * 5.4, 14.0, 21.6)]
        for a, b in zip(pts, pts[1:]):
            g.segment_between(a, b, 0.08, 0.06, 6, mat=1, geo=hd)
    MESHES.append(('mantis_head', hd, M, True, neck))
    for sx, side in ((-1, 'L'), (1, 'R')):
        S = Vector((sx * 0.65, 9.2, 9.7))
        C = Vector((sx * 1.05, 6.6, 12.0))
        Fe = Vector((sx * 1.2, 10.3, 14.3))
        Ti = Vector((sx * 1.05, 8.1, 12.7))
        Ta = Vector((sx * 1.0, 7.4, 13.9))
        nrm = (C - S).cross(Fe - C).normalized()            # plane of the folded raptorial leg
        if nrm.x * sx < 0:
            nrm = -nrm
        cx = chain(f'mn_{side}1_coxa', bk, S, [('swing', tuple(nrm), (-0.4, 0.4), '+ = coxa swings (raptorial leg '
                                                                                   'raised/lowered as a unit)')])
        fe = chain(f'mn_{side}1_femur', cx, C, [('flex', tuple(nrm), (-0.3, 0.3), 'coxa-femur; held in clips')])
        ti = chain(f'mn_{side}1_tibia', fe, Fe, [('flex', tuple(nrm), (-0.3, 0.3), 'femur-tibia; held folded in '
                                                                                  'clips (no strike)')])
        cg, fg, tg = g.Geo(), g.Geo(), g.Geo()
        g.segment_between(S, C, 0.62, 0.5, 12, mat=0, geo=cg)
        sphere(tuple(C), 0.4, fg, mat=0, seg=10, rings=6)
        g.segment_between(C, Fe, 0.72, 0.42, 12, mat=0, geo=fg)
        sphere(tuple(Fe), 0.3, tg, mat=1, seg=8, rings=5)
        g.segment_between(Fe, Ti, 0.4, 0.3, 10, mat=1, geo=tg)
        g.segment_between(Ti, Ta, 0.14, 0.1, 8, mat=1, geo=tg)
        for i in range(1, 7):
            p = C.lerp(Fe, i / 7.0)
            g.segment_between(tuple(p), tuple(p + Vector((0, -0.35, -0.95))), 0.14, 0.0, 6, mat=4, geo=fg)
        MESHES += [(f'mn_{side}1_coxa_mesh', cg, M, True, cx), (f'mn_{side}1_femur_mesh', fg, M, True, fe),
                   (f'mn_{side}1_tibia_mesh', tg, M, True, ti)]
    lims = {'yaw': (-0.7, 0.7), 'lift': (-0.5, 0.7), 'knee': (-0.8, 0.8), 'ankle': (-1.0, 1.0)}
    for sx, side in ((-1, 'L'), (1, 'R')):
        for idx, (A, K, F, T) in ((2, ((sx * 0.8, 4.4, 1.6), (sx * 6.2, 6.8, 4.4), (sx * 9.2, 0.15, 6.4),
                                       (sx * 9.7, 0.1, 7.6))),
                                  (3, ((sx * 0.8, 4.3, -0.9), (sx * 6.8, 7.1, -4.4), (sx * 9.9, 0.15, -9.6),
                                       (sx * 10.3, 0.1, -10.8)))):
            A, K, F, T = Vector(A), Vector(K), Vector(F), Vector(T)
            leg = f'{side}{idx}'
            out = Vector((F.x - A.x, 0, F.z - A.z)).normalized()
            hip, knee, ankle, n = leg_chain('mn', leg, A, K, F, sx, out, lims)
            fg, sg, tg = g.Geo(), g.Geo(), g.Geo()
            g.segment_between(A, K, 0.4, 0.32, 10, mat=0, geo=fg)
            sphere(tuple(K), 0.27, sg, mat=1, seg=8, rings=5)
            g.segment_between(K, F, 0.3, 0.2, 10, mat=0, geo=sg)
            g.segment_between(F, T, 0.14, 0.08, 6, mat=1, geo=tg)
            MESHES += [(f'mn_{leg}_femur', fg, M, True, hip), (f'mn_{leg}_tibia', sg, M, True, knee),
                       (f'mn_{leg}_tarsus', tg, M, True, ankle)]
            LEG_INFO[leg] = {'A': list(A), 'K': list(K), 'B': list(F), 'F': list(T), 'out': list(out),
                             'plane_normal': list(n), 'side': sx, 'radius_tip': 0.08}
    meta = {'depicts': 'praying mantis nymph (Mantidae; Tenodera sinensis style)',
            'scale': 'fly display scale (~3.3x life); ~11 mm early nymph -> ~37 viewport-mm',
            'legs': 'walking legs L2, L3, R2, R3; raptorial forelegs L1/R1 are held (not walking legs)'}

objs = []
for name, geo, mats, smooth, parent in MESHES:
    objs.append(g.mesh_object(name, geo, mats, smooth=smooth, parent=NODES[parent]))
root['neurofly_status'] = ('ILLUSTRATIVE: no predator stimulus or behaviour is implemented; decoration only and '
                           'must not change looming or retinal input')
root['neurofly_rig'] = f'neurofly-env-{ANIMAL}-rig-v1'
root['neurofly_frame'] = 'three.js +Y up, +Z forward, 1 unit = 1 viewport-mm (fly display scale), stands on y = 0'
tris = g.triangle_count(objs)
stem = os.path.join(ARGS['out'], f'env_{"jumping_spider" if ANIMAL == "spider" else "mantis_nymph"}_rig')
g.export_glb(stem + '.glb')
bpy.ops.wm.save_as_mainfile(filepath=stem + '.blend')
# Bind pose must reproduce the static asset exactly (same vertices in the root frame).
check = None
if ARGS['static']:
    bpy.context.view_layer.update()
    mine = {tuple(round(c, 3) + 0.0 for c in (o.matrix_world @ v.co)) for o in objs for v in o.data.vertices}
    sfile = os.path.join(ARGS['static'], f'env_{"jumping_spider" if ANIMAL == "spider" else "mantis_nymph"}.glb')
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=sfile)
    new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    bpy.context.view_layer.update()
    ref = {tuple(round(c, 3) + 0.0 for c in (o.matrix_world @ v.co)) for o in new for v in o.data.vertices}
    check = {'static_file': os.path.basename(sfile), 'unique_positions_rig': len(mine),
             'unique_positions_static': len(ref), 'only_in_rig': len(mine - ref), 'only_in_static': len(ref - mine),
             'rounding_viewport_mm': 0.001, 'triangles_rig': tris}
    for o in new:
        bpy.data.objects.remove(o)
    print('NEUROFLY_RIG static check', check)
with open(stem + '_joints.json', 'w') as fh:
    json.dump({'rig': root['neurofly_rig'], 'frame': root['neurofly_frame'], 'units': 'viewport-mm, radians',
               'triangles': tris, 'bind_pose': 'every DOF = 0 (identity); equals the static asset',
               'static_check': check, 'meta': meta, 'legs': LEG_INFO, 'joints': JOINTS}, fh, indent=2)
print(f'NEUROFLY_RIG {ANIMAL} triangles={tris} joints={sum(1 for j in JOINTS if "axis" in j)} dofs')
