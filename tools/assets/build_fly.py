# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Generate the stylised NeuroFly presentation fly (Drosophila melanogaster).

Headless:  blender --background --factory-startup --threads 2 \
               --python tools/assets/build_fly.py -- --out OUTDIR [--lod 0|1] [--render 1]

Writes OUTDIR/fly_hq_lod<N>.glb and OUTDIR/fly_hq_lod<N>.blend; with --render 1 also
transparent orthographic PNGs (top/side/front) and a 3/4 hero PNG rendered from the very
same objects that were exported.

Every part is placed in the frame of the procedural fly in web/app.js
(ArticulatedFly3DViewport.buildFlyMesh) so the opt-in loader can hang each mesh on
the existing pose groups without changing any transform.  See tools/assets/INTERFACE.md.

This is a stylised presentation surface, not an anatomical reconstruction: wings,
halteres, antennae, proboscis and bristles are static decoration and are not simulated.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import nf_geom as g  # noqa: E402

ARGS = g.parse_args({'out': '', 'lod': 0, 'render': 0, 'samples': 48, 'res': 768})
LOD = int(ARGS['lod'])
SEG = 28 if LOD == 0 else 12          # around each body ellipsoid
RINGS = 18 if LOD == 0 else 8
LSEG = 12 if LOD == 0 else 6          # around each leg tube
EYE_SUBDIV = 3 if LOD == 0 else 2
DETAIL = LOD == 0                      # bristles, leg setae, arista branches

# Leg layout copied verbatim from ArticulatedFly3DViewport.buildFlyMesh (web/app.js).
# Viewport order is L1, L2, L3, R1, R2, R3 = FlyGym lf, lm, lh, rf, rm, rh.
LEGS = [
    ('lf', -1, 0.8, -math.pi / 4), ('lm', -1, 0.0, -math.pi / 2), ('lh', -1, -0.8, -3 * math.pi / 4),
    ('rf', 1, 0.8, math.pi / 4), ('rm', 1, 0.0, math.pi / 2), ('rh', 1, -0.8, 3 * math.pi / 4),
]
THORAX_Y = 2.2                         # thoraxMesh.position.y in flyGroup
COXA_LEN, FEMUR_LEN, TIBIA_TARSUS_LEN = 1.0, 2.2, 2.4
TIBIA_LEN = 1.55                       # tarsus occupies the last 0.85 of the tibia group

scene = g.reset_scene()

M = {
    'cuticle': g.material('nf_cuticle_tan', (0.42, 0.26, 0.11), roughness=0.42),
    'scutum': g.material('nf_scutum_brown', (0.30, 0.18, 0.08), roughness=0.38),
    'head': g.material('nf_head_tan', (0.48, 0.30, 0.12), roughness=0.45),
    'eye': g.material('nf_eye_red', (0.50, 0.02, 0.015), roughness=0.28),
    'abd': g.material('nf_abdomen_tan', (0.55, 0.38, 0.16), roughness=0.5),
    'band': g.material('nf_abdomen_band', (0.07, 0.045, 0.03), roughness=0.45),
    'leg': g.material('nf_leg_tan', (0.45, 0.30, 0.13), roughness=0.5),
    'joint': g.material('nf_leg_joint', (0.25, 0.15, 0.07), roughness=0.5),
    'bristle': g.material('nf_bristle', (0.03, 0.02, 0.015), roughness=0.6),
    'wing': g.material('nf_wing_membrane', (0.80, 0.86, 0.92), roughness=0.15, alpha=0.28),
    'vein': g.material('nf_wing_vein', (0.28, 0.19, 0.10), roughness=0.4),
    'haltere': g.material('nf_haltere', (0.62, 0.50, 0.30), roughness=0.4),
}

fly = g.empty('neurofly_fly')                                   # == flyGroup frame
thorax_frame = g.empty('thorax_frame', g.trans(0, THORAX_Y, 0), fly)   # == thoraxMesh frame
parts = []


def body_part(name, geo, mats, smooth=True):
    obj = g.mesh_object(name, geo, mats, smooth=smooth, parent=thorax_frame)
    parts.append(obj)
    return obj


# ------------------------------------------------------------------ thorax
geo = g.Geo()
g.ellipsoid((0, 0.05, 0.05), (1.12, 1.18, 1.55), SEG, RINGS, geo=geo)
body_part('c_thorax', geo, [M['cuticle']])
# Scutum (dorsal plate) and scutellum, darker; slightly proud of the thorax surface.
geo = g.Geo()
g.ellipsoid((0, 0.55, 0.15), (0.92, 0.72, 1.25), SEG, RINGS, geo=geo)
g.ellipsoid((0, 0.92, -1.25), (0.52, 0.26, 0.42), SEG // 2, RINGS // 2, geo=geo)
body_part('c_scutum', geo, [M['scutum']])

if DETAIL:
    # Dorsocentral and scutellar macrochaetae: thin cones swept backward.
    geo = g.Geo()
    for x, z, length in [(-0.35, 0.55, 0.55), (0.35, 0.55, 0.55), (-0.38, -0.25, 0.6), (0.38, -0.25, 0.6),
                         (-0.75, 0.6, 0.45), (0.75, 0.6, 0.45), (-0.85, -0.3, 0.45), (0.85, -0.3, 0.45),
                         (-0.28, -1.25, 0.75), (0.28, -1.25, 0.75), (-0.5, -1.05, 0.6), (0.5, -1.05, 0.6)]:
        y = 1.12 if z > -1.0 else 1.06
        base = Vector((x, y, z))
        tip = base + Vector((x * 0.25, 0.32, -1.0)).normalized() * length
        g.segment_between(base, tip, 0.028, 0.004, 5, geo=geo)
    body_part('c_thorax_bristles', geo, [M['bristle']])

# ------------------------------------------------------------------ head
HEAD = Vector((0, 0.25, 1.75))
geo = g.Geo()
g.ellipsoid(HEAD, (0.92, 0.80, 0.55), SEG, RINGS, matrix=None, geo=geo)
body_part('c_head', geo, [M['head']])

# Compound eyes: faceted icospheres (flat shading suggests ommatidia).
for side, name in ((-1, 'l_eye'), (1, 'r_eye')):
    m = g.trans(HEAD.x + side * 0.6, HEAD.y + 0.05, HEAD.z + 0.08) @ g.rot_z(side * 0.25) @ g.scale3(0.42, 0.62, 0.5)
    verts, faces = g.icosphere(EYE_SUBDIV, m)
    body_part(name, g.Geo().extend(verts, faces), [M['eye']], smooth=False)

# Antennae: pedicel, funiculus (third segment) and feathered arista.
for side, s in (('l', -1), ('r', 1)):
    base = Vector((s * 0.17, HEAD.y + 0.35, HEAD.z + 0.48))
    ped_tip = base + Vector((s * 0.06, 0.02, 0.2))
    geo = g.Geo()
    g.segment_between(base, ped_tip, 0.09, 0.08, LSEG, geo=geo)
    body_part(f'{side}_pedicel', geo, [M['head']])
    geo = g.Geo()
    fun_c = ped_tip + Vector((0, -0.18, 0.06))
    g.ellipsoid(fun_c, (0.1, 0.22, 0.1), LSEG, max(6, RINGS // 2), matrix=None, geo=geo)
    body_part(f'{side}_funiculus', geo, [M['cuticle']])
    geo = g.Geo()
    a0 = fun_c + Vector((s * 0.08, 0.08, 0.04))
    a1 = a0 + Vector((s * 0.35, 0.28, 0.45))
    g.segment_between(a0, a1, 0.022, 0.006, 5, geo=geo)
    if DETAIL:
        axis = (a1 - a0)
        for k in range(1, 7):
            p = a0 + axis * (k / 7.0)
            for up in (1, -1):
                g.segment_between(p, p + Vector((0, up * 0.11, 0.03)), 0.008, 0.002, 3, geo=geo)
    body_part(f'{side}_arista', geo, [M['bristle']])

# Proboscis (rostrum + haustellum + labellum), retracted under the head.
geo = g.Geo()
r0 = Vector((0, HEAD.y - 0.55, HEAD.z + 0.15))
r1 = r0 + Vector((0, -0.32, 0.18))
g.segment_between(r0, r1, 0.16, 0.12, LSEG, geo=geo)
g.ellipsoid(r1 + Vector((0, -0.07, 0.04)), (0.16, 0.1, 0.13), LSEG, max(6, RINGS // 2), geo=geo)
body_part('c_proboscis', geo, [M['head']])

if DETAIL:
    geo = g.Geo()
    for x, z in [(-0.3, 1.6), (0.3, 1.6), (-0.15, 1.85), (0.15, 1.85), (-0.55, 1.45), (0.55, 1.45)]:
        base = Vector((x, HEAD.y + 0.76, z))
        g.segment_between(base, base + Vector((x * 0.3, 0.3, -0.25)), 0.02, 0.003, 4, geo=geo)
    body_part('c_head_bristles', geo, [M['bristle']])

# ------------------------------------------------------------------ abdomen (5 visible tergites)
TERGITES = [('c_abdomen12', 0.0, 0.62), ('c_abdomen3', 0.62, 1.30), ('c_abdomen4', 1.30, 1.98),
            ('c_abdomen5', 1.98, 2.62), ('c_abdomen6', 2.62, 3.45)]
ABD_TOP_Z = -1.0                       # anterior edge of tergite 1/2, tucked under the thorax
ABD_LEN = 3.45


def abd_radius(t):
    u = t / ABD_LEN
    return max(0.0, 1.0 * math.sin(math.pi * min(1.0, 0.18 + 0.82 * u)) ** 0.7 * (1.0 - 0.15 * u))


for name, t0, t1 in TERGITES:
    n = max(3, int(round((t1 - t0) / ABD_LEN * RINGS * 2)))
    prof = []
    for i in range(n + 1):
        t = t0 + (t1 - t0) * i / n
        lip = 1.04 if i == 0 and t0 > 0 else 1.0     # overlapping anterior lip reads as segmentation
        r = abd_radius(t) * lip
        if name == 'c_abdomen6' and i == n:
            r = 0.0
        prof.append((t, max(r, 0.02) if not (name == 'c_abdomen6' and i == n) else 0.0))
    dark_from = int(n * 0.55)
    geo = g.Geo()
    m = g.trans(0, -0.15, ABD_TOP_Z) @ g.rot_x(math.pi / 2)     # lathe axis -Y -> -Z (posterior)
    g.tube_down(prof, SEG, matrix=m, sx=1.0, sz=0.86, mat=(lambda j, d=dark_from: 1 if j >= d else 0), geo=geo)
    body_part(name, geo, [M['abd'], M['band']])

# ------------------------------------------------------------------ wings (folded at rest)
VEINS = {  # (s along span 0..1, c chord fraction: +1 anterior margin, -1 posterior margin)
    'costa': [(0.02, 0.85), (0.2, 1.0), (0.45, 1.0), (0.7, 0.92), (0.9, 0.6), (0.99, 0.15)],
    'L2': [(0.05, 0.5), (0.3, 0.6), (0.55, 0.75), (0.75, 0.85)],
    'L3': [(0.05, 0.3), (0.4, 0.3), (0.7, 0.28), (0.97, 0.12)],
    'L4': [(0.08, 0.0), (0.4, -0.12), (0.7, -0.2), (0.96, -0.3)],
    'L5': [(0.12, -0.3), (0.4, -0.55), (0.62, -0.72), (0.78, -0.85)],
    'acv': [(0.42, 0.29), (0.42, -0.12)],
    'pcv': [(0.6, -0.17), (0.56, -0.66)],
}
WING_LEN, WING_W = 4.0, 1.45


def wing_frame(side):
    hinge = Vector((side * 0.62, 0.95, -0.15))
    yaw = 0.25
    span = Vector((side * math.sin(yaw), -0.12, -math.cos(yaw))).normalized()
    up = Vector((0, 1, 0))
    chord = up.cross(span).normalized() * (-side)       # anterior margin faces outward
    normal = span.cross(chord).normalized()
    if normal.y < 0:
        normal = -normal
    return hinge, span, chord, normal


def wing_point(side, s, c):
    hinge, span, chord, normal = wing_frame(side)
    half = WING_W / 2 * math.sin(math.pi * min(0.999, 0.04 + 0.96 * s)) ** 0.55
    off = (0.45 if c >= 0 else 0.55) * 2 * half * c
    lift = 0.06 * math.sin(math.pi * s)                  # gentle camber
    return hinge + span * (s * WING_LEN) + chord * off + normal * lift


for side, name in ((-1, 'l_wing'), (1, 'r_wing')):
    n = 24 if LOD == 0 else 10
    verts, faces = [], []
    for i in range(n + 1):
        s = i / n
        verts += [tuple(wing_point(side, s, 1.0)), tuple(wing_point(side, s, -1.0))]
    for i in range(n):
        faces.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
    body_part(name, g.Geo().extend(verts, faces), [M['wing']], smooth=True)
    _, _, _, normal = wing_frame(side)
    geo = g.Geo()
    for vein, pts in VEINS.items():
        if LOD and vein in ('acv', 'pcv'):
            continue
        poly = [wing_point(side, s, c) + normal * 0.006 for (s, c) in pts]
        g.ribbon(poly, 0.035 if vein == 'costa' else 0.026, normal, geo=geo)
    obj = body_part(name + '_veins', geo, [M['vein']], smooth=False)

# ------------------------------------------------------------------ halteres
for side, name in ((-1, 'l_haltere'), (1, 'r_haltere')):
    base = Vector((side * 0.82, 0.35, -1.0))
    tip = base + Vector((side * 0.22, -0.12, -0.32))
    geo = g.Geo()
    g.segment_between(base, tip, 0.05, 0.035, LSEG // 2 + 2, geo=geo)
    g.ellipsoid(tip, (0.12, 0.11, 0.15), LSEG, max(6, RINGS // 2), geo=geo)
    body_part(name, geo, [M['haltere']])

# ------------------------------------------------------------------ legs
def leg_scale(leg):
    return {'f': 1.0, 'm': 0.95, 'h': 1.08}[leg[1]]


# Display pose (radians about each joint's local X): coxa, femur, tibia.
DISPLAY_POSE = {'f': (-0.35, -0.75, 1.55), 'm': (-0.45, -0.8, 1.6), 'h': (-0.4, -0.75, 1.5)}
leg_objs = []
for leg, side, z_off, base_angle in LEGS:
    k = leg_scale(leg)
    # Joint origins and the coxa yaw are the viewport's.  The extra rotations about the
    # local X axis (whose +Z points radially outward after the yaw) are a DISPLAY POSE
    # for previews and 2D sprites only; the dashboard loader ignores these empties and
    # keeps applying its own pose transforms to its own groups.
    splay = DISPLAY_POSE[leg[1]]
    coxa_joint = g.empty(f'{leg}_coxa_joint', g.trans(side * 1.0, -0.2, z_off) @ g.rot_y(base_angle)
                         @ g.rot_x(splay[0]), thorax_frame)
    femur_joint = g.empty(f'{leg}_femur_joint', g.trans(0, -COXA_LEN, 0) @ g.rot_x(splay[1]), coxa_joint)
    tibia_joint = g.empty(f'{leg}_tibia_joint', g.trans(0, -FEMUR_LEN, 0) @ g.rot_x(splay[2]), femur_joint)

    geo = g.Geo()
    g.tube_down([(0.0, 0.0), (0.0, 0.25 * k), (0.3, 0.29 * k), (0.75, 0.24 * k), (COXA_LEN, 0.2 * k),
                 (COXA_LEN, 0.0)], LSEG, geo=geo)
    leg_objs.append(g.mesh_object(f'{leg}_coxa', geo, [M['leg']], parent=coxa_joint))

    geo = g.Geo()
    # trochanter bulb, then the femur with a mid-shaft swelling.
    g.tube_down([(0.0, 0.0), (0.0, 0.19 * k), (0.18, 0.21 * k), (0.34, 0.15 * k)], LSEG, mat=1, geo=geo)
    fem = [(0.34, 0.15 * k), (0.6, 0.19 * k), (1.2, 0.205 * k), (1.8, 0.17 * k), (FEMUR_LEN, 0.13 * k),
           (FEMUR_LEN, 0.0)]
    g.tube_down(fem, LSEG, geo=geo)
    leg_objs.append(g.mesh_object(f'{leg}_trochanterfemur', geo, [M['leg'], M['joint']], parent=femur_joint))

    geo = g.Geo()
    g.tube_down([(0.0, 0.0), (0.0, 0.13 * k), (0.12, 0.125 * k), (0.9, 0.11 * k), (TIBIA_LEN - 0.08, 0.115 * k),
                 (TIBIA_LEN, 0.09 * k), (TIBIA_LEN, 0.0)], LSEG, geo=geo)
    if DETAIL:
        for j in range(5):
            t = 0.3 + j * 0.24
            a = j * 2.1
            base = Vector((0.115 * k * math.cos(a), -t, 0.115 * k * math.sin(a)))
            tip = base + Vector((math.cos(a) * 0.12, -0.16, math.sin(a) * 0.12))
            g.segment_between(base, tip, 0.012, 0.002, 3, mat=1, geo=geo)
    leg_objs.append(g.mesh_object(f'{leg}_tibia', geo, [M['leg'], M['bristle']], parent=tibia_joint))

    geo = g.Geo()
    t = TIBIA_LEN
    for j, length in enumerate([0.3, 0.16, 0.13, 0.11, 0.1]):
        r = (0.085 - 0.006 * j) * k
        g.tube_down([(t, 0.0), (t, r * 0.8), (t + 0.03, r), (t + length - 0.03, r * 0.9), (t + length, r * 0.7),
                     (t + length, 0.0)], max(6, LSEG - 2), geo=geo)
        t += length
    # Paired pretarsal claws, splayed sideways in the leg's local frame.
    for s in (-1, 1):
        g.segment_between((0, -t + 0.02, 0), (s * 0.06, -TIBIA_TARSUS_LEN - 0.02, 0.03), 0.02, 0.004, 4, mat=1,
                          geo=geo)
    leg_objs.append(g.mesh_object(f'{leg}_tarsus', geo, [M['leg'], M['bristle']], parent=tibia_joint))

all_objs = parts + leg_objs
tris = g.triangle_count(all_objs)
print(f'NEUROFLY_ASSET fly lod={LOD} objects={len(all_objs)} triangles={tris}')

stem = os.path.join(ARGS['out'], f'fly_hq_lod{LOD}')
g.export_glb(stem + '.glb')
bpy.ops.wm.save_as_mainfile(filepath=stem + '.blend')

if int(ARGS['render']):
    g.setup_render(scene, (int(ARGS['res']), int(ARGS['res'])), samples=int(ARGS['samples']))
    g.studio_lights(scale=10.0)
    ORTHO = 12.0                       # viewport-mm across the frame -> px/mm = res / 12
    target = (0, THORAX_Y - 0.4, -0.6)
    views = {
        'top': g.camera('cam_top', (0, 40, -0.6), (0, 0, -0.6), ortho_scale=ORTHO),
        'side': g.camera('cam_side', (40, THORAX_Y - 0.4, -0.6), target, ortho_scale=ORTHO),
        'front': g.camera('cam_front', (0, THORAX_Y - 0.4, 40), target, ortho_scale=ORTHO),
        'hero': g.camera('cam_hero', (9.5, 8.0, 8.5), (0, 1.2, -0.8), lens=55),
    }
    for view, cam in views.items():
        g.render_to(scene, cam, f'{stem}_{view}.png')
    print(f'NEUROFLY_ASSET renders ortho_scale={ORTHO} px_per_viewport_mm={int(ARGS["res"]) / ORTHO:.3f}')
