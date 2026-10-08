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

ARGS = g.parse_args({'out': '', 'lod': 0, 'render': 0, 'samples': 48, 'res': 768, 'sex': 'neutral', 'rig': 'v1'})
RIG2 = ARGS['rig'] == 'v2'   # v2: articulated wing roots, antennae and halteres; bristles seated on the cuticle
if ARGS['rig'] not in ('v1', 'v2'):
    raise SystemExit('--rig must be v1 or v2')
SEX = ARGS['sex']
if SEX not in ('neutral', 'female', 'male'):
    raise SystemExit('--sex must be neutral, female or male')
# Appearance only.  Choosing a variant never selects a brain dataset, changes
# physiology or implies sex-specific behaviour (see tools/assets/SEX_VARIANTS.md).
# Display scale relative to the female: David et al. 2003 (J Genet 82:79-88) report
# female/male ratios of 1.15 (thorax length) and 1.16 (wing length) at 25 C.
DISPLAY_SCALE = {'neutral': 1.0, 'female': 1.0, 'male': round(1 / 1.15, 4)}[SEX]
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


def body_part(name, geo, mats, smooth=True, parent=None):
    obj = g.mesh_object(name, geo, mats, smooth=smooth, parent=parent or thorax_frame)
    parts.append(obj)
    return obj


# ------------------------------------------------------------------ rig v2 joints
# Each articulated part hangs on a chain  <part>_root (translation = pivot, in the
# thorax frame or its parent joint) -> one node per rotational DOF (rest = identity,
# rotate about the listed axis) -> <part>_offset (translation = -pivot) -> meshes,
# whose vertices stay in the thorax frame.  At rest the chain is the identity, so the
# bind pose equals the v1 geometry, and a loader that lifts meshes by name still
# places them correctly (static).  See tools/assets/JOINT_CONTRACT.md.
JOINTS = []


def joint_chain(prefix, pivot, dofs, parent=None, parent_pivot=None):
    parent = parent or thorax_frame
    base = Vector(pivot) - (Vector(parent_pivot) if parent_pivot is not None else Vector((0, 0, 0)))
    node = g.empty(prefix + '_root', g.trans(*base), parent)
    JOINTS.append({'node': prefix + '_root', 'parent': parent.name, 'translation': [round(v, 5) for v in base],
                   'kind': 'pivot'})
    for dof, axis, lo, hi, note in dofs:
        node = g.empty(f'{prefix}_{dof}', None, node)
        a = Vector(axis).normalized()
        JOINTS.append({'node': f'{prefix}_{dof}', 'parent': node.parent.name, 'axis': [round(v, 5) for v in a],
                       'range_rad': [lo, hi], 'rest_rad': 0.0, 'note': note})
    rot_leaf = node
    offset = g.empty(prefix + '_offset', g.trans(*(-Vector(pivot))), rot_leaf)
    JOINTS.append({'node': prefix + '_offset', 'parent': rot_leaf.name, 'translation': [round(-v, 5) for v in pivot],
                   'kind': 'offset'})
    return rot_leaf, offset


def seat(x, z, ellipsoids, embed=0.03):
    """Highest cuticle point above (x, z) over the given ellipsoids, sunk by ``embed``."""
    best = None
    for (cx, cy, cz), (rx, ry, rz) in ellipsoids:
        q = 1 - ((x - cx) / rx) ** 2 - ((z - cz) / rz) ** 2
        if q > 0:
            y = cy + ry * math.sqrt(q)
            best = y if best is None else max(best, y)
    return None if best is None else best - embed


THORAX_SHELLS = [((0, 0.05, 0.05), (1.12, 1.18, 1.55)), ((0, 0.55, 0.15), (0.92, 0.72, 1.25)),
                 ((0, 0.92, -1.25), (0.52, 0.26, 0.42))]

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
        if RIG2:   # v1 placed some bases up to ~0.37 above the cuticle (floating hairs)
            y = seat(x, z, THORAX_SHELLS)
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
    ant_parent = fun_parent = None
    if RIG2:
        ped_axis = (ped_tip - base).normalized()
        # Positive abduct swings the tip away from the midline on both sides.
        ant_rot, ant_parent = joint_chain(f'{side}_antenna', base, [
            ('abduct', (0, s, 0), -0.4, 0.8, 'flybody antenna_abduct range; + = tip away from midline'),
            ('extend', (1, 0, 0), -0.2, 0.5, 'flybody antenna_extend range; + = tip down/forward'),
            ('twist', tuple(ped_axis), -0.1, 0.09, 'flybody antenna_twist range; about the pedicel axis')])
        fun_c0 = ped_tip + Vector((0, -0.18, 0.06))
        fun_axis = (fun_c0 + Vector((0, -0.22, 0)) - ped_tip).normalized()
        _, fun_parent = joint_chain(f'{side}_funiculus', ped_tip, [
            ('rotate', tuple(fun_axis), -0.1, 0.1,
             'passive rotation of funiculus+arista about the funiculus long axis (Gopfert & Robert 2002); '
             'amplitude stylised')], parent=ant_rot, parent_pivot=base)
    geo = g.Geo()
    g.segment_between(base, ped_tip, 0.09, 0.08, LSEG, geo=geo)
    body_part(f'{side}_pedicel', geo, [M['head']], parent=ant_parent)
    geo = g.Geo()
    fun_c = ped_tip + Vector((0, -0.18, 0.06))
    g.ellipsoid(fun_c, (0.1, 0.22, 0.1), LSEG, max(6, RINGS // 2), matrix=None, geo=geo)
    body_part(f'{side}_funiculus', geo, [M['cuticle']], parent=fun_parent)
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
    body_part(f'{side}_arista', geo, [M['bristle']], parent=fun_parent)

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
        y = HEAD.y + 0.76
        if RIG2:
            y = seat(x, z, [((HEAD.x, HEAD.y, HEAD.z), (0.92, 0.80, 0.55))])
        base = Vector((x, y, z))
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


for name, t0, t1 in (TERGITES if SEX == 'neutral' else []):
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


# Sexed abdomens (stylised from the cited descriptions, see SEX_VARIANTS.md):
# * female: longer, wider, tapering to a point; seven visible segments, each tergite
#   with a dark posterior band (Williams et al. 2008: A5/A6 pigment restricted to a
#   posterior stripe in females; A7 female-specific, Wang et al. 2011).
# * male: shorter, rounder, blunt tip; A5 and A6 fully pigmented, so the tip reads as
#   one fused dark cap (Kopp et al. 2000; Williams et al. 2008); A7 externally absent.
# Node names stay the FlyGym bodies; A7 and the terminalia are part of c_abdomen6.
SEXED_ABDOMEN = {
    'female': dict(length=3.95, rmax=1.12, power=0.6, taper=0.10, segments=[
        ('c_abdomen12', [(0.0, 0.62, 'band')]), ('c_abdomen3', [(0.62, 1.30, 'band')]),
        ('c_abdomen4', [(1.30, 1.98, 'band')]), ('c_abdomen5', [(1.98, 2.66, 'band')]),
        ('c_abdomen6', [(2.66, 3.30, 'band'), (3.30, 3.82, 'band'), (3.82, 3.95, 'tip')])]),
    'male': dict(length=3.05, rmax=0.98, power=0.42, taper=0.05, segments=[
        ('c_abdomen12', [(0.0, 0.60, 'band')]), ('c_abdomen3', [(0.60, 1.22, 'band')]),
        ('c_abdomen4', [(1.22, 1.84, 'band')]), ('c_abdomen5', [(1.84, 2.45, 'dark')]),
        ('c_abdomen6', [(2.45, 2.95, 'dark'), (2.95, 3.05, 'tip')])]),
}
if SEX != 'neutral':
    spec = SEXED_ABDOMEN[SEX]

    def sexed_radius(t, spec=spec):
        u = t / spec['length']
        return spec['rmax'] * math.sin(math.pi * min(1.0, 0.18 + 0.82 * u)) ** spec['power'] * (1.0 - spec['taper'] * u)

    for name, pieces in spec['segments']:
        geo = g.Geo()
        m = g.trans(0, -0.15, ABD_TOP_Z) @ g.rot_x(math.pi / 2)
        for k, (t0, t1, style) in enumerate(pieces):
            n = max(3, int(round((t1 - t0) / spec['length'] * RINGS * 2)))
            last = name == 'c_abdomen6' and k == len(pieces) - 1
            prof = []
            for i in range(n + 1):
                t = t0 + (t1 - t0) * i / n
                lip = 1.04 if i == 0 and t0 > 0 and style != 'tip' else 1.0
                r = 0.0 if (last and i == n) else max(0.02, sexed_radius(t) * lip)
                prof.append((t, r))
            if style == 'band':
                dark_from = int(n * 0.6)
                mat = (lambda j, d=dark_from: 1 if j >= d else 0)
            else:
                mat = 1
            g.tube_down(prof, SEG, matrix=m, sx=1.0, sz=0.86, mat=mat, geo=geo)
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
    droop = -0.12
    if RIG2:
        # v2 bind pose: the v1 folded wing passed through the scutum and abdomen.  This
        # hinge stays embedded in the thorax (ellipsoid form 0.97) and the wing, beyond
        # its 0.40 root (the base that inserts under the notum), clears thorax, scutum,
        # scutellum and every sexed abdomen; found by a numeric search, re-checked on
        # the exported mesh (JOINT_CONTRACT.md).  The folded wing rises ~8.5 degrees.
        hinge = Vector((side * 0.45, 1.06, -0.45))
        droop = 0.15
    span = Vector((side * math.sin(yaw), droop, -math.cos(yaw))).normalized()
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
    wing_parent = None
    if RIG2:
        hinge, span, chord, normal = wing_frame(side)
        # Opening (+) moves the tip laterally away from the midline on both sides.
        _, wing_parent = joint_chain(name, hinge, [
            ('sweep', (0, -side, 0), 0.0, 3.0,
             'about the body vertical; 0 = folded rest; 3.0 = flybody wing_yaw span (-1.5..1.5) from its '
             'folded springref'),
            ('elevate', tuple(span.cross(Vector((0, 1, 0))).normalized()), -1.7, 0.8,
             'about the horizontal axis normal to the span (rotates with the sweep); + = tip up; flybody wing_roll range (-1..1.5) relative to its rest 0.7'),
            ('pitch', tuple(span), -0.27, 3.92,
             'about the wing span axis (leading edge down = +); flybody wing_pitch range (-1.27..2.92) '
             'relative to its rest -1.0')])
    n = 24 if LOD == 0 else 10
    verts, faces = [], []
    for i in range(n + 1):
        s = i / n
        verts += [tuple(wing_point(side, s, 1.0)), tuple(wing_point(side, s, -1.0))]
    for i in range(n):
        faces.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
    body_part(name, g.Geo().extend(verts, faces), [M['wing']], smooth=True, parent=wing_parent)
    _, _, _, normal = wing_frame(side)
    geo = g.Geo()
    for vein, pts in VEINS.items():
        if LOD and vein in ('acv', 'pcv'):
            continue
        poly = [wing_point(side, s, c) + normal * 0.006 for (s, c) in pts]
        g.ribbon(poly, 0.035 if vein == 'costa' else 0.026, normal, geo=geo)
    obj = body_part(name + '_veins', geo, [M['vein']], smooth=False, parent=wing_parent)

# ------------------------------------------------------------------ halteres
for side, name in ((-1, 'l_haltere'), (1, 'r_haltere')):
    base = Vector((side * 0.82, 0.35, -1.0))
    hal_parent = None
    if RIG2:
        # v1 left the haltere base ~0.03 outside the thorax; seat it on the surface.
        c, r = Vector((0, 0.05, 0.05)), Vector((1.12, 1.18, 1.55))
        d = base - c
        f = math.sqrt((d.x / r.x) ** 2 + (d.y / r.y) ** 2 + (d.z / r.z) ** 2)
        base = c + d * (0.98 / f)
        _, hal_parent = joint_chain(name, base, [
            ('beat', tuple(Vector((side * 0.22, -0.12, -0.32)).cross(Vector((0, 1, 0))).normalized()), -0.2, 0.2,
             'flybody haltere joint range; + = knob up')])
    tip = base + Vector((side * 0.22, -0.12, -0.32))
    geo = g.Geo()
    g.segment_between(base, tip, 0.05, 0.035, LSEG // 2 + 2, geo=geo)
    g.ellipsoid(tip, (0.12, 0.11, 0.15), LSEG, max(6, RINGS // 2), geo=geo)
    body_part(name, geo, [M['haltere']], parent=hal_parent)

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
    if SEX == 'male' and leg in ('lf', 'rf'):
        # Sex comb: one row of short, dark, blunt teeth on the first tarsal segment of
        # the front leg, running along the leg axis (D. melanogaster's comb is rotated
        # to lie roughly parallel to it; Kopp 2011; Doucet et al. 2023).  The count of
        # ten and the side of the leg are stylised, not measured.
        r1 = 0.085 * k
        for j in range(10):
            tt = TIBIA_LEN + 0.05 + j * 0.022
            base = Vector((r1 * 0.95, -tt, 0.0))
            g.segment_between(base, base + Vector((0.075, -0.03, 0.0)), 0.014, 0.006, 4, mat=1, geo=geo)
    # Paired pretarsal claws, splayed sideways in the leg's local frame.
    for s in (-1, 1):
        g.segment_between((0, -t + 0.02, 0), (s * 0.06, -TIBIA_TARSUS_LEN - 0.02, 0.03), 0.02, 0.004, 4, mat=1,
                          geo=geo)
    leg_objs.append(g.mesh_object(f'{leg}_tarsus', geo, [M['leg'], M['bristle']], parent=tibia_joint))

all_objs = parts + leg_objs
tris = g.triangle_count(all_objs)
print(f'NEUROFLY_ASSET fly lod={LOD} objects={len(all_objs)} triangles={tris}')

if SEX != 'neutral' or RIG2:
    # Shared rig metadata (exported as glTF extras on the root node).
    fly['nf_rig'] = 'neurofly-viewport-fly-v2' if RIG2 else 'neurofly-viewport-fly-v1'
    fly['nf_units'] = 'viewport-mm (about 3x life size; rig frame of ArticulatedFly3DViewport)'
    fly['nf_variant'] = SEX
    fly['nf_display_scale'] = DISPLAY_SCALE
    fly['nf_display_scale_basis'] = 'relative to female; female/male thorax-length ratio 1.15 (David et al. 2003)'
    fly['nf_appearance_only'] = 'no brain dataset, physiology or behaviour is tied to this variant'
stem = os.path.join(ARGS['out'], f'fly_hq_lod{LOD}' if SEX == 'neutral' else f'fly_{SEX}_lod{LOD}')
if RIG2:
    stem = os.path.join(ARGS['out'], f'fly_{SEX}_v2_lod{LOD}')
    import json
    with open(stem + '_joints.json', 'w') as fh:
        json.dump({'rig': 'neurofly-viewport-fly-v2', 'frame': 'thorax frame of the v1 rig (viewport-mm, +Y up, +Z forward)',
                   'joints': JOINTS}, fh, indent=1)
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
    # Renders show the variant at its display scale about the root origin, so every
    # variant shares 64 px per (female-referenced) viewport-mm.  The GLB stays unscaled.
    fly.scale = (DISPLAY_SCALE,) * 3
    if RIG2:
        views['under'] = g.camera('cam_under', (0, -40, -0.6), (0, 0, -0.6), ortho_scale=ORTHO)
    for view, cam in views.items():
        g.render_to(scene, cam, f'{stem}_{view}.png')
    if RIG2:
        # Articulation check (after export: the files keep the rest/bind pose).  Values
        # lie inside the display-safe envelope of JOINT_CONTRACT.md.
        OPEN_POSE = {'wing_sweep': 1.5, 'wing_elevate': 0.3, 'wing_pitch': 0.3, 'antenna_abduct': 0.4,
                     'antenna_extend': 0.2, 'haltere_beat': 0.2}
        by_name = {j['node']: j for j in JOINTS if 'axis' in j}
        for side in ('l', 'r'):
            for dof, angle in OPEN_POSE.items():
                node = bpy.data.objects[f'{side}_{dof}']
                node.matrix_basis = g.matrix_to_blender(Matrix.Rotation(angle, 4, Vector(by_name[node.name]['axis'])))
        for view in ('top', 'side', 'under', 'hero'):
            g.render_to(scene, views[view], f'{stem}_open_{view}.png')
    print(f'NEUROFLY_ASSET renders ortho_scale={ORTHO} px_per_viewport_mm={int(ARGS["res"]) / ORTHO:.3f}')
