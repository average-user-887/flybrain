# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Generate the NeuroFly food / odour / wind / predator presentation props.

Headless:  blender --background --factory-startup --threads 2 \
               --python tools/assets/build_env.py -- --out OUTDIR --prop NAME [--render 1]

NAME is one of PROPS below.  Writes OUTDIR/env_<NAME>.glb and .blend; with
--render 1 also transparent orthographic top and side PNGs and a 3/4 hero PNG
rendered from the very same objects that were exported, plus
OUTDIR/env_<NAME>.render.json (camera centre, ortho scale, bounds, triangles).

Frame and units follow tools/assets/INTERFACE.md: three.js frame, +Y up, +Z
forward, one unit = one viewport-mm.  Every prop stands on y = 0 (its own floor;
the dashboard floor top is y = -0.02) and is centred on its reference point:
the droplet for the sugar source, the source marker for the odour emitter, the
pole for the wind vane, the body for each predator.

These props are PRESENTATION ONLY.  They are not a stimulus, a field, an encoder
input or a collision shape, and loading them must never change looming or retinal
input.  The two predators are ILLUSTRATIVE: no predator stimulus or behaviour is
implemented.  Food props are drawn at a nominal size, not at a simulated source
radius.  Predators use the fly GLB's display scale (the fly is about 8.3 viewport-mm
long for a ~2.5 mm animal, so about 3.3x life).

Every vertex is generated here.  No third-party mesh, texture or scan is used.
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import nf_geom as g  # noqa: E402

PROPS = ('fermenting_fruit', 'yeast_patch', 'sugar_water', 'odour_emitter', 'wind_vane',
         'jumping_spider', 'mantis_nymph')
ARGS = g.parse_args({'out': '', 'prop': '', 'render': 0, 'samples': 32, 'res': 768})
PROP = ARGS['prop']
if PROP not in PROPS:
    raise SystemExit('pass -- --prop one of ' + ', '.join(PROPS))

STATUS = {
    'decorative': 'DECORATIVE: presentation only; not a stimulus, field, encoder input or collision shape',
    'illustrative': ('ILLUSTRATIVE: no predator stimulus or behaviour is implemented; decoration only and '
                     'must not change looming or retinal input'),
}

scene = g.reset_scene()


# ------------------------------------------------------------------ helpers
def lathe(profile, seg, geo, matrix=None, mat_fn=None, rad_fn=None, loop=False):
    """Lathe [(y, r)] around Y.  ``rad_fn(a)`` scales r per angle, ``mat_fn(j, k)`` picks a slot.

    ``loop=True`` joins the last ring back to the first (a closed cross-section, e.g. a
    peel ring or a hollow tube wall) and adds no caps.  Otherwise both ends are capped.
    """
    mat_fn = mat_fn or (lambda j, k: 0)
    verts, faces, mats, rings = [], [], [], []
    for (y, r) in profile:
        if r <= 1e-9 and not loop:
            rings.append([len(verts)])
            verts.append((0.0, y, 0.0))
            continue
        ids = []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            rr = r * (rad_fn(a) if rad_fn else 1.0)
            ids.append(len(verts))
            verts.append((rr * math.cos(a), y, rr * math.sin(a)))
        rings.append(ids)
    pairs = [(j, j + 1) for j in range(len(rings) - 1)]
    if loop:
        pairs.append((len(rings) - 1, 0))
    for j, j2 in pairs:
        a, b = rings[j], rings[j2]
        for k in range(seg):
            k2 = (k + 1) % seg
            if len(a) == 1:
                faces.append((a[0], b[k2], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
            mats.append(mat_fn(j, k))
    if not loop:
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
    """Ellipsoid with its rings running along Z (ring 0 at the front, +Z); ``mat(j)`` bands it."""
    m = g.trans(*center) @ g.rot_x(pitch) @ g.rot_x(math.pi / 2) @ g.scale3(rx, rz, ry)
    return lathe(g.sphere_profile(rings), seg, geo, matrix=m,
                 mat_fn=(lambda j, k: mat(j)) if callable(mat) else (lambda j, k: mat))


def sphere(center, r, geo, mat=0, seg=12, rings=8, sy=1.0):
    return g.ellipsoid(center, (r, r * sy, r), seg, rings, mat=mat, geo=geo)


def box(center, size, geo, mat=0, matrix=None):
    hx, hy, hz = (s / 2 for s in size)
    cx, cy, cz = center
    v = [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sy in (-1, 1) for sz in (-1, 1) for sx in (-1, 1)]
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return geo.extend(v, f, mat, matrix=matrix)


def annulus(r0, r1, seg, geo, y=0.01, mat=0, dashes=0, duty=0.55):
    """A flat ring on the floor; ``dashes`` > 0 cuts it into that many dashes."""
    verts, faces = [], []
    n = seg if not dashes else dashes
    for d in range(n):
        a0 = 2 * math.pi * d / n
        a1 = a0 + 2 * math.pi / n * (duty if dashes else 1.0)
        steps = 2 if not dashes else max(2, seg // dashes)
        base = len(verts)
        for s in range(steps + 1):
            a = a0 + (a1 - a0) * s / steps
            verts += [(r0 * math.cos(a), y, r0 * math.sin(a)), (r1 * math.cos(a), y, r1 * math.sin(a))]
        for s in range(steps):
            i = base + 2 * s
            faces.append((i, i + 2, i + 3, i + 1))
    return geo.extend(verts, faces, mat)


def mat(name, color, **kw):
    return g.material('nf_env_' + name, color, **kw)


def glass(name, color, alpha=0.3, roughness=0.05, transmission=0.9):
    m = mat(name, color, roughness=roughness, alpha=alpha)
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    for key in ('Transmission Weight', 'Transmission'):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = transmission
            break
    if 'IOR' in bsdf.inputs:
        bsdf.inputs['IOR'].default_value = 1.4
    return m


PARTS = []           # (name, Geo, [materials], smooth)
ROOT_PROPS = {}


def part(name, geo, mats, smooth=True):
    PARTS.append((name, geo, mats, smooth))


def bezier(p0, p1, p2, t):
    return tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c for a, b, c in zip(p0, p1, p2))


# ------------------------------------------------------------------ props
def build_fermenting_fruit():
    """A banana cross-section slice going over: five-lobed peel, browning flesh, seed trefoil, bubbles."""
    R, H = 12.0, 3.2
    lobes = lambda a: 1.0 - 0.035 * abs(math.sin(2.5 * a))          # noqa: E731  five peel ridges
    rot = lambda a: math.sin(3 * a + 0.7) + 0.8 * math.sin(7 * a + 2.1) + 0.5 * math.sin(11 * a)  # noqa: E731
    M = [mat('flesh', (0.86, 0.76, 0.50), roughness=0.55),
         mat('flesh_core', (0.74, 0.62, 0.36), roughness=0.5),
         mat('ferment_brown', (0.36, 0.20, 0.07), roughness=0.4),
         mat('peel_yellow', (0.80, 0.58, 0.08), roughness=0.45),
         mat('peel_spot', (0.16, 0.08, 0.03), roughness=0.5),
         mat('seed', (0.05, 0.035, 0.03), roughness=0.3),
         glass('bubble', (0.85, 0.75, 0.55), alpha=0.45, roughness=0.02, transmission=0.6),
         glass('juice', (0.55, 0.32, 0.08), alpha=0.35, roughness=0.05, transmission=0.4),
         mat('ferment_rim', (0.66, 0.50, 0.26), roughness=0.5)]
    seg = 72
    top = [(H + 0.15, 0.0), (H + 0.14, 1.6), (H + 0.11, 3.5), (H + 0.06, 5.5), (H, 7.5), (H - 0.06, 9.2),
           (H - 0.2, 10.6), (H - 0.45, 11.4), (H - 0.8, 11.75), (0.4, 11.75), (0.0, 11.4), (0.0, 0.0)]

    def flesh_mat(j, k):
        if j <= 1:
            return 1
        if j in (6, 7):
            return 8                       # browning rim under the peel
        return 0
    fg = lathe(top, seg, g.Geo(), mat_fn=flesh_mat, rad_fn=lobes)
    part('fruit_flesh', fg, M)
    # Peel: closed cross-section around the rim.
    peel = [(0.15, 11.65), (0.0, 12.15), (0.15, 12.65), (H - 0.7, 12.8), (H - 0.3, 12.55), (H - 0.45, 11.7)]

    def peel_mat(j, k):
        a = 2 * math.pi * k / seg
        s = math.sin(13 * a + 1.3) + math.sin(29 * a + 0.4) * 0.8 + math.sin(5 * a) * 0.5
        return 4 if (s > 1.25 or (j == 3 and s > 0.6)) else 3
    part('fruit_peel', lathe(peel, seg, g.Geo(), mat_fn=peel_mat, rad_fn=lobes,
                             loop=True), M)
    sg = g.Geo()
    for i in range(3):
        a = math.pi / 2 + 2 * math.pi * i / 3
        m = g.trans(1.1 * math.cos(a), H + 0.13, 1.1 * math.sin(a)) @ g.rot_y(-a) @ g.scale3(0.55, 0.08, 0.3)
        lathe(g.sphere_profile(4), 10, sg, matrix=m, mat_fn=lambda j, k: 5)
    for i in range(9):
        a = 2 * math.pi * i / 9 + 0.3
        sphere((2.6 * math.cos(a), H + 0.12, 2.6 * math.sin(a)), 0.13, sg, mat=5, seg=6, rings=3, sy=0.5)
    part('fruit_seeds', sg, M)
    br = g.Geo()                           # soft fermentation bruises sitting on the cut face
    for (a, rr, sx, sz) in ((0.4, 7.6, 2.2, 1.5), (1.7, 8.6, 1.6, 1.1), (2.6, 6.4, 1.3, 1.0),
                            (3.9, 8.2, 2.6, 1.7), (5.1, 7.0, 1.4, 1.2), (5.8, 9.2, 1.2, 0.9)):
        m = (g.trans(rr * math.cos(a), H + 0.13 - 0.0027 * rr * rr, rr * math.sin(a)) @ g.rot_y(-a)
             @ g.scale3(sz, 0.09, sx))
        lathe(g.sphere_profile(6), 20, br, matrix=m, mat_fn=lambda j, k: 2)
    part('fruit_ferment_bruises', br, M)
    bg = g.Geo()
    rng = random.Random(11)
    for _ in range(9):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(6.0, 10.2)
        r = rng.uniform(0.25, 0.65)
        sphere((rr * math.cos(a), H - 0.05 + r * 0.6, rr * math.sin(a)), r, bg, mat=6, seg=12, rings=8)
    part('fruit_ferment_bubbles', bg, M)
    jg = lathe([(0.06, 0.0), (0.06, 12.9), (0.0, 13.3), (0.0, 0.0)], seg, g.Geo(), mat_fn=lambda j, k: 7,
               rad_fn=lambda a: 1.0 + 0.06 * math.sin(3 * a + 0.4) + 0.04 * math.sin(5 * a))
    part('fruit_juice_film', jg, M)
    ROOT_PROPS.update(nominal_size='24 viewport-mm slice; nominal, not a simulated source radius',
                      depicts='fermenting banana slice (cross-section)')


def build_yeast_patch():
    """A dome of live yeast paste with budding colonies on a disc of standard fly food."""
    M = [glass('fly_food', (0.62, 0.36, 0.10), alpha=0.85, roughness=0.35, transmission=0.2),
         mat('yeast_paste', (0.80, 0.68, 0.45), roughness=0.75),
         mat('yeast_colony', (0.90, 0.80, 0.56), roughness=0.6),
         mat('yeast_dark', (0.55, 0.42, 0.22), roughness=0.7)]
    seg = 64
    part('food_disc', lathe([(0.7, 0.0), (0.7, 8.6), (0.6, 9.0), (0.0, 9.0), (0.0, 0.0)], seg, g.Geo(),
                            mat_fn=lambda j, k: 0), M)
    blob = lambda a: 1.0 + 0.13 * math.sin(3 * a + 0.5) + 0.08 * math.sin(7 * a + 1.9) + 0.04 * math.sin(13 * a)  # noqa: E731,E501
    prof = [(0.7 + 1.25 * math.cos(math.pi / 2 * i / 8) ** 0.8, 6.3 * math.sin(math.pi / 2 * i / 8)) for i in range(9)]
    prof = [(prof[0][0], 0.0)] + prof[1:] + [(0.62, 6.45), (0.62, 0.0)]
    part('yeast_paste', lathe(prof, seg, g.Geo(), mat_fn=lambda j, k: 3 if (j >= 7 and k % 9 == 0) else 1,
                              rad_fn=blob), M)
    cg = g.Geo()
    rng = random.Random(5)
    n = 0
    while n < 34:
        a = rng.uniform(0, 2 * math.pi)
        rr = math.sqrt(rng.uniform(0, 1)) * 5.6 * blob(a)
        t = rr / (6.3 * blob(a))
        y = 0.7 + 1.25 * max(0.0, 1 - t * t) ** 0.55
        r = rng.uniform(0.3, 0.85) * (1 - 0.4 * t)
        sphere((rr * math.cos(a), y - r * 0.25, rr * math.sin(a)), r, cg, mat=2, seg=10, rings=6, sy=0.7)
        n += 1
    part('yeast_colonies', cg, M)
    ROOT_PROPS.update(nominal_size='18 viewport-mm patch; nominal, not a simulated source radius',
                      depicts='live yeast paste on standard fly food')


def build_sugar_water():
    """A sessile sucrose droplet fed by a glass capillary, with a solid contact-zone ring."""
    M = [glass('sugar_water', (0.25, 0.55, 0.95), alpha=0.6, roughness=0.02, transmission=0.7),
         glass('capillary_glass', (0.9, 0.95, 1.0), alpha=0.22, roughness=0.02, transmission=0.95),
         mat('clamp', (0.10, 0.11, 0.13), roughness=0.4, metallic=0.5),
         mat('contact_ring', (0.10, 0.42, 0.85), roughness=0.5)]
    a, h = 3.0, 1.8
    rs = (a * a + h * h) / (2 * h)
    cy = h - rs
    n = 10
    t_max = math.asin(a / rs)
    cap = [(cy + rs * math.cos(t_max * i / n), rs * math.sin(t_max * i / n)) for i in range(n + 1)]
    cap = [(h, 0.0)] + cap[1:] + [(0.0, 0.0)]
    part('sugar_droplet', lathe(cap, 40, g.Geo(), mat_fn=lambda j, k: 0), M)
    # Capillary: tip dips into the drop top, runs back (-Z) and up at 16 degrees.
    tilt = math.radians(16)
    length = 26.0
    tip = Vector((0.0, 1.2, -0.6))
    d = Vector((0.0, math.sin(tilt), -math.cos(tilt)))
    q = Vector((0, 1, 0)).rotation_difference(d)
    mtx = g.trans(*tip) @ q.to_matrix().to_4x4()
    wall = [(0.0, 0.42), (0.0, 0.62), (length, 0.62), (length, 0.42)]
    part('capillary_tube', lathe(wall, 20, g.Geo(), matrix=mtx, mat_fn=lambda j, k: 1, loop=True), M)
    part('capillary_liquid', lathe([(0.02, 0.0), (0.02, 0.4), (length * 0.78, 0.4), (length * 0.78, 0.0)],
                                   16, g.Geo(), matrix=mtx, mat_fn=lambda j, k: 0), M)
    end = tip + d * (length - 2.0)
    cl = g.Geo()
    box(tuple(end), (2.2, 2.2, 3.4), cl, mat=2, matrix=None)
    g.segment_between((end.x, end.y - 1.1, end.z), (end.x, 0.0, end.z), 0.35, 0.35, 12, mat=2, geo=cl)
    lathe([(0.25, 0.0), (0.25, 2.0), (0.0, 2.0), (0.0, 0.0)], 20, cl, matrix=g.trans(end.x, 0, end.z),
          mat_fn=lambda j, k: 2)
    part('capillary_clamp', cl, M, smooth=False)
    part('sugar_contact_marker', annulus(3.7, 4.2, 64, g.Geo(), mat=3), M, smooth=False)
    ROOT_PROPS.update(nominal_size='6 viewport-mm droplet, 26 viewport-mm capillary; nominal',
                      depicts='sucrose solution droplet at a glass capillary tip',
                      sense='taste on CONTACT (tarsi / proboscis); the solid ring marks the contact zone')


def build_odour_emitter():
    """An odour vial with a soaked cotton wick, plus a separable source marker (pin + dashed rings)."""
    M = [glass('vial_glass', (0.9, 0.95, 1.0), alpha=0.25, roughness=0.03, transmission=0.95),
         glass('odourant', (0.85, 0.65, 0.25), alpha=0.6, roughness=0.05, transmission=0.6),
         mat('cotton', (0.92, 0.90, 0.86), roughness=0.95),
         mat('marker_violet', (0.42, 0.12, 0.72), roughness=0.4),
         mat('marker_pole', (0.85, 0.85, 0.88), roughness=0.3, metallic=0.6)]
    seg = 40
    wall = [(0.0, 1.85), (0.0, 2.2), (5.6, 2.2), (5.8, 2.0), (5.6, 1.85)]
    part('vial_glass', lathe(wall, seg, g.Geo(), mat_fn=lambda j, k: 0, loop=True), M)
    part('vial_bottom', lathe([(0.25, 0.0), (0.25, 1.85), (0.0, 1.85), (0.0, 0.0)], seg, g.Geo(),
                              mat_fn=lambda j, k: 0), M)
    part('vial_odourant', lathe([(2.3, 0.0), (2.3, 1.8), (0.25, 1.8), (0.25, 0.0)], seg, g.Geo(),
                                mat_fn=lambda j, k: 1), M)
    fluff = lambda a: 1.0 + 0.08 * math.sin(5 * a + 0.3) + 0.05 * math.sin(9 * a + 1.0)  # noqa: E731
    wick = [(6.9, 0.0), (6.85, 0.8), (6.6, 1.5), (6.1, 1.95), (5.4, 1.8), (4.4, 1.8), (4.4, 0.0)]
    part('vial_wick', lathe(wick, 28, g.Geo(), mat_fn=lambda j, k: 2, rad_fn=fluff), M)
    mk = g.Geo()
    g.segment_between((0, 7.0, 0), (0, 10.6, 0), 0.12, 0.12, 10, mat=4, geo=mk)
    lathe([(12.0, 0.0), (11.2, 0.75), (10.4, 0.0)], 4, mk, mat_fn=lambda j, k: 3)
    annulus(4.6, 5.0, 96, mk, mat=3, dashes=16)
    annulus(7.2, 7.6, 128, mk, mat=3, dashes=24)
    part('odour_source_marker', mk, M, smooth=False)
    ROOT_PROPS.update(nominal_size='4.4 viewport-mm vial; marker rings 5 and 7.6 viewport-mm; nominal',
                      depicts='odour vial with soaked wick and an odour-source marker',
                      sense='volatile odour detected at a DISTANCE (antennae); the dashed rings mark the '
                            'source position only and are not a measured concentration or plume',
                      reference_point='origin = source marker = odour source position')


def build_wind_vane():
    """A wind vane whose rotor's local +Z points to where the air is going (downwind)."""
    M = [mat('vane_base', (0.18, 0.2, 0.23), roughness=0.5, metallic=0.3),
         mat('vane_pole', (0.8, 0.8, 0.82), roughness=0.3, metallic=0.7),
         mat('vane_arrow', (0.05, 0.55, 0.60), roughness=0.4),
         mat('vane_fin', (0.95, 0.55, 0.10), roughness=0.45)]
    st = g.Geo()
    lathe([(0.45, 0.0), (0.45, 2.6), (0.3, 3.0), (0.0, 3.0), (0.0, 0.0)], 40, st, mat_fn=lambda j, k: 0)
    g.segment_between((0, 0.4, 0), (0, 9.0, 0), 0.18, 0.15, 12, mat=1, geo=st)
    sphere((0, 9.1, 0), 0.32, st, mat=1)
    part('vane_stand', st, M)
    rt = g.Geo()
    y = 9.1
    g.segment_between((0, y, -5.5), (0, y, 3.4), 0.2, 0.2, 12, mat=2, geo=rt)
    g.segment_between((0, y, 6.0), (0, y, 3.4), 0.0, 0.85, 16, mat=2, geo=rt)
    fin = [(0, y - 1.5, -6.6), (0, y + 1.5, -6.6), (0, y + 0.3, -3.6), (0, y - 0.3, -3.6)]
    th = 0.12
    v = [(x + s * th, yy, z) for s in (-1, 1) for (x, yy, z) in fin]
    fins = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    rt.extend(v, fins, 3)
    # A second, horizontal fin so the tail also reads in the top view.
    hv = [(x - (yy - y), y + s * th, z) for s in (-1, 1) for (x, yy, z) in fin]
    rt.extend(hv, fins, 3)
    PARTS.append(('vane_rotor_parts', rt, M, False))
    ROOT_PROPS.update(nominal_size='13 viewport-mm vane, 9 viewport-mm tall; nominal',
                      depicts='wind vane / direction indicator',
                      direction_convention=('rotate node wind_vane_rotor about +Y; its local +Z (arrow tip) '
                                            'points where the air moves TO (downwind); the orange fin trails '
                                            'upwind. Not a measurement: show NOT SIMULATED when no wind field'))


def build_jumping_spider():
    """Zebra jumping spider (Salticus scenicus style): boxy carapace, big AME, banded legs and abdomen."""
    M = [mat('spider_black', (0.025, 0.022, 0.02), roughness=0.55),
         mat('spider_white', (0.80, 0.78, 0.72), roughness=0.8),
         mat('spider_eye', (0.01, 0.01, 0.012), roughness=0.04),
         mat('spider_brown', (0.16, 0.10, 0.06), roughness=0.6),
         mat('spider_lens', (0.30, 0.18, 0.06), roughness=0.03)]
    body = g.Geo()
    ellipsoid_z((0, 3.2, 3.0), 2.9, 2.0, 4.1, 28, 16, body, mat=lambda j: 1 if j in (9, 10) else 0, pitch=-0.06)
    ellipsoid_z((0, 2.5, -0.9), 0.7, 0.6, 0.9, 10, 6, body, mat=0)
    ellipsoid_z((0, 3.0, -5.0), 2.9, 2.3, 4.0, 28, 18,
                body, mat=lambda j: 1 if j in (4, 5, 9, 10, 14) else 0, pitch=0.12)
    part('spider_body', body, M)
    ey = g.Geo()
    for sx in (-1, 1):
        sphere((sx * 0.98, 3.75, 6.95), 0.88, ey, mat=4, seg=16, rings=10)
        sphere((sx * 2.05, 4.05, 6.55), 0.45, ey, mat=2, seg=12, rings=8)
        sphere((sx * 2.35, 4.75, 5.2), 0.17, ey, mat=2, seg=8, rings=5)
        sphere((sx * 2.4, 5.0, 3.6), 0.4, ey, mat=2, seg=12, rings=8)
    part('spider_eyes', ey, M)
    mo = g.Geo()
    for sx in (-1, 1):
        g.segment_between((sx * 0.6, 2.4, 6.7), (sx * 0.55, 1.4, 7.3), 0.45, 0.3, 10, mat=3, geo=mo)
        g.segment_between((sx * 1.2, 2.2, 6.6), (sx * 1.6, 2.0, 7.9), 0.28, 0.26, 10, mat=0, geo=mo)
        g.segment_between((sx * 1.6, 2.0, 7.9), (sx * 1.7, 1.0, 8.6), 0.26, 0.3, 10, mat=1, geo=mo)
        sphere((sx * 1.72, 0.85, 8.65), 0.32, mo, mat=1, seg=10, rings=6)
    part('spider_mouthparts_palps', mo, M)
    lg = g.Geo()
    legs = [(5.0, 30, 7.8, 0.55), (3.8, 72, 6.6, 0.44), (2.5, 112, 6.5, 0.42), (1.1, 150, 8.0, 0.46)]
    for sx in (-1, 1):
        for (z0, yaw, L, r) in legs:
            a = math.radians(yaw)
            dx, dz = sx * math.sin(a), math.cos(a)
            A = Vector((sx * 2.1, 2.5, z0))
            out = Vector((dx, 0, dz))
            K = A + out * (L * 0.36) + Vector((0, 3.0, 0))
            P = K + out * (L * 0.12) + Vector((0, -0.1, 0))
            B = A + out * (L * 0.82) + Vector((0, -1.3, 0))
            F = A + out * (L * 1.02) + Vector((0, -2.3, 0))
            g.segment_between(A, K, r, r * 0.9, 10, mat=0, geo=lg)
            sphere(tuple(K), r * 0.9, lg, mat=1, seg=8, rings=5)
            g.segment_between(K, P, r * 0.9, r * 0.82, 10, mat=1, geo=lg)
            g.segment_between(P, B, r * 0.82, r * 0.6, 10, mat=0, geo=lg)
            sphere(tuple(B), r * 0.6, lg, mat=1, seg=8, rings=5)
            g.segment_between(B, F, r * 0.6, r * 0.42, 10, mat=3, geo=lg)
    part('spider_legs', lg, M)
    ROOT_PROPS.update(depicts='zebra jumping spider (Salticidae; Salticus scenicus style)',
                      scale='fly display scale (~3.3x life); ~6 mm spider -> ~20 viewport-mm')


def build_mantis_nymph():
    """Chinese mantis nymph (Tenodera sinensis style): raptorial forelegs, long prothorax, upturned abdomen."""
    M = [mat('mantis_green', (0.22, 0.40, 0.10), roughness=0.5),
         mat('mantis_dark', (0.20, 0.28, 0.08), roughness=0.5),
         mat('mantis_eye', (0.45, 0.55, 0.22), roughness=0.2),
         mat('mantis_pupil', (0.02, 0.02, 0.02), roughness=0.3),
         mat('mantis_spine', (0.12, 0.10, 0.05), roughness=0.4)]
    bd = g.Geo()
    p0, p1, p2 = (0, 4.6, -1.6), (0, 3.9, -10.5), (0, 9.8, -15.5)
    n = 9
    for i in range(n):
        a, b = bezier(p0, p1, p2, i / n), bezier(p0, p1, p2, (i + 1) / n)
        r0 = 1.5 - 0.9 * (i / n) + 0.55 * math.sin(math.pi * i / n)
        r1 = 1.5 - 0.9 * ((i + 1) / n) + 0.55 * math.sin(math.pi * (i + 1) / n)
        g.segment_between(a, b, r0 * 0.94, r1, 14, mat=1 if i % 2 else 0, geo=bd)
    g.segment_between((0, 4.6, -1.6), (0, 4.8, 3.2), 1.0, 0.85, 14, mat=0, geo=bd)
    g.segment_between((0, 4.8, 3.2), (0, 6.2, 5.0), 0.85, 0.9, 14, mat=0, geo=bd)
    g.segment_between((0, 6.2, 5.0), (0, 10.4, 10.6), 0.8, 0.62, 14, mat=1, geo=bd)
    for sx in (-1, 1):
        m = g.trans(sx * 0.75, 5.6, 0.2) @ g.rot_x(0.15) @ g.scale3(0.6, 0.18, 1.6)
        lathe(g.sphere_profile(6), 10, bd, matrix=m, mat_fn=lambda j, k: 1)
    part('mantis_body', bd, M)
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
    part('mantis_head', hd, M)
    rl = g.Geo()
    for sx in (-1, 1):
        S = (sx * 0.65, 9.2, 9.7)
        C = (sx * 1.05, 6.6, 12.0)
        Fe = (sx * 1.2, 10.3, 14.3)
        Ti = (sx * 1.05, 8.1, 12.7)
        Ta = (sx * 1.0, 7.4, 13.9)
        g.segment_between(S, C, 0.62, 0.5, 12, mat=0, geo=rl)
        sphere(C, 0.4, rl, mat=0, seg=10, rings=6)
        g.segment_between(C, Fe, 0.72, 0.42, 12, mat=0, geo=rl)
        sphere(Fe, 0.3, rl, mat=1, seg=8, rings=5)
        g.segment_between(Fe, Ti, 0.4, 0.3, 10, mat=1, geo=rl)
        g.segment_between(Ti, Ta, 0.14, 0.1, 8, mat=1, geo=rl)
        cv, fv = Vector(C), Vector(Fe)
        for i in range(1, 7):
            p = cv.lerp(fv, i / 7.0)
            g.segment_between(tuple(p), tuple(p + Vector((0, -0.35, -0.95))), 0.14, 0.0, 6, mat=4, geo=rl)
    part('mantis_raptorial_forelegs', rl, M)
    wl = g.Geo()
    for sx in (-1, 1):
        for A, K, F, T in (((sx * 0.8, 4.4, 1.6), (sx * 6.2, 6.8, 4.4), (sx * 9.2, 0.15, 6.4), (sx * 9.7, 0.1, 7.6)),
                           ((sx * 0.8, 4.3, -0.9), (sx * 6.8, 7.1, -4.4), (sx * 9.9, 0.15, -9.6),
                            (sx * 10.3, 0.1, -10.8))):
            g.segment_between(A, K, 0.4, 0.32, 10, mat=0, geo=wl)
            sphere(K, 0.27, wl, mat=1, seg=8, rings=5)
            g.segment_between(K, F, 0.3, 0.2, 10, mat=0, geo=wl)
            g.segment_between(F, T, 0.14, 0.08, 6, mat=1, geo=wl)
    part('mantis_walking_legs', wl, M)
    ROOT_PROPS.update(depicts='praying mantis nymph (Mantidae; Tenodera sinensis style)',
                      scale='fly display scale (~3.3x life); ~11 mm early nymph -> ~37 viewport-mm')


BUILDERS = {'fermenting_fruit': build_fermenting_fruit, 'yeast_patch': build_yeast_patch,
            'sugar_water': build_sugar_water, 'odour_emitter': build_odour_emitter,
            'wind_vane': build_wind_vane, 'jumping_spider': build_jumping_spider,
            'mantis_nymph': build_mantis_nymph}
BUILDERS[PROP]()

illustrative = PROP in ('jumping_spider', 'mantis_nymph')
root = g.empty('env_' + PROP)
root['neurofly_status'] = STATUS['illustrative' if illustrative else 'decorative']
root['neurofly_frame'] = 'three.js +Y up, +Z forward, 1 unit = 1 viewport-mm, stands on y = 0'
for k, v in ROOT_PROPS.items():
    root['neurofly_' + k] = v
parent_for = {'vane_rotor_parts': g.empty('wind_vane_rotor', parent=root)} if PROP == 'wind_vane' else {}
objs = []
for name, geo, mats, smooth in PARTS:
    objs.append(g.mesh_object(name, geo, mats, smooth=smooth, parent=parent_for.get(name, root)))
tris = g.triangle_count(objs)
bpy.context.view_layer.update()
pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
lo = [min(p[i] for p in pts) for i in range(3)]
hi = [max(p[i] for p in pts) for i in range(3)]
# Blender (x, y, z) -> three.js (x, z, -y)
bb_lo = (lo[0], lo[2], -hi[1])
bb_hi = (hi[0], hi[2], -lo[1])
print(f'NEUROFLY_ASSET env {PROP} objects={len(objs)} triangles={tris} '
      f'bbox_three=({bb_lo[0]:.2f},{bb_lo[1]:.2f},{bb_lo[2]:.2f})..({bb_hi[0]:.2f},{bb_hi[1]:.2f},{bb_hi[2]:.2f})')
stem = os.path.join(ARGS['out'], 'env_' + PROP)
g.export_glb(stem + '.glb')
bpy.ops.wm.save_as_mainfile(filepath=stem + '.blend')

info = {'prop': PROP, 'triangles': tris, 'status': root['neurofly_status'],
        'bbox_three_min': [round(v, 3) for v in bb_lo], 'bbox_three_max': [round(v, 3) for v in bb_hi],
        **{k: v for k, v in ROOT_PROPS.items()}}
if int(ARGS['render']):
    res = int(ARGS['res'])
    g.setup_render(scene, (res, res), samples=int(ARGS['samples']))
    cx, cy, cz = ((a + b) / 2 for a, b in zip(bb_lo, bb_hi))
    span_top = max(bb_hi[0] - bb_lo[0], bb_hi[2] - bb_lo[2]) * 1.12
    span_side = max(bb_hi[1] - bb_lo[1], bb_hi[2] - bb_lo[2]) * 1.12
    ext = max(span_top, span_side)
    g.studio_lights(scale=ext * 0.9)
    cams = {
        'top': g.camera('cam_top', (cx, ext * 4, cz), (cx, 0, cz), ortho_scale=span_top),
        'side': g.camera('cam_side', (cx + ext * 4, cy, cz), (cx, cy, cz), ortho_scale=span_side),
        'hero': g.camera('cam_hero', (cx + ext * 1.25, cy + ext * 1.0, cz + ext * 1.4), (cx, cy * 0.7, cz), lens=50),
    }
    for view, cam in cams.items():
        scene.render.resolution_y = res if view != 'hero' else int(res * 0.75)
        g.render_to(scene, cam, f'{stem}_{view}.png')
    info['renders'] = {
        'top': {'look': '-Y (down); image top = +Z (forward)', 'center_three': [round(cx, 3), round(cz, 3)],
                'ortho_scale_viewport_mm': round(span_top, 3), 'px_per_viewport_mm': round(res / span_top, 3)},
        'side': {'look': '-X; image right = +Z', 'center_three_yz': [round(cy, 3), round(cz, 3)],
                 'ortho_scale_viewport_mm': round(span_side, 3), 'px_per_viewport_mm': round(res / span_side, 3)},
        'hero': {'note': '3/4 perspective preview, not to scale'},
        'engine': 'Cycles CPU, 2 threads', 'samples': int(ARGS['samples']), 'resolution_px': res}
    print(f'NEUROFLY_ASSET env {PROP} renders done')
with open(stem + '.render.json', 'w') as fh:
    json.dump(info, fh, indent=2)
