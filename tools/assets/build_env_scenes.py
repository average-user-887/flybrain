# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Composed presentation scenes from the W3 props and the W2 rig-v2 flies.

Headless:  blender --background --factory-startup --threads 2 \
               --python tools/assets/build_env_scenes.py -- --out OUTDIR --props ENV_DIR \
               --fly-female RIG/fly_female_v2_lod1.glb --fly-male RIG/fly_male_v2_lod1.glb \
               --scene NAME [--render 1]

NAME is one of SCENES.  Writes OUTDIR/scene_<NAME>.glb, .blend and .json (triangles, bounds,
placements) and, with --render 1, a 3/4 hero PNG and an orthographic top PNG of the very same
objects.  The props and flies are imported from their GLBs (only read; never re-saved).

Each scene adds original ground geometry: a gently undulating soil (or lab plate) disc coloured
by vertex colours, a leaf-litter and pebble dressing, and a lighting rig (a warm sun and a cool
fill exported as glTF punctual lights; studio area lights are render-only).

PRESENTATION ONLY.  A scene is a picture, not an arena, stimulus, odour field, taste contact or
encounter: nothing here is simulated, and the predator scene is ILLUSTRATIVE (no predator
stimulus or behaviour exists; it must not change looming or retinal input).  Every vertex of the
dressing is generated here; no third-party mesh, texture or scan is used.
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

ARGS = g.parse_args({'out': '', 'props': '', 'fly_female': '', 'fly_male': '', 'scene': '', 'render': 0,
                     'samples': 48, 'res': 1280})
SCENES = ('fermenting_fruit_patch', 'sugar_water_feeder', 'predator_encounter_illustrative')
NAME = ARGS['scene']
if NAME not in SCENES:
    raise SystemExit('pass -- --scene one of ' + ', '.join(SCENES))
BUDGET_TRIS = 40000
scene = g.reset_scene()
rng = random.Random(20261008)


# ------------------------------------------------------------------ materials
def vcol_material(name, roughness=0.9):
    """Principled BSDF whose base colour is the mesh's 'Col' colour attribute (exported as COLOR_0)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get('Principled BSDF')
    attr = nt.nodes.new('ShaderNodeVertexColor')
    attr.layer_name = 'Col'
    nt.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def paint(obj, colours):
    """colours: per-vertex RGB list (in vertex order of obj.data)."""
    me = obj.data
    layer = me.color_attributes.new('Col', 'BYTE_COLOR', 'CORNER')
    for loop in me.loops:
        r, gg, b = colours[loop.vertex_index]
        layer.data[loop.index].color = (r, gg, b, 1.0)


def noise2(x, z, seed=0.0):
    return (math.sin(0.31 * x + 1.7 + seed) * math.cos(0.27 * z - 0.4 + seed)
            + 0.5 * math.sin(0.83 * x - 0.6 * z + 2.1 + seed) + 0.25 * math.sin(1.9 * x + 1.3 * z + seed))


# ------------------------------------------------------------------ ground and dressing
def ground_disc(radius, rings, seg, height, palette, name, flat_inside=0.0, rim=None):
    """Polar-grid disc (three.js frame, top near y = 0) with vertex colours from palette noise."""
    geo = g.Geo()
    verts, cols = [(0.0, 0.0, 0.0)], []
    for i in range(1, rings + 1):
        r = radius * i / rings
        for k in range(seg):
            a = 2 * math.pi * k / seg
            x, z = r * math.cos(a), r * math.sin(a)
            h = 0.0 if r < flat_inside else height * noise2(x, z) * min(1.0, (r - flat_inside) / 6.0)
            edge = max(0.0, (r - 0.85 * radius) / (0.15 * radius))
            verts.append((x, h - 1.2 * edge * edge, z))
    faces = []
    for k in range(seg):
        faces.append((0, 1 + (k + 1) % seg, 1 + k))
    for i in range(rings - 1):
        b0, b1 = 1 + i * seg, 1 + (i + 1) * seg
        for k in range(seg):
            k2 = (k + 1) % seg
            faces.append((b0 + k, b0 + k2, b1 + k2, b1 + k))
    geo.verts, geo.faces, geo.mats = verts, faces, [0] * len(faces)
    obj = g.mesh_object(name, geo, [vcol_material(name + '_mat')], smooth=True, recalc=False)
    for v in verts:
        t = 0.5 + 0.5 * math.tanh(noise2(v[0] * 1.7, v[2] * 1.7, 3.0))
        c = [lerp(a, b, t) for a, b in zip(palette[0], palette[1])]
        speck = noise2(v[0] * 6.0, v[2] * 6.0, 9.0)
        if speck > 1.2:
            c = list(palette[2])
        cols.append(tuple(c))
    paint(obj, cols)
    return obj


def lerp(a, b, t):
    return a + (b - a) * t


def ground_height(x, z, height, flat_inside):
    r = math.hypot(x, z)
    return 0.0 if r < flat_inside else height * noise2(x, z) * min(1.0, (r - flat_inside) / 6.0)


def leaf(geo, cx, cz, base_y, length, width, yaw, curl, mat):
    """A dried leaf: elliptical blade, raised midrib, edges curling up; lies on the ground."""
    n = 7
    pts = []
    for i in range(n + 1):
        u = i / n
        w = width * math.sin(math.pi * u) ** 0.8
        for s in (-1.0, 0.0, 1.0):
            x = s * w * 0.5
            y = 0.05 + (0.08 if s == 0 else curl * (abs(s) * w * 0.5) ** 2 / max(width, 1e-3))
            pts.append((x, y, (u - 0.5) * length))
    faces = []
    for i in range(n):
        a, b = 3 * i, 3 * (i + 1)
        faces += [(a, a + 1, b + 1, b), (a + 1, a + 2, b + 2, b + 1)]
    m = g.trans(cx, base_y, cz) @ g.rot_y(yaw) @ g.rot_x(rng.uniform(-0.08, 0.08))
    geo.extend(pts, faces, mat, matrix=m)


def pebble(geo, cx, cz, base_y, r, mat):
    verts, faces = g.icosphere(1, g.trans(cx, base_y + r * 0.35, cz) @ g.rot_y(rng.uniform(0, 6.28))
                               @ g.scale3(r * rng.uniform(0.8, 1.2), r * 0.55, r * rng.uniform(0.8, 1.2)))
    geo.extend(verts, faces, mat)


def twig(geo, p0, p1, r0, r1, mat):
    g.segment_between(p0, p1, r0, r1, 8, mat=mat, geo=geo)


def dressing(radius, n_leaves, n_pebbles, keep_out, height, flat_inside, leaf_scale=1.0, twigs=2):
    """Leaf litter, pebbles and twigs, kept out of the circles in keep_out [(x, z, r)]."""
    M = [g.material('nf_leaf_brown', (0.30, 0.17, 0.07), roughness=0.75),
         g.material('nf_leaf_ochre', (0.55, 0.36, 0.12), roughness=0.75),
         g.material('nf_leaf_dark', (0.12, 0.08, 0.04), roughness=0.8),
         g.material('nf_pebble_grey', (0.32, 0.30, 0.28), roughness=0.85),
         g.material('nf_pebble_tan', (0.45, 0.38, 0.30), roughness=0.85),
         g.material('nf_twig', (0.22, 0.15, 0.09), roughness=0.8)]
    geo = g.Geo()

    def free(x, z, pad):
        if math.hypot(x, z) > radius * 0.85:
            return False
        return all(math.hypot(x - kx, z - kz) > kr + pad for kx, kz, kr in keep_out)
    placed = 0
    while placed < n_leaves:
        x, z = rng.uniform(-radius, radius), rng.uniform(-radius, radius)
        L = rng.uniform(5.0, 9.0) * leaf_scale
        if not free(x, z, L * 0.6):
            continue
        leaf(geo, x, z, ground_height(x, z, height, flat_inside), L, L * rng.uniform(0.38, 0.55),
             rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.8), rng.choice((0, 0, 1, 2)))
        placed += 1
    placed = 0
    while placed < n_pebbles:
        x, z = rng.uniform(-radius, radius), rng.uniform(-radius, radius)
        r = rng.uniform(0.5, 1.6)
        if not free(x, z, r + 0.5):
            continue
        pebble(geo, x, z, ground_height(x, z, height, flat_inside) - 0.1, r, rng.choice((3, 4)))
        placed += 1
    placed = 0
    while placed < twigs:
        x, z = rng.uniform(-radius * 0.7, radius * 0.7), rng.uniform(-radius * 0.7, radius * 0.7)
        a = rng.uniform(0, math.pi)
        L = rng.uniform(14, 22)
        x1, z1 = x + L * math.cos(a), z + L * math.sin(a)
        if not (free(x, z, 2.0) and free(x1, z1, 2.0) and free((x + x1) / 2, (z + z1) / 2, 2.0)):
            continue
        y0, y1 = ground_height(x, z, height, flat_inside), ground_height(x1, z1, height, flat_inside)
        twig(geo, (x, y0 + 0.45, z), (x1, y1 + 0.3, z1), 0.45, 0.25, 5)
        placed += 1
    return g.mesh_object('scene_dressing', geo, M, smooth=False)


# ------------------------------------------------------------------ placement
def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    return [o for o in new if o.parent is None][0], new


def place(path, x, z, yaw, base_y=0.0, kind='prop'):
    """Put a GLB root at three.js (x, base_y, z), turned by yaw about +Y.  Flies stand on base_y:
    they are lifted by their own lowest tarsus vertex (display pose; preview placement only)."""
    root, objs = import_glb(path)
    lift = 0.0
    scale = 1.0
    if kind == 'fly':
        scale = float(root.get('nf_display_scale', 1.0))
        root.scale = (scale,) * 3
        bpy.context.view_layer.update()
        lowest = min((o.matrix_world @ v.co).z for o in objs if o.type == 'MESH' and o.name.split('.')[0].endswith('_tarsus')
                     for v in o.data.vertices)
        lift = -lowest
    root.location = Vector(g.to_blender((x, base_y + lift, z)))
    root.rotation_mode = 'XYZ'
    root.rotation_euler = (0.0, 0.0, yaw)
    bpy.context.view_layer.update()
    return {'file': os.path.basename(path), 'root': root.name, 'x': x, 'z': z, 'yaw_rad': round(yaw, 4),
            'base_y': round(base_y, 3), 'stand_lift': round(lift, 4), 'display_scale': scale}


def prop(name):
    return os.path.join(ARGS['props'], f'env_{name}.glb')


def heading_to(x, z, tx, tz):
    """Yaw (about +Y) that turns a model's +Z forward toward (tx, tz)."""
    return math.atan2(tx - x, tz - z)


# ------------------------------------------------------------------ scenes
placements = []
SOIL = [(0.20, 0.13, 0.07), (0.33, 0.22, 0.12), (0.12, 0.08, 0.05)]
PLATE = [(0.82, 0.82, 0.80), (0.86, 0.86, 0.84), (0.78, 0.78, 0.76)]
FRUIT_TOP = 3.35        # flesh top of the fermenting-fruit slice near its centre (build_env.py, H + 0.15)

if NAME == 'fermenting_fruit_patch':
    R, H = 42.0, 0.35
    ground_disc(R, 22, 72, H, SOIL, 'scene_ground_soil', flat_inside=0.0)
    keep = [(0, 0, 15.5), (25, -16, 15.5), (-22, 12, 10.5), (-6, -18, 6), (12, 17, 6)]
    dressing(R, 26, 14, keep, H, 0.0)
    placements.append(place(prop('fermenting_fruit'), 0, 0, 0.3, ground_height(0, 0, H, 0)))
    placements.append(place(prop('fermenting_fruit'), 25, -16, 2.1, ground_height(25, -16, H, 0)))
    placements.append(place(prop('yeast_patch'), -22, 12, 0.0, ground_height(-22, 12, H, 0)))
    gy = ground_height(0, 0, H, 0)
    placements.append(place(ARGS['fly_female'], 3.0, 2.0, heading_to(3, 2, -1, -3), gy + FRUIT_TOP - 0.25, 'fly'))
    x, z = -6.0, -18.0
    placements.append(place(ARGS['fly_male'], x, z, heading_to(x, z, 0, 0), ground_height(x, z, H, 0), 'fly'))
    x, z = 12.0, 17.0
    placements.append(place(ARGS['fly_female'], x, z, heading_to(x, z, -22, 12), ground_height(x, z, H, 0), 'fly'))
    caption = 'Fermenting fruit patch (illustrative scene; nothing here is simulated)'
elif NAME == 'sugar_water_feeder':
    R = 34.0
    ground_disc(R, 18, 72, 0.0, PLATE, 'scene_ground_plate', flat_inside=R)
    rim = g.Geo()
    g.revolve([(-0.3, R - 0.6), (2.2, R - 0.6), (2.2, R + 0.2), (-0.3, R + 0.2)], 96, geo=rim)
    g.mesh_object('scene_dish_wall', rim, [g.material('nf_dish_wall', (0.85, 0.9, 0.95), roughness=0.1,
                                                     alpha=0.35)], smooth=True)
    placements.append(place(prop('sugar_water'), 0, 6, math.pi, 0.0))   # capillary runs away (+Z) from the flies
    x, z = 0.0, -2.2                       # head toward the droplet, front legs near the solid contact ring
    placements.append(place(ARGS['fly_female'], x, z, heading_to(x, z, 0, 6), 0.0, 'fly'))
    x, z = 11.0, -7.0
    placements.append(place(ARGS['fly_male'], x, z, heading_to(x, z, 0, 6), 0.0, 'fly'))
    caption = 'Sugar-water feeder (illustrative scene; taste is by contact, nothing here is simulated)'
else:
    R, H = 46.0, 0.4
    ground_disc(R, 24, 72, H, SOIL, 'scene_ground_soil', flat_inside=0.0)
    keep = [(0, 0, 7), (0, 24, 12), (-24, 6, 16)]
    dressing(R, 30, 12, keep, H, 0.0, twigs=3)
    # a large flat leaf as the mantis' perch
    perch = g.Geo()
    leaf(perch, -24, 6, ground_height(-24, 6, H, 0), 30.0, 14.0, 0.6, 0.2, 0)
    g.mesh_object('scene_perch_leaf', perch, [g.material('nf_leaf_green_dry', (0.30, 0.32, 0.10), roughness=0.7)],
                  smooth=False)
    gy = ground_height(0, 0, H, 0)
    placements.append(place(ARGS['fly_female'], 0, 0, 0.0, gy, 'fly'))
    placements.append(place(prop('jumping_spider'), 0, 24, heading_to(0, 24, 0, 0), ground_height(0, 24, H, 0)))
    placements.append(place(prop('mantis_nymph'), -24, 6, heading_to(-24, 6, 0, 0),
                            ground_height(-24, 6, H, 0) + 0.18))
    caption = 'Predator encounter: ILLUSTRATIVE (no predator stimulus or behaviour is simulated)'

# Root empty carrying the scene's status for any loader.
root = g.empty('scene_' + NAME)
root['neurofly_status'] = ('ILLUSTRATIVE scene: no predator stimulus or behaviour; decoration only; must not change '
                           'looming or retinal input' if 'predator' in NAME else
                           'DECORATIVE scene: presentation only; not an arena, stimulus, odour field or taste contact')
root['neurofly_caption'] = caption
for o in list(bpy.data.objects):
    if o.parent is None and o is not root and o.type in ('MESH', 'EMPTY'):
        o.parent = root
        o.matrix_parent_inverse = root.matrix_world.inverted()

# ------------------------------------------------------------------ lights (exported) + studio (render only)
sun_data = bpy.data.lights.new('scene_sun', 'SUN')
sun_data.energy = 3.0
sun_data.color = (1.0, 0.93, 0.82)
sun_data.angle = math.radians(6)
sun = bpy.data.objects.new('scene_sun', sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(35))
fill_data = bpy.data.lights.new('scene_fill', 'POINT')
fill_data.energy = 60000.0
fill_data.color = (0.75, 0.85, 1.0)
fill = bpy.data.objects.new('scene_fill', fill_data)
scene.collection.objects.link(fill)
fill.location = Vector(g.to_blender((-60, 70, 50)))
export_objs = [o for o in bpy.data.objects if o.type in ('MESH', 'EMPTY', 'LIGHT')]

meshes = [o for o in bpy.data.objects if o.type == 'MESH']
tris = g.triangle_count(meshes)
bpy.context.view_layer.update()
pts = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
lo = [min(p[i] for p in pts) for i in range(3)]
hi = [max(p[i] for p in pts) for i in range(3)]
stem = os.path.join(ARGS['out'], 'scene_' + NAME)
bpy.ops.object.select_all(action='DESELECT')
for o in export_objs:
    o.select_set(True)
bpy.ops.export_scene.gltf(filepath=stem + '.glb', export_format='GLB', export_yup=True, export_apply=True,
                          export_extras=True, export_cameras=False, export_lights=True, use_selection=True)
print(f'NEUROFLY_SCENE {NAME} triangles={tris} budget={BUDGET_TRIS}')
info = {'scene': NAME, 'caption': caption, 'triangles': tris, 'triangle_budget': BUDGET_TRIS,
        'within_budget': tris <= BUDGET_TRIS, 'status': root['neurofly_status'],
        'bbox_three_min': [round(lo[0], 2), round(lo[2], 2), round(-hi[1], 2)],
        'bbox_three_max': [round(hi[0], 2), round(hi[2], 2), round(-lo[1], 2)],
        'placements': placements, 'lights_exported': ['scene_sun (directional)', 'scene_fill (point)'],
        'units': 'viewport-mm (fly display scale); food props at nominal size'}

if int(ARGS['render']):
    res = int(ARGS['res'])
    g.setup_render(scene, (res, int(res * 9 / 16)), samples=int(ARGS['samples']))
    scene.render.film_transparent = False
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.42, 0.47, 0.55, 1.0)
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    g.studio_lights(scale=span * 0.6)
    for o in [o for o in bpy.data.objects if o.type == 'LIGHT' and o.name in ('KeyLight', 'FillLight', 'RimLight')]:
        o.data.energy *= 0.35
    # Caption in the hero render only (a text object, not exported).
    cu = bpy.data.curves.new('caption', 'FONT')
    cu.body = caption
    cu.size = span * 0.028
    cu.align_x = 'CENTER'
    cap = bpy.data.objects.new('render_caption', cu)
    cap.data.materials.append(g.material('caption_mat', (0.95, 0.62, 0.12) if 'predator' in NAME else
                                         (0.95, 0.95, 0.92), roughness=0.6))
    scene.collection.objects.link(cap)
    cap.location = Vector(g.to_blender((0, 0.3, span * 0.47)))
    focus = {'fermenting_fruit_patch': (0, 2, 0), 'sugar_water_feeder': (0, 2, 2), }.get(NAME, (-6, 3, 8))
    hero = g.camera('cam_hero', (focus[0] + span * 0.35, span * 0.42, focus[2] + span * 0.62), focus, lens=40)
    g.render_to(scene, hero, stem + '_hero.png')
    cf = {'predator_encounter_illustrative': (0, 2.5, 10)}.get(NAME, (focus[0], 2.5, focus[2]))
    cpos = {'predator_encounter_illustrative': (30, 13, 6)}.get(NAME, (focus[0] + 14, 11, focus[2] + 22))
    close = g.camera('cam_close', cpos, cf, lens=50)
    g.render_to(scene, close, stem + '_close.png')
    scene.render.resolution_y = res
    top = g.camera('cam_top', (0, span * 3, 0), (0, 0, 0), ortho_scale=span * 1.05)
    cap.hide_render = True
    g.render_to(scene, top, stem + '_top.png')
    info['renders'] = {'hero': 'perspective 16:9 with caption (caption not exported)', 'close': 'perspective close-up',
                       'top': f'orthographic, ortho_scale {round(span * 1.05, 2)} viewport-mm, image top = +Z',
                       'engine': 'Cycles CPU, 2 threads', 'samples': int(ARGS['samples'])}
bpy.ops.wm.save_as_mainfile(filepath=stem + '.blend')
with open(stem + '.json', 'w') as fh:
    json.dump(info, fh, indent=2)
print('NEUROFLY_SCENE done', NAME)
