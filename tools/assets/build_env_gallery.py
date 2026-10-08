# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Lay the environment props (and, for scale, the fly GLB) out in one review scene.

Headless:  blender --background --factory-startup --threads 2 \
               --python tools/assets/build_env_gallery.py -- --out OUTDIR [--fly FLY_GLB] [--render 1]

Imports the very GLBs written by build_env.py, so the gallery shows what a browser
would load.  Writes OUTDIR/env_gallery.blend and, with --render 1,
env_gallery_hero.png and env_gallery_top.png.  Review scene only: nothing here is a
stimulus, and the floor and labels are not part of any asset.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import nf_geom as g  # noqa: E402

ARGS = g.parse_args({'out': '', 'fly': '', 'render': 0, 'samples': 48, 'res': 1600})
ROWS = [
    [('fermenting_fruit', 'fermenting fruit'), ('yeast_patch', 'yeast patch'),
     ('sugar_water', 'sugar water\nCONTACT taste'), ('odour_emitter', 'odour source\nVOLATILE'),
     ('wind_vane', 'wind vane\ntip = downwind')],
    [('fly', 'fly\ndisplay scale'), ('jumping_spider', 'jumping spider\nILLUSTRATIVE'),
     ('mantis_nymph', 'mantis nymph\nILLUSTRATIVE')],
]
GAP = 14.0

scene = g.reset_scene()
label_mat = g.material('gallery_label', (0.92, 0.92, 0.9), roughness=0.6)
warn_mat = g.material('gallery_label_illustrative', (0.95, 0.62, 0.12), roughness=0.6)
floor_mat = g.material('gallery_floor', (0.05, 0.06, 0.08), roughness=0.9)


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in new if o.parent is None]
    return roots, new


def bounds(objs):
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ v.co for o in objs if o.type == 'MESH' for v in o.data.vertices]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi


def label(text, x, y, mat, size=1.8):
    cu = bpy.data.curves.new('label_' + text.split(chr(10))[0], 'FONT')
    cu.body = text
    cu.align_y = 'TOP'
    cu.size = size
    cu.align_x = 'CENTER'
    cu.extrude = 0.02
    ob = bpy.data.objects.new('label_' + text.split(chr(10))[0], cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = (x, y, 0.01)
    return ob


row_y = 0.0
placed = []
for row in ROWS:
    items = []
    for key, text in row:
        path = ARGS['fly'] if key == 'fly' else os.path.join(ARGS['out'], f'env_{key}.glb')
        if not path or not os.path.exists(path):
            print('NEUROFLY_ASSET gallery skip', key)
            continue
        roots, objs = import_glb(path)
        lo, hi = bounds(objs)
        items.append((key, text, roots, objs, lo, hi))
    x = 0.0
    row_depth = max((hi.y - lo.y) for *_, lo, hi in items)
    for key, text, roots, objs, lo, hi in items:
        width = hi.x - lo.x
        dx = x - lo.x
        dy = row_y - (lo.y + hi.y) / 2
        dz = -lo.z if key == 'fly' else 0.0      # stand the fly on the floor (preview only)
        for r in roots:
            r.location += Vector((dx, dy, dz))
        mat = warn_mat if 'ILLUSTRATIVE' in text else label_mat
        label(text, x + width / 2, row_y - row_depth / 2 - 2.5, mat)
        placed.append((key, x, width))
        x += width + GAP
    row_y -= row_depth + 18.0

lo, hi = bounds([o for o in bpy.data.objects if o.type == 'MESH'])
fl = g.Geo()
pad = 8.0
# Blender frame -> three.js frame for the floor quad (y = -0.02 in three.js)
fl.verts = [(lo.x - pad, -0.02, -(lo.y - pad - 6)), (hi.x + pad, -0.02, -(lo.y - pad - 6)),
            (hi.x + pad, -0.02, -(hi.y + pad)), (lo.x - pad, -0.02, -(hi.y + pad))]
fl.faces = [(0, 1, 2, 3)]
fl.mats = [0]
g.mesh_object('gallery_floor', fl, [floor_mat], smooth=False)
print('NEUROFLY_ASSET gallery placed ' + ' '.join(k for k, *_ in placed))
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ARGS['out'], 'env_gallery.blend'))

if int(ARGS['render']):
    res = int(ARGS['res'])
    g.setup_render(scene, (res, int(res * 0.6)), samples=int(ARGS['samples']))
    c = (lo + hi) / 2
    span = max(hi.x - lo.x, hi.y - lo.y) + 2 * pad
    g.studio_lights(scale=span * 0.6)
    ct = (c.x, 0.0, -c.y)                         # three.js frame
    hero = g.camera('cam_gallery_hero', (ct[0] - span * 0.12, span * 0.75, ct[2] + span * 0.95), ct, lens=40)
    g.render_to(scene, hero, os.path.join(ARGS['out'], 'env_gallery_hero.png'))
    top = g.camera('cam_gallery_top', (ct[0], span * 2, ct[2]), ct, ortho_scale=span)
    top.rotation_euler.z += 3.141592653589793      # labels read upright: image top = -Z (back row)
    g.render_to(scene, top, os.path.join(ARGS['out'], 'env_gallery_top.png'))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ARGS['out'], 'env_gallery.blend'))
    print('NEUROFLY_ASSET gallery renders done')
