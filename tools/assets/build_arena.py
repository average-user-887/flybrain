# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Generate the NeuroFly arena presentation shell (visual only).

Headless:  blender --background --factory-startup --threads 2 \
               --python tools/assets/build_arena.py -- --out OUTDIR [--render 1] [--fly OUTDIR/fly_hq_lod0.glb]

The shell is a plinth that sits entirely BELOW the dashboard's arena floor
(floor y = -0.02 viewport-mm; shell top y = 0 in its own frame, placed at
y = -0.06 by the loader).  It has no rim, wall, marker or tick: it must never read
as a boundary, a stimulus or a scale.  Both shapes are authored at unit size
(footprint 1 x 1, thickness 1) and the loader scales them to the arena's own
bounds plus a margin, so arena dimensions always come from the simulation.

Outputs: arena_shell.glb / .blend; with --render 1 transparent orthographic top
views of each shape (2D canvas backgrounds) and a perspective preview of the
shell with the fly GLB standing on it (same GLB the dashboard would load).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import nf_geom as g  # noqa: E402

ARGS = g.parse_args({'out': '', 'render': 0, 'samples': 48, 'res': 768, 'fly': ''})
CHAMFER = 0.025
SEG = 96

scene = g.reset_scene()
M = [
    g.material('nf_shell_top', (0.010, 0.015, 0.026), roughness=0.9),
    g.material('nf_shell_edge', (0.02, 0.028, 0.042), roughness=0.7),   # no glow: must not read as a wall
    g.material('nf_shell_side', (0.006, 0.009, 0.016), roughness=0.8),
]


def rect_shell():
    e = CHAMFER
    rings = [(0.0, 0.5 - e), (-e, 0.5), (-1.0, 0.5)]           # (y, half-size)
    verts, faces, mats = [], [], []
    for y, h in rings:
        verts += [(-h, y, -h), (h, y, -h), (h, y, h), (-h, y, h)]
    faces.append((3, 2, 1, 0)); mats.append(0)                 # top
    for r in range(2):
        for k in range(4):
            a, b = r * 4 + k, r * 4 + (k + 1) % 4
            faces.append((a, b, b + 4, a + 4)); mats.append(1 if r == 0 else 2)
    faces.append((8, 9, 10, 11)); mats.append(2)              # bottom
    geo = g.Geo()
    geo.verts = verts
    geo.faces = faces
    geo.mats = mats
    return geo


def round_shell():
    e = CHAMFER
    prof = [(0.0, 0.0), (0.0, 0.5 - e), (-e, 0.5), (-1.0, 0.5), (-1.0, 0.0)]
    return g.revolve(prof, SEG, mat=lambda j: 0 if j == 0 else (1 if j == 1 else 2))


shell_root = g.empty('arena_shell')
rect = g.mesh_object('shell_rect', rect_shell(), M, smooth=False, parent=shell_root)
rnd = g.mesh_object('shell_round', round_shell(), M, smooth=True, parent=shell_root)
objs = [rect, rnd]
print(f'NEUROFLY_ASSET arena triangles={g.triangle_count(objs)} rect={g.triangle_count([rect])} '
      f'round={g.triangle_count([rnd])}')
stem = os.path.join(ARGS['out'], 'arena_shell')
g.export_glb(stem + '.glb')
bpy.ops.wm.save_as_mainfile(filepath=stem + '.blend')

if int(ARGS['render']):
    res = int(ARGS['res'])
    g.setup_render(scene, (res, res), samples=int(ARGS['samples']))
    g.studio_lights(scale=2.0)
    # 2D views: each shape alone, unit footprint filling 1.04 of the frame.
    cam = g.camera('cam_top', (0, 20, 0), (0, 0, 0), ortho_scale=1.04)
    for shown, name in ((rect, 'rect'), (rnd, 'round')):
        rect.hide_render = shown is not rect
        rnd.hide_render = shown is not rnd
        g.render_to(scene, cam, f'{stem}_{name}_top.png')
    # Perspective preview: 60 x 40 mm rectangle shell (example size only) plus the fly GLB.
    rnd.hide_render = True
    rect.hide_render = False
    rect.matrix_basis = g.matrix_to_blender(g.trans(0, -0.06, 0) @ g.scale3(60 + 8, 0.6, 40 + 8))
    for light in [o for o in scene.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(light)
    g.studio_lights(scale=40.0)
    floor_geo = g.Geo()
    floor_geo.verts = [(-30, -0.02, -20), (30, -0.02, -20), (30, -0.02, 20), (-30, -0.02, 20)]
    floor_geo.faces = [(3, 2, 1, 0)]
    floor_geo.mats = [0]
    floor_mat = g.material('preview_dashboard_floor', (0.004, 0.007, 0.014), roughness=0.95, alpha=0.72)
    g.mesh_object('preview_dashboard_floor', floor_geo, [floor_mat], smooth=False)
    if ARGS['fly'] and os.path.exists(ARGS['fly']):
        bpy.ops.import_scene.gltf(filepath=ARGS['fly'])
    hero = g.camera('cam_hero', (52, 40, 58), (0, 0, 0), lens=45)
    scene.render.resolution_y = int(res * 0.625)
    g.render_to(scene, hero, f'{stem}_preview.png')
    print('NEUROFLY_ASSET arena renders done')
