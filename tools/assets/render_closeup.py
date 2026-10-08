# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Close-up render of one object in a built .blend (for inspecting small details).

    blender --background --threads 2 OUT/fly_male_lod0.blend \
        --python tools/assets/render_closeup.py -- lf_tarsus OUT/fly_male_lod0_sexcomb_closeup.png
"""
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nf_geom as g  # noqa: E402

name, out = sys.argv[sys.argv.index('--') + 1:][:2]
scene = bpy.context.scene
obj = bpy.data.objects[name]
bpy.context.view_layer.update()
pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
centre = sum(pts, Vector()) / len(pts)
target = max(pts, key=lambda p: p.z) * 0.6 + centre * 0.4
g.setup_render(scene, (512, 512), samples=24)
g.studio_lights(scale=4.0)
data = bpy.data.cameras.new('closeup')
data.lens = 100
cam = bpy.data.objects.new('closeup', data)
scene.collection.objects.link(cam)
cam.location = target + Vector((1.2, -1.6, 0.6))
cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
g.render_to(scene, cam, out)
