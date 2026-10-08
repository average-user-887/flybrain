# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Wind-vane orientation mapping from published wind telemetry, plus one labelled DEMO clip.

Headless:
    blender --background --factory-startup --threads 2 \
        --python scripts/animations/build_env_vane.py -- --props ENV_DIR --out OUTDIR

Reads ENV_DIR/env_wind_vane.glb (W3 static prop, only read) and writes OUTDIR/env_wind_vane_demo.glb,
.blend and env_wind_vane_mapping.json.

* LIVE orientation is NOT an animation: a player sets the rotor from telemetry every frame with
  the mapping in the JSON (sensory.wind_x / sensory.wind_y in mm/s, arena frame).  A present zero
  vector is CALM (0.00 mm/s); a missing/non-finite field is NOT SIMULATED. Both leave the vane at rest.
* The only clip, ``DEMO_vane_swing``, is a deterministic sinusoidal swing for gallery preview.  Its
  name, extras and metadata say DEMO; it must never be shown as measured or simulated wind.
* Odour props get no animation at all: odour fields are static scalar fields with no transport
  (no plume, advection, turbulence or emission rate).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'tools', 'assets'))

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402

import nf_geom as g  # noqa: E402

ARGS = g.parse_args({'props': '', 'out': '', 'fps': 30, 'seconds': 6.0, 'amp_deg': 60.0})
scene = g.reset_scene()
scene.render.fps = int(ARGS['fps'])
scene.frame_start = 1
src = os.path.join(ARGS['props'], 'env_wind_vane.glb')
bpy.ops.import_scene.gltf(filepath=src)
rotor = bpy.data.objects['wind_vane_rotor']
root = bpy.data.objects['env_wind_vane']
rotor.rotation_mode = 'QUATERNION'
n = int(round(float(ARGS['seconds']) * int(ARGS['fps'])))
amp = math.radians(float(ARGS['amp_deg']))
for i in range(n):
    yaw = amp * math.sin(2 * math.pi * i / n)
    rotor.matrix_basis = g.matrix_to_blender(Matrix.Rotation(yaw, 4, 'Y'))
    rotor.keyframe_insert('rotation_quaternion', frame=i + 1)
act = rotor.animation_data.action
act.name = 'DEMO_vane_swing__wind_vane_rotor'
tr = rotor.animation_data.nla_tracks.new()
tr.name = 'DEMO_vane_swing'
tr.strips.new('DEMO_vane_swing', 1, act)
rotor.animation_data.action = None
rotor.matrix_basis = Matrix.Identity(4)
root['neurofly_demo_clip'] = ('DEMO_vane_swing is a deterministic preview swing (DEMO), not measured or simulated '
                              'wind. Live orientation comes from telemetry via env_wind_vane_mapping.json.')
os.makedirs(ARGS['out'], exist_ok=True)
stem = os.path.join(ARGS['out'], 'env_wind_vane_demo')
bpy.ops.export_scene.gltf(filepath=stem + '.glb', export_format='GLB', export_yup=True, export_apply=True,
                          export_extras=True, export_cameras=False, export_lights=False, export_animations=True,
                          export_animation_mode='NLA_TRACKS', export_force_sampling=False,
                          export_optimize_animation_size=False)
tr.mute = True
bpy.ops.wm.save_as_mainfile(filepath=stem + '.blend')
mapping = {
    'node': 'wind_vane_rotor (child of env_wind_vane); rotate about +Y only',
    'telemetry': {'keys': ['sensory.wind_x', 'sensory.wind_y'], 'units': 'mm/s', 'frame': 'arena (x, y), as published '
                  'by the dashboard telemetry (see the W1 sensory contract)'},
    'frame_map': 'arena (x, y) -> three.js (x - cx, ., -(y - cy)) (INTERFACE.md arenaPointFor3D); a wind vector '
                 '(wx, wy) is therefore (wx, 0, -wy) in the viewport',
    'formula': 'rotor yaw theta = atan2(wx, -wy) radians about +Y; the rotor local +Z (arrow tip) then points where '
               'the air moves TO (downwind); the orange fin trails upwind',
    'checks': {'(wx, wy) = (0, +1)': 'theta = atan2(0, -1) = pi: arrow points to viewport -Z, i.e. arena +y',
               '(wx, wy) = (+1, 0)': 'theta = pi/2: arrow points to viewport +X, i.e. arena +x'},
    'calm': 'finite telemetry vector with speed below 1e-6 mm/s: leave the rotor at rest and show CALM (0.00 mm/s); never animate',
    'missing': 'non-finite values or no wind field: leave the rotor at rest and show NOT SIMULATED; never animate',
    'magnitude': 'the vane shows direction only; show the speed as a number with units, not as vane motion',
    'demo_clip': {'name': 'DEMO_vane_swing', 'seconds': float(ARGS['seconds']), 'amplitude_deg': float(ARGS['amp_deg']),
                  'label': 'DEMO: deterministic preview swing, not measured or simulated wind'},
    'odour': 'odour emitter and source marker have NO animation: odour fields are static, normalised scalar fields with '
             'no transport; no plume, advection, turbulence or emission-rate values are drawn',
    'source_glb': os.path.basename(src),
}
with open(os.path.join(ARGS['out'], 'env_wind_vane_mapping.json'), 'w') as fh:
    json.dump(mapping, fh, indent=2)
print('NEUROFLY_VANE done')
