# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Illustrative animation clips for the NeuroFly presentation fly (rig v2).

Headless:
    blender --background --factory-startup --threads 2 \
        --python scripts/animations/build_fly_clips.py -- \
        --glb RIG_V2_DIR/fly_<sex>_v2_lod<N>.glb --out OUTDIR [--render 1] [--res 256]

Reads a rig-v2 GLB and the ``*_joints.json`` written next to it (W2 joint contract,
tools/assets/JOINT_CONTRACT.md).  Writes into OUTDIR (a new versioned directory; the
rig's own outputs are only read, never re-saved):

* ``<stem>_clips.glb``: the same geometry with every clip as a named glTF animation
  (one NLA track per clip; every controlled node is keyed in every clip, so switching
  clips never leaves a joint in another clip's pose);
* ``<stem>_clips.blend``: the editable scene (one NLA track per clip);
* ``<stem>_clips.json``: clip list, frames, loop flags, DOFs, the measured envelope
  and penetration report;
* with ``--render 1``: one transparent sprite sheet per clip rendered from the SAME
  posed objects, and side/top/underside check sheets of the full grooming cycle.

PRESENTATION ONLY.  Every clip is an illustrative animation, not simulated behaviour.
It implies no aerodynamics, mechanosensation, grooming decision or neural output, and
it must never be overlaid on recorded or live body or neural telemetry.  The root,
thorax frame, joint origins and floor contact never move; only joint rotations are
keyed.  Display time is slowed (the wingbeat by about 180x).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'tools', 'assets'))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

import nf_geom as g  # noqa: E402

ARGS = g.parse_args({'out': '', 'glb': '', 'render': 0, 'res': 256, 'samples': 16, 'frames': 8, 'fps': 30, 'clips': '', 'debug': 0})
FPS = int(ARGS['fps'])

# ------------------------------------------------------------------ rig facts
# Leg layout and display pose: tools/assets/build_fly.py (v1 frames, unchanged in v2).
LEGS = {'lf': (-1, 0.8, -math.pi / 4), 'lm': (-1, 0.0, -math.pi / 2), 'lh': (-1, -0.8, -3 * math.pi / 4),
        'rf': (1, 0.8, math.pi / 4), 'rm': (1, 0.0, math.pi / 2), 'rh': (1, -0.8, 3 * math.pi / 4)}
DISPLAY_POSE = {'f': (-0.35, -0.75, 1.55), 'm': (-0.45, -0.8, 1.6), 'h': (-0.4, -0.75, 1.5)}
L0, L1, L2 = 1.0, 2.2, 2.4          # coxa, femur, tibia+tarsus (tarsus tip at -2.4 in the tibia group)
SHELLS = ('c_head', 'l_eye', 'r_eye', 'c_thorax', 'c_scutum', 'c_scutellum', 'c_proboscis', 'c_abdomen12',
          'c_abdomen3', 'c_abdomen4', 'c_abdomen5', 'c_abdomen6')
# Wing display-safe envelope (JOINT_CONTRACT.md): minimum elevate for a sweep.
MIN_ELEV = [(0, 0), (0.25, 0), (0.5, -0.1), (0.75, -0.9), (1.0, -0.8), (1.25, -0.6), (1.5, -0.3), (1.75, 0),
            (2.0, 0.2), (2.25, 0.4), (3.0, 0.4)]


def min_elev(sweep):
    for (s0, e0), (s1, e1) in zip(MIN_ELEV, MIN_ELEV[1:]):
        if s0 <= sweep <= s1:
            return e0 + (e1 - e0) * (sweep - s0) / (s1 - s0)
    return 0.4


# Conservative display-safe envelope from JOINT_CONTRACT.md, used for EVERY sample of every clip:
# pitch must be 0 while folded (sweep < 0.5) and otherwise lie inside the intersection of all the
# contract's measured pitch intervals, [-0.07, 0.13]; antenna uses the simple envelope
# (abduct [-0.4, 0.6], extend [-0.2, 0.2]) and the measured max-extend-per-abduct table.
PITCH_MAX, PITCH_MIN_ABS = 0.12, 0.06
ENV_LIMITS = {'wing_sweep': (0.0, 3.0), 'wing_elevate': (None, 0.8), 'wing_pitch': (-0.07, 0.13),
              'antenna_abduct': (-0.4, 0.6), 'antenna_extend': (-0.2, 0.2), 'antenna_twist': (-0.1, 0.09),
              'funiculus_rotate': (-0.1, 0.1), 'haltere_beat': (-0.2, 0.2)}
MAX_EXT = [(-0.4, 0.40), (0.0, 0.40), (0.1, 0.35), (0.3, 0.35), (0.4, 0.30), (0.5, 0.25), (0.6, 0.20),
           (0.7, 0.15), (0.8, -0.05)]


def max_extend(abduct):
    for (a0, e0), (a1, e1) in zip(MAX_EXT, MAX_EXT[1:]):
        if a0 <= abduct <= a1:
            return e0 + (e1 - e0) * (abduct - a0) / (a1 - a0)
    return -0.05


def ease(t):
    t = min(1.0, max(0.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def lerp(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------------ scene
scene = g.reset_scene()
scene.render.fps = FPS
bpy.ops.import_scene.gltf(filepath=ARGS['glb'])
OBJ = bpy.data.objects
root = OBJ['neurofly_fly']
thorax = OBJ['thorax_frame']
stem = os.path.splitext(os.path.basename(ARGS['glb']))[0]
with open(os.path.splitext(ARGS['glb'])[0] + '_joints.json') as fh:
    CONTRACT = json.load(fh)
AXES = {j['node']: Vector(j['axis']) for j in CONTRACT['joints'] if 'axis' in j}
RANGES = {j['node']: tuple(j['range_rad']) for j in CONTRACT['joints'] if 'axis' in j}
DOFS = sorted(AXES)
LEG_NODES = [f'{leg}_{j}_joint' for leg in LEGS for j in ('coxa', 'femur', 'tibia')]
for n in DOFS + LEG_NODES:
    OBJ[n].rotation_mode = 'QUATERNION'


def leg_mats(leg, yaw, c, f, t):
    side, z_off, _ = LEGS[leg]
    return (g.trans(side * 1.0, -0.2, z_off) @ g.rot_y(yaw) @ g.rot_x(c),
            g.trans(0, -L0, 0) @ g.rot_x(f), g.trans(0, -L1, 0) @ g.rot_x(t))


def rest_leg(leg):
    return (LEGS[leg][2],) + DISPLAY_POSE[leg[1]]


# The imported display pose must be exactly the documented one; otherwise stop.
for leg in LEGS:
    for name, m in zip(('coxa', 'femur', 'tibia'), leg_mats(leg, *rest_leg(leg))):
        got = OBJ[f'{leg}_{name}_joint'].matrix_basis
        want = g.matrix_to_blender(m)
        err = max(abs(got[i][j] - want[i][j]) for i in range(4) for j in range(4))
        if err > 1e-4:
            raise SystemExit(f'rig mismatch at {leg}_{name}_joint (max error {err:.2e}); check the joint contract')


def leg_ik(leg, target, alpha0, knee=1):
    """Planar IK in the leg's vertical plane through the coxa origin (thorax frame).

    Returns (yaw, c, f, t) for the coxa/femur/tibia joints so that the tarsus tip
    (0, -2.4, 0 in the tibia group) reaches ``target``.  alpha0 is the coxa angle from
    straight down toward the target; knee = +1 / -1 picks the elbow solution.
    """
    side, z_off, _ = LEGS[leg]
    o = Vector((side * 1.0, -0.2, z_off))
    p = Vector(target)
    dx, dz = p.x - o.x, p.z - o.z
    yaw = math.atan2(dx, dz)
    up, vp = math.hypot(dx, dz), p.y - o.y
    ku, kv = L0 * math.sin(alpha0), -L0 * math.cos(alpha0)
    wu, wv = up - ku, vp - kv
    d = min(L1 + L2 - 1e-3, max(abs(L1 - L2) + 1e-3, math.hypot(wu, wv)))
    aw = math.atan2(wu, -wv)
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))))
    a1 = aw + knee * beta
    nu, nv = ku + L1 * math.sin(a1), kv - L1 * math.cos(a1)
    a2 = math.atan2(up - nu, -(vp - nv))
    return (yaw, -alpha0, -(a1 - alpha0), -(a2 - a1))


def apply_pose(pose):
    for n in DOFS:
        a = pose.get(n, 0.0)
        OBJ[n].matrix_basis = g.matrix_to_blender(Matrix.Rotation(a, 4, AXES[n]))
    for leg in LEGS:
        ang = pose.get(leg) or rest_leg(leg)
        for name, m in zip(('coxa', 'femur', 'tibia'), leg_mats(leg, *ang)):
            OBJ[f'{leg}_{name}_joint'].matrix_basis = g.matrix_to_blender(m)


# ------------------------------------------------------------------ penetration test
bpy.context.view_layer.update()
shell_objs = [OBJ[n] for n in SHELLS if n in OBJ]


def world_bvh(objs):
    verts, polys = [], []
    for o in objs:
        base = len(verts)
        mw = o.matrix_world
        verts += [mw @ v.co for v in o.data.vertices]
        polys += [[base + i for i in p.vertices] for p in o.data.polygons]
    return BVHTree.FromPolygons(verts, polys)


SHELLS_W = []
for _o in shell_objs:
    _pts = [_o.matrix_world @ v.co for v in _o.data.vertices]
    _lo = Vector([min(p[i] for p in _pts) for i in range(3)])
    _hi = Vector([max(p[i] for p in _pts) for i in range(3)])
    SHELLS_W.append((world_bvh([_o]), _lo, _hi))
HINGE_EXCLUDE = 0.40 * float(root.matrix_world.to_scale()[0])
COXA_EXCLUDE = 0.45 * float(root.matrix_world.to_scale()[0])
MOVING = [o for o in OBJ if o.type == 'MESH' and (
    o.name.endswith(('_coxa', '_trochanterfemur', '_tibia', '_tarsus')) or 'wing' in o.name
    or o.name.endswith(('_pedicel', '_funiculus', '_arista', '_haltere')))]
RAY_DIRS = (Vector((1, 0, 0)), Vector((0, 0.6, 0.8)), Vector((-0.6, -0.8, 0.0)))


def _hits(bvh, p, d):
    n, q = 0, p.copy()
    for _ in range(64):
        hit = bvh.ray_cast(q, d)
        if hit[0] is None:
            break
        n += 1
        q = hit[0] + d * 1e-5
    return n


def depth_inside(p):
    """Depth of p inside any closed body shell (ray-parity vote over three directions), else 0."""
    best = 0.0
    for bvh, lo, hi in SHELLS_W:
        if not (lo.x <= p.x <= hi.x and lo.y <= p.y <= hi.y and lo.z <= p.z <= hi.z):
            continue
        votes = sum(_hits(bvh, p, d) % 2 for d in RAY_DIRS)
        if votes >= 2:
            near = bvh.find_nearest(p)
            best = max(best, near[3] if near[0] is not None else 0.0)
    return best


def inside_flags():
    """Per moving mesh: array of per-vertex depth inside a body shell (0 = outside)."""
    out = {}
    for o in MOVING:
        mw = o.matrix_world
        hinge, radius = None, 0.0
        if 'wing' in o.name:                            # wing insertion under the notum (as W2 measures)
            hinge, radius = OBJ[o.name[:2] + 'wing_root'].matrix_world.translation, HINGE_EXCLUDE
        elif o.name.endswith('_coxa'):                  # coxa base sits in its thoracic socket
            hinge, radius = OBJ[o.name + '_joint'].matrix_world.translation, COXA_EXCLUDE
        depth = []
        for v in o.data.vertices:
            p = mw @ v.co
            if hinge is not None and (p - hinge).length < radius:
                depth.append(0.0)
                continue
            depth.append(depth_inside(p))
        out[o.name] = np.array(depth, dtype=float)
    return out


apply_pose({})
bpy.context.view_layer.update()
REST_INSIDE = inside_flags()
LEG_MESHES = [o for o in MOVING if o.name[:2] in LEGS]
FLOOR_Z = min((o.matrix_world @ v.co).z for o in LEG_MESHES if o.name.endswith('_tarsus') for v in o.data.vertices)


def below_floor():
    """How far any leg vertex dips below the display-pose foot plane (static preview floor)."""
    low = min((o.matrix_world @ v.co).z for o in LEG_MESHES for v in o.data.vertices)
    return max(0.0, FLOOR_Z - low)


def new_penetration():
    """Vertices inside a shell now that were not inside at rest: (count, max depth, worst mesh)."""
    cur = inside_flags()
    count, depth, worst = 0, 0.0, ''
    for name, arr in cur.items():
        new = (arr > 0) & (REST_INSIDE[name] <= 0)
        if new.any():
            count += int(new.sum())
            m = float(arr[new].max())
            if m > depth:
                depth, worst = m, name
    return count, depth, worst


# ------------------------------------------------------------------ clips
def wing_pose(side, sweep, elev, pitch):
    return {f'{side}_wing_sweep': sweep, f'{side}_wing_elevate': elev, f'{side}_wing_pitch': pitch}


def both(fn, *a):
    pose = {}
    for s in ('l', 'r'):
        pose.update(fn(s, *a))
    return pose


def clip_wing_open_fold():
    frames = []
    n = 72
    for i in range(n):
        if i < 10:
            s = 0.0
        elif i < 34:
            s = ease((i - 10) / 24)
        elif i < 46:
            s = 1.0
        elif i < 70:
            s = 1.0 - ease((i - 46) / 24)
        else:
            s = 0.0
        sweep, elev = 1.5 * s, 0.3 * s
        pitch = PITCH_MAX * min(1.0, max(0.0, (sweep - 0.6) / 0.6))   # 0 until sweep 0.6 (folded rule)
        frames.append(both(wing_pose, sweep, elev, pitch))
    return frames, False


def clip_wingbeat_loop():
    frames = []
    n = 24
    for i in range(n):
        ph = i / n
        sweep = 1.55 + 1.2 * math.cos(2 * math.pi * ph)            # 0.35..2.75 rad, about 137 deg
        elev = min(0.75, max(min_elev(sweep) + 0.12, 0.12) + 0.04 * math.sin(4 * math.pi * ph))
        ramp = min(1.0, max(0.0, (sweep - 0.6) / 0.3)) * min(1.0, max(0.0, (1.95 - sweep) / 0.3))
        pitch = (PITCH_MAX if math.sin(2 * math.pi * ph) > 0 else -PITCH_MIN_ABS) * abs(math.sin(2 * math.pi * ph)) * ramp
        pose = both(wing_pose, sweep, elev, pitch)
        hal = -0.2 * math.cos(2 * math.pi * ph)                    # antiphase with the stroke
        pose.update({'l_haltere_beat': hal, 'r_haltere_beat': hal,
                     'l_funiculus_rotate': 0.03 * math.sin(2 * math.pi * ph),
                     'r_funiculus_rotate': 0.03 * math.sin(2 * math.pi * ph)})
        frames.append(pose)
    return frames, True


def clip_antenna_sweep():
    frames = []
    n = 60
    for i in range(n):
        ph = i / n
        ab = 0.1 + 0.35 * math.sin(2 * math.pi * ph)
        ex = 0.07 + 0.12 * math.sin(2 * math.pi * ph + math.pi / 2)
        pose = {}
        for s in ('l', 'r'):
            pose.update({f'{s}_antenna_abduct': ab, f'{s}_antenna_extend': ex,
                         f'{s}_antenna_twist': 0.07 * math.sin(2 * math.pi * ph),
                         f'{s}_funiculus_rotate': 0.09 * math.sin(4 * math.pi * ph)})
        frames.append(pose)
    return frames, True


# Grooming: tarsus-tip targets for the LEFT leg in the thorax frame; the right leg mirrors x.
def mirror(p):
    return (-p[0], p[1], p[2])


def sweep_path(p0, p1, n):
    return [tuple(lerp(a, b, k / (n - 1)) for a, b in zip(p0, p1)) for k in range(n)]


class Timeline:
    """Key poses -> per-frame poses.  A leg value is ('ik', target, alpha0, knee) or None (display pose)."""

    def __init__(self):
        self.keys = []          # (frame, legs{leg: spec}, dofs{node: angle})

    def key(self, frame, legs=None, dofs=None):
        self.keys.append((frame, legs or {}, dofs or {}))

    def leg_angles(self, leg, spec):
        if spec is None:
            return rest_leg(leg)
        if spec == 'tuck':
            yaw, c, _, _ = rest_leg(leg)
            a1, t = (1.6, 2.4) if leg[1] == 'f' else (2.0, 2.7)
            return (yaw, c, -(a1 + c), t)           # femur raised, tibia folded: foot lifted clear
        _, target, alpha0, knee = spec
        return leg_ik(leg, target, alpha0, knee)

    def frames(self):
        out = []
        for (f0, l0, d0), (f1, l1, d1) in zip(self.keys, self.keys[1:]):
            for f in range(f0, f1):
                t = ease((f - f0) / (f1 - f0))
                pose = {}
                for leg in LEGS:
                    a, b = l0.get(leg), l1.get(leg)
                    if isinstance(a, tuple) and isinstance(b, tuple):
                        tgt = tuple(lerp(x, y, t) for x, y in zip(a[1], b[1]))
                        pose[leg] = leg_ik(leg, tgt, lerp(a[2], b[2], t), b[3])
                    else:
                        ja, jb = self.leg_angles(leg, a), self.leg_angles(leg, b)
                        pose[leg] = tuple(lerp(x, y, t) for x, y in zip(ja, jb))
                for n in set(d0) | set(d1):
                    pose[n] = lerp(d0.get(n, 0.0), d1.get(n, 0.0), t)
                out.append(pose)
        last = self.keys[-1]
        pose = {leg: self.leg_angles(leg, last[1].get(leg)) for leg in LEGS}
        pose.update(last[2])
        out.append(pose)
        return out


CONFIGS = [(a0, k) for k in (-1, 1) for a0 in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2)]


def best_config(pair, targets, dofs=None):
    """Pick one (coxa angle, knee side) for a whole stroke: least new penetration over its key targets."""
    best = None
    for a0, k in CONFIGS:
        worst = 0.0
        for t in targets:
            pose = dict(dofs or {})
            for leg, tgt in ((pair[0], t), (pair[1], mirror(t))):
                pose[leg] = leg_ik(leg, tgt, a0, k)
            apply_pose(pose)
            bpy.context.view_layer.update()
            worst = max(worst, new_penetration()[1], below_floor())
        if best is None or worst < best[0] - 1e-4:
            best = (worst, a0, k)
    apply_pose({})
    if int(ARGS['debug']):
        print(f'NEUROFLY_DEBUG best_config {pair} worst={best[0]:.3f} a0={best[1]} knee={best[2]}')
    return best[1], best[2]


def front(target, cfg):
    return {'lf': ('ik', target, cfg[0], cfg[1]), 'rf': ('ik', mirror(target), cfg[0], cfg[1])}


def front_lr(tl, tr, cfg):
    return {'lf': ('ik', tl, cfg[0], cfg[1]), 'rf': ('ik', mirror(tr), cfg[0], cfg[1])}


def hind(target, cfg):
    return {'lh': ('ik', target, cfg[0], cfg[1]), 'rh': ('ik', mirror(target), cfg[0], cfg[1])}


def hind_lr(tl, tr, cfg):
    return {'lh': ('ik', tl, cfg[0], cfg[1]), 'rh': ('ik', mirror(tr), cfg[0], cfg[1])}


# Paths (thorax frame, left side), offset outside the measured head / eye / abdomen shells.
EYE_STROKE = [(-1.12, 1.05, 2.15), (-1.05, 0.55, 2.45), (-0.9, -0.05, 2.5), (-0.75, -0.45, 2.35)]
EYE_RETURN = (-1.35, 1.2, 2.3)
ANT_STROKE = [(-0.45, 1.25, 2.4), (-0.6, 0.95, 2.85), (-0.7, 0.45, 3.1)]
ANT_RETURN = (-0.85, 1.45, 2.6)
RUB_FRONT = [(-0.12, -1.5, 2.0), (-0.12, -1.5, 2.6)]
# Hind legs sweep the ventrolateral abdomen in the vertical plane x = -1 (just outside its widest point).
ABD_STROKE = [(-1.25, -0.25, -2.0), (-1.25, -0.3, -3.0), (-1.1, -0.3, -4.1), (-0.85, -0.3, -5.05)]
ABD_RETURN = (-1.9, -0.9, -2.4)
WING_STROKE = [(-1.75, 0.95, -2.0), (-1.8, 0.95, -3.0), (-1.6, 1.0, -4.2)]
WING_RETURN = (-2.3, 0.3, -2.2)
RUB_HIND = [(-0.12, -0.95, -5.6), (-0.12, -1.35, -5.15)]
ANT_HOLD = {'l_antenna_abduct': 0.15, 'r_antenna_abduct': 0.15, 'l_antenna_extend': 0.18, 'r_antenna_extend': 0.18}
WING_LIFT = {'l_wing_elevate': 0.25, 'r_wing_elevate': 0.25}


CHOSEN = {}


def head_groom(tl, f):
    """Front legs: eyes x2, rub, antennae x2, rub (Seeds et al. 2014: eyes before antennae)."""
    c_eye = best_config(('lf', 'rf'), EYE_STROKE + [EYE_RETURN])
    c_ant = best_config(('lf', 'rf'), ANT_STROKE + [ANT_RETURN], ANT_HOLD)
    c_rub = best_config(('lf', 'rf'), RUB_FRONT)
    CHOSEN.update(eye=c_eye, antenna=c_ant, front_rub=c_rub)
    tl.key(f, {})
    f += 10
    tl.key(f, {'lf': 'tuck', 'rf': 'tuck'})
    f += 12
    tl.key(f, front(EYE_RETURN, c_eye))
    for _ in range(2):
        for i, p in enumerate(EYE_STROKE):
            f += 6 if i else 8
            tl.key(f, front(p, c_eye))
        f += 10
        tl.key(f, front(EYE_RETURN, c_eye))
    for k in range(2):                                    # leg rubbing, legs in antiphase
        f += 8
        tl.key(f, front_lr(RUB_FRONT[k % 2], RUB_FRONT[(k + 1) % 2], c_rub))
    f += 10
    tl.key(f, front(ANT_RETURN, c_ant), ANT_HOLD)
    for _ in range(2):
        for i, p in enumerate(ANT_STROKE):
            f += 6 if i else 8
            tl.key(f, front(p, c_ant), ANT_HOLD)
        f += 10
        tl.key(f, front(ANT_RETURN, c_ant), ANT_HOLD)
    for k in range(2):
        f += 8
        tl.key(f, front_lr(RUB_FRONT[k % 2], RUB_FRONT[(k + 1) % 2], c_rub))
    f += 12
    tl.key(f, {'lf': 'tuck', 'rf': 'tuck'})
    f += 10
    tl.key(f, {})
    return f


def body_groom(tl, f):
    """Hind legs: abdomen x2, wings x2, rub (Seeds et al. 2014: abdomen before wings)."""
    c_abd = best_config(('lh', 'rh'), ABD_STROKE + [ABD_RETURN])
    c_wing = best_config(('lh', 'rh'), WING_STROKE + [WING_RETURN], WING_LIFT)
    c_rub = best_config(('lh', 'rh'), RUB_HIND)
    CHOSEN.update(abdomen=c_abd, wing=c_wing, hind_rub=c_rub)
    f += 10
    tl.key(f, {'lh': 'tuck', 'rh': 'tuck'})
    f += 12
    tl.key(f, hind(ABD_RETURN, c_abd))
    for _ in range(2):
        for i, p in enumerate(ABD_STROKE):
            f += 6 if i else 8
            tl.key(f, hind(p, c_abd))
        f += 10
        tl.key(f, hind(ABD_RETURN, c_abd))
    f += 10
    tl.key(f, hind(WING_RETURN, c_wing), WING_LIFT)
    for _ in range(2):
        for i, p in enumerate(WING_STROKE):
            f += 7 if i else 8
            tl.key(f, hind(p, c_wing), WING_LIFT)
        f += 10
        tl.key(f, hind(WING_RETURN, c_wing), WING_LIFT)
    for k in range(3):
        f += 8
        tl.key(f, hind_lr(RUB_HIND[k % 2], RUB_HIND[(k + 1) % 2], c_rub))
    f += 10
    tl.key(f, hind(ABD_RETURN, c_abd))
    f += 12
    tl.key(f, {'lh': 'tuck', 'rh': 'tuck'})
    f += 10
    tl.key(f, {})
    return f


def clip_groom_head():
    tl = Timeline()
    head_groom(tl, 0)
    return tl.frames(), False


def clip_groom_abdomen_wings():
    tl = Timeline()
    tl.key(0, {})
    body_groom(tl, 0)
    return tl.frames(), False


def clip_groom_full_cycle():
    tl = Timeline()
    f = head_groom(tl, 0)
    f += 6
    tl.key(f, {})
    body_groom(tl, f)
    return tl.frames(), False


CLIPS = [('wing_open_fold', clip_wing_open_fold), ('wingbeat_loop', clip_wingbeat_loop),
         ('antenna_sweep', clip_antenna_sweep), ('groom_head_forelegs', clip_groom_head),
         ('groom_abdomen_wings_hindlegs', clip_groom_abdomen_wings), ('groom_full_cycle', clip_groom_full_cycle)]


def check_ranges(pose):
    """Every contract / display-safe-envelope violation of one (possibly sub-frame) pose."""
    bad = []
    for n in DOFS:
        a = pose.get(n, 0.0)
        lo, hi = RANGES[n]
        elo, ehi = ENV_LIMITS[n[2:]]
        lo = max(lo, elo) if elo is not None else lo
        hi = min(hi, ehi) if ehi is not None else hi
        if a < lo - 1e-9 or a > hi + 1e-9:
            bad.append(f'{n} range')
    for s in ('l', 'r'):
        sw, el = pose.get(f'{s}_wing_sweep', 0.0), pose.get(f'{s}_wing_elevate', 0.0)
        if el < min_elev(sw) - 1e-9:
            bad.append(f'{s}_wing_elevate below min for sweep')
        if sw < 0.5 and abs(pose.get(f'{s}_wing_pitch', 0.0)) > 1e-9:
            bad.append(f'{s}_wing_pitch nonzero while folded')
        ab, ex = pose.get(f'{s}_antenna_abduct', 0.0), pose.get(f'{s}_antenna_extend', 0.0)
        if ex > max_extend(ab) + 1e-9:
            bad.append(f'{s}_antenna_extend above max for abduct')
    return bad


SUBSTEPS = 10


def envelope_report(frames, loop):
    """Dense check: every frame and SUBSTEPS linear sub-samples between frames (glTF LINEAR
    interpolation of a single-axis rotation is linear in angle), including the loop wrap."""
    stats = {n: [math.inf, -math.inf] for n in DOFS}
    violations = {}
    pairs = list(zip(frames, frames[1:])) + ([(frames[-1], frames[0])] if loop else [])
    samples = 0
    for a, b in pairs:
        for k in range(SUBSTEPS):
            t = k / SUBSTEPS
            pose = {n: lerp(a.get(n, 0.0), b.get(n, 0.0), t) for n in DOFS}
            samples += 1
            for n in DOFS:
                stats[n][0] = min(stats[n][0], pose[n])
                stats[n][1] = max(stats[n][1], pose[n])
            for v in check_ranges(pose):
                violations[v] = violations.get(v, 0) + 1
    last = {n: frames[-1].get(n, 0.0) for n in DOFS}
    for v in check_ranges(last):
        violations[v] = violations.get(v, 0) + 1
    per_joint = {}
    for n in DOFS:
        lo, hi = RANGES[n]
        elo, ehi = ENV_LIMITS[n[2:]]
        per_joint[n] = {'min': round(stats[n][0], 4), 'max': round(stats[n][1], 4),
                        'limit': [max(lo, elo) if elo is not None else lo, min(hi, ehi) if ehi is not None else hi]}
    return {'samples': samples + 1, 'substeps_per_frame': SUBSTEPS, 'violations': sum(violations.values()),
            'violation_kinds': violations, 'per_joint': per_joint,
            'coupled_rules': ['wing elevate >= contract minimum for the current sweep (linear table)',
                              'wing pitch == 0 while sweep < 0.5', 'antenna extend <= measured max for abduct']}


# ------------------------------------------------------------------ bake
if ARGS['clips']:
    CLIPS = [c for c in CLIPS if c[0] in ARGS['clips'].split(',')]
CONTROLLED = DOFS + LEG_NODES
report = {}
built = []
for clip, fn in CLIPS:
    frames, loop = fn()
    for o in (OBJ[n] for n in CONTROLLED):
        if o.animation_data:
            o.animation_data.action = None            # keep earlier clips' NLA tracks
    worst = (0, 0.0, '')
    floor_worst = 0.0
    range_issues = []
    for i, pose in enumerate(frames):
        apply_pose(pose)
        bpy.context.view_layer.update()
        if i % 2 == 0 or i == len(frames) - 1:
            c, d, w = new_penetration()
            if int(ARGS['debug']) and d > 0.03:
                print(f'NEUROFLY_DEBUG {clip} frame={i} depth={d:.3f} n={c} mesh={w}')
            if d > worst[1]:
                worst = (c, d, w, i)
            bf = below_floor()
            if int(ARGS['debug']) and bf > 0.01:
                print(f'NEUROFLY_DEBUG {clip} frame={i} below_floor={bf:.3f}')
            floor_worst = max(floor_worst, bf)
        range_issues += check_ranges(pose)
        for n in CONTROLLED:
            OBJ[n].keyframe_insert('rotation_quaternion', frame=i)
    for n in CONTROLLED:
        o = OBJ[n]
        act = o.animation_data.action
        act.name = f'{clip}__{n}'
        act.use_fake_user = True
        track = o.animation_data.nla_tracks.new()
        track.name = clip
        strip = track.strips.new(clip, 0, act)
        strip.extrapolation = 'NOTHING'
        o.animation_data.action = None
    report[clip] = {'frames': len(frames), 'fps': FPS, 'seconds': round(len(frames) / FPS, 3), 'loop': loop,
                    'max_new_penetration_viewport_mm': round(worst[1], 4),
                    'new_penetration_vertices_at_worst_frame': worst[0],
                    'worst_mesh': worst[2],
                    'max_below_foot_plane_viewport_mm': round(floor_worst, 4), 'worst_frame': worst[3] if len(worst) > 3 else None,
                    'envelope_check': envelope_report(frames, loop)}
    built.append((clip, frames))
    print(f'NEUROFLY_ANIM {stem} {clip} envelope_violations={report[clip]["envelope_check"]["violations"]} '
          f'frames={len(frames)} max_new_penetration={worst[1]:.4f} '
          f'mesh={worst[2]} below_floor={floor_worst:.3f} range_issues={len(set(range_issues))}')

apply_pose({})
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = False
        t.is_solo = False
out_stem = os.path.join(ARGS['out'], stem + '_clips')
bpy.ops.object.select_all(action='DESELECT')
kwargs = dict(filepath=out_stem + '.glb', export_format='GLB', export_yup=True, export_apply=True,
              export_extras=True, export_cameras=False, export_lights=False, export_animations=True,
              export_animation_mode='NLA_TRACKS', export_force_sampling=True, export_frame_step=1,
              export_optimize_animation_size=False)
bpy.ops.export_scene.gltf(**kwargs)
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = True                                  # .blend opens at rest; unmute a track to preview
bpy.ops.wm.save_as_mainfile(filepath=out_stem + '.blend')

meta = {
    'rig': CONTRACT.get('rig'), 'source_glb': os.path.basename(ARGS['glb']), 'fps': FPS,
    'controlled_nodes': CONTROLLED,
    'ik_configs_chosen': {k: {'coxa_angle_rad': v[0], 'knee_side': v[1]} for k, v in CHOSEN.items()},
    'label': 'Illustrative animation, not simulated behaviour.',
    'clips': report,
}

# ------------------------------------------------------------------ renders
if int(ARGS['render']):
    res = int(ARGS['res'])
    g.setup_render(scene, (res, res), samples=int(ARGS['samples']))
    g.studio_lights(scale=10.0)
    root.scale = (float(root.get('nf_display_scale', 1.0)),) * 3
    ortho = 12.0
    tgt = (0, 1.4, -0.9)
    views = {
        'side': g.camera('cam_side', (40, 1.4, -0.9), tgt, ortho_scale=ortho),
        'top': g.camera('cam_top', (0, 40, -0.9), (0, 0, -0.9), ortho_scale=ortho),
        'under': g.camera('cam_under', (0, -40, -0.9), (0, 0, -0.9), ortho_scale=ortho),
        'front': g.camera('cam_front', (0, 1.4, 40), tgt, ortho_scale=ortho),
        'hero': g.camera('cam_hero', (9.5, 8.0, 9.5), (0, 1.0, -0.6), lens=55),
    }
    tmp = os.path.join(ARGS['out'], '_tmp_frames')
    os.makedirs(tmp, exist_ok=True)

    def sheet(clip, frames, picks, view_list, name):
        rows = []
        for v in view_list:
            tiles = []
            for i in picks:
                apply_pose(frames[i])
                bpy.context.view_layer.update()
                p = os.path.join(tmp, f'{v}_{i}.png')
                g.render_to(scene, views[v], p)
                img = bpy.data.images.load(p)
                arr = np.array(img.pixels[:], dtype=np.float32).reshape(res, res, 4)
                bpy.data.images.remove(img)
                os.remove(p)
                tiles.append(arr)
            rows.append(np.concatenate(tiles, axis=1))
        full = np.concatenate(rows[::-1], axis=0)        # Blender pixel rows run bottom-up
        h, w = full.shape[:2]
        img = bpy.data.images.new(name, w, h, alpha=True)
        img.pixels = full.ravel()
        img.filepath_raw = os.path.join(ARGS['out'], name + '.png')
        img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)
        return {'file': name + '.png', 'frames': list(picks), 'views': view_list, 'tile_px': res,
                'ortho_scale_viewport_mm': ortho if all(v != 'hero' for v in view_list) else None}

    sheets = {}
    nf = int(ARGS['frames'])
    best_view = {'wing_open_fold': ['top', 'hero'], 'wingbeat_loop': ['top', 'front'],
                 'antenna_sweep': ['top', 'front'], 'groom_head_forelegs': ['side', 'front'],
                 'groom_abdomen_wings_hindlegs': ['side', 'top'], 'groom_full_cycle': ['side']}
    for clip, frames in built:
        n = len(frames)
        picks = [round(k * (n - 1) / (nf - 1)) for k in range(nf)] if not report[clip]['loop'] else \
            [round(k * n / nf) for k in range(nf)]
        sheets[clip] = sheet(clip, frames, picks, best_view[clip], f'{stem}_{clip}_sprites')
    # Full grooming cycle: side / top / underside checks at 12 evenly spaced frames.
    if 'groom_full_cycle' in dict(built):
        frames = dict(built)['groom_full_cycle']
        n = len(frames)
        picks = [round(k * (n - 1) / 11) for k in range(12)]
        sheets['groom_full_cycle_checks'] = sheet('groom_full_cycle', frames, picks, ['side', 'top', 'under'],
                                                  f'{stem}_groom_full_cycle_checks')
    os.rmdir(tmp)
    meta['sprite_sheets'] = sheets
    meta['render'] = {'engine': 'Cycles CPU, 2 threads', 'samples': int(ARGS['samples']),
                      'note': 'rendered from the same posed objects that are exported; rows = views in the '
                              'listed order, columns = frames; display scale applied as in the rig renders'}
    apply_pose({})
    root.scale = (1.0, 1.0, 1.0)

with open(out_stem + '.json', 'w') as fh:
    json.dump(meta, fh, indent=2)
print('NEUROFLY_ANIM done', stem)
