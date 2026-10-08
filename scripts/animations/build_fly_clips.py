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
scene.frame_start = 1
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


def leg_mats(leg, yaw, c, f, t, roll=0.0):
    """Coxa: yaw about +Y, then roll about the leg plane's horizontal axis (tilts the plane), then c."""
    side, z_off, _ = LEGS[leg]
    return (g.trans(side * 1.0, -0.2, z_off) @ g.rot_y(yaw) @ g.rot_z(roll) @ g.rot_x(c),
            g.trans(0, -L0, 0) @ g.rot_x(f), g.trans(0, -L1, 0) @ g.rot_x(t))


def rest_leg(leg):
    return (LEGS[leg][2],) + DISPLAY_POSE[leg[1]] + (0.0,)


# The imported display pose must be exactly the documented one; otherwise stop.
for leg in LEGS:
    for name, m in zip(('coxa', 'femur', 'tibia'), leg_mats(leg, *rest_leg(leg))):
        got = OBJ[f'{leg}_{name}_joint'].matrix_basis
        want = g.matrix_to_blender(m)
        err = max(abs(got[i][j] - want[i][j]) for i in range(4) for j in range(4))
        if err > 1e-4:
            raise SystemExit(f'rig mismatch at {leg}_{name}_joint (max error {err:.2e}); check the joint contract')


def leg_ik(leg, target, alpha0, knee=1, roll=0.0):
    """Planar IK in the leg's vertical plane through the coxa origin (thorax frame).

    Returns (yaw, c, f, t) for the coxa/femur/tibia joints so that the tarsus tip
    (0, -2.4, 0 in the tibia group) reaches ``target``.  alpha0 is the coxa angle from
    straight down toward the target; knee = +1 / -1 picks the elbow solution.
    """
    side, z_off, _ = LEGS[leg]
    rho = -side * roll                      # mirror-symmetric: the same roll tilts both sides alike
    o = Vector((side * 1.0, -0.2, z_off))
    p = Vector(target)
    dx, dz = p.x - o.x, p.z - o.z
    v = p.y - o.y
    r = math.hypot(dx, dz)
    if abs(rho) < 1e-9 or r < 1e-9:
        yaw = math.atan2(dx, dz)
    else:
        # plane normal n = cos(rho) X' + sin(rho) Y with X' = (cos yaw, 0, -sin yaw); require (p - o).n = 0
        phi = math.atan2(dz, dx)
        cval = max(-1.0, min(1.0, -math.tan(rho) * v / r))
        cands = [-phi + math.acos(cval), -phi - math.acos(cval)]
        yaw = max(cands, key=lambda y: dx * math.sin(y) + dz * math.cos(y))
    xp = (math.cos(yaw), 0.0, -math.sin(yaw))
    dvec = (math.sin(yaw), 0.0, math.cos(yaw))
    up = dx * dvec[0] + dz * dvec[2]
    vp = -math.sin(rho) * (dx * xp[0] + dz * xp[2]) + math.cos(rho) * v
    ku, kv = L0 * math.sin(alpha0), -L0 * math.cos(alpha0)
    wu, wv = up - ku, vp - kv
    d = min(L1 + L2 - 1e-3, max(abs(L1 - L2) + 1e-3, math.hypot(wu, wv)))
    aw = math.atan2(wu, -wv)
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))))
    a1 = aw + knee * beta
    nu, nv = ku + L1 * math.sin(a1), kv - L1 * math.cos(a1)
    a2 = math.atan2(up - nu, -(vp - nv))
    return tuple(wrap(x) for x in (yaw, -alpha0, -(a1 - alpha0), -(a2 - a1))) + (rho if roll else 0.0,)


def wrap(x):
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(x), math.cos(x))


def anatomical(ang):
    """Femur-tibia joint flexes the same way as in the display pose (tibia angle > 0, < pi)."""
    return 0.0 < ang[3] < math.pi


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
            yaw, c, _, _, _ = rest_leg(leg)
            a1, t = (1.6, 2.4) if leg[1] == 'f' else (2.0, 2.7)
            return (yaw, c, -(a1 + c), t, 0.0)      # femur raised, tibia folded: foot lifted clear
        _, target, alpha0, knee, roll = spec
        return leg_ik(leg, target, alpha0, knee, roll)

    def _path(self, kind, leg, a, b):
        """One candidate interpolation u in [0, 1] -> leg angles for a segment a -> b."""
        ja, jb = self.leg_angles(leg, a), self.leg_angles(leg, b)

        def jl(p, q, u):
            return tuple(x + wrap(y - x) * u for x, y in zip(p, q))
        if kind == 'target' and isinstance(a, tuple) and isinstance(b, tuple) and a[3] == b[3]:
            return lambda u: leg_ik(leg, tuple(lerp(x, y, u) for x, y in zip(a[1], b[1])),
                                    lerp(a[2], b[2], u), b[3], lerp(a[4], b[4], u))
        if kind == 'joint':
            return lambda u: jl(ja, jb, u)
        if kind in ('tuck', 'hover'):
            mid = self.leg_angles(leg, 'tuck') if kind == 'tuck' else self.leg_angles(leg, rest_t((leg,), HOVER)[leg])
            return lambda u: jl(ja, mid, u * 2) if u < 0.5 else jl(mid, jb, u * 2 - 1)
        return None

    def _score(self, paths, n):
        worst = 0.0
        for k in range(n + 1):
            u = ease(k / n)
            pose = {leg: fn(u) for leg, fn in paths.items()}
            if not all(anatomical(v) for v in pose.values()):
                return math.inf
            apply_pose(pose)
            bpy.context.view_layer.update()
            worst = max(worst, new_penetration()[1], below_floor())
        return worst

    def frames(self):
        """Per-frame poses.  Each segment picks the first clear path among: target-space IK,
        joint-space, via the tucked pose, via the hover pose (dense check of every frame)."""
        out = []
        self.choices = []
        for (f0, l0, d0), (f1, l1, d1) in zip(self.keys, self.keys[1:]):
            moving = [leg for leg in LEGS if l0.get(leg) != l1.get(leg)]
            best = None
            for kind in ('target', 'joint', 'tuck', 'hover'):
                paths = {leg: self._path(kind, leg, l0.get(leg), l1.get(leg)) for leg in moving}
                if any(p is None for p in paths.values()):
                    continue
                sc = self._score(paths, f1 - f0) if moving else 0.0
                if best is None or sc < best[0] - 1e-4:
                    best = (sc, kind, paths)
                if sc <= 0.0:
                    break
            self.choices.append((f0, f1, best[1], round(best[0], 4)))
            static = {leg: self._path('joint', leg, l0.get(leg), l1.get(leg)) for leg in LEGS if leg not in moving}
            for f in range(f0, f1):
                t = ease((f - f0) / (f1 - f0))
                pose = {leg: fn(t) for leg, fn in list(best[2].items()) + list(static.items())}
                for n in set(d0) | set(d1):
                    pose[n] = lerp(d0.get(n, 0.0), d1.get(n, 0.0), t)
                out.append(pose)
        apply_pose({})
        last = self.keys[-1]
        pose = {leg: self.leg_angles(leg, last[1].get(leg)) for leg in LEGS}
        pose.update(last[2])
        out.append(pose)
        if int(ARGS['debug']):
            for c in self.choices:
                if c[3] > 0:
                    print('NEUROFLY_DEBUG segment', c)
        return out


# Only knee = +1 keeps the femur-tibia joint flexing the anatomical way (as in the display pose).
CONFIGS = [(a0, 1, roll) for roll in (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75) for a0 in (-0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8)]


BEST_CACHE = {}
SHIFTS = (0.0, 0.15, 0.3, 0.45, 0.6)


def lateral(p, shift):
    """Move a LEFT-side target outward (more negative x) by shift."""
    return (p[0] - shift, p[1], p[2])


def best_config(pair, targets, dofs=None):
    """Pick (coxa angle, knee, coxa roll) and the smallest outward shift of the whole stroke so that
    the stroke, sampled densely between its key targets, neither pierces the body nor dips below the
    foot plane, with the femur-tibia joint flexing the anatomical way.  Returns (cfg, shift)."""
    key = (pair, tuple(targets), tuple(sorted((dofs or {}).items())))
    if key in BEST_CACHE:
        return BEST_CACHE[key]
    dense = []
    loop = list(targets) + [targets[0]]
    for p0, p1 in zip(loop, loop[1:]):
        for k in range(4):
            dense.append(tuple(lerp(x, y, k / 4) for x, y in zip(p0, p1)))
    best = None
    for shift in SHIFTS:
        for cfg in CONFIGS:
            worst = 0.0
            for t in dense:
                t = lateral(t, shift)
                pose = dict(dofs or {})
                for leg, tgt in ((pair[0], t), (pair[1], mirror(t))):
                    pose[leg] = leg_ik(leg, tgt, *cfg)
                    if not anatomical(pose[leg]):
                        worst = math.inf
                if worst == math.inf:
                    break
                apply_pose(pose)
                bpy.context.view_layer.update()
                worst = max(worst, new_penetration()[1], below_floor())
                if best is not None and worst >= best[0]:
                    break
            if best is None or worst < best[0] - 1e-4:
                best = (worst, cfg, shift)
        if best[0] <= 0.03:
            break
    apply_pose({})
    if int(ARGS['debug']):
        print(f'NEUROFLY_DEBUG best_config {pair} worst={best[0]:.3f} cfg={best[1]} shift={best[2]}')
    BEST_CACHE[key] = (best[1], best[2])
    return BEST_CACHE[key]


def front(target, cfg):
    return {'lf': ('ik', target, *cfg), 'rf': ('ik', mirror(target), *cfg)}


def front_lr(tl, tr, cfg):
    return {'lf': ('ik', tl, *cfg), 'rf': ('ik', mirror(tr), *cfg)}


def hind(target, cfg):
    return {'lh': ('ik', target, *cfg), 'rh': ('ik', mirror(target), *cfg)}


def hind_lr(tl, tr, cfg):
    return {'lh': ('ik', tl, *cfg), 'rh': ('ik', mirror(tr), *cfg)}


# Paths (thorax frame, left side), offset outside the measured head / eye / abdomen shells.
EYE_STROKE = [(-1.12, 1.05, 2.15), (-1.05, 0.55, 2.45), (-0.9, -0.05, 2.5), (-0.75, -0.45, 2.35)]
EYE_RETURN = (-1.35, 1.2, 2.3)
ANT_STROKE = [(-0.45, 1.25, 2.9), (-0.6, 0.95, 3.35), (-0.7, 0.45, 3.6)]
ANT_RETURN = (-0.85, 1.45, 3.1)
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


def rest_t(legs, up=0.0):
    """Display pose (or a point straight above its foot) as an IK target in the display-pose config."""
    out = {}
    for leg in legs:
        tip, a0, knee = REST_IK[leg]
        out[leg] = ('ik', (tip[0], tip[1] + up, tip[2]), a0, knee, 0.0)
    return out


def lift_off(tl, f, legs):
    """Rest -> foot straight up (target space, never below the foot plane) -> tucked."""
    tl.key(f, rest_t(legs))
    f += 6
    tl.key(f, rest_t(legs, HOVER))
    f += 8
    tl.key(f, {leg: 'tuck' for leg in legs})
    return f


def set_down(tl, f, legs):
    """Tucked -> foot straight above its rest point -> lowered onto the foot plane."""
    tl.key(f, {leg: 'tuck' for leg in legs})
    f += 8
    tl.key(f, rest_t(legs, HOVER))
    f += 8
    tl.key(f, rest_t(legs))
    return f


HOVER = 0.6


def head_groom(tl, f):
    """Front legs: eyes x2, rub, antennae x2, rub (Seeds et al. 2014: eyes before antennae)."""
    c_eye, s_eye = best_config(('lf', 'rf'), EYE_STROKE + [EYE_RETURN])
    c_ant, s_ant = best_config(('lf', 'rf'), ANT_STROKE + [ANT_RETURN], ANT_HOLD)
    c_rub, s_rub = best_config(('lf', 'rf'), RUB_FRONT)
    CHOSEN.update(eye=c_eye + (s_eye,), antenna=c_ant + (s_ant,), front_rub=c_rub + (s_rub,))
    eye_s, eye_r = [lateral(p, s_eye) for p in EYE_STROKE], lateral(EYE_RETURN, s_eye)
    ant_s, ant_r = [lateral(p, s_ant) for p in ANT_STROKE], lateral(ANT_RETURN, s_ant)
    rub = [lateral(p, s_rub) for p in RUB_FRONT]
    def via(f, legs, dofs=None):
        f += 14
        tl.key(f, rest_t(legs, HOVER), dofs)            # neutral hover between strokes of different configs
        return f
    f = lift_off(tl, f, ('lf', 'rf'))
    f += 10
    tl.key(f, front(eye_r, c_eye))
    for _ in range(2):
        for i, p in enumerate(eye_s):
            f += 9 if i else 12
            tl.key(f, front(p, c_eye))
        f += 10
        tl.key(f, front(eye_r, c_eye))
    f = via(f, ('lf', 'rf'))
    for k in range(2):                                    # leg rubbing, legs in antiphase
        f += 14 if k == 0 else 8
        tl.key(f, front_lr(rub[k % 2], rub[(k + 1) % 2], c_rub))
    f = via(f, ('lf', 'rf'))
    f += 14
    tl.key(f, front(ant_r, c_ant), ANT_HOLD)
    for _ in range(2):
        for i, p in enumerate(ant_s):
            f += 9 if i else 12
            tl.key(f, front(p, c_ant), ANT_HOLD)
        f += 10
        tl.key(f, front(ant_r, c_ant), ANT_HOLD)
    f = via(f, ('lf', 'rf'), ANT_HOLD)
    for k in range(2):
        f += 14 if k == 0 else 8
        tl.key(f, front_lr(rub[k % 2], rub[(k + 1) % 2], c_rub))
    f = via(f, ('lf', 'rf'))
    f += 10
    return set_down(tl, f, ('lf', 'rf'))


def body_groom(tl, f):
    """Hind legs: abdomen x2, wings x2, rub (Seeds et al. 2014: abdomen before wings)."""
    c_abd, s_abd = best_config(('lh', 'rh'), ABD_STROKE + [ABD_RETURN])
    c_wing, s_wing = best_config(('lh', 'rh'), WING_STROKE + [WING_RETURN], WING_LIFT)
    c_rub, s_rub = best_config(('lh', 'rh'), RUB_HIND)
    CHOSEN.update(abdomen=c_abd + (s_abd,), wing=c_wing + (s_wing,), hind_rub=c_rub + (s_rub,))
    abd_s, abd_r = [lateral(p, s_abd) for p in ABD_STROKE], lateral(ABD_RETURN, s_abd)
    wing_s, wing_r = [lateral(p, s_wing) for p in WING_STROKE], lateral(WING_RETURN, s_wing)
    rub = [lateral(p, s_rub) for p in RUB_HIND]
    def via(f, legs, dofs=None):
        f += 14
        tl.key(f, rest_t(legs, HOVER), dofs)            # neutral hover between strokes of different configs
        return f
    f = lift_off(tl, f, ('lh', 'rh'))
    f += 10
    tl.key(f, hind(abd_r, c_abd))
    for _ in range(2):
        for i, p in enumerate(abd_s):
            f += 9 if i else 12
            tl.key(f, hind(p, c_abd))
        f += 10
        tl.key(f, hind(abd_r, c_abd))
    f = via(f, ('lh', 'rh'))
    f += 14
    tl.key(f, hind(wing_r, c_wing), WING_LIFT)
    for _ in range(2):
        for i, p in enumerate(wing_s):
            f += 9 if i else 12
            tl.key(f, hind(p, c_wing), WING_LIFT)
        f += 10
        tl.key(f, hind(wing_r, c_wing), WING_LIFT)
    f = via(f, ('lh', 'rh'), WING_LIFT)
    f += 8
    tl.key(f, rest_t(('lh', 'rh'), HOVER))
    for k in range(3):
        f += 14 if k == 0 else 8
        tl.key(f, hind_lr(rub[k % 2], rub[(k + 1) % 2], c_rub))
    f = via(f, ('lh', 'rh'))
    f += 10
    return set_down(tl, f, ('lh', 'rh'))


def clip_groom_head():
    tl = Timeline()
    head_groom(tl, 0)
    return tl.frames(), False


def clip_groom_abdomen_wings():
    tl = Timeline()
    body_groom(tl, 0)
    return tl.frames(), False


def clip_groom_full_cycle():
    tl = Timeline()
    f = head_groom(tl, 0)
    f += 6
    body_groom(tl, f)
    return tl.frames(), False


# ------------------------------------------------------------------ batch 2: walking, takeoff prep, idle
def leg_tip(leg, ang):
    m0, m1, m2 = leg_mats(leg, *ang)
    return tuple(m0 @ m1 @ m2 @ Vector((0.0, -L2, 0.0)))


def rest_ik(leg):
    """(tip, alpha0, knee) that reproduce the display pose exactly through leg_ik."""
    ang = rest_leg(leg)
    tip = leg_tip(leg, ang)
    for knee in (1, -1):
        got = leg_ik(leg, tip, -ang[1], knee)
        if max(abs(a - b) for a, b in zip(got, ang)) < 1e-6:
            return tip, -ang[1], knee
    raise SystemExit('display pose not reproducible by IK for ' + leg)


REST_IK = {leg: rest_ik(leg) for leg in LEGS}
STANCE_LIFT = 0.04          # stance tips sit this far above the display-pose tip so no claw dips below it
TRIPOD_A = ('lf', 'rm', 'lh')   # alternating tripods (Mendes et al. 2013: three legs in stance at a time)


def step_tip(leg, phase, stride, lift, duty=0.5):
    """Tarsus tip for one leg at gait phase in [0, 1): stance (on the foot plane, moving back) then swing."""
    tip, _, _ = REST_IK[leg]
    x, y, z = tip
    y += STANCE_LIFT
    if phase < duty:                       # stance: front -> back, linear (constant body speed)
        u = phase / duty
        return (x, y, z + stride / 2 - stride * u)
    u = (phase - duty) / (1 - duty)        # swing: back -> front, raised arc
    e = ease(u)
    return (x, y + lift * math.sin(math.pi * u), z - stride / 2 + stride * e)


def clip_walk_tripod_loop():
    n = 48
    frames = []
    for i in range(n):
        pose = {}
        for leg in LEGS:
            ph = (i / n + (0.0 if leg in TRIPOD_A else 0.5)) % 1.0
            _, a0, knee = REST_IK[leg]
            pose[leg] = leg_ik(leg, step_tip(leg, ph, 1.1, 0.45), a0, knee)
        frames.append(pose)
    return frames, True


def shifted(leg, dz=0.0, dr=0.0, dy=0.0):
    """Rest tip moved dz along the body axis and dr radially outward in the floor plane."""
    tip, a0, knee = REST_IK[leg]
    side = LEGS[leg][0]
    return (tip[0] + side * dr, tip[1] + STANCE_LIFT + dy, tip[2] + dz), a0, knee


def clip_takeoff_prep():
    """Card & Dickinson 2008 order: T1/T3 adjust, T2 reposition, wing raise, pause, T2 extension."""
    keys = []                              # (frame, {leg: (target, a0, knee)}, dofs)

    def legs_at(spec):
        out = {}
        for leg in LEGS:
            if leg in spec:
                out[leg] = shifted(leg, *spec[leg])
            else:
                out[leg] = shifted(leg)
        return out
    t13 = {'lf': (0.25, 0.1), 'rf': (0.25, 0.1), 'lh': (-0.2, 0.1), 'rh': (-0.2, 0.1)}
    t2 = dict(t13, lm=(0.45, 0.0), rm=(0.45, 0.0))
    t2x = dict(t13, lm=(0.45, 0.75), rm=(0.45, 0.75))
    raise_ = {'l_wing_sweep': 0.4, 'r_wing_sweep': 0.4, 'l_wing_elevate': 0.78, 'r_wing_elevate': 0.78}
    keys = [(0, legs_at({}), {}), (10, legs_at({}), {}), (24, legs_at(t13), {}), (30, legs_at(t13), {}),
            (44, legs_at(t2), {}), (62, legs_at(t2), raise_), (74, legs_at(t2), raise_),
            (82, legs_at(t2x), raise_), (96, legs_at(t2x), raise_)]
    lifting = {24: ('lf', 'rf', 'lh', 'rh'), 44: ('lm', 'rm')}   # legs that step (swing arc) into that key
    frames = []
    for (f0, l0, d0), (f1, l1, d1) in zip(keys, keys[1:]):
        for f in range(f0, f1):
            t = ease((f - f0) / (f1 - f0))
            pose = {}
            for leg in LEGS:
                (p0, a0, k), (p1, _, _) = l0[leg], l1[leg]
                tgt = [lerp(a, b, t) for a, b in zip(p0, p1)]
                if leg in lifting.get(f1, ()):
                    tgt[1] += 0.35 * math.sin(math.pi * (f - f0) / (f1 - f0))
                pose[leg] = leg_ik(leg, tuple(tgt), a0, k)
            for nme in set(d0) | set(d1):
                pose[nme] = lerp(d0.get(nme, 0.0), d1.get(nme, 0.0), t)
            frames.append(pose)
    last = keys[-1]
    pose = {leg: leg_ik(leg, last[1][leg][0], last[1][leg][1], last[1][leg][2]) for leg in LEGS}
    pose.update(last[2])
    frames.append(pose)
    return frames, False


def clip_idle_antenna_twitch():
    n = 90
    events = [  # (start, length, {dof: peak})
        (8, 9, {'l_antenna_abduct': 0.22, 'l_funiculus_rotate': 0.05}),
        (30, 8, {'r_antenna_abduct': 0.18, 'r_antenna_extend': 0.08}),
        (52, 10, {'l_antenna_extend': 0.12, 'r_antenna_extend': 0.12, 'l_antenna_twist': 0.05,
                  'r_antenna_twist': 0.05}),
        (70, 7, {'r_antenna_abduct': -0.12, 'l_antenna_abduct': -0.1, 'r_funiculus_rotate': -0.05}),
    ]
    frames = []
    for i in range(n):
        pose = {}
        for f0, ln, peaks in events:
            if f0 <= i <= f0 + ln:
                w = math.sin(math.pi * (i - f0) / ln) ** 2
                for k, v in peaks.items():
                    pose[k] = pose.get(k, 0.0) + v * w
        frames.append(pose)
    return frames, True


CLIPS = [('wing_open_fold', clip_wing_open_fold), ('wingbeat_loop', clip_wingbeat_loop),
         ('antenna_sweep', clip_antenna_sweep), ('groom_head_forelegs', clip_groom_head),
         ('groom_abdomen_wings_hindlegs', clip_groom_abdomen_wings), ('groom_full_cycle', clip_groom_full_cycle),
         ('walk_tripod_loop', clip_walk_tripod_loop), ('takeoff_prep', clip_takeoff_prep),
         ('idle_antenna_twitch', clip_idle_antenna_twitch)]


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
    apply_pose({})                                  # explicit reset of EVERY controlled joint to bind/rest
    bpy.context.view_layer.update()
    frames, loop = fn()
    apply_pose({})
    bpy.context.view_layer.update()
    for o in (OBJ[n] for n in CONTROLLED):
        if o.animation_data:
            o.animation_data.action = None            # keep earlier clips' NLA tracks
    worst = (0, 0.0, '')
    floor_worst = 0.0
    range_issues = []
    # Pass 1, checks: earlier clips' NLA tracks are muted and no action is active, so the depsgraph
    # evaluates exactly the pose we set (a stale NLA evaluation leaked a key in v2).
    for i, pose in enumerate(frames):
        apply_pose(pose)
        bpy.context.view_layer.update()
        c, d, w = new_penetration()                 # every frame (W4 found a dip on an odd frame)
        if int(ARGS['debug']) and d > 0.03:
            print(f'NEUROFLY_DEBUG {clip} frame={i} depth={d:.3f} n={c} mesh={w}')
        if d > worst[1]:
            worst = (c, d, w, i)
        bf = below_floor()
        if int(ARGS['debug']) and bf > 0.01:
            print(f'NEUROFLY_DEBUG {clip} frame={i} below_floor={bf:.3f}')
        floor_worst = max(floor_worst, bf)
        range_issues += check_ranges(pose)
    # Pass 2, keys: set each pose and key it directly, with no depsgraph evaluation in between.
    for i, pose in enumerate(frames):
        apply_pose(pose)
        for n in CONTROLLED:
            OBJ[n].keyframe_insert('rotation_quaternion', frame=i + 1)   # clips start at frame 1
    for n in CONTROLLED:
        o = OBJ[n]
        act = o.animation_data.action
        act.name = f'{clip}__{n}'
        act.use_fake_user = True
        track = o.animation_data.nla_tracks.new()
        track.name = clip
        strip = track.strips.new(clip, 1, act)
        strip.extrapolation = 'NOTHING'
        track.mute = True                            # muted while later clips are authored and checked
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
              export_animation_mode='NLA_TRACKS', export_force_sampling=False, export_frame_step=1,
              export_optimize_animation_size=False)
bpy.ops.export_scene.gltf(**kwargs)


def patch_rest_pose(clip_glb, rig_glb):
    """The NLA export writes animated nodes' rest TRS from the evaluated NLA state, not the bind pose.
    Copy every node's rest translation/rotation/scale from the source rig GLB (same node names), so a
    joint a clip does not key stays exactly at the rig's bind/display pose.  JSON chunk only."""
    import struct as st

    def load(path):
        data = open(path, 'rb').read()
        clen = st.unpack_from('<I', data, 12)[0]
        return data, json.loads(data[20:20 + clen]), clen
    data, doc, clen = load(clip_glb)
    _, rig, _ = load(rig_glb)
    rn = {n.get('name'): n for n in rig['nodes']}
    for n in doc['nodes']:
        r = rn[n['name']]
        for key in ('translation', 'rotation', 'scale'):
            if key in r:
                n[key] = r[key]
            else:
                n.pop(key, None)
    js = json.dumps(doc, separators=(',', ':')).encode('utf-8')
    js += b' ' * ((4 - len(js) % 4) % 4)
    rest = data[20 + clen:]
    total = 12 + 8 + len(js) + len(rest)
    with open(clip_glb, 'wb') as fh:
        fh.write(st.pack('<III', 0x46546C67, 2, total) + st.pack('<II', len(js), 0x4E4F534A) + js + rest)


patch_rest_pose(out_stem + '.glb', ARGS['glb'])
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = True                                  # .blend opens at rest; unmute a track to preview
bpy.ops.wm.save_as_mainfile(filepath=out_stem + '.blend')

meta = {
    'rig': CONTRACT.get('rig'), 'source_glb': os.path.basename(ARGS['glb']), 'fps': FPS,
    'controlled_nodes': CONTROLLED,
    'ik_configs_chosen': {k: {'coxa_angle_rad': v[0], 'knee_side': v[1], 'coxa_roll_rad': v[2], 'outward_shift_viewport_mm': v[3]} for k, v in CHOSEN.items()},
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
                 'groom_abdomen_wings_hindlegs': ['side', 'top'], 'groom_full_cycle': ['side'],
                 'walk_tripod_loop': ['side', 'under'], 'takeoff_prep': ['side', 'front'],
                 'idle_antenna_twitch': ['front', 'top']}
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
