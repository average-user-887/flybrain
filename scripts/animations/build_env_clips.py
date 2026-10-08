# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Illustrative locomotion clips for the spider and mantis rigs (ENV-ANIM-01).

Headless:
    blender --background --factory-startup --threads 2 \
        --python scripts/animations/build_env_clips.py -- \
        --rig RIG_DIR/env_<animal>_rig.glb --params scripts/animations/env_anim_params.json \
        --animal spider|mantis --out OUTDIR [--render 1]

Reads the rig GLB, its ``*_joints.json`` contract and the parameter file (values extracted from
the research table, each labelled measured / derived / proxy / estimate).  Writes into OUTDIR (a new
versioned directory):

* ``<stem>_clips.glb``: every clip as a named glTF animation.  ``*_inplace`` clips keep
  ``<p>_motion`` at identity (feet slide back in stance, like a treadmill); ``*_rootmotion`` clips
  key ``<p>_motion`` so that stance feet stay planted in the world (no sliding statue).  Joint
  curves are identical between the two variants of a clip.
* ``<stem>_clips.blend``, ``<stem>_clips.json`` (clip list, timing, physical speed, playback
  multiplier, checks), sprite sheets rendered from the same posed objects, and a top-view
  heading/scale check.

Gait model: each leg has a phase offset; stance occupies ``duty`` of the cycle.  The body pose is
integrated from the commanded forward speed and yaw rate; a stance foot is planted at the world
position of its neutral (bind-pose) foot point at mid-stance, and a swing foot travels from its
lift-off point to the next planted point on a raised arc.  Legs are solved by planar IK in each
leg's plane (hip yaw + lift, knee, ankle keeping the tarsus at its bind-pose inclination).

PRESENTATION ONLY.  Illustrative animation with source-informed timing, not motion capture; no
predator stimulus, behaviour, contact physics, retinal input or neural response.
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

ARGS = g.parse_args({'out': '', 'rig': '', 'params': '', 'animal': '', 'render': 0, 'res': 320, 'samples': 16,
                     'frames': 8, 'debug': 0})
ANIMAL = ARGS['animal']
PR = 'sp' if ANIMAL == 'spider' else 'mn'
PARAMS = json.load(open(ARGS['params']))[ANIMAL]
FPS = int(PARAMS['fps'])

scene = g.reset_scene()
scene.render.fps = FPS
scene.frame_start = 1
bpy.ops.import_scene.gltf(filepath=ARGS['rig'])
OBJ = bpy.data.objects
stem = os.path.splitext(os.path.basename(ARGS['rig']))[0]
CONTRACT = json.load(open(os.path.splitext(ARGS['rig'])[0] + '_joints.json'))
AXES = {j['node']: Vector(j['axis']) for j in CONTRACT['joints'] if 'axis' in j}
RANGES = {j['node']: tuple(j['range_rad']) for j in CONTRACT['joints'] if 'axis' in j}
LEGS = CONTRACT['legs']
MOTION = OBJ[f'{PR}_motion']
SHIFT = OBJ[f'{PR}_body_shift']
DOFS = sorted(AXES)
CONTROLLED = DOFS + [MOTION.name, SHIFT.name]
for n in CONTROLLED:
    OBJ[n].rotation_mode = 'QUATERNION'
root = [o for o in OBJ if o.parent is None and o.type == 'EMPTY'][0]


def rot_axis(a, axis):
    return Matrix.Rotation(a, 4, Vector(axis))


# ------------------------------------------------------------------ body chain (three-frame matrices)
BODY = {'spider': 'sp_body', 'mantis': 'mn_body'}[ANIMAL]
BODY_PIVOT = Vector(next(j['translation'] for j in CONTRACT['joints'] if j['node'] == BODY + '_root'))


def body_matrix(pose):
    m = Matrix.Identity(4)
    for dof in ('pitch', 'roll', 'yaw'):
        n = f'{BODY}_{dof}'
        m = m @ rot_axis(pose.get(n, 0.0), AXES[n])
    sx, sy = pose.get('_shift', (0.0, 0.0))
    return g.trans(sx, sy, 0.0) @ g.trans(*BODY_PIVOT) @ m @ g.trans(*(-BODY_PIVOT))


# ------------------------------------------------------------------ leg IK
def plane_coords(leg):
    """Bind-pose leg in its hinge frame: u (horizontal, along the leg), up (+Y), n = u x up.
    Points need not be coplanar: the n-components are invariant under the hinge rotations."""
    L = LEGS[leg]
    A, K, B, F = (Vector(L[k]) for k in ('A', 'K', 'B', 'F'))
    u = Vector(L['out']).normalized()
    n = Vector(L['plane_normal']).normalized()

    def pc(p):
        d = p - A
        return Vector((d.dot(u), d.y)), d.dot(n)
    (k, _), (b, _), (f, cf) = pc(K), pc(B), pc(F)
    return {'A': A, 'u0': u, 'L1': k.length, 'L2': (b - k).length, 'a1': math.atan2(k.y, k.x),
            'a2': math.atan2(b.y - k.y, b.x - k.x), 'tar': f - b, 'c_tip': cf}


PLANE = {leg: plane_coords(leg) for leg in LEGS}


def leg_ik(leg, target_body):
    """Target tarsus tip (leg's body-moved frame) -> ({dof: angle}, reach shortfall)."""
    P = PLANE[leg]
    side = LEGS[leg]['side']
    dx, dz = target_body.x - P['A'].x, target_body.z - P['A'].z
    r = math.hypot(dx, dz)
    c = P['c_tip']
    # n = u x Y = (-uz, ux) in (x, z), so d.n = r sin(delta - alpha); keep it equal to the bind-pose offset c.
    delta = math.atan2(dz, dx)
    alpha = delta - math.asin(max(-1.0, min(1.0, c / max(r, 1e-6))))
    alpha0 = math.atan2(P['u0'].z, P['u0'].x)
    phi = alpha0 - alpha                       # rotation about +Y taking u0 to u
    phi = math.atan2(math.sin(phi), math.cos(phi))
    tu = math.sqrt(max(0.0, r * r - c * c))
    tv = target_body.y - P['A'].y
    bu, bv = tu - P['tar'].x, tv - P['tar'].y
    D = math.hypot(bu, bv)
    L1, L2 = P['L1'], P['L2']
    Dc = min(L1 + L2 - 1e-4, max(abs(L1 - L2) + 1e-4, D))
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + Dc * Dc - L2 * L2) / (2 * L1 * Dc))))
    a1 = math.atan2(bv, bu) + beta
    ku, kv = L1 * math.cos(a1), L1 * math.sin(a1)
    a2 = math.atan2(bv - kv, bu - ku)
    lift = math.atan2(math.sin(a1 - P['a1']), math.cos(a1 - P['a1']))
    knee = (a2 - a1) - (P['a2'] - P['a1'])
    knee = math.atan2(math.sin(knee), math.cos(knee))
    ankle = -(lift + knee)
    yaw = phi if side < 0 else -phi
    return ({f'{PR}_{leg}_hip_yaw': yaw, f'{PR}_{leg}_hip_lift': lift, f'{PR}_{leg}_knee_flex': knee,
             f'{PR}_{leg}_ankle_flex': ankle}, D - Dc)


# ------------------------------------------------------------------ gait
def yaw_m(theta):
    return Matrix.Rotation(theta, 4, 'Y')


class Body2D:
    """Body pose in the floor plane integrated from speed v(t) (along heading) and yaw rate w(t).
    Integration starts before t = 0 (with v(0), w(0)) so stance feet planted before the clip starts are
    where a steadily moving animal would have put them; the pose is then re-based so pose(0) = identity."""

    def __init__(self, v, w, dur, dt, pre):
        self.dt = dt
        self.t0 = -pre
        xs, zs, hs = [0.0], [0.0], [0.0]
        n = int(round((dur + pre) / dt)) + 2
        for i in range(n):
            t = self.t0 + i * dt
            tm = max(0.0, t + dt / 2)
            h = hs[-1] + w(tm) * dt
            hm = (hs[-1] + h) / 2
            xs.append(xs[-1] + v(tm) * math.sin(hm) * dt)
            zs.append(zs[-1] + v(tm) * math.cos(hm) * dt)
            hs.append(h)
        self.x, self.z, self.h = xs, zs, hs
        x0, z0, h0 = self._raw(0.0)
        c, s_ = math.cos(-h0), math.sin(-h0)
        # re-base: rotate by -h0 about the pose at t = 0 and translate it to the origin
        self.x = [c * (x - x0) + s_ * (z - z0) for x, z in zip(xs, zs)]
        self.z = [-s_ * (x - x0) + c * (z - z0) for x, z in zip(xs, zs)]
        self.h = [h - h0 for h in hs]

    def _raw(self, t):
        f = (t - self.t0) / self.dt
        i = max(0, min(len(self.x) - 2, int(f)))
        u = min(1.0, max(0.0, f - i))
        return tuple(a[i] + (a[i + 1] - a[i]) * u for a in (self.x, self.z, self.h))

    def at(self, t):
        return self._raw(t)

    def matrix(self, t):
        x, z, h = self.at(t)
        return g.trans(x, 0.0, z) @ yaw_m(h)


def neutral(leg):
    f = Vector(LEGS[leg]['F'])
    return Vector((f.x, f.y + PARAMS.get('stance_clearance', 0.0), f.z))


def gait_frames(dur, v, w, cycle, duty, offsets, lift_h, gait_on=lambda t: True, extra=None):
    """Return per-frame poses (dict: DOFs + '_motion': (x, z, heading)) and per-leg contact flags."""
    dt = 1.0 / FPS
    body = Body2D(v, w, dur + cycle, dt / 4, 2 * cycle)
    n = int(round(dur * FPS))
    frames, contacts = [], []
    for i in range(n):
        t = i / FPS
        M = body.matrix(t)
        Minv = M.inverted()
        pose = {}
        cflags = {}
        for leg, off in offsets.items():
            ph = (t / cycle + off) % 1.0
            on = gait_on(t)
            if not on or ph < duty:
                # stance: planted where the neutral foot was at mid-stance
                if on:
                    t_mid = t - (ph - duty / 2) * cycle
                else:
                    t_mid = t
                W = body.matrix(t_mid) @ neutral(leg)
                tgt = Minv @ W
                cflags[leg] = True
            else:
                s = (ph - duty) / (1 - duty)
                t_lift = t - (ph - duty) * cycle
                t_prev_mid = t_lift - duty / 2 * cycle
                t_next_mid = t_lift + (1 - duty) * cycle + duty / 2 * cycle
                W0 = body.matrix(t_prev_mid) @ neutral(leg)
                W1 = body.matrix(t_next_mid) @ neutral(leg)
                e = 0.5 - 0.5 * math.cos(math.pi * s)
                W = W0.lerp(W1, e)
                W.y += (lift_h(t) if callable(lift_h) else lift_h) * math.sin(math.pi * s)
                tgt = Minv @ W
                cflags[leg] = False
            bm = body_matrix(extra(t) if extra else {})
            ang, err = leg_ik(leg, bm.inverted() @ tgt)
            pose.update(ang)
            pose.setdefault('_tgt', {})[leg] = tuple(tgt)
        if extra:
            pose.update(extra(t))
        x, z, h = body.at(t)
        pose['_motion'] = (x, z, h)
        frames.append(pose)
        contacts.append(cflags)
    return frames, contacts, body


# ------------------------------------------------------------------ clips (parameters from PARAMS)
G = PARAMS['gait']
CYCLE, DUTY = G['cycle_s'], G['duty']
OFFS = G['phase_offsets']
SPEED = PARAMS['speed_viewport_mm_s']
LIFT = G['swing_height_viewport_mm']
TURN_W = PARAMS['turn']['yaw_rate_rad_s']
TURN_V = PARAMS['turn']['speed_fraction'] * SPEED
NCYC = int(PARAMS.get('loop_cycles', 2))


def idle_extra(t):
    out = {}
    if 'peer_amp_viewport_mm' in PARAMS['idle']:      # peering: body translates sideways, head keeps facing ahead
        out['_shift'] = (PARAMS['idle']['peer_amp_viewport_mm'] * math.sin(2 * math.pi * t / PARAMS['idle']['peer_period_s']), 0.0)
    for item in PARAMS['idle']['dofs']:
        node, amp, period, phase = item['node'], item['amp_rad'], item['period_s'], item.get('phase', 0.0)
        shape = item.get('shape', 'sine')
        if shape == 'sine':
            out[node] = out.get(node, 0.0) + amp * math.sin(2 * math.pi * (t / period + phase))
        else:                                   # brief twitch once per period: sin^2 bump of width w
            w = item.get('width_s', 0.2)
            tt = (t + phase * period) % period
            out[node] = out.get(node, 0.0) + (amp * math.sin(math.pi * tt / w) ** 2 if tt < w else 0.0)
    return out


def clip_idle():
    dur = PARAMS['idle']['duration_s']
    frames, contacts, _ = gait_frames(dur, lambda t: 0.0, lambda t: 0.0, CYCLE, DUTY, OFFS, LIFT,
                                      gait_on=lambda t: False, extra=idle_extra)
    return frames, contacts, True


def clip_walk():
    dur = CYCLE * NCYC
    return (*gait_frames(dur, lambda t: SPEED, lambda t: 0.0, CYCLE, DUTY, OFFS, LIFT)[:2], True)


def clip_start_stop():
    ramp = PARAMS['start_stop']['ramp_s']
    hold = PARAMS['start_stop']['cruise_s']
    dur = PARAMS['start_stop']['rest_s'] * 2 + ramp * 2 + hold
    r0 = PARAMS['start_stop']['rest_s']

    def v(t):
        if t < r0:
            return 0.0
        if t < r0 + ramp:
            return SPEED * (0.5 - 0.5 * math.cos(math.pi * (t - r0) / ramp))
        if t < r0 + ramp + hold:
            return SPEED
        if t < r0 + 2 * ramp + hold:
            return SPEED * (0.5 + 0.5 * math.cos(math.pi * (t - r0 - ramp - hold) / ramp))
        return 0.0
    # Stepping continues but stride and swing height scale with speed, so at rest the feet stay put.
    return (*gait_frames(dur, v, lambda t: 0.0, CYCLE, DUTY, OFFS, lambda t: LIFT * v(t) / SPEED)[:2], False)


def clip_turn(sign):
    T = PARAMS['turn']
    if ANIMAL == 'spider':                          # continuous on-the-spot turning (Land 1972), loops
        dur = CYCLE * NCYC
        return (*gait_frames(dur, lambda t: TURN_V, lambda t: sign * TURN_W, CYCLE, DUTY, OFFS, LIFT)[:2], True)
    # mantis: one orienting turn; head starts first (<= 40 ms lead, Yamawaki 2011), body follows by stepping
    lead, dur = T['head_lead_s'], CYCLE * 4

    def win(t):
        if t < lead:
            return 0.0
        if t < lead + 0.15:
            return 0.5 - 0.5 * math.cos(math.pi * (t - lead) / 0.15)
        if t < dur - 0.6:
            return 1.0
        if t < dur - 0.3:
            return 0.5 + 0.5 * math.cos(math.pi * (t - dur + 0.6) / 0.3)
        return 0.0

    def head(t):
        if t < 0.6:
            return {'mn_neck_yaw': sign * T['head_peak_rad'] * math.sin(math.pi * t / 1.2) ** 2 * (1 if t < 0.6 else 0)}
        return {'mn_neck_yaw': sign * T['head_peak_rad'] * max(0.0, 1 - (t - 0.6) / 0.8) ** 2}
    return (*gait_frames(dur, lambda t: 0.0, lambda t: sign * TURN_W * win(t), CYCLE, DUTY, OFFS,
                         lambda t: LIFT * win(t), extra=head)[:2], False)


def clip_scene_walk():
    """Composed short walk for the walking scene: start, walk, stop, turn left on the spot, walk, stop.
    Root motion is integrated from the same speed/yaw-rate schedule that places the feet."""
    sc = PARAMS['scene_walk']
    segs = sc['segments']                  # [kind, seconds]: rest | walk | turn_left | turn_right
    times, t = [], 0.0
    for kind, secs in segs:
        times.append((t, t + secs, kind))
        t += secs
    dur = t
    ramp = sc['ramp_s']

    def env(t, kinds):
        level = 0.0
        for a, b, k in times:
            if k in kinds and a - ramp < t < b + ramp:
                up = min(1.0, max(0.0, (t - a) / ramp)) if t < a + ramp else 1.0
                dn = min(1.0, max(0.0, (b - t) / ramp)) if t > b - ramp else 1.0
                level = max(level, (0.5 - 0.5 * math.cos(math.pi * min(up, dn))))
        return level
    v = lambda t: SPEED * env(t, ('walk',))                                               # noqa: E731
    w = lambda t: TURN_W * (env(t, ('turn_left',)) - env(t, ('turn_right',)))           # noqa: E731
    act = lambda t: max(env(t, ('walk',)), env(t, ('turn_left', 'turn_right')))          # noqa: E731
    return (*gait_frames(dur, v, w, CYCLE, DUTY, OFFS, lambda t: LIFT * act(t))[:2], False)


CLIPS = [('idle', clip_idle), ('walk_forward', clip_walk), ('start_stop', clip_start_stop),
         ('turn_left', lambda: clip_turn(+1)), ('turn_right', lambda: clip_turn(-1)),
         ('scene_walk', clip_scene_walk)]


# ------------------------------------------------------------------ apply, checks
def apply(pose, motion=True):
    for n in DOFS:
        OBJ[n].matrix_basis = g.matrix_to_blender(rot_axis(pose.get(n, 0.0), AXES[n]))
    sx, sy = pose.get('_shift', (0.0, 0.0))
    SHIFT.matrix_basis = g.matrix_to_blender(g.trans(sx, sy, 0.0))
    x, z, h = pose.get('_motion', (0.0, 0.0, 0.0)) if motion else (0.0, 0.0, 0.0)
    MOTION.matrix_basis = g.matrix_to_blender(g.trans(x, 0.0, z) @ yaw_m(h))


apply({})
bpy.context.view_layer.update()
MESHES = [o for o in OBJ if o.type == 'MESH']
LEG_MESH = [o for o in MESHES if any(o.name.startswith(f'{PR}_{leg}_') for leg in LEGS)]
BODY_MESH = [o for o in MESHES if o not in LEG_MESH and not o.name.startswith(('sp_palp', 'spider_palp',
                                                                                 'mn_L1', 'mn_R1'))]
FOOT_PLANE = min((o.matrix_world @ v.co).z for o in LEG_MESH for v in o.data.vertices)   # bind-pose lowest leg vertex


def bvh(objs):
    verts, polys = [], []
    for o in objs:
        base = len(verts)
        verts += [o.matrix_world @ v.co for v in o.data.vertices]
        polys += [[base + i for i in p.vertices] for p in o.data.polygons]
    return BVHTree.FromPolygons(verts, polys)


def inside_depth(tree, p):
    votes = 0
    for d in (Vector((1, 0, 0)), Vector((0, 0.6, 0.8)), Vector((-0.6, -0.8, 0.0))):
        n, q = 0, p.copy()
        for _ in range(64):
            hit = tree.ray_cast(q, d)
            if hit[0] is None:
                break
            n += 1
            q = hit[0] + d * 1e-5
        votes += n % 2
    if votes >= 2:
        near = tree.find_nearest(p)
        return near[3] if near[0] is not None else 0.0
    return 0.0


REST_INSIDE = None


def body_pen():
    """Leg vertices newly inside the body (vs bind pose), every leg mesh; hip region excluded (socket)."""
    tree = bvh(BODY_MESH)
    worst = 0.0
    for o in LEG_MESH:
        mw = o.matrix_world
        hip = None
        if o.name.endswith('_femur'):
            leg = o.name[len(PR) + 1:].split('_')[0]
            hip = mw @ Vector(g.to_blender(tuple(LEGS[leg]['A'])))
        for k, v in enumerate(o.data.vertices):
            p = mw @ v.co
            if hip is not None and (p - hip).length < 0.6:
                continue
            if REST_INSIDE is not None and REST_INSIDE.get((o.name, k)):
                continue
            d = inside_depth(tree, p)
            if d > worst:
                worst = d
    return worst


def rest_inside_map():
    tree = bvh(BODY_MESH)
    out = {}
    for o in LEG_MESH:
        for k, v in enumerate(o.data.vertices):
            if inside_depth(tree, o.matrix_world @ v.co) > 0:
                out[(o.name, k)] = True
    return out


REST_INSIDE = rest_inside_map()


def lowest_leg():
    return min((o.matrix_world @ v.co).z for o in LEG_MESH for v in o.data.vertices)


def foot_tip_world(leg):
    o = OBJ[f'{PR}_{leg}_tarsus']
    return o.matrix_world @ Vector(g.to_blender(tuple(LEGS[leg]['F'])))


def check_ranges(pose):
    bad = []
    for n in DOFS:
        lo, hi = RANGES[n]
        a = pose.get(n, 0.0)
        if a < lo - 1e-9 or a > hi + 1e-9:
            bad.append(n)
    return bad


# ------------------------------------------------------------------ bake
report, built = {}, []
for clip, fn in CLIPS:
    apply({})
    bpy.context.view_layer.update()
    frames, contacts, loop = fn()
    for variant in ('inplace', 'rootmotion'):
        name = f'{clip}_{variant}'
        if clip == 'idle' and variant == 'rootmotion':
            continue
        if clip == 'scene_walk' and variant == 'inplace':
            continue
        apply({})
        for o in (OBJ[n] for n in CONTROLLED):
            if o.animation_data:
                o.animation_data.action = None
        pen, low, slide, out_of_range, reach = 0.0, math.inf, 0.0, set(), 0.0
        mot_inv = None
        prev_tip = {}
        for i, pose in enumerate(frames):
            apply(pose, motion=(variant == 'rootmotion'))
            bpy.context.view_layer.update()
            if i % int(PARAMS.get('check_every', 1)) == 0:
                pen = max(pen, body_pen())
            low = min(low, lowest_leg())
            out_of_range |= set(check_ranges(pose))
            minv = MOTION.matrix_world.inverted()
            for leg, tg in pose.get('_tgt', {}).items():
                tip = minv @ foot_tip_world(leg)            # Blender frame of the motion node
                tip3 = Vector((tip.x, tip.z, -tip.y))
                reach = max(reach, (tip3 - Vector(tg)).length)
            if variant == 'rootmotion':
                for leg in LEGS:
                    tip = foot_tip_world(leg)
                    if contacts[i].get(leg) and contacts[i - 1].get(leg) if i else False:
                        dxy = math.hypot(tip.x - prev_tip[leg].x, tip.y - prev_tip[leg].y)
                        slide = max(slide, dxy)
                    prev_tip[leg] = tip
        for i, pose in enumerate(frames):
            apply(pose, motion=(variant == 'rootmotion'))
            for n in DOFS:
                OBJ[n].keyframe_insert('rotation_quaternion', frame=i + 1)
            MOTION.keyframe_insert('location', frame=i + 1)
            MOTION.keyframe_insert('rotation_quaternion', frame=i + 1)
            SHIFT.keyframe_insert('location', frame=i + 1)
        for n in CONTROLLED:
            o = OBJ[n]
            act = o.animation_data.action
            act.name = f'{name}__{n}'
            act.use_fake_user = True
            tr = o.animation_data.nla_tracks.new()
            tr.name = name
            tr.strips.new(name, 1, act).extrapolation = 'NOTHING'
            tr.mute = True
            o.animation_data.action = None
        x, z, h = frames[-1]['_motion']
        stance = {}
        for leg in LEGS:
            runs, start = [], None
            for i, c in enumerate(contacts):
                if c.get(leg) and start is None:
                    start = i
                if not c.get(leg) and start is not None:
                    runs.append([start, i - 1])
                    start = None
            if start is not None:
                runs.append([start, len(contacts) - 1])
            stance[leg] = runs
        report[name] = {'stance_key_intervals': stance,'frames': len(frames), 'fps': FPS, 'seconds': round(len(frames) / FPS, 4), 'loop': loop,
                        'root_motion': variant == 'rootmotion',
                        'displacement_per_clip_viewport_mm': [round(x, 3), round(z, 3)] if variant == 'rootmotion'
                        else [0.0, 0.0],
                        'heading_change_rad': round(h, 4) if variant == 'rootmotion' else 0.0,
                        'max_leg_body_penetration_viewport_mm': round(pen, 4),
                        'lowest_leg_vertex_minus_foot_plane': round(low - FOOT_PLANE, 4),
                        'max_stance_slide_per_frame_viewport_mm': round(slide, 4) if variant == 'rootmotion' else None,
                        'dofs_outside_display_limits': sorted(out_of_range)}
        built.append((name, frames, variant))
        report[name]['max_foot_target_error_viewport_mm'] = round(reach, 4)
        print(f'NEUROFLY_ENVCLIP {ANIMAL} {name} frames={len(frames)} pen={pen:.3f} reach_err={reach:.3f} '
              f'low={low - FOOT_PLANE:.4f} slide={slide:.4f} out={len(out_of_range)}')

apply({})
bpy.context.view_layer.update()
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = False
out_stem = os.path.join(ARGS['out'], stem + '_clips')
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.export_scene.gltf(filepath=out_stem + '.glb', export_format='GLB', export_yup=True, export_apply=True,
                          export_extras=True, export_cameras=False, export_lights=False, export_animations=True,
                          export_animation_mode='NLA_TRACKS', export_force_sampling=False,
                          export_optimize_animation_size=False)


def patch_rest_pose(clip_glb, rig_glb):
    """Copy every node's rest TRS from the rig GLB (the NLA export writes evaluated, not bind, values)."""
    import struct as st

    def load(path):
        data = open(path, 'rb').read()
        clen = st.unpack_from('<I', data, 12)[0]
        return data, json.loads(data[20:20 + clen]), clen
    data, doc, clen = load(clip_glb)
    _, rig, _ = load(rig_glb)
    rn = {n.get('name'): n for n in rig['nodes']}
    for n in doc['nodes']:
        r = rn.get(n['name'])
        if r is None:
            continue                                   # scene-only nodes (ground)
        for key in ('translation', 'rotation', 'scale'):
            if key in r:
                n[key] = r[key]
            else:
                n.pop(key, None)
    js = json.dumps(doc, separators=(',', ':')).encode('utf-8')
    js += b' ' * ((4 - len(js) % 4) % 4)
    rest = data[20 + clen:]
    with open(clip_glb, 'wb') as fh:
        fh.write(st.pack('<III', 0x46546C67, 2, 20 + len(js) + len(rest)) + st.pack('<II', len(js), 0x4E4F534A)
                 + js + rest)


patch_rest_pose(out_stem + '.glb', ARGS['rig'])
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = True
bpy.ops.wm.save_as_mainfile(filepath=out_stem + '.blend')

# ------------------------------------------------------------------ composed walking scene (ground + all clips)
scene_frames = dict((n, f) for n, f, v in built)['scene_walk_rootmotion']
path = [f['_motion'] for f in scene_frames]
cx = (min(p[0] for p in path) + max(p[0] for p in path)) / 2
cz = (min(p[1] for p in path) + max(p[1] for p in path)) / 2
radius = max(max(abs(p[0] - cx), abs(p[1] - cz)) for p in path) + PARAMS['sprite_extent_viewport_mm'] * 0.8
gg = g.Geo()
rings, seg = 12, 48
gg.verts = [(cx, -0.03, cz)] + [(cx + radius * i / rings * math.cos(2 * math.pi * k / seg), -0.03,
                                 cz + radius * i / rings * math.sin(2 * math.pi * k / seg))
                                for i in range(1, rings + 1) for k in range(seg)]
gg.faces = [(0, 1 + (k + 1) % seg, 1 + k) for k in range(seg)] + [
    (1 + i * seg + k, 1 + i * seg + (k + 1) % seg, 1 + (i + 1) * seg + (k + 1) % seg, 1 + (i + 1) * seg + k)
    for i in range(rings - 1) for k in range(seg)]
gg.mats = [0] * len(gg.faces)
ground = g.mesh_object('scene_ground', gg, [g.material('nf_scene_soil', (0.24, 0.16, 0.09), roughness=0.95)],
                       smooth=False, recalc=False)
ground['neurofly_note'] = 'walking-scene floor at y = -0.03 (just below the foot plane); not an arena or stimulus'
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = False
bpy.ops.object.select_all(action='DESELECT')
walk_scene = os.path.join(ARGS['out'], stem.replace('_rig', '') + '_walk_scene')
bpy.ops.export_scene.gltf(filepath=walk_scene + '.glb', export_format='GLB', export_yup=True, export_apply=True,
                          export_extras=True, export_cameras=False, export_lights=False, export_animations=True,
                          export_animation_mode='NLA_TRACKS', export_force_sampling=False,
                          export_optimize_animation_size=False)
patch_rest_pose(walk_scene + '.glb', ARGS['rig'])
for o in (OBJ[n] for n in CONTROLLED):
    for t in o.animation_data.nla_tracks:
        t.mute = True
apply({})
bpy.ops.wm.save_as_mainfile(filepath=walk_scene + '.blend')

meta = {'animal': ANIMAL, 'rig': CONTRACT['rig'], 'source_rig_glb': os.path.basename(ARGS['rig']), 'fps': FPS,
        'label': 'Illustrative animation with source-informed timing; not motion capture, not simulated behaviour.',
        'playback_multiplier_default': PARAMS['playback_multiplier_default'],
        'playback_note': 'clip times are physical seconds at the estimated speed; multiply playback rate by the '
                         'multiplier (default 1 = estimated real time)',
        'estimated_physical_speed_mm_s': PARAMS['speed_mm_s'], 'estimated_physical_speed_bl_s': PARAMS['speed_bl_s'],
        'viewport_mm_per_mm': PARAMS['viewport_mm_per_mm'], 'speed_viewport_mm_s': SPEED,
        'gait': G, 'turn': PARAMS['turn'], 'foot_plane_y': None, 'heading': '+Z forward (head), +Y up',
        'parameters': PARAMS, 'gait_label': PARAMS['labels']['gait'],
        'clips': report}

# ------------------------------------------------------------------ sprites (same posed objects)
if int(ARGS['render']):
    res = int(ARGS['res'])
    g.setup_render(scene, (res, res), samples=int(ARGS['samples']))
    g.studio_lights(scale=30.0)
    ext = PARAMS['sprite_extent_viewport_mm']
    cams = {'side': g.camera('cam_side', (60, ext * 0.25, 0), (0, ext * 0.25, 0), ortho_scale=ext),
            'top': g.camera('cam_top', (0, 80, 0), (0, 0, 0), ortho_scale=ext)}
    tmp = os.path.join(ARGS['out'], '_tmp_' + ANIMAL)
    os.makedirs(tmp, exist_ok=True)
    sheets = {}
    ground.hide_render = True                          # sprites are transparent, animal only
    for name, frames, variant in built:
        if variant != 'inplace':
            continue
        n = len(frames)
        nf = int(ARGS['frames'])
        picks = [round(k * n / nf) % n for k in range(nf)]
        rows = []
        for view in ('side', 'top'):
            tiles = []
            for i in picks:
                apply(frames[i], motion=False)
                bpy.context.view_layer.update()
                p = os.path.join(tmp, f'{view}_{i}.png')
                g.render_to(scene, cams[view], p)
                img = bpy.data.images.load(p)
                tiles.append(np.array(img.pixels[:], dtype=np.float32).reshape(res, res, 4))
                bpy.data.images.remove(img)
                os.remove(p)
            rows.append(np.concatenate(tiles, axis=1))
        full = np.concatenate(rows[::-1], axis=0)
        h, w = full.shape[:2]
        img = bpy.data.images.new(name, w, h, alpha=True)
        img.pixels = full.ravel()
        fname = f'{stem}_{name}_sprites.png'
        img.filepath_raw = os.path.join(ARGS['out'], fname)
        img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)
        sheets[name] = {'file': fname, 'frames': picks, 'views': ['side (looking -X, head left)',
                                                                 'top (looking down, head up)'],
                        'tile_px': res, 'ortho_scale_viewport_mm': ext, 'px_per_viewport_mm': round(res / ext, 3)}
    # walking-scene preview: fixed top camera over the ground, the animal translating (root motion)
    n = len(scene_frames)
    picks = [round(k * (n - 1) / 7) for k in range(8)]
    tiles = []
    span = radius * 2
    scam = g.camera('cam_scene_top', (cx, 120, cz), (cx, 0, cz), ortho_scale=span)
    ground.hide_render = False
    for i in picks:
        apply(scene_frames[i], motion=True)
        bpy.context.view_layer.update()
        p = os.path.join(tmp, f'scene_{i}.png')
        g.render_to(scene, scam, p)
        img = bpy.data.images.load(p)
        tiles.append(np.array(img.pixels[:], dtype=np.float32).reshape(res, res, 4))
        bpy.data.images.remove(img)
        os.remove(p)
    full = np.concatenate([np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)][::-1], axis=0)
    img = bpy.data.images.new('scene', full.shape[1], full.shape[0], alpha=True)
    img.pixels = full.ravel()
    fname = os.path.basename(walk_scene) + '_top_frames.png'
    img.filepath_raw = os.path.join(ARGS['out'], fname)
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    sheets['scene_walk_rootmotion'] = {'file': fname, 'frames': picks, 'views': ['top, fixed camera over the ground'],
                                       'tile_px': res, 'ortho_scale_viewport_mm': round(span, 3)}
    os.rmdir(tmp)
    meta['sprite_sheets'] = sheets
    apply({})
with open(out_stem + '.json', 'w') as fh:
    json.dump(meta, fh, indent=2)
print('NEUROFLY_ENVCLIP done', ANIMAL)
