# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Shared helpers for the NeuroFly presentation-asset scripts (Blender 4.5, bpy).

All geometry is computed in the *three.js viewport frame* (Y up, +Z = fly
forward, units = viewport millimetres) and converted to Blender's Z-up frame
only when a mesh or empty is created.  The glTF exporter's +Y-up conversion
then maps every node and vertex back to exactly the three.js frame, so the
numbers in these scripts are the numbers the dashboard sees.

Blender (x, y, z)  ->  glTF / three.js (x, z, -y)
three.js (X, Y, Z) ->  Blender (X, -Z, Y)          (``to_blender``)

Nothing here imports a third-party mesh: every vertex is generated.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

# Basis change: three = C @ blender.  C is a proper rotation (det +1).
C = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
CT = C.transposed()


def to_blender(v):
    x, y, z = v
    return (x, -z, y)


def matrix_to_blender(m4):
    """Express a three.js-frame 4x4 transform in Blender's frame (CT M C)."""
    c4 = C.to_4x4()
    return c4.transposed() @ m4 @ c4


def rot_x(a):
    return Matrix.Rotation(a, 4, 'X')


def rot_y(a):
    return Matrix.Rotation(a, 4, 'Y')


def rot_z(a):
    return Matrix.Rotation(a, 4, 'Z')


def trans(x, y, z):
    return Matrix.Translation((x, y, z))


def scale3(sx, sy, sz):
    return Matrix.Diagonal((sx, sy, sz, 1.0))


def parse_args(defaults):
    """Parse ``-- --out DIR --lod N`` style arguments after Blender's own."""
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    out = dict(defaults)
    i = 0
    while i < len(argv):
        key = argv[i].lstrip('-').replace('-', '_')
        if key in out and i + 1 < len(argv):
            out[key] = type(defaults[key])(argv[i + 1])
            i += 2
        else:
            raise SystemExit('unknown or incomplete argument: ' + argv[i])
    if not out.get('out'):
        raise SystemExit('pass -- --out DIR (an output directory outside the repository)')
    os.makedirs(out['out'], exist_ok=True)
    return out


def reset_scene():
    """Start from an empty scene.  Only ever used by these batch scripts."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    return scene


# ------------------------------------------------------------------ geometry
class Geo:
    """A vertex/face soup in the three.js frame with per-face material slots."""

    def __init__(self):
        self.verts = []
        self.faces = []
        self.mats = []

    def extend(self, verts, faces, mat=0, matrix=None):
        base = len(self.verts)
        for v in verts:
            p = Vector(v)
            if matrix is not None:
                p = matrix @ p
            self.verts.append(tuple(p))
        for f in faces:
            self.faces.append(tuple(base + i for i in f))
            self.mats.append(mat if not callable(mat) else mat(f, verts))
        return self


def revolve(profile, seg, matrix=None, sx=1.0, sz=1.0, mat=0, geo=None):
    """Lathe a profile [(y, r), ...] around the Y axis.  r == 0 makes a pole.

    ``mat`` may be a callable ``mat(ring_index) -> slot`` for banded colours.
    """
    geo = geo or Geo()
    verts, faces, ring_ids, face_mats = [], [], [], []
    for (y, r) in profile:
        if r <= 1e-9:
            ring_ids.append([len(verts)])
            verts.append((0.0, y, 0.0))
            continue
        ids = []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            ids.append(len(verts))
            verts.append((r * math.cos(a) * sx, y, r * math.sin(a) * sz))
        ring_ids.append(ids)
    for j in range(len(ring_ids) - 1):
        a, b = ring_ids[j], ring_ids[j + 1]
        slot = mat(j) if callable(mat) else mat
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            for k in range(seg):
                faces.append((a[0], b[(k + 1) % seg], b[k])); face_mats.append(slot)
        elif len(b) == 1:
            for k in range(seg):
                faces.append((a[k], a[(k + 1) % seg], b[0])); face_mats.append(slot)
        else:
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append((a[k], a[k2], b[k2], b[k])); face_mats.append(slot)
    # Close open ends with a fan so every part is watertight.
    for ids, flip in ((ring_ids[0], False), (ring_ids[-1], True)):
        if len(ids) > 1:
            c = len(verts)
            y = verts[ids[0]][1]
            verts.append((0.0, y, 0.0))
            for k in range(seg):
                f = (c, ids[(k + 1) % seg], ids[k])
                faces.append(f[::-1] if flip else f); face_mats.append(mat(0) if callable(mat) else mat)
    base = len(geo.verts)
    for v in verts:
        p = Vector(v)
        if matrix is not None:
            p = matrix @ p
        geo.verts.append(tuple(p))
    for f, m in zip(faces, face_mats):
        geo.faces.append(tuple(base + i for i in f))
        geo.mats.append(m)
    return geo


def sphere_profile(rings, ry=1.0):
    return [(ry * math.cos(math.pi * i / rings), math.sin(math.pi * i / rings)) for i in range(rings + 1)]


def ellipsoid(center, radii, seg, rings, matrix=None, mat=0, geo=None):
    """Ellipsoid whose rings run along Y; ``matrix`` is applied after placement."""
    rx, ry, rz = radii
    m = trans(*center) @ scale3(rx, ry, rz)
    if matrix is not None:
        m = matrix @ m
    return revolve(sphere_profile(rings), seg, matrix=m, mat=mat, geo=geo)


def tube_down(profile, seg, matrix=None, mat=0, geo=None, sx=1.0, sz=1.0):
    """Lathe along -Y: profile [(t, r)] with t >= 0 measured downward from the joint."""
    return revolve([(-t, r) for (t, r) in profile], seg, matrix=matrix, mat=mat, geo=geo, sx=sx, sz=sz)


def segment_between(p0, p1, r0, r1, seg, mat=0, geo=None):
    """A tapered cone from p0 to p1 (three.js frame)."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    length = d.length
    q = Vector((0, -1, 0)).rotation_difference(d.normalized())
    m = trans(*p0) @ q.to_matrix().to_4x4()
    return tube_down([(0.0, r0), (length, r1)], seg, matrix=m, mat=mat, geo=geo)


def icosphere(subdiv, matrix):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    verts = [tuple(matrix @ v.co) for v in bm.verts]
    idx = {v: i for i, v in enumerate(bm.verts)}
    faces = [tuple(idx[v] for v in f.verts) for f in bm.faces]
    bm.free()
    return verts, faces


def ribbon(points, width, normal, mat=0, geo=None):
    """A flat strip along a polyline (three.js frame), facing ``normal``."""
    geo = geo or Geo()
    n = Vector(normal).normalized()
    pts = [Vector(p) for p in points]
    verts = []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        side = t.cross(n).normalized() * (width / 2)
        verts += [tuple(p + side), tuple(p - side)]
    faces = [(2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1) for i in range(len(pts) - 1)]
    return geo.extend(verts, faces, mat)


# ------------------------------------------------------------------ objects
def material(name, color, roughness=0.5, metallic=0.0, alpha=1.0, emission=None):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Alpha'].default_value = alpha
    if emission is not None:
        bsdf.inputs['Emission Color'].default_value = (*emission[0], 1.0)
        bsdf.inputs['Emission Strength'].default_value = emission[1]
    if alpha < 1.0:
        for attr, value in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
            try:
                setattr(mat, attr, value)
            except (AttributeError, TypeError):
                pass
        mat.use_backface_culling = False
    mat.diffuse_color = (*color, alpha)
    return mat


def mesh_object(name, geo, materials, smooth=True, parent=None, recalc=True):
    """Create a mesh object whose local frame equals ``parent``'s (identity basis)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([to_blender(v) for v in geo.verts], [], geo.faces)
    me.update()
    for m in materials:
        me.materials.append(m)
    me.polygons.foreach_set('material_index', geo.mats)
    me.update()
    if recalc:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
    me.polygons.foreach_set('use_smooth', [smooth] * len(me.polygons))
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
        obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    return obj


def empty(name, matrix_three=None, parent=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.3
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
        obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = matrix_to_blender(matrix_three if matrix_three is not None else Matrix.Identity(4))
    return obj


def triangle_count(objs):
    total = 0
    for o in objs:
        if o.type == 'MESH':
            total += sum(len(p.vertices) - 2 for p in o.data.polygons)
    return total


# ------------------------------------------------------------------ export/render
def export_glb(path, objects=None):
    bpy.ops.object.select_all(action='DESELECT')
    if objects:
        for o in objects:
            o.select_set(True)
    kwargs = dict(filepath=path, export_format='GLB', export_yup=True, export_apply=True,
                  export_extras=True, export_cameras=False, export_lights=False,
                  use_selection=bool(objects))
    try:
        bpy.ops.export_scene.gltf(**kwargs)
    except TypeError:
        kwargs.pop('export_extras', None)
        bpy.ops.export_scene.gltf(**kwargs)


def setup_render(scene, res, samples=48, threads=2):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'                 # never the simulation GPU
    scene.cycles.samples = samples
    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    except (AttributeError, TypeError):
        pass
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = threads
    scene.render.resolution_x = res[0]
    scene.render.resolution_y = res[1]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.compression = 90
    scene.view_settings.view_transform = 'Standard'
    world = bpy.data.worlds.new('NeuroFlyStudioWorld')
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.55, 0.6, 0.7, 1.0)
    bg.inputs['Strength'].default_value = 0.55
    scene.world = world


def studio_lights(scale=10.0):
    specs = [('KeyLight', (0.6, -0.8, 1.0), 900, 1.0), ('FillLight', (-1.0, -0.2, 0.6), 300, 1.6),
             ('RimLight', (0.0, 1.0, 0.8), 450, 1.2)]
    out = []
    for name, d, energy, size in specs:
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy * (scale / 10.0) ** 2
        data.size = size * scale / 2
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = Vector(d).normalized() * scale * 1.8
        obj.rotation_euler = (-obj.location).to_track_quat('-Z', 'Y').to_euler()
        out.append(obj)
    return out


def camera(name, location_three, target_three, ortho_scale=None, lens=50):
    data = bpy.data.cameras.new(name)
    if ortho_scale:
        data.type = 'ORTHO'
        data.ortho_scale = ortho_scale
    else:
        data.lens = lens
    data.clip_start = 0.05
    data.clip_end = 2000
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    loc = Vector(to_blender(location_three))
    tgt = Vector(to_blender(target_three))
    obj.location = loc
    up = 'Y'
    if abs((tgt - loc).normalized().z) > 0.999:
        # Looking straight down: keep three.js +Z (fly forward, Blender -Y) at the image top.
        obj.rotation_euler = (tgt - loc).to_track_quat('-Z', 'Y').to_euler()
        obj.rotation_euler.z += math.pi
    else:
        obj.rotation_euler = (tgt - loc).to_track_quat('-Z', up).to_euler()
    return obj


def render_to(scene, cam, path):
    scene.camera = cam
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
