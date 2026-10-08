#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Measure built presentation assets and write the provenance/budget record.

    python3 tools/assets/measure_assets.py --out OUTDIR [--blender BLENDER] \
        [--write tools/assets/PROVENANCE.json]

Pure standard library (no Blender needed).  Reads every .glb and .png in OUTDIR,
counts triangles per glTF node instance, lists textures with their pixel sizes,
hashes outputs and the source scripts, and checks the phone-browser budgets.
Only file names (never absolute paths) are written.
"""
import argparse
import datetime
import hashlib
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SOURCES = ['tools/assets/nf_geom.py', 'tools/assets/build_fly.py', 'tools/assets/build_arena.py',
           'tools/assets/measure_assets.py', 'tools/assets/build_all.sh', 'web/hq_assets.js',
           'web/asset_preview.html', 'web/vendor/GLTFLoader.js']
BUDGETS = {  # provisional, from the asset card; not a measured frame-rate promise
    'fly_hq_lod0.glb': 30000, 'fly_hq_lod1.glb': 8000, 'arena_shell.glb': 20000, 'texture_max_px': 2048}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_glb(path):
    with open(path, 'rb') as fh:
        data = fh.read()
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(path + ' is not a glTF 2.0 binary')
    clen, ctype = struct.unpack_from('<II', data, 12)
    if ctype != 0x4E4F534A:
        raise ValueError(path + ' has no JSON chunk first')
    return json.loads(data[20:20 + clen].decode('utf-8'))


def glb_metrics(path):
    doc = read_glb(path)
    accessors = doc.get('accessors', [])
    meshes = doc.get('meshes', [])

    def mesh_tris(mesh):
        total = 0
        for prim in mesh.get('primitives', []):
            if prim.get('mode', 4) != 4:
                continue
            if 'indices' in prim:
                total += accessors[prim['indices']]['count'] // 3
            else:
                total += accessors[prim['attributes']['POSITION']]['count'] // 3
        return total

    nodes = doc.get('nodes', [])
    tris = sum(mesh_tris(meshes[n['mesh']]) for n in nodes if 'mesh' in n)
    images = doc.get('images', [])
    return {
        'triangles': tris, 'nodes': len(nodes), 'meshes': len(meshes),
        'materials': sorted(m.get('name', '') for m in doc.get('materials', [])),
        'alpha_blend_materials': sorted(m.get('name', '') for m in doc.get('materials', [])
                                        if m.get('alphaMode') == 'BLEND'),
        'textures': len(doc.get('textures', [])), 'images': len(images),
        'generator': doc.get('asset', {}).get('generator', ''),
        'node_names': sorted(n.get('name', '') for n in nodes),
    }


def png_size(path):
    with open(path, 'rb') as fh:
        head = fh.read(24)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(path + ' is not a PNG')
    return struct.unpack('>II', head[16:24])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--blender', default='')
    ap.add_argument('--write', default='')
    args = ap.parse_args()
    blender = ''
    if args.blender:
        try:
            first = subprocess.run([args.blender, '--version'], capture_output=True, text=True, timeout=60).stdout
            blender = next((ln.strip() for ln in first.splitlines() if ln.startswith('Blender')), '')
        except (OSError, subprocess.SubprocessError):
            blender = 'unavailable'
    outputs = {}
    problems = []
    max_tex = 0
    for name in sorted(os.listdir(args.out)):
        path = os.path.join(args.out, name)
        if name.endswith('.glb'):
            m = glb_metrics(path)
            entry = {'sha256': sha256(path), 'bytes': os.path.getsize(path), **m}
            if m['images']:
                problems.append(f'{name}: embeds {m["images"]} texture image(s); check each is <= 2048 px')
            if name in BUDGETS:
                entry['triangle_budget'] = BUDGETS[name]
                entry['within_budget'] = m['triangles'] <= BUDGETS[name]
                if not entry['within_budget']:
                    problems.append(f'{name}: {m["triangles"]} triangles > {BUDGETS[name]}')
            outputs[name] = entry
        elif name.endswith('.png'):
            w, h = png_size(path)
            outputs[name] = {'sha256': sha256(path), 'bytes': os.path.getsize(path), 'width': w, 'height': h}
        elif name.endswith('.blend'):
            outputs[name] = {'sha256': sha256(path), 'bytes': os.path.getsize(path),
                             'note': 'editable source scene; Blender .blend files are not byte-reproducible'}
    record = {
        'schema': 'neurofly.presentation-assets.v1',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'purpose': 'Presentation only. Stylised surface; not anatomical validation, not simulated biology.',
        'tools': {'blender': blender, 'python': sys.version.split()[0],
                  'three_js': 'r128 (web/vendor/three.min.js)',
                  'gltf_loader': 'three.js r128 examples/js/loaders/GLTFLoader.js (web/vendor/GLTFLoader.js)'},
        'sources': {rel: sha256(os.path.join(REPO, rel)) for rel in SOURCES if os.path.exists(os.path.join(REPO, rel))},
        'outputs': outputs,
        'budgets': {**BUDGETS, 'textures_used': sum(o.get('textures', 0) for o in outputs.values()),
                    'max_texture_px_used': max_tex, 'status': 'within budget' if not problems else problems},
        'render': {'engine': 'Cycles, CPU only, 2 threads', 'ortho_scale_viewport_mm': 12.0,
                   'note': 'fly ortho PNGs: px per viewport-mm = width / 12; head toward image top in *_top.png'},
        'licences': {
            'scripts': 'MIT (repository LICENSE)',
            'generated_assets': 'MIT (repository LICENSE); every vertex generated by the scripts, no third-party mesh',
            'three.js r128 + GLTFLoader.js': 'MIT, copyright 2010-2021 three.js authors (licenses/upstream/three.js-MIT.txt)',
            'Blender 4.5.3': 'GPL-2.0-or-later tool; its licence does not apply to generated output',
        },
    }
    text = json.dumps(record, indent=2, sort_keys=False) + '\n'
    if args.write:
        with open(args.write, 'w') as fh:
            fh.write(text)
    print(text)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
