#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Measure the environment props and write the source / licence / budget manifest.

    python3 tools/assets/env/measure_env.py --out OUTDIR [--blender BLENDER] \
        [--write tools/assets/env/ENV_MANIFEST.json]

Standard library only.  Reuses the GLB reader of measure_assets.py.  Hashes every
build output in OUTDIR (env_* files), every committed icon and every source script,
checks the phone budgets, and records the predator ecology citations.  Only file
names and repository-relative paths are written, never absolute paths.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ASSETS))
sys.path.insert(0, ASSETS)

from measure_assets import glb_metrics, png_size, sha256  # noqa: E402

SOURCES = ['tools/assets/nf_geom.py', 'tools/assets/build_env.py', 'tools/assets/build_env_gallery.py',
           'tools/assets/build_env_all.sh', 'tools/assets/env/make_icons.py',
           'tools/assets/env/contact_sheet.py', 'tools/assets/env/measure_env.py']
PREDATORS = ('env_jumping_spider.glb', 'env_mantis_nymph.glb')
BUDGET = {'prop_triangles': 8000, 'predator_triangles': 12000, 'texture_max_px': 1024,
          'note': 'provisional phone-browser budgets per GLB; no textures are used (material colours only)'}

ECOLOGY = {
    'summary': ('Both predator families are documented catching Drosophila in laboratory assays and both are '
                'reared on D. melanogaster. Field predation on wild D. melanogaster is not documented: Parigi et '
                'al. (2019) state that the natural predators of wild populations have not been documented. The '
                'models are therefore labelled ILLUSTRATIVE examples of plausible predators, not field ecology.'),
    'jumping_spider': {
        'model': 'Salticidae, styled after the zebra jumping spider Salticus scenicus',
        'evidence': [
            {'citation': ('Parigi A, Porter C, Cermak M, Pitchers WR, Dworkin I (2019). The behavioral repertoire of '
                          'Drosophila melanogaster in the presence of two predator species that differ in hunting '
                          'mode. PLoS ONE 14(5): e0216860.'),
             'doi': '10.1371/journal.pone.0216860',
             'finding': ('Salticus scenicus vs D. melanogaster in arenas: about 50% of spiders captured the fly '
                         'within 10 min; spiders were maintained on ~5 D. melanogaster per week.')},
            {'citation': ('Taylor PW, Jackson RR, Robertson MW (1998). A case of blind spider\'s buff?: prey-capture '
                          'by jumping spiders (Araneae, Salticidae) in the absence of visual cues. Journal of '
                          'Arachnology 26: 369-381.'),
             'url': 'https://www.biodiversitylibrary.org/part/228756',
             'finding': ('All 42 salticid species tested caught prey (house flies and Drosophila spp.) in at least '
                         'one procedure.')},
        ]},
    'mantis': {
        'model': 'Mantidae nymph, styled after the Chinese mantis Tenodera sinensis (T. aridifolia sinensis)',
        'evidence': [
            {'citation': 'Parigi et al. (2019), PLoS ONE 14(5): e0216860 (as above).',
             'doi': '10.1371/journal.pone.0216860',
             'finding': ('Tenodera aridifolia sinensis used as an ambush predator of D. melanogaster; juvenile '
                         'mantids were fed D. melanogaster. Captures occurred but were rare in the mantid '
                         'treatment ("very few individuals were captured").')},
        ],
        'caveat': ('Adult T. sinensis are far larger than the fly and take larger prey; the model is an early '
                   'nymph, the stage the cited study fed on D. melanogaster.')},
}


def blender_version(path):
    if not path:
        return ''
    try:
        first = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=60).stdout
        return next((ln.strip() for ln in first.splitlines() if ln.startswith('Blender')), '')
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--blender', default='')
    ap.add_argument('--write', default='')
    args = ap.parse_args()
    outputs, problems = {}, []
    for name in sorted(os.listdir(args.out)):
        path = os.path.join(args.out, name)
        if not name.startswith('env_') or name.endswith('.blend1'):
            continue
        entry = {'sha256': sha256(path), 'bytes': os.path.getsize(path)}
        if name.endswith('.glb'):
            m = glb_metrics(path)
            budget = BUDGET['predator_triangles'] if name in PREDATORS else BUDGET['prop_triangles']
            entry.update(triangles=m['triangles'], nodes=m['nodes'], meshes=m['meshes'],
                         materials=m['materials'], alpha_blend_materials=m['alpha_blend_materials'],
                         textures=m['textures'], images=m['images'], generator=m['generator'],
                         node_names=m['node_names'], triangle_budget=budget,
                         within_budget=m['triangles'] <= budget and m['images'] == 0)
            if not entry['within_budget']:
                problems.append(f'{name}: {m["triangles"]} triangles (budget {budget}), {m["images"]} images')
            side = os.path.join(args.out, name[:-4] + '.render.json')
            if os.path.exists(side):
                with open(side) as fh:
                    info = json.load(fh)
                entry['status'] = info.get('status')
                entry['bbox_three'] = {'min': info.get('bbox_three_min'), 'max': info.get('bbox_three_max')}
                entry['metadata'] = {k: v for k, v in info.items()
                                     if k not in ('prop', 'triangles', 'status', 'bbox_three_min', 'bbox_three_max',
                                                  'renders')}
                if 'renders' in info:
                    entry['renders'] = info['renders']
        elif name.endswith('.png'):
            entry['width'], entry['height'] = png_size(path)
        elif name.endswith('.blend'):
            entry['note'] = 'editable source scene; Blender .blend files are not byte-reproducible'
        outputs[name] = entry
    icon_dir = os.path.join(HERE, 'icons')
    icons = {f: sha256(os.path.join(icon_dir, f)) for f in sorted(os.listdir(icon_dir))}
    record = {
        'schema': 'neurofly.presentation-assets.env.v1',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'purpose': ('Presentation only: food, odour, wind and predator props, icons and a sense legend. Not a '
                    'stimulus, field, encoder input, collision shape or behaviour. Loading them must not change '
                    'physics, stimulus geometry, looming or retinal input.'),
        'status_labels': {
            'DECORATIVE': 'food / odour / wind props; the picture does not imply that a sense or field is simulated',
            'ILLUSTRATIVE': 'predators; no predator stimulus or behaviour is implemented',
        },
        'frame': ('three.js frame, +Y up, +Z forward, 1 unit = 1 viewport-mm (tools/assets/INTERFACE.md); each '
                  'prop stands on y = 0 and is centred on its reference point. Food props are a nominal size, '
                  'not a simulated source radius; predators use the fly display scale (~3.3x life).'),
        'sense_convention': {
            'sugar_taste': 'CONTACT (tarsi / proboscis): droplet, solid bar, solid contact ring',
            'volatile_odour': 'AIRBORNE at a distance (antennae): dashed rings and wavy lines; rings mark the source '
                              'position only, never a concentration or plume',
            'wind': 'arrow tip = where the air moves to (downwind); show NOT SIMULATED without a wind field',
        },
        'tools': {'blender': blender_version(args.blender), 'python': sys.version.split()[0],
                  'render': 'Cycles, CPU only, 2 threads; transparent PNG', 'icons': 'hand-written SVG; PNG via Inkscape'},
        'sources': {rel: sha256(os.path.join(REPO, rel)) for rel in SOURCES if os.path.exists(os.path.join(REPO, rel))},
        'icons': icons,
        'outputs': outputs,
        'budgets': {**BUDGET, 'textures_used': sum(o.get('textures', 0) for o in outputs.values()),
                    'status': 'within budget' if not problems else problems},
        'predator_ecology': ECOLOGY,
        'licences': {
            'scripts': 'MIT (repository LICENSE)',
            'generated_assets_and_icons': ('MIT (repository LICENSE). Original geometry and artwork: every vertex '
                                           'and path is generated by these scripts. No imported mesh, texture, scan '
                                           'or third-party artwork (no CC0 / CC-BY imports were needed).'),
            'Blender 4.5.3': 'GPL-2.0-or-later tool; its licence does not apply to generated output',
            'Inkscape': 'GPL tool used only to rasterise the SVGs; its licence does not apply to output',
        },
    }
    text = json.dumps(record, indent=2) + '\n'
    if args.write:
        with open(args.write, 'w') as fh:
            fh.write(text)
    print(text)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
