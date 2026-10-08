#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Write the animation manifest: hashes, budgets, clip list, dense envelope checks, sources.

    python3 scripts/animations/measure_clips.py --out CLIP_DIR --rig RIG_V2_DIR \
        [--write scripts/animations/ANIM_MANIFEST.json] [--sheet CLIP_DIR/anim_contact_sheet.png]

Runs verify_clip_envelope.py on every *_clips.glb against the rig's own *_joints.json and
fails (exit 1) on any envelope violation, any off-axis rotation or a GLB over budget.
Only file names are written.  --sheet needs Pillow.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'tools', 'assets'))
from measure_assets import glb_metrics, png_size, sha256  # noqa: E402

SOURCES = ['scripts/animations/build_fly_clips.py', 'scripts/animations/verify_clip_envelope.py',
           'scripts/animations/measure_clips.py', 'scripts/animations/build_clips_all.sh', 'tools/assets/nf_geom.py']
BUDGET = {'lod0_triangles': 30000, 'lod1_triangles': 8000, 'glb_bytes_max': 2_000_000, 'texture_max_px': 2048}

REFERENCES = [
    {'id': 'vaxenburg2025', 'used_for': 'joint axes and ranges of the rig (via the W2 joint contract)',
     'citation': 'Vaxenburg R, Siwanowicz I, Merel J, et al. (2025) Whole-body physics simulation of fruit fly '
                 'locomotion. Nature 643:1312-1320.', 'doi': '10.1038/s41586-025-09029-4'},
    {'id': 'fry2005', 'used_for': 'wingbeat amplitude and frequency (wingbeat_loop)',
     'citation': 'Fry SN, Sayaman R, Dickinson MH (2005) The aerodynamics of hovering flight in Drosophila. '
                 'J Exp Biol 208:2303-2318.', 'doi': '10.1242/jeb.01612',
     'values': 'free hovering D. melanogaster, Table 1: wingbeat frequency 218 +/- 7 Hz, stroke amplitude '
               '140 +/- 10 deg (n = 6); tethered flight 186-195 Hz'},
    {'id': 'deora2015', 'used_for': 'haltere beating in antiphase with the wing (wingbeat_loop)',
     'citation': 'Deora T, Singh AK, Sane SP (2015) Biomechanical basis of wing and haltere coordination in '
                 'flies. PNAS 112:1481-1486.', 'doi': '10.1073/pnas.1412279112',
     'caveat': 'shown in soldier and flesh flies, not measured in Drosophila here'},
    {'id': 'mamiya2011', 'used_for': 'antennal joints and movements (antenna_sweep, wingbeat_loop)',
     'citation': 'Mamiya A, Straw AD, Tomasson E, Dickinson MH (2011) Active and passive antennal movements '
                 'during visually guided steering in flying Drosophila. J Neurosci 31:6900-6914.',
     'doi': '10.1523/JNEUROSCI.0498-11.2011',
     'values': 'the scape-pedicel joint is moved by two muscles; the pedicel-funiculus joint moves passively; '
               'passive movements include small oscillations at wingbeat frequency'},
    {'id': 'gopfert2002', 'used_for': 'funiculus/arista rotation about the funiculus axis (via W2 contract)',
     'citation': 'Gopfert MC, Robert D (2002) The mechanical basis of Drosophila audition. J Exp Biol '
                 '205:1199-1208.', 'doi': '10.1242/jeb.205.9.1199'},
    {'id': 'seeds2014', 'used_for': 'grooming order, which legs groom which body part, sweep/rub alternation',
     'citation': 'Seeds AM, Ravbar P, Chung P, Hampel S, Midgley FM Jr, Mensh BD, Simpson JH (2014) A '
                 'suppression hierarchy among competing motor programs drives sequential grooming in '
                 'Drosophila. eLife 3:e02951.', 'doi': '10.7554/eLife.02951',
     'values': '"The priority order for cleaning the different body parts is: eyes > antennae > abdomen > wings > '
               'thorax." "Front leg cleaning movements are directed to the head whereas the hind legs clean the '
               'abdomen, wings, and thoraces." Cleaning alternates sweeps of the targeted region with leg rubbing.'},
    {'id': 'hampel2015', 'used_for': 'antennal grooming as a distinct leg movement program',
     'citation': 'Hampel S, Franconville R, Simpson JH, Seeds AM (2015) A neural command circuit for grooming '
                 'movement control. eLife 4:e08758.', 'doi': '10.7554/eLife.08758'},
]

STYLISED = [
    'Every clip is an illustrative animation, not simulated behaviour; it implies no aerodynamics, '
    'mechanosensation, grooming decision or connectome/neural output, and must never be overlaid on recorded or '
    'live body or neural telemetry (recorded poses always win).',
    'Display time is slowed: wingbeat_loop shows one stroke in 24 frames at 30 fps (1.25 Hz), about 175x slower '
    'than the cited 218 Hz.',
    'Wing stroke is a sweep about the body vertical from 0.35 to 2.75 rad (137.5 deg, inside the cited 140 +/- 10 '
    'deg) with elevation kept inside the display-safe envelope. Wing pitch is limited to [-0.07, 0.13] rad, so '
    'the large wing rotation at stroke reversal is NOT shown. The stroke plane, deviation and timing are not '
    'measured kinematics.',
    'Halteres beat at the clip rate in antiphase with the stroke at +/-0.2 rad (contract range); amplitude is '
    'stylised.',
    'antenna_sweep exaggerates the small real active antennal movements so they are visible; funiculus rotation '
    'is +/-0.09 rad, far larger than real sound-driven motion.',
    'Grooming legs are posed by planar inverse kinematics on the rig\'s existing coxa/femur/tibia empties (joint '
    'origins unchanged); the paths, timing and number of strokes (two per body part) are invented for display. '
    'Thorax (notum) grooming, the last item in the cited order, is omitted because the rig\'s planar legs cannot '
    'reach the notum without piercing the body. Abdominal strokes run along the side of the abdomen and wing '
    'strokes under the lifted wing edge.',
    'Body, root, thorax frame and the standing legs never move; real flies shift their body while grooming.',
    'Coxa bases (within 0.45 viewport-mm of their joint) and wing bases (within 0.40) sit in their sockets and '
    'are excluded from the penetration test, as in the W2 measurements. Leg-to-leg contact (rubbing) is not '
    'tested.',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--rig', required=True)
    ap.add_argument('--write', default='')
    ap.add_argument('--sheet', default='')
    a = ap.parse_args()
    variants, problems = {}, []
    for name in sorted(os.listdir(a.out)):
        if not name.endswith('_clips.glb'):
            continue
        stem = name[:-len('_clips.glb')]
        glb = os.path.join(a.out, name)
        joints = os.path.join(a.rig, stem + '_joints.json')
        env_json = os.path.join(a.out, stem + '_envelope.json')
        rc = subprocess.run([sys.executable, os.path.join(HERE, 'verify_clip_envelope.py'), glb, joints,
                             '--json', env_json], capture_output=True, text=True).returncode
        env = json.load(open(env_json))
        meta = json.load(open(os.path.join(a.out, stem + '_clips.json')))
        m = glb_metrics(glb)
        budget = BUDGET['lod0_triangles'] if stem.endswith('lod0') else BUDGET['lod1_triangles']
        files = {}
        for f in sorted(os.listdir(a.out)):
            if f.startswith(stem + '_') and not f.endswith('.blend1'):
                p = os.path.join(a.out, f)
                files[f] = {'sha256': sha256(p), 'bytes': os.path.getsize(p)}
                if f.endswith('.png'):
                    files[f]['width'], files[f]['height'] = png_size(p)
        clips = {}
        for clip, r in meta['clips'].items():
            e = env.get(clip, {})
            clips[clip] = {k: r[k] for k in ('frames', 'fps', 'seconds', 'loop', 'max_new_penetration_viewport_mm',
                                             'worst_mesh', 'max_below_foot_plane_viewport_mm')}
            clips[clip]['envelope_check_exported_glb'] = e
            if not e or e.get('violations', 1) != 0 or e.get('max_off_axis_quaternion_component', 1) > 2e-3:
                problems.append(f'{stem}/{clip}: envelope')
        if rc != 0:
            problems.append(f'{stem}: verifier exit {rc}')
        if m['triangles'] > budget or os.path.getsize(glb) > BUDGET['glb_bytes_max'] or m['images']:
            problems.append(f'{stem}: budget')
        variants[stem] = {'source_rig_glb': stem + '.glb', 'source_rig_glb_sha256': sha256(os.path.join(a.rig, stem + '.glb')),
                          'joints_json_sha256': sha256(joints), 'triangles': m['triangles'], 'triangle_budget': budget,
                          'textures': m['textures'], 'glb_animations': sorted(clips), 'clips': clips,
                          'ik_configs_chosen': meta.get('ik_configs_chosen'), 'sprite_sheets': meta.get('sprite_sheets'),
                          'files': files}
    record = {
        'schema': 'neurofly.presentation-animations.v1',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'label': 'Illustrative animation, not simulated behaviour.',
        'rig': 'neurofly-viewport-fly-v2 (W2 joint contract, tools/assets/JOINT_CONTRACT.md on claude/ui-assets-20261008)',
        'envelope': {
            'rule': 'W2 contract range AND display-safe envelope for every DOF over the WHOLE duration of every clip; '
                    'checked on the exported GLB (decoded quaternions, LINEAR interpolation, 20 sub-samples per frame, '
                    'loop wrap included); zero violations required; no runtime clamping',
            'limits': {'wing_sweep': [0.0, 3.0], 'wing_elevate': '>= contract minimum for the current sweep, <= 0.8',
                       'wing_pitch': '0 while sweep < 0.5, else [-0.07, 0.13] (intersection of all measured '
                                     'contract intervals)',
                       'antenna_abduct': [-0.4, 0.6], 'antenna_extend': '[-0.2, 0.2] and <= measured max for abduct',
                       'antenna_twist': [-0.1, 0.09], 'funiculus_rotate': [-0.1, 0.1], 'haltere_beat': [-0.2, 0.2]},
            'legs': 'leg empties (<leg>_coxa_joint, _femur_joint, _tibia_joint) are rotated only; their origins are '
                    'unchanged; no leg range is defined by the contract, so legs are checked by penetration and the '
                    'foot plane instead',
            'player_note': 'joints a clip leaves at rest are not keyed; a player resets to rest when it stops a clip '
                           '(three.js AnimationAction.stop does)',
        },
        'stylised_or_display_scaled': STYLISED,
        'references': REFERENCES,
        'budgets': {**BUDGET, 'status': 'within budget' if not any('budget' in p for p in problems) else problems},
        'sources': {s: sha256(os.path.join(REPO, s)) for s in SOURCES if os.path.exists(os.path.join(REPO, s))},
        'variants': variants,
        'problems': problems,
        'licences': {'scripts and generated clips': 'MIT (repository LICENSE); geometry is the W2 rig v2 (MIT), '
                                                    'animation curves generated by these scripts',
                     'Blender 4.5.3': 'GPL-2.0-or-later tool; its licence does not apply to output'},
    }
    if a.sheet:
        from PIL import Image, ImageDraw
        rows = []
        for stem in sorted(variants):
            for f in (f'{stem}_groom_full_cycle_checks.png', f'{stem}_wing_open_fold_sprites.png',
                      f'{stem}_wingbeat_loop_sprites.png', f'{stem}_antenna_sweep_sprites.png'):
                p = os.path.join(a.out, f)
                if os.path.exists(p):
                    rows.append((f, Image.open(p).convert('RGBA')))
        w = max(im.width for _, im in rows) // 2
        h = sum(im.height // 2 + 26 for _, im in rows)
        sheet = Image.new('RGBA', (w, h), (78, 84, 96, 255))
        d = ImageDraw.Draw(sheet)
        y = 0
        for f, im in rows:
            d.text((6, y + 6), f + '  (illustrative animation, not simulated behaviour)', fill=(240, 240, 235, 255))
            im = im.resize((im.width // 2, im.height // 2))
            sheet.alpha_composite(im, (0, y + 26))
            y += im.height + 26
        sheet.save(a.sheet)
        record['contact_sheet'] = {'file': os.path.basename(a.sheet), 'sha256': sha256(a.sheet)}
    text = json.dumps(record, indent=2) + '\n'
    if a.write:
        open(a.write, 'w').write(text)
    print(json.dumps({'problems': problems, 'variants': sorted(variants)}))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
