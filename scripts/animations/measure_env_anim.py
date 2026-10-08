#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Write the ENV-ANIM-01 manifest: sources, parameters (with classes), hashes, checks, licences.

    python3 scripts/animations/measure_env_anim.py --out OUTDIR [--write scripts/animations/ENV_ANIM_MANIFEST.json]

Runs verify_env_clips.py on both animals' clip GLBs (exit 1 on any violation) and hashes every file in
OUTDIR.  Standard library only; only file names are written.
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

SOURCES = ['tools/assets/build_env_rigs.py', 'tools/assets/env/ENV_RIG_CONTRACT.md',
           'scripts/animations/build_env_clips.py', 'scripts/animations/verify_env_clips.py',
           'scripts/animations/build_env_vane.py', 'scripts/animations/env_anim_params.json',
           'scripts/animations/measure_env_anim.py', 'scripts/animations/build_env_anim_all.sh', 'tools/assets/nf_geom.py']
REFERENCES = [
    {'id': 'shamble2017', 'citation': 'Shamble PS, Hoy RR, Cohen I, Beatus T (2017) Walking like an ant: a quantitative and '
     'experimental approach to understanding locomotor mimicry in the jumping spider Myrmarachne formicaria. Proc R Soc B '
     '284:20170308.', 'licence': 'CC BY 4.0', 'used_for': 'salticid tetrapod phase, stationary fraction, speeds (non-mimetic '
     'controls Salticus/Sitticus/Phidippus); values only, no files reused'},
    {'id': 'deagro2021', 'citation': 'De Agro M, Rossler DC, Kim K, Shamble PS (2021) Perception of biological motion by '
     'jumping spiders. PLoS Biol 19:e3001172.', 'licence': 'CC BY 4.0', 'used_for': 'stride timing (low confidence); no files reused'},
    {'id': 'hao2019', 'citation': 'Hao X, et al. (2019) Appl Bionics Biomech 4617212 (Grammostola rosea gait).', 'licence': 'CC BY',
     'used_for': 'duty factor proxy'},
    {'id': 'land1972', 'citation': 'Land MF (1972) Stepping movements made by jumping spiders during turns mediated by the '
     'lateral eyes. J Exp Biol 57:15-40. doi:10.1242/jeb.57.1.15', 'licence': 'abstract only', 'used_for': 'turn rates and stepping'},
    {'id': 'parigi2019', 'citation': 'Parigi A, et al. (2019) PLoS ONE 14(5):e0216860.', 'licence': 'CC BY 4.0',
     'used_for': 'body sizes (spider, first-instar mantis)'},
    {'id': 'grabowska2012', 'citation': 'Grabowska M, Godlewska E, Schmidt J, Daun-Gruhn S (2012) Quadrupedal gaits in hexapod '
     'animals - inter-leg coordination in free-walking adult stick insects. J Exp Biol 215:4255-4266.', 'licence': 'values cited only',
     'used_for': 'four-leg phase proxy, stride period, duty'},
    {'id': 'yamawaki2011', 'citation': 'Yamawaki Y, et al. (2011) J Insect Physiol 57:1010. doi:10.1016/j.jinsphys.2011.04.018',
     'licence': 'abstract only', 'used_for': 'head/prothorax/abdomen onset within 40 ms'},
    {'id': 'kral2012', 'citation': 'Kral K (2012) Eur J Entomol 109:295. doi:10.14411/eje.2012.039', 'licence': 'values cited only',
     'used_for': 'peering sway amplitude/speed (adult, scaled to nymph under a labelled assumption)'},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--write', default='')
    a = ap.parse_args()
    problems, animals = [], {}
    for stem in ('env_jumping_spider_rig', 'env_mantis_nymph_rig'):
        p = lambda s: os.path.join(a.out, stem + s)   # noqa: E731
        rc = subprocess.run([sys.executable, os.path.join(HERE, 'verify_env_clips.py'), p('_clips.glb'), p('.glb'),
                             p('_joints.json'), p('_clips.json'), '--json', p('_clips_verify.json')],
                            capture_output=True, text=True).returncode
        ver = json.load(open(p('_clips_verify.json')))
        meta = json.load(open(p('_clips.json')))
        joints = json.load(open(p('_joints.json')))
        if rc:
            problems.append(stem + ': GLB verification failed')
        for c, r in meta['clips'].items():
            if r['max_leg_body_penetration_viewport_mm'] > 0.0 or r['max_foot_target_error_viewport_mm'] > 1e-3 \
                    or r['dofs_outside_display_limits']:
                problems.append(f'{stem}/{c}: Blender check')
        m = glb_metrics(p('_clips.glb'))
        animals[stem] = {
            'rig': joints['rig'], 'bind_pose_vs_static_prop': joints['static_check'], 'triangles': m['triangles'],
            'clips': {c: {k: r[k] for k in ('frames', 'fps', 'seconds', 'loop', 'root_motion',
                                             'displacement_per_clip_viewport_mm', 'heading_change_rad',
                                             'max_leg_body_penetration_viewport_mm', 'max_foot_target_error_viewport_mm',
                                             'lowest_leg_vertex_minus_foot_plane')} for c, r in meta['clips'].items()},
            'glb_verification': {c: {k: v for k, v in r.items() if k != 'per_joint'} for c, r in ver.items()},
            'playback_multiplier_default': meta['playback_multiplier_default'],
            'estimated_physical_speed_mm_s': meta['estimated_physical_speed_mm_s'],
            'estimated_physical_speed_bl_s': meta['estimated_physical_speed_bl_s'],
            'gait_label': meta['gait_label'], 'parameters': meta['parameters'],
            'sprite_sheets': meta.get('sprite_sheets')}
    files = {}
    for f in sorted(os.listdir(a.out)):
        fp = os.path.join(a.out, f)
        if os.path.isfile(fp) and not f.endswith('.blend1'):
            files[f] = {'sha256': sha256(fp), 'bytes': os.path.getsize(fp)}
            if f.endswith('.png'):
                files[f]['width'], files[f]['height'] = png_size(fp)
    record = {
        'schema': 'neurofly.env-animation.v1',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'label': ('ILLUSTRATIVE animation with source-informed timing; not motion capture, not simulated behaviour; no '
                  'predator AI, contact physics, retinal stimulus or neural response'),
        'conventions': {'frame': 'three.js +Y up, +Z forward, viewport-mm', 'playback': 'clip seconds are estimated '
                        'physical seconds; playback multiplier default 1', 'inplace_vs_rootmotion': '*_inplace keeps '
                        '<p>_motion fixed (feet slide back in stance); *_rootmotion moves it so stance feet stay planted',
                        'turns': 'turn_left = heading rotates toward +X (the anatomical left in a right-handed +Y-up, '
                        '+Z-forward frame); leg side letters follow the existing viewport convention (L at -X), whose '
                        'handedness is a separate open renderer question',
                        'scale': 'predators use their own viewport-mm-per-mm factor (parameters), not the fly 3.3x'},
        'checks': {'glb': 'rest reset vs rig GLB; DOF limits over the whole duration at 240 Hz incl. loop seam; key and '
                   'seam continuity < 0.5 rad; FK ground check of every leg vertex; stance-foot slide per 240 Hz '
                   'sample <= 0.01 viewport-mm in root-motion clips (stance intervals from metadata); travel heading '
                   '+Z and speed within 5 % for walk_forward_rootmotion',
                   'blender': 'every frame: leg-into-body penetration (ray parity, hip sockets excluded), IK target '
                              'error, lowest leg vertex vs foot plane, display limits',
                   'not_checked': 'leg-to-leg contact; the composed walk-scene GLB is the clips GLB plus a ground '
                                  'node and is hashed but not re-verified separately; 2D sprites are checked only by '
                                  'their documented camera/scale, not by image analysis'},
        'references': REFERENCES,
        'environment': json.load(open(os.path.join(a.out, 'env_wind_vane_mapping.json'))),
        'sources': {s: sha256(os.path.join(REPO, s)) for s in SOURCES if os.path.exists(os.path.join(REPO, s))},
        'animals': animals, 'files': files, 'problems': problems,
        'licences': {'scripts, rigs, clips': 'MIT (repository LICENSE); original geometry and curves',
                     'external sources': 'numerical values cited with attribution; no external files, figures or '
                                         'videos were copied',
                     'Blender 4.5.3': 'GPL-2.0-or-later tool; its licence does not apply to output'},
    }
    if a.write:
        open(a.write, 'w').write(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'problems': problems}))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
