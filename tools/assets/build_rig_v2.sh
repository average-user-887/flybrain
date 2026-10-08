#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# Build rig v2 (articulated wings, antennae, halteres; seated bristles) for female and
# male at both LODs into a NEW versioned directory, with rest and open-pose checks
# (top/side/under/hero), measurements and a contact sheet.  CPU, 2 threads, nice 19,
# optionally serialised behind RENDER_LOCK.  Joint contract: JOINT_CONTRACT.md.
#
#   BLENDER=/path/to/blender [RENDER_LOCK=/path/to/lock] tools/assets/build_rig_v2.sh OUTDIR
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: BLENDER=... build_rig_v2.sh OUTDIR}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
if [ -n "${RENDER_LOCK:-}" ]; then L=(flock "$RENDER_LOCK"); else L=(); fi
for sex in female male; do
    for lod in 0 1; do
        "${L[@]}" nice -n 19 "$BLENDER" --background --factory-startup --threads 2 \
            --python "$HERE/build_fly.py" -- --out "$OUT" --sex "$sex" --lod "$lod" --rig v2 --render 1
    done
done
"${L[@]}" nice -n 19 "$BLENDER" --background --threads 2 "$OUT/fly_female_v2_lod0.blend" \
    --python "$HERE/render_closeup.py" -- c_thorax_bristles "$OUT/fly_female_v2_lod0_bristles_closeup.png" \
    hide=l_wing,r_wing view=side
python3 "$HERE/measure_assets.py" --out "$OUT" --blender "$BLENDER" --write "$HERE/PROVENANCE_RIG_V2.json" > /dev/null
python3 "$HERE/contact_sheet.py" "$OUT" "$OUT/fly_rig_v2_contact_sheet.png" --rig v2
echo "measured -> tools/assets/PROVENANCE_RIG_V2.json; sheet -> $OUT/fly_rig_v2_contact_sheet.png"
