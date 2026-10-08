#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# Build the illustrative fly animation clips for female and male at both LODs from the
# rig-v2 GLBs (W2 joint contract), render the sprite and check sheets, verify the joint
# envelope on the exported GLBs and write scripts/animations/ANIM_MANIFEST.json.
# CPU only, two threads; each Blender run optionally serialised behind RENDER_LOCK.
#
#   BLENDER=/path/to/blender [RENDER_LOCK=/path/to/lock] scripts/animations/build_clips_all.sh RIG_V2_DIR OUTDIR
#
# OUTDIR must be a NEW versioned directory outside the repository; RIG_V2_DIR is only read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIG="${1:?usage: build_clips_all.sh RIG_V2_DIR OUTDIR}"
OUT="${2:?usage: build_clips_all.sh RIG_V2_DIR OUTDIR}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
if [ -n "${RENDER_LOCK:-}" ]; then L=(flock "$RENDER_LOCK"); else L=(); fi
for sex in female male; do
    for lod in 0 1; do
        "${L[@]}" nice -n 19 "$BLENDER" --background --factory-startup --threads 2 \
            --python "$HERE/build_fly_clips.py" -- --glb "$RIG/fly_${sex}_v2_lod${lod}.glb" --out "$OUT" --render 1
    done
done
python3 "$HERE/measure_clips.py" --out "$OUT" --rig "$RIG" --write "$HERE/ANIM_MANIFEST.json" \
    --sheet "$OUT/anim_contact_sheet.png"
