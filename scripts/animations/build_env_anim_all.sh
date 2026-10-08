#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# ENV-ANIM-01: spider and mantis rigs, clips (in-place + root-motion), walking scenes, sprites, the
# wind-vane DEMO/mapping, verification and scripts/animations/ENV_ANIM_MANIFEST.json.  CPU, 2 threads.
#
#   BLENDER=... [RENDER_LOCK=...] scripts/animations/build_env_anim_all.sh ENV_DIR OUTDIR
#
# ENV_DIR holds the static W3 props (only read); OUTDIR must be a NEW versioned directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="$HERE/../../tools/assets"
ENV="${1:?usage: build_env_anim_all.sh ENV_DIR OUTDIR}"
OUT="${2:?}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
if [ -n "${RENDER_LOCK:-}" ]; then L=(flock "$RENDER_LOCK"); else L=(); fi
B=(nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python)
for a in spider mantis; do
    "${B[@]}" "$ASSETS/build_env_rigs.py" -- --out "$OUT" --animal "$a" --static "$ENV"
done
"${L[@]}" "${B[@]}" "$HERE/build_env_clips.py" -- --rig "$OUT/env_jumping_spider_rig.glb" \
    --params "$HERE/env_anim_params.json" --animal spider --out "$OUT" --render 1
"${L[@]}" "${B[@]}" "$HERE/build_env_clips.py" -- --rig "$OUT/env_mantis_nymph_rig.glb" \
    --params "$HERE/env_anim_params.json" --animal mantis --out "$OUT" --render 1
"${B[@]}" "$HERE/build_env_vane.py" -- --props "$ENV" --out "$OUT"
python3 "$HERE/measure_env_anim.py" --out "$OUT" --write "$HERE/ENV_ANIM_MANIFEST.json"
