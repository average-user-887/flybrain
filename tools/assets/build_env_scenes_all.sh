#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# Build the three composed presentation scenes (GLB + .blend + hero/close/top renders) and
# measure them into tools/assets/env/SCENES_MANIFEST.json.  CPU only, two threads.
#
#   BLENDER=... [RENDER_LOCK=...] tools/assets/build_env_scenes_all.sh ENV_DIR RIG_V2_DIR OUTDIR
#
# ENV_DIR holds env_*.glb (build_env_all.sh); RIG_V2_DIR holds fly_{female,male}_v2_lod1.glb (W2).
# OUTDIR must be a NEW versioned directory outside the repository; the inputs are only read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${1:?usage: build_env_scenes_all.sh ENV_DIR RIG_V2_DIR OUTDIR}"
RIG="${2:?}"
OUT="${3:?}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
if [ -n "${RENDER_LOCK:-}" ]; then L=(flock "$RENDER_LOCK"); else L=(); fi
for scene in fermenting_fruit_patch sugar_water_feeder predator_encounter_illustrative; do
    "${L[@]}" nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python "$HERE/build_env_scenes.py" -- \
        --out "$OUT" --props "$ENV" --fly-female "$RIG/fly_female_v2_lod1.glb" --fly-male "$RIG/fly_male_v2_lod1.glb" \
        --scene "$scene" --render 1
done
python3 "$HERE/env/measure_scenes.py" --out "$OUT" --env "$ENV" --rig "$RIG" --write "$HERE/env/SCENES_MANIFEST.json"
