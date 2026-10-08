#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# Build every environment/food/predator presentation prop headless, render its
# transparent PNGs, the icons and legend, the gallery scene and the contact sheet,
# then write tools/assets/env/ENV_MANIFEST.json.  CPU only, two threads.
#
#   BLENDER=/path/to/blender [RENDER_LOCK=/path/to/lockfile] tools/assets/build_env_all.sh OUTDIR [FLY_GLB]
#
# OUTDIR must be outside the repository.  FLY_GLB (optional) is fly_hq_lod0.glb
# from build_all.sh, shown in the gallery for scale.  When RENDER_LOCK is set each
# Blender run is serialised with flock on that file.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: BLENDER=... build_env_all.sh OUTDIR [FLY_GLB]}"
FLY="${2:-}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
run() {
    if [ -n "${RENDER_LOCK:-}" ]; then
        flock "$RENDER_LOCK" nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python "$@"
    else
        nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python "$@"
    fi
}
for prop in fermenting_fruit yeast_patch sugar_water odour_emitter wind_vane jumping_spider mantis_nymph; do
    run "$HERE/build_env.py" -- --out "$OUT" --prop "$prop" --render 1
done
python3 "$HERE/env/make_icons.py" --out "$HERE/env/icons"
run "$HERE/build_env_gallery.py" -- --out "$OUT" --fly "$FLY" --render 1
python3 "$HERE/env/contact_sheet.py" --out "$OUT" --icons "$HERE/env/icons"
python3 "$HERE/env/measure_env.py" --out "$OUT" --blender "$BLENDER" --write "$HERE/env/ENV_MANIFEST.json" > /dev/null
echo "measured -> tools/assets/env/ENV_MANIFEST.json"
