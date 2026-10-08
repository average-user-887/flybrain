#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# Build the female and male appearance variants (two LODs each), their orthographic
# PNGs, the measurements and a contact sheet.  CPU only, two threads, low priority,
# serialised behind an optional render lock.  Appearance only: see SEX_VARIANTS.md.
#
#   BLENDER=/path/to/blender [RENDER_LOCK=/path/to/lockfile] tools/assets/build_sex_variants.sh OUTDIR
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: BLENDER=... build_sex_variants.sh OUTDIR}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
run() {
    if [ -n "${RENDER_LOCK:-}" ]; then
        flock "$RENDER_LOCK" nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python "$@"
    else
        nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python "$@"
    fi
}
for sex in female male; do
    for lod in 0 1; do
        run "$HERE/build_fly.py" -- --out "$OUT" --sex "$sex" --lod "$lod" --render 1
    done
done
if [ -n "${RENDER_LOCK:-}" ]; then L=(flock "$RENDER_LOCK"); else L=(); fi
"${L[@]}" nice -n 19 "$BLENDER" --background --threads 2 "$OUT/fly_male_lod0.blend" \
    --python "$HERE/render_closeup.py" -- lf_tarsus "$OUT/fly_male_lod0_sexcomb_closeup.png"
python3 "$HERE/measure_assets.py" --out "$OUT" --blender "$BLENDER" --write "$HERE/PROVENANCE_SEX_VARIANTS.json" > /dev/null
python3 "$HERE/contact_sheet.py" "$OUT" "$OUT/fly_sex_variants_contact_sheet.png"
echo "measured -> tools/assets/PROVENANCE_SEX_VARIANTS.json; sheet -> $OUT/fly_sex_variants_contact_sheet.png"
