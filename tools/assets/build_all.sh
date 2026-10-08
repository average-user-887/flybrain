#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
#
# Build every presentation asset headless, measure it, and optionally stage it
# for web/asset_preview.html.  CPU only, two threads, low priority.
#
#   BLENDER=/path/to/blender tools/assets/build_all.sh OUTDIR [--stage]
#
# OUTDIR must be outside the repository.  --stage copies the .glb/.png files
# into web/assets/hq/ (git-ignored) so the preview page and ?assets=hq find them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:?usage: BLENDER=... build_all.sh OUTDIR [--stage]}"
BLENDER="${BLENDER:-blender}"
mkdir -p "$OUT"
run() { nice -n 19 "$BLENDER" --background --factory-startup --threads 2 --python "$@"; }
run "$HERE/build_fly.py" -- --out "$OUT" --lod 0 --render 1
run "$HERE/build_fly.py" -- --out "$OUT" --lod 1
run "$HERE/build_arena.py" -- --out "$OUT" --render 1 --fly "$OUT/fly_hq_lod0.glb"
python3 "$HERE/measure_assets.py" --out "$OUT" --blender "$BLENDER" --write "$HERE/PROVENANCE.json" > /dev/null
echo "measured -> tools/assets/PROVENANCE.json"
if [ "${2:-}" = "--stage" ]; then
    mkdir -p "$REPO/web/assets/hq"
    cp "$OUT"/*.glb "$OUT"/*.png "$REPO/web/assets/hq/"
    echo "staged -> web/assets/hq/"
fi
