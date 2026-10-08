#!/usr/bin/env python3
"""Build the OPTIONAL presentation-asset zip from an explicit whitelist.

The whitelist is every file web/asset_manifest.json references (fly and prop LODs,
clip files per LOD, joint contracts, sprites, sheets) plus the two files the
dashboard loader fetches (hq_assets.js FLY_URL / ARENA_URL).  Nothing else in the
staging directory is archived.  Entries are sorted and carry a fixed timestamp and
mode, so the zip is byte-reproducible from the same inputs.

    python3 tools/assets/package_hq_assets.py --staged web/assets/hq --out DIST/neurofly-hq-assets-0.5.1rc1.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXED_TIME = (2026, 10, 8, 0, 0, 0)
LOADER_FILES = ("fly_hq_lod0.glb", "arena_shell.glb")
META = {
    "LICENSE": ROOT / "LICENSE",
    "NOTICE": ROOT / "NOTICE",
    "provenance/PROVENANCE.json": ROOT / "tools/assets/PROVENANCE.json",
    "provenance/PROVENANCE_SEX_VARIANTS.json": ROOT / "tools/assets/PROVENANCE_SEX_VARIANTS.json",
    "provenance/PROVENANCE_RIG_V2.json": ROOT / "tools/assets/PROVENANCE_RIG_V2.json",
    "provenance/ENV_MANIFEST.json": ROOT / "tools/assets/env/ENV_MANIFEST.json",
    "provenance/ANIM_MANIFEST.json": ROOT / "scripts/animations/ANIM_MANIFEST.json",
    "provenance/JOINT_CONTRACT.md": ROOT / "tools/assets/JOINT_CONTRACT.md",
    "provenance/SEX_VARIANTS.md": ROOT / "tools/assets/SEX_VARIANTS.md",
    "provenance/INTERFACE.md": ROOT / "tools/assets/INTERFACE.md",
}
PREFIX = "assets/hq/"


def whitelist(manifest: dict) -> list[str]:
    """Paths relative to web/assets/hq that the gallery manifest and the loader use."""
    urls = set()
    for asset in manifest.get("assets", []):
        lods = [lod.get("level", i) for i, lod in enumerate(asset.get("lods", []))]
        for lod in asset.get("lods", []):
            urls.add(lod["url"])
        for clip in asset.get("clips", []):
            for level in lods:
                urls.add(clip["url"].replace("{lod}", str(level)))
        if isinstance(asset.get("joints"), str):
            for level in lods:
                urls.add(asset["joints"].replace("{lod}", str(level)))
        for sprite in asset.get("sprites", []):
            urls.add(sprite["url"])
    for sheet in manifest.get("sheets", []):
        urls.add(sheet["url"])
    out = set(LOADER_FILES)
    for url in urls:
        if not url.startswith(PREFIX) or ".." in url or url.startswith("/"):
            raise SystemExit(f"refusing a path outside {PREFIX}: {url}")
        out.add(url[len(PREFIX):])
    return sorted(out)


INSTALL = """# NeuroFly optional presentation assets ({version})

Optional. NeuroFly runs without these files. They change only how the fly and the
arena look in the opt-in views; they never change physics, stimuli, encoders, the
brain or recordings.

## Install

From the root of a NeuroFly checkout or installed package (the directory that holds
`web/`):

    mkdir -p web/assets/hq
    unzip neurofly-hq-assets-{version}.zip -d web/assets/hq

Check the files with `cd web/assets/hq && sha256sum -c SHA256SUMS`.

## Use

* Gallery: serve `web/` (for example `cd web && python3 -m http.server 8799 --bind 127.0.0.1`)
  and open `asset_gallery.html`.
* Dashboard 3D view and embodied replay: add `?assets=hq` to the URL (off by default).

## Contents

{count} asset files listed in `SHA256SUMS`, chosen by an explicit whitelist:
every file `web/asset_manifest.json` references, plus the loader's fly and arena GLBs.
Two clip sets are kept on purpose as labelled evidence and are shown in the gallery
under those labels: `anim/w3_v2_*` (FAILED: feet through the floor, loop pops) and
`anim/v1-early-PREVIEW_NOT_FINAL/`. The repaired, final clips are `anim/w3_v3_*`.
Animations are illustrative, not simulated behaviour.

Provenance (sources, hashes, budgets, references) is in `provenance/`. Licence: MIT
(`LICENSE`); third-party notices in `NOTICE`. No textures, recordings or editable
.blend files are included.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True, type=Path, help="web/assets/hq holding the built files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--version", default="0.5.1rc1")
    args = ap.parse_args()
    manifest = json.loads((ROOT / "web/asset_manifest.json").read_text())
    files = whitelist(manifest)
    missing = [f for f in files if not (args.staged / f).is_file()]
    if missing:
        print("missing staged files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    entries = {f: (args.staged / f).read_bytes() for f in files}
    sums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(entries.items()))
    entries["SHA256SUMS"] = sums.encode()
    entries["INSTALL.md"] = INSTALL.format(version=args.version, count=len(files)).encode()
    for name, src in META.items():
        entries[name] = src.read_bytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            zf.writestr(info, entries[name], compresslevel=9)
    print(f"{args.out}: {len(files)} assets + {len(entries) - len(files)} meta; "
          f"sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
