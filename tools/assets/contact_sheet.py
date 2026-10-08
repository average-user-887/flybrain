#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Contact sheet of the female/male variant renders (needs Pillow).

    python3 tools/assets/contact_sheet.py OUTDIR SHEET.png

Rows: female LOD0, male LOD0, female LOD1, male LOD1.  Columns: top, side, front, 3/4.
Every tile is placed unscaled except for one common factor, so the ortho tiles keep
the shared 64 px per (female-referenced) viewport-mm and the size difference shows.
"""
import os
import sys

from PIL import Image, ImageDraw

out, sheet = sys.argv[1], sys.argv[2]
views = ['top', 'side', 'front', 'hero']
rows = [('female', 0), ('male', 0), ('female', 1), ('male', 1)]
tile, label_h, pad = 320, 22, 8
img = Image.new('RGBA', (pad + len(views) * (tile + pad), pad + len(rows) * (tile + label_h + pad) + 20), (24, 30, 44, 255))
draw = ImageDraw.Draw(img)
checker = Image.new('RGBA', (tile, tile), (40, 48, 64, 255))
cd = ImageDraw.Draw(checker)
for y in range(0, tile, 16):
    for x in range(0, tile, 16):
        if (x // 16 + y // 16) % 2:
            cd.rectangle([x, y, x + 15, y + 15], fill=(52, 61, 80, 255))
for r, (sex, lod) in enumerate(rows):
    for c, view in enumerate(views):
        x, y = pad + c * (tile + pad), pad + r * (tile + label_h + pad)
        path = os.path.join(out, f'fly_{sex}_lod{lod}_{view}.png')
        img.alpha_composite(checker, (x, y + label_h))
        if os.path.exists(path):
            im = Image.open(path).convert('RGBA').resize((tile, tile), Image.LANCZOS)
            img.alpha_composite(im, (x, y + label_h))
        label = f'{sex} LOD{lod} {view}' + (' (ortho)' if view != 'hero' else ' (persp.)')
        draw.text((x + 2, y + 4), label, fill=(226, 232, 240, 255))
draw.text((pad, img.height - 14), 'Appearance only: no brain dataset, physiology or behaviour differs.',
          fill=(251, 191, 36, 255))
img.save(sheet)
print(sheet, img.size)
