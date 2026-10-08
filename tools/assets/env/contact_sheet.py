#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Compose one labelled contact sheet of every environment render, icon and the legend.

    python3 tools/assets/env/contact_sheet.py --out OUTDIR --icons tools/assets/env/icons

Needs Pillow.  Writes OUTDIR/env_contact_sheet.png (review image, not an asset).
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

PROPS = [('fermenting_fruit', 'fermenting fruit'), ('yeast_patch', 'yeast / food patch'),
         ('sugar_water', 'sugar water - CONTACT taste'), ('odour_emitter', 'odour emitter - VOLATILE'),
         ('wind_vane', 'wind vane - tip = downwind'), ('jumping_spider', 'jumping spider - ILLUSTRATIVE'),
         ('mantis_nymph', 'mantis nymph - ILLUSTRATIVE')]
CELL, PAD, BG = 300, 14, (78, 84, 96, 255)


def font(size, bold=False):
    for name in ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',):
        for d in ('/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/TTF'):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit(path, w, h):
    im = Image.open(path).convert('RGBA')
    im.thumbnail((w, h), Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--icons', required=True)
    a = ap.parse_args()
    views = ['top', 'side', 'hero']
    label_w = 250
    W = label_w + len(views) * (CELL + PAD) + PAD
    rows_h = len(PROPS) * (CELL + PAD)
    gallery = os.path.join(a.out, 'env_gallery_hero.png')
    legend = os.path.join(a.icons, 'env_legend.png')
    icons = sorted(f for f in os.listdir(a.icons) if f.startswith('icon_') and f.endswith('.png'))
    gal_h = int(W * 0.6) if os.path.exists(gallery) else 0
    leg_h = int((W - 2 * PAD) * 640 / 960) if os.path.exists(legend) else 0
    H = 70 + rows_h + 60 + 150 + gal_h + leg_h + 4 * PAD
    sheet = Image.new('RGBA', (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 18), 'NeuroFly W3 environment / food / predator assets - presentation only', font=font(26, True),
           fill=(240, 240, 235, 255))
    y = 70
    for key, text in PROPS:
        col = (245, 170, 40, 255) if 'ILLUSTRATIVE' in text else (235, 235, 230, 255)
        for i, line in enumerate(text.split(' - ')):
            d.text((PAD, y + 110 + i * 26), line, font=font(19, i == 0), fill=col)
        for j, v in enumerate(views):
            p = os.path.join(a.out, f'env_{key}_{v}.png')
            x = label_w + j * (CELL + PAD)
            d.rectangle((x, y, x + CELL, y + CELL), outline=(110, 116, 128, 255))
            if os.path.exists(p):
                im = fit(p, CELL, CELL)
                sheet.alpha_composite(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
                d.text((x + 6, y + 4), v, font=font(14), fill=(200, 205, 215, 255))
        y += CELL + PAD
    d.text((PAD, y + 10), 'Icons (SVG + PNG; dashed frame = ILLUSTRATIVE)', font=font(19, True),
           fill=(235, 235, 230, 255))
    y += 50
    x = PAD
    for f in icons:
        im = fit(os.path.join(a.icons, f), 110, 110)
        sheet.alpha_composite(im, (x, y))
        x += 130
    y += 150
    if gal_h:
        im = fit(gallery, W - 2 * PAD, gal_h)
        sheet.alpha_composite(im, (PAD, y))
        y += im.height + PAD
    if leg_h:
        im = fit(legend, W - 2 * PAD, leg_h)
        sheet.alpha_composite(im, (PAD, y))
        y += im.height + PAD
    sheet = sheet.crop((0, 0, W, y + PAD))
    out = os.path.join(a.out, 'env_contact_sheet.png')
    sheet.save(out)
    print('NEUROFLY_ASSET contact sheet', sheet.size)


if __name__ == '__main__':
    main()
