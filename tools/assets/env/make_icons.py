#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Project NeuroFly contributors
"""Write the environment icons and the sense legend as SVG, then PNG via Inkscape.

    python3 tools/assets/env/make_icons.py --out tools/assets/env/icons [--inkscape inkscape]

Standard library only (Inkscape is used for the PNG export when it is on PATH).
Every path is drawn here; no third-party artwork.

Shape carries the meaning and colour only supports it:
  * sugar taste / contact  -> filled droplet standing on a SOLID contact bar, solid ring
  * volatile odour         -> source dot with DASHED rings and wavy rising lines
  * wind                   -> arrow (tip = where the air goes) with streamlines
  * predators              -> DASHED tile frame = ILLUSTRATIVE (no stimulus or behaviour)
"""
import argparse
import math
import os
import shutil
import subprocess

SUGAR, ODOUR, WIND, FIN = '#1f6fd1', '#7a2bc2', '#0b7f88', '#e8870f'
INK, TILE, FOOD = '#26231f', '#f5f3ee', '#8a5a1c'
FONT = "font-family='DejaVu Sans, Verdana, sans-serif'"


def tile(body, dashed=False, title=''):
    dash = " stroke-dasharray='5 3.2'" if dashed else ''
    return (f"<title>{title}</title>"
            f"<rect x='2' y='2' width='60' height='60' rx='9' fill='{TILE}' stroke='{INK}' stroke-width='2'{dash}/>"
            + body)


def droplet(cx, cy, s, fill):
    return (f"<path d='M{cx} {cy - 13 * s} C{cx + 2 * s} {cy - 8 * s} {cx + 9 * s} {cy - 1 * s} {cx + 9 * s} {cy + 5 * s} "
            f"A{9 * s} {9 * s} 0 0 1 {cx - 9 * s} {cy + 5 * s} C{cx - 9 * s} {cy - 1 * s} {cx - 2 * s} {cy - 8 * s} "
            f"{cx} {cy - 13 * s} Z' fill='{fill}' stroke='{INK}' stroke-width='1.6'/>")


def icon_sugar():
    return tile(droplet(32, 30, 1.25, SUGAR)
                + f"<path d='M26 30 q3 -6 6 -9' stroke='#ffffff' stroke-width='2.2' fill='none' stroke-linecap='round'/>"
                + f"<rect x='12' y='50' width='40' height='5' rx='1.5' fill='{SUGAR}' stroke='{INK}' stroke-width='1.4'/>"
                + f"<path d='M8 40 l8 9 M16 49 l-1 -6 M16 49 l-6 0' stroke='{INK}' stroke-width='2' fill='none' "
                  "stroke-linecap='round'/>",
                title='Sugar water: taste on contact')


def icon_odour():
    arcs = ''.join(f"<circle cx='32' cy='44' r='{r}' fill='none' stroke='{ODOUR}' stroke-width='2.4' "
                   f"stroke-dasharray='4 3.5'/>" for r in (9, 16))
    waves = ''.join(f"<path d='M{x} 30 c-4 -4 4 -8 0 -12 c-4 -4 4 -8 0 -11' fill='none' stroke='{ODOUR}' "
                    "stroke-width='2.2' stroke-linecap='round'/>" for x in (24, 32, 40))
    return tile(arcs + waves + f"<circle cx='32' cy='44' r='4.5' fill='{ODOUR}' stroke='{INK}' stroke-width='1.4'/>",
                title='Odour source: volatile, detected at a distance')


def icon_wind():
    lines = ''.join(f"<path d='M8 {y} h{w}' stroke='{WIND}' stroke-width='2' stroke-linecap='round' "
                    "stroke-dasharray='6 4'/>" for y, w in ((18, 30), (46, 26)))
    return tile(lines + f"<path d='M10 32 H42' stroke='{WIND}' stroke-width='5' stroke-linecap='round'/>"
                f"<path d='M40 22 L56 32 L40 42 Z' fill='{WIND}' stroke='{INK}' stroke-width='1.5'/>"
                f"<path d='M8 25 L16 32 L8 39 Z' fill='{FIN}' stroke='{INK}' stroke-width='1.3'/>",
                title='Wind: arrow tip points where the air goes')


def icon_fruit():
    pent = ' '.join(f"{32 + 22 * math.cos(a):.1f},{32 + 22 * math.sin(a):.1f}"
                    for a in [(-90 + 72 * i) * 3.14159265 / 180 for i in range(5)])
    seeds = ''.join(f"<ellipse cx='{32 + 4 * c:.1f}' cy='{32 + 4 * s:.1f}' rx='3' ry='1.6' fill='{INK}' "
                    f"transform='rotate({d} {32 + 4 * c:.1f} {32 + 4 * s:.1f})'/>"
                    for c, s, d in ((0, -1, 0), (0.866, 0.5, 60), (-0.866, 0.5, -60)))
    return tile(f"<polygon points='{pent}' fill='#d9a91a' stroke='{INK}' stroke-width='1.8' stroke-linejoin='round'/>"
                f"<circle cx='32' cy='32' r='16.5' fill='#eedca5' stroke='#a07a30' stroke-width='1'/>"
                f"<path d='M20 38 a13 13 0 0 0 12 9 a9 9 0 0 1 -12 -9 Z' fill='{FOOD}' opacity='0.85'/>"
                f"<path d='M42 22 a13 13 0 0 1 4 9 a8 8 0 0 0 -4 -9 Z' fill='{FOOD}' opacity='0.85'/>"
                + seeds
                + ''.join(f"<circle cx='{x}' cy='{y}' r='{r}' fill='#ffffff' stroke='{FOOD}' stroke-width='1'/>"
                          for x, y, r in ((44, 40, 2.2), (39, 45, 1.5), (22, 25, 1.6))),
                title='Fermenting fruit')


def icon_yeast():
    dots = ''.join(f"<circle cx='{x}' cy='{y}' r='{r}' fill='#f3e3b8' stroke='#8a6d36' stroke-width='1'/>"
                   for x, y, r in ((22, 33, 3.2), (30, 29, 3.8), (39, 31, 3.2), (45, 36, 2.6), (26, 39, 2.4),
                                   (35, 37, 3.0), (18, 40, 2.0), (42, 41, 2.2)))
    return tile(f"<ellipse cx='32' cy='46' rx='26' ry='7' fill='{FOOD}' stroke='{INK}' stroke-width='1.6'/>"
                f"<path d='M12 44 C14 26 50 22 52 44 Z' fill='#d8c08a' stroke='{INK}' stroke-width='1.6'/>" + dots,
                title='Yeast / food patch')


def icon_spider():
    legs = ''
    for sx in (-1, 1):
        for (ax, ay, kx, ky, fx, fy) in ((5, -6, 15, -16, 18, -26), (6, -3, 18, -8, 26, -12),
                                         (6, 0, 18, 4, 26, 10), (5, 3, 14, 12, 20, 24)):
            legs += (f"<path d='M{32 + sx * ax} {30 + ay} L{32 + sx * kx} {30 + ky} L{32 + sx * fx} {30 + fy}' "
                     f"fill='none' stroke='{INK}' stroke-width='2.2' stroke-linejoin='round' stroke-linecap='round'/>")
    body = (f"<ellipse cx='32' cy='26' rx='8' ry='9' fill='{INK}'/>"
            f"<ellipse cx='32' cy='44' rx='8.5' ry='11' fill='{INK}'/>"
            + ''.join(f"<path d='M24.5 {y} h15' stroke='#ffffff' stroke-width='2'/>" for y in (40, 46))
            + f"<circle cx='29' cy='19.5' r='2.6' fill='#ffffff'/><circle cx='35' cy='19.5' r='2.6' fill='#ffffff'/>"
            + f"<circle cx='29' cy='19.5' r='1.4' fill='{INK}'/><circle cx='35' cy='19.5' r='1.4' fill='{INK}'/>")
    return tile(legs + body, dashed=True, title='Jumping spider (Salticidae): ILLUSTRATIVE')


def icon_mantis():
    g = INK
    return tile(f"<path d='M7 30 C9 40 16 46 26 45 L27 41 C19 41 13 37 11 29 Z' fill='{g}'/>"   # upturned abdomen
                f"<path d='M25 44 L30 40 L42 21' fill='none' stroke='{g}' stroke-width='3.4' "
                "stroke-linecap='round' stroke-linejoin='round'/>"                                 # thorax + prothorax
                f"<path d='M39 18 L52 15 L45 25 Z' fill='{g}' stroke='{g}' stroke-width='1.5' stroke-linejoin='round'/>"
                f"<circle cx='48.5' cy='16.8' r='1.6' fill='{TILE}'/>"
                f"<path d='M49 15 Q55 7 60 6 M47 15 Q51 8 55 5' fill='none' stroke='{g}' stroke-width='1'/>"
                f"<path d='M39 25 L40 35 L50 27 L47 34' fill='none' stroke='{g}' stroke-width='2.8' "
                "stroke-linejoin='round' stroke-linecap='round'/>"                                 # raptorial foreleg
                f"<path d='M41.5 32 l-1.5 2.5 M44 30 l-1.5 2.5 M46.5 28.5 l-1.5 2.5' stroke='{g}' stroke-width='1.2'/>"
                f"<path d='M29 41 L24 50 L20 57 M29 41 L36 49 L40 57 M26 43 L14 49 L8 57' fill='none' stroke='{g}' "
                "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/>",
                dashed=True, title='Praying mantis nymph (Mantidae): ILLUSTRATIVE')


ICONS = {'icon_sugar_contact': icon_sugar, 'icon_odour_volatile': icon_odour, 'icon_wind': icon_wind,
         'icon_fermenting_fruit': icon_fruit, 'icon_yeast_patch': icon_yeast,
         'icon_jumping_spider_illustrative': icon_spider, 'icon_mantis_illustrative': icon_mantis}


def svg(body, w=64, h=64):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<!-- SPDX-License-Identifier: MIT; Project NeuroFly; generated by tools/assets/env/make_icons.py -->"
            f"{body}</svg>\n")


def legend():
    rows = [
        ('icon_sugar_contact', 'Sugar taste  -  CONTACT',
         ['Gustatory. Detected only when the tarsi or proboscis TOUCH the solution.',
          'Solid droplet on a solid bar; a solid ring marks the contact zone. No range in air.']),
        ('icon_odour_volatile', 'Volatile odour  -  AIRBORNE, AT A DISTANCE',
         ['Olfactory. Molecules travel through air and reach the antennae.',
          'Dashed rings mark the SOURCE POSITION only, never a measured concentration or plume.']),
        ('icon_wind', 'Wind direction',
         ['Arrow tip points where the air moves TO (downwind); the orange fin trails upwind.',
          'If the engine has no wind field the display must say NOT SIMULATED.']),
        ('icon_fermenting_fruit', 'Food sources: fermenting fruit, yeast patch',
         ['In nature a food can be both an odour source and a taste contact.',
          'The picture does not say which is simulated; see the sensory capability table.']),
        ('icon_jumping_spider_illustrative', 'Predators  -  ILLUSTRATIVE (dashed frame)',
         ['Jumping spider and mantis nymph are decoration only: no predator stimulus or behaviour',
          'is implemented, and they do not change looming or retinal input.']),
    ]
    W, H, top, step = 960, 640, 92, 102
    parts = [f"<rect x='0' y='0' width='{W}' height='{H}' rx='14' fill='#fbfaf7' stroke='{INK}' stroke-width='2'/>",
             f"<text x='32' y='50' {FONT} font-size='28' font-weight='bold' fill='{INK}'>"
             "NeuroFly environment legend: taste vs odour</text>"]
    for i, (key, head, lines) in enumerate(rows):
        y = top + i * step
        parts.append(f"<g transform='translate(32 {y}) scale(1.25)'>{ICONS[key]()}</g>")
        if key == 'icon_fermenting_fruit':
            parts.append(f"<g transform='translate(116 {y}) scale(1.25)'>{icon_yeast()}</g>")
        if key == 'icon_jumping_spider_illustrative':
            parts.append(f"<g transform='translate(116 {y}) scale(1.25)'>{icon_mantis()}</g>")
        x = 212
        parts.append(f"<text x='{x}' y='{y + 24}' {FONT} font-size='21' font-weight='bold' fill='{INK}'>{head}</text>")
        for j, line in enumerate(lines):
            parts.append(f"<text x='{x}' y='{y + 50 + j * 22}' {FONT} font-size='16' fill='{INK}'>{line}</text>")
    parts.append(f"<text x='32' y='{H - 22}' {FONT} font-size='14' fill='#55504a'>Colour is supplementary: "
                 "each sense also has its own shape (solid bar and ring vs dashed rings and wavy lines). "
                 "Presentation only.</text>")
    return svg(''.join(parts), W, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--inkscape', default=shutil.which('inkscape') or '')
    ap.add_argument('--png-px', type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = {name + '.svg': svg(fn()) for name, fn in ICONS.items()}
    files['env_legend.svg'] = legend()
    for name, text in files.items():
        with open(os.path.join(args.out, name), 'w') as fh:
            fh.write(text)
    if args.inkscape:
        for name in files:
            src = os.path.join(args.out, name)
            dst = src[:-4] + '.png'
            width = 960 if name == 'env_legend.svg' else args.png_px
            subprocess.run([args.inkscape, src, '--export-type=png', f'--export-filename={dst}',
                            f'--export-width={width}'], check=True, capture_output=True)
    print('NEUROFLY_ASSET icons', len(files), 'svg', 'png' if args.inkscape else '(no inkscape: svg only)')


if __name__ == '__main__':
    main()
