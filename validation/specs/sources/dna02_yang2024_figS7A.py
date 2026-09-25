"""Read the DNa02 per-fly spike rates from Yang et al. 2024 (Cell) Figure S7A.

The supplementary PDF (NIHMS2118063-supplement-Supplementary_Figure_7.pdf, in
the Europe PMC supplementary files of PMC12778575) is vector graphics, so the
dot positions are exact numbers in the file.  The left panel is DNa02 (legend:
"n=4 cells in 4 flies for DNa02"); green dots = walking, purple = not walking.
The y axis is calibrated on its three tick marks (10, 40, 70 Hz).

    python3 dna02_yang2024_figS7A.py NIHMS2118063-supplement-Supplementary_Figure_7.pdf
"""
import json
import sys

import pdfplumber

GREEN, PURPLE = (0.192, 0.745, 0.188), (0.471, 0.353, 0.749)


def main(path):
    page = pdfplumber.open(path).pages[0]
    # DNa02 panel: y axis at x = 34.2, ticks are short horizontal lines ending there.
    ticks = sorted(l['top'] for l in page.lines
                   if abs(l['x1'] - 34.2) < 0.2 and l['top'] == l['bottom'] and l['top'] < 135)
    assert len(ticks) == 3, ticks            # 70, 40, 10 Hz from top to bottom
    y70, y40, y10 = ticks
    hz_per_pt = 60.0 / (y10 - y70)
    to_hz = lambda y: 10.0 + (y10 - y) * hz_per_pt
    pts = {'walking': set(), 'not_walking': set()}
    for c in page.curves:
        if c['x0'] > 100 or c['top'] > 135 or c['x1'] - c['x0'] > 1:
            continue
        colour = tuple(round(v, 3) for v in c.get('non_stroking_color') or ())
        key = 'walking' if colour == GREEN else 'not_walking' if colour == PURPLE else None
        if key:
            pts[key].add(round((c['top'] + c['bottom']) / 2, 2))
    out = {k: sorted(round(to_hz(y), 1) for y in v) for k, v in pts.items()}
    out['tick_check_hz_at_40_tick'] = round(to_hz(y40), 2)
    out['group_mean_hz'] = {k: round(sum(v) / len(v), 1) for k, v in out.items() if isinstance(v, list)}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
