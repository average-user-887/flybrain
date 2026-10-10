"""v7 data pin: digitise Behnia et al. 2014 (Nature 512:427, doi 10.1038/nature13427) figures.

No raw traces or source data were released with the paper (PMC author manuscript PMC4243710:
'has-supplement no'; Europe PMC hasSuppl N), so the numbers are digitised from the publisher's
figure images.  The images are NOT redistributed; this script reproduces the digitised numbers
from images whose sha256 is pinned in qualification/v7/V7_data_pin.json.

TRAINING (Fig. 3b, 3e): average linear temporal filters (mV contrast^-1 ms^-1, +/- s.e.m. band)
of Mi1 (N=7), Tm3 (N=11), Tm1 (N=15), Tm2 (N=14), from in vivo whole-cell current clamp during a
10 s full-field Gaussian flicker (50 % contrast s.d., exponential correlation time 10 ms, 240 Hz).

HELD-OUT (Fig. 2, 1 s flash column): onset and offset peak deflections (mV) of the same four types
for a 1 s full-field light flash from dark.  Written to a SEPARATE file; the fitting harness
never reads it (tests/test_v7_medulla.py enforces this).

Method (documented, deterministic):
  * axis calibration from the figure's own ticks and scale bars, located by pixel search and
    frozen below as constants (pixel coordinates in the 1052x1221 / 1051x1407 publisher JPEGs);
  * curve centre = mean row of pixels within RGB distance LINE_TOL of the trace colour, per
    image column; columns where another trace occludes the line are linearly interpolated and
    flagged;
  * s.e.m. band = contiguous run of band-tint or line pixels around the centre, per column;
  * lag 0 = the panel's y-axis line (validated: digitised peak times 70.5/55.8/53.8/43.6 ms vs
    the paper's text means 71+/-3.8, 53+/-5.2, 56+/-3.8, 43+/-2.7 ms);
  * resampled to a 1 ms grid by linear interpolation.
Uncertainty: y quantisation 1 px = 0.0042 mV contrast^-1 ms^-1; time 50 ms = 39+/-1 px
(+/-2.6 % scale), lag-0 +/-1.5 px (+/-1.9 ms); Fig. 2 amplitudes +/-1 px = +/-0.23 mV
(panel a) / 0.20 mV (panel b) plus +/-1 px on the 10 mV bar (+/-2.3 %).

  python scripts/v7/digitise_behnia2014.py --fig3 FIG3.jpg --fig2 FIG2.jpg --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

LINE = {'Mi1': (170, 85, 140), 'Tm3': (85, 168, 115), 'Tm1': (40, 85, 150), 'Tm2': (200, 50, 50)}
BAND = {'Mi1': (228, 180, 210), 'Tm3': (175, 210, 170), 'Tm1': (165, 178, 215), 'Tm2': (240, 170, 170)}
LINE_TOL = 45.0
BAND_TOL = 40.0

# Fig. 3 calibration (pixels).  y rows of the 0.0 and +/-0.5 ticks; x of the y-axis line (lag 0);
# 50 ms scale bar width (inclusive pixel run 771-809 in b, 767-805 in e).
FIG3 = {
    'b': dict(types=('Mi1', 'Tm3'), x_axis=626.5, y_zero=231.5, y_half=112.5, half=0.5,
              x_range=(627, 822), y_range=(60, 245), px_per_50ms=39.0),
    'e': dict(types=('Tm1', 'Tm2'), x_axis=630.0, y_zero=711.5, y_half=830.5, half=-0.5,
              x_range=(631, 822), y_range=(700, 905), px_per_50ms=39.0),
}
TEXT_PEAK_MS = {'Mi1': (71.0, 3.8, 7), 'Tm3': (53.0, 5.2, 11), 'Tm1': (56.0, 3.8, 15), 'Tm2': (43.0, 2.7, 14)}

# Fig. 2, 1 s flash column.  Light on at x 822, off at x 855.5 (intensity trace); 1 s = 33 px.
# 10 mV bars: panel a x 992 y 5-49 (44 px), panel b x 996 y 735-784 (49 px).
FIG2 = {
    'Mi1': dict(y_range=(20, 205), mv_px=10.0 / 44.0),
    'Tm3': dict(y_range=(215, 420), mv_px=10.0 / 44.0),
    'Tm1': dict(y_range=(750, 935), mv_px=10.0 / 49.0),
    'Tm2': dict(y_range=(945, 1125), mv_px=10.0 / 49.0),
}
FIG2_X = dict(base=(760, 818), on=822, off=856, offset_end=880, px_per_s=33.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB')).astype(float)


def _dist(img, rgb):
    return np.sqrt(((img - np.asarray(rgb, float)) ** 2).sum(-1))


def trace_columns(img, line_rgb, band_rgb, x_range, y_range, other_band=None, max_gap=3):
    """Per image column: (centre_row or nan, band_top, band_bottom).  Rows are image rows.

    Band pixels: light, saturated pixels closer to this trace's band tint than to the other
    trace's (``other_band``), or the line itself; the band is the run around the centre,
    bridging gaps of up to ``max_gap`` rows (dashed markers, the other trace's line)."""
    x0, x1 = x_range; y0, y1 = y_range
    sub = img[y0:y1, x0:x1]
    line = _dist(sub, line_rgb) < LINE_TOL
    sat = sub.max(-1) - sub.min(-1)
    own = _dist(sub, band_rgb)
    tint = (own < BAND_TOL * 2) & (sat > 20) & (sub.min(-1) > 110)
    if other_band is not None:
        tint &= own < _dist(sub, other_band)
    tint |= line
    out = []
    for k in range(x1 - x0):
        rows = np.flatnonzero(line[:, k])
        if len(rows) == 0:
            out.append((np.nan, np.nan, np.nan)); continue
        c = rows.mean()
        lo = hi = int(round(c))
        while True:
            nxt = [lo - g for g in range(1, max_gap + 2) if lo - g >= 0 and tint[lo - g, k]]
            if not nxt:
                break
            lo = nxt[0]
        while True:
            nxt = [hi + g for g in range(1, max_gap + 2) if hi + g < tint.shape[0] and tint[hi + g, k]]
            if not nxt:
                break
            hi = nxt[0]
        out.append((c + y0, lo + y0, hi + y0))
    return np.arange(x0, x1), np.array(out)


def _interp_nan(x, y):
    bad = np.isnan(y)
    if bad.all():
        raise ValueError('no trace pixels found')
    y = y.copy(); y[bad] = np.interp(x[bad], x[~bad], y[~bad])
    return y, bad


def digitise_filter(img, panel: dict, cell: str, t_max_ms: int = 240):
    other = [t for t in panel['types'] if t != cell][0]
    xs, cols = trace_columns(img, LINE[cell], BAND[cell], panel['x_range'], panel['y_range'], BAND[other])
    have = ~np.isnan(cols[:, 0])
    first, last = np.flatnonzero(have)[[0, -1]]
    xs, cols = xs[first:last + 1], cols[first:last + 1]
    centre, flag = _interp_nan(xs.astype(float), cols[:, 0])
    top, _ = _interp_nan(xs.astype(float), cols[:, 1])
    bot, _ = _interp_nan(xs.astype(float), cols[:, 2])
    unit_px = (panel['y_zero'] - panel['y_half']) / panel['half']      # px per unit (signed)
    val = (panel['y_zero'] - centre) / unit_px
    v_top = (panel['y_zero'] - top) / unit_px; v_bot = (panel['y_zero'] - bot) / unit_px
    half_w = np.abs(v_top - v_bot) / 2.0
    # a band narrower than the line itself (occluded or blended) falls back to the trace median
    med = float(np.median(half_w[~flag])) if (~flag).any() else 0.0
    half_w = np.where(flag | (half_w <= 0), med, half_w)
    t = (xs - panel['x_axis']) * 50.0 / panel['px_per_50ms']
    grid = np.arange(0, t_max_ms + 1, dtype=float)
    keep = grid <= t[-1]
    grid = grid[keep]
    lead = grid < t[0]                       # before the first coloured pixel: the trace starts at 0
    tt = np.concatenate([[0.0], t]) if t[0] > 0 else t
    vv = np.concatenate([[0.0], val]) if t[0] > 0 else val
    hh = np.concatenate([[med], half_w]) if t[0] > 0 else half_w
    ff = np.concatenate([[True], flag]) if t[0] > 0 else flag
    mean = np.interp(grid, tt, vv)
    sem = np.interp(grid, tt, hh)
    fl = np.interp(grid, tt, ff.astype(float)) > 0
    fl |= lead
    return dict(t_ms=grid, mean=mean, sem=sem, flag=fl)


def peak_time_ms(d, cell):
    sign = 1.0 if cell in ('Mi1', 'Tm3') else -1.0
    return float(d['t_ms'][int(np.argmax(sign * d['mean']))])


def digitise_fig2(img, cell):
    p = FIG2[cell]
    xs, cols = trace_columns(img, LINE[cell], BAND[cell], (FIG2_X['base'][0], FIG2_X['offset_end'] + 1),
                             p['y_range'])
    y, _ = _interp_nan(xs.astype(float), cols[:, 0])
    mv = -(y - np.median(y[(xs >= FIG2_X['base'][0]) & (xs <= FIG2_X['base'][1])])) * p['mv_px']
    on = (xs > FIG2_X['on']) & (xs <= FIG2_X['off'])
    off = (xs > FIG2_X['off']) & (xs <= FIG2_X['offset_end'])
    if cell in ('Mi1', 'Tm3'):        # ON cells: depolarise at onset, hyperpolarise at offset
        onset, offset = float(mv[on].max()), float(mv[off].min())
    else:                             # OFF cells: hyperpolarise at onset, depolarise at offset
        onset, offset = float(mv[on].min()), float(mv[off].max())
    return dict(onset_peak_mV=onset, offset_peak_mV=offset,
                amplitude_uncertainty_mV=round(p['mv_px'] * 1.0 + 0.023 * max(abs(onset), abs(offset)), 3))


def write_filter_csv(path, d):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_ms', 'filter_mV_per_contrast_per_ms', 'sem', 'interpolated'])
        for t, m, s, fl in zip(d['t_ms'], d['mean'], d['sem'], d['flag']):
            w.writerow([f'{t:.0f}', f'{m:.5f}', f'{s:.5f}', int(bool(fl))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig3', type=Path, required=True)
    ap.add_argument('--fig2', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    (a.out / 'training').mkdir(parents=True, exist_ok=True)
    (a.out / 'heldout').mkdir(parents=True, exist_ok=True)
    img3 = load_rgb(a.fig3)
    report = dict(fig3_sha256=sha256(a.fig3), fig2_sha256=sha256(a.fig2), training={}, heldout={})
    for panel in FIG3.values():
        for cell in panel['types']:
            d = digitise_filter(img3, panel, cell)
            path = a.out / 'training' / f'behnia2014_fig3_filter_{cell}.csv'
            write_filter_csv(path, d)
            pk = peak_time_ms(d, cell)
            m, sem, n = TEXT_PEAK_MS[cell]
            report['training'][cell] = dict(file=path.name, sha256=sha256(path), n_ms=int(len(d['t_ms'])),
                                            interpolated_ms=int(d['flag'].sum()), peak_ms=pk,
                                            peak_abs=float(np.abs(d['mean']).max()),
                                            text_peak_ms=m, text_sem_ms=sem, n_cells=n,
                                            peak_check=bool(abs(pk - m) <= 2 * sem + 3.0))
    img2 = load_rgb(a.fig2)
    held = {cell: digitise_fig2(img2, cell) for cell in FIG2}
    hp = a.out / 'heldout' / 'behnia2014_fig2_flash1s.json'
    hp.write_text(json.dumps(held, indent=1, sort_keys=True))
    report['heldout'] = dict(file=hp.name, sha256=sha256(hp), values=held)
    (a.out / 'digitise_report.json').write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
