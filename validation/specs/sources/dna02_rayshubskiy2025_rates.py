"""DNa02 spike rates from the raw whole-cell recordings of Rayshubskiy et al. 2025.

Data: Harvard Dataverse doi:10.7910/DVN/0NCLP1 (v1.2), folders ephys_data_a2_s_NN
(single DNa02 recordings, tethered fly walking on a ball in the dark).
Spike detection: the authors' own code, github.com/wilson-lab/
rayshubskiy_elife_102230_secondary_analysis_code, a2lib/spike_tools.py, with the
DNa02 settings of import_preprocess_data.ipynb (prominence 15, 10 s window).
Rates are spike counts in 1 s bins; seconds with the optogenetic stimulus on
are excluded.  "still" = the 10% of seconds with the least ball movement.

    python3 dna02_rayshubskiy2025_rates.py <path to secondary-analysis repo> a2_s_01=file.mat ...
"""
import json
import sys

import numpy as np
import scipy.io as sio


def rates(fn, spike_tools, prominence=15.0):
    m = sio.loadmat(fn, squeeze_me=True)
    fs, bfs = int(m['ephys_SR']), int(m['ball_SR'])
    n = (m['ephys_A'].size // fs) * fs
    v, stim = m['ephys_A'][:n].astype(float), m['stim'][:n].astype(float)
    proc = spike_tools.process_voltage_spike_detection(v, fs)
    t = np.flatnonzero(spike_tools.detect_spikes(proc, method='findpeaks', window_len=int(10 * fs),
                                                 prominence=prominence)) / fs
    T = n // fs
    c1 = np.histogram(t, bins=np.arange(T + 1))[0]
    ok = np.array([stim[i * fs:(i + 1) * fs].max() == 0 for i in range(T)])
    move = np.array([sum(np.mean(np.abs(m[k][i * bfs:(i + 1) * bfs])) for k in ('fwd', 'yaw', 'lat'))
                     for i in range(T)])
    still = ok & (move <= np.quantile(move[ok], 0.1))
    return dict(duration_s=int(T), spikes=int(t.size), mean_hz=round(float(c1[ok].mean()), 1),
                still_10pct_mean_hz=round(float(c1[still].mean()), 1),
                p95_1s_hz=float(np.percentile(c1[ok], 95)), max_1s_hz=int(c1[ok].max()))


def main(repo, pairs):
    sys.path.insert(0, repo)
    np.int, np.float = int, float                    # the authors' code uses the old aliases
    from a2lib import spike_tools
    out = {name: rates(fn, spike_tools) for name, fn in (p.split('=', 1) for p in pairs)}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
