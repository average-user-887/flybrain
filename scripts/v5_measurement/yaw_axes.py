"""Pre-lock anatomy: the posterior (front-to-back) direction of each eye, expressed in
the v4 photoreceptor encoder's own lattice-plane frame.

Anatomy only, no simulation and no response is read. Reuses the already-declared
lattice derivation of ``brainlab.io_map_photoreceptor._lattice`` (least squares of
L1 somaLocation on hex; docs/PHOTORECEPTOR_ENCODER.md §2.2): posteriority per hex
step is ``c1*dh1 + c2*dh2``. The v4 encoder places a cartridge at
``(u, w) = M @ (h1, h2)`` with ``M = [[1, 1/2], [0, sqrt(3)/2]]`` (times 5 deg), so
posteriority is ``c . M^-1 (u, w)`` and its gradient in the encoder plane is
``M^-T c``. The grating ``cos(k*(u cos t + w sin t) - wt)`` drifts toward +theta,
so a drift direction theta_post = atan2(g_w, g_u) moves the pattern posteriorly
(front-to-back) on that eye.
"""
import json
import math

import numpy as np

from brainlab.io_map_photoreceptor import _lattice, _load_tables

nodes = _load_tables(None)
M = np.array([[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]])
out = {}
for side in ('L', 'R'):
    lat = _lattice(nodes, side)
    c = np.array([lat['c1'], lat['c2']])
    grad = np.linalg.inv(M).T @ c
    theta = math.degrees(math.atan2(grad[1], grad[0])) % 360.0
    # How horizontal is that direction really: fraction of the 3-D displacement per
    # unit step along theta_post that is anterior-posterior (z).
    step_uw = np.array([math.cos(math.radians(theta)), math.sin(math.radians(theta))])
    dh = np.linalg.inv(M) @ step_uw
    disp = dh[0] * np.array(lat['lattice_vector_hex1_nm']) + dh[1] * np.array(lat['lattice_vector_hex2_nm'])
    out[side] = dict(c1=lat['c1'], c2=lat['c2'], theta_front_to_back_deg=theta,
                     displacement_nm_per_unit_step=[float(x) for x in disp],
                     fraction_xyz=[float(abs(x) / np.linalg.norm(disp)) for x in disp])
out['rule'] = __doc__.strip()
print(json.dumps(out, indent=2))
with open('docs/receipts/v5_raw/yaw_axes.json', 'w') as fh:
    json.dump(out, fh, indent=2)
