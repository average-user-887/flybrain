"""S3 guard: lattice-specific encoder fields may differ across arms; everything else is checked.
The real-file test uses the two completed v5 GPU comparators when NEUROFLY_S3_DIR points at them."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

GUARD = Path(__file__).resolve().parents[1] / 'scripts/v6/s3_guard.py'
PRE = '9658283e34b1c846b07c7631fc7853b4f06f7ddfbe0e38d329f9c243f99f946e'
COND = ['dir0', 'dir45', 'dir90', 'dir135', 'dir180', 'dir225', 'dir270', 'dir315', 'yaw_ccw', 'yaw_cw']


def _fake(d, tag, lattice, drop=None, enc_extra=None):
    enc = dict(i_max=20.0, spatial_period_deg=30.0)
    if lattice == 'malecns-hex-v2':
        enc.update(lattice=lattice, lattice_matrix=[[1, -0.5], [0, 0.866]], correction='post-finding')
    enc.update(enc_extra or {})
    conds = [c for c in COND if c != drop]
    meta = dict(prereg_sha256=PRE, lattice=lattice, encoder=enc, protocol=dict(step_ms=2.0),
                conditions_done=conds)
    z = {'trace_idx': np.arange(5)}
    for c in ['gray'] + conds:
        for f in ('f1_im', 'f1_re', 'mean_v', 'spikes'):
            z[f'{c}__{f}'] = np.zeros(5)
        z[f'{c}__pop_v'] = np.zeros((3, 2)); z[f'{c}__watch_v'] = np.zeros((3, 4))
    np.savez(d / f'{tag}.npz', **z)
    (d / f'{tag}.meta.json').write_text(json.dumps(meta))


def _guard(d, arms):
    return subprocess.run([sys.executable, str(GUARD), str(d), '--arms', ','.join(arms)], capture_output=True)


def test_lattice_fields_may_differ_but_shared_fields_may_not(tmp_path):
    _fake(tmp_path, 'v5_gpu_axial-v1', 'axial-v1'); _fake(tmp_path, 'v5_gpu_malecns-hex-v2', 'malecns-hex-v2')
    assert _guard(tmp_path, ['v5_gpu_axial-v1', 'v5_gpu_malecns-hex-v2']).returncode == 0
    _fake(tmp_path, 'v5_gpu_malecns-hex-v2', 'malecns-hex-v2', enc_extra=dict(i_max=10.0))
    assert _guard(tmp_path, ['v5_gpu_axial-v1', 'v5_gpu_malecns-hex-v2']).returncode == 2


def test_missing_condition_is_not_evaluable(tmp_path):
    _fake(tmp_path, 'v5_gpu_axial-v1', 'axial-v1', drop='yaw_cw')
    assert _guard(tmp_path, ['v5_gpu_axial-v1']).returncode == 2


@pytest.mark.skipif(not os.environ.get('NEUROFLY_S3_DIR'), reason='needs the completed v5 GPU comparator files')
def test_completed_v5_gpu_comparators_pass_the_guard():
    r = _guard(Path(os.environ['NEUROFLY_S3_DIR']), ['v5_gpu_axial-v1', 'v5_gpu_malecns-hex-v2'])
    assert r.returncode == 0, r.stdout.decode()
