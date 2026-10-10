"""v7 S3 guard (prereg v2): the v7 arm keeps BOTH the params-sha check and the pre-merge
part-consistency checks.  fail-old / pass-new: run with V7_GUARD_UNDER_TEST pointing at the
pre-patch guard to see these fail."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

GUARD = Path(os.environ.get('V7_GUARD_UNDER_TEST') or Path(__file__).resolve().parents[1] / 'scripts/v6/s3_guard.py')
PRE = '9658283e34b1c846b07c7631fc7853b4f06f7ddfbe0e38d329f9c243f99f946e'
V7 = 'a' * 64
A = ['dir0', 'dir45', 'dir90', 'dir135', 'yaw_ccw']
B = ['dir180', 'dir225', 'dir270', 'dir315', 'yaw_cw']


def _part(d, tag, conds, params=V7, engine='v6'):
    meta = dict(prereg_sha256=PRE, params_sha256=params, lattice='axial-v1', encoder=dict(i_max=20.0),
                protocol=dict(step_ms=2.0), conditions_done=conds, engine=engine)
    z = {'trace_idx': np.arange(5)}
    for c in ['gray'] + conds:
        for f in ('f1_im', 'f1_re', 'mean_v', 'spikes'):
            z[f'{c}__{f}'] = np.zeros(5)
        z[f'{c}__pop_v'] = np.zeros((3, 2)); z[f'{c}__watch_v'] = np.zeros((3, 4))
    np.savez(d / f'{tag}.npz', **z)
    (d / f'{tag}.meta.json').write_text(json.dumps(meta))


def _guard(d, *args):
    return subprocess.run([sys.executable, str(GUARD), str(d), *args], capture_output=True)


def test_v7_split_parts_are_checked_before_merge(tmp_path):
    _part(tmp_path, 'v7_axial-v1_A', A); _part(tmp_path, 'v7_axial-v1_B', B)
    r = _guard(tmp_path, '--params-sha256', V7, '--split', 'v7_axial-v1')
    assert r.returncode == 0, r.stdout.decode()


def test_v7_parts_with_a_foreign_params_sha_are_refused(tmp_path):
    _part(tmp_path, 'v7_axial-v1_A', A); _part(tmp_path, 'v7_axial-v1_B', B, params='b' * 64)
    r = _guard(tmp_path, '--params-sha256', V7, '--split', 'v7_axial-v1')
    assert r.returncode == 2 and b'params sha' in r.stdout


def test_v7_merged_arm_with_a_foreign_params_sha_is_refused(tmp_path):
    _part(tmp_path, 'v7_axial-v1', A + B, params='b' * 64)
    r = _guard(tmp_path, '--params-sha256', V7, '--arms', 'v7_axial-v1')
    assert r.returncode == 2 and b'params sha' in r.stdout


def test_v7_parts_that_disagree_are_refused(tmp_path):
    _part(tmp_path, 'v7_axial-v1_A', A); _part(tmp_path, 'v7_axial-v1_B', B, engine='v5')
    r = _guard(tmp_path, '--params-sha256', V7, '--split', 'v7_axial-v1')
    assert r.returncode == 2 and b'parts disagree on engine' in r.stdout
