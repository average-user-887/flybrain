"""v7 S3 launcher (prereg v2): verifies the preregistered environment, then runs the UNCHANGED
frozen runner scripts/v6/measure_tuning_v6.py.  It changes nothing in the protocol.

Checks before any simulation (exit 3 on any failure, nothing launched):
  * cwd is this checkout's root (the runner opens the relative outputs/brainlab/malecns_v1/graph.npz);
  * that graph.npz (after resolving links) has the pinned sha256 4b2f87cc...091d01;
  * connectome_data/ exists (photoreceptor IO and neuron tables);
  * the interpreter's numpy / scipy / numba versions equal the prereg's;
  * the params file has the frozen v7 sha256 and the S3 prereg has 9658283e...;
  * the runner, gates, guard, merge and analysis files have the prereg v2 sha256s.

  python scripts/v7/run_s3_v7.py --prereg-v2 P --prereg-v2-sha256 SHA --params F --params-sha256 SHA \\
      --lattice ARM --part A|B --outdir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH_REL = Path('outputs/brainlab/malecns_v1/graph.npz')
PARTS = {'A': 'dir0,dir45,dir90,dir135,yaw_ccw', 'B': 'dir180,dir225,dir270,dir315,yaw_cw'}
LATTICES = ('axial-v1', 'malecns-hex-v2')


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def preflight(pre: dict, params, params_sha, cwd=None, versions=None):
    """Return a list of failures (empty = ready)."""
    cwd = Path(cwd or os.getcwd()).resolve()
    env = pre['downstream_s3']['environment']
    bad = []
    if cwd != REPO.resolve():
        bad.append(f'cwd {cwd} is not the checkout root')
    g = cwd / GRAPH_REL
    if not g.exists():
        bad.append(f'{GRAPH_REL} missing')
    elif sha(g.resolve()) != env['graph_sha256']:
        bad.append('graph sha256 differs from the pin')
    if not (cwd / 'connectome_data').exists():
        bad.append('connectome_data missing')
    if versions is None:
        import numba, numpy, scipy
        versions = dict(numpy=numpy.__version__, scipy=scipy.__version__, numba=numba.__version__)
    for k, v in env['python_packages'].items():
        if versions.get(k) != v:
            bad.append(f'{k} {versions.get(k)} != {v}')
    if not Path(params).exists() or sha(params) != params_sha:
        bad.append('params sha256 mismatch')
    for rel, s in pre['downstream_s3']['unchanged_sha256'].items():
        f = cwd / rel
        if not f.exists() or sha(f) != s:
            bad.append(f'{rel} missing or sha256 differs')
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prereg-v2', type=Path, required=True); ap.add_argument('--prereg-v2-sha256', required=True)
    ap.add_argument('--params', type=Path, required=True); ap.add_argument('--params-sha256', required=True)
    ap.add_argument('--lattice', choices=LATTICES, required=True)
    ap.add_argument('--part', choices=tuple(PARTS), required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    a = ap.parse_args()
    raw = a.prereg_v2.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.prereg_v2_sha256:
        raise SystemExit('prereg v2 sha256 mismatch')
    pre = json.loads(raw)
    bad = preflight(pre, a.params, a.params_sha256)
    receipt = dict(python=sys.executable, cwd=str(Path.cwd()), failures=bad)
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / f'preflight_v7_{a.lattice}_{a.part}.json').write_text(json.dumps(receipt, indent=1))
    if bad:
        print('\n'.join(bad), file=sys.stderr)
        sys.exit(3)
    s3 = pre['downstream_s3']['unchanged_sha256']
    cmd = [sys.executable, 'scripts/v6/measure_tuning_v6.py', '--engine', 'v6', '--params', str(a.params),
           '--params-sha256', a.params_sha256, '--prereg', 'qualification/v6/S3_prereg.json',
           '--prereg-sha256', s3['qualification/v6/S3_prereg.json'], '--lattice', a.lattice,
           '--tag', f'v7_{a.lattice}_{a.part}', '--outdir', str(a.outdir), '--conditions', PARTS[a.part]]
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
