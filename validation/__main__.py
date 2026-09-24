"""CLI: ``python -m validation run <spec> --out <dir>`` / ``python -m validation check <spec>``."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m validation', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='run a spec and write receipt.json + trials.json into --out')
    run.add_argument('spec', type=Path)
    run.add_argument('--out', type=Path, required=True, help='new directory (must not exist)')
    run.add_argument('--backend', default='auto', choices=('auto', 'cpu', 'cuda'))
    run.add_argument('--graph-dir', type=Path)
    run.add_argument('--connectome-dir', type=Path)
    run.add_argument('--synthetic', action='store_true',
                     help='labelled synthetic typed graph (tests the code path; no scientific verdict)')
    run.add_argument('--seeds', type=int, nargs='+',
                     help='override the spec seeds (makes the run EXPLORATORY)')
    check = sub.add_parser('check', help='validate a spec and its bounds file, print its gates')
    check.add_argument('spec', type=Path)
    args = parser.parse_args(argv)
    from validation import harness
    if args.command == 'check':
        spec = harness.load_spec(args.spec)
        bounds = harness.load_bounds(spec)
        missing = [c['bound'] for c in spec['physiology']['checks'] if c['bound'] not in bounds['bounds']]
        if missing:
            raise SystemExit(f'bounds missing: {missing}')
        print(json.dumps(dict(id=spec['id'], status=spec['status'], sha256=spec['_sha256'],
                              gates=[(g['id'], g['verified']) for g in spec['behaviour']['gates']],
                              physiology=[c['id'] for c in spec['physiology']['checks']]), indent=1))
        return 0
    receipt = harness.run(args.spec, args.out, synthetic=args.synthetic, backend=args.backend, seeds=args.seeds,
                          graph_dir=args.graph_dir, connectome_dir=args.connectome_dir)
    return 0 if receipt['verdict'] in ('PASS', 'PASS_PROVISIONAL', 'EXPLORATORY', 'SYNTHETIC_PLUMBING_ONLY') else 1


if __name__ == '__main__':
    sys.exit(main())
