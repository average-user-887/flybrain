#!/usr/bin/env python3
"""Reproducible paired/reversed/frozen calibration for all experiment brains.

This tests associative readouts, isolation and persistence, not maze solving.
Creates a fresh run directory, never overwriting an existing run.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_brains import ExperimentBrain, PARADIGMS


def train(brain, pairs, reverse=False):
    brain.start_teaching(pairs, reverse)
    while brain.teaching:
        brain.teaching_step(0.02)
    return brain.probe()['discrimination']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs') / f'learning-battery-{time.time_ns()}')
    parser.add_argument('--pairs', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for pid in PARADIGMS:
        b = ExperimentBrain(pid, args.output / 'trained')
        frozen = ExperimentBrain(pid, args.output / 'frozen')
        frozen.learning_enabled = False
        baseline = b.probe()['discrimination']
        paired = train(b, args.pairs)
        frozen_score = train(frozen, args.pairs)
        b.save()
        restored = ExperimentBrain(pid, args.output / 'trained')
        exact = bool(np.array_equal(restored.circuit.w, b.circuit.w) and restored.brain_id == b.brain_id)
        reversal = train(b, args.pairs, reverse=True)
        row = dict(paradigm=pid, seed=b.seed, brain_id=b.brain_id, baseline=baseline,
                   paired=paired, reversed=reversal, frozen=frozen_score, restore_exact=exact,
                   passed=paired > 0.05 and reversal < -0.05 and frozen_score == 0.0 and exact)
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {pid:23} paired={paired:+.3f} reversed={reversal:+.3f} frozen={frozen_score:+.3f} restored={exact}", flush=True)
    report = dict(protocol='A cue→reward, B cue→punishment; reverse; matched seeded frozen control',
                  model='modular MB, 120 KCs', pairs=args.pairs,
                  scope='Associative readout calibration. Not evidence of improved arena navigation.',
                  all_passed=all(r['passed'] for r in rows), experiments=rows)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {args.output / "report.json"}')
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
