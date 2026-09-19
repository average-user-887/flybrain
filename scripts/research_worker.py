#!/usr/bin/env python3
"""Persistent, isolated arena training with paired frozen controls and shareable data.

Each assay/seed has separate retained trained and frozen brains. Evaluation uses
a fresh arena seed and frozen copies of their mushroom bodies. A difference is
an exploratory observation, not evidence of validated biological learning.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from experiment_brains import ExperimentBrain, PARADIGMS

METRICS = dict(zip(PARADIGMS, (
    'distance_mm', 'performance_index', 'spontaneous_alternation_rate', 'escape_latency_ms',
    'centrophobism_index', 'operant_learning_index', 'upwind_progress_mm',
    'time_to_collision_at_jump_ms', 'optomotor_gain', 'crossing_success',
    'total_sleep_minutes', 'courtship_index', 'path_tortuosity', 'composite_benchmark_score')))
PROTOCOL = ('Sensorimotor-v2: explicit heuristic reflexes, not a validated biological circuit. Arena experience with plasticity enabled versus a matched-seed frozen control; '
            'retained memory across rounds. Evaluation: fresh arena seed, copied MB memory, '
            'reset eligibility traces and frozen weights in both groups. Reinforcement remains '
            'present during evaluation. Raw trajectories sampled every simulated second; '
            'motion integrity checked every 20 ms. No resets inside a phase. '
            'Short runs cannot establish circadian rhythms or general multi-step learning.')


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as fh:
        json.dump(data, fh, allow_nan=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def phase(brain, steps, path, stop=None):
    """No trial respawns; preserve raw coordinates and inspect every integration step."""
    prev = brain.arena.fly.pos.to_tuple()
    distance = 0.0
    rests = jumps = 0
    max_step = 0.0
    last = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        for i in range(steps + 1):
            if stop and stop.is_set():
                raise InterruptedError('Worker stopped; incomplete cohort excluded')
            if i:
                last = brain.arena.step(.02)
                brain.steps += 1
            f = brain.arena.fly
            pose = f.pos.to_tuple()
            d = math.dist(prev, pose)
            if not all(math.isfinite(v) for v in (*pose, f.heading, f.speed)):
                raise ValueError('Non-finite trajectory')
            jumps += d > .32  # 3.5 mm/s maximum plus collision tolerance
            max_step = max(max_step, d)
            distance += d
            rests += i > 0 and f.behavioral_state == 'REST'
            if i % 50 == 0 or i == steps:
                fh.write(json.dumps(dict(step=i, sim_time_s=i*.02, x=f.pos.x, y=f.pos.y,
                    heading=f.heading, speed=f.speed, state=f.behavioral_state,
                    reward=last.get('reward', 0), punishment=last.get('punishment', 0),
                    metrics=last.get('paradigm_metrics', {})), allow_nan=False)+'\n')
            prev = pose
        fh.flush()
        os.fsync(fh.fileno())
    metrics = last.get('paradigm_metrics', {})
    if brain.paradigm == 'open-arena':
        metrics['distance_mm'] = distance
    return dict(metrics=metrics, distance_mm=distance, rest_seconds=rests*.02,
                max_step_mm=max_step, discontinuities=jumps, steps=steps,
                quality='valid' if jumps == 0 else 'invalid_motion')


def cohort(output, index, seeds, steps, stop=None, previous=None):
    pid = PARADIGMS[index % len(PARADIGMS)]
    seed = (index // len(PARADIGMS)) % seeds
    round_no = index // (len(PARADIGMS) * seeds)
    run = output / 'cohorts' / f'{index:07d}-{pid}-seed{seed}'
    # A interrupted attempt has a different directory and remains available for audit.
    run = run.with_name(run.name + '-' + str(time.time_ns()))
    run.mkdir(parents=True)
    row = dict(index=index, paradigm=pid, seed=seed, round=round_no,
               metric_name=METRICS[pid], started_at=time.time(), directory=str(run.relative_to(output)))
    try:
        for group in ('trained', 'frozen'):
            state = run / 'state' / group
            state.mkdir(parents=True)
            if previous:
                previous_file = output / previous['directory'] / 'state' / group / f'{pid}.json'
                initial = previous_file.read_bytes()
                (state / f'{pid}.json').write_bytes(initial)
                (run / f'{group}-initial-brain.json').write_bytes(initial)
            brain = ExperimentBrain(pid, state, seed=seed)
            brain.learning_enabled = group == 'trained'
            brain.arena.fly.learning_enabled = brain.learning_enabled
            before = brain.probe()
            train = phase(brain, steps, run / f'{group}-training.jsonl', stop)
            # New arena, no residual locomotor/hunger/compass state from training.
            evaluation_seed = 100000 + seed + round_no * seeds
            evaluation = ExperimentBrain(pid, run / 'evaluation' / group, seed=evaluation_seed)
            evaluation.arena.fly.circuit = copy.deepcopy(brain.circuit)
            evaluation.circuit.reset_transients()
            evaluation.learning_enabled = False
            evaluation.arena.fly.learning_enabled = False
            held_out = phase(evaluation, steps, run / f'{group}-evaluation.jsonl', stop)
            metric = held_out['metrics'].get(METRICS[pid])
            if isinstance(metric, bool):
                metric = int(metric)
            if metric is not None and not math.isfinite(metric):
                metric = None
            row[group] = dict(brain_id=brain.brain_id, before=before, after=brain.probe(),
                              training=train, evaluation=held_out, evaluation_seed=evaluation_seed,
                              metric=metric, weight_change_l2=brain.summary()['weight_change_l2'])
            if train['quality'] != 'valid' or held_out['quality'] != 'valid':
                raise ValueError('Motion integrity failed; cohort excluded from comparisons')
            # Saved inside this attempt. Only a complete ledger pair becomes a future parent.
            brain.save()
            (run / f'{group}-brain.json').write_bytes(brain.path.read_bytes())
        row['status'] = 'complete'
    except InterruptedError:
        atomic_json(run / 'result.json', {**row, 'status':'interrupted'})
        raise
    except Exception as exc:
        row.update(status='failed', error=str(exc))
    row['finished_at'] = time.time()
    atomic_json(run / 'result.json', row)
    return row


def bundle(output, rows, manifest, target):
    """Only this batch's generated scientific files; no workspace or secret scraping."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.zip.tmp')
    hashes = {}
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for row in rows:
            directory = output / row['directory']
            for path in sorted(directory.glob('*')):
                if path.suffix not in ('.json', '.jsonl') or not path.is_file():
                    continue
                name = str(path.relative_to(output))
                content = path.read_bytes()
                hashes[name] = hashlib.sha256(content).hexdigest()
                z.writestr(name, content)
        z.writestr('manifest.json', json.dumps({**manifest, 'sha256':hashes}, indent=2))
        z.writestr('README.txt', PROTOCOL + '\n\nModel: compact modular brain, 120 KCs. '
                   'Not the downloaded full MaleCNS connectome. Missing outcomes are null, '
                   'not zero or success. Each file is a separate continuous phase. '
                   'Do not join positions between files. Failed cohorts must be excluded.\n'
                   'Reproduce with scripts/research_worker.py using the manifest options and revision.\n')
    os.replace(tmp, target)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROJECT / 'outputs' / 'research-sensorimotor-v2')
    p.add_argument('--public-status', type=Path, default=PROJECT / 'web' / 'research-status.json')
    p.add_argument('--bundle', type=Path, default=PROJECT / 'web' / 'research-latest.zip')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--interval', type=float, default=300)
    p.add_argument('--once', action='store_true')
    args = p.parse_args(argv)
    if args.steps < 1 or args.seeds < 1 or args.interval < 0:
        p.error('steps/seeds must be positive and interval non-negative')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    ledger = args.output / 'cohorts.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT,text=True).strip()
    manifest = dict(schema_version=1, revision=revision, protocol=PROTOCOL, dt=.02,
                    steps=args.steps, seeds=args.seeds, python=sys.version, numpy=np.__version__,
                    source_sha256={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
                        for name in ('arena.py','maze.py','online_metrics.py','assay_response.py','assay_controls.py','circuit.py','surge_cast.py','vision.py','mechanosensory.py','metabolic.py','central_complex.py','locomotion.py','experiment_brains.py','scripts/research_worker.py')})
    if (args.output / 'manifest.json').exists():
        previous_manifest = json.loads((args.output / 'manifest.json').read_text())
        if any(previous_manifest[k] != manifest[k] for k in ('steps','seeds','dt')):
            raise ValueError('Protocol changed; use a new output directory')
    atomic_json(args.output / 'manifest.json', manifest)
    def publish(state, active=''):
        atomic_json(args.public_status, dict(updated_at=time.time(), state=state, active=active,
            completed=sum(r['status']=='complete' for r in rows), failed=sum(r['status']!='complete' for r in rows),
            recent=rows[-140:], protocol=PROTOCOL, revision=revision))
    try:
        while not stop.is_set():
            batch = []
            for _ in range(len(PARADIGMS)):
                index = rows[-1]['index']+1 if rows else 0
                publish('Training and evaluating', PARADIGMS[index % len(PARADIGMS)])
                previous = next((r for r in reversed(rows) if r['status']=='complete' and r['index'] % (len(PARADIGMS)*args.seeds) == index % (len(PARADIGMS)*args.seeds)), None)
                row = cohort(args.output, index, args.seeds, args.steps, stop, previous)
                row['revision'] = revision
                row['source_sha256'] = manifest['source_sha256']
                atomic_json(args.output / row['directory'] / 'result.json', row)
                with ledger.open('a') as fh:
                    fh.write(json.dumps(row, allow_nan=False)+'\n');fh.flush();os.fsync(fh.fileno())
                rows.append(row);batch.append(row)
                print(json.dumps({k:row[k] for k in ('index','paradigm','seed','status')}), flush=True)
            bundle(args.output, batch, manifest, args.bundle)
            if args.once:
                break
            deadline = time.monotonic()+args.interval
            while not stop.is_set() and time.monotonic()<deadline:
                publish('Waiting for next batch')
                stop.wait(min(30, max(0, deadline-time.monotonic())))
    except InterruptedError:
        pass
    finally:
        publish('Stopped' if stop.is_set() else 'Batch complete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
