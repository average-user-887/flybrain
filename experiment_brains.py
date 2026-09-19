"""Independent, inspectable learning state for each behavioral experiment.

Checkpoints restore learned MB state and CX working memory, not an exact replay of
an arena's random trajectory. No pickle or executable serialization is accepted.
The optional teaching protocol is a shared odor-conditioning calibration assay;
it is not evidence that every behavioral task has been learned.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from arena import Arena

PARADIGMS = (
    'open-arena', 't-maze', 'y-maze', 'heat-maze', 'buridan', 'visual-operant',
    'wind-tunnel', 'looming-escape', 'optomotor', 'gap-crossing', 'circadian-dam',
    'courtship', 'labyrinth', 'multisensory-sandbox',
)
ARRAYS = ('pn_tuning', 'w_pn_kc', 'w_kc_mbon_baseline', 'u', 'w', 'y_kc',
          'y_dan_pam', 'y_dan_ppl1')


class ExperimentBrain:
    def __init__(self, paradigm: str, directory: Path):
        if paradigm not in PARADIGMS:
            raise ValueError(f'Unknown experiment: {paradigm}')
        self.paradigm = paradigm
        self.directory = directory
        self.seed = int.from_bytes(hashlib.sha256(paradigm.encode()).digest()[:4], 'big')
        self.arena = Arena(paradigm=None if paradigm == 'open-arena' else paradigm,
                           brain_type='modular', seed=self.seed, num_flies=1, num_predators=0)
        self.brain_id = uuid.uuid4().hex
        self.created_at = time.time()
        self.trials = 0
        self.steps = 0
        self.elapsed = 0.0
        self.learning_enabled = True
        self.curve = []
        self.history = []
        self.teaching = None
        self.restored = False
        self.restore_error = None
        self.last_saved = None
        self._restore()
        self.arena.fly.learning_enabled = self.learning_enabled

    @property
    def circuit(self):
        return self.arena.fly.circuit

    @property
    def path(self):
        return self.directory / f'{self.paradigm}.json'

    def probe(self):
        """Forward readouts only: do not update weights, traces, RNG or arena."""
        result = {}
        for label, a, b in [('A', 1.0, 0.0), ('B', 0.0, 1.0)]:
            _, kc = self.circuit.encode_odor(a, b)
            approach, avoidance, valence = self.circuit.forward(kc)
            result[label] = {'approach': approach, 'avoidance': avoidance, 'valence': valence,
                             'active_kcs': int(np.count_nonzero(kc))}
        result['discrimination'] = result['A']['valence'] - result['B']['valence']
        return result

    def summary(self, details=False):
        weights = self.circuit.get_effective_weights()
        out = dict(paradigm=self.paradigm, brain_id=self.brain_id, seed=self.seed,
                   model='modular-mushroom-body', n_kc=self.circuit.n_kc,
                   synapses=int(weights.size), trials=self.trials, steps=self.steps,
                   learning_enabled=self.learning_enabled, restored=self.restored,
                   restore_error=self.restore_error, last_saved=self.last_saved,
                   weight_mean=float(weights.mean()), weight_std=float(weights.std()),
                   weight_change_l2=float(np.linalg.norm(weights - self.circuit.w_kc_mbon_baseline)),
                   probe=self.probe(), teaching=self.teaching,
                   curve=self.curve[-100:], history=self.history[-100:])
        if details:
            out['weights'] = weights.tolist()
            out['walls'] = [[w.p1, w.p2] for w in getattr(self.arena.paradigm, 'walls', [])]
        return out

    def log(self, kind, **fields):
        record = dict(kind=kind, timestamp=time.time(), brain_id=self.brain_id,
                      paradigm=self.paradigm, brain_step=self.steps, **fields)
        # The durable event ledger is append-only; bounded history is only a UI cache.
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f'{self.paradigm}.events.jsonl').open('a') as fh:
            fh.write(json.dumps(record, allow_nan=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        self.history.append(record)
        self.history = self.history[-200:]
        return record

    def start_teaching(self, pairs=8, reverse=False):
        if self.teaching:
            raise ValueError('Teaching is already running for this brain')
        if isinstance(pairs, bool) or not isinstance(pairs, int) or not 1 <= pairs <= 50:
            raise ValueError('pairs must be an integer from 1 to 50')
        self.circuit.reset_transients()
        self.teaching = dict(pair=0, pairs=pairs, tick=0, reverse=bool(reverse),
                             learning_enabled=self.learning_enabled, before=self.probe())
        self.log('teaching_started', pairs=pairs, reverse=bool(reverse), learning_enabled=self.learning_enabled)

    def teaching_step(self, dt):
        """A and B: 0.3 s cue, 0.3 s cue+US, 0.4 s washout, each. Arena is paused."""
        p = self.teaching
        # The daemon runs 20 ms ticks. Separate into two legal 10 ms plasticity bins.
        phase = p['tick'] % 100
        cue_b = phase >= 50
        within = phase % 50
        cue = 1.0 if within < 30 else 0.0
        us = 1.0 if 15 <= within < 30 else 0.0
        reward_cue = cue_b if p['reverse'] else not cue_b
        for _ in range(2):
            self.circuit.step(0.0 if cue_b else cue, cue if cue_b else 0.0,
                              reward=us if reward_cue else 0.0,
                              punishment=0.0 if reward_cue else us,
                              dt_seconds=dt / 2, learning=self.learning_enabled)
        p['tick'] += 1
        self.steps += 1
        if p['tick'] % 100 == 0:
            p['pair'] += 1
            self.log('teaching_pair', pair=p['pair'], reverse=p['reverse'],
                     learning_enabled=self.learning_enabled, probe=self.probe(),
                     weight_change_l2=float(np.linalg.norm(self.circuit.w)))
        if p['pair'] >= p['pairs']:
            self.log('teaching_completed', before=p['before'], after=self.probe(),
                     pairs=p['pairs'], reverse=p['reverse'], learning_enabled=self.learning_enabled)
            self.teaching = None
            self.circuit.reset_transients()
            self.save()

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        data = dict(schema_version=1, paradigm=self.paradigm, brain_id=self.brain_id,
                    seed=self.seed, created_at=self.created_at, saved_at=time.time(),
                    trials=self.trials, steps=self.steps, learning_enabled=self.learning_enabled, teaching=self.teaching,
                    curve=self.curve[-1000:], history=self.history[-200:],
                    circuit={name: getattr(self.circuit, name).tolist() for name in ARRAYS},
                    circuit_step_count=self.circuit.step_count,
                    cx={'epg': self.arena.fly.cx.epg.tolist(),
                        'goal_vector': self.arena.fly.cx.goal_vector.tolist(),
                        'has_goal': self.arena.fly.cx.has_goal})
        tmp = self.path.with_suffix('.json.tmp')
        with tmp.open('w') as fh:
            json.dump(data, fh, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self.last_saved = data['saved_at']
        return self.path

    def _restore(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if data['schema_version'] != 1 or data['paradigm'] != self.paradigm:
                raise ValueError('checkpoint schema or experiment mismatch')
            arrays = {k: np.asarray(data['circuit'][k], dtype=float) for k in ARRAYS}
            for k, value in arrays.items():
                if value.shape != getattr(self.circuit, k).shape or not np.isfinite(value).all():
                    raise ValueError(f'invalid circuit array: {k}')
            cx = data['cx']
            epg, goal = np.asarray(cx['epg'], dtype=float), np.asarray(cx['goal_vector'], dtype=float)
            if epg.shape != (16,) or goal.shape != (2,) or not np.isfinite(epg).all() or not np.isfinite(goal).all():
                raise ValueError('invalid CX state')
            # Validate before changing any live state. Invalid files remain untouched.
            for k, v in arrays.items():
                getattr(self.circuit, k)[:] = v
            self.circuit.step_count = int(data['circuit_step_count'])
            self.arena.fly.cx.epg[:] = epg
            self.arena.fly.cx.goal_vector[:] = goal
            self.arena.fly.cx.has_goal = bool(cx['has_goal'])
            for k in ('brain_id', 'created_at', 'trials', 'steps', 'learning_enabled', 'curve', 'history'):
                setattr(self, k, data[k])
            self.teaching = data.get('teaching')
            self.restored = True
            self.last_saved = data['saved_at']
        except (ValueError, KeyError, TypeError, OSError) as exc:
            # Refuse to continue with silently fresh weights or overwrite the evidence.
            raise ValueError(f'Cannot restore {self.path}: {exc}') from exc


class ExperimentBrains:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.instances = {}

    def get(self, paradigm):
        if paradigm not in PARADIGMS:
            raise ValueError(f'Unknown experiment: {paradigm}')
        if paradigm not in self.instances:
            self.instances[paradigm] = ExperimentBrain(paradigm, self.directory)
        return self.instances[paradigm]

    def catalog(self, active):
        return [dict(self.instances[p].summary(), state='active' if p == active else 'paused')
                if p in self.instances else dict(paradigm=p, state='saved' if (self.directory / f'{p}.json').exists() else 'not_started')
                for p in PARADIGMS]

    def save_all(self):
        return [brain.save() for brain in self.instances.values()]
