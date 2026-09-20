"""Local, append-only experiment records. No external telemetry."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import platform
import uuid
import numpy as np
from .connectome import ROOT, file_digest


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class Run:
    """A unique run with provenance, durable events, and binned neural data.

    Leaving the context marks completion/failure. A killed process remains
    'running', deliberately never mistaken for a completed experiment.
    """
    def __init__(self, graph, config, seed, root=None):
        self.path = Path(root or ROOT/'runs') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:12])
        self.path.mkdir(parents=True, exist_ok=False)
        self.index = 0
        self.trials = 0
        self.metadata = dict(schema_version=1, run_id=self.path.name, started_at=now(),
            status='initializing', seed=int(seed), config=config,
            interpretation='Numerical experiment; no demonstrated learning or biological validity')
        atomic_json(self.path/'run.json', self.metadata)
        try:
            source_dir = self.path/'source'
            source_dir.mkdir()
            hashes = {}
            for source in sorted((ROOT/'brainlab').glob('*.py')):
                data = source.read_bytes()
                (source_dir/source.name).write_bytes(data)
                hashes[source.name] = hashlib.sha256(data).hexdigest()
            self.metadata.update(graph_path=str(Path(graph).resolve()), graph_sha256=file_digest(Path(graph)),
                source_sha256=hashes, python=platform.python_version(), platform=platform.platform(),
                dependencies={name:version(name) for name in ['numpy','numba','pandas','pyarrow']},
                status='running')
            atomic_json(self.path/'run.json', self.metadata)
        except BaseException as error:
            self.metadata.update(status='failed', error=str(error), ended_at=now())
            atomic_json(self.path/'run.json', self.metadata)
            raise

    def __enter__(self):
        return self

    def event(self, kind, **fields):
        record = dict(schema_version=1, event=kind, timestamp=now(), **fields)
        line = json.dumps(record, allow_nan=False)+'\n'
        with (self.path/'events.jsonl').open('a') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def step(self, brain, currents, duration_ms, *, trial_id, phase):
        """Record actual input, nonzero spike counts, and terminal active state.

        Counts are per bin, NOT precise spike times. IDs retain all 64 bits.
        State samples are observations, not restartable checkpoints.
        """
        drive = np.ascontiguousarray(currents, dtype=np.float32)
        start = brain.sim_ms
        counts, elapsed = brain.step(drive, duration_ms)
        fired = np.flatnonzero(counts)
        driven = np.flatnonzero(drive)
        active = brain.active[:int(brain.nactive[0])]
        filename = f'bin-{self.index:06d}.npz'
        temporary = self.path/(filename+'.partial')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, spike_ids=brain.ids[fired], spike_counts=counts[fired],
                input_ids=brain.ids[driven], input_current=drive[driven],
                state_ids=brain.ids[active], voltage=brain.v[active],
                # v1: (k,) current-like synaptic state; v2: (2, k) excitatory/inhibitory
                # conductances.  The trailing axis is always the neuron axis.
                synaptic_state=brain.g[..., active], lif_dynamics=np.array(brain.dynamics))
        temporary.replace(self.path/filename)
        self.event('neural_bin', bin=self.index, trial_id=trial_id, phase=phase,
            sim_start_ms=start, sim_end_ms=brain.sim_ms, wall_seconds=elapsed,
            total_spikes=int(counts.sum()), firing_neurons=len(fired),
            finite_state=bool(np.isfinite(brain.v).all() and np.isfinite(brain.g).all()),
            artifact=filename, artifact_sha256=file_digest(self.path/filename))
        self.index += 1
        return counts, elapsed

    def trial(self, *, trial_id, condition, **measurements):
        """Future tasks may supply cue, delay_ms, action, reward, success, split."""
        self.event('trial_end', trial_id=trial_id, condition=condition, **measurements)
        self.trials += 1

    def __exit__(self, kind, error, traceback):
        self.metadata.update(status='failed' if kind else 'completed', ended_at=now(),
                             neural_bins=self.index, trials=self.trials)
        if error is not None:
            self.metadata['error'] = f'{kind.__name__}: {error}'
        atomic_json(self.path/'run.json', self.metadata)
        return False
