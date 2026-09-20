# Experiment recording

Run a reproducible full-brain baseline experiment from this workspace:

```bash
cd /path/to/flybrain
.venv/bin/python -m brainlab.experiment --seed 42 --repeats 2
.venv/bin/python -m brainlab.report
```

This randomizes six trials: two each of silent, lamina-tonic-only, and retina-plus-tonic stimulation. Each trial starts with a fresh brain, runs 100 ms of input, and observes 50 ms of recovery with no input. Neural observations are binned every 10 ms. The seed controls trial order; the model itself is deterministic. Repeated trials are reproducibility checks, not independent evidence of learning.

## What gets saved

Every invocation creates a unique directory under `runs/`. Nothing is uploaded.

| Artifact | Contents |
| --- | --- |
| `run.json` | Parameters, seed, schedule, graph SHA-256, dependency versions, timestamps, and completion/failure status |
| `source/` | Exact Python source snapshot, with hashes recorded in the manifest |
| `events.jsonl` | Flushed event stream: trial boundaries, phase, simulation time, wall time, spike totals, outcomes |
| `bin-*.npz` | Actual nonzero input currents, nonzero spike counts, active-neuron voltage and synaptic state, with exact integer neuron IDs |
| `runs/comparison.csv` | Rebuildable table of recorded trials across runs, including status, graph identity, condition, reward, and success |

The graph itself is stored once outside the run directory and identified by hash. Keep it with any archived experiment. NPZ observations contain binned counts, not precise spike times. State samples are not resumable checkpoints: queues and complete simulation state are not captured. Logging overhead is excluded from each bin's neural `wall_seconds`; the run's timestamps cover total elapsed time including recording.

A caught exception marks the run `failed` and keeps previously recorded data. A forcibly killed process leaves `running` status, which must be treated as incomplete. Comparison output includes status explicitly; do not pool incomplete results with completed experiments. Each event is flushed to disk; atomic artifact and manifest replacement avoid exposing partially written normal records.

## Use with the future learning task

```python
from brainlab.runs import Run

with Run(graph_path, task_config, seed=42) as run:
    # Task code owns trial state, input mapping, learning, and action selection.
    counts, seconds = run.step(brain, currents, 10.0,
                              trial_id=0, phase='cue')
    run.event('decision', trial_id=0, action='left', cue='blue')
    run.trial(trial_id=0, condition='delayed-choice',
              cue='blue', delay_ms=100, action='left',
              reward=1, success=True, split='evaluation')
```

The above decision and reward are API examples, not measured results. Future learning runs should record the learning rule and input/output mapping, weight changes or checkpoint references, training/evaluation split, and delay. The baseline runner does not implement a task, learning, rewards, or weight checkpoints. It leaves success/reward null, rather than reporting fictitious accuracy.

## Initial measured run

Run `20260914T210929-4d29ce92dce4` completed all six trials and recorded 90 neural bins on the full 166,700-neuron graph. Each repeat yielded:

- Silent: 0 spikes during stimulation.
- Tonic only: 28,523 spikes.
- Retina plus tonic: 52,282 spikes.

These are simulator activity measurements, not learning scores. Automated recorder tests cover exact neuron IDs, input replay, artifact/event agreement, failure preservation, and unique run directories.
