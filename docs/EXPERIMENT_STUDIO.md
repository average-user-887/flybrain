# Experiment studio (roadmap P5, first slice)

The studio is a browser page for building an experiment, queueing it, watching
the finished run in 3D, and comparing two runs side by side. It sits on the P2
pieces: the `neurofly_body` run queue, `body.nfbody` recordings and
`web/embodied_replay.html`.

```bash
python -m neurofly_studio serve \
    --graph-dir outputs/brainlab/malecns_v1 --connectome-dir connectome_data/malecns_v1
# open http://127.0.0.1:8782/
```

`serve` also runs the queue worker (one run at a time) unless `--no-worker` is
given, in which case run `python -m neurofly_body queue run outputs/studio-queue
--watch 5` yourself, for example in tmux. Only one studio worker runs per queue
(a lock file enforces it). The server binds to 127.0.0.1; reach it from another
computer through an SSH tunnel (`ssh -L 8782:127.0.0.1:8782 <server>`). Anyone
who can reach a non-loopback `--host` can queue runs.

## What the first slice does

| Roadmap item | State |
|---|---|
| Badges from the capability matrix | Done. Parsed from `docs/CAPABILITY_MATRIX.md` on every request: "Tested on v3" is **Validated**, "Mapped…" is **Mapped**, anything else is **Exploratory**. |
| Experiment builder | Done for optomotor, the one stimulus the embodied loop drives: rotation speed, contrast, simulated time, seed, repeats. The other 13 paradigms are listed with their badge and "Not in the studio yet". |
| Queue | Done: live list, cancel before start, failures with their reason. |
| Watch a run | Done: opens `embodied_replay.html` on the run's `body.nfbody` (1x, seek, speed). |
| Side-by-side comparison | Done: two replays and a table of descriptive numbers. Playback of the two sides is not synchronised yet. |
| Curated gallery | The page and format exist; no curated runs ship yet (they need Ryzen runs). |
| Silence a named neuron type | Done when the installed runner has `neurofly_body run --silence` (PR #31); hidden otherwise. Five groups for optomotor, both sides or one, paired by default with the same fly un-silenced. |
| Research API and CLI | `python -m neurofly_studio submit EXPERIMENT.json` and the HTTP API. Export to Parquet or NWB is still to decide. |

## Design decisions

**Every studio run is exploratory.** The experiment file has a `label` field
that can only be `"exploratory"`; anything else is refused with a pointer to
`neurofly validate run`. Preregistered specs under `validation/specs/` are
listed read-only with their SHA-256 and cannot be edited, re-run or overridden
from the studio. This keeps tweaked runs from ever being mistaken for
confirmatory results.

**Full accuracy only.** An experiment file cannot name the neural time step,
physics time step or recording rate; runs always use the runner's defaults
(2 ms neural steps, 0.1 ms physics, 50 recorded frames per simulated second),
and each run's `manifest.json` records them. There is no faster, less accurate
mode in the studio.

**Matched control by default.** The builder pre-ticks a paired
`output-disconnected` run with the same seed: the connectome runs and is
recorded, but its commands never reach the legs. It is the one control the
embodied runner supports today, and with the same seed any difference between
the pair comes from the brain-to-body coupling. Citizens can untick it.

**Parameter ranges are input ranges, not validity claims.** Rotation speed is
limited to ±12 rad/s (about ±690 °/s). Tethered-fly optomotor experiments since
Götz (1964) have used gratings moving from a few to several hundred degrees per
second. The embodied encoder takes retinal slip in rad/s and has no grating
wavelength, so temporal frequency is not defined here. Contrast runs from 0 to 1,
the runner's own limit. Simulated time runs from 0.5 to 60 s in 0.1 s steps, so
every run is a whole number of 2 ms steps and stays affordable on one GPU.
Repeats go up to 6 seeds, the size of one preregistered optomotor seed block.

**Silencing.** The builder offers groups of MaleCNS cell types named for this
circuit in `brainlab/io_map.py`: the T4/T5 motion detectors that receive the
stimulus (Maisak 2013), the HS cells (HSN, HSE, HSS) of the lobula plate,
DNa02 (steering; Yang 2024, Rayshubskiy 2025), DNp09 (forward walking; Bidaye
2020) and MDN (backward walking; Bidaye 2014). Each group can be silenced on
both sides or on one soma side. Researchers can name any cell type in the
experiment file; the runner refuses one that resolves to no neurons before the
run starts. Silencing uses the validation harness's clamp (input current held
at `SILENCE_DRIVE` every step). Under v3 that clamp can leak, so the queue,
gallery and comparison show `clamp_held` from the run's `summary.json`, and a
run where a silenced neuron spiked is marked "clamp leaked". When something is
silenced, the default control is the same seed with nothing silenced ("what
does this neuron do?"). The disconnected-brain control keeps the silencing.

**Modular controller is a researcher baseline.** Experiment files may set
`"controller": "modular"`; the citizen page never offers it. Modular runs do
not load the graph, and the page labels them.

**Comparison numbers are descriptive.** Mean body yaw velocity, turning gain
(fly ÷ world), net heading change (yaw unwrapped step by step) and total graph
spikes are read from `telemetry.jsonl`. One seed per side is an anecdote, and
the page says so.

**A separate small server.** The studio is `neurofly_studio.server` (standard
library only, default port 8782), not new endpoints in `neurofly_daemon.py`. The
daemon runs the live assays under a simulation lock. Queue runs are separate
processes and do not need that lock. Queue runs share the GPU with a running
daemon: that changes their speed, not their results.

**Browser safety.** POSTs need `Content-Type: application/json` and, when the
browser sends an `Origin`, the same host, so another web page cannot queue runs
through a visitor's browser. Run names are server-generated, run files are
served from a fixed list (`summary.json`, `manifest.json`, `body.nfbody`), and
only finished runs are served.

## Experiment files

```json
{
  "schema": "neurofly-studio-experiment-v1",
  "title": "Slow rotation, half contrast",
  "paradigm": "optomotor",
  "parameters": {"world_angular_velocity_rad_s": 2.0, "contrast": 0.5,
                 "duration_s": 5.0, "seed": 1},
  "repeats": 3,
  "silence": ["DNa02", "MDN:L"],
  "control": "intact",
  "controller": "connectome"
}
```

`python -m neurofly_studio submit exp.json --dry-run` prints the planned runs
and their exact `neurofly_body run` arguments. Without `--dry-run` it queues
them. Unknown fields are refused, as is any run the runner's own argument
parser would reject. Each queued run gets `QUEUE/studio/<run>.json` with the
experiment, its role, seed and paired run.

## HTTP API

| Method | Path | |
|---|---|---|
| GET | `/api/studio/catalog` | 14 paradigms: badge, matrix status, parameters, controls, preregistered specs |
| GET | `/api/studio/runs` | queue jobs (pending, running, done, failed) and curated runs |
| POST | `/api/studio/experiments` | queue an experiment file; returns the run names |
| POST | `/api/studio/runs/<name>/cancel` | drop a job that has not started |
| GET | `/api/studio/metrics/<queue\|curated>/<name>` | comparison numbers of a finished run |
| GET | `/api/studio/files/<queue\|curated>/<name>/<file>` | `summary.json`, `manifest.json` or `body.nfbody` |

## Curated runs

A curated run is a finished run directory under `experiment_data/curated/<name>/`
(or `--curated DIR`) with a `curated.json`: `title`, `paradigm`, `explanation`,
`role`, `pair` and `parameters`. They are produced on the reference machine,
checked with `replay-check`, and ship with the first public release (P6).

## Checks

- `tests/test_studio.py`: catalog and badges, experiment validation, planning,
  the HTTP API end to end with a stand-in runner, cancel, the cross-origin and
  content-type guards.
- `scripts/studio_headless_check.py --out DIR`: headless Chromium walk-through
  of every tab at desktop and phone width, with a stand-in runner
  (`tests/studio_fakes.py`: modular baseline and a stick-fly body, not a
  simulation). It supplements the live browser check in `AGENTS.md` and does
  not replace it.

## Next slices

1. Synchronised side-by-side playback.
2. The first curated runs, produced on the reference machine (an intact vs
   DNa02-silenced pair is the obvious first one).
3. Export format (Parquet or NWB) with provenance.
4. Looming and T-maze in the builder once the embodied loop drives those stimuli.
