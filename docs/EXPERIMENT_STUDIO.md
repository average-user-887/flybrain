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
| Side-by-side comparison | Done: two replays driven by one clock (play, pause, seek and speed for both), and a table of descriptive numbers. |
| Curated gallery | Done as the "Start here" section of the gallery, filled by `python -m neurofly_studio curate`. The first curated runs still have to be made on the reference machine (plan below). |
| Silence a named neuron type | Done when the installed runner has `neurofly_body run --silence` (PR #31); hidden otherwise. Five groups for optomotor, both sides or one, paired by default with the same fly un-silenced. |
| Research API and CLI | `python -m neurofly_studio submit EXPERIMENT.json`, the HTTP API, and run bundles (`export`, a Download button on every run). |

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
| GET | `/api/studio/export/<queue\|curated>/<name>.zip` | the run bundle |

## Run bundles (export)

`python -m neurofly_studio export RUN_DIR --out run.zip`, or Download on any
run card, gives one zip per run:

- the run's own files, unchanged: `manifest.json`, `summary.json`,
  `telemetry.jsonl`, `timing.jsonl`, `body.nfbody` and, if present,
  `replay_check.json`;
- `telemetry.parquet`: one row per 2 ms step, with nested fields flattened to
  dotted column names (`body.thorax.yaw_rad`, `motor.applied_cpg_drive.0`, …);
- `studio.json`: the experiment and its exploratory label;
- `bundle.json`: the format, the provenance (manifest subset) and the SHA-256 of
  every file;
- a README.

The zip is deterministic, so the same run always gives the same bytes and the
same SHA-256.

**Why Parquet, not NWB.** The roadmap left this choice to P5. A run today
records body kinematics, motor commands, per-type DN rates and whole-graph
spike counts. It has no per-neuron spike trains and no measured data, which
are what NWB's structure is for. Parquet opens directly in pandas, R (arrow)
and Julia, and pyarrow is already a dependency. The original JSON lines stay in
the bundle, so the conversion loses nothing. Revisit NWB once runs can export
per-neuron rasters.

## Curated runs

Curated runs are the citizen entry point: the first thing the gallery shows,
each with a plain-language explanation, Watch, Download and "Compare with
pair".

```bash
python -m neurofly_studio curate RUN_DIR [CONTROL_RUN_DIR] --name optomotor-intro \
    --title "The fly turns with the world" --explanation "..." [--control-explanation "..."]
```

A run can be curated only when it completed and `neurofly_body replay-check`
reproduced it bit for bit. Its `replay_check.json`, copied into the run
directory, must say `BIT_IDENTICAL` and name the run's trajectory hash.
Curation checks every run first and only then copies anything, and it never
overwrites. It copies `manifest.json`, `summary.json`, `body.nfbody` and
`replay_check.json` to `experiment_data/curated/<name>/` (and `<name>-control/`)
and writes `curated.json`: the explanation, role and pair, the parameters, the
comparison numbers computed from `telemetry.jsonl`, and the SHA-256 of every
copied file. `telemetry.jsonl` is left out to keep the install small; use
`--with-telemetry` to keep it. Curated runs keep the exploratory label:
curation checks reproducibility, not scientific validity.

The first set is planned in `experiment_data/curated_plan/`:
`optomotor-intro.json` (10 s, intact vs brain disconnected from the legs) and
`optomotor-dna02-silenced.json` (10 s, DNa02 silenced vs the same fly
unsilenced; needs `--silence`). Both use seed 1 and the connectome controller.

## Synchronised playback

The comparison view drives both replays from one clock. It calls
`window.embodiedReplay.seekTime(t)` in each same-origin replay frame, a small
additive hook in `web/embodied_replay.js` (`duration`, `seekTime`, `pause`).
Each replay can still be played on its own with its own controls.

## Checks

- `tests/test_studio.py`: catalog and badges, experiment validation, planning,
  silencing, the HTTP API end to end with a stand-in runner, cancel, the
  cross-origin and content-type guards, bundle determinism and checksums,
  and curation's replay requirement.
- `scripts/studio_headless_check.py --out DIR`: headless Chromium walk-through
  of every tab at desktop and phone width, with a stand-in runner
  (`tests/studio_fakes.py`: modular baseline and a stick-fly body, not a
  simulation). It supplements the live browser check in `AGENTS.md` and does
  not replace it.

## Next slices

1. Produce the first curated runs on the reference machine (plan above).
2. Per-neuron raster export, and NWB with it.
3. Looming and T-maze in the builder once the embodied loop drives those stimuli.
