# Changelog

All notable changes to Project NeuroFly are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Scientific results, including negative
ones, get their own section, because for this project they matter as much as code.

## [0.5.0rc1] - 2026-10-07 (PRERELEASE)

Release candidate on top of `v0.4.0`. **Prerelease, pending review and the owner's
decision.** The first GPU cohort kernel failed its preregistered CPU-reference gate.
This candidate changes its arrival arithmetic and passes the unchanged gate, but the GPU
is now only about 1.1–1.4× faster than the CPU (see Scientific results). Details:
`docs/RELEASE_NOTES_v0.5.0rc1.md`.

### Added

- **Cohort runs:** `neurofly cohort run|resume|verify` (`brainlab/cohort`). Many
  independent fixed-weight v3 brains share one read-only graph, each with its own seed,
  arena, encoder and decoder. The only assay is optomotor. Checkpoint/resume is
  byte-exact and works across engines. The CPU engine is the reference; `--engine gpu`
  (CUDA/CuPy) is optional.
- **Opt-in two-process mode:** `neurofly run --split` runs the simulation and the web
  server as separate processes, and `neurofly sim-serve` and `neurofly web-serve` start
  one side only. One process stays the default. This is a v0.5 prototype.

### Fixed

- `stop()` is idempotent after an acknowledged API shutdown. A later SIGTERM or SIGINT
  no longer writes a false `required_save_failed` row or exits 1.

### Scientific results

- **GPU cohort CPU-reference gate:**
  - **Original fixed-point kernel (`af87006`): FAIL.** First breach at tick 172, a g
    relative error of 1.006e-6.
  - **This candidate: PASS** (new run, same unchanged contract). Arrivals are added one
    at a time in float32, in ascending (pre, edge) order. The CPU uses its own
    active/queue order, so the two are not bit-identical: max g relative error 5.35e-7,
    0 spike mismatches over 200 ticks.
- **Compatibility:** cohorts written under the fixed-point kernel are refused on resume
  by the dynamics signature, and their files are left untouched.
- **Checks that pass:** cross-engine restore, batch invariance, isolation between
  flies, and byte-identical resume.
- **Throughput** (GTX 1660 Ti): batched GPU runs 545–643 brain ticks/s (0.1 ms ticks),
  against about 480 on the CPU. The fixed-point kernel's 4,150–4,850 ticks/s are
  historical.
- **Disclosure:** the cohort's input and output are declared, not native. The encoder
  drives T4/T5 directly, and the DNa02 yaw decoder is an engineered linear readout.

### Not in this release

- Learning.
- Dynamics v4/v5 in cohorts.
- The fly body in cohorts.
- AMD GPU acceleration for cohorts.
- A browser cohort UI.

## [0.4.0] - 2026-10-07

Everything merged to `master` since the `v0.3.0` tag (`b6031b7`, 24 September 2026),
plus the release candidate `cbaa876`. Pull-request numbers refer to
<https://github.com/average-user-887/flybrain/pulls>.

**Scope of this release.** v0.4.0 is a downloadable, **experimental research
instrument**. It does not simulate a whole fly, and it does not show that the model
learns. Its main scientific result is negative (see Scientific results). The CPU path
is the supported one. The AMD GPU engine is included but **experimental and
unqualified**. Docker does not build. Features the dashboard shows but does not yet
support are listed by ID in `docs/POST_V04_FEATURES.md` (NEXT-01 to NEXT-12).
Details: `docs/RELEASE_NOTES_v0.4.0.md`.

### Added

- **CUDA/CuPy GPU backend for LIF v3**, used automatically when a CUDA device is visible;
  `NEUROFLY_BRAIN_BACKEND=cpu|cuda|auto` overrides it. NVIDIA only.
- **LIF dynamics v4**: declared graded (non-spiking) transmission for listed cell
  classes, preregistered before measurement (`docs/LIF_DYNAMICS_SPEC.md` §7).
- **LIF dynamics v5**: per-receptor-class synaptic time constants, preregistered and
  pinned as a new version (`docs/LIF_DYNAMICS_SPEC.md` §9). v1-v4 unchanged.
- **Photoreceptor encoder**: a separately pinned input map that drives only R1-R6, so
  the graph, not the encoder, must compute motion (`docs/PHOTORECEPTOR_ENCODER.md`).
- **Validation harness** (`neurofly validate`) with preregistered specs for optomotor,
  looming escape and T-maze, and firing-rate bound files v1-v3 (#9, #15, #16, #22, #28, #29).
- **Embodied loop features** (`neurofly_body`): bit-identical replay from a seed (#23),
  declared DN decoder `dn-v2` for DNp09, DNa02, MDN and GF (#24), a modular baseline
  controller for the FlyGym body (#26), a run queue, body recording and 1x browser
  replay (#27), opt-in `--silence` for named cell types (#31), opt-in motor delay with
  pipelined brain/body overlap (#32), opt-in leg-load feedback onto campaniform
  sensilla (#33).
- **Bit-identical fast FlyGym controller loop** (about 5x faster body stepping on the
  measured host, `docs/receipts/body_speedup/`).
- **Deterministic run recordings** (`.nfrec`) with 1x replay in the dashboard (#8);
  recordings state whether the brain ran on the CPU or GPU (#13).
- **`neurofly full-sim`**: a multi-task lifelong run across the 14 paradigms (#7).
  Retired before release; see Removed.
- **Optional AMD GPU engine (experimental, unqualified)**: fixed v3 dynamics only,
  through Vulkan (`pip install ".[amd]"`, then `--brain-backend wgpu-amd`). It is
  never chosen automatically and refuses learning, plastic controllers and other
  dynamics. It has been tested only with small fixtures, not on a real AMD device with
  the real graph (`docs/AMD_STATE_ADAPTER.md`).
- **Durable observation recording**: each observation is written to an fsync'd
  journal before it is acknowledged; a damaged journal tail is quarantined, not
  silently dropped, and a failed write halts the run instead of losing data.
- **Per-assay trial clock**, saved with checkpoints and restored after a restart; a
  restart starts a new, explicitly marked measurement window. The dashboard names the
  clock (session, trial or graph) behind each time readout.
- Every graph instance records the compute engine it asked for and the one it got.
- **Science Guide** in the dashboard, separating cited fly research, the engineered
  preview and what v0.4 supports; unsupported controls point to their backlog ID.
- The capability matrix is packaged with the installed CLI (`neurofly capability`).
- `CITATION.cff`, this changelog, `docs/RELEASE_PLAN_v0.4.md`, `docs/OWNER_DECISIONS.md`,
  `docs/LITERATURE_BENCHMARKS.md` and `docs/LITERATURE_SCAN_2026-10-05.md`.

### Changed

- **v3 is the default dynamics everywhere**, always on its own transmitter-policy
  weights, in the daemon, the library and `full-sim` (#7, #10).
- The live dashboard stays responsive when the simulation runs far below real time (#14).
- The daemon keeps only the newest 20 checkpoints per assay (#21).
- The bridge steers with DNa02 L - R, like the daemon and the body decoder (#20).
- Looming escape fires only on a Giant Fiber spike (#22).
- Documentation reset to claim only what the receipts show (#6, #18); the README now
  documents the full data pipeline (download, normalize, prepare, verify), uses an
  HTTPS clone URL, and states plainly what runs without an NVIDIA GPU.
- `NOTICE` now lists the bundled three.js and gives full citations.

### Removed

- **Retired entry points** (`docs/RETIREMENT_INDEX.md`). Each now prints why and exits
  2 without loading data or writing output; the source stays in git history.
  `neurofly full-sim` and `experiments/full_connectome_simulation.py` were not a full
  connectome simulation (empty sensory input; motor output overwritten by the modular
  controller). `experiments/run_paradigm_battery.py` and
  `experiments/whole_brain_scientific_battery.py` drove the surrogate bridge, and the
  latter hardcoded significance claims. `sync_ecosystem.py` had a stale manifest, and
  `scripts/test_srv.py` loaded the full graph on import (now inert).
  `flybrain_scientific_instrument.html` is a notice pointing to the dashboard, and the
  orphaned `web/research/app.js` and `style.css` moved to `docs/archive/`.

### Fixed

- Dashboard truthfulness (October): stale activity from a previous assay is withheld
  after a switch; the gait proxy waits until the displayed brain has stepped; command
  results are matched to the dashboard's own request; a backend request stays
  "unknown" until its own acknowledgement; a refused backend switch becomes history
  once a rebuild is accepted; replay controls, age labels and timing tooltips follow
  playback; the optomotor stimulus phase is kept as scene state; training export is
  labelled as a report, not a brain checkpoint; the sandbox paradigm has one name.
- An explicit `--graph-dir` is honoured in every controller rebuild.
- **`neurofly record --state-dir` ends cleanly**: a normally finished run saved no
  final state and wrote no clean-shutdown marker, so the next run on the same
  directory restored an older checkpoint and reported `interrupted_unclean_shutdown`.
  It now runs the daemon's shutdown transaction (final checkpoint, then
  `session_end`); a failed final save leaves the run incomplete and the last good
  checkpoint current, and a killed process is still reported as unclean.
- **Recording header names the code that wrote it**: `provenance.code` is the run's
  origin from the run manifest, so a recording continuing saved state showed the
  parent's source hashes as if they identified the writer. It is kept unchanged and
  labelled by `provenance.code_scope`; the new `provenance.writer` hashes the code
  actually running (the installed distribution's RECORD-listed files, or git in a
  checkout, plus the loaded module files).
- **Recording header `initial_state.restored` on graph backends**: it read the runner's
  graph-bookkeeping file, so a recording continuing a checkpointed graph instance
  (for example a migrated linked child, which has no bookkeeping file) said
  `restored: false` while starting at the checkpoint's step. It now reports whether the
  registry restored the instance from a checkpoint, and the new
  `initial_state.restore_source` names the restored checkpoint version and step.
  Frames, traces, deltas and history are unchanged; older files still read.
- **No more silent freezes** (audit F). The simulation thread no longer dies on an
  exception outside the step (a full disk during a checkpoint froze the observatory for
  42 minutes behind an "online" status). Any loop failure is an honest halt naming the
  phase. A failed required save (storage error) stops the run at a step boundary and
  marks its result incomplete; GPU/compute errors (also during a checkpoint) and
  NaN/inf state always halt; only `--exploratory` mode keeps stepping "NOT SAVING"
  with a recorded gap. Incidents are append-only (status, manifest, checkpoint meta,
  `run_validity.jsonl`) and survive recovery and restart. A watchdog reports
  `stalled` / `dead` in status, frames and heartbeats, and the page shows a red
  "SIMULATION NOT ADVANCING" pill and a step age instead of LIVE. A rebuild restarts a
  dead thread. Startup falls back to the newest checkpoint that verifies, also when
  `CURRENT.json` is missing, and checkpoint numbers always continue above every file
  on disk; shutdown saves once. New options: `--exploratory`, `--exit-on-stall`,
  `--step-hard-limit`, `--keep-shutdown-checkpoints`, `NEUROFLY_CHECKPOINT_INTERVAL`.
  See `docs/LEARNING_OBSERVATORY.md`, "When something fails".
- Graph-controller node indices are resolved against the loaded graph, not a separately
  read neuron table (`a9c988c`).
- The cached WP5 optomotor loop is bound to the live graph instance, which stopped the
  simulation freezing on the second optomotor visit after an experiment switch
  (`8b8ba85`, evidence in `docs/receipts/switch-race-20261004/`).
- A step error now halts the run honestly and can be recovered from; the dashboard
  shows SIMULATION HALTED instead of LIVE DAEMON while halted (`85f075d`, `ce1d248`).
- `neurofly run` and the WP6 rule's time step (#18); `full-sim` checkpoint,
  weight serialization, heading and plasticity-update defects (24 September).
- Scheduler timing tests no longer depend on CI runner load (#12).

### Scientific results

- **Optomotor, v3, encoder-imposed direction selectivity: PASS_PROVISIONAL** (#19,
  `docs/receipts/validation/optomotor-yaw-v3-2.md`). 7/7 behaviour gates, 5/5
  physiology checks against unverified bounds; silencing DNa02 gives zero yaw. The
  preceding run v3-1 **FAILED** on HS firing above 50 Hz, and v3-2 made that check
  report-only after seeing the failure. The encoder supplies the direction selectivity,
  so this is not evidence that the connectome computes the response.
- **`E_inh` sensitivity**: the v3 optomotor verdict holds at −70 and −60 mV, with a
  third less effect at −60 mV; the verdict at −56 mV is unknown because the gate
  failed there (`docs/EINH_SENSITIVITY.md`). *Corrected 5 October 2026:* none of these
  is a measured adult value. No native adult central chloride reversal was found, so
  −60 and −56 mV are sensitivity points (`docs/EINH_SENSITIVITY.md` §7).
- **Negative, model-specific: our connectome model does not produce direction
  selectivity from photoreceptor input** under v3 (the signal dies at the first synapse), v4 (the signal
  crosses the graph but carries no direction) or v5 (receptor kinetics do not help)
  (`docs/WP5_OPTOMOTOR.md` §13-§14, `docs/receipts/lif_dynamics_v4.json`,
  `docs/receipts/lif_dynamics_v5.json`). Diagnosed causes: a 200-500x signal loss from
  photoreceptors to T4/T5, and T4/T5 inputs whose relative timing is wrong.
  These negatives belong to these model versions. They are not evidence that the
  biological connectome cannot compute motion. A later variant, v6a (branch
  `claude/gain-v6a`, not on `master`), failed its amplitude (stage) gate, so its
  motion test was never run (corrected 5 October 2026).
- **The scan is incomplete in the lamina**: about 55 % of L1-L3 cells receive no
  photoreceptor input in MaleCNS. By owner ruling this is reported as a result and is
  not filled in (`docs/OWNER_DECISIONS.md`, `docs/LITERATURE_SCAN_2026-10-05.md`).
  NeuroFly's engine graph was checked to contain exactly the scan's 25,582,938 edges.
- **Embodied loop**: reversing the visual stimulus reverses the FlyGym body's turn on
  2 of 2 seeds, through one DNa02 pair and an engineered decoder
  (`docs/receipts/embodied_mvp_verification.md`).
- **Reproducibility**: v3 GPU runs repeat spike for spike; v4/v5 GPU runs on the real
  graph do not repeat bit for bit (about 1e-5 mV, same spike count;
  `docs/receipts/lif_dynamics_v5.json`, F1).
- **Cross-machine determinism (CPU backend)**: bit-identical embodied runs on two AMD
  hosts (Zen 3, Python 3.12; Zen 2, Python 3.14) with the same library versions
  (`docs/receipts/determinism-ryzen-vs-amd-20261005.md`).
- **Benchmarks** for true v3 on the reference NVIDIA host, and GPU-vs-CPU parity (#11,
  `docs/receipts/ryzen/`).

### Known limitations

- No connectome paradigm has a non-provisional behavioural pass. The dashboard's
  default `./start_daemon.sh` fly is the hand-built modular controller.
- The AMD GPU engine is experimental and unqualified: no real-device, real-graph or
  browser run has been accepted for this release. Other non-NVIDIA GPUs have no GPU
  path. Without a GPU the CPU path is used, which is slow on the full graph. The
  connectome is not real-time on any measured host.
- The recording limits listed under "Known limitations" in
  `docs/RELEASE_NOTES_v0.4.0.md` (conflicting retries halt the run; outside edits to
  the journal force a slow full re-check; a restart starts a new measurement window).
- `experiments/data/` (the September lesion study report) is legacy output from the
  hand-built heuristic modules (not the connectome graph), not v0.4 evidence; it is no
  longer packaged in the wheel but stays in the repository.
- Not done for this release: the final check on the Steam Deck host and a real AMD
  device run, the exhaustive browser check of every control (110 controls), a working
  Docker image, splitting the daemon into separate processes, and a new GPU engine.
- The DNa02-to-walking link is an engineered decoder, not a ventral nerve cord model;
  the embodied fly moves about 1 mm/s.
- Cross-machine reproducibility is tested only on two AMD CPUs, CPU backend, identical
  library versions.
- The Docker image does not build (`validation/` is not copied). GPU acceleration needs
  a GPU library installed by hand. The daemon listens on all interfaces unless `--host`
  is given.
- Most sensory IO maps (all but optomotor and the WP6 visual-heading map) are declared
  but not verified against annotations.

## [0.3.0] - 2026-09-24

Tagged as `v0.3.0` (`b6031b7`); no changelog was kept before this version.

[0.5.0rc1]: https://github.com/average-user-887/flybrain/compare/v0.4.0...v0.5.0rc1
[0.4.0]: https://github.com/average-user-887/flybrain/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/average-user-887/flybrain/tree/v0.3.0
