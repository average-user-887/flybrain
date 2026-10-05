# Changelog

All notable changes to Project NeuroFly are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Scientific results, including negative
ones, get their own section, because for this project they matter as much as code.

## [0.4.0] - Unreleased

Everything merged to `master` since the `v0.3.0` tag (`b6031b7`, 24 September 2026).
Pull-request numbers refer to <https://github.com/average-user-887/flybrain/pulls>.

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
- No AMD or other non-NVIDIA GPU path; those machines run on the CPU, which is slow on
  the full graph. The connectome is not real-time on any measured host.
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

[0.4.0]: https://github.com/average-user-887/flybrain/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/average-user-887/flybrain/tree/v0.3.0
