# NeuroFly v0.4.0 release notes

*Text for the GitHub Release, 7 October 2026, built from commit `cbaa876`. Links are
relative to this file (`docs/`); turn them into absolute URLs at the release tag when
pasting into the Release.*

## In short

- **This is an experimental research instrument**, offered for download so others can
  run it, inspect it and check our results. It is not a finished product.
- **It does not simulate a whole fly**, and it does **not** show that the model brain
  learns. Learning controls in the dashboard are not supported in this release.
- **The main scientific result is negative:** driven only from its photoreceptors, our
  model of the fly's visual system does not tell which way things are moving. About
  55 % of the first-stage visual cells (lamina) in the brain scan have no
  photoreceptor input.
- **Supported:** the CPU path on Linux (tested on two x86-64 machines).
  **Included but experimental and unqualified:** the AMD GPU engine.
  **NVIDIA GPU:** optional, tested on one card. **Docker:** does not build.

## What NeuroFly is

NeuroFly runs a spiking model of the complete MaleCNS v1.0 *Drosophila* connectome
(166,700 neurons, 25,582,938 connections) and couples it to an articulated FlyGym/MuJoCo
fly body, with a browser dashboard and headless experiment runners. It is research
software, built to answer one question honestly: **does the wiring of a real fly's
brain, simulated as published, produce the fly's behaviour?**

**So far, it does not.** This release says so, and shows the evidence.

## The brain scan is used as published

- **NeuroFly adds no connections.** On 5 October 2026 the engine graph was checked
  against the MaleCNS source: exactly the same 25,582,938 connections, the identical
  set of (presynaptic, postsynaptic) pairs, and every weight equal to the synapse
  count × 0.275, with its sign from the scan's predicted transmitter
  ([`docs/OWNER_DECISIONS.md`](OWNER_DECISIONS.md)). The model's dynamics versions are
  declared interpretations of how transmission works (for example, v3 gives
  aminergic neurons no fast synaptic effect); they never change what is wired to what.
- **The scan is incomplete exactly where vision starts.** About 55 % of the lamina
  cells that should receive photoreceptor input receive none (L1 54.4 %, L2 54.2 %,
  L3 55.9 %). Nern *et al.* 2025 note that the lamina is the one optic-lobe neuropil
  the dataset does not cover completely
  ([`docs/LITERATURE_SCAN_2026-10-05.md`](LITERATURE_SCAN_2026-10-05.md) §A1).
- **On the real scan, our simulated visual system does not compute the direction of
  motion.** That is the result of this release for these model versions, and it is
  reported as it stands. It is not evidence that the fly's real connectome cannot
  compute motion. By
  owner ruling, no missing connection will be invented to make it work, not even as a
  labelled variant.

## Honest scientific status

- **The dashboard fly you see by default is the hand-built modular controller**, not
  the connectome (`./start_daemon.sh`). The connectome backends run, but none has
  passed a behavioural validation. See the
  [capability matrix](CAPABILITY_MATRIX.md).
- **Optomotor turning with motion detection supplied by us: provisional pass.** When
  the input encoder *imposes* direction selectivity on the T4/T5 motion detectors, the
  v3 model routes the signal to a turn that depends on the DNa02 descending neurons
  (silencing them gives zero yaw). Verdict PASS_PROVISIONAL
  ([`receipts/validation/optomotor-yaw-v3-2.md`](receipts/validation/optomotor-yaw-v3-2.md)):
  the gating bounds were unverified, the HS check was relaxed after an earlier run
  failed on it, and the encoder does the computation the circuit should do. The size
  of the effect depends on an assumed chloride reversal potential: a third smaller at
  −60 mV than at −70 mV, and unknown at −56 mV. These are sensitivity points, not
  measured adult values; no native adult central chloride reversal was found
  (corrected 5 October 2026, `EINH_SENSITIVITY.md` §7)
  ([`EINH_SENSITIVITY.md`](EINH_SENSITIVITY.md)).
- **Our connectome model does not produce direction selectivity from photoreceptor
  input** (a model-specific negative, not proof about the biological connectome) under v3, v4 (graded transmission) or v5 (receptor-class synaptic kinetics)
  ([`WP5_OPTOMOTOR.md`](WP5_OPTOMOTOR.md) §13-§14). Two diagnosed causes: the signal
  shrinks about 200-500x between the photoreceptors and T4/T5 (partly because of the
  lamina gap above), and the inputs to T4/T5 arrive in the wrong relative order.
- **The embodied loop turns the right way, through one pair of neurons.** Reversing
  the stimulus reverses the FlyGym body's turn on 2 of 2 seeds, driven through a single
  DNa02 pair and an engineered decoder that stands in for the ventral nerve cord
  ([`receipts/embodied_mvp_verification.md`](receipts/embodied_mvp_verification.md)).
  The fly moves about 1 mm/s; a real fly walks 10-20 mm/s.
- **Reproducibility.** v3 on the GPU repeats spike for spike. **v4 and v5 on the GPU
  are not bit-reproducible run to run** on the real graph (differences around 1e-5 mV,
  same spike counts; [`receipts/lif_dynamics_v5.json`](receipts/lif_dynamics_v5.json),
  falsifier F1). On the CPU backend, the same commit, seed, data and library versions
  gave bit-identical runs on two different AMD CPUs (Zen 3 and Zen 2) and Python 3.12
  and 3.14 ([`receipts/determinism-ryzen-vs-amd-20261005.md`](receipts/determinism-ryzen-vs-amd-20261005.md));
  Intel, ARM, other library versions and the GPU backend were not tested.

## What's new

- LIF dynamics **v4** (graded transmission) and **v5** (receptor-class kinetics), each
  preregistered before measurement; v3 remains the default.
- A **CUDA/CuPy GPU backend** for v3, used automatically on NVIDIA GPUs.
- A **validation harness** (`neurofly validate`) with preregistered specs.
- **Embodied-loop features**: seed replay, a declared DN decoder for DNp09, DNa02, MDN
  and GF, a modular baseline body controller, run queue and browser replay, cell-type
  silencing, motor delay and leg-load feedback.
- **Recording you can trust or see fail**: observations are written to a journal on
  disk before they are acknowledged; a damaged journal tail is set aside, not dropped;
  a failed save stops the run and says so instead of freezing behind an "online"
  status. Each assay keeps its own trial clock across restarts, and the dashboard names
  which clock each time readout shows.
- **Recordings say what wrote them**: the header names the code that ran, whether a
  saved brain was restored and from which checkpoint, and the requested and actual
  compute engine.
- **A Science Guide** in the dashboard separates cited fly research, the engineered
  preview and what this release supports. Unsupported controls name their backlog ID.
- **Optional AMD GPU engine** (experimental, unqualified; see below).
- **Fixes** to the live dashboard: returning to the optomotor assay after a switch
  no longer freezes the simulation; a step error shows SIMULATION HALTED and can be
  recovered; activity, gait and command results shown always belong to the brain on
  screen; replay controls and labels follow playback.
- `CITATION.cff`, `CHANGELOG.md`, and fuller attribution in `NOTICE`.

Full list: [`CHANGELOG.md`](../CHANGELOG.md).

## Install

Follow the [README](../README.md#installation): clone over HTTPS, `pip install -e
".[body,test]"`, then download, normalize, prepare and verify the MaleCNS data
(about 1.1 GB, CC BY 4.0).

The wheel attached to the Release (`neurofly-0.4.0-py3-none-any.whl`) contains the
program only, no connectome data. The data download and preparation steps run from
a source checkout, so the README route above is the one to follow.

- **Any machine runs on the CPU**, NVIDIA or not. On a Steam-Deck-class AMD CPU the
  full brain runs at about 1/40 of real time with the default v3 dynamics (about 1/670
  with the experimental v4), using about 1.2 GB of RAM; the hand-built modular
  controller runs faster than real time.
- **GPU acceleration needs an NVIDIA GPU with CUDA** and a GPU library (CuPy or
  numba's CUDA support) that the standard install does not provide; there is no
  tested one-line GPU install yet.
- **AMD GPU: present, but experimental and unqualified.** An optional engine runs the
  default v3 model on AMD GPUs through Vulkan (`pip install -e ".[amd]"`, then
  `neurofly run --backend connectome-fixed --dynamics v3 --brain-backend wgpu-amd`).
  It is never selected automatically, refuses learning and other model versions, and
  has only been tested with small test fixtures. No run on a real AMD GPU with the
  real graph has been accepted for this release, so its results are not qualified
  ([`AMD_STATE_ADAPTER.md`](AMD_STATE_ADAPTER.md)). Intel GPUs and Apple Silicon run
  on the CPU.
- **Start with `neurofly run --backend modular --host 127.0.0.1`** to see the dashboard
  without any data. The connectome backends need the prepared data, and the daemon
  listens on all interfaces unless you pass `--host`.
- **Docker does not build in this release**; use the editable install.

## Known limitations

- No connectome paradigm has a non-provisional behavioural pass.
- The connectome is not real-time on any measured host; the CPU path is slow on the
  full graph.
- Most sensory input maps are declared but not verified against annotations.
- The DNa02-to-walking link is engineered, and the embodied fly does not walk well.
- Cross-machine reproducibility is measured only on two AMD CPUs with the CPU backend.
- The Docker image does not build. NVIDIA GPU support needs the `gpu` extra; AMD GPU
  support is unqualified (above).

Recording and restart limits (what you may notice in a long run):

- **A conflicting retry stops the run.** If an old observation is sent again with
  different content, the recorder refuses it, keeps the earlier record unchanged and
  halts the run. Recovery needs an explicit action; nothing restarts on its own. The
  dashboard should then show a halted or incomplete state, never "online".
- **Some exact retries wait.** A repeated observation the recorder no longer has in
  memory shows as "pending" until it is checked, instead of "saved" at once. It can
  fill the queue (`ObservationQueueFull`); nothing is dropped.
- **Editing the journal from outside forces a full re-check**, which takes about
  0.13 s per MB. Above about 220 MB of history one re-check may exceed the 30 s stall
  limit. An edit that keeps the same size within the same file-timestamp tick is only
  noticed at the next open.
- **After a restart**, a scan of up to 4 MiB may happen while saving; this limits the
  bytes read, not the time taken. The measurement window starts again and is marked
  as a break; an earlier segment whose status cannot be established shows "unknown".
- **After restoring a save**, a trial runs one extra full observation window. If a
  graph trial ends between checkpoints, it can be counted again after a restore.
  Trial numbers in saves from older versions stay unknown.
- Scientific runs without a recorder halt or refuse to finish cleanly, by design.

## Not supported in v0.4

The dashboard shows some controls and experiments that this release does not
support. Each is tracked in [`POST_V04_FEATURES.md`](POST_V04_FEATURES.md):

- NEXT-01 connected assay controls that are preview-only (threat placement, spatial
  stimulus editing);
- NEXT-02 lesion and intervention tools;
- NEXT-03 connectome learning controls (teach, reverse, probe, freeze);
- NEXT-04 graph training history and brain export/import;
- NEXT-05 reproducible whole-connectome experiment batteries;
- NEXT-06 motion computation from photoreceptors, saccadic efference copy, richer
  cell dynamics;
- NEXT-07 working memory, place memory, operant memory, labyrinth planning;
- NEXT-08 circadian rhythm, moving courtship partner, courtship memory;
- NEXT-09 turbulent odor plumes and richer environments;
- NEXT-10 biological body control, gap crossing, multisensory coordination;
- NEXT-11 broader AMD support and performance;
- NEXT-12 a unified research workspace and dependable distribution.

## Not done for this release

These were planned checks or features, deferred by the owner on 7 October 2026. The
release makes no claim that depends on them:

- the final check on the Steam Deck host and a run on a real AMD GPU;
- the exhaustive browser check of all 110 dashboard controls (the main browser
  checks, all 14 assays, rapid switching and two tabs, were run);
- a working Docker image;
- splitting the daemon into separate processes;
- a new GPU engine.

## Data and credit

The connectome data come from the MaleCNS project (FlyEM at HHMI Janelia, University
of Cambridge, MRC LMB and Google Research), CC BY 4.0,
<https://male-cns.janelia.org/download/>. Please cite Berg *et al.* (2026), *Cell*
189(18):5504-5526.e15, <https://doi.org/10.1016/j.cell.2026.08.015>. The engine's
parameters follow Shiu *et al.* (2024), *Nature* 634:210-219; the body is NeuroMechFly
v2 via FlyGym (Wang-Chen *et al.* 2024, *Nature Methods* 21:2353-2362). See
[`NOTICE`](../NOTICE) and [`CITATION.cff`](../CITATION.cff).

## Feedback

Please report problems, confusing documentation and questions in
[GitHub Issues](https://github.com/average-user-887/flybrain/issues). Once Discussions
are enabled, use them for open-ended questions and ideas. Reports from machines we have
not tested (macOS, Windows/WSL2, AMD or Intel GPUs) are especially welcome.
