# NeuroFly v0.4.0 — draft release notes

*Draft text for the GitHub Release. Links are relative to this file (`docs/`); turn
them into absolute URLs at the release tag when pasting into the Release.*

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
- **Fixes** to the live dashboard: returning to the optomotor assay after a switch
  no longer freezes the simulation, and a step error now shows SIMULATION HALTED and can be recovered.
- `CITATION.cff`, `CHANGELOG.md`, and fuller attribution in `NOTICE`.

Full list: [`CHANGELOG.md`](../CHANGELOG.md).

## Install

Follow the [README](../README.md#installation): clone over HTTPS, `pip install -e
".[body,test]"`, then download, normalize, prepare and verify the MaleCNS data
(about 1.1 GB, CC BY 4.0).

- **Any machine runs on the CPU**, NVIDIA or not. On a Steam-Deck-class AMD CPU the
  full brain runs at about 1/40 of real time with the default v3 dynamics (about 1/670
  with the experimental v4), using about 1.2 GB of RAM; the hand-built modular
  controller runs faster than real time.
- **GPU acceleration needs an NVIDIA GPU with CUDA** and a GPU library (CuPy or
  numba's CUDA support) that the standard install does not provide; there is no
  tested one-line GPU install yet.
- **There is no AMD or other non-NVIDIA GPU path** in this release.
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
- The Docker image does not build; GPU support needs a manual library install.

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
