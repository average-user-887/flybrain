# Embodied MaleCNS MVP

This guide describes the current `neurofly_body` executable: a verified
MaleCNS v1.0 graph steps in lockstep with an articulated FlyGym body. It is a
small causal integration experiment, not a full biological fly simulation.

## Requirements

- Python 3.12, 3.13 or 3.14 for the body extra. FlyGym 2.1.0 declares
  `>=3.12,<3.15`; NeuroFly's core package still supports Python 3.11.
- A working MuJoCo/FlyGym install. The `body` extra pins FlyGym 2.1.0 and
  MuJoCo 3.9.0.
- The MaleCNS v1.0 source tables and prepared graph. The source tables total
  about 1.1 GB and are not distributed in this repository.

Install from a checkout:

```bash
python -m pip install -e ".[body]"
```

The same extra can be installed with `pip install neurofly[body]` after a
published package is available. On Python 3.11 the extra's FlyGym and MuJoCo
requirements are skipped by environment markers; use Python 3.12–3.14 for the
body runtime.

## Prepare the connectome

The current download and preparation commands use paths relative to a NeuroFly
source checkout. Run them there:

```bash
python -m brainlab.download
python -m brainlab.connectome
python -m brainlab.prepare
```

The downloader checks the source file hashes recorded in
`data-provenance/malecns_v1/source.lock.json`. Preparation writes the verified
graph to `outputs/brainlab/malecns_v1/graph.npz`; the normalized neuron table
is written under `connectome_data/malecns_v1/normalized/`. No MaleCNS graph or
source table is included in the wheel.

The wheel includes the dataset registry and graph identity pins required at
runtime. To run the installed wheel away from the checkout, prepare the data
from a source checkout first, then pass the resulting graph and connectome
directories explicitly with `--graph-dir` and `--connectome-dir`. The legacy
download/import/prepare commands still write under the checkout-root layout;
they are not yet a relocatable installed-data manager.

## Run

An intact run:

```bash
python -m neurofly_body run --duration 2 --output runs/embodied-intact \
  --mode intact --graph-dir outputs/brainlab/malecns_v1 \
  --connectome-dir connectome_data/malecns_v1 --seed 1
```

The causal control with the same initial seed:

```bash
python -m neurofly_body run --duration 2 --output runs/embodied-disconnected \
  --mode output-disconnected --graph-dir outputs/brainlab/malecns_v1 \
  --connectome-dir connectome_data/malecns_v1 --seed 1
```

To record an offscreen video, add `--video`; the CLI writes `body.mp4` inside
the output directory. Keep each run in a separate output directory. The CLI
refuses to substitute a synthetic graph for missing real data.

Compare the two runs using their saved manifests and time series. The
output-disconnected condition severs the decoded connectome command to the
body controller while leaving the rest of the setup in place. It is an
output-path control, not a biological lesion. A matched pair is a first
causal check; it does not by itself establish behavioral competence.

## What the loop means

The prepared graph contains 166,700 retained annotated neuronal entries and
25,582,938 directed edge pairs. Each graph edge has a synapse-contact count as
its weight. Edge pairs are not individual synaptic contacts, and the graph
does not encode receptor-specific kinetics or measured motor commands.

The embodied run uses the declared LIF v3 dynamics and the
`v3-modulatory-only` transmitter policy. Each 2 ms neural update advances the
model in 0.1 ms steps;
the MuJoCo body also advances in 0.1 ms steps. Engineered assistance currents
to named descending neurons are disabled.

The current interface makes two explicit engineering choices:

- DNa02 left/right spike rates are mapped to the opposite-side leg CPG
  magnitude commands used by FlyGym's stock hybrid turning controller. The
  gain, smoothing time and cap are controller parameters, not fitted
  biological values. Zero DNa02 rates produce zero CPG drive; there is no
  hidden tonic drive or minimum walking speed.
- A retinal-slip proxy is computed from commanded world angular velocity minus
  measured body yaw velocity and fed to the connectome's visual input. It is
  not a rendered retinal image or a calibrated optic-flow pathway.

FlyGym supplies the articulated body, physics and stock locomotion controller.
The adapter does not model the biological ventral nerve cord, muscles,
proprioceptive pathways or their synapses. The connectome output-to-CPG bridge
is engineered and should be read as a testable interface hypothesis.

## Limits and next work

- Synapses are fixed; this executable has no learning rule or learning claim.
- The direct graph-to-controller mapping is not a validated descending-neuron
  to motor-neuron pathway.
- The visual feedback is a scalar slip proxy, not a calibrated sensory model.
- The whole-connectome LIF dynamics and the engineered control interface have
  not been validated against fly behavior.
- Assay experiments, including the planned 14-assay learning work, are outside
  this MVP.

Useful source references: `brainlab/graph_identity.py` (identity checks),
`brainlab/transmitter_policy.py` (transmitter mapping),
`neurofly_body/decoder.py` (engineered CPG mapping), and
`neurofly_body/flygym_body.py` (FlyGym adapter).
