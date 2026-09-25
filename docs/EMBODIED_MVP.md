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

## Determinism and replay check

`telemetry.jsonl` holds only simulated quantities, so the same code, arguments,
seed, graph and brain backend give a byte-identical file. Its SHA-256 is the
run's `trajectory_sha256` (in `summary.json` and `manifest.json`). Wall-clock
measurements (`elapsed_ms` from the graph, time per step) go to `timing.jsonl`.
The manifest also stores the run arguments under `invocation`.

To check a finished run, re-run it from its manifest:

```bash
python -m neurofly_body replay-check runs/embodied-intact --output runs/embodied-intact-replay
```

This writes `replay_check.json` with the verdict `BIT_IDENTICAL` or
`DIVERGED`, both hashes and the first telemetry record that differs. The command
exits 1 when the runs diverge. CPU and GPU brains agree statistically but not
bit for bit, so run the replay on the same backend. The receipt records both
backends.

`ConnectomeServer.reset()` returns the server to its freshly built state. That
includes the GPU copy of the brain state and the optomotor encoder's random
stream, which restarts from `optomotor_seed`. So a second run in the same
process matches a run on a new server.

## Modular baseline controller

`--controller modular` drives the same FlyGym body with the arena's
phenomenological optomotor model (`vision.CompoundEyeVision`) instead of the
connectome. This is the researcher baseline named in the roadmap. It is not
derived from the connectome and makes no biological claim:

```bash
python -m neurofly_body run --controller modular --duration 10 --output runs/modular-10s --seed 1
```

It gets the same retinal-slip input as the connectome, with the drum at
infinity so only rotation matters. It walks with a tonic amplitude
(`--modular-forward-drive`, default 1.0) and turns by shrinking the amplitude
of the legs on the side it turns toward, scaled by the arena's clipped yaw
bias (`--modular-turn-gain`). Both values are assumptions. The manifest
records `controller_kind: modular-baseline`, and `replay-check` works as it
does for connectome runs. On the laptop CPU, a 1 s run with a 4 rad/s drum
walked forward and turned counter-clockwise with the drum at about
4.5 rad/s, at 0.38x real time. The replay was bit-identical.

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

- The default decoder `dn-v2` (`neurofly_body/decoder.py`,
  `DNCommandDecoder`) maps descending-neuron rates to the signed left/right
  commands of FlyGym's hybrid turning controller. The signs and sides come
  from the literature, but the gains are assumptions:

  | DN | Mapping | Source for the sign and side | Gain |
  |---|---|---|---|
  | DNp09 | forward amplitude of the opposite-side legs, so one active P9 turns the fly toward its own side | Bidaye et al. 2020, Neuron | 0.02 per Hz, assumed |
  | DNa02 | multiplies the same-side amplitude by `max(0, 1 - k·rate)`, which shortens the strides on that side; alone it produces no walking | Yang et al. 2024, Cell; Rayshubskiy et al. 2025, eLife | k = 0.01 per Hz, assumed |
  | MDN | when its drive beats the forward drive, both sides go negative (reverse stepping) | Bidaye et al. 2014, Science | 0.02 per Hz, assumed |
  | GF (DNp01) | any spike logs a `takeoff_command` event | von Reyn et al. 2014, Nat Neurosci | none |

  The rates are filtered with a 50 ms time constant (assumed), and the
  command is clipped to ±1.2, the NeuroMechFly v2 drive range. No published
  calibration from firing rate to speed exists for these neurons, so no gain
  was fitted to behaviour. Zero spikes give zero drive, and there is no tonic
  term. Some things are reported rather than faked: the legs-only body cannot
  take off, FlyGym's reverse replays the forward step backwards instead of the
  hindleg-led MDN program, and speed changes through stride amplitude only.
  The sides of DNp09, MDN, GF and DNa02 are resolved from annotated soma sides
  (`brainlab/io_map.py`, `resolve_locomotion_dns`). The run refuses to start
  when that map is missing.
- `--decoder dna02-crossed-v1` keeps the earlier MVP mapping, in which the
  DNa02 rates drive the opposite-side legs and DNa02 is the only source of
  propulsion. It contradicts Yang et al. 2024 and is kept only so that
  earlier runs can be reproduced. `replay-check` uses it for runs recorded
  before `--decoder` existed.
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
`neurofly_body/decoder.py` (declared DN-to-CPG decoders), and
`neurofly_body/flygym_body.py` (FlyGym adapter).
