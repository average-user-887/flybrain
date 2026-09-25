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

## Run queue

Runs are slower than real time, so experiments can be queued and left to run
one after another:

```bash
python -m neurofly_body queue add runs/queue optomotor-s1 -- --duration 10 --seed 1
python -m neurofly_body queue add runs/queue modular-s1 -- --controller modular --duration 10 --seed 1
python -m neurofly_body queue run runs/queue            # add --watch 30 to keep polling
python -m neurofly_body queue status runs/queue
```

`add` checks the run arguments right away. Each job runs in its own process,
in the order it was added. Its output goes to `runs/queue/runs/<name>`, its
log to `runs/queue/logs/<name>.log`, and the job file moves from `pending/`
through `running/` to `done/` or `failed/`. A finished job file records the
exit status, wall time, `trajectory_sha256` and real-time factor. If the
worker is killed during a job, the next `queue run` moves that job to
`failed/` rather than running it again, because its output may be partial.

## Replay in the browser

The loop runs slower than real time, so each run also writes `body.nfbody`:
the 3D positions of every body segment, the thorax yaw, the CPG command, leg
contacts, per-side DN rates (connectome controller), graph spikes per frame
and motor events, at `--record-fps` frames per simulated second (default 50;
the frame period must be a whole number of 2 ms neural steps; 0 turns it off).
The file holds no wall-clock data, so a replayed run gives a byte-identical
recording, and `replay-check` compares its frame hash too.

To watch a run at the fly's own speed, open `/embodied_replay.html` on the
dashboard (or `web/embodied_replay.html` from any static server) and pick the
`body.nfbody` file, drop it on the page, or pass `?src=<url>`. Playback is 1x
simulated time by default, with 0.25x to 4x, seeking and pause (space). The page
checks the frame SHA-256 against the file's end record where the browser allows
it (localhost or HTTPS); on a plain-HTTP LAN address it says the file is not
verified.

## Silencing cell types

`--silence CELL_TYPE` (repeatable; `CELL_TYPE:L` or `:R` for one annotated soma
side) clamps those neurons in a connectome run. It is off by default, and a run
without it is byte-identical to one made before the flag existed.

```bash
python -m neurofly_body run --duration 10 --seed 1 --silence DNa02 --output runs/dna02-silenced
```

The semantics match the validation harness's `dna02_silenced` condition. Every
2 ms step, after all sensory drive, each silenced neuron's input current is
replaced by `SILENCE_DRIVE` (-200, `brainlab/io_map.py`). Cell types are
matched on the prepared `cell_type` and sides on the annotated `somaSide`. A
target that resolves to no neurons stops the run before it starts. Under v3
conductance dynamics the clamp can leak, which is why the harness's gate O7
checks it. So every telemetry record counts the spikes of silenced neurons
(`neural.silenced`), and `summary.json` gets a `silenced` block with the
targets, neuron counts, map hash, `spikes_total` and `clamp_held`
(no silenced spike in the whole run). The manifest's `neural_backend.silence_map`
lists the silenced source IDs, and the recording header carries the same
`silenced` summary. `replay-check` repeats the flag. The modular baseline has
no neurons and refuses it.

## Motor delay and brain/body overlap (opt-in)

The default loop is sequential with zero motor latency. The body runs the
command decoded from this step's graph output, so the graph and the body cannot
run at the same time. Overlapping them therefore changes the model: the body
has to run a command from an earlier step. Both options below are off by
default. Without them, a run is byte-identical to one made before they existed
(checked against master on a 1 s modular run).

- `--motor-delay-steps N` adds N neural steps (N x 2 ms) of motor latency. The
  body executes the command decoded N steps earlier and zeros until then. This
  is the sequential reference for the overlapped mode. Each telemetry record
  keeps `decoded_cpg_drive` (this step's decode) and `applied_cpg_drive`
  (what the body ran) and adds `motor.delay_steps`. The manifest gives
  `lockstep.motor_delay_ms`.
- `--pipeline` (needs `--motor-delay-steps` >= 1) runs this step's body
  substeps in a worker thread while the graph computes. The body's input no
  longer depends on the graph's current step. The graph kernels release the
  GIL (`KERNEL_OPTIONS nogil=True`), and so do the GPU and MuJoCo steps. The
  result is bit-identical to the same delay without `--pipeline`, because the
  two steps share no state and the order in which results are joined is
  fixed. Tests and a 1 s FlyGym run confirm this.

Why a 2 ms delay is defensible: real flies are slower than that. The optomotor
response has a pure delay of about 20 ms (Theobald et al. 2010, J Exp Biol
213:1366), so one 2 ms step of latency lies inside the biological delay. It is
still a model change, and it is recorded in every run.

Measured cost:
- **Behaviour:** in a 1 s modular run (seed 1, 4 rad/s drum), the final yaw was
  2.89108 rad at zero latency and 2.89237 rad with a 2 ms delay, a difference of
  0.04 %.
- **Speed:** the modular controller is too cheap to show a speed-up (0.33x real
  time in both modes on a cloud CPU). The speed-up has to be measured with the
  connectome brain on the Ryzen: 2 s runs with `--motor-delay-steps 1`, with and
  without `--pipeline`, compared on `real_time_factor`.

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
