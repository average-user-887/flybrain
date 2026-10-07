# NeuroFly v0.5.0rc1 release notes (PRERELEASE)

*Release candidate, 7 October 2026. Base: tag `v0.4.0` (`5bd16d6`).*

> **PRERELEASE.** This candidate is not a release. Its new GPU cohort engine **fails**
> its preregistered CPU-reference gate (below). Whether v0.5.0 ships with this result,
> and how it is described, waits on the owner's decision.

## In short

- **New: `neurofly cohort`** runs many independent fixed-weight v3 brains (a cohort of
  flies) in the optomotor assay, on the CPU or, optionally, on an NVIDIA GPU.
- **New, opt-in: `--split`** runs the simulation and the web server as two processes.
  The default is still one process.
- **Fixed:** stopping the daemon (SIGTERM or SIGINT) after a clean API shutdown no
  longer reports a false save failure.
- Everything v0.4.0 says about what NeuroFly is and is not still holds. It does not
  simulate a whole fly, and it does not show that the model brain learns.

## Cohort runs

A cohort is B flies that share one read-only copy of the MaleCNS graph. Each fly has
its own seed (`--seed-base` + k), its own arena, its own encoder state and its own
decoder. Weights are fixed (v3 dynamics, no learning). The only assay is `optomotor`.

```
neurofly cohort run --out DIR [--flies 8] [--seconds 2.0] [--seed-base 1] \
                    [--engine cpu|gpu] [--checkpoint-every-ms MS] \
                    [--graph-dir DIR] [--connectome-dir DIR]
neurofly cohort resume DIR [--seconds S] [--engine cpu|gpu]
neurofly cohort verify [--engine cpu|gpu]
```

- `run` writes a new cohort into an empty directory: per-fly recordings, checkpoints
  and `cohort_manifest.json`.
- `resume` verifies a cohort directory and continues it exactly, on either engine.
- `verify` runs the preregistered CPU/GPU numerical contract
  (`brainlab/cohort/cohort_contract.json`).
- `--engine gpu` needs an NVIDIA GPU with CuPy. The CPU engine is the reference.

**Data, and running from the wheel.** The wheel contains the program only. The
connectome data are prepared once, in the matching source checkout, with the same steps
as v0.4.0:

```bash
git clone https://github.com/average-user-887/flybrain.git
cd flybrain
git checkout v0.5.0rc1
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
neurofly download-data          # about 1.1 GB, hash-checked (CC BY 4.0)
python -m brainlab.connectome   # normalizes the tables
python -m brainlab.prepare      # builds the engine graph
neurofly status                 # the graph line should end in (VERIFIED)
```

To run cohorts from the installed wheel in another environment, export the two data
locations once, pointing at that checkout. Every cohort command then reuses the same
prepared data:

```bash
export NEUROFLY_GRAPH_DIR=/path/to/flybrain/outputs/brainlab/malecns_v1
export NEUROFLY_CONNECTOME_DIR=/path/to/flybrain/connectome_data/malecns_v1
neurofly cohort run --out runs/cohort-a --flies 8 --seconds 2
neurofly cohort resume runs/cohort-a --seconds 2
neurofly cohort verify --engine gpu      # NVIDIA + CuPy only
```

How each command finds the data:
- `verify` has no directory options. It reads only these two variables.
- `run` also accepts `--graph-dir` and `--connectome-dir`. These options take precedence
  over the variables.
- `resume` loads the graph in the same way (options, else the variables), not from a
  path stored in the cohort. It then checks the loaded graph and I/O map against the
  `graph_sha256` and `io_map_sha256` recorded in `cohort_manifest.json`, and refuses to
  continue if either differs.
- Without the variables, each command falls back to directories inside the installed
  package, which hold no data.

**What the cohort does not show.** The input and output are declared, not computed by
the connectome. The `OptomotorEncoder` injects motion-selective drive directly into
T4/T5: the encoder imposes the direction, and photoreceptor motion computation is
bypassed. The `DNa02YawDecoder` is an engineered linear readout. There is no claim of
native motion computation or validated fly behaviour. `cohort run` prints this notice
and writes it into `cohort_manifest.json`.

## GPU cohort engine: measured results

These were measured on one host with a GTX 1660 Ti (6 GB).

### CPU-reference gate: **FAIL**

The gate was preregistered before measurement, and nothing was tuned afterwards.

- **FAIL:** 7 g points exceed 1e-6, worst 1.047e-6.
- 0 spike or refractory mismatches over 200 ticks.
- Max |ΔV| 1.14e-5 mV.
- The first breach was at tick 172, fly 4: g relative error 1.006e-6 against a bound of
  1e-6.
- The cause: the CPU accumulates in float32 and the GPU in fixed point.

The spike and voltage figures are descriptive and do not change the FAIL.

### Checks that pass

- Cross-engine restore works in both directions.
- Results do not depend on batch size: with B = 1, 8 and 32, all 32 flies are
  byte-identical.
- Flies are isolated from each other.
- Resume is byte-identical.
- Refusals are atomic.

### Throughput

**Units.** The brain advances in **ticks of 0.1 ms**. The cohort loop runs the Arena in
**steps of 20 ms**, and each Arena step is 200 brain ticks. The summary that `neurofly cohort run`
prints counts "fly-steps/s" in 20 ms Arena steps, including the Arena, encoder and
decoder work.

**How it was measured.** `scripts/cohort_bench.py` steps the engine alone, with the
preregistered constant drive. It runs 2000 ticks (0.2 simulated seconds) per fly per
run, in calls of 20 ticks. Each figure is the median of 3 runs on the GTX 1660 Ti and
on this host's CPU.

**Not included.** The Arena, encoder and decoder work in a real `cohort run` is not
timed. A real run is therefore somewhat slower than these figures.

**Columns.**
- *Brain ticks/s* is fly × ticks per wall second, summed over the cohort.
- *Arena steps/s* is the same rate in 20 ms Arena steps (brain ticks/s ÷ 200).
- *Simulated s per wall s, per fly* is how far each fly gets in one wall-clock second.

| Engine | Flies (B) | Brain ticks/s (0.1 ms) | Arena steps/s (20 ms) | Simulated s per wall s, per fly |
|---|---|---|---|---|
| CPU | 1 | 484 | 2.4 | 0.048 |
| CPU | 8 | 474 | 2.4 | 0.0059 |
| CPU | 32 | 473 | 2.4 | 0.0015 |
| GPU, serial (8 runs of B1) | 8 | 4068 | 20.3 | 0.051 |
| GPU, serial (32 runs of B1) | 32 | 4041 | 20.2 | 0.013 |
| GPU, batched | 1 | 4151 | 20.8 | 0.415 |
| GPU, batched | 8 | 4803 | 24.0 | 0.060 |
| GPU, batched | 32 | 4854 | 24.3 | 0.015 |

- **Where the numbers come from.** The rows for CPU B1/B8, serial 8 and batched B1/B8/B32
  were measured at `35859da`. CPU B32 and serial 32 were measured at the release
  candidate, with the machine otherwise idle (load average about 1) and after the GPU
  tests.
- **Serial and batched B1 are the same thing.** One run of B1 is the batched B1 row.
- **Peak device memory in use:** 1.05 GB batched (B32) and 0.42 GB serial.

- **CPU versus GPU:** batched GPU reaches about 4,800 brain ticks/s, against about 480 on
  the CPU.
- **Batching:** batching adds only about **1.2×** over running the same flies one after
  another on the GPU: 4803 against 4068 for 8 flies, and 4854 against 4041 for 32. Most of the gain comes from the GPU kernel itself, not from
  batching.
- **Start-up costs:**
  - GPU start-up: 0.2–0.65 s.
  - Graph upload: 0.2–0.4 s.
  - Kernel compile: 0.02 s (cached).
  - Warm-up: 3–54 ms.
  - Graph preparation on the host: 1.8 s.

## Opt-in two-process mode (`--split`)

`neurofly run --split` (or `--process-mode split`) starts a headless simulation process
and a separate web process (dashboard, REST, SSE). They are connected by a Unix socket.

- `neurofly sim-serve` and `neurofly web-serve` start one side only.
- **The default is unchanged:** one process. `--single-process` is accepted as an
  explicit no-op.
- The child processes exit with the launcher, even if the launcher is killed. On Linux
  the simulation still does its final save.
- **Bounded queue:** at most 64 commands can be waiting. A full queue is refused
  explicitly and never applied.
- While the simulation is unreachable, `/api/command_ack` answers "unknown", never
  success.
- **Status:** this is a v0.5 prototype.

## Fix: idempotent stop

A SIGTERM or SIGINT that arrived after an acknowledged `{"action": "shutdown"}` used to
queue a second shutdown that nothing applied. The daemon then waited out the timeout,
wrote a false `required_save_failed` row and exited with code 1.

`stop()` now returns the earlier acknowledged result. A failed or unfinished shutdown
keeps the existing path.

## Out of scope for this release

- Learning (plastic weights) in cohorts or anywhere else.
- LIF dynamics v4 and v5 in cohorts. Cohorts are fixed v3 only.
- The fly body. Cohorts use the tethered arena loop, not FlyGym/MuJoCo.
- AMD GPU acceleration for cohorts. The GPU engine is CUDA/CuPy only.
- A browser UI for cohorts. Cohorts are command line only.
