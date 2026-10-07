# NeuroFly v0.5.0rc1 release notes (PRERELEASE)

*Release candidate, 7 October 2026. Base: tag `v0.4.0` (`5bd16d6`).*

> **Experimental prerelease; not biological validation.** The first GPU cohort engine
> (fixed-point arrivals, frozen at `af87006`) **failed** its preregistered CPU-reference
> gate. This candidate replaces its arrival arithmetic and **passes** the unchanged gate,
> at about one eighth of the earlier GPU speed (below).

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
- `resume` verifies a cohort directory and continues it.
  - **Same GPU engine:** the continuation is byte-identical to an uninterrupted run.
  - **CPU↔GPU continuation is not bit-identical.** It is verified only within the
    contract's stated bounds: g relative error ≤ 4.5e-7 and 0 spike mismatches in the
    tick 100–199 window after the switch.
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

Measured on one host with a GTX 1660 Ti (6 GB), the only device visible to the runs.

### Arrival arithmetic (what changed and what did not)

- **This engine:** each neuron that receives spikes adds them one at a time,
  `g = float32(double(g) + double(w) × g_unit)` (inhibitory: minus), in ascending
  (presynaptic index, edge index) order.
- **CPU reference (unchanged):** the same per-arrival float32 rounding, but in its
  history-dependent active/queue order, which is not in general ascending.
- The two are **not bit-identical**. On the contract workload they agree to a g relative
  error of at most 5.35e-7.
- Equations, constants, weights, delay, refractory period and phase order are unchanged.
- **Superseded:** the first GPU cohort kernel (`af87006`) accumulated arrivals as
  deterministic quantized 64-bit fixed-point values, rounded on folding once per tick.
- The cohort dynamics signature now records the arrival arithmetic. A cohort written
  under the fixed-point kernel is refused on resume, before anything is restored or
  written, and its files are left untouched.

### CPU-reference gate

The gate (`tests/cohort_contract.json`) was preregistered before any run. The reference,
stimulus, seeds, window, tolerances and predicates are unchanged.

- **Original run, fixed-point kernel (`af87006`): FAIL.** The first breach was at tick
  172, fly 4: g relative error 1.006e-6 against a bound of 1e-6. Descriptively over 200
  ticks: 7 g points above 1e-6 (worst 1.047e-6), 0 spike or refractory mismatches, max
  |ΔV| 1.14e-5 mV. That result stands.
- **Cause:** the CPU rounds to float32 after each arrival. The fixed-point kernel used
  deterministic quantized fixed-point arrivals, rounded on folding. At the first
  divergent neuron, 64 arrivals at tick 104 left an 11-ulp difference, which persisted
  through decay. In that one 64-arrival example the fixed-point result agreed with an
  exact double sum. This is an example, not a general proof.
- **This candidate (per-arrival float32): PASS**, a new run.
  - 0 spike mismatches.
  - Max |ΔV| 7.6e-6 mV.
  - Max g relative error 5.35e-7.
  - Refractory exact over all 200 ticks.

### Checks that pass (this candidate)

- Cross-engine restore works in both directions within the contract bounds: max g
  relative error 4.5e-7 and 0 spike mismatches over ticks 100–199. It is not
  bit-identical.
- Results do not depend on batch size: with B = 1, 8 and 32, all 32 flies are
  byte-identical.
- Flies are isolated from each other.
- GPU resume is byte-identical.
- Refusals are atomic.
- Fixed-point-era cohorts are refused.

### Throughput

**Units.** The brain advances in **ticks of 0.1 ms**. The cohort loop runs the Arena in
**steps of 20 ms**, and each Arena step is 200 brain ticks. The column *20 ms
step-equivalents/s (engine only)* is derived: brain ticks/s ÷ 200. It is not a measured
count of Arena-loop steps. The summary that `neurofly
cohort run` prints counts "fly-steps/s" in 20 ms Arena steps, including the Arena,
encoder and decoder work.

**How it was measured.** `scripts/cohort_bench.py` steps the engine alone, with the
preregistered constant drive. It runs 2000 ticks per fly per run, in calls of 20 ticks.
Each figure is the median of 3 runs. The real Arena, encoder and decoder work is not
timed.

| Engine | Flies (B) | Brain ticks/s (0.1 ms), measured | 20 ms step-equivalents/s (engine only) | Simulated s per wall s, per fly |
|---|---|---|---|---|
| CPU | 1 | 484 | 2.4 | 0.048 |
| CPU | 8 | 474 | 2.4 | 0.0059 |
| CPU | 32 | 473 | 2.4 | 0.0015 |
| GPU, serial (8 runs of B1) | 8 | 536 | 2.7 | 0.0067 |
| GPU, serial (32 runs of B1) | 32 | 535 | 2.7 | 0.0017 |
| GPU, batched | 1 | 545 | 2.7 | 0.055 |
| GPU, batched | 8 | 643 | 3.2 | 0.0080 |
| GPU, batched | 32 | 606 | 3.0 | 0.0019 |

- **Where the numbers come from:**
  - GPU rows: this candidate (`d7ce108`).
  - CPU rows: reused, because the CPU code is unchanged. B1/B8 were measured at
    `35859da` and B32 at `af87006`.
- **This GPU engine is only about 1.1–1.4× faster than the CPU.** The new delivery
  pass walks every incoming edge of each receiving neuron in a single thread.
- **Historical, not this engine's speed:** the fixed-point kernel measured about 4,150
  to 4,850 brain ticks/s at B1/B8/B32.
- **Invalid:** an interrupted benchmark of the probe also opened a context on the
  desktop GPU. It is labelled invalid and not used.
- **Peak device memory:** 1.03 GB (B32) and 0.42 GB (B1).
- **Start-up costs:**
  - GPU start-up: about 3.1 s, mostly building and uploading the graph and its
    incoming-edge index.
  - Kernel compile: 0.15 s.
  - Warm-up: 3–62 ms.
  - Host graph preparation: 2.2 s.

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
