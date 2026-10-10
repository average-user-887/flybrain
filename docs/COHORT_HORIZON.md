# Cohort long-horizon characterisation (rc2 Lane 3)

Harness: `scripts/cohort_horizon.py`. Preregistration: `docs/cohort_horizon_prereg.json`
(committed before any real-graph run; the original file had SHA-256 `32bea168216179f752d2a6bd14b0e1e5e7ba6c698a3ee50d2d0c7195485fabdd`; this published copy differs from it only in the device-selection string, where a generic description replaces the private device UUID).
Nothing was tuned and no gate was relaxed.

## No general equivalence claim

This page reports what was observed for a finite set of flies, one stimulus family and two (plus one
declared) windows. **It makes no general long-horizon CPU/GPU equivalence claim, in either direction.**
It does not show that the CUDA engine diverges, and it does not show that it agrees. The cohort contract's
200-tick equivalence result stays exactly what it was (a 200-tick result).

## Clock units

One tick is 0.1 ms. One arena step is 20 ms = 200 ticks. So 2,000 ticks = 200 ms = 10 arena steps, and
20,000 ticks = 2 s = 100 arena steps (the default `--seconds 2.0` cohort). Every "tick" below is a 0.1 ms tick
(1-based: "tick t" = observed after t ticks), never an arena step and never a millisecond.

## What was run

* Real graph (MaleCNS v1.0, v3 transmitter policy, 166,700 neurons), the cohort runner's closed optomotor
  loop (Arena, OptomotorEncoder, DNa02 decoder), 8 flies with seeds 1-8, fresh state.
* The CPU reference (`CpuLoopCohortEngine`, `advance_v3`) drives the loop. The candidate replays the
  identical per-arena-step drive open loop, so a candidate divergence never changes the stimulus.
* Metrics against the cohort contract bounds, unchanged: max|dV| 1e-3 mV, g relative error 1e-6
  (g == 0 in the reference: |g| <= 1e-12). The run does not stop at the first break.

## Section A: EMULATION (CPU only, not CUDA)

Candidate = a CPU emulation of the GPU kernel's arrival order (ascending (pre, edge), per-arrival float32 via
double), after the F2 review's probe. `CUDA_VISIBLE_DEVICES=""`, peak RSS 1.12 GB (guard 3 GB), 1,721 s wall.
**Observation resolution: per tick** (chunk 1): counts of every neuron and the full v, g, refractory state
were compared after every single tick, so the ticks below are exact for this emulation. Raw data:
`result.json` and `curve.csv` in the evidence folder (curve every 200 ticks).

2,000-tick window (200 ms, 10 arena steps), 8 flies:

* No spike divergence in any fly (fraction 0/8). Max |dV| over all flies 7.6e-6 mV (about one float32
  step of v), far under 1e-3 mV.
* g relative error: all 8 flies exceed 1e-6 within the window. Per-fly maxima 1.7e-6 to 3.4e-6. The first
  tick above 1e-6 was between tick 330 and tick 831 (per fly). This is beyond the 200-tick window of the
  contract result (4.5e-7 there); by tick 200 the all-fly maximum was 3.6e-7.

20,000-tick window (2 s, 100 arena steps), 8 flies:

| fly (seed) | first spike divergence (exact tick) | what differed | max g rel before it | max abs dV before it |
|---|---|---|---|---|
| 0 (1) | 11,248 | neuron 143733: reference 0 spikes, emulation 1 | 3.8e-6 | 7.6e-6 mV |
| 1 (2) | 9,117 | neuron 96761: reference 1 spike, emulation 0 | 6.7e-6 | 7.6e-6 mV |
| 5 (6) | 16,794 | neuron 9935: reference 0 spikes, emulation 1 | 1.1e-5 | 7.6e-6 mV |
| 2,3,4,6,7 (3,4,5,7,8) | none observed in 20,000 ticks | | 1.0e-5 to 8.5e-5 over the full window | 7.6e-6 mV |

* Divergence-fraction curve (spike): 0/8 until tick 9,116; 1/8 from tick 9,117; 2/8 from 11,248;
  3/8 from 16,794 to the end. The fraction whose |dV| exceeded 1e-3 mV equals the spike fraction (the first
  break in dV coincides with the first spike mismatch in each of the three flies). The g-relative fraction is
  8/8 from about tick 831 onward.
* Every fly's first spike divergence is a single neuron differing by one spike. After it, state keeps
  drifting: the three diverged flies reach max |dV| 7.0 to 23.5 mV and cumulative mismatched-neuron
  observations of 2 (fly 1, seed 2), 134,890 (fly 0) and 46,010 (fly 5) by tick 20,000. The very large
  "max g relative" values after divergence (up to 1e44) are relative errors against near-underflow
  reference conductances and are post-divergence artefacts; use the "before it" columns for the
  pre-divergence regime.
* Refractory arrays first differ on the same tick as the first spike mismatch.

Caveat for Section A: it is the *emulation* of the order described in `gpu.py` (same libm exp on both
sides), not a measurement of the CUDA engine. It shows that this arrival-order difference alone is enough to
produce a first spike mismatch on the real graph inside the default 2 s run for 3 of 8 seeds.

Declared extension (100,000 ticks = 10 s, seeds 1-4, per tick, emulation): see Section C.

## Section B: actual CUDA (not emulation)

Run on the GTX 1660 Ti only (selected by device UUID, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUPY_GPU_MEMORY_LIMIT=4500000000`), the other GPU invisible. Kernel under test: the rc1 kernel,
`brainlab/cohort/gpu.py` at the branch base `38ec828` (git blob 1b9231c042d877d39d8659b62ed27f401fc93969,
file SHA-256 8949ca2a9354ea0f12c81b2c1670d2d5c6d85fba7610ffc721157a31fa9b698c). Lane 2's later kernel
(9da81f6) was reported byte-identical to it and was not run here. Same real graph, same 8 flies (seeds 1-8),
same closed-loop stimulus replayed to both engines, 20,000 ticks, 448 s wall.

**Observation resolution: 200-tick chunks (one 20 ms arena step).** Spike counts are compared per call and
state at the end of each call. A CUDA first divergence is therefore only known to lie in a 200-tick
range; **no exact tick is claimed**. (No chunk-1 bisection was run.)

2,000-tick window: no spike divergence (0/8); max |dV| 7.6e-6 mV; g relative error above 1e-6 in 8/8 flies
(per-fly maximum at tick 2,000: 1.1e-6 to 2.8e-6; first chunk-end above 1e-6 between tick 600 and 1,800).

20,000-tick window:

| fly (seed) | first spike divergence, in ticks | what differed | max g rel before it | max abs dV before it |
|---|---|---|---|---|
| 0 (1) | within [11,601, 11,800] | 10 neurons, reference 0 vs CUDA 1 spike (e.g. neurons 203, 746, 3421) | 1.0 | 0.34 mV |
| 5 (6) | within [16,801, 17,000] | neuron 9038: reference 1, CUDA 0 | 1.0 | 7.6e-6 mV |
| others (2,3,4,5,7,8) | none observed in 20,000 ticks | | 1.0 | 7.6e-6 mV, except seed 2: 0.057 mV |

* Spike-divergence fraction (chunk resolution): 0/8 up to tick 11,600, 1/8 in the chunk ending 11,800, 2/8
  in the chunk ending 17,000, to the end. Fraction with |dV| above 1e-3 mV: 1/8 by tick 11,400, 2/8 by tick
  11,800 (that includes seed 2, whose |dV| exceeded 1e-3 mV at tick 9,200 without any spike mismatch), 3/8 by 17,000.
* The CUDA g relative error reaches exactly 1.0 for every fly by tick 4,400 (before any spike divergence),
  i.e. CUDA g is 0 where the reference g is non-zero, at some neuron and some chunk end. The EMULATION never
  showed this before divergence (pre-divergence g relative maximum 8.5e-5). So CUDA differs from the
  emulation in at least this respect, and this run does not diagnose why; it is recorded as observed.
* Observed divergences are of different flies and at different ticks from the emulation (emulation:
  seeds 1, 2, 6; CUDA: seeds 1, 6). The emulation is therefore not a prediction of the CUDA outcome per fly.
* After a spike divergence the state drifts (max |dV| about 21 mV); the huge post-divergence relative
  g errors are not meaningful.

Raw data: `result.json`, `curve.csv` in the evidence folder.

## Section C: declared extension (EMULATION, 100,000 ticks = 10 s, seeds 1-4, per tick)

Completed (4,118 s wall, peak RSS 1.01 GB, CPU only), well before the 20:20 deadline. Emulation, not CUDA.
* Seeds 1 and 2 reproduce the 20,000-tick run exactly: first spike divergence at ticks 11,248 and 9,117.
* Seeds 3 and 4: no spike divergence in 100,000 ticks (max g relative 9.0e-5 and 2.6e-5, max |dV| 1.1e-5 mV).
  So the F2 review's tick-28,311 synthetic divergence has no counterpart among these two real-graph seeds up to tick 100,000.
* Fraction of the 4 flies with a spike divergence: 0 until tick 9,116, 1/4 from 9,117, 2/4 from 11,248, 2/4 to tick 100,000.
* Seed 2's divergence was only 2 mismatched neuron-observations in total, then the spike counts agreed again for the
  rest of the 100,000 ticks, while its |dV| reached 7.0 mV and g relative error reached 1.0. Seed 1 kept drifting
  (1.48 million mismatched neuron-observations by tick 100,000, max |dV| 23.8 mV).
* No claim beyond these 4 flies and this stimulus.


## Correction: the CUDA "g relative error 1.0" is a subnormal flush, not a dynamics difference

**What was seen.** In Section B, the CUDA g relative error reaches exactly 1.0 for every fly by tick
about 4,400, before any spike divergence.

**The cause.** This is a comparison artefact.
* CuPy 14.2 always compiles kernels with `-ftz=true` (`cupy/cuda/compiler.py`, the option is appended
  after the caller's). The GPU therefore flushes subnormal float32 values (below 1.18e-38) to zero.
* The CPU reference (numba/numpy) keeps subnormal values.
* An isolated, decaying conductance reaches the subnormal range after about 87 synaptic time constants.
  From then on the reference holds a tiny non-zero g where the GPU holds 0.
* In a 1-fly, 6,000-tick reproduction (seed 1), the first chunk with a relative error of 0.5 or more
  ended at tick 4,400. All 773 g entries over the bound in that chunk had a subnormal reference value
  (at most 1.13e-38) and a GPU value of exactly 0.
* max |dV| stayed at 7.63e-6 mV. The flush had no effect on V or spikes.
* The same result was obtained with the rc2 delivery kernel (`9da81f6`). Its kernel source differs from
  rc1, but its outputs are byte-identical to rc1's.

**The correction.** `compare_state` in `scripts/cohort_horizon.py` now treats reference values below the
smallest normal float32 (`np.finfo(np.float32).tiny`) as zero. These values then fall under the
`|g| <= 1e-12` check, and a GPU 0 passes it.
* The bounds, the CPU reference and the cohort contract are unchanged.
* No engine change was made. Bit-equality of g would need a CPU reference that emulates the flush.

**The raw results are kept as they were.** The original CUDA result files (`result.json`, `curve.csv`)
were produced with the old comparison (`nz = rg != 0`). Their g relative errors of 1.0 before a
divergence are this artefact.

**What the correction does not explain.** The later spike divergences (seed 1 in ticks 11,601-11,800,
seed 6 in ticks 16,801-17,000) and the very large relative g errors after them remain as reported. The
flush does not explain them.

## Reproduction

Command lines are in the evidence folder (`command_*.txt`); graph and connectome directories are supplied by
`--graph-dir` and `NEUROFLY_CONNECTOME_DIR` and are read-only.
