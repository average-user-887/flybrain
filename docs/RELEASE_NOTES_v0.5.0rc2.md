# NeuroFly v0.5.0rc2 release notes (PRERELEASE)

*Second release candidate, 7 October 2026. Base: `v0.5.0rc1` (`a4368a2`).*

> **Experimental prerelease; not biological validation.** This candidate fixes cohort
> resume and other correctness issues found in rc1, makes GPU cohorts faster with
> byte-identical outputs (the kernel source changed), and compresses checkpoints. Everything in
> [`RELEASE_NOTES_v0.5.0rc1.md`](RELEASE_NOTES_v0.5.0rc1.md) still applies unless it is
> changed here.

## If you used rc1: resume picked the CPU

In rc1, `neurofly cohort resume DIR` without `--engine` continued on the **CPU**, even
for a cohort recorded on the GPU. The rc1 workaround was to pass `--engine gpu`.

In rc2, resume uses the engine recorded in the cohort, matched by its exact id. If
that engine is unavailable or unknown, resume refuses before it writes anything. An
explicit `--engine` that differs from the recorded one is still allowed. It is recorded
in the manifest as a mixed-engine numerical realisation, and the earlier segment keeps
its own engine description. A CPU↔GPU continuation is not bit-identical (see below).

## Fixes

- **`cohort verify` without CuPy** prints `FAIL  engine-available` and exits 1.
- **Inputs checked before anything is created:**
  - `step_ms` must be finite, positive and a whole number of 0.1 ms ticks.
  - Non-finite `--seconds` and checkpoint intervals are refused.
  - Both checks happen before any cohort directory exists.
- **Stricter engine-state checks.** A state is refused atomically, for either engine,
  when:
  - its active list disagrees with its active flags;
  - its active list holds duplicates;
  - it has negative refractory counters.
- **Cohort manifest I/O.** The manifest declares only what the cohort uses:
  - the optomotor/yaw I/O and the encoder and decoder descriptions;
  - the daemon's I/O declaration, referenced by version and sha256 only.
- **Writer identity.** Module hashes are taken when the code is loaded. A later change
  to those files on disk is reported as `disk_drift_since_load`. Limit: the wheel
  RECORD and git fields still describe the disk at the time of the call.
- **Opt-in split mode (`--split`):**
  - Snapshots, status and pending requests are tied to one connection.
  - A command still pending when the connection drops resolves as "outcome unknown".
  - After a restart, telemetry from the earlier run is never shown.

## Faster GPU cohorts, byte-identical outputs

In rc1, 80 % of GPU engine time went into delivery: each receiving neuron scanned all
its incoming edges, about 253 edges scanned per arrival applied. rc2 keeps a per-fly
arrival bitmap, so delivery visits only the arrivals present. The order and the
per-arrival float32 rounding are unchanged.

- **The kernel source changed; its outputs are byte-identical** to the rc1 strict
  kernel's on the same device. This was checked for:
  - the per-tick reference workload;
  - cross-engine restore;
  - B = 1, 8 and 32;
  - isolation between flies;
  - resume;
  - three real cohort pairs, with 8/8 recordings and 16/16 checkpoint states.
- **Whole cohort** (`neurofly cohort run --engine gpu`, 8 flies × 2 s, GTX 1660 Ti,
  median of 3): **113.19 s → 42.88 s, 2.64× faster**.
- **Engine only** (`scripts/cohort_bench.py`, 2000 ticks, median of 3): B1 4.6×,
  B8 3.9×, B32 3.7×.
- The integrate phase is now the largest part of engine time.
- **Device memory:** 489.9 MB at B8, up from 464.4 MB.
- **Compatibility:** the dynamics signature is unchanged, so rc1 GPU cohorts still
  resume.
- **Provenance:** the engine description adds `delivery_index=csc-arrival-bitmap-v1`.

**Measured again on this candidate,** with the installed wheels, the same workload
(8 flies × 2 s, `--engine gpu`, GTX 1660 Ti only), interleaved runs and fresh stores,
median of 3:

| Wheel | Process wall (s) | Cohort loop wall (s) |
|---|---|---|
| v0.5.0rc1 | 111.46 | 105.93 |
| v0.5.0rc2 | 42.19 | 36.79 |
| Speed-up | **2.64×** | 2.88× |

## Checkpoint compression

Each checkpoint member is now stored with deterministic deflate, level 1.

- **Size:** a real-graph checkpoint is 1.53 MB, down from 16.6 MB.
- **Time:** writing a checkpoint (encode and fsync) takes about 76 ms per fly, against
  66 ms before. Reading is unchanged.
- **rc1 compatibility:** rc1 checkpoints still verify against their recorded sha256
  and still resume. A committed rc1-format store (`tests/fixtures/cohort_rc1_store`)
  resumes byte-exactly and is left unchanged.

## What CPU/GPU agreement means

The only established CPU/GPU agreement is the preregistered **200-tick (20 ms)**
CPU-reference contract (`brainlab/cohort/cohort_contract.json`). There is **no claim of
long-run CPU/GPU equivalence**.

On the faster kernel the contract passes 6/6. The g relative error is 5.35e-7, and
restore is 4.53e-7, against a preregistered bound of 1e-6. Both values are unchanged
from rc1.

- **Same engine:** resume is byte-identical.
- **Subnormals:** the GPU flushes subnormal float32 values to zero (CuPy compiles with
  -ftz=true); the CPU reference keeps them. No effect on V or spikes was observed within
  the 200-tick contract. The CPU reference and the contract are unchanged.
- **Long-horizon characterisation:** [`COHORT_HORIZON.md`](COHORT_HORIZON.md) describes
  observations beyond the contract window, for a finite set of flies, using emulation and
  CUDA. It makes no general equivalence claim in either direction.
- **CPU↔GPU:** continuation is not exact. It is verified only within the contract
  bounds.

## Known limitations

- In split mode, pause state is not kept across a simulation-process restart; a
  restarted sim resumes advancing.

## Unchanged

- **Out of scope:**
  - learning;
  - dynamics v4/v5 in cohorts;
  - the fly body in cohorts;
  - AMD GPU acceleration for cohorts;
  - a browser cohort UI.
- **Disclosure:** the cohort's encoder drives T4/T5 directly, and the DNa02 yaw decoder
  is an engineered linear readout. There is no claim of native motion computation or
  validated fly behaviour.
