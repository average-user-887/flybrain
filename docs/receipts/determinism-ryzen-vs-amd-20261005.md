# Cross-machine determinism: reference host vs AMD test host (gate G11)

Date: 2026-10-05. Gate: G11 in [`RELEASE_PLAN_v0.4.md`](../RELEASE_PLAN_v0.4.md),
"Cross-machine determinism stated". Label: **EXECUTED, both halves.**

## Verdict

Same commit, same seed, same data, CPU brain backend: **all three telemetry hashes are
bit-identical** on the two hosts. The recording frame hashes, record counts and total
spike counts also match. No divergence to locate.

## What was compared

- Commit `5c54b034af458b43dfe7c3cb347b294fa7c3faa8` (clean, detached), the same commit
  on both hosts.
- Data: `outputs/brainlab/malecns_v1/graph.npz` sha256 `4b2f87cc…1d01` on both hosts.
  Every data pin recorded in the run manifests is equal across the hosts: `graph_sha256`
  (after the `v3-modulatory-only` policy) `eab575f6…bbda`, `neuron_map_sha256`,
  `io_map_sha256`, `sensory_map_sha256`, `locomotion_dn_map_sha256` and
  `lif_dynamics_pin`.
- Commands, run from the repository root on both hosts:

```
export NEUROFLY_BRAIN_BACKEND=cpu MUJOCO_GL=egl
python -m neurofly_body run --controller modular    --duration 0.2 --seed 0 --graph-dir outputs/brainlab/malecns_v1 --connectome-dir connectome_data/malecns_v1 --output <scratch>/run-modular
python -m neurofly_body run --controller connectome --duration 0.2 --seed 0 --graph-dir outputs/brainlab/malecns_v1 --connectome-dir connectome_data/malecns_v1 --output <scratch>/run-connectome
python -m neurofly_body run --controller connectome --duration 1.0 --seed 0 --graph-dir outputs/brainlab/malecns_v1 --connectome-dir connectome_data/malecns_v1 --output <scratch>/run-connectome-1s
```

  The reference host also set `CUDA_VISIBLE_DEVICES=` (empty), so no CUDA device was
  visible. Its manifests record `brain_backend: cpu` for both connectome runs; the
  modular run has no brain.
- The hash compared is the sha256 of `telemetry.jsonl`, which equals the manifest's
  `trajectory_sha256`. Telemetry contains no host paths, so it is comparable across hosts.

## Hosts and versions

| | reference NVIDIA host | AMD test host (Steam Deck) |
|---|---|---|
| CPU | AMD Ryzen 5 5600X (Zen 3), CPU backend | AMD Custom APU 0932 (Zen 2), CPU backend |
| OS / glibc | Linux 7.0, glibc 2.39 | Linux 7.2 (valve), glibc 2.43 |
| Python | 3.12.3 | 3.14.6 |
| numpy | 2.5.3 | 2.5.3 |
| numba | 0.67.0 (llvmlite 0.49.0) | 0.67.0 |
| mujoco | 3.9.0 | 3.9.0 |
| flygym | 2.1.0 | 2.1.0 |
| scipy | 1.18.1 | 1.18.1 |
| pyarrow | 25.0.1 | not reported |
| pandas | 3.0.5 | not reported |

## Results

| run | records | reference host trajectory sha256 | AMD test host | match | frames sha256 | spikes |
|---|---|---|---|---|---|---|
| modular 0.2 s | 100 | `4da5c6e745790e3740d3a2819e66df6470e04d892fc476ed370d1bc008f65ece` | same | **yes** | `1bcea10e…` both | 0 / 0 |
| connectome 0.2 s | 100 | `80afbf6afd507cedf6f460943255c0b4b9b3d0ac53b83c0f3985284170e93a61` | same | **yes** | `06f32a91…` both | 17,812 / 17,812 |
| connectome 1.0 s | 500 | `ac48cabf6673abfa63ad6e08b29deb5af2881d16f8a5812d99ef4bffa06e1238` | same | **yes** | `1580f280…` both | 89,827 / 89,827 |

Wall time on the reference host: 1.2 s, 4.6 s and 23.4 s (real-time factor 0.165, 0.043,
0.043). On the AMD test host: 4 s, 11 s and 41 s.

The AMD test host also ran `replay-check` on its own modular and connectome 0.2 s runs:
both were BIT_IDENTICAL on that host.

## What this does and does not show

- It shows that the MuJoCo/FlyGym physics and the numba CPU LIF kernel give
  bit-identical trajectories on two different x86-64 CPUs (Zen 3 and Zen 2) and two
  different Python minor versions (3.12 and 3.14), with the same numpy, numba, mujoco
  and flygym versions.
- It does **not** test Intel CPUs, ARM, or different numpy, numba or mujoco versions.
  Those library versions were identical on both hosts, so this result says nothing about
  reproducibility across library upgrades.
- It does **not** cover the CUDA backend. CPU-vs-GPU runs are not expected to be
  bit-identical; that comparison belongs to `scripts/gpu_parity.py`.
- Runs were 0.2 s and 1.0 s of simulated time. A divergence that only appears later in
  a longer run is not excluded, but none had appeared by 500 records.

Statement for G11: *with the CPU brain backend and the same commit, seed, data and
library versions, NeuroFly embodied runs are bit-identical across the two tested AMD
x86-64 hosts.*

## Note added with telemetry format 2

The hashes above are telemetry **format 1** digests of commit `5c54b034`, and they
stay valid for that commit. Telemetry format 2 (`neurofly_body/runner.py`,
`TELEMETRY_FORMAT_VERSION`) moves the static neural identity block out of every
record into `manifest.json`, so a format-2 run of the same arguments has a different
`trajectory_sha256`. The physics, the spike counts and the `body.nfbody` frame
digests do not change with the format. The cross-host comparison has to be repeated
on a format-2 commit before its hashes can be quoted for that code.
