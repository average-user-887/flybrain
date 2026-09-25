# Receipts index

Every capability claim in the README and in
[`CAPABILITY_MATRIX.md`](../CAPABILITY_MATRIX.md) has to point at a file listed here.
This index records which engine produced each receipt and what it can and cannot support.
It was written on 2026-09-24 as part of ROADMAP Phase 0.

Engine labels: **v1** is current-based LIF (runaway, ~10⁶ spikes/s). **v2** is
conductance-based (~4 × 10⁶ spikes/s). **v3** is conductance-based with per-sign PSP
calibration and the `v3-modulatory-only` transmitter policy. **modular** is the
hand-built controller, not the connectome.

## Known defect: "v3" equations on v1 weights

Until PR #4, a daemon started with `NEUROFLY_LIF_DYNAMICS=v3` ran the v3 equations on
the pinned v1 weights. It did not apply the transmitter policy, so 541 aminergic neurons
kept 435,541 fast out-edges that v3 sets to zero (`lif_dynamics_v3.json`,
`fixed_point.v3-primary`). Anything the daemon produced under a "v3" label before PR #4
is suspect and must be re-run.

`neurofly full-sim` (`experiments/full_connectome_simulation.py`) has the same defect
and PR #4 does not fix it: it built `Brain(graph_path, dynamics="v3")` from the raw
graph without applying the transmitter policy. PR #7 fixed this. Outputs from before
PR #7 are v3 equations on v1 weights.

The paths that do apply the policy are `brainlab/cosim_server.py` (used by
`neurofly_body` and the WP5 scripts), `scripts/gpu_parity.py`, and the `brain-malecns`
stage of `scripts/benchmark.py`.

## Committed receipts

| Receipt | Engine | What it shows | Status |
|---|---|---|---|
| `graph_identity.json` | n/a | MaleCNS v1.0 graph: 166,700 neurons, 25,582,938 edges, SHA-256 pinned. DN cell types and DNa02 sides verified. Olfactory and wind channels and other DN roles are **not** verified. | Valid |
| `lif_dynamics_diagnosis.json` | v1, v2 | The v1 membrane is unbounded; v2 is bounded. Small-network self-sustain probes. | Valid, historical |
| `lif_dynamics_v2.json` | v1, v2 | Full WP5 confirmatory set under v2: verdict **NULL**, network 4.08 × 10⁶ spikes/s. | Valid (negative result) |
| `lif_dynamics_v3.json` | v3 | Calibration probes A/B pass. Fixed point −45.13 mV under the v3 policy. Probe C: one exploratory 2 s full-graph run per direction. | Valid, exploratory only |
| `wp5_optomotor.json` | v1 | Preregistered optomotor verdict POSITIVE, from a runaway engine. One-sided; null at contrast 0.5. | v1 only; the claim was withdrawn 2026-09-20 |
| `connectome_closed_loop_optomotor.json` | v1 | 1 s (50 steps) of the optomotor loop running | Smoke test only; re-run on v3 |
| `connectome_closed_loop_looming.json` | v1 | 1 s of looming; `escape_triggered: true` | Smoke test only; re-run on v3 |
| `connectome_closed_loop_wp6_plasticity.json` | v1 (plastic) | 1 s of Buridan with the ER→EPG rule; 2,036 spikes, bump phase 0.0 | Smoke test only; re-run on v3 |
| `wp4_registry_resources.json` | v1 | Registry memory and switching costs; connectome-fixed ran at 2.48x real time on a 1 s R1–R6 drive | Engineering measurement; v1 activity level |
| `integration/wp5_live_loop.json` | synthetic graph | The live path refuses to fake the optomotor map on a synthetic graph | Valid (software) |
| `integration/firefox_headless_receipt*.json` | modular | Dashboard checks in headless Firefox | Valid (software) |
| `live-signoff/`, `live-signoff-fresh/` | modular | Owner's live dashboard: all 14 assays switch; speed, pause, tab sync and error banner checked in real Firefox | Valid (software) |
| `wp1_wp2/determinism_receipt.json` | modular | Wind tunnel is bit-identical at 1x, 20x and 100x | Valid (modular) |
| `body_speedup/cloud_before.json`, `cloud_after.json` | n/a (body only) | FlyGym body alone: 0.107x to 0.55x real time on a 4-vCPU cloud Xeon, with trajectories bit-identical to the stock controller (`tests/test_fast_controller.py`). The "after" run has `git_dirty: true` because it was measured before the commit. | Valid; the Ryzen figure (0.648x, PR #3) has no committed receipt |
| `wp1_wp2/stress_*`, `gil_starvation_probe.json`, `lock_profile_receipt.json`, `firefox_baseline_*` | modular | At a requested 100x the modular daemon achieves about 30x (12-thread Linux host) | Valid (modular) |
| `ryzen/bench-ryzen-1f4a58a.json` | v3 (policy applied) | Ryzen 5600X + GTX 1660 Ti at `1f4a58a` (PR #11). Full daemon, open-arena, 5 s: 0.0433x CPU, 0.4296x GPU. Brain only, MaleCNS, 0.5 s: 0.0418x CPU, 0.3841x GPU. The GPU brain stage reports `active_neurons_end: 0` (CPU 157,141), which looks like a reporting gap, not a speed issue. | Valid |
| `ryzen/gpu-parity-malecns-1f4a58a.json` | v3 (policy applied) | 2 s MaleCNS, CPU vs GPU per-neuron rate r 0.9986, 0.47 % spike difference; CPU vs 1e-5 mV nudged CPU r 0.9993; two GPU runs identical | Valid |
| `validation/optomotor-yaw-v3-1.md` | v3 (policy applied) | Preregistered optomotor confirmatory run under `validation/specs/optomotor_v3.json`: behaviour 7/7, physiology 5/7, verdict **FAIL**. The raw `receipt.json` is still only on the Ryzen. | Valid (negative result); rerun under `optomotor_v3_2.json` pending |

## Results reported without a committed receipt

| Claim | Where it is stated | Problem |
|---|---|---|
| Embodied co-simulation: 26.98° yaw intact vs 0.02° disconnected, `runs/embodied-video/body.mp4` | `docs/archive/SESSION_HANDOFF_2026-09-23.md` | No run directory or manifest is committed. `neurofly_body` does use the v3 policy, so a re-run can produce a real receipt. |
| Full daemon: 0.398x real time on GPU vs 0.0405x on CPU | PR #2 description | Measured before PR #4 with `NEUROFLY_LIF_DYNAMICS=v3`, so the daemon ran v3 equations on v1 weights. Superseded by `ryzen/bench-ryzen-1f4a58a.json` (0.43x / 0.043x on true v3). |
| Brain on the laptop i5-1334U: 0.0085x | PR #2 description | No committed receipt. |
| Body on the Ryzen: 0.648x | PR #3, `docs/ROADMAP.md` | No committed receipt; the committed body receipts are from a cloud Xeon. |
| `experiment_data/ryzen_battery/` (12 paradigms, 3 trials each, 2026-09-18) | tracked data | No controller or dynamics identity is recorded. Every trial is identical (for example, optomotor gain 8.8 × 10⁻²³ and HS rate exactly 76.0 Hz in all three). Not evidence. |

## Re-run queue

This is the batch for the Ryzen once PRs #1, #2 and #4 are on master. Each re-run should
commit its JSON here with `dynamics`, `transmitter_policy`, `graph_sha256`, backend
(CPU or GPU) and git commit recorded.

1. **WP5 optomotor confirmatory set on v3.** Preregistered, 24 runs (intact, DNa02
   silenced, sham, shuffled × 6 seeds). *Run 2026-09-24 through the validation
   harness (`neurofly validate`), not `scripts/wp5_optomotor.py`: verdict FAIL
   ([`validation/optomotor-yaw-v3-1.md`](validation/optomotor-yaw-v3-1.md)). The rerun
   under `validation/specs/optomotor_v3_2.json` (seeds 100-105) is pending, and the
   raw v3-1 `receipt.json` still has to be copied in.* This is the ROADMAP P1 gate.
   Nothing else on the connectome can be called tested until one passes.
2. **Closed-loop optomotor on v3**, replacing `connectome_closed_loop_optomotor.json`
   (v1, 1 s). Use a duration long enough to cover the preregistered block schedule,
   not 1 s.
3. **Closed-loop looming on v3**, replacing `connectome_closed_loop_looming.json`
   (v1, 1 s).
4. **WP6 plasticity smoke run on v3**, replacing
   `connectome_closed_loop_wp6_plasticity.json` (plastic v1, 1 s). Run it only after
   item 1, since WP6 depends on the DNa02 decoder.
5. *Done in PR #11 (`ryzen/bench-ryzen-1f4a58a.json`).* **Daemon benchmark after PR #4**: `scripts/benchmark.py --stages
   daemon-connectome,daemon-connectome-cuda`, committed as a receipt so the daemon
   speed comes from true v3.
6. *Done in PR #11 (`ryzen/`).* **Commit the PR #2 GPU receipts**: `scripts/benchmark.py --stages
   brain-malecns,brain-malecns-cuda` and `scripts/gpu_parity.py` output.
7. **Embodied intact vs output-disconnected pair** (`neurofly embodied`, seed 1, 2 s),
   committing both manifests, so the 26.98° claim has a receipt or is dropped.
8. **Any daemon registry brain or run saved with a "v3" label before PR #4** (outside
   git, under `outputs/registry/` on the Ryzen). Discard them or re-run them; they
   are v3 equations on v1 weights.
9. **`neurofly full-sim` outputs from before PR #7** (under `outputs/full_simulation/`).
   They are v3 equations on v1 weights; re-run them on current master.
10. **`experiment_data/ryzen_battery/`**: re-run with identity recorded, or move it out
    of the tracked tree.
