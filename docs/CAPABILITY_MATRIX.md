# Project NeuroFly — 14-Paradigm Capability Matrix

**Version**: 2.0 · **Date**: 2026-09-24 · **Reference**: `docs/ROADMAP.md` Phase 0 (ground truth)

Version 1.0 of this matrix marked the connectome as "Validated" for open arena,
optomotor and the multisensory sandbox. No receipt supports that. This version
claims only what a file in [`docs/receipts/`](receipts/README.md) shows. The owner decided on
2026-09-24 that anything without a v3 receipt is labelled **Mapped, untested on v3**.

---

## Status vocabulary

| Label | Meaning |
|---|---|
| **Tested on v3** | A receipt from the v3 engine with the v3 transmitter policy exists and meets its preregistered rule. **No paradigm has reached this yet.** |
| **Mapped, untested on v3** | The paradigm runs and its sensory and motor channels are named in code, but there is no v3 behavioural receipt. |
| **IO pinned** | The sensory and motor neuron sets are resolved from MaleCNS annotations and pinned by digest in `brainlab/io_map.py`. Only the optomotor map and the WP6 visual-heading map are pinned. |
| **IO not verified** | The channel is declared, but `docs/receipts/graph_identity.json` states that it is not verified ("Other DN roles and the olfactory/wind sensory channels are NOT verified"). |

What the engines are, briefly (full spec in [`LIF_DYNAMICS_SPEC.md`](LIF_DYNAMICS_SPEC.md)):

- **v1** is current-based LIF. Its membrane is unbounded (−200 mV observed) and after
  the first stimulus the whole graph runs away at about 10⁶ spikes/s. Every positive
  connectome result in this repository comes from v1.
- **v2** adds reversal potentials. Its network runs at about 4 × 10⁶ spikes/s, and its
  preregistered optomotor verdict is NULL.
- **v3** adds per-sign PSP calibration and makes aminergic neurons modulatory-only.
  Its receipt (`lif_dynamics_v3.json`) covers calibration probes and one exploratory
  full-graph run per direction. The first confirmatory optomotor run (spec
  `optomotor_v3.json`, 2026-09-24) **FAILED** under its preregistered rule: behaviour
  7/7, physiology 5/7 (HS above the 50 Hz ceiling). See
  [`receipts/validation/optomotor-yaw-v3-1.md`](receipts/validation/optomotor-yaw-v3-1.md).

---

## 14-paradigm matrix

"Modular" is the hand-built controller (researcher baseline). "Connectome" is the
MaleCNS v1.0 graph (166,700 neurons, 25,582,938 synapses).

| # | Paradigm | Sensory ingress (declared) | Motor readout (declared) | Modular controller | Connectome IO | Connectome status | Connectome plasticity | Receipts |
|---|---|---|---|---|---|---|---|---|
| 1 | Open Arena (`open-arena`) | Antennal PNs, ommatidia, JON-C/E | DNa02, DNp09 | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 2 | T-Maze (`t-maze`) | Olfactory ORNs (code drives `ORN_DM1` food, `ORN_DA2` danger) | DNa02, DNp09 | Runs in live UI | Not verified | Mapped, untested on v3 | None on the connectome. MB learning exists in the modular controller only. | live UI sign-off only |
| 3 | Y-Maze (`y-maze`) | Olfactory PNs, antennal mechanosensory | DNa02, DNp09 | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 4 | Heat-Maze (`heat-maze`) | Antennal thermosensory | DNa02, DNp09 | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 5 | Buridan (`buridan`) | Vertical stripes (visual) | DNa02, DNa01 | Runs in live UI | Visual-heading map pinned (WP6) | Mapped, untested on v3 | WP6 ER→EPG rule implemented. Only a 1 s v1 smoke run. | `connectome_closed_loop_wp6_plasticity.json` (v1, 1 s) |
| 6 | Visual Operant (`visual-operant`) | Visual quadrants, thermal reinforcement | DNa02 | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 7 | Wind Tunnel (`wind-tunnel`) | JON-C/E → WED, olfactory PNs | DNa02, DNp09 | Runs in live UI; deterministic at 1x/20x/100x | Not verified | Mapped, untested on v3 | None | `wp1_wp2/determinism_receipt.json` (modular) |
| 8 | Looming Escape (`looming-escape`) | LC4 / LPLC2 | Giant fiber (DNp01) | Runs in live UI | Not verified | Mapped, untested on v3 | n/a | `connectome_closed_loop_looming.json` (v1, 1 s) |
| 9 | Optomotor (`optomotor`) | T4/T5 subtypes (encoder imposes direction selectivity) | DNa02 L/R | Runs in live UI | **Pinned** (`OPTOMOTOR_IO_PIN`) | Mapped, untested on v3. v1 POSITIVE but an engine artefact. v2 NULL. v3 confirmatory run v3-1 FAIL (behaviour passed, HS rate check failed); rerun v3-2 pending. | n/a | `validation/optomotor-yaw-v3-1.md` (v3, FAIL), `wp5_optomotor.json` (v1), `lif_dynamics_v2.json` (v2), `lif_dynamics_v3.json` probe C (v3, 1 seed), `connectome_closed_loop_optomotor.json` (v1, 1 s) |
| 10 | Gap Crossing (`gap-crossing`) | Leg FeCO, campaniform sensilla | CPG cadence and elevation | Runs in live UI | Not verified | Mapped, untested on v3 | n/a | live UI sign-off only |
| 11 | Circadian DAM (`circadian-dam`) | Photoperiod | Locomotor arousal | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 12 | Courtship (`courtship`) | Visual target, cVA (code drives `ORN_DA1`, the Or67d ORNs) | P1 → DNa02 | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 13 | Labyrinth (`labyrinth`) | Antennal touch, contact normals | DNa02, DNp09 | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |
| 14 | Multisensory Sandbox (`multisensory-sandbox`) | All of the above | DNa02, DNp09, MDN, GF | Runs in live UI | Not verified | Mapped, untested on v3 | None | live UI sign-off only |

"Runs in live UI" means the modular backend loaded and ran each assay in the owner's
dashboard during the automated Firefox sign-off
(`docs/receipts/live-signoff/`, `docs/receipts/live-signoff-fresh/`, 14 of 14 assays).
That is a software check. The modular controller's behaviour has not been compared with
published fly data for any paradigm.

Unit and integration tests (for example `tests/test_maze.py`,
`tests/test_assay_responses.py`) show that code paths execute and are deterministic.
They are not behavioural evidence and are no longer cited as such.

---

## What the receipts do show

- **Optomotor, v1** (`wp5_optomotor.json`): preregistered verdict POSITIVE, turning
  index +0.193 [+0.133, +0.243], 6/6 seeds, DNa02 silencing abolishes yaw. The receipt
  itself says every number comes from the runaway v1 engine (~10⁶ spikes/s, DNa02_R held
  near −200 mV). The effect is one-sided and absent at contrast 0.5. The claim was
  withdrawn on 2026-09-20 (`OPEN_SOURCE_PLAN.md`, "Correction").
- **Optomotor, v2** (`lif_dynamics_v2.json`, `optomotor_rerun.v2`): 24 runs, verdict
  **NULL**. Silencing DNa02 does not abolish the turning (paired −0.0115
  [−0.0247, +0.0059]). The network sits at 3.6 × 10⁶ spikes/s with no stimulus.
- **v3 calibration** (`lif_dynamics_v3.json`): the membrane stays within
  [−70, −45] mV, the random-network probe does not self-sustain at weight gains 1 to 4 (it does at 8
  and 16, `probe_b[0].sweep`), and the high-conductance
  fixed point is −45.13 mV (0.13 mV below threshold) under the v3 policy. Probe C is one
  2 s full-graph run per direction, seed not replicated: DNa02 L/R 8 / 0 Hz for leftward
  motion and 2 / 5 Hz for rightward, network 0.9–1.0 × 10⁵ spikes/s during the stimulus
  and 3–4.5 × 10⁴ spikes/s in the gray period after it. That is direction-consistent,
  but it is exploratory, not the preregistered confirmatory set.
- **Closed-loop connectome runs** (`connectome_closed_loop_*.json`): each is 1 s of
  simulated time (50 steps) with controller `brainlab-lif-v1`. They show that the loop
  executes. They are not behavioural evidence.

## Open gates

1. The preregistered v3 optomotor confirmatory run (ROADMAP P1) has not passed. Run
   v3-1 FAILED on physiology on 2026-09-24; the rerun under `optomotor_v3_2.json` is
   pending. Until one passes, no connectome paradigm can move to **Tested on v3**.
2. The IO maps for every paradigm except optomotor and Buridan are unverified.
3. The receipts that need re-running are listed in
   [`docs/receipts/README.md`](receipts/README.md#re-run-queue).

---

## Batch architectures and shared pathways (declared design, not evidence)

These are the anatomical routes each batch is meant to use. They describe the design;
the matrix above says what has been tested.

### Batch A: Visual-Motor Pathway
* **Paradigms**: Optomotor, Buridan, Looming Escape, Visual Operant
* **Route**: photoreceptors → lamina (L1–L5) → medulla (Mi1, Tm1–Tm4, Tm9) → lobula
  plate (T4a–d, T5a–d) and lobula columnar neurons (LC4, LPLC2).
* **Readout**: `DNa02` for yaw; `DNp01` (giant fiber) for looming escape.

### Batch B: Olfactory & Mechanosensory Pathway
* **Paradigms**: T-Maze, Y-Maze, Wind Tunnel, Courtship
* **Route**: ORNs → antennal lobe glomeruli → PNs → lateral horn and mushroom-body
  calyx; Johnston's organ (JON-C/E) → AMMC / wedge → WPNs.
* **Readout**: `DNa02` for steering, `DNp09` for forward drive, `P1` for courtship.

### Batch C: Thermal & Spatial Pathway
* **Paradigms**: Heat-Maze, Gap Crossing, Circadian DAM
* **Route**: antennal thermosensory neurons → SEZ / posterior lateral protocerebrum;
  leg FeCO and campaniform sensilla → thoracic neuromeres; photoperiod → s-LNv.
* **Readout**: `DNa02` for thermal avoidance; CPG amplitude and phase for gap crossing;
  `DNp09` tonic drive for activity bouts.

### Batch D: Complex & Composite Assays
* **Paradigms**: Open Arena, Labyrinth, Multisensory Sandbox
* **Route**: all of the above at once, with arbitration between obstacle reflexes,
  foraging and escape.

---

## WP6 plasticity protocol (specified, not validated)

* **Plastic subset**: ring neuron to compass neuron synapses (`ER4d` + `ER2` → `EPG`),
  3,081 directed edges (0.012 % of the graph).
* **Modulation**: the octopaminergic `EL` cluster gates a depression-only rule.
* **Evidence so far**: one 1 s run on `brainlab-lif-plastic-v1`
  (`connectome_closed_loop_wp6_plasticity.json`: 2,036 spikes in total, final EPG bump
  phase 0.0). WP6 was specified on top of the WP5 decoder, whose causal claim was
  withdrawn, so it needs the v3 optomotor result first. See
  [`WP6_PLASTICITY_SPEC.md`](WP6_PLASTICITY_SPEC.md).
