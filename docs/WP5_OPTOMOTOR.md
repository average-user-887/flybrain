# WP5 — one causal full-graph sensory → brain → motor loop (optomotor yaw)

> **Status on 2026-09-24:** there is **no valid optomotor result on the current
> engine.** v1 was POSITIVE, but as an engine artefact (§1–§10). v2 was NULL (§11).
> The preregistered v3 confirmatory set has **not** been run; its only v3 data so far
> is one exploratory run per direction (`receipts/lif_dynamics_v3.json`, probe C).
> Running it is the ROADMAP P1 gate. See `receipts/README.md`.

Run of 19 September 2026 on the pinned MaleCNS v1.0 graph
(`graph_sha256 4b2f87cc…091d01`, 166,700 neurons, 25,582,938 edges), backend
`connectome-fixed` through `ExperimentRegistry`. Preregistration:
[`wp5_optomotor_prereg.json`](wp5_optomotor_prereg.json). Machine-readable
result: [`receipts/wp5_optomotor.json`](receipts/wp5_optomotor.json).

> **Everything in §1–§10 is the v1 run and is left exactly as it was.** The two
> dynamics defects in §6 qualifiers 1 and 3 have since been addressed as a
> declared dynamics change (`docs/LIF_DYNAMICS_SPEC.md`, LIF **v2**,
> conductance-based with reversal potentials, controller version
> `brainlab-lif-v2`). **The same preregistered protocol re-run under v2 gives a
> different answer. Read [§11](#11-v2-re-run-conductance-based-dynamics) before
> quoting any number below.** The numbers in §1–§10 remain valid *only* as
> results of the `brainlab-lif-v1` controller.

**Result by the preregistered rule: POSITIVE, with four material qualifiers.**
A graph-mediated, stimulus-locked, side-specific yaw command exists and is
abolished by silencing DNa02, but it is carried almost entirely by leftward
rotation at full contrast, and the surrounding dynamics are not physiological.
Read the qualifiers before quoting the verdict.

## 1. What was broken before this work

* `brainlab/cosim_server.py` split the R1-R6 dataframe rows in half and called
  the halves `visual_l` / `visual_r`. Both halves mixed eyes
  (`outputs/rethink-audit/connectome-mapping-receipt.json`).
* `DN_CHANNELS` in `brainlab/graph_identity.py` had the DNa02 sides swapped:
  `dna02_l` was node 332 = body 10360 = `DNa02_R` (`somaSide R`), and `dna02_r`
  was node 131957 = body 523769 = `DNa02_L`. Every steering sign derived from
  `dna02_diff` was therefore mirrored.
* Two engineered inputs bypassed the sensory pathways entirely: a direct 80-unit
  injection into DNp01 on `looming_trigger`, and a tonic 14.5–21.75 unit drive
  into DNb01 on every step.

## 2. Verified side assignment (the DNa02 question)

`somaSide` from the released `annotations.feather`, cross-checked against the
graph's own wiring:

| channel (now) | node | body | `somaSide` | `instance` | share of \|output weight\| on the soma side |
|---|---|---|---|---|---|
| `dna02_l` | 131957 | 523769 | L | `DNa02_L` | 92.0 % |
| `dna02_r` | 332 | 10360 | R | `DNa02_R` | 92.5 % |

`rootSide` is null for DNa02 (it is populated for photoreceptors, not for these
central neurons), so the soma annotation plus the output laterality is the
evidence. Each DNa02 sends ~92 % of its absolute output weight to same-side VNC
and central neurons, i.e. the axon stays ipsilateral. Rayshubskiy et al. 2020
(bioRxiv 2020.04.04.024703) report that DNa02 activity predicts, and unilateral
activation evokes, **ipsilateral** turning. Naming the channels by soma side is
therefore the mapping that makes the steering sign correct.

`brainlab/graph_identity.py:39-47` now holds the corrected map and
`DN_EXPECTED_SIDES`; `verify_graph` (`graph_identity.py:168-180`) fails if a
lateralised channel's soma side does not match. This changes
`io_map_sha256` from `f64f8a21…e0bb7` to `c571ed62…6165a`, so WP4 checkpoints
written under the old map are refused by `ExperimentRegistry.read_checkpoint`.
That is the intended behaviour: the mapping changed, so the identity changed.
`docs/receipts/graph_identity.json` carries the amendment.

## 3. The loop

Four stages, in separate objects, `brainlab/io_map.py`:

1. **Sensor encoder** (`OptomotorEncoder`, `io_map.py:174-227`). Input: retinal
   slip in rad/s (+ = pattern rotating counter-clockwise seen from above, i.e.
   leftward) and a contrast in [0, 1]. Leftward rotation means the **left** eye
   sees front-to-back motion and the **right** eye back-to-front, so it drives
   `T4a + T5a` of the left eye and `T4b + T5b` of the right eye; rightward
   rotation mirrors this. Cells are resolved by cell type and annotated
   `somaSide`, sorted by body ID, matched to equal counts (835 T4 + 826 T5 per
   population), never by row order. Drive per cell is a rectified sinusoid at
   the stimulus temporal frequency (1.5 Hz here) with a per-cell random phase
   and 10 % multiplicative noise, peak 20 units (the same amplitude as the WP4
   validation stimulus, not tuned). The resolved map is hashed and pinned:
   `OPTOMOTOR_IO_PIN = 228c1b69…66c7b`; a different map raises
   `GraphUnavailable`.
   *Why T4/T5 and not photoreceptors:* the LIF proxy has no graded potentials
   and no delay lines, so it cannot compute direction selectivity from R1-R6.
   Direction selectivity is therefore an explicit encoder assumption (subtype
   assignment from Maisak et al. 2013), not a graph result. HS/VS are **not**
   driven — they are recorded as the first graph-computed stage.
2. **Graph.** `ExperimentRegistry.activate('optomotor', 'connectome-fixed')`,
   fixed released weights, 2 ms steps of the 0.1 ms LIF. Nothing else is
   injected: no tonic drive, no DN injection, no looming trigger.
3. **Motor decoder** (`DNa02YawDecoder`, `io_map.py:230-266`).
   `yaw = 0.02 rad/s/Hz × (filtered rate DNa02_L − filtered rate DNa02_R)`,
   50 ms exponential filter, ipsilateral convention. Linear, unclipped, no
   rectification, no floor, no fallback: zero spikes give exactly zero yaw, and
   both signs are reachable. Each DN's contribution is logged separately.
4. **Body.** Not simulated here (open-loop tethered assay); the caller
   integrates yaw. Section 8 lists the arena hooks.

## 4. Preregistered design

Predeclared before any confirmatory run
(`docs/wp5_optomotor_prereg.json`, sha256 `b05f48c6…a49c6` at declaration;
`9af619d4…5e168` after the pilot clause cut the seed list). Fixed schedule for
every condition and seed: 500 ms gray, then 8 blocks of 1 s with 500 ms gray
between, directions counterbalanced, contrasts 1.0 and 0.5, drum speed
45 °/s, spatial period 30 ° (1.5 Hz), 2 ms steps, seeds 0–5.

Primary outcome: turning index `TI = mean over blocks of s × mean yaw`
(rad/s; balanced design, so a constant turning bias cancels). Conditions:
intact; DNa02 silenced (both DNa02 clamped at −200 units, Kir2.1-like);
sham (encoder runs with the identical RNG stream, output not delivered);
degree-preserving shuffled graph (the `post` array permuted once with
`default_rng(777)`, preserving every in- and out-degree and every source's
weights, explicitly labelled as a control graph, run in registry test mode);
and the modular baseline (`vision.CompoundEyeVision.get_optomotor_yaw_bias`).

The pilot (seed 999, 3.5 s) measured only runtime and encoder firing; its DN
outputs were dropped before printing. It projected 69.9 min > the 60 min
budget, so the prereg's own clause cut seeds from 10 to 6. No other parameter
changed.

## 5. Results (6 seeds, mean [95 % bootstrap CI])

| condition | TI (rad/s) | TI contrast 1.0 | TI contrast 0.5 | DNa02 asymmetry (Hz) | network rate (spikes/s) |
|---|---|---|---|---|---|
| intact | **+0.193 [+0.133, +0.243]** | +0.370 [+0.269, +0.463] | +0.015 [−0.007, +0.033] | +10.4 [+7.2, +13.1] | 1.02 × 10⁶ |
| DNa02 silenced | 0.000 (identically) | 0.000 | 0.000 | 0.0 | 1.01 × 10⁶ |
| sham (no input) | 0.000 (identically) | 0.000 | 0.000 | 0.0 | 0 |
| shuffled graph | −0.024 [−0.035, −0.012] | −0.048 [−0.071, −0.023] | −0.000 | −1.25 [−1.83, −0.60] | 8.97 × 10⁴ |
| modular baseline | +0.298 (deterministic) | +0.300 | +0.296 | n/a | n/a |

Paired differences (per seed): intact − silenced and intact − sham
= +0.193 [+0.133, +0.243]; intact − shuffled = +0.217 [+0.156, +0.268];
Cohen's dz 2.5–2.8; 6 of 6 seeds positive.

**Neural response to left vs right stimuli** (intact, mean rate per cell):

| population | leftward stimulus (s = +1) | rightward stimulus (s = −1) |
|---|---|---|
| HS L / HS R | 142.9 / 0.0 Hz | 0.0 / 127.6 Hz |
| H2 L / H2 R | 0.0 / 65.3 Hz | 123.8 / 0.0 Hz |
| DNa02 L / DNa02 R | **21.3 / 0.2 Hz** | **4.0 / 3.7 Hz** |
| DNa01 L / DNa01 R | 0.0 / 0.4 Hz | 0.1 / 0.0 Hz |

The sensory stage is cleanly eye-specific and side-swapping: the ipsilateral HS
of the stimulated eye responds, H2 responds to the contralateral eye's
back-to-front motion — exactly the pathway the wiring predicts. The structural
prediction (computed from the graph before the results were read,
`structural_evidence` in the receipt) is that leftward rotation drives DNa02_L
through two routes, HS_L → DNa02_L (two-hop weight product +7.5 × 10⁴) and
H2_R → DNa02_L (+4.7 × 10⁴), with the mirror for rightward rotation
(HS_R → DNa02_R +9.1 × 10⁴, H2_L → DNa02_R +4.0 × 10⁴).

**Motor dependence.** Across the 48 intact blocks, the stimulus-aligned HS
asymmetry correlates with the DNa02 asymmetry at r = 0.64; the DNa02 asymmetry
correlates with aligned yaw at r = 0.998, which is by construction (the decoder
is a linear map of DNa02 rates) and is *not* evidence. The causal evidence is
the silencing condition: clamping DNa02 leaves the upstream response intact
(HS asymmetry +134.6 vs +135.2 Hz, network rate unchanged) and removes the
motor output entirely (yaw identically zero, every step of every seed). The
sham condition shows the loop is input-driven: with no delivered input the
graph produces zero spikes, so zero yaw.

**Conditional secondary — closed loop.** The prereg allowed a closed-loop run
only if the primary was positive. With `slip = s·ω − yaw` fed back each 2 ms
step (seeds 0–4), the mean absolute retinal slip falls from 0.785 rad/s
(open loop, by definition) to 0.647–0.710 rad/s, a 10–18 % reduction. The loop
stabilises gaze partially and in the correct direction; it does not null the
slip, which follows from the one-sided response in §6.1.

## 6. The qualifiers (why this is not "the fly does optomotor")

1. **The effect is one-sided.** Leftward rotation drives DNa02_L to 21 Hz while
   DNa02_R stays silent; rightward rotation produces no net rightward bias
   (4.0 vs 3.7 Hz). An exploratory run with every direction negated (seeds 0–2,
   declared as exploratory after the confirmatory result, `exploratory_mirror`
   in the receipt) reproduces the same left dominance (TI +0.217, DNa02_L
   19–30 Hz for leftward, DNa02_R ≤ 6 Hz for rightward), so this is not an
   artefact of the block order. DNa02_R sits at −120 to −200 mV in the proxy,
   i.e. it is held down by the network state. The wiring is symmetric; the
   dynamics are not.
2. **Contrast 0.5 gives nothing.** TI at half contrast is +0.015
   [−0.007, +0.033]. The whole result rides on full contrast.
3. **The network runs away.** After the first stimulus the graph enters a
   self-sustained state of about 10⁶ spikes/s (≈6 Hz mean per neuron) that
   persists through the gray periods; the "gray" baseline is not rest, and
   DNa02_L fires a few spikes there too. Membrane potentials reach −200 mV
   because the LIF proxy has no reversal potentials. These are properties of
   `brainlab/engine.py`, not of the connectome.
4. **The graph is not computing direction selectivity.** The encoder supplies
   it. What the graph contributes is the side-specific routing from T4/T5
   through HS/H2 to DNa02 — which the shuffled control shows is wiring-specific
   (shuffling degrees-preserved destroys the response: HS ≈ 1 Hz, TI −0.024).

Honest summary: **a causal, wiring-dependent, stimulus-locked loop from
eye-specific motion input to a steering descending neuron and to a yaw command
exists and is demonstrated. A symmetric optomotor response is not.**

## 7. Compute cost

Sustained, whole loop (encoder + 166,700-neuron LIF + decoder + logging), 2 ms
control steps, single core:

| condition | simulated s per wall s |
|---|---|
| intact | 0.106 |
| DNa02 silenced | 0.111 |
| shuffled graph | 0.132 |
| sham (silent graph) | 5.55 |

75 s of simulated time cost 708 s of wall time; the whole confirmatory set was
32 m 54 s. Peak RSS 826 MiB (one graph plus one shuffled `post` array; a single
graph instance is ~600 MiB). Real time is therefore **about 10× away** at this
activity level — the cost is dominated by the self-sustained ~10⁶ spikes/s, as
the silent sham (5.6× real time) shows. Any real-time claim for the live
dashboard is unsupported for this loop today.

## 8. Integration hooks needed to run this live

**Status: landed in the live path (see the commit that follows `ca7b71b`).** The arena
sends slip and contrast for the optomotor assay only, the graph yaw replaces (never
adds to) the modular bias, the tethered assay runs at speed 0, and a missing
`optomotor` block reports `graph-unmapped-io` instead of a silent zero. Receipt:
`docs/receipts/integration/wp5_live_loop.json`. The live-UI sign-off is still pending.

Server side is done: `brainlab/cosim_server.py` accepts
`sensory["optomotor_slip_rad_s"]` (+ optional `optomotor_contrast`) and returns
`reply["optomotor"] = {yaw_rad_s, contributions, rate_l, rate_r, io_map_sha256}`,
lists `engineered_assistance_applied` on every step, exposes
`engineered_assistance_enabled` in `/status`, and takes
`--no-engineered-assistance` (constructor `engineered_assistance=False`) to gate
the DNp01 injection and the tonic DNb01 drive off. `visual_l` / `visual_r` are
now eye-specific R1-R6 by annotated `rootSide`.

Still needed, in files owned by other agents (**not** changed here):

* `arena.py` (~line 1012 and ~1270, `compute_steering`): pass
  `optomotor_slip_rad_s = radians(drum_velocity_deg_s) − fly.angular_velocity`
  and `optomotor_contrast = visual_contrast` into the connectome sensory packet.
* `arena.py` ~line 1087: for the `optomotor` assay under a graph backend, use
  the reply's `optomotor.yaw_rad_s` as the yaw command **instead of**
  `vision.get_optomotor_yaw_bias()` — do not add the two. With the modular
  backend nothing changes.
* `arena.py`: when the graph backend is selected, the fly's forward speed has no
  graph-derived source in this loop (DNb01/DNp09 drive is engineered assistance
  and is gated off), so the tethered optomotor assay must run with speed 0 and
  report it, rather than borrowing the modular walking drive.
* `connectome_bridge.py` / `connectome_client.py`: forward the two new sensory
  keys, surface `optomotor`, `engineered_assistance_applied` and
  `engineered_assistance_enabled` in telemetry, and treat a missing
  `optomotor` block (synthetic graph, unresolved map) as an explicit
  unsupported state, not as zero.
* `neurofly_daemon.py` / `web/app.js`: show `engineered_assistance_enabled`,
  `optomotor_io_map_sha256` and the achieved sim-s-per-wall-s next to the run
  identity; at ~0.1× real time the optomotor assay cannot be run at 1× on this
  hardware and the UI must say so instead of silently dropping frames.

## 9. Files, tests, reproduction

* `brainlab/io_map.py` (new): map resolution + pin, encoder, decoder, loop.
* `brainlab/graph_identity.py:36-47, 168-180`: corrected DNa02 channels, soma
  side verification.
* `brainlab/cosim_server.py:45-56, 67-86, 151-171, 207, 212-222, 240-246,
  296-306, 385-415`: assistance gate and reporting, eye-specific retina,
  optomotor input/reply.
* `scripts/wp5_optomotor.py` (new): pilot, confirmatory, exploratory mirror,
  closed-loop secondary. `scripts/wp5_receipt.py` (new): receipt assembly.
* `tests/test_wp5_optomotor.py` (new, 11 tests): pin and annotated sides,
  eye/direction selection, RNG stream independence from the stimulus, matched
  left/right drive, zero-spikes-zero-yaw, no clipping or floor, sham delivers
  nothing, silencing clamps, server assistance gate. Whole suite:
  `PYTHONPATH=. ./.venv/bin/python -m pytest -q -p no:cacheprovider tests/`
  → 358 passed.

Reproduce:

```
NEUROFLY_GRAPH_DIR=<graph dir> PYTHONPATH=. .venv/bin/python scripts/wp5_optomotor.py \
    --out outputs/wp5/<stamp>            # ~33 min, one graph in memory at a time
PYTHONPATH=. .venv/bin/python scripts/wp5_receipt.py outputs/wp5/<stamp> \
    <structure.json> docs/receipts/wp5_optomotor.json
```

## 10. Next dependency (WP6)

WP6 needs a written plasticity specification before any code. This loop gives
it a usable substrate and three constraints:

* The frozen encoder and decoder of `brainlab/io_map.py` are the ones to hold
  fixed during a learning comparison, so behaviour changes cannot be decoder
  training. Their pins (`OPTOMOTOR_IO_PIN`, `io_map_sha256`) belong in the WP6
  manifest.
* A plastic edge subset must be declared from identified connections; the
  T4/T5 → HS/H2 → DNa02 routes above are the candidate set for an optomotor
  learning assay, with primary sources for which of them are plastic.
* The runaway self-sustained state and the absent rightward response are
  prerequisites, not details: a learning rule evaluated on top of a network at
  10⁶ spikes/s with one dead output side will measure the proxy's dynamics. WP6
  should either declare adaptation/normalisation as an explicit dynamics change
  (a new controller version, re-pinned) or restrict claims to the left-rotation
  condition.

## 11. v2 re-run (conductance-based dynamics)

**Everything above is the v1 result and is unchanged.** This section reports the
same preregistered protocol (`docs/wp5_optomotor_prereg.json`, same sha256, same
seeds, same blocks, same primary outcome `TI`) under the declared conductance-based
dynamics of [`LIF_DYNAMICS_SPEC.md`](LIF_DYNAMICS_SPEC.md) — controller version
`brainlab-lif-v2`. Only the engine differs. Receipts:
[`receipts/lif_dynamics_diagnosis.json`](receipts/lif_dynamics_diagnosis.json),
[`receipts/lif_dynamics_v2.json`](receipts/lif_dynamics_v2.json).

> **Status: the confirmatory v2 set is COMPLETE** (24 of 24 runs, finished
> 20 September 2026; receipt regenerated). The preregistered verdict under v2 is
> **NULL**: silencing DNa02 does not remove the residual turning
> (intact − silenced `TI` = −0.011 [−0.025, +0.006], CI includes zero, 5 of 6
> seeds negative), so the yaw that remains is not DNa02-mediated. The numbers in
> §11.2 are the seed-0 preview that was written before the set finished; §11.4
> carries the full result.

### 11.1 Diagnosis of the two defects

**The dead right side was dynamical, not anatomical — as §6.1 suspected.**
DNa02_L and DNa02_R have input budgets within 2 % of each other (L: 1,138
in-edges, Σ excitatory weight 4,427.5, Σ inhibitory 2,160.7; R: 1,201 in-edges,
4,389.8 / 2,256.4). Under v1, DNa02_R is driven to −184 mV during leftward
rotation and −70 to −123 mV during rightward rotation — far below any
physiological chloride reversal — and emits **0 spikes in either direction**.
Under v2 the same neuron cannot go below −70 mV, sits at −51 mV, and fires.

**The unbounded membrane is fixed; the runaway is not.** Probe A (two neurons,
inhibitory weight swept over three decades):

| inhibitory weight | v1 min V | v2 min V |
|---|---|---|
| −40 | −71.4 mV | −56.98 mV |
| −400 | −245.9 mV | −66.66 mV |
| −4000 | −1991.5 mV | −69.75 mV |

v1 scales without limit; v2 asymptotes at `E_inh`. The conductance quantum is
calibrated, not fitted: the unitary EPSP at rest is 0.04332 mV under v1 and
0.04372 mV under v2.

Probe B (2,000-neuron random recurrent net, 200 ms input then 800 ms free) shows
v2 self-sustains at a **lower** recurrent gain than v1 — at weight gain 8, v1
leaves 2.46 Hz/neuron and v2 leaves 67.9 Hz/neuron. §3.3 item 4 of the spec
predeclared this as a possible consequence of giving both signs one conductance
quantum: the unitary IPSP at rest becomes 18/52 = 0.346× the v1 IPSP, so
inhibition loses hyperpolarising authority even as it gains shunting authority.

**Why (analysis of the declared model, not a parameter change).** With
`E_exc = 0`, `E_inh = −70` and one quantum per unit weight, the high-conductance
fixed point is `r·E_inh/(1+r)` where `r = ĝ_i/ĝ_e`. The MaleCNS graph under the
coarse transmitter-sign proxy has `r = 0.619` (13.0 M inhibitory vs 21.1 M
excitatory weight), giving a fixed point of **−26.75 mV against a −45 mV
threshold** — suprathreshold, so any sustained input drives the whole network to
its refractory limit. A subthreshold fixed point needs `r > 1.80`; the graph is
2.91× short. v1 hid this behind unbounded hyperpolarisation. The 0.275 mV per
synapse scale was itself calibrated *inside* a current-based model; carrying it
into a conductance model without recalibrating the overall synaptic gain is what
leaves the graph suprathreshold. (An alternative calibration — matching the v1
IPSP rather than the v1 EPSP — gives an inhibitory quantum 2.889× the excitatory
one, within 1 % of the 2.91× needed. That is recorded as an observation for a
future declared v3. It was **not** adopted here: choosing it after seeing this
result would be tuning.)

### 11.2 Optomotor, v1 vs v2, intact, seed 0

Same trace, same metric, computed identically for both. **n = 1; the v1 column's
six-seed values from §5 are given for scale, not for comparison.**

| measure | v1 (seed 0) | v2 (seed 0) |
|---|---|---|
| `TI` (rad/s) | +0.072 | +0.052 |
| `TI` contrast 1.0 | +0.175 | +0.076 |
| `TI` contrast 0.5 | −0.030 | +0.027 |
| blocks with yaw aligned to stimulus | **4 of 8** | **8 of 8** |
| DNa02 asymmetry `A` (Hz) | +3.9 | +2.8 |
| DNa02_L / DNa02_R, leftward (Hz) | 11.8 / **0.0** | 307.0 / **304.0** |
| DNa02_L / DNa02_R, rightward (Hz) | 4.3 / **0.3** | 307.5 / **310.0** |
| network rate, stimulus (spikes/s) | 1.22 × 10⁶ | 4.08 × 10⁶ |
| network rate, gray (spikes/s) | 1.06 × 10⁶ | 3.58 × 10⁶ |
| DNa02_L / DNa02_R during gray (Hz) | 8.4 / 0.0 | 272.2 / 272.9 |
| membrane range (mV) | −307.3 … −45.0 | **−54.5 … −45.0** |
| sustained speed (sim s per wall s) | 0.106 | 0.029 |

Read the two middle rows together. Under v1 the right side is dead in *both*
directions, and only half the blocks turn the right way; the positive `TI` comes
from leftward blocks alone. Under v2 **both** DNa02s respond and the sign of the
L−R difference follows the stimulus in **every** block, including rightward ones
— the §6.1 defect is gone. But the signal is now ±3 Hz riding on a 307 Hz
saturated background, the gray periods are noisier still (273 Hz per DNa02,
3.6 × 10⁶ spikes/s network-wide), and the resulting `TI` is *smaller*.

### 11.3 Honest verdict

* The **membrane-bound** defect (§6 qualifier 3, second half) is **fixed**.
  −200 mV is now impossible by construction; the observed range is −54.5 to
  −45.0 mV.
* The **dead right side** (§6 qualifier 1) is **fixed**, and its cause is
  confirmed to be the missing inhibitory reversal potential, not the wiring.
* The **runaway** (§6 qualifier 3, first half) is **not fixed — it is worse**:
  4.1 × 10⁶ spikes/s against 1.0 × 10⁶, and the gray periods are further from
  rest than before, not closer. Reversal potentials alone were never going to
  fix it, and the spec said so before the run.
* The **optomotor claim** under v2 is **NULL** by the preregistered rule, on the
  complete 24-run set. See §11.4. The v1 verdict (POSITIVE, one-sided) stands as
  the `brainlab-lif-v1` result and is not retracted; it is now known to have
  depended on an engine property (DNa02_R held below any chloride reversal).
* **WP6 must not build on either version as it stands.** A learning rule
  evaluated on a network at 4 × 10⁶ spikes/s with a suprathreshold
  high-conductance fixed point measures the engine, exactly as §10 warned. The
  next declared change is a synaptic-gain recalibration, argued from physiology
  *before* it is measured, not after.


### 11.4 Completed v2 confirmatory set (24 runs, 6 seeds, all conditions)

| condition | `TI` mean [95 % CI] | seeds positive | DNa02 L/R, leftward (Hz) | network rate (spikes/s) |
|---|---|---|---|---|
| intact | +0.0287 [+0.0221, +0.0384] | 6/6 | 305.8 / 304.2 | 4.08 × 10⁶ |
| DNa02 silenced | **+0.0402 [+0.0309, +0.0495]** | 6/6 | 241.4 / 240.2 | 4.08 × 10⁶ |
| sham (no input) | 0.0000 | 0/6 | 0.0 / 0.0 | 0 |
| shuffled graph | −0.0209 [−0.0328, −0.0085] | 0/6 | 0.0 / 2.1 | 9.02 × 10⁴ |

Paired: **intact − silenced `TI` = −0.0115 [−0.0247, +0.0059]**, dz −0.54, 5 of 6
seeds negative. Silencing the decoder's own output neurons does **not** abolish the
turning, and numerically increases it. The preregistered causal requirement is
therefore not met: **NULL**.

Gray-period baseline, intact: 3.58 × 10⁶ spikes/s with DNa02_L 272.5 Hz and
DNa02_R 273.0 Hz **with no stimulus at all**. The ±3 Hz stimulus-linked asymmetry
rides on a ~273 Hz saturated background, which is why a small residual `TI`
survives DNa02 silencing: it is not a decoded steering command.

Compute: 0.030 simulated s per wall s intact (v1: 0.106), 41 min for the intact
condition alone, about 2 h for the set.

**Reading.** v2 removes the artefact that produced the v1 result and does not
replace it with a real one. Neither version supports a graph-mediated optomotor
claim: v1's was an engine artefact, and v2 has no causal effect to claim. The
blocking problem is the synaptic-gain calibration (§11.1), not the wiring.
