# Photoreceptor encoder: can the graph compute direction selectivity itself?

**Status: PRE-REGISTRATION. Sections 0–6 were written and hashed before any
response of any neuron to this encoder was measured.** The locked verbatim copy
is `docs/receipts/photoreceptor_encoder_declaration.locked.md` and its sha256 is
recorded in `docs/receipts/photoreceptor_encoder.json` under `declaration_lock`,
exactly as `docs/receipts/einh_sensitivity_declaration.locked.md` was done for
`docs/EINH_SENSITIVITY.md`. Sections 7 onward are filled in afterwards and are
not permitted to change 0–6.

Everything anatomical quoted in §2 (edge signs, cell counts, hex coordinates,
soma positions) was computed from the pinned graph and the released annotations
**before** the declaration was locked, and is anatomy, not response. No neuron
had been driven by this encoder when §0–§6 were written.

**Nothing in the existing WP5 loop is modified.** `brainlab/io_map.py`, its
`OptomotorEncoder`, its pin `OPTOMOTOR_IO_PIN = 228c1b69…66c7b`, and every number
in `docs/WP5_OPTOMOTOR.md` §1–§13 stay exactly as they are and stay reproducible.
The photoreceptor encoder is a **new, separately pinned io-map variant** in a new
module with its own sha256. The dynamics are unchanged: v3, `dynamics_pin('v3')`
byte-identical, primary `E_inh = −70 mV` for comparability with §12, and the
declared variant `v3-einh-60` for §13's literature-supported value.

---

## 0. Why this exists

`docs/WP5_OPTOMOTOR.md` §3, §6 qualifier 4, §12.4 item 4 and
`docs/LIF_DYNAMICS_SPEC.md` §6.9 all record the same limitation in the same words:
the WP5 encoder **imposes** direction selectivity rather than computing it. It
drives `T4a+T5a` of one eye and `T4b+T5b` of the other according to the stimulus
direction, using the subtype-to-preferred-direction assignment of Maisak *et al.*
2013 as an engineering input. The graph then routes that already-directional
signal through HS/H2 to DNa02. The routing is graph-computed and shuffle-sensitive;
the *selectivity* is not.

The owner's constraint is that the connectome must do the work. This program
therefore delivers the stimulus one stage earlier than any previous NeuroFly
measurement — to photoreceptors only — and asks whether the graph's own motion
circuitry (R1-R6 → lamina → medulla → T4/T5) produces direction selectivity.

**A null here is a result, not a failure.** If the selectivity does not appear,
that is a statement about this LIF proxy on this graph, and §4 declares in advance
exactly what evidence will be produced to localise and attribute it. That trace is
the deliverable in that case and is reported as the headline.

## 1. What this engine can and cannot represent (checked in the code, before measuring)

These are facts about `brainlab/engine.py` at this HEAD, verified by reading it,
not predictions.

1. **There is no graded transmission path of any kind.** All three declared
   dynamics (`advance`, `advance_v2`, `advance_v3`) transmit only on a threshold
   crossing: a neuron's membrane is integrated, compared with
   `V_THRESHOLD_MV = −45`, and only if it crosses does anything appear in the
   delay queue and reach a postsynaptic conductance. A subthreshold membrane
   excursion, of any size and any duration, transmits **nothing**. The only
   continuous quantity that enters the network is the external `drive` array,
   i.e. the encoder's own input; it is not a transmission mechanism between
   neurons.
2. **The only temporal parameters available are global and identical for every
   neuron:** `tau_m = 20 ms`, `tau_syn = 5 ms`, axonal delay `1.8 ms`,
   refractory `2.2 ms`, `dt = 0.1 ms`. There are no per-cell-type kinetics, no
   H-current, no dendritic compartments, no receptor kinetics and no calcium
   state. A delay difference between two pathways onto the same target can arise
   here **only** from a difference in the number of synaptic hops (1.8 ms each,
   plus the 5 ms/20 ms filtering that both pathways share) or from a difference in
   the strength/sign pattern that shifts the time at which threshold is crossed.
3. **The graph has no spontaneous activity under v3.** `docs/WP5_OPTOMOTOR.md`
   §12.3 measured the pre-stimulus gray as *exactly* 0 spikes/s. Zero basal
   firing is inherited from the source model (Shiu *et al.* 2024), which states
   it as a limitation of its own.

**This is inherited, not invented.** Shiu *et al.* 2024 (*Nature* 634:210–219,
DOI `10.1038/s41586-024-07763-9`), the source of every LIF constant in this engine
(verified against `philshiu/Drosophila_brain_model/model.py`: `v_0 −52 mV`,
`v_rst −52 mV`, `v_th −45 mV`, `t_mbr 20 ms`, `tau 5 ms`, `t_rfc 2.2 ms`,
`t_dly 1.8 ms`, `w_syn 0.275 mV`), state that their model "does not account for
gap junctions, non-spiking neurons, internal state or long-range neuropeptides",
that they treat "each neuron identically as a spiking neuron", and that "circuits
with extensive basal inhibition, not captured by the model because of the zero
basal firing rate, may be poorly simulated". This project therefore reproduces a
declared limitation of the published model it descends from; it has not dropped a
mechanism the source had. The same is true of the aminergic-as-excitatory
convention, which Shiu *et al.* also adopt and which v3 departs from deliberately
(`LIF_DYNAMICS_SPEC.md` §6.3.1).

**The biology this is being asked to reproduce is graded.** Literature status is
marked per item; leads relayed to this session but not retrieved here are marked
as such and are **not** relied on for any criterion.

* Photoreceptors and lamina monopolar cells respond to light with **graded**
  potentials; the medulla neurons that feed T4/T5 were recorded by patch clamp as
  graded cells, and Mi1/Tm3 (ON) and Tm1/Tm2 (OFF) carry the delayed and
  non-delayed arms. Behnia, R., Clark, D.A., Carter, A.G., Clandinin, T.R. &
  Desplan, C. (2014), *Processing properties of ON and OFF pathways for*
  Drosophila *motion detection*, **Nature** 512:427–430,
  DOI `10.1038/nature13427`. *(Retrieved in this session.)*
* T4 direction selectivity arises from **simple integration of spatially offset
  fast excitation and offset, delayed inhibition**, measured by *in vivo*
  whole-cell (i.e. graded voltage) recordings and reproduced by a passive
  conductance-based single-cell model. Gruntman, E., Romani, S. & Reiser, M.B.
  (2018), *Simple integration of fast excitation and offset, delayed inhibition
  computes directional selectivity in* Drosophila, **Nature Neuroscience**
  21:250–257, DOI `10.1038/s41593-017-0046-4`. *(Retrieved in this session.)*
* The temporal tuning of the detectors is set by the **dynamics of their input
  elements**, not by conduction delay. Arenz, A., Drews, M.S., Richter, F.G.,
  Ammer, G. & Borst, A. (2017), *The temporal tuning of the* Drosophila *motion
  detectors is determined by the dynamics of their input elements*, **Current
  Biology** 27:929–944, DOI `10.1016/j.cub.2017.01.051`. *(Retrieved in this
  session.)*
* T4/T5 subtype preferred directions: Maisak, M.S. *et al.* (2013), **Nature**
  500:212–216 — already cited by the existing encoder. Reported calcium-imaging
  direction-selectivity indices of roughly **0.7–0.9** with near-complete
  null-direction suppression were **relayed to this session and not retrieved
  here**; they are used only as an order-of-magnitude bar in §3, never as a
  criterion.
* The relevant delay is of the order of **tens of milliseconds** and is
  attributed to membrane and synaptic filtering and to offset delayed inhibition
  (Behnia 2014; Arenz 2017; Gruntman 2018). The specific figure "20–30 ms" was
  relayed and is not independently established here; §3 uses only the
  order-of-magnitude statement.

**Consequence, declared now.** This engine has 1.8 ms of hop delay and one global
5 ms synaptic time constant. It can express a **spatial** offset between inputs to
a T4/T5 cell exactly as the connectome wires it, and it can express **sign**
(the coarse transmitter proxy makes Mi9 and L1 glutamatergic-inhibitory, Mi4/CT1/C3
GABAergic-inhibitory, Mi1/Tm1/Tm2/Tm3/Tm9/L2 cholinergic-excitatory). What it
cannot express is a cell-type-specific temporal filter. Whether spatial offset plus
sign plus one shared filter is enough is precisely the empirical question below.

## 2. The encoder, declared completely

### 2.1 What is driven, and nothing else

Only `R1-R6` photoreceptors receive drive. Resolution is by **cell type** and the
released **`rootSide`** annotation — the same rule `brainlab/cosim_server.py`
already uses for the eye-specific retina, because R1-R6 have no soma in the volume
and `somaSide` is null for them. Row order is never used. Anatomy of the pinned
graph, computed before locking:

| fact | value |
|---|---|
| `R1-R6` nodes | 3,377 |
| `rootSide` | L 1,112 / R 2,265 |
| nodes with no out-edges | 18 |
| median out-degree | 4 |
| sign of **every** photoreceptor out-edge | **negative** (histamine → inhibitory under `brainlab/transmitters.py`) |
| largest targets by Σ\|weight\| | L2, L1, L3, then Lai, T1, L4, C3 |

No other neuron receives any input: no tonic drive, no T4/T5 drive, no looming, no
DN injection, and `cosim_server`'s `ENGINEERED_ASSISTANCE` paths are not used.
HS/H2/VS/T4/T5/DNa02 are **recorded only**.

### 2.2 Retinal layout: derived from anatomy, not asserted

A photoreceptor must be given a position in the visual field. The layout is
derived in five steps, each from released data, none from any response:

1. **Column assignment.** Each `R1-R6` node is assigned the optic-lobe column of
   its dominant lamina target: among its out-edges onto `L1`, `L2` or `L3` with a
   non-null `assignedOlHex1`/`assignedOlHex2`, the (hex1, hex2) pair carrying the
   largest Σ\|weight\|. This resolves **3,335 of 3,377** photoreceptors, and the
   assignment is essentially unambiguous: the dominant column carries a mean
   **99.8 %** (median 100 %) of that photoreceptor's lamina output weight, and
   3,293 of the 3,335 project to exactly one column. The 42 unresolved
   photoreceptors (18 with no out-edges, 24 with no hex-annotated lamina target)
   receive **no drive**, and that is recorded in the map.
2. **Volume axes, from anatomy.** `somaLocation` centroids of the VNC leg
   neuromeres run T1 → T2 → T3 at z = 75.6, 94.2, 120.4 µm, so **z is the
   anterior→posterior axis**; `L1` somata of `somaSide` L and R sit at
   x = 86.1 µm and 10.6 µm, so **x is the left–right axis**; the VNC lies at
   y ≈ 57 µm against the brain's y ≈ 29 µm, so **y is the dorsal→ventral axis**.
3. **Which hex direction is azimuth.** Least squares of `L1` `somaLocation` on
   (hex1, hex2) within each eye gives the two lattice vectors. Their **difference**
   is horizontal — left eye (−829, +174, −752) nm, right eye (+945, +71, −625) nm,
   i.e. the dorsoventral component is ≤ 15 % — while their **sum** is vertical —
   (−143, −1138, 0) and (+29, −1159, −201) nm, i.e. essentially pure y with no
   anterior–posterior component. So **azimuth runs along (hex1 − hex2) and
   elevation along (hex1 + hex2)**, in both eyes, and the choice is fixed by
   anatomy rather than guessed. Residual rms of the planar fit is 3.5–3.9 µm
   against a column pitch of ~0.8 µm, which is the curvature of the lamina shell
   and is why step 4 works in hex space rather than in nanometres.
4. **Azimuth in degrees.** Let `ê` be the unit azimuth vector of that eye
   (the lattice-vector difference, with the dorsoventral component projected out,
   signed so that `ê · ẑ > 0`, i.e. pointing posteriorly). With
   `c_k = (lattice vector k · ê) / d_col` and `d_col` the eye's mean column pitch,
   the posteriority of a column is

       p(hex1, hex2) = Δφ · ( c₁·hex1 + c₂·hex2 ) ,  Δφ = 5.0 °/ommatidium

   and `p` is centred by subtracting the eye's median. `Δφ = 5.0°` is the standard
   *Drosophila* interommatidial angle (reported range ≈ 4.6–5.5°; Land, M.F. 1997,
   *Visual acuity in insects*, **Annual Review of Entomology** 42:147–177). It is a
   literature constant, it is not fitted, and it enters only as the number of
   columns per grating period (30° / 5° = 6 columns). The absolute offset is
   immaterial because the pattern is periodic.
5. **Azimuth around the fly.** With ψ measured counter-clockwise from straight
   ahead seen from above (front 0°, left 90°, back 180°, right 270°), which is the
   sign convention `docs/wp5_optomotor_prereg.json` already uses,

       ψ = 90° + p   (left eye)        ψ = 270° − p   (right eye)

   Elevation is ignored: the grating is a vertical sinusoidal grating, constant
   along elevation, which is what a yaw-rotating drum is. Declared as a
   simplification: the real drum's bars are also vertical, so what is lost is the
   eye's vertical sampling structure, not the stimulus.

### 2.3 The luminance field and the drive rule

One moving sinusoidal grating, in the prereg's own units, with the prereg's own
values (`docs/wp5_optomotor_prereg.json`, sha256 recorded in every output):
angular velocity ω = 0.7854 rad/s (45 °/s), spatial wavelength λ = 30°, temporal
frequency f = ω/λ = 1.5 Hz, direction s = ±1, contrast c ∈ {1.0, 0.5}, control
step 2 ms. Nothing is re-chosen.

Luminance at azimuth ψ, in normalised units L ∈ [0, 1] where 0 is black, 1 is
white and 0.5 is the gray screen:

    L(ψ, t) = 0.5 · ( 1 + c · cos( 2π·ψ/λ − 2π·f·t·s ) )

s = +1 moves the pattern toward increasing ψ, i.e. counter-clockwise seen from
above (leftward), which by §2.2 step 5 is front-to-back on the left eye and
back-to-front on the right — the same physical stimulus the existing encoder's `s`
means.

**Drive, primary arm P1 (luminance-faithful).**

    drive_i(t) = i_max · L(ψ_i, t) · (1 + σ·ξ_i(t)) ,  rectified at 0

with `i_max = 20.0` and `σ = 0.1` **taken unchanged from the preregistration**,
where they are declared as "equal to the WP4 validation stimulus amplitude; not
tuned". `ξ` is standard normal, drawn for every photoreceptor on every step
regardless of whether drive is delivered, so the RNG stream does not depend on the
stimulus or on the sham condition — the same discipline the existing encoder uses
and `tests/test_wp5_optomotor.py` already tests for.

**Why this scaling is derived and not tuned.** The peak per-cell drive is the
prereg's `i_max`, unchanged. The number of driven cells is 3,335 photoreceptors
against the existing encoder's 3,322 driven T4/T5 cells (2 populations × (835 T4 +
826 T5)) — within 0.4 %. The per-cell drive is the same rectified sinusoid, at the
same amplitude, with the same noise. The total input energy delivered to the graph
is therefore matched to §12/§13 **by construction**, and the single thing that
changes is **where it enters and that its phase is retinotopic instead of random**.
No amplitude was chosen by looking at a response, and none will be: if the graph's
answer is "too weak" or "too strong", that is the reported result.

Under P1 the gray screen is a uniform mean-luminance field, `L = 0.5`, so
`drive = 0.5·i_max = 10` on every resolved photoreceptor during gray. That is what
"gray" physically means in the preregistered assay — a lit drum with no contrast,
not darkness — and it is the arm in which an inhibitory first synapse has a
baseline to modulate in both directions. It is therefore the primary.

**Declared arm P2 (contrast-only / light-adapted).**

    drive_i(t) = i_max · c · 0.5 · ( 1 + cos(2π·ψ_i/λ − 2π·f·t·s) ) · (1 + σ·ξ)

i.e. the same field with the DC term removed, so gray delivers exactly zero drive.
This is the arm that matches §12/§13's convention that gray is zero input (and
hence the arm in which Q1's "quiet baseline" means what it meant there), and it is
a crude proxy for photoreceptor light adaptation. **P1 is the primary and P2 is a
declared sensitivity arm.** Both are chosen now, both will be reported whatever
they show, and neither may be promoted after the fact. No third parameterisation
will be run: if both fail, the answer is that they failed.

### 2.4 Identity and pins

The resolved map — per-eye photoreceptor `source_id` lists, their azimuths, the
monitor populations, and the layout rule string — is hashed into a new pin
`PHOTORECEPTOR_IO_PIN` in the **new** module `brainlab/io_map_photoreceptor.py`.
`brainlab/io_map.py` is not edited; `OPTOMOTOR_IO_PIN = 228c1b69…66c7b` keeps its
meaning, and the new map carries its own sha256 and its own name. The decoder is
the **existing** `DNa02YawDecoder` imported unchanged (gain 0.02 rad/s/Hz, τ 50 ms,
ipsilateral, unclipped, zero spikes → zero yaw), and the silencing clamp is the
existing `SILENCE_DRIVE = −200`.

Dynamics: **v3**, `dynamics_pin('v3')` unchanged, transmitter policy
`v3-modulatory-only` with `unclear_mode = 'excitatory'` (the §12 primary). Primary
`E_inh = −70 mV` for direct comparability with §12; the declared variant
`v3-einh-60` (§13's literature-supported value) is run as a second configuration
if budget allows, exactly as §13 defined it.

### 2.5 What is recorded

Per-cell spike rates, by cell type **and** annotated side, for the whole pathway,
so that §4's trace can be produced regardless of outcome:
`R1-R6` (by `rootSide`), `L1`, `L2`, `L3`, `L4`, `L5`, `Lai`, `T1`, `C2`, `C3`,
`Mi1`, `Mi4`, `Mi9`, `Tm1`, `Tm2`, `Tm3`, `Tm4`, `Tm9`, `CT1`, `Dm9`,
`T4a`, `T4b`, `T4c`, `T4d`, `T5a`, `T5b`, `T5c`, `T5d`,
`HS` (HSN+HSE+HSS), `H2`, `VS`, `LLPC1`, `LPi34`, `TmY15`, `PFL3`,
`DNa01`, `DNa02`, `DNa03`, `DNb01`, `DNp09`, `MDN`, plus network rate, membrane
range, and the DNa02 membrane and spike counts. Sides come from `somaSide` for
everything except `R1-R6`, which use `rootSide`.

## 3. Predictions, stated before measuring

**Do I expect T4/T5 to show direction selectivity? No — I expect a null, and I
expect the signal to die at or just after the first synapse.** The reasons are
structural and were computable before the run:

* **P1 — the first synapse is inhibitory into a silent network.** Every
  photoreceptor out-edge is negative (§2.1). Under v3 the graph has exactly zero
  spontaneous activity (§1 item 3). Inhibiting a neuron that is not firing has no
  observable consequence in a threshold-transmission engine. In the animal the
  signal is carried by *disinhibition* — light closes the histamine-gated channels
  and L1/L2 depolarise — which requires either a graded membrane or a maintained
  baseline. Arm P1 exists precisely to supply a baseline (uniform mean luminance
  makes photoreceptors fire tonically, so a grating modulates that rate up and
  down); whether the resulting modulation of an inhibitory input can push L1/L2
  and the medulla above `−45 mV` at all is the first thing §4 measures.
* **P2 — graded cells must spike or say nothing.** Every cell in this pathway is
  graded in the animal (§1). Here each must cross `−45 mV`. If the pathway's
  responses are subthreshold, the measured rates will be zero at some stage and
  the signal dies there; that is an engine property, not a connectome property.
* **P3 — no cell-type-specific delay.** The delay the correlator needs is of the
  order of tens of milliseconds and comes from per-cell-type filtering in the
  animal. This engine offers 1.8 ms per hop and one global 5 ms synaptic time
  constant (§1 item 2). At f = 1.5 Hz a 1.8 ms delay is 0.27 % of a cycle ≈ 1° of
  phase. Even if every other stage works, the correlator's temporal term is
  therefore expected to be small: an HR-type detector's response scales roughly as
  sin(spatial phase offset) · sin(ω·τ), and with one column of offset (≈ 60° of
  spatial phase at λ = 30°) and ω·τ ≈ 0.05–0.2 rad, the predicted direction
  selectivity is **of order 5–20 %, i.e. DSI ≲ 0.2**, against the ≈ 0.7–0.9
  reported for real T4/T5. Any positive result should be read against that gap.
* **P4 — no calcium stage.** Reported T4/T5 selectivity is measured largely in
  calcium, and the voltage-to-calcium transformation is itself part of the
  nonlinearity. A single-compartment LIF has no calcium state, so even a correct
  voltage computation would be reported here in the wrong variable.

**What would count as the approach working.** T4/T5 populations showing DSI of
consistent sign per subtype and eye (T4a/T5a of the left eye preferring s = +1,
T4b/T5b of the left eye preferring s = −1, mirrored on the right), an HS/H2
asymmetry inheriting that sign, and the existing gate's R1 passing at DNa02. In
that case the connectome, not an adapter, is doing the work — and §12/§13's
numbers become directly comparable.

**Falsifiers, each reported whichever way it comes out; none licenses a retune.**

* **G1 — the photoreceptors themselves do not fire.** Then the drive scaling is
  too small *by the derivation*, and that is reported, not adjusted.
* **G2 — the photoreceptors fire but nothing downstream does.** The inhibition-only
  first synapse (prediction P1). Reported with the L1/L2/L3 input budget as
  evidence.
* **G3 — the lamina and medulla respond but T4/T5 show DSI ≈ 0** (|DSI| < 0.05 in
  every subtype × eye). Then spatial offset plus sign plus one global filter is not
  enough, which is prediction P3.
* **G4 — T4/T5 are selective but the sign is inconsistent with the subtype map.**
  Then the graph computes *something* directional that does not agree with Maisak
  *et al.*; reported as such, with no relabelling of subtypes to fit.
* **G5 — T4/T5 are selective but the selectivity does not survive to DNa02.**
  Reported with the stage at which the sign is lost.
* **G6 — the network runs away** (gray rate ≥ 1.0 × 10⁵ spikes/s, i.e. Q1 fails).
  Then the photoreceptor drive, not the encoder's target, is what the v3 quiet
  baseline depended on. Q1 is not relaxed; the set is not run.

## 4. The diagnostic produced if selectivity does not emerge

This is declared in advance so it cannot be assembled to suit the answer. If the
gate's R1 fails, **the confirmatory set is not run** (§5) and the deliverable is:

1. **A stage-by-stage table**, per direction and per eye: mean per-cell spike rate
   and the fraction of cells that fired at least once, for
   `R1-R6 → L1/L2/L3 → Mi1/Tm3/Tm1/Tm2/Tm9/Mi9/Mi4/C3/CT1 → T4a-d/T5a-d → HS/H2/VS → DNa02/DNa01`,
   plus the DSI of every population and the network rate and membrane range.
   **The first stage whose rate is zero in both directions is where the signal
   dies**, and it is named.
2. **The membrane evidence for whether that death is subthreshold or absent.** For
   the first silent stage, the distribution of the membrane potential reached
   (max V per cell over the rotation window) against `−45 mV`. A population that
   sits at −52 to −46 mV is being *silenced by the threshold*; one that sits at
   `E_inh` is being *hyperpolarised*; one that never moves is receiving nothing.
3. **An attribution, by these pre-declared rules:**
   * **(a) graded cells never reach threshold** — if the first silent stage has
     max V within (`E_inh`, −45) mV for most cells, i.e. it *is* responding,
     subthreshold, and the engine discards it.
   * **(b) inhibition into silence** — if the first silent stage's only arriving
     weight is negative and its membrane sits at or below `V_rest`.
   * **(c) no tens-of-milliseconds delay mechanism** — if the pathway *does* reach
     T4/T5 with non-zero rates but DSI ≈ 0 in every subtype. Then the pathway
     conducts and the correlator does not correlate, which is the temporal-filter
     gap of §1 item 2 and prediction P3. Reported with the measured per-stage
     latency (time from stimulus onset to first spike per stage) as supporting
     evidence.
   * **(d) something else** — anything not covered above, described explicitly and
     not forced into (a)–(c).
4. **The structural reason, from the graph itself**: for the first silent stage,
   the Σ excitatory and Σ inhibitory in-weight and how much of it comes from the
   driven photoreceptors, so the reader can see whether any excitatory route into
   that stage exists at all.

**This trace is reported as the headline if it happens**, with the honest statement
that it says what the engine would need in order to be a fair test of the
connectome — not that the connectome cannot compute motion.

## 5. Acceptance gate and outcome metric — both reused verbatim

**Nothing here is new and nothing is weakened.**

* **Gate: Q1 / R1 / R2 exactly as `docs/LIF_DYNAMICS_SPEC.md` §6.7 defines them**,
  evaluated by the *same code*, copied verbatim from
  `outputs/wp5/v3-verify-20260926/gate.py` as §13 did:
  * **Q1 — quiet baseline.** In both gray windows: network rate < 1.0 × 10⁵
    spikes/s **and** DNa02_L and DNa02_R each < 20 Hz.
  * **R1 — responsiveness.** During rotation: network rate > 0, the sign of the
    DNa02 L−R rate difference follows the stimulus in **both** directions, and
    |L−R| ≥ 1 Hz in at least one direction.
  * **R2 — bounded membrane.** min V ≥ `E_inh` in every window.
  * Probe schedule identical to probe C: 500 ms gray, 1 s rotation, 500 ms gray,
    one run per direction, contrast 1.0, seed 0.
  * **The binding budget rule of §6.7 applies unchanged: if Q1 or R1 fails, the
    24-run confirmatory set is not run, and the failure is the result.**
* **Outcome metric: the preregistered `TI` of `docs/wp5_optomotor_prereg.json`,
  unchanged**, computed by the unchanged code path — `TI` = mean over the 8 blocks
  of `s_b` × mean decoded yaw in block b, with the same 4 conditions (intact,
  DNa02-silenced, sham, degree-preserving shuffle), the same seeds 0–5, the same
  block schedule, the same decoder, the same bootstrap (10,000 resamples,
  `default_rng(20260919)`) and the same POSITIVE/NEGATIVE/NULL decision rule. The
  result is therefore directly comparable with §12 (+0.0671 at `E_inh` −70) and
  §13 (+0.0452 at −60).
* **Additional reported measure, not a criterion:** the direction-selectivity
  index per population, `DSI = (r₊ − r₋)/(r₊ + r₋)` from the probe's rotation
  windows, where `r±` is the mean per-cell rate at `s = ±1`; undefined and
  reported as null when `r₊ + r₋ = 0`. Sign convention: positive DSI = prefers
  `s = +1`. Expected signs are in §3.

## 6. What this program may not do

* **No tuning.** `i_max`, `σ`, `λ`, `f`, `ω`, `Δφ`, the contrast values, the
  schedule, the seeds, the decoder, the gate thresholds, the transmitter policy,
  `E_inh` and the metric are all fixed above and inherited from documents already
  locked. If the response is too weak or too strong, that is the finding. Any
  further parameterisation would be a new pre-registration.
* **No change to the existing loop.** `brainlab/io_map.py`, `OPTOMOTOR_IO_PIN`,
  `docs/wp5_optomotor_prereg.json`, the v1/v2/v3 pins and §1–§13 of
  `docs/WP5_OPTOMOTOR.md` are untouched, so every published number stays
  reproducible.
* **No engine change.** In particular, graded or threshold-linear transmission is
  **not** implemented here, however clearly the diagnostic points at it. That
  would be a new declared dynamics version with its own pre-registration and pin,
  and mixing it into a measurement already under way would destroy the
  attribution this program exists to make.
* **No relaxation of the gate to reach the confirmatory set**, and no confirmatory
  set if R1 fails.
* **Model rates are not comparable in kind to the biological measurements.**
  Where a biological rate is quoted beside a model rate — for instance DNa02
  bursting at ≈ 93–128 spikes/s during steering in Rayshubskiy *et al.* (relayed
  to this session, **not retrieved here**) against v3's 6.5 Hz, or HS cells
  responding with ≈ 5 mV graded deflections rather than the 140 Hz the v1 engine
  produced — the comparison is flagged as a category difference, not a calibration.
