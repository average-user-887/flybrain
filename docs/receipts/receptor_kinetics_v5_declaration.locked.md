# LOCKED DECLARATION — LIF dynamics v5, per-receptor-class synaptic kinetics

Locked 2026-10-04, before v5 was run on the real graph for anything except a
200 ms no-stimulus wall-clock cost probe with a placeholder kinetics table
(no response recorded or read; `docs/receipts/v5_raw/speed_probe.json`), and
before any v5 or v4 measurement under the protocol of §9.9. Unit tests on
synthetic graphs (v5 with v4 kinetics bit-identical to v4 on CPU and GPU) had
been run; they measure the implementation, not the model. This file is the
hashed, verbatim declaration; `docs/LIF_DYNAMICS_SPEC.md` §9 reproduces it and
cites this file's sha256. If the two ever differ, this file is the declaration.

Nothing below was chosen after looking at a v5 result. Every time constant is
either a published number with its source or is labelled DECLARED ASSUMPTION
where it is used.

---

## 9.1 Why v5 exists, and what the literature says about the premise

v4 (§7, `WP5_OPTOMOTOR.md` §13) carried a photoreceptor-only grating to DNa02,
and T4/T5 showed no direction selectivity: tuning flat within 0.04 mV, every
subtype hyperpolarising in every direction, no anti-parallel subtype pairs. The
named blocker was that the engine has **one** synaptic time constant (5 ms) and
**one** delay (1.8 ms) for every synapse, so the mechanism Gruntman, Romani &
Reiser 2018 identify in T4 — fast excitation plus offset, slower inhibition —
cannot be expressed.

v5 tests the most literal reading of that diagnosis: give every synapse the
published kinetics of its **receptor class**. Before choosing parameters, the
published record was retrieved (§9.3). It says something this declaration has
to state first, because it predicts the outcome:

* **Measured receptor-class kinetics in *Drosophila* are milliseconds.**
  Nicotinic mEPSC decay 1.4 ms and GABAergic mIPSC decay 3.7 ms in the same
  preparation (Su & O'Dowd 2003); adult *in situ* nicotinic decay about 5 ms at
  the soma (Gu & O'Dowd 2006); histamine-gated chloride channel kinetics
  sub-millisecond to 6 ms (Pantazis *et al.* 2008); no *Drosophila* GluCl
  synaptic decay has been published at all.
* **The time constants T4/T5 models need are 50–300 ms, and every model
  examined puts them in the input cells, not the receptors.** Gruntman 2018's
  fitted inhibitory conductance decays in 154–307 ms, but it lumps the whole
  input pathway (upstream cell + synapse) and is fitted to T4 voltage. Borst
  2018 uses instantaneous synapses and 50 ms low-pass / 250 ms high-pass
  *input-cell* filters; Zavatone-Veth *et al.* 2020 a 150 ms input filter with
  instantaneous synapses; Groschner *et al.* 2022 the measured input-neuron
  voltages themselves as conductance time courses; Kohn *et al.* 2021 conclude
  that input *filter shape* is "necessary and sufficient"; the connectome-wide
  models (Lappalainen *et al.* 2024; Borst 2025) use instantaneous synapses and
  per-cell-type membrane time constants or intrinsic currents (an H-current in
  L1/L2).

So receptor-class kinetics are one to two orders of magnitude too fast to be
the delay line, and the literature locates the timing in cell-level filtering.
**This declaration therefore predicts that v5 will not produce direction
selectivity (§9.7),** and v5 is run as the decisive test of that one
hypothesis, at the cost it actually has (§9.8). If it fails, the deliverable is
the diagnosis that the missing ingredient is cell-level temporal filtering, not
synaptic kinetics, and §9.11 names what that would require.

v1–v4 are not deleted, superseded or reinterpreted. All stay selectable, keep
their pins (v1 `1b17bfc6…f146`, v2 `1ea33ba1…eb56`, v3 `5739c4f4…4a41`, v4
`7ae09665…ed44`), stay bit-reproducible, and every number published under them
stands. The default stays v3.

## 9.2 Granularity: per receptor class, and why

Three granularities were considered.

| granularity | what it can express | what it cannot | assumptions added | cost |
|---|---|---|---|---|
| **per receptor class** (4 classes) | inhibition slower than excitation everywhere it is so; class-specific shunting time course; applies brain-wide with no cell type singled out | any difference between two cells of the same transmitter (Mi1 vs Tm3; Tm1/Tm2/Tm4 vs Tm9 — all cholinergic); any cell-level filter | 4 time constants, 3 of them published | 3 kinetic channels; ≈ v4 or faster (§9.8) |
| per cell type | the measured input-cell filters (Behnia 2014; Arenz 2017) | nothing in principle about the inputs | ≈ one filter per type, most unpublished as numbers (Arenz 2017's table could not be retrieved); filters measured *in the animal* already contain the upstream circuit, so imposing them on top of the graph double-counts it and hands the answer to the model | per type, a filter state per neuron |
| per connection | everything | — | ≈ 25 M parameters with no source | prohibitive |

**Adopted: per receptor class.** Reasons, in order:

1. It is the coarsest granularity, and the only one for which published
   *Drosophila* numbers exist for most classes.
2. It is a property of the synapse type, applied identically to all 166,700
   neurons. Nothing is chosen for the motion pathway; the connectome has to do
   the work with the same kinetics every other circuit gets. Per-cell-type
   filters for exactly the T4/T5 inputs would be the hand-built adapter the
   owner's constraint excludes, and would make a positive result uninformative.
3. It is exactly the hypothesis the v4 write-up named, so testing it is the
   honest next step even though §9.1 predicts it fails.

The literature (§9.1) says this granularity is **not** sufficient for a
Reichardt / Barlow–Levick motif at 1.5 Hz. It is chosen because it is the
hypothesis on the table and the only one implementable without importing the
answer; it is not chosen in the expectation that it succeeds.

## 9.3 The receptor classes and their declared kinetics

Class = the fast ionotropic receptor implied by the **presynaptic** neuron's
released transmitter label (`normalized/neurons.feather`, the column the v3
policy already reads). One class per neuron, because the table records one
transmitter per neuron.

| transmitter | neurons | class | reversal (unchanged) |
|---|---|---|---|
| acetylcholine | 103,720 | `nicotinic` (nAChR) | E_exc = 0 mV |
| GABA | 22,069 | `gaba_a` (Rdl) | E_inh = −70 mV |
| glutamate | 29,302 | `glucl` (GluClα) | E_inh = −70 mV |
| histamine | 7,891 | `hiscl` (ort / HisCl1) | E_inh = −70 mV |
| unclear (2,999), nan (178) | 3,177 | `nicotinic` — DECLARED ASSUMPTION: their v3 sign is the declared excitatory proxy and nicotinic is the only excitatory fast class | E_exc |
| dopamine, octopamine, serotonin | 541 | `nicotinic` — irrelevant: zero fast weight under v3 | — |

Decay time constants (single exponential; rise neglected, §9.4):

| class | **primary** τ (ms) | source / status | upper arm S2 τ (ms) | source / status |
|---|---|---|---|---|
| nicotinic | **1.4** | Su & O'Dowd 2003, *J Neurosci* 23:9246, Table 1: Kenyon-cell mEPSC decay 1.4 ± 0.1 ms (n = 38); cultured pupal neurons, room temperature. Lee & O'Dowd 1999 (embryonic culture) 2.1 ± 0.2 ms agrees. | 5.23 | Gu & O'Dowd 2006, *J Neurosci* 26:265, Table 1: adult *in situ* KC mEPSC 5.23 ± 0.41 ms (OK107, n = 14), measured at the soma and slowed by the cable (the authors' own caveat). |
| gaba_a (Rdl) | **3.7** | Su & O'Dowd 2003, Table 1: KC mIPSC decay 3.7 ± 0.9 ms (n = 12), same preparation as the nicotinic value. No adult *in situ* value exists. | 13.8 | DECLARED EXTRAPOLATION: 5.23 × (3.7 / 1.4). |
| glucl | **3.7** | **DECLARED ASSUMPTION = gaba_a.** No *Drosophila* GluCl synaptic decay is published (Liu & Wilson 2013 report none; Cully 1996 and Molina-Obando 2019 say only "rapidly desensitizing"). A nematode GluCl IPSC (17–40 ms; Atif *et al.* 2019) is not transferable and is not used. | 13.8 | same assumption |
| hiscl | **1.0** | Pantazis *et al.* 2008, *J Neurosci* 28:7250, Table 3: slow Lorentzian component of the ort (HCLA) homomer, 1.0 ± 0.2 ms. **Channel-noise kinetics in S2 cells, not a PSC decay; no PSC decay is published.** Consistent with "no detectable synaptic delay" and LMCs peaking before photoreceptors (Mansour *et al.* 2026; Pantazis 2008). | 6.1 | Pantazis 2008: slowest histamine-receptor Lorentzian (HCLB homomer), 6.1 ± 1.0 ms. |

All voltage-clamp numbers above were recorded at room temperature; no
temperature correction is applied (DECLARED). The primary table has three
distinct values, so three kinetic channels (1.4 ms; 3.7 ms; 1.0 ms).

**Transmission delay: unchanged, 1.8 ms for every class.** No class-specific
synaptic delay has been published for *Drosophila*; at the photoreceptor
synapse none is detectable. Gruntman 2018 found no relative delay was needed.

## 9.4 Equations

For K kinetic channels (classes sharing a τ share a channel), each neuron
carries an excitatory and an inhibitory conductance per channel, `ĝ_e,k`,
`ĝ_i,k`, in leak units. The membrane and its integrator are v4's with

    ĝ_e = Σ_k ĝ_e,k ,   ĝ_i = Σ_k ĝ_i,k
    g_tot = 1 + ĝ_e + ĝ_i
    V_inf = (E_leak + ĝ_e·E_exc + ĝ_i·E_inh + I_ext) / g_tot
    V     ← V_inf + (V − V_inf)·exp(−dt·g_tot/tau_m),  clamped to [E_inh, E_exc]
    ĝ_·,k ← ĝ_·,k · exp(−dt/τ_k)

An arrival over edge e from presynaptic neuron i in channel k = chan(i), with
emission x (1 for a delivered spike, `r(V_i)·dt/1000` for a graded cell, both
exactly as v4):

    Δĝ_(sign of w_e),k(post) = |w_e| · g_unit_(sign) · q_k · x
    q_k = (1 − exp(−dt/τ_k)) / (1 − exp(−dt/τ_ref)),   τ_ref = 5 ms

**The charge-preserving quantum q_k is a derivation, not a fit.** It is the
unique per-channel scale that keeps, in the engine's own discrete time, the
time-averaged conductance per unit release rate equal to v4's. Consequences:
every tonic set point v4 derived (§7.4) is unchanged, the PSP *area* at rest
is unchanged to first order, and **only the time course of each synaptic event
changes** — which is exactly the variable the v4 diagnosis named. The
alternative, preserving each unitary PSP's *peak* (the v3 rule, §6.2), would
rescale every class's efficacy and change the operating point at the same time
as the timing, so a v5 effect could not be attributed to timing. Rejected for
that reason. With every τ = 5 ms there is one channel, q = 1.0 exactly, the
state is (2, n), and v5 is bit-identical to v4 by construction (§9.5, S0).

Not modelled, declared: rise times (0.4–0.8 ms measured, below the timescales
of §9.7), desensitisation, GABA_B / mGluR metabotropic components, short-term
plasticity, receptor subunit heterogeneity within a class, temperature.

Every other constant — `dt`, `tau_m`, `V_rest`, threshold, reset, refractory
period, both reversals, both per-sign quanta, the membrane bounds, the graded
class list `v4-optic-lobe-graded`, the release function and its baseline, the
v3 transmitter policy — is v4's, unchanged.

## 9.5 Identity, pins, state, arms

* `dynamics_version = 'v5'`, controller `brainlab-lif-v5`; declared dict
  `graph_identity.LIF_DYNAMICS_V5`, pin `dynamics_pin('v5')`. v1–v4 dicts and
  pins are byte-unchanged (tested). Default stays v3.
* `Brain(..., dynamics='v5', kinetics=<table>)`; tables live in
  `brainlab.receptor_kinetics.KINETICS`. The per-neuron class assignment and
  the table are hashed (`kinetics_sha256`) into every snapshot; a snapshot is
  refused by a brain with a different version, graded set or kinetics hash.
* Synaptic state is (2K, n); the v4 release ring is kept.
* CPU reference kernel `engine.advance_v5`; GPU `cupy_v5.CupyV5State`.

Declared arms (none can change the primary verdict):

* **S0 — `v4-equivalent`** (every τ = 5 ms): must be bit-identical to v4 on CPU
  and on the GPU, on synthetic graphs and on the real graph (§9.9 a3).
* **S2 — `v5-receptor-class-upper`** (§9.3 right-hand columns): the slowest
  defensible receptor kinetics, run through the same protocol. Tests whether
  the verdict depends on where in the published range the values sit.

No other table is run. If the primary and S2 fail, the answer is that they
failed; no third table is tried and no τ is adjusted.

## 9.6 What v5 can and cannot express, stated before measuring

Arithmetic at the declared 1.5 Hz: a first-order filter with τ delays a
sinusoid by `atan(2πfτ)`. τ = 1.4 ms → 0.76°; 3.7 ms → 2.0°; 13.8 ms → 7.4°.
The primary's excitation/inhibition phase difference is **≈ 1.2°**, i.e. an
equivalent 2.3 ms; S2's ≈ 4.6°, about 8.6 ms. v4's is 0°. The T4 models of
§9.1 need tens to hundreds of ms. The intrinsic delay of any one hop is still
dominated by `tau_m = 20 ms` (≈ 10.7° at 1.5 Hz), which v5 does not change.

## 9.7 Predictions

* **P1 (certain by construction, checked):** each class's unitary PSP decays
  with its declared τ; PSP area at rest equals v4's within 5 %; S0 is
  bit-identical to v4.
* **P2 (high confidence): no T4/T5 subtype passes the gate.** D1 (§9.9) DSI
  < 0.2 or response magnitude < 0.5 mV in every subtype, as in v4.
* **P3 (high confidence): v5 is close to v4 everywhere.** Per population, the
  D1 tuning curve differs from v4's re-measured one by less than 0.05 mV in
  every direction, the D2 median per-cell F1 DSI differs by less than 0.05, and
  the D4 flicker phase differs by less than 5° (≈ 9 ms). S2 is allowed up to
  twice these.
* **P4 (moderate): no anti-parallel T4a/T4b or T4c/T4d pairs** under D1 or D2.
* **P5 (moderate): HS stays sub-millivolt** to the yaw grating and its sign
  does not depend on direction (F5 of §7.8 again).
* **P6 (moderate-high): the DNa02 clause fails** — the v4 standing left bias
  (L − R > 0) persists in both yaw directions.
* **P7 (cost, measured pre-lock):** with K = 4 placeholder channels v5 ran at
  0.0081 simulated s per wall s against v4's 0.0045 on the same GPU, because
  its per-channel matrices drop the explicit zeros v4's two all-edge matrices
  carry. Predicted: v5 primary (K = 3) no slower than v4.

## 9.8 Falsifiers

Each is reported whichever way it comes out; none licenses changing a τ, a
class assignment, a gain or the protocol.

* **F1 — S0 not bit-identical to v4** (synthetic or real graph). Then v5 changed
  something other than kinetics and nothing it shows is attributable.
* **F2 — a unitary PSP does not decay with its declared τ**, or its area differs
  from v4's by > 5 %. Implementation error.
* **F3 — the gate passes.** Falsifies P2 and the §9.1 reading that receptor
  kinetics are too fast to matter; reported as the headline positive result,
  and (d) is then run.
* **F4 — v5 departs from v4 by more than P3's bounds.** Falsifies the claim that
  millisecond kinetics are negligible here; reported with where it happens.
* **F5 — the instrument fails its null controls**: R1-R6 D1 DSI ≥ 0.1, or R1-R6
  D2 median per-cell F1 DSI ≥ 0.05. A photoreceptor cannot be direction
  selective; if the instrument says otherwise, no DSI at that scale is a
  finding (v4's 500 ms window gave R1-R6_L a DSI of 0.505).
* **F6 — saturation or collapse**: > 50 % of graded cells at a membrane bound in
  the gray window (§7.8 F2/F3, inherited).
* **F7 — the verdict depends on the arm**: S2 passes the gate where the primary
  fails, or the reverse. Then the verdict rests on where in the published range
  τ is taken, and is reported as such.

## 9.9 Predeclared measurements, cheap before expensive

Raw output of every measurement is written straight into
`docs/receipts/v5_raw/` and committed as soon as it finishes.

**(a) Unit level — CPU, synthetic graphs, seconds.**
 a1. Two-neuron probes, one per class: a presynaptic spiking cell of that class
 fires once onto a passive graded target; report peak, time-to-peak, decay τ
 (fitted from 20 % of peak onward), area of the target's PSP, for the primary,
 S2 and S0.
 a2. The same with a graded presynaptic cell stepped by +4 mV for 50 ms: the
 target's conductance time course per class.
 a3. S0 bit-identity: v5 `v4-equivalent` vs v4, CPU and GPU, synthetic graph
 (tests) and real graph (200 ms of the gray protocol, every state array
 compared).

**(b)/(c) Real graph — GPU.** The v4 photoreceptor protocol of §7.10(c) —
same `PhotoreceptorGratingEncoder`, same `PHOTORECEPTOR_IO_PIN`, i_max 20, λ
30°, f 1.5 Hz, contrast 1, no noise, eight lattice-plane directions 0°…315°,
graded policy `v4-optic-lobe-graded`, E_inh −70 mV — with these **declared
protocol changes**, applied identically to v4, primary and S2:

* *Gray:* 1,000 ms settle from rest, then a 1,000 ms baseline window (v4
  measured its baseline during the first 500 ms after reset, i.e. inside the
  settling transient).
* *Grating:* every condition starts from the same post-gray state (snapshot;
  the run is deterministic), 500 ms of grating to pass the onset transient,
  then a **2,000 ms response window = exactly 3 cycles** of the 1.5 Hz grating
  (v4: 500 ms = 0.75 cycles).
* *Yaw conditions:* the eight lattice directions apply the same direction to
  both eyes, so none of them is a yaw rotation. From anatomy (the declared
  lattice derivation of `PHOTORECEPTOR_ENCODER.md` §2.2, computed in
  `scripts/v5_measurement/yaw_axes.py`, `docs/receipts/v5_raw/yaw_axes.json`),
  front-to-back is 123.13° on the left eye and 123.94° on the right eye in the
  encoder's lattice frame — so v4's 135° and 315° were near-bilateral
  front-to-back and back-to-front, not rotations. Two conditions are added:
  **yaw CCW** (left eye 123.13°, right eye 303.94°: front-to-back on the left,
  back-to-front on the right, i.e. leftward motion seen from above) and
  **yaw CW** (left 303.13°, right 123.94°), via
  `PhotoreceptorGratingEncoder.encode_per_eye` (bit-identical to `encode` when
  both eyes share a direction; tested).
* *Flicker:* one spatially uniform condition (zero spatial frequency, same f,
  mean and contrast) to read each population's temporal phase.

Sampling: every 2 ms control step. Per neuron and window: mean V, the first
Fourier component `F1 = (2/N) Σ V(t_k) e^{−i2πf t_k}`, spike count.

Derived (`scripts/v5_measurement/analyse_tuning.py`, fixed now):

* **D1 — the gate statistic, verbatim from §7.10(c):** per population, mean V
  in the response window minus the gray window (graded, mV) or rate minus gray
  rate (spiking, Hz); rectify at 0; pref = argmax over the eight directions;
  `DSI = (R⁺_pref − R⁺_anti)/(R⁺_pref + R⁺_anti)`; magnitude = max |R|.
  *Declared limitation, stated before measuring:* in a linear network a
  population's mean (DC) response to a drifting grating does not depend on
  direction at all; D1 can register direction selectivity only through a
  nonlinearity (shunting, the membrane bounds, spiking). It is kept as the gate
  so v4 and v5 are judged on identical terms.
* **D2 — per-cell F1 direction selectivity (diagnostic):** per graded cell,
  `A(θ) = |F1(θ)|`; over the four axes, `d = (A(θ) − A(θ+180))/(A(θ) + A(θ+180))`,
  keep the axis with the largest |d|, preferred direction = θ or θ+180 by its
  sign. Report per population the median and 90th percentile of |d|, the
  fraction of cells with |d| ≥ 0.2, the histogram of preferred directions, and
  the |d|-weighted circular mean preferred direction. Any linear network whose
  paths all share one temporal kernel gives |d| = 0 exactly (opposite
  directions give complex-conjugate sums), so D2 is the measure that sees
  space–time inseparability, which is what a correlator needs.
* **D3 — null controls:** R1-R6 and L1/L2 under D1 and D2 (F5).
* **D4 — flicker phase:** per population the phase lag of the mean F1 relative
  to the drive, its sign relative to light (same / inverted), and the
  equivalent delay; compared with the published input-cell order (Behnia
  *et al.* 2014: filter peaks Mi1 71 ms vs Tm3 53 ms, Tm1 56 ms vs Tm2 43 ms;
  Arenz 2017, qualitative: Mi4 and Mi9 slow and sustained, Mi1 and Tm3 fast
  and transient).
* **D5 — HS:** per-cell mean-V response to yaw CCW and CW minus gray, in mV,
  against ~5 mV in the animal and Schnell *et al.* 2010's sign (HS depolarises
  to front-to-back motion on its own eye, hyperpolarises to back-to-front);
  HS_L's front-to-back is CCW, HS_R's is CW.
* **D6 — DNa02:** rates in gray and every condition; L − R.

**Runs, in this order, one graph instance at a time on the GTX 1660 Ti:**
1. a3 real-graph S0 check (minutes).
2. v4, full protocol (the like-for-like reference under the corrected protocol).
3. v5 primary, full protocol.
4. v5 S2, full protocol.

**The gate on (d), reused verbatim from §7.10:** (c) must show **both** DSI
≥ 0.2 in at least one T4/T5 subtype whose response magnitude is at least
0.5 mV (D1), **and** a direction-dependent signal reaching DNa02,
|rate(DNa02_L) − rate(DNa02_R)| ≥ 1 Hz with the sign of the difference
following the stimulus in both horizontal directions. *Operational definition
of the second clause, declared now because v4's eight directions contain no
horizontal rotation:* yaw CCW must give L − R ≥ +1 Hz and yaw CW must give
L − R ≤ −1 Hz (syndirectional, the optomotor sign; DNa02 steers
ipsilaterally, Rayshubskiy *et al.* 2020; the WP5 decoder's `+` = CCW). The
v4-style reading (L − R reversing between θ and θ+180 among the eight
directions) is reported beside it and does not decide the gate.

**(d) Only if the gate passes for the primary:** the preregistered optomotor
protocol through master's validation harness (`validation/specs`, a new v5
spec), compared with the v3-2 receipt. **If the gate fails, (d) is not run**,
and the deliverable is the diagnostic: where selectivity is absent and what
the next blocker is.

## 9.10 What v5 does not claim

* Not a biophysical model: one compartment, single-exponential conductances, no
  rise, no desensitisation, no metabotropic receptors, no gap junctions.
* The class of a synapse is inferred from the presynaptic transmitter label;
  receptor expression on the postsynaptic side is not modelled.
* The GluCl time constant is an assumption; the HisCl one is a channel-noise
  number, not a PSC decay; the S2 inhibitory values are an extrapolation.
* Charge preservation is a modelling choice that isolates timing; it is not a
  measured property of these synapses.
* v5 imposes no cell-type-specific filter. If v5 fails, that is a statement
  about receptor kinetics, not about the connectome.

## 9.11 If the gate fails: the next blocker, stated in advance

The candidates, ranked by what §9.1's literature says:

1. **Cell-level temporal filtering of the T4/T5 inputs** — sustained/low-pass
   Mi4, Mi9, Tm9, CT1 against transient/band-pass Mi1, Tm3, Tm1, Tm2, Tm4 —
   which the models attribute to intrinsic conductances (e.g. an H-current in
   L1/L2, Borst 2025) or fit as per-type membrane time constants (Lappalainen
   *et al.* 2024). A faithful engine needs per-cell-type intrinsic dynamics with
   their own sources; fitting them to motion responses would be a trained
   model, not a test of the connectome.
2. **Gap junctions** in the lamina and CT1.
3. **Dendritic compartmentalisation** in T4/T5 (Gruntman's EM-reconstructed
   multicompartment T4) and CT1 (Meier & Borst 2019).
4. **The incompletely sampled retinotopic lattice** (824 lit cartridges of
   ≈ 1,600; v4 §13.2).

## 9.12 References added for v5

Retrieval status as recorded by this session (FULL = full text read; ABSTRACT =
abstract only; NOT = could not be retrieved).

* Su, H. & O'Dowd, D.K. (2003). *J Neurosci* 23:9246–9253. PMC6740836. FULL.
* Lee, D. & O'Dowd, D.K. (1999). *J Neurosci* 19:5311–5321. PMC6782340. FULL.
* Lee, D., Su, H. & O'Dowd, D.K. (2003). GABA receptors containing Rdl
  subunits mediate fast inhibitory synaptic transmission in *Drosophila*
  neurons. *J Neurosci* 23:4625–4634. PMC6740792. FULL (decay values plotted
  only).
* Gu, H. & O'Dowd, D.K. (2006). Cholinergic synaptic transmission in adult
  *Drosophila* Kenyon cells *in situ*. *J Neurosci* 26:265–272. PMC6674319. FULL.
* Kazama, H. & Wilson, R.I. (2008). *Neuron* 58:401–413. PMC2429849. FULL (no
  unitary EPSC decay reported).
* Gouwens, N.W. & Wilson, R.I. (2009). *J Neurosci* 29:6239–6249. PMC2709801.
  FULL (model-inferred dendritic conductance decay 0.6–1.1 ms).
* Wilson, R.I. & Laurent, G. (2005). *J Neurosci* 25:9069–9079. PMC6725763.
  FULL (no IPSC kinetics).
* Liu, W.W. & Wilson, R.I. (2013). *PNAS* 110:10294–10299. PMC3690841. FULL
  (no GluCl kinetics).
* Rohrbough, J. & Broadie, K. (2002). *J Neurophysiol* 88:847–860. ABSTRACT.
* Cully, D.F. *et al.* (1996). *J Biol Chem* 271:20187–20191. ABSTRACT.
* Molina-Obando, S. *et al.* (2019). *eLife* 8:e49373. PMC6845231. FULL.
* Atif, M. *et al.* (2019). *PLoS Pathog* 15:e1007570. PMC6368337. FULL
  (nematode; not used).
* Hardie, R.C. (1989). *Nature* 339:704–706. ABSTRACT.
* Gengs, C. *et al.* (2002). *J Biol Chem* 277:42113–42120. ABSTRACT.
* Zheng, Y. *et al.* (2002). *J Biol Chem* 277:2000–2005. ABSTRACT.
* Pantazis, A. *et al.* (2008). Distinct roles for two histamine receptors
  (hclA and hclB) at the *Drosophila* photoreceptor synapse. *J Neurosci*
  28:7250–7259. PMC6670387. FULL.
* Zheng, L. *et al.* (2006). *J Gen Physiol* 127:495–510. PMC2151524. FULL.
* Juusola, M., Uusitalo, R.O. & Weckström, M. (1995). *J Gen Physiol*
  105:117–148. ABSTRACT.
* Mansour, N. *et al.* (2026). *Nat Commun* 17. PMC13144400. FULL.
* Skingsley, D.R., Laughlin, S.B. & Hardie, R.C. (1995). *J Comp Physiol A*
  176. NOT.
* Behnia, R. *et al.* (2014). *Nature* 512:427–430. PMC4243710. FULL.
* Arenz, A. *et al.* (2017). *Curr Biol* 27:929–944. ABSTRACT (fitted filter
  table not retrieved).
* Serbe, E. *et al.* (2016). *Neuron* 89:829–841. ABSTRACT.
* Fisher, Y.E. *et al.* (2015). *Curr Biol* 25:3178–3189. ABSTRACT.
* Strother, J.A. *et al.* (2017). *Neuron* 94:168–182. ABSTRACT. (Cited in §7.5
  as *Curr Biol* 27:3132; the correct citation is this one.)
* Meier, M. & Borst, A. (2019). *Curr Biol* 29:1545–1550. ABSTRACT.
* Kohn, J.R. *et al.* (2021). *Curr Biol*. PMC8725177. FULL.
* Gruntman, E., Romani, S. & Reiser, M.B. (2018). *Nat Neurosci* 21:250–257.
  PMC5967973. FULL, including Supplementary Table 1.
* Gruntman, E., Romani, S. & Reiser, M.B. (2019). *eLife* 8:e50706.
  PMC6917495. FULL.
* Borst, A. (2018). *PLoS Comput Biol* 14:e1006240. PMC6016951. FULL.
* Zavatone-Veth, J.A., Badwan, B.A. & Clark, D.A. (2020). *J Vis* 20(2):2.
  PMC7343402. FULL.
* Groschner, L.N. *et al.* (2022). *Nature* 603:119–123. PMC8891015. FULL.
* Ramos-Traslosheros, G. & Silies, M. (2021). *Nat Commun* 12. PMC8371135. FULL.
* Lappalainen, J.K. *et al.* (2024). *Nature* 634:1132–1140. PMC11525180. FULL.
* Borst, A. (2025). *J Comput Neurosci* 53:507–520. PMC12672718. FULL.
* Schnell, B. *et al.* (2010) and Rayshubskiy, A. *et al.* (2020): as cited in
  §7.11 and §8.
