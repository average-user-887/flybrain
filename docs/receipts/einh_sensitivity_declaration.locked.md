# E_inh sensitivity: is the v3 optomotor result a property of the connectome or of one assumed constant?

**Status: PRE-REGISTRATION. Written and hashed before any value in §4, §5 or §6
was measured.** §1–§3 (the literature, the sweep, the criteria) and — critically
— §4 (the decision rule) were fixed first. §5 and §6 are filled in afterwards and
are not permitted to change §1–§4. The lock is recorded in
`docs/receipts/einh_sensitivity.json` under `declaration_lock`.

Nothing here changes the calibration, the transmitter policy, the preregistered
outcome, the preregistration file or the Q1/R1/R2 gate. `E_inh` is the only thing
that moves, and the inhibitory conductance quantum follows it because
`LIF_DYNAMICS_SPEC.md` §6.2 *derives* that quantum from the driving force at rest
(`g_unit_inh = 1/(V_rest − E_inh)`). That coupling is the declared rule, not a
second knob, and it is the reason a sweep of `E_inh` is a sweep of the whole
inhibitory scale.

**v1, v2 and v3 are untouched.** Their pins, numbers and checkpoints stand. Each
swept value is a **new labelled variant** `v3-einh<value>` with its own pin from
`brainlab.graph_identity.dynamics_variant_pin`; `dynamics_pin('v3')` is
byte-unchanged at
`5739c4f4ffb84979d90ad430d2405bfc0b7736c7c75f2a6e676ef606dffb4a41`, and
`dynamics_variant('v3', None)` is byte-identical to `LIF_DYNAMICS_V3`.

---

## 0. Why this exists

`docs/WP5_OPTOMOTOR.md` §12 reports a POSITIVE preregistered optomotor verdict
under v3 (intact `TI` +0.0671 [+0.0524, +0.0829], 6/6 seeds; DNa02-silenced and
sham identically zero; shuffled −0.0151; paired dz 3.18). Every quantity in the
v3 engine that touches inhibition is derived from one number that no
*Drosophila* measurement supplies: `E_inh = −70 mV`, declared in spec §3.1 as an
**engineering assumption** and re-declared in §6.9 as one.

Two facts already on record make that assumption load-bearing rather than
cosmetic:

* the v3 high-conductance fixed point clears threshold by **0.13 mV** (spec §6.5,
  "This is a knife edge and is declared as one"); and
* declared arm **S2** (`E_inh = −56 mV`) **failed R1** at probe level — +11 Hz
  leftward and +2 Hz rightward, i.e. left-biased in *both* directions — while
  declared arm **S1** (`unclear_mode='zero'`) passed the probes with a *larger
  and more symmetric* asymmetry (+18 / −18 Hz) than the primary (+8 / −3 Hz).
  Neither arm was ever run at confirmatory level.

So the headline result may be an artefact of one assumed constant, in exactly the
way v1's was an artefact of an unbounded membrane. This document decides, in
advance, what each possible outcome will be taken to mean.

---

## 1. What the literature actually supports

All papers below were retrieved (2026-09-26) and are cited from what was
actually read. Where a full text was paywalled, that is stated and the value is
attributed to the retrievable abstract or quotation rather than asserted.

### 1.1 The measured numbers, and why they disagree

Fast inhibition in *Drosophila* is chloride throughout — GABA via Rdl, glutamate
via GluClα, histamine via ort/HisCl — so `E_inh = E_Cl`. `E_Cl` is not a property
of the receptor; it is set by the chloride gradient across the membrane. In a
whole-cell recording the pipette **imposes** the intracellular chloride, so a
reported "GABA reversal potential" is in large part a readout of the internal
solution. That is the single most important thing to understand about this
literature, and it is demonstrated directly:

| source | preparation | reported reversal | internal Cl⁻ context |
|---|---|---|---|
| Grolleau *et al.* 2000, *Br J Pharmacol* 130:1833–1842, DOI [10.1038/sj.bjp.0703507](https://doi.org/10.1038/sj.bjp.0703507) | RDL homomer in a *Drosophila* S2 cell line | `E_GABA = −1.4 mV` | near-symmetrical Cl⁻; "close to the calculated chloride equilibrium potential" |
| Lee, Su & O'Dowd 2003, *J Neurosci* 23:4625–4634 | cultured **embryonic** *Drosophila* neurons | x-intercepts **−44.5 mV and 0 mV** in two conditions | explicitly "close to the theoretical equilibrium potential for chloride in the two conditions" — the same cells give either number depending on the pipette |
| Rohrbough & Broadie 2002, *J Neurophysiol* 88:847–860, DOI [10.1152/jn.2002.88.2.847](https://doi.org/10.1152/jn.2002.88.2.847) | **larval** central neurons | **−56 ± 3 mV** (6 of 9 neurons) | "in reasonable agreement with an ionic dependence on Cl⁻". Full text is paywalled (HTTP 403); the value is taken from the retrievable quotation and matches the value spec §3.1 already cites. |
| Su & O'Dowd 2003, *J Neurosci* 23:9246–9253 | **adult** Kenyon cells in situ | **−37 ± 3 mV** against a *theoretical* −45 mV for their internal | already cited in spec §3.1; again pipette-determined |
| Wilson & Laurent 2005, *J Neurosci* 25:9069–9079 | **adult** antennal lobe, in vivo whole cell | GABA pulse from threshold (−39 ± 1 mV) hyperpolarised local-neuron somata to **−56 ± 2 mV** | internal: 140 K-aspartate, 10 HEPES, **1 KCl**, 4 MgATP, 0.5 Na₃GTP, 1 EGTA; external 103 mM NaCl |

**The Wilson & Laurent row is the most informative and the most awkward.** Their
pipette contains ~1–5 mM Cl⁻ against ~103 mM outside, so the Nernst `E_Cl` for
that internal is roughly **−120 mV**. A GABA pulse should therefore have driven
the soma far below −70 mV. It stopped at **−56 mV**. Whatever the mechanism
(incomplete dialysis of the processes where the synapses are, active Cl⁻
regulation, or a mixed conductance), the *operative* endpoint of GABAergic
hyperpolarisation in an intact adult *Drosophila* central neuron is near −56 mV
even under a pipette that should have imposed −120 mV.

### 1.2 The one adult constraint that does not depend on a pipette

Gouwens & Wilson 2009, *J Neurosci* 29:6239–6249,
[PMC2709801](https://pmc.ncbi.nlm.nih.gov/articles/PMC2709801/), adult antennal
lobe PNs: *"Lowering [Cl⁻]e did not significantly change the resting membrane
potential, so we conclude that there is little chloride current flowing at rest
in PNs."* They put the true resting potential at **−55 to −60 mV** with intact
ORN input and **≈ −65 mV** with ORN input removed.

"Little chloride current at rest" means `E_Cl ≈ V_rest`. That is an *adult*,
*in vivo*, pipette-independent inference, and it puts adult central `E_Cl` at
**about −55 to −65 mV**.

### 1.3 There is no single adult value to find

* Schellinger *et al.* 2021/2022 (bioRxiv [10.1101/2021.07.16.452737](https://doi.org/10.1101/2021.07.16.452737);
  published as *Chloride oscillation in pacemaker neurons regulates circadian
  rhythms through a chloride-sensing WNK kinase signalling cascade*, PMID
  [35303418](https://pubmed.ncbi.nlm.nih.gov/35303418/)) measure intracellular
  Cl⁻ in **adult** sLNv pacemaker neurons with ClopHensor and find it **rises and
  falls across the day**, Ncc69-dependently. `E_Cl` in an adult *Drosophila*
  neuron is a function of time of day.
* *Chloride-dependent mechanisms of multimodal sensory discrimination and
  nociceptive sensitization in Drosophila*, **eLife** 2022,
  [elifesciences.org/articles/76863](https://elifesciences.org/articles/76863):
  `kcc` is down- and `ncc69` up-regulated in class III neurons, "suggesting these
  neurons may maintain relatively high intracellular Cl⁻". `E_Cl` is also a
  function of cell type.

### 1.4 GluCl has no reversal potential of its own

The *Drosophila* GluClα literature (Cully *et al.* 1996, *J Biol Chem*
271:20187–20191, PMID [8702744](https://pubmed.ncbi.nlm.nih.gov/8702744/); Kane
*et al.* 2000, *PNAS* 97:13949–13954, DOI
[10.1073/pnas.240464697](https://doi.org/10.1073/pnas.240464697); Liu & Wilson
2013, *PNAS* 110:10294–10299) establishes GluClα as a glutamate-gated **chloride**
channel. **No separate GluCl reversal potential is reported**, and none could be:
a Cl⁻-selective channel reverses at `E_Cl`, whatever `E_Cl` happens to be. So
GluClα supplies no independent support for −70 mV.

*Correction to the framing of this task:* the project's own documents do **not**
cite −70 mV for GluClα. `docs/LIF_DYNAMICS_SPEC.md` §6.3 cites Liu & Wilson 2013
only for "glutamate is inhibitory, via a chloride conductance", and §3.1 already
labels −70 mV an engineering assumption with the two *Drosophila* measurements
(−56, −37 mV) named against it. The project has been honest about this from the
start. What it has not done is *measure* the consequence.

### 1.5 The honest answer

> **The adult *Drosophila* central chloride reversal potential is not well
> constrained. Every direct measurement is a readout of the experimenter's
> pipette, it varies with cell type, and within one cell type it varies with time
> of day. There is no single value to adopt.**
>
> **What the literature does constrain is a range, and −70 mV is outside it.**
> The two pipette-independent adult constraints — GABA hyperpolarisation stopping
> at −56 ± 2 mV in intact adult neurons (Wilson & Laurent 2005), and negligible
> resting chloride current with `V_rest` at −55 to −65 mV (Gouwens & Wilson 2009)
> — put adult central `E_Cl` **at or just below rest, roughly −55 to −65 mV**. No
> paper retrieved here supports −70 mV for adult *Drosophila*. −70 mV is a
> textbook mammalian-neuron convention.

### 1.6 A structural consequence that must be stated before the sweep

This engine's `V_rest` is **−52 mV** (`brainlab/engine.py`, from Shiu *et al.*
2024) — which is itself *above* every measured adult *Drosophila* resting
potential in §1.1–§1.2. The v3 calibration is
`g_unit_inh = 1/(V_rest − E_inh)`, which **diverges as `E_inh` approaches
`V_rest`** and is undefined for `E_inh ≥ −52 mV`. So the literature-supported
range sits in the region where the declared calibration is most extreme:

| `E_inh` (mV) | `V_rest − E_inh` | `g_unit_inh` | ratio to `g_unit_exc` |
|---|---|---|---|
| −70 (the v3 assumption) | 18 mV | 1/18 | 2.889 |
| −66 | 14 | 1/14 | 3.714 |
| −63 | 11 | 1/11 | 4.727 |
| −60 | 8 | 1/8 | 6.500 |
| −58 | 6 | 1/6 | 8.667 |
| −56 (Rohrbough & Broadie; Wilson & Laurent endpoint) | 4 | 1/4 | **13.000** |

**This is declared in advance as a confound, not discovered afterwards.** A
failure at −56 mV cannot be cleanly attributed to the reversal potential alone,
because the same move multiplies the inhibitory conductance by 4.5×. The two are
inseparable *by the declared calibration rule*, exactly as spec §6.7 says of
arm S2 ("this arm tests the calibration and the reversal together, as it must").
The sweep is therefore designed to run **intermediate** values, so that the
result is a curve rather than two endpoints, and so that a monotone degradation
can be distinguished from a cliff.

---

## 2. The sweep

**Values of `E_inh` (mV), fixed here:** **−70, −66, −63, −60, −58, −56.**

* −70 is the v3 engineering assumption, carried so that every other value has a
  same-code reference point.
* −56 is arm S2's value: the larval measurement (Rohrbough & Broadie 2002) and,
  coincidentally, the adult hyperpolarisation endpoint (Wilson & Laurent 2005).
* −60 is the centre of the pipette-independent adult range of §1.2 and is
  declared here as **the single most literature-supported value**.
* −63 and −58 bracket −60 inside that range; −66 sits between the supported
  range and the assumption, so the sweep can show whether behaviour changes
  smoothly or at a cliff.
* Nothing at or above −52 mV can be run: the declared calibration is undefined
  there (§1.6). This is a limit of the model, and it is recorded as one.

**Arms, both run at every value:**

* **A-primary** — the v3 primary transmitter policy, `unclear_mode='excitatory'`.
* **A-S1** — declared arm S1, `unclear_mode='zero'`.

That is 6 values × 2 arms × 2 directions = **24 probe-C runs**, ~32 s each.

**Everything else is byte-unchanged:** `docs/wp5_optomotor_prereg.json`
(sha256 `9af619d4…`), the pinned MaleCNS graph, the WP5 encoder and decoder, the
`v3-modulatory-only` policy, seeds **0–5**, the four conditions (intact,
`dna02_silenced`, `sham_no_input`, `shuffled_graph`), the primary outcome
`TI = mean over blocks of s × mean yaw`, the bootstrap and its rng seed, and the
POSITIVE/NULL decision rule.

**Gate, quoted from `docs/LIF_DYNAMICS_SPEC.md` §6.7, unchanged:**

> * **Q1 — quiet baseline.** In both gray windows of probe C: network rate
>   < 1.0 × 10⁵ spikes/s **and** DNa02_L and DNa02_R each < 20 Hz.
> * **R1 — responsiveness.** During rotation: network rate > 0, the sign of the
>   DNa02 L−R rate difference follows the stimulus in **both** directions, and
>   |L−R| ≥ 1 Hz in at least one direction.
> * **R2 — bounded membrane.** min `V` ≥ `E_inh` in every window.
>
> **If Q1 or R1 fails, the 24-run confirmatory set is not run.** The failure is
> the result and is reported as such.

**Declared limitation of the gate, stated before it is applied.** Probe C measures
DNa02 rates over a 1 s rotation window, so `rate_l_hz` and `rate_r_hz` are integer
spike counts. R1's bidirectional sign test is therefore decided by differences of
1–3 spikes. It is a coarse instrument; it is the *declared* instrument, and it is
not being changed here. Where a gate outcome turns on a single spike, that is
recorded explicitly in §5 and weakens any reading built on it.

### 2.1 Which values get the expensive 24-run set

Declared in advance, and capped at **3** confirmatory sets (~1.5–2 h each):

1. **A-S1 at `E_inh = −70 mV`** — unconditional. Arm S1 passed the probes with a
   larger, more symmetric asymmetry than the primary and has never been run at
   confirmatory level. This is the direct test of falsifier **F5**.
2. **A-primary at the gate-passing value closest to −60 mV other than −70 mV** —
   because −60 mV is declared in §2 as the most literature-supported value. Ties
   break toward the more negative value.
3. If and only if no A-primary value other than −70 mV passes the gate: **A-S1 at
   the gate-passing value closest to −60 mV other than −70 mV**, so that at least
   one literature-range value is measured at confirmatory level in *some* arm.

A value whose gate fails is **not** run confirmatorily, per the quoted rule. If
neither (2) nor (3) yields a candidate, no second set is run and that is the
result.

---

## 3. Predictions, computed before any simulation

Static arithmetic over the pinned graph, A-primary weights
(Σ excitatory 2.0773 × 10⁷, Σ inhibitory 1.3050 × 10⁷, as
`brainlab.transmitter_policy._balance` already reports), high-conductance fixed
point `V* = E_inh · r/(1+r)`, threshold −45 mV:

| `E_inh` | `r = ĝ_i/ĝ_e` | `V*` | margin below threshold |
|---|---|---|---|
| −70 | 1.8148 | **−45.132** | +0.132 mV |
| −66 | 2.3333 | −46.200 | +1.200 |
| −63 | 2.9697 | −47.130 | +2.130 |
| −60 | 4.0833 | −48.197 | +3.197 |
| −58 | 5.4444 | −49.000 | +4.000 |
| −56 | 8.1666 | −49.891 | +4.891 |

**Prediction P1.** Every swept value is subthreshold in the mean, and
*increasingly* so as `E_inh` rises toward rest. −70 mV is the **least** quiet of
the swept values by this arithmetic. So Q1 is expected to pass everywhere, and
the risk is at the other end: the network becomes progressively harder to drive.

**Prediction P2.** R1, not Q1, is where failures are expected, and they are
expected to appear as *loss of one direction* rather than loss of both — the
already-observed +8/−3 Hz primary asymmetry has a factor ~2.7 left/right
imbalance at −70 mV, and stronger inhibition should extinguish the weaker
(rightward) side first. Arm S2's +11/+2 Hz at −56 mV is consistent with exactly
that. **If R1 fails across the literature range for this reason, it is a loss of
sensitivity, not evidence of a leftward-biased fly**, and it must be reported as
the model running out of dynamic range.

**Prediction P3 — the verdict is genuinely open, and this document does not
predict that it is robust.** A monotone collapse of `TI` toward zero as `E_inh`
approaches the literature range is at least as likely as robustness, on the
arithmetic above. Whichever happens is reported with equal prominence.

**Prediction P4.** Arm A-S1 at −70 mV is expected to give a *larger* `TI` than
the primary, since its probe-level asymmetry is 2.3× larger and symmetric
(+18/−18 vs +8/−3). A larger `TI` in the arm would **not** be a better result; it
would show the primary verdict is sensitive to the treatment of 2,999 unlabelled
neurons, which is what F5 asks.

---

## 4. THE DECISION RULE — committed before any number in §5 or §6 is known

Let **G(x)** be the Q1/R1/R2 gate outcome at configuration *x*, and **V(x)** the
preregistered verdict (POSITIVE requires all four preregistered clauses,
including intact−silenced CI excluding zero). Let the **literature-supported
range** be `E_inh ∈ [−65, −55] mV`, fixed by §1.2, which within this sweep means
**−63, −60, −58, −56**. Exactly one of the following readings will be applied,
tested in this order:

**R-ARTEFACT.** *If* V(−70 mV, A-primary) = POSITIVE **and** there exists a
gate-passing configuration elsewhere in the sweep whose verdict is NULL or
NEGATIVE, *then*: **the positive optomotor result is an artefact of the assumed
value of `E_inh`.** The sign or the causal clause flips while the model is still
inside its own declared measurable regime, so the result is produced by the
choice of constant rather than by the connectome. The v3 POSITIVE is then
retained only as a measurement of the `brainlab-lif-v3` engine at `E_inh = −70
mV` and is withdrawn as a claim about the fly, as prominently as it was made.
WP6 does not build on it.

**R-ARM-CONTINGENT (F5).** *If* V(−70 mV, A-S1) differs in verdict class from
V(−70 mV, A-primary), *then* falsifier F5 has fired and spec §6.6's own words
apply: *"then the verdict rests on 2,999 unlabelled neurons and no claim survives
either way."* This is reported even when R-ARTEFACT already applies, since it is
an independent contingency.

**R-CONTINGENT.** *If* V(−70 mV, A-primary) = POSITIVE **and** no configuration
in the literature-supported range is measured POSITIVE — whether because its gate
failed or because its verdict was not POSITIVE — *then*: **the positive result is
contingent on an unsupported assumption.** It holds at a value of `E_inh` that no
*Drosophila* measurement supports, and fails or cannot be evaluated at every
value the measurements do support. This is the same class of finding as v1's, one
level up: not a coding error, but a result that exists only at a chosen constant.
It must be reported in `WP5_OPTOMOTOR.md` §13 and in §12's own reading, and the
POSITIVE verdict must carry this qualifier wherever it is quoted. WP6 does not
build on it.

**R-ROBUST.** *If* V = POSITIVE at every gate-passing configuration in the sweep,
**and** at least one of those is a value of `E_inh` other than −70 mV, **and** at
least one is inside the literature-supported range, **and** every one of those has
intact−silenced CI excluding zero, *then*: **the positive result is robust to
`E_inh` across the physiologically admissible range.** It is then a result about
the graph and the encoder, not about the constant, and `E_inh = −70 mV` is
demoted from load-bearing assumption to arbitrary-within-range choice.

**R-PARTIAL.** *If* V = POSITIVE at −70 mV and at one or more gate-passing values
other than −70 mV, but **no** gate-passing POSITIVE configuration lies inside the
literature-supported range, *then*: **the result is robust to the value of `E_inh`
only outside the range the literature supports.** This is strictly weaker than
R-ROBUST and is reported as such: it establishes that the result is not a
knife-edge coincidence at one number, and it does **not** establish that it
survives a physiologically supported reversal potential.

**R-REGIME-LIMITED.** *If* the gate fails at every configuration except
`E_inh = −70 mV` (in both arms), *then*: **the measurable regime itself exists
only at the assumed constant.** No sensitivity statement can be made, because
there is nothing admissible to compare against. The correct reading is that the
v3 POSITIVE deserves no confidence beyond "reproducible under one declared engine
configuration", and that the blocking problem is the model's dynamic range — most
likely the `V_rest = −52 mV` / `E_Cl ≈ −56 mV` collision of §1.6 — not the
connectome.

**Confidence statement, also fixed in advance.** Under R-ROBUST the optomotor
result deserves the confidence WP5 §12 currently claims for it, with §12.4's
existing engine-contingency qualifier and no new one. Under R-PARTIAL it deserves
that confidence explicitly conditioned on `E_inh < −65 mV`, a condition the
literature does not support. Under R-CONTINGENT, R-REGIME-LIMITED or
R-ARM-CONTINGENT it deserves the status of an **engine-configuration-dependent
result**: reproducible, correctly preregistered, causally clean *within its
configuration*, and not evidence that the MaleCNS connectome computes an
optomotor turn. Under R-ARTEFACT it deserves none, and §12's POSITIVE is
withdrawn as a biological claim.

**Forbidden, restated.** No value of `E_inh`, no calibration, no transmitter
policy, no metric, no seed set, no preregistration file and no gate threshold may
be changed to obtain a better reading. `g_unit_inh` continues to be derived as
`1/(V_rest − E_inh)`. If the reading is R-CONTINGENT or R-ARTEFACT, **that is the
finding**, and it is reported as prominently as the POSITIVE was.

---

## 5. Gate results

*Filled in after §1–§4 were hashed. See `docs/receipts/einh_sensitivity.json`.*

## 6. Confirmatory verdicts and the applied rule

*Filled in after §5.*
