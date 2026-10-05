# Literature benchmarks and prior art

Compiled 27 September 2026 from four parallel research passes, then **verification of the
load-bearing claims by direct retrieval**. Every item below carries its status:

- **VERIFIED** — the orchestrator retrieved the source and confirmed the claim, quoted here.
- **LEAD** — reported by a research pass but **not** independently confirmed. Do not cite
  from this file; retrieve the paper first.
- **CORRECTED** — a research pass asserted something that direct retrieval disproved.

This file exists because a first pass produced plausible but wrong claims, including one
that would have driven an engine change. Treat unverified research output as a search
index, never as evidence.

---

## 1. The model our engine descends from

**Shiu, Sterne, Spiller et al. (2024), "A Drosophila computational brain model reveals
sensorimotor processing", Nature 634:210–219.** Open access: PMC11446845.

**VERIFIED — parameters.** Our engine matches their published source code
(`philshiu/Drosophila_brain_model/model.py`) exactly:
`v_0` −52 mV, `v_rst` −52 mV, `v_th` −45 mV, `t_mbr` 20 ms (commented as
`0.002 uF × 10 MOhm`), `tau` 5 ms, `t_rfc` 2.2 ms, `t_dly` 1.8 ms, `w_syn` 0.275 mV.

> **CORRECTED:** a research pass reported the membrane constant as 11 ms. It is 20 ms.
> Our spec's value was right.

**VERIFIED — non-spiking neurons.** Quoted from the paper: *"The model does not account
for gap junctions, non-spiking neurons, internal state or long-range neuropeptides"*, and
it treats *"each neuron identically as a spiking neuron and ignore[s] neural morphology as
well as different neurotransmitter receptor dynamics."* Also: *"circuits with extensive
basal inhibition, not captured by the model because of the zero basal firing rate, may be
poorly simulated in our model."*

> **CORRECTED:** a research pass claimed Shiu et al. approximate graded neurons with a
> threshold-linear rule and that we had failed to inherit it. **They do not.** Our engine
> reproduces a limitation the published model states about itself; it did not drop a
> mechanism the source had. This distinction matters for how our own null results read.

**VERIFIED — transmitter signs.** *"GABAergic and glutamatergic neurons are inhibitory,
and … each neuron is either exclusively inhibitory or excitatory"*, a neuron counting as
inhibitory when more than half its presynaptic sites are predicted inhibitory.
**Dopaminergic, octopaminergic and serotonergic neurons are assigned to the excitatory
category.** So the aminergic-as-fast-excitation problem flagged in
`docs/WP6_PLASTICITY_SPEC.md` is inherited from the published convention, not invented here.

---

## 2. Does connectivity alone determine function?

**LEAD — Lappalainen, Turaga et al. (2024), Nature 634:1132–1140.** Reported: a
connectome-constrained network of ~64 optic-lobe cell types, with connectivity fixed
(~604 synaptic signs, ~2,355 synapse counts) but **734 free parameters still fitted by
gradient descent** — per-cell-type time constants, resting potentials and synaptic scaling.
Reported conclusion: connectivity is a powerful constraint but **insufficient on its own**
to determine function.

Why it matters here: NeuroFly runs **fixed** weights with generic parameters and no
task-driven fitting. That is a legitimate and more conservative position, but it means our
model is an empirical test of a hypothesis these authors report as false in its strong
form. Any null we obtain should be read against that.

---

## 3. Transmitter prediction accuracy

**LEAD — Eckstein, Bates et al. (2024), Cell 187:2574–2594.** Reported accuracies for
neurotransmitter prediction from EM: ~91–97 % for acetylcholine, glutamate and GABA;
~85–90 % dopamine; **~33–38 % serotonin** (their least reliable class). Also reported:
one fast transmitter assumed per neuron (co-transmission violates this), and **glutamate's
sign cannot be inferred structurally** — excitatory via iGluR or inhibitory via GluClα
depending on the receptor.

Why it matters here: our one-label-per-neuron proxy is defensible for the core
sensorimotor pathway and unreliable for exactly the modulatory circuits WP6 depends on.

---

## 4. Motion vision — what direction selectivity reportedly requires

All **LEAD**; several citations in the source pass looked misattributed and must be
retrieved before use.

- Both Hassenstein–Reichardt (preferred-direction enhancement) and Barlow–Levick
  (null-direction suppression) motifs are reported in T4/T5.
- Minimal motif: spatial offset + **temporal delay of about 20–30 ms** + a nonlinearity.
  The delay reportedly arises from membrane and synaptic filtering, **not** conduction
  delay (~1 ms).
- Pathway: R1–R6 → L1 (ON) / L2, L3 (OFF) → medulla (Mi1, Tm3 → T4; Tm1, Tm2, Tm4, Tm9
  → T5) → T4/T5 → LPTCs (HS, VS, H2).
- **The upstream pathway is reported to be non-spiking and graded** — R1–R6, L1, L2, the
  medulla columnar cells, and the HS tangential cells. H2 is reported as spiking.
- Reported T4/T5 direction-selectivity index: **about 0.7–0.9**, with near-complete
  null-direction suppression. That is the quantitative bar for "the graph computed it".

---

## 5. Behavioural and physiological bands

All **LEAD**. Useful as plausibility ranges, not as validation targets, until retrieved.

| quantity | reported range | our model |
|---|---|---|
| DNa02 firing during steering | **93–128 spikes/s** (burst) | **~6.5 Hz** under v3 |
| HS cell response | graded, **~5 mV**, non-spiking | spiked at **142.9 Hz** under v1 |
| optomotor turning gain | ~0.9–1.1 (near unity) | not measured in these units |
| spatial wavelength optimum | 20–24° | 30° used in the WP5 protocol |
| temporal frequency optimum | 1–4 Hz walking, 5–10 Hz flight | 1.5 Hz used |
| contrast threshold | ~5–15 % Michelson | null at 50 % contrast (§13) |

**Read the last column honestly.** Our DNa02 rate is 15–20× below the biological burst
range, and our v1 HS "firing rate" is not merely wrong in magnitude but wrong in kind,
since those cells do not spike. Model rates and biological rates are not comparable
quantities here, and should not be presented side by side without that caveat.

---

## 6. Plasticity rules for WP6

All **LEAD**; retrieve before implementing.

**ER → EPG (site A).** Reported anti-Hebbian depression with co-activity, potentiation
with postsynaptic activity alone (Fisher, Lu, D'Alessandro & Wilson 2019, Nature
576:121–125); dopamine as a "when to learn" signal (Fisher et al. 2022, Nature
612:316–322); and a **preprint** reporting presynaptic depression instructed by
octopaminergic EL neurons, with postsynaptic activity dispensable. Remapping is reported
to stabilise within about 5 minutes. **No published equation or rate constants.**

**KC → MBON (site B).** Reported ~80 % depression after odour–dopamine pairing (Hige et
al. 2015, Neuron 88:985–998); bidirectional sign set by pairing order within a window
under ~1 s, via distinct DopR1/DopR2 pathways (Handler et al. 2019, Cell 178:60–75).
**No published equation**, and the DopR1/DopR2 balance is unquantified.

**Verdict for both:** the qualitative rule, the locus and the instructive signal are
published; **the learning rate and time constants are not**, and must be declared as
engineering assumptions rather than sourced values.

---

## 7. What this changes

1. A spiking-only engine is being asked to run a pathway that is graded in the animal, and
   the published model this engine descends from **declares that same limitation**. A null
   in the photoreceptor-level optomotor test is therefore an expected structural outcome,
   not a defect unique to this project — and it is worth reporting as such.
2. Any future graded-transmission engine is a **new declared dynamics version** with its
   own pre-registration and pin. It must not be introduced mid-measurement.
3. We now have external plausibility bands. Our DNa02 output is far below them. That
   belongs in any statement about the optomotor result.
4. WP6's instructive signals sit in the transmitter classes with the weakest prediction
   accuracy, and its rate constants are unpublished. Both are assumptions to declare.

---

## 8. Addendum — how this bears on the validation receipts in this repository

Added when these findings were ported onto `master`, 27 September 2026.

**The HS ceiling judgement now has sources.** `docs/receipts/validation/optomotor-yaw-v3-1.md`
records a FAIL caused solely by HS_L at 60 Hz and HS_R at 53 Hz against a 0–50 Hz LPTC
bound, and `optomotor-yaw-v3-2.md` makes those checks report-only, giving the reason that
"HS cells signal with graded potentials and no published HS spike rate exists", while
honestly noting the change was prompted by the failure.

The research above supports that reasoning and supplies the citations it was missing
(all still **LEAD** status — retrieve before citing):

- HS cells are reported to be **graded, non-spiking** neurons with responses of roughly
  **5 mV** to wide-field motion, resting near −55 mV (Joesch et al., J. Neurophysiol.
  104:2826–2841, 2010). H2, by contrast, is reported as spiking.
- Shiu et al. (2024), whose parameters this engine uses, state plainly that their model
  "does not account for … non-spiking neurons" and treats "each neuron identically as a
  spiking neuron" (**VERIFIED**, PMC11446845).

So an HS *spike rate* is not a quantity that can be compared against biology at all: the
cell does not spike in the animal, and the engine has no graded mode in which it could
fail to. Making the check report-only is defensible, but the honest statement is stronger
than "no published rate exists" — it is that **the model represents these cells in a
regime they do not occupy**, and that this is an inherited limitation of the published
approach rather than a threshold that needs a better number.

**The DNa02 rates are far below the animal.** The v3-2 receipt reports DNa02_L at 4.4 Hz
and DNa02_R at 1.8 Hz when driven, against a 0–100 Hz bound. Real DNa02 is reported to
burst at **93–128 spikes/s** during steering (Rayshubskiy et al., eLife 13:RP102230 —
**LEAD**). The bound is therefore satisfied from roughly 20× below the biological range,
which is worth stating wherever the pass is quoted: the gate is not evidence that the
rate is right, only that it is not absurd.

**What would raise confidence**, in the spirit of the existing bounds files: replace
`verified: false` bounds with measured ones where the literature supplies them, and mark
explicitly which quantities (HS response amplitude in mV, graded LPTC signalling) cannot
be gated at all until the engine has a graded mode.
