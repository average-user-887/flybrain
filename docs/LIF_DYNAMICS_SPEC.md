# Declared LIF dynamics for `brainlab` — v1 current-based, v2 conductance-based, v3 recalibrated

Written **before** the v2 implementation and before any v2 measurement, as
required by `docs/CLAUDE_IMMEDIATE_PLAN.md` ("declare the parameters and their
sources first"). Nothing in this document was chosen after looking at a v2
optomotor result. If the corrected dynamics weaken or abolish the WP5
optomotor effect, that is the result.

This specification exists because WP5 (`docs/WP5_OPTOMOTOR.md` §6, qualifiers 1
and 3) recorded two defects that are properties of `brainlab/engine.py`, not of
the connectome:

1. **Runaway.** After the first stimulus the 166,700-neuron graph enters a
   self-sustained state of ≈10⁶ spikes/s that persists through the gray
   periods. The "gray" baseline is not rest.
2. **Unbounded membrane potential and a dead right side.** Membrane potentials
   reach −200 mV. DNa02_R sits at −120 to −200 mV and never fires, although its
   input budget is within 2 % of DNa02_L's (measured: DNa02_L 1138 in-edges,
   Σ excitatory weight 4427.5, Σ inhibitory 2160.7; DNa02_R 1201 in-edges,
   4389.8 / 2256.4 — `docs/receipts/lif_dynamics_diagnosis.json`).

Both follow from one modelling gap: **synapses in v1 are voltage-equivalent
currents with no reversal potential.** Inhibition therefore has unlimited
hyperpolarising authority and no shunting effect, and excitation has no
saturation.

v1 is **not deleted or reinterpreted.** It remains selectable and remains the
dynamics under which every existing WP1–WP5 result was produced.

**§6 adds v3** and was written under the same rule: before the v3 engine was run
on anything larger than a two-neuron test graph, and before the static
excitation/inhibition arithmetic of §6.5 was computed. §1–§5 are the v1/v2
document and are unchanged.

---

## 1. Common structure (both versions)

A leaky integrate-and-fire point neuron per connectome node, single-exponential
synaptic kinetics, fixed axonal delay, absolute refractory period, all-edge
propagation over the prepared CSR graph. Edge weight is
`synapse_count × transmitter_sign × 0.275`, where the sign is the declared
coarse fast-transmission proxy of `brainlab/transmitters.py` (ACh +; GABA,
glutamate, histamine −). No receptor kinetics, no neuromodulation, no gap
junctions, no dendrites, no adaptation, no ion-channel model.

### 1.1 Parameters shared by v1 and v2

| symbol | value | units | source |
|---|---|---|---|
| `dt` | 0.1 | ms | integration timestep; engineering choice (`brainlab/engine.py`), 1/50 of the fastest time constant |
| `tau_m` | 20 | ms | Shiu et al. 2024 LIF model (`t_mbr`), from `C·R = 2 nF · 10 MΩ` |
| `tau_syn` | 5 | ms | Shiu et al. 2024 (`tau`) |
| `V_rest` = `E_leak` | −52 | mV | Shiu et al. 2024 (`v_0`) |
| `V_reset` | −52 | mV | Shiu et al. 2024 (`v_rst`) |
| `V_threshold` | −45 | mV | Shiu et al. 2024 (`v_th`) |
| `t_refractory` | 2.2 | ms | Shiu et al. 2024 (`t_rfc`) |
| `t_delay` | 1.8 | ms | Shiu et al. 2024 (`t_dly`) |
| `w_syn` | 0.275 | mV of synaptic weight per anatomical synapse | Shiu et al. 2024 (`w_syn`) |

`w_syn` is the *weight* parameter, not the realised PSP. In this engine's
kernel one unit of weight produces a peak PSP of
`(tau_syn/(tau_m − tau_syn))·(e^{−t_p/tau_m} − e^{−t_p/tau_syn}) = 0.1575` mV at
rest, so a single anatomical synapse gives a **0.0433 mV** peak EPSP. Measured:
v1 0.04332 mV, v2 0.04372 mV (probe A, `docs/receipts/lif_dynamics_diagnosis.json`).

Primary source for the whole shared set: Shiu, P.K., Sterne, G.R., Spiller, N.,
*et al.* (2024), *A Drosophila computational brain model reveals sensorimotor
processing*, **Nature** 634:210–219 (preprint bioRxiv 2023.05.02.539144);
parameter values read from the released implementation,
`github.com/philshiu/Drosophila_brain_model` (`model.py`: `v_0 = -52 mV`,
`v_rst = -52 mV`, `v_th = -45 mV`, `t_rfc = 2.2 ms`, `t_mbr = 20 ms`,
`tau = 5 ms`, `t_dly = 1.8 ms`, `w_syn = 0.275 mV`).

These are **model** parameters from a published *Drosophila* whole-brain model,
not independent electrophysiological measurements. Shiu et al.'s model is
itself current-based, so it supplies no reversal potentials.

### 1.2 Inherited engineering properties, unchanged in v2

Declared so that v2 is not mistaken for a biophysical model:

* Synaptic state is **zeroed on every spike** (`g ← 0` at reset). This follows
  the Brian2 `(unless refractory)` schedule the upstream probe used. It is an
  engineering property; real synaptic conductance does not vanish when the
  postsynaptic cell fires.
* Synaptic input arriving at a **refractory** neuron is **discarded**, not
  queued (`brainlab/engine.py`, delivery loop).
* No spike-frequency adaptation, no short-term plasticity, no
  after-hyperpolarisation conductance.
* External input (`Brain.step(currents, …)`) is a per-neuron **current** in
  mV-equivalent units, not a conductance. It is the only channel the WP5
  encoder and the Kir2.1-like silencing clamp use.
* The transmitter-sign proxy is coarse and never receptor physiology.

---

## 2. v1 — current-based (existing; unchanged)

    tau_m dV/dt = (E_leak − V) + I_ext + g(t)
    tau_syn dg/dt = −g,   g ← g + w  on each arriving spike

`g` and `I_ext` are in mV. Integration is the exact two-exponential kernel
(`coupling = (e^{−dt/tau_m} − e^{−dt/tau_syn})/3`, where
`1/3 = tau_syn/(tau_m − tau_syn)`), so v1 is analytically exact between spikes.

**Consequence, and the defect:** `V` has no lower or upper bound. A sustained
net inhibitory input of `w̄` drives `V → E_leak + w̄` without limit; the
observed −200 mV is exactly this. Inhibition is purely hyperpolarising and
never shunting, so it cannot regulate gain, only subtract.

Identifier: `dynamics_version = "v1"`, controller version `brainlab-lif-v1`.

---

## 3. v2 — conductance-based with reversal potentials (new)

    tau_m dV/dt = (E_leak − V) + ĝ_e(t)(E_exc − V) + ĝ_i(t)(E_inh − V) + I_ext
    tau_syn dĝ_e/dt = −ĝ_e,   ĝ_e ← ĝ_e + |w|·G_unit  on an excitatory (w > 0) spike
    tau_syn dĝ_i/dt = −ĝ_i,   ĝ_i ← ĝ_i + |w|·G_unit  on an inhibitory (w < 0) spike

`ĝ_e`, `ĝ_i` are conductances **in units of the leak conductance**
(dimensionless). Everything else is as in §1.

### 3.1 New parameters

| symbol | value | units | source / status |
|---|---|---|---|
| `E_exc` | 0 | mV | **Measured, Drosophila.** Fast excitation in the adult brain is nicotinic-cholinergic; nAChRs are non-selective cation channels. Lee, D. & O'Dowd, D.K. (1999), *J. Neurosci.* 19:5311–5321, report the peak *I–V* of cholinergic mEPSCs in *Drosophila* central neurons reversing **near 0 mV**. Su, H. & O'Dowd, D.K. (2003), *J. Neurosci.* 23:9246–9253, report +8.9 ± 1.7 mV in Kenyon cells. 0 mV is the rounded canonical value. |
| `E_inh` | −70 | mV | **Engineering assumption, literature-bounded.** GABA (Rdl), glutamate (GluClα) and histamine (ort/HisCl) fast inhibition in *Drosophila* are all chloride conductances, so `E_inh = E_Cl`. Reported *Drosophila* values depend on the recording internal solution and are therefore not directly transferable: −56 ± 3 mV in larval central neurons (Rohrbough, J. & Broadie, K. 2002, *J. Neurophysiol.* 88:847–860) and −37 ± 3 mV in Kenyon cells against a theoretical −45 mV for that internal (Su & O'Dowd 2003). −70 mV is the value used here, chosen as the conventional physiological `E_Cl` for a low intracellular chloride neuron, **below rest so that inhibition remains hyperpolarising as well as shunting**. It is *not* a measured adult *Drosophila* number and is marked as an assumption everywhere it appears. |
| `G_unit` | 1/(`E_exc` − `V_rest`) = 1/52 ≈ 0.019231 | leak-conductance units per mV-equivalent weight unit | **Derived, not fitted.** Fixed by requiring that one unitary cholinergic synapse at rest produce the same peak EPSP as v1 (0.0433 mV, §1.1). Verified: v1 0.04332 mV, v2 0.04372 mV. No free parameter is introduced. |
| `V` bounds | [`E_inh`, `E_exc`] = [−70, 0] | mV | Hard clamp applied after each update. Needed only because `I_ext` is a current channel (encoder drive, Kir-like clamp) that could otherwise push `V` outside the conductance-bounded range. |

**Declared sensitivity arm (predeclared, not a tuning pass).** Because `E_inh`
is the one assumed value, the diagnostic probe (§5, probe C only — *not* the
confirmatory optomotor run) is additionally reported at `E_inh = −56 mV`, the
measured *Drosophila* larval value. The confirmatory optomotor re-run uses
−70 mV only.

### 3.2 Integration

Exponential Euler with the conductances held at their start-of-step values
within each `dt` (`brainlab/engine.py: advance_v2`):

    g_tot = 1 + ĝ_e + ĝ_i
    V_inf = (E_leak + ĝ_e·E_exc + ĝ_i·E_inh + I_ext) / g_tot
    V     ← V_inf + (V − V_inf)·exp(−dt·g_tot / tau_m)
    V     ← min(max(V, E_inh), E_exc)
    ĝ_e   ← ĝ_e·exp(−dt/tau_syn);  ĝ_i ← ĝ_i·exp(−dt/tau_syn)

This is unconditionally stable and, with `ĝ_e = ĝ_i = 0`, reduces exactly to
v1's leak term. It is *not* v1's exact two-exponential kernel: with
time-varying conductances no closed form exists, so v2 is first-order accurate
in `dt` in the synaptic term. `dt = 0.1 ms` is 1/50 of `tau_syn`.

The spike, delay-queue, refractory and reset schedule is **identical** to v1
(threshold check each `dt`, delivery 18 ticks later, reset of the spiking
neuron after delivery), so any difference between versions is attributable to
the synaptic model alone.

### 3.3 What changes, relative to v1

1. **Bounded membrane potential.** `V ∈ [−70, 0] mV` always. −200 mV becomes
   impossible by construction.
2. **Shunting inhibition.** The effective membrane time constant becomes
   `tau_m / g_tot` and the effective input resistance `1/g_tot`. Inhibition now
   divides as well as subtracts — the physiological gain-control mechanism v1
   lacks entirely.
3. **Saturating excitation.** Excitatory driving force `(E_exc − V)` shrinks as
   `V` depolarises.
4. **Weaker hyperpolarisation per unit inhibitory weight.** At rest the
   unitary IPSP is `(V_rest − E_inh)/(E_exc − V_rest) = 18/52 = 0.346×` the v1
   IPSP for the same weight, while the shunt it contributes is new. This is a
   direct consequence of the calibration in §3.1 (one conductance quantum for
   both signs) and is *not* a tuning choice. **It may make the network more
   active, not less.** Predeclared: the runaway measure in §5 is reported
   whichever way it comes out.
5. **External drive is shunted.** `I_ext` contributes `I_ext / g_tot` to
   `V_inf`. In a high-conductance state the encoder's 20-unit T4/T5 drive is
   attenuated; at rest (`g_tot ≈ 1`) it is identical to v1.
6. **The Kir2.1-like silencing clamp still silences.** `SILENCE_DRIVE = −200`
   (`brainlab/io_map.py`) now pins `V` at the floor `E_inh = −70 mV`, which is
   below threshold, so the WP5 silencing control is unchanged in effect. Its
   *mechanism* is now "clamped to the chloride reversal", which is what a
   Kir2.1 experiment approximates, rather than "−200 mV".

### 3.4 Identity, state and checkpoint compatibility

* `dynamics_version = "v2"`; controller version `brainlab-lif-v2`
  (`brainlab-lif-plastic-v2`, `brainlab-lif-readout-v2` for the other graph
  backends). `provenance.RunManifest.create` derives this from the dynamics
  dict, so every manifest, telemetry identity packet and export carries it.
* The declared dict is `brainlab.graph_identity.LIF_DYNAMICS_V2`; its pin is
  `graph_identity.dynamics_pin('v2')`.
* The active version is selected per `Brain` (`Brain(..., dynamics='v2')`) or
  process-wide by `NEUROFLY_LIF_DYNAMICS=v2`. The default is **v1**, so every
  existing script, test and checkpoint keeps its current meaning.
* **Old checkpoints are refused, never reinterpreted.** The synaptic state
  array `g` has shape `(n,)` under v1 and `(2, n)` under v2 (row 0 excitatory,
  row 1 inhibitory). `Brain.restore_state` rejects a shape mismatch with an
  error naming both dynamics versions. A v1 checkpoint cannot be silently
  loaded into a v2 brain or the reverse.

---

## 4. What v2 does *not* claim

* It is **not** a biophysical model of a *Drosophila* neuron. It has one
  compartment, two synaptic conductances, no active channels, no adaptation and
  a coarse transmitter-sign proxy.
* `E_inh = −70 mV` is an assumption (§3.1).
* Reversal potentials do not make the direction selectivity of §3 of
  `docs/WP5_OPTOMOTOR.md` graph-computed; that remains an encoder assumption.
* A conductance-based model is not by itself a cure for runaway. Whether the
  self-sustained state survives is an empirical question answered in §5, and
  the honest answer is reported whatever it is.

---

## 5. Predeclared measurements (v1 and v2)

Receipts: `docs/receipts/lif_dynamics_diagnosis.json` (probes) and
`docs/receipts/lif_dynamics_v2.json` (re-run).

**Probe A — voltage bound.** Two neurons, 0 driven at 20 units, edge 0→1 with
weight −40, 2 s. Report `min V₁`. Pass for v2: `min V₁ ≥ E_inh`.

**Probe B — self-sustained state.** 2,000-neuron sparse random recurrent net
(40 out-edges each, 80 % excitatory by edge count, seed 11), 200 ms input to
10 % of neurons, then 800 ms with no input. Report the free-window rate. No
pass/fail threshold is declared: this is a comparison, reported both ways.

**Probe C — the real graph.** 500 ms gray, 1 s rotation, 500 ms gray, one
direction per run, on the pinned MaleCNS graph with the WP5 encoder. Report
network rate and DNa02 L/R rate and `V` in each window, per direction, per
dynamics version, plus the `E_inh = −56 mV` sensitivity arm.

**Re-run — WP5 preregistered optomotor protocol**, unchanged script, unchanged
preregistration (`docs/wp5_optomotor_prereg.json`), unchanged primary outcome
`TI = mean over blocks of s × mean yaw`, same seeds 0–5, same four conditions.
Only `--dynamics v2` differs. Reported side by side with the v1 numbers,
including the left/right breakdown that §6.1 of `docs/WP5_OPTOMOTOR.md` flagged.

---

---

## 6. v3 — recalibrated conductance-based dynamics with a transmitter-class policy

**Written before any v3 network measurement.** The calibration rule, the
transmitter policy, the predicted consequences and the falsification criteria
below were fixed and hashed before the v3 engine was run on anything larger than
a two-neuron test graph, and before the static excitation/inhibition arithmetic
of §6.5 was computed. The locked declaration is
`sha256 e7bdc18d8e0811af57ee999f2c04bb3a5217fc5e490c06ee8c5ba6aa537f4d2a`,
written 2026-09-20T16:44:31+02:00, and is reproduced verbatim in
`docs/receipts/lif_dynamics_v3.json` under `declaration_lock`.

v1 and v2 are **not** deleted, superseded or reinterpreted. Both stay
selectable, both keep their pins, and every number already published under them
stands as a result of that controller version.

### 6.1 Why a recalibration is needed at all

§11.1 of `docs/WP5_OPTOMOTOR.md` established the defect analytically. In the
high-conductance limit the membrane of a v2 neuron sits at

    V* = E_inh · r / (1 + r),   r = ĝ_i / ĝ_e

and the MaleCNS graph under the coarse sign proxy has a total inhibitory weight
0.619× its total excitatory weight. With v2's single conductance quantum that is
`r = 0.619`, hence `V* = −26.75 mV` against a `−45 mV` threshold: **the fixed
point of the whole network is suprathreshold**, so any sustained input saturates
it. This is not a property of the connectome. It is a property of importing a
voltage-calibrated synaptic scale into a conductance model without recalibrating
it.

### 6.2 The declared calibration: per-sign PSP preservation

The upstream model (Shiu et al. 2024) is current-based. Its only statement about
synaptic strength is a **voltage**: `w_syn = 0.275 mV` of weight per anatomical
synapse, applied with a `+` sign for excitation and a `−` sign for inhibition.
In that model the statement is symmetric — a unitary inhibitory event
hyperpolarises by exactly as much as a unitary excitatory event depolarises.

When that statement is carried into a conductance model, the conductance quantum
is not given; it has to be derived, and the derivation needs an invariant. There
are exactly two candidates:

| invariant preserved | consequence |
|---|---|
| **the PSP the source model specifies, per sign** | `g_exc = w/(E_exc − V_rest)`, `g_inh = w/(V_rest − E_inh)`; the two quanta differ by the ratio of the two driving forces |
| the conductance, one quantum for both signs | the excitatory PSP is preserved and the inhibitory PSP silently becomes `(V_rest − E_inh)/(E_exc − V_rest) = 18/52 = 0.346×` what the source specifies |

v2 chose the second. That choice is an **additional assumption the upstream model
never makes**: nothing in Shiu et al. asserts that an inhibitory synapse opens
the same conductance as an excitatory one, and the arithmetic consequence — a
threefold weakening of every inhibitory synapse in the graph — was not visible
in the declaration, only in the behaviour.

**v3 preserves the PSP, per sign.** This is the calibration that adds nothing to
the source:

    g_unit_exc = 1 / (E_exc  − V_rest) = 1/52 ≈ 0.019231    (unchanged from v2)
    g_unit_inh = 1 / (V_rest − E_inh)  = 1/18 ≈ 0.055556    (new)
    ratio g_unit_inh / g_unit_exc = 52/18 = 2.888…

Both quanta are in units of the leak conductance. Everything else — `E_exc = 0`,
`E_inh = −70`, the membrane bounds, the exponential-Euler integration, the
spike/delay/refractory/reset schedule, `dt`, `tau_m`, `tau_syn`, `V_rest`,
`V_reset`, `V_threshold` — is identical to v2, and the spike schedule is
identical to v1, so any difference is attributable to this one change plus the
policy of §6.3.

**This is a derivation, not a fit. It has no free parameter.** It is fixed
entirely by `E_exc`, `E_inh` and `V_rest`, all of which were declared in §3.1
before v2 ran. If `E_inh` is moved by a sensitivity arm, `g_unit_inh` moves with
it — `Brain` derives it as `1/(V_rest − E_inh)` rather than storing it — because
the calibration is a statement about driving forces, not a number.

**Physiological standing.** The proposition being asserted is that in a real
neuron a unitary GABA_A / GluCl / HisCl event and a unitary nicotinic event
produce comparable-magnitude PSPs at rest. That is what the source model
asserts, and it is the weaker of the two claims available: the alternative
asserts comparable *conductances*, which — because the chloride driving force at
rest is roughly a third of the cation driving force — implies inhibitory PSPs a
third the size of excitatory ones throughout the brain. No measurement in
*Drosophila* known to us supports that asymmetry, and the *Drosophila* central
recordings that exist (Rohrbough & Broadie 2002; Su & O'Dowd 2003) report
cholinergic and GABAergic currents of comparable amplitude in the same cells.
**It is not, however, an independently measured unitary conductance ratio, and
this specification does not claim it is. It is the faithful port of a declared
model parameter, and it is marked as such.**

### 6.3 The transmitter classes, one by one

| class | neurons | v1 / v2 fast weight | **v3 fast weight** | basis |
|---|---|---|---|---|
| acetylcholine | 103,720 | `+ count × 0.275` | unchanged | nicotinic cation channels; Lee & O'Dowd 1999 |
| GABA | 22,069 | `− count × 0.275` | unchanged | Rdl chloride channel; Su & O'Dowd 2003 |
| glutamate | 29,302 | `− count × 0.275` | unchanged | GluClα chloride channel; Liu & Wilson 2013 |
| histamine | 7,891 | `− count × 0.275` | unchanged | ort/HisCl chloride channels |
| **dopamine** | **392** | `+ count × 0.275` | **0** | receptors are GPCRs (§6.3.1) |
| **octopamine** | **101** | `+ count × 0.275` | **0** | receptors are GPCRs (§6.3.1) |
| **serotonin** | **48** | `+ count × 0.275` | **0** | receptors are GPCRs (§6.3.1) |
| **`unclear`** | **2,999** | `+ count × 0.275` | **unchanged in the primary** (§6.3.2) | switchable, declared |
| unlabelled | 178 | (no out-edges) | same switch as `unclear` | — |

#### 4.3.1 Dopamine, octopamine and serotonin become modulatory-only

**The claim.** In *Drosophila*, the receptors for dopamine, octopamine and
serotonin are G-protein-coupled. Blenau, W. & Baumann, A. (2001), *Molecular and
pharmacological properties of insect biogenic amine receptors: lessons from*
Drosophila melanogaster *and* Apis mellifera, **Archives of Insect Biochemistry
and Physiology** 48:13–38, DOI `10.1002/arch.1055`, reviews the cloned
*Drosophila* dopamine, octopamine, tyramine and serotonin receptors and
identifies them as members of the GPCR superfamily signalling through cAMP and
Ca²⁺ second messengers. Evans, P.D. & Maqueira, B. (2005), *Insect octopamine
receptors: a new classification scheme based on studies of cloned* Drosophila
*G-protein coupled receptors*, **Invertebrate Neuroscience** 5:111–118, DOI
`10.1007/s10158-005-0001-z`, classifies the *Drosophila* octopamine receptors
into α-adrenergic-like (Oamb), β-adrenergic-like (Octβ1–3R) and
octopamine/tyramine classes — all GPCRs, none an ionotropic channel.

**Why that matters here and not elsewhere.** This engine has exactly one
synaptic mechanism: a conductance with a 5 ms exponential decay opening 1.8 ms
after a presynaptic spike. A GPCR cascade is not that mechanism — it is slower
by one to three orders of magnitude, it does not open a channel directly, and it
typically modulates excitability or synaptic gain rather than injecting charge.
Mapping an aminergic spike onto a 5 ms excitatory conductance is therefore not an
approximation of neuromodulation; it is a different mechanism wearing its name.
**Zero fast weight is the more accurate of the two available statements.** The
engine cannot represent what these neurons do, so it should not pretend to.

**What is lost.** These neurons keep their identity, their edges and their place
in the graph; only the fast weight of their out-edges is zeroed. They remain
available to WP6 as the modulatory signal `m(t)` — which is exactly the
double-counting `docs/WP6_PLASTICITY_SPEC.md` §3.5 flagged and option (b) of its
Q5. 435,541 of 25,582,938 edges (1.70 %) carry zero weight as a result.

**The honest limitation, stated in advance.** Aminergic neurons in *Drosophila*
can co-release a fast transmitter: Croset, V., Treiber, C.D. & Waddell, S.
(2018), *Cellular diversity in the* Drosophila *midbrain revealed by single-cell
transcriptomics*, **eLife** 7:e34550, DOI `10.7554/eLife.34550`, find fast
transmitter markers co-expressed in aminergic populations. The released
`neurotransmitter` field records **one** predicted transmitter per neuron, so the
data cannot say which of these 541 neurons also releases acetylcholine or
glutamate. v3 therefore replaces one known error (all of them fast excitatory)
with a different known error (none of them fast anything). The second is the
smaller claim, and it is the one WP6's own spec prefers, but it is a claim, and
the sensitivity arm in §6.6 exists because of it.

#### 4.3.2 The 2,999 `unclear` neurons: kept excitatory, declared, switchable

These are handled **separately** from the aminergic populations, because the
label means something entirely different. `unclear` is the *classifier's*
abstention — the synapse-image transmitter predictor (Eckstein, N., Bates, A.S.,
Champion, A., *et al.* 2024, *Neurotransmitter classification from electron
microscopy images at synaptic sites in* Drosophila melanogaster, **Cell**
187:2574–2594.e23, DOI `10.1016/j.cell.2024.03.016`) did not reach a confident
consensus for that cell. It is a statement about the evidence, not about the
neuron. There is no physiological claim that an `unclear` neuron makes no fast
synapse; overwhelmingly it makes one, and we do not know its sign.

The three options and the decision:

* **`exclude`** — zero the neuron's out-edges *and* every edge onto it, removing
  it from the simulated network. Rejected for the primary: it deletes 2,999
  neurons' worth of real anatomy on the strength of a missing label, and it
  changes the effective graph far more than either alternative.
* **`zero`** — zero its out-edges only. Rejected for the primary: it asserts
  "this neuron makes no fast synapse", which is a *stronger* and less supported
  claim than the one it replaces. It is retained as a declared sensitivity arm
  (§6.6) precisely because it is the arm that removes the excitatory bias.
* **`excitatory` — ADOPTED for the primary.** Leave them exactly as v1 and v2
  had them: the declared ambiguous sign `+1` of `brainlab/transmitters.py`. This
  is the only option that keeps v3 a **single** attributable change relative to
  v2 plus the aminergic change, and it inherits an assumption already declared
  and already reviewed rather than introducing a new one.

The decision is recorded as `unclear_mode` in every v3 manifest and receipt, and
is switchable at the command line. It is chosen on this reasoning and **not** on
its effect on the excitation/inhibition balance; §6.5 reports that effect as a
consequence.

The primary is therefore: **aminergic neurons lose their fast weight, `unclear`
neurons do not.** Policy name `v3-modulatory-only`, `unclear_mode='excitatory'`.

### 6.4 Identity, state and checkpoints

* `dynamics_version = 'v3'`; controller version `brainlab-lif-v3`
  (`brainlab-lif-plastic-v3`, `brainlab-lif-readout-v3`), derived by
  `provenance.controller_version_for` exactly as v2's was.
* The declared dict is `brainlab.graph_identity.LIF_DYNAMICS_V3`; its pin is
  `graph_identity.dynamics_pin('v3')`.
* **The pinned `graph.npz` is never rewritten.** The transmitter policy is
  applied to the weight array in memory by
  `brainlab.transmitter_policy.apply_to_shared`, which returns a `SharedGraph`
  with its **own** `graph_sha256` and a label naming the policy. v1 and v2 keep
  loading the identical bytes they always loaded and stay bit-reproducible.
* `ExperimentRegistry.read_checkpoint` already refuses a checkpoint whose
  `graph_sha256` differs, so a v2 checkpoint cannot enter a v3 run or the
  reverse. In addition, **every `Brain` snapshot now records its dynamics
  version** and `restore_state` refuses a mismatch by name: v2 and v3 share the
  `(2, n)` synaptic state shape, so the shape check alone could not separate
  them. A snapshot written before this field existed makes no claim and still
  falls back to the shape check, which separates v1 from v2 as before.
* Default dynamics stays **v1**. v3 does not become the default by existing; §6.7
  states what it would have to pass first, and any change of default is proposed
  to the owner, never taken.

### 6.5 Predicted consequences — computed before the run, reported whatever happens

Static arithmetic over the pinned graph (`brainlab.transmitter_policy._balance`,
recorded in the receipt):

| configuration | Σ excitatory weight | Σ inhibitory weight | `r = ĝ_i/ĝ_e` | high-conductance fixed point |
|---|---|---|---|---|
| v2 quanta, no policy (the v2 run) | 2.110 × 10⁷ | 1.305 × 10⁷ | 0.619 | **−26.75 mV** (suprathreshold by 18.25 mV) |
| v3 quanta, no policy | 2.110 × 10⁷ | 1.305 × 10⁷ | 1.787 | **−44.88 mV** (suprathreshold by 0.12 mV) |
| **v3 quanta + v3 policy (the primary)** | 2.077 × 10⁷ | 1.305 × 10⁷ | **1.815** | **−45.13 mV** (subthreshold by 0.13 mV) |
| v3 quanta + policy + `unclear_mode=zero` | 2.021 × 10⁷ | 1.305 × 10⁷ | 1.865 | −45.57 mV (subthreshold by 0.57 mV) |

Threshold is −45 mV; a subthreshold fixed point needs `r > 1.80`.

**This is a knife edge and is declared as one.** The primary configuration clears
the threshold by 0.13 mV, about 0.3 % in `r`. The aminergic change contributes
0.028 of that margin and is what carries the configuration across the line; that
is a consequence of a decision taken on independent grounds (the owner's, on
§3.5 of the WP6 spec) and of a calibration rule locked before this table was
computed, but it is a coincidence and is reported as one rather than as a
design.

Predictions, in order of confidence:

1. **Network rate falls sharply from v2's 4.1 × 10⁶ spikes/s.** High confidence.
   The mean-field fixed point moves 18.4 mV and inhibition gains a factor 2.889
   in authority. Predicted: at least an order of magnitude, i.e. below
   ~4 × 10⁵ spikes/s during stimulation.
2. **The gray periods get much closer to rest than v2's 3.58 × 10⁶ spikes/s.**
   Moderate confidence. The network is only marginally subthreshold *in the
   mean*, and it is heterogeneous: a neuron whose own inhibitory share is below
   the graph average has a suprathreshold local fixed point and can sustain
   activity regardless of the global number. **Silence is not predicted.** What
   is predicted is that the *self-sustained* state is no longer guaranteed by the
   mean-field arithmetic, which it was under both v1 and v2.
3. **The membrane stays bounded in [−70, 0] mV.** Certain by construction.
4. **Both DNa02s remain responsive and side-specific.** Moderate confidence:
   this was v2's one genuine gain (the §6.1 dead-right-side defect is a v1
   artefact of unbounded hyperpolarisation, which v3 also lacks), and v3 changes
   nothing about the encoder, the decoder or the HS/H2 wiring.
5. **The optomotor verdict is genuinely open.** Explicitly: a smaller `TI` than
   v1's, a null, or a negative are all possible outcomes, and *this document
   does not predict a positive one*. What v3 is expected to deliver is a
   **measurable regime** — a network whose baseline is not saturation — not a
   positive result. If a quieter, better-behaved network still shows no
   DNa02-mediated turning, that is the finding, and it is a more informative
   finding than v2's, because it can no longer be blamed on saturation.
6. **Cost.** v3 does more work per delivered spike than v1 and the same as v2 per
   spike, but should emit far fewer spikes, so it is predicted to run **faster**
   than v2's 0.030 simulated s per wall s.

### 6.6 What would falsify this approach

v3 is wrong, or at least not the fix, if any of the following is observed. Each
is reported whichever way it comes out; none of them licenses a retune.

* **F1 — the runaway survives.** Gray-period network rate stays above 10⁶
  spikes/s. Then the saturation is not explained by the E/I conductance balance
  and the analysis in §6.1 is incomplete.
* **F2 — the network dies.** Zero or near-zero spikes during stimulation, i.e.
  the marginal fixed point tipped the graph into silence. Then 2.889 overshoots
  and the conclusion is that no single global scale makes this graph both quiet
  and responsive, which is itself a result about the proxy.
* **F3 — the measured unitary PSPs do not match the declared calibration.**
  Checked directly on a two-neuron graph: the v3 unitary IPSP at rest must equal
  the v1 unitary IPSP to within the integrator's own error. *(Measured before
  any network run: v1 −0.04332 mV, v2 −0.01514 mV, v3 −0.04367 mV; v3 is within
  0.8 % of v1, the residual being exponential-Euler versus v1's exact kernel.
  v2's 0.346× shortfall is visible directly.)*
* **F4 — the membrane leaves [E_inh, E_exc].** Would mean the implementation,
  not the model, is wrong.
* **F5 — the result depends on the `unclear` decision.** Reported by the
  sensitivity arm below. If `unclear_mode='zero'` changes the optomotor verdict,
  then the verdict rests on 2,999 unlabelled neurons and no claim survives
  either way.

### 6.7 Declared acceptance criteria, and what is run only if they pass

These were fixed in the locked declaration, before any v3 simulation. They are
evaluated on the **cheap** probes (§6.8), not on the confirmatory set.

* **Q1 — quiet baseline.** In both gray windows of probe C: network rate
  < 1.0 × 10⁵ spikes/s **and** DNa02_L and DNa02_R each < 20 Hz.
* **R1 — responsiveness.** During rotation: network rate > 0, the sign of the
  DNa02 L−R rate difference follows the stimulus in **both** directions, and
  |L−R| ≥ 1 Hz in at least one direction.
* **R2 — bounded membrane.** min `V` ≥ `E_inh` in every window.

**If Q1 or R1 fails, the 24-run confirmatory set is not run.** The failure is
the result and is reported as such. This is a budget rule and an integrity rule
at once: it removes the option of running the expensive set and then deciding
what to make of it.

**Declared sensitivity arms** (pre-registered here, before any of them runs; each
is labelled as an arm and none of them can change the primary verdict):

* **S1 — `unclear_mode='zero'`**, probe C only, both directions. Tests F5.
* **S2 — `E_inh = −56 mV`**, probe C only, both directions: the measured
  *Drosophila* larval chloride reversal (Rohrbough & Broadie 2002). Under v3 the
  inhibitory quantum follows it to `1/4`, so this arm tests the calibration and
  the reversal together, as it must.

No other configuration is run. If the primary fails, the answer is that it
failed — a third gain is not tried.

### 6.8 Predeclared measurements

1. **Probe A** (two neurons, unbounded-voltage check and unitary PSP
   calibration), **probe B** (2,000-neuron random recurrent net, self-sustained
   state as a function of recurrent gain) and the **fixed-point calculation** of
   §6.5, all three run for v1, v2 and v3 side by side. Seconds to minutes.
2. **Probe C** — 500 ms gray, 1 s rotation, 500 ms gray on the pinned MaleCNS
   graph with the unchanged WP5 encoder, one run per direction, plus arms S1 and
   S2. Minutes. **Q1/R1/R2 are evaluated here.**
3. **Only if Q1 and R1 pass:** the WP5 preregistered optomotor protocol,
   unchanged script, unchanged preregistration
   (`docs/wp5_optomotor_prereg.json`, same sha256), unchanged primary outcome
   `TI`, same seeds 0–5, same four conditions, same verdict rule. Only
   `--dynamics v3` differs. Reported beside the v1 and v2 numbers.

Receipt: `docs/receipts/lif_dynamics_v3.json`. WP5 §12 carries the verdict.

### 6.9 What v3 does *not* claim

* It is **not** a biophysical model. One compartment, two conductances, no
  active channels, no adaptation, a coarse sign proxy for the fast classes and a
  coarse class proxy for the slow ones.
* `E_inh = −70 mV` remains an engineering assumption (§3.1), and the v3
  inhibitory quantum is derived from it, so it inherits that status.
* The 2.889 ratio is **not** a measured *Drosophila* unitary conductance ratio.
  It is the ratio of driving forces at rest, which is what preserving the source
  model's declared PSP requires.
* Zeroing the aminergic fast weight does **not** model neuromodulation. It
  declines to mis-model it. There is still no `m(t)` in this engine; that is
  WP6's to build.
* Nothing here makes the WP5 encoder's direction selectivity graph-computed. It
  remains an encoder assumption, unchanged since §3 of `docs/WP5_OPTOMOTOR.md`.
* A recalibrated E/I balance is not by itself a cure for saturation. Whether the
  self-sustained state survives is empirical, it is answered in the receipt, and
  the honest answer is reported whatever it is.


## 7. References

* Shiu, P.K. *et al.* (2024). A *Drosophila* computational brain model reveals
  sensorimotor processing. *Nature* 634:210–219. Preprint: bioRxiv
  2023.05.02.539144. Implementation: `github.com/philshiu/Drosophila_brain_model`.
* Lee, D. & O'Dowd, D.K. (1999). Fast excitatory synaptic transmission mediated
  by nicotinic acetylcholine receptors in *Drosophila* neurons.
  *J. Neurosci.* 19:5311–5321.
* Su, H. & O'Dowd, D.K. (2003). Fast synaptic currents in *Drosophila*
  mushroom body Kenyon cells are mediated by α-bungarotoxin-sensitive nicotinic
  acetylcholine receptors and picrotoxin-sensitive GABA receptors.
  *J. Neurosci.* 23:9246–9253.
* Rohrbough, J. & Broadie, K. (2002). Electrophysiological analysis of synaptic
  transmission in central neurons of *Drosophila* larvae.
  *J. Neurophysiol.* 88:847–860.
* Liu, W.W. & Wilson, R.I. (2013). Glutamate is an inhibitory neurotransmitter
  in the *Drosophila* olfactory system. *PNAS* 110:10294–10299. (GluCl is a
  chloride conductance; basis for treating glutamate as inhibitory.)
* Wilson, R.I. & Laurent, G. (2005). Role of GABAergic inhibition in shaping
  odor-evoked spatiotemporal patterns in the *Drosophila* antennal lobe.
  *J. Neurosci.* 25:9069–9079.
* Maisak, M.S. *et al.* (2013). A directional tuning map of *Drosophila*
  elementary motion detectors. *Nature* 500:212–216. (T4/T5 subtype assignment
  used by the WP5 encoder; unchanged here.)
* Rayshubskiy, A. *et al.* (2020). Neural control of steering in walking
  *Drosophila*. bioRxiv 2020.04.04.024703. (DNa02 ipsilateral steering;
  unchanged here.)

Added for v3 (§6):

* Blenau, W. & Baumann, A. (2001). Molecular and pharmacological properties of
  insect biogenic amine receptors: lessons from *Drosophila melanogaster* and
  *Apis mellifera*. *Archives of Insect Biochemistry and Physiology*
  48:13–38. DOI [10.1002/arch.1055](https://doi.org/10.1002/arch.1055).
  (*Drosophila* dopamine, octopamine, tyramine and serotonin receptors are
  cloned GPCRs signalling through cAMP/Ca²⁺; basis for §6.3.1.)
* Evans, P.D. & Maqueira, B. (2005). Insect octopamine receptors: a new
  classification scheme based on studies of cloned *Drosophila* G-protein
  coupled receptors. *Invertebrate Neuroscience* 5:111–118. DOI
  [10.1007/s10158-005-0001-z](https://doi.org/10.1007/s10158-005-0001-z).
  (All three *Drosophila* octopamine receptor classes are GPCRs.)
* Croset, V., Treiber, C.D. & Waddell, S. (2018). Cellular diversity in the
  *Drosophila* midbrain revealed by single-cell transcriptomics. *eLife*
  7:e34550. DOI [10.7554/eLife.34550](https://doi.org/10.7554/eLife.34550).
  (Fast-transmitter markers co-expressed in aminergic populations; the declared
  limitation of §6.3.1.)
* Eckstein, N., Bates, A.S., Champion, A., *et al.* (2024). Neurotransmitter
  classification from electron microscopy images at synaptic sites in
  *Drosophila melanogaster*. *Cell* 187:2574–2594.e23. DOI
  [10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016).
  (What the `unclear` label means; basis for §6.3.2.)
* Shiu, P.K. *et al.* (2024). DOI
  [10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9).
  (DOI added; the entry above is the same paper.)
