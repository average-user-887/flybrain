# Declared LIF dynamics for `brainlab` — v1 current-based, v2 conductance-based, v3 recalibrated, v4 hybrid graded/spiking

Written **before** the v2 implementation and before any v2 measurement, as
required by `docs/archive/CLAUDE_IMMEDIATE_PLAN.md` ("declare the parameters and their
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

**§7 adds v4**, the hybrid graded/spiking mode, under the same rule; see §7 and its
locked declaration. **§9 adds v5**, per-receptor-class synaptic kinetics, under the
same rule; see §9 and its locked declaration.

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
  *(Update 2026-09-24: since PR #10 the library default is v3 as well
  (`graph_identity.py`, `Brain(..., dynamics='v3')`); v1 is opt-in. The daemon, `neurofly run`,
  defaults to v3 and applies the v3 transmitter policy (PR #4). Before PR #4,
  `NEUROFLY_LIF_DYNAMICS=v3` in the daemon ran the v3 equations on v1 weights. See
  `docs/receipts/README.md`.)*
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
  to the owner, never taken. *(Update 2026-09-24: the owner made v3 the default for
  the daemon (PR #4) and the library (PR #10), as a roadmap decision, before any v3
  behavioural gate passed.)*

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



---

## 7. v4 — hybrid graded/spiking transmission (new)

**Written before the v4 engine was run on anything larger than a two-neuron
test graph, before any graded class was resolved against the real graph, and
before any v4 network measurement.** The graded membrane equation, the release
function and its derivation, the declared graded cell classes, the predicted
consequences, the falsification criteria and the stop rule before the
expensive protocol were all fixed and hashed first. The locked declaration is

    sha256 2634c824476c9b799bd2c0c656040255360cd10f8837e2d874a97af5f939872e

written 2026-09-27T11:15:16+02:00, and is reproduced verbatim in
[`receipts/graded_transmission_v4_declaration.locked.md`](receipts/graded_transmission_v4_declaration.locked.md).
If this section and that file ever differ, **that file is the declaration.**

v1, v2 and v3 are **not** deleted, superseded or reinterpreted. All three stay
selectable, keep their pins, stay bit-reproducible, and every number published
under them stands as a result of that controller version. The default stays
v3.

## 7.1 Why a graded mode is needed at all

Shiu, P.K., Sterne, G.R., Spiller, N., *et al.* (2024), *Nature* 634:210–219
(open access, PMC11446845) — the source of every LIF constant this engine uses
(§1.1) — states its own scope explicitly: the model "does not account for gap
junctions, non-spiking neurons, internal state or long-range neuropeptides",
and it treats "each neuron identically as a spiking neuron". v1, v2 and v3
inherit that limitation exactly, because they inherit the model.

Large parts of the *Drosophila* visual system do not spike in the animal.
Photoreceptors R1–R6, the lamina monopolar cells L1–L5, the medulla columnar
interneurons and the lobula plate tangential cells including HS signal with
**graded** membrane potential and tonic transmitter release. In a spiking-only
engine their signal cannot leave the cell at all until it crosses −45 mV, and
the subthreshold part of it — which is the whole of it, for these cells — is
discarded.

This has bitten in three recorded places.

1. **The v3-1 validation FAILED on an HS spike rate.** Physiology checks P4/P5
   measured HS_L at 60 Hz and HS_R at 53 Hz against a 0–50 Hz ceiling
   (`docs/receipts/validation/optomotor-yaw-v3-1.md`). `optomotor_v3_2.json`
   made those checks report-only and recorded honestly that the change was
   prompted by the failure. Both records are right about the deeper point: an
   HS spike rate is not a biologically comparable quantity, because HS cells
   are non-spiking (Fujiwara *et al.* 2017) or spikelet-only (Joesch *et al.*
   2008). The check could not be fixed by changing its bound; the quantity was
   wrong.

2. **Driving the graph from photoreceptors under v3 produces nothing.** A
   retinotopically correct moving grating delivered only to R1–R6 under v3
   (measured by a separate session, relayed to this one before this
   declaration was written, and to be reproduced here rather than copied)
   gives: R1–R6 firing at 38.6 / 39.6 Hz with 98–99 % of cells active, and
   then nothing. L1, L2 and L3 fire at 0 Hz with their **maximum** membrane
   potential over the entire rotation at exactly V_rest = −52.00 mV; every
   population downstream of the lamina — medulla, T4/T5, HS/H2/VS, DNa02 —
   sits at exactly −52.00 mV and never moves; decoded yaw is identically
   0.000000. The signal dies at the first synapse.

3. **Direction selectivity is still supplied by the encoder, not the graph.**
   `brainlab/io_map.py` says so in its own docstring, and every spec version
   since WP5 §3 has declared it. Removing that assumption means driving R1–R6
   and letting the graph compute, which requires the graded pathway of (2) to
   conduct.

### 7.1.1 Graded transmission alone is not sufficient, and this declaration says why

Every out-edge of every R1–R6 in the pinned graph is **negative**: R1–R6 are
labelled histaminergic (3,377 of 3,377), and the coarse fast-transmission
proxy of `brainlab/transmitters.py` treats histamine as inhibitory, which is
correct — *Drosophila* photoreceptors release histamine onto ort/HisCl
chloride channels on L1/L2. In the animal this pathway works by
**disinhibition**: photoreceptor release is tonic, L1/L2 are tonically
inhibited, and a light decrement releases them. v1–v3 have exactly zero
spontaneous activity, so there is nothing to release, and an inhibitory first
synapse onto a silent network can only push silent cells further down. That
is precisely the −52.00 mV ceiling measured in (2).

A graded mode that rectified release from silence — release zero at rest,
rising only with depolarisation — would reproduce the same null one stage
later: L1/L2 would be pushed below rest and would still have nothing to
release from. **A maintained baseline is therefore not optional for this
pathway, and §7.3 provides one, as a consequence of the anchors the release
function is derived from rather than as an addition.** §7.4 states the
consequences for the network's resting state, and §7.8 predeclares the ways
this can fail.

The third dependency is temporal: one global `tau_syn = 5 ms` and one global
1.8 ms hop delay cannot express a 20–50 ms delay line. §7.7 P4 and §7.8 F6
predeclare this as the most likely reason a *working* graded pathway would
still fail to produce a Reichardt-style correlator, and it is **not** fixed
here. v4 adds graded transmission and a maintained baseline; it does not add
cell-type-specific synaptic kinetics.

v1, v2 and v3 are **not** deleted, superseded or reinterpreted. All three stay
selectable, keep their pins, and every number published under them stands as a
result of that controller version. The default stays v3; a change of default
is proposed to the owner, never taken (§7.6).

## 7.2 The graded membrane equation

A neuron whose cell class is **declared graded** (§7.5) is integrated as a
passive membrane with the same conductance model, the same constants and the
same integrator as v3, and with **no threshold, no reset and no refractory
period**:

    tau_m dV/dt = (E_leak − V) + ĝ_e(t)(E_exc − V) + ĝ_i(t)(E_inh − V) + I_ext
    tau_syn dĝ_e/dt = −ĝ_e
    tau_syn dĝ_i/dt = −ĝ_i

integrated by the same exponential Euler step as v2/v3, with the conductances
frozen within each `dt`:

    g_tot = 1 + ĝ_e + ĝ_i
    V_inf = (E_leak + ĝ_e·E_exc + ĝ_i·E_inh + I_ext) / g_tot
    V     ← V_inf + (V − V_inf)·exp(−dt·g_tot / tau_m)
    V     ← min(max(V, E_inh), E_exc)
    ĝ_e   ← ĝ_e·exp(−dt/tau_syn);  ĝ_i ← ĝ_i·exp(−dt/tau_syn)

| symbol | value | units | source / status |
|---|---|---|---|
| `dt` | 0.1 | ms | unchanged (§1.1) |
| `tau_m` | 20 | ms | unchanged; Shiu *et al.* 2024 `t_mbr` |
| `tau_syn` | 5 | ms | unchanged; Shiu *et al.* 2024 `tau` |
| `E_leak` = `V_rest` | −52 | mV | unchanged; Shiu *et al.* 2024 `v_0` |
| `E_exc` | 0 | mV | unchanged (§3.1); Lee & O'Dowd 1999 |
| `E_inh` | −70 | mV | unchanged (§3.1); **engineering assumption**, inherited |
| `V` bounds | [−70, 0] | mV | unchanged (§3.1) |
| `V_threshold`, `V_reset`, `t_refractory` | — | — | **not applicable to a graded cell** |

**No new membrane parameter is introduced.** A graded cell in v4 is a v3 cell
with the spike mechanism removed. The membrane time constant a graded cell
actually shows is `tau_m / g_tot`, i.e. it shortens under load exactly as a v3
cell's does; that is a property of the conductance model, not a new constant.

## 7.3 The release function, and how its scale is derived

A declared graded cell has no spikes, so it needs a rule that turns its
membrane potential into synaptic drive on its targets. v4 declares

    r(V) = r_max · (V − E_inh) / (E_exc − E_inh),     r_max = 1 / t_refractory

with `r` in release events per second, and delivers over each out-edge `e`,
every timestep, after the same 1.8 ms transmission delay the spiking path
uses:

    Δĝ_e-or-i(target of e) = |w_e| · g_unit_<sign of w_e> · r(V_pre) · dt/1000

using the **same** weight array, the **same** per-sign conductance quanta
`g_unit_exc = 1/52` and `g_unit_inh = 1/(V_rest − E_inh) = 1/18` (§6.2), the
**same** sign convention, and the **same** rule that an arrival at a
refractory target is dropped (§1.2). Numerically:

| quantity | value | where it comes from |
|---|---|---|
| `r_max` | 1 / 2.2 ms = **454.545** s⁻¹ | `t_rfc = 2.2 ms`, Shiu *et al.* 2024, already declared in §1.1 |
| `r(E_inh) = r(−70 mV)` | 0 s⁻¹ | lower membrane bound, already declared in §3.1 |
| `r(E_exc) = r(0 mV)` | 454.545 s⁻¹ | upper membrane bound, already declared in §3.1 |
| `r(V_rest) = r(−52 mV)` | **116.883** s⁻¹ (25.71 % of `r_max`) | consequence, not a choice |
| `r(V_threshold) = r(−45 mV)` | 162.338 s⁻¹ | consequence |

**Why this is a derivation and not a fit.** Three requirements fix it
completely, and each of them is already in this document:

1. **The unit in which a connectome edge weight means anything is the PSP per
   presynaptic release event** (§1.1: `w_syn = 0.275 mV` *per anatomical
   synapse per event*). Graded release therefore has to be expressed as an
   event *rate*, because that is the only thing that keeps one weight array
   meaning the same thing for a spiking and for a graded presynaptic cell.
   That fixes the **form** `Δĝ = w · g_unit · r · dt`, and it makes a graded
   cell at release rate `r` deliver exactly the mean conductance that a
   spiking cell firing at `r` Hz delivers in v3 — which is what lets the two
   populations interoperate at all.
2. **Release must vanish where the membrane can go no lower, and be maximal
   where it can go no higher.** The membrane bounds are `[E_inh, E_exc]`,
   declared in §3.1 before v2 ran. So the two anchors of `r(V)` are not new
   numbers.
3. **The largest event rate this engine can represent is its own refractory
   ceiling**, `1/t_rfc`, declared in §1.1. Anything larger would let a graded
   cell out-drive any spiking cell in the same graph on the same weights;
   anything smaller would require a new constant with no source.

**There is no free parameter.** `r_max`, both anchors and both conductance
quanta are already-declared values, and `r(V_rest)` is a consequence of them.
If `E_inh` is moved by a sensitivity arm, `r(V)` moves with it, because the
release function is a statement about the membrane's own range and not a
number.

**The engineering assumption is the linearity, and it is declared as one.** No
graded input–output curve measured in *Drosophila* is expressed in this
engine's units, so there is nothing to fit a shape to. A straight line between
two declared anchors is the only choice that adds no shape parameter. Real
graded synapses are sigmoid, saturate, and depress; v4 has none of that, and
§7.9 lists it.

## 7.4 The maintained baseline, stated explicitly

Because the zero of `r(V)` sits at `E_inh` and not at `V_rest`, a graded cell
at rest releases at `r(V_rest) = 116.883` s⁻¹. **Graded transmission in v4 is
therefore bidirectional about a set point: hyperpolarisation below rest
reduces release and depolarisation above rest increases it, and both are
transmitted.** This is the property §7.1.1 says the photoreceptor pathway
cannot work without, and it is a consequence of anchoring `r(V)` on the
membrane bounds rather than something added to obtain it. Had the zero been
placed at `V_rest` instead, every hyperpolarising graded signal in the visual
pathway — including the entire OFF channel — would have been rectified away,
and the declaration would have been strictly less faithful for the sake of a
quieter network.

Predeclared consequences, reported whichever way they come out:

* **v4 breaks the zero-spontaneous-activity property that v1–v3 inherit from
  Shiu *et al.*** Every declared graded cell releases from `t = 0`, with no
  stimulus. A v4 "gray" window is a tonic-release steady state, not silence,
  and is **not** comparable with a v1/v2/v3 gray window. The v3 acceptance
  criterion Q1 ("network rate < 1.0 × 10⁵ spikes/s in gray") is **not**
  inherited by v4 and is not applied to it; §7.8 F2 and F3 replace it with
  bounds on the steady state itself.
* Under the primary graded declaration this is 95,545 cells (§7.5), i.e. 57 %
  of the graph, releasing tonically. Whether the remaining spiking population
  then sits at a usable operating point is an **empirical question**, it is
  answered in §7.10, and the answer is reported whatever it is.
* An alternative was considered and rejected: a separate declared resting
  release level `r_0`. Rejected because it is a free parameter with no source,
  and because the anchored form already supplies a baseline.

## 7.5 Which cell classes are declared graded, and on what evidence

Resolved by `cell_type` and `superclass` strings from the released
`neurons.feather`, and by `somaSide` / `rootSide` / `instance` from
`annotations.feather` — **never by dataframe row order**. The resolved node set
is hashed and pinned, and a different resolution raises
`GraphUnavailable`, exactly as the WP5 IO map does.

### Tier 1 — named cell types with direct evidence of graded (non-spiking) signalling in *Drosophila*

| cell types | n | evidence |
|---|---|---|
| `R1-R6` | 3,377 | Photoreceptors respond with graded depolarisation and tonic histamine release; no action potentials. Hardie 1991; Juusola & Hardie 2001 *J Gen Physiol* 117:3–25. |
| `R7d R7p R7y R8d R8p R8y R7_unclear R8_unclear R7R8_unclear` | 631 | Same class of cell, same signalling mode. |
| `L1 L2 L3 L4 L5` | 8,884 | Graded, non-spiking lamina monopolar cells. Clark *et al.* 2011 *Neuron* 70:1165 (L1/L2 whole-cell and imaging); Freifeld *et al.* 2013 *Neuron* 78:1075 (L2 graded voltage). |
| `Lai` | 86 | Lamina amacrine; graded, no spikes reported. |
| `Mi1 Tm3 Tm1 Tm2 Tm4 Tm9` | 10,804 | Whole-cell recordings: graded voltage responses, no spikes. Behnia *et al.* 2014 *Nature* 512:427–430; Yang *et al.* 2016 *Neuron* 92:227; Arenz *et al.* 2017 *Curr Biol* 27:929. |
| `Mi4 Mi9` | 3,547 | Graded ON/OFF medulla interneurons; Strother *et al.* 2017 *Curr Biol* 27:3132; Arenz *et al.* 2017. Evidence is largely calcium imaging, so weaker than the row above; declared graded on class membership. |
| `CT1` | 2 | Non-spiking and electrotonically compartmentalised. Meier & Borst 2019 *Curr Biol* 29:3277. |
| `T4a T4b T4c T4d T5a T5b T5c T5d T4_unclear T5a_unclear` | 13,580 | Whole-cell recordings of T4 and T5 show graded voltage responses with no action potentials. Gruntman, Romani & Reiser 2018 *Nat Neurosci* 21:250–257; Wienecke, Leong & Clandinin 2018 *Neuron* 100:1058. |
| `HSN HSE HSS HST H1 H2 VS VST1 VST2 VSm` | 46 | Lobula plate tangential cells: graded responses, at most small irregular spikelets. Joesch *et al.* 2008 *Curr Biol* 18:368–374 (VS: graded with small irregular action potentials); Schnell *et al.* 2010 *J Neurophysiol* 103:1646 (HSN/HSE/HSS whole-cell); Fujiwara *et al.* 2017 *Nat Neurosci* 20 (HS non-spiking during walking). |

### Tier 2 — declared extension by anatomical class (ENGINEERING ASSUMPTION)

**Primary: every remaining neuron of `superclass ∈ {ol_intrinsic, ol_sensory}`.**

The reasoning, stated as an assumption and not as physiology: every
*Drosophila* optic lobe neuron that has been recorded intracellularly to date
signals gradedly, and none has been reported to spike. The alternative — a
boundary drawn part-way along the pathway at the last cell type somebody has
patched — is itself an untested assumption, and a worse one, because it would
put a spiking stage in the middle of a graded cascade and make every result
depend on where the literature happens to stop. Tier 2 is the coarser but more
consistent statement, and it is switchable (§7.6, arm S1).

`visual_projection` neurons are **excluded** from tier 2, because lobula
columnar cells do spike (von Reyn *et al.* 2017 *Nat Neurosci* 20:1247, LC4
and LPLC2 spike trains) — with the single exception of the lobula plate
tangential cells named in tier 1, which are annotated `visual_projection` but
are not columnar. `visual_centrifugal`, all central-brain and all VNC classes
stay spiking.

### The primary declared graded set

| component | neurons |
|---|---|
| `superclass == 'ol_intrinsic'` | 89,403 |
| `superclass == 'ol_sensory'` | 6,098 |
| lobula plate tangential types annotated `visual_projection` (`HSN HSE HSS HST H2 VS VST1 VST2 VSm`) | 44 |
| **total declared graded (primary)** | **95,545** of 166,700 (57.3 %) |

Every tier-1 cell type above is a subset of this set; the tiers are a statement
about the strength of the evidence, not two different mechanisms. Policy name
**`v4-optic-lobe-graded`**.

Everything else — the whole central brain, the VNC, the descending neurons
including DNa02, the lobula columnar visual projection neurons — remains a v3
spiking cell, with v3's threshold, reset and refractory period, on v3's
weights under the v3 transmitter policy.

## 7.6 Interoperation, identity, pins and checkpoints

**Spiking → graded.** Unchanged: an arriving spike adds `|w|·g_unit_<sign>` to
the graded cell's conductance, which its membrane integrates. A graded cell
can be excited, inhibited and shunted by spiking cells exactly as a v3 cell
is.

**Graded → spiking.** The continuous conductance of §7.3 drives the v3
membrane of the spiking target exactly as a spike train would, and can carry
it over threshold. No extra mechanism, no separate pathway, no adapter.

**Graded → graded** and **spiking → spiking** follow from the two above.

* `dynamics_version = 'v4'`; controller version `brainlab-lif-v4`
  (`brainlab-lif-plastic-v4`, `brainlab-lif-readout-v4`), derived by
  `provenance.controller_version_for` exactly as v2's and v3's are.
* The declared dict is `brainlab.graph_identity.LIF_DYNAMICS_V4`; its pin is
  `graph_identity.dynamics_pin('v4')`.
* **v4 runs on the v3 weights under the v3 transmitter policy**
  (`v3-modulatory-only`, `unclear_mode='excitatory'`), unchanged and applied in
  memory by `transmitter_policy.apply_to_shared`, so the pinned `graph.npz` is
  still never rewritten and v1/v2 still load the identical bytes.
* **v1, v2 and v3 are unchanged and remain bit-reproducible.** v4 adds a
  kernel; it edits no v3 arithmetic. §7.7 P2 / §7.8 F4 make that a checked
  claim, not an intention.
* **Checkpoints keep mutually refusing each other.** A v4 snapshot records
  `dynamics='v4'` and `restore_state` refuses a version mismatch by name, as
  it already does between v2 and v3. v4 additionally carries the graded mask
  and the delayed-release ring buffer in its state, so its snapshot dict is not
  interchangeable with v3's even by shape. The resolved graded node set is
  hashed into every manifest and receipt (`graded_set_sha256`), so a
  checkpoint written under one graded declaration is refused by a run under
  another.
* **The default dynamics stays v3.** v4 does not become the default by
  existing. Any change of default is proposed to the owner and never taken.

**Declared arms** (pre-registered here, before any of them runs; each is
labelled as an arm, and none of them can change the primary verdict):

* **S0 — no class declared graded.** Must be bit-identical to v3 (§7.8 F4).
* **S1 — tier 1 only** (the named cell types with direct evidence, 40,957
  neurons), primary probes only. Tests F7.

No other configuration is run. If the primary fails, the answer is that it
failed; a third release scale is not tried.

## 7.7 What is predicted, before measuring

1. **P1 — subthreshold transmission works, and is the whole point.** High
   confidence. A graded cell whose membrane sits below −45 mV delivers a
   conductance to its targets proportional to `(V − E_inh)`; under v3 it
   delivers exactly nothing. Checked on a three-neuron graph.
2. **P2 — v4 with no declared graded class is bit-identical to v3.** Certain
   by construction and checked directly.
3. **P3 — HS becomes a membrane-potential readout and emits no spikes at
   all.** Certain by construction (HS is declared graded), so the v3-1 FAIL's
   60 Hz / 53 Hz becomes *not applicable* rather than *passing*. The
   comparable quantity is the HS depolarisation in mV to wide-field motion.
   The figure this project has been given for the animal is **about 5 mV**;
   the *Drosophila* LPTC literature reports single- to low-tens-of-millivolt
   graded responses depending on cell, velocity and contrast (Joesch *et al.*
   2008; Schnell *et al.* 2010). **Predicted: a v4 HS response between 0.1 and
   20 mV.** Outside that window is a mismatch and is reported as one (F5).
   The *sign* prediction is separate and stronger: HS should depolarise to
   front-to-back motion on its own eye and hyperpolarise to back-to-front
   (Schnell *et al.* 2010), and that is reported per direction.
4. **P4 — direction selectivity in T4/T5 is NOT predicted to reach the
   animal's value, and this document predicts a small one.** Reported T4/T5
   DSI in the animal is about 0.7–0.9. v4 has one global `tau_syn = 5 ms` and
   one global 1.8 ms hop delay, so it cannot express the 20–50 ms delay line
   that a Hassenstein–Reichardt or Barlow–Levick correlator needs at the
   declared 1.5 Hz temporal frequency. The only asymmetry available to it is
   the one synaptic hop that differs in *sign* (Mi9 glutamate-inhibitory,
   Mi4 GABA-inhibitory and Mi1 cholinergic onto T4; Tm1/Tm2/Tm4/Tm9 onto T5),
   and in this engine those hops have identical kinetics.
   **Predicted: |DSI| < 0.2 in every T4/T5 subtype.** If T4a and T4b
   nevertheless prefer opposite directions along one lattice axis, and T4c/T4d
   along another, that is a genuine positive finding and is reported as one.
   If they do not, the deliverable is the trace showing at which stage the
   direction information is absent, and the conclusion is about the engine's
   single global synaptic time constant, not about the connectome.
5. **P5 — the network will be far more active than v3, and may saturate.**
   Moderate confidence. 95,545 cells releasing tonically at ~117 s⁻¹
   equivalent is a large maintained conductance that v3 never had. The
   spiking population's rates and membrane distribution are reported
   whichever way they come out; saturation is F2 and is a result, not a
   licence to retune `r_max`.
6. **P6 — the specific v3 signature disappears.** Moderate-to-high
   confidence. Under v4 the lamina cells should no longer sit with their
   maximum pinned at exactly −52.00 mV: photoreceptor release is tonic and
   inhibitory, so L1/L2/L3 should rest *below* −52 mV and move *upward* on a
   light decrement. If their maximum is still exactly −52.00 mV, the graded
   path is not conducting and F1 applies.
7. **P7 — cost.** v4 does strictly more work per tick than v3, because graded
   out-edges are traversed every tick instead of only on a spike. Predicted:
   at least an order of magnitude slower than v3 per simulated second on the
   same hardware, and the real-graph measurements are GPU-only.

## 7.8 What would falsify this approach

Each is reported whichever way it comes out; none of them licenses a retune of
`r_max`, of the graded class list, or of any other parameter.

* **F1 — no subthreshold transmission.** A graded cell held below −45 mV
  produces no measurable conductance change in its targets, or the lamina
  cells' maximum membrane potential is still exactly −52.00 mV under
  photoreceptor drive. Then the implementation, not the model, is wrong.
* **F2 — the graded population saturates the network.** Declared graded cells
  pinned at `E_exc`, or the spiking population at its refractory ceiling, in
  the no-stimulus window. Then the derived `r_max` is too large for this
  graph, and the conclusion is that no single global release scale makes this
  graph both responsive and stable — a result about the proxy.
* **F3 — the graded population collapses.** Declared graded cells pinned at
  `E_inh` with release near zero, so nothing is transmitted.
* **F4 — S0 is not bit-identical to v3.** Then v4 has changed v1–v3 and the
  change is not attributable.
* **F5 — HS response outside 0.1–20 mV**, or HS depolarising to
  back-to-front motion on its own eye.
* **F6 — no direction selectivity emerges** (|DSI| < 0.2 in every T4/T5
  subtype, or T4a/T4b not anti-parallel). This is the *predicted* outcome
  (P4). It falsifies the claim that graded transmission is *sufficient* for
  the connectome to compute direction selectivity; it does not falsify the
  graded model itself, and the diagnostic trace is then the result.
* **F7 — the result depends on which graded class list is used.** If arm S1
  (tier 1 only) changes the verdict, then the verdict rests on the tier-2
  extension and no claim survives either way.

## 7.9 What v4 does *not* claim

* It is **not** a biophysical model. One compartment per cell, two synaptic
  conductances, no active channels, no adaptation, and a coarse
  transmitter-sign proxy.
* **No gap junctions.** The Shiu *et al.* limitation quoted in §7.1 is only
  half addressed: non-spiking neurons are now represented, gap junctions,
  internal state and neuropeptides are not. CT1's and the lamina's electrical
  coupling are absent.
* **No cell-type-specific synaptic kinetics.** One `tau_syn = 5 ms` and one
  1.8 ms delay for the entire brain. This is the declared reason a working
  graded pathway may still fail to compute motion (§7.7 P4).
* **The release function is linear, unsaturating and undepressing.** Real
  graded synapses, and the photoreceptor terminal in particular, are none of
  those.
* **No photoreceptor physiology.** No light adaptation, no contrast gain
  control, no rhabdomeric nonlinearity, no separate R7/R8 spectral channels.
  A photoreceptor in v4 is a passive membrane driven by an external current.
* **Histamine is a chloride conductance via the coarse sign proxy**, not ort /
  HisCl receptor physiology.
* `E_inh = −70 mV` remains the engineering assumption of §3.1, and `r(V)`
  inherits that status because it is anchored on it.
* **A graded engine is not by itself a demonstration that the connectome
  computes anything.** Whether direction selectivity emerges is empirical, it
  is answered in §7.10, and the honest answer is reported whatever it is.

## 7.10 Predeclared measurements, cheap before expensive

Receipt: `docs/receipts/lif_dynamics_v4.json`.

**(a) Unit level — seconds, CPU, synthetic graphs.**
 a1. A three-neuron graph `0 → 1 → 2` with node 1 declared graded. Node 0 is
 driven so that node 1's membrane stays strictly below −45 mV. Report node 2's
 conductance and membrane deflection under v3 and under v4. Pass for v4:
 node 2 moves; under v3 it must not.
 a2. A graded cell downstream of a spiking cell responds to its spikes.
 a3. **S0 bit-identity**: the full v3 probe A and probe B of §6.8 re-run under
 v4 with an empty graded mask; every array must compare bit-identical to v3.
 a4. Release calibration: a graded cell clamped at `V` delivers the mean
 conductance a v3 spiking cell at `r(V)` Hz delivers, at `V ∈ {−70, −52, −45,
 −20, 0}` mV.

**(b) HS physiology — minutes, GPU, real graph.** The declared
photoreceptor grating of (c), wide-field, in both horizontal directions.
Report HS membrane potential in mV per cell and per eye, relative to the
no-stimulus window, against P3's 0.1–20 mV window and the sign prediction.
Report the HS spike count, which must be exactly zero.

**(c) The real question — minutes to an hour, GPU, real graph.**

*Stimulus.* A drifting sinusoidal grating delivered as `I_ext` to
**R1-R6 only**, resolved by `cell_type == 'R1-R6'` with eye side from
`instance` / `rootSide` (`R1-R6_L` 1,112, `R1-R6_R` 2,265). Retinotopic
position comes from the graph's own wiring plus the released hex annotation:
an R1–R6 carries no `assignedOlHex*` of its own, so its **cartridge** is the
`(assignedOlHex1, assignedOlHex2)` of the L1–L5 cells it synapses onto in the
pinned graph (3,293 of 3,377 contact exactly one cartridge; 42 are assigned by
majority; 42 reach no hex-annotated lamina cell and are **left undriven**).
Hex axial coordinates are mapped to a planar retinotopic frame by the standard
axial-to-cartesian transform `u = h1 + h2/2`, `w = (√3/2)·h2`, scaled by a
declared interommatidial angle of **5°** (engineering assumption; the
*Drosophila* value is about 4.5–5.5° and nothing here depends on its exact
value, since only the ratio of spatial period to spacing matters).

    I(cell, t) = I_max · 0.5 · (1 + contrast · cos(2π(u' cos θ + w' sin θ)/λ − 2π f t))

with `I_max = 20.0` mV-equivalent (taken **unchanged** from the WP5 encoder's
`i_max`, so the comparison is like-for-like and no new drive scale is
introduced), `contrast = 1.0`, `λ = 30°` and `f = 1.5 Hz` (both unchanged from
`optomotor_v3_2.json`), `u'`,`w'` the cartridge position in degrees, and `θ`
the drift direction in the retinotopic plane. **No noise**, so the measurement
is deterministic and bit-reproducible on one backend; there are no seeds and
no bootstrap.

*Protocol.* Eight directions `θ ∈ {0°, 45°, …, 315°}`. Each direction: 500 ms
at contrast 0 (no-stimulus window), then 1,000 ms of grating; the response
window is the **last 500 ms** of the grating. The network starts from rest for
each direction.

*Which lattice axis is "horizontal" is deliberately not assumed.* The eight
directions are defined in the hex lattice's own plane. If the graph computes
direction selectivity, the T4/T5 subtypes' preferred directions will *tell* us
the lattice orientation, and the falsifiable prediction is that T4a and T4b
prefer opposite directions along one axis and T4c/T4d along another
(Maisak *et al.* 2013 assigns a = front-to-back, b = back-to-front,
c = upward, d = downward).

*Reported per population, per eye side, per direction.* Graded populations:
mean membrane potential in the response window minus the no-stimulus window,
in mV. Spiking populations: firing rate in Hz. Populations: `R1-R6`, `L1`–`L5`,
`Mi1`, `Mi4`, `Mi9`, `Tm1`, `Tm2`, `Tm3`, `Tm4`, `Tm9`, `CT1`, `T4a`–`T4d`,
`T5a`–`T5d`, `LPi` (pooled), `HS` (HSN+HSE+HSS+HST), `H2`, `VS`, `DNa02_L`,
`DNa02_R`.

*Direction-selectivity index.* Responses are rectified at zero
(`R⁺ = max(R, 0)`); for each population,
`DSI = (R⁺_pref − R⁺_anti)/(R⁺_pref + R⁺_anti)` where `pref` is the direction
with the largest `R⁺` and `anti` is `pref + 180°`; undefined (reported NaN)
when both are zero. The full eight-point tuning curve and the preferred
direction are reported alongside, and the DSI of a population whose response
is sub-millivolt is reported with that magnitude next to it, because a DSI
computed on noise-level responses is not a finding.

**(d) The preregistered optomotor protocol — only if (c) succeeds.** Run
through master's validation harness (`validation/specs`, a new v4 spec), not
the older Q1/R1/R2 gate, and compared with the v3-2 receipt.

**Declared gate on (d).** (c) must show **both**:
 * `DSI ≥ 0.2` in at least one T4/T5 subtype whose response magnitude is at
   least 0.5 mV, **and**
 * a direction-dependent signal reaching DNa02: `|rate(DNa02_L) −
   rate(DNa02_R)| ≥ 1 Hz` with the sign of the difference following the
   stimulus in both horizontal directions.

**If (c) fails this gate, (d) is not run.** The failure is the result, and the
deliverable is the per-population trace showing where the signal dies and why.
This is a budget rule and an integrity rule at once: it removes the option of
running the expensive protocol and then deciding what to make of it.

## 7.11 References added for v4

* Shiu, P.K. *et al.* (2024). *Nature* 634:210–219. DOI
  [10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9).
  PMC11446845. The scope statement quoted in §7.1 ("does not account for gap
  junctions, non-spiking neurons, internal state or long-range neuropeptides";
  "each neuron identically as a spiking neuron").
* Juusola, M. & Hardie, R.C. (2001). Light adaptation in *Drosophila*
  photoreceptors. *J Gen Physiol* 117:3–25. (Graded photoreceptor responses.)
* Clark, D.A., Bursztyn, L., Horowitz, M.A., Schnitzer, M.J. & Clandinin, T.R.
  (2011). Defining the computational structure of the motion detector in
  *Drosophila*. *Neuron* 70:1165–1177. (L1/L2 graded.)
* Freifeld, L., Clark, D.A., Schnitzer, M.J., Horowitz, M.A. & Clandinin, T.R.
  (2013). GABAergic lateral interactions tune the early stages of visual
  processing in *Drosophila*. *Neuron* 78:1075–1089. (L2 graded voltage.)
* Behnia, R., Clark, D.A., Carter, A.G., Clandinin, T.R. & Desplan, C. (2014).
  Processing properties of ON and OFF pathways for *Drosophila* motion
  detection. *Nature* 512:427–430. (Whole-cell, graded: L1, L2, Mi1, Tm3,
  Tm1, Tm2.)
* Yang, H.H. *et al.* (2016). Subcellular imaging of voltage and calcium
  signals reveals neural processing in vivo. *Cell* 166:245–257 / Yang &
  Clandinin, *Neuron* 92:227 (2016). (Graded medulla signalling.)
* Arenz, A., Drews, M.S., Richter, F.G., Ammer, G. & Borst, A. (2017). The
  temporal tuning of the *Drosophila* motion detectors is determined by the
  dynamics of their input elements. *Curr Biol* 27:929–944.
* Strother, J.A. *et al.* (2017). The emergence of directional selectivity in
  the visual motion pathway of *Drosophila*. *Curr Biol* 27:3132–3146.
  (Mi4/Mi9.)
* Meier, M. & Borst, A. (2019). Extreme compartmentalization in a
  *Drosophila* amacrine cell. *Curr Biol* 29:3277–3284. (CT1 non-spiking.)
* Gruntman, E., Romani, S. & Reiser, M.B. (2018). Simple integration of fast
  excitation and offset, delayed inhibition computes directional selectivity
  in *Drosophila*. *Nat Neurosci* 21:250–257. (T4 whole-cell: graded, no
  spikes.)
* Wienecke, C.F.R., Leong, J.C.S. & Clandinin, T.R. (2018). Linear summation
  underlies direction selectivity in *Drosophila*. *Neuron* 100:1058–1061.
* Joesch, M., Plett, J., Borst, A. & Reiff, D.F. (2008). Response properties
  of motion-sensitive visual interneurons in the lobula plate of *Drosophila
  melanogaster*. *Curr Biol* 18:368–374. (Graded with small irregular
  spikes.)
* Schnell, B. *et al.* (2010). Processing of horizontal optic flow in three
  visual interneurons of the *Drosophila* brain. *J Neurophysiol*
  103:1646–1657. (HSN/HSE/HSS; excited front-to-back, inhibited
  back-to-front.)
* Fujiwara, T., Cruz, T.L., Bohnslav, J.P. & Chiappe, M.E. (2017). A faithful
  internal representation of walking movements in the *Drosophila* visual
  system. *Nat Neurosci* 20. (HS non-spiking during walking.)
* von Reyn, C.R. *et al.* (2017). Feature integration drives probabilistic
  behavior in the *Drosophila* escape response. *Nat Neurosci* 20:1247–1257.
  (LC4 / LPLC2 spike; basis for excluding lobula columnar cells from tier 2.)
* Maisak, M.S. *et al.* (2013). A directional tuning map of *Drosophila*
  elementary motion detectors. *Nature* 500:212–216. (T4/T5 subtype preferred
  directions; here a *prediction to be tested*, not an encoder assumption.)

*Citation status.* The graded/non-spiking claims above are the reason each
class is declared graded. Where this project has read only an abstract, it is
already recorded so in `validation/specs/optomotor_v3_2.json` for Schnell
2010, Joesch 2008 and Fujiwara 2017. The remaining entries in this section are
cited from the published record for the specific property named in the table
and have not been re-read in full text by this session; they are listed so
that a reader can check the declaration against the source, and any that does
not support the property attributed to it invalidates that row of §7.5 rather
than the mechanism of §7.2–§7.4.


---

## 8. References

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


---

## 9. v5 — per-receptor-class synaptic kinetics (new)

**Written before v5 was run on the real graph for anything except a 200 ms
no-stimulus wall-clock cost probe, and before any v5 or v4 measurement under the
protocol of §9.9.** The receptor classes, every time constant and its source,
the charge-preserving quantum, the predictions, the falsifiers and the gate on
the expensive protocol were fixed and hashed first. The locked declaration is

    sha256 3f5996b5c132bd5d0ebb099446761344da11add9dc7340933d105a8269756b6b

written 2026-10-04T16:08:34+02:00, and is reproduced verbatim in
[`receipts/receptor_kinetics_v5_declaration.locked.md`](receipts/receptor_kinetics_v5_declaration.locked.md).
If this section and that file ever differ, **that file is the declaration.**
§1–§8 are unchanged.

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

---

## Correction (5 October 2026): the `E_inh` literature in §3.1 and §6.7

Left unedited above because §3.1 and §6.7 are part of locked declarations. Retrieved
against the primary papers on 5 October 2026 (details and quotations in
`docs/EINH_SENSITIVITY.md` §7):

* **Su & O'Dowd 2003** (−37 ± 3 mV against a theoretical −45 mV) recorded **cultured
  neurons from late-stage pupal central brain** (Kenyon cells identified by GFP), in
  whole-cell mode with a ≈ 24 mM Cl⁻ internal, not adult Kenyon cells in situ.
* **Rohrbough & Broadie 2002**: larval ventral-nerve-cord neurons, primarily motor
  neurons. The abstract says GABA and glutamate responses "reversed near normal
  resting potential". The **−56 ± 3 mV value and the pipette chloride could not be
  verified** (full text not retrievable), so "the measured *Drosophila* larval value"
  in §3.1 and §6.7 (arm S2) is unverified. It is in any case a whole-cell reversal,
  which the pipette sets.
* No paper retrieved reports a **native** adult central `E_Cl`. `E_inh` is
  unconstrained by the literature, so −70 mV (the assumption here), −60 mV and −56 mV
  are all sensitivity points, and none is physiologically privileged.

§3.1 already labels −70 mV an engineering assumption, and that stands. What is
corrected is the standing of the comparison values.
