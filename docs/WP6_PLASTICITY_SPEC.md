# WP6 — plasticity model specification (specification and research only)

Written 20 September 2026 against the worktree `neurofly-openready` at `abea3d6`,
on the pinned MaleCNS v1.0 graph (`graph_sha256 4b2f87cc…`, 166,700 neurons,
25,582,938 directed edges). Machine-readable companion:
[`wp6_plastic_subset.json`](wp6_plastic_subset.json).

> **Implementation Update (v0.3.0)**: The specification below was implemented in
> `brainlab/wp6_plasticity.py` (`VisualHeadingPlasticityRule`) and verified by unit tests in
> `tests/test_wp6_plasticity.py` following Rule R1 (§3.2). The historical specification
> text is preserved below.

> **Status on 2026-09-24:** the unit tests show that the rule is applied as specified.
> They say nothing about learning. The only full-graph run is a 1 s smoke test on
> `brainlab-lif-plastic-v1` (`receipts/connectome_closed_loop_wp6_plasticity.json`).
> §1.1 below calls the DNa02 decoder "causally validated" because of WP5, but that
> WP5 claim was withdrawn on 2026-09-20 (v1 artefact, v2 NULL). WP6 cannot be
> evaluated until the v3 optomotor confirmatory run exists (ROADMAP P1).

Every edge count below was measured read-only from `graph.npz` and
`connectome_data/malecns_v1/` on 20 September 2026 (see §9 for the method).
Every citation was retrieved during this session; two are flagged as not
re-verified and one as a preprint. Nothing else is cited.

---

## 0. The one-sentence proposal

Declare the **visually driven ring-neuron → compass-neuron synapses
(ER4d + ER2 → EPG, 3,081 edges, 0.012 % of the graph)** as the plastic subset,
depress them under a two-factor rule gated by the graph's own octopaminergic
`EL` population, keep the rest of the 25.58 M edges fixed, read behaviour out
through WP5's already-frozen DNa02 yaw decoder, and treat the whole thing as a
model hypothesis — not as biology the connectome release handed us.

---

## 1. Candidate learning assays, ranked by what is verifiable in this graph today

The ranking criterion is the one WP6 sets: *are the sensory and the modulatory
mappings actually verifiable in this graph today?* Not *which assay is most
famous*, and not *which would look best on the dashboard*.

State of play for the three real candidates:

| | **A. Visual heading learning (ER→EPG)** | **B. Olfactory conditioning (KC→MBON)** | **C. Optomotor gain (T4/T5→HS→DNa02)** |
|---|---|---|---|
| Plastic site has primary sources | **Yes** — Fisher 2019, Kim 2019, Fisher 2022, Plitt 2025 (preprint) | **Yes, strongest in the fly** — Hige 2015, Aso 2014, Handler 2019, Li 2020, Huang 2024 | **No source found** for experience-dependent plasticity at these synapses |
| Plastic site identifiable in this graph | Yes: 282 ER, 50 EPG, 11,719 edges | Yes: 4,064 KC, 97 MBON, 61,210 edges | Yes, but the bottleneck is 6 (HS→DNa02) + 2 (H2→DNa02) edges |
| Modulatory population identifiable | Yes: `EL` ×18 (predicted octopaminergic), `ExR2` ×4 (predicted dopaminergic) | Yes: `PAM` ×316, `PPL1` ×16, compartment-specific | **No candidate modulator identified** |
| Sensory encoder verified | **No** — ER receptive fields are physiological, not annotated. Assumption, same *class* as WP5's T4/T5 assumption | **No** — no odour model, no verified ORN/glomerulus encoder | **Yes** — pinned `OPTOMOTOR_IO_PIN`, WP5 |
| Motor decoder verified & frozen | **Yes** — EPG→PFL3 (113 edges) →DNa02 (24 edges) → WP5 `DNa02YawDecoder` | **No** — all 97 MBONs together make 376 edges onto all 1,310 DNs; no validated decoder exists | Yes (same decoder) |
| Sign constraint available from the release | Yes: all 282 ER predicted GABAergic → all 11,719 edges negative | Yes: all 4,064 KC predicted cholinergic → all 61,210 edges positive | Yes |
| First-target verdict | **Recommended** | Second; neural-level only, not behavioural, until a decoder exists | Not a learning target |

### 1.1 Recommendation

**Take a visual paradigm first — but the honest visual paradigm is heading-map
learning at ER→EPG, not optomotor gain adaptation.** Reasons:

1. **It is the only candidate where the plastic site, the modulatory population
   and the motor readout are all identifiable in this graph and the readout is
   already frozen and causally validated.** WP5 proved that clamping DNa02
   abolishes the yaw command while leaving the upstream response intact. That
   is the decoder WP6 needs to hold fixed, and EPG reaches it through a short,
   anatomically identified route that the graph contains (EPG→PFL3→DNa02;
   Westeinde 2024 report that DNa02 is the only DN receiving direct PFL3 input,
   and this graph shows exactly 24 PFL3→DNa02 edges and 0 PFL3→DNa01 edges).
2. **The unverified piece is the encoder, and it is the same kind of assumption
   WP5 already declared and got reviewed.** WP5 could not compute direction
   selectivity from photoreceptors in a LIF proxy, so it imposed it at T4/T5 and
   said so. Here we cannot compute a ring neuron's visual receptive field from
   anatomy either, so we impose bar azimuth tuning at ER and say so. One
   declared encoder assumption, not a stack of them.
3. **Odour conditioning needs two unverified mappings, not one.** There is no
   odour encoder and — the harder problem — no verified path from MBON activity
   to a motor command. Across the whole graph, all 97 MBONs make just 376 edges
   (4,181 synaptic contacts) onto all 1,310 descending neurons. Building an
   MBON→motor decoder for this experiment would mean training or hand-tuning a
   decoder in the same experiment that is supposed to show internal plasticity,
   which is precisely the confound WP6 forbids ("freeze the encoder and motor
   decoder so changes in behaviour cannot be explained by decoder training").
4. **Optomotor gain adaptation is the best-verified loop and the worst learning
   model.** No primary source was found for experience-dependent plasticity at
   T4/T5→HS or HS→DNa02, and the whole side-specific signal passes through 8
   edges. A rule applied there is a scalar gain controller with a learning-rule
   name on it. Running it would satisfy the software gate and fail the honesty
   gate.

So: **odour conditioning stays a candidate, as the plan says, and should not be
the first target.** If the plan author prefers the olfactory route anyway, §6.6
gives the honest version of it: a *neural-level* replication of Hige 2015
(compartment-specific depression of the MBON response to a paired odour, with
odour specificity and retention), with no behavioural claim and no decoder.

### 1.2 What is explicitly *not* proposed

Not proposed: plasticity on all 25.58 M edges; plasticity chosen by weight
magnitude or by "edges the task uses"; reward delivered to arbitrary synapses;
any rule applied to an assay whose sensory pathway is unmapped (heat, social,
gap, danger-odour — see `NEUROFLY_RETHINK.md` barrier 5). Those stay
`unmapped` in the WP7 capability matrix.

---

## 2. The plastic subset

### 2.1 Declared subset (primary)

| item | value |
|---|---|
| Presynaptic types | `ER4d` (26 neurons), `ER2_a`, `ER2_b`, `ER2_c`, `ER2_d` (41 neurons) — 67 total |
| Postsynaptic types | `EPG` + `EPGt` — 50 neurons |
| **Plastic edges** | **3,081** |
| Synaptic contacts represented | 40,919 |
| Fraction of the graph | 0.012 % |
| Sign in the prepared graph | 3,081 / 3,081 negative (all 282 ER neurons predicted GABAergic) |
| Permitted change | magnitude only, monotone toward 0; sign flip forbidden |

Wider variant, pre-declared as the *secondary* subset for a sensitivity analysis
only: **all ER→EPG, 282 → 50 neurons, 11,719 edges, 124,264 contacts, 0.046 %
of the graph**, also 100 % inhibitory. Per-ER-type edge counts onto EPG are in
the JSON companion; the largest single contributors are ER4d (1,195 edges),
ER2_c (920), ER3d_b (854), ER3w_b (845), ER5 (781).

### 2.2 Why these edges

* **They are the synapses the primary literature says are remapped by
  experience.** Fisher 2019 showed each compass (E-PG) neuron is inhibited by
  visual cues at particular azimuths, that the map differs between individuals,
  and that visuomotor experience remaps it. Kim 2019 showed correlated compass
  and visual-neuron activity drives the plasticity that converts an arbitrary
  visual scene into a stable heading signal. Fisher 2022 showed dopamine
  released during turning gates it. Plitt 2025 (preprint) localises the
  depression to the ER presynaptic terminal and identifies octopaminergic EL
  neurons as the instructive signal.
* **The release's own annotation supplies the sign constraint.** Every ER
  neuron in this build is predicted GABAergic, so every one of the 3,081 edges
  is negative. Depression of an inhibitory synapse means *less* inhibition of
  the compass at the trained azimuth — a directional prediction the run can be
  scored against, not a free parameter.
* **ER4d and ER2 are the visually driven families.** Ring neurons of the R2/R4d
  classes are the ones shown to carry visual feature and orientation tuning
  (Seelig & Jayaraman 2013), and in this graph they are the ER types with
  substantial TuBu (anterior-optic-tubercle→bulb) input: TuBu→ER4d 50 edges /
  1,323 contacts, TuBu→ER2* 93 edges / 2,610 contacts, out of 568 TuBu→ER edges
  overall. The other ER families (ER3*, ER5, ER1) include sleep- and
  state-related populations and are excluded from the primary subset for that
  reason.
* **They are few enough to audit.** 3,081 edges can be listed, hashed, plotted
  and checked by hand. 25.58 M cannot.

### 2.3 Why not the others

* Not KC→MBON in the first experiment: §1.1 point 3.
* Not EPG→ER (5,201 edges, excitatory) or EPG→EPG (893): no source claims these
  are the learned synapses; they are part of the recurrent attractor and
  changing them changes the dynamics rather than the visual map.
* Not ExR2→EPG / ExR2→ER or EL→ER: these are the *modulatory* inputs. They are
  read as a learning signal, not modified.
* Not T4/T5→HS, HS→DNa02, EPG→PFL3, PFL3→DNa02: these stay fixed so that any
  behavioural change must come from the declared subset. PFL3→DNa02 in
  particular must stay fixed, otherwise the experiment becomes decoder training
  by another name.

### 2.4 The full graph is kept

All 166,700 neurons and all 25,582,938 edges remain loaded and simulated. The
plastic subset is a **declared index array over those edges**, and learning is a
sparse `delta` array added to the released weights — the mechanism
`experiment_registry.GraphInstance` already implements (`_materialize`,
`plastic_edges`, `plastic_delta`). No edge is removed, no subgraph is extracted,
and the graph identity hash is unchanged by learning. The rule's own description
hash goes into the run manifest alongside `graph_sha256` and
`io_map_sha256`, so a run made under a different rule is a different identity.

---

## 3. The rule

### 3.1 Signals

| symbol | meaning | source of the value | units |
|---|---|---|---|
| `s_i(t)` | spikes of presynaptic ER neuron *i* in this control step | graph | spikes / step |
| `a_i(t)` | presynaptic eligibility trace, low-pass of `s_i` | computed | dimensionless, normalised to [0,1] |
| `s_j(t)` | spikes of postsynaptic EPG neuron *j* | graph | spikes / step |
| `p_j(t)` | postsynaptic trace (variant R1b only) | computed | [0,1] |
| `m(t)` | modulatory signal: low-pass population rate of the 18 `EL` neurons, normalised to a declared baseline | graph (primary) or declared encoder (fallback, §5) | [0,1] |
| `g_e` | magnitude of plastic weight on edge *e = (i→j)* | state | graph weight units = synaptic contacts × 0.275 (upstream mV-equivalent) |
| `g_e⁰` | released value of that magnitude | graph, immutable | same |

The sign of every plastic edge is fixed at `−1` (GABAergic). The rule only ever
moves `g_e`.

### 3.2 Update equation (rule **R1**, primary: two-factor presynaptic depression)

Per plastic edge *e = (i→j)*, once per **2 ms control step** (`Δt = 0.002 s`;
the underlying LIF integration step stays 0.1 ms and is **not** changed):

```
a_i   ← a_i + (1 − exp(−Δt/τ_pre)) · ( ŝ_i − a_i )          τ_pre  = 0.20 s
m     ← m   + (1 − exp(−Δt/τ_mod)) · ( m̂_EL − m )           τ_mod  = 0.50 s

Δg_e  = − η · a_i · m · g_e                                  (depression term)
        + (g_e⁰ − g_e) · Δt / τ_rec                          (recovery term)

g_e   ← clip( g_e + Δg_e , g_min · g_e⁰ , g_e⁰ )
w_e   = − g_e                                                (sign never changes)
```

where `ŝ_i` is the step's spike count of neuron *i* divided by a declared
saturating constant `s_sat` (so `ŝ_i ∈ [0,1]`), and `m̂_EL` is the EL population
rate divided by a declared saturating rate `r_sat`.

Variant **R1b** (declared now so it is not invented later, to be chosen *before*
confirmatory runs, not after seeing results): multiply the depression term by
`p_j` as well, giving the three-factor correlational form of Kim 2019 /
Fisher 2019. R1 (presynaptic, postsynaptic activity dispensable) follows
Plitt 2025; R1b follows Kim 2019. **Which one is primary is an open question for
the plan author (§8, Q4).** The two make a testable different prediction —
under R1 the depression is azimuth-specific but compass-phase-independent;
under R1b it requires bump/cue coincidence — and the spec deliberately does not
decide it by trying both and keeping the winner.

### 3.3 Parameters, units, bounds, initial values

| parameter | proposed value | units | status |
|---|---|---|---|
| `Δt` (rule step) | 0.002 | s | matches the WP5 control step; **assumption** |
| `τ_pre` | 0.20 | s | **assumption** (order of a GCaMP-resolvable presynaptic eligibility window) |
| `τ_mod` | 0.50 | s | **assumption**, informed by Fisher 2022 (modulator tracks moment-to-moment rotational speed) |
| `τ_rec` | 60 | s | **assumption** — recovery toward released anatomy; sets the retention horizon (§6) |
| `η` | 0.02 | per step, dimensionless (multiplies `g_e`) | **assumption**, to be fixed by the pilot to give a measurable but non-saturating change within the training budget; fixed before confirmatory runs |
| `g_min` | 0.0 | fraction of `g_e⁰` | depression to silence permitted; **potentiation above the released value is not** |
| `g_max` | 1.0 | fraction of `g_e⁰` | sourced: the reported ER→EPG change is depression |
| `s_sat` | 50 | spikes/s equivalent | **assumption**, re-pinned after the LIF fix |
| `r_sat` | 30 | spikes/s (EL population mean) | **assumption**, re-pinned after the LIF fix |
| initial `g_e` | `g_e⁰` (delta = 0) | — | **the released anatomy is the initial condition.** No pre-training, no random init, no scaling |

Hard invariants, checked every step and asserted in the unit tests WP6 requires:
`sign(w_e)` never changes; `0 ≤ g_e ≤ g_e⁰`; edges outside the declared index
array are bit-identical to the released weights at every step and after every
checkpoint round-trip; with `m ≡ 0` no weight changes at all; with
`learning_enabled = False` no weight changes at all.

### 3.4 The reward / modulatory signal, and the olfactory equivalent

There is no scalar "reward" in R1. The head-direction system's learning signal
is a **when-to-learn** signal, not a good/bad signal: Fisher 2022 showed the
dopaminergic neurons of the head-direction network are active when the fly turns
and their activity scales with rotational speed; Plitt 2025 (preprint) assigns
the instructive role at ER→EPG terminals to octopamine from EL neurons. Both
populations exist in this graph and are identifiable:

| population | neurons | predicted transmitter | edges → ER | edges → EPG |
|---|---|---|---|---|
| `EL` | 18 | octopamine | 4,488 (of 4,514 octopaminergic edges onto ER) | 300 |
| `ExR2` | 4 | dopamine | 495 | 187 |

For the olfactory alternative the equivalent is the classical one: PAM
(appetitive) and PPL1 (aversive) dopaminergic neurons, compartment by
compartment (Aso 2014; Li 2020). This graph contains `PAM` ×316 and `PPL1` ×16;
`PPL101` makes 4,936 edges onto KCs (and, notably, **0** direct edges onto
MBON01), i.e. the release does show the compartmental convergence the rule
would use.

### 3.5 A confound the prepared graph creates, which must be declared

In `graph.npz` every dopaminergic, octopaminergic, serotonergic and
`unclear`-transmitter neuron is mapped to **positive (excitatory)** weight:

```
acetylcholine 103,720   glutamate 29,302   gaba 22,069   histamine 7,891
unclear 2,999   dopamine 392   unassigned 178   octopamine 101   serotonin 48
negative fraction of all 25,582,938 edges: 0.384
```

So EL and ExR2 spikes already act as fast ionotropic excitation onto ER and EPG
in the LIF proxy. If WP6 *also* reads them as `m(t)`, their activity is counted
twice, by two different mechanisms, one of which (fast excitation from an
aminergic neuron) is not what the biology does. Options, to be decided by the
plan author (§8, Q5): (a) keep the excitatory edges and declare the
double-counting in the manifest; (b) declare a new controller version in which
aminergic edges carry zero fast weight and act only through `m(t)`, re-pinning
the graph identity; (c) run both as a sensitivity analysis. This spec's
preference is (b) with (a) as the labelled control, because (b) is the smaller
lie, but it changes the graph identity and therefore needs a decision. The new
`v2` conductance dynamics does not fix this: an aminergic edge is still a fast
excitatory conductance there, just a bounded one.

---

## 4. What the released connectome data does and does not supply

**Supplies** (MaleCNS v1.0, as imported into `connectome_data/malecns_v1/`):

* 166,700 retained neuronal entries, with stable body IDs, and 25,582,938
  directed connections standing for 124,177,617 synaptic contacts. These three
  numbers are not interchangeable and this spec keeps them apart.
* Cell type, instance, `somaSide`/`rootSide`, superclass/class/subclass, and for
  some sensory neurons `receptorType` (`annotations.feather`, 36 columns).
* A **predicted** neurotransmitter per neuron and per cell type, with confidence
  and occasional ground truth (`neurotransmitters.feather`, 1,835,518 rows). In
  the prepared graph every neuron's transmitter is recorded as
  `source_consensus_prediction_or_ground_truth` — a single field that does not
  distinguish the two.
* Contact counts per connection, which our pipeline turns into weights by
  `synapse_count × transmitter_sign × 0.275`.

**Does not supply:**

* **Any learning rule.** Not the site, not the sign of the change, not the
  learning rate, not the time constants, not the eligibility window.
* **Plasticity annotations of any kind.** No edge in the release is marked
  plastic or fixed. The subset in §2 comes from the physiology papers in §7,
  mapped onto the release's cell types by us.
* **Synaptic sign as ground truth.** Sign here is our binary projection of a
  *predicted* transmitter (2,999 neurons are `unclear` and are currently
  projected as excitatory). Glutamate is projected as inhibitory, which is a
  convention, not a measurement.
* **Receptor identity at a synapse.** Handler 2019 show DopR1 and DopR2 direct
  depression versus potentiation at KC→MBON; the release cannot tell us which
  receptor a given synapse carries, so any bidirectional rule's sign assignment
  is ours.
* **Physiological strength.** A contact count is not a conductance. The 0.275
  scale, the 20 mV-equivalent encoder drive and every LIF constant in
  `LIF_DYNAMICS` are engineering.
* **Neuromodulator dynamics** — release, diffusion, volume transmission,
  receptor kinetics, compartment-level dopamine "zones". A modulatory signal in
  this model is a number we compute from spike counts.
* **Behaviour, function, or any within-animal variation.** It is one male
  animal, one fixation, one reconstruction.

Consequently: **anything in §3 that is not in §7 is our hypothesis.** The
release constrains it; it does not supply it.

---

## 5. Engineering assumptions, separated from sourced claims

### 5.1 Sourced (see §7 for the papers)

S1. ER→EPG synapses are a site of experience-dependent plasticity in the heading
system (Fisher 2019; Kim 2019).
S2. Ring neurons of the R2/R4d families carry visual feature/orientation tuning
(Seelig & Jayaraman 2013).
S3. ER→EPG is inhibitory (GABAergic ring neurons; also the release's own
prediction for all 282 ER neurons here).
S4. The plasticity is gated by an activity-dependent modulatory signal tied to
turning: dopamine (Fisher 2022), and at the ER terminal, octopamine from EL
neurons (Plitt 2025 — **preprint, not peer reviewed**).
S5. PFL3 conveys the compass-versus-goal comparison to steering, and DNa02 is
the DN that receives direct PFL3 input (Westeinde 2024; Rayshubskiy 2025).
S6. DNa02 activity predicts and unilateral activation evokes ipsilateral turning
(Rayshubskiy 2025; used already in WP5).
S7. In the mushroom body, coincidence of KC activity with compartment-specific
dopaminergic activity depresses KC→MBON synapses; compartments are the unit
(Hige 2015; Aso 2014; Li 2020); the temporal order matters and two dopamine
receptors direct opposite signs (Handler 2019); the depression is expressed
presynaptically and heterogeneously (eNeuro 2023); memory has interacting short-
and long-term components (Huang 2024, the paper `circuit.py`'s baseline rule
already implements).

### 5.2 Engineering assumptions (ours, not the fly's, not the release's)

A1. **The ER visual encoder.** Bar azimuth → drive on ER4d/ER2 neurons with
assigned preferred azimuths. Receptive fields are not in the release; we assign
them (proposed: evenly spaced preferred azimuths per neuron, fixed by a pinned
RNG, hashed into the IO map). Same class of assumption as WP5's T4/T5
direction selectivity, and it must be pinned the same way.
A2. **Which ER types are "the visual ones"** — ER4d + ER2 here, on the basis of
S2 plus TuBu input in this graph. A defensible cut, not a measurement.
A3. All rule parameters in §3.3 (`η`, `τ_pre`, `τ_mod`, `τ_rec`, `s_sat`,
`r_sat`, `g_min`, `g_max`, the 2 ms rule step).
A4. **The functional form.** Multiplicative depression with linear recovery is
a modelling choice; no paper gives this equation for this synapse.
A5. **Reading `m(t)` from EL spike counts** rather than from simulated
octopamine release, and the fallback of computing `m(t)` from the commanded yaw
magnitude if the graph's EL population turns out to be silent or saturated.
The fallback is the weaker assumption and must be labelled as an *engineered
modulatory signal* in the manifest if used.
A6. **That the LIF proxy can host a heading representation at all.** It has no
reversal potentials, no graded transmission, no adaptation, and (today) a
self-sustained ~10⁶ spikes/s state. This is the biggest assumption in the
document and §8 treats it as a blocker rather than a caveat.
A7. Sign projection of the release's predicted transmitters, including 2,999
`unclear` neurons as excitatory and glutamate as inhibitory.
A8. That an open-loop/closed-loop bar-fixation arena is an adequate stand-in for
the tethered-flight and walking-ball rigs the source papers used.

---

## 6. Evaluation design

Predeclared in full, and to be frozen as a prereg JSON (`docs/wp6_*_prereg.json`,
hashed at declaration) exactly as WP5 did, **before** any confirmatory run.

### 6.1 Frozen components

Frozen for every condition and every seed, hash-pinned in the manifest:

* the ER visual encoder (A1) and its RNG stream,
* the modulatory read-out definition (`m(t)`),
* the **WP5 motor decoder** `DNa02YawDecoder` (`gain 0.02 rad/s/Hz`, `τ 50 ms`,
  `OPTOMOTOR_IO_PIN 228c1b69…66c7b`) — unchanged, untrained, never fitted,
* every non-declared edge in the graph,
* all LIF constants and the graph hash.

Only the 3,081 declared magnitudes move. Any change to a frozen component is a
new controller version with a new identity, not a tweak.

### 6.2 Primary outcome (predeclared)

**Cue-anchoring index `A`** = the resultant-vector length (circular
concentration, 0–1) of the offset between the EPG population activity phase and
the true bar azimuth, measured across *frozen-weight probe blocks* (plasticity
disabled during probes, so the probe measures what was learned, not learning).

Predeclared success: `A_post − A_pre > 0` with a 95 % bootstrap CI excluding 0
in the **plasticity-on** condition, **and** CIs overlapping 0 in the
plasticity-off and yoked conditions, **and** a measured depression concentrated
on edges whose ER preferred azimuth matches the trained bar position.

### 6.3 Secondary outcomes

1. **Behaviour**: closed-loop bar-fixation error (mean |bar azimuth|, degrees)
   in frozen-weight probes, using the frozen decoder. Behaviour is secondary on
   purpose: with the engine in its current state a null here is expected and
   uninformative about the rule.
2. **Weight record**: distribution of `g_e/g_e⁰` over the 3,081 edges, fraction
   changed by >5 %, azimuth-specificity of the change, sign-integrity assertion.
3. **Retention**: `A` re-measured after the retention interval (§6.4).
4. **Held-out**: `A` at a bar azimuth never trained (generalisation / specificity).

WP6's warning is adopted verbatim: a changing weight, on its own, is not
evidence of learning, and neither is an increasing training-time signal.

### 6.4 Schedule, budget, seeds

Per run (one instance, one seed, one condition):

| phase | simulated duration | plasticity |
|---|---|---|
| settle (dark) | 5 s | off |
| pre probe: 3 × bar sweeps | 8 s | **frozen** |
| training: closed-loop bar at the trained azimuth, modulator live | 40 s | **on** |
| post probe | 8 s | **frozen** |
| retention gap (dark) | 30 s | off |
| retention probe | 8 s | **frozen** |
| held-out probe (untrained azimuth) | 7 s | **frozen** |
| **total** | **106 simulated s** | |

Seeds: 0–4 (five), fixed in the prereg. Training budget: 40 simulated s per run,
fixed; it is not extended because a condition failed to learn.

### 6.5 Comparison conditions

1. **plasticity-on** — the declared rule, live modulator.
2. **plasticity-off** — identical in every other way (`learning_enabled=False`);
   identical RNG stream, identical stimulus schedule.
3. **yoked / shuffled reward** — the `m(t)` time series recorded from a
   plasticity-on run of a *different seed*, replayed here, preserving its
   distribution and destroying its contingency with this run's activity.
4. **pathway intervention** — the 18 `EL` neurons clamped (Kir-like, the WP5
   `SILENCE_DRIVE = −200` mechanism). Removes the instructive signal while
   leaving the sensory pathway intact. Predicted: behaves like plasticity-off.
5. *(if budget allows)* **degree-preserving shuffled graph**, as in WP5,
   labelled a control graph.

All five share initial state, exposure budget and stimulus schedule. Training
and frozen-weight testing are separated in time, never interleaved.

### 6.6 The olfactory fallback design, if the plan author prefers it

Not behavioural. Plastic subset: KC→MBON01 (2,109 edges, 40,888 contacts) in the
γ1pedc compartment; modulator: `PPL101` (2 neurons, 4,936 edges onto KCs).
Outcome: depression of the MBON01 response to the paired odour relative to an
unpaired odour, measured at frozen weights — the measurement Hige 2015 made.
Conditions: paired, unpaired (odour and DAN activation separated in time),
reverse-order (backward pairing; Handler 2019 predicts a different sign), DAN
silenced. No motor decoder is involved and no behavioural claim is made. This is
cheaper and better-sourced than the visual design, and it is **not** a learning
demonstration in the embodied sense WP6 is aiming at.

### 6.7 What counts as a NULL result

Declared in advance, and reported as a result rather than a reason to re-tune:

* `A_post − A_pre` CI includes 0 in the plasticity-on condition → **null**: the
  declared rule does not produce cue anchoring in this model. Retained,
  published in the capability matrix, and the assay stays `learning evaluated —
  null`.
* Weights change but `A` does not → **null**, and explicitly labelled as the
  case WP6 warns about.
* `A` improves identically in plasticity-off or yoked → **null**, and evidence
  that the change was stimulus exposure or engine drift, not the rule.
* The engine's runaway state makes `A` unmeasurable (no identifiable EPG phase)
  → **not a null but a blocked run**: reported as *not evaluable on this engine
  version*, and the assay stays `integrated`, not `learning evaluated`.
* A positive result is **not** a claim that the fly learns this way. It is:
  *with this declared rule on these 3,081 declared edges in this LIF proxy, the
  measured index moved and the three controls did not.*

---

## 7. Feasibility on this machine

Measured costs, from WP5 §7 on this hardware (single core, whole loop, 2 ms
control steps): **0.106 simulated s per wall s** for the intact graph,
0.111 silenced, 5.55 for a silent (sham) graph; peak RSS 826 MiB with one graph
plus a shuffled array, **≈590–600 MiB for one graph instance**.

Rule overhead: 3,081 edges updated per 2 ms step = 1.54 M scalar updates per
simulated second, against ~10⁶ spikes/s propagating through 25.6 M edges. Below
1 % — but it will be measured in the pilot, not assumed.

Memory overhead: `plastic_delta` 3,081 × float32 = 12 KiB; traces ~1 KiB;
checkpoint delta < 20 KiB. Negligible. One instance at a time, as WP4 requires.

**Cost of the design in §6.4–6.5:**

| | simulated s | wall s at 0.106 | |
|---|---:|---:|---|
| one run | 106 | 1,000 | ≈ 16.7 min |
| 4 conditions × 5 seeds | 2,120 | 20,000 | **≈ 5.6 h** |
| + condition 5 (shuffled graph) × 5 | +530 | +4,015 | ≈ 6.7 h total |
| + pilot (1 seed × 2 conditions) | 212 | 2,000 | ≈ 0.6 h |
| **total** | | | **≈ 6–7.3 wall hours** |

**Verdict: feasible.** It is an overnight, single-core, one-instance-at-a-time
campaign on this machine, with ~0.9 GiB peak RSS, and it fits the pattern WP5
already ran (32 m 54 s for its confirmatory set).

**What is *not* feasible**, stated plainly so it is not attempted: the
"complete" version of this design — 6 conditions × 10 seeds × 300 simulated s —
costs 18,000 simulated s ≈ **47 wall hours** of continuous single-core compute,
and that is before any repeat. It should not be scheduled. Likewise, running
this assay live in the dashboard at 1× is impossible: 0.106× real time means the
UI must report the achieved rate rather than pretend (WP5 §8).

Two things could change the number, in opposite directions, and the pilot must
re-measure rather than extrapolate:

* The LIF fix should make it **much cheaper** if it removes the self-sustained
  ~10⁶ spikes/s state — the silent-graph figure (5.55 sim s/wall s) shows cost
  is activity-dominated, so a physiological rate could plausibly buy back a
  large factor. Do not bank on it in the prereg.
* Closed-loop training with a live modulator may raise network activity and cost
  more than the WP5 open-loop measurement.
* The new conductance-based `v2` dynamics (§8.1) costs more per step than `v1`
  (an extra conductance array and a division per active neuron) but should fire
  far less. Net direction unknown; the pilot must measure it under `v2` before
  the prereg is sealed. Every number in the table above is a `v1` extrapolation.

**Smaller honest design, if the pilot comes in worse than 0.05 sim s/wall s:**
drop condition 5, cut seeds to 3, and cut training to 25 s and probes to 6 s
(70 simulated s/run) → 12 runs × 22 min ≈ 4.4 h. Below three seeds the
uncertainty on `A` is not worth reporting, and the correct decision is then to
run the olfactory neural-level fallback (§6.6), which needs no arena and no
closed loop.

---

## 8. Open questions and dependencies

### 8.1 Blocking dependency: the LIF dynamics fix

Another agent is repairing `brainlab/engine.py` now, and work landed while this
spec was being written: `brainlab/graph_identity.py` now declares two versioned
dynamics (`LIF_DYNAMICS_V1`, `LIF_DYNAMICS_V2`) with `docs/LIF_DYNAMICS_SPEC.md`
as their specification, and `engine.advance_v2` is conductance-based with
reversal potentials (`E_EXC_MV = 0.0`, `E_INH_MV = −70.0`, membrane bounded,
shunting inhibition). The default is still `v1`, selected by
`NEUROFLY_LIF_DYNAMICS`. **WP6 should be specified and run against `v2`**, and
the dynamics version belongs in the WP6 manifest and prereg next to the graph
hash — v1 and v2 are different controllers and a learning result under one says
nothing about the other. This spec was written before any v2 measurement
existed, so every rate-dependent constant in §3.3 (`s_sat`, `r_sat`, `η`) is
provisional until the pilot re-measures them under v2.

WP5 §6 documented the two properties of v1 that would wreck any learning
measurement made on top of them, and which v2 is meant to address:

1. **Runaway.** After the first stimulus the graph enters a self-sustained
   ~10⁶ spikes/s state (≈6 Hz per neuron) that persists through "gray" periods,
   with membrane potentials at −200 mV because the proxy has no reversal
   potentials. A rule whose eligibility trace is a low-pass of spike counts
   would be driven by this state, not by the stimulus, and `τ_pre`/`s_sat` fixed
   against today's rates would be wrong tomorrow.
2. **One dead side.** DNa02_R does not respond; rightward rotation produces
   4.0 vs 3.7 Hz. A heading experiment on a network with one non-responsive
   hemisphere cannot produce a symmetric cue-anchoring measurement, and any
   apparent learning would be confounded with the asymmetry.

**Therefore: no WP6 confirmatory run before the engine fix lands, is re-pinned,
and is shown to have (a) a quiet baseline between stimuli and (b) bilateral
responsiveness — demonstrated by re-running the WP5 optomotor confirmatory set
under `v2`, not asserted.** The pilot and the rule's unit tests (small-circuit
verification, bounds, no-update conditions, checkpoint recovery) can be written
and run against synthetic test graphs in the meantime — they do not depend on
the fix. `s_sat`, `r_sat` and `η` must be re-pinned after the fix, in the pilot,
before the prereg is sealed.

### 8.2 Questions for the plan author

**Q1.** Do you accept a **neural** primary outcome (cue-anchoring index) with
behaviour as a secondary? A behavioural primary on today's engine would almost
certainly return a null that says more about the engine than about the rule.

**Q2.** Is ER→EPG the right first target, or do you want the olfactory
neural-level design (§6.6) first because its sources are stronger? They are not
mutually exclusive; the visual one is the one with a frozen decoder.

**Q3.** Is an **assumed ER visual encoder** (A1) acceptable, given that it is
the same class of assumption WP5 declared at T4/T5? If not, WP6 has no visual
learning target at all, because the LIF proxy cannot derive ring-neuron
receptive fields from photoreceptors.

**Q4.** R1 (presynaptic two-factor, Plitt 2025 preprint) or R1b (three-factor
correlational, Kim 2019) as the **primary** rule? A preprint is the most direct
source for R1; does a preprint count as a primary source for a released
artifact? Choosing after seeing results is not an option.

**Q5.** The aminergic-excitation confound (§3.5): keep the excitatory EL/ExR2
edges and declare the double-counting, or declare a new controller version in
which aminergic edges carry no fast weight (changing the graph identity)?

**Q6.** Ownership and sequencing: `brainlab/**` is being edited by another agent
right now, and the rule needs a new module plus registration through
`ExperimentRegistry(plasticity_rule=…)`. Who lands it, and after which commit?

**Q7.** Checkpoint identity: registering a rule adds `dynamics.plasticity_rule`
to the manifest (`experiment_registry.py:310`) and therefore changes the run
identity. Confirm that existing WP4/WP5 `connectome-fixed` checkpoints should
keep working unchanged and that plastic checkpoints are a separate lineage.

**Q8.** Retention horizon: `τ_rec = 60 s` makes the retention interval
meaningful within an affordable run. Is a longer, more biological horizon wanted
at the cost of an unaffordable evaluation?

---

## 9. Method for the numbers in this document

Read-only, from a scratch directory outside the checkout, using the worktree's
`.venv`: `connectome_data/malecns_v1/normalized/neurons.feather` (cell type,
predicted transmitter) joined to `annotations.feather` (`somaSide`, `instance`)
on `source_id`/`bodyId`, and the CSR arrays of
`outputs/brainlab/malecns_v1/graph.npz` (`ptr`, `post`, `weight`). Edges between
two type sets were counted by iterating the source rows of the CSR and masking
targets; "synaptic contacts" is `|weight| / 0.275`, the inverse of the prepared
graph's scale. Nothing was written into the root checkout and no service or
process was touched.

Counts recorded: ER 282 (26 types), EPG+EPGt 50, ExR 26 (ExR2 ×4), EL 18,
TuBu 156, PFL3 24, PFL2 12, PFL1 14, PEN/PEG 60, Delta7 46, KC 4,064,
MBON 97 (37 types), PAM 316, PPL1 16, T4 6,865, T5 6,720, HS 6, H2 2,
DNa02 2, DNa01 2.

---

## 10. Sources

Retrieved during this session (20 September 2026) unless marked otherwise.

1. **Fisher YE, Lu J, D'Alessandro I, Wilson RI (2019)** Sensorimotor experience
   remaps visual input to a heading-direction network. *Nature* 576:121–125.
   doi:10.1038/s41586-019-1772-4
2. **Kim SS, Hermundstad AM, Romani S, Abbott LF, Jayaraman V (2019)**
   Generation of stable heading representations in diverse visual scenes.
   *Nature* 576:126–131. doi:10.1038/s41586-019-1767-1
3. **Fisher YE, Marquis M, D'Alessandro I, Wilson RI (2022)** Dopamine promotes
   head direction plasticity during orienting movements. *Nature* 612:316–322.
   doi:10.1038/s41586-022-05485-4 (an Author Correction exists: PMC10247362)
4. **Plitt MH, Turner-Evans DB, Co JC, Layden A, Eddison M, Ray RP, Jayaraman V,
   Fisher YE (2025)** Octopamine instructs head direction plasticity. *bioRxiv*
   preprint, 15 December 2025. doi:10.64898/2025.12.11.693783 —
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12724720/
   **PREPRINT — not peer reviewed.** Its two-factor, postsynaptically
   independent claim is the basis of rule R1 and is the least settled source here.
5. **Seelig JD, Jayaraman V (2013)** Feature detection and orientation tuning in
   the *Drosophila* central complex. *Nature* 503:262–266. doi:10.1038/nature12601
6. **Hulse BK, Haberkern H, Franconville R, Turner-Evans DB, Takemura S, et al.
   (2021)** A connectome of the *Drosophila* central complex reveals network
   motifs suitable for flexible navigation and context-dependent action
   selection. *eLife* 10:e66039. doi:10.7554/eLife.66039
7. **Westeinde EA, et al., Wilson RI (2024)** Transforming a head direction
   signal into a goal-oriented steering command. *Nature* 626:819–826.
   doi:10.1038/s41586-024-07039-2 (Author Correction: doi:10.1038/s41586-024-08245-8)
8. **Rayshubskiy A, et al. (2025)** Neural circuit mechanisms for steering
   control in walking *Drosophila*. *eLife* 13:RP102230.
   doi:10.7554/eLife.102230 — the peer-reviewed version of the bioRxiv preprint
   (2020.04.04.024703) cited in `brainlab/io_map.py` and `docs/WP5_OPTOMOTOR.md`.
   **Suggest updating those citations to this version.**
9. **Hige T, Aso Y, Modi MN, Rubin GM, Turner GC (2015)** Heterosynaptic
   plasticity underlies aversive olfactory learning in *Drosophila*. *Neuron*
   88(5):985–998. PMID 26637800.
   https://www.sciencedirect.com/science/article/pii/S0896627315009824
   (DOI not separately retrieved; PMID and publisher URL verified.)
10. **Aso Y, et al. (2014)** The neuronal architecture of the mushroom body
    provides a logic for associative learning. *eLife* 3:e04577.
    doi:10.7554/eLife.04577
11. **Li F, Lindsey JW, Marin EC, et al., Rubin GM (2020)** The connectome of the
    adult *Drosophila* mushroom body provides insights into function. *eLife*
    9:e62576. doi:10.7554/eLife.62576
12. **Handler A, et al. (2019)** Distinct dopamine receptor pathways underlie the
    temporal sensitivity of associative learning. *Cell* 178(1):60–75.e19.
    PMID 31230716. (Citation confirmed via the FlyBase reference report
    FBrf0242767; DOI not separately retrieved.)
13. **Huang C, Luo J, et al. (2024)** Dopamine-mediated interactions between
    short- and long-term memory dynamics. *Nature* 634:1141–1149.
    doi:10.1038/s41586-024-07819-w — **this is the paper `circuit.py` cites for
    its baseline mushroom-body rule, and the citation checks out.**
14. *Dopamine-Dependent Plasticity Is Heterogeneously Expressed by Presynaptic
    Calcium Activity across Individual Boutons of the Drosophila Mushroom Body.*
    *eNeuro* 10(10):ENEURO.0275-23.2023.
    https://www.eneuro.org/content/10/10/ENEURO.0275-23.2023 — **author list not
    verified in this session**; cited only for the presynaptic-expression point.
15. **Ofstad TA, Zuker CS, Reiser MB (2011)** Visual place learning in
    *Drosophila melanogaster*. *Nature* 474:204–207. (DOI not separately
    retrieved.) Cited only as evidence that visual place learning is a real
    behaviour in this animal.
16. **MaleCNS v1.0.** *Sexual dimorphism in the complete connectome of the
    Drosophila male central nervous system.* bioRxiv 2025.10.09.680999 (posted
    9 October 2025); data portal https://male-cns.janelia.org/. **Preprint.**
    Cited for what the release contains.
17. **Maisak MS, et al. (2013)** A directional tuning map of *Drosophila*
    photoreceptor pathways. *Nature* 500:212–216. — **not re-verified in this
    session**; carried over from `brainlab/io_map.py`, where it supports the
    T4/T5 subtype assignment. Anyone quoting it should re-check it.

Claims I could not source, and which are therefore **assumptions** rather than
citations: any plasticity at T4/T5→HS/H2 or HS→DNa02; any quantitative learning
rate, eligibility-trace time constant or bound for ER→EPG in physical units; and
any statement that the LIF proxy's dynamics resemble the fly's.
