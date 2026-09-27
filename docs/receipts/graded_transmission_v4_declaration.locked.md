# LOCKED DECLARATION — LIF dynamics v4, hybrid graded/spiking transmission

Locked 2026-09-27, before the v4 engine was run on anything larger than a
two-neuron test graph, before any graded class was resolved against the real
graph, and before any v4 network measurement. This file is the hashed,
verbatim declaration; `docs/LIF_DYNAMICS_SPEC.md` §7 reproduces it and cites
this file's sha256. If the two ever differ, this file is the declaration.

Nothing below was chosen after looking at a v4 result. Where a number is not
derived from an already-declared constant, it is marked as an engineering
assumption in the place where it is used.

---

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

--- END OF LOCKED DECLARATION ---
