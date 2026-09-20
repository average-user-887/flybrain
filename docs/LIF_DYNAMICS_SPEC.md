# Declared LIF dynamics for `brainlab` — v1 (current-based) and v2 (conductance-based)

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

## 5. Predeclared measurements

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

## 6. References

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
