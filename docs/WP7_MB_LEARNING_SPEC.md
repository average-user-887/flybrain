# WP7 — mushroom-body odour learning: frozen specification (P4)

Written 25 September 2026 on branch `wp7-draft`, from `master` at `4264226`,
against the pinned MaleCNS v1.0 graph (`graph_sha256 4b2f87cc5c92fa2a…`,
166,700 neurons, 25,582,938 directed edges). Design and literature:
[`design/mb_learning.md`](design/mb_learning.md) (PR #17), with its census
[`design/mb_circuit_census.json`](design/mb_circuit_census.json). Machine-readable
preregistration drafts:

- [`validation/specs/mb_e0_kc_regime_v3.json`](../validation/specs/mb_e0_kc_regime_v3.json) (E0)
- [`validation/specs/mb_e1_hige2015_v3.json`](../validation/specs/mb_e1_hige2015_v3.json) (E1)

> **Status: FROZEN, pending the owner gate. No real-graph simulation has been
> run under this spec.** Every number that depends on a run is a plan. The
> gates, thresholds, seeds, conditions and parameters below were fixed before
> any E0 or E1 output existed. Results will be recorded as they come out,
> including failures, **without retuning**. The one value left open is
> η_γ1pedc, and the procedure that sets it (§6.2.3) is frozen too.

What this spec does not claim:

- It does not claim that the model learns. It says how learning will be
  tested and what counts as a pass, a fail and a null.
- Anything tagged ASSUMPTION or NOT VERIFIED below is a modelling choice, not
  biology.

---

## 0. The proposal in one paragraph

On the v3 graph, KC→KC synapses get no fast weight (D3) and DPM is relabelled
GABAergic (D5). That is the **P4 primary graph**. On it, the KC→MBON synapses
in the γ1pedc and γ2α′1 compartments become plastic under **MB-R1**:
dopamine-gated depression of KC synapses that were recently active, driven by
the PPL1 spikes of the same compartment and hemisphere. η for γ2α′1 is
0.25 × η_γ1pedc. The unconditioned stimulus (US) is current injected into the
PPL1 neurons, as in an optogenetic experiment (D4). Two experiments come first:

- **E0**, without plasticity, checks that KC odour coding is sparse and
  odour-specific on the primary graph. Four comparison graphs are run with it.
- **E1** reproduces Hige 2015 at KC→MBON11. η_γ1pedc is calibrated once, on 5
  calibration seeds, to Hige's 80 % depression. The test then runs on 10 fresh
  confirmatory seeds against five null controls.

The behavioural T-maze experiment (E2) is declared in §6.3. It is gated
behind E1 and behind the P3 naive T-maze gate, and it gets its own
preregistered spec later.

---

## 1. Freeze and recording rules

1. **What is frozen.** Every field of the two spec JSONs except `status`,
   `status_reason`, `freeze` and `calibration.result`.
   - `freeze.frozen_content_sha256` is the sha256 of the canonical JSON
     (`sort_keys`, compact separators, UTF-8) of the spec with those fields
     removed.
   - `tests/test_wp7_graph_policy.py::test_wp7_specs_are_frozen` recomputes
     it, so any edit fails CI. It also checks the sha256 of the T-maze spec
     that the encoder is taken from.
   - E0: `872875ed24a3165475343f9e73e7ab782ec051dcf3a0b84932b83cfc03ebf455`.
   - E1: `5f45ba6f50dffa736a0ad77d81d21eaaf6bc4de5416724f032e4aa56ef033401`.
2. **Owner gate.** Only the owner changes `status` from `draft` to
   `preregistered`, in a commit that changes nothing else. That happens only
   after all three of these hold:
   - (a) the Phase 1 gate is signed off;
   - (b) implementation items I1–I4 (§8) are merged, with their synthetic
     plumbing tests passing;
   - (c) the sidecar files of §2.3 and §4 exist and are hash-recorded.
   A run on a `draft` spec is `EXPLORATORY` by the harness rule, so it cannot
   produce a confirmatory verdict.
3. **Calibration.** η_γ1pedc is written into `calibration.result` by one
   commit, before any confirmatory seed runs. That commit contains the
   calibration receipt path and nothing else.
4. **Recording.** Every receipt is committed as it comes out, pass or fail,
   under `docs/receipts/validation/wp7-*.md`, in the format of
   `optomotor-yaw-v3-1.md`. That includes E0 on all five variants and every
   calibration attempt.
5. **Amendments.** If anything needs to change after a real-graph run, the
   change goes into a new spec id (`-2`) with an `amendment` block, following
   `optomotor_v3_2.json`. That block says what changed, what prompted it and
   whether the result was already known. The earlier verdict stays on record.
   No parameter, threshold, seed or condition is retuned after seeing output.

---

## 2. Graph identity

### 2.1 Base graph

- The pinned prepared graph: `graph_sha256
  4b2f87cc5c92fa2aa4a3879ac520d55edffe3e577d8828bfd054e12800091d01` and
  `neuron_map_sha256 2103b520…` (`brainlab/graph_pins.json`,
  `docs/receipts/graph_identity.json`).
- Each weight is `count(float32) · sign(int8) · 0.275` (`brainlab/prepare.py`).
  The DPM neurons' predicted transmitter is dopamine, which the prepared
  graph stored with the ambiguous sign +1.
- LIF dynamics are v3 (`brainlab/engine.py::advance_v3`). The transmitter
  policy is `v3-modulatory-only` with `unclear_mode='excitatory'`, the v3
  default: every dopaminergic, octopaminergic and serotonergic out-edge gets
  weight 0.

### 2.2 The five graph variants

The options are declared in `brainlab/transmitter_policy.py` (§2.5):

| variant id (spec `graph_variants`) | `kc_kc` | `dpm` | role |
|---|---|---|---|
| **V2_primary** (= the spec's own `dynamics`) | `modulatory-only` | `gaba` | **P4 primary graph**; gated in E0; the only graph used in E1/E2 |
| V0_released | `released` | `released` | plain v3 (KC→KC as released, DPM silent); reported |
| V1_kc0_dpm_silent | `modulatory-only` | `released` | reported |
| V3_kckc_released_dpm_gaba | `released` | `gaba` | reported |
| V4_calyx_kept_dpm_gaba | `modulatory-only-except-calyx` | `gaba` | reported; blocked until the calyx sidecar exists (§2.3) |

`dpm='released'` leaves DPM under its released label. Under v3 that label is
dopamine, so its output is **silent**. The policy report lists
`dpm_released_labels` whenever `dpm='gaba'`, so the silent reading of
`released` is checked on every run rather than assumed. (The census records
both DPM neurons as `dopamine`.)

**Identity.** `apply_to_shared` computes a new `graph_sha256` over
`ptr, post, weight, ids` for every variant, so different weights give a
different hash. The dataset name gains a suffix, for example
`malecns_v1+v3-modulatory-only(excitatory)+mb(kc_kc=modulatory-only,dpm=gaba)`.
With the calyx option it also carries the first 12 hex digits of the sidecar
sha256. The label names the variant.

- The `v3-modulatory-only` substring that `GraphInstance` checks is kept.
- A checkpoint written on one variant is refused on another (registry
  `graph_sha256` check).
- The variant hashes need the real graph, so they are recorded at the first
  load on the GPU host (§9 step 0), before any simulation. They are not
  computed here.

### 2.3 D3: KC→KC fast weight

- **Query.** A KC is a neuron whose released `type` matches `^KC`: 4,064
  neurons. KC→KC means both endpoints are KCs: 642,933 engine edges and
  1,153,845 contacts (census `apl.kc_to_kc`).
- **`modulatory-only`.** Every KC→KC edge gets weight 0. Nothing else
  changes: KC→MBON, KC→APL, KC→DPM, KC→DAN, APL→KC and PN→KC stay as
  released. The report gives `kc_kc_edges` and `kc_kc_edges_zeroed`. On the
  real graph both must equal 642,933, otherwise the E0 run stops (paradigm
  check, I3).
- **`modulatory-only-except-calyx`.** A KC→KC edge keeps
  `c_calyx · 0.275 · sign`, computed with the same float32/int8 arithmetic as
  `prepare.py`, where `c_calyx` is its number of synapses whose
  **postsynaptic** point lies in a calyx ROI. Every other KC→KC edge gets 0.
- **The runtime graph cannot support the calyx option by itself.**
  `graph.npz` holds one weight per neuron pair and no synapse ROI.
  - The calyx split needs the released synapse tables that the census already
    uses for compartments:
    `syn-partners-male-cns-v1.0-minconf-0.5.feather` (sha256 `959d8ef4…`)
    and `syn-points-male-cns-v1.0-minconf-0.5.feather` (sha256 `c16b1b63…`).
  - **What is implemented** is the application side:
    `apply_policy(kc_kc_calyx=(edges, counts))`,
    `calyx_edges_from_pairs` and `load_kc_kc_calyx`. These check that every
    listed pair is an existing KC→KC edge, that it is listed once, and that its
    count does not exceed the edge's released contact count.
  - **What is not implemented** is the generator (item I1), and so no calyx
    sidecar exists yet.
- **Calyx sidecar format** (`data-provenance/malecns_v1/derived/wp7_kc_kc_calyx.npz`,
  written with `np.savez_compressed`, no pickle):
  - `pre_index`, `post_index`: prepared-graph node indices (int64);
  - `calyx_synapses`: int64;
  - `table_synapses`: all KC→KC synapses of that pair in the synapse table
    (int64);
  - `roi_names`: the ROI strings counted as calyx (unicode).
- **Generator rules (I1):**
  1. A synapse is placed by its postsynaptic point: the census join on body
     and coordinates, `kind='PostSyn'`.
  2. Calyx means the released **primary** ROI matches `^(CA|ACA)\((L|R)\)$`.
     The exact MaleCNS ROI names are NOT VERIFIED. The generator prints every
     primary ROI name that holds KC→KC postsynaptic points, and it stops if
     none matches.
  3. The generator stops if any `calyx_synapses` exceeds the released edge
     count. The loader refuses such a file anyway.
  4. The generator reports the number of edges where `table_synapses` differs
     from the released count.
  5. The sidecar's sha256 goes into the E0 receipt and the graph identity.

### 2.4 D5: DPM transmitter

- **Query.** `type` matches `^DPM$`: 2 neurons, one per side, predicted
  dopamine.
- **`gaba`.** DPM is treated as GABAergic. It is removed from the modulatory
  set, and each of its out-edges gets `-|w_released|`, which is exactly
  `count · (−1) · 0.275`. The released contact counts are kept (DPM→KC: 4,082
  edges, 32,795 contacts).
- Its serotonin is not modelled, like every other aminergic signal. Fast
  GABA kinetics are an approximation (design §6 D5 caveat).
- **Precedence.** The `unclear_mode` drops (`zero` and `exclude`) apply after
  both options and win over them. No neuron that WP7 touches is `unclear`, so
  this matters only for edges onto an excluded neuron.

### 2.5 Code (implemented here)

**`brainlab/transmitter_policy.py`**:

- `KC_KC_MODES = ('released', 'modulatory-only',
  'modulatory-only-except-calyx')` and `DPM_MODES = ('released', 'gaba')`. The
  first entry of each is the default.
- `apply_policy(..., kc_kc=, dpm=, cell_types=, kc_kc_calyx=)`.
- `apply_to_shared(..., kc_kc=, dpm=, kc_kc_calyx_file=)`, which loads the cell
  types from `normalized/neurons.feather` (`cell_type`) only when an option is
  non-default.
- `describe(kc_kc=, dpm=)`, `graph_variant_tag`, `load_cell_types`,
  `calyx_edges_from_pairs`, `load_kc_kc_calyx`.

The options are refused under the legacy policy, and unknown values are
refused.

**Defaults are unchanged.** With the default options:

- the weights are byte-identical to the previous v3 policy;
- the report and `describe()` output are identical (no new keys);
- the dataset name and label are identical, so `graph_sha256` is too.

`brainlab/brain.py`, the daemon, the engine and every existing spec call the
policy with defaults and are therefore unaffected.

**`validation/harness.py`**:

- `load_graph` passes `dynamics.kc_kc`, `dynamics.dpm` and
  `dynamics.kc_kc_calyx_file` to the policy. They are absent from every
  existing spec, so those specs behave as before.
- `apply_graph_variant` and `python -m validation run --graph-variant ID`
  select one of a spec's **declared** `graph_variants`. An undeclared id is
  refused. The receipt records `graph.variant`.
- `paradigm_module` raises a clear `SpecError` for the two declared but
  unimplemented WP7 paradigms. `run()` checks this before it creates any
  output.

**Tests**: `tests/test_wp7_graph_policy.py` (§10).

---

## 3. The plasticity rule

### 3.1 State and initial value

The state is one conductance magnitude `g_ijk ≥ 0` for every plastic triple:
KC *i* → MBON *j* in compartment *k*. The triples are listed in §4.

The released value is set so that the weights at t = 0 are **exactly** the
released weights:

```
g⁰_ijk  = w_ij · n_ijk / N_ij          n_ijk: synapse-table synapses of (i→j) in k (plastic compartments only)
w_fix,ij = w_ij · (1 − Σ_k n_ijk / N_ij)  N_ij : all synapse-table synapses of (i→j)
w_ij(t)  = w_fix,ij + Σ_k g_ijk(t)
```

Here w_ij is the P4-primary weight of the engine edge (positive: KCs are
cholinergic).

- When the synapse table and the released edge count agree, this is the
  design's `0.275 × synapses`.
- When they disagree, it keeps the released total rather than inventing
  weight.
- The engine receives `plastic_delta_ij = Σ_k (g_ijk − g⁰_ijk)`, which is
  exactly 0 at t = 0 (`GraphInstance`).

### 3.2 MB-R1 (primary rule, every plastic compartment)

The rule runs once per Δt = 2 ms of simulated time, on the spike counts summed
over that window. The LIF step stays 0.1 ms. The order inside one update is
fixed:

```
1. e_i    ← e_i · exp(−Δt/τ_e) + s_i                        every KC i in the plastic set
2. d_k,h  = Σ spikes in this window of the DANs modulating k on hemisphere h
3. g_ijk  ← g_ijk − η_k · min(1, e_i / e_sat) · d_k,h · g_ijk  depression, KC eligibility × dopamine
4. g_ijk  ← g_ijk + (g⁰_ijk − g_ijk) · Δt / τ_rec              recovery toward g⁰
5. g_ijk  ← clip(g_ijk, g_min · g⁰_ijk, g⁰_ijk)
6. plastic_delta_ij = Σ_k (g_ijk − g⁰_ijk); push the changed edges (Brain.update_weights)
```

- **Precision.** Arithmetic is in float64 on the host. The delta is stored as
  float32, as `GraphInstance.plastic_delta` is.
- **Upload.** An update uploads only the edges whose delta changed, which is
  exact. Step 4 changes an edge only while it is below g⁰.
- **Hemisphere h.** h is the side suffix of the synapse's released ROI (for
  example `g1(R)`). The census strips that suffix, and the sidecar must keep
  it (§4).
- **Which DAN counts for h.** For compartment k, the modulating DAN on side h
  is the neuron of the declared type (below) whose DAN→KC synapses lie mostly
  in ROIs with suffix h. The generator records this mapping, and it must be
  one neuron per type and side.
- **Order.** Dopamine that arrives before any KC activity meets e_i ≈ 0
  (only spontaneous KC spikes), so backward pairing does almost nothing. This
  is intended (Hige 2015).

### 3.3 Parameters

The source tags are the design doc's: READ, ABSTRACT, NOT VERIFIED, plus
ASSUMPTION and CALIBRATED.

| parameter | value | tag | basis |
|---|---|---|---|
| Δt (rule step) | 2 ms | engineering | harness control step |
| τ_e | 1.0 s | ASSUMPTION | Handler 2019 READ (+0.5 s depresses, +6 s minimal); Hige 2015 READ (a pulse 0.8 s after onset works) |
| e_sat | 5 spikes | ASSUMPTION | Honegger 2011 READ (5–10 spikes per odour); Turner 2008 READ (2.2–4.9 spikes in 2 s) |
| η_γ1pedc (g1, PED) | calibrated, per DAN spike | CALIBRATED | §6.2.3, on E1 calibration seeds only |
| η_γ2α′1 (g2, a′1) | 0.25 × η_γ1pedc | ASSUMPTION (D1) | Aso 2012 READ: MV1 memory "slight but significant" |
| τ_rec | 3 h (10,800 s) | ASSUMPTION | Hige 2015 READ (little recovery in 40 min); Aso 2012 READ (MP1 memory decays over ~9 h) |
| g_min | 0 × g⁰ | READ | Hige 2015: 90 % charge-transfer reduction |
| modulators | g1, PED: PPL101 + PPL102; g2, a′1: PPL103 | design D1, §2.5 | census DAN→KC ROIs. PPL102 has no PED synapses in the census; kept per the design, and its effect is negligible (1 neuron per side, 502 g1 synapses) |
| secondary set η (a′2, a2, a3, a′3; modulators PPL105, PPL106, PPL104) | η_γ1pedc / 60 | ASSUMPTION | Hige 2015 (α2sc needs 120 pulses); Aso & Rubin 2016. Secondary-set arm only, not in E0/E1 |

### 3.4 MB-R2 at γ2α′1 only (sensitivity arm)

MB-R2 adds the following to MB-R1, **only for k ∈ {g2, a′1}**. It runs between
steps 3 and 4 of §3.2:

```
a_k,h  ← a_k,h · exp(−Δt/τ_a) + d_k,h
g_ijk  ← g_ijk + η_p · min(1, a_k,h / a_sat) · s_i · (g⁰_ijk − g_ijk)
```

| parameter | value | tag |
|---|---|---|
| τ_a | 1.2 s | ASSUMPTION from Handler 2019 READ (−1.2 s potentiating interval, measured at γ2/γ4/γ5) |
| a_sat | 1 spike | ASSUMPTION |
| η_p | **= η_γ2α′1** | ASSUMPTION, NOT VERIFIED |

- **η_p deviates from the design doc**, which said "calibrated". No
  potentiation magnitude with a source exists (design D2), so there is nothing
  to calibrate it against. It is frozen equal to the depression rate of the
  same compartment and declared as such. It only ever feeds a reported arm.
- MB-R2 is not used at γ1pedc, where Hige 2015 found no backward effect. It is
  never primary.
- It runs in E2 on the unpaired and backward conditions (design §3.4) and is
  reported, not gated. It is not run in E1, because E1 reads γ1pedc.

### 3.5 What the rule does not do

- There is no postsynaptic factor (Hige 2015 Fig. 4).
- A weight never goes above its released value: Cohn 2015's DAN-only
  potentiation cannot happen, and this is a declared limitation.
- A sign never flips.
- Nothing outside the plastic set changes (checked every run, gate
  E1_G8).
- WP6 R1's low-passed modulator is not reused (design §2.2).

---

## 4. The plastic set

- **Definition (census query, `scripts/mb_circuit_census.py::compartment_split`).**
  A KC→MBON synapse is plastic iff both hold:
  1. its **postsynaptic** point lies in the released `subprimary` ROI, or the
     `primary` ROI where the subprimary is unspecified, of a primary
     compartment: `g1`, `PED`, `g2` or `a′1`, side stripped for membership;
  2. the MBON's released `instance` name places its dendrite in that
     compartment (`mbon_named_compartments`).
- **Expected counts.** The primary set must reproduce these exactly or the
  run stops:
  - 86,981 synapses;
  - 18,799 (edge, compartment) pairs;
  - 16,250 engine edges;
  - 13 MBON types (census `compartments.plastic_sets.primary`).
- **Secondary set.** The primary compartments plus `a′2`, `a2`, `a3` and
  `a′3`: 145,003 synapses, 28,810 pairs, 25,910 edges. Sensitivity arm only.
- **Fixed.** Everything else stays fixed: PAM compartments, all DAN edges,
  APL, DPM, PN→KC, every MBON output and the whole path to DNa02.
- **Sidecar (I1)**, `data-provenance/malecns_v1/derived/wp7_plastic_set_primary.npz`,
  with one row per (edge, compartment, side):
  - `pre_index`, `post_index`, `edge_index` (CSR index in the pinned graph);
  - `compartment` (`g1|PED|g2|a'1`) and `side` (`L|R`, from the ROI suffix);
  - `n_ijk`, and `N_ij` repeated per row.
  It also carries the DAN mapping `dan_index` per (compartment, side). Its
  sha256 is recorded in the E1 receipt and in the rule's `describe()`, which
  goes into the manifest next to `graph_sha256` (design §2.6).
- **Side consistency.** For every row, the census check that the ROI side
  matches the KC's `somaSide` is **reported**, not enforced. The ROI side is
  where the synapse is, so it decides h.

---

## 5. US drive (D4)

Every result is labelled **"punishment = direct PPL1 activation"**. It is
compared with DAN-activation data first, and with shock data only by analogy.

- **E1 (Hige 2015 protocol).**
  - Four 1-ms current pulses into PPL101 (L and R) at t0 + 0.2, 0.7, 1.2 and
    1.7 s.
  - Amplitude 2 · (V_th − V_rest) / (1 − e^(−1 ms/τ_m)) = 2 · 7 / (1 − e^(−0.05))
    = **287.1 mV-equivalent**. That is twice the drive that brings an isolated
    resting cell to threshold within the pulse. Refractory time (2.2 ms) is
    longer than the pulse, so each pulse gives at most one spike.
  - The 2-ms control step that contains a pulse is split into 1 ms with the
    pulse and 1 ms without. Gate E1_G0 requires exactly 4 PPL101 spikes per
    neuron in the pulse windows.
- **E2 (T-maze, design §3.1).**
  - Twelve 1.25-s pulses every 5 s, from 1 s after CS+ onset.
  - Per-neuron drive `drive_for_rate(30 Hz)`, the same inversion as the P3
    encoder. The 30 Hz target is NOT VERIFIED.
  - Targets: PPL101, PPL102 and PPL103 on both sides, plus PPL106 in the
    secondary arm only.
- **Silencing.** `SILENCE_DRIVE = −200` (`brainlab/io_map.py`), the value the
  optomotor specs use.

---

## 6. Experiments

**Seeds.** No seed is used in more than one role (asserted by
`test_wp7_seed_sets_are_disjoint`).

| experiment | role | seeds |
|---|---|---|
| E0 | gated (V2_primary) and reported (other variants) | 0–9 |
| E1 | calibration of η_γ1pedc only | 100–104 |
| E1 | confirmatory | 200–209 |
| E1 | CPU/GPU parity (reported) | 200, 201, 202 (a subset of the confirmatory seeds, rerun on the CPU) |

### 6.1 E0: is the MB in a learnable regime? (no plasticity)

- **Graph.** All five variants of §2.2. Only V2_primary is gated.
- **Encoder.** The P3 T-maze encoder, by reference: the `encoder` block of
  `tmaze_odour_naive_v3.json` at sha256 `7eb6f62b…`. Odours are presented
  bilaterally.
- **Protocol per seed** (20 s):
  - 2 s of air warm-up;
  - then three blocks of 2 s air, 2 s stimulus and 2 s air, with the stimuli
    air, OCT and MCH;
  - OCT before MCH on even seeds, MCH before OCT on odd seeds;
  - the state is reset once per seed, never between blocks;
  - the control step is 2 ms.
- **Measures:**
  - **KC responder**: stimulus-window spikes minus pre-window spikes ≥ 2.
    This threshold is a judgement, NOT VERIFIED.
  - **Responding fraction** = responders / 4,064, per odour and per side.
  - **Overlap** = |R_OCT ∩ R_MCH| / min(|R_OCT|, |R_MCH|). An empty set makes
    it undefined, which makes gate G3 INCONCLUSIVE.
  - PPL101/102/103, MBON11/12, DPM and APL rates are reported.

| gate | statistic | pass iff | verified |
|---|---|---|---|
| E0_G1 | KC responding fraction, OCT | 95 % CI ⊂ [0.01, 0.20] | false (band is a judgement around Honegger 2011, Turner 2008) |
| E0_G2 | KC responding fraction, MCH | 95 % CI ⊂ [0.01, 0.20] | false |
| E0_G3 | OCT/MCH responder overlap | 95 % CI upper < 0.60 | false (Hige 2015: 30–33 %, definition not matched) |
| P1, P2 | KC spontaneous / driven mean rate | in `KC` bound: [0, 1] Hz / [0, 5] Hz | **true** (firing_rate_bounds_v2) |
| P3–P5 | ORN driven; brain mean; brain max | in bound | ORN true; brain bounds false |

The PN and MBON rates, and the per-side fractions, are reported only.

**Decision.** E1 runs only if E0 on V2_primary is PASS or PASS_PROVISIONAL.
The latter is the best possible here, because G1–G3 are judgements.

If V2_primary fails:

- E1 does not run and the failure is recorded.
- No other variant is promoted after the fact. Changing the primary graph
  needs a new spec version with an amendment.

The comparison variants' verdicts are reported and decide nothing. They
measure what D3 and D5 do to sparseness.

**Cost.** 20 s × 10 seeds × 5 variants = 1,000 simulated seconds, about 42 min
of GPU time at the measured 0.40× real time.

### 6.2 E1: Hige 2015 in silico (neural level)

#### 6.2.1 Protocol

- **Setup.** The primary graph and plastic set with MB-R1. OCT is the CS+ in
  every seed, as in Hige Fig. 1, which is also the calibration target. MCH is
  the CS−. The state is reset once per seed, and weights persist through the
  run.

| phase | time (ms) | content |
|---|---|---|
| pre-test | 0–22,000 | air; CS+ (OCT) 1 s at 5,000; CS− (MCH) 1 s at 15,000 |
| pairing | t0 = 22,000 | per condition (below) |
| wait | to 82,000 | air (60 s from t0) |
| post-test | 82,000–104,000 | as the pre-test: CS+ at 87,000, CS− at 97,000 |

- **Response.** MBON11 spike count summed over both MBON11 neurons, 0–1,400 ms
  from odour onset, minus the count in the 1,400 ms before onset (Hige's
  window).
- **Depression.** D = (R_pre − R_post) / R_pre per seed and odour. If
  R_pre ≤ 0, the gates that use D are INCONCLUSIVE. That seed is not
  excluded.

#### 6.2.2 Conditions (E1)

| condition | at t0 | prediction under MB-R1 | gate |
|---|---|---|---|
| paired | OCT for 1 s; PPL101 pulses at t0 + 0.2/0.7/1.2/1.7 s | D(CS+) ≫ D(CS−) | E1_G2 |
| backward | pulses as paired; OCT from t0 + 2.2 s (0.5 s after the last pulse) | no change | E1_G3 |
| DAN only | pulses; no odour | no change (Cohn 2015's potentiation not representable) | E1_G4 |
| odour only | OCT for 1 s; no pulses | no change | E1_G5 |
| PPL101 silenced | OCT for 1 s; PPL101 held at −200 from t0 − 1 s to t0 + 3 s, no pulses | no change | E1_G6 |
| plasticity off | as paired, `learning_enabled=False` | no change | E1_G7 |

#### 6.2.3 Calibration (before the confirmatory run)

- **What is set.** η_γ1pedc only, on the paired condition, calibration seeds
  100–104, with every other parameter at its frozen value.
- **Target.** The mean of D(CS+) over the 5 seeds is 0.80 ± 0.02 (Hige 2015:
  80 ± 5.7 %).
- **Search.** Bisection on log10 η over [1e-4, 1] per DAN spike, at most 20
  evaluations. Each evaluation runs all 5 seeds.
- **Failure.** If the target is not bracketed or not reached, calibration
  FAILS, E1 does not run, and the result is recorded. Nothing else is changed
  to rescue it.
- **Record.** The result goes into `calibration.result` by one commit
  (§1.3).
- **Worst-case cost.** 20 × 5 × 104 s ≈ 2.9 simulated hours, about 7.2 GPU
  hours.

#### 6.2.4 Criteria (confirmatory seeds 200–209, stated before any run)

| gate | statistic | pass iff | verified |
|---|---|---|---|
| E1_G0 | PPL101 spikes per neuron in the 4 pulse windows (paired, backward, DAN only) | all equal 4 | true (precondition) |
| E1_G1 | PPL101 spikes in the clamp window (silenced) | all equal 0 | true (precondition) |
| E1_G2 | D(CS+) − D(CS−), paired, per seed | 95 % CI lower > 0 | true (Hige 2015 direction) |
| E1_G3–G7 | (mean R_post − mean R_pre) / mean R_pre for CS+, per control | 95 % CI ⊂ [−0.10, 0.10] | false (tolerance is a judgement) |
| E1_G8 | weights changed outside the plastic set, every run | all equal 0 | true |
| P1–P4 | KC spontaneous/driven; brain mean/max (paired) | in bound | KC true; brain false |

These are reported and never gated:

- D(CS+) against 0.80 ± 2 × 0.057. This is in-sample for η.
- D(CS−) against Hige's 0.27 ± 2 × 0.071. This is the out-of-sample
  specificity test, because η was fitted on CS+ only.
- The MBON12 CS+ change after PPL101-only pairing (compartment specificity).
- The MBON11 driven rate.
- CPU/GPU agreement of D(CS+) and D(CS−) on seeds 200–202, within their seed
  CIs (design §2.7).
- The sensitivity arms: τ_e 0.5 s and 2 s, and η_γ2α′1 × 1. η is not
  recalibrated for them, and they run on the confirmatory seeds after the
  primary run.

**Verdict.** The harness rule applies. The best possible outcome is
PASS_PROVISIONAL, because G3–G7 carry judgement tolerances.

A caution recorded before the run: an equivalence gate at ±10 % with 10 seeds
can fail from trial noise alone. A null control that FAILs or is INCONCLUSIVE
for that reason is recorded as exactly that, and the tolerance is not widened
afterwards.

**Cost.**

- Confirmatory: 104 s × 6 conditions × 10 seeds = 6,240 s, about 4.3 GPU
  hours.
- CPU parity: all 6 conditions on 3 seeds = 1,872 s, about 12.7 h on the Ryzen
  CPU at 0.041×.

### 6.3 E2: the behavioural T-maze (declared; not frozen for running)

E2 needs three things first: E1 PASS or PASS_PROVISIONAL, the P3 naive T-maze
Gate 3 (L0), and the verified Tully & Quinn values from the citation thread
(design §3.3). It gets its own spec (`mb_e2_tmaze_v3.json`) before it runs.

The protocol is design §3.1, with the US of §5 and a reciprocal CS+
assignment. The metric is PI_learn = (n_CS− − n_CS+) / choosers, averaged over
the reciprocal halves, using the harness bootstrap. The criteria are declared
now so they cannot drift:

| condition | change from paired | criterion |
|---|---|---|
| paired | §3.1 | **L1**: PI_learn CI lower > 0 |
| unpaired | the same pulses in the 45-s gap, ≥ 20 s after CS+ offset | **L2**: paired − unpaired CI lower > 0; **L3**: CI ⊂ [−0.2, 0.2] |
| backward | pulses in the 15 s before CS+ | **L4**: CI ⊂ [−0.2, 0.2] |
| DAN silenced | PPL1 at −200 throughout | L3 |
| plasticity off | rule disabled | L3 |
| shuffled graph | harness shuffle | L3 |
| γ1pedc only (D1) | only g1 and PED plastic | reported |
| MB-R2 at γ2α′1 (D2) | on unpaired and backward | reported |

**L5** (neural, from the same runs): the CS+ responses of MBON11 and MBON12
are depressed at test, the CS− responses less so, and nothing changes
outside the plastic set.

Magnitudes are reported, never gated.

---

## 7. Pass/fail summary (declared before any run)

| stage | pass | fail | not evaluable |
|---|---|---|---|
| E0 (V2_primary) | G1, G2, G3 and P1–P5 pass → E1 may run | any gate or check fails → E1 does not run; recorded | an undefined overlap or missing data → INCONCLUSIVE; E1 does not run |
| E1 calibration | target reached within 20 evaluations | target not bracketed or not reached → E1 does not run | — |
| E1 | G0–G8 and P1–P4 pass | any fails | R_pre ≤ 0 in a seed, or an undefined ratio → INCONCLUSIVE |
| E2 | L1–L4 (spec to follow) | any fails | — |

What counts as a NULL for P4 is inherited from WP6 §6.7: E1 passes but E2's
L1 fails. That outcome is reported as "the rule acts where declared; the
readout does not see it" (design §4 risk 1). It is not a reason to train the
decoder.

---

## 8. Implementation status

| item | content | status |
|---|---|---|
| I0 | D3 and D5 graph options, variant identity, harness variant selection | **done in this branch** (§2.5) |
| I1 | census extension: plastic-set sidecars (primary, secondary) and the KC→KC calyx sidecar, with the DAN (compartment, side) mapping; needs the 6.8 GB and 13 GB synapse tables | not implemented |
| I2 | `brainlab/mb_plasticity.py`: MB-R1 and MB-R2 with the `VisualHeadingPlasticityRule` interface (`edges`, `update(delta, pre, post, full_counts)`, `substeps`, `describe`); bit-exact replay test on a recorded count sequence (design §2.7 check 1) | not implemented |
| I3 | `validation/paradigms/mb_e0_kc_regime.py` and `mb_e1_hige2015.py`: runner option that keeps plastic weights across the E1 test blocks, pulse-step splitting, the E0/E1 samples named in the specs, a `--sensitivity-arm` selector restricted to declared arms, and synthetic plumbing tests | not implemented |
| I4 | `scripts/wp7_calibrate_eta.py`: the bisection of §6.2.3, writing a calibration receipt | not implemented |

The rule is not implemented here. It needs I1's data to be meaningful, and
the task asked for the spec first.

---

## 9. Commands for the GPU host

These are to be run once I1–I4 exist and the owner gate is passed. Nothing
below has been run. `STAMP=$(date +%Y%m%dT%H%M%S)`.

```bash
# 0. Provenance, before any simulation
python -m brainlab.graph_identity --out outputs/wp7/graph_identity-$STAMP.json
python scripts/mb_circuit_census.py --out outputs/wp7/census-$STAMP.json \
    --synapses connectome_data/malecns_v1/syn-partners-male-cns-v1.0-minconf-0.5.feather \
    --points   connectome_data/malecns_v1/syn-points-male-cns-v1.0-minconf-0.5.feather \
    --wp7-sidecars data-provenance/malecns_v1/derived          # --wp7-sidecars is item I1
sha256sum data-provenance/malecns_v1/derived/wp7_*.npz
python -m validation check validation/specs/mb_e0_kc_regime_v3.json
python -m validation check validation/specs/mb_e1_hige2015_v3.json

# 1. E0: primary first, then the four comparison variants
for V in V2_primary V0_released V1_kc0_dpm_silent V3_kckc_released_dpm_gaba V4_calyx_kept_dpm_gaba; do
  python -m validation run validation/specs/mb_e0_kc_regime_v3.json \
      --graph-variant "$V" --backend cuda --out "outputs/validation/wp7-e0-$V-$STAMP"
done

# 2. E1 calibration (only if E0 V2_primary is PASS/PASS_PROVISIONAL); seeds 100-104
python scripts/wp7_calibrate_eta.py validation/specs/mb_e1_hige2015_v3.json \
    --backend cuda --out "outputs/wp7/e1-calibration-$STAMP"            # item I4
#    -> commit calibration.result only; owner flips status to preregistered

# 3. E1 confirmatory (seeds 200-209 from the spec)
python -m validation run validation/specs/mb_e1_hige2015_v3.json \
    --backend cuda --out "outputs/validation/wp7-e1-$STAMP"

# 4. Reported only: CPU/GPU parity (EXPLORATORY by the seeds override) and sensitivity arms
python -m validation run validation/specs/mb_e1_hige2015_v3.json \
    --backend cpu --seeds 200 201 202 --out "outputs/validation/wp7-e1-cpu-parity-$STAMP"
for A in tau_e_0p5s tau_e_2s eta_g2a1_x1; do
  python -m validation run validation/specs/mb_e1_hige2015_v3.json \
      --backend cuda --sensitivity-arm "$A" --out "outputs/validation/wp7-e1-arm-$A-$STAMP"   # item I3
done
```

Each receipt is committed as it comes out (§1.4).

---

## 10. Tests in this branch

`tests/test_wp7_graph_policy.py` uses small synthetic graphs only, with no
connectome download. It covers:

- **Defaults.** Byte-identical weights, reports and `describe()` against a
  reimplementation of the previous v3 policy, on a toy MB and on a random
  300-neuron graph, for every `unclear_mode`.
- **D3.** Exactly the KC→KC edges are zeroed. The calyx option keeps its
  counts with `prepare.py` arithmetic, and a full count reproduces the
  released weight bit for bit. Bad calyx input is refused: a non-KC edge, a
  count above the released one, a duplicate, or a missing or extra table.
- **Calyx sidecar.** Pair-to-edge mapping and the loader.
- **D5.** GABA sign with the released counts. `released` stays silent, and
  PPL101 stays modulatory-only.
- **Precedence and validation.** `unclear_mode='exclude'` still wins. The
  options are refused under the legacy policy, for unknown values, and
  without cell types.
- **Identity.** The five variants give five distinct `graph_sha256` values.
  The dataset still contains `v3-modulatory-only`. CSR `ptr` and `post` are
  unchanged.
- **Specs.** Both specs load through the harness and their bounds exist.
  Every declared variant is valid, undeclared ones are refused, and the
  paradigm is refused as not implemented. Seed sets are disjoint. The frozen
  content hash and the encoder-source hash match.

---

## 11. Left unspecified or unverified

- **ROI names.** The MaleCNS calyx ROI names (`CA(L)`/`CA(R)`, accessory
  calyx) are NOT VERIFIED. The generator must print them.
- **Count agreement.** Whether synapse-table counts equal the released edge
  counts is unknown. §3.1 is exact either way, and the calyx loader refuses
  counts above the released one.
- **Variant hashes.** The graph hashes of the five variants need the real
  graph and are recorded at step 0.
- **η_p (MB-R2)** is set equal to η_γ2α′1 by assumption, a deviation from the
  design's "calibrated" (§3.4).
- **Judgement values.** The KC responder threshold (≥ 2 spikes), the overlap
  definition, the ±10 % null tolerance, the 287.1 mV-equivalent pulse
  amplitude and the 30 Hz E2 drive are judgements or assumptions, not
  measurements.
- **PPL102 at PED** is kept as a modulator per the design, although the census
  shows no PPL102 synapses in PED.
- **Tully & Quinn 1985** values (E2) remain ABSTRACT only until the citation
  thread verifies them.
- **Not modelled.** No SEZON01 shock route (design D4 follow-up), no mAChR-B
  lateral inhibition (D3), and no DPM serotonin (D5).

## 12. Sources

As in [`design/mb_learning.md`](design/mb_learning.md) §7. The spec JSONs
carry the harness citation records for `hige2015neuron`, `turner2008`,
`honegger2011` and `cohn2015`, and a `census` record for the repository counts.
