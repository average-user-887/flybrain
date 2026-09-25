# Mushroom-body odour learning: circuit, rule and T-maze protocol (P4 design)

**Status**: design and research only, written 2026-09-24. Nothing in the engine,
the daemon or the validation harness changes with this document. The Phase 1
gate has not been signed off, so every number here that depends on a run is a
plan, not a result.

**Roadmap link**: this is the groundwork for the P4 WP7 spec in
[`docs/ROADMAP.md`](../ROADMAP.md) ("Brain training"). It follows the format of
[`WP6_PLASTICITY_SPEC.md`](../WP6_PLASTICITY_SPEC.md) and reuses its machinery.

**Reproducing the numbers**: every count in §1 comes from
[`scripts/mb_circuit_census.py`](../../scripts/mb_circuit_census.py), and its
full output is committed as [`mb_circuit_census.json`](mb_circuit_census.json).
It reads the released MaleCNS v1.0 files pinned in `brainlab/datasets.json`,
checks their sha256 against `data-provenance/malecns_v1/source.lock.json`, and
applies the same node and edge policy as the simulation graph. The edge count
it retains (25,582,938) is checked against the pinned graph. Two optional,
larger released tables give the compartment split in §1.4. They are not in
the lock file, so their hashes are recorded here:

- `syn-partners-male-cns-v1.0-minconf-0.5.feather` (6.8 GB), sha256
  `959d8ef4173b35382a3e6acfaf5167c795b6d10b877572d146af04e1b487bc07`
- `syn-points-male-cns-v1.0-minconf-0.5.feather` (13 GB), sha256
  `c16b1b63186c4d4f28939decea7444451f0f5f6f7ef1bb5dab5b2a7058f8f284`

**Literature**: every source in §7 was checked on Crossref on 2026-09-24. Each
value carries a status tag:

- **READ**: the sentence was found in the full text.
- **ABSTRACT**: only the abstract was available.
- **NOT VERIFIED**: no primary source was read. The value is then an assumption.

---

## 0. The proposal in brief

- **Plastic synapses**: KC→MBON synapses in the compartments innervated by the
  PPL1 punishment dopamine neurons. The primary set is γ1pedc and γ2α′1, the
  two compartments whose DANs induce short-term aversive odour memory
  (Aso 2012, Aso & Rubin 2016). The other three PPL1 compartments form a
  declared secondary set.
- **Rule**: dopamine-gated depression of recently active KC synapses (rule
  MB-R1). A KC eligibility trace is multiplied by the PPL1 spike count of that
  compartment in the same step. Order matters: dopamine before odour does
  nothing. There is no postsynaptic factor, following Hige 2015. Depression
  recovers toward the released weight on a timescale of hours. A bidirectional
  variant (MB-R2, following Handler 2019) is declared now so it is not invented
  later.
- **Punishment**: current injected into the PPL1 DANs during the CS+, as an
  optogenetic experiment does. Electric shock can't be modelled because its
  pathway into PPL1 is polysynaptic and unmapped (§1.8).
- **Readout**: unchanged from the P3 naive T-maze spec: the DNa02 left/right
  spike difference. Nothing downstream of the MBONs is trained.
- **Behavioural target**: a positive learned PI, with the controls at zero.
  The PI sign and definition are the same as the P1 harness's. Magnitudes are
  reported against the literature but not gated, because the published values
  couldn't be read (§3.3).
- **First experiment**: an in-silico replication of Hige 2015 at a single
  synapse class (KC→MBON11, the γ1pedc output). It is neural-level only, needs
  no body and no maze, and takes about 6.5 GPU hours (§5). It comes before
  any behavioural claim.
- **Biggest risk**: the anatomy from the MB output to DNa02 is thin (§1.7). The
  rule can work while the T-maze PI stays null. That is why the neural test
  comes first and the behavioural result is reported as a separate verdict.

---

## 1. The mushroom-body circuit in MaleCNS v1.0

### 1.1 Annotation queries

The simulation graph keeps every annotated body with a superclass, excluding
Glia (`brainlab.connectome.normalize_nodes`, 166,700 neurons). Populations are
regular expressions on the released annotation columns:

| population | column | regex | neurons |
|---|---|---|---:|
| Kenyon cells (KC) | `type` | `^KC` | 4,064 |
| MB output neurons (MBON) | `type` | `^MBON` | 97 (37 types) |
| PAM dopamine neurons | `type` | `^PAM\d` | 316 (15 types) |
| PPL1 dopamine neurons | `type` | `^PPL1\d` | 16 (8 types) |
| APL | `type` | `^APL$` | 2 |
| DPM | `type` | `^DPM$` | 2 |
| MB-C1 | `type` | `^MB-C1$` | 5 |
| OA-VPM3/4 (octopaminergic) | `type` | `^OA-VPM[34]$` | 4 |
| Antennal-lobe projection neurons | `class` | `^ALPN$` | 686 |
| All DANs by class | `class` | `^DAN$` | 340 |
| Descending neurons | `superclass` | `^descending_neuron` | 1,316 |

The `class` column agrees with the type queries: `Kenyon_Cell` 4,064 and `MBON`
97. `DAN` has 340 members, which is the 332 PAM+PPL1 neurons plus the PPL2
neurons.

### 1.2 Populations

| | count | per side (L/R) | released transmitter (consensus) | notes |
|---|---:|---|---|---|
| KC | 4,064 | 2,019 / 2,045 | ACh 4,064 | γ 1,557; α′β′ 695; αβ 1,810; 2 unassigned |
| MBON | 97 | 48 / 49 | ACh 50, Glu 26, GABA 21 | 37 types; compartment in the `instance` name |
| PAM | 316 | 158 / 158 | DA 316 | PAM01–PAM15 |
| PPL1 | 16 | 8 / 8 | DA 16 | PPL101–PPL108, one per side |
| APL | 2 | 1 / 1 | GABA | |
| DPM | 2 | 1 / 1 | **DA (predicted)** | the literature says serotonin and GABA; see §4 |

Compared with the hemibrain's right hemisphere (Li 2020, READ: 1,927 KCs,
of which 701 γ, 337 α′β′ and 889 αβ; 5 PPL1 and 150 PAM DANs onto the MB),
MaleCNS has about 2,030 KCs per side, with class proportions close to the
hemibrain's.

One annotation asymmetry matters for a left/right readout. `KCab-s` has 234
neurons on the left and 423 on the right, and `KCab-c` and `KCab-m` lean the
other way. The αβ totals per side are 885 L and 925 R. This looks like subtype
labelling, not missing cells, but the T-maze readout compares the two sides,
so it is recorded here.

### 1.3 PN→KC input

| measure | value |
|---|---:|
| ALPN→KC edges / synaptic contacts | 22,586 / 390,928 |
| distinct PN inputs per KC (≥3 contacts): mean, median, 10–90 % | 5.1, 5, 2–8 |
| same at ≥1 contact: mean, median | 5.6, 6 |
| KCs with no PN input at ≥3 contacts | 283 (7 %) |
| by lobe system (≥3 contacts): γ, α′β′, αβ | 5.8, 4.5, 4.8 |

This is a little below the light-microscopy counts: 6.4 claws per KC (Aso 2014, citing
Butcher 2012, READ) and 2 to 11 claws, averaging 7 (Caron 2013, READ). One
PN bouton feeds each claw. Of the 283 KCs with no PN input, 185 are `KCg-d`
and 87 are `KCab-p`. Those are the accessory-calyx classes that take visual
rather than olfactory input, so they will not carry the odour in this
experiment.

Where KC input comes from, by presynaptic class (contacts onto all KCs):

| presynaptic class | contacts | share |
|---|---:|---:|
| Kenyon_Cell (KC→KC) | 1,153,845 | 55.4 % |
| ALPN | 390,928 | 18.8 % |
| cb_intrinsic (includes APL) | 289,084 | 13.9 % |
| DAN | 225,127 | 10.8 % |
| MBON | 11,181 | 0.5 % |
| visual_projection | 11,094 | 0.5 % |

The KC→KC and DAN→KC contacts are axo-axonal, in the lobes. A point-neuron LIF
treats them as input at the soma. What that means for sparse coding is the
second risk in §4.

### 1.4 Compartments, their DANs and their MBONs

The compartment of each MBON and DAN type is in its released `instance` name,
for example `MBON11(y1pedc>a/B)` or `PPL101(y1ped)`. PPL106–108 carry no
compartment in the name, so they were placed by their DAN→KC and DAN→MBON
synapses:

- **PPL106** is α3. 6,264 of its 6,287 KC contacts are onto αβ KCs, and its top
  MBON target is MBON14(α3), with 860 contacts.
- **PPL107 and PPL108** barely touch KCs (13 and 1 contacts). They are the PPL1
  neurons that project outside the MB lobes and are not MB reinforcement
  neurons.

The synapse-point table resolves the lobes into compartments (`subprimary` ROI:
g1–g5, a1–a3, a′1–a′3, b1, b2, b′1, b′2). This section's compartment split of
KC→MBON synapses comes from it. See §1.4.1.

**Valence of the DANs.** PPL1 (Claridge-Chang 2009, ABSTRACT: "the PPL1
cluster") signals punishment:

- PPL1-γ1pedc (MB-MP1) and PPL1-γ2α′1 (MB-MV1) each induce aversive odour
  memory when activated (Aso 2010 READ; Aso 2012 READ).
- MP1 memory "decayed rapidly over 9 hours", while MV1 memory was "moderate"
  initially but "still significantly present after 9 hours" (Aso 2012, READ).

PAM signals reward (Liu 2012, Burke 2012, ABSTRACT), with one exception:
PAM-β2β′2a (MB-M3) induces a labile aversive memory (Aso 2010, READ). Its
MaleCNS type is `PAM03(B2B'2a)`.

**The PPL1 compartments** (DAN→KC contacts and DAN→MBON contacts from the
census; the MBON list is every MBON type whose instance names the compartment):

| compartment | DAN (MaleCNS type, per side) | DAN→KC contacts | MBONs with dendrites there | MBON transmitter |
|---|---|---:|---|---|
| γ1pedc | PPL101 (1), PPL102 γ1 (1) | 11,482 + 590 | MBON11 γ1pedc>α/β; also MBON20, MBON25, MBON25-like (γ1γ2), MBON30 (γ1γ2γ3) | GABA; GABA, Glu, Glu, Glu |
| γ2α′1 | PPL103 (1) | 20,538 | MBON12 γ2α′1; MBON32, MBON34, MBON35, MBON25-like (γ2); MBON31 α′1a; MBON15 α′1, MBON15-like α′1α′2; MBON33 γ2γ3; also the multi-compartment γ MBONs above | ACh; GABA, Glu, ACh, Glu; GABA; ACh, ACh; ACh |
| α′2α2 | PPL105 (1) | 8,147 | MBON13 α′2; MBON18 α2sc; MBON23 α2sp; MBON19 α2p3p | ACh |
| α3 | PPL106 (1) | 6,287 | MBON14 α3; MBON19 α2p3p | ACh |
| α′3 | PPL104 (1) | 2,198 | MBON16 α′3ap; MBON17 α′3m; MBON17-like; MBON28 α′3a | ACh |

The PAM compartments (γ3, γ4, γ5, β1, β2, β′1, β′2, α1) are listed with their
DANs and MBONs in `mb_circuit_census.json` (`dan`, `kc_to_mbon`). They are not
plastic in the aversive experiment.

#### 1.4.1 KC→MBON synapses per compartment

**Method.** Each KC→MBON synapse is placed by its postsynaptic point, and each
DAN→KC synapse by its presynaptic point. The census joins the synapse-partner
table to the synapse-point table on body and coordinates, and reads the
released `subprimary` ROI. All 686,685 synapses matched.

- The pedunculus has no subprimary ROI, so where the subprimary is
  unspecified, the `primary` ROI is used (for example `PED`). γ1pedc is
  therefore g1 + PED.
- The PED ROI covers the whole pedunculus, and several MBONs have a few
  synapses just across a ROI boundary. So a synapse counts as belonging to a
  compartment only if it lies in that ROI **and** the MBON's released instance
  name places its dendrite there. For example, MBON22 (calyx) has 5,510 PED
  synapses that stay fixed.

Of the 61,210 KC→MBON engine edges, 35,755 (58 %) have all their synapses in
one ROI. The rest span two or more, which is why the rule keeps a state per
(edge, compartment) pair (§2.2).

The DANs confirm the compartment assignment. PPL101 puts 7,280 synapses in g1
and 2,817 in PED. PPL102 puts 502 in g1. PPL103 puts 13,952 in g2 and 2,709 in
α′1. PPL105 puts 4,659 in α2 and 3,076 in α′2. PPL106 puts 6,248 in α3.
PPL104 puts 1,829 in α′3.

KC synapses in the PPL1 compartments, onto the MBONs named there:

| compartment (ROI) | MBON type: KC synapses there |
|---|---|
| γ1 (g1) + pedunculus (PED) | MBON11 34,214 (g1 25,025 + PED 9,189); MBON30 3,472; MBON20 2,822; MBON25 646; MBON25-like 462 |
| γ2 (g2) | MBON12 13,175; MBON32 10,480; MBON30 4,786; MBON35 4,112; MBON20 1,014; MBON33 869; MBON25 356; MBON25-like 347; MBON34 73 |
| α′1 (a′1) | MBON12 4,401; MBON31 4,186; MBON15-like 991; MBON15 575 |
| α′2, α2 | MBON18 13,075 (α2); MBON13 9,093 (α′2); MBON23 2,486 (α2); MBON15-like 1,173; MBON19 999; MBON17-like 321 |
| α3 | MBON14 22,110; MBON19 38 |
| α′3 | MBON16 3,315; MBON17 2,292; MBON28 1,984; MBON17-like 1,136 |

The declared plastic sets (§2.5), from `compartments.plastic_sets` in the JSON:

| set | compartments | KC→MBON synapses | (edge, compartment) pairs | engine edges | share of KC→MBON synapses |
|---|---|---:|---:|---:|---:|
| primary | γ1pedc, γ2, α′1 | 86,981 | 18,799 | 16,250 | 18.8 % |
| secondary | + α′2, α2, α3, α′3 | 145,003 | 28,810 | 25,910 | 31.3 % |

The engine has 25,582,938 edges, so the primary set is 0.064 % of the graph.

### 1.5 KC→MBON, APL, DPM and the recurrent loops

| block | engine edges | contacts | pre / post neurons |
|---|---:|---:|---|
| KC→MBON (all) | 61,210 | 463,640 | 4,063 / 97 |
| KC→MBON11 (γ1pedc>α/β) | 4,184 | 41,460 | 3,623 / 2 |
| KC→MBON12 (γ2α′1) | 4,126 | 22,603 | 2,177 / 4 |
| KC→APL | 4,693 | 210,352 | 4,063 / 2 |
| APL→KC | 4,633 | 196,200 | 2 / 4,063 |
| KC→DPM / DPM→KC | 4,237 / 4,082 | 95,527 / 32,795 | |
| KC→DAN | 145,663 | 281,862 | 4,061 / 339 |
| MBON→DAN (feedback) | 2,589 | 11,309 | 97 / 338 |
| MBON→MBON | 1,606 | 26,259 | 97 / 97 |
| KC→KC | 642,933 | 1,153,845 | 4,064 / 4,064 |

The KC→MBON total (61,210 edges) agrees with the count in WP6 §1.

**MBON11 input.** MBON11 takes 27,820 contacts from γ KCs, 13,150 from αβ KCs
(the pedunculus part of γ1pedc) and 490 from α′β′ KCs. The hemibrain gives
the same picture: KC input per MBON ranges "from 122 (MBON10) to 1694
(MBON11)" (Li 2020, READ).

**Correction to WP6 §6.6.** That section describes the olfactory fallback as
"KC→MBON01 (2,109 edges, 40,888 contacts) in the γ1pedc compartment". MBON01
is `MBON01(y5B'2a)`, the γ5β′2a output, which is glutamatergic, avoidance-coding
and PAM-innervated. The γ1pedc output neuron, the one Hige 2015 recorded, is
**MBON11**. The edge and contact counts quoted there are MBON01's. Its modulator
(PPL101) is correct.

### 1.6 What the v3 transmitter policy does to this circuit

v3 (`brainlab/transmitter_policy.py`) gives zero fast weight to every
dopaminergic, octopaminergic and serotonergic neuron, and makes them
"available to a plasticity rule as a modulatory signal m(t) and in no other
way". For the MB:

- **All 332 PAM and PPL1 neurons** lose their fast output. That includes
  225,127 DAN→KC contacts and all DAN→MBON contacts, for example
  PPL101→MBON11 at 2,311. This is what the rule needs: DAN spikes count only
  through m(t). The DANs still receive input (KC→DAN 281,862 contacts,
  MBON→DAN 11,309), so their activity is shaped by the network.
- **DPM** is predicted dopaminergic in the release, so v3 silences its output
  too. The predictor's own authors list this as a misprediction, and DPM is
  GABAergic and serotonergic. Decision D5 (§6) relabels it.
- **APL** is GABAergic and keeps its inhibitory feedback onto all KCs.
- **OA-VPM3/4** are octopaminergic and silenced.

### 1.7 From MBONs to the steering readout

The P3 naive T-maze spec (`tmaze_odour_naive_v3.json` on the
`validation/specs/`, merged in PR #9) decides the choice from the side with more
DNa02 spikes. Learning must reach DNa02 through fixed synapses.

| measure | value |
|---|---|
| DNa02 total input (contacts), L / R | 23,957 / 24,168 |
| direct MBON→DNa02 contacts | 250 (MBON31 133, MBON32 104, MBON26 7, MBON27 6), about 0.5 % of DNa02 input |
| MBON→all DNs | 409 edges, 4,347 contacts, from 49 MBONs onto 170 DNs |
| minimum hops MBON→DNa02, v3 fast edges with ≥5 contacts | 1 for MBON26, 31, 32; 2 for most γ MBONs, **including MBON11 and MBON12**; 3 for most αβ and α′ MBONs (for example 02, 06, 07, 13, 14, 18, 23) |

The two-hop input share
Σ_X [c(M,X)/C_in(X)]·[c(X,DNa02)/C_in(DNa02)] ranks how much of DNa02's input
is reachable from each MBON type through one intermediate neuron. It is an
anatomical ranking, not dynamics:

| MBON | compartment | transmitter | two-hop share (×10⁻³) |
|---|---|---|---:|
| MBON26 | β′2d | ACh | 4.68 |
| MBON32 | γ2 | GABA | 3.06 |
| MBON31 | α′1a | GABA | 1.47 |
| MBON27 | γ5d | ACh | 0.78 |
| MBON35 | γ2 | ACh | 0.66 |
| MBON12 | γ2α′1 | ACh | 0.64 |
| MBON01 | γ5β′2a | Glu | 0.42 |
| MBON11 | γ1pedc>α/β | GABA | **≈0.01** |

The γ2α′1 compartment holds four of the six MBONs with the most reach to
DNa02 (MBON32, MBON31, MBON35, MBON12), including both MBONs with substantial
direct DNa02 synapses (MBON31 and MBON32). MBON11, the best-characterised plastic synapse, has almost none. In the
fly, MBON11's role in expressing aversive memory runs through feed-forward
inhibition of other MBONs (Perisse 2016, READ), and the model has those
MBON→MBON edges (26,259 contacts). This is why the primary plastic set includes
γ2α′1 as well as γ1pedc (§2.5), and why the first experiment reads MBON11
directly rather than behaviour (§5).

### 1.8 Where a punishment signal could enter PPL1

The PPL1 MB DANs receive 7,145 (PPL104) to 39,125 (PPL101) contacts. More than
98.7 % come from `cb_intrinsic` neurons, mostly KCs: KCg-m alone gives 13,836
of PPL101's input. Direct input from ascending neurons is at most 0.8 %
(PPL102). There is no identified, direct somatosensory route for an electric
shock in the graph. So the unconditioned stimulus is modelled as current
injected into the PPL1 DANs, the in-silico counterpart of CsChrimson or dTrpA1
DAN activation (Claridge-Chang 2009; Aso 2010, 2012; Aso & Rubin 2016;
Hige 2015). It is declared as such.

---

## 2. The plasticity rule

### 2.1 What the literature supports

| claim | source | status |
|---|---|---|
| One pairing of a 1-s odour with PPL1-γ1pedc activation (four 1-ms pulses at 2 Hz, from 0.2 s) cuts MBON-γ1pedc's response to that odour by 80 ± 5.7 % (118 → 24 spikes). The unpaired odour falls 27 ± 7.1 % (110 → 83). | Hige 2015 Neuron, Fig. 1 | READ |
| Synaptic charge transfer falls 90 ± 3.7 %. | Hige 2015, Fig. 3D | READ |
| A single 1-ms pulse 0.8 s after odour onset is enough. | Hige 2015 | READ |
| Backward order (odour 0.5 s after the last pulse) has no effect. | Hige 2015, Fig. 1G–J | READ |
| Postsynaptic spikes are dispensable (depression persists with MBON spiking blocked by 83 %). | Hige 2015, Fig. 4 | READ |
| KC odour responses and excitability don't change; the expression site (pre or post) is left open. | Hige 2015 | READ |
| The depression lasts ≥40 min with "only a small sign of recovery". | Hige 2015, Fig. S4 | READ |
| It is compartment-specific: PPL1-γ2α′1 pairing leaves MBON-γ1pedc unchanged. | Hige 2015 | READ |
| α2sc needs 1 min of odour with 120 pulses; the 1-s protocol does nothing there. | Hige 2015 | READ |
| DAN 0 to +0.5 s after KC activity depresses; DAN 1.2 s *before* KC activity potentiates; a 6-s interval gives minimal plasticity. DopR1 is needed for depression and DopR2 for potentiation. (Measured at γ4/γ5/γ2 MBONs, not γ1pedc.) | Handler 2019 | READ |
| DAN activation alone potentiates KC→MBON transmission; paired with KC activity it depresses. | Cohn 2015 | READ |
| Behaviour: PPL1-γ1pedc stimulation within 30 s after the onset of a 10-s odour gives aversive memory. DAN stimulation 20–60 s *before* the odour gives appetitive memory. γ1pedc learns in one trial and forgets fast; α3 learns slowly. | Aso & Rubin 2016 | READ |
| MP1 (γ1pedc) memory is robust initially and decays over 9 h. MV1 (γ2α′1) memory is moderate and still present at 9 h. | Aso 2012 | READ |
| Aversive learning depresses the CS+ drive to MBON-γ1pedc>α/β, whose output is needed for short-term memory only, and which inhibits avoidance MBONs. | Perisse 2016 | READ (summary) |
| Aversive learning potentiates the avoidance MBONs' odour responses, through the network. | Felsenberg 2018 | READ (summary) |
| A fitted eligibility time constant or recovery time constant. | none found | NOT VERIFIED |

### 2.2 Rule MB-R1 (primary): dopamine-gated depression of eligible KC synapses

The state is one conductance magnitude g for every plastic (KC *i* → MBON *j*,
compartment *k*) triple. The released value is g⁰ = 0.275 × (KC *i*→MBON *j*
synapses in compartment *k*). An engine edge (*i*→*j*) whose synapses lie in
several compartments gets the sum of its triples plus the fixed part in
non-plastic compartments (§2.6).

Once per control step Δt (the harness uses 2 ms; the LIF step stays 0.1 ms):

```
e_i   ← e_i · exp(−Δt/τ_e) + s_i                   KC eligibility (spike-count trace)
d_k,h = spikes of compartment k's PPL1 DANs on hemisphere h in this step

g     ← g − η_k · min(1, e_i / e_sat) · d_k,h · g   depression, only after KC activity
g     ← g + (g⁰ − g) · Δt / τ_rec                   slow recovery toward the released weight
g     ← clip(g, g_min · g⁰, g⁰)                      depression only; the sign never changes
w_ij  = w_fixed,ij + Σ_k g_ijk                       written to the engine weight array
```

*h* is the hemisphere of the KC: the MB is ipsilateral, and the compartment ROI
in the synapse table carries the side, which the implementation will check.

**Why this form:**

- **Order.** Dopamine multiplies the *current* KC eligibility. Dopamine that
  arrives before the odour finds e_i = 0 and does nothing. That reproduces
  Hige 2015's null backward result and Tully & Quinn's "backward conditioning
  produced no learning" (ABSTRACT). WP6's R1 low-passes the *modulator*, which
  would let dopamine that precedes the odour still depress, so it is
  deliberately not reused here.
- **No postsynaptic factor**, following Hige 2015 Fig. 4.
- **Heterosynaptic and odour-specific by construction.** Only KCs active
  shortly before the DAN spikes are depressed, so specificity is set by the
  overlap of the KC odour representations. Hige 2015 measured 30–33 % overlap
  between OCT and MCH KCs. The model's overlap is measured in experiment E0.
- **Multiplicative in g**, so repeated pairings saturate rather than drive the
  weight negative.

### 2.3 Variant MB-R2 (declared now): bidirectional timing

MB-R2 adds a DopR2-like potentiation term. A dopamine eligibility trace a_k,h
(time constant τ_a) potentiates synapses of KCs that fire *after* dopamine:

```
a_k,h ← a_k,h · exp(−Δt/τ_a) + d_k,h
g     ← g + η_p · min(1, a_k,h / a_sat) · s_i · (g⁰ − g)   potentiation back toward g⁰
```

This makes backward pairing (Handler 2019: −1.2 s) potentiate, and reverses a
prior depression (Handler: "could be reversed through a single backward
conditioning trial"). It is **not** primary, for two reasons:

- the potentiation data come from γ4/γ5/γ2, not from γ1pedc;
- the capped form (g ≤ g⁰) can't express Cohn 2015's potentiation *above*
  baseline from DAN activation alone.

Decision D2 (§6) keeps MB-R1 primary everywhere and runs MB-R2 only at
γ2α′1, as a sensitivity arm.

### 2.4 Parameters

| parameter | value | units | status and basis |
|---|---|---|---|
| Δt, rule step | 2 | ms | engineering; equals the harness control step |
| τ_e, KC eligibility | 1.0 | s | **assumption**, bounded by Handler 2019 (depresses at +0.5 s, minimal at +6 s) and Hige 2015 (a pulse 0.8 s after onset works). Sensitivity arms 0.5 s and 2 s. |
| e_sat | 5 | spikes | **assumption**; KCs fire "typically 5 to 10" spikes per odour (Honegger 2011, READ); responders fire 2.2–4.9 spikes in 2 s (Turner 2008, via `firing_rate_bounds_v2.json`) |
| η for γ1pedc | calibrated | per DAN spike | **calibrated once** on the paired condition of E1 to Hige 2015's 80 % depression, on calibration seeds only, then frozen (§5). Not tuned on behaviour. |
| η for γ2α′1 | 0.25 × η_γ1pedc | | **assumption** (D1). MV1 activation gives a "slight but significant" memory and is not required for 2-min memory (Aso 2012, READ). Sensitivity arm ×1. |
| η for α′2α2, α3, α′3 (secondary set) | η_γ1pedc / 60 | | **assumption** from Hige 2015 (α2sc needs a 1-min, 120-pulse protocol, not 1 s) and Aso & Rubin 2016 (α3 barely learns from one pairing). The ratio is a declared guess. |
| τ_rec, recovery | 3 | h | **assumption**. Hige 2015 shows little recovery in 40 min, and MP1 memory decays over about 9 h (Aso 2012). Much longer than a run, so depression is effectively permanent inside one experiment. |
| g_min | 0 | fraction of g⁰ | Hige 2015's 90 % charge reduction allows near-complete depression |
| τ_a, η_p, a_sat (MB-R2 only) | 1.2 s, calibrated, 1 | | **assumption** from Handler 2019's −1.2 s potentiating interval |
| DAN drive for the US | the per-neuron current that gives the target rate through `drive_for_rate` | | **assumption**: target 30 Hz during each pulse, a CsChrimson-like range. NOT VERIFIED; PPL1 in-vivo rates were not read. |

### 2.5 The plastic synapse set

- **Primary**: every KC→MBON synapse in the g1 or PED ROI (modulator PPL101 +
  PPL102) or in the g2 or α′1 ROI (modulator PPL103), onto an MBON whose
  instance name places its dendrite in that compartment. That is 86,981
  synapses on 16,250 engine edges (§1.4.1). `scripts/mb_circuit_census.py`
  already computes the membership; the rule class will load the same list.
- **Secondary** (sensitivity analysis only, declared now): the primary set plus
  α′2, α2, α3 and α′3, with the slow η above.
- **Fixed**: everything else, including PAM compartments, every edge onto or out
  of the DANs, APL, all PN→KC, all MBON→anything, and the whole path to DNa02.
  A behavioural change can then only come from the declared set. This is WP6's
  principle, kept.

### 2.6 How it fits the v3 engine and the WP6 code

The existing pieces carry over with small, declared changes:

- **Weights.** `experiment_registry.GraphInstance` already keeps a sparse
  `plastic_delta` over declared edge indices, adds it to the released weights
  in `_materialize`, calls `rule.update(delta, pre_counts, post_counts,
  full_counts=counts)` after every `Brain.step`, and checkpoints
  `plastic.edges` and `plastic.delta`. An MB rule class with the same interface
  as `VisualHeadingPlasticityRule` drops in.
  - It needs `full_counts` (the PPL1 spikes), which the registry already
    passes.
  - Engine edges that span several compartments need a per-triple state that
    is summed into the edge's delta. That is internal to the rule.
- **Transmitter policy.** v3 already makes DANs modulatory-only. This removes
  the WP6 §3.5 double-counting confound for the MB: a DAN spike acts only
  through d_k,h.
- **Rule identity.** The rule's `describe()` goes into the manifest next to
  `graph_sha256`, as in WP6. A run under a different rule or plastic set is a
  different identity.
- **The P1 harness runner resets the brain to rest before each trial.** For
  learning, the transient state (v, g, refractory, queue) must reset between
  test trials while the plastic weights persist. `Brain.snapshot_state` and
  `restore_state` already exclude the weights, so this needs a runner option,
  not an engine change.
- **Not reused from WP6 R1.** The low-passed modulator, which breaks the order
  rule (§2.2), and the per-step clip of the delta only (recovery works
  differently here).

### 2.7 GPU backend

Plastic weights already work on the CUDA backend:

- `Brain.update_weights(edges)` pushes changed edge weights to the device
  through `update_edges`, which scatters new fixed-point increments
  (`brainlab/cuda_engine.py`).
- A shared read-only device copy is duplicated before the first write, and
  in-place edits of `Brain.weight` fail loudly on the GPU.
- The rule itself runs on the host from the per-step spike counts, which the
  GPU backend already copies back every step.

Cost and parity:

- **Cost.** The primary set is 16,250 engine edges (18,799 edge-compartment
  pairs, §1.4.1). Updating them per 2-ms step is a few hundred microseconds of
  numpy against a step that takes about 5 ms of wall time at the measured
  0.40× real time. Re-uploading every plastic edge is about 130 KB per step.
  Uploading only edges whose weight changed (only when a PPL1 DAN fired) is an
  exact optimisation.
- **Parity.** GPU spike trains agree with the CPU reference closely but not
  bit for bit (`cuda_engine.py` docstring). Learned weights will therefore
  differ slightly between backends. The parity checks are:
  1. **Exact**: given the same recorded spike-count sequence, the rule's
     weight trajectory is bit-identical on both backends, because the rule
     runs on the host in both. This is a unit test.
  2. **Statistical**: the E1 outcomes (depression of CS+ and CS−) agree between
     CPU and GPU within their seed CIs, as the roadmap requires for any GPU
     plasticity path.
  3. Fixed-point increments resolve 2⁻³² leak units per edge, far below any
     weight change the rule makes.

No new GPU kernel is needed for MB-R1. A device-side rule is an optimisation
for later, and it would need its own parity test.

---

## 3. T-maze conditioning protocol and targets

### 3.1 Protocol

This is modelled on Tully & Quinn 1985 as the literature summarises it
(ABSTRACT; the full protocol text was not read):

- Twelve shocks during a 1-min CS+, then a 1-min CS− without shock.
- About 150 flies per group; testing is a 2-min choice.
- A modern write-up states the protocol explicitly (Okray 2025, SECONDARY):
  "pairing a 1-min presentation of one odor with 12 electric shocks then with a
  1-min presentation of a different odor without punishment ... giving the
  flies 2 min to choose".

The in-silico version, per simulated fly:

| phase | duration | stimulus |
|---|---:|---|
| rest | 5 s | clean air (the encoder's spontaneous ORN rates) |
| CS+ | 60 s | odour A, bilateral; 12 PPL1 drive pulses of 1.25 s every 5 s, starting 1 s after odour onset, to PPL101, PPL102 and PPL103 (D4); PPL106 as well in the secondary arm |
| gap | 45 s | air |
| CS− | 60 s | odour B, bilateral, no DAN drive |
| delay | 120 s | air (the "3-min" test point) |
| test | 2 × (0.5 s pre + 2 s decision) | the P3 naive T-maze trial, reciprocal sides, transient state reset between the two trials, weights kept |

- **Odours and encoder**: OCT and MCH with the same DoOR encoder, drive
  inversion, noise and DNa02 decoder as `tmaze_odour_naive_v3.json`.
  Reciprocal design: half the seeds use OCT as CS+, half use MCH.
- **Cost**: about 295 simulated seconds per fly. At the measured full-brain GPU
  rate of 0.40× real time, that is about 12 min per fly; on the Ryzen CPU at
  0.041×, about 2 h. 50 seeds for one condition is about 10 GPU hours, so the
  five conditions in §3.4 are a multi-night batch. A reduced protocol (10-s CS,
  3 pulses, Aso & Rubin 2016's short training) is declared as the pilot.

### 3.2 PI definition, the same as the P1 harness

- The harness's convention is `PI(a) = (choices of a − choices of b) /
  choosers`, resampled by seed. Non-choosers are excluded and their rate is
  gated.
- The learned index is **PI_learn = PI(CS−) = (n_CS− − n_CS+) / choosers**,
  averaged over the two reciprocal halves (OCT as CS+ and MCH as CS+).
- Positive means the CS+ is avoided, which is Tully & Quinn's sign. The
  bootstrap, the Clopper-Pearson choice-rate gate and the verdict rule are the
  harness's own.

### 3.3 Targets to preregister

| id | claim | value | basis and status |
|---|---|---|---|
| L0 | The naive spec passes first: flies choose (T0), OCT and MCH avoided vs air (T2, T3); the OCT/MCH balance is reported, not gated | the P3 naive spec's gates | P3 Gate 3 decides this; it is a precondition, not a P4 result. The literature check (PR #15) found the 50:50 split is an experimenter concentration calibration, so the P4 protocol also needs the naive split measured and reported next to the learned PI. |
| L1 | Learned avoidance after paired training | PI_learn 95 % CI lower bound > 0 | direction only. Tully & Quinn 1985 (ABSTRACT): "Typically, 95% of trained flies avoided the shock-associated odor", i.e. PI ≈ 0.9. That PI is a derivation and the paper's exact PIs were NOT READ. **Not gated on magnitude.** |
| L2 | Paired beats unpaired | PI_learn(paired) − PI_learn(unpaired) CI lower bound > 0 | Tully & Quinn 1985 (ABSTRACT): non-associative controls "did not alter our associative learning index" |
| L3 | Controls at zero | PI_learn CI within ±0.2 for unpaired, DAN-silenced and plasticity-off | the tolerance mirrors the P3 balance gate and is a judgement |
| L4 | Backward pairing (US before CS+) | PI_learn CI within ±0.2 under MB-R1 | Tully & Quinn 1985 (ABSTRACT): "produced no learning"; Hige 2015 (READ): no synaptic effect. Aso & Rubin 2016 (READ) find *appetitive* memory for DAN 20–60 s before the odour, which MB-R1 cannot produce. That difference is recorded, not tuned away. |
| L5 | The rule acts where declared | CS+ MBON11 and MBON12 responses depressed at test, CS− less so; no weight changes outside the plastic set | neural-level, from E1 |
| R | Magnitude, reported only | PI_learn next to ≈0.9 (Tully & Quinn, ABSTRACT, derived) and the optogenetic PPL1 PIs (Aso 2012; Aso & Rubin 2016; figures only, NOT VERIFIED) | not gated |

**Overlap with the "Verify validation citations" thread.** The two share one
source and one convention:

- **Tully & Quinn 1985** (`tully1985` in the harness specs, DOI
  10.1007/BF01350033). Both threads could only read the abstract. The harness
  draft uses it for the naive OCT/MCH balance. This design uses it for the
  learned PI and for the backward and non-associative controls.
- **The PI convention of Li 2013** (`li2013`, reciprocal half-PIs).

This document uses exactly the harness's citation keys and PI convention.
Before the P4 spec is preregistered, every tully1985 value here must be
replaced by whatever that thread finally verifies. If they disagree, the
thread's number wins and L1 and L2 are re-read against it. No other T-maze
values overlap: the naive odour-vs-air numbers (Akalal 2006, Li 2013 Fig. 1I)
are P3's and are not used here.

### 3.4 Conditions

| condition | what changes | expected under MB-R1 |
|---|---|---|
| paired | as §3.1 | PI_learn > 0 |
| unpaired | the same 12 DAN pulses, delivered in the 45-s air gap, starting ≥20 s after CS+ offset | ≈ 0 |
| backward | the DAN pulses in the 15 s before CS+ onset | ≈ 0 |
| DAN silenced | PPL1 clamped with the harness's silencing current throughout | ≈ 0 |
| plasticity off | the rule disabled (`learning_enabled=False`) | ≈ 0; equals naive |
| shuffled graph | the roadmap's shuffled control, built by the harness | ≈ 0 |
| γ1pedc only (D1) | only γ1pedc plastic, same US | > 0 if the fly's 2-min necessity data hold in the model; reported next to paired |
| R2 at γ2α′1 (D2) | MB-R2 potentiation in γ2α′1, run on unpaired and backward | reported, not gated |

---

## 4. Risks

1. **The readout may not see the learning.**
   - Direct MBON→DNa02 input is 0.5 % of DNa02's synapses, and MBON11's
     two-hop reach is hundreds of times smaller than MBON26's (§1.7).
   - The rule can depress exactly the right synapses while the PI stays at
     zero.
   - *Mitigations*: the neural outcome (L5, E1) is its own verdict; γ2α′1 is in
     the primary set because its MBONs have the most reach; the decoder stays
     frozen.
   - A behavioural null with a neural positive is reported as exactly that. It
     is not a reason to train the decoder.
2. **Sparse KC coding may not survive the point-neuron model.**
   - 55 % of KC input contacts are KC→KC and 11 % are DAN→KC. Both are
     axo-axonal in the lobes, yet a LIF point neuron adds them at the soma.
     (DAN→KC is zero under v3; KC→KC is not.)
   - Flies activate about 5 % of KCs per odour (Honegger 2011, READ; up to 17 %
     for some odours), with APL feedback setting that sparseness (Lin 2014,
     READ).
   - If KC→KC excitation makes the model's KCs fire densely, OCT and MCH
     representations overlap and learning can't be odour-specific.
   - *Mitigation*: decision D3 (§6) gives KC→KC no fast weight, because the
     literature shows these synapses are axo-axonal and act through inhibitory
     mAChR-B. E0 measures both graphs.
3. **The US is dopamine-neuron drive, not shock.** There is no mapped
   nociceptive route into PPL1 (§1.8). Results compare to optogenetic DAN data
   first, and to shock data only by analogy. Shock also recruits other DANs
   (for example PAM03, MB-M3), which the model will not see.
   - SEZON01 carries shock to PPL101 and is required for shock learning
     (Meschi 2024), but it isn't typed in MaleCNS v1.0. A SEZON01-driven US
     arm is a follow-up once it is identified (D4).
4. **Parameters are partly unknown.**
   - τ_e, τ_rec, the η ratios and the DAN drive rate are assumptions (§2.4).
   - η for γ1pedc is calibrated on Hige 2015 only, on calibration seeds, before
     any behavioural run.
   - A small preregistered sweep (τ_e 0.5 and 2 s; η_γ2α′1 ×1) is allowed,
     and results are reported as "learning under assumptions X".
5. **The left/right choice needs lateralised MB output.**
   - The P3 encoder gives each antenna only its own arm's odour
     (`contralateral_fraction 0`, a declared design choice).
   - Real flies also sample sequentially.
   - The KC subtype labels are asymmetric (`KCab-s` 234 L vs 423 R, §1.2).
6. **Compartment attribution.** Engine edges are neuron pairs, and 42 % of
   KC→MBON edges have synapses in more than one ROI. The per-synapse ROI split
   (§1.4.1) handles this. The ROI boundaries come from the release and are not
   re-derived here. The pedunculus is one ROI, not split into γ1pedc and the
   rest, which is why membership also needs the MBON's named compartment.
7. **Transmitter labels.**
   - DPM is predicted dopaminergic, a documented misprediction. D5 relabels it
     GABA; its fast-inhibition kinetics are an approximation.
   - Two PPL2 neurons are `unclear`.
   - DAN→MBON direct synapses carry no fast effect under v3, although they
     exist.
8. **Compute.** About 12 GPU minutes per fly. The full condition set is several
   nights on the Ryzen. The pilot uses the short protocol.
9. **Backend drift.** GPU and CPU spike trains are not bit-identical, so
   learned weights differ. The parity checks are in §2.7.

---

## 5. The first experiments

### E0: is the MB in a learnable regime? (no plasticity)

This is the P3 T-maze encoder with OCT, MCH and air, 10 seeds, 2-s odour
presentations. It measures:

- the fraction of KCs that spike per odour, per side;
- the OCT/MCH overlap of the active KC sets;
- PPL101 and PPL103 spontaneous and odour-evoked rates;
- MBON11 and MBON12 evoked rates.

**Pass** (all reported; only the first two decide whether E1 runs):

- KC responding fraction per odour between 1 % and 20 %. The literature has
  5–6 % on average with a maximum of 17 % (Honegger 2011), and 6 ± 5 % by
  single-cell recording (Turner 2008). The KC rate must also stay inside the
  harness's verified KC bound (driven ≤ 5 Hz, `firing_rate_bounds_v2.json`).
- OCT/MCH active-set overlap below 60 %. Hige 2015 measured 30–33 %; the bound
  is a judgement.

About 20 simulated seconds per seed and 200 in total, which is minutes on the
GPU. E0 runs on five graph variants (D3, D5): KC→KC fast weight {released, zero,
zero except calyx} × DPM {GABA, silent}. The P4 primary graph is KC→KC zero,
DPM GABA. If the primary graph still fails the KC bounds, learning waits and
the failure is reported.

### E1: Hige 2015 in silico (neural-level)

- **Recording**: MBON11 spike counts over 0–1.4 s from odour onset, minus
  spontaneous, as Hige counted them.
- **Tests**: before pairing, then 60 s after it. Each test is OCT and MCH, 1 s
  each, 10 s apart.
- **Pairing**: 1-s OCT with PPL101 driven by four 1-ms pulses at 2 Hz starting
  0.2 s after onset. The drive is one pulse-length current per pulse.

| condition | what changes | prediction under MB-R1 |
|---|---|---|
| paired | as above | CS+ response strongly depressed; CS− less |
| backward | the odour starts 0.5 s after the last pulse | no change (Hige 2015) |
| DAN only | pulses, no odour | no change (Cohn 2015 reports potentiation, which MB-R1 cannot produce: a declared limitation) |
| odour only | OCT, no pulses | no change |
| PPL101 silenced | a clamp during pairing | no change |
| plasticity off | the rule disabled | no change |

**Calibration.** η_γ1pedc is set on 5 calibration seeds of the paired
condition so that the mean CS+ depression is 80 %. It is then frozen, and the
spec is committed.

**Evaluation**, on 10 fresh seeds, gated:

- CS+ depression > CS− depression (CI above 0);
- backward, DAN-only, odour-only, silenced and plasticity-off within ±10 % of
  their pre-pairing response;
- no weight change outside the plastic set.

Reported, not gated:

- the CS− depression against Hige's 27 ± 7.1 %, which tests odour specificity
  out of sample, because η was fitted only on CS+;
- CPU/GPU agreement on 3 seeds.

**Cost**: about 105 simulated seconds per run (two 22-s test blocks, the
pairing and the 60-s wait) × 6 conditions × 15 seeds = 9,450 simulated
seconds, about 6.5 h on the GPU at 0.40×, one overnight run. It needs no body, no maze
and no behavioural decoder, so it can run as soon as the P1 harness and the
rule class exist.

A γ2α′1 version (PPL103 → MBON12 and MBON32) follows the same template, once
E1 passes.

---

## 6. Design decisions (researched 2026-09-25)

The owner asked for these five to be settled from the literature. Sources are
numbered as in §7. Each decision is part of the design from now on, and each
is revisited only if a named result contradicts it.

### D1. Plastic set: γ1pedc + γ2α′1, with γ1pedc as the stronger site

**Decision**: keep both compartments plastic. η for γ2α′1 is now **0.25 ×
η_γ1pedc** (it was 1×), and 1× becomes the sensitivity arm. A declared
**γ1pedc-only** arm runs in every behavioural batch.

Evidence:

- **γ1pedc is necessary and sufficient for short-term memory.** Blocking
  MB-MP1 (PPL1-γ1pedc) during training impaired memory at 2 min, 2 h and
  9 h (Aso 2012, READ). One pairing depresses KC→MBON-γ1pedc by 80 %
  (Hige 2015, READ). Its activation gives the strongest immediate memory
  (Aso & Rubin 2016, READ).
- **γ2α′1 carries shock, but more weakly for this assay.**
  - MB-MV1 (PPL1-γ2α′1) responds strongly to electric shock (Mao & Davis
    2009, READ: the lower stalk/junction response was among the largest;
    Berry 2018, READ).
  - Its activation gives a "slight but significant" aversive memory
    (Aso 2012, READ).
  - Blocking it spares 2-min and 2-h memory and impairs only 9-h memory
    (Aso 2012, READ).
- **The γ1pedc-only arm tests the readout risk.** γ2α′1 is where the MBONs
  with the most reach to DNa02 sit (§1.7). So a learned PI that appears only
  when γ2α′1 is plastic, and vanishes in the γ1pedc-only arm, would disagree
  with the fly's 2-min necessity data. It is reported that way.
- **α3 and α′2α2 stay in the secondary set.** α3 responds to shock but gives
  "barely detectable" immediate memory (Aso & Rubin 2016, READ). The evidence
  that α′2α2 contributes to 3-min memory is thin.

### D2. Rule: MB-R1 is primary; MB-R2 potentiation only at γ2α′1, as a sensitivity arm

**Decision**: MB-R1 (forward-only depression) is the primary rule in every
compartment. MB-R2's backward potentiation is declared **only for γ2α′1**, as
a preregistered sensitivity arm run on the unpaired and backward conditions.

Evidence:

- **γ1pedc**: at this synapse itself, backward order gave "no change in odor
  responses" (Hige 2015, READ). No study found backward potentiation at
  γ1pedc.
- **γ2 and γ2α′1**: backward potentiation and reversal *are* measured there.
  - Handler 2019 (READ) found it at γ2 with DAN activation 1.2 s before KC
    activity, and with real shock 3 s before the odour.
  - Berry 2018 (READ) found that DAn-γ2α′1 activation "is sufficient for the
    bidirectional modulation".
- **Behaviour**: DAN activation 20–60 s before the odour gives appetitive
  memory for PPL1-γ1pedc (Aso & Rubin 2016, READ). That is behavioural only,
  at a timescale MB-R1's 1-s eligibility can't represent. It stays a recorded
  mismatch under L4.

**Why R2 is not primary at γ2α′1 either**: its potentiation magnitude η_p
has no sourced value. Putting an uncalibrated term inside the headline result
would trade accuracy for completeness.

### D3. KC→KC synapses carry no fast weight (declared graph policy)

**Decision**: a v3 policy option `kc_kc='modulatory-only'` gives every KC→KC
edge zero fast weight, the same treatment v3 already gives aminergic outputs.
It is the P4 primary graph. E0 runs both graphs, so the effect is measured,
not assumed. It is a new graph identity with its own `graph_sha256`, and
nothing about the v3 default for other paradigms changes.

This is no longer conditional on E0. The evidence says these synapses do not
excite the soma:

- **They are axonal.**
  - "KCs make 48% of their synapses onto other KCs in the adult α lobe", but
    there is "no direct evidence that they are functional synapses"
    (Takemura 2017, READ).
  - "almost no KC-KC interactions are observed between dendrites, and the vast
    majority of KC-KC interactions are between axons" (Manoim 2022, READ).
- **Their effect is inhibitory and modulatory.** They act through mAChR-B and
  "suppress both odor-evoked calcium responses and dopamine-evoked cAMP
  signals in neighboring KCs" (Manoim 2022, READ).
- **KC axons show no fast response to acetylcholine.** "local ACh application
  to the MB lobes did not elicit Ca²⁺ transients in KCs" (Barnstedt 2016,
  READ).
- **In the point-neuron model they would do the opposite.** They make up 55 %
  of KC input contacts (§1.3) and would arrive as fast excitation at the soma.

**Caveat**: the hemibrain also annotates KC→KC synapses in the calyx (11.9 %
of calyx input, sign unknown; Li 2020, READ). The census's synapse ROIs can
separate calyx from lobe KC→KC synapses. A calyx-only-kept arm is declared for
E0.

A lateral-inhibition model of the mAChR-B effect is out of scope for P4.
Zero is the conservative choice.

### D4. The US is PPL1 drive, labelled as optogenetic-style

**Decision**: yes, and the drive follows D1's weighting.

- PPL101 and PPL102 (γ1pedc, γ1) get the full pulse drive (§3.1).
- PPL103 (γ2α′1) gets the same pulses.
- PPL106 (α3) gets them only in the secondary arm.

Every result is labelled *"punishment = direct PPL1 activation"* and is
compared first with DAN-activation data.

Evidence:

- **Shock partly arrives through a known ascending cell type we can't find in
  MaleCNS.**
  - Meschi 2024 (READ) shows ascending SEZON01 neurons "synapse onto PPL101
    (γ1pedc), PAM01 (γ5), and PAM02 (β′2a)" and "are required to convey the
    reinforcing effects of electric shock".
  - But blocking them does not abolish learning ("other ascending pathways
    contribute").
  - No MaleCNS annotation column names SEZON01; checked `type`, `instance`,
    `flywireType`, `hemibrainType`, `synonyms`, `mancType` and `group`.
- **The DANs that carry punishment share input.** "PPL101 (γ1pedc), PPL103
  (γ2α′1) and PAM12 (γ3) DANs share input, which supports the idea that they
  are driven in parallel in response to aversive/punishing cues" (Li 2020,
  READ). That supports driving them together.
- **Activation substitutes for shock.** PPL1 activation does so
  (Claridge-Chang 2009; Aso 2010, 2012; Aso & Rubin 2016). So an
  activation-driven US has a direct experimental counterpart, and a shock
  ingress would need a mapping the release doesn't provide.
- **Follow-up**: identify the SEZON01 homologue in MaleCNS (by morphology or
  the FlyWire match) and add "shock via SEZON01 drive" as a second US arm.
  That is a separate task.

### D5. DPM is relabelled GABA (+5-HT), never dopamine

**Decision**: the transmitter policy gets a declared per-type override,
`DPM: gaba`. DPM's fast output becomes inhibitory, and its serotonin stays
unmodelled like every other aminergic signal. E0 compares it with the
release's label (DPM silent), and the effect on KC sparseness is reported.

Evidence:

- **The dopamine label is a known error of the predictor itself.** Eckstein
  2024 (READ) lists DPM among its "major mispredictions": it is "predicted to
  be dopaminergic" in both hemibrain and FlyWire, although it expresses "the
  fast-acting transmitter GABA the monoamine serotonin and several
  neuropeptides".
- **DPM inhibits the MB through GABA-A receptors.**
  - "DPM neurons release GABA and 5HT, but not ACh or dopamine".
  - "DPM neurons inhibit the MBs via activation of GABAA receptors"
    (Haynes 2015, READ).
- **Caveat**: Haynes measured a picrotoxin-sensitive chloride rise in
  explants and did not show fast synaptic kinetics. DPM output is also needed
  mainly during consolidation (Keene 2004 and 2006, ABSTRACT). A fast
  GABAergic DPM is therefore the closest available approximation, not a
  measured property. That is why E0 reports both labels.

### What this changes in the rest of the document

- §2.4: η for γ2α′1 is now 0.25 × η_γ1pedc.
- §3.1: the PPL1 drive follows D4.
- §3.4: the γ1pedc-only arm and the MB-R2 at γ2α′1 arm are added.
- §5 E0 runs the four graph variants (KC→KC on/off × DPM GABA/silent) plus
  the calyx-kept arm.
- The engine and policy changes D3 and D5 need are implementation work for
  the WP7 spec. They are not in this PR.

## 7. Sources

Checked on Crossref on 2026-09-24. Status as in the header.

1. Hige T, Aso Y, Modi MN, Rubin GM, Turner GC (2015) Heterosynaptic plasticity underlies aversive olfactory learning in *Drosophila*. *Neuron* 88(5):985–998. doi:10.1016/j.neuron.2015.11.003 (PMC4674068). READ.
2. Handler A, Graham TGW, Cohn R, Morantte I, Siliciano AF, Zeng J, Li Y, Ruta V (2019) Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning. *Cell* 178(1):60–75.e19. doi:10.1016/j.cell.2019.05.040. READ.
3. Aso Y, Rubin GM (2016) Dopaminergic neurons write and update memories with cell-type-specific rules. *eLife* 5:e16135. doi:10.7554/eLife.16135. READ. (Four-quadrant arena, not a T-maze.)
4. Aso Y, Herb A, Ogueta M, Siwanowicz I, Templier T, Friedrich AB, Ito K, Scholz H, Tanimoto H (2012) Three dopamine pathways induce aversive odor memories with different stability. *PLoS Genet* 8(7):e1002768. doi:10.1371/journal.pgen.1002768. READ (PIs in figures only).
5. Aso Y, Siwanowicz I, Bräcker L, Ito K, Kitamoto T, Tanimoto H (2010) Specific dopaminergic neurons for the formation of labile aversive memory. *Curr Biol* 20(16):1445–1451. doi:10.1016/j.cub.2010.06.048. READ.
6. Cohn R, Morantte I, Ruta V (2015) Coordinated and compartmentalized neuromodulation shapes sensory processing in *Drosophila*. *Cell* 163(7):1742–1755. doi:10.1016/j.cell.2015.11.019. READ.
7. Tully T, Quinn WG (1985) Classical conditioning and retention in normal and mutant *Drosophila melanogaster*. *J Comp Physiol A* 157:263–277. doi:10.1007/BF01350033. ABSTRACT only.
8. Tempel BL, Bonini N, Dawson DR, Quinn WG (1983) Reward learning in normal and mutant *Drosophila*. *PNAS* 80(5):1482–1486. doi:10.1073/pnas.80.5.1482. ABSTRACT only; for the later appetitive (PAM) version.
9. Tanimoto H, Heisenberg M, Gerber B (2004) Event timing turns punishment to reward. *Nature* 430:983. doi:10.1038/430983a. ABSTRACT only.
10. Aso Y, Hattori D, Yu Y, et al., Rubin GM (2014) The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife* 3:e04577. doi:10.7554/eLife.04577. READ.
11. Aso Y, Sitaraman D, Ichinose T, et al., Rubin GM (2014) Mushroom body output neurons encode valence and guide memory-based action selection in *Drosophila*. *eLife* 3:e04580. doi:10.7554/eLife.04580. READ.
12. Li F, Lindsey JW, Marin EC, et al., Rubin GM (2020) The connectome of the adult *Drosophila* mushroom body provides insights into function. *eLife* 9:e62576. doi:10.7554/eLife.62576. READ.
13. Claridge-Chang A, Roorda RD, Vrontou E, Sjulson L, Li H, Hirsh J, Miesenböck G (2009) Writing memories with light-addressable reinforcement circuitry. *Cell* 139(2):405–415. doi:10.1016/j.cell.2009.08.034. ABSTRACT only.
14. Liu C, Plaçais P-Y, Yamagata N, et al., Tanimoto H (2012) A subset of dopamine neurons signals reward for odour memory in *Drosophila*. *Nature* 488:512–516. doi:10.1038/nature11304. ABSTRACT only.
15. Burke CJ, Huetteroth W, Owald D, et al., Waddell S (2012) Layered reward signalling through octopamine and dopamine in *Drosophila*. *Nature* 492:433–437. doi:10.1038/nature11614. ABSTRACT only.
16. Perisse E, Owald D, Barnstedt O, Talbot CB, Huetteroth W, Waddell S (2016) Aversive learning and appetitive motivation toggle feed-forward inhibition in the *Drosophila* mushroom body. *Neuron* 90(5):1086–1099. doi:10.1016/j.neuron.2016.04.034. READ (summary).
17. Felsenberg J, Jacob PF, Walker T, et al., Waddell S (2018) Integration of parallel opposing memories underlies memory extinction. *Cell* 175(3):709–722.e15. doi:10.1016/j.cell.2018.08.021. READ (summary).
18. Honegger KS, Campbell RAA, Turner GC (2011) Cellular-resolution population imaging reveals robust sparse coding in the *Drosophila* mushroom body. *J Neurosci* 31(33):11772–11785. doi:10.1523/JNEUROSCI.1099-11.2011. READ.
19. Lin AC, Bygrave AM, de Calignon A, Lee T, Miesenböck G (2014) Sparse, decorrelated odor coding in the mushroom body enhances learned odor discrimination. *Nat Neurosci* 17(4):559–568. doi:10.1038/nn.3660. READ (introduction).
20. Liu X, Davis RL (2009) The GABAergic anterior paired lateral neuron suppresses and is suppressed by olfactory learning. *Nat Neurosci* 12(1):53–59. doi:10.1038/nn.2235. ABSTRACT only.
21. Caron SJC, Ruta V, Abbott LF, Axel R (2013) Random convergence of olfactory inputs in the *Drosophila* mushroom body. *Nature* 497:113–117. doi:10.1038/nature12063. READ.
22. Turner GC, Bazhenov M, Laurent G (2008) Olfactory representations by *Drosophila* mushroom body neurons. *J Neurophysiol* 99(2):734–746. doi:10.1152/jn.01283.2007. READ by the validation literature check (`firing_rate_bounds_v2.json`, PR #15): spontaneous 0.1 ± 0.4 spikes/s; an odour evokes spikes in 6 ± 5 % of KCs; responders fire 4.9 ± 3.0 (α′β′) and 2.2 ± 1.2 (αβ) spikes in 0–2 s.
23. Hige T, Aso Y, Rubin GM, Turner GC (2015) Plasticity-driven individualization of olfactory coding in mushroom body output neurons. *Nature* 526:258–262. doi:10.1038/nature15396. READ (no rates in the text).
24. Berry JA, Cervantes-Sandoval I, Nicholas EP, Davis RL (2012) Dopamine is required for learning and forgetting in *Drosophila*. *Neuron* 74(3):530–542. doi:10.1016/j.neuron.2012.04.007. ABSTRACT only.
25. Okray Z, et al. (2025) T-maze aversive conditioning protocol. *Cold Spring Harb Protoc*. doi:10.1101/pdb.prot108566. SECONDARY, for the protocol wording only.
26. Bennett JEM, Philippides A, Nowotny T (2021) Learning with reinforcement prediction errors in a model of the *Drosophila* mushroom body. *Nat Commun* 12:2569. doi:10.1038/s41467-021-22592-4. Related model.
27. Springer M, Nawrot MP (2021) A mechanistic model for reward prediction and extinction learning in the fruit fly. *eNeuro* 8(3):ENEURO.0549-20.2021. doi:10.1523/ENEURO.0549-20.2021. Related model.
28. Jiang L, Litwin-Kumar A (2021) Models of heterogeneous dopamine signaling in an insect learning and memory center. *PLoS Comput Biol* 17(8):e1009205. doi:10.1371/journal.pcbi.1009205. Related model.
29. Shiu PK, et al., Scott K (2024) A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634:210–219. doi:10.1038/s41586-024-07763-9. The LIF basis of this engine; no plasticity.

30. Mao Z, Davis RL (2009) Eight different types of dopaminergic neurons innervate the *Drosophila* mushroom body neuropil. *Front Neural Circuits* 3:5. doi:10.3389/neuro.04.005.2009. READ.
31. Berry JA, Phan A, Davis RL (2018) Dopamine neurons mediate learning and forgetting through bidirectional modulation of a memory trace. *Cell Rep* 25(3):651–662. doi:10.1016/j.celrep.2018.09.051. READ.
32. Villar ME, et al. (2022) *Curr Biol* 32:4576. doi:10.1016/j.cub.2022.08.058. ABSTRACT only.
33. Riemensperger T, Völler T, Stock P, Buchner E, Fiala A (2005) Punishment prediction by dopaminergic neurons in *Drosophila*. *Curr Biol* 15:1953–1960. doi:10.1016/j.cub.2005.09.042. ABSTRACT only.
34. Takemura S, et al. (2017) A connectome of a learning and memory center in the adult *Drosophila* brain. *eLife* 6:e26975. doi:10.7554/eLife.26975. READ.
35. Manoim JE, et al. (2022) Lateral axonal modulation is required for stimulus-specific olfactory conditioning in *Drosophila*. *Curr Biol* 32:4438. doi:10.1016/j.cub.2022.09.007. READ.
36. Barnstedt O, et al., Waddell S (2016) Memory-relevant mushroom body output synapses are cholinergic. *Neuron* 89(6):1237–1247. doi:10.1016/j.neuron.2016.02.015. READ.
37. Bielopolski N, et al. (2019) Inhibitory muscarinic acetylcholine receptors enhance aversive olfactory learning in adult *Drosophila*. *eLife* 8:e48264. doi:10.7554/eLife.48264. READ.
38. Meschi E, et al. (2024) *Neuron* 112:2315. doi:10.1016/j.neuron.2024.04.035. READ; cited for SEZON01→PPL101 and its need in shock learning.
39. Galili DS, et al. (2014) Converging circuits mediate temperature and shock aversive olfactory conditioning in *Drosophila*. *Curr Biol* 24:1712. doi:10.1016/j.cub.2014.06.062. ABSTRACT only.
40. Otto N, et al. (2020) Input connectivity reveals additional heterogeneity of dopaminergic reinforcement in *Drosophila*. *Curr Biol* 30:3200. doi:10.1016/j.cub.2020.05.077. READ.
41. Haynes PR, Christmann BL, Griffith LC (2015) A single pair of neurons links sleep to memory consolidation in *Drosophila melanogaster*. *eLife* 4:e03868. doi:10.7554/eLife.03868. READ.
42. Eckstein N, et al. (2024) Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187:2574. doi:10.1016/j.cell.2024.03.016. READ.
43. Keene AC, et al., Waddell S (2004) Diverse odor-conditioned memories require uniquely timed dorsal paired medial neuron output. *Neuron* 44:521. doi:10.1016/j.neuron.2004.10.006. ABSTRACT only.
44. Keene AC, Krashes MJ, Leung B, Bernard JA, Waddell S (2006) *Drosophila* dorsal paired medial neurons provide a general mechanism for memory consolidation. *Curr Biol* 16:1524. doi:10.1016/j.cub.2006.06.022. ABSTRACT only.
45. Lee PT, et al. (2011) Serotonin-mushroom body circuit modulating the formation of anesthesia-resistant memory in *Drosophila*. *PNAS* 108:13794. doi:10.1073/pnas.1019483108. ABSTRACT only.
46. Waddell S, Armstrong JD, Kitamoto T, Kaiser K, Quinn WG (2000) The amnesiac gene product is expressed in two neurons in the *Drosophila* brain that are critical for memory. *Cell* 103:805. doi:10.1016/S0092-8674(00)00183-5. ABSTRACT only.

No peer-reviewed whole-brain LIF model with MB learning was found for
2024–2026.
