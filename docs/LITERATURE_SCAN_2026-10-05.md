# Literature scan, mid-2025 to 5 October 2026

A worker retrieved every source below on 5 October 2026. Each item is labelled:
- **PEER-REVIEWED**, **PREPRINT**, **CONFERENCE** or **CODE/REPO** for the kind of source;
- **full text**, **abstract** or **search snippet** for how much was read.

Search-snippet items are weak. The orchestrator spot-checked the claims that drive
decisions (§A1 and §C) before acting on them. Before citing anything from here, check
whether it has been confirmed, and retrieve the paper first if it hasn't. The earlier
scan with per-claim verification is `docs/LITERATURE_BENCHMARKS.md`.

---

## A. The most important new developments for NeuroFly

1. **About half the MaleCNS lamina gets no photoreceptor input.** The worker checked
   our local `connectome_data/malecns_v1/` read-only:
   - There are 3,377 R1-6 bodies, against roughly 10,600 expected (about 1,770
     columns per side × 6).
   - Lamina cells with **zero** R1-6 synapses: L1 54.4 % (967/1,776), L2 54.2 %,
     L3 55.9 %.
   - Where the input exists it is large: a median of 123 R→L1 synapses.
   - Nern et al. 2025 (Nature, doi 10.1038/s41586-025-08746-0) say the data covers
     every optic-lobe neuropil "except for the most peripheral neuropil, the lamina",
     and that R1-6 are undercounted.
   - An independent project (brainfly) reports 962 of 1,769 columns without
     photoreceptor input.

   **Consequence:** the ~500× signal loss from photoreceptors to T4/T5 that we
   measured mixes columns that receive the stimulus with roughly 55 % that cannot.
   Part of it is a gap in the data, not physiology.

   **VERIFIED by the orchestrator on 5 October 2026**, on the hash-pinned engine
   graph (`outputs/brainlab/malecns_v1/graph.npz`).
   - The photoreceptor label is `R1-R6`; there are 3,377 cells.
   - Of those, 3,359 have outputs, 14,785 out-edges in total. That matches v4's
     "all 14,785" in `docs/WP5_OPTOMOTOR.md` §13.
   - Cells with zero R1-R6 input: L1 967/1,776 = 54.4 %, L2 965/1,779 = 54.2 %,
     L3 990/1,772 = 55.9 %.
   - Median input weight where present: 33.8 for L1 (about 123 synapses at
     0.275 per synapse), 34.5 for L2, 6.3 for L3.
   - A first attempt matched the label `R1-6` and so selected no cells at all. The
     label must be matched exactly.
2. **Borst 2025** (J Comput Neurosci 53:507–520, PMC12672718, PEER-REVIEWED, full text)
   modelled 65 cell types with conductance-based synapses. In that model, the
   transient-versus-sustained split between Mi1/Tm3/Tm1/Tm2 is "inherited from
   first-order interneurons", through an **H-current in L1/L2**. Without it, "all
   model neurons now responded in a sustained way".
3. **Pang et al. 2025** (Current Biology 35:333, PEER-REVIEWED) found a second
   lamina mechanism. Recurrent feedback that depends on L2 makes the L1/L2
   responses biphasic, while photoreceptors respond in a single phase. Plain LIF has
   neither this nor the H-current.
4. **The only other published whole-brain LIF optic-lobe model also failed at
   T4/T5.** Liew et al. (arXiv 2609.01330, Sep 2026, PREPRINT) report "most T4 and
   T5 neurons had firing rates close to zero", even though they drove L1–L3
   directly. Our negative result is not unusual.
5. **Wiring alone underdetermines function.**
   - Beiran & Litwin-Kumar 2025 (Nature Neuroscience, doi
     10.1038/s41593-025-02080-4): about 7–10 recorded neurons remove the ambiguity
     for low-dimensional dynamics.
   - Lappalainen 2024: models with random parameters get contrast polarity right
     but give "poor predictions of direction selectivity".
   - Karaneen/Schomburg/Chklovskii 2026 (PREPRINT): even task-fitted models are
     unstable.
6. **Pugliese, Tuthill, Brunton et al. 2025** (bioRxiv 10.1101/2025.09.12.675944,
   PREPRINT, full text) is the closest published template for the owner's
   "connectome does the work" rule.
   - It finds a walking central pattern generator in nerve-cord connectomes **with no
     fine-tuning**. Each neuron's parameters are drawn from distributions based on
     physiology, over many replicates.
   - Gain and threshold are normalised by neuron size, which the authors call
     "crucial".
   - Two predictions were confirmed by optogenetics.
7. **Embodied "whole-fly" claims elsewhere rely on hand-built decoders and trained
   body controllers.** Eon Systems (Mar 2026) say so themselves: "no quantitative
   validation". Brunton et al. 2026 made a *worm* connectome produce realistic fly
   walking through a trained interface and called it "biologically meaningless".
   Realistic-looking walking is not evidence.
8. **Our base software has changed.**
   - FlyGym 2.0.0 (2 Apr 2026) was a rewrite with a new API; 2.1.0 followed on
     24 Jun 2026.
   - flyvis v1.2.0 was released on 6 Aug 2026.
   - MaleCNS v1.0 was released on 8 Jun 2026. Its paper is Berg et al., Cell,
     3 Sep 2026, doi 10.1016/j.cell.2026.08.015 (from a search snippet).

## B. Findings by question (condensed; labels as retrieved)

**Whole-brain simulation and closed-loop bodies**
- **Eon Systems post** (Mar 2026; full post read):
  - Shiu-style LIF on FlyWire, with the Lappalainen visual model "piped in".
  - Hand-picked descending neurons drive the body; the body controllers were trained
    by imitation learning.
  - Their code is brain-only (GPL-2.0).
  - A "91 % embodied accuracy" figure seen elsewhere is not supported by the post.
- **Jin et al.** (arXiv 2602.17997, PREPRINT):
  - FlyWire weights fixed; trainable non-spiking descriptors; deep reinforcement
    learning on the flybody body.
  - Treats glutamate and histamine as *excitatory*.
  - Turning error: 8.29° with the real connectome, 13.55° rewired, 125° with a
    random graph.
  - "An SNN baseline failed catastrophically." This is training, not testing.
- **Pugliese 2025** (see A6):
  - Rate model with rectified tanh, on MANC (4,604 neurons), replicated on FANC,
    MaleCNS and BANC.
  - Parameters: gain 1±0.1, threshold 7.5±0.6, maximum rate 200±10 Hz, τ 20±2 ms.
  - Gain is divided by median-normalised volume and threshold multiplied by it.
  - Optogenetic tests: p 0.032–0.042 (n=7) and p 0.004–0.036 (n=10).
- **Others (PREPRINTS):**
  - Wang et al. (Sandia, arXiv 2508.16792) ran the FlyWire LIF on Loihi 2: about
    82× faster than Brian2 on the sugar experiment, validated only against Brian2.
  - An (bioRxiv 10.64898/2026.09.19.752860) pruned the Shiu model: 79 % of feeding
    output survives with 6 % of the edges.
  - Chen & Xi (Dec 2025): escape suppressing feeding; abstract only.
- **GitHub projects** (one checked against its code, the rest from READMEs):
  - **brainfly**, cloned and checked: T4/T5 direction selectivity comes *only*
    from porting flyvis's motion-trained optic lobe. Its own LIF eye failed.
  - **Fly.exe** zeroes all 66,533 photoreceptor output edges and injects light one
    synapse downstream.
  - **therealfly** reports a nerve-cord oscillation that 5 of 6 scrambled
    connectomes reproduce.
  - **FLYCNS** forces graded neurons to spike.
  - **None of these** gets T4/T5 direction selectivity from an unfitted connectome.

**Inferring unknown parameters**
- **Beiran & Litwin-Kumar 2025** (above): "single-neuron parameters are often not
  recovered accurately" even when the activity fits.
- **Li, Ping, Zhang, Wang 2026** (bioRxiv 10.64898/2026.08.21.745055, PREPRINT):
  - Rate model on FlyWire, with weights and per-neuron τ fitted to whole-brain
    calcium imaging from one fly.
  - Held-out correlation r 0.423, against a ceiling of 0.473.
- **Duan/Dong/Fiete 2025** (PREPRINT, abstract): a head-direction ring attractor
  from cell-type-level parameters inferred with self-supervision.
- **Zhou & Hasler 2026** (PREPRINT, full results):
  - Connectome-constrained versus random networks, **untrained**: r 0.26 and 0.22.
    Neither survives correction for multiple comparisons.
  - Comparison with biological T4/T5 tuning: uninformative (r −0.015).
  - The abstract claims more than its results show.
- **Currier & Clandinin 2025** (Cell, PEER-REVIEWED; abstract and snippet only):
  connectome predictions are good for orientation tuning and "surprisingly poor"
  for receptive-field size. "Physiology is a stronger predictor of wiring than
  wiring is of physiology."
- **Lappalainen 2024** (foundational; methods read):
  - 734 fitted parameters, with τ initialised at 50 ms.
  - Models with random parameters fail at direction selectivity.

**Graded transmission in large models**
- **Borst 2025**:
  - Output rectified at −50 mV; every cell has τ 40 ms.
  - Leak reversal −50 mV, except **L1–L3 at −20 mV**, i.e. a tonically
    depolarised lamina.
  - 130 fitted gains.
- **flyvis** uses threshold-linear graded release. Pugliese uses a rectified-tanh
  rate model; Li et al. use ReLU.
- **Pang 2025** fitted L2 with τ 13.1 ms, feedback τ 561 ms and feedback weight 12.7.
- No peer-reviewed whole-brain model mixing spiking and graded neurons was found.

**Fly motion vision**
- **Borst 2025** (see A2):
  - Fitted to calcium-imaging receptive fields of 13 *input* types (white noise,
    not motion): about 7 % cost passive, about 3 % with the H-current.
  - Fitted H-conductance ranks L1 ≫ L2 > L4, L5 ≫ L3.
  - Predicts that HCN knockdown impairs T4/T5 direction selectivity; direction
    selectivity itself was **not tested**.
  - Code: github.com/axelborst/temporal_filtering.
- **Pang 2025** (see A3).
- **Henning/Silies 2025** (PREPRINT, eLife reviewed preprint 108529; abstract only):
  - C2/C3 GABAergic feedback is required for T4/T5 direction selectivity, and
    sharpens ON responses.
- **Groschner et al. 2022** (Nature, foundational):
  - Direction selectivity comes from cholinergic Mi1/Tm3 excitation coinciding with
    release from Mi9 GluClα inhibition.
  - Mi9 releases tonically in darkness: removing GluClα depolarises T4 by 12 mV
    at rest.
- **Arenz 2017** (snippet): Mi1/Tm3 are fast and band-pass, Mi4/Mi9 slow and
  low-pass. "Peak delay timing is not sufficient": the full filter shape matters.
- **Not found:** any new measurement of the Mi1-versus-Tm3 lag since mid-2025, and
  **any T4/T5 direction selectivity from an unfitted connectome model**.

**Datasets and resources**
- **MaleCNS:**
  - v0.9 on 5 Oct 2025, v1.0 on 8 Jun 2026.
  - Per-T-bar transmitter predictions available (2.7 GB). **No gap junctions.**
  - 166,691 neurons, 11,691 types.
- **Local check of transmitter labels:**
  - R1-6 histamine, L1 glutamate, L2/L3 acetylcholine, Mi1/Tm3 acetylcholine, Mi4
    GABA, Mi9 glutamate, Tm1/Tm2/Tm9 acetylcholine, CT1 GABA, C2/C3 GABA.
  - These agree with the ground-truth labels.
- **Nern 2025:** the lamina is excluded. About 50 optic-lobe types have aminergic
  predictions far above plausible numbers, so **serotonin labels in the optic lobe
  are unreliable**.
- **BANC** (Nature, 8 Jun 2026, doi 10.1038/s41586-026-10735-w; snippet).
  synister_banc v1.0.4 (Zenodo 10.5281/zenodo.20350570).
- **Gauthey et al. 2026** (Nature Communications, doi 10.1038/s41467-026-72437-1):
  - Whole-brain imaging at 28 volumes/s, about 48,000 regions of interest.
  - Data on Zenodo: 10.5281/zenodo.17613016 (raw) and 10.5281/zenodo.17618684
    (processed).
- No adult electron-microscopy gap-junction map exists that the worker could find.

## C. What this means for NeuroFly's two blockers

1. **Signal loss.** Part of it is structural (the incomplete lamina). Any
   population-averaged amplitude from photoreceptors to T4/T5 includes columns that
   get no stimulus.
   - Others work around this by injecting at L1–L3 (Liew), injecting one synapse
     downstream (Fly.exe), or filling in the missing input (brainfly).
   - Separately, a depolarised lamina operating point (Borst: −20 mV leak reversal
     in L1–L3) lets graded release move in both directions. A threshold at or above
     rest would clip hyperpolarising signals.
2. **Input timing.** New work places the fast/slow split in the lamina (H-current,
   recurrent feedback), not in receptor kinetics and not mainly in the Mi1-versus-Tm3
   lag. With identical passive cells and no lamina H-current, every downstream cell
   should be sustained, which is consistent with our wrong ordering.
3. **Our "measured parameters only" approach.** The field's direction points to a
   real ceiling: no study gets T4/T5 direction selectivity from an unfitted
   connectome. Two encouraging exceptions:
   - Pugliese: parameters drawn from physiological distributions plus size
     normalisation, with no tuning, gives a validated circuit.
   - Borst: fits only *input* neurons to non-motion stimuli, which arguably counts
     as independent measurement.

   Whether that kind of fit is allowed is an **owner decision**.

## D. Recommendations recorded by the scan

- **Read in full:** Borst 2025, Pang 2025, Pugliese 2025, Beiran & Litwin-Kumar
  2025, Currier & Clandinin 2025, and Nern 2025 (its lamina and data-limitation
  sections), plus Groschner 2022.
- **Report signal loss per column,** restricted to columns that receive
  photoreceptor input. (Sent to the v6a worker on 5 Oct 2026.)
- **Candidates to import without breaking the owner's constraint**, each needing an
  independent source:
  - an L1/L2 H-current;
  - a depolarised lamina operating point;
  - ensembles drawn from physiological distributions, with size normalisation.
- **Things in our model to check:**
  - population-mean metrics on MaleCNS vision;
  - uniform passive properties;
  - treating the Mi1-versus-Tm3 lag as the main timing defect;
  - trusting optic-lobe serotonin, dopamine and octopamine labels.
- **Tooling:** decide on a move to FlyGym 2.x.

## E. Looked for but not found or not verified

- New measurements of Mi1/Tm3 timing or of input-cell intrinsic properties.
- The full text of Vieira et al. (PNAS 2026; title only).
- Any T4/T5 direction selectivity from an unfitted connectome.
- Per-transmitter accuracy for MaleCNS or BANC.
- Any adult electron-microscopy gap-junction map.
- The full text of the BANC paper.
- The authors of the Cell 2026 "organization of visual pathways" paper.
- The full text of flybody.
- New photoreceptor-to-lamina gain measurements (only a light search, so as not to
  duplicate v6a).
- The Liew et al. venue.
- The cell-type count in Currier & Clandinin (43 or 91, depending on version).
- The full text of Henning in eLife.
- The "20fly8" hybrid project.
