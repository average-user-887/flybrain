# Project NeuroFly — 14-Paradigm Capability Matrix

**Version**: 1.0 · **Date**: 2026-09-24 · **Reference**: `docs/ROADMAP.md` Phase 4 (Step 4.1)

---

## Preamble & Ground Truth Standards

This matrix documents the verified capabilities of Project NeuroFly across all 14 classical *Drosophila* neuroethology paradigms.

In accordance with the project's strict scientific honesty policy:
- **✅ Working / Mapped**: Pathway verified in code and validated in unit/integration tests or physics co-simulation.
- **⚠️ Partial / Tested**: Implemented and probed, but exhibiting specific physiological limitations or under ongoing recalibration.
- **⬜ Unmapped / In Progress**: Biological pathway declared in connectome literature but sensory ingress or motor readout is not yet frozen in this codebase.
- **N/A**: Not applicable (e.g. reflex assays that do not undergo experience-dependent synaptic plasticity).

---

## 14-Paradigm Capability Matrix

| # | Paradigm | Primary Sensory Ingress (Cell Types) | Descending Motor Readout | Modular Controller | Connectome (MaleCNS v1.0, v3) | Plasticity / Learning | Evidence / Receipts |
|---|---|---|---|---|---|---|---|
| 1 | **Open Arena** (`open-arena`) | Multimodal (Antennal PNs, ommatidia, JON-C/E wind) | DNa02 (yaw torque), DNp09 (thrust) | ✅ Working | ✅ Validated (causal steering) | ✅ Active (MB KC→MBON) | `docs/receipts/lif_dynamics_v3.json`, `runs/embodied-video/body.mp4` |
| 2 | **T-Maze** (`t-maze`) | Bilateral Olfaction (54 AL PNs: DL5, DM1, DM2, DP1m) | DNa02 (odor avoidance), DNp09 (thrust) | ✅ Working | ✅ Mapped (PN→MB/LH→DNa02) | ✅ Active (MB KC→MBON) | `tests/test_sensory_ingress_advanced.py`, `tests/test_maze.py` |
| 3 | **Y-Maze** (`y-maze`) | Bilateral Olfaction (PNs) + Antennal Mechanosensory | DNa02 (turn bias), DNp09 | ✅ Working | ✅ Mapped (CX bump→DNa02) | ⚠️ Working-memory trace | `tests/test_maze.py`, `tests/test_containment_all_paradigms.py` |
| 4 | **Heat-Maze** (`heat-maze`) | Antennal Thermoreception (Gr28b, TRPA1, BmPr) | DNa02 (thermal turn), DNp09 | ✅ Working | ✅ Mapped (Thermo→SEZ→DNa02) | ⚠️ Non-associative | `tests/test_assay_responses.py`, `assay_controls.py` |
| 5 | **Buridan's Paradigm** (`buridan`) | Visual Vertical Stripes (medulla Tm, lobula LC) | DNa02 (orientation), DNa01 (course hold) | ✅ Working | ✅ Mapped (Visual→CX→PFL3→DNa02) | ✅ Heading map (WP6) | `tests/test_assay_responses.py`, `docs/WP6_PLASTICITY_SPEC.md` |
| 6 | **Visual Operant** (`visual-operant`) | Visual Quadrants + Thermal Reinforcement | DNa02 (yaw torque → drum displacement) | ✅ Working | ✅ Mapped (Optic→DNa02 + Thermal shock) | ⚠️ Operant torque | `tests/test_assay_responses.py`, `web/live_assays.js` |
| 7 | **Wind Tunnel** (`wind-tunnel`) | Johnston's Organ (JON-C/E) → Wedge + Olfactory PNs | DNa02 (casting yaw), DNp09 (upwind surge) | ✅ Working | ✅ Mapped (JON→WED→WPN→DNp09/DNa02) | ✅ Odor-gated anemotaxis | `tests/test_whole_brain.py`, `connectome_bridge.py` |
| 8 | **Looming Escape** (`looming-escape`) | Optical Expansion (Lobula Col4 / LPLC2) | Giant Fiber (`DNp01` / GF ballistic takeoff) | ✅ Working | ✅ Mapped (LPLC2/LC4→GF monosynaptic) | N/A (Innate reflex) | `tests/test_whole_brain.py`, `connectome_bridge.py` |
| 9 | **Optomotor Drum** (`optomotor`) | Retinal Slip (T4a/T5a front-to-back, T4b/T5b back-to-front) | DNa02 (ipsilateral compensatory yaw) | ✅ Working | ✅ Validated (WP5 / Phase 1 & 2) | ✅ Saccadic efference copy | `docs/WP5_OPTOMOTOR.md`, `tests/test_wp5_live_loop.py` |
| 10 | **Gap Crossing** (`gap-crossing`) | Foreleg FeCO (joint angle) + CS (cuticular load) | CPG stepping cadence & elevation | ✅ Working | ✅ Mapped (FeCO/CS→CPG gait) | N/A (Biomechanics) | `tests/test_biomechanics_closed_loop.py`, `tests/test_embodied_decoder.py` |
| 11 | **Circadian DAM** (`circadian-dam`) | Photoperiod (Visual eyelets / CRY) | BPN & DNp09 (arousal / locomotion cadence) | ✅ Working | ✅ Mapped (s-LNv clock→DN gating) | ⚠️ Diurnal turnover | `tests/test_assay_responses.py`, `maze.py` |
| 12 | **Courtship Chamber** (`courtship`) | Visual Target + Male Pheromone cVA (Or67d PNs) | P1 command cluster → DNa02 / Wing vibration | ✅ Working | ✅ Mapped (Or67d→DA1→LH→P1 pursuit) | ⚠️ Song suppression | `tests/test_assay_responses.py`, `maze.py` |
| 13 | **Labyrinth Maze** (`labyrinth`) | Antennal Touch + Contact Normal Vectors | DNa02 (Coulomb wall sliding), DNp09 | ✅ Working | ✅ Mapped (Mechanosensory→DNa02) | ⚠️ Spatial memory | `tests/test_collision_physics.py`, `tests/test_maze.py` |
| 14 | **Multisensory Sandbox** (`multisensory-sandbox`) | Full Composite (Visual, Odor, Thermal, Wind, CS) | Full Motor Complement (DNa02, DNp09, MDN, GF) | ✅ Working | ✅ Validated (Multi-rate co-sim) | ✅ Multi-modal learning | `tests/test_multisensory_benchmark.py`, `tests/test_embodied_telemetry.py` |

---

## Batch Architectures & Shared Pathways (Step 4.2)

### Batch A: Visual-Motor Pathway
* **Paradigms**: Optomotor, Buridan, Looming Escape, Visual Operant
* **Sensory Route**: Photoreceptors $\to$ Lamina (L1–L5) $\to$ Medulla (Mi1, Tm3, Tm1, Tm2, Tm4, Tm9) $\to$ Lobula Plate (T4a–d, T5a–d) and Lobula Columnar (LC4, LPLC2).
* **Motor Readout**:
  * Fine yaw stabilization & orientation: `DNa02` (bilateral ipsilateral steering).
  * Ballistic looming escape: `DNp01` (Giant Fiber, monosynaptic from LC4/LPLC2).

### Batch B: Olfactory & Mechanosensory Pathway
* **Paradigms**: T-Maze, Y-Maze, Wind Tunnel, Courtship
* **Sensory Route**:
  * Olfactory: Antennal ORNs $\to$ 54 AL Glomeruli $\to$ Cholinergic Projection Neurons (PNs) $\to$ Lateral Horn (innate valence) and Mushroom Body Calyx (associative learning).
  * Wind / Anemotaxis: Johnston's Organ (JON-C/E) $\to$ Antennal Mechanosensory and Motor Center (AMMC) / Wedge (WED) $\to$ Wedge Projection Neurons (WPNs).
* **Motor Readout**:
  * Bilateral steering away from aversive or toward attractive plume: `DNa02`.
  * Forward surging drive along wind vector: `DNp09`.
  * Courtship song & pursuit: `P1` cluster.

### Batch C: Thermal & Spatial Pathway
* **Paradigms**: Heat-Maze, Gap Crossing, Circadian DAM
* **Sensory Route**:
  * Thermal: Antennal cold/warm sensory neurons $\to$ Subesophageal Zone (SEZ) / Posterior Lateral Protocerebrum.
  * Leg Proprioception: Femoral Chordotonal Organ (FeCO) & Campaniform Sensilla (CS) $\to$ Thoracic neuromeres $\to$ Ascending mechanosensory feedback.
  * Circadian: Light photoperiod $\to$ Small ventral lateral neurons (s-LNv) expressing Pigment Dispersing Factor (PDF).
* **Motor Readout**:
  * Thermal avoidance yaw: `DNa02`.
  * Chasm reach & step elevation: Kuramoto-Hopf CPG amplitude and phase coupling.
  * Locomotor activity bouts: `BPN` / `DNp09` tonic arousal.

### Batch D: Complex & Composite Assays
* **Paradigms**: Open Arena, Labyrinth, Multisensory Sandbox
* **Sensory Route**: Simultaneous concurrent multisensory ingress across all primary modalities.
* **Motor Readout**: Multi-channel arbitration between obstacle sliding reflexes, foraging pursuit, and escape overrides.

---

## Scientific Plasticity Protocol (Step 4.3 — WP6)

* **Plastic Synapse Subset**: Visually driven ring neuron to compass neuron connections (`ER4d` + `ER2` $\to$ `EPG`).
* **Edge Count**: **3,081 directed synapses** (0.012% of the 25.58M connectome edges).
* **Modulatory System**: Octopaminergic `EL` cluster gating depression-only STDP.
* **Behavioral Readout**: Compass bump azimuth stabilization and landmark visual anchoring.
