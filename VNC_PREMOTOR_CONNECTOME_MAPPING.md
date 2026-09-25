# MANC & VNC PREMOTOR CONNECTOME MAPPING: DESCENDING-TO-MOTOR CIRCUITS & PROPRIOCEPTIVE FEEDBACK
## Comprehensive Neurobiological Specification and Biophysical Connectome Blueprint for Drosophila Leg Motor Control

---

### Executive Summary

The Ventral Nerve Cord (VNC) of *Drosophila melanogaster* is the functional and anatomical analogue of the vertebrate spinal cord, containing approximately **15,800 central neurons, 6,500 sensory axons, and ~74 million post-synaptic sites** (MANC Connectome, Takemura et al. 2024, Marin et al. 2024, Cheong et al. 2024). While the central brain (Protocerebrum, Central Complex, Mushroom Body) computes navigation decisions, valence representations, and motor intentions, the execution of terrestrial walking, grooming, jumping, and flight steering is mediated by modular premotor microcircuits partitioned across thoracic leg neuromeres (Prothoracic T1, Mesothoracic T2, Metathoracic T3).

A central architectural discovery of the complete synaptic-resolution MANC connectome is that **direct descending neuron (DN) to motor neuron (MN) synapses are remarkably rare (< 8% of descending synaptic output)**. Instead, descending commands converge onto developmental **hemilineage-derived premotor interneuron communities** that:
1. Pool descending command channels with local sensory feedback from proprioceptors (Femoral Chordotonal Organ [FeCO] and Campaniform Sensilla [CS]).
2. Coordinate reciprocal agonist-antagonist joint flexion/extension via ipsilateral GABAergic interneurons (**Hemilineage 13A**).
3. Enforce inter-limb phase locking, tripod gait stability ($\Delta\Phi = \pi$), and bilateral symmetry breaking via midline-crossing contralateral GABAergic interneurons (**Hemilineage 13B**).
4. Relay behavioral state transitions, locomotion pace, and motor efference copies back to the brain via Ascending Neurons (ANs).

This document establishes the rigorous connectomic, synaptic, and mathematical specification mapping descending commands (DNa02, DNa01, DNp09, BPN, MDN, DNp01/GF) into VNC leg motor control.

---

## 1. Descending Command Ingress to Thoracic Neuromeres (T1, T2, T3)

### 1.1 The Premotor Bottleneck: Indirect vs. Direct Connectivity

In the adult male VNC (MANC v1.0), over 1,300 descending neuron axons enter the cervical connective from the brain. Tracing their post-synaptic targets reveals a strictly layered organization:

```
[Brain: CX (PFL3/PFL2) + MB (MBONs) + Vision (LPTCs/LCs)]
                          │
                          ▼
           [Lateral Accessory Lobe (LAL)]
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
[Steering DNs: DNa02, DNa01]       [Locomotion DNs: DNp09, BPN, MDN, DNp01]
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼  (Cervical Connective)
      =======================================================
                         VNC Neuromeres
      =======================================================
                          │
        ┌─────────────────┴─────────────────┐
        ▼ (~8% direct)                      ▼ (>92% indirect)
  [Leg Motor Neurons]               [Premotor Interneurons]
  (Ti-Ext, Ti-Flex,                 - Hemilineage 13A (Ipsilateral GABA)
   Tr-Lev, Tr-Dep,                  - Hemilineage 13B (Contralateral GABA)
   Co-Pro, Co-Ret)                  - Hemilineage 9A  (Excitatory Cholinergic)
        ▲                                   │
        └───────────────────────────────────┘
                          ▲
                          │ Proprioceptive Feedback
            [FeCO (Claw, Hook) & CS Sensilla]
```

### 1.2 Target Neuromere Distribution of Primary Descending Lines

| Descending Neuron Class | Neurotransmitter | Primary Target Neuromeres | Primary Premotor Hemilineage Targets | Functional Behavioral Role |
|:---|:---|:---|:---|:---|
| **DNa02** (L/R) | Acetylcholine | T1, T2 (ipsilateral bias) | 13A, 9A, 19A | High-gain asymmetric steering yaw torque; modulates inside leg stance duration and outside leg swing amplitude |
| **DNa01** (L/R) | Acetylcholine | T1, T2, T3 | 13A, 8A | Low-gain sustained course stabilization; counteracts drift |
| **DNp09** (P9) | Acetylcholine | T1, T2, T3 (bilateral) | 9A, 11A, 24A | Pursuit forward locomotion; increases stepping frequency (up to 14 Hz) and propulsive thrust |
| **BPN** (Brain Peduncular) | Acetylcholine | T1, T2, T3 | 9A, 13B | Baseline exploration cadence; modulated by metabolic state (hunger boosts BPN via sNPF/dopamine) |
| **MDN** (Moonwalker) | Acetylcholine | T1, T2, T3 (pan-thoracic) | 13A, 13B, 10B | Reversal of metachronal stepping wave (T3→T2→T1); activates backwards walking while shunting forward CPG |
| **DNp01** (Giant Fiber) | Acetylcholine | T2 (Mesothoracic neuromere) | TTMn (Tergo-trochanteral MN), PSI (Peripherally Synapsing Interneuron) | Ballistic looming escape jump; simultaneous middle leg extension and wing depression |

---

## 2. VNC Premotor Circuitry: Hemilineages as Functional Building Blocks

### 2.1 Developmental Hemilineage Logic

Neurons in the Drosophila VNC are generated by a stereotyped set of ~30 neuroblasts (NBs) per hemisegment. Each neuroblast divides asymmetrically to produce ganglion mother cells (GMCs), which divide once into two sibling post-mitotic neurons, defining two **hemilineages** (A and B) under the control of Notch signaling (Notch-ON vs. Notch-OFF):

```
                        Neuroblast (NB)
                              │
                    Ganglion Mother Cell (GMC)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        Hemilineage A                 Hemilineage B
          (Notch-ON)                   (Notch-OFF)
```

### 2.2 Hemilineage 13A: Ipsilateral Agonist-Antagonist Reciprocal Inhibition
- **Origin**: Neuroblast NB4-2 (Notch-ON sibling).
- **Neurotransmitter**: **GABA** (Inhibitory, ionotropic $Rdl$ / metabotropic $GABA_B$).
- **Projection**: Confined strictly to the **ipsilateral leg neuropil** of the corresponding thoracic neuromere.
- **Synaptic Connectivity**:
  - Receives direct excitation from descending steering lines (DNa02, DNa01) and sensory feedback from the Femoral Chordotonal Organ (FeCO).
  - Directly synapses onto antagonist motor neuron pools:
    $$\text{13A}_{\text{ext}} \xrightarrow{\text{GABA}} \text{MN}_{\text{flexor}}, \quad \text{13A}_{\text{flex}} \xrightarrow{\text{GABA}} \text{MN}_{\text{extensor}}$$
- **Computational Role**: Implements **reciprocal inhibition**. When an extensor motor neuron is recruited during the stance phase, 13A actively silences flexor motor neurons, preventing co-contraction and enabling compliant joint articulation.

### 2.3 Hemilineage 13B: Contralateral Inter-Limb Coordination & Stance Phase Locking
- **Origin**: Neuroblast NB4-2 (Notch-OFF sibling).
- **Neurotransmitter**: **GABA** (Inhibitory).
- **Projection**: Axons cross the midline through the **ventral accessory commissure (VAC)** and arborize profusely in the **contralateral leg neuropil**.
- **Synaptic Connectivity**:
  - Innervates the contralateral premotor 13A network and contralateral premotor rhythm generators.
  - Receives heavy monosynaptic input from ipsilateral Campaniform Sensilla (CS) load sensors.
- **Computational Role**: Implements **contralateral cross-inhibition (Cruse Rule 4 & Tripod Alternation)**.
  $$\frac{d\Phi_R}{dt} = \omega + K_{13B} \sin(\Phi_L - \Phi_R - \pi)$$
  When the left leg is under load (stance phase, active CS feedback), 13B sends strong inhibitory barrages across the commissure to the right leg's swing initiators, ensuring that contralateral homologous legs never swing simultaneously ($\Delta\Phi = \pi$).

### 2.4 Hemilineage 9A: Excitatory Drive and Rhythm Generation
- **Origin**: Neuroblast NB3-3.
- **Neurotransmitter**: **Acetylcholine** (Excitatory nicotinic nAChR).
- **Projection**: Local and intersegmental within the ventral leg neuropil.
- **Synaptic Connectivity**: Direct excitatory driver of leg motor neurons controlling the trochanter depressor (Tr-Dep, primary stance weight support) and tibia extensor (Ti-Ext).
- **Computational Role**: Core component of the local non-spiking Central Pattern Generator (CPG), sustaining tonic rhythmicity under descending neuromodulatory drive (DNp09/BPN).

---

## 3. Proprioceptive Feedback Microcircuits: FeCO and Campaniform Sensilla

### 3.1 Femoral Chordotonal Organ (FeCO) Functional Sub-Modalities

The FeCO inside the femur consists of ~150 sensory neurons classified into three functionally and anatomically segregated receptor types (Mamiya et al. 2018, Phelps et al. 2021):

```
       FeCO Sensory Receptor Clusters
       ┌─────────────────────────────┐
       │   Femur-Tibia (FT) Joint    │
       └──────────────┬──────────────┘
                      │
     ┌────────────────┼────────────────┐
     ▼                ▼                ▼
[Claw Neurons]  [Hook Neurons]   [Club Neurons]
  (Position)      (Velocity)      (Vibration)
     │                │                │
     │ Target:        │ Target:        │ Target:
     │ Ventral leg    │ Medial leg     │ Dorsal leg
     │ motor neurons  │ premotor 13A   │ acoustic /
     │ (reflex loops) │ (presynaptic   │ seismic ANs
     │                │  gating)       │
```

1. **Claw Neurons (Joint Position Receptors)**:
   - Firing rate is proportional to static femur-tibia angle $\theta_{\text{FT}}$ (range: $20^\circ\text{--}160^\circ$).
   - Divided into **Extension Claws** (active at large angles, $\theta > 100^\circ$) and **Flexion Claws** (active at acute angles, $\theta < 50^\circ$).
   - Form direct monosynaptic resistance reflex circuits: flexion claws excite extensor motor neurons, maintaining posture against gravity.

2. **Hook Neurons (Joint Velocity & Direction Receptors)**:
   - Fire phasically in response to angular velocity $\dot{\theta}_{\text{FT}}$, with strict direction-selectivity (extension-sensitive vs. flexion-sensitive).
   - Undergo **presynaptic GABAergic inhibition** from local 13A interneurons during voluntary saccades, shunting sensory feedback that would otherwise oppose rapid active movements.

3. **Club Neurons (Vibration & Substrate Mechanics)**:
   - High-frequency phase-locking ($100\text{--}1000\text{ Hz}$).
   - Project dorsally, feeding into ascending pathways (ANs) that alert the brain to approaching ground vibrations (predator footsteps).

### 3.2 Campaniform Sensilla (CS): Load Feedback and Cruse's Walknet Rules

Campaniform sensilla are dome-shaped cuticular mechanoreceptors located at points of high mechanical stress: trochanter, femur base, and pretarsal claws.

1. **Load Feedback Mechanism**:
   - Under ground reaction force $F_{\text{load}} > 0$, CS fire sustained spike trains ($50\text{--}250\text{ Hz}$).
2. **Cruse's Walknet Rule 1 (Stance Maintenance under Load)**:
   $$\text{Pr}(\text{Swing Initiation}) = \begin{cases} 0 & \text{if } F_{\text{CS}} > F_{\text{threshold}} \\ f(\Phi, \theta_{\text{PEP}}) & \text{if } F_{\text{CS}} \le F_{\text{threshold}} \end{cases}$$
   As long as a leg carries body weight ($F_{\text{CS}} > 0$), swing initiation is strictly vetoed via 13A/13B inhibitory loops, preventing catastrophic postural collapse.

---

## 4. Giant Fiber Escape Circuit: From Optic Tectum to Flight Takeoff

The Giant Fiber (GF / DNp01) system is the fastest known sensorimotor circuit in *Drosophila*, executing an unconditioned, all-or-none escape takeoff in **< 8 milliseconds** following visual looming detection:

```
[Visual Threat: Rapid Expanding Dark Loom]
                 │
                 ▼
[Lobula Collinear / Radial Detectors: LC4, LPLC2]
                 │
                 │ (Monosynaptic Chemical & Gap Junctions)
                 ▼
[Giant Fiber (DNp01 / Source IDs: 10001 R, 10010 L)]
                 │
                 │ (Giant axon: ~6-8 µm diameter, conduction > 5 m/s)
                 ▼  (Entering T2 Neuromere)
     ┌───────────┴───────────┐
     │                       │
     ▼ (Mixed Synapse)       ▼ (Electrical Gap Junction)
[Peripherally Synapsing   [Tergo-Trochanteral Motor Neuron (TTMn)]
 Interneuron (PSI)]                  │
     │                               ▼
     ▼ (Cholinergic)       [Middle Leg Trochanter Depressor Muscle]
[Dorsal Longitudinal                 │
 Motor Neurons (DLMn)]               ▼
     │                     [Rapid Extension -> Jump Takeoff (t = 0 ms)]
     ▼
[Wing Depressor Muscles]
     │
     ▼
[Wing Depression & Flight Initiation (t = +4 ms)]
```

- **Synaptic Latencies**:
  - LC4/LPLC2 $\rightarrow$ GF: $\sim 1.2\text{ ms}$.
  - GF $\rightarrow$ TTMn: $\sim 0.6\text{ ms}$ (electrical coupling via innexins).
  - TTMn $\rightarrow$ TTM muscle: $\sim 1.0\text{ ms}$.
  - **Total latency from looming threshold to jump**: $\approx 6.5\text{--}8.0\text{ ms}$.

---

## 5. Mathematical Formulation for Whole-Brain Co-Simulation

### 5.1 Kuramoto-Hopf Tripod CPG with Hemilineage Coupling

Each of the 6 legs ($i \in \{\text{L1, L2, L3, R1, R2, R3}\}$) is modeled as an adaptive nonlinear limit-cycle oscillator with phase $\Phi_i$ and radius $r_i$:

$$\dot{\Phi}_i = 2\pi f_{\text{step}}(\text{DN}_{\text{drive}}) + \sum_{j \ne i} K_{ij}^{13B} \sin(\Phi_j - \Phi_i - \Delta\Phi_{ij}^*) + I_{\text{CS}}(F_{\text{load}, i})$$

Where:
- $f_{\text{step}}(\text{DN}_{\text{drive}}) = f_{\text{base}} + \alpha (\text{DNp09} + \text{BPN})$: Firing rate of descending walking drives frequency from 3 Hz to 14 Hz.
- $K_{ij}^{13B}$: Contralateral inhibitory coupling weight mediated by Hemilineage 13B interneurons ($K_{13B} \approx 2.5\text{ s}^{-1}$).
- $\Delta\Phi_{ij}^*$: Ideal phase difference ($\pi$ for contralateral homologous legs, $\pi$ for ipsilateral adjacent legs, enforcing alternating tripod gait).
- $I_{\text{CS}}(F_{\text{load}, i})$: Campaniform sensilla stance-holding term ($\le 0$ when under load, delaying swing phase onset).

### 5.2 Backward Locomotion Reversal (MDN Moonwalker Mechanism)

When MDN fires above threshold ($\text{MDN} > 30\text{ Hz}$):
$$\dot{\Phi}_i = -2\pi f_{\text{step}} + \sum_{j} K_{ij}^{13B} \sin(\Phi_j - \Phi_i + \Delta\Phi_{ij}^*)$$
The sign of phase progression is inverted, reversing the metachronal sequence from anterior-to-posterior (T1$\rightarrow$T2$\rightarrow$T3) to posterior-to-anterior (T3$\rightarrow$T2$\rightarrow$T1).

---

## 6. Synthesis and Integration Blueprint

| Component | Biological Circuit (MANC / MaleCNS) | Digital Implementation in Simulation | Verification Test |
|:---|:---|:---|:---|
| **Contralateral Coordination** | Hemilineage 13B (GABAergic midline-crossing interneurons) | Kuramoto-Hopf phase coupling with strict $\Delta\Phi = \pi$ target | `test_tripod_anti_phase_relationship` |
| **Ipsilateral Joint Antagonism** | Hemilineage 13A (Ipsilateral reciprocal inhibition) | Stance/swing mutual exclusion and stance force gating | `test_walknet_rule_one_stance_hold` |
| **Looming Escape Takeoff** | LC4/LPLC2 $\rightarrow$ GF (DNp01) $\rightarrow$ TTMn/PSI | Optical expansion rate detector triggering 42 mm/s ballistic surge | `test_dnp01_giant_fiber_escape_trigger` |
| **Olfactory Plasticity** | Antennal Lobe PNs $\rightarrow$ KCs $\rightarrow$ MBONs + PAM/PPL1 | 120-KC sparse coding + Huang/Luo anti-Hebbian rate rule | `test_appetitive_learning_shifts_valence` |
| **Aversive Avoidance** | MBON-avoidance $\rightarrow$ LAL steering $\rightarrow$ MDN backward walk | Negative MB valence triggers MDN backward rate | `test_aversive_learning_triggers_mdn_retreat` (archived with the root duplicate `docs/archive/root_duplicates_20260924/test_whole_brain.py`; not in `tests/`) |
| **Empirical Data Export** | Tully & Quinn (1985) T-maze assay protocol | `ScientificDataLogger` & `LearningAssay` producing CSV/JSON/NPZ | `test_assay_saves_experiment_data` |

This specification provides the definitive connectomic and biophysical substrate linking the 166,700-neuron MaleCNS v1.0 central brain with the 23,000-neuron MANC v1.0 ventral nerve cord for embodied simulation.
