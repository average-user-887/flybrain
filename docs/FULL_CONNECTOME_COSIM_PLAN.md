# Full-connectome co-simulation plan: APPROVED 2026-09-21, IMPLEMENTED in v0.3.0

> **Implementation Note (v0.3.0)**: Server-side co-simulation (`neurofly_body`), FlyGym 2.1 / MuJoCo 3.9
> integration, and the Three.js 3D dashboard viewport were implemented and delivered in Release v0.3.0.

---

# Implementation Plan: Adapting Project NeuroFly to Full Whole-Brain Connectome & Embodied Walking

## Executive Summary & State-of-the-Art Research

To transition Project NeuroFly from its current **compact modular controller** to a **true embodied whole-brain simulation**, we conducted deep research across the foremost open-source Drosophila connectome projects published between 2024 and 2026:

1. **`erojasoficial-byte/fly-brain` (2024–2025)**:
   - **Scale**: 138,639 neurons and 15+ million synapses from FlyWire v783.
   - **Embodiment**: Coupled directly to **NeuroMechFly v2** inside the **MuJoCo** physics engine.
   - **Key Architectural Lesson**: Employs a dedicated **"Brain-Body Bridge"**. High-level behavioral decisions emerge from connectome spiking activity and are channeled through Descending Neurons (DNs) into lower-level Central Pattern Generators (CPGs), while ascending sensory and proprioceptive signals flow back into the brain graph.
2. **`ZeroXClem/closed-loop-fly` (2024–2025)**:
   - **Scale**: MaleCNS v1.0 connectome in closed loop.
   - **Environment**: Runs directly in the browser using **WebGPU and Three.js** with neural simulation executing inside a WebWorker (~0.17x real-time on RTX 3070).
   - **Key Architectural Lesson**: Maps compound eye ommatidia directly to optic lobe column entries and decodes bilateral descending outputs to steer in-browser 3D agents.
3. **`philshiu/Drosophila_brain_model` (Nature 2024)**:
   - **Ground-Truth Biophysics**: Published benchmark for Drosophila leaky integrate-and-fire (LIF) parameters ($V_{\text{rest}} = -52\text{ mV}$, $V_{\text{reset}} = -52\text{ mV}$, $V_{\text{thresh}} = -45\text{ mV}$, $\tau = 5\text{ ms}$, $t_{\text{ref}} = 2.2\text{ ms}$, $w_{\text{unit}} = 0.275\text{ mV}$).
   - **Key Finding**: Validated transmission paths from sensory receptor neurons through intermediate interneurons to descending premotor neurons (e.g. sugar feeding and grooming pathways).
4. **`NeLy-EPFL/flygym` (EPFL Ramdya Lab / NeuroMechFly v2)**:
   - **The Gold Standard for Drosophila Biomechanics**: Simulates an articulated fruit fly body in MuJoCo with 18 actuated degrees of freedom (Coxa, Femur, Tibia per leg), cuticular adhesion, and ground reaction forces.
   - **Descending Locomotion Interface**: Driven by a **two-value descending signal** ($\Delta_{\text{yaw}}$ for turning, $v_{\text{fwd}}$ for thrust) that modulates left and right Kuramoto CPG oscillators.

---

## User Review Required

> [!IMPORTANT]
> **Architectural Execution Choice: Server-Side Co-Simulation (Recommended) vs. Client-Side WebGPU**
> - **Option A (Recommended: Multi-Rate Co-Simulation on AMD Ryzen)**:
>   The full MaleCNS v1.0 connectome (166,700 neurons, 25.5M synapses, 1.4 GB) and the 3D walking physics run on your AMD Ryzen 3900X (24 threads, 128 GB RAM) via Numba/C++ and MuJoCo/FlyGym. The server steps at high speed and streams 60 FPS 3D joint angles, leg contacts, descending neuron firing rates, and arena coordinates to the web dashboard via SSE/WebSockets.  
>   *Advantages*: Runs seamlessly on any client browser/laptop (no GPU bottleneck); utilizes the already-verified 166.7k dataset and `brainlab/` kernel on Ryzen.
> - **Option B (In-Browser WebGPU Single-Worker)**:
>   Port the connectome weights to quantized WebGPU buffers running entirely on the user's browser GPU (similar to `ZeroXClem/closed-loop-fly`).  
>   *Disadvantages*: Requires high-end client GPUs (RTX 3070+ only achieves ~0.17x real-time; standard work laptops will thermal throttle or crash with Out-Of-Memory errors on a 166.7k graph).

---

## Open Questions

> [!NOTE]
> 1. **Visual Presentation Mode**: In the web dashboard, do you want:
>    - **(Recommended) Integrated 3D Articulated View**: A Three.js 3D viewport rendering the articulated fly walking, showing leg joints flexing, tarsal ground contact, and antenna deflections in real-time alongside the arena map.
>    - **High-Resolution 2D Biomechanical Stage**: Keep the current top-down 2D arena canvas but upgrade the fly icon with true articulated 6-limb kinematic joint rendering (CTr, FTi, TiTa angles derived from the CPG).
> 2. **Connectome Plasticity Scope**:
>    - **(Recommended) Hybrid Biological Plasticity**: Run the 166.7k connectome with resting homeostatic stability and GluCl$\alpha$ inhibition, while allowing dopamine-modulated synaptic plasticity in the Mushroom Body ($W_{\text{KC}\to\text{MBON}}$) and premotor decoders, matching Shiu et al. and erojasoficial-byte.
>    - **Uniform Static Connectome**: Run the 166.7k graph purely with fixed static anatomical weights.

---

## Proposed Changes

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PROPOSED RE-ARCHITECTURE                                 │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ Layer                          │ Implementation Target                                 │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ 1. Sensory Ingress             │ 72-Ray Visual Ingress + Johnston's Organ Antennae      │
│                                │   ──► 6,098 Sensory Nodes in MaleCNS v1.0             │
│ 2. Whole-Brain Spiking Graph   │ MaleCNS v1.0 (166,700 Neurons / 25.58M Synapses)      │
│                                │   ──► Numba / C++ LIF Spiking Kernel on Ryzen         │
│ 3. Descending Premotor Bridge  │ 1,314 Descending Neurons (DNa02, DNp09, MDN, GF)      │
│                                │   ──► Firing Rate Differential Decoders               │
│ 4. 6-Limb Walking Engine       │ 18-DOF Articulated Kuramoto/FlyGym Tripod CPG         │
│                                │   ──► Cruse Rule 1 Stance/Swing & Joint Torques       │
│ 5. Interactive Web Dashboard   │ Three.js 3D Articulated Model + Connectome HUD        │
│                                │   ──► Real-time Descending Spike Rasters & Gait Telemetry│
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

### Component 1: Sensory Ingress to Full Connectome (`connectome_bridge.py`, `brainlab/`)

Connect the environment observations directly to the 6,098 sensory neuron entries in MaleCNS v1.0:

#### [MODIFY] [`connectome_bridge.py`](../connectome_bridge.py)
- **Visual Mapping**: Map 72-ommatidia compound eye visual rays into the optic lobe columnar entry neurons (R1–R6, L1, L2, T4/T5, LC4, LPTC HS/VS) using Poisson rate-to-spike generation.
- **Olfactory & Mechanosensory Mapping**: Route odor concentrations directly into Antennal Lobe Projection Neurons (PNs) and wind deflection drag into Johnston's Organ Neurons (JON-C/E).

#### [MODIFY] [`brainlab/engine.py`](../brainlab/engine.py)
- Integrate Shiu et al. (Nature 2024) baseline parameters alongside Claude's v3 PSP conductance calibration ($E_{\text{exc}} = 0\text{ mV}$, $E_{\text{inh}} = -70\text{ mV}$, $g_{\text{unit}}^{\text{exc}} = 1/52$, $g_{\text{unit}}^{\text{inh}} = 1/18$).
- Enforce tonic background current injection to maintain physiological resting state ($V_m \approx -52\text{ mV}$) without runaway epileptic synchronization.

---

### Component 2: Descending Premotor Bottleneck & Motor Decoding (`connectome_bridge.py`, `locomotion.py`)

Extract motor commands directly from the descending neuron populations in the full connectome:

#### [MODIFY] [`connectome_bridge.py`](../connectome_bridge.py)
- Monitor real-time spike counts over sliding $20\text{ ms}$ windows across identified descending pairs:
  - **`DNa02` (Turning Yaw)**: $\Delta\omega_{\text{yaw}} = \alpha \cdot (R_{\text{DNa02\_R}} - R_{\text{DNa02\_L}})$.
  - **`DNp09` (Forward Velocity / Plume Surge)**: $v_{\text{thrust}} = v_0 + \beta \cdot R_{\text{DNp09}}$.
  - **`MDN` (Moonwalker Reversal)**: Reverses phase direction of CPG oscillators when obstacles or noxious heat are detected.
  - **`DNp01` (Giant Fiber Looming Takeoff)**: Triggers sudden ballistic flight transition upon rapid visual disk expansion ($d\theta/dt > 1.2\text{ rad/s}$).

---

### Component 3: Articulated 6-Leg Walking Engine (`locomotion.py`, `arena.py`)

Implement true 6-limb walking kinematics and dynamics derived from NeuroMechFly v2 / FlyGym:

#### [MODIFY] [`locomotion.py`](../locomotion.py)
- Upgrade the Kuramoto-Hopf oscillator network to output explicit 3D joint angles for all 6 legs:
  - **Coxa-Trochanter (CTr)**: Protraction / retraction ($[-20^\circ, +40^\circ]$).
  - **Femur-Tibia (FTi)**: Joint flexion / extension ($[30^\circ, 110^\circ]$).
  - **Tibia-Tarsus (TiTa)**: Pitch and claw ground contact ($[-15^\circ, +35^\circ]$).
- Implement **Cruse's Rule 1** (Campaniform Sensilla CS cuticular load gating): A leg cannot initiate swing phase until adjacent legs bear ground reaction load ($F_{\text{normal}} > 2.5\ \mu\text{N}$).

#### [MODIFY] [`arena.py`](../arena.py)
- Transmit the full 18-joint state, 6 leg contact states, and descending neuron rates in every telemetry snapshot published by `neurofly_daemon.py`.

---

### Component 4: Dashboard Adaptation (`web/app.js`, `web/index.html`)

Upgrade the web dashboard to visually expose the running full connectome and articulated walking:

#### [MODIFY] [`web/index.html`](../web/index.html)
- Add an interactive **3D Articulated Walking Viewport** (using Three.js) in the center stage or as a toggleable overlay next to the 2D arena.
- Add a **Connectome Premotor Deck**: Real-time spike activity indicators and rate bars for `DNa02_L/R`, `DNp09`, `MDN`, and `GF`.
- Update the Identity Bar to reflect:
  `Controller: connectome (MaleCNS v1.0 — 166,700 neurons, 25.5M synapses)`

#### [MODIFY] [`web/app.js`](../web/app.js)
- Parse the 18-joint telemetry angles from the daemon SSE stream and animate the 3D articulated fly skeleton.
- Render ground contact indicators (green pads for stance, blue for swing) showing the alternating tripod walking gait in real-time.

---

## Verification Plan

### 1. Automated Spiking & Premotor Tests
- `pytest tests/test_connectome_stability.py`: Verify that the 166,700-neuron graph achieves stable resting rates ($< 5\text{ Hz}$ mean background) without runaway epileptiform bursting.
- `pytest tests/test_biomechanics_closed_loop.py`: Verify that asymmetric visual stimulus produces asymmetric `DNa02` spikes ($> 20\text{ Hz}$ differential), which successfully induces differential leg cadence and yaw steering in the CPG.

### 2. Physical Walking Kinematics Verification
- `python scripts/behavior_audit.py`: Verify that 6-limb stepping produces valid tripod coordination (Tripod A vs Tripod B phase shift $\Delta\Phi \approx \pi$), zero backward slippage during forward walking, and proper obstacle-triggered MDN reversal.

### 3. Live Browser Verification (Firefox 156 / Headless)
- `python scripts/live_ui_signoff.py`: Verify live in the browser that:
  - The identity bar shows `Controller: connectome (MaleCNS v1.0)`.
  - The 3D articulated fly actively flexes leg joints, walks across all 14 paradigms, and turns toward odor plumes and away from looming discs.
  - Zero dropped frames, zero console errors, data age $< 0.1\text{ s}$.
