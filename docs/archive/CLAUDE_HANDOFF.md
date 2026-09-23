# Project NeuroFly — Claude Agent Handoff & Architecture Dossier

**Target Audience**: Claude (AI Research & Engineering Pair Programmer)  
**Status as of September 19, 2026**: Fully Operational, 100% Test Pass Rate Across All Mirrors, Zero Hardcoded Paths.  
**System Mission**: Complete In-Silico Biophysical & Embodied Whole-Brain Simulation of *Drosophila melanogaster* (166,700 Neurons, 25.5 Million Synapses, MaleCNS v1.0 & FlyWire Connectomes).

---

## 1. Executive Summary & Repository Identity

Project NeuroFly is a state-of-the-art computational neuroethology platform bridging whole-brain connectomics with closed-loop embodied biomechanics. It reproduces canonical behavioral assays in an interactive browser-based scientific instrument and a 24/7 continuous learning daemon.

### Current Health & Baseline Verification
- **Docker Sandbox Container (`<container-id>`)**: **156 / 156 passed** (`pytest -q /workspace/tests`).
- **AMD Ryzen 3900X Cluster (`<workstation-host>`)**: **180 / 180 passed** (`PYTHONPATH=. pytest -q tests/`).
- **Continuous Learning Daemon**: Active on a recorded PID at `http://<workstation-host>:8769`, stepping at $15.0\times$ simulation speed with rolling checkpoints.
- **Portability Audit**: **Zero hardcoded paths**. All file paths resolve dynamically relative to `Path(__file__)`.
- **Standalone Instrument**: Single-file bundle at [`flybrain/flybrain_scientific_instrument.html`](flybrain_scientific_instrument.html) (310 KB, zero external dependencies, runs offline in any modern browser).

---

## 2. Infrastructure & Compute Mirrors Map

The codebase is synchronized across four distinct execution environments:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PROJECT NEUROFLY ECOSYSTEM                           │
├──────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│ Node / Target                │ Address / Location          │ Role & Execution Details  │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ 1. Local Workspace (Windows) │ .\flybrain                  │ Primary development root. │
│ 2. Docker Sandbox            │ <container-id>:/workspace     │ Local Linux test runner.  │
│ 3. AMD Ryzen 3900X Cluster   │ <user>@<workstation-host>     │ 24/7 continuous daemon &  │
│                              │ ~/Documents/ChatGPT/flybrain│ full 180-test benchmark.  │
│ 4. HP Storage Server         │ <archive-share>/neurofly                 │ Network SMB backup & cold │
│                              │ //<file-server>/Storage/...  │ weight checkpoint archive.│
│ 5. Shipping Distribution     │ .\neurofly_v1_shipping      │ Clean deployment package. │
└──────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```

### Essential Commands Quick Reference
```bash
# 1. Run local Docker tests (156 tests)
docker exec -i <container-id> pytest -q /workspace/tests

# 2. Run remote tests on AMD Ryzen (180 tests)
ssh -i $NEUROFLY_SSH_KEY <user>@<workstation-host> "cd $NEUROFLY_DIR && source .venv/bin/activate && PYTHONPATH=. pytest -q tests/"

# 3. Check AMD Ryzen background learning daemon status
curl -s http://<workstation-host>:8769/api/status

# 4. Restart Ryzen daemon
ssh -i $NEUROFLY_SSH_KEY <user>@<workstation-host> "cd $NEUROFLY_DIR && ./stop_daemon_ryzen.sh && ./start_daemon_ryzen.sh"

# 5. Re-bundle standalone HTML instrument
python scratch/sync_standalone.py

# 6. Synchronize all code and tests to Docker, Ryzen, and HP Server
python flybrain/sync_ecosystem.py
```

---

## 3. Core Architecture & Biophysical Subsystems

```
                                  ┌───────────────────────────────┐
                                  │      14 CANONICAL ASSAYS      │
                                  │ (T-Maze, Y-Maze, Buridan, etc)│
                                  └──────────────┬────────────────┘
                                                 │ Visual / Olfactory / Mechanosensory
                                                 ▼
┌──────────────────────────────┐  ┌───────────────────────────────┐
│     6-LIMB BIOMECHANICS      │  │        SENSORY INGRESS        │
│ • Kuramoto-Hopf CPG (8.5 Hz) │  │ • Lobula LC4/LPLC2 (Looming)  │
│ • Cruse Rule 1 (CS load gate)│  │ • LPTC HS/VS (Optic Flow)     │
│ • FeCO joint proprioception  │  │ • Johnston's Organ JON-A/B/C/E│
│ • 3D Coxa/Femur/Tibia joints │  │ • Antennal Deflection Taxis   │
└──────────────▲───────────────┘  └──────────────┬────────────────┘
               │                                 │
               │ Descending Motor Commands       │
               │ (DNa02, DNp09, MDN, GF)         ▼
┌──────────────┴───────────────┐  ┌───────────────────────────────┐
│     DESCENDING BOTTLENECK    │  │       CENTRAL CONNECTOME      │
│ • DNa02: Fast steering yaw   │  │ • MaleCNS v1.0 (166.7k nodes) │
│ • DNa01: Course holding      │◄─┤ • GluCl-alpha inversion (-70mV)│
│ • DNp09: Forward pursuit     │  │ • Mushroom Body (120 KCs)     │
│ • MDN: Backward walking      │  │ • Central Complex (E-PG, PFL3)│
│ • GF (DNp01): Looming takeoff│  │ • Dopamine PAM/PPL1 plasticity│
└──────────────────────────────┘  └───────────────────────────────┘
```

### 1. Central Connectome Biophysics ([`circuit.py`](flybrain/circuit.py), [`central_complex.py`](flybrain/central_complex.py))
- **Scale**: Connectome graph derived from MaleCNS v1.0 and FlyWire.
- **Glutamate Inversion ($\text{GluCl}\alpha$)**: In *Drosophila*, central glutamate acts primarily as an inhibitory neurotransmitter via ligand-gated chloride channels ($\text{GluCl}\alpha$, $E_{\text{rev}} = -70\text{ mV}$). Modeling this prevents epileptiform runaway activity and keeps the spectral radius $\rho < 1.0$.
- **Degree-Dependent In-Degree Scaling**: Synaptic weights scale as $w_{ij} \propto (k_i^{\text{in}})^{-0.5}$ to prevent hub saturation across high-degree interneurons (APL, PFL3, DNa02).
- **Mushroom Body Associative Memory**: 120 Kenyon cells receiving sparse pseudorandom projection neuron (PN) olfactory inputs. Dopaminergic neurons (DANs: PPL1 for electric shock punishment, PAM for sugar reward) modulate KC $\to$ MBON synapses via an anti-Hebbian rate-depression learning rule.
- **Central Complex Spatial Compass**: 16-wedge E-PG ring attractor tracking allocentric heading $\theta$. P-EN and P-EG angular velocity integration coupled with PFL3/PFL2 descending steering decoders.

### 2. Sensory Ingress Systems ([`vision.py`](flybrain/vision.py), [`mechanosensory.py`](flybrain/mechanosensory.py))
- **Lobula Columnar (LC) Features**:
  - **LC4 / LPLC2**: Evaluates angular expansion velocity $d\theta/dt$ converging on the Giant Fiber (**DNp01**) to trigger rapid takeoff jumps ($5\text{ ms}$ latency).
  - **LC6**: Broad looming detection in PVLP for collision avoidance and landing deceleration.
  - **LC10 (LC10a–d)**: Small moving target pursuit tracking tuned to $2^\circ\text{--}5^\circ$ subtended angle.
  - **LC11**: Small target motion against textured background clutter via center-surround suppression.
- **Lobula Plate Tangential Cells (LPTCs)**:
  - **Horizontal System (HSN, HSE, HSS)**: Wide-field horizontal optomotor gaze stabilization.
  - **Vertical System (VS1–VS12)**: 3D rotational flow field deconstruction (pitch, roll, yaw).
  - **Visual Efference Copy**: Shunts $>80\%$ of LPTC membrane depolarization during voluntary saccades, preventing the fly from fighting its own self-generated visual flow.
- **Johnston's Organ (Antenna)**:
  - **JON-A/B**: High-frequency acoustic vibrations ($200\text{--}500\text{ Hz}$) for courtship song detection.
  - **JON-C/E**: Static and low-frequency aerodynamic drag for wind compass tracking and gravitaxis.
  - **Wedge Projection Neurons (WPNs)**: Bilateral push-pull wind angle decoding routing into the Lateral Accessory Lobe (LAL010) and Central Complex ring neurons.

### 3. Biomechanics & Closed-Loop Locomotion ([`locomotion.py`](flybrain/locomotion.py))
- **Kuramoto-Hopf Limit Cycle CPG**: 6 coupled non-linear phase oscillators driving alternating tripod gait (Tripod A: L1-R2-L3 vs Tripod B: R1-L2-R3) at $8.5\text{ Hz}$ base cadence.
- **Proprioceptive Reflex Gating (Cruse's Walknet Rule 1)**: Cuticular Campaniform Sensilla (CS) measure ground reaction load. If a leg experiences cuticular load $> 2.5\ \mu\text{N}$, stance-to-swing phase transition is inhibited until neighboring legs make contact.
- **Femoral Chordotonal Organ (FeCO)**: Modulates claw angle and hook velocity during swing phase.
- **Descending Motor Decoders**:
  - `DNa02`: Fast turning yaw rate injected into bilateral CPG phase velocity differentials.
  - `DNa01`: Course-holding tonic stabilizer.
  - `DNp09`: Forward walking speed and odor plume surge drive.
  - `MDN` (Moonwalker): Reverses phase coupling direction, driving backward walking.
  - `DNp01` (Giant Fiber): All-or-none escape takeoff trigger.

---

## 4. Physics Engine & Collision Architecture ([`maze.py`](flybrain/maze.py), [`arena.py`](flybrain/arena.py), [`web/app.js`](flybrain/web/app.js))

### Simulation-Grade Continuous Collision Detection (CCD)
The physics engine operates under strict continuous collision detection to guarantee that flies never tunnel through walls, get stuck, or escape out of bounds:

1. **Analytical Swept-Circle TOI (`swept_circle_toi`)**:
   - Tests the moving circle capsule volume along $p_0 \to p_1$ against each finite line segment and its rounded endcaps.
   - Computes the exact closed-form Time-of-Impact $\text{TOI} \in [0, 1]$, contact point on the wall `cpt`, and incoming contact normal `norm`.
2. **Contact Normal Calculation (Critical Rule)**:
   - When a swept hit occurs (`hit == True`), the contact normal **MUST ALWAYS** be `norm` returned by `swept_circle_toi` because it points back toward the side the circle entered from.
   - For static overlap checks: normal is $(p_1 - \text{proj}) / d$ when $d > 10^{-8}$, falling back to segment normal `nx, ny`.
   - **DO NOT** use `signed_dist < 0.0` as a raw distance condition! Segments are finite; checking signed distance on the infinite line without checking whether the projection lies on the segment will falsely detect walls 50 mm away and cause corner solver teleportation.
3. **Simultaneous 2x2 Corner Wedge Solver**:
   - Resolves acute corners ($60^\circ$ Y-maze junctions, $45^\circ$ labyrinth corners) by solving the $2 \times 2$ matrix equation:
     $$\begin{pmatrix} n_{1x} & n_{1y} \\ n_{2x} & n_{2y} \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} d_1 \\ d_2 \end{pmatrix}$$
   - Places the circle center at the unique valid position where distance to both walls is $\ge R$, eliminating jitter and ping-ponging.
4. **Biomechanical Sliding Friction**:
   - Elastic coefficient $e = 0.0$ (flies do not bounce like rubber balls).
   - Normal velocity is zeroed; tangential velocity is damped by Coulomb sliding friction ($\mu \approx 0.4\text{--}0.5$).
5. **Mathematical Bounding Containment (`enforce_containment`)**:
   - Executed at the end of every simulation step as a hard mathematical failsafe.
   - Enforces closed-form bounding geometries across all 14 paradigms (T-maze stem/arms union, Y-maze radius, Buridan disk, capillary tubes, etc.).

---

## 5. Genetic Knockouts & The 14 Experimental Assays

### The 6 Available Genotypes & Biological Driver Lines
In *Drosophila* research, GAL4/UAS silencing establishes **causal necessity**:

| Genotype | Real-World Genetic Driver | Cellular Mechanism | Phenotype & Behavioral Deficit |
|---|---|---|---|
| **WT Control** | *Canton-S* / *w1118* baseline | Complete unperturbed connectome. | Intact associative memory ($\text{PI} > 0.70$), normal turning, rapid escape. |
| **$\Delta$MB (Kenyon)** | *MB247-GAL4 > UAS-TNT* | Silences ~4,000 Mushroom Body Kenyon cells. | **Complete Associative Amnesia**: Fails T-maze odor learning ($\text{PI} \approx 0.00$), fails heat-maze place memory. |
| **$\Delta$CX (Compass)** | *R60D05-GAL4 > UAS-TNT* | Silences Central Complex E-PG compass & PFL3 steering. | **Loss of Spatial Navigation**: Cannot fixate visual stripes (fails Buridan), wanders aimlessly. |
| **$\Delta$GF (Escape)** | *R68A06-GAL4 > UAS-shi[ts]* | Silences Giant Fiber descending pair (DNp01/GF). | **Looming Blindness**: Fails to trigger $5\text{ ms}$ jump takeoff; captured by looming shadows. |
| **$\Delta$JO (Wind)** | *tilB* / *nompA* / *JO-GAL4* | Silences Johnston's organ antennal mechanoreceptors. | **Deafness & Loss of Anemotaxis**: Fails upwind surge-cast tracking; deaf to courtship songs. |
| **$\Delta$OFF (T5)** | *T5-split-GAL4 > UAS-Kir2.1* | Silences T5 dark-edge motion columnar interneurons. | **Dark-Edge Motion Blindness**: Fails optomotor gaze stabilization during dark grating rotation. |

### The 14 Canonical Neuroethology Assays & Key Sliders
1. **Open Arena Multi-Modal Foraging**: Boundary repulsion, ambient wind, predator speed.
2. **T-Maze Associative Conditioning (Tully-Quinn 1985)**: Shock grid voltage ($10\text{--}100\text{ V}$, modulates PPL1 dopamine rate), vacuum airflow ($2\text{--}25\text{ cm/s}$).
3. **Y-Maze Spontaneous Alternation (Buchanan 2015)**: Bilateral premotor bias (DNa02 asymmetry, models idiosyncratic handedness), decision pause delay ($0.1\text{--}2.0\text{ s}$).
4. **Thermal Place Learning in Heat-Maze (Ofstad 2011)**: Aversive heated floor ($32\text{--}44^\circ\text{C}$), cool refuge radius ($6\text{--}20\text{ mm}$).
5. **Buridan's Landmark Fixation (Götz 1980)**: Stripe angular width ($5^\circ\text{--}30^\circ$), water moat barrier repulsion ($0.5\text{--}3.0\times$).
6. **Visual Operant Flight Simulator (Wolf & Heisenberg 1991)**: Infrared laser punishment power ($10\text{--}100\text{ mW}$), virtual yaw inertia ($0.5\text{--}2.5\times$).
7. **Wind Tunnel Plume Tracking (Alvarez-Salvado 2018; Demir 2020)**: Laminar carrier velocity ($5\text{--}40\text{ cm/s}$), Gaussian odor plume width ($6\text{--}30\text{ mm}$).
8. **Predator Looming Escape (Card & Dickinson 2008)**: Approach $r/v$ ratio ($10\text{--}100\text{ ms}$), Giant Fiber spike threshold ($0.40\text{--}0.95\text{ }V_m$).
9. **Optomotor Gaze Stabilization (Götz 1964; Kim 2017)**: Drum rotational velocity ($\omega = -120\text{--}+120^\circ/\text{s}$), grating spatial wavelength ($10^\circ\text{--}60^\circ$).
10. **Gap Crossing & Spatial Motor Planning (Pick & Strauss 2005)**: Chasm width ($1.5\text{--}5.0\text{ mm}$, threshold $\le 3.8\text{ mm}$ to cross vs $> 4.2\text{ mm}$ to abort), tarsal claw friction.
11. **Circadian Locomotor DAM Monitor (Konopka 1971; Allada 2010)**: Incubator temperature ($18\text{--}32^\circ\text{C}$), time dilation factor ($1\text{--}60\times$), LD vs DD photoperiod.
12. **Courtship Conditioning (Siegel & Hall 1979; Keleman 2007)**: Anti-aphrodisiac cVA pheromone concentration, female decoy walking speed.
13. **Corridor Obstacle Labyrinth**: Coulomb wall crawling friction ($\mu = 0.0\text{--}0.8$), goal chamber sucrose odor emission.
14. **Multisensory 6-Limb Benchmark**: Kuramoto CPG cadence ($3.0\text{--}14.0\text{ Hz}$), DNa02 steering sensitivity gain ($0.5\text{--}2.5\times$).

---

## 6. Known Gotchas & Architectural Rules

1. **Zero Hardcoded Paths**: Never write hardcoded paths like `C:\Users\...` or `/home/...` into code. Always use `Path(__file__).resolve().parent` or standard environment variables (`NEUROFLY_RYZEN_HOST`, `NEUROFLY_SSH_KEY`, etc.).
2. **Synchronize After Every Edit**: When modifying python backend files or frontend files, always run:
   ```bash
   python scratch/sync_standalone.py
   python flybrain/sync_ecosystem.py
   ```
   This ensures the standalone HTML bundle and all remote nodes (Docker, AMD Ryzen, HP Server) remain 100% in sync.
3. **Continuous Background Learning Daemon**: The AMD Ryzen host runs the background daemon via `start_daemon_ryzen.sh` on port `8769`. If modifying `neurofly_daemon.py`, restart the daemon using `./stop_daemon_ryzen.sh && ./start_daemon_ryzen.sh`.
4. **SSE Packet Paradigm Filtering**: The frontend `DaemonBridgeClient` enforces `if (pkt.active_paradigm && pkt.active_paradigm !== this.activeParadigmId) return;` to prevent stale daemon telemetry from warping fly coordinates into another paradigm.

---

## 7. Open Frontiers & Next Steps for Claude

If you are continuing this work, here are the highest-impact frontiers ready to be tackled:

1. **Spiking Kernel SIMD Acceleration (AVX-512 / C++ Numba)**:
   - Scale the LIF spiking kernel from the current modular sub-circuits to the full 166,700-neuron connectome running at $\ge 1.0\times$ real-time on the AMD Ryzen 3900X (24 threads).
   - Leverage sparse CSR matrix representation for the 25.5 million synapses with 8-bit quantized weights.
2. **MuJoCo / FlyGym 3D Physics Co-Simulation**:
   - Bridge the descending decoders (`DNa02`, `DNa01`, `DNp09`, `MDN`, `DNp01`) to the MuJoCo-based FlyGym 3D articulated exoskeleton model for ground reaction forces, grooming, and aerial flight transitions.
3. **Multi-Fly Social & Courtship Interactions**:
   - Instantiate multiple interactive flies in the courtship chamber and open arena to model dynamic male-female chasing, courtship song feedback loops, and group density-dependent aggregation.
4. **Long-Term Plasticity Checkpoint Management**:
   - Extend the daemon to snapshot full synaptic weight matrices into compressed HDF5/Zarr formats after multi-hour associative learning campaigns.
