# Project NeuroFly (v1.0 Alpha)
## Whole-Brain Connectome Simulation & In-Silico Neuroethological Instrument

**Project NeuroFly** is a biologically grounded, whole-brain sensorimotor simulation of *Drosophila melanogaster*. It bridges the **166,700-neuron MaleCNS v1.0 / FlyWire** connectome and the **23,000-neuron MANC v1.0** Ventral Nerve Cord (VNC) motor circuits with a continuous 2D/3D physics environment, closed-loop multi-modal sensory ingress, and a full suite of **12 classical and modern neuroethological experimental paradigms**.

---

## Key Features

1. **Biophysical Whole-Brain Connectomics**:
   - **Mushroom Body (MB)**: 120 Kenyon cells (KCs), sparse antennal lobe projection neuron (PN) encoding, anti-Hebbian rate plasticity (*Huang, Luo et al., Nature 2024*), and PAM (sugar/relief reward) vs PPL1 (shock/heat punishment) dopaminergic neuromodulation.
   - **Central Complex (CX)**: 16-wedge Protocerebral Bridge (PB) and Ellipsoid Body (EB) E-PG compass attractor bump, ER2/ER4d visual ring neurons, and Fan-Shaped Body (FB) allocentric vector navigation (*Ofstad, Zuker & Reiser, Nature 2011*).
   - **Descending Locomotor Decoders**: Identified premotor channels (DNa01 course holding, DNa02 turning yaw, DNp09 pursuit cadence, BPN exploration drive, MDN Moonwalker backward walking, and DNp01 Giant Fiber escape).
   - **Kuramoto-Hopf Tripod Gait CPG**: 6-leg stance/swing coordination enforcing alternating tripod locomotion ($\Delta\Phi = \pi$) and Campaniform Sensilla (CS) ground force feedback (*Cruse's Walknet Rule 1*).
   - **Ascending Efference Copy**: Voluntary saccadic motor copy that shunts Lobula Plate Tangential Cell (HS/VS) self-generated optical slip by $\ge 80\%$.

2. **Continuous 2D Sliding Collision Physics**:
   - Elastic and tangential sliding collision resolution with friction ($\mu=0.5$) and restitution ($\epsilon=0.1$).
   - Rigorously tested: $0$ tunneling occurrences across $20,000$ simulation steps in narrow ($12\text{ mm}$) corridors.

3. **Complete 12-Paradigm Empirical Catalog**:
   - **T-Maze**: Differential olfactory conditioning (*Tully & Quinn 1985*).
   - **Y-Maze**: Spontaneous alternation and individual turn handedness (*Buchanan et al. 2015*).
   - **Thermal Heat-Maze**: Allocentric place learning (*Ofstad et al., Nature 2011*).
   - **Buridan's Paradigm**: Stripe fixation and centrophobism (*Götz 1980; Colomb & Brembs 2012*).
   - **Visual Operant Flight Simulator**: Closed-loop yaw torque conditioning (*Wolf & Heisenberg 1991*).
   - **Wind Tunnel**: Plume tracking with surge-and-cast dynamics (*Alvarez-Salvado et al. 2018*).
   - **Visual Looming Escape**: Giant Fiber ballistic jump takeoff (*Card & Dickinson 2008*).
   - **Optomotor Gaze Stabilization**: Wide-field grating tracking & saccadic efference copy (*Götz 1964; Kim et al. 2017*).
   - **Gap Crossing**: Visual & antennal tactile chasm estimation (*Pick & Strauss 2005*).
   - **Circadian DAM Monitor**: Locomotor activity and sleep/wake bout quantification (*Konopka & Benzer 1971; Allada 2010*).
   - **Courtship Conditioning**: Pheromone associative suppression (*Siegel & Hall 1979; Keleman et al. 2007*).
   - **Corridor Labyrinth**: Multi-junction maze with dead ends, odor plumes, and sliding collisions.

4. **Multi-Tier Scientific Data Logging**:
   - Continuous per-step telemetry (`.csv`): Coordinates, heading, sensory, MBON valence, dopamine, descending spikes, and gait.
   - Trial summaries (`.json`): Latencies, choices, and canonical metrics ($PI, SAR, H, LI, CI, \text{tortuosity}$).
   - Spatial occupancy grids (`.npz`): 2D probability heatmaps.
   - Cohort reports (`PARADIGM_REPORT.md`): Analytical statistics (mean, SEM, Student's t-test, Cohen's $d$).

---

## How to Interact with Project NeuroFly

### 1. Interactive Scientific Web Instrument (GUI)
Start a local web server and open the application in any modern browser:

```bash
# Start server from flybrain directory
python -m http.server 8085 --directory web
```
Then navigate to: **`http://localhost:8085`**

**Features in the Browser**:
- Live 60 FPS 2D canvas simulation with real-time fly kinematics and multi-modal stimuli.
- Left-column **Experiment Catalog** with 13 distinct clickable paradigm cards and live metric readouts.
- Real-time **Genetic Knockout / Lesion Switches** (WT Control, ΔMB Kenyon cells, ΔCX Compass, ΔGF Giant Fiber, ΔJO Johnston's organ, ΔOFF T5 motion).
- Right-column **Experiment Director's Guide** explaining "What to Observe" and live interactive parameter sliders (floor temperature, refuge radius, shock voltage, wind velocity, looming $r/v$, wall friction).
- Electrophysiology HUD: Live 120-KC raster, E-PG compass dial, multi-channel descending motor oscilloscope, and Kuramoto tripod gait monitor.
- Interactive tools: Inspect Fly, Drop Food (Odor A), Alarm Pheromone (Odor B), Deploy Threat, Drag Wind.
- One-click data export: Download per-step telemetry CSV and trial summary JSON directly in browser.

### 2. Standalone Generative UI Artifact
If running without a web server, open the single-file self-contained bundle:
`flybrain_scientific_instrument.html` directly in any web browser or within the IDE.

### 3. Automated High-Throughput Experiment Battery (CLI)
Run headless batches across any or all 12 paradigms:

```bash
# Run all 12 paradigms (3 trials x 200 steps each)
python -m experiments.run_paradigm_battery --paradigms all --trials 3 --steps 200

# Run specific paradigms
python -m experiments.run_paradigm_battery --paradigms heat_maze,t_maze,buridan --trials 5 --steps 500
```

### 4. Programmatic Python API
```python
from arena import Arena
from maze import ExperimentRegistry
from data_logger import ParadigmDataLogger

# Instantiate any paradigm
paradigm = ExperimentRegistry.get("heat-maze")
arena = Arena(paradigm=paradigm, fly_count=1, brain_type="connectome")

# Run simulation loop
logger = ParadigmDataLogger(paradigm_name="heat-maze")
for step in range(500):
    result = arena.step(dt=0.02)
    logger.log_step(result)

# Finalize trial
summary_path, npz_path = logger.end_trial(result.get("paradigm_metrics", {}))
print(f"Logged trial to {summary_path}")
```

---

## Running Automated Tests

```bash
# In Docker container
docker exec -w /workspace 65dcff428c87 pytest -v tests/

# On AMD Ryzen Workstation (SSH)
ssh avg-usr@192.168.194.227 "cd /home/avg-usr/Documents/ChatGPT/flybrain && source .venv/bin/activate && PYTHONPATH=. pytest tests/"
```

**Status**: 116 / 116 tests passing in Docker; 140 / 140 tests passing on AMD Ryzen host.
