# Project NeuroFly

An embodied *Drosophila* behavioural simulator with a streaming "always-on"
learning daemon, a browser dashboard, and a separate CPU spiking substrate
built from the MaleCNS v1.0 connectome. It is research software in alpha; the
section **What this simulates today** is the authoritative description of what
the code does. Everything else is roadmap.

## What this simulates today

| Layer | What runs | Where |
| --- | --- | --- |
| **Arena and paradigms** | A 2D continuous arena with one fly (optionally predators), swept-circle collision detection, corner resolution and per-paradigm containment. 13 registered assays plus an open arena: T-maze, Y-maze, heat-maze, Buridan, visual operant, wind tunnel, looming escape, optomotor, gap crossing, circadian DAM, courtship, corridor labyrinth, multisensory benchmark. | `arena.py`, `maze.py` |
| **The fly's "brain" (default, `brain_type="modular"`)** | Hand-built, rate-based modules with literature-chosen parameters: a mushroom-body circuit (40 projection neurons, 120 Kenyon cells, 2 output neurons, PAM/PPL1 dopamine) with the Huang et al. 2024 anti-Hebbian plasticity rule; a 16-wedge E-PG compass with a fan-shaped-body goal vector; surge-and-cast plume logic; a 72-ommatidium compound eye; a 16-wedge Johnston's organ; a metabolic state; and a six-leg tripod CPG. **None of these are derived from connectome wiring.** They are compact phenomenological models. | `circuit.py`, `central_complex.py`, `surge_cast.py`, `vision.py`, `mechanosensory.py`, `metabolic.py`, `locomotion.py` |
| **`brain_type="connectome"` surrogate** | A larger in-process rate model (54 glomeruli, 72 ommatidia, named descending channels DNa01/DNa02/DNp09/MDN/DNp01, LAL arbitration, efference copy, Kuramoto-Hopf leg oscillators). Despite the name it is also hand-built; the code calls it the *surrogate*. | `connectome_bridge.py` |
| **What learns** | Online: KC→MBON weights in the mushroom-body module change with dopamine during trials, in the arena, the daemon and the dashboard. Offline: `brainlab.learning` trains an *external* linear readout on spikes from the full graph with fixed synaptic weights. | `circuit.py`, `brainlab/learning.py` |
| **Full-scale connectome substrate (`brainlab/`)** | Downloads MaleCNS v1.0 (CC BY 4.0), keeps every released edge among 166,700 annotated neurons (25,582,938 directed connections), builds a CSR graph and steps a fixed-weight leaky integrate-and-fire model in a Numba kernel validated against Brian2. 100 ms of activity takes about 0.3 s on one CPU. It has **no plasticity, no calibrated sensory input and no motor output**; stimulation targets are chosen by hand. | `brainlab/`, `FULL_BRAIN_TEST.md`, `RUNS.md` |
| **Bridging the two (optional RPC mode)** | `brainlab/cosim_server.py` can expose the full graph over HTTP and `connectome_client.py` can drive it from the arena. Sensory node groups are looked up by cell type; the *descending-neuron read-out indices are hard-coded integers whose mapping to real DN types has not been verified*. This path is exercised only by unit tests against a 2,500-neuron synthetic graph; the daemon does not use it. | `brainlab/cosim_server.py`, `connectome_client.py` |
| **Continuous daemon** | Steps the modular fly 24/7, streams telemetry over SSE, accepts intervention commands, writes per-experiment weight checkpoints and append-only learning records (`docs/DATA_SCHEMA.md`). Has an opt-in read-only public mode (`docs/PUBLIC_STREAMING.md`). | `neurofly_daemon.py`, `stream_gateway.py`, `learning_recorder.py` |
| **Browser dashboard** | `web/app.js` is an independent JavaScript re-implementation of the arena and the modular brain that runs at 60 FPS in the browser, with knockout switches and sliders. When a daemon is reachable it overlays the daemon's fly. The physics must be kept in step with the Python by hand. | `web/` |
| **Data logging and batteries** | Per-step CSV, per-trial JSON, occupancy NPZ, cohort statistics and Markdown reports; a headless battery runner and a lesion study script. | `data_logger.py`, `experiments/` |

Things the documentation used to claim that are **not** true of the code:

- The fly you watch in the arena, the daemon or the dashboard is **not** driven by
  166,700 neurons or 25.5 million synapses. Those numbers describe the
  `brainlab` graph, which runs separately with fixed weights.
- No **FlyWire** or **MANC** data is used anywhere. Both are cited by name; the
  VNC document (`VNC_PREMOTOR_CONNECTOME_MAPPING.md`) is a written
  specification, and `locomotion.py` is a CPG model, not a MANC-derived circuit.
- "Zero tunnelling over 20,000 steps" is a unit test of the collision engine in
  a straight corridor (`tests/test_collision_physics.py`). The live daemon fly
  was observed pinned against an arena wall; containment and trial advancement
  are being fixed (`docs/OPEN_SOURCE_PLAN.md`, Phase 1).
- Knockout "genotypes" flip flags on the phenomenological modules (e.g. clamp
  MB valence, decouple the compass). They are not lesions of a connectome graph.

## Roadmap

In rough order (see `docs/OPEN_SOURCE_PLAN.md`):

1. Containment and trial-advancement fixes in the physics, mirrored in the dashboard.
2. Public read-only stream of the daemon (implemented, not yet published).
3. Verified descending/sensory neuron mappings for the RPC bridge, then a real
   closed loop between the arena and the full graph at reduced real-time factor.
4. Plasticity inside the full graph with HDF5/Zarr weight checkpoints.
5. Multi-fly social behaviour.

## Learning observatory

The new `web/research.html` shows actual daemon weights, cue probes, trial
measurements and per-experiment teaching records. Every experiment now retains
its own seeded brain when you switch away or restart. Start and verification
instructions: [Learning observatory](docs/LEARNING_OBSERVATORY.md).
On this Linux workstation, `python3 scripts/observatory.py start` starts both
local services persistently; use `status`, `logs`, `restart`, or `stop` to manage
them. Open <http://127.0.0.1:8780/research.html>.

The controlled teaching battery checks paired, reversed and frozen-learning
conditions in all 14 brain instances. This demonstrates cue memory and isolation;
improved navigation and multi-step planning still require behavioral evidence.

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
PYTHONPATH=. .venv/bin/python -m pytest -q tests/        # no connectome download needed
```

### Behavioural battery (headless)

```bash
PYTHONPATH=. .venv/bin/python -m experiments.run_paradigm_battery --paradigms all --trials 3 --steps 200
PYTHONPATH=. .venv/bin/python -m experiments.run_paradigm_battery --paradigms heat_maze,t_maze,buridan --trials 5 --steps 500
```

Outputs land in `./experiment_data/battery/<paradigm>/` (git-ignored).

### Python API

```python
from arena import Arena
from data_logger import ParadigmDataLogger

arena = Arena(paradigm="heat-maze", num_flies=1, brain_type="modular")
logger = ParadigmDataLogger(paradigm_name="heat-maze", output_dir="experiment_data/demo")
for _ in range(500):
    result = arena.step(0.02)
    logger.log_step(result)
summary_path, npz_path = logger.end_trial(result.get("paradigm_metrics", {}))
```

### Continuous daemon and dashboard

```bash
./start_daemon_ryzen.sh                     # port 8769, writes outputs/learning/*.jsonl
curl -s http://localhost:8769/api/status
python -m http.server 8770 --directory web  # then open http://localhost:8770
./stop_daemon_ryzen.sh
```

Endpoints: `GET /api/status`, `GET /api/telemetry`, `GET /api/paradigms`,
`GET /api/stream` (SSE), `POST /api/command`. The daemon binds `0.0.0.0` by
default; on a shared network prefer `--host 127.0.0.1` or public mode.

### Public, read-only stream

```bash
export NEUROFLY_PUBLIC=1
export NEUROFLY_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/python neurofly_daemon.py --host 127.0.0.1 --public
```

In public mode `POST /api/command` answers `403` unless the request carries
`Authorization: Bearer $NEUROFLY_ADMIN_TOKEN`, concurrent SSE clients are
capped (default 50) and the stream is throttled (default 10 Hz). Put a
TLS-terminating proxy in front. Details: `docs/PUBLIC_STREAMING.md`.

### Full MaleCNS graph (optional, ~1.4 GB download, several GB RAM)

```bash
.venv/bin/python -m brainlab.download      # verifies SHA-256 against brainlab/datasets.json
.venv/bin/python -m brainlab.connectome
.venv/bin/python -m brainlab.prepare
.venv/bin/python -m brainlab.validate_full # see FULL_BRAIN_TEST.md for expected numbers
.venv/bin/python -m brainlab.experiment --seed 42 --repeats 2 && .venv/bin/python -m brainlab.report
```

`BRAINLAB.md`, `RUNS.md` and `LEARNING_DEMO.md` describe that pipeline and its
recorded results.

## Data the daemon keeps

`outputs/learning/trials.jsonl` and `telemetry_summary.jsonl` are append-only,
fsync'd, rotated by size and never deleted by the daemon. Set
`NEUROFLY_DATA_DIR` to move them. Schema: `docs/DATA_SCHEMA.md`.

## Repository layout

```
arena.py maze.py ...          behavioural simulator and the modular brain modules
neurofly_daemon.py            24/7 runner + HTTP/SSE API
stream_gateway.py             public-mode policy (auth, client cap, throttling)
learning_recorder.py          durable JSONL learning records
data_logger.py experiments/   scientific logging, batteries, lesion study
web/                          browser dashboard (independent JS simulation)
brainlab/                     MaleCNS import + fixed-weight LIF kernel
tests/                        pytest suite (CI: .github/workflows/tests.yml)
docs/                         plans, audits, schemas
licenses/ NOTICE              third-party licenses and attribution
data-provenance/ data/        MaleCNS provenance records and two example skeletons
```

`connectome_data/`, `outputs/`, `runs/` and `upstream/` are git-ignored
working directories.

## Status

The suite is exercised on Python 3.12
(`PYTHONPATH=. pytest -q tests/`). CI runs the same command on every pull
request. Test counts in older documents (116, 140, 156, 180) refer to earlier
snapshots.

## License and attribution

Code: MIT (`LICENSE`). Portions derive from
[DOOMFLY](https://github.com/nftechie/doomfly) (MIT). The MaleCNS v1.0
connectome is © the MaleCNS collaboration under CC BY 4.0; the mushroom-body
rule follows Huang, Luo et al. (Nature 2024, CC BY 4.0). Full notices and the
list of what is and is not bundled: `NOTICE`, `licenses/`.

Contributions: `CONTRIBUTING.md`. Release audit: `docs/RELEASE_AUDIT.md`.
