# Audit E: code inventory (stale controllers, dead code, duplicate implementations)

**Date:** 5 October 2026 · **Tree audited:** `origin/master` at `b0abd8f` · **Report only:** nothing was
deleted, moved or edited.

This inventory supports the owner's request to "remove stale controllers and other functions
that do not work, and build up from the core". Every item gets a class:

| Class | Meaning |
|---|---|
| **CORE** | Needed for the working product: the daemon and dashboard (modular and connectome backends), the embodied FlyGym run, the graph pipeline, the CLI and validation. |
| **OPTIONAL** | Supported optional feature, reachable from a supported entry point and tested. |
| **RESEARCH** | Keep, but separate from the product (measurement scripts, one-off studies). |
| **STALE** | Not reachable from a supported entry point, or superseded. Candidate to retire. |
| **BROKEN** | Does not run as committed. Fix or retire. |
| **UNKNOWN** | Needs an owner or legal decision. |

A **receipt** marker (R) means a published receipt or locked declaration cites the item, or the
item produced one. Retire those items to an archive path. Do not delete them.

## 1. Method and evidence

| Check | How | Result |
|---|---|---|
| Static import graph | AST walk of all 243 tracked `.py` files. Covers `import`, `from … import`, relative imports and constant-string `importlib.import_module`/`__import__`. | Importer lists in the tables below |
| Dynamic and subprocess entry points | grep for `-m <module>`, `scripts/*.py`, systemd `ExecStart` (`scripts/observatory.py:38-60`), `Dockerfile`, `start_daemon.sh`, `run_brain.sh`, `pyproject.toml [project.scripts]` | Section 2 |
| Compile | `py_compile` on every tracked `.py` | 0 failures |
| Imported-name check | Every `from <repo module> import name` is resolved against that module's top-level names | 1 failure: `scripts/wp5_photoreceptor_probe.py:48` |
| Dry import | Each library module and each `__main__`-guarded script is imported in a fresh process from a neutral working directory, with `PYTHONPATH` set to the tree and the GPU hidden | 4 failures (Section 6) |
| `--help` smoke | Every `-m` entry point and argparse script | Section 6 |
| Test suite | `pytest -q tests/` on CPU (`CUDA_VISIBLE_DEVICES=""`), no MaleCNS data in the tree | **841 passed, 39 skipped, 0 failed** (4 min 48 s). Skips: no GPU, no real graph, no FlyGym physics flag, no selenium interpreter. |
| Last change | `git log -- <file>` | The repository history starts at `256f7fb` (2026-09-19, initial import). "1 commit @ 256f7fb" means the file has not changed since import. |
| Receipt and doc citations | Path, basename and dotted-module matches in `docs/receipts/**` and active docs | Counts for `arena`, `maze`, `circuit`, `vision` and `locomotion` are inflated, because assay names such as `t-maze` and `open-arena` also match. |

## 2. Entry points and what they reach

| Entry point | How it starts | Reaches (main modules) | Status |
|---|---|---|---|
| `neurofly` console script (`neurofly/cli.py`) | `pyproject.toml [project.scripts]`, `Dockerfile` ENTRYPOINT | `run` → `neurofly_daemon`; `full-sim` → `experiments.full_connectome_simulation`; `embodied` → `neurofly_body.cli`; `download-data` → `brainlab.download`; `status` → `brainlab.graph_identity`, `brain`, `gpu_probe`, `io_map`; `validate` → `validation`; `record` → `neurofly.recording`; `capability` → `docs/CAPABILITY_MATRIX.md` | CORE |
| `neurofly_daemon.py` | `start_daemon.sh`, `neurofly run`, `scripts/observatory.py` (systemd) | `arena`, `maze`, `assay_controls`, `stream_gateway`, `learning_recorder`, `experiment_brains`, `provenance`. Graph backends lazily add `experiment_registry`, `brainlab.*` and `brainlab.wp6_plasticity`. | CORE |
| `python -m neurofly_body` | `neurofly embodied`, `neurofly_body/run_queue.py` (subprocess per job) | `runner`, `flygym_body`, `fast_controller`, `decoder`, `modular`, `body_recording`, `brainlab.cosim_server.ConnectomeServer` (in-process class) | CORE |
| `python -m neurofly_studio` | documented in `docs/EXPERIMENT_STUDIO.md` | `server`, `catalog`, `curate`, `export`, `metrics`, `redact`, `neurofly_body.run_queue` | OPTIONAL |
| `python -m validation` | `neurofly validate` | `harness`, `runner`, `cells`, `stats`, `synthetic`, `paradigms/*`, `brainlab.*`, `experiment_registry` | CORE |
| `python -m brainlab.download` / `.connectome` / `.prepare` / `.graph_identity` | README install steps, `Dockerfile` comments | data pipeline | CORE |
| `python -m brainlab` (`run_brain.sh`), `brainlab.experiment`, `.report`, `.validate_full`, `.learning`, `.demo_server` | `BRAINLAB.md`, `RUNS.md`, `FULL_BRAIN_TEST.md`, `LEARNING_DEMO.md` (all four carry a "v1 engine, will not reproduce" banner) | `brainlab.brain`, `connectome`, `runs` | RESEARCH (pre-v3 tools from the DOOMFLY extraction) |
| `python -m brainlab.cosim_server` (HTTP, port 8768) | only `connectome_client.py` talks to it | `ConnectomeServer` | STALE as a server (the class is CORE) |
| `python -m brainlab.measure_registry` | receipt `wp4_registry_resources.json` | registry | RESEARCH (R) |
| `experiments/*.py` | manual (`python -m experiments.…`) | Section 4.4 | mixed |
| `scripts/*.py` | manual, CI (`check_private_infra.sh`), tests | Section 5 | mixed |
| Standalone pages `demo.html`, `viewer.html`, `learning.html`, `flybrain_scientific_instrument.html` | opened as files, or `learning.html` through `brainlab.demo_server` | none (static) | Section 4.5 |
| `web/` | served by `neurofly_daemon.py:2038,2175` (any file under `web/`), `neurofly_studio/server.py:314`, `http.server` in `scripts/observatory.py:55` | Section 4.5 | mixed |

**Daemon backends actually selectable.** `DAEMON_BACKENDS = ("modular",) + GRAPH_BACKENDS`
(`neurofly_daemon.py:76`, `provenance.py:69`). That means `modular`, `connectome-fixed`,
`connectome-plastic` and `connectome-with-trained-readout`. The two bridge backends in
`provenance.py:42-67` (`bridge-surrogate`, `hybrid-bridge-rpc-experimental`) are **not**
selectable from the daemon or the CLI. The daemon always builds `Arena(brain_type="modular")`
(`neurofly_daemon.py:905`). A graph backend replaces the steering through
`graph_controller` (`arena.py:1317`).

## 3. Controllers, brains and backends

| Controller / implementation | Where | Used by (prod) | Reachable from a supported entry point | Tests | Duplicates | Last change | Docs say | Class |
|---|---|---|---|---|---|---|---|---|
| Modular arena controller (geometric steering + assay reflexes) | `arena.py:1262-1520` (`compute_steering`), `assay_response.py:16` | daemon (`--backend modular`, default of `start_daemon.sh`) | yes | 16 test files import `arena` | `neurofly_body/modular.py` is a separate embodied baseline | arena 2026-09-25 (13 commits); assay_response 1 commit @ da7d4bc 09-19 | "hand-built, not the connectome" (README:25,134; `docs/LEGACY_MODULAR.md`) | **CORE** (as the labelled baseline) |
| Modular sub-circuits: `circuit.MushroomBodyCircuit`, `surge_cast`, `central_complex`, `metabolic`, `mechanosensory`, `vision.CompoundEyeVision`, `locomotion.TripodGaitCPG` | top level | `arena.FlyState.__init__` builds **all** of them for **every** fly, whatever the backend (`arena.py:121-127`). `vision` is also used by `neurofly_body/modular.py`. | yes | `test_flybrain`, `test_sensory_ingress_advanced`, `test_biomechanics_closed_loop` | none | 1-5 commits; `central_complex`, `metabolic` and `surge_cast` unchanged since 09-19 | `LEGACY_MODULAR.md`: "older … phenomenological" | **CORE** while `arena.py` is core. To shrink this, refactor `FlyState` to build them only for `modular`. |
| `circuit.MushroomBodySimulator`, `demonstrate_basic_learning` | `circuit.py:245,311` | none | no | `test_flybrain` only | — | 09-19 | — | **STALE** (test-only helper) |
| `circuit.py:16-17` `from doom_learning_v6.rule import …` | `circuit.py` | the import always fails and the fallback always runs | — | — | — | 09-19 | — | **STALE** (DOOMFLY leftover; dead branch) |
| Modular per-assay brain bank | `experiment_brains.py` (`ExperimentBrain`, `ExperimentBrains`) | daemon | yes | 9 test files | parallel to `experiment_registry` (graph) by design | 3 commits, 09-19 | `NEUROFLY_RETHINK.md` | **CORE** |
| Graph backends `connectome-fixed`, `-plastic`, `-with-trained-readout` | `experiment_registry.py:163,296` (`GraphInstance`, `ExperimentRegistry`), `neurofly_daemon.py:106` (`GraphArenaController`) | daemon (default of `neurofly run`) | yes | 7 + 22 test files | Name collision: `maze.ExperimentRegistry` (`maze.py:2566`) is a different class (the paradigm registry) | 09-25 / 10-05 | README: "exist and run, none validated" | **CORE** |
| WP6 plasticity rule | `brainlab/wp6_plasticity.py` | daemon, `experiment_registry`, `full-sim` | yes (`connectome-plastic`) | `test_wp6_plasticity` | `experiment_registry.TestOnlyCoactivityRule` (test fixture) | 09-24 | spec `WP6_PLASTICITY_SPEC.md` | **CORE** |
| Trained readout | `brainlab/learning.py:16` (`Readout`) | `experiment_registry.py:196` | yes | `test_learning` | — | 09-19 | `LEARNING_DEMO.md` (v1 banner) | **CORE** (the class). The `main()` demo is RESEARCH. |
| `ConnectomeBridge` (`bridge-surrogate`, `hybrid-bridge-rpc-experimental`) | `connectome_bridge.py:83` | `arena.py:36,132` only when `brain_type='connectome'`. That happens only in `experiments/run_paradigm_battery.py` (**its default**, lines 136-137) and `experiments/whole_brain_scientific_battery.py:119,225,266`. | **no** (not in `DAEMON_BACKENDS`; `provenance.py` marks both `scientific=False`) | 7 test files | Duplicates the modular MB/CX/LAL/CPG and the DN decoding (4.2) | 8 commits, last 09-25 | `NEUROFLY_RETHINK.md:52-61`: "hand-built controller … not the dashboard's default" | **STALE** (R: `integration/wp5_live_loop.json`) |
| `ConnectomeClient` (RPC) | `connectome_client.py:44` | `connectome_bridge` (rpc mode) | no | `test_provenance_registry`, `test_whole_brain` | — | 5 commits, last 10-05 | `RELEASE_AUDIT.md:30` | **STALE** (R: same receipt) |
| `ConnectomeServer` class | `brainlab/cosim_server.py:70` | `neurofly_body/cli.py:246,266` (in-process) | yes | 11 test files | DN readout overlaps `neurofly_body.decoder` | 13 commits, last 10-05 | `EMBODIED_MVP.md` | **CORE** |
| Its HTTP layer (`CoSimHTTPHandler`, `run_server`) | `cosim_server.py:540-611` | `connectome_client` only | no | indirectly | — | — | — | **STALE** (goes with the bridge) |
| Full-sim single-brain loop | `experiments/full_connectome_simulation.py` | `neurofly full-sim` | yes | `test_daemon_dynamics` | A third controller loop beside the daemon registry and the embodied runner. It builds `Arena(brain_type="modular")` with "Placeholder flag" (line 395). | 9 commits, last 09-24 | CHANGELOG #7. Absent from the capability matrix. | **OPTIONAL**. Question for the owner: keep it as a product command? |
| Embodied connectome loop | `neurofly_body/runner.py`, `cli.py`, `flygym_body.py`, `fast_controller.py`, `interfaces.py` | `neurofly embodied` | yes | 8+ test files; physics tests gated by `NEUROFLY_RUN_PHYSICS=1` | — | 09-25 | `EMBODIED_MVP.md`: "the MVP" | **CORE** |
| Embodied modular baseline | `neurofly_body/modular.py` | `neurofly_body/cli.py` (`--controller modular`) | yes | `test_embodied_modular` | reuses `vision.CompoundEyeVision` (good) | 09-25 | module docstring | **CORE** (baseline) |
| `env_adapter.FlyBrainEnv` (Gymnasium) | `env_adapter.py:58` | none | no | `test_flybrain` | wraps the modular sub-circuits a second time | 1 commit @ 09-19 | docstring mentions "Doom learning v6 / Brainlab pipelines" | **STALE** |
| Lesion `FlyState` battery | `experiments/lesion_study.py` | none | no (manual) | `test_lesion_study` | — | 09-19 | not documented | **RESEARCH** |

## 4. Duplicate implementations

### 4.1 LIF engines (`brainlab/engine.py`) and GPU builds

| Engine | CPU kernel | GPU build | How selected | Used by product? | Tests | Receipts | Class |
|---|---|---|---|---|---|---|---|
| v1 current-based | `engine.py:69 advance` | none | `--dynamics v1` / `NEUROFLY_LIF_DYNAMICS=v1` | selectable only | `test_lif_dynamics` | `lif_dynamics_diagnosis.json`, `lif_dynamics_v2.json`, the root `*.md` v1 results | **RESEARCH** (R). Keep for reproducing old receipts. |
| v2 conductance | `engine.py:107 advance_v2` | none | `--dynamics v2` | selectable only | `test_lif_dynamics` | `lif_dynamics_v2.json` | **RESEARCH** (R) |
| **v3** (default, `graph_identity.py:365`) | `engine.py:168 advance_v3` | `cuda_engine.py` (numba.cuda) **and** `cupy_engine.py` (CuPy, a line-for-line translation) | default | **yes** | `test_lif_dynamics`, `test_cuda_engine` (all 7 GPU tests skipped without GPU/CUDASIM) | `lif_dynamics_v3.json` | **CORE** |
| v4 graded | `engine.py:242 advance_v4` + `graded_policy.py` | `cupy_v4.py` | env `NEUROFLY_LIF_DYNAMICS=v4` only (the daemon's `--dynamics` accepts only v1-v3, `neurofly_daemon.py:2357`) | experimental | `test_graded_v4` | `lif_dynamics_v4*.json`, `graded_transmission_v4_declaration.locked.md` | **OPTIONAL/RESEARCH** (R) |
| v5 receptor kinetics | `engine.py:355 advance_v5` + `receptor_kinetics.py` | `cupy_v5.py` (no test imports it directly) | env only | experimental | `test_kinetics_v5` (GPU parts skipped) | `lif_dynamics_v5.json`, `v5_raw/` | **OPTIONAL/RESEARCH** (R) |

The v3 GPU path has two builds. `pyproject.toml` (the `gpu` extra comment) says not to install
numba-cuda, so on a user machine `cuda_engine.make_state` (`cuda_engine.py:70-85`) always picks
the **CuPy** build. The numba.cuda kernel runs only under `NUMBA_ENABLE_CUDASIM=1` in tests, or
with `NEUROFLY_CUDA_IMPL=numba`. Both builds are recent (09-24 to 10-05) and documented. This is
a candidate to collapse later, but it is not stale.

### 4.2 Sensory encoders and motor decoders

| Function | Implementations | Which are live | Notes |
|---|---|---|---|
| Optomotor encoder | (a) `brainlab/io_map.py:159 OptomotorEncoder` (T4/T5 drive imposed by the encoder; the pinned WP5 map); (b) `brainlab/io_map_photoreceptor.py:259 PhotoreceptorGratingEncoder` (R1-R6 only, WP5 photoreceptor null result); (c) `brainlab/photoreceptor_io.py:201 PhotoreceptorGratingEncoder` (R1-R6 for LIF v4/v5) | (a) daemon `GraphArenaController`, `validation/paradigms/optomotor.py`, `neurofly status`; (b) scripts only (`v5_measurement/yaw_axes.py`, the broken `wp5_photoreceptor_probe.py`); (c) tests plus v4/v5 measurement scripts | (b) and (c) are two implementations of the same idea, with the **same class and function names** (`PhotoreceptorIOMap`, `resolve_photoreceptor_io`, `PhotoreceptorGratingEncoder`) and separate pins. Both carry receipts. Consolidate later. Do not delete either. |
| Sensory injection into the graph (embodied) | `cosim_server.ConnectomeServer.step` (`cosim_server.py:337`); `io_map.LegLoadEncoder` (`io_map.py:578`) | CORE | — |
| Sensory ingress (surrogate) | `connectome_bridge.py` (vision, olfaction, JO, proprioception) | STALE | hand-built; duplicates the modular sub-circuits |
| DN → 2D yaw | `io_map.py:212 DNa02YawDecoder` | daemon, validation | CORE |
| DN → FlyGym CPG | `neurofly_body/decoder.py:10 DNa02CPGDecoder` (`--decoder dna02-crossed-v1`), `decoder.py:92 DNCommandDecoder` (`dn-v2`, the default, `neurofly_body/cli.py:71`) | embodied | Two decoders by design (versioned). Keep both: recorded runs name them. |
| DN readout on the server | `cosim_server.py:504 _locomotion_dn_reply` | embodied | CORE |
| DN decoding (surrogate) | `connectome_bridge.py:234-235` and onward (`dna02_rate_l/r`, `MDN`, `GF`, …) | STALE | the third DN decoding path |

### 4.3 Daemons and servers

| Server | File | Port | Started by | Class |
|---|---|---|---|---|
| Main daemon + dashboard API | `neurofly_daemon.py` | 8769 | `neurofly run`, `start_daemon.sh`, Docker | **CORE** |
| Observatory (systemd) | `scripts/observatory.py` → daemon on 8781, `http.server` for `web/` on 8780, `scripts/research_worker.py` | 8780/8781 | manual install of user services | **OPTIONAL** (the owner's running observatory; `live_ui_signoff.py` targets it) |
| Co-simulation RPC | `brainlab/cosim_server.py run_server` | 8768 | manual | **STALE** (only the RPC bridge uses it) |
| Learning demo | `brainlab/demo_server.py` → `learning.html`, `learning-data.json` | 8767 | `LEARNING_DEMO.md` | **RESEARCH**. No tests; v1-era results. |
| Experiment studio | `neurofly_studio/server.py` | configurable | `python -m neurofly_studio serve` | **OPTIONAL** |
| Stub daemon (test harness) | `scripts/browser_optomotor_stub_daemon.py` | — | `scripts/browser_firefox_check.py` | **RESEARCH** (test harness) |
| Manual graph smoke | `scripts/test_srv.py` (unguarded, loads the full graph at import) | — | `docs/archive/CODEX_HANDOFF.md` only | **STALE** (superseded by `neurofly status` and the tests) |

### 4.4 Experiment runners and batteries

| Runner | Controller it actually drives | Reachable | Tests | Receipts / data | Class |
|---|---|---|---|---|---|
| Daemon live assays | modular or graph | `neurofly run` | many | live-signoff receipts | CORE |
| `validation/` harness | graph (`experiment_registry`) | `neurofly validate` | `test_validation_harness`, `test_wp7_graph_policy` | `validation/specs/*` (preregistered) | CORE |
| `experiments/full_connectome_simulation.py` | graph, single brain | `neurofly full-sim` | `test_daemon_dynamics` | README mention | OPTIONAL |
| `experiments/run_paradigm_battery.py` | **surrogate `ConnectomeBridge` by default** (`brain_type="connectome"`, `connectome_mode="surrogate"`, lines 136-137) | manual | `test_paradigm_logger` (subprocess) | produced `experiment_data/ryzen_battery/**` (120 files, 2026-09-18, e.g. optomotor gain 0.0000) | **STALE**. It labels the surrogate run as connectome. |
| `experiments/whole_brain_scientific_battery.py` | surrogate `ConnectomeBridge` ("Full MaleCNS v1.0 ConnectomeBridge" in its docstring is wrong) | manual. It has no argparse, so `--help` starts a run that writes `./experiment_data/` into the working directory. | none | none | **STALE** |
| `experiments/lesion_study.py` | modular `FlyState` with ablation switches | manual | `test_lesion_study` | `experiments/data/lesion_*` (09-19) | RESEARCH |
| `scripts/research_worker.py`, `scripts/learning_battery.py` | modular `ExperimentBrains` | observatory / manual | `test_research_worker` | `web/research-status.json` (runtime) | OPTIONAL / RESEARCH |
| `scripts/generate_phase7_receipts.py` | daemon runner, `connectome-fixed`/`-plastic` | manual | none | **R**: wrote `connectome_closed_loop_{optomotor,looming,wp6_plasticity}.json` (lines 73, 136, 175) | RESEARCH (R) |
| `brainlab.experiment` / `.learning` / `.report` / `.validate_full` | raw `Brain`, v3 by default | manual | `test_learning`, `test_runs` only | root docs (v1 results) | RESEARCH |

### 4.5 Dashboards and pages

| Page | Size | Served / used by | Last change | Class |
|---|---|---|---|---|
| `web/index.html` + `app.js`, `live_assays.js`, `replay.js`, `training.js`, `training.css`, `vendor/*` | 90 KB + 343 KB | daemon, observatory | 10-05 (23 commits on app.js) | **CORE** |
| `web/studio.html`, `studio.js`, `embodied_replay.html`, `embodied_replay.js` | — | studio server; `body.nfbody` replay | 09-25 / 10-05 | **OPTIONAL** |
| `web/research.html` | 6 lines | a redirect to `index.html` (`web/research.html:4`) | 09-19 | **STALE** (harmless; keep the redirect or drop it) |
| `web/research/app.js`, `web/research/style.css` | — | **nothing**: no page loads them | 09-19 | **STALE** |
| `flybrain_scientific_instrument.html` | 317 KB | **nothing**. A single-file fork of the dashboard titled "v1.0 Release". Its bundler `scratch/sync_standalone.py` is not in the repository (`RELEASE_AUDIT.md:54`). It lacks the identity-rejection and backend-switch logic that `web/app.js` has. It still carries LAN/host strings (`RELEASE_AUDIT.md:31`). Listed in `sync_ecosystem.py`. | 09-24 | **STALE** |
| `demo.html` (1.4 MB, generated), `viewer.html`, `build_demo.py`, `data/12781.swc`, `data/556329.swc`, `data/manifest.json` | — | standalone offline skeleton viewer for two DNge104 neurons. `build_demo.py` runs at import (no `main`). | 09-19 (1 commit) | **UNKNOWN** (owner decision, already open in `RELEASE_AUDIT.md:53,115`). Not product. |
| `learning.html`, `learning-data.json` | 10 KB, 45 KB | `brainlab/demo_server.py`, `scripts/check_learning_ui.cjs` | 09-19 | **RESEARCH** (v1-era readout demo) |

## 5. Scripts

| Script | Purpose | Runs on master? | Tests / CI | Receipt | Class |
|---|---|---|---|---|---|
| `check_private_infra.sh`/`.py`, `private_infra_allowlist.txt` | release-hygiene guard | yes | CI + `test_private_infra_guard` | `REDACTIONS.md` | **CORE** (tooling) |
| `live_ui_signoff.py` | AGENTS.md real-Firefox sign-off | yes (`--help` ok) | — | **R** live-signoff, switch-race, pr30 | **CORE** (tooling) |
| `browser_firefox_check.py` + `browser_optomotor_stub_daemon.py` | headless Firefox dashboard check | yes | `test_browser_recovery` (skipped without selenium) | **R** `integration/firefox_headless_receipt*.json` | OPTIONAL |
| `benchmark.py`, `gpu_parity.py` | host speed, GPU vs CPU | yes | `test_benchmark` | **R** README, `determinism-ryzen-vs-amd-20261005.md` | OPTIONAL |
| `containment_audit.py`, `behavior_audit.py`, `timing_stress.py`, `gil_starvation_probe.py` | audits | yes | `test_containment_all_paradigms` (subprocess) | **R** `wp1_wp2/lock_profile_receipt.json` | RESEARCH (R) |
| `dashboard_headless_check.py` | Playwright dashboard check | **no**: `ModuleNotFoundError: playwright`. Playwright is in no extra in `pyproject.toml`. | — | **R** `wp1_wp2/lock_profile_receipt.json` | **BROKEN** (undeclared dependency). Retire it in favour of the Firefox checks, or declare the dependency. |
| `studio_headless_check.py` | Playwright studio walk-through | **no** (same) | — | — (`EXPERIMENT_STUDIO.md` documents it) | **BROKEN** (undeclared dependency) |
| `wp5_optomotor.py`, `wp5_receipt.py`, `lif_dynamics_diagnosis.py`, `lif_dynamics_v2_receipt.py`, `_make_partial_receipt.py` | WP5 and LIF v2 receipts | yes (`--help` ok, or assemble-only) | — | **R** `wp5_optomotor.json`, `lif_dynamics_v2.json` | RESEARCH (R) |
| `wp5_photoreceptor_probe.py` | photoreceptor DS probe | **no**: `ImportError` at line 48 (`active_e_inh_mV`, `dynamics_variant`, `dynamics_variant_pin` do not exist in `brainlab/graph_identity.py`) | — | **R** `photoreceptor_encoder.json` | **BROKEN** (see 6.1) |
| `wp5_photoreceptor_receipt.py` | assembles the receipt above | yes (assemble-only) | — | **R** | RESEARCH (R) |
| `v4_measurement/*` (20 files, no `__main__` guards, shared `_paths.py`) | LIF v4 measurement → `lif_dynamics_v4*.json`. `apply_docs.py` and `append_spec.py` patch docs in place. | names resolve; needs the real graph (not run) | — | **R** (`apply_docs.py:42` → `lif_dynamics_v4.json`) | RESEARCH (R). Mark `apply_docs`/`append_spec` as one-shot. |
| `v5_measurement/*` (10 files, unguarded) | LIF v5 measurement → `lif_dynamics_v5.json`, `v5_raw/` | names resolve; needs the graph (not run) | — | **R** (`build_receipt.py:2,10-23`) | RESEARCH (R) |
| `generate_phase7_receipts.py` | see 4.4 | not run (needs the graph) | — | **R** | RESEARCH (R) |
| `mb_circuit_census.py` | WP7 MB census | yes | — | `docs/design/mb_circuit_census.json` | RESEARCH |
| `observatory.py`, `research_worker.py`, `learning_battery.py` | observatory services | yes | `test_research_worker`, `test_observatory_service` | — | OPTIONAL |
| `test_srv.py` | manual full-graph smoke (unguarded) | not run (loads the graph at import) | — | — | **STALE** |
| `check_learning_ui.cjs` | node check of `learning.html` | not run | — | — | RESEARCH (goes with `learning.html`) |
| `containment_harness.js`, `js_syntax_check.js` | gjs checks of `web/app.js` | not run | — | — | OPTIONAL (tooling) |
| `validation/specs/sources/dna02_yang2024_figS7A.py` | source-figure digitiser | **no**: needs `pdfplumber` (undeclared) | — | produced `dna02_rates_read.json` (spec input) | RESEARCH (keep with the spec) |
| `sync_ecosystem.py` | copy the tree to remote/Docker/archive targets | runs, but **its `SYNC_MANIFEST` (`sync_ecosystem.py:35`) omits `neurofly/`, `neurofly_body/`, `neurofly_studio/`, `validation/`, `provenance.py`, `experiment_registry.py`, `experiment_brains.py`, `assay_controls.py`, `assay_response.py`, `online_metrics.py`, `pyproject.toml`**. A synced target cannot import the daemon. It does include the stale `flybrain_scientific_instrument.html`, `connectome_bridge.py` and `env_adapter.py`. | — | — | **BROKEN** (retire; git and the Docker image cover this) |

## 6. Dead and broken code

### 6.1 Broken on master

1. **`scripts/wp5_photoreceptor_probe.py:48`** imports three names that do not exist on master.
   They were added by commit `520efcb` ("pre-registered E_inh sweep", 2026-09-26). That commit is
   **only on the local branch `worktree-neurofly-openready`: it is on no remote branch**
   (`git branch -r --contains 520efcb` is empty). Commit `45156eb` ported the probe without its
   dependency.
2. **Related reproducibility gap (severe for receipts):** `docs/EINH_SENSITIVITY.md` and
   `docs/receipts/einh_sensitivity.json` (ported docs-only in `69935e9`) describe `v3-einh<value>`
   variants, `NEUROFLY_LIF_E_INH_MV` and `gate_sweep.py`. None of that code is on master or on
   GitHub. The `photoreceptor-einh60` arm of `photoreceptor_encoder.json` depends on it too.
   **Recommendation:** push `520efcb` (or its graph_identity part) to a remote archive branch
   before any cleanup, then either port it or retire the probe with a pointer to that commit.
3. `scripts/dashboard_headless_check.py` and `scripts/studio_headless_check.py` need `playwright`,
   which is in no extra.
4. `validation/specs/sources/dna02_yang2024_figS7A.py` needs `pdfplumber` (undeclared).
5. `sync_ecosystem.py`: the manifest is out of date (Section 5).
6. Entry points that ignore `--help` and **run**: `experiments/whole_brain_scientific_battery.py`
   (writes `./experiment_data/` into the working directory), `brainlab/prepare.py` and
   `brainlab/validate_full.py` (they need data and traceback without it), `build_demo.py`,
   `scripts/test_srv.py` and every `scripts/v4_measurement`/`v5_measurement` file. This is a
   hazard for a beginner, not dead code.

Everything else compiles, imports and passes its tests. All other `-m` entry points print usage.

### 6.2 Dead or unreachable (no product importer)

| Item | Evidence | Class |
|---|---|---|
| `circuit.py:16-17` DOOMFLY `doom_learning_v6` import and its fallback | the package exists nowhere | STALE |
| `circuit.MushroomBodySimulator`, `demonstrate_basic_learning` (`circuit.py:245-358`) | only `tests/test_flybrain.py` | STALE |
| `env_adapter.py` | only `tests/test_flybrain.py` | STALE |
| `connectome_bridge.py`, `connectome_client.py`, `cosim_server.run_server`/`CoSimHTTPHandler` | reached only from two stale experiment batteries and tests | STALE (R) |
| `data_logger.py` (1,334 lines) | only `experiments/run_paradigm_battery.py`, `whole_brain_scientific_battery.py` and tests | STALE with the batteries |
| `experiments/whole_brain_scientific_battery.py` | no importer, no test, no receipt | STALE |
| `brainlab/__main__.py`, `run_brain.sh`, `brainlab/experiment.py`, `report.py`, `validate_full.py`, `demo_server.py` | no tests; documented only in root docs whose v1 results no longer reproduce | RESEARCH (archive candidates) |
| `web/research/app.js`, `web/research/style.css` | no page loads them | STALE |
| `flybrain_scientific_instrument.html` | no importer, no server route | STALE |
| `docs/archive/root_duplicates_20260924/*.py` | already archived byte-identical copies, which still import live modules | already retired |

### 6.3 Data files

| Data | Read by | Class |
|---|---|---|
| `brainlab/datasets.json`, `brainlab/graph_pins.json`, `data-provenance/malecns_v1/*` | `brainlab.download`, `graph_identity`, `connectome_client` | CORE |
| `validation/specs/*.json` incl. superseded `firing_rate_bounds_v1/v2`, `optomotor_v3`, `optomotor_v3_2` | the harness and preregistration | CORE. Keep superseded specs as preregistration records. |
| `experiment_data/curated_plan/*.json` | studio experiment plans (`neurofly_studio`) | OPTIONAL |
| `experiment_data/ryzen_battery/**` (120 files) | **nothing reads them**. Surrogate-bridge battery output from 2026-09-18. Host-named path flagged in `RELEASE_AUDIT.md:34`. | STALE (archive) |
| `experiments/data/lesion_*` | nothing reads them (output of `lesion_study.py`) | RESEARCH |
| `learning-data.json` | `learning.html`, written by `brainlab.learning` | RESEARCH |
| `data/*.swc`, `data/manifest.json` | `build_demo.py` | UNKNOWN (goes with `demo.html`) |
| `docs/design/mb_circuit_census.json`, `docs/wp5_optomotor_prereg.json`, `docs/wp6_plastic_subset.json` | specs | keep |
| `licenses/upstream/npm/*` (19 React/shadcn/vinext packages), `Freedoom-BSD-3-Clause.txt`, `ViZDoom-MIT.txt`, `dependency-notices.json` ("pinned in doom-ui/package-lock.json") | notices for DOOMFLY game/UI components that are **not** in this repository | **UNKNOWN**. Needs legal review before removal. `brainlab` was extracted from DOOMFLY (`UPSTREAM.json`, `brainlab/__init__.py:1`), so the `DOOMFLY-*` and `Shiu-model-MIT` notices must stay. |

### 6.4 Docs describing removed, legacy or never-ported features

| Doc | Issue |
|---|---|
| `BRAINLAB.md`, `RUNS.md`, `FULL_BRAIN_TEST.md`, `LEARNING_DEMO.md` (root) | v1-engine results that do not reproduce under the v3 default. Each has a banner saying so. They also keep absolute-path items (`RELEASE_AUDIT.md:32`, open). Candidates to move under `docs/archive/` or `docs/history/`. |
| `VNC_PREMOTOR_CONNECTOME_MAPPING.md` (root) | a literature blueprint, written in the bridge era. Cites `ScientificDataLogger`/`LearningAssay` as "empirical data export" (line 236). Not current architecture. |
| `docs/FULL_CONNECTOME_COSIM_PLAN.md:88-106` | plans `[MODIFY] connectome_bridge.py`. The architecture moved to `neurofly_body` + `ConnectomeServer`. |
| `docs/EINH_SENSITIVITY.md` | describes code that is not on master (6.1). |
| `docs/WP7_MB_LEARNING_SPEC.md` | cites `brainlab/mb_plasticity.py`, `scripts/wp7_calibrate_eta.py`, `validation/paradigms/mb_e0_kc_regime.py`. These are future work under a frozen spec. Fine, but not present. |
| `docs/REPOSITORY_LAYOUT.md` | calls `neurofly_body` "the publication-facing MVP" and the daemon "preserved modular baseline". The README now presents the daemon with graph backends as a main product. The docs disagree about what "core" is (see Section 8). |
| `env_adapter.py:11`, `brainlab/__init__.py:1`, `brainlab/connectome.py:1`, `brainlab/transmitters.py:1` | docstrings still say "DOOMFLY" / "Doom learning v6". This is provenance, but it confuses a new reader. |

## 7. Receipt reproducibility: archive, never delete

Retiring any of these must move them to an archive path, such as `archive/<date>/…` or a
retained archive branch. Receipts cite them by path or were produced by them.

| Item | Receipts |
|---|---|
| `connectome_bridge.py`, `connectome_client.py`, `cosim_server` HTTP layer | `integration/wp5_live_loop.json` |
| `scripts/wp5_optomotor.py`, `wp5_receipt.py`, `lif_dynamics_diagnosis.py`, `lif_dynamics_v2_receipt.py`, `_make_partial_receipt.py` | `wp5_optomotor.json`, `lif_dynamics_v2.json`, `lif_dynamics_diagnosis.json` |
| `scripts/wp5_photoreceptor_probe.py`, `wp5_photoreceptor_receipt.py`, `brainlab/io_map_photoreceptor.py` | `photoreceptor_encoder.json`, `photoreceptor_encoder_declaration.locked.md`, `v5_raw/yaw_axes.json` |
| `scripts/generate_phase7_receipts.py` | `connectome_closed_loop_{optomotor,looming,wp6_plasticity}.json` (the capability matrix cites the WP6 one) |
| `scripts/v4_measurement/*`, `brainlab/photoreceptor_io.py`, `graded_policy.py`, `cupy_v4.py` | `lif_dynamics_v4*.json`, `graded_transmission_v4_declaration.locked.md` |
| `scripts/v5_measurement/*`, `receptor_kinetics.py`, `cupy_v5.py` | `lif_dynamics_v5.json`, `v5_raw/*`, `receptor_kinetics_v5_declaration.locked.md` |
| `brainlab/measure_registry.py` | `wp4_registry_resources.json` |
| `scripts/containment_audit.py`, `timing_stress.py`, `gil_starvation_probe.py`, `dashboard_headless_check.py` | `wp1_wp2/lock_profile_receipt.json` |
| `scripts/browser_firefox_check.py`, `live_ui_signoff.py` | `integration/firefox_headless_receipt*.json`, `live-signoff*/`, `switch-race-20261004/`, `pr30-review-20261005/` |
| `scripts/gpu_parity.py`, `benchmark.py` | README performance table, `determinism-ryzen-vs-amd-20261005.md` |
| `engine.advance` (v1), `advance_v2` and the `--dynamics v1/v2` switches | v1/v2 receipts and the four root docs |
| `docs/receipts/switch-race-20261004/scripts/*`, `pr25-review-20261005/*.py`, `pr30-review-20261005/scripts/*` | these receipt scripts import `neurofly_daemon`, `brainlab`, `experiment_registry` and `neurofly_studio.redact` internals. Refactoring those APIs breaks re-running them at HEAD (they stay valid at their commits). |
| Local-only commit `520efcb` (branch `worktree-neurofly-openready`) | `einh_sensitivity.json`, the `photoreceptor-einh60` arm. **Push it to a remote archive branch first.** |

## 8. Proposed core and retirement plan (input for the orchestrator)

### 8.1 Proposed minimal core: a working product

- **CLI:** `neurofly/` (cli, recording, `__main__`), `pyproject.toml`, `Dockerfile`, `start_daemon.sh`, `stop_daemon.sh`.
- **Graph pipeline:** `brainlab/download.py`, `connectome.py`, `prepare.py`, `transmitters.py`, `transmitter_policy.py`, `graph_identity.py`, `graph_pins.json`, `datasets.json`, `data-provenance/`.
- **Engine:** `brainlab/brain.py`, `engine.py` (v3; v1/v2 kept for receipts), `gpu_probe.py`, `cupy_engine.py` + `cuda_engine.py` (the v3 GPU path).
- **I/O maps:** `brainlab/io_map.py` (optomotor, visual heading, locomotion DNs, silence, leg load).
- **Live daemon and dashboard:** `neurofly_daemon.py`, `stream_gateway.py`, `learning_recorder.py`, `provenance.py`, `experiment_registry.py`, `brainlab/wp6_plasticity.py`, `brainlab/learning.py` (`Readout`), `brainlab/runs.py`, `arena.py`, `maze.py`, `online_metrics.py`, `assay_controls.py`, `experiment_brains.py`, the modular sub-circuits (`circuit`, `surge_cast`, `central_complex`, `metabolic`, `mechanosensory`, `vision`, `locomotion`, `assay_response`), and `web/index.html` + `app.js`, `live_assays.js`, `replay.js`, `training.js/.css`, `vendor/`.
- **Embodied:** `neurofly_body/*`, `brainlab/cosim_server.py` (`ConnectomeServer` class).
- **Validation:** `validation/*` + specs.
- **Tooling:** `scripts/check_private_infra.*`, `scripts/live_ui_signoff.py`, and `tests/`.

**Owner decision needed:** the docs disagree on what "core" is. `REPOSITORY_LAYOUT.md` says
`neurofly_body` is the MVP and the daemon is a preserved baseline. The README presents the
daemon with graph backends as the main product. "Build up from the core" needs one of these
named as the primary product.

### 8.2 Retirement candidates (archive path; risk notes)

| # | Candidate | Risk |
|---|---|---|
| 1 | `connectome_bridge.py`, `connectome_client.py`, the `cosim_server` HTTP server, and the `bridge-surrogate`/`hybrid-bridge-rpc-experimental` entries in `provenance.BACKENDS` | **Medium.** `arena.py:28-37` imports `ConnectomeBridge` unconditionally inside a `try` whose `except` falls back to *relative* imports. Deleting the file therefore breaks `import arena` and the whole daemon. Edit `arena.py` first: drop `brain_type='connectome'`, `connectome_mode` and the bridge branches at `arena.py:132,1273,1333`. 7 test files must be retired or rewritten. Receipt-bearing, so archive. |
| 2 | `experiments/run_paradigm_battery.py`, `experiments/whole_brain_scientific_battery.py`, `data_logger.py`, `experiment_data/ryzen_battery/**` | Low. Goes with item 1. `test_paradigm_logger` and `test_whole_brain` must go too. Alternatively fix `run_paradigm_battery` to default to `modular` and label it honestly. |
| 3 | `env_adapter.py`, `circuit.MushroomBodySimulator`/`demonstrate_basic_learning`, the `doom_learning_v6` import branch | Low. Only `test_flybrain` cases depend on them. |
| 4 | `flybrain_scientific_instrument.html`, `web/research/` (and optionally `web/research.html`) | Low. No consumers. Removes the stale LAN/host strings that `RELEASE_AUDIT.md` item 9 (line 31) flags. |
| 5 | `sync_ecosystem.py` | Low. Broken manifest; git and Docker replace it. |
| 6 | `scripts/test_srv.py` | Low. |
| 7 | `scripts/dashboard_headless_check.py`, `scripts/studio_headless_check.py` | Low/medium. Either declare `playwright` in an extra or retire them in favour of the Firefox/selenium checks. The first is receipt-bearing, so archive it. |
| 8 | DOOMFLY-era brainlab tools: `brainlab/__main__.py`, `run_brain.sh`, `experiment.py`, `report.py`, `validate_full.py`, `demo_server.py`, `learning.html`, `learning-data.json`, `scripts/check_learning_ui.cjs`, root `BRAINLAB.md`/`RUNS.md`/`FULL_BRAIN_TEST.md`/`LEARNING_DEMO.md` | Low/medium. Move them to a `research/` or `archive/` area. Keep `brainlab.learning.Readout` and `brainlab.runs` in core, because `experiment_registry` and the readout backend use them. |
| 9 | `demo.html`, `viewer.html`, `build_demo.py`, `data/*.swc`, `data/manifest.json` | Owner decision (already open in RELEASE_AUDIT). |
| 10 | `scripts/wp5_photoreceptor_probe.py` | **First** push `520efcb` to a remote archive branch. Then fix it (port the `graph_identity` variant API) or archive it with a pointer. |
| 11 | `licenses/upstream/npm/*`, `Freedoom`, `ViZDoom` notices | Legal review only. Not a code task. |

### 8.3 Research-only: keep, but separate from the product

Move these into a clearly labelled `research/` tree without changing their content, or leave
them in place with a README:

- `scripts/v4_measurement/`, `scripts/v5_measurement/`, `scripts/wp5_*`, `scripts/lif_dynamics_*`, `scripts/_make_partial_receipt.py`, `scripts/generate_phase7_receipts.py`
- `brainlab/measure_registry.py`, `brainlab/io_map_photoreceptor.py`, `brainlab/photoreceptor_io.py`, the v4/v5 engine paths and `cupy_v4/v5`
- `scripts/mb_circuit_census.py`, `experiments/lesion_study.py` + `experiments/data/`
- `scripts/containment_audit.py`, `behavior_audit.py`, `timing_stress.py`, `gil_starvation_probe.py`

Moving a receipt-cited script changes the path that the receipt cites. Prefer leaving a
one-line stub or a README at the old path that names the new location.

### 8.4 Consolidation candidates (do later, not by deletion)

- Merge `photoreceptor_io.py` and `io_map_photoreceptor.py` into one module with two pinned variants. The class names are currently identical.
- Rename `maze.ExperimentRegistry` (the paradigm registry) to end the name collision with `experiment_registry.ExperimentRegistry`.
- Collapse the numba.cuda v3 build into CuPy only, once the tests use the CuPy build or a CPU parity test instead of CUDASIM.
- Build the modular sub-circuits in `arena.FlyState` only for the `modular` backend. Today every connectome-backend fly also builds an MB, CX, JO, vision model and CPG it does not use.
- Make all entry points that run on import or ignore `--help` (Section 6.1, item 6) parse arguments first.
