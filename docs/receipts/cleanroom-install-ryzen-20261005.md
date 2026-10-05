# Clean-room install, NVIDIA Linux (Ryzen), 5 October 2026: gates G1 and G3

Tester: a Claude worker acting as a stranger. It did not use the owner's checkout, venv,
`outputs/` or `connectome_data/`, or any knowledge of how the owner's machine is set up.
When something failed, the failure was recorded first. Only then was it worked around,
and the workaround is described.

## Verdict

| gate | verdict | why |
|---|---|---|
| **G1** clean-room install on NVIDIA Linux, README followed verbatim | **FAIL** | Following the README literally, `neurofly run` exits at startup (B1, B2) and `docker build` fails (B3). With the workarounds below, everything else passed: the install, 706 tests with 0 failures, the dashboard in real Firefox (7/7 sign-off) and the documented embodied experiment (bit-identical replay). |
| **G3** data path documented and working | **FAIL** | `neurofly download-data` fetches and hash-checks the three Janelia tables (1.1 GB in 36 s). But the graph identity check then fails on a fresh install (B1). The two preparation steps are not in the README (B2), and no attribution is shown (m5). |

Total time from clone to a working dashboard was 4 min 49 s of machine time
(15:06:55 to 15:11:44 CEST). That includes this tester's diagnosis of B1 and B2. The
browser-confirmed live dashboard came at about 15:15, and the 7/7 sign-off finished at
15:17:54. A real stranger without the workarounds does not get there at all on the
README alone. With the workarounds in hand, expect about 5 minutes on a fast connection.

## Environment

| item | value |
|---|---|
| clone | `git clone https://github.com/average-user-887/flybrain.git` (HTTPS) → `9a6c1c57743b47017f5c2b75d837132ee3f8c682` ("docs: release plan for v0.4.0"). `origin/master` later moved to `93f0b18`, which changes docs only and leaves the README untouched. |
| location | `<worktree>/.cleanroom-20261005/flybrain` (not a git worktree; left on disk) |
| Python | system `/usr/bin/python3` 3.12.3, fresh `python3 -m venv .venv` |
| caches | `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `MPLCONFIGDIR` redirected into the clean room; `PYTHONPATH` and all `NEUROFLY_*` unset |
| OS / GPU | Ubuntu, kernel 7.0.0; NVIDIA driver 580.178.04 (CUDA 13.0); GPU 0 Quadro P620 2 GB (display), GPU 1 GTX 1660 Ti 6 GB; no CUDA toolkit installed |
| browser | Firefox 157.0 (snap), geckodriver `/snap/bin/geckodriver`, selenium 4.50.0 in a separate venv |
| ports used | 8791 (daemon), 8792 (static web). The owner's 8769/8780/8781 were never contacted. Every dashboard URL carried an explicit `?daemon=` or was served by our own daemon. |

## Step log

Wall times come from the step wrapper. Severity codes: **B** = BLOCKER, **M** = MAJOR,
**m** = MINOR, **n** = NOTE. They refer to the findings list below.

| # | command (cwd = clone) | wall | result |
|---|---|---|---|
| 1 | `git clone https://github.com/average-user-887/flybrain.git` | ~5 s | OK, 82 MB. The README gives `git@github.com:...` (SSH) instead (m1). |
| 2 | `python3 -m venv .venv` | 2.0 s | OK |
| 3 | `pip install -e ".[body,test]"` | 39.9 s | OK. It resolved numpy 2.5.3, numba 0.68.0, **pandas 3.0.6**, pyarrow 25.0.1, flygym 2.1.0, mujoco 3.9.0, brian2 2.10.1, pytest 9.1.1. No CuPy or CUDA package was installed (M1). |
| 4 | `neurofly --help` | <1 s | OK, 8 subcommands |
| 5 | `neurofly status` (before data) | 2.8 s | Graph not found. It suggests `NEUROFLY_GRAPH_DIR`, but nothing says how to make `graph.npz` (B2). |
| 6 | `neurofly capability` | <1 s | OK, prints `docs/CAPABILITY_MATRIX.md` |
| 7 | `neurofly download-data --help` | **36.2 s** | **Ignored `--help` and downloaded the full 1.1 GB data set** (m3). All 3 files were SHA-256 verified. No licence or attribution was printed (m5). |
| 8 | `neurofly status` | 1.1 s | Still "Prepared graph not found": the download does not prepare the graph (B2) |
| 9 | *(found in `docs/EMBODIED_MVP.md` and `BRAINLAB.md`, not the README)* `python -m brainlab.connectome` | 14.5 s | OK, 166,700 neurons and 25,582,938 retained edges. Peak RSS 1.8 GB. |
| 10 | `python -m brainlab.prepare` | 2.2 s | OK. `graph.npz` sha256 `4b2f87cc…` **matches the pin**. |
| 11 | `neurofly status` | 1.4 s | **`Neuron map hash mismatch … neurons.feather: 416b4a36…`** (B1) |
| 12 | workaround: `pip install pandas==3.0.5` (the version in `requirements-lock.txt`), then rerun step 9 | 5.4 s + 13.5 s | `neurons.feather` → `2103b520…` = pin; `neurofly status`: **166,700 neurons, 25,582,938 synapses, Graph SHA-256 4b2f87cc… (VERIFIED)**, WP6 3081 edges VERIFIED |
| 13 | `docker build -t neurofly:latest .` | 25.1 s | **FAIL**: `error: package directory 'validation' does not exist` (B3) |
| 14 | `neurofly run --port 8791 --paradigm multisensory-sandbox` | HTTP ready in 3 s | OK. Backend **connectome-fixed** (the default), v3, CPU. |
| 15 | Firefox → `http://127.0.0.1:8791/` (the README's URL pattern) | live in 1.2 s | The dashboard renders, the LIVE DAEMON pill shows, the clock advances and there are no JS errors. Screenshot: `ff_cpu/00-readme-url.png`. Curl gets JSON at `/`; a browser gets the HTML. |
| 16 | `scripts/live_ui_signoff.py --web "http://127.0.0.1:8791/?daemon=http://127.0.0.1:8791" --daemon http://127.0.0.1:8791` (run with `CUDA_VISIBLE_DEVICES=`) | 1 min 51 s | **7/7 PASS**: initial load, all 14 assays, rapid selection, two-tab sync, speed, pause/resume, restore. 14 distinct canvases. |
| 17 | `pytest tests/` (GPU hidden: `CUDA_VISIBLE_DEVICES=`) | 3 min 12 s | **706 passed, 20 skipped, 0 failed**. Skip reasons are listed below. |
| 18 | `NEUROFLY_RUN_PHYSICS=1 pytest tests/test_embodied_*.py tests/test_fast_controller.py` | 24.7 s | **58 passed** |
| 19 | `pytest -v tests/test_wp6_plasticity.py` | (in step 17) | passed |
| 20 | `./scripts/check_private_infra.sh` | not run | It calls `git grep`, and this tester was limited to clone and `git log` inside the clean clone. An equivalent `grep -r` found no private IPs and no UNC paths, but 50 tracked files mention the owner's username or `<redacted-host>` (m6). |
| 21 | GPU: `numba.cuda` on a fresh install | n/a | `cuda.is_available()` False: `libNVVM cannot be found`. CuPy is absent, so `cuda_available()` is False and everything silently runs on the CPU (M1). |
| 22 | workaround A: `pip install "numba-cuda[cu12]"` | 9.1 s | numba-cuda 0.30.4 compiles nothing: `AttributeError: module 'numpy' has no attribute 'row_stack'` (numpy 2.5). Worse, it makes `cuda_available()` True, so the engine would choose a broken path. Uninstalled. |
| 23 | workaround B: `pip install cupy-cuda12x` (with nvrtc and runtime wheels already present from step 22) | 5.3 s | `cuda_available()` True. Default device = 1660 Ti. |
| 24 | GPU tests, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 pytest tests/test_cuda_engine.py tests/test_graded_v4.py tests/test_kinetics_v5.py` | 4.5 s | **38 passed, 3 failed**: `ImportError: libcusparse.so.12` in the CuPy v4/v5 tests (M1). There was no disk space to add `nvidia-cusparse-cu12` at that moment (see "Environment incidents"). |
| 25 | **Experiment** (from `docs/EMBODIED_MVP.md`): `python -m neurofly_body run --duration 2 --output runs/embodied-intact --mode intact --graph-dir outputs/brainlab/malecns_v1 --connectome-dir connectome_data/malecns_v1 --seed 1`, 1660 Ti | 20.5 s (sim 14.3 s) | `status: complete`. RTF 0.139, 179,831 graph spikes, motor output present, 7 GF takeoff commands logged as "UNSUPPORTED: legs-only body". Trajectory `b6d92fb7…`, brain_backend `cuda`. The GPU was shared with another worker at about 97 % utilisation. |
| 26 | `python -m neurofly_body replay-check runs/embodied-intact --output runs/embodied-intact-replay` | 19.7 s | **BIT_IDENTICAL** (trajectory and frame hashes) |
| 27 | Same experiment with `CUDA_VISIBLE_DEVICES=` | 55.1 s (sim 51.2 s) | `complete`. RTF 0.039, brain_backend `cpu`, 179,257 spikes, trajectory `c79cbde7…`. That this differs from the GPU run is expected and documented. |
| 28 | GPU daemon: `neurofly run --port 8791` with the 1660 Ti, plus Firefox | ready in 6 s | Live in Firefox, 0 errors, 408 MiB VRAM. Achieved 1.19x of the 5x requested, overloaded (the GPU was contended). The CPU daemon reached 2.0 to 3.1x. |
| 29 | Repro of B1 on the daemon: a `neurons.feather` identical except for `pandas_version: 3.0.6` in its metadata, then `neurofly run` | <3 s | Hash `416b4a36…`, **byte-for-byte the fresh-install hash**. The daemon prints `Cannot start backend 'connectome-fixed': GraphUnavailable: Neuron map hash mismatch` and **exits**. With `--backend modular` it starts and runs at 4.98x. |

### Test skips (step 17, GPU hidden)

| count | test | reason given |
|---|---|---|
| 7 | `test_cuda_engine.py` | needs an NVIDIA GPU or `NUMBA_ENABLE_CUDASIM=1` (expected with GPU hidden; all 7 pass on the 1660 Ti with CuPy) |
| 3 | `test_graded_v4.py`, `test_kinetics_v5.py` | no CuPy GPU on this host (these fail with libcusparse missing when CuPy is present, step 24) |
| 6 | `test_embodied_physics.py`, `test_fast_controller.py` | needs `NEUROFLY_RUN_PHYSICS=1` (all pass with it, step 18) |
| 1 | `test_browser_recovery.py` | needs `NEUROFLY_BROWSER_PYTHON` pointing at a Python with selenium (nothing documents this) |
| 1 | `test_provenance_registry.py::test_real_graph_matches_pins` | `NEUROFLY_GRAPH_DIR` not set. The default graph location existed and verified, but the test only runs when the env var is set. |
| 2 | `test_wall_progress_fixtures.py` | needs `NEUROFLY_RETAINED_BRAINS` (owner's checkpoints; not available to a stranger) |

## Findings, in order of severity

### BLOCKER

**B1. The graph identity pin depends on the pandas version, so a fresh README install fails verification and the daemon will not start.** *(code defect, triggered by the environment; time-dependent)*
`brainlab/graph_pins.json` pins `neuron_map_sha256` as a SHA-256 of the **bytes** of
`normalized/neurons.feather`. Feather files carry pandas schema metadata that includes
`"pandas_version"`. The README install uses unpinned dependencies and resolved pandas
3.0.6, which writes a file hashing to `416b4a36…`, while the pin `2103b520…` was made
with pandas 3.0.5 (`requirements-lock.txt`). Step 29 shows that changing only that
metadata string reproduces the mismatch exactly. Effects:
- `neurofly status` reports the graph as unavailable;
- `neurofly run` uses the default backend `connectome-fixed`, prints `GraphUnavailable` and exits, so the README's "open the dashboard" step cannot be reached;
- the next pandas or pyarrow release will break the pin again even for someone who installs from the lock file, unless the lock is enforced.

`graph.npz` itself matched its pin. Workaround: `pip install pandas==3.0.5` and rerun
`python -m brainlab.connectome`. Fix direction (for the owner of the code): hash the
neuron table's contents in a version-independent form, or strip or normalize the pandas
metadata before writing. Separately, make the README install from
`requirements-lock.txt`, or pin pandas.

**B2. The README's data step stops halfway: nothing tells the user to run `brainlab.connectome` and `brainlab.prepare`.** *(README defect)*
`neurofly download-data` downloads and verifies only the three source tables. The
graph (`outputs/brainlab/malecns_v1/graph.npz`) and the normalized neuron table come
from `python -m brainlab.connectome` then `python -m brainlab.prepare`. Those commands
appear only in `docs/EMBODIED_MVP.md` and `BRAINLAB.md`. After following the README,
`neurofly status` says "Prepared graph not found … Set NEUROFLY_GRAPH_DIR …". That
points the user at an env var rather than at the missing step. The default daemon needs
the graph, so the README route to the dashboard is blocked. This tester did not
separately run the daemon with the graph absent; that case goes through the same
`GraphUnavailable` path as B1, and `start_daemon.sh` documents that "a missing graph
stops the daemon". The steps take only 17 s together. Folding them into
`download-data` (or adding a `prepare-data` subcommand) would close this.

**B3. `docker build` fails.** *(code/packaging defect)*
The Dockerfile copies `neurofly/ brainlab/ experiments/ neurofly_body/ web/ *.py` but
not `validation/`. `pyproject.toml` lists `validation` and `validation.paradigms` as
packages, so `pip install -e` fails with `package directory 'validation' does not exist`
(log: `.cleanroom-20261005/log_docker_build.txt`). Even with that fixed, the image has
no data and no data step. Its default command runs the daemon with the default
`connectome-fixed` backend, which needs the graph (B1/B2). `docker run -p 8769:8769`
also has the port issue in m2. No workaround was attempted because disk was critically
low at the time; see "Environment incidents".

### MAJOR

**M1. On an NVIDIA machine the README install runs everything on the CPU without telling the user. Every route to the GPU needs outside knowledge.** *(README and packaging)*
The README says "The GPU is the default for v3 brains whenever CuPy or numba.cuda can
see a CUDA device". But no extra installs either one, and no page says how to. With the
default install:
- numba 0.68's `numba.cuda` sees the device but cannot compile (`libNVVM cannot be found`; no CUDA toolkit), so `cuda_available()` is False;
- `neurofly status` does not report the brain backend, and the daemon's "running on the CPU backend" message is a `log.info` that never prints, so the user cannot tell;
- the obvious fix, `pip install numba-cuda[cu12]`, installs 0.30.4, which is incompatible with numpy 2.5 (`np.row_stack`). It also flips `cuda_available()` to True, so the engine would pick a path that cannot compile;
- `pip install cupy-cuda12x` works for v3 here, but only because the nvrtc and runtime wheels from the numba-cuda attempt were still present. A bare `cupy-cuda12x` without a CUDA toolkit is expected to fail at the first kernel compile; this was not tested in isolation. The v4/v5 CuPy paths also need `libcusparse.so.12`, and 3 GPU tests fail without it. `cupy-cuda12x[ctk]` is presumably the right instruction, but it adds about 1 to 2 GB.

Suggested: a `gpu` extra (for example `cupy-cuda12x[ctk]`), a README line, and a
`neurofly status` row that says "brain backend: CUDA (GTX 1660 Ti) / CPU (reason)".

**M2. The default daemon backend contradicts the docs, and there is no way to run without data.** *(README, with a code/doc mismatch)*
`neurofly run` defaults to `--backend connectome-fixed` (`neurofly_daemon.py:2311`). But
`start_daemon.sh` defaults to `modular`, and the release plan says the live dashboard
fly is the modular controller by default. The README mentions neither `--backend` nor
the fact that the default needs the 1.1 GB data and the preparation steps. `--backend
modular` starts without any graph (checked, 4.98x). A stranger who just wants to see
the dashboard is never told that.

### MINOR

- **m1.** The README clone command uses SSH (`git@github.com:…`). A stranger without a GitHub SSH key gets "Permission denied (publickey)". Use the HTTPS URL. *(README)*
- **m2.** Port 8769 is the default everywhere (README, Docker `-p 8769:8769`, `start_daemon.sh`), and nothing says what to do if it is taken. Also, `web/app.js` probes, in order, the page origin, then `<page host>:8769` (or 8781 when the page is on 8780), then `localhost:8769`. A dashboard page served from a static server therefore attaches silently to whatever daemon holds 8769. On this host that is the owner's live daemon, so this tester always passed `?daemon=`. The daemon also binds `0.0.0.0` by default, which exposes its command API to the LAN without a token. *(code/README)*
- **m3.** `neurofly download-data --help` ignores the flag and starts the 1.1 GB download. `status` and `capability` also ignore all arguments. Only `neurofly --help` works. *(code)*
- **m4.** Disk needs are under-stated: the README says 1.1 GB, but the real footprint is 1.7 GB venv + 1.4 GB `connectome_data` + 0.2 GB graph, plus about 30 MB of v3 checkpoints per assay visited (453 MB in `outputs/registry-v3` after the 14-assay sign-off). Normalization peaks at about 1.8 GB RAM. *(README)*
- **m5.** G3 says "attribution shown", but `download-data` prints no licence (CC BY 4.0) or citation for the Janelia data. *(code)*
- **m6.** Privacy/G9: 50 tracked files contain the owner's username or `<redacted-host>`. 19 scripts in `scripts/v4_measurement/` and the scripts in `docs/receipts/switch-race-20261004/scripts/` hard-code `<redacted-path>/...` paths, so they cannot run for anyone else. `check_private_infra.sh` does not look for home paths, and it needs a git checkout because it uses `git grep`. No secrets were seen. *(repo hygiene)*
- **m7.** Browser tooling is undocumented: selenium is in no extra, `test_browser_recovery.py` silently skips without `NEUROFLY_BROWSER_PYTHON`, and `live_ui_signoff.py` defaults to the owner's ports 8780/8781 and a profile root `~/snap/firefox/common/neurofly-signoff`. *(README/test)*
- **m8.** Counts differ between documents: the README says ">520 tests" while the suite has 726. The release plan says "717 passed, 9 skipped". A stranger without a GPU, CuPy or retained brains sees 706 passed and 20 skipped. *(README)*

### NOTE

- **n1.** Device order: with no env vars, CUDA picks the 1660 Ti as device 0 (fastest first), so a user who follows nothing gets the right card. `CUDA_VISIBLE_DEVICES=1` alone selects the **Quadro P620** (checked: CuPy device name), because CUDA numbering differs from `nvidia-smi` numbering unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. The README gives no device-selection guidance, so a user only hits this by copying `nvidia-smi` indices.
- **n2.** On the default `connectome-fixed` backend the fly sits pressed against the arena wall: 4,041 of the first 4,472 steps were near the wall, it moved 0 mm, and the brain-activity panel showed 0.00 Hz. This matches the honest status, but a beginner will read it as "broken". One line in the README would prevent that.
- **n3.** The dashboard badge says `v1.0-release`, but the package version is 0.3.0 (G6).
- **n4.** Daemon speed: requested 5x; achieved 2.0 to 3.1x on the CPU and 1.19x on the GPU, `overloaded: true` in both. The GPU run shared the card with another worker at about 97 % utilisation, so it is not a clean measurement. The README performance table quotes 0.043x for the CPU "full daemon, connectome v3", which is hard to square with 2 to 3x here. It may be a different workload; worth checking for G10.
- **n5.** Fresh-install dependency drift: the unpinned install resolved pandas 3.0.6 and numpy 2.5.3 on release day. B1 is one consequence; numba-cuda being incompatible with numpy 2.5 is another.

## Environment incidents (not project defects)

- Disk: the root filesystem started at 8.3 GB free and fell to **7.9 MB free** at about 15:21, through activity outside this test. To avoid starving the owner's live services, this tester deleted its own clean-room pip cache (606 MB) and temporarily uninstalled the CuPy/NVIDIA wheels it had added. Free space later recovered to about 15 GB through other activity, and CuPy was reinstalled for the GPU daemon check. Some other session wrote `<redacted-path>/tmp/venv-pd305` and `venv-pd306` at the same time. The failed Docker build left about 0.6 GB of build cache, not pruned because pruning would also touch other users' caches.
- The GTX 1660 Ti was busy (about 97 %, about 1 GB in use) with another worker throughout. GPU use here was kept to under 2 minutes in total, at a peak of 408 MiB.

## Clean-room artefacts (left on disk)

`<worktree>/.cleanroom-20261005/`: `flybrain/` (the clone, with `.venv`, data, `runs/`,
`outputs/`), `log_*.txt` (one per step, each with the exact command, env and wall time),
`signoff_cpu_run1/` (sign-off JSON and 9 screenshots), `ff_cpu/` and `ff_gpu/`
(first-look Firefox JSON and screenshots), and `mm/` (the B1 reproduction). Every
process this tester started has been stopped, and ports 8791 and 8792 are free.
