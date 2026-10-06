# Retirement index

Entry points and pages retired from the supported product on 2026-10-05, before the
v0.4 release. Nothing was deleted from history: every path below is readable and
runnable at its last historical commit, and all of them are unchanged at `713ba82`,
the `master` commit this retirement branched from. To look at or rerun an old
version, check out a commit, for example:

```bash
git show 713ba82:experiments/full_connectome_simulation.py
git worktree add ../neurofly-713ba82 713ba82
```

Shared implementations were **not** removed. `connectome_bridge.py`,
`connectome_client.py`, `brainlab/cosim_server.py` (`ConnectomeServer`, used in-process
by `neurofly embodied`), `data_logger.py` and `UnifiedConnectomeBrain` are still
imported by `arena.py`, the body CLI or the tests. Only the misleading entry points
were disabled. Citations, checksums and saved-brain formats are unchanged.

"Last historical commit" is the last commit that changed the path before this
retirement. Where the historical command, environment or receipt is not recorded
anywhere in the repository, it says **unknown**.

| Old path | Why unsupported | Last historical commit | Historical command and environment | Receipt dependency | Supported replacement |
|---|---|---|---|---|---|
| `neurofly full-sim` (`neurofly/cli.py`, `cmd_full_sim`) | Not a full connectome simulation. The loop fed the brain empty sensory input, because `Arena.get_sensory_inputs` does not exist (`hasattr` fell back to `{}`), and `arena.step` overwrote the brain's motor output with the modular controller. | `7d5a363` (CLI; `full-sim` wiring unchanged since `dc0465d`) | `neurofly full-sim [--steps-per-task 200] [--tasks all] [--output-dir outputs/full_simulation] [--no-plasticity] [--seed 42]`. Host, GPU and graph identity of past runs: unknown. | None committed. Untracked outputs under `outputs/full_simulation/` from before PR #7 are v3 equations on v1 weights (`docs/receipts/README.md`, item 9). | `neurofly run --backend connectome-fixed` (dashboard on port 8769; without `--backend`, auto-detection may choose the modular controller); `neurofly record --backend connectome-fixed --paradigm P --seconds S --out F` for a headless run; `neurofly validate run <spec> --out <dir>` for preregistered tests. |
| `experiments/full_connectome_simulation.py` (direct or `-m`; `main()`) | Same as above. The module stays importable because `tests/test_daemon_dynamics.py` uses `UnifiedConnectomeBrain`. | `5040a77` | `python experiments/full_connectome_simulation.py` with the same flags. Environment: unknown. | Same as above. | Same as above. |
| `experiments/run_paradigm_battery.py` (direct or `-m`; `main()`) | Defaults to `brain_type="connectome"`, `connectome_mode="surrogate"`: the surrogate `ConnectomeBridge`, labelled as connectome. Its metrics are not maintained. `run_battery` stays importable only for the `ParadigmDataLogger` format tests (`tests/test_paradigm_logger.py`). | `256f7fb` | Documented: `python run_paradigm_battery.py --paradigms all --trials 3 --steps 500`. The exact command that produced `experiment_data/ryzen_battery/` (3 trials x 200 steps, 2026-09-18) and its environment: unknown; no controller or dynamics identity was recorded. | `experiment_data/ryzen_battery/**`, kept unchanged and labelled surrogate smoke output, not validation (`experiment_data/ryzen_battery/README.md`). | `neurofly validate run <spec> --out <dir>`; `neurofly record` for single-paradigm runs. |
| `experiments/whole_brain_scientific_battery.py` (direct or `-m`; `run_complete_scientific_battery()`) | Drives the surrogate `ConnectomeBridge`, not the MaleCNS graph, and its report template hardcodes significance claims and conclusions ("p < 0.001", "necessary and sufficient") that no run measured. Before retirement it had no argument parser, so even `--help` ran it and wrote `./experiment_data/`. | `256f7fb` | `python experiments/whole_brain_scientific_battery.py` (no arguments; wrote `./experiment_data/whole_brain_battery/`). Environment: unknown. | None. | `neurofly validate run <spec> --out <dir>`. |
| `flybrain_scientific_instrument.html` (now a retired-page notice) | An unmaintained single-file fork of the dashboard ("v1.0 Release"). It lacks the identity-rejection and backend-switch logic of `web/app.js`, carried the LAN daemon-candidate entries flagged in `docs/RELEASE_AUDIT.md` item 9, and its bundler `scratch/sync_standalone.py` was never in the repository. Nothing served or loaded it. | `6198ce6` | Opened directly as a file in a browser. Bundling command and environment: unknown. | None. | The dashboard: `neurofly run --host 127.0.0.1 --port 8769`, then `http://localhost:8769` (`web/index.html`). |
| `web/research/app.js`, `web/research/style.css` (moved to `docs/archive/web_research_20261005/`) | Orphaned duplicate UI. Callers checked before the move with `git grep` over all HTML, JS and Python, the tests and the docs: no page, script, server route or test loads them; only the audit notes mention them. | `d5a8e7c` (last commit whose `web/research.html` loaded them; `2553a9e` turned that page into a redirect, asset blobs unchanged) | Loaded by `web/research.html` up to `d5a8e7c`. Environment: not applicable. | None. | The dashboard's Training & Data tab (`web/index.html`). `web/research.html` still redirects there and is kept. |
| `sync_ecosystem.py` (`main()`) | Its `SYNC_MANIFEST` omits current product and provenance modules (`neurofly/`, `neurofly_body/`, `neurofly_studio/`, `validation/`, `provenance.py`, `experiment_registry.py`, `experiment_brains.py`, `assay_controls.py`, `assay_response.py`, `online_metrics.py`, `pyproject.toml`), so a synced target cannot import the daemon. | `c474057` | `python sync_ecosystem.py [--remote-host H --remote-dir D] [--docker-target C:/path] [--archive-dir A] [--skip-tests]`, or the `NEUROFLY_*` environment variables. Targets used historically: unknown. | None. | `git` (clone, fetch or push a branch) on the target host. |
| `scripts/test_srv.py` (now an inert notice) | Loaded and stepped the full MaleCNS graph at module top level, so importing it or passing `--help` did the heavy work. | `6eba269` | `python scripts/test_srv.py` (no arguments). `docs/archive/CODEX_HANDOFF.md` records that nobody could confirm it had been run. Environment: unknown. | None. | `neurofly status` (graph identity and compute backend); the `ConnectomeServer` tests in `tests/`. |

## Still supported, kept on purpose

- `web/research.html`: a redirect to `index.html` for old bookmarks.
- `experiments/lesion_study.py`, `scripts/research_worker.py` and the other research
  tools listed in `docs/receipts/audit-20261005/E-code-inventory.md` were out of scope
  for this retirement.

## How each retirement behaves

Every retired entry point prints a notice naming the reason, the replacement and
this file, then exits with status 2. The notice is printed before NumPy, the arena,
the bridge or the graph are imported and before any output directory is created.
`--help` prints the same notice and exits 0, with no side effects.

## Preview controls retired on 2026-10-06

- Removed unused T5 lesion and wing angle controls; their displayed values had no implemented consumer. General biological references remain.
- Removed manual DN setpoint controls and thermal/odor flash buttons from the local preview. DN values were overwritten by autonomous updates; the flash actions changed MB reward/punishment without implementing the advertised sensory pulses. No trained state is reset by this cleanup.
- CPG cadence, GF escape and wind gust remain explicitly labeled preview actions. The former optogenetic turn is named Rotate preview fly and changes only preview heading. Actual temperature/odor field controls remain available.
- This source retirement does not establish a browser pass; root retains prechange evidence and owns postchange verification.
