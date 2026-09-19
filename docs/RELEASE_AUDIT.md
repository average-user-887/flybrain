# Open-source release audit

Date: 2026-09-19. Scope: the whole worktree (`.venv`, `.git` and the
git-ignored `upstream/`, `connectome_data/`, `runs/`, `outputs/` excluded
from the findings unless stated). Method: recursive grep for private
addresses (`192.168.*`, `10.*`, `172.16-31.*`, ZeroTier), hostnames, SSH key
names, usernames, Windows/UNC paths, container ids, tokens/passwords; size
scan for binaries; review of `licenses/`, `UPSTREAM.json`, `data-provenance/`,
`brainlab/datasets.json`, `data/manifest.json`.

Status legend: **fixed** = done in this pass (files owned by the release
worker); **open** = needs the orchestrator or the file's owner; **decision** =
owner call.

## 1. Private infrastructure

| # | File:line | Finding | Suggested change | Status |
| --- | --- | --- | --- | --- |
| 1 | `README.md:105-111` | `docker exec ... 65dcff428c87`, `ssh avg-usr@192.168.194.227`, `/home/avg-usr/...` | rewritten | fixed |
| 2 | `sync_ecosystem.py:65,89-90,222-226` | SSH key name `repo_audit_linux`, UNC `//192.168.1.23/Storage/neurofly`, `Z:/neurofly`, default host `avg-usr@192.168.194.227`, container id | rewritten: all targets opt-in via `NEUROFLY_REMOTE_HOST/REMOTE_DIR/DOCKER_TARGET/ARCHIVE_DIR/SSH_KEY`; no defaults | fixed |
| 3 | `neurofly_daemon.py:6` | "AMD Ryzen workstation" in docstring | neutral wording | fixed |
| 4 | `start_daemon_ryzen.sh`, `stop_daemon_ryzen.sh` | file names carry the host name; content is generic | rename to `start_daemon.sh` / `stop_daemon.sh` via `git mv` (orchestrator); content updated to document `NEUROFLY_PUBLIC`, `NEUROFLY_ADMIN_TOKEN`, `NEUROFLY_DATA_DIR` and pass extra flags through | open (rename) |
| 5 | `CLAUDE_HANDOFF.md:14-16, 33-37, 45-59, 188-195` | container id `65dcff428c87`, `192.168.194.227`, `avg-usr@`, `~/.ssh/repo_audit_linux`, daemon PID, `Z:\neurofly`, `//192.168.1.23/Storage/...`, Windows dev root, `scratch/sync_standalone.py` (not in repo) | Either move to `docs/INTERNAL_HANDOFF.md` with section 2 and the command block removed, or keep out of the public tree. Sections 3-7 are fine to publish once the scale claims match the README | open |
| 6 | `arena.py:84, 465` (line numbers drift while the physics agent edits; search `connectome_host`) | `connectome_host: str = '192.168.194.227'` | `os.environ.get("NEUROFLY_CONNECTOME_HOST", "127.0.0.1")` | open (owner: physics agent) |
| 7 | `connectome_bridge.py:28, 63` | "Remote RPC client connecting to Ryzen workstation"; `rpc_host: str = "192.168.194.227"` | same env default; neutral wording | open |
| 8 | `connectome_client.py:4-5, 24, 42, 75` | "remote Ryzen ... spiking engine"; `host: str = "192.168.194.227"` | same env default; neutral wording | open |
| 9 | `web/app.js:3322` (search `192.168` and `Ryzen cluster daemon`) and `flybrain_scientific_instrument.html:4615, 4668, 5341` | `list.push(\`http://192.168.194.227:${port}\`)` in the daemon candidate URL list; "Ryzen cluster daemon" tooltips | drop the LAN entry (keep `window.location.hostname`, `localhost`, `127.0.0.1`, plus an optional `?daemon=` query/`data-daemon-url` attribute); neutral wording | open (owner: web agent) |
| 10 | `BRAINLAB.md:10, 31, 37-40`; `RUNS.md:6`; `FULL_BRAIN_TEST.md:26`; `LEARNING_DEMO.md:8, 50` | absolute `/home/avg-usr/Documents/ChatGPT/flybrain` paths; `/home/avg-usr/.cache/codex-runtimes/.../node` | replace with relative commands (`.venv/bin/python -m brainlab...`, `node scripts/check_learning_ui.cjs`) | open |
| 11 | `brainlab/cosim_server.py:5` | "on Ryzen / remote compute nodes" | neutral wording | open (cosmetic) |
| 12 | `experiment_data/ryzen_battery/**` (108 files, 1.9 MB, tracked) | directory named after the host; `PARADIGM_REPORT.md` embeds that path; the data is a 3×200-step smoke battery with all-zero performance indices | `git rm -r --cached` and ignore (regenerable with one command), or `git mv` to `examples/battery_smoke/` if an example is wanted. Nothing to delete on disk | decision |
| 13 | `docs/OPEN_SOURCE_PLAN.md:3, 6` | "Ryzen worktree", "live Ryzen tree" | harmless internal wording; leave or neutralise | open (cosmetic) |
| 14 | `.gitignore` | `upstream/` (unanchored) also ignored `licenses/upstream/`, so the CC BY 4.0 text and MIT notices referenced by the DOOMFLY notices were never tracked | anchored to `/upstream/`; added `outputs/`, `experiment_data/battery/`, `.env*`, `*.pid`, `*.log`. Orchestrator: `git add licenses/upstream` | fixed (needs `git add`) |

No ZeroTier addresses, `10.x`/`172.16-31.x` addresses, e-mail addresses, or
Windows user-profile paths were found outside the items above. The
`.venv` symlink in the worktree points at a machine path but is ignored.

## 2. Secrets

None found. No tokens, passwords, API keys, `.env` files or private keys in
tracked files. The new public mode reads its token from `NEUROFLY_ADMIN_TOKEN`
only and never logs or persists it (`learning_recorder` deliberately omits
`argv` from `session.json`).

## 3. Large and generated files

| Path | Size | Tracked | Note |
| --- | --- | --- | --- |
| `demo.html` | 1.4 MB | yes | generated by `build_demo.py` from `viewer.html` + `data/*.swc`; regenerable. Decision: keep as the offline skeleton demo or drop from git |
| `flybrain_scientific_instrument.html` | 317 KB | yes | single-file bundle produced by `scratch/sync_standalone.py`, which is **not** in the repository; either add the bundler or document that this file is hand-maintained |
| `app.js`, `index.html` (repository root) | 99 KB, 32 KB | yes | stale copies of `web/app.js` / `web/index.html` (differ by ~3,700 / ~2,400 diff lines). Recommend `git rm` (orchestrator) and point everything at `web/` |
| `test_whole_brain.py`, `whole_brain_scientific_battery.py` (root) | 29 KB, 23 KB | yes | byte-identical to `tests/test_whole_brain.py` and `experiments/whole_brain_scientific_battery.py`; recommend `git rm` of the root copies |
| `learning-data.json`, `learning.html` | 45 KB, 10 KB | yes | recorded cue-learning demo (LEARNING_DEMO.md); fine to keep |
| `data/*.swc` | 1.2 MB | yes | two MaleCNS skeletons, CC BY 4.0, hashed in `data/manifest.json`; fine |
| `experiment_data/ryzen_battery/` | 1.9 MB | yes | see item 12 |
| `connectome_data/` | 1.4 GB | ignored | downloaded source tables + normalized derivatives; correct |
| `runs/` | 195 MB | ignored | brainlab run artefacts; correct (`RUNS.md` documents them) |
| `upstream/` | 26 MB | ignored | DOOMFLY + neuprint-python reference checkouts; correct |

`tests/test_daemon.py:20` writes into `<repo>/outputs/test_daemon` instead of
`tmp_path`; harmless (ignored) but worth switching to `tmp_path`.

## 4. Third-party code and data attribution

| Item | License | Evidence in tree | Bundled? | Action |
| --- | --- | --- | --- | --- |
| DOOMFLY (nftechie) | MIT | `licenses/DOOMFLY-LICENSE`, `UPSTREAM.json` (commit pinned) | code derived (`brainlab/`, plasticity rule in `circuit.py`) | attributed in `NOTICE` and `LICENSE` |
| Shiu et al. 2024 brain model | MIT | `licenses/upstream/Shiu-model-MIT.txt` | constants only, no code | attributed in `NOTICE` |
| neuprint-python (HHMI) | BSD-3-Clause | `upstream/neuprint-python/LICENSE` | no (ignored reference checkout) | noted in `NOTICE` |
| MaleCNS v1.0 | CC BY 4.0 (per release page, recorded in `licenses/DOOMFLY-THIRD_PARTY.md`; full text `licenses/upstream/CC-BY-4.0.txt`) | `brainlab/datasets.json`, `data-provenance/malecns_v1/source.lock.json` (URLs + SHA-256), `data/manifest.json` | two skeletons + provenance report; source tables downloaded on demand | attributed in `NOTICE`; compatible with MIT code |
| Huang, Luo et al. 2024 (Nature) | CC BY 4.0 | cited in `circuit.py` | rule adapted, no article material | attributed in `NOTICE` |
| FlyWire | CC BY 4.0 (public release), **not verified here** | name only (`README`, `connectome_bridge.py`, `web/app.js`, `CLAUDE_HANDOFF.md`) | **no data present** | README now states no FlyWire data is used. Do not add FlyWire data without recording its license and citation |
| MANC v1.0 | CC BY 4.0 (Janelia), **not verified here** | name only (`VNC_PREMOTOR_CONNECTOME_MAPPING.md`, battery report strings) | **no data present** | same as FlyWire |
| ViZDoom / Freedoom / doom-ui npm packages | MIT / BSD-3 / various | `licenses/DOOMFLY-THIRD_PARTY_NOTICES.md`, `licenses/upstream/npm/**` | no (DOOMFLY components not shipped) | notices preserved verbatim; `NOTICE` explains they are inherited |

`experiments/whole_brain_scientific_battery.py:386, 418, 477` and the root
copy write "MaleCNS v1.0 / FlyWire / MANC v1.0" and "166,700 neurons, 25.58M
synapses & MANC v1.0 (23,000 VNC neurons)" into generated reports. These
strings should be changed to describe the modular model actually used by the
battery (owner: whoever owns `experiments/`).

## 5. Public-stream safety (implemented)

- `stream_gateway.py` + `neurofly_daemon.py --public`: `POST /api/command`
  returns `403` unless `Authorization: Bearer <NEUROFLY_ADMIN_TOKEN>` matches;
  SSE clients capped (`503` + `Retry-After`), stream throttled, command body
  capped at 64 KiB, `/api/status` reports the policy. Private mode is unchanged.
- `learning_recorder.py`: append-only JSONL with fsync and non-destructive
  rotation; documented in `docs/DATA_SCHEMA.md`.
- Pre-existing bug fixed in `run_daemon()`: the SIGTERM handler called
  `server.shutdown()` from the thread running `serve_forever()`, which
  deadlocks; the stop script masked it with `kill -9` after 7.5 s. The handler
  now lets `SystemExit` unwind `serve_forever()`.
- Still open for a public link: reverse proxy with TLS and per-IP limits;
  `web/app.js` must handle `403` from `/api/command` and stop embedding the
  LAN address (item 9).

## 6. Checklist before the first public push

- [x] `LICENSE` (MIT) and `NOTICE` added — owner to confirm the copyright line
      and MIT vs Apache-2.0 (see decisions)
- [x] README rewritten with "What this simulates today"
- [x] `.gitignore` anchored; `licenses/upstream/` now trackable (`git add`)
- [x] `sync_ecosystem.py`, README, daemon docstring scrubbed
- [x] CI workflow with a private-infrastructure grep guard
- [x] `CONTRIBUTING.md`
- [ ] Items 5-12 above in files owned by others
- [ ] Remove or relocate root-level duplicates (`app.js`, `index.html`,
      `test_whole_brain.py`, `whole_brain_scientific_battery.py`)
- [ ] Decide on `demo.html` / `flybrain_scientific_instrument.html` bundling
- [ ] `git mv` the daemon scripts to neutral names
- [ ] Owner confirmation before publishing any stream URL

## 7. Decisions for the owner

1. **Code license.** Recommendation: **MIT**, matching upstream DOOMFLY so the
   derived `brainlab/` code and the plasticity rule need no dual notice.
   Apache-2.0 would also be compatible (MIT code can be included in an
   Apache-2.0 project with its notice kept) and adds an explicit patent grant,
   at the cost of a longer NOTICE discipline. `LICENSE` has been added as MIT
   with "Project NeuroFly contributors"; confirm the copyright holder name or
   switch to Apache-2.0 before the first public commit.
2. **Data license compatibility.** MaleCNS v1.0 is CC BY 4.0 as recorded in the
   inherited DOOMFLY notice. CC BY 4.0 data alongside MIT code is fine provided
   attribution is kept (`NOTICE`). If the owner wants the two bundled
   skeletons and the provenance report to stay in the repository, no further
   action; if not, they can be moved behind `brainlab.download`. FlyWire and
   MANC are not used; if they are added later their license terms (both
   believed CC BY 4.0, unverified in this tree) must be recorded first.
3. **`experiment_data/ryzen_battery/`** — untrack, or rename to an
   `examples/` directory (item 12).
4. **`CLAUDE_HANDOFF.md`** — keep out of the public tree, or publish a
   redacted `docs/INTERNAL_HANDOFF.md` (item 5).
