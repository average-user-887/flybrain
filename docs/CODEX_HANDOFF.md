# Historical handoff to Codex — 2026-09-21

> This is the pre-MVP handoff, not the current installation guide. Private
> deployment identifiers have been redacted; original operational copies were
> archived outside the public checkout. Start with `README.md` and
> `docs/EMBODIED_MVP.md` for the implemented MVP.

Written by the Claude orchestrator after an Antigravity (Gemini) session ran out of quota mid-task.
Everything under "Verified" was checked directly on the hosts at handoff time. Everything under
"Claimed, not verified" comes from Antigravity's chat transcript and must be re-checked before it is relied on.

## 1. The goal (owner's words, condensed)

Full biological simulation of a fruit fly with real scientific value, open source, so people can run
experiments on their own PC. Client–server: the simulation runs on the **AMD Ryzen 3900X** host
(most powerful machine in the fleet), and **the connectome controls all joints and biological functions**.
Browsers are thin clients.

Owner-approved direction (2026-09-21 14:40): **server-side co-simulation on Ryzen** + **integrated
Three.js 3D articulated-fly viewport** in the dashboard. Details: `docs/FULL_CONNECTOME_COSIM_PLAN.md`.
Nothing from that plan has been implemented.

## 2. Hosts, checkouts, mirrors

| Where | Path | Role |
|---|---|---|
| Ryzen 3900X (`<operator>@<simulation-host>`, ZeroTier) | `~/Documents/ChatGPT/flybrain`, branch `master` | **Canonical repo. Work here.** Holds `connectome_data/malecns_v1/` (~1.4 GB, gitignored) and `.venv/`. |
| Ryzen worktree | `.claude/worktrees/neurofly-openready`, branch `worktree-neurofly-openready` (at `a544bfa`) | Locked. **Hosts the running observatory + dashboard (see §4). Do not remove it** while those run. |
| Ryzen, nested repo | `neurofly-site/` (branch `main`, own `.git`, no remote) | Next.js download site. Ignored by the parent repo. |
| hpserver storage | `/mnt/hpserver-storage/neurofly/` on Ryzen = SMB `//<archive-host>/Storage/neurofly` | Archive mirror. `snapshots/<sha>/` + `CURRENT_HANDOFF.md`. |
| Windows laptop | `<laptop-checkout>` | Mirror. Historical `.git` snapshot restore; Git is now installed and the mirror is verified. |
| Windows laptop | `...\Sluzbowy_hp\neurofly_v1_shipping` | Plain export (not a git repo) at `a544bfa`. Stale by docs-only commits; refresh only when shipping. |

There is **no git remote anywhere**. Sync = snapshot, not push.

### Sync procedure (end of every session)
On Ryzen, from the repo root with a clean tree:
```bash
SHA=$(git rev-parse --short=7 HEAD); S=/mnt/hpserver-storage/neurofly/snapshots/$SHA; mkdir -p $S
git bundle create $S/neurofly-openready-$SHA.bundle --all
git archive --format=tar.gz -o $S/flybrain-$SHA.tar.gz HEAD
tar czf $S/flybrain-dotgit.tar.gz .git
(cd $S && sha256sum *.bundle *.tar.gz > SHA256SUMS && sha256sum -c SHA256SUMS)
git bundle verify $S/neurofly-openready-$SHA.bundle
```
Then update `/mnt/hpserver-storage/neurofly/CURRENT_HANDOFF.md` (revision, verification line, progression).
Laptop: copy `flybrain-dotgit.tar.gz` down, **rename** the old `.git` to `.git.pre-<sha>-<date>`, extract,
and copy changed working files. Never delete old snapshots or old `.git` copies; retire by rename.

## 3. Verified state at handoff

- `master` HEAD is the commit that adds this file; tree clean (`git status` empty). Worktree and
  `neurofly-site` are clean.
- History since the last test run: `a544bfa` (last code) → `6eba269` (Antigravity leftovers: live
  sign-off receipts, `scripts/test_srv.py`, `.gitignore` for `/.claude/ /build/ /neurofly-site/`) →
  docs commits (plans + this handoff). **No code changed after `a544bfa`.**
- Snapshots on hpserver exist for `a544bfa`, `6eba269`, `f18eb8b` and the handoff commit, each with verified `SHA256SUMS` and bundle.
- Laptop mirror `.git` matches the latest snapshot. Retired copies: `.git.pre-6eba269-20260921`, `.git.pre-f18eb8b-20260921`, and one for the final sync.
- Tests were **not** re-run by the orchestrator. Last reported full run: 506 passed, 4 skipped, 101 s, at `a544bfa` (Antigravity report; see §5).

## 4. Running services on Ryzen (leave them unless the owner says otherwise)

| PID | Port | What | Runs from |
|---|---|---|---|
| 4110817 | 0.0.0.0:8769 | `neurofly_daemon.py --speed 15 --paradigm multisensory-sandbox` (continuous background learning) | main checkout `.venv` |
| 2085848 | 127.0.0.1:8781 | observatory daemon, `--paradigm t-maze --speed 3 --continuous`, output `outputs/observatory-live/` | **worktree** `.venv` |
| 289423 | 127.0.0.1:8780 | `python -m http.server` serving `web/` (the dashboard) | **worktree** |

Other things on the fleet that are not part of flybrain: hpserver uvicorn apps on :3003 and :8000.
An Antigravity desktop instance (`/opt/google-antigravity`, pid 9793) and a stale `claude -r` (pid 8310,
cwd `$HOME`, started 2026-09-17) are still running on Ryzen. Neither is editing the repo. The owner decides whether to close them.

## 5. What the system actually is today (be precise; past overclaims were withdrawn)

**Live dashboard fly = hand-built modular controller, NOT the connectome.** The identity bar says:
`Controller: modular — "Hand-built modular controller with explicit assay reflexes. Not the connectome."
Assists: ON (wall_avoidance_reflex, contact_turn) · Motor: modular`.

- Modular controller: 120-KC mushroom body (PAM/PPL1 dopamine), 16-wedge E-PG compass,
  2D Kuramoto-Hopf tripod CPG, hard-coded assay reflexes (`assay_response.py`). 14 assays.
- Body: 2D arena; the fly is a circle with 6 animated legs. No mass, torque, contact or gravity.
- Connectome: MaleCNS v1.0, 166.7k neurons / ~25.6M directed edges, imported under `connectome_data/malecns_v1/`.
- Engine: `brainlab/` Numba LIF, dynamics v1 (current), v2 (conductance), v3 (PSP-preserving
  conductance calibration, `docs/LIF_DYNAMICS_SPEC.md`). Steps the whole graph in isolation
  (reported ~0.29 s wall per 100 ms sim). `brainlab/cosim_server.py` → `ConnectomeServer.step(sensory_dict, duration_ms)`
  returns DN readouts (`dna02_rate_l/r`, `dna02_diff`, `dnp09_rate`, `mdn_rate`, `total_spikes`). Smoke test:
  `python scripts/test_srv.py`, untracked until today and **not yet run by anyone we can confirm**.
- Scientific record: WP5 optomotor preregistered causal test, verdict **NULL** (`4552423`); causal-loop
  claim withdrawn (`e882744`). WP6 plasticity spec written (`docs/WP6_PLASTICITY_SPEC.md`), not implemented.
  Open owner decisions are recorded in commit `30b2597` / `docs/`.

### Claimed by Antigravity, not verified by the orchestrator
- 506 passed / 4 skipped at `a544bfa`; 8/8 live UI sign-off (`docs/receipts/live-signoff-fresh/`, screenshots are real files).
- "14/14 assays learn": `scripts/learning_battery.py` paired/reversed/frozen deltas of ±1.5–2.1. Note that this
  exercises the **modular 120-KC mushroom body**, not the connectome, and the exact ± symmetry means it is
  a mechanism check, not a behavioural learning result. Do not present it as the fly learning.
- Daemon uptime and step counts (plausible given process ages above).
- Research references in the plan: `philshiu/Drosophila_brain_model` (Shiu et al., Nature 2024) and
  `NeLy-EPFL/flygym` (NeuroMechFly v2) are established. **`erojasoficial-byte/fly-brain` and
  `ZeroXClem/closed-loop-fly` and their numbers are unverified. Check they exist before citing or copying anything.**
- Numbers inside the plan (6,098 sensory nodes, 1,314 DNs, joint ranges, 2.5 µN Cruse threshold, <5 Hz
  resting rate) are unsourced targets, not measurements.

## 6. Plan files

- `docs/FULL_CONNECTOME_COSIM_PLAN.md`: **the plan the owner approved on 2026-09-21** (Antigravity session
  `139759bc`). Four components: sensory ingress → full graph; DN readout → locomotion; 18-DOF articulated
  walking (FlyGym/MuJoCo or upgraded CPG); dashboard 3D viewport + premotor deck + identity bar
  `Controller: connectome`. Its open question on plasticity scope (hybrid MB plasticity vs static graph) is **still undecided by the owner**.
- Cleanup correction (2026-09-21): the former `docs/ANTIGRAVITY_PLAN_2026-09-17.md`
  was a misplaced **Uroboros** plan, not a NeuroFly plan. It was archived outside
  the checkout before removal; its original contents remain in Git history at
  `4db4b88`. See `docs/REPOSITORY_LAYOUT.md` for the active project layout.

## 7. Recommended first slice (smallest honest end-to-end)

1. On Ryzen: `source .venv/bin/activate && pytest -q` → record result; `python scripts/test_srv.py` → record DN
   rates and wall-time. If the graph goes silent or runs away, fix that before anything else (tonic drive / calibration).
2. **Optomotor on the connectome:** a new controller backend (the provenance system from WP4 already supports
   named backends, see `f1395b1`, `5c9519e`) where HS/VS-proxy input from the rotating drum feeds the graph via
   `ConnectomeServer`, and `dna02_diff` / `dnp09_rate` drive yaw/thrust of the existing 2D body with **assists OFF**.
   The identity bar must say `Controller: connectome`. Keep `modular` as the labelled baseline.
3. Receipt: preregister the metric (as WP5 did), run both backends, commit the numbers whatever they show.
4. Only then add the body: FlyGym/MuJoCo in a separate process on Ryzen, stepped in lock-step with the LIF
   kernel, streaming joint angles + contacts over the existing SSE path to a Three.js viewport.
5. Streaming/timing budget: the graph runs ~0.3× real time on CPU at best. Design the UI for slower-than-real-time
   with a visible sim-clock, and don't fake 60 FPS motion.

## 8. Rules that bind you

- `AGENTS.md` in the repo root: owner's standing rules, including the live-UI sign-off procedure. Read it first.
- Every capability claim needs a receipt in `docs/receipts/` or a committed test. Label assists and
  non-connectome paths in the UI. Report failures and NULL results as plainly as successes.
- Never delete evidence (receipts, snapshots, old `.git` copies); retire by dated rename.
- End each session: clean `master`, snapshot + `CURRENT_HANDOFF.md` on hpserver, laptop mirror refreshed.
