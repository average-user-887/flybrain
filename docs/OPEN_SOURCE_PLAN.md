# NeuroFly — Open-Source Readiness Plan

> **Superseded as a plan** by [`ROADMAP.md`](ROADMAP.md) v2.0 (2026-09-24); the phases below are the
> old plan's. The branch named here no longer exists; `master` on GitHub is authoritative.
> The "about 0.1 simulated seconds per wall second" figure below is a v1-engine number; true-v3
> speeds are in `docs/receipts/ryzen/`. See [`SUPERSEDED_PLANS.md`](SUPERSEDED_PLANS.md).

Branch: `worktree-neurofly-openready` (Ryzen worktree). Started 2026-09-19.

## Phase 0 — Baseline (done)
- Snapshot of the live Ryzen tree committed (`7fe4eb6`); the tree had ~4.6k lines of
  uncommitted work synced from the Windows development root.
- Baseline: 180/180 tests pass (`PYTHONPATH=. .venv/bin/python -m pytest -q tests/`).
- Dashboard served from `web/` on port 8770; daemon on port 8769.

## Phase 1 — Containment and physics (script-controlled)
Observed live: the daemon fly sits at exactly `x = 158.5`, speed ≈ 0, sliding along the
wall. It is clamped against the boundary by the containment failsafe instead of steering away.
- Headless containment harness: every paradigm, many seeds and headings, asserting the
  fly stays inside bounds, never tunnels, and does not stay wall-pinned.
- Fix the root cause in the Python physics and mirror the fix in `web/app.js`.
- The daemon must advance trials. `trials_completed` stays 0 after ~2M steps.

## Phase 2 — Dashboard
- Load the dashboard headless, capture console errors and screenshots for each paradigm, and fix them.
- The JS containment must match the Python containment.

## Phase 3 — Open-source hygiene
- Remove private infrastructure: hard-coded LAN or ZeroTier IPs, SSH key names, and hostnames.
- Add a LICENSE, align the README claims with what the code actually simulates, add
  data-provenance notes for the connectome datasets, and add CI.

## Phase 4 — Data collection and public streaming
- The daemon persists per-trial learning data as append-only JSONL with checkpoints.
- A read-only public stream: GET/SSE only, with `POST /api/command` off by default or
  behind a token, CORS locked down, and a viewer page that works over a tunnel.
- Publishing the stream link is an outward-facing step and needs owner confirmation.

## Phase 5 — Research development
- Whole-connectome sparse kernel, multi-fly social behaviour, plasticity checkpoints (HDF5/Zarr).

## 2026-09-19 continuation

- Dashboard collision fixes committed as `8cc4412`; all 14 paradigms pass a
  12-seed × 3,000-step stress harness (504,000 total steps).
- Separate persistent experiment brains, controlled teaching/reversal, frozen
  controls, read-only probes, and a live research observatory are implemented.
- See `LEARNING_OBSERVATORY.md` for commands, scope, evidence and remaining
  scientific limitations. Public release and full-connectome closed-loop work
  remain separate milestones.

## 2026-09-19 evening — remediation work packages 1-5

Executed against `docs/archive/CLAUDE_IMMEDIATE_PLAN.md` (Astra's accepted plan). Each work
package ran in its own agent with disjoint file ownership; no running service was
touched, and no checkpoint or evidence bundle was deleted.

| WP | Commit | Outcome |
|----|--------|---------|
| 1+2 | `85989e7` | The frozen view at 100x was caused by the simulation loop re-taking the lock every step (HTTP waits p95 1.9 s) against a 1.2 s client reconnect probe. Deadline scheduler, lock-free snapshot delivery, command acks, visible error and data-age states. Command p95 19.6 ms; 1x/20x/100x bit-identical. Host ceiling is about 30x, stated honestly in the UI. |
| 3 | `ca4a086` | The wall-avoidance reflex and contact turn are engineered behavioural assists; now named, logged options, on by default. They were hiding three real solver defects, now fixed. Containment passes 84/84 with assists on and off. |
| 4 | `f1395b1` | The real graph is pinned and verified (166,700 neurons, 25,582,938 edges). Named backends, no silent surrogate on RPC fault, per-assay instances with atomic checkpoints and isolation tests. |
| integration | `5c9519e` | Run identity, motor provenance and controller faults flow through arena, daemon and UI; stale packets rejected; assists default off for graph backends. |
| 5 | `ca7b71b` | DNa02 left/right were swapped, mirroring every graph steering sign; corrected and verified against annotations. Preregistered causal test: intact turning index +0.193 [+0.133, +0.243], DNa02-silenced and sham identically zero, shuffled graph -0.024. |

### Honest status

- One causal graph-mediated loop (optomotor) is demonstrated, with a one-sided
  response, a null at contrast 0.5 and a runaway LIF proxy. It is not validated
  biological optomotor behaviour, and it costs about 0.1 simulated seconds per
  wall second, so it cannot run live at 1x.
- No plasticity rule is declared, so `connectome-plastic` refuses to start (WP6).
- The other 13 assays have no verified sensory or motor mapping for graph backends.
- Tests: 458 passed, 4 skipped. Firefox 156 headless passes; Chromium is untested.
- **The live-UI sign-off required by `AGENTS.md` is still pending.** It needs the
  observatory brain service restarted so it loads this code, then a check of all 14
  assays in the running browser.

## 2026-09-20 — dynamics v2 and the WP6 specification

| Item | Commit | Outcome |
|------|--------|---------|
| WP6 spec | `bd8f59c` | Recommends ER4d+ER2 -> EPG heading-map learning (3,081 edges, 0.012 % of the graph, all inhibitory, depression-only) over optomotor gain or olfactory conditioning, because it is the only candidate whose plastic site, modulator and motor readout are all identifiable here. Feasible in 6-7 wall hours; the 47-hour design is excluded. Specification only. |
| Release redactions | `f709cb9` | Private infrastructure removed from the handoff, the guides and the standalone bundle. |
| Dynamics v2 | `001b692` | Under v1, DNa02_R sat at -184 mV, below any chloride reversal, and never fired: **WP5's one-sided optomotor result was an engine property, not anatomy.** v2 bounds the membrane and both sides now respond, but the runaway is 4x worse. |

### The blocking scientific problem

The graph's inhibition-to-excitation ratio is 0.619 where subthreshold operation needs
more than 1.80, so the high-conductance fixed point is -26.75 mV against a -45 mV
threshold and sustained input saturates the network. The root cause is that Shiu et
al.'s 0.275 mV per synapse was calibrated inside a current-based model. A recalibration
that would close the gap was identified and deliberately **not** adopted, because
choosing it after seeing the result would be tuning. It belongs in a declared v3,
argued from physiology before measurement.

**Consequently WP6 must not start on either dynamics version.** A learning rule on a
network with a suprathreshold fixed point measures the engine, not the connectome.

### Open decisions for the owner

1. Adopt a declared v3 synaptic-gain recalibration (argued first, measured second), or
   restrict all graph claims to what v1 and v2 actually support?
2. Aminergic and `unclear` neurons are mapped to positive excitatory weights, so
   modulators already act as fast excitation; a three-factor learning rule would count
   them twice. Resolve before WP6.
3. WP6 target: ER->EPG heading learning, or the olfactory neural-level replication?
4. Release decisions left open: renaming the `*_ryzen.sh` scripts, the stale root-level
   duplicates of `app.js`/`index.html`, and the tracked `experiment_data/ryzen_battery/`.
   *(2026-09-24: the scripts are now `start_daemon.sh`/`stop_daemon.sh` and the root
   duplicates are in `docs/archive/root_duplicates_20260924/`; `ryzen_battery` is still open.)*
5. **The live-UI sign-off still requires restarting the observatory service**, which
   this session is not permitted to do.

### Correction (2026-09-20, completed v2 set)

The entry above for 19 September states that one causal graph-mediated loop was
demonstrated. **That claim is withdrawn.** The completed 24-run v2 set (commit
`4552423`) shows:

- Under v1 the positive optomotor result was an engine artefact: DNa02_R was held
  at -184 mV, below any chloride reversal, and never fired.
- Under v2, with the membrane bounded, the preregistered verdict is **NULL**.
  Silencing DNa02 does not abolish the residual turning (paired intact - silenced
  -0.0115 [-0.0247, +0.0059]); it slightly increases it. The residual is not a
  decoded steering command, but noise on a ~273 Hz saturated background.

NeuroFly currently has **no validated graph-mediated sensorimotor loop**. The
software provenance, isolation, physics and transport work stands; the scientific
claim does not. The next step is a declared v3 synaptic-gain recalibration argued
from physiology before measurement, not another run of the same protocol.
