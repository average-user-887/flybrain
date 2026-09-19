# NeuroFly — Open-Source Readiness Plan

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

Executed against `docs/CLAUDE_IMMEDIATE_PLAN.md` (Astra's accepted plan). Each work
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
