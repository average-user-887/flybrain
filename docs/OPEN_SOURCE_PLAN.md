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
