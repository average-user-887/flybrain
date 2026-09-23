# Antigravity plan of 2026-09-17 (SUPERSEDED, reference only)

Older Antigravity session 0ebdf394, written 2026-09-17. It is not the plan the owner approved on
2026-09-21; that one is docs/FULL_CONNECTOME_COSIM_PLAN.md. Kept because it was committed in 6eba269.

---

# Phase 10 Completion & 2-Host Multi-Architecture Benchmarking Tool Delivery Plan

This plan establishes the concrete steps to close out Phase 10 (P10 v2), resolve all remaining defects and acceptance gaps in the Uroboros repository, and deliver the 2-host multi-architecture LLM benchmarking tool across AMD Ryzen (CUDA / Docker / dedicated memory) and Valve Steam Deck (Vulkan / rootless Podman / shared memory).

---

## Sources of Truth & Handoff Locations

### Repository & Local Checkout
- **Repository Root:** `/home/avg-usr/Documents/Codex/uroboros`
- **Active Handoff:** [CURRENT_HANDOFF.md](file:///home/avg-usr/Documents/Codex/uroboros/docs/state/CURRENT_HANDOFF.md)
- **Compact Local Handoff (worktree-p10v2-ryzen branch):** [P10_FINAL_HANDOFF.md](file:///home/avg-usr/Documents/Codex/uroboros/docs/state/P10_FINAL_HANDOFF.md)
- **Standing Rules & Authority:** [AGENTS.md](file:///home/avg-usr/Documents/Codex/uroboros/AGENTS.md), [CLAUDE.md](file:///home/avg-usr/Documents/Codex/uroboros/CLAUDE.md), [CONSOLIDATION_PLAN.md](file:///home/avg-usr/Documents/Codex/uroboros/CONSOLIDATION_PLAN.md), [CONSOLIDATION_LEDGER.md](file:///home/avg-usr/Documents/Codex/uroboros/docs/state/CONSOLIDATION_LEDGER.md)

### Storage Mount (`/mnt/hpserver-storage/uroboros-transfer/`)
- [P10-HARNESS-HANDOFF-RYZEN-20260916.md](file:///mnt/hpserver-storage/uroboros-transfer/P10-HARNESS-HANDOFF-RYZEN-20260916.md): Ryzen harness transition handoff at window close.
- [P10-HARNESS-HANDOFF-DECK-20260916.md](file:///mnt/hpserver-storage/uroboros-transfer/P10-HARNESS-HANDOFF-DECK-20260916.md): Steam Deck harness transition handoff at window close.
- [DECK-FINAL-HANDOFF.md](file:///mnt/hpserver-storage/uroboros-transfer/p10v2-deck-FINAL-HANDOFF/DECK-FINAL-HANDOFF.md): Deck final handoff with 5-model coverage matrix and Vulkan memory limits.
- [p10-dashboard-correctness-20260916.md](file:///mnt/hpserver-storage/uroboros-transfer/p10-dashboard-correctness-20260916.md): Live dashboard correctness repair checklist items 1–5.

---

## Current Status: Why Phase 10 is Not Yet Complete

As recorded in the cross-host handoffs at the close of the 8-hour window (2026-09-16 17:15:30Z):

### Delivered & Verified:
1. **2-Host Multi-Architecture Benchmark Coverage:**
   - **Ryzen (CUDA/Docker):** 36/36 document diagnostic cases + 30/30 benchmark cases (18 BFCL, 8 IFEval, 4 HumanEval).
   - **Steam Deck (Vulkan/RADV/rootless Podman):** 36 document cases + 30 benchmark cases across 5 models (`Qwen3.5-4B`, `Claude-Reasoning`, `LFM2.5-1.2B`, `SmolLM3`, `gemma-3-4b-it`).
2. **Dashboard Host-Side Projections (D-2):**
   - Host projects `uroboros.run-projection.v2` into `<state_root>/projections/` to avoid adding heavy Inspect dependencies to the read-only dashboard image.
   - Historical assessment visibility restored (5 assessments render on Ryzen, 11 on Deck).
   - Document evidence displays on its own discrete axes (`coverage`, `task_success`, `critical_violation`), validated 5 of 5 exact against disk by both hosts.
   - Per-model cited run counts (`accepted_run_count`) and `completed_sample_count` repaired at the server merge point.
3. **Core Parser & Role Fixes:**
   - Markdown code fence parser bug in `dsh_research` fixed (`_strip_code_fence`).
   - Precondition check decoupled from `dsh-research` role (`cli.py:3831`).
   - Multi-partition reader unreadability resolved with `_per_case_partition`.

### Partially Completed / Open Defects:
1. **`build_compare` with Per-Case Runs:**
   - `build_compare` still fails with `duplicate_eval_task` when comparing two runs that used per-case partitioning.
2. **`last_tested_at` Unknown Everywhere:**
   - `created_at` is caller-supplied and defaults to `None`. For `uroboros.lifecycle.v2` bundles (as present on Deck), anchored timestamps (`utc_ns`) exist and should be derived per-bundle.
3. **Deck Deployed Candidate Stale:**
   - Deck's provisioned dashboard at `127.0.0.1:18081` runs the older image/wheel (`db128038`) and does not reflect historical sibling binds or document axes (though its hand-run container at `18090` does).
4. **Build of `cd3d87e` Missing on Share:**
   - `p10-ryzen-build-319dac0` is the latest build on `/mnt/hpserver-storage/uroboros-transfer/`; no wheel or image for `cd3d87e` has been published.
5. **Fixture Audit for Presentation Testing:**
   - Optional fields in test fixtures default to `None`, leaving populated template branches untested.
   - Document column presentation is tested only on the live container because test fixtures use the `task-a` mock name instead of real task names (`dsh_document_evidence`, `bfcl`).
6. **Unmet Qualification Gates:**
   - S4.1 (compatibility execution) and S4.2 (export staging) remain unimplemented (`compatibility_not_executed`).
   - A7 (launch flag tuning optimization) undemonstrated.

---

## User Review Required

> [!IMPORTANT]
> **Branch Integration on Main**:
> The `main` branch currently sits at `cd3d87e`. The local worktree branch `worktree-p10v2-ryzen` contains two documentation commits ahead of `main`:
> 1. `a23c036` ("P10 v2 Ryzen: final cumulative closeout")
> 2. `3258fac` ("P10 v2 Ryzen: compact final P10 handoff", creating `P10_FINAL_HANDOFF.md`)
> These will be fast-forwarded/integrated into `main`.

> [!IMPORTANT]
> **Three-Claims Boundary Preservation**:
> The benchmarking tool is delivered for diagnostic execution and multi-architecture comparisons. All benchmark outcomes (BFCL, IFEval, HumanEval, and 36 document cases) are nonqualifying diagnostic observations. They must not be mislabeled as role qualification without executing the full qualification pipeline.

---

## Open Questions

> [!NOTE]
> 1. **Export Staging & Compatibility Execution**:
>    Should S4.1/S4.2 be marked as deferred to a dedicated qualification phase (retaining the honest `COMPATIBILITY_NOT_EXECUTED` reason code), or should an offline compatibility receipt verification pipeline be staged?
> 2. **Candidate Packaging & Share Deployment**:
>    After fixing the remaining software defects, should a fresh candidate wheel and dashboard image (`cd3d87e` + fixes) be built and published to `/mnt/hpserver-storage/uroboros-transfer/` so the Steam Deck can update its provisioned install?

---

## Proposed Changes

### Documentation & Repository State

#### [MODIFY] [CURRENT_HANDOFF.md](file:///home/avg-usr/Documents/Codex/uroboros/docs/state/CURRENT_HANDOFF.md)
- Merge `a23c036` final cumulative closeout section into `main`.

#### [NEW] [P10_FINAL_HANDOFF.md](file:///home/avg-usr/Documents/Codex/uroboros/docs/state/P10_FINAL_HANDOFF.md)
- Bring `P10_FINAL_HANDOFF.md` from `worktree-p10v2-ryzen` onto `main`.

---

### Projections & Comparators (`uroboros/`)

#### [MODIFY] [projections.py](file:///home/avg-usr/Documents/Codex/uroboros/uroboros/projections.py)
- In `build_compare()`: permit repeated task names if the reports indicate `_per_case_partition`, removing the spurious `duplicate_eval_task` error when comparing diagnostic runs.
- In `_status_for_bundle()` and `load_report()`: if `created_at` is `None`, inspect `records/lifecycle.json`. If `schema == "uroboros.lifecycle.v2"` and anchored intervals exist with `utc_ns`, convert the earliest interval to an ISO 8601 UTC timestamp. If absent or v1, preserve `None`.

#### [MODIFY] [run_projection.py](file:///home/avg-usr/Documents/Codex/uroboros/uroboros/run_projection.py)
- In `build_projection()`: pass the bundle-derived timestamp into the projection payload so `uroboros.run-projection.v2` carries accurate timestamps when available.

---

### Dashboard Server & UI (`uroboros/dashboard/`)

#### [MODIFY] [server.py](file:///home/avg-usr/Documents/Codex/uroboros/uroboros/dashboard/server.py)
- Ensure `_diagnostic_suites()` outputs comprehensive per-run resolution metadata in `/api/v1/assessments` so callers can distinguish resolved, unsupported, unreadable, and missing runs explicitly.

#### [MODIFY] [data.py](file:///home/avg-usr/Documents/Codex/uroboros/uroboros/dashboard/data.py)
- Widen projection reading on `/runs` to surface available performance fields (memory domain, duration breakdown) alongside server-measured token rates.

---

### Test Suite (`tests/`)

#### [MODIFY] [test_projections.py](file:///home/avg-usr/Documents/Codex/uroboros/tests/test_projections.py)
- Add unit tests for `build_compare()` with per-case partitioned reports.
- Add unit tests for `created_at` timestamp extraction from v2 lifecycle bundles vs v1 lifecycle bundles.

#### [MODIFY] [test_dashboard_consolidated_history.py](file:///home/avg-usr/Documents/Codex/uroboros/tests/test_dashboard_consolidated_history.py)
- Add test cases using real suite names (`dsh_document_evidence`, `bfcl`) to verify that suite rows and document evidence columns (`coverage`, `task_success`, `critical_violation`) render accurately in automated tests.

#### [MODIFY] [test_dashboard_server.py](file:///home/avg-usr/Documents/Codex/uroboros/tests/test_dashboard_server.py)
- Add tests verifying populated model and run fixture presentation.

---

## Verification Plan

### Automated Tests
1. **Targeted Subsystem Tests**:
   ```bash
   /home/avg-usr/Documents/Codex/uroboros/.venv-v30/bin/python -m pytest tests/test_projections.py tests/test_run_projection.py tests/test_dashboard_server.py tests/test_dashboard_consolidated_history.py -v
   ```
2. **Lint & Style Check**:
   ```bash
   /home/avg-usr/Documents/Codex/uroboros/.venv-v30/bin/ruff check uroboros tail tests
   ```
3. **Full Integration Test Gate**:
   ```bash
   /home/avg-usr/Documents/Codex/uroboros/.venv-v30/bin/python -m pytest tests/ tail/tests/ --ignore=tests/test_qualification_public_lifecycle.py --ignore=tests/test_pty_acceptance.py -q
   /home/avg-usr/Documents/Codex/uroboros/.venv-v30/bin/python -m pytest tests/test_qualification_public_lifecycle.py tests/test_pty_acceptance.py -v
   ```

### Manual & Share Verification
1. **Candidate Build & Share Publication**:
   - Build updated wheel and candidate package.
   - Verify artifacts and generate `SHA256SUMS` in `/mnt/hpserver-storage/uroboros-transfer/p10-ryzen-build-<sha>/`.
2. **Dashboard Health Verification**:
   - Verify `http://127.0.0.1:18081` serves HTTP 200 across all 10 endpoints.
   - Confirm `/assessments` displays both benchmark and document evidence suites with accurate per-run accounting.
