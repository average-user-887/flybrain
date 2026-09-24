# Superseded Plan Documents

As of 2026-09-22, these documents are **historical references** — their
technical findings and evidence remain valid, but their roadmaps, milestones,
and capability claims are superseded by [`docs/ROADMAP.md`](ROADMAP.md).

Where these documents conflict with each other, the resolution is recorded in
the consolidated roadmap. Where they contain valid evidence (probe results,
audit receipts, withdrawn claims), that evidence is cited by reference.

## Documents and their status

| Document | Date | Author | Why superseded |
|---|---|---|---|
| [`docs/archive/CLAUDE_HANDOFF.md`](archive/CLAUDE_HANDOFF.md) | Sep 19 | ChatGPT | Presents the modular controller as if it is the connectome. Overclaims (e.g. "100% operational whole-brain simulation") were [withdrawn](NEUROFLY_RETHINK.md) by a later audit. §3–7 architecture descriptions are still accurate for the modular controller. |
| [`docs/OPEN_SOURCE_PLAN.md`](OPEN_SOURCE_PLAN.md) | Sep 19 | Claude | 5-phase plan partially executed (WP1–5). Phase 5 was never started. The WP5 causal-loop claim was [withdrawn](OPEN_SOURCE_PLAN.md#correction-2026-09-20-completed-v2-set). Open-source hygiene items are carried forward into Phase 5 of the roadmap. |
| [`docs/NEUROFLY_RETHINK.md`](NEUROFLY_RETHINK.md) | Sep 19 | Claude | **The most honest document.** Its audit findings and decision record are fully incorporated into the roadmap's "Current State" section. Not superseded as evidence, only as a plan. |
| [`docs/archive/CLAUDE_IMMEDIATE_PLAN.md`](archive/CLAUDE_IMMEDIATE_PLAN.md) | Sep 19 | Claude | 8 work packages with gates. WP1–4 completed. WP5 returned NULL. WP6–8 completed in v0.3.0. |
| [`docs/FULL_CONNECTOME_COSIM_PLAN.md`](FULL_CONNECTOME_COSIM_PLAN.md) | Sep 21 | Antigravity | **Owner-approved architecture.** Its 4-component structure (sensory ingress, DN bottleneck, articulated walking, 3D dashboard) was implemented in v0.3.0. |
| [`docs/archive/CODEX_HANDOFF.md`](archive/CODEX_HANDOFF.md) | Sep 21 | Claude (orchestrator) | Honest handoff. Infrastructure details are operational records, not part of the product plan. |
| [`docs/REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) | Sep 21 | Codex | Cleaned repository layout. `docs/EMBODIED_MVP.md` is delivered and active. |
| [`docs/archive/ROADMAP_v1.0.md`](archive/ROADMAP_v1.0.md) | Sep 24 | Mixed | Marked phases "delivered" on the strength of v1 results. Replaced by roadmap v2.0, which resets claims to v3 receipts. |

## Documents that remain authoritative

These are **not** superseded and coexist with the roadmap:

| Document | Role |
|---|---|
| [`docs/LIF_DYNAMICS_SPEC.md`](LIF_DYNAMICS_SPEC.md) | Technical specification for v1/v2/v3 dynamics. Referenced by Phase 1. |
| [`docs/WP6_PLASTICITY_SPEC.md`](WP6_PLASTICITY_SPEC.md) | Plasticity model specification. Referenced by Phase 4. |
| [`docs/RELEASE_AUDIT.md`](RELEASE_AUDIT.md) | Open-source hygiene checklist. Referenced by Phase 5. |
| [`docs/DATA_SCHEMA.md`](DATA_SCHEMA.md) | Telemetry and data export schema. |
| [`docs/receipts/`](receipts/) | Evidence bundles. Never superseded or deleted. |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Contributor guide. |
