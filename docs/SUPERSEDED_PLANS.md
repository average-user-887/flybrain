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
| [`CLAUDE_HANDOFF.md`](../CLAUDE_HANDOFF.md) | Sep 19 | ChatGPT | Presents the modular controller as if it is the connectome. Overclaims (e.g. "100% operational whole-brain simulation") were [withdrawn](NEUROFLY_RETHINK.md) by a later audit. §3–7 architecture descriptions are still accurate for the modular controller. |
| [`docs/OPEN_SOURCE_PLAN.md`](OPEN_SOURCE_PLAN.md) | Sep 19 | Claude | 5-phase plan partially executed (WP1–5). Phase 5 was never started. The WP5 causal-loop claim was [withdrawn](OPEN_SOURCE_PLAN.md#correction-2026-09-20-completed-v2-set). Open-source hygiene items are carried forward into Phase 5 of the roadmap. |
| [`docs/NEUROFLY_RETHINK.md`](NEUROFLY_RETHINK.md) | Sep 19 | Claude | **The most honest document.** Its audit findings and decision record are fully incorporated into the roadmap's "Current State" section. Not superseded as evidence, only as a plan. |
| [`docs/CLAUDE_IMMEDIATE_PLAN.md`](CLAUDE_IMMEDIATE_PLAN.md) | Sep 19 | Claude | 8 work packages with gates. WP1–4 completed. WP5 returned NULL. WP6–8 never started. The work package structure is preserved in Phase 4 of the roadmap; the ordering is revised based on the owner's approval of the FlyGym direction. |
| [`docs/FULL_CONNECTOME_COSIM_PLAN.md`](FULL_CONNECTOME_COSIM_PLAN.md) | Sep 21 | Antigravity | **Owner-approved architecture.** Its 4-component structure (sensory ingress, DN bottleneck, articulated walking, 3D dashboard) is the backbone of Phases 2–3 in the roadmap. Its unverified reference projects have been fact-checked; see roadmap §Architecture. |
| [`docs/CODEX_HANDOFF.md`](CODEX_HANDOFF.md) | Sep 21 | Claude (orchestrator) | Honest handoff. Its "recommended first slice" is incorporated into Phase 1–2 of the roadmap. Infrastructure details are operational records, not part of the product plan. |
| [`docs/REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) | Sep 21 | Codex | References `docs/EMBODIED_MVP.md` which does not exist. Layout description is accurate; the missing MVP doc is replaced by the roadmap's Phase 2 acceptance criteria. |

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
