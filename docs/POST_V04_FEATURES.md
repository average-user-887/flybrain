# Features to restore or implement after v0.4

Owner request recorded 2026-10-07. Status: planned, not implemented or promised
for a particular release. Unsupported buttons are capability requests, not a
reason to discard the capability permanently. Reimplement useful functions in
the supported product rather than resurrecting misleading legacy entry points.
The [capability matrix](CAPABILITY_MATRIX.md) describes current scientific
support; the [retirement index](RETIREMENT_INDEX.md) preserves historical paths.

## Release boundary

This backlog does not defer v0.4 defects: truthful labels, working advertised
controls, authoritative stimulus rendering, recording and checkpoint recovery,
saved-brain preservation, and browser acceptance remain current release gates.
AMD support stays in the v0.4 qualification scope. No unsupported button becomes
enabled merely because it appears here.

## Prioritized backlog

Order is a proposed sequence within each area; dependencies and evidence govern
scheduling. Each item needs a bounded implementation card before coding.

| ID | Capability to restore or build | Acceptance before enabling or claiming support |
|---|---|---|
| NEXT-01 | Connected assay controls currently preview-only, including threat placement and spatial stimulus editing | Inventory every disabled/retired control and map it to a real daemon capability or an explicit design rejection. Commands must change the intended stimulus, acknowledge applied state, record provenance, and survive assay switches. Verify each control in the running browser. Keep existing working airflow controls. |
| NEXT-02 | Anatomical lesion and intervention tools | Select neurons from identified anatomy; record exact IDs and intervention parameters; support reversible interventions and sham controls without changing stored source wiring. Verify causal effects and browser feedback. |
| NEXT-03 | Genuine connectome learning controls: teach, reverse, probe and freeze | Wire supported plasticity to identified KC/DAN/MBON circuitry. Demonstrate retained synaptic changes, held-out conditioning results and appropriate controls. Fixed-connectome mode remains explicitly non-learning; the modular helper cannot stand in for graph learning. |
| NEXT-04 | Graph training history and portable brain export/import | Show graph-owned events and checkpoint provenance; export actual neural state separately from summary reports. Verify same-model round trips and declared cross-version migrations; preserve source data and refuse incompatible imports visibly. Existing saved-state preservation is a v0.4 gate. |
| NEXT-05 | Reproducible whole-connectome experiment batteries | Replace retired full-sim and scientific-battery entry points with supported execution and validation APIs. Use actual sensory input and graph motor output, adequate assay duration, preregistered outcomes and per-run identity. Never hardcode conclusions or significance. |
| NEXT-06 | Photoreceptor-driven motion computation, saccadic efference copy and richer cell dynamics | Evaluate physiology-grounded hypotheses with motion behavior held out. Require measured direction selectivity and physiological activity from supported input pathways; report missing anatomy. No hidden direction-selective adapter or invented wiring. Efference copy is a proposal: it needs an identified pathway in the connectome and measured suppression, not a hand-set factor such as the browser preview's 85%. Research hypotheses are not established fixes. |
| NEXT-07 | Working memory, visual place memory, pattern-specific operant memory and labyrinth planning | Give each behavior its own protocol, neural mechanism, memory-retention test and causal controls. An arena, score or scripted controller is not evidence of learning. |
| NEXT-08 | Circadian oscillator, moving courtship partner and courtship memory | Separate environmental stimulus support from neural behavior. Validate an endogenous rhythm or retained courtship learning against declared controls before upgrading claims. |
| NEXT-09 | Turbulent odor plumes and richer spatial environments | Specify reproducible stimulus fields, seeds and sensor sampling; record and replay the same field and render its authoritative state in 2D/3D. Validate neural navigation separately. |
| NEXT-10 | Biological body control, gap-crossing planning and multisensory coordination | Add supported VNC, proprioceptive and muscle mechanisms with anatomical provenance. Measure contact, reach and coordination against declared criteria; label any external CPG or scripted leg motion. |
| NEXT-11 | Broader AMD dynamics/plasticity support and performance improvements | Retain current AMD support. Qualify each additional model on real hardware against the reference, including numerical tolerance, save/restore, failure behavior and measured resources. Require independent Antigravity audit before AMD merge. |
| NEXT-12 | Unified research workspace and dependable distribution | Restore useful functions from retired research pages inside the maintained dashboard, with current identity checks and recording. Use supported package/install/update paths instead of incomplete sync scripts. Test fresh installs and visible browser workflows; duplicate pages are not a feature requirement. |

## Completion rules

- Attach exact code and graph identities, data provenance and reproducible evidence
  to each completed item. Record software functionality separately from biological
  validation; negative scientific results remain valid results.
- Honor [owner decisions](OWNER_DECISIONS.md): the connectome does the computation;
  fitting permission is not expanded by this backlog. Preserve saved brains and
  published history. New dependencies require verified licensing.
- Update the capability matrix only to the level the evidence supports. For UI
  work, click the real controls after the final change, inspect rendering and
  errors, and restore the user's starting state under the project browser rule.
- If a requested capability is rejected or replaced, keep its backlog entry and
  explain the decision and supported alternative instead of silently deleting it.
