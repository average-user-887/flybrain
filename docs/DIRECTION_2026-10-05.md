# NeuroFly direction and acceptance gates — 5 October 2026

Status: v0.4 release HOLD. Direction set by Codex under the owner's delegation on
5 October. Claude owns implementation, worker coordination and candidate assembly;
Codex owns scope, independent audit and acceptance. Existing owner decisions remain
binding. This is an engineering direction document, not a claim of owner approval
for new scientific assumptions.

## Destination and evidence

The destination is a biologically grounded fly simulation for studying neural
computation, learning and pattern recognition. The next deliverable is a reliable,
reproducible connectome research instrument. A connectome constrains wiring; it does
not by itself establish cell dynamics, synaptic efficacy, plasticity, sensory
transduction, muscle control or behavioural validity. Each layer needs its own
evidence and declared uncertainty.

The current dataset is MaleCNS v1.0, a male central nervous system including the
nerve cord, from the Janelia/Cambridge/MRC/Google collaboration. It is distinct from
the FlyWire female brain. The prepared graph contains 166,700 nodes and 25,582,938
directed neuron-pair edges weighted by synapse counts; do not call that edge count
the number of individual synapses. Source: <https://male-cns.janelia.org/>;
anatomy and coverage: <https://www.nature.com/articles/s41586-025-08746-0>.

No invented connections, graph completion or behavioural computation in adapters.
The permitted fitting remains non-motion input-cell physiology with the tested
behaviour held out. A new fit outside this permission must be scoped separately;
the WP7 plan does not silently expand it. Engineered interventions may remain as
explicit circuit probes or controls, never as evidence of sensory computation.

## Decisions on the four handoff questions

1. **Core:** one scientific execution and recording contract shared by the daemon,
   dashboard, validation harness and embodied runner. The dashboard is the user
   instrument; the embodied runner is the route toward a physical fly. Converge
   incrementally. Do not start two rewrites or claim that today's CPG/decoder is a
   biological VNC. The current 2D arena is an explicit abstraction.
2. **Retirement:** remove misleading paths from supported/default operation first.
   Archive obsolete bridge/RPC entry points, surrogate batteries labelled as
   connectome, orphan dashboard forks, unused research assets and broken sync
   helpers after checking callers. Preserve receipt-linked source, dependencies,
   pinned commits and a reproduction index. Keep modular only as an explicitly
   engineered baseline, and Studio/body tools as bounded research tools. Do not
   expand the 14 modular behaviours as a substitute for neural simulation.
3. **Save failures:** required scientific recording, checkpoint or provenance failure
   stops the experiment at a safe boundary and marks its result incomplete/invalid.
   Preserve the last good checkpoint and in-memory state; never substitute a fresh
   brain silently. Explicit exploratory mode may continue with persistent
   DEGRADED / NOT SAVING status, bounded retry and a recorded gap. Optional diagnostic
   logging failure may degrade. Compute/device failure always halts, including a
   CUDA error encountered while checkpointing. Do not classify every exception as
   ordinary disk I/O failure. Recovery must be visible and retain the failure event.
4. **v0.4:** may ship honest negative science after the instrument works. Positive
   motion vision, learning and a completed biological body are not release gates.
   Passing unit tests or a moving decorative fly is insufficient. Keep the release
   on hold until the acceptance gates below pass. State supported configurations
   explicitly; two Linux hosts do not prove that it runs on every PC.

## Audit snapshot and limitations

At intake the live checkout was `b0abd8f`, while fetched integration master was
`69e52db`. During the audit E_inh restoration landed at `3e9bc5d`, and the A–F audit
bundle and summary at `f5c8814`. These moving heads are not one tested candidate.

- Audits B/C report modular: 1 fully working assay of 14; connectome: 0 fully
  working, 2 partial, 5 no behaviour, 7 misleading. Consult the per-assay receipts.
- Audit D's 68 working controls of 97 does not mean the other 29 are all proven
  broken: categories include misleading, dead, no visible effect and unverified.
- Source inspection confirms the unconditional DNb01 current and +5 speed floor
  in the general graph controller despite Assistance OFF.
- Codex reproduced the real 3D toggle failure on the running observatory: clicking
  View: 2D Arena produced `Cannot read properties of undefined (reading 'x')` and
  suspended rendering while the live step counter advanced. Switching back to 2D
  plus Resume view recovered it. The starting wind-tunnel, connectome-plastic,
  0.5x, running state was preserved. No saved brains or service processes were
  changed. This was a targeted reproduction, not a full browser sign-off.
- Worker browser receipts on isolated/headless services are diagnostic evidence.
  Final acceptance still requires the owner's running UI under AGENTS.md.
- Freeze work was at `d21d7ca` with an uncommitted browser-check edit; body port at
  `ad4d2d7` with final numbers pending when inspected. Require new exact candidates.
  Neither those branch labels nor old test totals establish an integrated pass.
- No full suite, large-graph hash comparison or new biological simulation was run
  by Codex for this direction audit. Reported worker results retain their original
  commit and hardware scope.

## Work order and ownership

Maintain at least two implementation/test lanes: a Claude worker on NVIDIA and a
Claude worker on the AMD Steam Deck. Use the owner's requested Opus 5.5 where the
session actually supports it; report the actual model rather than assuming it.
The NVIDIA Claude coordinates day-to-day execution. The Deck independently tests
and may propose fixes on its own branch; it does not validate NVIDIA-only behavior.
CPU execution on an AMD host is AMD CPU coverage, not AMD GPU coverage. An AMD GPU
port is deferred until correctness and performance measurements justify one.

| Priority | Work package | Responsible lane | Acceptance evidence |
|---|---|---|---|
| P0 | Finish freeze/watchdog candidate and error classification | NVIDIA implementation; Deck fault replication | Every F1–F7 path visible; heartbeat separate from simulated progress; paused/slow CPU distinguished from dead; required capture failure invalidates run; real compute errors halt |
| P0 | Truthful graph execution and UI | NVIDIA | Hidden DNb01 drive/floor absent from unassisted mode; every input/intervention and decoder declared; zero input/output controls; modular panels absent from graph runs; absent data distinct from zero |
| P0 | Restore user controls and assay contracts | NVIDIA; Deck independent browser pass | 2D/3D works; invalid actions disabled with reasons; outcomes survive trial completion; durations permit specified outcomes; no success/learning score from immobility or an empty denominator |
| P1 | Clean install and architecture matrix | Deck T7–T9; NVIDIA counterpart | Same pinned candidate, fresh install, exact dependencies, full test counts/skips, actual compute backend, memory and sim-s/wall-s, real browser evidence |
| P1 | Body verdict port and reproducibility repairs | NVIDIA; Deck CPU repeat | Integrate on current master; preserve privacy/install/Studio work; stimulus reversal, null/lesion/channel-swap controls, both decoder identities, exact commands and receipts |
| P1 | Documentation and retirement manifest | NVIDIA, reviewed by Codex | Current release state/claims consistent; archival path and reproduction command for each retired receipt dependency |
| P2 | Physiology and causal computation | NVIDIA research; Deck replication | Preregistered stages below; no behavioural tuning to make the demonstration work |

For unsupported assays, keep a visible capability entry with a precise reason;
disable execution or label it an unvalidated probe. Do not fabricate outcomes.
v0.4 needs at least one real-graph experiment that completes with a valid trace and
an honest result, including a negative result or an explicitly labelled downstream
circuit probe. Fourteen scientifically positive assays are not required.

## Scientific gates after instrument repair

1. **Identity and physiology:** immutable topology/source checks; dynamics, units,
   input maps, output maps and assistance pinned in each run. Audit sensory target
   identity and dose-response, baseline, transient and offset behavior. Report
   anatomically connected subsets alongside whole-population coverage. Reproduce
   the odor/thermal/cVA persistent activity observation with null, offset and
   numerical controls before attributing it to anatomy or claiming loss of all
   stimulus information. Neither similar population totals nor persistent firing
   alone proves the absence of information.
2. **Visual computation:** measure transfer and non-motion timing from photoreceptors
   through successive cell classes. Freeze permitted calibration before held-out
   direction/contrast/speed tests. v3/v4/v5 negatives are model-specific. v6a failed
   the amplitude gate; its subsequent motion verdict was not run. Never describe
   these results as proof that the biological connectome cannot compute motion.
3. **Learning and recognition:** first establish distinguishable sensory responses
   and sparse, odor-specific KC activity. Test plasticity only on existing declared
   edges, with paired/unpaired/backward, plasticity-off, dopamine-blocked, retention
   and novel-stimulus controls. Separate network changes from a trained readout or
   hand-coded memory. Fitting a depression target and checking it with fresh seeds
   is calibration robustness, not independent biological validation. Respect the
   owner's narrow fitting permission before selecting learning parameters.
4. **Embodiment:** attach validated neural computations to measured body dynamics;
   show the contributions of CPGs, reflexes and decoders through causal controls.
   Expand toward VNC, proprioception and muscles in measured stages. Do not label
   a complete software loop a full biological fly.

E_inh correction ticket: little chloride current at rest does not imply chloride
reversal near rest without knowing conductance (`I = g(V - E)`). A GABA-induced
voltage endpoint is not itself a reversal measurement. Independently check the
primary evidence before privileging -60 mV or a claimed adult range. Code restoration
and result reproduction do not validate that interpretation. Preserve the locked
historical declaration and append a dated correction or new preregistration.

## Review, integration and release gates

Each handoff to Codex names: base and candidate SHA, worker/model and hardware,
scope and non-goals, dirty files, exact tests and skips, scientific controls,
browser actions and limitations, raw receipt locations, rollback and next blocker.
No claim is promoted solely because its author says PASS.

Claude assembles candidates; Codex reviews before master integration, deployment
or release. Use ordinary fast-forwards/merges, never rewrite published history.
Avoid wholesale refactors in the freeze fix. Serialize changes to shared controller
and UI files. Changes made after validation require appropriate revalidation on the
new exact candidate.

Before v0.4:

- A clean supported NVIDIA install and AMD CPU install reproduce the same declared
  experiment on the same commit/data/seed. Numerical acceptance distinguishes exact
  checks from predeclared tolerances; never transfer a v3 CPU result to all GPU engines.
- Runtime, provenance, data-integrity and truthful-assistance gates above pass.
- The real dashboard is tested after final deployment and existing-tab reload.
  Visit all 14 assays; test the reported transition, rapid consecutive selections,
  and two-tab synchronization; click real controls and inspect arena, assay, clock,
  tools, telemetry and browser errors. Preserve brains and restore the starting
  assay, speed and pause state. Record what was actually tested.
- Every enabled assay has a duration, terminal outcome, valid metric and capability
  label. Negative, unavailable, incomplete and untested are distinct states.
- Current documentation, privacy guard, install path and measured performance agree
  with receipts. No personal paths or private coordination data enter public docs.
- A single pinned candidate has independent review and both hardware receipts.

This direction does not authorize an unattended service restart during another
worker's test. Coordinate the live deployment window and preserve recovery data.
