# Claude remediation plan: independent learning connectome experiments

Prepared 19 September 2026 against implementation revision `355f50e`.
Read [NEUROFLY_RETHINK.md](NEUROFLY_RETHINK.md) first. The user accepted the six
recommendations from the grilling interview and requested this remediation plan.
This document records the agreed direction and implementation gates. This turn
prepares the plan only; no implementation or research restart is included.

## Workspace and non-negotiable constraints

Canonical running checkout:
`<repository-worktree-root>`

The root checkout is older and has separate work. Do not synchronize or overwrite
it. The actual prepared full graph currently resides in that root checkout;
the active worktree's default RPC graph path does not exist. Resolve this explicitly.

* Keep all 14 assay entries and the original dark dashboard style.
* Preserve all brain checkpoints, datasets, failed runs and user changes.
* Do not add behavioral rules merely to produce expected-looking curves.
* Never substitute a synthetic graph or surrogate without an explicit mode and
  a new, recorded run identity. The real graph may legitimately fail a task.
* Do not claim Firefox support from an in-app-browser run. Do not count executed
  timesteps as proof of movement, responsiveness or learning.
* `scripts/dashboard_headless_check.py` is pre-existing untracked work. Inspect
  its assumptions; do not overwrite it or count its POST-blocked mode as a live
  control test.
* No public upload or remote is configured. Prepare a reviewable release without
  publishing it or contacting contributors unless the user authorizes that step.

The main daemon is paused at wind-tunnel, 1×; the background research service is
stopped. Installed services remain enabled, so these are not guaranteed to remain
paused after a service restart or login. Do not silently resume research while
the user is reviewing the direction. The temporary port-18784 audit process was
stopped. Existing API/web ports are 8781/8780.

## Agreed decision contract

1. Research first. A failed run is a retained experimental result, not a reason
   to insert a hidden controller that makes the fly succeed.
2. Use open released connectome data and this computer only. No paid compute or
   real-time performance promise. Measure memory and throughput before scaling.
3. Internal synaptic plasticity is the learning target. A trained external
   decoder and the existing modular brains remain separately labeled baselines.
   Released anatomy does not establish that our dynamics or plasticity are biological.
4. Physics enforces physical constraints without steering toward task success.
   Preserve immobility or wall pushing as a behavioral outcome within predefined
   trial limits; distinguish that outcome from a faulty collision solver.
5. Every assay gets its own neural state, learning state and random state. Run
   one full-brain experiment at a time initially; share only immutable data.
6. Preserve all 14 experiments and their visual style. First validate one causal
   sensory–brain–motor pathway, then internal learning, then extend to every assay.

The target is a learning-capable full-connectome instance in every experiment,
not guaranteed successful learning in every task. Unsupported sensory mappings
must be visible and block scientific claims, rather than trigger a surrogate.
Optomotor yaw is the initial engineering candidate; verify its mapping before
committing to a learning protocol. A physical robot target is deferred until the
simulation interface and measured compute budget support that decision.

The work packages below are ordered by dependency, not calendar promises.

## Work package 1 — capture failures and make them observable

Before changing model dynamics, reproduce the reported Firefox failure and the
recorded 100× disconnect. Capture version, OS, viewport/zoom, URL, code revision,
controller mode, request/achieved speed, console errors and network timing.

Relevant files: `web/app.js`, `neurofly_daemon.py`, `stream_gateway.py`.

Add structured error reporting around startup, packet parsing/application and
rendering. A failure must identify the assay/run/step and freeze or recover in a
defined way. Do not silently reset the brain, continue recording invented state,
or create an endless loop of swallowed exceptions. Display last valid data age.
Preserve the complete snapshot on pause/disconnect, including metrics and labels.

Acceptance:

* Inject a malformed packet and a renderer exception in a test build: show the
  error, keep raw-data integrity, recover through an explicit supported path.
* Pause retains the measured assay metrics; simulated clock and weights stop.
* Disconnect preserves labels, units and the last measured frame. Preview data
  must not replace it under the same run identity.
* Reproduce/test in **actual Firefox** and Chromium. If unavailable, report that
  browser as untested; do not close its issue.

## Work package 2 — fix timing and separate computation from delivery

Use one authoritative scientific simulator. Keep any local preview explicitly
illustrative and isolated from scientific exports. Never change integration dt
merely because the user changes display speed.

Replace the current pacing threshold with deadline-aware scheduling. Maintain
requested speed and measured achieved speed separately. Profile shared-lock
waits and request latency; publish immutable telemetry snapshots without holding
the simulation's main lock during delivery. Commands need explicit application
steps and acknowledgments. Choose backpressure/decimation policy deliberately.

Acceptance:

* With identical seeds, initial state and step-indexed stimuli, compare state and
  learned weights at the same simulated steps under requested 1×, 20× and 100×.
  Define numerical tolerance before running; inspect divergence rather than
  adjusting the threshold until tests pass.
* Report achieved rate, integration dt, simulation time, snapshot age and dropped/
  decimated display frames. Overload reduces achieved speed visibly.
* On documented hardware, run an isolated 60-wall-second stress check; require
  no stale-stream disconnect, and record command response latency. Proposed
  responsiveness gate: p95 <250 ms and maximum <1 s, or document and agree a
  different target before claiming support. Bound compute to meet the gate.
* Display trajectory segments/events, not only widely spaced endpoint snapshots.
  Rendering interpolation is labeled display-only and never saved as raw data.
  Trails must not imply a straight path through a wall where the actual path turned.

Evidence to reproduce: `outputs/rethink-audit/browser-speed-100.json`,
`browser-freeze.json`, `timing_diagnostic.py`, `timing_receipt.json`.

## Work package 3 — replace weak movement criteria

Relevant files: `arena.py`, `maze.py`, `scripts/containment_audit.py`,
`scripts/behavior_audit.py`, `tests/test_containment_all_paradigms.py`.

Create deterministic fixtures for head-on contact, concave/convex corners,
narrow corridors, a cue behind a wall, equal antenna signals, reverse movement
and restored trained checkpoints. Test beyond 12 simulated seconds and include
the user's retained-brain conditions using copies.

Record attempted motor commands, realized displacement, contact normals, solver
corrections, time near walls, rolling progress and repeated stalls. Distinguish
intentional tethering, rest, feeding and gap probing from numerical trapping.
Do not disguise a locomotion policy failure as a collision-engine failure.

Acceptance:

* Physics fixtures using known escape commands fail on numerical trapping or
  unexplained jumps. Behavioral trials report immobility and repeated wall
  pushing as failed task outcomes, not automatically as physics defects.
  A step-count assertion is insufficient for either claim.
* Each fixture declares its expected physical constraint and progress criterion
  in advance. Arena containment, controller task success and biological validity
  are reported separately.
* In the live browser, all 14 assays remain selectable at tested rates with no
  hidden resets. Review the same trajectory in raw data and playback.

## Work package 4 — make controller provenance impossible to confuse

Define explicit backends such as `modular`, `connectome-fixed`,
`connectome-plastic`, and `connectome-with-trained-readout`. An experimental hybrid, if retained, must be
named separately. Do not represent any of these as interchangeable.

Every run needs graph/data hashes, neuron-map hash, controller version, dynamics
parameters, units, seed and RNG state, source revision, learned parameter
locations, intervention schedule and reset/fallback events. The UI and export
must display the same identity. Complete replay snapshots must include world,
brain transients and random state, not just trained weights.

Acceptance:

* Missing graph or mapping fails explicitly in scientific mode. Synthetic mode
  requires an explicit test option and is conspicuously labeled.
* Resolve the real graph path and verify its IDs/hashes before starting.
* Disconnecting RPC does not leave unreported surrogate commands driving the fly.
* Same snapshot and same input sequence replay to the documented tolerance.

Create an experiment registry with separate instance IDs and checkpoint paths
for all 14 assays. Share the immutable graph and stable neuron map; keep membrane
state, refractory state, synaptic traces, plastic weight changes, learning-rule
state, RNG and world state per instance. Profile whether sparse plastic deltas
save memory; do not assume sparse storage is always smaller. Inactive instances
are checkpointed and do not train in the background.

Isolation acceptance:

* Train instance A and verify B's saved mutable state is unchanged. Switch A→B→A
  and compare A's continuation against an uninterrupted reference at equal steps.
* Switching acknowledges the newly active instance only after its world and brain
  snapshot are ready. Packets carry instance/run IDs; stale packets are rejected.
* Checkpoint writes are atomic and versioned. Interrupted writes retain the last
  valid checkpoint. Never reinterpret old modular weights as graph weights.
* Record peak RAM, checkpoint size/load time and sustained throughput on this
  machine. If the graph exceeds the budget, report the blocker; any reduced
  circuit is a separately named experiment requiring an explicit scope decision.

## Work package 5 — prove one causal full-graph control loop

Initial candidate: optomotor yaw. Keep every
other assay in the roster with an honest capability status.

Relevant files: `brainlab/cosim_server.py`, `connectome_client.py`,
`connectome_bridge.py`, `brainlab/brain.py`, `arena.py`.

Resolve sensory/output neurons through stable IDs, cell types and verified side
annotations. Do not split dataframe row order into left/right. Pin graph identity.
The currently selected DN cell types match the installed graph; side assignments
and mapping stability still require verification. Current nominal visual channels
mix both eyes—see the mapping audit receipt.

Separate sensor encoding, neural simulation, motor decoding and body dynamics.
All graph output values, including zero, must have their documented effect.
Avoid positive-only overrides that preserve surrogate movement when the graph
is silent. Log each motor contribution and any engineered assistance.

Acceptance:

* Matched left/right and contrast stimuli produce a recorded neural response;
  a claimed motor effect has a measured, repeatable dependence on that response.
* Compare intact graph, selected output silenced, disconnected/sham graph and
  modular baseline. Keep the stimulus schedule fixed for causal comparisons.
* Report a failed or absent effect honestly. Do not tune a direct stimulus→motor
  rule and present it as graph-mediated success.
* Establish sustained compute cost before promising real-time operation.

## Work package 6 — implement and evaluate internal learning

Before coding a rule, write a short model specification with primary sources:
which anatomically identified connections are plastic, what local signals and
reward/modulatory signals update them, update units and timestep, bounds/sign
constraints, initial values, and what remains an engineering assumption. Do not
apply arbitrary plasticity to every edge or claim that the release supplied the
learning rule. Keep the full graph while restricting learning to declared edges.

Select the first learning assay by verified sensory and modulatory mappings.
Odor conditioning is a candidate, not an established capability. Freeze the
encoder and motor decoder for the main internal-learning comparison so changes
in behavior cannot be explained by simultaneous decoder training.

Acceptance:

* Verify updates against the declared mathematical rule on a small circuit,
  including no-update conditions, bounds, sign handling and checkpoint recovery.
* Predeclare task outcome, trial duration, training budget, retention interval,
  seed list and analysis before confirmatory runs. Use a separate pilot to estimate
  runtime/variability and choose a feasible sample count on this computer.
* Compare matched instances with plasticity enabled, plasticity disabled and
  shuffled/yoked reward; include a relevant pathway intervention. Keep initial
  state and exposure budgets matched and separate training from frozen-weight tests.
* Record the actual internal weight changes, acquisition, held-out performance,
  retention and uncertainty. A changing weight or increasing training reward alone
  is not evidence of useful learning. Retain null and negative outcomes.
* Do not mark biological learning validated: this is learning in a model constrained
  by released connectivity, with an explicit hypothesis about plasticity.

## Work package 7 — extend the learning backend across the roster

Create a 14-row capability matrix, one row per existing assay: sensory encoder,
verified neuron IDs, motor outputs/units, reward protocol, plastic subset, instance
ID, checkpoint, outcome metric, controls, and evidence links. Use statuses such as
unmapped, integrated, causally tested and learning evaluated; report positive/null/
negative learning results separately from software readiness.

Integrate assays in batches according to shared validated input/output pathways.
Each row must use its own full-connectome learning instance, pass isolation and
physics checks, and complete its declared evaluation before its status advances.
An assay with an unresolved input pathway stays visible but explicitly unsupported
for scientific graph runs. Do not invent a neural mapping to finish the checklist.

Store a machine-readable run manifest, step-indexed stimuli/actions/rewards,
neural summaries, contact events, reset reasons and checkpoint hashes. Capture
targeted spike/weight traces with a declared sampling policy and bounded storage;
do not attempt to save every neuron at every timestep by default. Exports include
failed/aborted runs and link each result to its instance and source revision.
Test replay/export on representative runs and verify no cross-assay data mixing.

The roster gate is complete when every assay has an independent learning-capable
graph backend and an honest evaluated outcome, not when every fly solves its task.

## Work package 8 — community release and robotics adapter

Define these contracts during the earlier integration work and package them once
provenance and the first causal loop are credible: timestamped `SensorPacket`, `Brain.step(dt)`, `MotorCommand`,
`World.step(dt)`, checkpoint and recorder contracts. Include units and coordinate
frames, controller capabilities, errors and stale-data semantics.

Start robotics with recorded sensor playback and a simulated actuator adapter.
Then choose a physical robot with the user. Actuation needs bounded outputs,
deadline monitoring and a stop on stale/missing commands. Keep robot wall time
separate from accelerated simulation time.

Release acceptance:

* One canonical entry point and a portable foreground launcher; systemd optional.
* A clean clone can run a small synthetic test, and a separately documented path
  retrieves/verifies the real data. No local absolute path is required.
* Preserve MIT/NOTICE/data attribution; remove conflicting science-guide claims.
* Document all 14 experiments with supported controller combinations, expected
  inputs/outputs, learning status and known failures. No invented success scores.
* Attach real Firefox/Chromium artifacts for 1×/20×/100×, rapid switches,
  pause/resume, reconnect and background-tab return. Keep `AGENTS.md`'s final
  browser rule; a headless/unit pass alone cannot sign off the UI.

## Suggested small commits

1. Reproduction fixtures, diagnostic receipts and visible error states.
2. Fixed-step pacing, achieved-speed telemetry and responsive snapshot delivery.
3. Wall-progress fixtures and targeted physical corrections supported by them.
4. Explicit controller identity, validated mapping and failure semantics.
5. One graph-mediated assay with causal controls and reproducible traces.
6. Declared internal plasticity and matched learning evaluations.
7. Per-assay integration, capability matrix and reproducible run exports.
8. Portable packaging and a simulation-first robotics interface.

## First implementation session and stop conditions

Start with work package 1: confirm the checkout/revision, preserve checkpoints,
reproduce Firefox and acceleration failures, and capture a baseline receipt.
Then fix observable transport/rendering failures before changing neural dynamics.
Make small commits with evidence and keep the current dashboard layout intact.

Do not progress past a gate by hiding a missing graph, substituting a controller,
changing the declared metric after seeing results, or treating an untested browser
as passed. If local resources or missing anatomical annotations block a stage,
report the measured constraint and proposed alternatives. Timing/browser fixes
can proceed independently of plasticity model research; scientific claims cannot
precede mapping, provenance and causal validation.

Report after each work package: problem, evidence, change, tests, browser result,
remaining uncertainty and next dependency. Do not close the broad project goal
because a fly animates or a test count is large.
