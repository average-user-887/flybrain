# Claude handoff: stabilize NeuroFly and make brain identity auditable

Prepared 19 September 2026 against implementation revision `355f50e`.
Read [NEUROFLY_RETHINK.md](NEUROFLY_RETHINK.md) first. This is a proposed work
sequence. The user has requested a stop and a grilling interview; do not treat
this document as approval to begin a redesign before the open decisions are answered.

## Workspace and non-negotiable constraints

Canonical running checkout:
`/home/avg-usr/Documents/ChatGPT/flybrain/.claude/worktrees/neurofly-openready`

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

## First: establish the decision contract

Await answers to the current interview frontier:

1. Is the first deliverable a connectome research platform, a reliable robot
   controller, or an educational simulator?
2. Is the initial resource budget this computer, funded compute, or a longer
   contributor-driven effort?

Recommended direction: a connectome-controller research toolkit with explicit
modular baselines, initially measured on this computer. Subsequent questions must
settle acceptable decoding/engineered assistance, where learning must occur,
the first causal assay and the robotics target. Keep those decisions separate
from facts the code can establish.

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

* Tests explicitly fail on a motionless non-exempt agent, repeated wall-pushing
  loops and unexplained jumps. A step-count assertion is insufficient.
* Each fixture declares its expected physical constraint and progress criterion
  in advance. Arena containment, controller task success and biological validity
  are reported separately.
* In the live browser, all 14 assays remain selectable at tested rates with no
  hidden resets. Review the same trajectory in raw data and playback.

## Work package 4 — make controller provenance impossible to confuse

Define explicit backends such as `modular`, `connectome-fixed`, and
`connectome-with-trained-readout`. An experimental hybrid, if retained, must be
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

## Work package 5 — prove one causal full-graph control loop

Recommended first candidate, subject to the interview: optomotor yaw. Keep every
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

## Work package 6 — community release and robotics adapter

Only after provenance and the first causal loop are credible, define a minimal
interface: timestamped `SensorPacket`, `Brain.step(dt)`, `MotorCommand`,
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
6. Portable packaging and a simulation-first robotics interface.

Report after each work package: problem, evidence, change, tests, browser result,
remaining uncertainty and next dependency. Do not close the broad project goal
because a fly animates or a test count is large.
