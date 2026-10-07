# NeuroFly learning-record schema (schema_version 1)

The daemon (`neurofly_daemon.py`) persists learning data as append-only
[JSON Lines](https://jsonlines.org/) under a data directory:

| Precedence | Source |
| --- | --- |
| 1 | `--data-dir PATH` |
| 2 | `NEUROFLY_DATA_DIR` |
| 3 | `<repository>/outputs/learning/` (git-ignored) |

`--no-record` disables recording entirely. Writing is done by
`learning_recorder.py`, a poller thread that reads the runner's in-memory
ledger under its lock once a second and appends anything new. It never
touches the simulation loop, and a recorder failure is logged rather than
propagated.

## Files

```
<data_dir>/
  session.json              manifest of the current/most recent daemon start
  sessions.jsonl            one manifest line per daemon start (append-only)
  trials.jsonl              one line per trial milestone (append-only)
  telemetry_summary.jsonl   one line per summary interval (append-only)
  trials.jsonl.<UTC stamp>  rotated segments, see "Rotation"
  telemetry_summary.jsonl.<UTC stamp>
```

Every line is one JSON object. Every record carries:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | string | `"trial"`, `"observation"`, `"telemetry_summary"` |
| `schema_version` | int | `1` |
| `session_id` | string | `<UTC YYYYMMDDTHHMMSS>-<pid>-<6 hex>`; identifies one daemon start |
| `recorded_at` | float | Unix seconds when the line was written (wall clock) |

Timestamps are Unix epoch seconds as floats. Numbers from NumPy are converted
to plain JSON numbers; arrays become lists.

## `session.json` / `sessions.jsonl`

Written once at daemon start. `session.json` is replaced atomically on the
next start; `sessions.jsonl` keeps every start.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version`, `session_id` | | as above |
| `started_at`, `started_at_iso` | float, string | daemon start time |
| `pid` | int | daemon process id |
| `python`, `platform` | string | interpreter and OS build (no hostname) |
| `prior_trial_lines_on_disk` | int | lines already in `trials.jsonl` when this session began |
| `paradigm` | string | paradigm requested at launch (`--paradigm`) |
| `sim_speed` | float | effective speed multiplier after clamping |
| `port` | int | HTTP port |
| `public` | bool | whether the public read-only mode was on |
| `backend` | string | controller backend chosen at launch (`--backend`) |
| `daemon_run_id` | string | the daemon process's run id (telemetry `run_id`) |
| `manifest` | object | full run manifest of the run active at start (see "Run manifest") |

The command line is deliberately **not** recorded (it could contain paths or
tokens). No hostname or IP address is recorded.

## `trials.jsonl` (`type: "trial"`)

One line per entry the runner appends to its `trial_history`, i.e. whenever a
paradigm reports `trial_complete` or the food-collection milestone fires
(see `ContinuousExperimentRunner._record_trial_milestone`). Fields beyond the
common header are copied verbatim from the runner:

| Field | Type | Meaning |
| --- | --- | --- |
| `trial` | int | session trial number: **restarts at 1 on every daemon start** and increases across assay switches |
| `assay_trial` | int | the assay's own trial number from its saved trial clock (see "Clocks"); continues across restarts |
| `trial_known` | bool | false when `assay_trial` is not a true ordinal because the state predates trial clocks |
| `brain_trial` | int | the assay brain's saved count of completed trials |
| `sim_seconds` | float | the trial's elapsed time when it ended, including time restored from a checkpoint |
| `paradigm` | string | active paradigm id (`t-maze`, `heat-maze`, ...) |
| `step` | int | daemon-session simulation step at which the milestone was recorded |
| `metric` | float | `performance_index` / `pi` / `learning_index` from the paradigm metrics, else `0.5` |
| `timestamp` | float | wall-clock time of the milestone |
| `run_id`, `instance_id`, `backend` | string | controller identity of the run that produced the trial |

Because `trial` restarts per session, the unique key of a trial is
`(session_id, trial)`. Within a session a trial number is written at most
once, even if the runner's in-memory ledger is truncated or re-read.

## `trials.jsonl` (`type: "observation"`)

`LearningRecorder.record_observation` is the strict journal API for
metric-contract/1.2 terminal envelopes. The recorder thread reaches it through
the runner-owned publication queue. It accepts the immutable envelope
itself, never a mutable `last_terminal` wrapper or a live provisional snapshot.
The full key is `(daemon_run_id, run_id, instance_id, segment_id,
presentation_id)`; integer trial numbers do not participate, so multiple
presentations in one trial remain distinct.

The row stores `observation_key` plus the complete validated envelope in
`observation`. Header timestamps and recorder-session fields are outside the
immutable payload comparison. A successful receipt includes the full key,
canonical payload SHA-256 and physical file/line/byte location. It is returned
only after the complete line, file and any creation/rotation directory entry
are synced. A same-key, same-payload retry re-scans and re-syncs the existing
line; a different payload conflicts. Partial or malformed ledger tails are
reported without truncation or repair.

The supported ownership model is one `LearningRecorder`, drained by one
`RecorderThread`, per data directory. The recorder takes a nonblocking,
OS-backed exclusive lock on the stable `.neurofly-recorder.lock` sidecar before
it opens writers or changes session/journal files, and holds it until `close()`.
Direct observation calls on that recorder share the trials-writer lock. A
second recorder instance or process for the same resolved directory fails
before session/journal mutation; a clean close or process exit releases the
kernel lock. Recorder objects inherited across a process fork are rejected.

## Live metric-contract/1.2 telemetry

Every live telemetry packet carries two separate objects:

- `observation` is a detached, validated provisional schema 1.2 envelope. Its
  identity includes the actual daemon run, manifest run, controller instance,
  activation, assay, backend and retained `brain_id`. `segment_id` and
  `segment_start_sim_s` bind its relative producer clocks to daemon simulation
  time. Live records have `final: false`, no completion receipt and no terminal
  pose.
- `observation_lifecycle` is mutable runtime status. It reports the current
  producer status, segment origin, durability availability and the requested
  CLI policy. In `--continuous` mode an explicit `--trial-seconds` remains
  visible here as requested but unused; the canonical producer config retains
  the contract's continuous policy. Its `phase` is `observing`, `hold`,
  `waiting_for_save`, `measurement_ended`, or `durable_transition_blocked`.
  A captured terminal summary exposes its full observation key, canonical
  payload digest, capture step and whether the exact durable receipt arrived.
  Hold ticks continue to advance the producer clock without changing the
  already frozen payload. Once the hold closes, `waiting_for_save` prevents a
  new presentation or segment until that terminal's exact receipt is durable.

`observation_publication` remains the separate persistence view. A pending
candidate is not `last_terminal`; that field appears only with the exact durable
journal receipt. Mutable lifecycle fields are never inserted into the immutable
observation payload.

Its durability view also reports the recorder's background-thread state,
progress age, and active write name/age. A recorder that exits after it was
started has a 2-second startup/death grace. A live poll loop gets its configured
poll interval plus 2 seconds to report progress while no simulator step or
command is in progress. A recorder write that has not returned for 30 seconds
is considered stuck. The watchdog treats these conditions as required-save
failures in scientific mode: the run becomes incomplete and halts, while the
queued frozen terminal and writer ownership remain intact. Recorders attached
only for synchronous/manual polling have not been started and are therefore
outside this background-thread watchdog. Each watchdog episode remains in a
bounded diagnostic history. A late write receipt does not clear its halt; an
explicit recovery rearms the watchdog only after the same recorder is alive,
inside its progress bound and no longer writing. Exploratory mode resolves its
episode after independently observed healthy progress and does not duplicate
the same active failure on every watchdog tick.

## `telemetry_summary.jsonl` (`type: "telemetry_summary"`)

One line every `--summary-interval` seconds (default 60) plus one final line
at shutdown. Built by `learning_recorder.summarise_runner` from the runner's
last telemetry packet:

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | float | when the summary was assembled |
| `uptime_sec` | float | seconds since daemon start |
| `paradigm` | string | active paradigm id |
| `step` | int | total simulation steps so far |
| `sim_speed` | float | current speed multiplier |
| `current_trial` | int | next trial number |
| `trials_completed` | int | `len(trial_history)` |
| `fly` | object | `x`, `y` (mm), `heading` (rad), `speed` (mm/s), `state` (behavioural state string) |
| `mb_weights_mean`, `mb_weights_std` | float or null | mushroom-body KC→MBON weight statistics from the telemetry packet |
| `learning_curve_tail` | list of float | last 10 trial metrics |
| `metrics` | object | the paradigm's `paradigm_metrics` dict at that instant (paradigm-specific keys) |
| `achieved_speed` | float or null | measured simulation speed (see `timing`) |
| `identity` | object | controller identity of the packet (see "Identity") |
| `motor_source`, `motor_assists_enabled`, `controller_fault` | | motor provenance at that instant (see "Motor provenance") |

## Live telemetry packet (`GET /api/stream`, `GET /api/telemetry`)

Not written to disk by the recorder, but exported by the dashboard (JSON/CSV)
and summarised above. Units: mm, s, rad, rad/s, mm/s unless stated.
All fields listed here are present on every packet.

### Timing (`timing`)

The integration step is fixed (`integration_dt_s` = 0.02 s) at every requested
speed; the wall-clock deadline only decides *when* the next step runs.

| Field | Type | Meaning |
| --- | --- | --- |
| `step` (top level and `timing.step`) | int | simulation steps since this daemon process started |
| `sim_time_s` | float | `step * integration_dt_s` |
| `sim_time_scope` (top level and `timing`) | string | always `"daemon_session"`: `step` and `sim_time_s` restart from zero with every daemon process; the retained brain, world and trial clock do not (see "Clocks") |
| `requested_speed` | float | speed the user asked for (0.1–100) |
| `achieved_speed` | float | measured simulated seconds per wall second over the last ~2 s; 0 while paused |
| `overloaded` | bool | achieved < 90 % of requested: this computer cannot keep up (no steps are skipped) |
| `steps_in_frame` | int | steps simulated since the previous published snapshot (decimation) |
| `snapshot_seq`, `publish_hz` | int, float | snapshot counter and target publication rate |
| `schedule_rebases`, `forgiven_wall_s`, `max_batch_hold_ms` | | scheduler diagnostics |
| `command_latency` | object | `count`, `p50_ms`, `p95_ms`, `max_ms` of recent command round trips |

SSE also sends `event: heartbeat` (`server_time`, `seq`) after 1 s without a new
snapshot and `event: stream` (`sent`, `decimated_snapshots`, `stream_hz`) once a
second. Delivery is latest-value-wins: intermediate snapshots are skipped, never
queued.

### Clocks (`clocks`; also in the ack, `/api/status` and checkpoint records)

Three clocks, each named for what it counts. Earlier packets and recordings do
not carry `clocks`; their values are kept as recorded.

| Field | Meaning |
| --- | --- |
| `session` | `scope` `"daemon_session"`, `daemon_run_id`, `step`, `elapsed_s`: this daemon process's step count and simulated time (the packet's `step`/`sim_time_s`) |
| `trial` | the active assay's trial clock, `neurofly.assay-trial-clock.v1`: `elapsed_s`, `current_trial`, `elapsed_known`, `trial_known`, `reason` |
| `graph` | graph runs only: `scope` `"retained_graph_instance"`, `instance_id`, `step` (the instance's saved step index), `elapsed_s` (its neural simulated time). Restored from the checkpoint with the brain; it is not the trial's elapsed time |

The trial clock is saved with the state it belongs to: in the same atomic,
versioned graph checkpoint as the neural and world state (`meta.trial_clock`),
or in the modular brain JSON (`trial_clock`). A restart restores it only from
that state; for a graph run that is the checkpoint the restore selected,
including an older version after a damaged newest one, never the graph-run
helper JSON. `trial_elapsed_s` and `trial` in the packet are its values.

State saved before trial clocks existed restores with `elapsed_known` and
`trial_known` false and `reason` `"checkpoint_predates_trial_clock"`; nothing
is inferred from brain, world or session counters. The values then count only
what was observed since that restore (`elapsed_s` is a lower bound) and are
saved as unknown again. A trial that starts while the daemon runs (a natural
end or `reset_trial`) makes `elapsed_known` true; an unknown trial ordinal stays
unknown. The dashboard shows Unknown for exactly these flags.

The observation window (metric-contract/1.2) still starts again with a new
segment after a restart, so a restored trial runs one more full window; its
`sim_seconds` includes the restored elapsed time.

### Path (`path`)

`[[step, x, y], ...]`: the measured position after each of the most recent
steps (up to 240) of the current trial segment, so a decimated display can draw
the path actually taken between frames. It is cleared at every segment boundary
(trial end, manual reset, experiment switch); positions are never connected
across segments. Display interpolation is not part of the data.

### Command acknowledgement (`POST /api/command` reply, `ack`)

| Field | Type | Meaning |
| --- | --- | --- |
| `action` | string | the command |
| `applied` | bool | whether it took effect |
| `applied_step`, `applied_sim_time_s` | int, float | daemon-session step boundary at which it was applied |
| `applied_sim_time_scope` | string | always `"daemon_session"` |
| `clocks` | object | the clocks after the command (see "Clocks") |
| `latency_ms` | float | request receipt to application (a queued command: includes the wait for the running step) |
| `run_id` | string | daemon process run id |
| `paradigm` | string | active assay after the command |
| `identity` | object | identity **after** the command (see below) |
| `halted_by_error` | string or null | the step error that still halts the simulation after this command, else null |

### Error halt (`error`, `halted`, `error_detail`; also in `/api/status`)

An exception inside a simulation step halts the run: nothing advances, and the
run is never stepped past a broken state. While halted, `/api/status` has
`"status": "error"` and `"halted": true` (`paused` stays as the user set it),
stream frames carry `error`, `halted: true` and `error_detail` (`message`,
`type`, `step`, `sim_time_s`, `paradigm`, `backend`, `instance_id`, `at`,
`recover`), `timing.achieved_speed` is 0, and the dashboard pill reads
`SIMULATION HALTED · ERROR`. Pausing, resuming or changing speed does not lift
the halt (their ack carries `halted_by_error`). A successful `switch_paradigm`
(re-selecting the same assay retries it) or a `switch_backend` that changes the
backend rebuilds the controller and world and lifts it; that reply carries
`cleared_error`, and `/api/status` `cleared_errors` keeps the last 16 lifted
halts. A failed switch leaves the halt in place.

For `switch_paradigm` the reply is sent only after the target's brain snapshot
(graph backends: `ExperimentRegistry.activate`) and world snapshot
(`Arena.restore_world`) are restored. The reply's `identity` is the newly
active run.

Commands are applied only at step boundaries. When a step is still running
after `command_reply_wait_s` (1 s; on a slow CPU one step can take seconds), the
reply is `{"status": "queued", "applied": false, "command_id": ...}` instead of
waiting. The command stays queued and is never dropped; its full result (the
reply it would have had, plus `command_id`) appears in the `command_acks` list
(latest 8) of every stream frame after it is applied. The dashboard waits for
that acknowledgement before it treats the command as done.

SSE `heartbeat` events carry `step_in_progress_s` (wall seconds the current step
has run, 0 between steps) and `last_step_wall_s`; `timing` in frames and
`/api/status` carries the same two fields.

### Liveness and saving (`status`, `liveness`, `persistence`, `recording_error`)

Added after audit F (docs/receipts/audit-20261005/F-robustness.md), where the
simulation thread died on a full disk while `/api/status` still said `online`.
`/api/status`, every stream frame and every SSE `heartbeat` (which also carries
`step`, `paused`, `halted` and `error`) report whether the simulation advances and
whether it saves:

After an unexpected simulation-loop exit, status and heartbeats additionally carry
`observation_validity_update`: a detached `neurofly-observation-validity-update/1`
notice with full `identity` (including `brain_id`), `segment_id`,
`validity` (`invalidated`, or `exploratory_degraded` in exploratory mode), and
`reason: unexpected_loop_exit`. No new engine observation or pose is sampled.
Heartbeats carry `identity`, `brain_id` and `segment_id` from a completed packet
or this notice. The dashboard rejects qualified heartbeats from another owner
or segment, including one superseded by an acknowledged switch. A matching
notice downgrades the live display and labels its values as the last pre-fault
frame; it does not alter recorded frames, metric values or saved terminals.
A verified rebuild uses normal new-owner frames; prior incomplete-run evidence remains.

| field | meaning |
|---|---|
| `status` | `error` (halted -- including a run stopped by a failed required save --, stalled or the simulation thread is dead), else `degraded` (exploratory mode not saving, or a diagnostic log failing), else `online`. A paused run is `online` with `paused: true` |
| `mode` | `scientific` (default) or `exploratory` (`--exploratory`) |
| `result_validity` | `run_id`, `state` (`valid_so_far` or `incomplete`), `incidents` (append-only: `reason` such as `required_save_failed`, `not_saved_exploratory_gap`, `compute_failure_while_saving`, `halted_step`, `halted_device`, `interrupted_unclean_shutdown`; `channel`, `failure_class`, `error`, `errno`, `step`, `at`, `gap_open`, `recovered_at`, `recovered_step`), `other_runs_incomplete`, `unwritten_records`. Also kept in `<output-dir>/run_validity.jsonl` (append-only; it also records `session_start`, `activate` and a clean `session_end`). On activation a run reloads its incidents from that ledger (malformed lines are skipped and counted), and a run that was active in a session without a clean end gets an `interrupted_unclean_shutdown` incident, so a crash never returns a run to valid |
| `liveness.state` | `advancing`, `paused`, `halted`, `slow` (one step has run > 2 s), `stalled` (no step for longer than `stall_threshold_s` while it should step, or one step longer than `step_hard_limit_s`), `dead` (the simulation thread has ended), `not_started`, `stopped` |
| `liveness.step`, `last_advance_age_s` | the step counter and the wall seconds since it last increased |
| `liveness.sim_thread_alive` | whether the simulation thread runs |
| `liveness.stall_threshold_s` | `max(10 s, 20 x dt / speed, 3 x last_step_wall_s)` |
| `persistence.state` | `ok`, `failing` (a write raised) or `disk_low` (free space below 2 x the last checkpoint + 512 MB, so the checkpoint was skipped before the disk filled) |
| `persistence.failing` | per channel: required `checkpoint`, `brain_save`, `trial_ledger`, `events_ledger`, `recording`, `learning_records`, `provenance`; diagnostic `telemetry_summary`, `halt_log`. Each: `error`, `errno`, `failure_class` (`persistence` / `compute` / `software`), `required`, `path`, `since`, `failures`, `backoff_s`, `next_retry_at` |
| `persistence.reason`, `summary`, `since` | `"disk full"` when any channel failed with ENOSPC or is `disk_low` |
| `persistence.last_ok_save_at`, `last_ok_save_age_s` | the last checkpoint that was written |
| `recording_error` | a `--record` / `record_start` capture that failed: the recording stopped and is INVALID (its `.partial` file is kept); the run stops (scientific mode) or continues (exploratory) |

After a requested raw recording fails in scientific mode, selecting or resetting
an assay cannot remove the recording obligation or resume without a writer.
Repairing storage and writing a healthy checkpoint alone is insufficient. Use
`record_start` with a **new name** to create a replacement that successfully
captures its initial frame, then select the assay to recover. Alternatively stop
and restart the daemon with `--record NEW_NAME` and the **same saved-brain output
directory**. The failed filename/prefix is preserved and cannot be reused. The
failed run remains incomplete; a fresh run has its own validity history.
`recording_error.run_id` identifies that failed capture's run, and its message
describes artifact/history facts. Current execution state comes from `halted`
and `liveness`, not a frozen recording message. Unrecorded scientific runs do not
require raw recording; exploratory mode retains visible unsaved continuation.

`/api/status` also has `loop_failure` (type, message, phase and traceback tail of
whatever ended the simulation thread) and `thread_failures` (uncaught exceptions
in any `NeuroFly-*` thread, recorded by a `threading.excepthook` backstop).
With status `dead`, `halted` is true and `error` starts with
`simulation thread stopped:`; `POST /api/command` then refuses everything except
`switch_paradigm` / `switch_backend`, whose success restarts the thread (the reply
has `loop_restarted: true`). `timing.achieved_speed` is computed when it is read,
from the steps of the last few seconds, so it drops to 0 when nothing advances.

An exception anywhere in the simulation loop (not only inside `arena.step`) is an
honest halt whose `error_detail.phase` names where it happened (`step`, `publish`
after 3 consecutive failed frames, `trial bookkeeping`, `command`, `persistence`,
`device`) and whose `error_detail.failure_class` is `persistence` (a storage error
on a required save: the run stops, scientific mode), `compute` (CUDA/CuPy, memory or
NaN/inf state: always halts) or `software` (any other error: halts). In
`--exploratory` mode a storage error on a required save does not halt: the run keeps
stepping with `status: "degraded"`, the checkpoint is retried after 30 s, doubling to
at most 10 minutes, and the gap is an incident. A save-failure halt is lifted only by a
switch whose recovery checkpoint succeeds. These health fields are wall-clock state and
are not part of `.nfrec` recordings; `neurofly record` returns `status` `complete`,
`incomplete` or `invalid` and exits non-zero unless complete.

### Identity (`identity`, also in `/api/status`, the ack and exports)

Compact form of the run manifest (`provenance.RunManifest.identity()`):

| Field | Type | Meaning |
| --- | --- | --- |
| `run_id` | string | the controller run (modular: new per activation; graph: per registry instance) |
| `instance_id` | string | modular: the experiment brain id; graph: the registry instance id |
| `assay` | string | assay of that run |
| `backend` | string | `modular`, `connectome-fixed`, `connectome-plastic`, `connectome-with-trained-readout` |
| `controller_version` | string | versioned controller implementation |
| `graph_sha256`, `neuron_map_sha256`, `io_map_sha256` | string or null | pinned graph identity; null for modular |
| `synthetic` | bool | a synthetic test graph (never a scientific result) |
| `test_mode` | bool | an explicit test option was used |
| `label` | string | human-readable description; conspicuous for synthetic/non-scientific runs |
| `activation` | int | switch counter of this daemon process; increases on every activation |
| `daemon_run_id` | string | equals the packet's top-level `run_id` |

Graph run manifests also carry `graph_io` (`version`, `sha256`). A linked run
created with `--continue-io-state` adds `lineage`: the parent run and instance,
the exact validated parent checkpoint version/step/SHA-256, prior and new I/O
identities, explicit continuation choice, inherited validity and interventions
with source-run attribution, and the method-change interventions. Unknown legacy
validity is recorded as incomplete evidence; it is never presented as pristine.

The graph registry v2 index adds `current_instances`, keyed by
`<assay>|<backend>`. Every historical entry remains in `instances`; selection
uses the explicit current ID. Registry v1 is accepted for migration, then
upgraded atomically on the first compatible selection or linked migration.
Readers that only understand registry v1 must reject v2 instead of selecting
the first historical entry.

If an existing graph run has a missing or different I/O identity, startup
refuses before migration writes and prints the deliberate continuation option:
`--continue-io-state`. That option validates a retained checkpoint and publishes
a new child store/run. It never relabels or overwrites the parent.

Stale packets: after a switch ack, a consumer rejects packets from the same
daemon process (`run_id == ack.identity.daemon_run_id`) whose
`identity.activation` is lower than the ack's, or equal with a different
`run_id`/`instance_id` (`neurofly_daemon.identity_rejection`, mirrored by
`identityRejection` in `web/app.js`). Higher activations (another dashboard
switched later) and a restarted daemon are accepted.

### Motor provenance (`motor`, `controller_fault`)

Graph controller replies identify their effective input/output method with
`graph_io.version` and `graph_io.sha256`. `input_stage` lists each enabled probe,
its biological entry stage, injected current, resolved cell count and observed
spiking for the step. `decoder` states the active DN-to-motor equations.

`dn_rates` uses JSON `null` when a requested DN pool has no resolved neurons;
`dn_unavailable` gives the corresponding reason. The decoder still receives a
numeric 0 Hz for that absent pool so its numerical behavior remains defined.

For graph runs, `connectome.raw_motor_command` is the direct decoder output.
`motor.record.raw_motor_command` repeats it beside `applied_motor_command`, the
command after arena primitives and wall steering, and
`engineered_motor_primitive`. A DNp01 graph spike decodes to a raw 35 mm/s
`ESCAPE`; the arena records the bounded escape primitive that applies 3.5 mm/s
for 0.2 s. When no primitive is active, `engineered_motor_primitive` is `null`.

| Field | Type | Meaning |
| --- | --- | --- |
| `controller_backend` | string | backend of the arena's controller |
| `motor_assists` | object | `wall_avoidance_reflex`, `contact_turn`: engineered assists enabled (bool each) |
| `motor_assists_enabled` | bool | any assist enabled. Default ON for `modular` and `bridge-surrogate`, OFF for graph backends and the RPC hybrid |
| `motor_source` | string | who produced this step's motor command (below) |
| `motor_source_normal` | bool | `motor_source` is one of `modular`, `surrogate`, `graph-rpc`, `graph` |
| `motor_halted` | bool | no motor command: speed and yaw are exactly zero, no assist acts |
| `controller_fault` | string or null | controller fault (also top-level `controller_fault`) |
| `assist_totals` | object | cumulative counts: steps near walls/in contact, reflex and contact-turn use, solver and failsafe corrections |
| `record` | object | this step: `controller_yaw` (rad/s), `controller_speed` (mm/s), `wall_reflex_yaw`, `contact_turn_rad`, `attempted_mm`, `realized_mm`, `solver_correction_mm`, `failsafe_correction_mm`, `overlap_correction_mm`, `wall_gap_mm`, `near_wall`, `in_contact`, `tethered`, `halted`, `contacts` |

Abnormal `motor_source` values: `halted-rpc-fault` (RPC graph unreachable under
the default `on_rpc_fault='halt'`), `surrogate-fallback-TEST` (explicit test
option: the hand-built surrogate drives), `graph-unmapped-io` (graph backend,
no verified sensory encoder/motor decoder for the assay in the live arena, so
no motor command), `halted-no-instance`. The dashboard shows a banner for any
of them, for `synthetic`/`test_mode` runs and for a `controller_fault`.

## Run manifest (`GET /api/manifest`)

`{"identity": ..., "manifest": ...}`. The manifest is
`provenance.RunManifest.to_dict()` (schema `neurofly.run-manifest.v1`):
backend, assay, instance and run ids, seed, graph identity, dynamics with
units, learned parameter locations, source revision and file hashes,
intervention schedule and events (activations, checkpoints, restores). Modular
manifests are written to `<output-dir>/manifests/<run_id>.json`; graph
manifests live in the registry (`<output-dir>/registry-v3/<assay>/<backend>/<instance_id>/manifest.json`
under the default v3 dynamics; v1 brains stay in `<output-dir>/registry/`).
The dashboard's JSON export embeds `identity`, `manifest` and `motor`; its CSV
rows carry `controller_run_id`, `instance_id`, `backend`, `synthetic`,
`motor_source`, `assists` and `controller_fault`.

## Graph backends and world snapshots

`--backend connectome-fixed|connectome-plastic|connectome-with-trained-readout`
loads one verified graph (`--graph-dir` or `NEUROFLY_GRAPH_DIR=/path/to/malecns_v1`;
missing or mismatching graph = startup error) and one experiment registry under
`<output-dir>/registry`. Every checkpoint (periodic, manual, on switch and on
shutdown) is a registry checkpoint whose `meta.world_state` holds
`Arena.snapshot_world()`: format `neurofly.world-state.v1`, a readable
`summary` (fly pose and velocities, RNG states, assists) and the complete
encoded `state` (arrays as base64 of raw bytes, large integers as strings), so
restoring and continuing is bit-exact. `meta.trial_clock` holds the assay's
trial clock (see "Clocks"); it is runner bookkeeping, not world state, and is
absent in checkpoints saved before trial clocks. Graph-run bookkeeping (curves, event
logs) is kept under `<output-dir>/graph-bookkeeping/<backend>/`, never in the
modular brain files. `--test-synthetic-graph` runs a graph backend on a small
synthetic graph for tests; such runs are labelled synthetic everywhere.

## Rotation and durability

- Each `append` flushes and `fsync`s before returning, so a line is on disk
  once the poller has run (bounded by the 1 s poll interval).
- When a file would exceed 64 MiB it is renamed to `<name>.<UTC stamp>` and a
  fresh file is started. Nothing is ever truncated, rewritten or deleted;
  rotated segments sort chronologically by name.
- To read everything: concatenate the rotated segments in name order followed
  by the live file, e.g. `cat trials.jsonl.* trials.jsonl | jq .`.

## Relation to the older checkpoint files

`outputs/checkpoints/checkpoint_<paradigm>_<tag>_<epoch>.json` are the
run-summary snapshots. New snapshots also point to the active brain checkpoint;
they are no longer deleted by the daemon. Actual MB weights, traces and CX state
are atomically saved per experiment under `<output-dir>/brains/<paradigm>.json`.
The adjacent `<paradigm>.events.jsonl` keeps teaching, probes and trial records.
See `LEARNING_OBSERVATORY.md` for the learning-state schema and its scope.

Global trial records now additionally include `brain_id` and `brain_trial`.
The session-wide `trial` stays monotonic across switches for recorder compatibility;
`assay_trial` is the assay's own trial number from its saved trial clock.
An unavailable scalar `metric` is null; it is never synthesized as 0.5.

## External stimulus scene phase

Optomotor `scene.drum_angle_deg` is declared external-stimulus kinematics in
degrees: a newly created stimulus or explicit trial reset starts at zero. Arena
steps advance it by configured velocity × simulation dt, modulo 360.
Reversal and contrast changes
preserve known phase; pause holds it. This phase does not feed the neural encoder
or change drive, wiring, learning or LIF dynamics. The existing world snapshot
retains phase. Historical worlds without it restore an unknown phase (`null`),
which remains unavailable until an explicit new presentation or trial reset.
Remote 2D/3D cues require a finite scene angle; they do not reconstruct it from
current velocity and elapsed time.

## Reading in Python

```python
from learning_recorder import read_jsonl
trials = read_jsonl("outputs/learning/trials.jsonl")
by_session = {}
for t in trials:
    by_session.setdefault(t["session_id"], []).append(t["metric"])
```
