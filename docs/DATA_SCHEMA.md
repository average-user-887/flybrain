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
| `type` | string | `"trial"`, `"telemetry_summary"` |
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
| `trial` | int | trial counter, **restarts at 1 on every daemon start** |
| `paradigm` | string | active paradigm id (`t-maze`, `heat-maze`, ...) |
| `step` | int | simulation step at which the milestone was recorded |
| `metric` | float | `performance_index` / `pi` / `learning_index` from the paradigm metrics, else `0.5` |
| `timestamp` | float | wall-clock time of the milestone |
| `run_id`, `instance_id`, `backend` | string | controller identity of the run that produced the trial |

Because `trial` restarts per session, the unique key of a trial is
`(session_id, trial)`. Within a session a trial number is written at most
once, even if the runner's in-memory ledger is truncated or re-read.

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
| `step` (top level and `timing.step`) | int | simulation steps since daemon start |
| `sim_time_s` | float | `step * integration_dt_s` |
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
| `applied_step`, `applied_sim_time_s` | int, float | step boundary at which it was applied |
| `latency_ms` | float | request receipt to application (a queued command: includes the wait for the running step) |
| `run_id` | string | daemon process run id |
| `paradigm` | string | active assay after the command |
| `identity` | object | identity **after** the command (see below) |

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

Stale packets: after a switch ack, a consumer rejects packets from the same
daemon process (`run_id == ack.identity.daemon_run_id`) whose
`identity.activation` is lower than the ack's, or equal with a different
`run_id`/`instance_id` (`neurofly_daemon.identity_rejection`, mirrored by
`identityRejection` in `web/app.js`). Higher activations (another dashboard
switched later) and a restarted daemon are accepted.

### Motor provenance (`motor`, `controller_fault`)

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
manifests live in the registry (`<output-dir>/registry/<assay>/<backend>/<instance_id>/manifest.json`).
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
restoring and continuing is bit-exact. Graph-run bookkeeping (curves, event
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
The session-wide `trial` stays monotonic across switches for recorder compatibility.
An unavailable scalar `metric` is null; it is never synthesized as 0.5.

## Reading in Python

```python
from learning_recorder import read_jsonl
trials = read_jsonl("outputs/learning/trials.jsonl")
by_session = {}
for t in trials:
    by_session.setdefault(t["session_id"], []).append(t["metric"])
```
