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
pre-existing rolling checkpoints (last 50 kept). They remain unchanged and are
a convenience snapshot, not the durable record. The JSONL files above are the
source of truth for a run's history.

## Reading in Python

```python
from learning_recorder import read_jsonl
trials = read_jsonl("outputs/learning/trials.jsonl")
by_session = {}
for t in trials:
    by_session.setdefault(t["session_id"], []).append(t["metric"])
```
