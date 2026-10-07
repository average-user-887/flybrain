# Run recordings (`.nfrec`) and 1x replay

The full-accuracy simulation runs slower than real time (about 0.4x on the Ryzen
GPU, far less on a CPU). A run is therefore recorded once and played back at 1x
in the dashboard, with the same panels live mode uses. The same file is the
unit of open data: it carries the provenance needed to cite and re-run it.

## Making a recording

Headless, as fast as the machine computes (the usual way):

```bash
neurofly record --paradigm t-maze --seconds 60 --out recordings/t-maze-naive
neurofly record --paradigm optomotor --backend connectome-fixed --seconds 30 \
    --out recordings/optomotor-v3 --raster io
# step-exact interventions: [{"step": 1500, "cmd": {"action": "reset_trial"}}]
neurofly record --paradigm t-maze --seconds 60 --schedule inputs.json --out recordings/t-maze-reset
```

Graph backends use the v3 LIF dynamics by default, like the daemon;
`--dynamics v1` (or `NEUROFLY_LIF_DYNAMICS`) selects another version, and the
header records it as `provenance.lif_dynamics`. `--state-dir` continues a saved brain; without it every recording starts from the
paradigm's naive brain in a temporary directory, so two identical commands give
the same file.

From the running daemon: start it with `--record NAME` (records from the first
step), or send `{"action": "record_start", "name": "NAME"}` and
`{"action": "record_stop"}` to `POST /api/command`. Files land in
`<output-dir>/recordings/`. `GET /api/recordings` lists them and
`GET /api/recordings/NAME.nfrec` serves one. A recording covers one graph, so
`switch_backend` is refused while recording; paradigm switches are allowed and
show up in the frames.

## Playing it back

In the dashboard, **Replay** opens a chooser: the connected daemon's recordings,
or any local `.nfrec` file. `?replay=<url>` opens one directly. Playback follows
the recorded simulated time: 1x is the fly's real time, whatever the run cost to
compute. The replay bar has play/pause (the navbar Pause button does the same),
seek, speed (0.5x to 10x) and **Exit replay**, which reconnects to the live
daemon. Frames go through `DaemonBridgeClient.handleDaemonPacket`, the same path
as SSE frames, so every live panel shows recorded data. The **[4] Brain Activity**
panel shows per-region rates live and in replay, and the spike raster in replay.

## File layout

One gzip stream (`mtime=0`, empty file name) of UTF-8 JSON lines, keys sorted,
compact separators, non-finite numbers written as `null`.

| `k` | When | Content |
|---|---|---|
| `header` | first line | `format` = `neurofly-run-recording`, `version` = 2 (the Python reader also accepts legacy 1), `record_every`, `start_step`, `frame_dt_s`, `provenance`, `channels`, `label` |
| `f` | one per recorded step (frame 0 = the start state) | the telemetry packet (docs/DATA_SCHEMA.md) minus wall-clock and random-ID fields, plus `i`, `activity`, `spikes` |
| `e` | when an input was applied | `kind` (`command`, `simulation_error`), `step`, `cmd` |
| `end` | last line | `frames`, `events`, `first_step`, `last_step`, `frames_sha256` (SHA-256 of all frame lines, newline included) |

An `e` event at step `n` was applied after frame `n` and before step `n + 1`.
Only commands that were accepted and can change the simulation are logged
(`set_speed`, `set_paused`, `probe_brain` and checkpoint or recording control are not).

### `provenance`

`backend`, `assay`, `lif_dynamics` (graph backends; `null` for modular), `brain_backend` (`cpu` or `cuda` for graph backends; `null` for modular), `seed`, `controller_version`, `label`, `synthetic`,
`test_mode`, `graph` (graph, neuron map and IO map SHA-256, neuron and edge
counts; host paths removed), `dynamics` (the run manifest's model description),
`code` (git commit, dirty flag, SHA-256 of the backend's source files),
`code_scope`, `writer` (both below),
`software` (Python and NumPy versions), `params` (`dt_s`, `graph_step_ms`,
`trial_length_s`, `continuous`, `motor_assists`), `initial_state` (see below)
and `inputs` (the step schedule given to `neurofly record`).

### Code origin and executing writer

`code` is the **origin** of the run: the code identity stored in the run manifest
when the run, or the saved instance it continues, was created. A recording that
continues saved state therefore shows the parent's code here, even when newer code
wrote the file; it is never rewritten. `code_scope` says which it is:
`run_manifest_origin`, or `running_runner_source` when the runner had no manifest
source. Files written before October 2026 have no `code_scope`; their `code` has the
same origin meaning.

`writer` (added October 2026) identifies the code that wrote **this file**, hashed
when the header is written:

- `role`: `executing_writer`.
- `loaded_module_sha256`: SHA-256 of the files the interpreter actually imported
  for the recorder, daemon runner, registry, provenance, brains, arena and graph
  engine (modules that were not loaded are absent).
- `kind` `installed_distribution` (the imported code is the installed `neurofly`
  package): `distribution`, `version`, `files` (count), `files_sha256` (SHA-256 of
  the sorted JSON map of every RECORD-listed file inside site-packages, hashed from
  disk, without pip's per-install `INSTALLER`, `REQUESTED`, `direct_url.json`,
  `RECORD`), `ident` (`name-version+files:files_sha256`, independent of the venv),
  and `record_mismatches` (files whose bytes differ from RECORD; empty when intact).
  Git is not needed.
- `kind` `source_tree` (running from a checkout): `version`, `commit` and `dirty`
  from git (null without git).

### `provenance.initial_state`

What the recording's first frame starts from:

- `restored`: true when the brain was loaded from saved state instead of starting
  naive. Graph backends: the registry restored the graph instance from a checkpoint
  in this process (it then logs a `restore` event in the instance's `events.jsonl`).
  Modular controller: the brain's saved file was loaded. For a daemon recording
  started later in a run (`record_start`) it describes the process's starting
  state, not the recording's first step.
- `restore_source` (added October 2026): `null` when `restored` is false; otherwise
  `{"kind": "graph_checkpoint", "checkpoint_version", "step_index", "fallback"}`
  (the checkpoint version and step that were restored; `fallback` true when the
  newest checkpoint could not be used and an older one was) or
  `{"kind": "modular_brain_file"}`. The instance itself is identified in the frames
  (`identity.instance_id`, recording-local alias).
- `graph_step_index` (graph backends only): the graph instance's step index at the
  first frame. It can be larger than `restore_source.step_index` when the
  recording starts after the restore.
- `brain_steps`: the runner's brain step count. On graph backends this is runner
  bookkeeping (`graph-bookkeeping/`), which a restored instance may not have; use
  `graph_step_index` for the neural state.

Files written before October 2026 have no `restore_source`, and on graph backends
their `restored` reflects only whether the runner's graph-bookkeeping file was
found, so it can be false for a restored instance; there, judge restoration by
`graph_step_index` and the instance's `restore` event. Readers ignore unknown
header fields, so the format version is unchanged (2).

### `channels`

Where each required channel lives in a frame:

- **body pose**: `fly`, `body_position_mm`, `body_quaternion_wxyz`, `joint_angles_rad`, `leg_contacts`, `biomechanics`
- **stimulus**: `stimuli`, `sensory`, `scene`, `assay_state`, `live_assay`
- **decoded motor channels**: `dn_rates`, `descending`, `motor`, `motor_drives`
- **per-region activity**: `activity`, a list aligned with `channels.regions.names`.
  Graph backends: mean spike rate per neuron (Hz) over the graph step that ended at
  the frame. The real MaleCNS graph is grouped by the annotated `superclass`
  (neuprint ROI/neuropil tables are not part of the downloaded data; the format
  takes any partition, identified by `membership_sha256`). The synthetic test graph
  is grouped by its IO channels. The modular controller has no neurons to group:
  it records KC mean, PAM, PPL1 and MB valence in model units.
- **spike raster** (optional): `spikes`, a flat `[slot, count, slot, count, ...]`
  list of non-zero spike counts of the neurons in `channels.raster.neurons`.
  `--raster io` (default) keeps the annotated IO neurons (DN channels, sensory
  inputs, EPG), `all` keeps every neuron, `none` turns it off. Graph backends only.

`null` means "not measured for this frame" (for example before the first graph
step), never zero.

### Determinism

The same code, seed, parameters, start state and inputs give a byte-identical
file (`tests/test_run_recording.py`). Nondeterministic source identities and operational wall-clock data live outside
the file. Version 2 declares `header.projection.identity_namespace` as
`recording-local/1`; frames declare `recording_context.mode = "replay"` and
`original_durable_evidence_verified = false`.

- Run/manifest IDs become `r0`, daemon IDs `d0`, brain/instance IDs share the
  `b0` namespace, segment IDs become `s0`, config IDs `c0`, and control IDs `q0`,
  in deterministic first appearance order. Equal brain/instance IDs stay equal.
  Derived `segment:<UUID>` and `<UUID>:<presentation-index>` references become
  `segment:s0` and `s0:<index>` everywhere, including nested observations,
  lifecycle status, publication keys and overrides. Missing/null/invalid values
  remain unchanged; aliases do not repair invalid source identities.
- All scientific metric records, evidence, lifecycle state, invalidity,
  simulated/relative timing, provenance and brain summary fields remain in the
  recording. Unknown science fields are retained. `null`, zero, false and
  unavailable measurements remain distinct. Only known operational clock leaves
  in recorder/watchdog and brain-save/history metadata are replaced by null.
  An arbitrary scientific `timestamp` attribute stays scientific data.
- Top-level wall-clock/viewer/health fields (`timestamp`, `timing`, `path`, speed
  and operational status) are preserved as reversible operational patches in
  the sidecar. Unique original IDs are stored once in its inverse crosswalk.
  `source_telemetry_from_frame()` reconstructs the privacy-redacted source
  packet using the frame plus this matching sidecar. Operational paths are
  interned once, and per-frame rows store only changed fields or removed patches.
  Source packets are never
  mutated by recording.
- Original terminal observation payload and receipt/hash/key are stored together
  once per distinct wrapper in sidecar `projection.source_evidence`. The frame
  uses `recording_payload_sha256` or `recording_payload_ref`, and a clearly
  unverified `recording_source_evidence` reference. It never substitutes the new
  digest into an original durable receipt. Original lifecycle durability is
  historical source state, not certification of replay bytes.
- If existing privacy redaction changes the original source wrapper, the public
  sidecar marks it `source_payload_redacted` and unverified. Exact original proof
  is preserved locally in `.source-evidence-private/NAME.nfrec.source.json`
  (directory mode 0700, file 0600), tied to the recording SHA-256. This private
  archive is excluded from recording listings and the `.nfrec` download API;
  `public_recording_artifacts()` exports only the public `.nfrec`, `.nfrec.json`
  and `.nfrec.done` files, never
  recursively bundle the private directory. The recording is independently
  replayable without the private archive. Redacted public evidence cannot verify
  an original payload digest; an original receipt's transport consistency alone
  is not a new durability proof.

The completion protocol binds the public sidecar to the exact `.nfrec` digest.
Durable command event bytes, command SHA-256, ownership checks, file-descriptor
fsync barrier and applied-ACK receipt validation remain unchanged (C1/C2).

Version 2 requires replay UI integration: accept both file versions, preserve
recording-local top and nested observation identities consistently (including
brain ID), label typed metric observations as replay, and keep the saved-durable
panel unavailable unless original evidence is separately verified. The private
CARD15B replay parser accepts versions 1 and 2; its deployment and running-browser
verification are separate gates. This writer repair does not claim browser acceptance.

### Completion (which files count as finished)

A recording is written to a hidden `.NAME.nfrec.partial`. Finishing it writes the
end record, closes and fsyncs the file, writes the sidecar `NAME.nfrec.json`
(with `"completion_protocol": 2`), renames the file to `NAME.nfrec`, fsyncs the
directory, and only then writes the completion record `NAME.nfrec.done` (the
file's SHA-256 and the sidecar's). Listings, the `/api/recordings/NAME` download and
`read_recording()` accept a current-protocol recording only when that record exists
and matches both files. A failure before that completion record is published leaves
an artifact those readers reject. If the final directory fsync fails after publication
and cleanup cannot withdraw the record, however, the fully formed artifact may remain
readable/listable. Its durability acknowledgement and scientific save receipt are
INVALID, the run is marked incomplete, and a required scientific run halts. Recordings
from before the protocol (sidecar without `completion_protocol`) are still accepted.

A recording made on another machine can still differ in the last bits of float
math (a different NumPy, numba or GPU); the header's `software` and `code`
fields say what produced it.

### Size

Version 2 retains typed observations and full science summaries, while the
sidecar adds inverse IDs and operational patches. Original terminal proof is
deduplicated; complete telemetry is not duplicated each frame. Size depends on
observation/evidence channels and should be measured for the chosen assay.

A bounded modular T-maze check (120 steps, 121 frames) measured version 2
`.nfrec` at 45,755 bytes versus legacy 44,871 bytes (+2.0%). The reversible
sidecar was 178,437 bytes versus legacy metadata-only 1,240 bytes; preservation
of operational chronology is an added cost, with sparse deltas reducing an
initial 1.07 MB projection. One local run took 1.55 seconds versus 0.52 seconds;
this is a small-fixture format-overhead observation, not a production throughput
benchmark. The writer now appends every operational delta and distinct public/private
proof to hidden `.NAME.nfrec.projection.partial/` JSONL scratch (0700 directory,
0600 files), with a disk-backed digest index. A bounded eight-entry detached
content cache skips repeated original-wrapper and projected-payload hashing;
typed content equality distinguishes false, zero and null, and changed payloads
cannot reuse an old declared digest. Only the terminal subtree is re-normalized,
not the entire telemetry for each receipt. Last-frame patches stay in RAM;
per-frame/different-proof rows do not. Identity crosswalk/path dictionaries and
baseline manifest IDs still grow with distinct transitions/schema; this is not a
claim of constant memory for arbitrarily expanding telemetry.

Final public sidecar and private proof retain the existing version-2 JSON schema,
assembled incrementally from scratch. The recording and sidecar digests stream
in 1 MiB blocks. There is no recording version bump or retained-recording
migration; version-1 and earlier version-2 recordings remain readable. The reader
and browser still load complete recordings/sidecars, an independent resource gate.

Scratch rows have incremental counts/digests checked during assembly, so even
valid JSON truncation or mutation refuses completion. Scratch has explicit
active/complete/failed status, row counts, and completion
binding to final recording/sidecar digests. The `.done` protocol remains the sole
public certification. A capture/finalization evidence failure revokes the writer
and preserves incomplete rows; no recovery silently skips a row. If disk failure
also prevents the failure marker, missing completion remains authoritative.
Abort cleanup closes only existing recording handles; append/index operations
leave no extra long-lived file handles. Successful scratch is retained as local
writer evidence, duplicating final sidecar/proof rows on disk. The filesystem
must enforce private POSIX modes and SQLite locking; unsupported permissions
refuse initialization rather than exposing exact proof. Public exports
remain the exact nfrec/json/done allowlist. Never recursively export scratch or
`.source-evidence-private`. Long-run disk usage, expanding telemetry metadata,
reader/browser memory and actual engine throughput remain separate gates.

Legacy frames were the full telemetry packet, uncompressed ~6 KB, about 7 to 47 KB per
simulated second after gzip at `--record-every 1`: 10 KB/s for the modular T-maze,
47 KB/s for the modular open arena (120 KC rates per frame), 7 KB/s for the
synthetic graph. `--record-every N` divides that by about N; replay then draws
one frame per N steps.

The writer authenticates each scratch digest-to-evidence reference with a
per-writer keyed MAC before reuse; altering SQLite cannot select an absent or
other proof after cache eviction. JSONL count/digest checks still bind the rows.
The ephemeral key is not exported, and interrupted writers are not resumable.
Finalizers serialize, while abort revokes immediately under a memory-only gate.
No filesystem work occurs in that gate. All abort cleanup, status writes and
completion withdrawal run asynchronously, including when `defer_cleanup` is
omitted; callers may wait on the cleanup thread explicitly outside runner locks.
The finalizer checks ownership before and after completion I/O and acknowledges
logical completion only under a final memory-only check after required fsyncs.
Pending/revoked writers expose `recording_writer_invalid_reason(path)` for the
same-process certifier; they must never certify while an in-flight syscall stalls.
Cleanup attempts to withdraw `.done`, hide the final file and mark scratch failed.
A syscall already in progress cannot be canceled. An external reader or process
crash before cleanup may observe apparently complete but unacknowledged disk
history. This is an explicit ambiguity, not cross-process atomic revocation.
Failed withdrawal retains a small same-process revoked-path tombstone; long-run
error accumulation and disk cleanup remain separate operational limits.

### Standalone scheduled command ownership

`ContinuousExperimentRunner` with an explicit `output_dir` may provision a locally
owned durable observation writer when `schedule_command` is first used. Initialization
is outside the simulation lock, and its private directory is
`output_dir/scheduled-records/<daemon-run-id>`. Set
`standalone_scheduled_records=False` to opt out. `run_daemon` always disables this
automatic ownership: it keeps its configured attached writer, including the explicit
`--no-record` refusal policy. An attached writer is never replaced or closed by the
standalone owner.

Scheduled commands use the same lifecycle transaction and final ACK barrier as live
commands. No successor tick or second same-step command runs while its evidence is
pending. Offline `record_run` releases the simulation lock during persistence waits.
Initialization failure leaves no scheduled entry; rejected inputs halt the schedule.
Where the source remains sound, rejection is durably recorded as the explicit
`scheduled_command_refused` evidence action with the requested command and
`applied:false`; it never denotes an applied biological command. Locally owned writer
cleanup runs off the simulation lock. A stalled cleanup makes bounded `stop()` return
false while cleanup retains ownership, rather than silently reporting a clean stop.

The three recorder liveness ages (`start_age_s`, `last_progress_age_s`,
`active_write_age_s`) under both observation durability paths are operational
wall-clock evidence. Frames preserve their keys with null projected values; reversible
sidecar deltas retain the exact originals. Recorder durability/status, false values,
unavailable reasons, and scientific timing remain in the frame.
Only the exact root `observation_lifecycle.durability.recorder` and
`observation_publication.durability.recorder` age paths are projected. Identically
named nested scientific provenance remains canonical frame content. Scheduled input
shape, integer step, stop state and past-step admissibility are checked before writer
creation and rechecked afterward; a race rejection closes only its unused newly
owned writer off-lock while retaining its scratch evidence.
