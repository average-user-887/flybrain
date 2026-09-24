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
| `header` | first line | `format` = `neurofly-run-recording`, `version` = 1, `record_every`, `start_step`, `frame_dt_s`, `provenance`, `channels`, `label` |
| `f` | one per recorded step (frame 0 = the start state) | the telemetry packet (docs/DATA_SCHEMA.md) minus wall-clock and random-ID fields, plus `i`, `activity`, `spikes` |
| `e` | when an input was applied | `kind` (`command`, `simulation_error`), `step`, `cmd` |
| `end` | last line | `frames`, `events`, `first_step`, `last_step`, `frames_sha256` (SHA-256 of all frame lines, newline included) |

An `e` event at step `n` was applied after frame `n` and before step `n + 1`.
Only commands that were accepted and can change the simulation are logged
(`set_speed`, `set_paused`, `probe_brain` and checkpoint or recording control are not).

### `provenance`

`backend`, `assay`, `lif_dynamics` (graph backends; `null` for modular), `seed`, `controller_version`, `label`, `synthetic`,
`test_mode`, `graph` (graph, neuron map and IO map SHA-256, neuron and edge
counts; host paths removed), `dynamics` (the run manifest's model description),
`code` (git commit, dirty flag, SHA-256 of the backend's source files),
`software` (Python and NumPy versions), `params` (`dt_s`, `graph_step_ms`,
`trial_length_s`, `continuous`, `motor_assists`), `initial_state` (brain steps
and whether a saved brain was restored) and `inputs` (the step schedule given
to `neurofly record`).

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
file (`tests/test_run_recording.py`). Everything that would differ between runs
lives outside the file:

- dropped from frames: `timestamp`, `timing`, `run_id`, `identity`, `brain_id`,
  `sim_speed`, `path` (the player rebuilds it from earlier frames), the
  brain summary's `history` and `last_saved`;
- renamed: segment UUIDs become `s0`, `s1`, ... in order of appearance;
- written to the sidecar `NAME.nfrec.json` instead: creation time, host, process
  id, daemon run id, manifest run ids, and the file's own SHA-256 and size.

A recording made on another machine can still differ in the last bits of float
math (a different NumPy, numba or GPU); the header's `software` and `code`
fields say what produced it.

### Size

Frames are the full telemetry packet, uncompressed ~6 KB, about 7 to 47 KB per
simulated second after gzip at `--record-every 1`: 10 KB/s for the modular T-maze,
47 KB/s for the modular open arena (120 KC rates per frame), 7 KB/s for the
synthetic graph. `--record-every N` divides that by about N; replay then draws
one frame per N steps.
