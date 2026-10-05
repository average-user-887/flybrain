# Audit F: daemon robustness and failure handling

Date: 2026-10-05. Code audited: `master` at `b0abd8f`. Nothing in the code was changed.
This receipt diagnoses the problem and proposes fixes.

## 1. The incident, reproduced

**What the owner saw.** At 15:22 the observatory daemon (`connectome-fixed`, t-maze, 3x,
`--checkpoint-interval 30`) hit `OSError: [Errno 28] No space left on device` during a periodic
checkpoint. After that, `/api/status` kept reporting `status: "online"`, `paused: false`,
`halted: false`, while `total_steps` never advanced again. The page showed old data. The
process kept running until it was restarted at 16:04, 42 minutes later.

**The traceback in the service journal** (hostname and paths removed; line numbers are from the
deployed tree, which has the same structure as `master`):

```
Exception in thread NeuroFly-SimLoop:
  neurofly_daemon.py  _run_loop        -> self._advance_one()
  neurofly_daemon.py  _advance_one     -> result = self.step_once(publish=False)
  neurofly_daemon.py  step_once        -> self.save_checkpoint("periodic")
  neurofly_daemon.py  save_checkpoint  -> self.registry.checkpoint(world_state=...)
  experiment_registry.py checkpoint    -> atomic_write_bytes(path, data)
  provenance.py       atomic_write_bytes -> stream.write(data)
OSError: [Errno 28] No space left on device
```

The first line is the important one: **the simulation thread died.** The process did not die.

**How it happens (file:line on `master`):**

1. `step_once` calls `save_checkpoint("periodic")` at `neurofly_daemon.py:1287`. That call sits
   outside the `try` that guards `arena.step` (`:1252-1262`). Only an exception from
   `arena.step` goes through `_halt_on_error`.
2. `_advance_one` (`:1085-1098`) has `try/finally` and no `except`. `_run_loop` (`:1018-1083`)
   has no handler at all. The `OSError` therefore ends the `NeuroFly-SimLoop` thread. The
   `with self.lock` block releases the lock while the exception unwinds, so no lock is left held.
3. The thread is `daemon=True` (`:975`). Python prints "Exception in thread" to stderr and the
   process continues. The HTTP server, the SSE threads and the recorder thread are separate, so
   they keep running.
4. **No state flag changes.** `self.running` stays `True` and `last_error` stays `None`.
   `_status_payload` (`:1952-1958`) bases its answer only on `last_error` and `paused`, so it
   reports `online`, not halted and not paused. `timing_snapshot` (`:1143`) returns the last
   value stored in `sched_stats["achieved_speed"]`. Only the dead thread updated that value, so
   status keeps claiming the run achieves about 1.6x (and `overloaded: true`) indefinitely.
5. **Nothing new is published.** Only the dead thread called `_publish_snapshot`.
   `_stream_snapshots` (`:2217`, `while runner.running`) keeps serving: each new client gets the
   last frame once, then a heartbeat every second. The heartbeat (`:2237`) has no step counter
   and reports `step_in_progress_s: 0`.
6. **The documented recovery reports success and changes nothing.** `dispatch_command`
   (`:1600`) checks `_loop_active()`. That returns false for a dead thread, so the command runs
   directly under the lock. `switch_paradigm` rebuilds the arena and returns `status: "ok"`
   with `halted_by_error: null`, but no thread exists to step it. Only a process restart helps.
7. systemd cannot detect this. The unit uses `Restart=always`, which acts only when the
   process exits, and it has no `WatchdogSec`.

**Deterministic reproduction** (`F-robustness-repro/repro.py ckpt_enospc`). The unmodified
daemon runs on its own port with the real MaleCNS graph on the CPU. A wrapper raises
`OSError(ENOSPC)` from `experiment_registry.atomic_write_bytes` once the fault is armed. No disk
is filled. The fault is disarmed right after it fires, so the disk is free again for the rest
of the test.

| window | steps advanced | status / halted / paused | SimLoop thread | SSE data frames | SSE heartbeats |
|---|---|---|---|---|---|
| healthy, 5 s | 311 | online / false / false | alive | 142 | 0 |
| after the fault, 10 s | **0** | **online / false / false** | **dead** | 1 (the old frame) | 9 |
| after "re-select assay", 8 s | **0** | online / false / false (command answered `ok`) | dead | 1 | 7 |

The traceback from the reproduction matches the owner's journal frame for frame
(`_run_loop -> _advance_one -> step_once:1287 -> save_checkpoint:1893 -> checkpoint:431`).

**Two details I could not reproduce.** The owner reported 88% CPU and "frames at 60 Hz". In
the reproduction (CPU backend), the process drops to 0.1-0.2% CPU once the thread dies, and
the stream sends no new data frames. The page's own 60 fps render loop draws the same stale
pose continuously, which looks like frames arriving. The 88% CPU probably comes from the GPU
build's runtime or from the browser; this was not verified. Neither detail changes the
mechanism.

## 2. Survey of failure paths

Method: `F-robustness-repro/repro.py all` runs one scenario per daemon launch. For each it arms
the fault, waits until it fires, disarms it, observes for 10 s, sends the documented recovery
(re-select the assay), observes again, then sends SIGTERM. Startup cases are in
`F-robustness-repro/startup.py`. Raw results are in `F-robustness-results.jsonl`.

Categories:
- **keeps stepping**: the simulation continues.
- **honest halt**: the simulation stops and says so.
- **SILENT FREEZE**: the simulation stops while status still claims `online`.
- **honest crash**: the process exits with a clear error.

| # | failure | trigger (injected) | observed behaviour | category | mechanism (file:line) | proposed fix |
|---|---|---|---|---|---|---|
| 1 | disk full, periodic graph checkpoint (**the incident**) | `atomic_write_bytes` raises ENOSPC | thread dies; 0 steps; status online; recovery answers `ok` and does nothing | **SILENT FREEZE** | `neurofly_daemon.py:1287` outside the guard; `_advance_one:1085`, `_run_loop:1018` have no handler | F1 + F2 + F3 |
| 2 | GPU error while a checkpoint copies device state (GPU lost) | same call raises a CUDA `RuntimeError` | same as #1 | **SILENT FREEZE** | same path; device-to-host copy happens in `registry.checkpoint` (`experiment_registry.py:417-438`) | F1 + F2 + F3 |
| 3 | disk full, modular brain save at checkpoint start | `ExperimentBrain.save` ENOSPC | same as #1 | **SILENT FREEZE** | `save_checkpoint:1884` `brains.save_all()`, reached from `:1287` | F1 + F2 |
| 4 | disk full writing the events ledger at a trial end | `ExperimentBrain.log` ENOSPC (`--trial-seconds 1`, non-continuous) | same as #1 | **SILENT FREEZE** | `step_once:1273` -> `_end_trial:1350` -> `_record_trial_milestone:1567` (`log`) and `:1570` (`save`) | F1 + F2 |
| 5 | disk full writing a `--record` replay | `RunRecorder.capture` ENOSPC | same as #1 | **SILENT FREEZE** | `step_once:1282` (and `:1248` in teaching mode) | F1 + F2 (stop the recording and report it) |
| 6 | exception while assembling a telemetry frame | `_assemble_telemetry` raises | same as #1 | **SILENT FREEZE** | `_publish_snapshot:1174`, called from `_run_loop:1041/1054/1082` | F1 (a publish failure becomes an honest halt after N retries) |
| 7 | step error **while the disk is full** | `arena.step` raises, then `log` raises ENOSPC | thread dies after `last_error` is set; status shows `error` at first. Re-selecting the assay **clears the halt**: status becomes `online`, not halted, 0 steps | honest halt that recovery **turns into a SILENT FREEZE** | the halt bookkeeping write `step_once:1257` sits inside the `except`; recovery runs inline because `_loop_active()` is false (`:1600`) -> `_clear_error` | F1 + F3; never write to disk inside the halt path without a guard |
| 8 | CUDA error inside the connectome step | `GraphArenaController.__call__` raises | `status: error`, `halted: true`, error text shown; frames keep coming; re-selecting the assay resumes (543 steps / 8 s) | honest halt | `step_once:1252-1262`, `_halt_on_error:1295` | works as designed |
| 9 | exception in world or body step | `Arena.step` raises | same as #8; resumes (661 steps / 8 s) | honest halt | same | works as designed |
| 10 | step that never returns (kernel hang, stalled I/O) | analysis, not injected | the heartbeat reports `step_in_progress_s` growing; the page shows "STEP RUNNING Ns" with no upper bound | slow, unbounded (borders on silent) | `step_in_progress_s:1132`, `app.js:3954-3960` | F3: stall after a hard limit, for example 300 s |
| 11 | disk full in the learning recorder (trials and telemetry summary) | `JsonlWriter.append` ENOSPC | keeps stepping (522 steps / 10 s); `[Recorder] error` printed to the journal; `/api/status` never mentions it | keeps stepping (records lost silently) | `learning_recorder.py:328-335`: the error counter is not exposed | F2: put recorder errors in status |
| 12 | exception in an HTTP handler | `_status_payload` raises | that request closes with no reply (`RemoteDisconnected`); stepping continues (711 steps / 10 s) | keeps stepping | `socketserver` handles each request in its own thread | answer 500 with the error as JSON |
| 13 | disk full at shutdown | brain save raises during `stop()` | `stop()` runs twice (signal handler and `finally`), both raise, exit code 1; recorder final flush and PID-file removal are skipped | honest crash (with data loss) | `stop:979-990` not idempotent; `_signal_handler:2527`, `finally:2550` | F5 |
| 14 | checkpoint hash mismatch at startup | 4 bytes appended to the current `.npz` | exits with code 2: "Cannot start backend ... CheckpointCorrupt" | honest crash | `experiment_registry.py:499-506`; caught at `run_daemon:2477` | F6: fall back to the newest checkpoint that verifies (19 more are kept) |
| 15 | checkpoint file missing at startup | current `.npz` deleted | exit 1, raw `FileNotFoundError` traceback | honest crash (unfriendly message) | `run_daemon:2477` catches only `RuntimeError` | F6 |
| 16 | empty `CURRENT.json` / truncated `registry.json` | file truncated | exit 1, raw `JSONDecodeError` traceback | honest crash (unfriendly message) | `experiment_registry.py:323`, `current_pointer:415` | F6 |
| 17 | corrupt brain JSON at startup | `graph-bookkeeping/.../t-maze.json` truncated | exit 1, `ValueError: Cannot restore ...` traceback | honest crash (unfriendly message) | `experiment_brains.py:160-189`; `ValueError` is not caught at `:2477` | F6 |
| 18 | GPU disappears | analysis: covered by #2 and #8 | failing inside a step gives an honest halt; failing inside a checkpoint gives a SILENT FREEZE; the status device probe is guarded (`:1988`) | mixed | as #2 and #8 | F1 |

Under systemd (`Restart=always`, `StartLimitBurst=5` per 60 s), cases 14-17 restart about five
times and then the unit stays failed. The page then shows "disconnected", which is honest. It
still needs manual repair, even though older checkpoints that verify are on disk.

## 3. Silent-freeze mechanisms, ranked

1. **The simulation thread dies on any exception outside `arena.step`, and nothing notices**
   (#1-#6). There is one root cause and six known entry points: checkpoint, brain save,
   events-ledger write, replay capture, telemetry assembly, and any future code added to
   `step_once` or `_run_loop`. Thread death is never turned into state. Status, the heartbeat
   and the command path all read flags that only the dead thread could have changed. Disk-full
   alone reaches this in four different ways. This is the incident.
2. **Status and the page cannot tell "not advancing" from "fine".** Status has no step age and
   no thread-liveness check, and it carries a frozen `achieved_speed`. The page marks a frame as
   fresh even when the step has not changed, and the heartbeat carries no step. This is why #1
   looked like a live run instead of an error.
3. **Recovery works only while the thread is alive** (#7 and the recovery column of #1-#6).
   With a dead loop, commands run inline and `_clear_error` lifts the halt, so an honest halt
   can be turned into a silent freeze by the very action the halt message tells the user to
   take.
4. **The halt path writes to disk without a guard** (#7). On a full disk, honest halts become
   thread deaths.
5. **No bound on a step that never returns** (#10). The page shows "STEP RUNNING" indefinitely.
   This is honest, but nothing ever ends it.

## 4. The dashboard side

Logic involved: `web/app.js` `updateFreshness` (`:3906-3952`), `lastValidDataTime` (`:4038`),
heartbeat handling (`:3994-3998`), and the halt pill (`:4073`).

When the loop dies, no new frames arrive, so the page's *data age* grows and the pill turns
amber: "● LIVE DAEMON · STALE DATA". The pill still says LIVE DAEMON. Nothing explains that the
simulation has stopped, nothing turns red, and nothing points to the cause or the fix. That is
the "static page" the owner saw.

The page cannot detect "steps not advancing while not paused" in general:
- `lastValidDataTime` is updated by every accepted frame, **including frames whose `step` did
  not change**. A daemon that keeps publishing the same step while not paused would read LIVE.
  The existing guard for halts covers only frames that carry `halted`/`error`.
- Heartbeats carry `seq`, `server_time`, `step_in_progress_s` and `last_step_wall_s`. They
  carry no `step`, `paused` or liveness, so the page cannot tell a dead loop from an idle one.

## 5. Resource hygiene

These are measurements on the observatory output directory, not on the system disk.
- **Graph checkpoints**: one `.npz` per instance per checkpoint, about **16.9 MB** each. At the
  observatory's 30 s interval that is about **2.0 GB/hour written** (SSD wear). Retention
  (`keep_checkpoints=20`) caps the footprint at about **340 MB per assay instance**, up to about
  4.7 GB if all 14 assays have been visited. Currently 2.4 GB. Every write needs about 17 MB
  free, and nothing checks for it first.
- **Daemon JSON checkpoints** (`checkpoints/`, 70-75 MB): `_prune_periodic_checkpoints`
  (`:1915`) prunes only the *active* assay's periodic records. **13,642** open-arena periodic
  files remain from before retention existed (43 MB). `final_shutdown` records are never pruned.
  They are written **twice per restart** (about 3.5 MB each) because `stop()` runs from both the
  signal handler and `finally`: 30 such files for t-maze so far.
- **Learning records**: `telemetry_summary.jsonl` grows about 87 KB/hour and the per-instance
  `events.jsonl` about 24 KB/hour. Both are small. `JsonlWriter` rotates at 64 MiB and never
  deletes, so they are unbounded but slow.
- **Logs**: request logging is suppressed (`log_message`), so the journal gets almost nothing
  from the daemon.
- **Disk-full attribution**: at the steady state above, the daemon's own footprint is bounded
  (a few GB at most). It did not fill a 231 GB system disk by itself, but it needs 17 MB
  of headroom every 30 s and was the first writer to fail. Large consumers elsewhere on the same
  disk (agent worktrees with their own `outputs/`, caches) are outside this audit.
- **Memory**: the deployed process reported a 2.3 GB memory peak over its 19 h run (from
  systemd). Three samples of the fresh process gave RSS 1,327 / 1,338 / 1,351 MB at 4, 15 and
  17 min uptime. The cause is not established; it may be warm-up. In-process lists that grow
  without a limit: `trial_history` (`:1572`) and `learning_curve` (`:1565`). They are small in
  continuous mode. A `tracemalloc` comparison over an hour is needed before claiming a leak.

## 6. Proposed fix design

**F1. A simulation loop that cannot die silently.**
- Wrap each `_run_loop` iteration in `try/except Exception`. Route the exception through
  `_halt_on_error(exc, phase=...)`, where phase is one of step, publish, persistence or command,
  then continue the loop in the halted state. That keeps it publishing at the paused rate and
  draining commands.
- Wrap the whole loop as well: if it exits while `running` is still true, set
  `self.loop_failure = {type, message, traceback, at}`.
- Make `_halt_on_error` itself safe to fail: guard its `active_brain.log(...)` call (#7).
- Install `threading.excepthook` for daemon threads, recording any uncaught exception into a
  `thread_failures` field that status reports.
- `_status_payload` must check `sim_thread.is_alive()`. A dead loop while `running` is true
  means `status: "error"`, `halted: true` and `error: "simulation thread stopped: ..."`.
- `dispatch_command` must not run the command inline when the loop *should* be running. A
  successful rebuild (`switch_paradigm` / `switch_backend`) restarts the thread if it is dead.
  The reply's `ack` must reflect the real result.

**F2. Persistence faults degrade the run; they do not stop it.**
- Move the periodic checkpoint out of `step_once` into `_run_loop`, between batches, still
  under the lock and at a step boundary, so determinism is unchanged.
- On `OSError` (or any exception) from `save_checkpoint`, a trial ledger write or a replay
  capture:
  - record `persistence = {state: "failing", error, errno, path, failures, last_ok_at,
    next_retry_at}`;
  - **back off**: 30 s, then 60 s, and so on up to 10 min. Retrying every step would attempt a
    17 MB write every 20 ms;
  - keep stepping;
  - stop an active `--record` capture with a visible `recording_error`.
- Before each graph checkpoint, call `shutil.disk_usage`. If free space is below
  2 x the last checkpoint size + 512 MB, skip the write and report `persistence.state =
  "disk_low"` **before** ENOSPC happens.
- Expose `RecorderThread.errors` and its last error the same way (#11).
- Status: `status: "degraded"` while persistence is failing.
- Owner decision: whether, after a long outage (for example 10 x the interval), the daemon
  should escalate to an honest halt with `--halt-on-persistence-failure`. Default proposal:
  stay degraded and visible. The simulation is correct; only its durability is at risk.

**F3. Watchdog: detect steps that are not advancing, in the daemon.**
- The sim thread stamps `last_advance_wall = time.monotonic()` in `_advance_one`.
- Status and every frame carry
  `liveness = {state, step, last_advance_age_s, sim_thread_alive, step_in_progress_s}`, where
  `state` is one of:
  - `advancing`;
  - `paused`;
  - `halted`;
  - `slow`: a step is in progress below the hard limit;
  - `stalled`: not paused, not halted, `_can_step()` true, and
    `last_advance_age_s > max(10 s, 20 x dt/speed, 3 x last_step_wall_s)` with no step in
    progress, **or** `step_in_progress_s > step_hard_limit_s` (default 300 s);
  - `dead`: the thread is not alive.
- `stalled` and `dead` set `status: "error"`.
- `achieved_speed` is computed from `(step, wall)` samples at read time, so a frozen value can
  no longer read as 1.6x.
- A small `NeuroFly-Watchdog` thread evaluates this every 2 s. On entering stalled or dead it
  logs one line to the journal. Optionally (`--exit-on-stall S`, enabled in the observatory
  unit) it calls `os._exit(70)` after S seconds of `dead` or `stalled`. `Restart=always` then
  resumes from the last checkpoint. For systemd, use `Type=notify` with `WatchdogSec=120`, and
  send `WATCHDOG=1` only while the state is advancing, paused or halted.
- The SSE heartbeat gains `step`, `paused` and `liveness`.

**F4. Watchdog: detect steps that are not advancing, on the page.**
- Keep `lastStepAdvanceTime`, updated only when a frame's `step` increases (or the run, assay
  or segment changes).
- State `stopped` (red pill, "● SIMULATION NOT ADVANCING · Ns"): the daemon reports `stalled`
  or `dead`, **or** the client sees no step advance for more than
  `max(10 s, 20 x dt/requested_speed)` while frames or heartbeats say it is not paused and not
  halted. The client check is a fallback for daemons without F3.
- The tooltip gives the daemon's error and the fix ("restart the daemon" for `dead`; persistence
  errors name the path and "free disk space").
- Show **step age** next to data age.
- An amber "NOT SAVING: disk full (last saved 12 min ago)" banner when `persistence.state` is
  not ok.
- Never show the word LIVE while the step is not advancing and the run is not paused.
- The browser check required by `AGENTS.md`: inject #1 into a local daemon, watch the pill go
  red within N s, re-select an assay and see it resume.

**F5. Shutdown.**
- Make `stop()` idempotent (one final checkpoint, not two).
- Wrap the final checkpoint so a failure is logged and the recorder flush and PID-file removal
  still run.
- Exit non-zero only after cleanup.

**F6. Startup.**
- On `CheckpointCorrupt`, a missing file or an unreadable `CURRENT.json`, try the retained
  checkpoints from newest to oldest. Restore the first one that verifies and log a `restore`
  event naming the skipped versions.
- Refuse only when none verifies.
- Catch `ValueError` and `OSError` alongside `RuntimeError` in `run_daemon` so every startup
  refusal prints one plain sentence (path plus what to do) instead of a traceback.

**F7. Hygiene.**
- Prune periodic JSON checkpoints for *all* assays, not only the active one.
- Keep the newest few `final_shutdown` records.
- Consider a longer observatory checkpoint interval (300 s takes write churn from about 2 GB/h
  to about 0.2 GB/h).
- Add a size budget for `registry-v3` to `keep_checkpoints`.

**Tests to add with the fix:**
- Each of #1-#7 as a unit test with a fault wrapper, like `faultd.py`. Each test asserts that
  the loop stays alive, that status shows `degraded` or `error`, that steps continue (F2) or a
  halt is reported (F1), and that recovery resumes stepping.
- A dead-thread test: kill the loop and assert status says `dead` within 3 s.
- Shutdown idempotence.
- Startup fallback to an older checkpoint.

## 7. Reproduction

Scripts are in `F-robustness-repro/`. Raw per-scenario results are in `F-robustness-results.jsonl`.

```
# from a checkout with the prepared MaleCNS graph; uses port 8851 only
python docs/receipts/audit-20261005/F-robustness-repro/repro.py ckpt_enospc   # the incident
python docs/receipts/audit-20261005/F-robustness-repro/repro.py all           # the survey
python docs/receipts/audit-20261005/F-robustness-repro/startup.py <a run's out dir>
```

- `faultd.py` runs the unmodified `run_daemon()` with wrappers that raise ENOSPC, a CUDA-style
  `RuntimeError` or a generic `RuntimeError` when a control file exists. Prefixing the spec
  with `once:` makes the fault transient. It also writes the live thread list every second.
- `probe.py` measures what a user sees over a window: steps advanced, status flags, SSE data
  frames versus heartbeats, whether the sim thread is alive, and CPU use.
- Environment: `NEUROFLY_BRAIN_BACKEND=cpu`, so no GPU is used. `AUDIT_SCRATCH` sets where run
  directories go; the default is the system temp directory.
