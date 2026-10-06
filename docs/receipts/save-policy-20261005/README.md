# Save policy, failure classes and record integrity: receipt (5 October 2026)

Supersedes the "degrade and keep stepping by default" behaviour in
`../no-silent-freeze-20261005/`. Policy: Codex direction of 5 October 2026, decision 3.
In the default scientific mode, a failed required save stops the run at a step boundary
and marks its result incomplete. Explicit `--exploratory` mode continues NOT SAVING with
a recorded gap. GPU/compute and software failures always halt. Incidents are
append-only and survive recovery and restart.

Code under test: the receipts below were produced on commit `542a04f` (branch
`claude/no-silent-freeze`). The later forward commits (`f746928`, `198bb27`,
`6bad215`) change the following; their targeted tests and the full suite ran on
`6bad215`:

- the recording-completion durability (`neurofly/recording.py`);
- the ledger validator's overflow handling;
- the whitespace E_inh guard.

The page and the scheduler are unchanged in those commits.

The daemon logs in `repro_results.jsonl` preserve the `542a04f` wording that a failed
recording is "never listed as finished"; that is historical output, not a universal
storage guarantee. With the completion-durability correction in `6bad215`, a final
directory-sync failure is an INVALID save receipt and makes the scientific run
incomplete and halted. If cleanup is also denied, the fully formed artifact may still
remain readable/listable.

All runs used private ports 8861 (daemon) and 8860 (web) and the real MaleCNS v1.0
graph. Disk faults are injected by `tests/fixtures/silent_freeze/faultd.py`, which raises
the errno a full or failing disk would. No disk was filled.

## Fault scenarios per class (`repro_results.jsonl`)

`tests/fixtures/silent_freeze/repro.py all`. The protocol is the same for each
scenario:

1. Arm the fault.
2. Observe for 10 s.
3. Re-select the assay.
4. Observe for 8 s.

"Before" is audit F on `b0abd8f`, where the scenario existed then.

| class | scenario | before | now: after the fault | now: after re-select |
|---|---|---|---|---|
| storage | checkpoint ENOSPC / EIO, brain save, trial ledger, `--record` | silent freeze | honest stop, required save failed, result INCOMPLETE | resumes; recovery checkpoint written first, result stays INCOMPLETE |
| storage | checkpoint ENOSPC with the disk still full at re-select | silent freeze | honest stop | still stopped (re-select refused) |
| storage, exploratory | checkpoint ENOSPC, `--record` ENOSPC | (n/a) | keeps stepping, `degraded`, gap recorded | keeps stepping |
| compute (GPU) | genuine CuPy `CUDARuntimeError` from the connectome step and from the checkpoint device copy, also in exploratory mode; NaN brain state | (n/a) | honest halt, failure class `compute`, INCOMPLETE | resumes after a fresh checkpoint |
| software | telemetry exception, world-step exception, step error while the disk is full | silent freeze / honest halt | honest halt, failure class `software` | resumes |
| diagnostic | learning-recorder telemetry summary | keeps stepping (invisible) | keeps stepping; reported | |
| shutdown | disk full at the final checkpoint | exit 1 after a double stop | exit 1 after a single stop with full cleanup | |

## Startup with damaged state (`startup_results.jsonl`)

| damaged state | result |
|---|---|
| corrupt or missing newest checkpoint, or empty `CURRENT.json` | restores the newest checkpoint that verifies and serves; the run is marked `restored_older_checkpoint` |
| truncated `registry.json`, or a corrupt brain file | exit 2 with one plain sentence |

## Real browser (`browser/silent_freeze_browser_check.json`, 43/43)

Run on Firefox 157 (headless) against our own daemons, using real clicks.

1. Healthy run: the pill reads LIVE.
2. Disk full in scientific mode:
   - the pill reads `SIMULATION STOPPED · NOT SAVED` within 2.9 s;
   - the banner reads `STOPPED · RESULT INCOMPLETE`;
   - the step is frozen.
3. A new tab opened during that fault shows the same state, never DISCONNECTED or the local engine.
4. Re-select with the disk still full: the run stays stopped.
5. Disk freed, then re-select: the pill returns to LIVE and the `RESULT INCOMPLETE` banner stays.
6. CuPy error: the pill reads `SIMULATION HALTED · GPU/COMPUTE ERROR`, then recovers after re-select.
7. Slow CPU (every step takes 3 s): the pill is never NOT ADVANCING.
8. A step hung past `--step-hard-limit 12`: the pill turns red after 12 s and recovers when the step returns.
9. Simulation thread death:
   - the pill turns red after 1.3 s;
   - a new tab opened while it is dead shows NOT ADVANCING;
   - re-select restarts the thread.
10. Pause held for 14 s: the pill reads `DAEMON CONNECTED · PAUSED`, never NOT ADVANCING.
11. Transport loss (the daemon is stopped with SIGSTOP): DISCONNECTED after 20 s. After SIGCONT the page reconnects INTO the halt (`SIMULATION HALTED · ERROR`); re-select recovers.
12. Exploratory daemon: the pill reads `LIVE DAEMON · EXPLORATORY`, the banner reads `EXPLORATORY · NOT SAVING`, and the step advances. A new tab shows the same state.

The browser shows no page or console errors. Some screenshots listed in the JSON were left out of this
receipt to keep it small.

## Live UI sign-off (`live_ui_signoff_run1.json`, `live_ui_signoff_run2.json`)

`scripts/live_ui_signoff.py` ran twice back to back against the private daemon on the
GPU. Each run passed 7/7: initial load, all 14 assays, rapid selection, two-tab sync,
speed, pause/resume, restore.

## Limitations

- Headless Firefox only, on our own daemons. This is not the owner's running
  observatory, so the AGENTS.md final sign-off on the deployed UI is still owed.
- A genuine GPU failure cannot be induced. The compute class is exercised with the
  genuine CuPy runtime error type, raised at the engine layer.
