# No silent freeze: receipt (5 October 2026)

Fix for audit F (`docs/receipts/audit-20261005/F-robustness.md`, branch `claude/audit-f`):
the observatory's simulation thread died on a full disk during a periodic checkpoint
while `/api/status` kept saying `online` and the page kept showing old data.
Design F1-F7 of that audit is implemented on branch `claude/no-silent-freeze`;
user-facing behaviour is described in `docs/LEARNING_OBSERVATORY.md` ("When something
fails") and the fields in `docs/DATA_SCHEMA.md` ("Liveness and saving").

All runs used a private daemon on port 8861 (and a static web server on 8860), the real
MaleCNS v1.0 graph, backend `connectome-fixed`. The fault runs used the CPU kernel, like
the audit; the browser runs used the GPU. No disk was filled: the audit's fault wrapper
(`tests/fixtures/silent_freeze/faultd.py`) raises `OSError(ENOSPC)` exactly as a full
disk would. Paths in these files are shortened to `<run>/<file>`.

## Audit scenarios, before and after (`repro_results.jsonl`)

`tests/fixtures/silent_freeze/repro.py all`: arm the fault, observe 10 s, re-select the
assay (the documented recovery), observe 8 s. "Before" is the audit's own measurement
on `master` `b0abd8f` (`F-robustness-results.jsonl`).

| scenario | before: after fault / after recovery | now: after fault | now: after recovery |
|---|---|---|---|
| 1 disk full, periodic checkpoint | **silent freeze**, 0 / 0 steps, "online" | keeps stepping, `degraded` (886 steps / 10 s) | stepping, `degraded` until the 30 s retry |
| same, disk stays full through recovery | (not measured) | keeps stepping, `degraded` | stepping, `degraded` |
| 2 GPU error in checkpoint | **silent freeze** | keeps stepping, `degraded` | stepping |
| 3 disk full, brain save | **silent freeze** | keeps stepping, `degraded` | stepping |
| 4 disk full, events ledger at trial end | **silent freeze** | keeps stepping | stepping |
| 5 disk full, `--record` capture | **silent freeze** | keeps stepping; recording stopped, `recording_error`, `degraded` | stepping |
| 6 exception assembling a frame | **silent freeze** | honest halt (`telemetry publication failed`) | resumes (843 steps / 8 s) |
| 7 step error while disk full | honest halt, then recovery -> **silent freeze** (0 steps, "online") | honest halt, thread alive | resumes (1097 steps / 8 s) |
| 7 with the disk STILL full at recovery | (not measured) | honest halt | resumes, `degraded` (events ledger not saving) |
| CUDA error in the step / world step error | honest halt | honest halt | resumes |
| learning recorder disk full | keeps stepping (error invisible) | keeps stepping; errors now in status | stepping |
| exception in the status handler | connection dropped | HTTP 500 with the error as JSON | stepping |
| disk full at shutdown | exit 1, `stop()` ran twice, PID file and recorder flush skipped | `stop()` once, final checkpoint reported not saved, PID removed, records flushed, exit 1 | |

Every one of the seven silent freezes is now a degraded-but-stepping run or an honest
halt, and the documented recovery works in all of them, including with the disk still full.

## Startup with damaged state (`startup_results.jsonl`)

`tests/fixtures/silent_freeze/startup.py`: newest checkpoint corrupt (hash mismatch) or
missing -> restores version 3 and serves; empty `CURRENT.json` -> restores the newest
version that verifies and serves; truncated `registry.json` or a corrupt brain JSON ->
exit 2 with one plain sentence naming the file and what to do. No traceback in any case.

## Real browser (`browser/`)

`scripts/silent_freeze_browser_check.py`, Firefox 157 headless (selenium + geckodriver),
own daemon and web server, 20/20 checks:

- a. healthy: pill `LIVE DAEMON`, step age resets every sample (`a-healthy-live.png`);
- b. disk full: amber `NOT SAVING: disk full (last saved N s ago)` within 0.3 s of the
  first failed write while the step keeps advancing and the pill stays LIVE; status
  `degraded` (`b-disk-full-not-saving.png`);
- c. disk freed: banner gone after the 30 s back-off retry (24 s after freeing);
- d. a step hung for 25 s with `--step-hard-limit 12`: `STEP RUNNING`, then red
  `SIMULATION NOT ADVANCING` (`d-hung-step-not-advancing.png`); back to LIVE when the
  step returns;
- e. simulation thread killed: red `SIMULATION NOT ADVANCING · 1s` after 1.3 s (threshold
  10 s), never LIVE, step frozen, status `dead` (`e-thread-dead-not-advancing.png`);
  clicking the T-maze card restarted the thread and the pill returned to LIVE with an
  advancing step (`e-recovered-live.png`);
- no page or console errors.

`signoff-run1-07-paused.png` shows the paused pill (`DAEMON CONNECTED · PAUSED`, step age
`paused`), which no longer reads LIVE.

## Live UI sign-off (`live_ui_signoff_run1.json`, `live_ui_signoff_run2.json`)

`scripts/live_ui_signoff.py` twice back to back against the private daemon (GPU):
7/7 scenarios each (initial load, all 14 assays, rapid selection, two-tab sync,
speed 1/20/100x, pause/resume, restore).

## Tests

`tests/test_no_silent_freeze.py` (25 tests) reproduces the seven freezes in process with
the same fault wrappers, plus the dead-thread, stall, exit-on-stall, free-space,
halt-on-persistence-failure, shutdown, startup-fallback and retention cases.
