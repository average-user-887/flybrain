# PR #30 review (release gate G13), 5 October 2026

Branch `claude/review-pr30` = PR #30 head `1706886` + `origin/master` `9a6c1c5` + review fixes.
Everything below ran on the review branch's final code, except where a row says otherwise.
Ports were 8793 (daemon), 8795 (dashboard), 8796 (studio). The owner's live services were
not touched. Paths are written as `<repo>/`, `<scratch>/`, `<home>/`.

## Scope of the PR

It adds a separate stdlib server (`neurofly_studio/`), `web/studio.{html,js}`, a small additive
hook in `web/embodied_replay.js`, docs and tests. It does **not** change `neurofly_daemon.py` or
the dashboard (`web/app.js`, `web/index.html`). The daemon and dashboard checks below are
regression checks of those unchanged paths, as AGENTS.md requires.

## Results

| check | result | evidence |
|---|---|---|
| Full test suite (real graph present, GTX 1660 Ti) | **759 passed, 9 skipped, 0 failed**: master's 717 plus 42 studio tests (38 from the PR, the #31-dependent one now running, and 3 added in review) | skips are the same 9 as on master: selenium interpreter, FlyGym physics opt-in, retained-brain fixtures |
| `tests/test_switch_recovery.py` (switch-freeze regression) | 7 passed | |
| API switching loop, real graph, CPU brain | settled: 2 passes, 28 switches, 0 freezes; rapid: 3 passes, 42 switches, 0 freezes; optomotor visited 5 times; final status online | `api-switching/` |
| `scripts/live_ui_signoff.py`, real Firefox 157 (headless), real MaleCNS graph, `connectome-fixed`, **CPU** brain backend | **runs 5 and 6 back to back: 7/7 each**, no console or app errors | `signoff/run5`, `signoff/run6` (+ screenshots) |
| earlier sign-off runs on the same daemon | run 1: 6/7 (`pause_resume.clock_frozen`, the known harness timing quirk); run 2: 7/7; run 3: 6/7 (`rapid_selection`: the daemon was still on `looming-escape` 35 s after the burst); run 4: 6/7 (`all_assays`: the `circadian-dam` switch was not acknowledged while a 4 s step was running) | `signoff/run1`–`run4` |
| Studio features in real Firefox, real runs (FlyGym body + MaleCNS v3, CPU) | all exercised, no page errors | `studio/` |

Runs 3 and 4 failed while the machine was shared: load average 7–9, another session's daemon,
and the GPU worker. The CPU backend reached about 0.04× of the requested speed during the
bursts, and command latency went up to 4.2 s against the dashboard's 2 s fetch timeout. Both
failures are in unchanged master code paths, and both cleared on rerun. The earlier 7/7
receipts used the GPU backend. Run 3 deserves a follow-up on master: confirm that a coalesced
click is never dropped when the daemon is slower than the dashboard's fetch timeout.

### Studio walk-through (`scripts/studio_firefox.py`)

- **Gallery:** the honest lead text shows. The empty state is followed by a curated pair and
  four finished runs. Watch on the curated card opens the replay (25 frames, SHA-256 verified).
- **Build:** 14 paradigm cards, all badged `Mapped` (no paradigm is "Tested on v3" in the
  matrix, so nothing shows Validated); only optomotor is buildable. The three preregistered
  optomotor specs are listed read-only with their hashes. Five silence groups appear.
  Controls: with nothing silenced, `output-disconnected` is preselected and `intact` is
  disabled; silencing DNa02 switches the default to `intact`; un-silencing switches it back.
  Queued: DNa02 silenced + intact control (seed 11); intact + disconnected (seed 12);
  a 2-repeat left-only T4/T5 set (4 runs). A contrast of 3 is refused by the server and shown
  ("Not queued: contrast: between 0.0 and 1.0").
- **Queue:** live rows. Cancel removed all 4 pending "cancel me" runs while another run was
  running. Worker line: "running · 1 running · 2 waiting".
- **Runs:** 4 real 0.5 s runs completed at about 0.043× real time. The DNa02 clamp held
  (2 neurons, 0 spikes).
- **Curation:** `neurofly_body replay-check` gave BIT_IDENTICAL for both runs of the seed-12
  pair (CPU). `neurofly_studio curate` then installed the pair. Compare with pair shows both
  runs and the metrics table.
- **Download:** HTTP 200 `application/zip` with attachment disposition. Firefox saved the zip,
  all `bundle.json` checksums match, and it contains no local paths, host name or account name.
- **Compare (synced):** both replays load (25 frames each). At 0.25×, both frames advance and
  stay equal (6/6 → 10/10 → 18/18). Pause holds (19/19 after 1 s). Seeking to 50% puts both on
  frame 11. 4× plays both to the last frame and stops.
- **Phone width (390 px):** no horizontal overflow.

The numbers in these 0.5 s runs are anecdotes, as the page itself says. For example, the
seed-12 intact fly turned at −0.001 rad/s and the disconnected one at 0.011 rad/s. Nothing
here is an optomotor result.

## Defects found and fixed on the review branch

1. **Privacy: local paths in shared and committed run files.** A real run's `manifest.json`
   (8 paths) and the `body.nfbody` header (3 paths) record `/home/<user>/…` and
   `/media/<user>/…`. Bundles (made to be shared) and curated runs (committed to the public
   repo) copied them unchanged. They now pass through `neurofly_studio/redact.py`:
   `<repo>`, `<home>`, `<local-path>/<last>`, `<host>`, `<user>`. Frame and trajectory hashes
   still verify. On the four real runs, 0 paths remain. Tests added.
2. **Overclaim in user-facing text.** The optomotor card said the rotation "drives
   direction-selective motion neurons of the connectome", and the gallery said the body is
   "driven by the full MaleCNS connectome". Both now say that an engineered encoder imposes the
   direction on T4/T5, that an engineered decoder drives the body, and that there is no passing
   validation. The seed help no longer implies cross-machine bit-identity, which G11 has not
   measured yet.
3. **DNS rebinding.** The same-host Origin check compares Origin with Host, and a rebinding
   page controls both. On a loopback bind, requests must now name a loopback Host
   (421 otherwise). Test added.
