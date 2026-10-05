# Functional audit D: every interactive control in the dashboard and the experiment studio

Audit date: 5 October 2026. Code audited: `origin/master` at `b0abd8f`, in a detached worktree
with the MaleCNS graph and annotation data linked in (gitignored). **Nothing was fixed. This
document only reports.**

## How it was tested

- **Services:** an audit daemon on port 8841, the dashboard served as static files on 8840
  (`index.html?daemon=http://127.0.0.1:8841`, so the page could never attach to another daemon),
  and the studio on 8842. The CPU brain backend was used for everything (`NEUROFLY_BRAIN_BACKEND=cpu`,
  no GPU). The owner's services on 8769, 8780 and 8781 were not contacted. Everything started
  for this audit was stopped afterwards.
- **Browser:** real Firefox (headless) driven by Selenium. Every control was clicked or set the
  way a user would. The effect was then checked on screen (DOM state, canvas changes, banners,
  screenshots) and in the daemon (`/api/status`, `/api/telemetry`, `/api/observatory`, command
  replies). A fetch hook recorded every `/api/command` the page sent, together with the reply.
- **Backends:** `modular`, then a switch made from the dashboard's own Controller dropdown to
  `connectome-fixed` (the real 166.7K-neuron graph), and then to `connectome-plastic` and back.
  Standalone preview (no daemon reachable) was tested separately with
  `?daemon=http://127.0.0.1:8849`, a port with nothing listening.
- **Studio:** real queued runs (FlyGym body with the MaleCNS v3 graph on the CPU, 0.5 s each).
  One pair was curated with the studio CLI (`replay-check` gave BIT_IDENTICAL for both runs),
  so that the gallery's curated path could be exercised.
- **Evidence:** `D-evidence/logs/*.jsonl` holds one JSON line per action, with the command sent,
  the daemon's reply and the before and after values. `D-evidence/scripts/` holds the harness
  and launch scripts, and `D-evidence/screenshots/` holds the key screenshots. Local paths are
  redacted to `<scratch>`, `<repo>` and `<home>`.

Verdicts: **WORKS**, **BROKEN** (does the wrong thing or fails), **NO VISIBLE EFFECT**
(accepted, but nothing a user can see changes), **MISLEADING** (the UI implies an effect that
does not happen), **DEAD** (wired to nothing, or an API action that nothing in the UI calls).
"Live" means connected to a daemon. "Preview" means the in-browser standalone engine.

## Headline findings

1. **The View 2D/3D toggle is broken on every backend and in preview.** On the first 3D frame,
   `ArticulatedFly3DViewport.updatePose()` (`web/app.js:6471`) reads `arena.fly.pos.x`, but the
   browser fly has `x` and `y`, not `pos`. The render phase throws
   `can't access property "x", fly.pos is undefined`, the red RENDER ERROR banner appears, and
   the whole view is suspended. The 3D canvas stays black, with the fly fixed at the origin.
   Switching back to 2D leaves the 2D arena frozen too, until the user clicks "Resume view".
   Screenshot: `modular-3d-on.png`.
2. **Controls that are disabled in live mode look enabled.** In live mode `HUD.update()`
   (`web/app.js:5787-5788`) disables Deploy Threat, all six lesion buttons, the Limb &
   Neuro-Stim deck and the guide sliders, but no CSS styles them as disabled. They look
   clickable and do nothing; the reason exists only as a hover tooltip. A disabled Deploy Threat
   click also leaves the previous tool active, so the next arena click places a different
   stimulus. This was observed: an alarm odor was placed after Deploy Threat was clicked.
3. **The Training & Data tab offers modular-only actions on graph runs.** On `connectome-fixed`,
   Teach, Reverse cues and Probe memory stay enabled and fail only after the click
   ("acts on the modular mushroom body, which is not the controller of this connectome-fixed
   run"). Freeze learning answers "Recorded: set learning", but a fixed-weight graph has nothing
   to freeze.
4. **The Controller dropdown cannot show or select `connectome-with-trained-readout`**, which
   the daemon supports. While the daemon runs that backend, the dropdown is blank
   (`selectedIndex = -1`). A refused switch (during a recording) gives no message; the dropdown
   silently snaps back. With no daemon, the dropdown POSTs `/api/controller` to the static page
   server, gets HTTP 501, and shows nothing.
5. **The ΔOFF (T5) lesion and the Wing Extension slider are dead code**, even in preview. The
   value is stored and never read. The lesion card still describes an expected deficit.
6. **The research download links (`research-status.json`, `research-latest.zip`) return 404**
   on both the static server and the daemon, unless `scripts/research_worker.py` happens to be
   running.
7. **The core live loop works on both backends.** Pause, speed, reset, all 14 assay switches
   (including a rapid burst), every connected assay parameter and intervention, the canvas
   stimulus tools in the open arena, recording replay, the telemetry exports and the modular
   training actions changed the daemon state as intended. The studio worked end to end.

## Dashboard (`web/index.html`, `web/app.js`, `web/live_assays.js`, `web/training.js`, `web/replay.js`)

| control | where (file:line) | expected | observed | verdict | evidence |
|---|---|---|---|---|---|
| Pause / Resume | index.html:1143, app.js:6589 | pause/resume the daemon | `set_paused` ok. Daemon `paused=true`, 0 steps advanced during the pause, label Resume/Pause. Same on connectome-fixed. Preview: local engine frozen | WORKS | live-modular, live-connectome, preview |
| Speed cycle button | index.html:1145, app.js:5361 | next speed | `set_speed 2`, daemon `sim_speed=2.0`, label, stat and select synced | WORKS | live-* `btnSpeedToggle` |
| Speed select | index.html:1146, app.js:5369 | set speed | 5x gave daemon 5.0, achieved 5.0x (modular). Connectome on CPU achieved 0.021x, reported honestly in "Achieved" | WORKS | live-* `selectSpeed=5` |
| Controller dropdown (live) | index.html:1160, app.js:5376 | switch the daemon backend | modular→connectome-fixed in 17 s, →connectome-plastic in 3 s, →modular. Identity bar follows | WORKS | backend.jsonl |
| Controller dropdown: trained-readout backend | index.html:1160-1164 | list every daemon backend | no `connectome-with-trained-readout` option; when the daemon runs it, the select is blank | MISLEADING | backend.jsonl, `backend-trained-readout.png` |
| Controller dropdown: refused switch | app.js:5376-5380 | show why a switch failed | daemon refused ("Stop the recording before switching backend"); no message shown, the select silently reverts | MISLEADING | raster.jsonl `selectBackend-while-recording` |
| Controller dropdown (no daemon) | app.js:5381-5390 | switch the local controller | POST `/api/controller` to the page's own origin: HTTP 501 from the static server, nothing shown, the select keeps the new value | BROKEN | preview.jsonl |
| Reset Trial | index.html:1166, app.js:5356, 1137 | new trial | `reset_trial`: trial 4→5, new `segment_id`, `transition=manual_reset`. Preview: trial 1→2. During a replay: does nothing, with no message | WORKS | live-* `btnResetTrial` |
| Connection pill | index.html:1131, app.js:3682 | reconnect / choose an offered daemon | reconnect keeps the page LIVE. The "offered daemon" path was not testable (it needs port 8769, which is off limits) | WORKS (partly tested) | live-* `clusterStatusPill` |
| "Training & Data" nav button | index.html:1129, app.js:3709 | open the Training tab | Training tab active, panel shown (a duplicate of the deck tab) | WORKS | live-* `researchLink` |
| Replay button / chooser | index.html:1130, replay.js:419 | list recordings | lists the daemon's `.nfrec` files with assay, backend, duration and size | WORKS | v3r-modular |
| Replay chooser Close | index.html:1193 | hide the chooser | hidden | WORKS | v3r-modular |
| Replay list entry | replay.js:409 | load a recording | 2397 frames, "digest OK", replay bar shown, time advancing | WORKS | v3r-modular |
| Open .nfrec file | index.html:1192 | load a local file | loads, "digest OK" | WORKS | v3r-modular `replayFileInput` |
| `?replay=URL` | replay.js:438 | auto-load | loads and plays | WORKS | replayparam.jsonl |
| Replay Play/Pause | index.html:1199 | toggle playback | paused clock holds (5.78→5.78 s) | WORKS | v3r-modular |
| Replay seek | index.html:1200 | jump | 0.8 → 38.34/47.92 s | WORKS | v3r-modular |
| Replay speed | index.html:1202 | playback rate | 5x: 5.1 s of recording per 1 s of wall time | WORKS | v3r-modular |
| Navbar Pause / Speed during a replay | app.js:6590, 5050 | control the replay | Pause toggles the replay, not the daemon; Speed 2 sets the replay to 2x | WORKS | v3r-modular |
| Assay card during a replay | app.js:5072-5076 | blocked, with an explanation | blocked, but the explanation is overwritten by the replay's state label within a frame, so nothing is visible | NO VISIBLE EFFECT | v3r-modular `card-click-in-replay` |
| Exit replay | index.html:1210 | return to live | reconnected, LIVE DAEMON | WORKS | v3r-modular |
| Error banner Resume view | index.html:1173, app.js:6201 | re-enable the view | the canvas updates again (after `?inject=render` and after the 3D fault) | WORKS | v3r-modular |
| Error banner Dismiss | index.html:1174 | hide the banner | hidden | WORKS | v3r-modular |
| Preview wall-reflex checkbox | index.html:1185, app.js:6207 | local preview reflex only (says so) | live: no command, daemon assists unchanged, but it stays ticked next to "Assists: OFF" on graph runs. Preview: toggles the local flag | WORKS (preview only) | live-* `previewWallAssist` |
| Inspect Fly tool | index.html:1215, app.js:655 | inspect the fly | selects "select" mode; clicking the fly does nothing in live or preview mode, and no inspector exists | NO VISIBLE EFFECT | live-* `toolSelect` |
| Drop Food (Odor A) | index.html:1216, app.js:660 | place food | live open arena: `place_stimulus food`, scene food 2→3 (modular and connectome). Preview: local food 1→2. Other assays: disabled (tooltip only) | WORKS | live-* `toolFood` |
| Alarm Pheromone (Odor B) | index.html:1217 | place a hazard odor | hazards 2→3 on both backends; preview 0→1 | WORKS | live-* `toolAlarm` |
| Deploy Threat | index.html:1218, app.js:5787 | place a predator | live: always disabled, but looks enabled; the click keeps the previous tool, and the next arena click placed an **alarm** instead. Preview: predators 1→2 | MISLEADING | live-* `toolPredator` |
| Drag Wind | index.html:1219, arena daemon handler neurofly_daemon.py:1765 | drag to set wind | a single click (not a drag) sets the direction toward the centre at a fixed 15 mm/s, overriding the Airflow slider. Wind changed (-0.3,0)→(-11.8,-9.3) | WORKS (label says drag) | live-* `toolWind` |
| View: 2D/3D toggle | index.html:1220, app.js:6565, **6471** | 3D articulated view | TypeError `fly.pos is undefined` on the first frame; RENDER ERROR banner; black 3D canvas, fly at (0,0,0); the 2D view stays frozen after switching back until Resume view. Modular, connectome and preview | BROKEN | v3r-modular `3d-on`/`3d-off`, `modular-3d-on.png` |
| Cam: Orbit/Chase | index.html:1221, app.js:6579 | change the camera | label and mode toggle; the camera does not move (rendering is suspended by the fault above) | NO VISIBLE EFFECT | live-* `btnCameraMode` |
| 14 experiment cards | index.html:950-1079, app.js:5061, 4355 | switch assay | all 14 switched on modular (≤0.6 s) and connectome-fixed (≤1.6 s): daemon `active_paradigm`, badge, active card and arena follow. Rapid burst of 4: landed on the last one on both backends | WORKS | live-* `card:*`, `rapid-card-switch` |
| Lesion WT / ΔMB / ΔCX / ΔGF / ΔJO | index.html:1091-1095, app.js:5307 | genetic lesion | live: disabled but looks enabled, nothing happens (the page footer says they act on the standalone preview). Preview: flags set and read by the local engine | MISLEADING (live) / WORKS (preview) | live-modular `btnLesion*`, preview |
| Lesion ΔOFF (T5) | index.html:1096, app.js:684-688, 4988 | T5 OFF-pathway lesion | sets `lesion='DELTA_OFF'`, which nothing reads; no effect even in preview, while the card describes a deficit | DEAD | preview `btnLesionOFF`, grep |
| Science guide open / × / Esc / backdrop | index.html:1114, 1708, app.js:5332-5348 | modal | opens and closes all four ways | WORKS | live-modular |
| Deck tabs (Guide / Assay Tools / Limb / Training) | index.html:1280-1283, app.js:5101 | switch panels | the correct panel is shown, the others hidden | WORKS | live-* `tab*` |
| Guide "Standalone preview parameters" sliders (2 per assay) | app.js:5266 | tune the preview | live: disabled (labelled standalone). Preview: wind slider set the local wind to (-40,0) | WORKS (preview only) | preview `guide:*` |
| Live assay parameter sliders (6 parameters in 5 assays) | live_assays.js:21, daemon set_param | change the model | windStrength, floorTemp, windVelocity, plumeWidth, patternSpeed and gapWidth: `set_param` ok, value echoed in telemetry, "Applied and recorded" (both backends) | WORKS | live-* `live_param:*` |
| Live assay interventions (10 actions in 8 assays) | live_assays.js:22, assay_controls.py:72 | intervene | food, reverse_arms, rotate_stripes, buridan contrast, reverse_heat, shift_plume, loom, reverse_grating, optomotor contrast and receptivity all applied; scene or assay state changed. Optomotor "contrast" toggles 0.9→0→1.0 (it does not restore 0.9) and is not drawn on screen; it is visible only as an HS rate change | WORKS | live-* `live_action:*` |
| Legacy assay panel (14 assays × 2 sliders + 3 buttons) | app.js:4433-5000, 5509-5540 | preview levers | mounted only without a live daemon; all 84 controls ran without errors in preview. Their `inject_stimulus` and `set_param` branches can never fire (see DEAD API) | WORKS (preview only) | preview `preview-assay:*` |
| Enable Manual Stim | index.html:1357, app.js:5125 | hand-drive the premotor neurons | live: disabled but looks enabled. Preview: the label switches to "DIRECT NEURO-STIMULATION" in every assay, but the overrides are read only in multisensory-sandbox (app.js:1956) | MISLEADING | preview, live-modular |
| DNa02 / DNp09 / MDN sliders | index.html:1398-1414, app.js:5155-5175 | override drives | values are stored; effective only in preview multisensory-sandbox; live disabled | MISLEADING (outside the sandbox) | preview |
| CPG cadence slider | index.html:1422, app.js:5176 | gait frequency | preview: `cpg.baseFreq` 8→14; live disabled | WORKS (preview only) | preview |
| Wing extension slider | index.html:1430, app.js:5183 | song wing angle | `overrideWings` is stored and never read anywhere | DEAD | preview, grep |
| Trigger GF Looming | index.html:1440, app.js:5197 | escape | preview: `escapeActive=true`; live disabled | WORKS (preview only) | preview |
| Thermal Flash (38°C) | index.html:1441, app.js:5204 | heat pulse | preview: sets `temp=40` (the label says 38), which the sandbox step overwrites on the next tick; only the MB plasticity call persists | MISLEADING | preview |
| Food Odor Puff | index.html:1442, app.js:5212 | odor and pursuit | preview: DNp09 forced to 65 is overwritten before the next frame (observed 5.36→4.35) | NO VISIBLE EFFECT | preview |
| Anemotaxic Wind Gust | index.html:1443, app.js:5218 | 2 s gust | preview: wind (-35,0) for 2 s, then reset to a hard-coded (-15,0); the user's wind of (-40,0) was lost | BROKEN | preview |
| Download Telemetry (CSV) | index.html:1266, app.js:5396 | CSV | 2863 rows with identity columns (run, brain, backend, segment) | WORKS | live-modular `csv-content` |
| Export Trial Summary (JSON) | index.html:1267, app.js:5401 | JSON | identity, manifest and telemetry included | WORKS | live-modular `json-content` |
| Clear Buffer | index.html:1268, app.js:5406 | clear | count dropped to 8 and refilled | WORKS | live-modular |
| Teach A+ / B− | index.html:1299, training.js:47 | condition the retained brain | modular: arena paused, 8 pairs, discrimination 0.51→1.88. connectome-fixed: enabled, then fails with "acts on the modular mushroom body…" | WORKS (modular) / MISLEADING (graph) | live-* `trainingTeach` |
| Reverse cues | index.html:1300, training.js:48 | reverse | modular: discrimination 1.88→−1.88; graph: same refusal as Teach | WORKS (modular) / MISLEADING (graph) | live-* |
| Probe memory | index.html:1301, training.js:49 | probe | modular: probe logged; graph: refusal after the click | WORKS (modular) / MISLEADING (graph) | live-* |
| Freeze / Enable learning | index.html:1302, training.js:50 | toggle plasticity | modular: `learning_enabled` true↔false, label flips. connectome-fixed: "Recorded" although fixed weights have no learning | WORKS (modular) / MISLEADING (fixed graph) | live-* |
| Save brain | index.html:1303, training.js:51 | checkpoint | `checkpoints/checkpoint_open-arena_instrument_*.json` written (both backends) | WORKS | live-* `trainingSave-files` |
| Export snapshot | index.html:1304, training.js:52 | JSON | 127 KB (modular), 52 KB (graph) | WORKS | live-* `trainingExport` |
| "Download research summary" / "Shareable data bundle (ZIP)" | index.html:1335 | download | HTTP 404 on 8840 and 8841; the files exist only while `scripts/research_worker.py` runs | BROKEN (default setup) | live-modular `research-download-links` |
| Brain-activity raster | index.html:1682, replay.js:121 | spike raster for recorded graph runs | replay of a connectome recording: 883 neurons, 9601 spikes drawn; hidden again after Exit | WORKS | raster.jsonl |

## Experiment studio (`web/studio.html`, `web/studio.js`, `web/embodied_replay.*`, `neurofly_studio/`)

| control | where (file:line) | expected | observed | verdict | evidence |
|---|---|---|---|---|---|
| Tabs Gallery / Build / Queue / Compare | studio.html:94-97, studio.js:29 | switch section | the correct section is visible; `aria-selected` follows | WORKS | studio-build, studio-results |
| Gallery "Start here" (curated) | studio.js:79 | curated demo runs | master ships no `experiment_data/curated`, so every new user sees "No curated runs are installed yet". After CLI curation, the cards render | WORKS (empty by default) | studio-build `studio-load` |
| Watch (gallery and queue) | studio.js:70, 106 | open the 3D replay | `embodied_replay.html?src=…` opens a new tab: 25 frames, SHA-256 verified | WORKS | studio-results `Watch` |
| Download (bundle) | studio.js:72 | zip | `auditd-intro.zip`, 8 files, no local paths or account name | WORKS | studio-results `Download(bundle)` |
| Compare with pair | studio.js:215 | open both runs side by side | Compare tab; intact on the left, control on the right; metrics table filled | WORKS | studio-results |
| Compare… (unpaired run) | studio.js:73 | pick a second run | not exercised: every run had a pair | NOT TESTED | — |
| Build button (per paradigm) | studio.js:250 | open the builder | only optomotor is buildable; 13 cards say "Not in the studio yet" (stated) | WORKS | studio-build `build-catalog` |
| Parameter slider ↔ number | studio.js:289 | stay in sync | range −6 → number −6; number 0.5 → range 0.5; seed range hidden | WORKS | studio-build |
| Silence selects (5 groups × both/L/R) | studio.js:280 | clamp cell types | silencing DNa02 switches the control to "intact"; un-silencing switches it back | WORKS | studio-build |
| Control radios | studio.js:296 | choose a matched control | "intact" disabled until something is silenced | WORKS | studio-build |
| Repeats select | studio.html:107 | seeds in a row | 2 repeats with a control queued 4 runs | WORKS | studio-build |
| Name field | studio.html:104 | run title | used as the queue and card title | WORKS | studio-build |
| Queue experiment (submit) | studio.js:318 | validate and queue | valid: "Queued 2 runs". Contrast 3: "Not queued: contrast: between 0.0 and 1.0" | WORKS | studio-build |
| Queue table | studio.js:100 | live status | pending / running / done rows; the worker line updates | WORKS | studio-build `queue-rows` |
| Cancel (pending run) | studio.js:230 | remove from the queue | 4 pending runs removed; running and done runs have no Cancel (stated: "You can close this page") | WORKS | studio-build `Cancel` |
| Real runs (worker) | neurofly_studio/server.py | execute | 4 runs finished in about 35 s wall each; the DNa02 clamp held (2 neurons, 0 spikes) | WORKS | studio-cmp |
| Compare selects (left / right) | studio.js:211-212 | choose runs | metrics table, including silencing rows; the selection survives the 5 s auto-refresh | WORKS | studio-cmp |
| Play both / Pause both | studio.js:206 | synced playback | both frames advance equally (9/9 → 17/17); a pause holds (17/17 after 1.2 s) | WORKS | studio-results `synced compare` |
| Synced seek | studio.js:207 | seek both | 50% puts both on frame 11 | WORKS | studio-results |
| Synced speed | studio.js:197 | rate | 0.25x and 4x respected; stops at the end | WORKS | studio-results |
| Embodied replay: Open .nfbody | embodied_replay.html:51 | load a file | 25 frames, verified | WORKS | embodied.jsonl |
| Embodied replay: Play / Space / Seek / Speed | embodied_replay.js:272-276 | transport | play advanced to the last frame; a pause holds; seek to 0 works; Space toggles playback | WORKS | studio-results, embodied.jsonl |
| Embodied replay: Follow fly | embodied_replay.html:85 | camera follow | the checkbox toggles; the camera effect could not be verified headless | NOT VERIFIED | embodied.jsonl |

Studio observation (not a control fault): in the curated pair, the intact fly and the
"brain disconnected from legs" control give identical metrics to three decimals (turning
−0.006 rad/s each, 39,043 spikes each) over 0.5 s. A beginner pressing "Compare with pair"
to "see what the brain contributes" sees no difference at this duration.

## Daemon API actions (`POST /api/command`, `neurofly_daemon.py:1650-1820`) and endpoints

| action / endpoint | where | UI caller | observed | verdict |
|---|---|---|---|---|
| `switch_paradigm` | daemon:1655 | assay cards | all 14 assays on each backend plus a rapid burst on each, all acknowledged | WORKS |
| `switch_backend` | daemon:1668 | Controller dropdown | modular↔fixed↔plastic; refuses while recording | WORKS |
| `switch_controller` (alias) | daemon:1668 | none (no caller anywhere outside the daemon) | — | DEAD |
| `probe_brain`, `teach_brain` | daemon:1685-1699 | Training tab | modular OK; refused on graph backends while the buttons stay enabled | WORKS (modular) |
| `set_learning` | daemon:1701 | Freeze learning | works; a no-op on connectome-fixed | WORKS |
| `set_paused`, `set_speed` | daemon:1715, 1723 | navbar | as above | WORKS |
| `set_param`, `assay_action` | daemon:1734 | live assay panel | as above | WORKS |
| `place_stimulus` | daemon:1748 | arena tools (open arena only) | food / alarm / wind OK; predator not supported (the tool is disabled) | WORKS |
| `inject_stimulus` | daemon:1770 | legacy preview panel only (app.js:4482, 4567, 4708, 4913-4915), which is never mounted while connected | always returns an error by design | DEAD |
| `reset_trial` | daemon:1773 | Reset Trial | works; the `keep_memory=false` variant has no UI | WORKS |
| `record_start`, `record_stop` | daemon:1795, 1805 | **none in the dashboard** (CLI `neurofly record` and `--record` only) | works over the API (2397-frame and 154-frame recordings, replayed fine) | NOT REACHABLE FROM UI |
| `save_checkpoint` | daemon:1811 | Save brain | works | WORKS |
| `POST /api/controller` | daemon:2290 | dropdown fallback when not connected, which goes to the page's own origin, where only the daemon-served page could answer | static page: 501 | DEAD (in practice) |
| `GET /api/telemetry`, `/api/paradigms` | daemon:2056, 2125 | none (scripts only) | — | no UI caller |
| `GET /api/brains` | daemon:2088 | none anywhere (`/api/brain` is used by `scripts/observatory.py`) | — | DEAD |

## Code that exists but cannot be reached

- `web/research/app.js` and `web/research/style.css`: `web/research.html` immediately
  redirects to `index.html`, and nothing else loads them. They are a second, orphaned
  teach/probe/freeze/save UI. **DEAD.**
- The legacy `ASSAY_CONFIGS` panel (app.js:4433-5000): reachable only in standalone preview. Its
  daemon branches (`inject_stimulus`, `set_param`) can never fire.
- `DELTA_OFF` lesion and `overrideWings`: written, never read.
- `overrideGf` and `overrideLegs` (app.js:1092, 1094): initialised in paradigm state. `overrideGf`
  is read in the sandbox (app.js:1966), but no UI control sets it (the GF button writes `dn`
  directly). `overrideLegs` is never read.
- Root-level legacy pages `flybrain_scientific_instrument.html`, `viewer.html`, `learning.html`
  and `demo.html`: not served by the daemon (404) and not linked from the dashboard. The
  instrument page still sends `inject_stimulus`. Outside area D's scope; listed for the cleanup.
- The `assay_action` / `set_param` paths have no UI in 6 assays (y-maze, circadian-dam,
  labyrinth, multisensory-sandbox; t-maze and others have actions only). This is stated in the
  panel ("No adjustable model parameters in this assay").

## Verdict counts (rows in the three tables above)

| verdict | dashboard | studio | API | total |
|---|---|---|---|---|
| WORKS (including "preview only", "partly tested", "label says drag" and "empty by default") | 38 | 21 | 9 | 68 |
| MISLEADING (including mixed rows marked "MISLEADING (graph / live / fixed graph)") | 11 | 0 | 0 | 11 |
| BROKEN | 4 | 0 | 0 | 4 |
| NO VISIBLE EFFECT | 4 | 0 | 0 | 4 |
| DEAD | 2 | 0 | 4 | 6 |
| NOT REACHABLE FROM UI / no UI caller | 0 | 0 | 2 | 2 |
| NOT TESTED / NOT VERIFIED | 0 | 2 | 0 | 2 |
| **rows** | 59 | 23 | 15 | 97 |

Mixed rows count once under their worst verdict. Rows marked "WORKS (modular) / MISLEADING
(graph)" count as MISLEADING.

## Recommendations (report only; nothing was changed)

**Remove** (dead, or misleading with no real function behind it):
- The ΔOFF (T5) lesion button and its `LESION_INFO` card (no implementation).
- The Wing Extension slider (`overrideWings` is never read).
- The "Inspect Fly" tool, unless an inspector is built (it does nothing).
- `web/research/app.js` and `style.css` (orphaned duplicate UI).
- The `switch_controller` alias, the `/api/brains` route, and the `inject_stimulus` action
  together with the legacy panel's calls to it.
- The `POST /api/controller` fallback in the dropdown (it can only reach a static server).
- The research download links, or make them conditional on the worker's files existing.

**Repair** (real function, wrong behaviour):
- The 3D viewport: `updatePose()` must read `fly.x`/`fly.y` (app.js:6471). Today the toggle
  freezes the whole view.
- Live-disabled controls (Deploy Threat, lesions, Limb deck, guide sliders): give them a visible
  disabled style or hide them in live mode. A disabled Deploy Threat must not leave the
  previous tool armed.
- Training tab on graph backends: disable Teach, Reverse and Probe (and Freeze on
  connectome-fixed) with the reason shown, instead of failing after the click.
- Controller dropdown: add `connectome-with-trained-readout` and show refusal messages.
- Manual Stim and its sliders: either work in every assay or say "multisensory sandbox only".
  The Thermal Flash label (38°C) does not match the value set (40) and is overwritten on the
  next tick. Odor Puff is overwritten before it can be seen.
- Wind Gust: restore the previous wind, not a hard-coded (-15,0).
- "Drag Wind": rename it ("Click to set wind") or implement a drag; it also overrides the
  Airflow slider's magnitude.
- The blocked assay-card click during a replay needs a message that stays visible.
- Recording has no dashboard control: a beginner cannot make the `.nfrec` files that the
  Replay button lists. Add Record/Stop, or say in the chooser how to record.
- Studio: ship at least one curated pair, so "Start here" is not empty for every new user.
