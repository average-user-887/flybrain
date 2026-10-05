# Functional audit A: dashboard core (liveness, staleness, refresh)

Date: 5 October 2026. Code under test: `master` at `b0abd8f`. Report only; nothing was fixed.

## What was done

- Real Firefox (headless, geckodriver, selenium). The observer never clicked anything; it read the visible
  text, the arena state and canvas contents, and took screenshots at t = 0, 5, 15, 30 and 60 s.
- Own daemon on a private port, CPU only, started twice: `--backend modular` and `--backend connectome-fixed`
  (the prepared MaleCNS graph, LIF v3, numba CPU kernel). Pages opened three ways: served by the daemon
  itself (the README route), a static server with `?daemon=<url>`, and a static server with no `?daemon=`.
- Daemon kill and restart with two tabs open; assay switch and pause sent from outside the tabs; a
  10-minute run with CPU sampling; a checkpoint write failure (output directories made read-only, the
  same failure class as the disk-full stop seen on the owner's observatory today).
- The owner's observatory page was opened read-only twice, about 40 minutes apart, with no clicks.

## Verdict table

| # | Function | Expected (UI / docs) | Observed | Verdict | Evidence |
|---|---|---|---|---|---|
| 1 | Page loads and connects (daemon-served page, README route) | README: "open the daemon URL to see the fly, watch premotor firing rates" | Connects within 3 s. Pill `LIVE DAEMON`, address badge, identity bar filled. | WORKS | `A-modular-t60.png` |
| 2 | Arena and fly move, modular | Live pose follows the daemon | Fly moves continuously (x 1.2 -> 20.5 -> 1.7 -> 42.3 -> 18.4 mm over 60 s); 21 visible fields change; arena, scope, curve and compass canvases change. | WORKS | `A-modular-t60.png`, samples |
| 3 | Arena and fly move, connectome-fixed | README: "the fly may sit almost still" | The fly is not still: every trial it runs straight in +x at a constant 25 mm/s, reaches the wall in about 3 sim s and then sits pinned at x = 73.49 mm, speed 0, for the rest of the trial. In the 10-minute run it was pinned in 17 of 21 samples. All five displayed DN rates read 0.0 Hz the whole time. Arena, scope, KC and compass canvases identical at t = 0 and t = 60 s. | BROKEN (user sees a static fly) | `A-connectome-fixed-t60.png`; SSE trace: 25 mm/s at heading 0 with `dn_rates` all 0.0 |
| 4 | Owner's live page (connectome-fixed, t-maze, `--continuous`) | A live experiment | Step counter and sim time advance (15529 -> 19667 in 60 s; 63531 -> 64359 later). Fly at exactly (70, 55.5) mm, the top of the T stem, in every sample across about 40 minutes. MB 0/120, PAM/PPL1 0 Hz, CPG 0.0 Hz, motor trace flat, compass 0. Pill `LIVE DAEMON`, data age 0.0 s, no banner. Achieved speed 1.4x then 0.55x against 3x requested. Same position reproduced on our daemon: in t-maze the graph run walks up the stem at 25 mm/s and stops at (70, 55.5); in `--continuous` there is no respawn, so it stays there indefinitely. | MISLEADING (LIVE over a fly that cannot move) | `A-owner-live-t60.png`; two read-only observations |
| 5 | Numbers update continuously | Step, sim time, rates and metrics live | Step, sim time, FPS, trial and elapsed time update on both backends. On connectome-fixed every neural readout stays at 0 (DN rates, PAM/PPL1, KC count, CPG frequency). | WORKS (counters) / MISLEADING (readouts) | samples |
| 6 | Data-age indicator | "Time since the last valid daemon frame was received and applied" | Accurate for transport: 0.0-0.1 s while frames arrive, counts up with `(frozen)` when the daemon dies. It measures frame arrival, not whether anything in the simulation changes, so a fly frozen for 40 min shows 0.0 s. | WORKS as labelled; does not catch a frozen experiment | samples |
| 7 | Daemon killed | Disconnected state, last frame kept | Within 1 s: pill `DISCONNECTED · FROZEN VIEW`, data age turns red `(frozen)`, view keeps the last frame. Never claimed LIVE. | WORKS | `A-daemon-killed-t30.png` |
| 8 | Daemon restarted | Recovers without manual reload | Both tabs reconnected automatically, 5 s (modular) and 10 s (connectome) after the daemon answered, adopted the new run id and the daemon's assay. Back-off grows to 30 s on long outages (`web/app.js:3796-3801`), so recovery can lag up to 30 s. | WORKS | timeline samples |
| 9 | Assay switch from elsewhere, two tabs | Tabs follow without reload | External `switch_paradigm` to t-maze: both tabs changed badge, arena and fly within one 5 s sample. Pause and resume also followed in both tabs. | WORKS | timeline samples |
| 10 | Simulation thread dies (checkpoint write fails) | `SIMULATION HALTED · ERROR` pill and recovery instructions (`web/app.js:3926-3932`) | The sim thread died with `PermissionError` in `save_checkpoint`. The page shows `LIVE DAEMON · STALE DATA` (amber) with a growing data age, "Achieved 3.0x", no error banner and no halt message, for as long as it runs. `/api/status` keeps reporting `online`, `halted: false`, `error: null`. Restoring write permission did not help. The documented recovery (select an assay) answered `{"status": "ok"}` but the step count stayed at 6000. Only a daemon restart recovers. | BROKEN | `A-checkpoint-fail-t115.png`; status JSON |
| 11 | Page opened without `?daemon=` from a static server | Some clear statement that it is not live | Runs the in-browser local engine: the fly moves, the step counter climbs and the MB panel lights up, while the run-state line says "Connecting to live experiment" indefinitely and the controller dropdown shows "Connectome v3 Fixed (Real 166.7K)". The pill says `LOCAL ENGINE · DAEMON FOUND, CLICK TO CONNECT`. | MISLEADING | `A-static-no-daemon.png` |
| 12 | New code deployed while a tab is open | Updated UI | No version check or reload prompt. Scripts load without cache-busting (`web/index.html:1691-1696`) and the static server sends `Last-Modified` without `Cache-Control`, so open tabs keep the old JS until a manual reload. The owner's observatory serves the main checkout's working tree, not `master`: its `app.js` differs from `b0abd8f` and its `index.html` dates from 24 September (it shows `v1.0-release`, with no daemon-address badge). | BROKEN for deployments | file comparison; owner screenshot header |
| 13 | Long run (10 min, connectome-fixed, one tab) | Keeps updating | Updated for all 10 minutes (sim time 196 -> 821 s). Requested 5x, achieved 1.0-1.3x, shown in amber. Daemon 57-71 % of one core, about 0.8 GiB RSS. Firefox 71-90 % of one core, about 0.65 GiB, for one headless tab. Each SSE client receives about 650 KB/s (29 MB in 45 s). | WORKS (heavy) | CPU samples |
| 14 | Identity bar, connectome-fixed | Says what drives the fly | `Motor: graph`, `Assists: OFF`, `Assistance: OFF`, no banner. A tonic +12 current is injected into DNb01 on every step (`neurofly_daemon.py:410-412`). It produces the 25 mm/s run (`neurofly_daemon.py:443`: speed = 1.5·DNp09 + 0.4·DNb01 + 5), and DNb01 is not among the displayed DN rates. The controller docstring says unmapped assays get "zero speed and zero yaw ... `graph-unmapped-io`" (`neurofly_daemon.py:119-124`); the running code does not do that. | MISLEADING | telemetry `motor`, `dn_rates` |
| 15 | Panels under connectome backend | Show the running brain | "[1] Mushroom body (120 KCs)", the dopamine meters and "Wild-Type Control ... Compact modular model: 120 Kenyon cells" stay on screen. The streamed `brain` block is the modular mushroom body (`model: modular-mushroom-body`, `n_kc: 120`) even though the graph drives the fly. "Composite benchmark score 62.99 / 100" while the fly is pinned to a wall. | MISLEADING | `A-connectome-fixed-t60.png`, telemetry |
| 16 | Compute line | Which device runs the brain | On `master` it appears only as a hover tooltip on the daemon-address badge, not as visible text. With the GPU hidden it reads "GPU not used: numba.cuda.is_available() is False ... CUDARuntimeError", which does not mention that `CUDA_VISIBLE_DEVICES` was empty. On the owner's (older) page it is absent. | MISLEADING (hidden) | `/api/status` `compute`, badge title |
| 17 | Speed control at the `start_daemon.sh` default | Shows the running speed | `start_daemon.sh` defaults to 15x, but the dropdown has no 15x option and shows blank. | BROKEN (cosmetic) | `A-modular-t60.png` |
| 18 | README statements | Match behaviour | "the daemon currently listens on all network interfaces (`--host 0.0.0.0`) unless you pass `--host`" is wrong: the default is 127.0.0.1 (`neurofly_daemon.py:2331`). "The fly may sit almost still ... pick optomotor to see activity" does not describe the run-to-the-wall-and-pin behaviour, and optomotor on the graph is tethered at speed 0. "Watch premotor firing rates": on connectome-fixed they are all 0.0. | MISLEADING | README "Running the dashboard" |

## Five most serious problems, in order

1. **A dead simulation thread is reported as alive, and recovery commands lie.** Any exception outside
   `arena.step` escapes `_advance_one`/`_run_loop` (`neurofly_daemon.py:1072`, `1091`, `1287`) and kills the
   `NeuroFly-SimLoop` thread. A checkpoint write is enough (`save_checkpoint` -> `experiment_brains.py:153`).
   Only `arena.step` is wrapped by `_halt_on_error` (`neurofly_daemon.py:1253-1256`). The HTTP threads keep
   running, so the status endpoint says `online`/`halted: false`, heartbeats keep the stream open, and the page
   shows `LIVE DAEMON · STALE DATA` for ever. A switch returns `ok` and changes nothing. This is the owner's
   "frozen after a disk-full checkpoint failure" incident, reproduced.
2. **On connectome-fixed the fly cannot do anything visible.** Every displayed DN rate is 0 Hz. A hidden
   tonic DNb01 current makes the fly run straight ahead at a constant 25 mm/s with yaw 0 into the nearest wall,
   where it stays (`neurofly_daemon.py:410-412`, `442-443`). In the owner's `--continuous` t-maze run there is no
   respawn, so the fly has been fixed at (70, 55.5) mm for the whole observation. This is the owner's "static".
3. **LIVE means "frames are arriving", not "the experiment is alive".** The pill and data age only measure
   transport (`web/app.js:3907-3952`), and pose stagnation, all-zero readouts and a dead sim thread are never
   surfaced as such. The owner's page reads `LIVE DAEMON`, data age 0.0 s, over a fly that has not moved in 40 minutes.
4. **The identity bar and panels misdescribe the connectome run.** `Assistance: OFF` and `Motor: graph`
   appear while an engineered tonic drive moves the fly. The docstring promises `graph-unmapped-io` with zero
   motion for unmapped assays (`neurofly_daemon.py:119-124`), but `motor_source` is `graph` and no banner
   appears. Modular mushroom-body panels, the 120-KC wild-type text and a 62.99/100 composite score stay on screen.
5. **The page the owner looks at is not the code that was tested, and nothing tells the user to reload.**
   The observatory serves the main checkout's working tree (older `index.html` and a different `app.js` from
   `b0abd8f`). Scripts carry no version, and there is no new-version check, so a deployment reaches an open tab
   only after a manual reload. The in-browser local engine also animates a fake run when the page is opened
   without `?daemon=`, under "Connecting to live experiment" and a "Connectome v3 Fixed" label.

## Why the automated sign-off passes

`scripts/live_ui_signoff.py` checks that the step counter advances, that badges and packet assays match after
switches, and that pill and identity text are present. It never checks that the fly's position changes,
that any neural readout is non-zero, or what happens when the simulation thread dies while HTTP survives.
All seven scenarios can pass on a fly pinned to a wall.

## Not tested

- A real disk-full (ENOSPC). A read-only output directory was used instead, which fails through the same
  `save_checkpoint` path.
- Interaction through the UI controls (clicked assay switch, speed, lesions, tools). This audit was
  observation-only; switching and pausing were driven through the command API.
- GPU backend, `connectome-plastic` and `connectome-with-trained-readout`, and public/read-only mode.
- The 3D viewport. Its WebGL canvas cannot be compared through `toDataURL` without `preserveDrawingBuffer`,
  so "unchanged" for that canvas is not evidence.
- Non-headless rendering and frame rate on a real display. CPU figures are for headless Firefox.
- Side effect: while the page was open without `?daemon=`, it probed `/api/status` on the default port 8769,
  which is the page's own discovery logic. It did not connect or send commands.
