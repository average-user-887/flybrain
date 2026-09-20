# Live-UI sign-off — owner's running observatory (2026-09-20)

AGENTS.md requires the final functional verification for dashboard, rendering, control,
telemetry and experiment-switching changes to happen in the running browser. This is that
check, run against the **owner's live services**, not an isolated copy:

- Web UI: `http://127.0.0.1:8780/` (static server for `web/`), loaded fresh so the deployed
  JavaScript is what ran.
- Daemon: `http://127.0.0.1:8781` (modular backend, commands enabled, not public mode). It
  was never stopped, restarted or signalled.

**What kind of check this is.** It was an *automated* session in **real Firefox 156.0**
(`/usr/bin/firefox` driven by `/snap/bin/geckodriver` 0.37.1 through selenium 4.49), running
**headless** at a 1600×914 viewport. Every control was operated as a real DOM click or a real
`<select>` change on the deployed page — experiment cards, the speed selector, the pause
button, the error banner's "Resume view" button — and every assertion is taken from the live
page (`window.arena`, `window.neuroflyDiagnostics`, the rendered identity bar, the canvas)
cross-checked against `/api/status`. It is a real browser against the real services, but it
is **not** a human looking at the screen; see "What a human still needs to eyeball".

Harness: `scripts/live_ui_signoff.py`. Structured results: `live_ui_signoff.json`
(per-assay table, per-scenario checks, timestamps, samples). Screenshots: `*.png` in this
directory; every one listed below was opened and inspected, not merely saved.

## Result: all eight scenarios passed

| Scenario | Result | Evidence |
| --- | --- | --- |
| Fresh load of the deployed page | pass | `01-initial-load.png` |
| All 14 assays via the real experiment cards | pass (14/14) | `02-assay-optomotor.png`, `02-assay-labyrinth.png`, `03-identity-bar.png` |
| Rapid consecutive selections (5 clicks, 0.35 s apart) | pass | `04-after-rapid.png` |
| Two-tab synchronisation | pass | `05-tabA-follows-tabB.png` |
| Speed control 1x / 20x / 100x | pass | `06-speed-100x.png` |
| Pause / resume with the real button | pass | `07-paused.png` |
| Error state and recovery (browser-side injected fault) | pass | `09-error-state-injected-render-fault.png` |
| Restore of the owner's state | pass | `08-restored-open-arena-1x.png`, `10-final-state-after-signoff.png` |

### On load
Arena rendered (T-maze, the daemon's unit default at the time); the shown assay matched
`/api/status`; the sim clock and step counter advanced over 3 s; the identity bar showed
`Controller: modular` with the label "Hand-built modular controller with explicit assay
reflexes. Not the connectome.", `Assists: ON (wall_avoidance_reflex, contact_turn)`,
`Motor: modular`, `Fault: none` and the run id; `Achieved: 3.0x` and `Data age: 0.0s` were
both live; the tools panel showed all five tools; no app errors and no `console.error` output.

### The 14 assays
Each card was clicked and the switch was only accepted once the command ack's identity
matched the identity of the packet being rendered. For every assay the arena geometry
changed (14 distinct canvas hashes, 12 distinct world-bound rectangles — `y-maze`/`heat-maze`
and `looming-escape`/`visual-operant` share a bounding box but render different geometry and
different canvases), the fly stayed inside the arena bounds, UI steps and daemon steps kept
advancing across the switch, the navbar badge and identity bar matched the daemon's
`active_paradigm`, `activation` counters advanced 2 → 15 with ack == rendered packet
throughout, and no errors appeared. The per-assay table is in the JSON.

### Rapid selections
`t-maze → wind-tunnel → looming-escape → heat-maze → y-maze` at 0.35 s intervals. The UI
settled on `y-maze`, the daemon agreed, and 12 samples over the following ~5 s never showed a
packet from an earlier selection: the documented stale-identity rule rejected the in-flight
packets (`rejectedIdentityPackets` 1 → 2). Daemon `total_steps` kept increasing across the
burst, so there was no hidden reset.

### Two tabs
A second tab was opened on the same URL and switched to `buridan`. The first tab followed:
badge `BURIDAN`, packet assay `buridan`, still `LIVE DAEMON`, steps still advancing, no errors.

### Speed
Requested vs achieved was reported honestly. 1x → achieved 1.0x; 20x → 19.98–20.03x;
100x → achieved 21.9–25.3x with the daemon reporting `overloaded: true`, and the UI showing
the measured value in amber next to `Speed: 100x` (visible in `06-speed-100x.png`). The pill
stayed `● LIVE DAEMON` at every speed, data age stayed under 3 s, and the view never froze.
The host was under a concurrent CPU-heavy workload from another session, which is why 100x is
unattainable here — that is the readout doing its job, not a defect.

### Pause / resume
The real Pause button froze the clock (1688.58s) and the step counter (84429) for 4 s,
`/api/status` reported `paused: true`, the button became `Resume`, `Achieved: paused`, the
pill stayed LIVE (no false DISCONNECTED), and the last measured metrics stayed on screen
(centrophobism 0.66, wind-tunnel upwind progress 22.17 mm). Resume restarted the counters and
`/api/status` returned to `paused: false`.

### Error state
Loading `?inject=render` — a one-shot, browser-only fault built into `web/app.js`, which
never reaches the daemon — produced the banner
`RENDER ERROR: Injected renderer fault (test build) — assay open-arena · run 7deb34e1 ·
step 89278 · view suspended (render); last measured frame kept`, with working "Resume view"
and "Dismiss" buttons. Clicking the real Resume button restored live rendering with no
further errors.

## Owner's state — restored

`outputs/observatory-live/pre-integration-restart.json` recorded assay `open-arena`,
`sim_speed 1.0`, `paused false`. Restored through the real controls and confirmed by
`/api/status`: `{"active_paradigm": "open-arena", "sim_speed": 1.0, "paused": false,
"status": "online"}`, with the dashboard showing `OPEN ARENA`, `Speed: 1x`, `Achieved: 1.0x`,
`● LIVE DAEMON` and a running Pause button (`10-final-state-after-signoff.png`).

## Saved brains — not touched by this check

Nothing under `outputs/` was written, renamed or deleted by the sign-off. A checksum listing
was taken before and after. The 8 `*.events.jsonl` files are byte-identical. The 14
`*.json` retained brains differ byte-for-byte, because the running daemon rewrites all of
them every 30 s under its own `--checkpoint-interval 30`
(`neurofly_daemon.py:796`). This was confirmed with a control: with **no browser open and no
commands sent**, all 14 changed again over 75 idle seconds. Brain identity and accumulated
history are intact — every `brain_id`, `seed` and `created_at` (2026-09-19) is unchanged and
the counters continue upward (e.g. `t-maze` 803 trials / 1 680 343 steps, `open-arena` 66
trials / 3 662 269 steps). Both listings and the continuity table are in the JSON.

## What was NOT tested (remaining limitations)

- **No human eyes on the live screen.** This was headless automation. Visual quality — colour,
  contrast, font rendering, animation smoothness, layout on a real monitor — is asserted only
  from the PNGs listed above.
- **One viewport only** (1600×914, devicePixelRatio 1). No narrow/mobile layout, no second
  monitor, no zoom level other than 100 %.
- **Firefox only.** No Chromium/WebKit run.
- **Tools panel presence, not tool behaviour.** The five tool buttons were confirmed present
  and enabled; Drop Food, Alarm Pheromone, Deploy Threat, Drag Wind and Inspect Fly were not
  exercised, and neither were the Assay Tools & Levers, Training & Data, genetic-lesion
  controls, Reset Trial or the CSV/JSON export buttons.
- **Two tabs, one direction.** Tab A was confirmed to follow a switch made in tab B. The
  reverse direction, and three or more tabs, were not tested. A genuinely stale packet was
  observed being rejected only during the rapid-selection burst.
- **No disconnect test.** The daemon must not be stopped, so DISCONNECTED/reconnect behaviour
  was not re-verified here; it is covered by the isolated-daemon receipt
  `docs/receipts/integration/firefox_headless_receipt.json`.
- **Graph/connectome backends were not exercised.** The live daemon runs the `modular`
  backend, so the synthetic-graph identity banner and the WP5 optomotor loop were not part of
  this sign-off (also covered by the isolated-daemon receipt).
- **Console-error capture has a small blind spot.** App-level errors are captured by the
  page's own global handler from the moment `app.js` parses; the `console.error` wrapper is
  installed immediately after `driver.get()` returns, so a console error emitted by other code
  in the first few milliseconds of the document would be missed. None were observed.
- **Speeds 0.5x, 2x, 3x, 5x, 10x and 50x** were not individually exercised; 1x, 20x and 100x
  were.
- **Sustained soak.** The whole session was roughly 12 minutes of wall clock; long-run drift
  was not assessed.

## What a human still needs to eyeball

1. Open `http://127.0.0.1:8780/` on the real monitor and confirm the arena animates smoothly
   rather than merely producing correct frames.
2. Click through a few assay cards and watch the transition itself — automation checks the
   settled state, not whether the change looks abrupt or flickers.
3. Exercise the five arena tools and the Assay Tools & Levers panel by hand.
4. Check the layout at the window size actually used day to day, and at a different zoom.
