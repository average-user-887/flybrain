# Audit B: the 14 assays on the modular backend (5 October 2026)

**Question.** The owner reports that "none of the assays work". The automated sign-off only
checks that each assay loads and the step counter moves. This audit asks a different question:
does each **experiment** do what it claims, judged the way a user would judge it?

**Scope.** Report only. Nothing was fixed.

## Method

- **Code:** GitHub `master` at `b0abd8f`, in a detached worktree.
- **Daemon:** `neurofly_daemon.py --backend modular --port 8821 --speed 5`, CPU only, with output
  and data in a scratch folder. The dashboard was served statically on port 8820 with
  `?daemon=http://127.0.0.1:8821`. Neither the owner's daemons nor the GPU were used.
- **Browser:** real Firefox 1600×1000, headless, driven by selenium and geckodriver.
- **Procedure for each assay:**
  1. Click its catalogue card.
  2. Open *Assay Tools & Levers* and set the speed with the real speed selector: 5x by
     default, 20x for T-maze and circadian, 10x for labyrinth, 1x for looming.
  3. Watch for 40–90 s of wall time, which is 40–450 s of simulated time and 5–100+ trials.
     Once a second, record the streamed packet (fly pose and state, `metrics`, trial number,
     trial-end transition and learning curve) and the DOM metric shown to the user.
  4. Take screenshots at the start, middle and end.
  5. Click every connected live action, and move every connected slider through a
     `change` event on the real input. Watch each for 15–20 s.
- **Extra browser checks:**
  - Gap crossing at 5 mm, set right after *Reset Trial*.
  - Looming at 1x in 0.1 s steps.
  - Switching T-maze → looming → T-maze.
  - Five rapid catalogue clicks (y-maze → buridan → courtship → labyrinth → wind-tunnel).
- **Offline control probe:** to support the suspected causes, a fresh `Arena(paradigm=…)` was
  stepped at `dt = 0.02` s for 120 s with **no** trial resets. This shows when each natural
  endpoint would fire. It does not replace the browser observation.
- **Shutdown:** the daemon, the web server and Firefox were stopped afterwards. The browser was
  left on open-arena at 5x, the daemon's starting state.

Compact screenshots are in [`B-shots/`](B-shots/). Arena crops are 70 % scale.

## The cross-cutting defect: every trial is cut off after 20 simulated seconds

Each paradigm keeps a `TrialManager` with a step limit:

- 1000 steps by default (`maze.py:784, 928, 1083, 1286, 1396, 1498, 1715, 1807, 2011`)
- 500 for looming (`maze.py:1615`)
- 1440 for circadian (`maze.py:1899`)
- 1500 for labyrinth and multisensory (`maze.py:2111, 2295`)

The daemon integrates at `dt = 0.02` s (`neurofly_daemon.py:775`) and ends the trial when the
manager runs out (`neurofly_daemon.py:1337-1339`). That gives these trial lengths:

| Paradigm | Trial length |
|---|---|
| Most paradigms | **20 s** |
| Looming | 10 s |
| Circadian | 28.8 s |
| Labyrinth and multisensory | 30 s |

The advertised `trial_length_s = 60` is never reached. The modular fly also rests most of the
time. Without odour the surge-cast engine's stop rate is 0.78/s and its walk rate 0.2/s
(`surge_cast.py:123, 127`), so it walks about 20 % of the time.

With no reset, the offline probe reaches each natural endpoint only after the cap has already
fired:

| Assay | Natural endpoint, no resets (3 seeds) | Trial cap |
|---|---|---|
| T-maze first arm choice | 27–40 s | 20 s |
| Heat-maze refuge | 24.6–28.1 s (the live brain often makes it in 11–18 s) | 20 s |
| Wind-tunnel source | 58.7 s | 20 s |
| Gap-crossing landing (3.5 mm) | 45–61 s | 20 s |
| Labyrinth goal | never in 120 s | 30 s |
| Y-maze | 1 arm entered in 120 s | 20 s |
| Circadian sleep bout | needs 5 min of immobility | 28.8 s |

In the browser this looks like this: the fly walks partway, then teleports back to the spawn
point, and the metric resets to zero or shows "Not observed". This repeats for ever.

## Results

| # | Assay | Claim (UI card / docs / code) | Observed in the browser | Verdict | Evidence | Suspected cause |
|---|---|---|---|---|---|---|
| 0 | **open-arena** | "Open Arena Foraging": bilateral odour navigation to food, learned odour value, food contact; "Place food near fly" and airflow controls. | The fly moves and steers in the odour field (state SURGE throughout). Over 7 trials of 60 s (370 s simulated) it **never ate**: the food positions never changed, and the closest approach per trial was 9.0–11.9 mm against a 3.5 mm contact radius. "Place food near fly" worked: the food was eaten within about 3 s and respawned. The airflow slider worked (speed rose from 0.2 to 2.9 mm/s). There is no assay metric (`metrics {}`), the learning curve is all `null`, and the card shows "Odor value 0.1–0.4". | **PARTLY** | `B-shots/open-arena_b_mid.jpg`. Samples: food unchanged for t = 0–74 s; eaten 3.2 s after the action. | The fly orbits slowly between the hazard and food plumes at low speed (local approach braking, `arena.py:1447-1448`). No foraging metric is defined for this paradigm. |
| 1 | **t-maze** | "T-Maze Pavlovian Conditioning" (Tully & Quinn): the fly chooses an arm, choices are recorded, PI and learning curve update; reward/shock reversal. | Over **19 trials at 20x** the fly never entered an arm. x stayed in 52.9–70.0, and an arm needs x < 45 or x > 95. Every trial ended `max_duration_steps` with `first_choice: null`, PI 0, choices 0/0. The learning curve is a flat row of **0** (not "no data"), and MB weight mean stayed 1.0, so nothing was learned. *Reverse arms* swapped the scene (`cs_plus_arm` → arm_b), but no choice followed (7 more trials, all timed out). | **BROKEN** | `B-shots/t-maze_c_end.jpg` (fly at the hub, PI 0.00, trial 71). Offline probe: first choice at 27–40 s. | 20 s cap (`maze.py:784`, `neurofly_daemon.py:1339`) shorter than the walk to an arm. The arm threshold x < 45 / x > 95 (`maze.py:875`) needs about 25 mm into an arm. With no choice, PI is reported as 0.0 (`maze.py:910`), which the curve plots as a real value. |
| 2 | **y-maze** | "Y-Maze Spontaneous Alternation": arm sequence, SAR and handedness. | 19 trials. The fly enters **at most one arm per trial, and always arm C**. The choice sequence is "C" or "", SAR 0, triads 0, handedness 0. It rests about 40 % of samples. The learning curve is all 0. | **BROKEN** | `B-shots/y-maze_c_end.jpg`. Probe: one arm in 120 s on every seed. | 20 s cap (`maze.py:928`). The spawn heading of 0 rad points straight down arm C (`arena.py:1005`), and the fly rests most of the time (`surge_cast.py:123-127`). SAR needs ≥ 3 entries in one trial. |
| 3 | **heat-maze** | "Thermal Heat-Maze **Place Learning**" (Ofstad 2011), "CX PLACE MEM": the fly escapes 36.5 °C onto a cool refuge, and latency should fall with training. | The thermal escape works. Most trials end `refuge_reached` with latency 11.3–18.1 s; about 1 in 6 time out at 20 s. Latency **does not fall** across 30 trials (11.9, 11.3, 11.9, 18.1, timeout, 11.3, …). The learning curve is all `null` (latency is not a curve metric), and MB weights are unchanged. The floor-temperature slider worked: at 30 °C the thermal dose dropped from about 57 to 24–37. | **PARTLY** (escape reflex) / MISLEADING on "place learning" | `B-shots/heat-maze_b_mid.jpg`; transition log in the samples. | No place memory is implemented (the limitation text in `assay_controls.py:9` admits it, but the card and title claim it). `escape_latency_ms` is missing from `TRIAL_METRIC_KEYS` (`neurofly_daemon.py:644-649`). |
| 4 | **buridan** | "Buridan's Landmark Fixation": walking between two opposite stripes, centrophobism, stripe crossings. | The fly spawns at the centre facing stripe 0 exactly. It walks or rests on the line y = 60 and moves at most 12 mm per 20 s trial. Centrophobism is **0.00** (100 % of time in the centre), crossings 0, and "fixation" a trivial 0.99998. *Rotate stripes* worked: the fly turned to the new stripe and one crossing was recorded. *Contrast* toggled fixation to about −0.2. | **BROKEN** (stimulus wiring is fine; the assay never develops) | `B-shots/buridan_c_end.jpg`. Metrics at 5 trial ends: CI 0, crossings 0. | Spawn on the stripe axis (`arena.py:1008`) plus the stripe yaw drive `sin(bearing)` (`assay_response.py:22`), which holds the fly on that axis. 20 s cap (`maze.py:1286`) and about 80 % REST. The fly never reaches the perimeter, so the paradigm's back-and-forth is impossible. |
| 5 | **visual-operant** | "Visual Operant Torque", "Learning Index LI" (Wolf & Heisenberg): operant conditioning of yaw to avoid heat sectors. | The tethered fly stays fixed and the drum turns. LI is **0.94 from the first trial** and flat over 18 trials (0.938–0.946). MB weights are unchanged. *Reverse heat* gave LI 0.89 on the transition trial and 0.99+ on the very next trial, with no relearning period. It is an immediate heat-escape reflex, not learning. | **MISLEADING** | `B-shots/visual-operant_b_mid.jpg`. Learning curve 0.942, 0.944, 0.94, … then 0.89, 0.994, 0.998. | Heat makes the yaw a constant `1.5 × wall_turn_dir` (`assay_response.py:47-48`), which spins the drum out of the punished sector. The limitation text says "no operant memory", but the card calls it a Learning Index and the learning curve plots it. |
| 6 | **wind-tunnel** | "Wind Tunnel Odor Plume Tracking", surge-and-cast, upwind progress, source reached. | The fly surges straight upwind along the plume, **60.7 mm per trial**, every trial identical. The source is 160 mm away and is **never reached**. 100 % of steps are SURGE, so `surge_to_cast_ratio` is **1000**, and that value is plotted as the "learning curve". *Shift plume* gave casting and only 22–35 mm of progress. The airflow and plume-width sliders took effect (width 10 mm restored 100 % surge). | **PARTLY** | `B-shots/wind-tunnel_c_end.jpg`. Transitions: `upwind_progress_mm` 60.689 ×6, ratio 1000. | 20 s cap (`maze.py:1498`). Probe: source at 58.7 s. The ratio divides by `max(1, cast_steps)` (`maze.py:1595`) and goes straight into the curve (`neurofly_daemon.py:1553`, scale 1.0). |
| 7 | **looming-escape** | "Visual Looming Giant Fiber Escape" (Card & Dickinson): a looming disc expands and the fly escapes; "Jump TT Collision" metric. | The disc does appear (θ shown, GF threshold 65°). But a trial lasts **19 steps (0.38 s)**: at 1x the disc restarts about 2.6 times a second (26 trials in 10 s). The fly **never visibly escapes**: x stayed 40.0–40.5 mm, and state ESCAPE was seen in 2 of 39 samples. The card and the outcome panel always show **"Not observed"**. The stored jump TTC is a constant 20 ms. *Present looming disk again* is redundant, because trials re-loom by themselves. | **BROKEN** (as experienced) | `B-shots/extra_loom_2.jpg` (disc at 28°, metric "Not observed", outcome chart "Outcome not observed"). | The trial ends on `escape_initiated` (`neurofly_daemon.py:641, 1328-1330`). `_end_trial` respawns the fly and clears `assay_escape_remaining` (`arena.py:1081`) in the same step that started the escape, and it discards the terminal metrics from the stream (`neurofly_daemon.py:1275`). The geometric threshold makes TTC deterministic (`maze.py:1626, 1663`). |
| 8 | **optomotor** | "Optomotor Gaze Stabilization": the tethered fly turns with the drum; gain; grating direction and contrast controls. | The position is fixed and the heading turns with the grating. Gain held at **0.54–0.59** across 18 trials. *Reverse grating* reversed the turning: slip changed sign and gain stayed positive. *Contrast 0* dropped the gain to about 0.00 and HS to its 40 Hz baseline. The speed slider was tested while contrast was still 0, so that one test is uninformative. | **WORKS** | `B-shots/optomotor_b_mid.jpg`. Learning curve about 0.55 → about 0 after contrast 0. | — (the efference-copy claim was not tested; `is_saccade` never triggered) |
| 9 | **gap-crossing** | "Gap Crossing & Tactile Probing" (Pick & Strauss): the fly probes, then crosses (≤ 3.8 mm) or turns back (> 4.2 mm). | At 3.5 mm, **2 of 16 trials crossed**. The rest timed out at 20 s, mostly before reaching the gap. At 5 mm (set right after Reset Trial), 3 of 9 trials reached the gap. At the first probing step (probing 20 ms) the trial ended `decision_abort` and the fly was respawned, so the **pause and turn-back are never seen**. Changing the gap while the fly is near it is refused, and the slider snaps back ("Reset the fly before changing the gap…"); this is intended. | **PARTLY** | `B-shots/gap-crossing_c_end.jpg`, `B-shots/extra_gap5_a.jpg`. | 20 s cap (`maze.py:1807`). Probe: landing at 45–61 s. `decision_outcome == "ABORT"` ends the trial immediately (`neurofly_daemon.py:1333`), before the PROBE/ABORT motor sequence in `assay_response.py:54-68` can play. |
| 10 | **circadian-dam** | "Circadian DAM Sleep Monitor": 12:12 light/dark, sleep = ≥ 5 min immobility, sleep minutes and bouts. | Over 27 trials at 20x: **sleep minutes 0, bouts 0** every time, and 0–1 beam crossings per trial. Lights never change. The card shows "0.00 min" for ever. | **BROKEN** | `B-shots/circadian-dam_c_end.jpg`. Probe without resets for 600 s: sleep 0 (rests are about 5 s long). | Trial cap 1440 steps = 28.8 s (`maze.py:1899`) against a 5-minute sleep criterion (`maze.py:1963`). The light phase is `time/60000 % 1440` minutes (`maze.py:1927`), so lights stay on for 12 h of simulated time and never switch within a trial. |
| 11 | **courtship** | "Courtship **Conditioning** & Wing Song" (Siegel & Hall): the male courts, a mated female rejects him, and courtship is suppressed afterwards. | The default female is **mated**: the fly avoids her and CI is **0** in all 19 trials. Rejection kicks are 0. *Toggle virgin* worked immediately: the fly approaches, stops 2 mm away in COURTSHIP, and CI is 0.98 every trial. Rejection kicks are **impossible**: they need COURTSHIP state with a mated female, and that state only arises for a virgin. So no conditioning can occur. | **MISLEADING** (approach/avoid works; conditioning does not exist) | `B-shots/courtship_c_end.jpg`, `B-shots/courtship_d_live_action_receptivity.jpg`. Curve 0,0,…,0.98,0.981. | COURTSHIP is set only when `attraction > .1` (`assay_response.py:39-40`). Kicks need that state together with a mated female (`maze.py:2061, 2069`). |
| 12 | **labyrinth** | "Corridor Obstacle Labyrinth": odour-guided navigation to the goal, dead ends, collisions, tortuosity. | Over 27 trials the fly **never leaves the first corridor** (x ≤ 23, behind wall 1 at x = 25) and **never reaches the goal**. `wall_collision_count` and `dead_end_entries` are always 0. Only tortuosity (about 1.2) changes. | **BROKEN** | `B-shots/labyrinth_c_end.jpg`. Probe: no goal in 120 s on 3 seeds. | 30 s cap (`maze.py:2111`). A weak, distant goal odour and about 50 % REST. The collision counter re-checks a position the arena has already resolved, so it never fires (`maze.py:2189-2192`). `dead_end_entries` would count steps, not entries (`maze.py:2205-2208`). |
| 13 | **multisensory-sandbox** | "Multisensory 6-Limb Benchmark": a composite sensorimotor score from coordination, integration, efficiency and smoothness. | The fly moves (it surges upwind up to 64 mm), but it never reaches the food or refuge. The **composite score is highest when the fly rests at the origin** (74.9–75.3) and falls as it moves (65.6). The locomotor coordination index is **1.0000 in every sample**. Efficiency stays below 0.04. There are no controls in this panel. | **MISLEADING** | `B-shots/multisensory-sandbox_c_end.jpg`. Curve 0.75 → 0.61 as the fly travels. | All six leg phases advance by the same `dphi`, so L1/R1 are always π apart and coordination ≡ 1 (`maze.py:2452-2486`). Resting gives zero jerk, so smoothness = 1. The composite score (`maze.py:2546`) therefore rewards immobility. |

### Switching and controls

- **Switching:** T-maze → looming → T-maze landed on the right assay each time. The fly pose
  was the T-maze's own (resumed), with no browser errors.
- **Rapid switching:** five rapid clicks ended on wind-tunnel in the daemon, the packet and the
  badge. The step counter kept advancing and there were 0 entries in `neuroflyErrors`.
  `B-shots/extra_rapid_final.jpg`.
- **Controls:** every connected control acknowledged and changed the simulation as labelled:
  - food
  - reverse arms
  - floor temperature
  - rotate stripes
  - stripe contrast
  - reverse heat
  - wind velocity, plume width and shift plume
  - loom
  - reverse grating and grating contrast
  - gap width (refused near the gap, by design)
  - receptivity

  The one exception is open-arena airflow: the slider shows "0" for the real 0.3 mm/s,
  because its step is 1. **Controls are not where the failures are.**
- **Speed:** at 20x the achieved speed was 9.7x on the T-maze; 7.0x at 10x on the labyrinth.

## Counts

| Verdict | Count | Assays |
|---|---|---|
| WORKS | 1 | optomotor |
| PARTLY | 4 | open-arena, heat-maze, wind-tunnel, gap-crossing |
| BROKEN | 6 | t-maze, y-maze, buridan, looming-escape, circadian-dam, labyrinth |
| MISLEADING | 3 | visual-operant, courtship, multisensory-sandbox |
| UNTESTABLE | 0 | — |

Heat-maze is also MISLEADING on its "place learning" title.

## Most common failure causes

1. **The 20 s trial cap (8 assays).** `TrialManager` step limits are counted in 20 ms steps, so
   1000 steps = 20 s (`maze.py:623` and each paradigm's `__init__`). The daemon honours that
   limit before its own 60 s `trial_length_s` (`neurofly_daemon.py:1337-1339`). In T-maze,
   Y-maze, wind-tunnel, gap-crossing, labyrinth, circadian and Buridan, the trial ends before
   the fly can reach the outcome the assay measures. Heat-maze loses about 1 trial in 6 to it.
2. **A sluggish fly.** The surge-cast stop/go rates leave the fly walking about 20 % of the time
   outside odour (`surge_cast.py:123-127`). That makes the cap fatal and leaves "REST" as the
   most common state on screen.
3. **Trial ends that erase the outcome.** Looming and gap-ABORT end the trial in the step where
   the response starts. The fly is respawned at once (`arena.py:1070-1083`) and the terminal
   metrics are dropped from the stream (`neurofly_daemon.py:1275`). The user never sees the
   escape or the turn-back, and the card reads "Not observed".
4. **Learning claims with no learning mechanism behind them.** The titles promise place learning
   (heat-maze), operant learning (visual-operant), conditioning (T-maze, courtship) and a
   learning curve for every assay. In 13 assays the MB weight mean never moved (1.0 → 1.0); it
   changed only in multisensory (1.0 → 1.05) and the open arena. Reflex scores (LI 0.94, CI 0.98,
   ratio 1000) or zeros from "no choice made" are plotted as learning curves
   (`neurofly_daemon.py:644-649, 1553`).
5. **Degenerate metrics.** These are listed per assay above:
   - coordination ≡ 1 (`maze.py:2486`)
   - surge/cast ratio 1000 (`maze.py:1595`)
   - labyrinth collisions ≡ 0 (`maze.py:2189`)
   - PI = 0 with no choice (`maze.py:910`)
   - Buridan fixation ≈ 1 by spawn geometry
   - circadian sleep unreachable

## Limitations

- Headless Firefox. Arena behaviour was judged from the streamed packets, DOM values and
  screenshots, not by watching the animation continuously.
- One fresh daemon run with one brain per assay. The trials of one assay share that brain, as
  they do on the owner's dashboard. Brains saved on the owner's own daemons were not touched.
- The offline probe uses fresh brains and seeds 1, 2, 3 and 42. It only shows when the
  endpoints would fire. The verdicts rest on the browser runs.
- The connectome backends are out of scope; that is area B on the modular backend only.
