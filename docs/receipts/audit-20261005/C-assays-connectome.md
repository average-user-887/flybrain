# Audit C: the 14 assays on the connectome backends

**Date:** 5 October 2026 · **Code:** `origin/master` at `b0abd8f`, unmodified · **Graph:** MaleCNS v1.0
(`graph_sha256 eab575f6…`), LIF v3 on CUDA (GTX 1660 Ti) · **Scope:** report only, nothing fixed.

## How this was measured

1. **Headless, real daemon code path.** A `ContinuousExperimentRunner` was built exactly as the
   daemon builds it (`connectome-fixed`, then `connectome-plastic`) and every assay was stepped
   through `step_once()`: 3,000 steps (60 s simulated, three to eight trials of up to 20 s) per
   assay on fixed and 1,000 steps on plastic (7 assays). The input currents handed to the graph were
   captured at every step. Evidence: `C-evidence/headless-fixed-3000steps.json`,
   `headless-plastic-1000steps.json`, script `harness.py`.
2. **Channel probe.** 500 steps per assay, recording the largest current put into each sensory
   channel, the spikes of each channel and of each DN, and the cell types that fired most
   (`channels-fixed.json`, `channels.py`).
3. **Persistence probe.** Each encoder channel was driven for 1 s from rest and then switched off
   for 2 s (`persistence-probe.json`, `storm.py`).
4. **Optomotor direction check.** Drum at +30, −30 and 0 deg/s on fresh instances
   (`optomotor-direction.json`, `opto.py`).
5. **Real Firefox (Selenium, headless)** against a daemon started the way the owner's
   observatory is started (`--paradigm t-maze --speed 3 --continuous --trial-seconds 120`, default
   backend, which is connectome-fixed). I clicked each of the 14 assay cards and sampled the page
   4 times over 30 s, then switched the controller to Connectome Plastic and visited t-maze,
   buridan and optomotor before restoring fixed + t-maze (`browser-walk-fixed.json`,
   `browser-walk-plastic.json`, three screenshots).
   The daemon log showed **no errors, exceptions or halts**, and the dashboard's error banner never
   appeared.

## How the graph backend actually drives the fly (applies to every assay except optomotor)

`GraphArenaController.__call__` (`neurofly_daemon.py`):

- **Tonic motor injection (confirms audit A).** `currents[DNb01] += 12.0` on every step (l. 412),
  and `forward_speed = min(35, max(0, 1.5·DNp09 + 0.4·DNb01 + 5.0))` (l. 443). Both DNb01 neurons
  fire once per 20 ms step because of that current, so the "decoded" rate is 50 Hz, which gives
  0.4·50 + 5 = **25 mm/s**, and 35 mm/s when a step has two spikes. The fly's forward motion is
  current injected into the output neuron and read straight back out. The connectome computes none
  of it. A correction to audit A: in the four assays where the graph goes into the high-activity
  state described below (open arena, heat maze, wind tunnel, courtship), the network inhibits DNb01
  (8 spikes in 500 steps instead of 1,014). Speed then falls to the **+5 mm/s floor**, sometimes
  15, and DNa02 fires sporadically, so those flies do turn.
- **Yaw** = 0.02·(DNa02_L − DNa02_R) Hz. With a silent graph it is exactly 0, so the fly walks in a
  straight line. Graph backends switch the wall-avoidance and contact-turn assists **off** by design
  (`Arena.MOTOR_ASSIST_DEFAULT_ON`), so the fly drives into the first wall and stays there.
- **Sensory "encoders"** are hand-written gains that inject current into whole annotated cell types
  (l. 340–410): ORN_DM1 for food (35×conc), ORN_DA2 for danger, ORN_DA1 for cVA, the first 50
  `JO-*` cells for wind (0.15×wind_speed), thermosensory class for |T−24|, all 311 LC4/LPLC2 cells
  for looming once θ > 0.15 rad, and ER4d/ER2 at a constant 20 plus EL at 18 whenever stripes
  exist. The positions and bearings of the stripes are not used. **None of these maps is verified**
  (`CAPABILITY_MATRIX.md`: "IO not verified"). The photoreceptor branch
  (`retina_photoreceptors_l/_r`, l. 355) is dead code because no paradigm produces those keys. The
  wind branch reads `wind_speed`, but the multisensory sandbox emits `wind_magnitude` (key
  mismatch). `feco_proprio` resolves to 0 cells.
- **The code contradicts its own documentation.** The class docstring (l. 119) says "Every other
  assay has no verified mapping, so no sensory drive is delivered and no motor command is decoded:
  the fly receives zero speed and zero yaw (`graph-unmapped-io`)". The code above does the opposite,
  and `graph-unmapped-io` is never emitted for these assays. `ENGINEERED_ASSISTANCE_ENABLED = False`
  (l. 227) and the comment in `arena.py:1316` ("applied as given: no floor") both describe a
  controller that does not exist: the DNb01 drive and the +5 floor are engineered assistance.

### Two engine-level facts that shape every result (science, not UI)

- **All-or-nothing ignition with latching.** Driving ORN_DM1 at 6 (mV-equivalent) does nothing. At
  8 it ignites a state of about 6,400 spikes per 20 ms step (about 320,000 spikes/s across about
  9,000 neurons, dominated by antennal-lobe local neurons lLN1_bc, lLN2P_*, lLN2X*). That state
  **persists unchanged for at least 2 s after the stimulus is removed** (off-period mean 6,394 /
  6,400 / 6,396 spikes per step after food, thermo or cVA drive). Food odour, heat and cVA produce
  the **same** state: the same top cell types, and EPG/ER/EL counts identical to within a few
  percent. In that state DNa02 fires sparsely with a consistent left bias (L:R = 27:9, 22:11, 18:9,
  102:15). Looming drive latches a smaller state (about 960 spikes per step after offset). ER-ring
  drive does not spread: ER fires, EPG stays silent, and everything stops at offset. This is the v3
  engine running on the unmodified scan. It is a modelling result (v3 has no spontaneous activity
  and no stabilising mechanism that stops this runaway), not a dashboard bug. Its consequence is
  that, once ignited, the brain's activity carries **no information about the stimulus**.
- **Silence below threshold.** v3 has zero spontaneous activity. In 8 of the 14 assays no encoder
  ever reaches threshold, so exactly two neurons in the whole graph ever spike: the two injected
  DNb01 cells.

## Per-assay results (connectome-fixed; plastic in the next section)

"Contact" is the fraction of sampled steps in which the fly touched a wall.

| # | Assay | Claim (card / guide) | Stimulus that reaches the graph | Brain | Body / arena | Metric over trials | Verdict | Cause |
|---|---|---|---|---|---|---|---|---|
| 0 | open-arena | Multi-modal foraging | ORN_DM1 and ORN_DA2 at up to 35 and 31 (odour ≤ 1.0) | Ignites the latched state: 6,565 spikes/step, about 9,100 neurons; DNa02 L/R ≤ 150/100 Hz, always L-biased | 5–15 mm/s (floor), yaw ≤ 3 rad/s, contact 40 % | none defined | **MISLEADING** | Motion is the +5 floor plus stimulus-independent latched activity, while the identity bar says "Assistance OFF, Motor: graph". MB panel shows modular KC activity (63/120 active). Software (invented encoder, floor) plus engine (latching). |
| 1 | t-maze | Pavlovian odour conditioning, PI | ORN_DM1/DA2 at ≤ 1.55 / 1.78 (odour at the fly ≤ 0.044): **subthreshold** | Only the 2 injected DNb01 fire (2 spikes/step); DNa02 never | Runs up the stem at 25 mm/s, pins on the far junction wall (contact 92 %), never turns | PI 0.00, 0 choices, 3/3 trials | **NO BEHAVIOUR** | Software: encoder gain cannot reach threshold at maze concentrations, and the only motion is the tonic DNb01 injection. No MB learning exists on the connectome (matrix). |
| 2 | y-maze | Spontaneous alternation | Nothing (no odour, no encoder) | 2 DNb01 spikes/step | Straight run, pinned (contact 97 %) | SAR 0.00, 0 triads | **NO BEHAVIOUR** | No sensory mapping exists. Honest scientific answer: nothing to do. Motion is still the injected drive. |
| 3 | heat-maze | Thermal place memory | Thermosensory class (25 cells) at ≤ 29.6 | Ignites the same latched AL state (6,543/step); sporadic DNa02, MDN once (REVERSE) | 5–25 mm/s, contact 46 % | refuge never reached in 3/3 trials; "time in target quadrant 98 %" | **MISLEADING** | Spawn is the centre with heading 0 (east), which is inside the target-quadrant zone, so the 98 % comes from geometry, not memory. A thermal input produces odour-network activity. Software plus engine. |
| 4 | buridan | Stripe fixation (visual) | ER4d/ER2 (67 cells) at constant 20 and EL at 18 whenever stripes exist. **Not stripe-dependent** | ER fires about 124 spikes/step, EPG **0**, DNs 0 | 25 mm/s straight east, pinned at x = 108.5 (contact 90 %) | centrophobism 0.95, "stripe fixation 1.0", 3/3 trials | **MISLEADING** | The pinned fly scores as a perfect "wild-type" centrophobe. There is no visual pathway (photoreceptor code dead; ER drive bypasses vision). Software. |
| 5 | visual-operant | Operant torque learning, LI | Nothing (no visual or laser input; the fly never enters a punished quadrant) | 2 DNb01 spikes/step | Tethered, yaw exactly 0 | **LI 1.00, 100 % safe**, 3/3 trials | **MISLEADING** | A motionless fly reports perfect learning. Software (metric has no "no torque" guard; no encoder). |
| 6 | wind-tunnel | Surge-cast plume tracking | ORN_DM1 ≤ 17.3 (ignites); JO wind at 3.75: **subthreshold** (25 mm/s × 0.15) | Latched AL state; JO cells silent | 5–25 mm/s, contact 39 %, MDN reversals | upwind progress 44 mm, surge/cast 0.36–0.52 | **MISLEADING** | Spawn faces upwind (heading 0), so the floor drive alone produces "upwind progress". Wind never reaches the graph. Software. |
| 7 | looming-escape | Looming → giant-fibre escape | All 311 LC4/LPLC2 uniformly at 15+40θ once θ > 0.15 rad | LPLC2/LC4 → **DNp01 (GF) fires through the real graph** (6,139 GF spikes in 10 s) | ESCAPE in 177/300 samples; escape run is a fixed 35 mm/s (plus a 3.5 override for 0.2 s) | escape in 8/8 trials; jump at θ ≈ 6.7° (plastic) while the UI shows "GF Thresh 65°"; TTC 260–340 ms | **PARTLY** | LC4/LPLC2 → GF is a genuine connectome path. The looming detection is a hand threshold in the encoder, the drive is bilateral and uniform, the escape motor is hard-coded, and the latched residue keeps GF firing. The UI's 65° threshold describes the modular model. |
| 8 | optomotor | Gaze stabilisation, gain | WP5 encoder: T4/T5 subtypes per eye (3,322 cells), **direction selectivity imposed by the encoder** | DNa02 L/R up to 35/22 Hz; HS about 64 Hz | Tethered, speed 0; yaw ≤ 0.7 rad/s | gain 0.20 / 0.19 / 0.24 (3 trials); drum +30 → +8.1 °/s, −30 → −2.7 °/s, 0 → 0 | **PARTLY** | Turns with the drum in both directions, but asymmetrically (left bias again). Provisional, input-imposed (matrix). Banner is honest about speed and forward drive, but EPG wedges are hard-coded to 0 in this branch. |
| 9 | gap-crossing | Tactile probing, motor planning | Nothing (`feco_proprio` = 0 cells) | 2 DNb01 spikes/step | Straight 25 mm/s along y = 10 | **"crossing_success" in 8/8 trials**, card "CROSSED" | **MISLEADING** | The injected forward drive "crosses" the gap: no probing, no leg sensing. Software. |
| 10 | circadian-dam | Sleep/wake | Nothing (photoperiod not mapped) | 2 DNb01 spikes/step | Runs to the tube end, pinned (contact 92 %) | sleep 0 min, 1 beam crossing | **NO BEHAVIOUR** | No mapping. Scientifically nothing to say; the motion is injected. |
| 11 | courtship | Courtship and wing song | ORN_DA1 (cVA) at ≤ 18.8 | Ignites the same latched AL state (6,381/step) | 5–25 mm/s, contact 89 % | courtship index 0.00, 3/3 trials | **NO BEHAVIOUR** | cVA input produces the generic latched state, not a courtship pathway (no P1 readout). Engine plus unverified encoder. |
| 12 | labyrinth | Corridor navigation | ORN_DM1 at ≤ 0.95 (odour 0.026): subthreshold | 2 DNb01 spikes/step | Pinned after 8.5 mm (contact 99 %) | goal never reached, tortuosity 1.00 | **NO BEHAVIOUR** | Encoder subthreshold; the motion is injected. |
| 13 | multisensory-sandbox | 6-limb benchmark, index /100 | ORN ≤ 2.8, cVA ≤ 1.05 (subthreshold); wind **not delivered** (`wind_magnitude` vs `wind_speed`) | 2 DNb01 spikes/step | Straight 73 mm into the wall (contact 86 %) | **benchmark 63.2/100** (63.7 live), locomotor coordination 1.00, smoothness 1.00 | **MISLEADING** | A fly that walked into a wall is scored 63/100 with "perfect coordination". Software (key mismatch, metric). |

**Verdict counts (connectome-fixed):** WORKS 0 · PARTLY 2 (looming-escape, optomotor) ·
NO BEHAVIOUR 5 (t-maze, y-maze, circadian-dam, courtship, labyrinth) · MISLEADING 7 (open-arena,
heat-maze, buridan, visual-operant, wind-tunnel, gap-crossing, multisensory-sandbox) · BROKEN 0.
Nothing crashed, the daemon log is clean, and every assay switch was acknowledged. The defects are
in what the controller does and how the results are presented, not in the plumbing.

## connectome-plastic

Headless runs (1,000 steps, 7 assays) and the browser (t-maze, buridan, optomotor) show **the same
behaviour as fixed**: t-maze pinned with PI 0, buridan pinned with centrophobism 0.95, y-maze SAR 0,
sandbox 63.1/100, looming escapes, optomotor gain 0.29. The WP6 rule (ER→EPG, 3,081 edges) changes
weights in buridan (mean Δ 0.009, 1.8 % of the HUD scale) even though EPG never fires there. In
optomotor the HUD reaches 85 % (mean Δ 0.43, max 3.3) while the compass reads 0°. In t-maze the
deltas are 0. Plasticity therefore has no visible behavioural consequence in any assay. Verdicts are
unchanged: plastic adds nothing that works.

## Is the UI honest? (cross-cutting)

- **Identity bar:** "Assistance OFF · Motor: graph · Fault: none" on all 14 assays. False for 13 of
  them: the tonic DNb01 drive and the +5 mm/s floor are engineered motor assistance. The
  `graph-unmapped-io` notice ("the graph gets no input and commands no motion") exists in `app.js`
  but is never triggered.
- **DN HUD:** shows DNa02/DNp09/MDN/GF only. DNb01, the neuron that actually moves the fly, is not
  shown, so the page displays "DNp09 0.0 Hz" while the fly runs at 25 mm/s.
- **Brain-activity panel (superclass Hz):** it works and is truthful about the counts. In the 8
  silent assays it shows one non-zero bar ("descending_neuron 0.08 Hz", the two injected DNb01), so
  it looks dead. In the latched assays it shows 37–62 Hz in efferent_ascending and 3–10 Hz
  elsewhere, with no indication that this activity is stimulus-independent and self-sustaining.
- **Central-complex compass, labelled "CONNECTOME EPG":** wedges are assigned by position in the
  neuron table (`w_idx = k·16/len`, l. 462), not by anatomical glomerulus, so the displayed bump
  angle (25°, −43°, −53°, −77° …) has no meaning. In silent assays and in optomotor (hard-coded
  zeros) it reads 0°.
- **Mushroom-body panel:** shows the modular model's KC encoding (`neural.kc_hz` from
  `fly.circuit.encode_odor`, l. 1533) on the connectome backend: 63/120 KCs "active" in open arena.
  On the connectome nothing is learned in the MB.
- **Left-panel lesion card:** says "Compact modular model … The downloaded whole connectome runs
  separately" while the connectome is selected. ΔMB/ΔCX/ΔJO/ΔOFF do nothing on graph backends;
  only ΔGF/LC4 suppresses the escape.
- **Assay cards and the director's guide:** they show modular-era claims and controls ("Odor
  associations affect steering", shock voltage, PPL1) and headline numbers (LI 1.00, centrophobism
  0.90, "CROSSED", 63.7/100) next to a fly that is not doing anything.
- **Optomotor:** the only assay with an honest banner (sub-real-time speed, forward drive gated off).

## Walls

Yes, the fly gets stuck. In 10 of the 12 free-walking assays it ends up against a wall
(contact 39–99 %; t-maze, y-maze, buridan, circadian, labyrinth, sandbox and courtship all above
85 %). With the assists off (a deliberate decision) and yaw exactly 0 from a silent graph, nothing
can turn it away. In the owner's continuous mode (120 s trials, no automatic respawn) it stays
pinned for the whole trial. The real-time factor is 0.35–0.4× in silent assays and 0.11–0.16× in
latched ones.

## Why the owner sees "none of the assays work" and a "static" page

On the default observatory (connectome-fixed, t-maze, continuous) the fly runs straight up the stem
on injected DNb01 current, hits the far wall within about 1.5 s, and stays there. Two neurons out of
166,700 fire. Every DN on the HUD reads 0, and the activity panel shows a single sliver. The leg
cadence is 0 Hz because the fly's actual speed is 0. That is the "static" page, and it is reported
accurately: on today's encoders and v3 dynamics **the connectome is silent in t-maze**. Switching
assays shows one of three pictures:

1. the same silence plus a straight run into a wall (8 assays);
2. a seizure-like latched state that any suprathreshold odour, heat or cVA drive ignites. It is
   identical across modalities, persists after the stimulus, and produces left-biased random
   turning at the 5 mm/s floor (4 assays);
3. the two partly real cases: looming, where LC4/LPLC2 input reaches the giant fibre through the
   real wiring, and optomotor, which follows the drum direction with an encoder-imposed direction
   signal.

### Science versus software

- **Scientific limits (the scan and the v3 engine):** no spontaneous activity; all-or-nothing
  ignition into a stimulus-independent latched state; no photoreceptor-to-direction computation;
  incomplete lamina. These are honest results and should be reported, not worked around.
- **Software defects:** the tonic DNb01 injection and the +5 floor presented as "graph" motor
  output with "Assistance OFF"; unverified hand-tuned encoders, some subthreshold at the
  concentrations the assays produce (t-maze, labyrinth, sandbox, wind/JO) and one key mismatch
  (`wind_magnitude`); dead photoreceptor code; ER drive that ignores the stripes; assay metrics that
  reward a pinned or motionless fly (centrophobism, operant LI, gap crossing, heat quadrant, wind
  upwind progress, sandbox score); modular panels (MB, lesion card, guide) shown on the connectome
  backend; an EPG compass with arbitrary wedge assignment; a docstring and comments that describe a
  zero-input, zero-motion controller that the code does not implement.

The honest description of today's dashboard on the connectome backend: **no assay shows a behaviour
that the connectome computes from its sensory input**. Looming is the closest: a real LC4/LPLC2 → GF
relay behind an invented trigger. Optomotor is provisional and input-imposed, as the capability
matrix already says.

## Evidence index (`C-evidence/`)

`headless-fixed-3000steps.json`, `headless-plastic-1000steps.json`: per-assay summaries.
`channels-fixed.json`: injected current, channel spikes and top cell types.
`persistence-probe.json`: ignition and latching. `optomotor-direction.json`: ±30 and 0 deg/s.
`browser-walk-fixed.json`, `browser-walk-plastic.json`: Firefox samples.
`screenshot-*.png`: t-maze pinned, buridan pinned, looming escape. Scripts: `harness.py`,
`channels.py`, `storm.py`, `opto.py`, `browser.py`, `browser_plastic.py` (paths redacted to
`<SCRATCH>`).
