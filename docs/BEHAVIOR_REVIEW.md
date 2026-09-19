# Behavior review — 19 September 2026

The previous containment checks did **not** establish that the assays reacted
correctly. The user's concern was justified. This review separates continuity,
stimulus-response wiring, and actual task learning.

## Verified defects and corrections

* The heat maze ran a permanent backward-walking override on hot ground. It now
  uses an explicit local thermal-search reflex, with bilateral temperature inputs.
  A heated tethered fly turns instead of issuing an ineffective translation.
* Stripe and social inputs were passed through unused. Stripe orientation and
  pheromone approach/avoidance now influence the modular motor commands.
* Wind-tunnel and labyrinth `odor_conc` was copied to both antennae, deleting its
  directional information. Both antennae now sample the actual field. The
  multisensory assay now passes its own wind vector instead of the default wind.
* Weak odor steering, undamped visual feedback and constant speed near a point
  source caused poor approach or orbiting. Normalized bilateral steering, a
  visual response time constant and local approach braking correct these model
  defects. The new gains are heuristic, **not fitted biological parameters**.
* Any positive reinforcement triggered feeding; relief, crossing success and
  social reward were treated as food. Only explicit food contact now starts a
  bounded feeding bout, and metabolism advances in seconds in every arena.
* A gap's CROSS/ABORT classification did not actuate movement. The explicit
  threshold controller now pauses, crosses or turns away without changing pose
  instantaneously. This is **not learned planning or limb biomechanics**.
* Corner contact could reverse the wall-avoidance turn each tick; strong odor
  attraction could also cancel avoidance. Contact steering now takes priority
  and retains the turn direction selected from all nearby boundaries.
* T-maze choices counted occupancy ticks. They now count arm entries after a hub
  return. These repeated entries still are not independent animals or trials.
* Several sliders/actions changed browser preview state or acknowledged a no-op.
  The live tools panel now exposes only connected, validated daemon controls.
  Unsupported legacy injections return an error. Applied interventions are logged.
  The live tools plot uses measured speed/yaw instead of illustrative response curves.
* Some visible fields were stale/default values: stripe fixation, path length,
  source progress, courtship wings and scene changes now follow their daemon data.
  Open-arena neural telemetry now describes the inputs used for the recorded step.
* A restarted daemon's step-zero packet could be rejected as stale. Ordering now
  includes run identity. The live outcome chart uses actual simulation timestamps,
  missing outcomes and segment boundaries, rather than waiting indefinitely for
  trial completions in continuous mode. Motor traces sample distinct ticks.
* The DAM viewport clipped 13 of its 16 drawn tubes. The entire monitor is visible
  and explicitly labels its single simulated animal. Optomotor zero contrast now
  removes the visible grating and returns the HS proxy to its baseline; the wind
  display shows the field sampled by the fly. Unstreamed neural readouts are labeled.

## What each experiment currently establishes

| Assay | Connected response / measurement | Limit |
|---|---|---|
| Open arena | Bilateral odor steering, learned odor value, food contact, real spatial stimulus placement | Compact 2D model; not a flight simulation |
| T-maze | Odor identity, reinforcement reversal, arm entries | Association and entry statistics; not automatically a valid cohort PI |
| Y-maze | Exploration, wall avoidance, alternation sequence | No learned working-memory task |
| Heat maze | Local thermal escape and refuge detection | No visual place memory |
| Buridan | Rotatable stripes, contrast, orientation and occupancy | Innate fixation; not learned navigation |
| Visual operant | Tethered yaw, closed-loop heat sectors, heat reversal | Reflex avoidance; no pattern-specific operant memory |
| Wind tunnel | Bilateral plume signal, wind, source position/width | Stationary Gaussian plume; no turbulent filaments |
| Looming | Expanding disk and repeatable bounded escape | One stimulus presentation at a time |
| Optomotor | Grating direction/contrast changes yaw, position fixed | Reflex model, not a calibrated gain curve |
| Gap crossing | Width-dependent probe, cross or abort | Explicit controller, not learned reachability |
| Circadian DAM | Beam crossings, actual simulated minutes and immobility | No endogenous circadian oscillator or light-driven entrainment |
| Courtship | Virgin/mated cues change approach/avoidance and wing state | No courtship memory; female stationary |
| Labyrinth | Bilateral goal odor, boundaries, path and goal measurements | No stored map or multi-step planner |
| Multisensory | Actual odor/wind/temperature inputs, contact and movement | Limb animation and composite score are illustrative; not all modalities drive decisions |

Each live Director's Guide and tools panel states these limits. The compact
120-KC learning model is separate from the downloaded whole connectome.

## Reproducible checks and data interpretation

`tests/test_assay_responses.py` compares matched inputs: mirrored plumes, rotated
stripes, heat versus safe sectors, opposite social cues, narrow versus wide gaps,
visible versus zero-contrast gratings and repeated looming. It also checks all
14 capability maps, actual command effects, telemetry inputs and arm-entry counts.
The full Python suite passed **316 tests** after the sensorimotor corrections.

In the controlled foraging regression (seed 42, 20-ms integration), a conditioned
brain first encountered food at about 39.9 simulated seconds; its matched naive
control did not encounter food within 120 seconds. This is one regression fixture,
not a population result, generalization claim or proof of learning in other tasks.

`scripts/behavior_audit.py` records 60 simulated seconds in every assay with seeds
4, 5 and 6. It retains failed goal searches, state occupancy, motor trajectories,
stimuli, reflex drives and maximum step displacement. Motion validity is separate
from successful task performance. The baseline is in
`outputs/behavior-review/before.json`; final runs are recorded separately.

### Completed validation receipt

* Core sensorimotor revision: `da7d4bc`. The final behavior audit is
  `outputs/behavior-review/final.json`: **42 runs, 126,000 integration steps**,
  maximum displacement **0.070 mm per 20 ms**, no nonfinite poses or jumps.
* All three heat-maze runs reached the refuge (13.1–29.1 simulated seconds), and
  all three wind-tunnel runs reached the source (58.7 seconds). All three
  labyrinth runs failed to reach the goal within 60 seconds; their missing
  latencies remain `null`. These outcomes are retained, not turned into passes.
* `outputs/behavior-review/containment-final.txt`: **84 starts, 151,200 steps**,
  all 14 arenas contained, zero teleports; maximum continuous wall pinning 1.12 s
  against the audit's 2-s threshold. Tethered assays intentionally stay in place.
* Browser review covered every assay on an isolated daemon. Connected actions
  were exercised for landmarks, heat sectors, plume location, looming, grating
  direction/contrast and social cues without altering the user's saved brains.
* The 316-test Python suite passed after the motor repairs. After the final HS
  contrast correction, all 45 assay/maze tests passed again. The JavaScript
  harness verifies live ownership for all 14 assays, restart packet handling,
  measured chart samples, missing data and segment boundaries, plus arena stress.

The trajectories validate software continuity, not biological realism. The
multisensory composite/limb scores and several legacy assay-specific counters
remain model proxies; use the raw trajectories and explicit cue-response tests
for movement analysis. In particular, odor-presence classifications in the wind
tunnel are not an independent measurement of motor-state occupancy, and the
labyrinth's internal collision counter is not the arena integrator's contact log.

The persistent research worker now starts a separate
`outputs/research-sensorimotor-v2` cohort lineage. Older cohorts are preserved in
`outputs/research-live`; do not pool their behavior/choice statistics with v2.
Source hashes cover the motor, vision, metabolism and sensory modules as well as
the circuit and assay code. Interventions in the interactive arena do not alter
the independent background research cohorts.

## Biological references, not implementation validation

* [Álvarez-Salvado et al., 2018](https://elifesciences.org/articles/37815): odor-on
  upwind movement and odor-offset search motivate the navigation comparison.
* [Ofstad et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3169673/): visual
  place learning requires spatial-memory evidence; merely reaching a cool region
  through local thermotaxis does not reproduce that experiment.
* [Characterizing approach behavior in Buridan's paradigm](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0245990):
  landmark fixation motivates a rotation/contrast manipulation, not a target score.

The reference papers describe biological experiments; their citation is not a
claim that these simplified controllers reproduce their neural mechanisms.
