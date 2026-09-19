# Research data and trajectory integrity

The default local UI is the original dark instrument at `http://127.0.0.1:8780/index.html`.
The former research page redirects there. Open **Training & Data** for retained
brains, cue probes, actual weights, research cohorts, and downloads.

## Live observation

The daemon owns pose, time, trial boundaries and exported samples. Browser assay
steps are suspended while connected; an outage freezes the last frame instead
of silently substituting synthetic local measurements. The scientific CSV has
one row per received daemon tick, not one fabricated sample per animation frame.
CSV is a bounded, potentially downsampled live-view buffer, not a complete raw run.

The local service uses `--continuous`: there are no automatic 20-second respawns.
Pause freezes simulation time, learning and motion. Reset starts a new segment.
Changing experiments or restarting a process also starts a segment. Never join
positions across different `run_id`/`segment`/`brain_id` values. UI trails clear at
these boundaries. Coordinates in CSV are the original daemon coordinates in mm;
display offsets center some arenas and do not change exported measurements.

REST is a modeled stochastic locomotion state, not necessarily a simulation
failure. Feeding, tethered assays, teaching, and explicit pauses can also limit
translation. Tethered visual-operant and optomotor assays allow yaw while fixing
position. Errors are reported and stop valid-step counting. The recorded state,
speed and phase make stationary intervals interpretable.

Earlier live CSVs mixed local timing with remote poses. Treat those as unvalidated
pre-repair output. Retain them as historical evidence; use the new records with
explicit source and segment fields for analysis. This repair does not certify
previous files or validate all biological model assumptions.

## Background research

`neurofly-research.service` runs `scripts/research_worker.py` independently of the
interactive daemon. It performs one 14-assay batch, waits five minutes, then
continues. Seeds rotate across three independent cohorts. Every assay/seed has a
retained plastic brain and a separate matched frozen control. A completed pair
is the parent of its next round; interrupted/failed pairs are never parents.
Neither group modifies the interactive brain bank.

Each group gets 3,000 training steps (60 simulated seconds), then 3,000 evaluation
steps in a fresh arena with a different seed. Evaluation copies the trained
mushroom-body state, clears eligibility traces, and freezes both groups' weights.
Other transient state (hunger, locomotion, compass) starts fresh. Reinforcement
remains present during evaluation: this is not an unreinforced extinction probe.
The frozen group gets the same initial circuit/arena seed and exposure duration,
but its actions may produce different sensory histories. This is not a yoked
stimulus control. Each phase runs continuously without respawn.

At 20-ms resolution the worker checks finite positions and maximum displacement.
Raw pose/reward/metrics rows are retained every simulated second, plus the final
step. Thus the exported trajectory is explicitly downsampled; the integrity
summary covers all integration steps. Discontinuous cohorts fail quality checks
and are excluded from comparisons. Missing endpoints are `null`, not zero or
success. Frozen weights, brain identities, initial/final probes, elapsed time,
rest durations, and all raw metric keys are retained.

Outputs are append-only cohort attempts under `outputs/research-sensorimotor-v2/cohorts/`,
with `cohorts.jsonl` and a manifest. The web tab reads the atomic
`web/research-status.json`; `web/research-latest.zip` contains the latest batch's
raw phase records, initial/final checkpoint snapshots, protocol, source revision,
Python/NumPy versions, source hashes and file checksums. The download is suitable
for sharing as an exploratory simulation dataset. It is not uploaded publicly.

The older `outputs/research-live` lineage is preserved separately. Its motor and
choice statistics predate the [behavior corrections](BEHAVIOR_REVIEW.md); do not
pool them with the sensorimotor-v2 results.

```bash
# Managed worker only; interactive UI/brain remain running.
systemctl --user stop neurofly-research.service
systemctl --user start neurofly-research.service
journalctl --user -u neurofly-research.service -n 30 --no-pager

# Independent one-batch run (do not reuse the live worker's output path).
.venv/bin/python scripts/research_worker.py --once --steps 3000 --seeds 3 \
  --output outputs/my-research --public-status outputs/my-research/status.json \
  --bundle outputs/my-research/share.zip
```

The managed services resume at user login and restart on failure. They cannot run
while the computer is asleep or powered off. Use distinct output paths for
different protocols; do not combine rounds with different step/seed settings.

## Interpretation

This is the compact modular simulator (120 Kenyon cells), not learning inside the
downloaded 166,700-neuron MaleCNS graph. Odor calibration establishes association
and isolation; it does not establish place learning, planning, or learning in
every assay. Graphs show individual paired observations without invented
significance or success thresholds. A single good trajectory is not a learning
result. Short circadian windows cannot establish a daily rhythm. Curriculum and
multi-step task learning remain research targets requiring causal controls and
task-specific sensory representations.

## Long runs and validation record

Scalar means and path statistics now update incrementally while retaining only
2,048 display samples. Cumulative counts, mean values, distance and the original
path origin survive eviction; long runs do not repeatedly scan the entire path.
Heat-maze `path_length` is measured in mm. In revision `2553a9e` and older, that
particular raw metric counted samples; use the phase summary's `distance_mm`
when analyzing those historical cohorts. Each cohort identifies its code revision.

The 2026-09-19 validation included browser inspection of all 14 arenas and all
restored renderers; daemon pause/resume, eight-pair teaching and a recorded
non-mutating probe; 378,000 headless physics steps over 14 assays, three seeds
and three starting poses; plus 54,000 further steps for the updated looming and
optomotor inputs. No escapes or unexplained teleports were detected. The browser
ownership regression checks all 14 assays for 5,000 simulated calls each without
allowing the local engine to alter a remote pose, clock or recording.
The full Python suite passed with 291 tests, including cumulative-statistic
retention after display-history eviction.
