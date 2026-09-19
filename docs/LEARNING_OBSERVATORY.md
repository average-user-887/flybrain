# Learning observatory — 2026-09-19

The current development branch is `worktree-neurofly-openready`, under
`.claude/worktrees/neurofly-openready`. The main checkout is an earlier snapshot.
No Git remote is configured; commits are local.

## Run the lab

Start the persistent local lab (Linux with a systemd user session):

```bash
python3 scripts/observatory.py start
python3 scripts/observatory.py status
```

The launcher works from any directory when called by its absolute path. It
installs three managed user services, starts them, and waits for the actual page
and brain API to respond. They survive terminal/chat-session closure, restart
on process failure, and start at user login. No root privileges are needed.
The web/API services bind to localhost; the research worker uses no network listener. Existing brain data stays in `outputs/observatory-live`.
The launcher refuses to take over ports held by unrelated processes.

```bash
python3 scripts/observatory.py logs      # recent logs for all services
python3 scripts/observatory.py restart   # graceful save and restart
python3 scripts/observatory.py stop      # stop now; retain learned brains
python3 scripts/observatory.py disable   # also disable automatic login startup
```

Service names: `neurofly-observatory-brain.service` and
`neurofly-observatory-web.service`, plus `neurofly-research.service`. Their generated unit files live in
`~/.config/systemd/user/`. Automatic startup follows the logged-in user session;
it does not promise availability while the computer is asleep or powered off.
Do not start a second daemon against the same output directory.

For development on a host without systemd, use two foreground terminals:

```bash
.venv/bin/python neurofly_daemon.py --host 127.0.0.1 --port 8781 \
  --speed 3 --continuous --trial-seconds 120 --output-dir outputs/observatory-live \
  --data-dir outputs/observatory-live/learning \
  --pid-file outputs/observatory-live/daemon.pid
.venv/bin/python -m http.server 8780 --bind 127.0.0.1 --directory web
```

Those foreground processes need to stay open. The persistent launcher is the
recommended option for this workstation.

Open <http://127.0.0.1:8780/index.html>. For another daemon, append
`?daemon=http://localhost:PORT`. The UI uses that explicit endpoint only.
The original dark instrument is the primary interface. Its Training & Data tab
contains saved weights, probes and background cohorts. Pose, clock, cue activity,
compass and recorded telemetry come from the daemon while connected. Limb
geometry is illustrative. Some old assay/lesion controls remain standalone-preview
controls, as labeled; they are not validated interventions on the saved brain.
The background worker runs separately and never switches the interactive assay.
See [Research protocol](RESEARCH_PROTOCOL.md) for exports, controls and limitations.

## Each experiment owns its memory

`ExperimentBrains` lazily creates one seeded `ExperimentBrain` per paradigm.
Each owns a distinct Arena, mushroom-body circuit, central-complex working
memory, trial counter, learning curve and event history. Switching suspends the
old instance and resumes the selected instance. Only the selected experiment
runs; this is not simultaneous training of fourteen brains.

Checkpoints under `<output-dir>/brains/<paradigm>.json` contain all PN tuning,
PN→KC connectivity, KC→MBON baselines, efficacy deviations, filtered weights,
eligibility traces, CX bump/vector, brain ID, counters, controls and recent
history. Saves use fsync and atomic replacement. An invalid checkpoint is
rejected without overwriting it or silently substituting a fresh brain.
Teaching can resume at its saved phase after restart. Behavioral arenas restart
at spawn: these are learning-state checkpoints, not exact trajectory replays.

`<paradigm>.events.jsonl` is an append-only, fsync'd ledger of teaching pairs,
probes, controls and completed trials. The UI shows bounded recent history;
the full ledger remains on disk. Existing global recorder events keep their
session-wide monotonic `trial`, and additionally carry `brain_id` and
`brain_trial`. Unknown scalar outcome metrics are null, not fabricated 0.5s.

## Controlled teaching and probing

A calibration protocol exposes the selected brain to A→reward and B→punishment.
Each cue lasts 600 ms; reinforcement starts after 300 ms; 400 ms washout follows.
A complete A/B pair takes 2 simulated seconds, integrated in 10 ms bins.
The behavioral arena pauses during this protocol. Reversal swaps contingencies.
The frozen control blocks both teaching and arena synaptic updates. Pure probes
call encoding and forward readout without changing weights, traces or RNG.

This is **odor-association calibration**, shared across experiment instances.
It does **not** prove learning of visual place memory, courtship, reflexes,
maze solving, or multi-step planning. The UI provides specific interpretation
and control suggestions for all 14 environments. Some arena trials time out
without a choice; they must not be presented as successful learning trials.

## API

- `GET /api/observatory`: one consistent snapshot of status, active brain,
  telemetry and the full brain catalog, copied under a single runner lock.
- `GET /api/brains`: catalog, active/paused/saved/not-started states and summaries.
- `GET /api/brain`: active brain details, 240 actual weights and arena walls.
- Telemetry includes `brain_id` and brain summary; plasticity statistics now
  read `get_effective_weights()` instead of the nonexistent `.weights` field.
- `POST /api/command`: `switch_paradigm`, `teach_brain` (`pairs`: 1–50,
  `reverse`: boolean), `probe_brain`, `set_learning` (`enabled`: boolean),
  `save_checkpoint`. Existing commands remain available.
- Public-stream authorization applies to all new commands. The observatory
  acts as a read-only viewer when public commands need an admin token.

## Verification

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/
PYTHONPATH=. .venv/bin/python scripts/learning_battery.py
# Requires gjs (SpiderMonkey); no browser or DOM dependencies:
gjs scripts/containment_harness.js web/app.js 12 3000 --stress
```

On 2026-09-19, all 263 Python tests passed. The learning battery passed all 14 experiments: positive paired
readouts, negative reversed readouts, zero frozen-control change and exact
weight/identity restoration. Output:
`outputs/resume-20260919/learning-battery/report.json`.
These are matched seeded controls, not population-level biological evidence.

The dashboard physics passed 504,000 stressed steps across 14 paradigms with
no escapes, penetrations above the harness tolerance, teleports or wall freezes.
The local browser was used to verify actual teaching, isolated brain switching,
the frozen control, all 14 live experiment views, and restoration of the same
trained brain identity after a real daemon restart. The earlier standalone dashboard remains a separate
artifact; it is not automatically rebuilt from the newer `web/` source.

## Scientific corrections in this update

- Preserve A/B cue identities and bilateral antenna differences independently.
  Reward-contingency reversal no longer implicitly relabels physical cues.
- Integrate arena plasticity over actual elapsed seconds in legal ≤10 ms bins.
- Both MBON pathways use the same prior KC eligibility trace. Previously PAM
  advanced the shared trace and PPL1 advanced it again, creating an artificial
  timing asymmetry. The bridge regression test now uses forward conditioning
  with a 500 ms cue lead, washout and a frozen held-out readout.
- Browser collision correction projects overlap without snapping separating
  motion back into a corner. Translational collision impulses no longer add a
  heading torque that cancels wall avoidance. Departing walls cannot cancel an
  approaching wall's reflex at a junction.

## Next research milestones

1. Quantify held-out behavioral learning against frozen controls across seeds.
   The current calibration proves memory mechanics, not improved navigation.
2. Audit remaining timing conventions in vision, CX, metabolism and locomotion;
   the plasticity clock is fixed, but other modules retain legacy fixed bins.
3. Add task-specific sensory representations and teaching contingencies for
   non-olfactory assays, then validate them independently.
4. Align Python and browser mechanics, rebuild the standalone instrument, and
   finish the release hygiene audit. Full-connectome RPC remains separate.

## Availability repair — 2026-09-19

The earlier foreground web and daemon processes ended with their tool sessions.
The page now runs under the user's service manager. The browser shows a recovery
panel when the API is down, disables interventions, labels retained values stale,
and reconnects without reloading. Poll requests cannot accumulate while offline.
Saved progress for inactive brains is read from cached checkpoint metadata after
a restart; it no longer disappears from the collection chart. Opening the full
arena instrument selects the daemon's actual experiment without sending a switch.

Verification of this repair: all 263 Python tests pass, including user-service
file validation, ownership/idempotence, saved-catalog restoration and coherent
HTTP snapshots. Terminating only the managed web process triggered automatic
restart with a new PID while the learning daemon kept advancing. An isolated
read-only browser fixture returned 503 errors before recovering; the UI showed
its offline panel, disabled interventions, then reconnected without a reload.
Both the observatory and full instrument were verified in the browser.
