# Assay semantics: metric contract v1.2 producers

| | |
|---|---|
| **Contract** | `metric-contract/1.2` (base contract plus the v1.1 and v1.2 amendments). Schema id `neurofly.metric/1.2`; every assay spec is `<assay>/1.2`. |
| **Stage** | C0, the producer stage: `maze.py`, `online_metrics.py`, `assay_controls.py`, `tests/test_metric_contract.py`. |
| **Status** | Engineering contract. It is **not** scientific validation and **not** release acceptance. Intermediate C0 code is not a repaired user-facing product: the daemon, arena and web still read the legacy views until C1 and C2. |

Every window, threshold and hysteresis value below is an **ENGINEERING default**. None of them claims
comparability with a published protocol. Such a claim would need verified primary sources and a
`spec_version` change.

`online_metrics.check_spec()` is the current C0 validation boundary. It accepts only the v1.2 schema, contract and matching assay spec version, and rejects unsupported versions instead of reinterpreting them. Historical fixtures remain byte-preserved; this repository has no historical metric-envelope reader to extend. C1/C2 must add consumer validation and integral-unit display in the daemon/status and UI paths before browser acceptance.

## 1. What the producer publishes

Each paradigm in `maze.py` (13 assays), plus `online_metrics.OpenArenaObserver` (open arena), provides:

| Call | Shape and rule |
|---|---|
| `observation_spec()` | Static, JSON-finite declaration: `assay`, `spec_version`, `mode`, `window_s` (the spec default), `hold_s`, `terminal_events`, `re_presentation_triggers`, `logged_interventions` (open arena only), `headline_metric`, `companion_metrics`, `curve_metrics`, `learning_claim:"none"`, `required_states`, `window_params`, `default_scope`, plus `light_schedule` for DAM. |
| `observation_status()` | `{state, end_reason, terminal_event, measurement_end_rel_s, presentation_index, presentation_start_rel_s, presentation_elapsed_s, segment_elapsed_s, hold_remaining_s, config_id, effective_window_s, automatic_end, window_end_rel_s}`. |
| `get_metric_records()` | `{name: record}`. A record is the v1 §2.1 record plus `scope` (`segment` or `presentation`). `final` is false. |
| `get_evidence()` | `{name: {type, data}}` from the closed evidence types. Every assay adds `interventions` (`intervention_log`), and adds `terminal_events_observed` when a terminal event occurred without ending the measurement. |
| `freeze_observation(end_reason=None)` | The producer part of a frozen envelope. It is built **before** any mutation and its records carry `final:true`. With no argument, the measurement's own end reason is used. |

Time is kept in integer microseconds from the `dt` actually passed in. Every timestamp is
**segment-relative**: event `t_rel_s`, `measurement_end_rel_s`, `censored_at_rel_s`, `window_end_rel_s` and
the stimulus events. Every latency value is elapsed from the presentation start `P`. A pre-motor sample of
step `n` is stamped at the segment time before that step's `dt` is added, so event latencies are one `dt`
earlier than in the legacy code, which increments time first. A post-solver contact is stamped `t + dt`
with the same step.

## 2. Interfaces for the integration stage (C1)

### 2.1 Observation configuration (v1.1 A1, I-19)

```python
from online_metrics import build_observation_config, ConfigError
cfg = build_observation_config(paradigm.observation_spec(), config_id='<daemon-issued unique id>',
                               override_window_s=args.trial_seconds,   # or None
                               continuous=args.continuous,             # bool
                               override_source='cli:--trial-seconds',  # or 'cli:--continuous' / 'api'
                               set_at_sim_s=sim_time_s, manifest_run_id=run_id)
paradigm.configure_observation(cfg)              # before the first sample of the segment
paradigm.reset_trial(config=cfg)                 # a new segment, with its config
paradigm.begin_next_presentation(reason, cfg)    # a new presentation, with its config
```

- `ObservationConfig = {config_id, effective_window_s, automatic_end, window_source, override, hold_s}` exactly as in
  amendment A1. `validate_observation_config()` normalises it and raises `ConfigError` on a bad shape.
- `configure_observation()` after the first sample of the current presentation raises `ConfigError` and changes
  nothing.
- The producer ends a measurement only from `effective_window_s` and `automatic_end`. With `automatic_end:false`
  no window end occurs. Terminal events are then recorded, with their latencies, but they do not end the
  measurement.
- When no config has been delivered, the producer uses its spec default (`config_id "spec-default:<assay>"`).
  An explicitly delivered config persists across later presentations and segments until a new one is
  delivered.
- A config delivered with `begin_next_presentation` or `reset_trial` that changes a policy field logs
  `policy:<field>` interventions (`effective_window_s`, `automatic_end`, `window_source`, `hold_s`, and `override`,
  whose value is the override source) at the new presentation start.
- `build_observation_config` rules:
  - `continuous=True` gives `automatic_end:false` with `window_source:"continuous_flag"` for every assay.
  - A continuous-mode assay (open arena, DAM) never has an automatic end. A window override is carried in
    `override` for provenance, but it is not applied, and `window_source` stays `spec` (see ambiguity Q2).
  - Otherwise, an override window becomes `effective_window_s` with `window_source:"override"`.

### 2.2 Accepted actions (v1.1 A2, I-7)

```python
pf = assay_controls.preflight(arena, 'set_param' | 'assay_action', name, value=None)
# -> {accepted, message, normalized_value, closes_presentation, intervention_only, kind, name}
```

`preflight` is pure. It reads only the action, the value and the current state, including the fly position
for the near-gap rule. It shares the single `_validate()` with `set_parameter` and `act`, which run it again as
a defence. The daemon sequence, under one hold of the runner lock:

1. `pf = preflight(...)`. If it is rejected, return `{status:"error", message: pf["message"]}`. Nothing changes.
2. If `pf["closes_presentation"]`: `envelope = paradigm.freeze_observation("re_presentation_user")`; ledger and
   `last_terminal`.
3. Apply with `set_parameter(arena, name, value)` (it returns the float, which is what the daemon logs today;
   `detailed=True` returns `{applied, value, presentation_closed}`) or `act(arena, name)` (it returns that dict).
4. If `closes_presentation`: `paradigm.begin_next_presentation(f"user_{name}", config)`. For the open arena, use the
   `OpenArenaObserver`.
5. If `pf["intervention_only"]` (open-arena `food`): `observer.log_intervention("food", None)`. Nothing closes.
6. Log the intervention.

`set_parameter` and `act` **never** call `begin_next_presentation` themselves.
`would_close_presentation` was withdrawn by v1.1 and does not exist.

### 2.3 Sampling hooks (I-8 to I-11, I-13)

- `paradigm.begin_sample(arena_step, dt)` immediately before `paradigm.step(fly, dt)` (pre-motor, unchanged order).
- `paradigm.observe_contact(arena_step, dt, "post_solver", in_contact, normals, source)` once, after the solve
  and containment. It raises `ValueError` if `arena_step` is not the step just sampled.
- `paradigm.provenance = {controller_states, gf_source, stimulus_entry_stage, motor_assists_enabled}` before each
  sample. State-derived records are `unsupported` until `controller_states` lists the state.
- Looming connectome mode: `paradigm.record_gf_spike()` on DNp01 ESCAPE. It is stamped at the latest pre-motor
  sample, so the arena must call it after that tick's `paradigm.step` (see A11).
- Open arena (no paradigm): `observer = OpenArenaObserver(airflow_mm_s=-arena.wind[0])`, then once per step at
  the pre-motor point `observer.observe(fly, food_contact, dt, airflow_mm_s=-arena.wind[0])`. The proposed arena
  attribute is `arena.open_arena_observer`.
- DAM compressed light schedule (a labelled software test only): `paradigm.set_light_schedule(light_s, dark_s)`.

### 2.4 Freeze, end reasons and completeness

- `freeze_observation()` returns the envelope fields that the producer owns:
  - schema, contract, `assay` and `spec_version`;
  - mode, `config_id`, `effective_window_s` and `window_s` (alias), `automatic_end`, `window_source`, `override` and `hold_s`;
  - `window_params` and `window_end_rel_s`, plus `stimulus_collision_rel_s` for looming;
  - `light_schedule`, `sample_point`, `dt_s` and `dt_fixed`;
  - the presentation index, start and elapsed time, state, completeness, end reason, terminal event and `measurement_end_rel_s`;
  - records and evidence.

  The daemon adds identity, provenance, `segment_id`, `presentation_id`, the sim times, validity and `terminal_pose_post_step`.
- End reasons: `window_elapsed`, `terminal_event:<name>`, `re_presentation_user`, `policy_change`, `manual_reset`,
  `experiment_selected`, `backend_switch`, `fault_halt`, `shutdown`.
- `completeness_for(end_reason, window_elapsed, automatic_end)` is the shared rule. The daemon owns the published value (A3).
  - `re_presentation_user` is `complete` when the window had elapsed or there is no automatic end. That is the
    open-arena airflow case of amendment A4.3.
  - `policy_change` is `incomplete` unless the window had elapsed.
- On `fault_halt`, pending outcomes and latencies become `invalidated`. Measured prefix counts are kept, and the
  daemon sets `validity:"invalidated"`.
- Hold: a display hold follows only a **terminal event** (`hold_s` from the config). During it the status is
  `measurement_ended`; afterwards it is `closed`. A window end goes straight to `closed`, with no hold (A5).
  Samples taken during a hold advance the clock but never enter the measurement.

## 3. Rules shared by all assays

| Rule | Definition |
|---|---|
| Walking / immobile | Walking is \|speed\| > 0.5 mm/s at the pre-motor sample. Immobile is \|speed\| ≤ 0.5 mm/s with no beam crossing. Both are operational proxies (ENGINEERING); neither means "no motion" biologically. |
| Terminal-sample duration (A4) | With an automatic end, the sample at which a terminal event is detected ends the measurement at its own time `t`. Its forward interval `[t, t+dt)` is **excluded** from duration sums; positions up to the event pose are included. |
| Scope (I-20) | Each record carries `scope`. The default is `presentation` for presentation-mode assays and for assays with re-presentation triggers (t_maze, heat_maze, buridan, visual_operant, wind_tunnel, looming_escape, optomotor, gap_crossing, courtship), and `segment` otherwise (y_maze, circadian_dam, labyrinth, multisensory). The open arena declares both (§5). |
| Physical hysteresis across presentations | Re-arm and contact state is physical, so it survives an in-place presentation change: the T-maze `last_arm`, the Y-maze `armed`, labyrinth dead-end re-arm, contact `prev` and the open-arena contact state. A fly already inside an arm at a reversal does not make a new entry until it re-arms. Counts and first choices are reset (see Q4). |
| First occurrence | Terminal events (refuge, source, goal, landing, turn) are recorded once per presentation. Later re-entries never overwrite the first time. |
| Censoring | A pending value cut by the window, by another terminal event or by an interruption is `censored`, with `counts.censored_at_rel_s` (segment-relative) and `counts.censored_after_s` (elapsed from P). A fault gives `invalidated`. |
| Evidence caps | `event_sequence` evidence keeps the first 1,000 events; record counts stay exact. DAM `activity_bins` is unbounded (1,440 bins per simulated day). |

### Deadline-outcome invariant (v1.1 A5)

The rule covers `crossing_success`, `refuge_reached`, `source_reached`, `goal_reached`, `escape_initiated`,
`escape_completed` and `turn_complete`:

| Situation | value | available | reason |
|---|---|---|---|
| Observing, no event yet | null | false | `pending` |
| Event occurred | true | true | null |
| Completed without the event: the window was fully observed, or the measurement ended at another declared terminal event | false | true | null |
| Interrupted (`manual_reset`, a switch, `re_presentation_user`, `policy_change`, `shutdown`) | null | false | `censored` |
| `fault_halt` | null | false | `invalidated` |

`false` together with a non-null reason is forbidden. `get_metric_records()` asserts this on every read.

## 4. Engineering defaults

| Constant (`maze.py`) | Value | Use |
|---|---|---|
| `WALKING_SPEED_MM_S` | 0.5 mm/s | walking / immobility proxy |
| `YMAZE_REARM_RADIUS_MM` | 12.0 mm | Y-maze re-arm: hub radius 10 mm plus 2 mm hysteresis |
| `LABYRINTH_REARM_MM` | 1.0 mm | dead-end re-arm distance outside the zone rectangle |
| `GAP_TURN_COMPLETE_RAD` | 0.4 rad | mirrors the gap ABORT controller's existing `abs(error) < .4`; not tuned (A10) |
| `DAM_BOUT_US` | 300 s | immobility-bout criterion, accumulated in µs, never compressed |
| `BURIDAN_WALK_MIN_US` | 1 s | walking-alignment eligibility |
| Windows | §9.1 of v1 | t_maze 120 s, y_maze 300, heat 300, buridan 300, visual_operant 120, wind 120, looming `t_collision_s + 1.6` (2.0), optomotor 60, gap 120, courtship 600, labyrinth 300, multisensory 120; open arena and DAM are continuous |
| Holds | — | heat 2 s, wind 1 s, gap 1 s, labyrinth 2 s |

Controller gains and the 0.4 rad decision threshold in `assay_response.py` are unchanged.

## 5. Records per assay

Units are from the closed vocabulary. A **proxy** is named as one.

- **open_arena** (`OpenArenaObserver`, continuous)
  - Segment scope: `observation_s` (s); `food_contacts` (count of onsets); `time_to_first_contact_s` (s from the segment start; `pending` while observing, `censored` on interruption); `distance_mm`.
  - Presentation scope: `presentation_observation_s`, `presentation_food_contacts`, `presentation_distance_mm`, and `airflow_mm_s` (the condition: the airflow at the presentation's first sample).
  - Evidence: `contact_events`, `interventions`.
  - An airflow change (`windStrength`) closes a presentation; food placement is only logged.
- **t_maze**
  - `arm_entry_preference_index` = (n₊ − n₋)/(n₊ + n₋), noted "arm-entry preference index over repeated visits; not a conditioning PI or evidence of learning".
  - `cs_plus_entries`, `cs_minus_entries`.
  - `first_choice` (label `cs_plus`/`cs_minus`; `counts.arm`) and `first_choice_latency_s`, both immutable within a presentation.
  - Evidence: `first_choice_pose`, `entries`.
  - Entry: arm_a with x < 45 or arm_b with x > 95. Re-arm: 63 ≤ x ≤ 77.
- **y_maze**
  - `physical_entries`.
  - `spontaneous_alternation_rate` = alternating / triads of the collapsed sequence; fewer than 3 collapsed symbols gives `insufficient_events`.
  - `handedness_index` = (R − L)/(R + L); `left_turns`, `right_turns`, `total_triads`, `alternating_triads`.
  - Evidence: `physical_visit_sequence` (`collapse:"none"`) and `alternation_sequence` (`consecutive_repeats`).
- **heat_maze**
  - `escape_latency_s` and `refuge_reached`.
  - `thermal_dose_degC_s` = Σ max(0, T − 25 °C)·dt at pre-motor samples (kind `integral`, unit `degC*s`; see A6).
  - `target_quadrant_fraction` and `path_length_mm`.
  - `place_learning` is unsupported.
  - Evidence: `refuge_entry_pose`.
- **buridan**
  - `observation_s`, `walking_s`, `displacement_mm`, `path_length_mm`.
  - `heading_alignment` and `walking_stripe_alignment`: the mean cos of the nearest-stripe bearing, time-weighted in µs, kind `mean`, unit `index` (A7), noted "alignment, not navigation success". Walking alignment needs ≥ 1 s of walking; otherwise `insufficient_events`.
  - `centre_fraction` (within 25 mm of the centre) and `centrophobism_index` = 1 − that.
  - `stripe_traversals` is unsupported: heading-side flips are not traversals.
- **visual_operant**
  - `safe_occupancy_fraction` and `occupancy_index`, noted "occupancy, not learned avoidance".
  - Per sector, `mean_yaw_command_{safe,punished}_rad_s` (**signed**; positive = counter-clockwise, the sign of `angular_velocity`/`yaw_torque`) and `mean_abs_yaw_command_{safe,punished}_rad_s`. They are proxies: "controller yaw command used as torque proxy".
  - Both carry `counts {n, sum_signed_rad_s, sum_abs_rad_s}`. n = 0 gives `not_observed`; n > 0 is available, a genuine 0.0 included.
  - `yaw_samples_safe` and `yaw_samples_punished`; `operant_learning` is unsupported.
  - Sector and yaw come from the legacy step of the same sample.
- **wind_tunnel**
  - `observation_s`, `upwind_displacement_mm` ("displacement, not evidence of wind sensing"), `path_length_mm`.
  - `odor_contact_s` (concentration > 0.05).
  - `source_reached` and `time_to_source_s`.
  - Gated on `controller_states`: `surge_s`, `cast_s`, `rest_s`, `other_state_s`, and `surge_cast_ratio` = surge_s/cast_s.
- **looming_escape**
  - `escape_initiated` (from an explicit GF event only: geometric θ ≥ `gf_threshold_deg` at a pre-motor sample, or `record_gf_spike()`), `initiation_latency_s`, `ttc_at_initiation_ms`, `theta_at_initiation_deg`.
  - `escape_completed` (the escape motor observed active, then ended) and `escape_displacement_mm`.
  - `gf_source`.
  - Evidence: `presentation_events` contains `stimulus_onset` (at P), `stimulus_collision` (at P + `t_collision_s`; a stimulus event, never an escape), `gf_event`, `escape_motor_start` and `escape_motor_end`.
  - Initiation ends nothing.
- **optomotor**
  - `gain` = Σ(yaw·dt)/Σ(drum·dt) in deg/s, a ratio of integrals: positive when following in either direction; a zero drum gives `zero_denominator`.
  - `mean_fly_yaw_deg_s` and `mean_retinal_slip_deg_s`.
  - `stimulus_entry_stage` comes from provenance; `mean_hs_firing_rate` is unsupported.
- **gap_crossing**
  - `decision_outcome` (`CROSS`/`ABORT`, "paradigm geometric threshold decision, not tactile planning").
  - `crossing_success`, "crossed by the declared deadline".
  - `time_to_cross_s` (P to `landed`).
  - `turn_complete`: after `abort`, the first sample with \|wrap(π − heading)\| < 0.4 rad.
  - `probing_duration_s` and `decision_latency_s`.
  - Evidence: `decision_sequence`.
  - ABORT is not terminal; `landed` and `turn_complete` are.
- **circadian_dam**
  - `beam_crossings`.
  - `immobility_bouts_300s`, `bout_immobility_min` and `current_immobile_s`, noted "operational immobility proxy (speed ≤ 0.5 mm/s, no beam crossing); not validated sleep".
  - `mean_bout_min` is a derived ratio of qualifying bout-minutes to bout count, with unit `min` (with 0 bouts, `zero_denominator`); `light_phase`; and `circadian_rhythm` (unsupported).
  - Evidence: `activity_bins` (60 s) and `light_phases`.
  - The default is LD 12 h/12 h of real simulated time, with ZT0 at the segment start. DD is also available. A compressed schedule is a labelled software test.
- **courtship**
  - `proximity_fraction` (time within 3.5 mm of the female position) and `min_distance_mm`.
  - Gated on states: `approach_s`, `avoid_s`, `courtship_state_s`, `courtship_state_fraction`.
  - Unsupported: `courtship_conditioning`, `learned_suppression`, `wing_song`, `rejection_kicks`.
- **labyrinth**
  - `wall_contact_onsets` and `wall_contact_s`, from the post-solver handoff only (`not_observed` without it).
  - `dead_end_entries` and `dead_end_s`.
  - `path_length_mm`, `net_displacement_mm`, and `tortuosity` (with net < 1e-9 mm, `zero_denominator`).
  - `goal_reached` and `time_to_goal_s`.
- **multisensory**
  - `distance_mm` (from pre-motor samples), `wall_contact_onsets`, `wall_contact_s`.
  - `odor_a_exposure_s` (odour A ≥ 0.5) and `heat_exposure_s` (T > 35 °C).
  - `mean_abs_speed_jerk_mm_s3`, a proxy: mean \|Δacceleration\|/dt from the sampled speed.
  - Unsupported ("scripted leg phases are illustrative; no neural coordination"): composite, coordination, integration, efficiency, smoothness and the energy proxy.

## 6. Legacy views (C3 mapping; unchanged at C0)

`get_metrics()` and the step outputs are unchanged at C0. S-12 checks them against fixtures generated from
`0f144ca`. At C3, `legacy[k] = record.value` (null when unavailable), following v1 §8, with these v1.1 rows:

- `mean_torque_safe` and `mean_torque_punished` come from the **signed** records `mean_yaw_command_*_rad_s` and are
  null when not observed. They are never reconstructed from the absolute means.
- `escape_latency_ms` and the other `*_ms` latencies are the record value × 1000. Consumers never compute a latency from
  timestamps.
- `crossing_success` comes from the deadline-outcome record, so it is null when censored.

## 7. Resolved hand-over questions (Deck A1–A15)

| # | Resolution |
|---|---|
| A1 | `set_parameter()` returns the float by default, because the daemon logs it. `detailed=True` returns `{applied, value, presentation_closed}`. `act()` returns that dict (its return value is unused today). |
| A2 | The daemon drives re-presentation: `preflight` → freeze → `set_parameter`/`act` → `begin_next_presentation("user_<name>", config)`. Nothing in `assay_controls` begins a presentation implicitly. |
| A3 | The daemon owns completeness. The producer exposes `completeness_for()` and fills a default into `freeze_observation()`. |
| A4 | The forward interval of a terminal sample is excluded from durations (§3). |
| A5 | There is no hold after `window_elapsed`; the hold follows terminal events only. |
| A6 | v1.2 declares `thermal_dose_degC_s` as kind `integral`, unit `degC*s`; its excess-temperature calculation is unchanged. |
| A7 | Buridan alignment is kind `mean`, unit `index`, time-weighted in µs (identical to the per-sample mean at fixed dt). |
| A8 | `curve_metrics` are declared only for `arm_entry_preference_index`, `spontaneous_alternation_rate` and `walking_stripe_alignment`, because v1 lists none. |
| A9 | Superseded by v1.1 A5: `turn_complete` follows the deadline-outcome invariant. After a CROSS that ends at `landed` it is false/available (completed at another terminal event); while observing it is pending. |
| A10 | The 0.4 rad threshold is an inline literal in `assay_response.py`, mirrored as `GAP_TURN_COMPLETE_RAD`. N1-17 should pin both. |
| A11 | `record_gf_spike()` is stamped at the latest pre-motor sample; the arena order must be documented in C1. |
| A12 | A new looming presentation restarts the legacy stimulus clock in place (as `act('loom')` does). |
| A13 | The legacy pose extractor defaults speed to 1.2 mm/s for dict flies without `speed`, and that value reaches the walking and immobility proxies. Real `FlyState` objects always carry speed. Not changed (legacy helper). |
| A14 | Evidence caps (§3). |
| A15 | Interfaces outside §10: `arena.open_arena_observer` and `set_light_schedule(light_s, dark_s)`. |

## 8. Open contract questions found in C0 (reported, not guessed)

- **Q1 — CF-3 wording.** Amendment CF-3 says that with `automatic_end:false` and a refuge entry, "`refuge_reached`
  stays live (A5 invariant: still `pending` while observing)". The A5 table says an event that occurred is
  `true`. C0 publishes `true`, with `final:false` (live), plus the latency. Please confirm.
- **Q2 — window override on continuous-mode assays.** `--trial-seconds` is a "global override", but OA-5a and DAM
  forbid automatic ends. C0 carries the override block on continuous-mode assays but does not apply it.
- **Q3 — `--trial-seconds` on presentation assays.** The override replaces the dynamic looming window
  (`t_collision_s + 1.6`) and the 60 s/120 s presentation windows. Is that intended?
- **Q4 — hysteresis across presentations.** Re-arm and contact state survives an in-place re-presentation (§3),
  so a fly sitting in a T-maze arm at a reversal is not counted as a new-condition entry until it re-arms.
- **Q5 — fault after an event.** On `fault_halt`, an outcome that had already occurred stays `true`. Only pending
  values become `invalidated`, and the envelope validity marks the run. Should a fault also invalidate events
  that were measured before it?
- **Q6 — spec_version (resolved by v1.2).** Every current assay publishes `<assay>/1.2` with schema
  `neurofly.metric/1.2` and contract `metric-contract/1.2`. Historical documents and records keep their original identifiers.
- **Q7 — policy-change completeness when the window had already elapsed.** It is treated like
  `re_presentation_user`: `complete` only if the window had elapsed.
- **Q8 — open-arena `airflow_mm_s`.** It is the condition at the presentation's first sample. If the airflow
  changes without the A2 flow (no freeze), the record keeps the first value.

## 9. Literature anchors (non-binding)

The Deck read results (`DECK_CODEX_08_ANCHORS`) do not change any window, because the windows are engineering
defaults. Recorded for any later comparability claim:
- Buridan sessions in Colomb et al. 2012 were 900 s.
- Buchanan et al. 2015 is a handedness study, not a spontaneous-alternation study.
- The 5-minute heat-maze trials in Ofstad 2011 allow leaving the cool tile. The `until_terminal` refuge end differs from that protocol.

## 10. Vector → test map (`tests/test_metric_contract.py`)

| Vectors | Tests |
|---|---|
| S-1 | `test_S_1_all_14_specs_static_finite_and_windows_exact` |
| S-2 | `test_S_2_vocabularies_are_closed` |
| S-3 | `test_S_3_validate_finite_returns_nested_paths` |
| S-4 | `test_S_4_unexpected_nonfinite_raises_metricfault` |
| S-5, S-6, S-7, S-8 | `test_S_5_S_6_S_7_S_8_degenerate_states_are_finite_json_and_consistent[*]` (4 end-reason variants); S-8 also `test_S_8_final_is_set_only_by_the_freeze_helper` |
| S-9 | `test_S_9_state_derived_records_unsupported_without_provenance` |
| S-10 | `test_S_10_begin_next_presentation_resets_timing_and_only_presentation_state` |
| S-11 | `test_S_11_premotor_times_before_dt_and_postsolver_contacts_plus_dt` |
| S-12 | `test_S_12_legacy_get_metrics_unchanged_against_713ba82_fixture` |
| S-13 | `test_S_13_deadline_outcome_invariant_across_all_outcome_booleans` |
| OA-1…OA-4, OA-5a, OA-5b, OA-6, OA-7, OA-8 | `test_OA_1_…`, `test_OA_2_…`, `test_OA_3_…`, `test_OA_4_…`, `test_OA_5a_…`, `test_OA_5b_…`, `test_OA_6_…`, `test_OA_7_…`, `test_OA_8_…` |
| TM-1…TM-7 | `test_TM_1_…`, `test_TM_2_and_TM_3_single_entries`, `test_TM_4_…`, `test_TM_5_…`, `test_TM_6_…`, `test_TM_7_…` |
| YM-1…YM-6 | `test_YM_1_…` … `test_YM_6_…` |
| HM-1…HM-5 | `test_HM_1_…` … `test_HM_5_…` |
| BU-1…BU-4 | `test_BU_1_…` … `test_BU_4_…` |
| VO-1a, VO-1b, VO-2…VO-4 | `test_VO_1a_…`, `test_VO_1b_…`, `test_VO_2_…`, `test_VO_3_…`, `test_VO_4_…` |
| VO-5…VO-8 + L-1 (extended) | `test_VO_5_to_VO_8_signed_and_absolute_yaw_and_L_1_extension[VO-5…VO-8]` |
| VO-9 (+ L-1 note) | `test_VO_9_empty_sector_signed_and_absolute_not_observed` |
| WT-1…WT-7 | `test_WT_1_…` … `test_WT_7_…` |
| LE-1…LE-4 | `test_LE_1_…`, `test_LE_2_…`, `test_LE_3_…`, `test_LE_4_…` |
| LE-T9-3 | `test_LE_T9_3_below_threshold_false_escape_cannot_initiate_or_end` |
| LE-6 (replaced by TS-2, kept) | `test_LE_6_stimulus_collision_is_not_an_escape` |
| connectome GF | `test_LE_connectome_gf_spike_initiates_at_latest_premotor_sample` |
| OM-1…OM-6 | `test_OM_1_and_OM_2_…`, `test_OM_3_…`, `test_OM_4_…`, `test_OM_5_…`, `test_OM_6_…` |
| GC-1, GC-2/GC-6, GC-3, GC-4, GC-5, GC-7 | `test_GC_1_…`, `test_GC_2_and_GC_6_…`, `test_GC_3_…`, `test_GC_4_…`, `test_GC_5_…`, `test_GC_7_…` |
| DAM-1…DAM-7 | `test_DAM_1_to_DAM_3_…`, `test_DAM_4_…`, `test_DAM_5_…`, `test_DAM_6_…`, `test_DAM_7_…` |
| CO-1…CO-4 | `test_CO_1_…` … `test_CO_4_…` |
| LB-1…LB-6 | `test_LB_1_…` … `test_LB_6_…` |
| MS-1, MS-2 | `test_MS_1_…`, `test_MS_2_…` |
| CF-1…CF-5 | `test_CF_1_…` … `test_CF_5_…`; also `test_CF_reset_trial_accepts_the_segment_config` and `test_CF_build_observation_config_rules` |
| PF-1…PF-5 | `test_PF_1_…` … `test_PF_5_…` |
| TS-1…TS-5 | `test_TS_1_…` … `test_TS_5_…` |
| §5.6 (producer side of N1-2) | `test_paradigm_without_spec_runs_legacy_step_and_refuses_observation` |

A paradigm subclass without `OBSERVATION_SPEC` (for example the WP3 physics test fixture) runs its legacy step
unchanged, and its `observation_spec()`, `observation_status()` and `get_metric_records()` raise
`maze.MissingObservationSpec("assay <name> has no observation spec")`. The daemon must refuse to activate it.
