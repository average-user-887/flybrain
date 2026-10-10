# Sensory-delivery repair and provenance addendum (2026-10-08)

| | |
|---|---|
| **Pins** | Released `8bae810` (v0.5.0). Candidate `claude/ui-assets-20261008` (a pre-rewrite commit, not in the published master history); it differs from the release in the touched files only by one web-asset list line in `neurofly_daemon.py`. |
| **Branch** | `claude/sensory-delivery-fix-20261008`, from `8bae810`. |
| **Scope** | Key/schema defects in the wind and the experimental RPC sensory delivery, nothing else. |
| **Unchanged** | First-N cell selection, laterality, JO subtypes, gains, thresholds, formulas, synapses, dynamics, science jobs and results. The graph-I/O declaration was bumped to v3 in a follow-up (§5). |
| **Status** | This is an engineering repair. A delivered current is **not** evidence of sensing or behaviour. |

## 1. Reproduction at the boundary (before → after)

Method:
- Small synthetic fixtures with one distinct neuron per channel.
- The value recorded is the current actually written into that neuron.
- The RPC packet goes through the client's JSON encoding (`connectome_client._json_safe`) into `ConnectomeServer.step`.
- Scripts were kept outside the repository; the same cases are now in `tests/test_sensory_delivery_boundary.py`.

### 1.1 Graph controller (`GraphArenaController`; connectome-fixed/-plastic, the daemon default)

Six real arena steps per assay; values are `input_stage[].current_injected`.

| Assay | jon_wind, release | jon_wind, repair | Other channels |
|---|---|---|---|
| multisensory-sandbox (`wind_magnitude` = 15 mm/s) | **0.0** | **2.25** (= 0.15 × 15) | identical: food 0.2617, danger 0.2176, cVA 0.0463 |
| wind-tunnel (`wind_speed` = 25 mm/s) | 3.75 | 3.75 | identical: food 17.129 |
| t-maze, open-arena (wind vector only) | 0.0 | 0.0, now labelled NOT DELIVERED | identical |
| courtship, looming-escape | 0.0 | 0.0 | identical: cVA 13.038 |

### 1.2 RPC bridge → server (`hybrid-bridge-rpc-experimental`)

Representative packet: odour 0.5/0.5, cVA 0.6, wind (−25, 0) mm/s with the fly at rest, 30 °C, and a looming predator.

| Channel (server key) | Release | Repair | Formula (unchanged) |
|---|---|---|---|
| orn_food (`mean_odor`) | **0.0** | 15.0 | min(45, 30·c) |
| courtship_cva (`pheromone_cva`) | **0.0** | 21.0 | min(40, 35·c) |
| jon_wind (`wpn_wind_speed`) | **0.0** | 3.75 | min(35, 0.15·v), v > 5 |
| thermo (`temperature_c`) | **0.0** | 20.0 | min(40, 4·(T − 25)) |
| visual_looming (`looming_trigger`) | 30.0 | 30.0 | unchanged |

Edge cases sent directly to the server:

| Case | Release | Repair |
|---|---|---|
| Explicit zero beside a fallback key | 0 | 0 |
| Both cVA keys present | 21 (single injection) | 21 (single injection) |
| Empty packet | 0 | 0 |
| Unknown keys only (`wind_speed`, `cva_rate`, `temperature`, `pn_dm1_norm`, `bogus`) | 0 | 0, now listed in `unconsumed_keys` |
| **Non-finite inputs** | **45 / 40 / 35 / 40** (`min(cap, inf)`) | **0**, with status NOT DELIVERED |

## 2. Fixes, and why each mapping is a true identity

1. **Multisensory wind** (`neurofly_daemon.resolve_wind_speed`).
   - The multisensory sandbox's `wind_magnitude` and the wind tunnel's `wind_speed` are both `hypot(world airflow vector)` in mm/s, with no baseline and no normalisation (`maze.py` `MultisensoryLimbBenchmark.sample_stimuli` and `WindTunnelParadigm.sample_stimuli`).
   - The first present key wins, so an explicit 0 is never replaced and the probe injects once.
   - A non-numeric, non-finite or negative value is NOT DELIVERED.
   - The `jon_wind` `input_stage` row gains `stimulus_key` and `delivery`.
2. **RPC packet** (`connectome_bridge.encode_sensory`). Four keys are added, each holding a quantity the bridge already computes. No existing key changed.
   - `mean_odor` = the bridge's own `0.5·(odor_left + odor_right)`: the raw bilateral concentration, arbitrary units 0–1. It is **not** `pn_dm1_norm`, which is a PN rate after adaptation, hunger gain and divisive normalisation, and was not aliased.
   - `pheromone_cva` = the bridge's clipped cVA concentration, 0–1. It is **not** `cva_rate`, a Hill-transformed rate in Hz.
   - `wpn_wind_speed` = the bridge's own `self.wpn_wind_speed`: relative airspeed in mm/s, the same variable the bridge already reports under that name.
   - `temperature_c` = the absolute temperature in °C. The server subtracts its own 25 °C baseline. It was **not** mapped to `temperature_excess`.
3. **Server** (`brainlab/cosim_server._scalar_input`).
   - For odour, cVA, temperature and wind, the first present key wins, preserving the existing precedence (`temperature_excess` over `temperature_c`; `pheromone_cva` over `courtship_cva`).
   - Non-finite or non-numeric values inject nothing.
   - Each `/step` reply now carries `sensory_delivery` (the key read, its value and status per channel) and `unconsumed_keys`.

**Evidence limits.** The server's expected meanings are inferred from its code: key names, gains, thresholds and the 25 °C subtraction. No server input schema document exists. The match is exact for `wpn_wind_speed` and `temperature_c`. For `mean_odor` and `pheromone_cva` it rests on the bridge's own variable name and concentration scale, and on the parallel `GraphArenaController` probes, which read raw concentrations.

## 3. Still NOT DELIVERED (labelled, not fixed)

- **Wind in assays that publish only a `wind` vector**: t-maze, open arena and others. The field exists, but no scalar key is published. This is **one open mapping decision**:
  - Should `hypot(wind)` be delivered as the JO wind speed for vector-only assays?
  - If yes, the t-maze vacuum airflow (15 mm/s) would start reaching JO cells in the default observatory assay.
  - Until that is decided, the probe reports `NOT DELIVERED: ... no vector-to-probe mapping is defined`.
- Graph-path non-finite guards for food, danger, cVA and temperature were **not** added. They were not part of a proven mismatch. Some paradigms (wind tunnel, multisensory) already reject non-finite stimuli at their source (`_finite_stimulus`); the others do not.
- Distinct modelling limitations are tracked separately and left untouched: first-N row-order pools, side bias, mixed JO subtypes, no lateralised input, gains.

## 4. Affected-result inventory (nothing rewritten)

**Graph path, multisensory wind** (JO input was absent before this commit):

- `docs/receipts/audit-20261005/C-assays-connectome.md` row 13 and `C-evidence/*`. That audit itself recorded "wind not delivered".
- Multisensory visits in the 14-assay UI sign-off receipts:
  - `docs/receipts/{switch-race-20261004,save-policy-20261005,pr30-review-20261005,no-silent-freeze-20261005}/**/live_ui_signoff*.json`;
  - `docs/receipts/cleanroom-install-nvidia-20261005.md`;
  - `docs/receipts/audit-20261005/D-evidence/logs/*.jsonl`.
  - These are UI/transport evidence and carry no wind-sensing claim.
- Local saved graph instances `outputs/**/registry*/multisensory-sandbox/connectome-fixed/*`: five registries (`registry-v3`, `observatory-live/registry-v3`, and three test registries).
  - Under v3 (§5), these and **every other saved graph instance of every assay** refuse an ordinary resume.
  - Continuing them needs `--continue-io-state`, which creates linked v3 children; the v2 parents stay unchanged.
  - Deploying this candidate therefore needs that flag (or fresh instances) on the owner's daemon. No owner store was touched here.
  - In the first repair commit (pre-rewrite commit, not in the published master history) only the per-step `input_stage[jon_wind].stimulus_key/delivery` fields marked the change. v3 supersedes that.

**RPC path.** No saved result found.
- The daemon never builds `connectome_mode='rpc'`. It is reachable only programmatically and in tests.
- No receipt in `docs/` or `experiment_data/` and no local `outputs/` JSON names `hybrid-bridge-rpc-experimental`.
- `experiment_data/ryzen_battery/*` ran the **surrogate** bridge, which sends no packet, and is unaffected.

**Unaffected**: wind-tunnel and looming paths (values identical), cohort runs (optomotor encoder only), and the modular controller.

## 5. Graph-I/O version bump (Astra decision 1–3, 2026-10-08)

- **Version change.** `GRAPH_IO_VERSION` is now `graph-arena-io-v3-unassisted`.
  - v3 hash: `220a31c25c6c236957bb0fbab316111223c978d8d810ecef88910ca8c838fedb`.
  - v2 hash, as released in v0.5.0: `23d421f937ed485e5cf8d70835c2a50d56b8f88be217e4e66e5ad3909eef6191`.
- **What the hashed `jon_wind` probe now declares.**
  - `accepted_keys`: `wind_speed`, then `wind_magnitude`.
  - `units`: mm/s, world airflow speed, no baseline.
  - `precedence`: the first present key wins; an explicit 0 is kept; one injection per step.
  - `invalid_input`: a bool, non-numeric, non-finite or negative value is NOT DELIVERED.
  - `vector_only_input`: NOT DELIVERED.
  - A `supersedes` block names v2 and what it dropped.
- **Saved v2 lineages never change in place.** They use the existing linked-child route, unchanged.
  - An ordinary resume raises `IOContinuationRequired`. The message names the v2 and v3 versions and hashes and `--continue-io-state`, and nothing is written.
  - `--continue-io-state` creates a linked child. The child has `lineage.prior_graph_io` = v2 and `new_graph_io` = v3, and copies the brain state, RNG and world state unchanged.
  - The parent store is byte-identical before and after.
  - Identity checks were not weakened.
- **AMD (`wgpu-amd`) keeps refusing explicit continuation.** This is a known limitation: a saved v2 store on AMD must continue on CPU or CUDA (`brainlab/runtime_backend.require_fixed_amd`).
- **Tests.** `tests/test_sensory_io_continuation.py` uses fixture lineage hashes only.
- **RPC path.** It has no saved lineage, and graph-I/O does not apply to it. Its keys are reported per step in `sensory_delivery`.

## 6. Backlog: explicit world-vector-to-magnitude wind mapping (deferred)

Vector-only wind stays NOT DELIVERED. Proposal for a bounded follow-up card:

1. **Producers that publish a `wind` vector with no scalar key.**
   - `maze.py:1967` t-maze: piecewise vacuum of (0, −15), (15, 0) or (−15, 0) mm/s, chosen at `maze.py:1955-1960`.
   - `maze.py:2106` y-maze: (0, 0).
   - `maze.py:2269` heat-maze: (0, 0).
   - `maze.py:3306` labyrinth: (0, 0).
   - Assays without a `wind` key inherit `arena.wind` (default (−0.3, 0), `arena.py:676/710`) at `arena.py:1732-1736`.
   - The open arena (`paradigm=None`) passes `arena.wind` as `wind_vector` (`arena.py:1395`, `:1417`). Its scalar control is `assay_controls.set_parameter('windStrength')`.
2. **Scalar producers already accepted.** `maze.py:2660` (`wind_speed`) and `maze.py:3565` (`wind_magnitude`).
3. **Proof the card must give before any delivery.**
   - Each vector is world-frame airflow in mm/s, not body-relative airspeed. The t-maze vector is a hand-set "vacuum" flow.
   - The mapping is `hypot(vx, vy)`, with no fly-velocity subtraction (the graph path does none today).
   - The direction stays undelivered.
4. **Then:**
   - publish `wind_speed` at each producer, or add one declared vector rule to `accepted_keys`;
   - bump graph-I/O again;
   - add boundary tests.
   - Expected effect: the t-maze reaches 2.25 on jon_wind in the default observatory assay; the zero-flow assays are unchanged.

## 7. Original note (superseded by §5)


The graph-I/O declaration and `GRAPH_IO_SHA256` were deliberately left unchanged. The probe's declared stimulus (`wind_speed_mm_s`) is still the correct quantity.

Bumping the declaration would make every saved graph instance of **every** assay refuse to resume without `--continue-io-state`, and would make AMD resume refuse. That is outside this bounded repair.

The consequence is that a resumed multisensory instance changes input mid-lineage with no manifest-level marker. Whether to bump the version for that reason is left to the reviewer.
