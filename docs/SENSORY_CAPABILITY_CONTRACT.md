# Sensory capability contract (W1, asset and sensory batch)

| | |
|---|---|
| **Base** | pre-rewrite commit `f96bdc3`, not in the published master history (branch `claude/ui-assets-20261008`). Every `file:line` below refers to that commit. |
| **Scope** | Inventory and display contract only. No code, encoder, physics or brain weight was changed, and no simulation was run for this document. |
| **Consumer** | W4, the sole web-UI integrator: environmental inspector and sensory overlays. |
| **Rules applied** | A decorative model is never evidence of detection. Delivering an input is never evidence of a neural response. A biological claim appears only with a primary citation. An observed response is quoted only from an existing receipt. |

## 0. Summary

1. **Every odour in the engine is a static, normalised scalar field.** There is no chemical identity, no transport, no turbulence and no wind advection.
   - Concentrations are dimensionless, in `[0, 1]`, and in **arbitrary units**: a Gaussian or exponential of distance (`arena.py:302-320`, `maze.py:1941-1971`, `maze.py:2642-2665`, `maze.py:3512-3570`).
   - The wind-tunnel "plume" is a fixed lateral Gaussian with exponential downwind decay. Nothing is solved or advected (`maze.py:2644-2651`).
2. **Wind is a uniform vector per assay, in mm/s, in the arena frame.** No flow field, obstacle flow or turbulence exists. The open arena accepts only scalar airflow toward −x (`arena.py:851-865`, `neurofly_daemon.py:4962-4963`).
3. **The default graph controller (connectome-fixed/-plastic) has seven sensory probes into named MaleCNS v1.0 cell types** (`neurofly_daemon.py:455-485`, `:656-721`; declared `status: unverified` in `provenance.py:106-136`).
   - All of them are **non-lateralised**. The bilateral antennal mean drives one pooled population, so left/right odour or wind differences never reach the graph.
   - Each pool is the first ≤ 50 cells **in table row order**. That order is side-biased: ORN_DM1 is 35 R / 15 L, ORN_DA1 is 43 R / 7 L, and the JO pool is 46 L / 4 R.
4. **Several assays deliver their stimulus to the graph under a key the controller never reads.**
   - Multisensory emits `wind_magnitude` (`maze.py:3565`) while the probe reads `wind_speed` (`neurofly_daemon.py:708`).
   - T-maze, open arena and labyrinth emit a wind vector but no `wind_speed`.
   - Result: **Johnston's-organ (JO) input reaches the graph only in the wind tunnel**, and only when airflow exceeds 3 mm/s.
5. **The `hybrid-bridge-rpc-experimental` path has a packet/key mismatch.**
   - `connectome_bridge.py:687-717` sends `pn_dm1_norm`, `odor_diff`, `wind_speed`, `cva_rate` and `temperature`.
   - The server reads `mean_odor`, `wpn_wind_speed`, `pheromone_cva`/`courtship_cva` and `temperature_excess`/`temperature_c` (`brainlab/cosim_server.py:345, 385, 393, 401`).
   - So in that mode odour, cVA, wind and temperature never reach ORN, JO or thermo cells. Only `looming_trigger` (and optomotor) arrive.
   - This was found by static reading only and is unverified by a run. It is reported, not fixed: fixing it requires an encoder change.
6. **Taste (sugar, bitter, water) has no environmental field, no encoder and no probe.** `food_contact` is a boolean distance test feeding a metabolic counter (`arena.py:1760-1768`). MaleCNS contains 1,428 cells of class `gustatory`, but none is driven.
7. **No live predator exists.** The daemon always builds arenas with `num_predators=0` (`neurofly_daemon.py:2197`, `:4771`) and rejects predator placement (`neurofly_daemon.py:4964-4965`). The only "predator" input is the looming-escape assay's abstract expanding disc.
8. **CO2, geosmin as such, acids, esters as distinct odorants, and cVA via Or65a are not simulated.**
   - The `odor_b` "danger" channel drives ORN_DA2 (the geosmin glomerulus, Or56a) as a generic aversive odour.
   - The `odor_a` "food" channel drives only ORN_DM1 (Or42b).

## 1. Inventory: assays, fields, units, telemetry

### 1.1 Antennal sampling geometry

- **Antenna positions.** Two points at `antenna_length = 2.0` mm from the body centre, at heading ± `antenna_angle = π/4` (`arena.py:81-82`, `arena.py:157-167`). This geometry is engineered: the two sample points are about 2.8 mm apart.
- **Paradigm assays.** The field is sampled at both antennae (`arena.py:1292-1312`). The bilateral mean then **overwrites** `stimuli['odor_a']` and `stimuli['odor_b']` (`arena.py:1742-1747`). Left/right values are kept only in `fly.sensory_input`, which the daemon does **not** publish.
- **Open arena** (`paradigm=None`, `neurofly_daemon.py:2196`). Sampled the same way (`arena.py:1314-1331`); the published values are the antennal means (`arena.py:2106`, `neurofly_daemon.py:4059-4062`).
- **Not sampled at the antennae.** cVA, aphrodisiac, temperature and wind are read at the **body centre** (`arena.py:1715-1718`, `:1752-1753`). The exception is temperature, which also gets per-antenna `temperature_left`/`temperature_right` (`arena.py:1310-1311`, `:1744`).

### 1.2 Per-assay sensory fields (paradigm `sample_stimuli`)

| Assay | Odour field (arb. units, 0–1) | Wind (mm/s) | Other chemosensory/contact | Source |
|---|---|---|---|---|
| open arena | `odor_a` food: sum of Gaussians, σ 18 mm. `odor_b` hazard: σ 14 mm. Clipped to 1. A predator kill adds an `odor_b` source of 1.8 (dead code in the live daemon: no predators). | `arena.wind`, default (−0.3, 0). Scalar control 0–40 toward −x | Food eaten within 3.5 mm, then respawned | `arena.py:676, 729-731, 1266-1282, 1966-1978, 1994-2007`; `assay_controls.py:33-35` |
| t_maze | `odor_a`/`odor_b`: Gaussians σ 22 mm at the arm ends (15, 50) and (125, 50); `odor_cs_plus`/`odor_cs_minus` | Piecewise vacuum ±15 (vector only, no `wind_speed`) | `food_contact` within 4 mm of the CS+ source | `maze.py:1941-1971` |
| wind_tunnel | `odor_conc`: lateral Gaussian σ = `filament_sigma` (3.5 mm) around the nozzle y, × exp(−downwind/250 mm). Static, not advected. | (−25, 0) plus `wind_speed`, `wind_direction_rad` | – | `maze.py:2602-2665`; controls `assay_controls.py:42-45` |
| looming_escape | – | – | Visual only: `theta_rad`, `theta_deg`, `expansion_rate_rad_s`, `time_to_collision_s`, `gf_membrane_potential_mv` (the last is modular-model output, not an input) | `maze.py:2751-2770` |
| courtship | `cva_concentration` = exp(−d/3 mm) if the female is mated; `aphrodisiac_concentration` if virgin | – | `female_type`, `inter_fly_distance_mm`; a "bitter pheromone" flag when mated and < 2.5 mm (modular/bridge only) | `maze.py:3164-3180`; `arena.py:1752-1754` |
| labyrinth | `odor_conc` = exp(−d/35 mm) from the goal | (0, 0) | `food_contact` < 4 mm | `maze.py:3295-3310` |
| multisensory | `odor_a` food at (+45, +45), `odor_b` repellent at (−45, −45), both σ 20. `odor_cva` = `cva_concentration`, 0.8·Gaussian σ 18 at (+45, −45) | `wind` (−15, 0), `wind_magnitude`, `egocentric_wind`, `jo_antenna_deflect_un` | `food_contact` < 4 mm; thermal terrain | `maze.py:3393-3400, 3512-3570` |
| heat_maze, buridan, visual_operant, optomotor, gap, DAM, y_maze | none relevant here (thermal/visual) | some emit a constant `wind` vector | – | `maze.py` (per class) |

**Preview-only code** (never a live field; label it so):

- `env_adapter.py:167-185` holds a downwind Gaussian plume used only by tests (`tests/test_flybrain.py`).
- The standalone browser preview has its own food, alarm, predator and wind tools and defaults (`web/app.js:716-719`, `:797-811`). Once connected to the daemon, only food and alarm placement in the open arena is forwarded (`web/app.js:787-795`, `neurofly_daemon.py:4961-4965`).

### 1.3 Encoders into the graph (default graph controller)

`GraphArenaController.__call__` (`neurofly_daemon.py:551-719`) merges `assay_stimuli` with the antennal `sensory` dict (`:650-653`).

| Probe | Stimulus key read | Current formula (arbitrary model units) | Threshold | MaleCNS cells driven (pool / total of type) | Source |
|---|---|---|---|---|---|
| `orn_food` | `mean_a`, else `odor_conc`, else `odor_a` | min(40, 35·s) | > 0.001 | ORN_DM1, first 50 of 74 (35 R / 15 L) | `:656-661`, `:465` |
| `orn_danger` | `mean_b`, else `odor_b` | min(45, 40·s) | > 0.001 | ORN_DA2, all 48 (21 L / 20 R / 7 unknown) | `:663-668`, `:466` |
| `courtship_cva` | `cva_concentration`, else kwarg `cva_odor` | min(35, 30·s) | > 0.001 | ORN_DA1, first 50 of 204 (43 R / 7 L); all are `fru_high` | `:700-705`, `:467` |
| `jon_wind` | `wind_speed` (only the wind tunnel emits it) | min(35, 0.15·v) | > 3 mm/s | First 50 `JO-*` cells of 672, mixing subtypes (EV, A, B, C, ED, FD …), 46 L / 4 R. No direction input. | `:707-713`, `:468` |
| `looming` | `looming_theta` / `theta_rad` / `theta_deg` | min(55, 15 + 40·θ) | θ > 0.15 rad | All LC4 (126) and LPLC2 (185), uniformly | `:686-697`, `:469` |
| `thermo` | `temperature` | min(40, 2.5·\|T − 24\|) | > 2 °C | annotation class `thermosensory` (25) | `:715-721`, `:481-483` |
| `photoreceptor_l/_r` | `retina_photoreceptors_l/_r` | 20·mean·contrast | – | R1-R6 by rootSide, ≤ 100 per eye | `:671-684` |
| `optomotor` | `optomotor_slip_rad_s` | WP5 encoder (pinned) | – | T4/T5 by eye | `brainlab/io_map.py` |

- **Cell counts** were read from the local MaleCNS `normalized/neurons.feather` and `annotations.feather` (rootSide), which matched pinned `graph_pins.json`. Counting is a table read, not a simulation.
- **Taste.** No sugar, bitter or water probe exists. The graph has 1,428 `gustatory` cells (putative ppk23 269, ppk25 257, IR52b 225, with leg, wing and labellar bristles and taste pegs), and none is driven.
- **Odour receptors.** The graph has 53 ORN glomerular types. Only DM1, DA2 and DA1 are driven. ORN_V (CO2, 55 cells), DM2, DM4, VA2, DP1m and VM2 are present and undriven.
- **Default dataset** is MaleCNS v1.0 (male, brain and VNC; `brainlab/datasets.json`, `graph_pins.json`). FlyWire 783 (female, brain only) is an optional separate dataset.

### 1.4 Modular and bridge paths (not the connectome)

- **Modular controller.** Mushroom-body KC encoding of `odor_a`/`odor_b`, a Johnston's-organ wind model (`mechanosensory.py`; `arena.py:1501-1512`), surge-cast, and chemotaxis using **bilateral** `diff_a`/`diff_b` (`arena.py:1565-1569`). This is a hand-built model; its steering is not connectome evidence.
- **Hand-built bridge** (`connectome_bridge.py`).
  - It labels array indices with glomeruli (`:145-151`): DM1 "vinegar Or42b", DM2 "esters Or22a", DA2 "geosmin Or56a", DA1 "cVA Or67d", DP1m "acid Ir64a", V "CO2 Gr21a/Gr63a".
  - Only DM1, DM2 and DA1 are ever assigned (`:591-609`). DA2, DP1m and V are declared and never used.
  - These are indices in a 54-element surrogate array, **not graph cells**.
  - The "Ir64a → DP1m" label is not supported by the abstract of Ai et al. 2010, which reports DC4.
- **RPC packet mismatch.** See §0.5: `connectome_bridge.py:687-717` versus `brainlab/cosim_server.py:345-406`.

### 1.5 What the daemon publishes (per frame)

All of these come from `_assemble_telemetry`, `neurofly_daemon.py:4056-4270`.

- **`stimuli`** (`:4192`): the full paradigm `sample_stimuli` dict, plus:
  - antennal-mean `odor_a`/`odor_b`;
  - `temperature_left`/`temperature_right`;
  - `wind` as a list.
- **`sensory`** (`:4221-4228`): `temp` (°C, 1 dp), `odor_a`, `odor_b`, `cva` (3 dp), `wind_x`, `wind_y` (mm/s, 2 dp).
- **`scene`** (`:4196-4210`), for source positions:
  - `food_pos`, `repellent_pos`, `pheromone_pos`, `goal_pos`, `hotspot_pos`, `cool_pos`, read from paradigm attributes (`:249-271`);
  - `food`, `hazards` (open arena);
  - `predators` (always empty live);
  - `nozzle_pos`, `filament_sigma`, `female_pos`, `female_type`, `cs_plus_arm` where present.
- **`fly`**: `x`, `y` (mm), `heading` (rad), `speed`, `radius`.
- **`connectome.input_stage[]`** (`:784-799`), one row per probe:
  - the declared probe fields: `name`, `cell_types`, `formula`, `threshold`, `status`;
  - `current_injected` (null when no cells are resolved);
  - `n_cells`, `n_spiking_this_step`, `unavailable`.
- **`connectome.dn_rates`**: per-step descending-neuron rates in Hz.

## 2. Primary biology (verified citations)

Each claim was checked against the Crossref record and the PubMed abstract (and, for Semmelhack 2009, the open full text). A claim not in an abstract is marked unverified.

| Stimulus | Detection pathway in D. melanogaster | Valence / caveats | Citation |
|---|---|---|---|
| Apple cider vinegar (blend) | At 3 ppm it activates DM1, DM4, DP1m, DM2, VA2 and VM2. Only DM1 (Or42b) and VA2 (Or92a) are needed for attraction, and each is sufficient. | Attractive at low concentration. At 32 ppm DM5 (Or85a) is recruited and attraction falls. Silencing DM4 or VM2 leaves attraction intact. | Semmelhack & Wang 2009, Nature 459:218, doi:10.1038/nature07983 |
| Acids (acidity) | IR64a neurons projecting to DC4 respond selectively to acids | Activation causes avoidance. The paper is about acidity, not vinegar attraction. | Ai et al. 2010, Nature 468:691, doi:10.1038/nature09537 |
| Acetic acid + yeast (social/sexual context) | Ir75a neurons | Increases female receptivity. Context-specific. | Gorter et al. 2016, Sci Rep 6:19441, doi:10.1038/srep19441 |
| Esters / general odours | Receptor-to-neuron map and odour-space coding. A single receptor can excite or inhibit depending on the odorant. | The specific mapping "ethyl acetate → Or42b/Or59b" is **unverified** in the abstracts. Treat it as unsupported until it is checked in the full text. | Hallem, Ho & Carlson 2004, Cell 117:965, doi:10.1016/j.cell.2004.05.012; Hallem & Carlson 2006, Cell 125:143, doi:10.1016/j.cell.2006.01.050 |
| CO2 | ab1C neurons co-expressing Gr21a + Gr63a, which are jointly sufficient; projects to the V glomerulus | **State-dependent valence.** Walking flies avoid CO2 from about 0.1 %. Flying flies track it (via Ir64a/Orco, not Gr21a/Gr63a). Active foraging flies are attracted (via IR25a). | Suh et al. 2004, Nature 431:854, doi:10.1038/nature02980; Jones et al. 2007, Nature 445:86, doi:10.1038/nature05466; Kwon et al. 2007, PNAS 104:3574, doi:10.1073/pnas.0700079104; Wasserman et al. 2013, Curr Biol 23:301, doi:10.1016/j.cub.2012.12.038; van Breugel et al. 2018, Nature 564:420, doi:10.1038/s41586-018-0732-8 |
| cVA (male pheromone) | Or67d → DA1, requiring LUSH and SNMP. Or65a mediates chronic and learned effects. | **Sex-specific.** In males it inhibits courtship of males and acutely raises aggression; chronic exposure via Or65a lowers aggression. In females it promotes receptivity. The DA1 lateral-horn projection is dimorphic and FruM-dependent. A 4-neuron, 3-synapse circuit links antenna to descending neurons. | Kurtovic et al. 2007, Nature 446:542, doi:10.1038/nature05672; Xu et al. 2005, Neuron 45:193, doi:10.1016/j.neuron.2004.12.031; Benton et al. 2007, Nature 450:289, doi:10.1038/nature06328; Ejima et al. 2007, Curr Biol 17:599, doi:10.1016/j.cub.2007.01.053; Liu et al. 2011, Nat Neurosci 14:896, doi:10.1038/nn.2836; Datta et al. 2008, Nature 452:473, doi:10.1038/nature06808; Ruta et al. 2010, Nature 468:686, doi:10.1038/nature09554 |
| Geosmin | Only Or56a neurons → DA2 | DA2 is necessary and sufficient for aversion, overriding other odours | Stensmyr et al. 2012, Cell 151:1345, doi:10.1016/j.cell.2012.09.046 |
| Sugar (**contact taste, not volatile**) | Gr5a (trehalose), Gr64f as co-receptor with Gr5a/Gr64a, on the labellum, legs and pharynx. Sweet and bitter (Gr66a) cells are separate, with segregated projections. Gr43a senses fructose, including haemolymph fructose in the brain. | Contact only. Gr43a valence depends on satiety. | Dahanukar et al. 2001, Nat Neurosci 4:1182, doi:10.1038/nn765; Jiao et al. 2008, Curr Biol 18:1797, doi:10.1016/j.cub.2008.10.009; Thorne et al. 2004, Curr Biol 14:1065, doi:10.1016/j.cub.2004.05.019; Wang et al. 2004, Cell 117:981, doi:10.1016/j.cell.2004.06.011; Miyamoto et al. 2012, Cell 151:1113, doi:10.1016/j.cell.2012.10.024 |
| Wind (Johnston's organ) | Wind-sensitive JO neurons respond tonically to static arista deflection, and subsets encode direction. They are distinct from sound neurons; separate gravity and sound groups exist. Direction is decoded from the **left–right difference** in antennal displacement (wedge projection neurons). | Ablating the wind neurons abolishes wind-induced locomotor suppression. Upwind turning on odour needs the antennal mechanoreceptors, in walking and in flight. | Yorozu et al. 2009, Nature 458:201, doi:10.1038/nature07843; Kamikouchi et al. 2009, Nature 458:165, doi:10.1038/nature07810; Suver et al. 2019, Neuron 102:828, doi:10.1016/j.neuron.2019.03.012; Bhandawat et al. 2010, J Exp Biol 213:3625, doi:10.1242/jeb.040402; Álvarez-Salvado et al. 2018, eLife 7:e37815, doi:10.7554/eLife.37815 |
| Odour plume navigation | In flight: surge upwind on contact (~190 ms), cast on loss (~450 ms). In walking: an upwind run on odour onset and a local search on offset. | Plume structure matters, which a static Gaussian cannot represent | Budick & Dickinson 2006, J Exp Biol 209:3001, doi:10.1242/jeb.02305; van Breugel & Dickinson 2014, Curr Biol 24:274, doi:10.1016/j.cub.2013.12.023; Álvarez-Salvado 2018 (above) |
| Looming (predator proxy) | LPLC2 (size) and LC4 (velocity) synapse onto the giant fibre (DNp01). GF spike timing selects short versus long takeoff. | Lateralised, retinotopic input in the fly; uniform in the model | Ache et al. 2019, Curr Biol 29:1073, doi:10.1016/j.cub.2019.01.079; von Reyn et al. 2014, Nat Neurosci 17:962, doi:10.1038/nn.3741 |
| Sex-specific chemosensation | FruM in about 2 % of male neurons, including pheromone ORNs. IR84a FruM+ OSNs respond to food odours and promote male courtship. ppk23/ppk29 fru+ leg neurons are contact-pheromone sensors. | Male-specific circuitry. The female FlyWire brain does not represent male FruM arbors. The default MaleCNS graph is male. | Stockinger et al. 2005, Cell 121:795, doi:10.1016/j.cell.2005.04.026; Grosjean et al. 2011, Nature 478:236, doi:10.1038/nature10428; Thistle et al. 2012, Cell 149:1140, doi:10.1016/j.cell.2012.03.045 |
| Datasets | FlyWire is one **adult female** brain; MaleCNS v1.0 is male, brain and VNC | Never mix them. Appearance sex (W2) must not imply that the dataset changed. | Dorkenwald et al. 2024, Nature 634:124, doi:10.1038/s41586-024-07558-y; Schlegel et al. 2024, Nature 634:139, doi:10.1038/s41586-024-07686-5; Zheng et al. 2018, Cell 174:730, doi:10.1016/j.cell.2018.06.019; MaleCNS paper as recorded in `brainlab/datasets.json` |

## 3. Capability table

Column notes:

- **Observed** quotes only existing receipts. The connectome audit `docs/receipts/audit-20261005/C-assays-connectome.md` ran on `49dbb4d` with LIF v3, **before** the removal of the tonic DNb01 drive and the forward floor (`provenance.py:137-138` `removed_undisclosed_drive`). Its responses are historical and are not re-validated for `f96bdc3`.
- **Wiring** names cells in the default MaleCNS graph that our encoder actually drives. "Present, undriven" means the cells exist in the graph but no stimulus reaches them.

| Stimulus | Biological detection (cited §2) | Environmental field? | Adapter encoding? | Connectome wiring (MaleCNS) | Observed model response (receipts only) | UNSUPPORTED |
|---|---|---|---|---|---|---|
| Generic food odour (`odor_a`/`odor_conc`) | Vinegar → DM1/VA2 (Semmelhack 2009) | Yes: static Gaussian/exponential, arbitrary 0–1 | Yes, graph `orn_food` from the antennal mean | ORN_DM1, 50/74, side-biased | Audit C: subthreshold in t-maze (≤ 1.55) and labyrinth (≤ 0.95); suprathreshold food drive "ignites the latched state" (open arena, wind tunnel). No odour-guided graph behaviour. | Chemical identity, concentration units, bilateral graph input, VA2/DM2/DM4 recruitment, plume dynamics |
| Vinegar / acetic acid specifically | DM1+VA2 (attraction); Ir64a/DC4 for acidity (aversive) | No (only an unnamed `odor_a`) | No | ORN_DC4/VA2 present, undriven | – | **Yes**: entirely |
| Ethyl acetate / fruit esters | Ester coding, OR repertoire (Hallem 2004, 2006); a specific Or42b/Or59b mapping is unverified | No | No | – | – | **Yes** |
| CO2 | Gr21a/Gr63a, V glomerulus; state-dependent valence | No | No (the bridge `GLOM_V` is never assigned) | ORN_V (55) present, undriven | – | **Yes** |
| Geosmin | Or56a → DA2, aversive | No as geosmin. `odor_b` "hazard/repellent/CS−" is routed to DA2. | Yes, graph `orn_danger` (generic aversive) | ORN_DA2, 48/48 | Audit C: ORN_DA2 ≤ 1.78 in t-maze (subthreshold) | Labelling `odor_b` as geosmin. It is a generic aversive odour on the DA2 channel. |
| cVA | Or67d → DA1 (and Or65a); sex-specific valence | Yes: courtship exp(−d/3) at the body centre when mated; multisensory Gaussian | Yes, graph `courtship_cva` | ORN_DA1, 50/204 (fru_high), side-biased | Audit C: courtship cVA ≤ 18.8 ignites the generic latched AL state; courtship index 0.00; no P1 readout | Or65a path; sex-specific downstream readout; antennal sampling |
| Alarm / stress odour | (Stress odour includes CO2, per Suh 2004) | Open arena `odor_b` hazards; predator-kill source is dead live code | Via `orn_danger` | ORN_DA2 | Audit C open arena: ignition and latching | Biological identity of the "alarm" odour |
| Sugar taste | Contact GRNs (Gr5a, Gr64f, Gr43a) | **No.** Only a boolean `food_contact` within 3.5–4 mm. | **No** | 1,428 gustatory cells present, undriven | Modular audit B: open-arena eating counted by distance; not a neural response | **Yes**: taste, concentration, labellar/tarsal contact |
| Bitter / contact pheromone | Gr66a; ppk23/ppk29 | No field. Courtship "bitter" flag only. | Modular/bridge only (`gr32a_rate`); no graph probe | ppk23/ppk25 putative cells present, undriven | – | **Yes** in graph |
| Wind speed | Wind JO neurons (Yorozu 2009) | Yes: uniform vector per assay, mm/s | Graph `jon_wind` **only if `wind_speed` is present** (wind tunnel only) and > 3 mm/s | First 50 JO-* of 672, mixed subtypes, 46 L / 4 R | Audit C: wind tunnel JO 3.75 subthreshold, "JO cells silent"; multisensory "wind not delivered" (key mismatch still present at `maze.py:3565`) | Delivery in t-maze, multisensory and open arena; subtype selection |
| Wind direction | L–R antennal displacement difference (Suver 2019) | Vector available (`wind_x`/`wind_y`, `egocentric_wind` in multisensory) | **No** graph encoding; modular JO and bridge only | – | – | **Yes** in graph |
| Odour plume / transport | Intermittent plume structure drives surge/cast | **No** solver. The wind-tunnel shape is static. | – | – | Battery `experiment_data/ryzen_battery/wind_tunnel/*` (backend not recorded): 200/200 SURGE, 1.5 mm upwind in 4 s. Audit B (modular): never reaches the source. | **Yes**: show NOT SIMULATED |
| Looming / predator approach | LPLC2 + LC4 → GF | Yes: abstract disc θ(t), not a predator body | Yes, graph `looming` (uniform, bilateral) | LC4 126 + LPLC2 185 → DNp01 (2) | `docs/receipts/connectome_closed_loop_looming.json` (connectome-fixed): escape_triggered true. Audit C: "LC4/LPLC2 → GF is a genuine connectome path" (PARTLY). Detection threshold and escape motor are hand-set. | Direction/laterality, a predator body, a predator moving in the live arena |
| Predator body (spider, mantis) | – | **No** live predators (`num_predators=0`) | No | – | – | **Yes**: decorative only |
| Temperature | (out of this card's scope) | Yes, °C, body centre plus per-antenna values | Graph `thermo` | thermosensory class (25) | Audit C: heat ignites the AL latched state | – |

## 4. Display contract for W4

### 4.1 Telemetry keys actually available (per frame packet)

| Display element | Key(s) | Unit / range | Availability |
|---|---|---|---|
| Fly pose (for drawing the antennal sample points) | `fly.x`, `fly.y`, `fly.heading` | mm, mm, rad (+ CCW) | all assays |
| Antennal sample locations | **derived**: `fly.(x,y) + 2.0·(cos, sin)(heading ± π/4)` | mm | Not published. Draw them, labelled "derived from model constants `arena.py:81-82`". |
| Odour at the antennae (food channel) | `sensory.odor_a` (= `stimuli.odor_a`) | arbitrary normalised units, 0–1 | Mean of the two antennae only. The **left/right values are not published**, so show "L/R not available". |
| Odour at the antennae (aversive channel) | `sensory.odor_b` | arbitrary, 0–1 | Same as above |
| Odour at the body centre (assay field) | `stimuli.odor_conc` (wind tunnel, labyrinth); `stimuli.odor_cs_plus`/`odor_cs_minus` (t-maze) | arbitrary, 0–1 | per assay |
| cVA | `sensory.cva` (= `stimuli.cva_concentration`) | arbitrary, 0–1, **body centre** | courtship, multisensory |
| Wind | `sensory.wind_x`, `sensory.wind_y` (arena frame); `stimuli.wind_speed`, `stimuli.wind_direction_rad` (wind tunnel); `stimuli.wind_magnitude`, `stimuli.egocentric_wind` (multisensory) | mm/s; rad. `wind_direction_rad` is the direction the air flows **toward**. `egocentric_wind` is the upwind bearing relative to heading. | Uniform field: draw one arrow, not a vector field |
| Source markers | `scene.food_pos`, `repellent_pos`, `pheromone_pos`, `goal_pos`, `nozzle_pos` (+ `filament_sigma`, mm), `food[]`, `hazards[]` (open arena), `female_pos`, `female_type`, `cs_plus_arm` | mm | Where present. **No emission rate exists.** Emission is a fixed unit amplitude; show "emission: unit amplitude (model constant)". |
| Food contact | `stimuli.food_contact` | boolean (distance < 3.5–4 mm) | Label "contact by distance, not taste" |
| Looming | `stimuli.theta_deg`, `theta_rad`, `expansion_rate_rad_s`, `time_to_collision_s` | deg, rad, rad/s, s | looming_escape |
| Delivered receptor input | `connectome.input_stage[]`: `name`, `cell_types`, `current_injected`, `n_cells`, `n_spiking_this_step`, `unavailable`, `status` | model current units (arbitrary) | graph backends only. `status` is `unverified` for every probe except optomotor. |
| Neural response | `connectome.dn_rates.*` (Hz), `n_spiking_this_step` per probe | Hz, count | graph backends only |

**Rules for the display:**

- Show three things in separate panels: field concentration, delivered input (`current_injected`) and neural response (`n_spiking_this_step`, DN rates).
- Delivered input with zero spikes means delivered but not detected: the input did not drive the cells to spike.
- `current_injected: null` means no cells were resolved for that probe; show UNAVAILABLE, not zero.
- Every concentration legend must read "arbitrary normalised units (0–1), not ppm or molar".

### 4.2 Must be shown as NOT SIMULATED

- Odour transport, advection by wind, turbulence, filaments, flow around obstacles.
  - The wind-tunnel plume is a static analytic shape.
  - Any particle animation must be labelled "illustrative" and must carry its equation.
- Chemical identity: vinegar, acetic acid, ethyl acetate/esters, CO2, geosmin-as-geosmin, yeast volatiles.
  - Allowed label for `odor_a`: "generic attractive odour → ORN_DM1 (Or42b) channel".
  - Allowed label for `odor_b`: "generic aversive odour → ORN_DA2 (Or56a) channel".
- Sugar, bitter or water taste; gustatory neuron input; sugar-water concentration.
- Bilateral (left vs right) olfactory or wind input to the connectome, and wind direction in the connectome.
- JO wind input in any assay except the wind tunnel above 3 mm/s. In particular, show it as NOT DELIVERED in multisensory (key mismatch) and in t-maze and open arena (no `wind_speed`).
- Predator bodies, predator movement and predator ecology. Looming is an abstract disc, and predator decoration must not alter `theta`.
- Emission rate, source depletion, odour lifetime.
- In `hybrid-bridge-rpc-experimental` mode, odour, cVA, wind and temperature are NOT DELIVERED to the graph (packet key mismatch, §0.5).
- Sex-specific chemosensory processing beyond the fixed dataset.
  - The default graph is the male MaleCNS. Appearance sex selection must not imply anything else.
  - The Or65a and female-receptivity pathways are not modelled.

## 5. Backlog (bounded follow-up cards; none is done here)

1. **Multisensory wind key.** Emit `wind_speed`, or read `wind_magnitude` (`maze.py:3565` versus `neurofly_daemon.py:708`). This is an encoder change and needs a card.
2. **RPC bridge packet.** Supply `mean_odor`, `wpn_wind_speed`, `pheromone_cva` and `temperature_c`, or align the server keys (`connectome_bridge.py:687-717` versus `brainlab/cosim_server.py:345-406`). Verify with a run.
3. **Lateralised probes.**
   - Split ORN_DM1, ORN_DA2 and ORN_DA1 pools by rootSide, and feed them `left_a`/`right_a`.
   - Select JO wind subtypes by annotation rather than row order.
   - Publish `left_a`, `right_a`, `left_b` and `right_b` in telemetry.
4. **Odour identity layer**, for example vinegar → DM1+VA2 at the cited concentration caveat, and CO2 → ORN_V with state-dependent valence. Each needs a mapping card and receptor-specific citations.
5. **Gustatory contact channel**: a sugar field plus contact at the tarsi or labellum, mapped to annotated GRNs, with citations.
6. **Plume transport model**, as an explicit solver with declared assumptions, before any live plume rendering.
7. **Bridge label cleanup.** `connectome_bridge.py:150` labels DP1m with Ir64a; the cited abstract says DC4. It also has unused DA2, DP1m and V indices. This is text and code hygiene.
