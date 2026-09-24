# Validation harness (roadmap P1)

The harness turns "does the simulated fly do X?" into a preregistered,
repeatable verdict. A **spec** declares the stimulus, encoder, decoder, seeds,
controls, the behavioural gates with their published targets and citations,
and the physiological firing-rate checks. The **harness** runs the spec on the
brain and writes a **receipt**. The owner decided on 2026-09-24 that validity
means both things at once: behaviour matches the literature *and* firing
rates are physiological.

```bash
neurofly validate check validation/specs/optomotor_v3.json      # schema, bounds, gates
neurofly validate run   validation/specs/optomotor_v3.json --out outputs/validation/<stamp>
neurofly validate run   validation/specs/looming_gf_v3.json --synthetic --seeds 0 1 --out /tmp/x   # code path only
```

`python -m validation ...` is the same command. `--backend auto|cpu|cuda`
selects the brain backend. `auto` uses the GPU for v3 when a CUDA device is
usable, and `NEUROFLY_BRAIN_BACKEND` overrides it. The receipt records which
backend ran. The real graph is found through `NEUROFLY_GRAPH_DIR` and
`NEUROFLY_CONNECTOME_DIR`, as everywhere else.

## Layout

| path | what |
|---|---|
| `validation/specs/*.json` | the specs (`flybrain.validation-spec.v1`) and the firing-rate bounds (`flybrain.firing-rate-bounds.v1`) |
| `validation/specs/sources/` | literature data the specs were derived from (DoOR 2.0 extract with its source commit) |
| `validation/harness.py` | spec checks, gate statistics, physiology checks, verdict rule, receipt |
| `validation/paradigms/` | `optomotor`, `looming`, `tmaze_odor`; each is encoder, graph, decoder, then per-seed samples |
| `validation/runner.py` | one brain per graph, reset to rest before every trial; spike counts only, so CPU and GPU receipts mean the same thing |
| `validation/cells.py` | populations resolved by cell type and annotated side (`somaSide`, else `rootSide`), lowest source IDs, hashed into the receipt |
| `validation/synthetic.py` | a small typed synthetic graph with planted pathways, used for tests only |

## Verdicts

| verdict | meaning |
|---|---|
| `PASS` | confirmatory; every gate and physiology check passed; every gating value is verified |
| `PASS_PROVISIONAL` | as `PASS`, but some gating target or bound is marked `verified: false` |
| `FAIL` | confirmatory; at least one gate or physiology check failed |
| `INCONCLUSIVE` | confirmatory; something could not be evaluated (e.g. an undefined interval) |
| `EXPLORATORY` | the spec is a draft, the seeds were changed, the tree was dirty, or the spec is uncommitted |
| `SYNTHETIC_PLUMBING_ONLY` | synthetic graph; no claim |

A run is **confirmatory** only if all of these hold: the graph is real, the
spec status is `preregistered`, the spec's own seeds are used, the source tree
is clean, and the spec file is committed. The receipt names the commit that
last touched the spec, so it shows the spec predates the run.

## What each receipt carries

- the spec content, its sha256 and commit, and the bounds file's sha256
- the graph identity (the v3-transformed graph carries its own sha256) and the transmitter-policy report
- the dynamics version, the declaration and its pin
- the brain backend of every runner and the CUDA device name
- the git commit and dirty flag
- seeds, resolved populations with their hash, and per-trial results (`trials.json`)
- every gate with its statistic and CI, the reported-only items, and the physiology checks
- per window: whole-brain mean and max rate, fraction of active neurons, and neurons above 300 Hz
- simulated/wall time

## The three flagship specs

| spec | status | gates | what is verified |
|---|---|---|---|
| `optomotor_v3.json` | **preregistered** | syndirectional TI; each direction alone; half contrast; minus sham; minus shuffled graph; DNa02-silenced yaw exactly 0 | Every behaviour gate. Duistermars et al. 2007 and Mano et al. 2023 were read in full. The rest are causal controls of the model. |
| `looming_gf_v3.json` | draft | GF-spike fraction falls with log10(r/v); no GF spike without a stimulus | The slope's sign comes from von Reyn et al. 2014, but only the Fig. 1 caption and abstract were read. The encoder ramps and the stimulus elevation still need owner review. |
| `tmaze_odour_naive_v3.json` | draft | the fly chooses; OCT vs MCH balanced (PI within ±0.2); OCT and MCH avoided vs air | None of the targets were read. Tully & Quinn 1985 is paywalled, and the Li et al. 2013 control values were not extracted. The ORN encoder is DoOR 2.0 data from a pinned commit. |

The optomotor spec keeps the WP5 stimulus, encoder, decoder, schedule, seeds
and controls exactly as they were. Only the gates are new. The receipt also
reports the old WP5 decision rule (`paradigm_extra.wp5_rule`), so the v3
answer can be put next to the v1 receipt.

**No size is gated.** The decoders' gains are unit conversions, not fits, so
decoded yaw in rad/s cannot be compared with a fly's turning speed. For the
same reason the looming fraction (per loom) is reported beside the published
short-mode fraction (per takeoff), not gated against it.

**Physiology bounds are all unverified.** The one exception is ORN spontaneous
rate, which comes from the Hallem & Carlson 2006 data inside DoOR.data. The
brain-wide mean and the single-neuron maximum are declared design choices.
The best verdict any spec can reach today is therefore `PASS_PROVISIONAL`.

## Declared engineering assumptions

- **Optomotor.** Direction selectivity is put in by the encoder: T4/T5
  subtypes by eye (inherited from WP5). DNa02 steers ipsilaterally.
- **Looming.** LPLC2 is driven by angular size and LC4 by angular velocity,
  as clipped linear ramps with amplitude 20, the WP5 encoder amplitude. The
  ramp end points (10 to 60 deg, 1000 deg/s) are declared, not fitted. The
  stimulus is frontal and drives both eyes. There is no direct DNp01 input.
  Only GF-mediated (short-mode) takeoffs are decoded.
- **T-maze.** ORN rate = Hallem spontaneous rate (median 12.5 Hz where
  missing) + 200 Hz × DoOR response. The rate is turned into a drive by
  inverting the resting LIF rate, and the achieved rates are measured, not
  assumed. The antenna ipsilateral to an arm gets that arm's odour and the
  contralateral antenna gets none. The choice is decoded from the DNa02 L-R
  spike count. The body and the maze are not simulated; that is P2/P3.
- **All paradigms.** Every trial starts from rest. The v3 network has no
  spontaneous activity, so trial-to-trial variance comes only from encoder
  noise.

## Library check still needed (before the drafts can be preregistered)

1. von Reyn et al. 2014 Supp. Figs 2-6 and von Reyn et al. 2017: takeoff probability per r/v, angular size at takeoff, and the GF response to receding and dimming controls.
2. Tully & Quinn 1985 Table 1: naive OCT/MCH balance, and the 3-min PI needed for P4.
3. Li et al. 2013 Fig. 1I: naive OCT-vs-air and MCH-vs-air PIs.
4. Firing rates: Bhandawat et al. 2007 and Wilson et al. 2004 (PN), Turner et al. 2008 (KC), Hige et al. 2015 (MBON), Rayshubskiy et al. 2025 figures (DNa02), von Reyn et al. 2014/2017 (GF, LC4, LPLC2), Schnell et al. 2010 (HS).

When a value is read, set its `verified` flag in the spec or bounds file,
record where it was read, and commit the change before any run that uses it.
