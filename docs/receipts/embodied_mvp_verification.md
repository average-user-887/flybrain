# Independent verification of `neurofly_body` (embodied MVP)

Verified 26 September 2026 against the ROOT checkout (`<repo>`,
branch `master`, embodied HEAD `f79d288`). The component is **not** on the
`worktree-neurofly-openready` branch; this file is the evidence record, and the
defects below are for whoever owns `master`. Nothing in the root checkout was
modified. Run artefacts: `<scratch>/embodied/`.

Environment: the root checkout's own `.venv`, genuine NeLy-EPFL FlyGym 2.1.0 from
PyPI, MuJoCo 3.9.0, CUDA brain backend. `flygym_demo` ships inside the FlyGym
distribution (51 RECORD entries), so it is stock, not a local fabrication.

## Verdict

The component is runnable as shipped and substantially honest. It is the first
component in this project whose headline claim survived being attacked. Thirteen of
its fifteen claims verified by execution; the two remaining verified by inspection.

**But its own shipped evidence does not establish what a reader will take it to
establish**, and controls it does not ship were needed to decide that the causal
loop is real.

## What the runs showed

All runs 5.0 s. The last row is the component's own control; rows 2-5 are conditions
it does not ship.

| condition | DNa02_L spikes | DNa02_R | final yaw (rad) | planar displacement |
|---|---|---|---|---|
| seed 0, world +4 rad/s | 30 | 3 | **+2.607** | 5.35 mm |
| seed 0, world -4 rad/s | 1 | 19 | **-1.195** | 3.29 mm |
| seed 0, world 0 | 9 | 10 | +0.004 | 3.56 mm |
| seed 1, world +4 | 26 | 0 | **+2.117** | 3.99 mm |
| seed 1, world -4 | 3 | 15 | **-1.017** | 3.29 mm |
| seed 0, +4, disconnected | 28 | 4 | +0.007 | **0.06 mm** |

Reversing the stimulus reverses the turn, 2 of 2 seeds in each direction. Zero
stimulus gives no net turn but still walks. Cutting the output stops locomotion.

## The causal control is sound

`mode` changes exactly one line of behaviour (`runner.py:312-314`); everything else
it touches is labelling. Over 2,500 steps of `--mode output-disconnected`:
`applied_cpg_drive` is `[0.0, 0.0]` in every record, the decoder still runs and still
logs non-zero output, the graph still steps (421,508 spikes vs 406,322 intact, same
graph and IO-map SHAs, same seed), and CPG magnitudes collapse to exactly zero, so
there is no hidden drive. Intact and disconnected are byte-identical until the first
step with non-zero applied drive.

## What it does NOT establish

1. **The intact/disconnected pair alone proves almost nothing about the connectome.**
   It proves the FlyGym CPG moves the body when driven and not when it isn't. Noise,
   a constant or a sine through the same decoder would produce the same gap. The
   claim that *neural output* drives the body rests on the stimulus reversal above,
   which is **not shipped and not mentioned in the README**. This is the main gap.
2. **The sample is two neurons wide.** `dna02_l` and `dna02_r` are single node indices
   out of 166,700, and the whole motor command derives from 1-30 spikes. The README's
   0.5 s example yields about three spikes; a 0.02 s run yields zero and makes the
   intact run byte-identical to its own control.
3. **No behavioural competence.** About 1 mm/s against 10-20 mm/s for a real fly. The
   fly shuffles and turns; it does not walk well.
4. **Baseline walking is not neutral.** At zero stimulus the fly still travels 3.56 mm,
   because both DNa02 neurons fire spontaneously (about 2 Hz). "No tonic or fallback
   walking drive" is true of the decoder, but the system does have a spontaneous
   walking drive from graph baseline activity. That distinction is unstated.
5. **No closed-loop DN modulation demonstrated.** Body feedback is wired, but its
   effect on the DN readout is not resolvable at these spike counts.

## Defects (patches proposed, none applied)

| # | Where | Problem | Proposed fix |
|---|---|---|---|
| D1 | `runner.py:315-316, 363` | `steps_with_nonzero_applied_drive` reads as sustained neural drive but is the 50 ms integrator ringdown of ~30 spikes (2,473/2,500) | report spike counts, mean drive, and steps above 1 % of max |
| D2 | `brainlab/cosim_server.py:377-381` | `raw_rate_l_hz` is a single neuron's count over a 2 ms bin, so it is quantized to {0, 500} Hz against a spec whose DNa02 band is < 20 Hz; three rate conventions coexist per record | rename to `dna02_spikes_l` plus `dna02_bin_rate_l_hz`; document that `filtered_rate_*_hz` is the only WP5-comparable quantity |
| D3 | `flygym_body.py:156-201` | FlyGym's stock leg-retraction and stumbling corrections and phase-driven adhesion stay active in BOTH modes, including when the command is zero (they explain the control's residual joint motion), but `last_info` is computed and discarded; `engineered_assistance_enabled: false` refers only to the neural backend and is easy to misread as covering the body | expose `last_info` and CPG phases; add the limitation explicitly |
| D4 | `neurofly_body/README.md` | Overstates the refusals: the CLI hardcodes dynamics and policy (`cli.py:161-170`), so these are library-level guards, not input validation. `docs/EMBODIED_MVP.md` is materially more honest | bring the README up to the doc; state the control is an output-path control, not a biological lesion |
| D5 | `README.md` | The 0.5 s worked example is below useful sample size | document 5 s; warn in `summary.json` when DNa02 spikes < 10 |
| D6 | `runner.py:129` | Output-directory collision is detected only after a full graph load | check in `cli.py:_run` before the imports |
| D7 | telemetry | About 8 kB per 2 ms record; the static identity block is re-serialised every step (~4 MB per simulated second) | keep identity in the manifest, write `identity_sha256` per record (deliberate format break) |
| D8 | `runner.py:364-370` | `any_neural_motor_output: true` at the top of a disconnected control's summary | rename to `decoder_produced_output`; add `motor_output_reached_body` |

## Recommendation

Ship the **stimulus-sign reversal** (ideally plus a rate-matched shuffled-drive
control) as the component's own built-in verdict. It is already possible with
existing flags and costs about two minutes of wall time. With it, the defensible
claim is: *stimulus-sign-appropriate turning of an articulated fly driven through a
single DNa02 pair of the real v3 MaleCNS graph, replicated across seeds, absent when
the output is cut.* Without it, the shipped evidence supports only *the decoder moves
the body*, while the README invites the stronger reading.
