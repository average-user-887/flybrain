# NeuroFly embodied MVP

This isolated package couples the verified MaleCNS fixed-weight graph to the stock
FlyGym 2.1 articulated fly. It does not modify or depend on the web dashboard or
the 14 assay controller. `docs/EMBODIED_MVP.md` has the install, data preparation,
decoder, lockstep and option details.

## The evidence: reverse the stimulus, does the turn reverse?

The intact/disconnected pair below shows only that the FlyGym CPG moves the body
when driven and does not when it is not. Noise, a constant or a sine through the
same decoder would produce the same gap. What distinguishes the neural signal is
the **stimulus-sign reversal**, and the command that runs it is built in:

```bash
NEUROFLY_BRAIN_BACKEND=cpu python -m neurofly_body verdict --output runs/embodied-verdict \
  --duration 5 --seeds 0 1 --full-controls --reuse-check-duration 5 \
  --graph-dir outputs/brainlab/malecns_v1 --connectome-dir connectome_data/malecns_v1
```

It runs both signs of the world angular velocity on two seeds, on one loaded graph
and body, plus a zero-stimulus baseline, the output-disconnected control and two
rate-matched drive controls. It exits 0 only on PASS: the turn follows the stimulus
sign in both directions on every seed, with at least 10 DNa02 spikes per reversal
condition, and the reuse check (the first condition re-run after all the others)
reproduces the same telemetry bytes over the whole condition. A failed reuse
check makes the verdict INVALID. A skipped one (`--no-reuse-check`) or one shorter
than a condition (the default 0.1 s window is a smoke comparison) makes it
UNVERIFIED, never PASS or FAIL. Pin the brain backend: CPU and CUDA agree
statistically, not bit for bit. It writes `verdict.json` and `verdict.md`. Eight
5 s conditions plus a 5 s reuse check take about 25 minutes on a CPU brain
backend.

Measured on the CPU backend, with both decoders:

- **`dn-v2` (the default decoder): FAIL.** The DNa02 left/right asymmetry does
  reverse with the stimulus on both seeds, but DNp09 is all but silent, so there
  is no forward drive for DNa02 to shorten, and MDN spikes drive a small symmetric
  reverse command. Every condition turns less than 0.04 rad. Receipt:
  [`docs/receipts/embodied_stimulus_reversal.md`](../docs/receipts/embodied_stimulus_reversal.md).
- **`dna02-crossed-v1` (legacy decoder): PASS.** Seed 0 turns +2.607 / -1.195 rad,
  seed 1 +3.107 / -1.017 rad; the turn is absent with the output cut, reverses
  when the left and right drive channels are exchanged (-2.457 rad), and same-sign
  turning survives the one fixed time permutation of the drive, at a smaller
  magnitude (+2.607 -> +1.966 rad); one permutation does not show that the time
  course is irrelevant. Scope: CPU backend, two seeds, motion sign imposed on
  T4/T5 by the encoder, engineered decoder. Receipt:
  [`docs/receipts/embodied_stimulus_reversal_legacy_decoder.md`](../docs/receipts/embodied_stimulus_reversal_legacy_decoder.md).

The legacy decoder makes DNa02 the only source of propulsion, which the literature
contradicts (see `docs/EMBODIED_MVP.md`), so its PASS is the claim *a DNa02
asymmetry that follows the stimulus can steer the articulated body*, not the claim
that the graph walks and steers the fly.

## Single runs

```bash
python -m neurofly_body run --duration 5 \
  --output outputs/embodied/intact-001 --mode intact
```

Five seconds is the documented example because the motor command is built from a
few descending neurons: a 5 s run yields tens of DNa02 spikes, a 1 s run a handful.
`summary.json` carries `decoder_input_spikes` (spikes of every population the
decoder reads), `dna02_spike_count`, and `dna02_spike_count_warning` whenever the
DNa02 count is below 10.

The output directory must not already exist; the collision is rejected before the
graph load. The run writes `manifest.json`, one neural/body record per 2 ms to
`telemetry.jsonl`, `timing.jsonl`, `summary.json` and, by default, `body.nfbody`.

The output-path control, with a fresh directory and the same seed and stimulus:

```bash
python -m neurofly_body run --duration 5 \
  --output outputs/embodied/disconnected-001 --mode output-disconnected
```

This still advances the same graph, sensory feedback and decoder and logs the
decoded command, but replaces the command reaching the FlyGym CPG with exact
zeros. It is an **output-path control, not a biological lesion**. `summary.json`
says which is which: `decoder_produced_output` is true in both modes,
`motor_output_reached_body` only in `intact`.

## What stays switched on in every mode

`output-disconnected` zeroes the CPG magnitude command and nothing else. FlyGym's
stock leg-retraction correction, stumbling correction and phase-driven tarsal
adhesion keep running in both modes, including at a command of exactly `[0, 0]`
(the CPG *phase* keeps advancing while its magnitude is zero), and they are what
moves the joints of the control. Their per-step values are in `telemetry.jsonl`
under `body.flygym_corrections` and `body.cpg_phases_rad`.
`engineered_assistance_enabled: false` refers to the graph's injected currents only.

## What the refusals actually are

The library fails closed unless the backend reports the verified real v3
configuration (`runner.validate_real_v3_status`). These are **library-level
guards, not input validation**: the CLI hardcodes the dynamics, transmitter
policy, unclear-transmitter mode and assistance flag, so no command-line argument
could ask for anything else. The guards matter for callers that build a backend
themselves.

## Reading the numbers

- `neural.dna02_spikes_l/r` and `neural.locomotion_dn.<population>_spikes` are
  integer spike counts per 2 ms bin.
- Bin rates (`neural.dna02_bin_rate_l/r_hz`, the legacy `neural.dna02_rate_l/r`,
  `neural.locomotion_dn.<population>_rate_hz`, dn-v2's `motor.decoder.raw_rates_hz`)
  divide a count by the bin, so one neuron's rate is quantized to multiples of
  500 Hz. Only the decoder's filtered rates (`filtered_rates_hz` in dn-v2,
  `filtered_rate_l/r_hz` in the legacy decoder) are comparable with the WP5 DNa02
  bands (under 20 Hz).
- Prefer `steps_with_applied_drive_above_1pct_of_max`, `mean_applied_drive_l/r`
  and `decoder_input_spikes` to `steps_with_nonzero_applied_drive`, which counts
  the decoder integrator's ringdown.

## Determinism

The same code, arguments, seed, graph and brain backend give a byte-identical
`telemetry.jsonl`; check a finished run with
`python -m neurofly_body replay-check RUN_DIR --output NEW_DIR`. Each intact or
output-disconnected verdict condition can be replayed the same way. CPU and CUDA
brains agree statistically, not bit for bit.

Telemetry format 2 keeps the static neural identity block in `manifest.json` and
writes one `identity_sha256` per record. A format-2 run has a different
`trajectory_sha256` from a format-1 run of the same arguments.

## Scientific scope

The DN-to-CPG decoders are explicitly labelled engineered bridges. This is a
testable neural-output to joint/body causal seam, not a ventral nerve cord, and it
does not model muscles or proprioceptive pathways. It does not establish
behavioural competence: the fly travels a few millimetres in 5 s, against
10-20 mm/s for a real fly. No closed-loop DN modulation is demonstrated.
