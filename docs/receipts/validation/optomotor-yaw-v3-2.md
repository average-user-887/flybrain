# optomotor-yaw-v3-2: confirmatory run, 2026-09-24 — PASS_PROVISIONAL

- Spec: `validation/specs/optomotor_v3_2.json` (status `preregistered`), committed at `ca5be44` before the run
- Bounds: `validation/specs/firing_rate_bounds_v2.json`
- Host: Ryzen GPU, real MaleCNS graph, LIF v3 with the `v3-modulatory-only` transmitter policy; clean tree
- Seeds: 100-105 (new; v3-1 used 0-5); conditions: intact, sham_no_input, dna02_silenced, shuffled_graph
- Run: 2026-09-24 17:26-17:44, 300 s simulated in 1065 s wall (0.28x real time)
- Receipt: `~/Documents/ChatGPT/flybrain/outputs/validation/optomotor-v3-2-20260924-1726/receipt.json` on the Ryzen, sha256 `16669c11faba8076cc2e70bc923da6729123704a05f479a4d8683c6313f64768`. The raw JSON has not been copied into the repo. The values below were relayed from it and have not been re-read from the file.

## Read this first: the pass is not blind on HS

The previous run, v3-1, **FAILED** only because the HS cells fired at 60 Hz (left) and 53 Hz (right) against a 0-50 Hz ceiling ([record](optomotor-yaw-v3-1.md)). This spec, v3-2, was written **after** that failure was seen. It made the HS checks report-only, and it changed nothing else that could affect the verdict. The reason given is that HS cells signal with graded potentials and no published HS spike rate exists (see the spec's `amendment` block). That reason may be right, but the change was prompted by the failure. So this PASS is not an independent test of HS physiology. HS behaved as before (60 Hz and 53 Hz, see below). The v3-1 FAIL stays on record against `optomotor_v3.json`.

The only other thing that changed was the seeds, which made this a replication on fresh trials. Every behaviour gate, threshold, the stimulus, the encoder, the decoder and the controls are identical to v3-1. The three bounds that gate (DNa02, brain mean, brain max) have the same numbers in the v1 and v2 bounds files.

## Result

| id | check | mean | 95% CI | result |
|---|---|---|---|---|
| O1 | syndirectional turning index | 0.079 | [0.070, 0.088] | pass |
| O2 | leftward rotation alone | 0.131 | [0.111, 0.150] | pass |
| O3 | rightward rotation alone | 0.028 | [0.013, 0.042] | pass; positive on 5 of 6 seeds; weakest gate, as in v3-1 |
| O4 | half contrast | 0.044 | [0.033, 0.057] | pass |
| O5 | intact minus sham | 0.079 | [0.070, 0.088] | pass (identical to O1, so the sham's turning index was presumably 0; inferred from the relayed values, not read from the receipt) |
| O6 | intact minus shuffled graph | 0.090 | [0.075, 0.105] | pass |
| O7 | DNa02 silenced: yaw exactly 0 | | | pass on all 6 seeds |
| P1 | DNa02_L driven | 4.4 Hz | bound 0-100 | pass |
| P2 | DNa02_R driven | 1.8 Hz | bound 0-100 | pass |
| P3 | DNa02_L spontaneous | 0 Hz | bound 0-20 | pass |
| P6 | brain mean, driven | 0.47 Hz | bound 0-5 | pass |
| P7 | brain max single neuron, driven | 194 Hz | bound 0-300 | pass |
| **Verdict** | | | | **PASS_PROVISIONAL**: 7/7 behaviour gates and 5/5 physiology checks; provisional because all 5 gating bounds are `verified: false` |

Turning indices are in decoder units (rad/s). They are not comparable with a fly's turning speed.

**Reported, not gated** (the same items v3-1 reported, plus HS):

- Contrast ratio (half ÷ full contrast) 0.39 [0.26, 0.54], against the 0.7 plateau cut-off. That cut-off is a judgement based on Mano 2023, so this item was reported-only in v3-1 as well. The model's response falls with contrast more than the fly's does.
- HS_L driven 60 Hz and HS_R driven 53 Hz. These are the v3-1 numbers again, still above the 50 Hz ceiling that no longer gates.

## Unverified bounds and what would verify them

The five checks that gate all use bounds marked `verified: false`, so the best verdict this spec can reach today is `PASS_PROVISIONAL`.

| bound | used by | range | why unverified | what would verify it |
|---|---|---|---|---|
| `DNa02` | P1-P3 | spont. 0-20 Hz, driven 0-100 Hz | No absolute DNa02 rate appears in the text of Rayshubskiy et al. 2025 or Yang et al. 2024; the rates exist only in figures. Figure axes (up to 120 spikes/s) hint that 100 Hz may be too tight. | Take the rates from those papers' source data or supplementary tables. If none are published, the owner can accept digitised figure values. The paywalled full texts may need the owner's library access. |
| `brain_mean` | P6 | 0-5 Hz | A design choice that excludes the v1 runaway, not a literature value. Shiu et al. 2024 warn that absolute rates in these models are unreliable. | No single paper gives a whole-brain mean spike rate for the fly. Either the owner reclassifies this bound as a declared model-sanity limit (outside the verified/unverified scheme), or it is grounded in whole-brain activity estimates, which would need a literature search. |
| `brain_max_single_neuron` | P7 | 0-300 Hz | A design choice, set below the LIF refractory ceiling of about 450 Hz. | The same as `brain_mean`: owner acceptance as a sanity limit, or a published maximum sustained rate for central fly neurons. |
| `LPTC` (HS) | reported only | driven 0-50 Hz | HS cells are graded or spikelet-only; no HS rate in Hz is published. | Compare membrane potential rather than spikes. The LIF proxy only spikes, so this needs a graded-output readout or an owner decision to drop the bound. |

Verifying a bound means reading its value from the source, setting `verified: true` with where it was read, and committing that before any run that uses it. Under the harness rules, that change would go in a new bounds file and a new spec version; `optomotor_v3_2.json` stays as it is.
