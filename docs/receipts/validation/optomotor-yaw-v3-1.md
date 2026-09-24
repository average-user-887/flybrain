# optomotor-yaw-v3-1: confirmatory run, 2026-09-24 — FAIL

- Spec: `validation/specs/optomotor_v3.json` (status `preregistered`), code at `ee4c2c9`
- Bounds: `validation/specs/firing_rate_bounds_v1.json`
- Host: Ryzen, real MaleCNS graph, LIF v3 with the `v3-modulatory-only` transmitter policy
- Seeds: 0-5; conditions: intact, sham_no_input, dna02_silenced, shuffled_graph
- Receipt: `outputs/validation/optomotor-v3-20260924-1645/receipt.json` on the Ryzen, sha256 `a8702d6f…`. The raw JSON has not been copied into the repo yet. The values below were relayed from it and have not been re-read from the file.

| component | result |
|---|---|
| Behaviour | **PASS, 7/7 gates**: TI 0.065 rad/s, 95% CI [0.051, 0.079], positive on all 6 seeds; each direction passes on its own (rightward is weakest, CI lower bound just above 0); half contrast passes; sham, shuffled graph and DNa02-silenced controls all pass |
| Physiology | **FAIL, 5/7 checks**: P4 `HS_L` driven 60 Hz and P5 `HS_R` driven 53 Hz, both above the 0-50 Hz `LPTC` bound (`verified: false`) on every seed |
| **Verdict** | **FAIL** under the preregistered rule (behaviour AND physiology) |

This record stays unchanged. The HS ceiling was later judged a weak check, because HS cells signal with graded potentials. It was reworked in a new spec version, `optomotor_v3_2.json`, which was declared before any rerun. The FAIL above still stands against `optomotor_v3.json`.
