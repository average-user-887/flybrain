# Path A verdict (frozen contract sha256 cd69cb6d…aa1c97, rows bound to code 6bc971f)

These are the results of the frozen analysis with nothing changed after the runs: 336 runs (42 conditions × 8 seeds), v3 engine, CPU.

| Test | Verdict | Basis |
|---|---|---|
| A1 JO → antennal grooming (blind) | **PASS** | JO-CE drives aBN1 monotonically (63.6 → 134.4 Hz across 25–200 Hz). JO-F does not: aBN1/JO-CE ratio is 0.000 at 100 Hz and 0.001 at 200 Hz, despite 131 direct excitatory contacts. This matches the Shiu 2024 Fig. 5h calcium imaging. The olfactory irrelevant-cell control gives 0 on aBN1/aDN. |
| A2 LC4/LPLC2 → GF → TTMn (blind) | **FAIL** | GF part holds: GF rises monotonically, 238 Hz (LC4) and 236 Hz (LPLC2) at 200 Hz. TTMn/GF is 0.29 (LC4) and 0.25 (LPLC2), below the 0.5 relay criterion. The irrelevant-cell control is invalid: LC15 drives GF to 15–18 % of the LC4 response, above the 10 % limit. |
| CAL sugar → MN9 | CONSISTENT (not evidence) | Monotone rise; MN9 at 100 Hz is 94.6 % of its maximum (Shiu reports about 80 %). The sugar-GRN identity is a name match. |

Sub-results reported separately; none of them changes a verdict.
- **A1 S1:** aDN1 and aDN2 rise with JO-CE (holds).
- **A1 S2:** silencing aBN1 abolishes the aDN1 and aDN2 response (ratio 0, holds).
- **A1 S3:** front-leg (ProLN) motor neurons rise with JO-CE (4.3 → 5.7 Hz population mean). The JO-F value (0.28 Hz at 200 Hz) is only nominally > 0. It sits at the irrelevant-ORN level (0.23–0.25 Hz). At motor-neuron level the model therefore does NOT reproduce the JO-F-elicited grooming that Hampel 2020 reports as behaviour.
- **A2 S1 fails:** silencing GF leaves TTMn at 10–69 % of its response, so other chemical routes drive TTMn (e.g. via GFC2).
- **Shuffled-input counterfactual (labelled offline, not NeuroFly wiring):** it removes the aBN1 response (ratio 0) and almost all of the GF response (0.013 and 0.002).

Limitations:
- **A2 relay:** the biological GF → TTMn relay is largely electrical, and gap junctions are absent from the data. The P2 failure is consistent with that missing data, but it is still a FAIL.
- **GF firing is unrealistic:** GF fires at about 200 Hz in the model, whereas the biological GF spikes sparsely. The LIF proxy exposes no graded depolarisation.
- **A1 is a reproduction, not a new prediction:** the same prediction was already published on FlyWire (Shiu 2024). A1 is blind for NeuroFly, but it reproduces a published model result on a second animal.
- **Reconstruction:** every JO cell is "RT Hard to trace".
- **Determinism:** 175 earlier rows without provenance (code d30adaf) are kept, marked UNVERIFIED and not used. All 172 that overlap the re-run are identical.
