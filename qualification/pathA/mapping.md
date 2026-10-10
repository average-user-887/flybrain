# Path A phase 1: pathway mapping in MaleCNS v1.0 (read-only, no simulation)

Produced by `scripts/pathA_mapping.py`; full data in `mapping.json` (body IDs, node indices,
all shortest paths). Graph `graph.npz` sha256 `4b2f87cc…1d01` (pinned MaleCNS v1.0, 166,700
neurons, 25,582,938 chemical edges). Signs are the presynaptic cell's sign under the engine's
declared v3 transmitter policy (ACh +; GABA, Glu, His −; aminergic 0). Path cost = 1/contacts.
**Electrical synapses are not in the data.**

## Cell sets

| Set | MaleCNS rule | n | NT | Identity source | Confidence |
|---|---|---|---|---|---|
| JO-CE | type JO-CA1/CA2/CL/CM/ED*/EV* | 335 (L203/R132) | ACh | MaleCNS types = FlyWire types (flywireType) | type-level OK; all "RT Hard to trace" |
| JO-F | type JO-FD1/FD2/FV | 78 | ACh | as above | as above |
| aBN1 | type SAD093 | 2 | ACh | VFB FBbt_00111645: aBN1 = hemibrain/FlyWire SAD093; MaleCNS SAD093 | **name chain, corroborated by wiring** (899 JO-CE contacts in, 422 to aDN1, 290 to aDN2) |
| aDN1 | type DNg62 | 2 | ACh | released MaleCNS synonym "Hampel 2015: aDN1" | good |
| aDN2 | type DNge078 | 2 | ACh | released synonym "Hampel 2015: aDN2" (mancType DNfl023) | good |
| front-leg MNs | vnc_motor, exitNerve ProLN | 81 | Glu 68 / unclear 12 | released exit nerve | good (readout only) |
| aBN2 | — | — | — | several LB23 cells/side; **not identified in MaleCNS** | not used |
| LC4 / LPLC2 | type | 126 / 185 | ACh | type names | good |
| GF | DNp01 | 2 | ACh | synonym "Kennedy and Broadie 2018: GF" | good |
| TTMn | TTMn | 2 | Glu | type, exit nerve PDMNp | good |
| sugar GRN (CAL) | LB3b/LB3c/LB3d/LB4b, rootSide R | 36 | ACh | FlyWire v783 types of 20/21 Shiu 2024 sugar root IDs (LB3c 9, LB3d 5, LB4b 4, LB3b 2); MaleCNS flywireType says only "LB3" | **UNCERTAIN name match** |
| MN9 | MN9 | 2 | ACh | type (flywireType CB0701) | good |

Homonym warning: AOTU103m carries the synonym "aDN" (Lee 2002, sexually dimorphic); it is NOT the
grooming aDN and is excluded.

## Connectivity (chemical contacts)

| Stage | Direct contacts (sign) | Best excitatory path (contacts per hop) |
|---|---|---|
| JO-CE → aBN1 | 899 (+), 150 pairs, both aBN1 | direct (33) |
| JO-F → aBN1 | 131 (+), 40 pairs | direct (15); but JO-F → GNG301/GNG516 (GABA) → aBN1 (152/128/64 contacts, −) |
| JO-CE → aDN1 / aDN2 | 0 / 0 | JO → aBN1 → aDN1 (33, 155); → aDN2 (33, 139) |
| aBN1 → aDN1 / aDN2 | 422 (+) / 290 (+) | direct |
| aDN1 → ProLN MNs | 16 (+), 7/81 MNs | aDN1 → DNge039 → Fe reductor MN (122, 116) |
| aDN2 → ProLN MNs | 535 (+), 10/81 MNs | direct (Ta levator MN 140) |
| JO-CE / JO-F → ProLN MNs | 0 / 0 | 2 hops via DNg15 → Ti extensor MN |
| LC4 → GF | 6,362 (+), all 126 cells | direct |
| LPLC2 → GF | 4,862 (+), all 185 cells | direct |
| GF → TTMn | **90 (+), 2 pairs (70 and 20)** | direct; TTMn's largest inputs are inhibitory (IN13A022 987 −) and GFC2 (481 +) |
| sugar → MN9 | 0 | 4 hops, e.g. LB3d → ANXXX462a → GNG585 → GNG108 → MN9 |

## Risks stated before any run

- **A2 electrical relay:** GF → TTMn is largely electrical (shakB) in biology; only 90 chemical
  contacts carry it here. A TTMn failure may be a data limitation, and will still be reported as FAIL.
- **JO reconstruction:** every JO cell is "RT Hard to trace"; inputs may be incomplete.
- **aBN1 identity** rests on a VFB name mapping plus wiring corroboration, not on a released MaleCNS synonym.
- **Sugar GRN identity** for the calibration is a name match only.
- A1 is blind for NeuroFly but not novel: Shiu 2024 published the same prediction on FlyWire.
