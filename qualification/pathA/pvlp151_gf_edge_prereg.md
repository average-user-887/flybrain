# PVLP151 → GF direct-edge deletion: proposed preregistration (for root review, not run)

**Status: proposal only.** This preregistration is not frozen and cannot run yet: the
runner support it needs is not implemented. Nothing runs before root review and GO.
The JSON record is `pvlp151_gf_edge_prereg.json`, written by
`scripts/pathA_make_gf_edge_prereg.py`. The read-only inventory it draws on is
`pvlp151_gf_inventory.json`, written by `scripts/pathA_pvlp151_gf_inventory.py`.

**Question.** The accepted all-output clamp of PVLP151 changed GF. Does deleting only
the existing PVLP151 → GF edges reproduce that GF change, or are indirect recurrent
routes needed? This question was raised after the accepted result and is labelled as
such. The analysis is descriptive only.

## Inventory (anatomy, not causal contribution)

- **Direct edges:** 8 edges, one from each of the 4 PVLP151 cells to each of the 2 GF
  cells.
  - They carry 603 contacts, all cholinergic and all excitatory (+) under v3.
  - The sum of their v3 weights is 165.8.
  - That is 3.4 % of PVLP151's 17,808 output contacts, spread over 1752 edges.
  - The contact counts match the edges.arrow synapse counts.
- **Two-hop routes PVLP151 → X → GF:** 382 intermediate cells across 135 types.
  - The sum of min-leg contacts is 2926.
  - The largest intermediate types are LPLC2, PVLP122, DNp103 and AVLP396 (excitatory),
    and PVLP024, CL367 and PVLP010 (inhibitory).
- **Feedback GF → PVLP151:** 5 edges, 18 contacts.

## Arms

Each arm runs on LC15@200, LC4@200, LPLC2@200 and the no-input baseline (LC4_0),
with seeds 0–7 and the inputs used before.

1. **Intact:** a sham replay of the frozen condition.
2. **Direct-edge deletion:** the 8 listed edges are set to weight 0 in an in-memory copy
   of the weights. PVLP151 itself is not clamped, and the graph file is untouched.
3. **All-output clamp:** the accepted intervention, unchanged.

A seed-0 empty-deletion sham runs the deletion code path with no edges removed. It must
equal the intact row.

## Shams and controls

- Intact rows must reproduce the frozen A2 counts.
- Clamp rows must reproduce the accepted 33110a4 clamp counts.
- The empty-deletion sham must equal its intact row.
- All-neuron controls are reported for every pair of arms: how many neurons changed
  their count, and the summed absolute count difference.

## Gates

- **G1:** the clamp holds (PVLP151 is silent in every clamp row).
- **G2:** each row records a clean reset plus a hash of the saved post-reset state.
- **G3:** all arms of a condition and seed receive identical inputs.
- **G4:** the deletion is exact, checked by a hash of the weight array actually used.
- **G5:** the shams above hold.
- **G6:** every row carries the full provenance pins, and every shard's exit code is
  captured as 0.

These gates close Astra's two limits on the accepted run: shard exit codes were not
captured, and the reset was evidenced by assertions rather than saved state.

## Report

The report is descriptive only. For each seed it gives absolute Hz and the paired
differences deletion − intact, clamp − intact and deletion − clamp. There is no
threshold, no "reproduces" verdict and no pathway fraction.

## Compute

There are 99 rows:
- 72 driven runs;
- 3 shams;
- 24 no-input rows that take about 0 s.

That is about 0.62 core-hours, or about 13 min in 3 single-thread CPU shards, within a
1 h budget. No GPU is used.

## Code needed after review (none implemented yet)

- a `zero_edges` field, with the edge indices checked against PVLP151 × GF;
- per-row hashes of the weight array and the post-reset state;
- an exit-code-capturing launcher;
- an analyser extension;
- tests, including negative fixtures.
