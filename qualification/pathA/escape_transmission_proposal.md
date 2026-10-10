# GF downstream diagnostic: root review proposal

1. **Decision requested:** accept the fixed design in `escape_transmission_proposal.json`, then prepare its independently checked launch package. This proposal authorizes no simulation and is not launch-ready.
2. **Question:** directly activate GF body 10001 alone, 10010 alone, or both at fixed 200 Hz; compare downstream per-cell counts with and without the ten frozen GFC2 cells clamped. Keep the accepted v3 engine, 0.275 scale, 0.1 ms tick, 1 s duration, drive/clamp amplitudes and eight seeds unchanged.
3. **Static evidence:** GF10001→TTMn800146 has 70 contacts; GF10010→TTMn804642 has 20. Both are positive under the pinned v3 policy. GF→GFC2 has 12 edges/144 contacts; GFC2→TTMn has 12 edges/481 contacts. The inventory verifies every reported edge against the separately pinned Arrow counts.
4. **Confounds to report:** GF→X→TTMn has 54 intermediate cells across 22 types, including eight of ten GFC2 cells. Other intermediate types include inhibitory second legs. GFC2→GF carries 13 positive contacts; TTMn→GF and TTMn→GFC2 carry 3 and 29 negative contacts under the proxy policy. Contact totals measure anatomy; no route share, necessity or mediation follows.
5. **Identity limits:** GF cells are labelled Roughly traced, targets Reviewed; rootSide is missing. Use body IDs, never infer left/right. Electrical edges remain absent from this prepared graph. A2 FAIL and all accepted original evidence stay unchanged.
6. **Inputs/controls:** 48 GF-driven rows (three input sets × two arms × eight seeds), 16 no-input rows, eight direct TTMn controls and three seed-0 empty-clamp shams = 75 rows. Direct TTMn stimulation tests drive/readout operation only. Clamp uses the accepted negative drive; no weights change.
7. **Preregistered readouts:** GF, both TTMn separately, all GFC2 separately, PSI, DLMn, PVLP151 and full sparse network counts. Report every seed, paired clamp-minus-intact mean/sample SD and lower/equal/higher counts; never select a gain, rate or threshold after seeing results.
8. **Success/failure:** all identity/state/input/controls/completeness/exit gates passing means a VALID descriptive diagnostic; any failure means INVALID and stops interpretation. TTMn counts above no-input in eight seeds mean consistently observed, one to seven seed-dependent, zero absent at this fixed input. Each outcome is retained; none validates biology or regrades A2.
9. **Evidence gates:** preserve full shared reset fixture and float32 v3 weights, per-row canonical state/weight hashes, exact input event bytes, independently regenerated RNG/input hashes, all raw counts and exclusive wrapper exit/log/startup receipts. Clamped GFC2 and both no-input networks must be zero; empty-clamp shams must exactly reproduce intact counts.
10. **Budget:** one CPU process, one thread, CUDA disabled; historical cost suggests about 0.5 core-hour plus setup/audit. Direct-GF runtime is unmeasured. Hard timeout 3500 s inside the child wrapper, one-hour card budget; stop on integrity failure, preserve partial output, no auto-expansion.
11. **Before root launches:** create a separate derived contract and bounded audit wrapper around unchanged `run_one`, with exact input retention; implement an independent analyzer and negative gate tests; freeze external source/contract/prereg/analyzer pins. The previous 99-row analyzer does not cover this design unchanged.
12. **Reproduce the read-only inventory** into a new output file; directories are supplied privately through environment variables. The helper imports no Brain or dynamics runner, refuses output overwrite and rechecks all four data pins after analysis:

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -B scripts/pathA_escape_inventory.py \
  --graph-dir "$NEUROFLY_GRAPH_DIR" --connectome-dir "$NEUROFLY_CONNECTOME_DIR" \
  --out /tmp/escape-inventory-review.json
python -B -m pytest -q tests/test_pathA_escape_inventory.py
```

13. **Exact evidence pins:** `escape_transmission_inventory.json` holds all four data hashes, accepted source `457d703038d6e7b8b77c71cd3ee933e00979a552`, accepted prereg `eb60ac9de13a0abd86e17ad7f1a45347220d074a1d6ab5d193f4add009fdbf8c`, contract, helper and v3 weight hashes. The proposal additionally pins engine, runner and reset fixture/state. SHA-256 of this proposal is bound by the review commit, rather than self-reported as launch authorization.
