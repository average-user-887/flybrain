# Old-to-new SHA map: escape-asymmetry Path A stack

The 17 commits 7847aab4..d42985cd were rebased onto origin/master 6fdff749 on 2026-10-10
as branch `sci/escape-asym-20261010`. The rebase applied without conflicts; commit contents and
frozen prereg/result files are unchanged. The old branch is kept as
`codex/escape-asymmetry-prep-20261008-retired-20261010`.

| Old SHA | New SHA | Subject |
|---|---|---|
| 05cb8fd6ef77 | 7b92e1e9619b | Path A: MaleCNS pathway mapping and frozen known-circuit contract |
| 419978aa759a | 3f8a6ee79688 | Path A: bind every result row to contract, engine, graph and source |
| 67a4339f9c73 | 4d77629de5a5 | Path A: refuse provenance records with missing or malformed identity |
| 3da078fb81e9 | 827ca1bd6ddc | Path A: results of the frozen battery (A1 PASS, A2 FAIL, CAL consistent) |
| a81ec3f64c19 | f46f9189d163 | Path A: PVLP151 counterfactual prereg, first draft (superseded by the next commit) |
| 95971cb3876f | 73a1131d28a2 | Path A: PVLP151 counterfactual prereg reconciled to Astra's scope (not run) |
| b81f2eaa3336 | fb18ef67fcd9 | Path A: CF analyser G3 rejects missing, null or malformed input hashes |
| 5b0dbd904780 | 7fbe9e0eaa05 | Path A: pin the transmitter table and refuse impossible sparse counts |
| 832326f32d20 | 456f0f382768 | Path A: hash list of the original frozen A2 count files for the PVLP151 CF |
| d4b8636474e7 | 5b4228d3aa79 | Path A: PVLP151->GF edge inventory and direct-edge-deletion prereg proposal |
| 5b24e1271c55 | 182f0a570fdd | Path A: direct-edge deletion runner flag, reset fixture, shard launcher, analyser |
| 57ff2527bbca | 47ef7ce8c27b | Path A: write the reset fixture compressed (same arrays, small enough to commit) |
| cfaa1fcbec75 | 97080b420390 | Path A: freeze the PVLP151->GF direct-edge prereg with external pins (not run) |
| 9a4d83ed29ef | e7ec0e63c75c | Path A: checker-only amendment closing two G6 false passes (not run) |
| bb6610401dae | 22a6e801dd84 | Prepare static GF downstream inventory and bounded diagnostic proposal |
| a2c6e2eb6f16 | 0f7e16442efd | Freeze audited 75-row GF diagnostic preparation without launching |
| d42985cdd748 | 3191187f9551 | Prepare bounded GF chemical-edge deletion diagnostic |

Not reworded: two strings naming a storage host remain in
`qualification/pathA/pvlp151_cf_clamp_counts.sha256.json:9` and
`qualification/pathA/pvlp151_cf_original_counts.sha256.json:7`. Both files are sha256-pinned
by the frozen `qualification/pathA/pvlp151_gf_edge_prereg.json` (lines 29 and 32), so editing
them would break the freeze.
