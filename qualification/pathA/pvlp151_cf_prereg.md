# A2 PVLP151 counterfactual: preregistration (prepared, not run)

**LABELLED OFFLINE COUNTERFACTUAL.** This is not NeuroFly wiring and not a fix.
The original A2 FAIL (contract `cd69cb6d…aa1c97`) stays unchanged whatever this
shows. No edge is added: no gap junction and no invented connection.
The frozen record is `pvlp151_cf_prereg.json`, written by
`scripts/pathA_make_cf_prereg.py`. If this note and the JSON disagree, the JSON wins.

**Question.** Does the LC15 irrelevant-cell control's drive onto GF depend on the
LC15 → PVLP151 → GF route? The traced counts are 1689 contacts for LC15 → PVLP151
and 603 for PVLP151 → GF. LC15 has no direct contact with GF.

**Method.** The 4 PVLP151 cells (bodies 10173, 11677, 11826, 12275) are clamped using
the frozen contract's own silencing method: a constant `silence_drive` holds the cell
at the E_inh floor. The v3 engine transmits only spikes, so the clamp is equivalent to
zeroing exactly the cells' 1752 outgoing existing edges. Their incoming edges, every
weight and the graph file are untouched. `tests/test_pathA_cf_silence.py` checks this
in two ways. On a toy graph, the clamp gives the same spike counts as zeroing the
cells' outgoing edges. On the pinned graph, only PVLP151's outgoing edges are removed
and the sha256 is unchanged.

**Run.** The run uses the frozen runner `scripts/pathA_run.py` with the frozen contract
plus `--cf-silence-prereg` and `--cf-silence-prereg-sha256`. Without that flag, the
runner behaves exactly as before.

**Conditions.** The runner repeats these A2 conditions with PVLP151 clamped, at the
same rates, with all 8 seeds (0–7) and identical Poisson input trains:
- LC4 at 25, 50, 100, 150 and 200 Hz
- LPLC2 at the same rates
- LC15 at 100 and 200 Hz

That makes 96 counterfactual rows. Each one is paired with the frozen intact A2 row of
the same condition and seed. Three reproduction rows are added: the intact LC4@200,
LPLC2@200 and LC15@200 conditions at seed 0, run with the new code.

**Readouts.** GF, TTMn, PVLP151, DLMn (10 cells) and PSI (2 cells).

**Validity gates.** If any gate fails, the result is INVALID and no verdict is given.
- **V1:** PVLP151 is silent in every counterfactual row.
- **V2:** the three reproduction rows match the frozen A2 counts bit for bit.
- **V3:** all 96 rows are present, each bound to the frozen contract, graph and engine
  and to this prereg's sha256.

**Prediction (written before any run).** Claude expects SUPPORTED. The
counter-hypothesis is that other existing polysynaptic LC15 → GF routes carry the
drive. The secondary expectations are reported only:
- LC4 → GF changes little.
- LPLC2 → GF falls.
- TTMn, DLMn and PSI are reported without a predicted direction.

**Decision rule.** This rule is proposed by Claude. Root accepts or replaces it before
the run, and any change produces a new sha256. Define
M_r = GF(LC15@r, clamped) / GF(LC15@r, intact), using seed means, for r = 100 and 200.
- **SUPPORTED:** both M ≤ 0.5, and GF is lower under the clamp in 8 of 8 seeds at both rates.
- **REJECTED:** both M ≥ 0.8.
- **PARTIAL:** neither of the above, but GF is lower in 8 of 8 seeds at both rates.
- **INCONCLUSIVE:** anything else.

The thresholds are generic fractions: 0.5 means a majority of the drive and 0.8 means
at most a fifth. No counterfactual output exists yet. The earlier "≤ 24 Hz" threshold
was post hoc and is withdrawn, not reused. Some results are reported but not graded:
- the control ratios GF and TTMn of LC15 over LC4 under the clamp, compared with the
  frozen 0.1 limit;
- all paired ratios, as mean ± SD over seeds.

The analysis script is `scripts/pathA_cf_analyse.py`.

**Compute.** CPU only, 1 thread per process, no GPU. There are 99 runs of 1000 ms each,
at about 35–40 s per run (the frozen A2 timing). That is about 1.1 core-hours: about
66 min in one process, or about 17 min in 4 shards. The budget is 2 h, and the
contract's stop rule applies. The run starts only after Root approves it and the
cores are free. It must not run alongside S3.
