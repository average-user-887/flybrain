# A2 PVLP151 counterfactual: preregistration (prepared, not run)

**LABELLED OFFLINE COUNTERFACTUAL.** This is a model counterfactual. It is not proof
that biology uses this route, and it does not repair A2. The original A2 FAIL
(contract `cd69cb6d…aa1c97`) and the unresolved electrical GF–TTMn wiring stay as they
are. No edge is added: no gap junction and no invented connection. The graph file and
the stored A2 results are never changed.

The frozen record is `pvlp151_cf_prereg.json`, written by
`scripts/pathA_make_cf_prereg.py`. If this note and the JSON disagree, the JSON wins.
The scope follows Astra's draft card and preparation review of 8 Oct 2026.

**Question.** Does the observed LC15 response depend on the existing outputs of
PVLP151? The traced counts are 1689 contacts for LC15 → PVLP151 and 603 for
PVLP151 → GF. LC15 has no direct contact with GF.

**Intervention.** The 4 PVLP151 cells (bodies 10173, 11677, 11826, 12275) are clamped
using the frozen contract's own silencing method: a constant `silence_drive` holds the
cell at the E_inh floor for the whole run. The v3 engine transmits only spikes, so the
clamp stops transmission on exactly the cells' 1752 outgoing existing edges.

**Run.** The run uses the frozen runner `scripts/pathA_run.py` with
`--cf-silence-prereg` and `--cf-silence-prereg-sha256`. Duration and dt are the
existing 1000 ms and 0.1 ms, and the network is reset to rest before every run.

**Conditions.** The conditions are LC15, LC4 and LPLC2 at 200 Hz, with seeds 0–7.
Each is run three ways:
- an intact sham replay;
- PVLP151 clamped;
- a reference that sets PVLP151's outgoing weights to zero in an in-memory copy (seed 0 only).

The no-input baseline (A2_LC4_0) is run intact and clamped.

**Readouts.** GF, TTMn, PVLP151, PSI, DLMn and GFC2.

**Proof from the real run.** These gates are checked in the run itself. The toy-graph
and pinned-graph tests are supporting evidence only.
- **G1:** PVLP151 has zero spikes in every clamped row.
- **G2:** every row records that, after reset, the membrane is at rest and the
  conductances, refractory counters, delay queue, active set and clocks are all zero.
- **G3:** the replay, clamp and zero-weight rows of one condition and seed carry the
  same sha256 for the input train.
- **G4:** at seed 0, the zero-weight reference matches the clamp row on every neuron
  except PVLP151.
- **G5:** each intact replay matches the stored frozen A2 counts bit for bit. Only after
  that are the frozen rows used as paired controls.
- **G6:** all rows are present and bound to the contract, graph, frozen engine and this
  prereg's sha256. The code sha is recorded, and graph.npz is unchanged after the run.

If any of G1–G4 or G6 fails, the result is INVALID. If only G5 fails, the same-code
replay is still the paired control, the frozen rows are not used, and the mismatch is
reported first.

**Report.** The report is descriptive only. There is no pass threshold and no
SUPPORTED or REJECTED verdict, and zero or contrary effects are valid outcomes. It
contains:
- absolute Hz for every seed, intact and clamped, with the baseline;
- the paired differences, with their mean ± SD;
- how many seeds went down, went up or stayed the same;
- a percent change only where the intact value is not zero.

The differences are **total network consequences** of removing PVLP151's output from a
nonlinear recurrent network. They are not fractions of the response assigned to a
pathway. Before the run, Claude expects GF under LC15 to fall, LC4 → GF to change
little and LPLC2 → GF to fall. This is recorded so it cannot be revised afterwards. It
is not a criterion.

The analysis script is `scripts/pathA_cf_analyse.py`.

**Compute.** CPU only, 1 thread per process, no GPU. The plan has 67 rows:
- 48 driven runs of about 36–41 s each, matching the frozen A2 @200 timing;
- 3 zero-weight audit runs;
- 16 no-input runs that take about 0 s.

That is about 0.55 core-hours: about 33 min in one process, or about 11 min in 3
shards. The budget is 1 h, with the contract's stop rule.

**Before launch.** An independent audit of source, IDs, intervention and state reset
is required, followed by Root GO. The run must be scheduled so that it does not compete
with S3, the replay or protected services.
