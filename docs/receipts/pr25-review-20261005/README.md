# PR #25 review (release gate G13), 5 October 2026

Branch `claude/review-pr25` = PR #25 head `0c183b9` + `origin/master` `9a6c1c5` + review fixes.
No simulation was run; WP7 still has no real-graph result.

## What was checked

| check | how | result |
|---|---|---|
| Full test suite before fixes | `pytest tests/` on the merge, real graph present, GTX 1660 Ti | 733 passed, **2 failed**, 9 skipped (`suite_before_fix.txt`) |
| Cause of the 2 failures | `test_wp7_specs_are_frozen[E0, E1]` | master `9b1023f` repointed `tmaze_odour_naive_v3.json` to `firing_rate_bounds_v3.json` after the PR pinned that file's sha256 (`7eb6f62b…`). Only `physiology.bounds_file` changed; the T-maze `encoder` block is byte-identical. |
| Full test suite after fixes | same | 738 passed, 1 failed, 9 skipped; the failure was `test_timing_determinism.py::test_requested_speed_does_not_change_state_or_weights`, a wall-clock pacing assertion (`wall < wall_1x / 3`) that fails while the machine is loaded; its determinism assertions passed. Rerun alone: passed (`suite_after_fix.txt`). With the extra test below the WP7 file has 22 tests. |
| Defaults leave the v3 graph untouched (real graph) | `real_graph_defaults.py`: master's and the PR's `apply_to_shared` with default options on the pinned MaleCNS graph (`4b2f87cc…`, 166,700 neurons, 25,582,938 edges) | weights byte-identical, identity, report and `describe()` equal; v3 graph hash `eab575f6…` unchanged (`real_graph_defaults.json`) |
| Spec counts against the real graph | same script, non-default variants | KC neurons 4,064; KC→KC edges 642,933 zeroed exactly; DPM neurons 2, released label `dopamine`; 4,934 DPM out-edges relabelled; CSR `ptr`/`post` unchanged; each variant has its own graph hash and `+mb(...)` dataset suffix. These preview hashes are a review check, not the spec's §9 step 0 record. |

## Fixes on the review branch

1. **Encoder-source re-pin** (`repin.py`). `encoder.from_spec_sha256` → `405cc81a…` in both
   WP7 specs, and the two frozen content hashes recomputed (E0 `0e81043c…`, E1 `f7b8fbca…`).
   This is the second such re-pin before any run, and it follows the PR's own precedent
   (`0c183b9`). It is recorded in spec §1.1.
2. **Stale hashes in the spec document.** Spec §1 still listed the frozen hashes from before
   the PR's own re-pin (`872875ed…`, `5f45ba6f…`). It now lists the current ones.
3. **v3-only guard.** `validation.harness.load_graph` now refuses `kc_kc`/`dpm` options
   unless `dynamics.version == 'v3'`. v4 and v5 take their graded and receptor-kinetics
   classes from the *released* transmitter table, so a DPM relabelled GABA in the weights
   would still be dopaminergic to them. A new test covers v1, v2, v4 and v5.
4. **Privacy.** "the Ryzen" (a host name) in spec §6.2.4 now reads "the reference machine's CPU".

## For the owner gate (not changed here)

The WP7 specs still use `firing_rate_bounds_v2.json`. Master has since moved the T-maze and
looming v3 specs to `firing_rate_bounds_v3.json`, which supersedes v2 (the KC bounds are
identical; MBON and the brain-wide bounds changed, and the brain bounds became verified).
Whether WP7 should follow is a change to frozen gate content. That decision belongs to the
spec's author or the owner before the status flips to `preregistered`.

Paths in the scripts are written as `<repo>/` and `<scratch>/`.
