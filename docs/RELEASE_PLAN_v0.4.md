# Release plan — NeuroFly v0.4.0

Owner of this plan: the Claude agent responsible for day-to-day implementation, working
under direction and acceptance gates set by a separate Codex agent. The owner delegated
those roles on 5 October 2026; see `docs/OWNER_DECISIONS.md` and
`docs/DIRECTION_2026-10-05.md`.

Status: **HOLD** (5 October 2026). A functional audit
(`docs/receipts/audit-20261005/SUMMARY.md`) found that the dashboard and assays do not
work as claimed from a user's point of view. The release waits until the acceptance
gates in `docs/DIRECTION_2026-10-05.md` pass on one reviewed, pinned candidate. Where
that direction document and this plan differ, the direction document governs.

---

## 1. What "full release" means for v0.4

A stranger with an ordinary PC, NVIDIA or not, can:

1. **install** NeuroFly from GitHub by following the README, with no knowledge of
   the owner's machines;
2. **obtain the MaleCNS data** through a documented, licensed path;
3. **run** the dashboard and at least one experiment end to end;
4. **read an honest account** of what the simulated fly does and does not do,
   with every scientific claim traceable to a committed receipt;
5. **reproduce** a published result from a pinned commit and seed, and know which
   parts are bit-reproducible and which are not;
6. **report problems** through GitHub Issues and Discussions.

v0.4 is *not* the release in which the connectome is shown to compute behaviour. The
current evidence says it does not yet (see §3), and the release must say so plainly
rather than wait for, or imply, a positive result.

---

## 2. Where things stand (5 October 2026)

| item | state |
|---|---|
*Updated later on 5 October 2026. Test counts are given only for the exact commit they
were run on.*

| item | state |
|---|---|
| Release | **HOLD**; see the status above |
| Code on GitHub | master moved several times on 5 October (install fixes, documentation, privacy guard, PRs #25/#30, hostname redaction, E_inh code restoration, audit receipts). None of these heads is a reviewed release candidate. |
| Last exact-commit full suite on master | `3e9bc5d`: 893 passed, 9 skipped, 0 failed (reference NVIDIA host, real graph, GPU). Later master commits are documentation only and have not had a full-suite run. |
| Functional audit (5 Oct) | modular controller: 1 of 14 assays fully works; connectome: 0 of 14 fully work (2 partial, 5 no behaviour, 7 misleading); 97 controls checked, several broken or misleading; 7 silent-freeze paths. See `docs/receipts/audit-20261005/`. |
| Plumbing sign-off (`scripts/live_ui_signoff.py`) | It checks page load, counters and switching, **not** whether experiments work. Its earlier 7/7 passes are not acceptance evidence. |
| GitHub releases | **none**, only a bare tag `v0.3.0` (24 Sep) |
| Version in `pyproject.toml` | 0.3.0 |
| Licence | MIT (code); MaleCNS data CC-BY 4.0; FlyGym/MuJoCo Apache-2.0 (see `NOTICE`) |
| Discussions | **on** |
| AMD / non-NVIDIA | Earlier AMD CPU testing at `5c54b03` passed (full suite 0 failures; cross-machine bit-identical embodied runs). Verification of the current fixes on AMD is pending. CPU only; no AMD GPU engine exists. |

---

## 3. Honest scientific status the release must carry

These are the findings the release notes and README must state, each with its receipt:

- **The live dashboard fly is the hand-built modular controller by default**, not the
  connectome. The connectome backends exist and run; none has passed a behavioural
  validation on the current engine.
- **Optomotor (WP5):** with an encoder that *imposes* direction selectivity on T4/T5,
  the v3 graph routes the signal to a DNa02-mediated turn (PASS_PROVISIONAL,
  `docs/receipts/validation/optomotor-yaw-v3-2.md`). The magnitude depends on an
  assumed chloride reversal potential (`docs/EINH_SENSITIVITY.md`).
- **The connectome does not compute direction selectivity** from photoreceptor input
  under v3, v4 (graded transmission) or v5 (receptor kinetics)
  (`docs/WP5_OPTOMOTOR.md` §13–§14). Two diagnosed causes: the signal attenuates
  ~500× before T4/T5, and paired T4/T5 inputs arrive in the wrong order.
- **Embodied loop:** stimulus-sign-correct turning of the FlyGym body, driven through a
  single DNa02 pair, replicated on 2 seeds, with a channel-swap control
  (`docs/receipts/embodied_stimulus_reversal.md`). An engineered decoder bridges the
  graph to the CPG; it is not a biological VNC.
- **Reproducibility:** v3 is bit-reproducible on the GPU; **v4/v5 GPU runs are not
  bit-reproducible run to run** on the real graph (differences ~1e-5 mV, same spike
  counts — `docs/receipts/lif_dynamics_v5.json`, falsifier F1). Cross-machine
  reproducibility is being measured (T3 below).

---

## 4. Release gates

Each gate has an owner, a check and a receipt. The release is cut only when every
**MUST** gate is green.

| # | gate | level | owner | check |
|---|---|---|---|---|
| G1 | Clean-room install works on NVIDIA Linux | MUST | orchestrator | fresh clone from GitHub into a new directory, README followed verbatim |
| G2 | Clean-room install works on AMD (CPU) | MUST | AMD test host (T1) | same, on the AMD test host |
| G3 | Data path documented and working | MUST | orchestrator | `neurofly` download command fetches and verifies MaleCNS from Janelia; attribution shown |
| G4 | Test suite green on NVIDIA and AMD | MUST | orchestrator, AMD test host (T2) | 0 failures; every skip explained |
| G5 | Real-browser dashboard sign-off on both | MUST | orchestrator, AMD test host (T5) | `scripts/live_ui_signoff.py` 7/7, two runs back to back |
| G6 | README and capability matrix match the receipts | MUST | orchestrator | every claim in README traceable; stale items fixed (status date, 18-DOF vs measured 66 joint DOFs, GPU reproducibility wording, test count) |
| G7 | Release notes and CHANGELOG | MUST | orchestrator | `CHANGELOG.md` covering 0.3.0 → 0.4.0; notes carry §3 verbatim in substance |
| G8 | Citation and attribution | MUST | orchestrator | `CITATION.cff`; MaleCNS CC-BY 4.0 attribution in README and NOTICE |
| G9 | No secrets or personal data in the published tree | MUST | orchestrator | **Not clean (corrected 5 Oct 2026).** The earlier entry here said the 29 commits published on 5 Oct were scanned clean; that was wrong. An external audit on 5 Oct 2026 found personal data in the tree and in commit metadata: the local username and home-directory paths (163 occurrences in 46 tracked files), agent scratch paths, the workstation hostname in 7 receipts, internal host and share names in docs, and a tracked `.claude/settings.json`. Branch `claude/sanitize-tree` removes these from the current tree (receipts redacted with placeholders, logged in `docs/receipts/REDACTIONS.md`; measurement scripts made repository-relative), untracks `.claude/settings.json`, and extends `scripts/check_private_infra.sh` (run in CI) so they cannot be reintroduced. **Accepted residual (owner decision, 5 Oct 2026, `docs/OWNER_DECISIONS.md`):** published history is not rewritten, because a rewrite would change every cited commit ID and could not truly erase anything. Older commits keep personal e-mail addresses in their author/committer metadata, private session links in some commit messages, and the removed strings in their trees; this is documented and accepted. GitHub e-mail privacy is on and the repository's commit identity is now a noreply address. G9 passes when the current tree is clean, the guard passes in CI, and no new commit carries personal data. **Correction (5 October 2026):** the session-link part of the history is not "documented and accepted". The owner ruling covers *not rewriting* history. Links published since the audit remain an unresolved exposure, not one approved after the fact. Counts by scope: `master` at `713ba82` has 244 commits, of which 145 carry session links and 223 fail a metadata rule; the range `5c54b03..713ba82` has 61 commits, of which 44 carry session links. Across all 52 live branch heads (Codex remote audit), 153 reachable commit bodies carry session links, 17 of them outside the ancestry of `69e52db`, and 128 commits have identity findings, all inside that ancestry. See the factual note in `docs/OWNER_DECISIONS.md`. A clean tree and the metadata guard prevent new exposure; they do not erase what has already been published. |
| G10 | Performance stated honestly | MUST | orchestrator, AMD test host (T4) | measured sim-s/wall-s on GPU, the reference host CPU and AMD CPU, with peak memory |
| G11 | Cross-machine determinism stated | SHOULD | AMD test host (T3) | same seed and commit on the reference host and the AMD test host; hashes compared |
| G12 | Discussions on, issue templates, CONTRIBUTING current | MUST | owner (toggle) + orchestrator | see §7 |
| G13 | Open PRs #25 and #30 resolved | MUST | orchestrator | each reviewed: merged with sign-off, or closed/deferred with a stated reason |
| G14 | Embodied extra installs and `verdict` runs | SHOULD | orchestrator, AMD test host (T6) | stimulus-reversal PASS on a clean install |

---

## 5. Work order

1. **Now:** publish current master (done, `5c54b03`). Pin the AMD test host to it (done).
2. **Docs pass (G6, G7, G8):** README status and claims, CHANGELOG, CITATION.cff,
   NOTICE check. Branch `claude/release-v0.4`.
3. **PR review (G13):** #25 is documents only, so review it for honesty and merge. #30 changes
   the dashboard, so code review, full suite and the real-browser sign-off, then merge or defer.
4. **Clean-room install on the reference NVIDIA host (G1, G3)** in parallel with **AMD test host T1–T5 (G2, G4,
   G5, G10, G11)**. Every install defect becomes a fix on a branch.
5. **Version bump** to 0.4.0 in `pyproject.toml`; regenerate anything that embeds the
   version.
6. **Release candidate:** tag `v0.4.0-rc1`, push, have the AMD test host repeat T1/T2 against
   the tag. Fix, re-tag rc2 … as needed.
7. **Release:** tag `v0.4.0`, publish the GitHub Release with notes, announce in
   Discussions.

Science work (v6a graded-synapse gain, the literature refresh) continues in parallel
and **does not block** v0.4. A result that lands before the release is included with
its receipt; one that lands after goes into 0.4.x or 0.5.

---

## 6. Explicitly out of scope for v0.4

- An AMD GPU (ROCm/HIP/Vulkan) engine. Decide for 0.5 from the AMD test host's CPU numbers.
- Any claim that the connectome computes behaviour on its own.
- Real-time performance.

---

## 7. Things that need the owner

Kept to what cannot be done from this machine without credentials:

- **Turn on Discussions** (GitHub → Settings → General → Features → Discussions), or
  provide a fine-grained token so the orchestrator can do it and publish Releases
  through the API. Pushing code works over SSH already; creating a GitHub *Release*
  object and toggling repository features need the API.

---

## 8. How integration works now

The orchestrator's sessions are worktree-isolated and cannot move the `master` ref
of the main checkout, which the live services run
from. Therefore:

- **GitHub `master` is the integration point.** The orchestrator lands work by pushing
  fast-forwards to `origin/master`, only after the full suite passes on the exact
  commit. Never a force-push.
- The main checkout follows with `git pull --ff-only` in the
  main checkout. Until it pulls, the running services keep the code
  they started with, which is safe.
- Branches stay on GitHub after merging, for provenance.

## 9. Coordination with the AMD test host

Through dated files on the project's archive share
(`ORCHESTRATOR_TO_DECK_*` and `DECK_*`). The first test scope (T1–T6) was sent on
5 October 2026 and pinned to `5c54b03`.
