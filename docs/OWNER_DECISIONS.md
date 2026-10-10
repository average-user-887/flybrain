# Owner decisions

The project owner's rulings that constrain how NeuroFly's scientific results are built
and read, newest first. Every result in this repository must comply with all of them.
An entry may be superseded only by a later, dated entry here; earlier entries are
never edited.

---

## 2026-10-10 — Catch-up: what changed after 6 October, and the history rewrite

This entry brings the log up to date with facts that the repository itself shows. It
adds no new ruling and edits no earlier entry; entries below are as written on their
dates, including the commit IDs they cite (see the map at the end of this entry).

- **The 5 October "no rewrite" ruling was superseded in fact.** The public `master`
  history was rewritten once on 9 October 2026; commit `6fdff749` records it as an
  "owner-approved history rewrite". The owner's own wording of that approval is not
  transcribed in this repository.
- **Direction of 7 October 2026 (requested by the owner).** Milestones, in order: ship
  `v0.4.0` as an experimental instrument; split the brain and web processes; a batched
  sparse GPU engine; only then biology, starting with graded potentials in the early
  visual layers. The connectome is the featured backend. A separate, always-labelled
  reference-behaviour adapter is allowed for presentation only, and a FlyWire female
  brain is allowed as a separate graph that is never spliced into the MaleCNS graph.
  GPU position: CPU supported, AMD experimental and unqualified.
- **Shipped since 6 October** (see `CHANGELOG.md`): `v0.4.0` (7 October, ending the
  v0.4 HOLD recorded below), `v0.5.0` (8 October: split processes and the connectome
  as defaults, a labelled reference-fly walking demo, a separate FlyWire 783 female
  graph imported by CLI), and the pre-releases `v0.5.0rc1`, `v0.5.0rc2`, `v0.5.1rc1`
  and `v0.5.1rc2`.
- **Operating assignments.** The "GPT/Codex orchestration and implementation"
  assignment of 6 October below is not current. Later assignments are not recorded in
  this repository.

**Commit IDs cited in the entries below** (old to current; same subject and author
date): `713ba82` is now `0f144cab`, `5c54b03` is now `c2b641d7`, `69e52db` is now
`9c58db42`.

---

## 2026-10-06 (later) — GPT/Codex operating assignments

The owner's current assignment is **GPT/Codex orchestration and implementation**,
with **SOL workers for bounded tasks**. Independent **Antigravity review remains**.
This supersedes the Claude implementation, coordination and candidate-assembly
assignments in the earlier leadership, release-ownership and AMD-lane entries.
Those entries remain verbatim as historical decisions, not current worker assignments.

- **AMD GPU work remains mandatory and isolated on its own branch.** Its merge
  requires both the root Codex review approval and independent Antigravity approval
  of the exact candidate commit.
- **Main questions remain with the owner.** Worker assignment does not transfer
  authority to grant new scientific permissions or accept a release.
- Existing scientific rules, saved-data obligations and acceptance gates remain
  unchanged. This operating assignment records no browser pass or release acceptance.

---

## 2026-10-06 — Carry saved learning across application releases

The owner requires existing training data and compatible saved brains to carry
forward into the release. Application upgrades must not force collection or
training from scratch. Preserve checkpoints, learned parameters, raw records and
their original provenance; keep durable stores independent of replaceable code.
Scientific-model incompatibility requires an explicit, validated migration or a
precise refusal while retaining the original data, never a silent reset.
See `docs/TRAINING_CONTINUITY.md` for the implementation boundary and release gate.

---

## 2026-10-05 (later still) — Isolated AMD GPU acceleration lane. **Supersedes the AMD GPU deferral in `docs/DIRECTION_2026-10-05.md`.**

The owner assigned the Claude worker on the AMD Steam Deck to implement AMD GPU
acceleration. Earlier the same day, the direction document had deferred an AMD GPU
port; this decision replaces that deferral. The entry headed "(latest)" below is
older than this one and is left unedited, per the rule above.

Conditions set by the owner:

- **Isolation.** The work happens on a separate branch that does not interfere with
  `master`.
- **Merge gate.** Merging requires **both** the Codex review approval **and** the
  independent Antigravity audit approval.

Codex implementation scope under the owner's delegation (these are not the owner's own
words; changing them requires Codex review, not a new owner decision):

- The branch records its base commit. It must not touch the integration candidate or
  any default backend.
- Each approval is tied to the exact candidate commit. A later substantive change voids
  an approval until it is re-reviewed.
- **Recommended first target:** the existing fixed-weight v3 LIF stepping path, with the
  same equations, topology, weights, units and inputs.
- **Refusal:** unsupported dynamics or plasticity must refuse, or fall back visibly
  under a documented policy.
- **Comparisons:**
  - Numerical comparison criteria are registered before the final comparison.
  - Gains, wiring and tolerances are never adjusted to obtain a pass.
  - GPU execution and useful acceleration are reported as separate findings; a result
    that does not accelerate is reported as such.
- **Dependencies:**
  - GPU dependencies are optional, and the backend is chosen explicitly.
  - An unavailable device gives a safe error.
  - The CPU and NVIDIA paths are preserved.
  - No system driver or kernel changes are part of any automatic rollout.

The owner's standing scientific rules (entries below) apply unchanged. The main repair
work (audits A–F) continues independently. This decision approves no biological
approximation, invented connection or adapter-computed behaviour.

---

## 2026-10-05 (latest) — Project leadership delegation

The owner delegated **project direction and scope** to a separate Codex agent, and kept
a Claude agent responsible for day-to-day implementation, worker coordination and
assembling integration candidates.
- **Codex** reviews each candidate independently before master integration,
  deployment or release.
- **The owner's earlier rulings** in this file stay binding on both agents.
- **Codex's engineering direction and acceptance gates** are recorded separately in
  `docs/DIRECTION_2026-10-05.md`. They are implementation decisions under this
  delegation, not owner rulings on new scientific assumptions.

Release status following this delegation: v0.4 is on **HOLD** until the acceptance gates
in that direction document pass.

---

## 2026-10-05 — Personal data in git history: no rewrite

**Context.** An external audit on 5 October 2026 found personal data in the public
repository. The orchestrator verified it.
- **In the current tree:** a local username, absolute home paths, a hostname and
  internal share names. These are being removed going forward, with a CI guard to
  stop them coming back.
- **In git history, which cannot be cleaned without a rewrite:**
  - the owner's personal email on 27 commits, all merges made in GitHub's web
    interface;
  - a local machine email on 194 commits;
  - private Claude session links in 105 commit messages.

**Ruling.** The published history is **not rewritten**.

Rewriting would change every commit ID the project's receipts, pins and test reports
cite. It also could not truly erase anything: old pull-request commits stay reachable
on GitHub, and existing clones keep their copies.

**Instead:**
- the owner turned on GitHub email privacy, so web merges no longer expose the
  personal address;
- the repository's commit identity is now the GitHub noreply address;
- the current tree is sanitised, and a guard blocks reintroduction.

Approved by the owner on 5 October 2026 ("no rewrite, email privacy is on").

**Factual note (5 October 2026), added after the ruling; the ruling text above is
unchanged.** The figure of 105 commit messages with session links was the count when
the audit was taken, and commits published since then have added to it. Recounts with
their exact scopes:

- **`master` at `713ba82`** (its full ancestry): 244 commits; 145 of them carry a
  session link in the commit message; 223 fail at least one commit-metadata rule
  (message, author or committer) of the metadata guard under review.
- **Range `5c54b03..713ba82`**: 61 commits, 44 with a session link in the message.
- **All refs, from the independent Codex remote privacy audit of 5 October 2026**
  (52 live branch heads, `master` at `713ba82`): 153 reachable commit bodies contain
  a session link, and 17 of these are outside the ancestry of `69e52db`. There are
  128 reachable commits with author/committer identity findings, all inside that
  ancestry.

The two sets of counts differ because they cover different scopes: `master` alone
versus every commit reachable from any live branch. Neither supersedes the other.

The ruling covers **not rewriting** history. It does not approve the links published
on or after 5 October 2026, and it does not accept them after the fact. Those links
remain an **unresolved exposure** under the no-rewrite constraint. A clean current
tree and a commit-metadata guard prevent new exposure; they do not erase what has
already been published.

---

## 2026-10-05 (later) — The scan stays the scan. **Supersedes Ruling 1 below.**

> "We just need to make sure we're not contaminating the brain scan with our made up
> connections that make it work. If the scan is incomplete and unfunctional — say so."

- **No invented connections, in any form.** Neither a headline result nor a labelled
  variant may add, copy, interpolate or "complete" connections that are not in the
  scan. This withdraws the "completed lamina" variant allowed earlier the same day.
- **Incompleteness is a result.** Where the scan is incomplete and the circuit does not
  function on it, the project says so, plainly and prominently. That includes the lamina,
  where about 55 % of cells have no photoreceptor input. A non-functional result on the
  real scan is reported as such; it is not repaired.
- **Ruling 2 (fitting to non-motion recordings, with the tested behaviour held out)
  is unaffected.** It changes cell properties, not connections.

**Evidence the scan is currently uncontaminated** (checked 5 October 2026 on the
pinned engine graph `outputs/brainlab/malecns_v1/graph.npz` against the source
`connectome_data/malecns_v1/normalized/edges.arrow`):
- The engine has 25,582,938 edges, the same count as the source, and the
  `(pre, post)` edge set is identical. No edge is added or missing.
- Every weight is exactly `synapse_count × 0.275`, with its sign from the predicted
  transmitter: `|weight| / synapse_count` is 0.275000 for every edge.
- Declared run-time interpretations act on transmission only, never on connectivity,
  and do not alter the stored graph. The v3 transmitter policy, for example, gives
  aminergic neurons no fast weight.

---

## 2026-10-05 — Two edge cases of "the connectome does the work"

**Context.** About 55 % of lamina cells in MaleCNS receive no photoreceptor input,
because the dataset's lamina is incomplete (Nern et al. 2025). Verified on the pinned
engine graph:
- photoreceptor label `R1-R6`, 3,377 cells;
- zero-input L1 54.4 %, L2 54.2 %, L3 55.9 %.

Details are in `docs/LITERATURE_SCAN_2026-10-05.md` §A1. A literature scan also found
that no published model obtains T4/T5 direction selectivity from an unfitted
connectome.

**Ruling 1 — missing wiring.** Measurements and gates default to the columns whose
lamina cells actually receive photoreceptor input. Filling in the missing connections
(a "completed lamina") is permitted **only as a separately labelled variant**, never as
a headline result.

**Ruling 2 — fitting.** Cell properties may be fitted to **non-motion** physiological
recordings of input cells (for example responses to white noise or flicker), provided
the tested behaviour is **strictly held out**. For motion, that means T4/T5 direction
selectivity and steering are never fitted against. Fitting to the outcome being tested
remains forbidden.

Approved by the owner on 5 October 2026.

---

## 2026-10-05 — Release ownership

The owner handed full responsibility for the project to the Claude orchestrator, with
the goal of a full public release (v0.4.0), and authorised publishing to GitHub. See
`docs/RELEASE_PLAN_v0.4.md`.

---

## 2026-09-27 — Full connectome, not hand-built adapters

> "Full connectome, not hand-built adapters. We can adapt the connectome to our inputs
> somehow but it still has to do the work."

Encoders and decoders at the edges are legitimate if they are declared, logged and
pinned. An adapter that performs the computation the graph is meant to perform is not.
The concrete case is the optomotor encoder that drives T4/T5 by direction: it imposes
direction selectivity instead of letting the circuit compute it.
