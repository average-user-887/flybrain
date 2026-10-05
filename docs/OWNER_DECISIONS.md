# Owner decisions

The project owner's rulings that constrain how NeuroFly's scientific results are built
and read, newest first. Every result in this repository must comply with all of them.
An entry may be superseded only by a later, dated entry here; earlier entries are
never edited.

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
