# Owner decisions

The project owner's rulings that constrain how NeuroFly's scientific results are built
and read, newest first. Every result in this repository must comply with all of them.
An entry may be superseded only by a later, dated entry here; earlier entries are
never edited.

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
