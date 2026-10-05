# Functional audit, 5 October 2026: summary and proposed rebuild

**Why this audit happened.** The owner reported that "none of the assays work, the page
doesn't refresh, displays stale data and is static." The automated browser sign-off was
passing 7/7 at the time. It checks plumbing (pages load, counters advance, switching
does not freeze) and never checks whether an experiment does what it claims. Six
independent audits checked every function the way a user would, each on its own
isolated daemon in real Firefox. Detailed reports are in this folder (A–F); this page
gives the conclusions.

## Verdicts

| area | result |
|---|---|
| **A** Dashboard core | Updates, reconnection and two-tab sync work. **"LIVE" only means frames are arriving**, so a stuck fly, all-zero readouts or a dead simulation can still show LIVE with data age 0. |
| **B** 14 assays, modular controller | **1 works** (optomotor), 4 partly, 6 broken, 3 misleading. |
| **C** 14 assays, connectome | **0 work**; 2 partly (looming-escape, optomotor), 5 show no behaviour, 7 misleading. |
| **D** 97 controls (dashboard and studio) | 68 work, 11 misleading, 4 broken (2D/3D toggle broken everywhere), 4 no visible effect, 6 dead. The studio's controls all work. |
| **E** Code inventory | A clear core exists, beside a retired bridge stack, surrogate batteries, stale pages and helpers, duplicate encoders and decoders, and six servers. |
| **F** Robustness | **7 silent-freeze paths.** Any exception outside the arena step kills the simulation thread while status still says online. This caused the owner's frozen page on 5 October. |

## Root causes (few, shared)

1. **A hidden engineered drive on the connectome backends.**
   - The daemon injects a constant current into DNb01 every step and adds a +5 mm/s
     speed floor.
   - The identity bar meanwhile says "Assistance OFF · Motor: graph".
   - In 8 assays these two injected neurons are the only cells firing in the
     166,700-neuron brain, and they drive the fly straight into a wall.
   - This breaks the owner's rule that the connectome must do the work, and it is
     mislabelled on top of that.
2. **Trials are cut off at 20 simulated seconds,** while the outcomes need 25–60 s
   (8 modular assays). The fly also walks only about 20 % of the time.
3. **Some trials end at the instant of the response,** which erases it (looming-escape,
   gap-crossing).
4. **Learning is claimed and plotted as a curve where nothing learns;** several metrics
   are meaningless by construction or reward a stuck fly (B, C).
5. **Sensory gains on the connectome are unverified** and often below firing threshold
   at the assays' own stimulus strengths. A wind key mismatch means wind never arrives.
6. **The simulation thread can die silently** (F), and the page cannot detect a
   simulation that is not advancing (A).
7. **Modular panels are shown on the connectome backend** (the 120-Kenyon-cell mushroom
   body, an EPG compass in table order), plus dead and misleading controls (D).

**What is science, not software.** On the unmodified scan the v3 engine has no
spontaneous activity, does not compute motion direction from photoreceptors, and has an
incomplete lamina. Strong odour, heat or cVA input tips the network into the same latched
high-activity state whichever sense delivers it. These are results to report, not bugs to
hide.

## Proposed rebuild (for the owner and Codex to decide)

**Core: one dashboard and daemon, with two honest modes.**
- **Connectome mode, the scientific product.**
  - Remove the hidden drive and floor. The fly moves only when the scan's own
    descending neurons fire.
  - Show the real state in plain words: "brain silent", "runaway state, not
    stimulus-specific", or "responding".
  - Show only connectome panels, with DNb01 among the displayed rates.
  - Verify each assay's sensory mapping, or mark it unmapped. An assay with no verified
    input says "not testable on the connectome yet" instead of showing a score.
- **Modular mode, a clearly labelled hand-built demonstrator.**
  - Fix the 20 s cap, the response-erasing trial ends and the meaningless metrics.
  - Drop learning claims where nothing learns.
  - Label it "hand-built, not the connectome" everywhere.
- **Reliability first:** the silent-freeze fix and a liveness watchdog in the daemon and
  on the page, so the page never shows LIVE while nothing advances.

**Retire** (archive, never delete where a receipt depends on it):
- the hidden drive;
- the bridge stack (after untangling `arena.py`'s import);
- the surrogate batteries that are labelled as connectome;
- the stale dashboard fork;
- `web/research/`;
- dead controls and API actions;
- the broken sync script.

**Repair:** the 2D/3D toggle, controls that are disabled but look enabled, and the
training tab on graph backends.

**Keep separate:** the embodied FlyGym runner and the experiment studio, as research
tools.

## Decisions needed

1. **Which is the core product:** the dashboard and daemon (recommended above), or the
   embodied runner?
2. **Approve the retirement list.**
3. **When saving fails:** keep stepping in a visible degraded state (interim choice), or
   halt?
4. **Does v0.4 ship with the honest negative science as it stands,** or wait for the
   rebuilt core?
