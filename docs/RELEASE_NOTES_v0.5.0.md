# NeuroFly v0.5.0 release notes

*8 October 2026. Base: `v0.5.0rc2`.*

> **Research software; not biological validation.** This release changes the launch
> defaults, adds a labelled reference-fly walking demo and a separate FlyWire 783
> female graph, and records the known-circuit results below, including a failure.
> Everything in [`RELEASE_NOTES_v0.5.0rc2.md`](RELEASE_NOTES_v0.5.0rc2.md) still applies
> unless it is changed here.

## Launch defaults

- **Split processes by default.** The simulation and the dashboard daemon run as
  separate processes without any flag. An installed launch with no process or backend
  overrides was observed using split processes and the connectome backend.
- **The connectome is the default backend.**
- **Visible no-graph fallback.** Without a prepared graph, `neurofly run` says so and
  shows a banner instead of silently substituting another controller.

## Reference-fly walking demo (not the connectome)

A reference controller from FlyGym 2.1.0 (Apache-2.0) can walk the fly in the live
arena. It is **not the connectome** and is labelled as a non-connectome reference
wherever it appears.

- Pause/resume, recorded replay and return to the preserved connectome run were
  exercised in the live arena.
- Diagnostic summaries carry reference time and steps, with the connectome run's
  context kept separate.
- This demonstrates presentation and recording. It is **not connectome locomotion**.

## FlyWire 783 female brain (separate dataset, CLI import only)

- A separate graph of 139,255 neurons, kept apart from the default male graph.
- **You obtain the data yourself** and import it from the command line.
- **The annotation licence is unresolved.** A licence inquiry has been sent. The
  package ships no annotation data and no derived annotation tables; follow the
  documented restrictions of the sources you download.
- Checks done: signs follow the known neurotransmitter first, weights are
  count x sign x 0.275, a full-brain CPU checkpoint/resume reproduced its arrays and
  spike counts, and restoring into the wrong graph is refused without changing the
  store. This was a synthetic 2 ms state-integrity smoke test, **not biological
  validation**.
- There is no female dashboard or body mapping.

## Known-circuit tests

- **Grooming (A1): PASS.** This is a reproduction of Shiu 2024 on a known circuit,
  not embodied grooming.
- **Escape (A2): FAIL.** It remains a failure. Missing electrical synapses are one
  possible explanation, and it is unproven.
- A sugar calibration was run; it is calibration only.

## What is not claimed

- **No motion or direction-selectivity success is claimed.** Controlled non-motion
  outputs were checked against the frozen contract, and no direction selectivity was
  established. In the lattice comparison both motion gates failed: the largest T4/T5
  response was 0.0471 mV against a required 0.5 mV, and the DNa02 left-minus-right
  difference stayed positive in both yaw directions.
- **v5 CPU/GPU equivalence is unproven.** The same nominal protocol gave different
  DNa02 results on CPU and GPU; the cause is not diagnosed.
- AMD GPU support remains experimental. An installed CPU smoke test passed on AMD
  hardware; that is not an AMD device qualification.
- Every UI assay selection was clicked and observed, and that is not evidence that
  every assay's biology works.

## Known cosmetic backlog

- The replay sidebar shows "undefined" for Duration and Drum.
- Repeatedly selecting the reference controller can bypass the requested assay
  validation. This is non-blocking, and the walking labels stay honest.

## Review

Scoped acceptance came from the maintainers' own checks and an independent static
review. The review was not an independent browser run.
