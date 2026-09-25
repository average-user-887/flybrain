# Full MaleCNS test — 2026-09-14

> **Engine note (2026-09-24):** the results quoted here were produced by the original current-based **v1** LIF engine (runaway, about 10⁶ spikes/s; see `docs/LIF_DYNAMICS_SPEC.md`). Since PR #10 the commands below run **v3** by default (`Brain(graph)` applies v3 and its transmitter policy), so a repeat will not reproduce these numbers. Set `NEUROFLY_LIF_DYNAMICS=v1` to reproduce them.

Downloaded all three MaleCNS v1.0 source tables and verified their SHA-256 hashes against the preserved DOOMFLY source lock. Imported and compiled **166,700 neurons, 25,582,938 directed connections, and 124,177,617 synaptic contacts**.

Every released connection between retained neuronal entries is kept, including 10,299,701 weight-one edges and 101 self-connections. Raw segmentation objects outside the upstream neuronal inclusion policy are excluded and accounted for in `connectome_data/malecns_v1/normalized/report.json`. The full raw source tables remain on disk.

## Results

- Existing numerical and graph-integrity suite: **19 passed**.
- No-input control, 10 ms: **0 spikes**; resting state unchanged.
- R1–R6-only stimulation, 100 ms: **30,393 spikes**, with **0 downstream spikes**. This input alone does not establish propagation in the extracted model.
- Tonic lamina-only input, 100 ms: **28,523 spikes**.
- Combined retinal + tonic lamina input, 100 ms: **52,282 spikes in 10,119 neurons**. Of these, **3,179 spikes in 1,237 neurons** were outside directly stimulated cells. Counts changed in 7,474 neurons relative to tonic input alone; these totals do not imply that all downstream spikes were caused by retinal input.
- A fresh replay using ten 10 ms calls exactly matched the single 100 ms call's spike counts, membrane potentials, synaptic state, and refractory state.
- State remained finite and all synaptic weights remained unchanged.

Uniform currents: 20 model units to 3,377 R1–R6 cells, plus 12 units to 7,114 L1/L2/L3/L5 cells. Tonic lamina stimulation follows the upstream graded-cell proxy. These are artificial inputs, not calibrated vision.

The combined 100 ms simulation interval took **0.293 seconds** on CPU in this run (about 2.9× slower than real time). The validation process peaked at **1,148 MiB RSS**, including multiple model instances and data loading. This is a short workload measurement, not a sustained-performance benchmark.

Machine-readable evidence: `outputs/brainlab/malecns_v1/validation.json`.
Prepared graph: `outputs/brainlab/malecns_v1/graph.npz`.

## Repeat

From the repository root:

```bash
.venv/bin/python -m brainlab.validate_full
.venv/bin/python -m pytest -q
```

This establishes that the complete retained graph loads and the fixed-weight numerical simulator responds reproducibly. It does not establish biological validity, perception, learning, or game-playing ability. No GPU is required for these tests.
