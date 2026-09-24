# Brainlab — DOOMFLY without Doom

> **Engine note (2026-09-24):** the results quoted here were produced by the original current-based **v1** LIF engine (runaway, about 10⁶ spikes/s; see `docs/LIF_DYNAMICS_SPEC.md`). Since PR #10 the commands below run **v3** by default (`Brain(graph)` applies v3 and its transmitter policy), so a repeat will not reproduce these numbers. Set `NEUROFLY_LIF_DYNAMICS=v1` to reproduce them.

Experiment logging and comparison are ready: see [RUNS.md](RUNS.md).

The original repository was preserved locally in `upstream/doomfly` (git-ignored, not in this repository). `UPSTREAM.json` records its exact commit. The standalone package is `brainlab/`; it has no imports from the original repository and no game dependencies.

## Run now

```bash
bash ./run_brain.sh
```

This drives the first neuron of a **synthetic three-neuron chain**, then writes spike counts to `outputs/brainlab/demo.json`. It verifies that stimulation and propagation run without Doom. It is not a whole-fly experiment. First use includes Numba compilation time.

## What was extracted

- The fixed-weight Numba LIF propagation kernel, unchanged from upstream.
- Versioned MaleCNS download registry, source hashes, neuron inclusion rules, and loss-accounted connectivity importer.
- All-edge CSR graph preparation and the explicitly approximate neurotransmitter-sign rule.
- Independent Brian2 numerical comparison and upstream graph-integrity tests.
- Original licenses, attribution, and source snapshot available in the reference checkout (shallow clone).

The input API is now `Brain.step(currents, duration_ms)`: one current per neuron, outputting a spike-count array and elapsed wall time. Explicit nonzero current activates a neuron; activity can then propagate through every retained outgoing connection. There is no hidden background stimulation or readout-to-action mapping.

Excluded from the standalone package: ViZDoom, arenas, game assets, camera adapters, movement/attack decoding, damage rewards, game-conditioned plasticity, game server, web spectator UI, and deployment configuration. The fixed-weight baseline is the extracted model; experimental v6 learning and the optional C++ optimization remain in the preserved source, not this package. Existing anatomy viewer files remain usable separately.

## Run with the actual MaleCNS graph

The full graph data is **downloaded, checksum-verified, imported, and smoke-tested**. See `FULL_BRAIN_TEST.md` for results. The following pipeline reproduces the download of roughly 1.1 GB of source tables; normalization and graph preparation require additional disk space and several GB of RAM. No graph cropping or weak/self-edge pruning is added.

From the repository root:

```bash
.venv/bin/python -m brainlab.download
.venv/bin/python -m brainlab.connectome
.venv/bin/python -m brainlab.prepare
bash ./run_brain.sh \
  --graph outputs/brainlab/malecns_v1/graph.npz \
  --neuron-id 12781 --current 20 --duration-ms 100 \
  --output outputs/brainlab/malecns-response.json
```

Neuron 12781 is an example explicit stimulation target, not a calibrated sensory input. Numerical model: 0.1 ms steps, 20 ms membrane and 5 ms synaptic time constants, 1.8 ms delay, 2.2 ms refractory period, -52 reset/rest, -45 threshold, and upstream synaptic scaling 0.275. These are modeling assumptions, not physiological facts recovered from the connectome. No learning or behavioral validity is claimed.

## Environment and tests

An isolated `.venv` is prepared. To recreate it, use Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m pytest
```

The tests compare spikes and membrane/synaptic state against Brian2 at two input cadences and check edge preservation and annotation filtering. They do not validate biology. A separate full-graph run records workload-specific timing in `FULL_BRAIN_TEST.md`.

## Attribution

Derived from https://github.com/nftechie/doomfly (original code MIT); see `licenses/DOOMFLY-LICENSE` and preserved third-party notices. MaleCNS is the FlyEM/HHMI Janelia, Cambridge/MRC LMB, and Google Research dataset, CC BY 4.0. Historical failed experiments and original documentation remain in the upstream repository; their results are not results for this extraction.
