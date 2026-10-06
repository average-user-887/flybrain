# Carrying learning across releases

Application updates must reuse compatible saved brains and preserve their history.
A new application version is not a reason to retrain or create a replacement brain.

| Saved material | Upgrade behavior |
| --- | --- |
| Raw observations, trial logs and recordings | Retain original files, identities and schema. Read older records with their original meaning. They are not automatically eligible training examples or new-contract measurements. |
| Learned plasticity deltas and trained readout weights | Restore when graph, neuron/edge ordering, input/output mapping and learning semantics are compatible. An application commit or version change alone is compatible. |
| Dynamic state, random state and learning traces | Restore for exact continuation when the checkpoint contains them. Missing legacy state must be disclosed; do not claim exact continuation from weights alone. |
| Different dynamics, topology or learning rules | Retain the original checkpoint. Require a separately validated migration or run it with its original model; never silently reset or reinterpret it. |

## Reuse existing stores now

Remember both existing directories once, replacing these example paths with your
actual retained paths. This command records their locations; it does not move,
copy, merge or initialize any brains:

```bash
neurofly storage use --output-dir /path/to/existing/outputs \
  --data-dir /path/to/existing/learning
neurofly storage show
```

On Linux, the selection is saved in `$XDG_CONFIG_HOME/neurofly/storage.json`
(default `~/.config/neurofly/storage.json`), or
`$NEUROFLY_CONFIG_HOME/storage.json` when explicitly configured. Configuration-home
overrides must be absolute. A saved missing or damaged selection refuses startup
instead of silently choosing a fresh directory. `storage show` checks directory
availability; the registry checks scientific compatibility when loading a brain.

After stopping the old process cleanly, start the new application with the original
model and assay options. `neurofly run` automatically uses the remembered roots.
Explicit per-run directory flags override that selection independently;
`NEUROFLY_DATA_DIR` overrides the remembered data root. Without a saved selection,
legacy defaults remain unchanged. Direct daemon scripts and existing service
launchers still need explicit directory flags; this command configures the CLI.

To select the directories explicitly for one run:

```bash
neurofly run --backend connectome-plastic --dynamics v3 \
  --graph-dir /path/to/verified/graph \
  --output-dir /path/to/existing/outputs \
  --data-dir /path/to/existing/learning
```

Select the original backend and assay, not necessarily those in the example.
Keep both directories; `--data-dir` alone does not locate saved brains. Do not run
two writers against the same stores, copy a live store as a migration, or overwrite
one brain with another. Test upgrades and rollback on a consistent copy first.

The supported `--continue-io-state` path creates a verified linked child for an
accepted input/output-method change and retains its parent. It does not convert
arbitrary graph topology, dynamics or learning rules. Fixed-connectome runs have
state and history to retain, but no learned synaptic weights to retrain.

## Release acceptance requirements

1. Remember explicitly selected persistent stores across installations; unresolved
   or missing existing data must not silently become a fresh brain.
2. Restore nonzero learned arrays, learning traces, readout rate, identity and the
   next deterministic step across an application-only upgrade. Checkpoints now
   persist WP6 traces per brain and readout rate. Older files missing these fields
   retain learned weights and record the transient/default-rate reset explicitly.
3. Preserve historical bytes and parent checkpoints through upgrade and rollback.
4. Refuse incompatible or damaged state before mutation, with a precise reason.
5. Demonstrate retained-data upgrades on NVIDIA and AMD, then show restored state
   in the running dashboard. These combined release checks are still pending.

Existing observations can inform later research without recollecting them. A new
scientific model may still need new simulation responses and held-out validation;
historical observations do not establish that the new model behaves correctly.
