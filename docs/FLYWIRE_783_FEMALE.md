# FlyWire 783 (female): a separate, labelled dataset

FlyWire 783 is the whole female adult brain (FAFB), imported as its **own** dataset
`flywire_783_female`, labelled **"FlyWire 783 (female)"**. It covers the brain only, with no
ventral nerve cord (VNC). MaleCNS v1.0 stays the default. The two graphs are never spliced,
their IDs are never mapped onto each other, and a store saved under one is refused by the other.

## Build it locally

```
python -m brainlab.flywire download   # fetch or re-verify the pinned sources (~1.1 GB)
python -m brainlab.flywire import     # nodes, edges, graph.npz, report (~30 s, ~1.4 GB RAM)
python -m brainlab.flywire verify     # check the build against brainlab/graph_pins_flywire_783_female.json
```

Outputs go to `connectome_data/flywire_783_female/` and `outputs/brainlab/flywire_783_female/`.
Checkpoints go to `outputs/registry-flywire_783_female/`. All three are git-ignored.
For brain-only runs, use `brainlab.flywire.load_shared()`.

## Sources (pinned in `data-provenance/flywire_783_female/source.lock.json`)

- Zenodo 10676866, version 783.0, CC BY 4.0. Each file is pinned by byte size, Zenodo
  MD5 and SHA-256: `proofread_connections_783.feather`, `proofread_root_ids_783.npy` and
  the two `per_neuron_neuropil_count_*` tables.
- `flyconnectome/flywire_annotations` at tag v3.2.0, commit
  `a83b2776d60d5764cef36b927f5f9679c16c47a2`. The file
  `supplemental_files/Supplemental_file1_neuron_annotations.tsv` is pinned by git blob
  `02e72f6c…` and SHA-256 `b214970b…`.
- Attribution: the FlyWire Consortium; Buhmann et al. 2021; Heinrich et al. 2018; Eckstein,
  Bates et al. 2024; Dorkenwald et al. 2024. For the annotations: Berg et al. 2026,
  Schlegel et al. 2024, Tastekin et al. 2026, Matsliah et al. 2024 and Dorkenwald et al. 2024.

## Method (hashed into the graph identity as `method_sha256`)

- **Nodes:** every proofread root ID, 139,255 in all. This includes 616 isolated cells and
  14 connected cells that have no annotation row. Those 14 carry
  `annotation_status = no_annotation_row` and an `unknown` transmitter. There are 7
  annotation rows whose IDs are not proofread roots; they are excluded and counted.
- **Edges:** the source has one row per pair per neuropil. Those rows are summed, giving
  16,847,997 rows → 15,091,983 pairs. Synapses are conserved exactly: 54,492,922 in and
  54,492,922 out, with 0 rows excluded. No threshold is applied.
- **IDs:** kept as exact 64-bit integers. They are never passed through float.
- **Sign:** `known_nt` is used first and `top_nt` is the fallback. The result then goes
  through the unchanged `brainlab.transmitters` rule:
  - acetylcholine is +;
  - GABA, glutamate and histamine are −;
  - conflicting, modulator-only or missing labels get the declared +1.

  `x-negative` means the cell was found *not* to use x. It carries no sign and is never
  read as "inhibitory". The full counts are in `normalized/report.json` under `sign`.
- **Weight:** `synapse_count × sign × 0.275`, the same arithmetic as MaleCNS. **Using the
  same 0.275 does not mean the two datasets are calibrated equivalently.** They differ in
  animal, sex, EM volume, synapse detector and transmitter predictor. Photoreceptor
  synapses are also under-detected in FlyWire (Matsliah et al. 2024).

## Inputs and outputs: unavailable

No encoder or decoder is declared for this dataset (`brainlab.flywire.IO_DECLARATION`):

- The MaleCNS DN channels are MaleCNS node indices and are never reused.
- The T4/T5 encoder reads MaleCNS columns.
- The public annotations carry no column (ommatidium) index, and no sourced mapping was
  found in either pinned source.

So the daemon, the dashboard and the embodied assays cannot run this graph. Pointed at
it, they refuse it by name. The Path B right-eye transfer run waits for a genuine,
sourced column mapping.

## Licence gate (open)

The annotation repository states no licence:

- the GitHub API reports `license: null`;
- there is no LICENSE file at the pinned commit;
- the README makes citation requests only.

So the TSV, and every table derived from it, stays local and is not shipped. That covers
`neurons.feather`, the signed `graph.npz` weights and the reports. Downloading the TSV
yourself is not licence clearance, and NeuroFly claims none. This gate stays open until
the maintainers answer.

Draft question for the maintainers (not sent):

> Hello — we'd like to use `Supplemental_file1_neuron_annotations.tsv` (v3.2.0) in NeuroFly,
> an open-source connectome simulator. The repository has no LICENSE file. Under what terms
> may we redistribute this table, or tables derived from it (per-neuron cell type and
> neurotransmitter sign), with full citation? Is it covered by CC BY 4.0, like the Zenodo
> 783 release? Thank you.
