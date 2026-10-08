# NeuroFly v0.5.1rc1 release notes (PRERELEASE)

*Release candidate, 8 October 2026. Base: `v0.5.0` (`33242c7`).*

> **Experimental prerelease; not biological validation.** This candidate repairs two
> sensory delivery paths and adds opt-in presentation tools. Everything in
> [`RELEASE_NOTES_v0.5.0.md`](RELEASE_NOTES_v0.5.0.md) still applies unless it is
> changed here.

## Read first: saved connectome runs need `--continue-io-state`

The graph I/O declaration moves from `graph-arena-io-v2-unassisted` to
`graph-arena-io-v3-unassisted`, because the multisensory sandbox's airflow now
reaches the JO wind probe. A run saved under v2, or with no recorded I/O, is refused
on an ordinary resume, and nothing is written. Restart with `--continue-io-state` to
create a linked child run from a verified checkpoint. The parent run is never changed
or reinterpreted.

## Fixed

- **Multisensory wind reaches the graph.** The JO wind probe reads `wind_speed`,
  else `wind_magnitude`; both are the world airflow speed in mm/s. The first key
  present wins, so an explicit 0 is kept, and a non-finite value is not delivered.
  The `jon_wind` input row reports `stimulus_key` and `delivery`. Assays that publish
  only a wind vector stay NOT DELIVERED. Gains, thresholds and formulas are unchanged.
  A delivered current is not evidence of sensing
  ([`SENSORY_DELIVERY_REPAIR_20261008.md`](SENSORY_DELIVERY_REPAIR_20261008.md)).
- **Experimental RPC bridge.** It now sends the odour, cVA, wind and temperature keys
  the server reads, and the server no longer injects non-finite inputs at the cap.

## Added (opt-in, presentation only, off by default)

- `?assets=hq`: a stylised fly and an arena shell for the dashboard 3D view and the
  embodied replay. The loader uses the neutral fly (`fly_hq_lod0`) only. The female and
  male appearance variants (two LODs each) are shown in the standalone gallery only
  until the new renderer is integrated; there is no variant selector in the dashboard or
  replay. Appearance never selects a brain dataset, physiology or behaviour. Recorded
  poses always drive the surface.
- `web/asset_gallery.html`: a standalone, manifest-driven review page with LOD, layer
  and overlay toggles, a contact sheet, and fly rig v2 with nine illustrative clips.
  Every clip carries the label "illustrative animation, not simulated behaviour" and is
  checked against the documented joint envelope and the floor.
- `web/env_inspector.html`: a read-only environment inspector following
  [`SENSORY_CAPABILITY_CONTRACT.md`](SENSORY_CAPABILITY_CONTRACT.md). It shows field
  concentration, delivered receptor input and neural response separately.
- Food, odour, wind and illustrative predator props, with icons and a sense legend.

## Optional asset download

The built GLB/PNG files are not in the wheel. `neurofly-hq-assets-0.5.1rc1.zip` holds
exactly the files the gallery manifest and the loader reference, with `SHA256SUMS`,
provenance and an `INSTALL.md`: unzip it into `web/assets/hq`. It is built
reproducibly by `tools/assets/package_hq_assets.py`.
