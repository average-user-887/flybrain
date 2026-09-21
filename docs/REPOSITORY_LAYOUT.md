# Repository layout and cleanup

The publication-facing MVP is the `neurofly_body` package: a verified MaleCNS
graph coupled to an existing physical body. The older modular simulator and its
14 assays remain available as separate baselines, not evidence of connectome
learning or complete biological control.

| Path | Purpose |
|---|---|
| `brainlab/` | Dataset download/normalization, pinned graph, neuron dynamics and server |
| `neurofly_body/` | Minimal physical-body adapter and reproducible run CLI |
| `tests/` | Unit/integration tests; optional real-data/body tests are explicit |
| `docs/EMBODIED_MVP.md` | MVP protocol, outputs and limitations |
| `docs/receipts/` | Versioned evidence, not generated runtime storage |
| `web/`, top-level arena/daemon modules | Preserved modular baseline and legacy dashboard |
| `connectome_data/`, `outputs/`, `runs/` | Ignored datasets and generated runs; not distributed as source |
| `licenses/`, `NOTICE` | Third-party and data attribution |

## Cleanup on 21 September 2026

- Moved laptop-only research drafts, duplicate learning pages, a DOOM experiment
  directory, temporary scripts, a whole-brain output directory and three old
  `.git.pre-*` backups into a dated sibling archive outside the checkout.
- Preserved the divergent laptop standalone HTML in that archive, verified its
  SHA256, and restored the tracked file to the canonical committed version.
- Removed the misplaced Uroboros plan from active documentation after preserving
  a copy outside each checkout. Git history at `4db4b88` also retains it.
- Moved the separately versioned `neurofly-site` repository out of the Ryzen
  checkout into a sibling directory, retaining its `.git` and commit. Verified
  it was clean and no process had a working directory inside it before moving.
- Preserved the locked worktree and running observatory/dashboard services.
  Old network snapshots and the loose network export were not deleted.

No unique source, saved brain, experiment result or Git history was discarded.
Deployment paths and archive inventories are operational records, not portable
installation instructions. Public users should follow the MVP quick start.

Do not reintroduce local Git-backup directories, other projects, source-data
copies, environment directories or generated videos into source control.
