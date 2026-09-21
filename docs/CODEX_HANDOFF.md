# Handoff to Codex — 2026-09-21

## Where the work happens

- **Canonical checkout:** AMD Ryzen 3900X host, `~/Documents/ChatGPT/flybrain`, branch `master`.
  Run your session there. The connectome (`connectome_data/malecns_v1/`, ~1.4 GB, gitignored)
  and the Numba LIF kernel only run on that host. There is **no git remote**. Mirrors are copies.
- **hpserver mirror:** `/mnt/hpserver-storage/neurofly/` (mounted on Ryzen; SMB `//192.168.1.23/Storage/neurofly`).
  Each sync writes `snapshots/<shortsha>/` (git bundle `--all`, `git archive` tar.gz, `.git` tar.gz,
  `SHA256SUMS`) and updates `CURRENT_HANDOFF.md`. Older snapshots are kept, never deleted.
- **Laptop mirror:** `C:\Users\Łukasz Wolny\Documents\Sluzbowy_hp\flybrain` (Windows, git is not installed).
  Synced by extracting the snapshot's `.git` tarball; the previous `.git` is renamed, not deleted.
- Read `AGENTS.md` first. It is the owner's standing rules for the project, including the live-UI sign-off procedure.

## State at handoff

- `master` is clean. The worktree `.claude/worktrees/neurofly-openready` (branch
  `worktree-neurofly-openready`, at `a544bfa`, one commit behind `master`) is clean and locked by an old
  Claude session; it can be unlocked and removed once nobody needs it.
- `neurofly-site/` is a **separate git repo** (Next.js download site, branch `main`, clean, no remote).
  It is ignored by the parent repo.
- Last full test run: **506 passed, 4 skipped** at `a544bfa`. Not re-run since; the later commits are
  docs, receipts, and `scripts/test_srv.py` only. Re-run the suite before your first change.
- An Antigravity instance may still be running on Ryzen (`/opt/google-antigravity`). It is not
  authorized to work on this repo any more; the owner is closing it.

## What the product is today (be precise about this)

The live browser dashboard (`flybrain_scientific_instrument.html` + daemon) is driven by the
**hand-built modular controller** (120-KC mushroom body, 16-wedge E-PG compass, Kuramoto-Hopf CPG,
explicit assay reflexes). The identity bar says so. **The 166.7k-neuron MaleCNS connectome is NOT
driving the fly.** `brainlab/` can step the full graph on the Ryzen CPU in isolation
(`brainlab/cosim_server.py`, `ConnectomeServer`; smoke test: `python scripts/test_srv.py`), but it is not
wired to the body. Earlier causal-loop claims were withdrawn (see commits `e882744`, `4552423`: WP5
preregistered verdict was NULL). Do not reintroduce overclaims. Every capability statement needs a
receipt under `docs/receipts/`.

## The goal the owner set

Client-server: the full connectome runs on Ryzen and controls **all joints and biological functions**
of the fly; browsers are thin clients that let people run experiments. The approved (not implemented)
direction is in `docs/FULL_CONNECTOME_COSIM_PLAN.md`:
server-side co-simulation on Ryzen (LIF kernel + 3D body physics, FlyGym/NeuroMechFly v2 in MuJoCo),
descending-neuron readout (DNa02 / DNa01 / DNp09 / MDN / DNp01) driving the body, sensory ingress back
into the graph, streamed to a Three.js 3D viewport over SSE/WebSocket.

That plan was written by another agent from web research and has **not been verified**. Treat its
citations (e.g. projects named `erojasoficial-byte/fly-brain`, `ZeroXClem/closed-loop-fly`) and numbers as
leads to check, not facts. `philshiu/Drosophila_brain_model` (Shiu et al., Nature 2024) and
`NeLy-EPFL/flygym` are real, established references.

## Suggested first steps

1. On Ryzen: `git status`, run the full test suite, run `python scripts/test_srv.py`; record results.
2. Verify the plan's references and pick the smallest end-to-end slice, e.g. one assay (optomotor) where the
   DNa02 left/right rate difference from the real graph steers the body, with the modular controller
   disabled and the identity bar reporting `Controller: connectome`.
3. Keep the modular controller as an explicit, labelled baseline so results can be compared honestly.
4. End each session with a clean commit on `master`, a new hpserver snapshot, updated `CURRENT_HANDOFF.md`,
   and the laptop mirror refreshed.
