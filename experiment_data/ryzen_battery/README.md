# Surrogate smoke output, not validation

These reports were written on 2026-09-18 by `experiments/run_paradigm_battery.py`,
which is now retired. Its default `brain_type="connectome"` drove the **surrogate
`ConnectomeBridge`**, not the MaleCNS connectome graph, and no controller or dynamics
identity was recorded. Every trial in a paradigm is identical (3 trials x 200 steps).

Nothing here is behavioural evidence or validation of the connectome model. The files
are kept unchanged as a historical record. See `docs/RETIREMENT_INDEX.md` and
`docs/receipts/README.md`.
