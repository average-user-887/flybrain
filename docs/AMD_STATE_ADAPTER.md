# Optional AMD v3 state adapter

`brainlab.amd_state_adapter.AmdV3StateAdapter` is an isolated, explicitly
constructed facade. Brain can explicitly select it with `backend='wgpu-amd'` or
`NEUROFLY_BRAIN_BACKEND=wgpu-amd`; `auto` retains its CPU/CUDA policy. There is no
automatic AMD selection or fallback after an explicit AMD failure. Source wiring
now includes daemon/registry admission, cached telemetry and optional packaging.
It supports the fixed baseline v3 constants only; it makes no runtime,
performance, learning or biological qualification claim.

The underlying files are byte-identical to source revision
pre-rewrite revision `1e93391ef79f68729c63ba450d13489c15123783` (not in the published master history; no commit with an identical tree exists there):

| File | SHA256 |
| --- | --- |
| `brainlab/wgpu_v3.py` | `f2c362496034980091a3cb2b1bb14b7b0dca68d1a3fdf3ab588ff6c83bd7649b` |
| `brainlab/wgpu_v3_math.wgsl` | `a4f666380fc4ccb4b2737462a475484b0c37f14e71470bfb6a4d28355f8e494e` |

Construction leaves transient state unloaded. `restore_state(snapshot)` requires
all nine Brain v3 arrays, `cursor`, `total_spikes`, `sim_ms` and explicit
`dynamics='v3'`. Arrays retain exact dtype, shape, bytes and active-set ordering,
including inactive array tails and pending queue storage. There is no legacy,
learned-state or alternate-dynamics conversion. Validation and detachment happen
before device writes. Caller snapshots and exported snapshots do not share model
storage. Calls require exclusive access, like Brain's existing state operations.

`snapshot_state()` reads and detaches complete transients. `step(f32_currents,
duration_ms)` returns detached counts and elapsed seconds and updates engine
clocks. A call is limited to 1–1024 ticks of 0.1ms, with the engine's uint32 cursor
ceiling. `reset_state()` is an explicit destructive baseline transient reset;
it never occurs automatically. It clears every transient and all three clocks.

The immutable capability mapping declares fixed weights and transient
checkpoint support; weight mutation, learning and learned-state restore are
unsupported. Weight assignment, edge updates and learning requests raise before
mutating model state. The weight accessor returns a detached read-only copy.

Transport or device-arithmetic failure poisons the adapter. Advance and export
then refuse without device activity until a complete explicit restore/reset
succeeds. A failed transfer may have changed device buffers; no prior state is
claimed valid or exported. Invalid inputs preserve a previously sound state.

Tests use tiny CSR arrays and fake byte transport with deterministic test ticks.
They exercise the retained engine's real upload/snapshot/advance API, but do not
execute WGSL or establish neural arithmetic conformance. No GPU, real graph,
browser or real daemon verification is included. A clean noneditable wheel check
verifies the packaged frozen shader, optional dependency and lazy imports.

Brain routes complete restore/snapshot/reset through the adapter. Host transients
change after successful restore and readback, and device failures cannot publish
partially restored host state. Snapshots retain exact active ordering rather than
rebuilding it from flags. The AMD weight assignment and edge-update paths refuse
before touching Brain's host weights; CPU/CUDA behavior retains the existing paths.
`Brain.compute_capabilities()` exposes state/weight facts only and does not grant
controller learning support. Nonbaseline v3 sensitivity constants and other LIF
dynamics are refused before adapter construction. Explicit reset uses -52mV only.

`--brain-backend wgpu-amd --backend connectome-fixed --dynamics v3` explicitly
selects the source-wired runtime. The selector is captured once and passed to
the registry and each Brain. The ordinary `auto` policy initially resolves
through the existing CPU/CUDA Brain path; the registry keeps that resolved
engine for its later instances. AMD is never an auto candidate. The optional
`amd` package extra pins `wgpu==0.32.0`; CPU installations need no wgpu import.

AMD admission refuses plastic/readout/modular controllers, learning enable,
toy teaching and graph-I/O state migration before controller/store mutations.
AMD starts with learning disabled and uses graph steps of 1–1024 exact 0.1ms
ticks (20ms by default). Startup reads selected fixed manifests and checkpoint
metadata/learning fields before PID or output writes, without loading
full transient arrays. Both saved manifest and checkpoint must declare the exact
current graph-I/O version and hash; absent, old or contradictory evidence refuses
before PID/output writes. The saved state remains available to its compatible
CPU/CUDA engine; AMD does not offer graph-I/O migration. Unsupported learned
arrays, reset provenance and changed saved dynamics refuse rather than being
discarded or reinitialized. Fixed CPU
v3 checkpoints can restore through AMD with the same instance identity and exact
arrays/clocks. Existing CPU/CUDA learned-state and migration policy is retained.
Runtime target preparation repeats learned-state admission off the runner lock,
before target-store staging; command capability admission itself performs no I/O.
Interrupted graph-I/O migration stores remain untouched on AMD.

`compute_info()` uses cached identity from the actual engine device's
`adapter_info`, including vendor/device IDs, backend and driver description;
missing identity stays unknown. It does not probe CUDA for AMD telemetry.
Capabilities and poisoned state accompany that identity. `neurofly status`
labels a requested AMD engine separately from a running device identity.

These are source and tiny-fixture contracts. A reviewed wheel and successful
installation establish no GPU/runtime/performance or biological acceptance.
Production selection remains gated on separate bounded real-device and browser
verification; none is claimed here.
