# NeuroFly

NeuroFly's current MVP couples a verified, fixed-weight MaleCNS v1.0 spiking
graph to an articulated FlyGym fly. The graph runs in the loop with FlyGym's
MuJoCo physics and locomotion controller. This is an experimental connectome
to body integration, not a complete biological fly or a learning system.

## What runs now

- The source graph retains 166,700 annotated neuronal entries and 25,582,938
  directed edge pairs. Each edge weight is an aggregate contact count; these
  are not 25.6 million individual synapses. The source covers brain and ventral
  nerve cord. The release files are downloaded on demand and are not bundled.
- A fixed-weight LIF model runs with the declared `v3` dynamics and transmitter
  policy. Engineered assistance into named descending neurons is disabled for
  the embodied run.
- An explicit, engineered mapping converts left/right DNa02 spike rates into
  left/right leg-controller drive. A retinal-slip proxy closes a body-motion
  feedback loop. These mappings are experimental controller choices, not
  verified biological pathways.
- `intact` and `output-disconnected` runs support a causal control comparison.
  Output-disconnected removes the connectome-to-body command; it is not a
  neuron lesion.

The physical body uses FlyGym 2.1.0 and MuJoCo 3.9.0. The body extra requires
Python 3.12 through 3.14; the core package remains installable on Python 3.11.
See [the embodied MVP guide](docs/EMBODIED_MVP.md) for setup, data preparation,
commands, outputs and limitations.

## Quick start

From a Python 3.12–3.14 environment, install the body dependencies:

```bash
python -m pip install -e ".[body]"
```

The MaleCNS source tables are about 1.1 GB. From a NeuroFly source checkout,
download and prepare them:

```bash
python -m brainlab.download
python -m brainlab.connectome
python -m brainlab.prepare
```

Then run the closed loop from that checkout (or point an installed package at
the prepared directories):

```bash
python -m neurofly_body run --duration 2 --output runs/embodied-intact \
  --mode intact --graph-dir outputs/brainlab/malecns_v1 \
  --connectome-dir connectome_data/malecns_v1 --seed 1
```

Repeat with `--mode output-disconnected` and the same seed for the control.
Add `--video` to save `body.mp4` inside the output directory. Details
and the out-of-checkout data-path limitation are in the [guide](docs/EMBODIED_MVP.md).

## Scope

This MVP does not demonstrate learning, task performance, or a complete
brain-to-muscle pathway. The connectome has fixed weights; the actuator-side
controller is FlyGym's stock locomotion model with a small engineered
connectome-output interface. There is no validated VNC/muscle mapping, no
calibrated retinal model, and no assay battery in this executable. Learning
and the 14-assay suite are later work.

The repository still contains earlier modular arena, dashboard, daemon and
assay code. Those components are outside this MVP; see
[legacy modular code](docs/LEGACY_MODULAR.md).

## Data and attribution

MaleCNS v1.0 data are available under CC BY 4.0. NeuroFly downloads the source
tables from the public release URLs and verifies their recorded SHA-256 hashes.
FlyGym is Apache-2.0; the optional MuJoCo runtime is also Apache-2.0. Project
code is MIT. See [NOTICE](NOTICE) and `licenses/` for attribution and license
details.
