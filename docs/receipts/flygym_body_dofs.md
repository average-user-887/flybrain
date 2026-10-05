# FlyGym body: how many joints the simulated fly has

Measured 5 October 2026 on the project's reference NVIDIA host, FlyGym 2.1.0 and MuJoCo 3.9.0 (the
versions the `body` extra pins), by building the same fly that
`neurofly_body/flygym_body.py` builds (`make_locomotion_fly(..., add_adhesion=True)`
from FlyGym's own `flygym_demo.complex_terrain`).

## Result

| quantity | value | FlyGym call |
|---|---|---|
| leg joint degrees of freedom (FlyGym's `JointDOF` objects) | **66** (11 per leg x 6 legs) | `fly.get_jointdofs_order()` |
| of which position-actuated (driven by the walking controller) | **42** (7 per leg x 6 legs) | `fly.get_actuated_jointdofs_order("position")` |
| MuJoCo joints in the compiled model | 67 (the 66 hinge joints plus one free joint for the body in the world) | `sim.mj_model.njnt` |
| MuJoCo actuators | 48 | `sim.mj_model.nu` |

Per leg, the 11 joint DOFs are thorax-coxa yaw/pitch/roll, coxa-trochanterfemur
pitch/roll, femur-tibia pitch, tibia-tarsus1 pitch, and four tarsal pitch joints
(tarsus1-2 ... tarsus4-5). The 7 actuated ones are the first seven; the four distal
tarsal joints are passive. The 6 MuJoCo actuators beyond the 42 position actuators
are the per-leg tarsal adhesion actuators (`nmf/lf_tarsus5-adhesion` ... one per
leg), added by `add_adhesion=True`.

So the earlier description of an "18-DOF" body was wrong: the body has 66 leg joint
DOFs, 42 of them actuated.

## Reproduce

From a neutral directory, in an environment with the `body` extra installed:

```python
from importlib.metadata import version
import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

print("flygym", version("flygym"))
fly = make_locomotion_fly(name="nmf", add_adhesion=True, colorize=False)
print("jointdofs", len(fly.get_jointdofs_order()))
print("actuated (position)", len(fly.get_actuated_jointdofs_order("position")))
world = FlatGroundWorld()
world.add_fly(fly, spawn_position=np.array([0, 0, 0.5]),
              spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]))
model = Simulation(world, timestep=1e-4).mj_model
print("mujoco njnt", model.njnt, "nu", model.nu)
```

Output on this run:

```
flygym 2.1.0
jointdofs 66
actuated (position) 42
mujoco njnt 67 nu 48
```
