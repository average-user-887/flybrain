# Dashboard 3D viewport: left/right and heading check (2026-10-08)

Status: **repaired on the renderer branch after the orchestrator's GO** (the
proposal below, applied as written). `tests/test_viewport_handedness.js` now checks
the real viewport class: head direction in the Top camera against the 2D arena at
eight headings, and recorded left coxae projected to screen space on the anatomical
left, procedurally and at HQ display scales 1 and 0.8696. The pre-repair mapping
fails 4 of those 5 tests.

## What was checked

`scripts/check_viewport_handedness.js <body.nfbody>` compares two things.

* **Recorded FlyGym frames** (MuJoCo world, z up, millimetres, right-handed). The
  forward direction is the horizontal part of (head − thorax); FlyGym 2 puts both
  eye bodies on the head joint. Anatomical left is up × forward. Each frame is also
  turned rigidly about z to nine headings, which keeps its handedness.
* **The viewport before the repair** (`ArticulatedFly3DViewport.updatePose` at d614118). Arena
  (x, y) is drawn at world (x, ·, −y). The root is rotated about Y by
  −heading + π/2, and the L coxae sit at local x = −1.

## Result

The check covered a 5 s FlyGym 2.1 reference walk (250 frames; the fixture keeps 5
of them) at 9 headings each, 2,250 frames in all.

| Check | Result |
| --- | --- |
| Recorded `lf/lm/lh_coxa` on the anatomical left | 2250 / 2250 |
| Recorded `rf/rm/rh_coxa` on the anatomical right | 2250 / 2250 |
| Recorded `l_wing`, `l_haltere`, `l_pedicel` on the left (and `r_*` on the right) | 2250 / 2250 |
| Recorded `yaw` agrees with thorax→head (cosine > 0.95) | 2250 / 2250 |
| Viewport: drawn head direction · true heading | **cos(2·heading)**: correct at 0° and 180°, reversed at 90° and 270°, sideways at 45° |
| Viewport: L legs relative to the head it draws | **always on the drawn fly's right** |

So the recorded data are consistent, but the viewport draws a **mirror image** of
the fly. Its arena→world map flips one axis, and the root rotation does not undo
that flip. Two things follow:

1. The head points along the heading reflected about the arena x axis. A fly
   walking "north" in the 2D arena faces "south" in 3D, while its path and the
   2D view are correct.
2. Relative to the head it draws, the L legs and every `l_*` part are on the
   right. This confirms the note in `tools/assets/INTERFACE.md` §2. Relative to
   the true heading, the side they land on changes with the heading.

## Repair (proposed, then applied after the GO)

Make the fly transform proper with two lines in `updatePose`:

```js
this.flyGroup.rotation.set(0, this.lastHeading + Math.PI / 2, 0);   // was -heading + pi/2
this.flyGroup.scale.set(-s, s, s);  // setDisplayScale: mirror local X once
```

With these two lines the head points along the heading at every angle, and every
part authored at local −X (L legs, `l_wing`, `l_eye` and so on, procedural and HQ
alike) lands on the anatomical left. No asset or leg table changes are needed;
three.js flips the face winding for a negative-determinant matrix. The
stand/floor scan is unaffected, because it is computed with the root unrotated.
Things to review with the fix:

* The chase camera uses the root's own rotation (fixed on this branch), so it
  follows.
* The coxa yaw sign (`rotation.y = baseAngle + coxa·0.8`) mirrors with the leg, so
  protraction and retraction should be checked against a recorded walk once the
  side is fixed.
* Every consumer of the signed scale uses its magnitude. The contact-shadow size
  reads `Math.abs(scale.y)`, and the stand scan works in y only.
* Contact colours follow the leg index, not the position, so they are unchanged.

## Separate backlog finding (NOT repaired here; telemetry interpretation)

When the FlyGym body supplies `joint_angles_rad` (`neurofly_daemon.py`,
`c_bridge.last_body_obs`), the packet carries the first 18 of FlyGym's 66 joint
DOFs in `get_jointdofs_order()` order: all 11 DOFs of `lf`, then 7 of `lm`. The
viewport reads them as six legs × (coxa, femur, tibia). On that path, legs L2 to
R3 are therefore drawn from the wrong joints. The modular daemon's synthetic
18-value stream does match the viewport's layout. A fix would need the packet to
say its joint order, which is a telemetry change outside this presentation card.
Truthful limitation until then: on the FlyGym-body path, the dashboard leg angles
for every leg except L1's coxa yaw are not those legs' joints. Body position,
heading and the embodied replay page (which draws recorded body positions) are not
affected.
