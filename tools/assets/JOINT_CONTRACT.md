# NeuroFly fly rig v2: joint contract (W2, 2026-10-08)

This is a presentation rig. Clips built on it are illustrative animation, not
simulated behaviour. A clip must never be overlaid on recorded or live body or
neural telemetry, and recorded poses always win. The recorded body and leg frames
are unchanged: thorax, head, abdomen and the six legs use exactly the v1 frames
in `tools/assets/INTERFACE.md`.

## Files

Build: `build_fly.py --rig v2 --sex female|male [--lod 0|1]` on branch
`claude/ui-assets-20261008`. Outputs go to the versioned directory
`output/w2-rig-v2-20261008/`, never to the old canonical outputs:

* `fly_<sex>_v2_lod<N>.glb` and `.blend`;
* `fly_<sex>_v2_lod<N>_joints.json`, the machine-readable version of the table
  below, written by the build.

The joint table is identical for female and male and for both LODs. The root
extras say `nf_rig = "neurofly-viewport-fly-v2"`.

## Conventions

* Frame: the v1 thorax frame (`thorax_frame`), in viewport-mm, with +Y up, +Z
  forward (head) and the viewport's side convention (`l_*` parts at −X;
  unchanged, and the left/right question stays separately scoped).
* Each articulated part is a node chain: `<part>_root` (a translation to the
  pivot) → one node per rotational degree of freedom → `<part>_offset` (a
  translation of −pivot) → meshes. The mesh vertices stay in the thorax frame, so
  at rest every chain is the identity: **bind pose = rest = all angles 0**.
* To animate a DOF node, set its quaternion to an axis-angle rotation by angle θ
  (radians) about the listed `axis`. The axis is expressed in the node's own frame,
  which equals its parent frame at rest; the order is intrinsic (parent DOF
  first). Never translate a DOF node, the `_root` or the `_offset`, and never
  change the GLB root or `thorax_frame` to hide clipping.
* Sign conventions are chosen to be side-symmetric. Positive values mean the same
  anatomical motion on the left and the right: sweep + = wing opens away from the
  midline; elevate + = tip up; antenna abduct + = tip away from the midline;
  extend + = tip down; haltere beat + = knob up.
* Mesh names are unchanged (`l_wing`, `l_wing_veins`, `l_pedicel`,
  `l_funiculus`, `l_arista`, `l_haltere`, and the `r_*` equivalents), so the v1
  loader, which lifts meshes by name, still shows a static rest pose. To animate,
  a loader must keep the `<part>_root` subtree intact.

## Joint table (radians; rest = 0 for every DOF)

| node | parent | axis (node frame) / translation | source range | rest | note |
| --- | --- | --- | --- | --- | --- |
| `l_antenna_root` | `thorax_frame` | translation (-0.1700, 0.6000, 2.2300) | — | — | pivot |
| `l_antenna_abduct` | `l_antenna_root` | (0.0000, -1.0000, 0.0000) | [-0.4, 0.8] | 0 | flybody antenna_abduct range; + = tip away from midline |
| `l_antenna_extend` | `l_antenna_abduct` | (1.0000, 0.0000, 0.0000) | [-0.2, 0.5] | 0 | flybody antenna_extend range; + = tip down/forward |
| `l_antenna_twist` | `l_antenna_extend` | (-0.2860, 0.0954, 0.9535) | [-0.1, 0.09] | 0 | flybody antenna_twist range; about the pedicel axis |
| `l_antenna_offset` | `l_antenna_twist` | translation (0.1700, -0.6000, -2.2300) | — | — | offset |
| `l_funiculus_root` | `l_antenna_twist` | translation (-0.0600, 0.0200, 0.2000) | — | — | pivot |
| `l_funiculus_rotate` | `l_funiculus_root` | (0.0000, -0.9889, 0.1483) | [-0.1, 0.1] | 0 | passive rotation of funiculus+arista about the funiculus long axis (Gopfert & Robert 2002); amplitude stylised |
| `l_funiculus_offset` | `l_funiculus_rotate` | translation (0.2300, -0.6200, -2.4300) | — | — | offset |
| `r_antenna_root` | `thorax_frame` | translation (0.1700, 0.6000, 2.2300) | — | — | pivot |
| `r_antenna_abduct` | `r_antenna_root` | (0.0000, 1.0000, 0.0000) | [-0.4, 0.8] | 0 | flybody antenna_abduct range; + = tip away from midline |
| `r_antenna_extend` | `r_antenna_abduct` | (1.0000, 0.0000, 0.0000) | [-0.2, 0.5] | 0 | flybody antenna_extend range; + = tip down/forward |
| `r_antenna_twist` | `r_antenna_extend` | (0.2860, 0.0954, 0.9535) | [-0.1, 0.09] | 0 | flybody antenna_twist range; about the pedicel axis |
| `r_antenna_offset` | `r_antenna_twist` | translation (-0.1700, -0.6000, -2.2300) | — | — | offset |
| `r_funiculus_root` | `r_antenna_twist` | translation (0.0600, 0.0200, 0.2000) | — | — | pivot |
| `r_funiculus_rotate` | `r_funiculus_root` | (0.0000, -0.9889, 0.1483) | [-0.1, 0.1] | 0 | passive rotation of funiculus+arista about the funiculus long axis (Gopfert & Robert 2002); amplitude stylised |
| `r_funiculus_offset` | `r_funiculus_rotate` | translation (-0.2300, -0.6200, -2.4300) | — | — | offset |
| `l_wing_root` | `thorax_frame` | translation (-0.4500, 1.0600, -0.4500) | — | — | pivot |
| `l_wing_sweep` | `l_wing_root` | (0.0000, 1.0000, 0.0000) | [0.0, 3.0] | 0 | about the body vertical; 0 = folded rest; 3.0 = flybody wing_yaw span (-1.5..1.5) from its folded springref |
| `l_wing_elevate` | `l_wing_sweep` | (0.9689, 0.0000, -0.2474) | [-1.7, 0.8] | 0 | about the horizontal axis normal to the span (rotates with the sweep); + = tip up; flybody wing_roll range (-1..1.5) relative to its rest 0.7 |
| `l_wing_pitch` | `l_wing_elevate` | (-0.2447, 0.1483, -0.9582) | [-0.27, 3.92] | 0 | about the wing span axis (leading edge down = +); flybody wing_pitch range (-1.27..2.92) relative to its rest -1.0 |
| `l_wing_offset` | `l_wing_pitch` | translation (0.4500, -1.0600, 0.4500) | — | — | offset |
| `r_wing_root` | `thorax_frame` | translation (0.4500, 1.0600, -0.4500) | — | — | pivot |
| `r_wing_sweep` | `r_wing_root` | (0.0000, -1.0000, 0.0000) | [0.0, 3.0] | 0 | about the body vertical; 0 = folded rest; 3.0 = flybody wing_yaw span (-1.5..1.5) from its folded springref |
| `r_wing_elevate` | `r_wing_sweep` | (0.9689, -0.0000, 0.2474) | [-1.7, 0.8] | 0 | about the horizontal axis normal to the span (rotates with the sweep); + = tip up; flybody wing_roll range (-1..1.5) relative to its rest 0.7 |
| `r_wing_pitch` | `r_wing_elevate` | (0.2447, 0.1483, -0.9582) | [-0.27, 3.92] | 0 | about the wing span axis (leading edge down = +); flybody wing_pitch range (-1.27..2.92) relative to its rest -1.0 |
| `r_wing_offset` | `r_wing_pitch` | translation (-0.4500, -1.0600, 0.4500) | — | — | offset |
| `l_haltere_root` | `thorax_frame` | translation (-0.7807, 0.3356, -0.9497) | — | — | pivot |
| `l_haltere_beat` | `l_haltere_root` | (0.8240, 0.0000, -0.5665) | [-0.2, 0.2] | 0 | flybody haltere joint range; + = knob up |
| `l_haltere_offset` | `l_haltere_beat` | translation (0.7807, -0.3356, 0.9497) | — | — | offset |
| `r_haltere_root` | `thorax_frame` | translation (0.7807, 0.3356, -0.9497) | — | — | pivot |
| `r_haltere_beat` | `r_haltere_root` | (0.8240, -0.0000, 0.5665) | [-0.2, 0.2] | 0 | flybody haltere joint range; + = knob up |
| `r_haltere_offset` | `r_haltere_beat` | translation (-0.7807, -0.3356, 0.9497) | — | — | offset |

## Display-safe envelope (measured on the exported meshes)

The source ranges come from the flybody model. They are re-expressed about this
rig's simpler pivots, so the stylised body intersects some combinations. The
following were measured by an exact vertex test of the exported GLB against the
thorax, scutum, scutellum, sexed abdomen, head and eye shells. The first 0.40
viewport-mm of the wing, where it inserts under the notum, is excluded. Values
are the same for both sexes.

* **Wing** (per side): sweep may use 0 to 3.0. Elevate must be at least the value
  for that sweep (linear interpolation is fine), and at most 0.8:

  | sweep | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.25 to 3.0 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | minimum elevate | 0 | 0 | −0.1 | −0.9 | −0.8 | −0.6 | −0.3 | 0 | 0.2 | 0.4 |

  Pitch (at pitch 0 the wing is flat, as at rest) is safe within these intervals
  for each (sweep, elevate): (0.5, 0) [−0.27, 0.13]; (0.5, 0.3) [−0.27, 0.63];
  (1.0, 0) [−0.27, 0.63]; (1.0, 0.3) [−0.27, 1.43]; (1.5, 0) [−0.27, 0.83];
  (1.5, 0.3) [−0.27, 1.33]; (1.5, 0.6) [−0.27, 2.23]; (2.0, 0.3) [−0.07, 1.13];
  (2.0, 0.6) [−0.27, 1.53]. Keep pitch at 0 when the wing is folded (sweep < 0.5).
* **Antenna**: the simple envelope is abduct in [−0.4, 0.6] and extend in
  [−0.2, 0.2]. Measured maximum extend for a given abduct: −0.4 to 0.0 → 0.40;
  0.1 to 0.3 → 0.35; 0.4 → 0.30; 0.5 → 0.25; 0.6 → 0.20; 0.7 → 0.15; 0.8 → −0.05.
  The full twist range (−0.1 to 0.09) and funiculus rotate (±0.1) are clear.
* **Haltere**: beat ±0.2 is clear.

## Sources

* Joint axes and ranges: the flybody fruit-fly model, Vaxenburg R, Siwanowicz I,
  Merel J, et al. (2025) *Whole-body physics simulation of fruit fly locomotion*,
  *Nature* 643:1312–1320, doi:10.1038/s41586-025-09029-4, as distributed in FlyGym
  2.1 (`flybody/fruitfly.xml`). It gives wing yaw [−1.5, 1.5] with folded rest 1.5,
  wing roll [−1, 1.5] with rest 0.7, wing pitch [−1.27, 2.92] with rest −1.0,
  antenna abduct [−0.4, 0.8], twist [−0.1, 0.09], extend [−0.2, 0.5] and haltere
  [−0.2, 0.2]. Here they are mapped onto sweep, elevate and pitch relative to its
  folded rest. **The mapping is a presentation approximation, not a calibrated
  transform** between the two models' frames.
* Funiculus rotation: Göpfert MC, Robert D (2002) *J Exp Biol* 205:1199–1208,
  doi:10.1242/jeb.205.9.1199. The funiculus and arista form one receiver that
  rotates about the funiculus's longitudinal axis. The ±0.1 rad display amplitude
  is stylised; real sound-driven amplitudes are far smaller.
* Antennal movement in flight: Mamiya A, Straw AD, Tómasson E, Dickinson MH (2011)
  *J Neurosci* 31:6900–6914, doi:10.1523/JNEUROSCI.0498-11.2011. They report
  active antennal movements, small tonic passive deflections and passive
  oscillation at wingbeat frequency.
* Grooming order (for W3's clip): Seeds AM et al. (2014) *eLife* 3:e02951,
  doi:10.7554/eLife.02951. They report a suppression hierarchy that produces a
  sequential, head-first grooming order. I verified only the abstract; W3 should
  take the detailed order from the paper's figures.
* Wing kinematics are not modelled. Any wingbeat clip is illustrative and slowed
  down. It implies no aerodynamics, mechanosensation or neural command.

## Changes from v1

These are bind-pose geometry fixes, not new biology:

1. The folded wing hinge moves from (±0.62, 0.95, −0.15) to (±0.45, 1.06, −0.45)
   and rises about 8.5° instead of drooping. The v1 folded wing passed through
   the scutum and abdomen; the v2 rest pose is clear and does not cross the
   midline.
2. Floating bristles are fixed. The v1 thorax and head macrochaetae floated up to
   about 0.37 viewport-mm above the cuticle; v2 seats each base 0.03 below the
   surface of the thorax, scutum, scutellum or head shell.
3. The haltere base, which sat about 0.03 outside the thorax, is seated on the
   surface.

Legs, body segments, feet and floor contact are unchanged. The floor bounds stay
those of `SEX_VARIANTS.md`, and the v2 measurements are in the build provenance.
