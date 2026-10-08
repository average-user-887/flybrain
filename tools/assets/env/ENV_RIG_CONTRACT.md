# Environment animal rigs: joint, frame and unit contract (ENV-ANIM-01, W3)

Presentation only. The jumping spider and the mantis nymph are ILLUSTRATIVE: no predator stimulus or
behaviour exists, and nothing here may change looming or retinal input, physics, the connectome or any
recorded pose. Clips built on these rigs are illustrative animation with source-informed timing; they are
not motion capture.

## Frame, units, chain convention

* Frame: three.js, +Y up, +Z forward (head), 1 unit = 1 viewport-mm (the fly display scale, about 3.3x
  life). The animal stands on y = 0; the lowest bind-pose tarsus vertex defines its foot plane.
* Chains follow the W2 fly rig: `<joint>_root` (translation to the pivot) -> one node per rotational DOF
  -> `<joint>_offset` (translation back) -> meshes. Mesh vertices stay in the root frame, so the bind pose
  is every DOF = 0 and equals the static prop exactly (checked: identical unique vertex positions and
  triangle counts against `env_jumping_spider.glb` / `env_mantis_nymph.glb`).
* Animate a DOF node only by a rotation of angle theta (radians) about its listed axis (node frame = parent
  frame at rest). Never translate a DOF, `_root` or `_offset` node.
* `<p>_motion` (child of the GLB root) is the only locomotion carrier: translation in the floor plane
  (x, z) and yaw about +Y. In-place clips keep it at identity; root-motion clips key it. The GLB root is
  never animated.
* `<p>_body_shift` (child of `<p>_motion`) is the one translating DOF node: body translation relative to
  the planted feet (x lateral, for the mantis peering sway; y bob), limits x [-3, 3], y [-0.6, 0.6]. The
  legs re-solve so the feet stay planted.
* Leg side letters follow the existing viewport convention (L at -X, as the W2 fly). `turn_left` rotates
  the heading toward +X, which is the anatomical left in a right-handed +Y-up, +Z-forward frame. The
  handedness question is open on the separate renderer card; the spider gait is left/right symmetric.
* Legs: hip has two DOFs (`yaw`, + = protraction on both sides; `lift`, + = levation), then `knee_flex`
  and `ankle_flex` (+ = distal part rotates up in the leg plane). The leg plane is the vertical plane through
  the hip along the bind-pose leg direction; `plane_normal` per leg is in the joints JSON.
* Limits are DISPLAY limits (artistic estimates chosen to avoid body/ground intersection), not measured
  joint ranges. Every exported clip is checked against them over its whole duration.

## zebra jumping spider (Salticidae; Salticus scenicus style)

Rig id `neurofly-env-spider-rig-v1`, 6012 triangles. eight walking legs L1..L4 / R1..R4 (spider legs I-IV); pedipalps are not legs. Scale: fly display scale (~3.3x life); ~6 mm spider -> ~20 viewport-mm body length.

| node | parent | axis / translation | display range | meaning |
| --- | --- | --- | --- | --- |
| `sp_motion` | `root` | locomotion | x, z + yaw | root-motion carrier: translation in the floor plane (x, z) and yaw about +Y; identity in in-place clips |
| `sp_body_shift` | `sp_motion` | translation | {'x': [-3.0, 3.0], 'y': [-0.6, 0.6]} | body translation relative to the planted feet (lateral peering sway x, bob y); the only translating DOF besides <p>_motion; legs re-solve to keep feet planted |
| `sp_body_root` | `sp_body_shift` | translation (0.000, 3.000, 0.000) | - | pivot |
| `sp_body_pitch` | `sp_body_root` | (1.0000, 0.0000, 0.0000) | [-0.15, 0.15] rad | + = nose down |
| `sp_body_roll` | `sp_body_pitch` | (0.0000, 0.0000, 1.0000) | [-0.12, 0.12] rad | + = left side up |
| `sp_body_yaw` | `sp_body_roll` | (0.0000, 1.0000, 0.0000) | [-0.2, 0.2] rad | + = turn left (toward +X) |
| `sp_palp_L_root` | `sp_body_offset` | translation (-1.200, 2.200, 6.600) | - | pivot |
| `sp_palp_L_flick` | `sp_palp_L_root` | (1.0000, 0.0000, 0.0000) | [-0.35, 0.25] rad | + = palp tip down/back (palp signalling, idle only) |
| `sp_palp_R_root` | `sp_body_offset` | translation (1.200, 2.200, 6.600) | - | pivot |
| `sp_palp_R_flick` | `sp_palp_R_root` | (1.0000, 0.0000, 0.0000) | [-0.35, 0.25] rad | + = palp tip down/back (palp signalling, idle only) |
| `sp_L1_hip_root` | `sp_body_offset` | translation (-2.100, 2.500, 5.000) | - | pivot |
| `sp_L1_hip_yaw` | `sp_L1_hip_root` | (0.0000, 1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_L1_hip_lift` | `sp_L1_hip_yaw` | (-0.8660, 0.0000, -0.5000) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_L1_knee_root` | `sp_L1_hip_offset` | translation (-3.504, 5.500, 7.432) | - | pivot |
| `sp_L1_knee_flex` | `sp_L1_knee_root` | (-0.8660, 0.0000, -0.5000) | [-1.0, 1.0] rad | + = distal part up |
| `sp_L1_ankle_root` | `sp_L1_knee_offset` | translation (-5.298, 1.200, 10.539) | - | pivot |
| `sp_L1_ankle_flex` | `sp_L1_ankle_root` | (-0.8660, 0.0000, -0.5000) | [-1.2, 1.2] rad | + = distal part up |
| `sp_L2_hip_root` | `sp_body_offset` | translation (-2.100, 2.500, 3.800) | - | pivot |
| `sp_L2_hip_yaw` | `sp_L2_hip_root` | (0.0000, 1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_L2_hip_lift` | `sp_L2_hip_yaw` | (-0.3090, 0.0000, -0.9511) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_L2_knee_root` | `sp_L2_hip_offset` | translation (-4.360, 5.500, 4.534) | - | pivot |
| `sp_L2_knee_flex` | `sp_L2_knee_root` | (-0.3090, 0.0000, -0.9511) | [-1.0, 1.0] rad | + = distal part up |
| `sp_L2_ankle_root` | `sp_L2_knee_offset` | translation (-7.247, 1.200, 5.472) | - | pivot |
| `sp_L2_ankle_flex` | `sp_L2_ankle_root` | (-0.3090, 0.0000, -0.9511) | [-1.2, 1.2] rad | + = distal part up |
| `sp_L3_hip_root` | `sp_body_offset` | translation (-2.100, 2.500, 2.500) | - | pivot |
| `sp_L3_hip_yaw` | `sp_L3_hip_root` | (0.0000, 1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_L3_hip_lift` | `sp_L3_hip_yaw` | (0.3746, 0.0000, -0.9272) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_L3_knee_root` | `sp_L3_hip_offset` | translation (-4.270, 5.500, 1.623) | - | pivot |
| `sp_L3_knee_flex` | `sp_L3_knee_root` | (0.3746, 0.0000, -0.9272) | [-1.0, 1.0] rad | + = distal part up |
| `sp_L3_ankle_root` | `sp_L3_knee_offset` | translation (-7.042, 1.200, 0.503) | - | pivot |
| `sp_L3_ankle_flex` | `sp_L3_ankle_root` | (0.3746, 0.0000, -0.9272) | [-1.2, 1.2] rad | + = distal part up |
| `sp_L4_hip_root` | `sp_body_offset` | translation (-2.100, 2.500, 1.100) | - | pivot |
| `sp_L4_hip_yaw` | `sp_L4_hip_root` | (0.0000, 1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_L4_hip_lift` | `sp_L4_hip_yaw` | (0.8660, 0.0000, -0.5000) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_L4_knee_root` | `sp_L4_hip_offset` | translation (-3.540, 5.500, -1.394) | - | pivot |
| `sp_L4_knee_flex` | `sp_L4_knee_root` | (0.8660, 0.0000, -0.5000) | [-1.0, 1.0] rad | + = distal part up |
| `sp_L4_ankle_root` | `sp_L4_knee_offset` | translation (-5.380, 1.200, -4.581) | - | pivot |
| `sp_L4_ankle_flex` | `sp_L4_ankle_root` | (0.8660, 0.0000, -0.5000) | [-1.2, 1.2] rad | + = distal part up |
| `sp_R1_hip_root` | `sp_body_offset` | translation (2.100, 2.500, 5.000) | - | pivot |
| `sp_R1_hip_yaw` | `sp_R1_hip_root` | (0.0000, -1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_R1_hip_lift` | `sp_R1_hip_yaw` | (-0.8660, 0.0000, 0.5000) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_R1_knee_root` | `sp_R1_hip_offset` | translation (3.504, 5.500, 7.432) | - | pivot |
| `sp_R1_knee_flex` | `sp_R1_knee_root` | (-0.8660, 0.0000, 0.5000) | [-1.0, 1.0] rad | + = distal part up |
| `sp_R1_ankle_root` | `sp_R1_knee_offset` | translation (5.298, 1.200, 10.539) | - | pivot |
| `sp_R1_ankle_flex` | `sp_R1_ankle_root` | (-0.8660, 0.0000, 0.5000) | [-1.2, 1.2] rad | + = distal part up |
| `sp_R2_hip_root` | `sp_body_offset` | translation (2.100, 2.500, 3.800) | - | pivot |
| `sp_R2_hip_yaw` | `sp_R2_hip_root` | (0.0000, -1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_R2_hip_lift` | `sp_R2_hip_yaw` | (-0.3090, 0.0000, 0.9511) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_R2_knee_root` | `sp_R2_hip_offset` | translation (4.360, 5.500, 4.534) | - | pivot |
| `sp_R2_knee_flex` | `sp_R2_knee_root` | (-0.3090, 0.0000, 0.9511) | [-1.0, 1.0] rad | + = distal part up |
| `sp_R2_ankle_root` | `sp_R2_knee_offset` | translation (7.247, 1.200, 5.472) | - | pivot |
| `sp_R2_ankle_flex` | `sp_R2_ankle_root` | (-0.3090, 0.0000, 0.9511) | [-1.2, 1.2] rad | + = distal part up |
| `sp_R3_hip_root` | `sp_body_offset` | translation (2.100, 2.500, 2.500) | - | pivot |
| `sp_R3_hip_yaw` | `sp_R3_hip_root` | (0.0000, -1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_R3_hip_lift` | `sp_R3_hip_yaw` | (0.3746, -0.0000, 0.9272) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_R3_knee_root` | `sp_R3_hip_offset` | translation (4.270, 5.500, 1.623) | - | pivot |
| `sp_R3_knee_flex` | `sp_R3_knee_root` | (0.3746, -0.0000, 0.9272) | [-1.0, 1.0] rad | + = distal part up |
| `sp_R3_ankle_root` | `sp_R3_knee_offset` | translation (7.042, 1.200, 0.503) | - | pivot |
| `sp_R3_ankle_flex` | `sp_R3_ankle_root` | (0.3746, -0.0000, 0.9272) | [-1.2, 1.2] rad | + = distal part up |
| `sp_R4_hip_root` | `sp_body_offset` | translation (2.100, 2.500, 1.100) | - | pivot |
| `sp_R4_hip_yaw` | `sp_R4_hip_root` | (0.0000, -1.0000, 0.0000) | [-0.9, 0.9] rad | + = protraction (tip forward) |
| `sp_R4_hip_lift` | `sp_R4_hip_yaw` | (0.8660, -0.0000, 0.5000) | [-0.6, 0.9] rad | + = levation (leg raised) |
| `sp_R4_knee_root` | `sp_R4_hip_offset` | translation (3.540, 5.500, -1.394) | - | pivot |
| `sp_R4_knee_flex` | `sp_R4_knee_root` | (0.8660, -0.0000, 0.5000) | [-1.0, 1.0] rad | + = distal part up |
| `sp_R4_ankle_root` | `sp_R4_knee_offset` | translation (5.380, 1.200, -4.581) | - | pivot |
| `sp_R4_ankle_flex` | `sp_R4_ankle_root` | (0.8660, -0.0000, 0.5000) | [-1.2, 1.2] rad | + = distal part up |

## praying mantis nymph (Mantidae; Tenodera sinensis style)

Rig id `neurofly-env-mantis-rig-v1`, 3436 triangles. walking legs L2, L3, R2, R3; raptorial forelegs L1/R1 are held (not walking legs). Scale: fly display scale (~3.3x life); ~11 mm early nymph -> ~37 viewport-mm.

| node | parent | axis / translation | display range | meaning |
| --- | --- | --- | --- | --- |
| `mn_motion` | `root` | locomotion | x, z + yaw | root-motion carrier: translation in the floor plane (x, z) and yaw about +Y; identity in in-place clips |
| `mn_body_shift` | `mn_motion` | translation | {'x': [-3.0, 3.0], 'y': [-0.6, 0.6]} | body translation relative to the planted feet (lateral peering sway x, bob y); the only translating DOF besides <p>_motion; legs re-solve to keep feet planted |
| `mn_body_root` | `mn_body_shift` | translation (0.000, 4.600, 0.000) | - | pivot |
| `mn_body_pitch` | `mn_body_root` | (1.0000, 0.0000, 0.0000) | [-0.12, 0.12] rad | + = front down |
| `mn_body_roll` | `mn_body_pitch` | (0.0000, 0.0000, 1.0000) | [-0.15, 0.15] rad | + = left side up |
| `mn_body_yaw` | `mn_body_roll` | (0.0000, 1.0000, 0.0000) | [-0.2, 0.2] rad | + = turn left (toward +X) |
| `mn_neck_root` | `mn_body_offset` | translation (0.000, 10.500, 10.800) | - | pivot |
| `mn_neck_yaw` | `mn_neck_root` | (0.0000, 1.0000, 0.0000) | [-0.6, 0.6] rad | + = head turns left (toward +X) |
| `mn_neck_pitch` | `mn_neck_yaw` | (1.0000, 0.0000, 0.0000) | [-0.3, 0.3] rad | + = head down |
| `mn_L1_coxa_root` | `mn_body_offset` | translation (-0.650, 9.200, 9.700) | - | pivot |
| `mn_L1_coxa_swing` | `mn_L1_coxa_root` | (-0.9910, 0.0393, -0.1279) | [-0.4, 0.4] rad | + = coxa swings (raptorial leg raised/lowered as a unit) |
| `mn_L1_femur_root` | `mn_L1_coxa_offset` | translation (-1.050, 6.600, 12.000) | - | pivot |
| `mn_L1_femur_flex` | `mn_L1_femur_root` | (-0.9910, 0.0393, -0.1279) | [-0.3, 0.3] rad | coxa-femur; held in clips |
| `mn_L1_tibia_root` | `mn_L1_femur_offset` | translation (-1.200, 10.300, 14.300) | - | pivot |
| `mn_L1_tibia_flex` | `mn_L1_tibia_root` | (-0.9910, 0.0393, -0.1279) | [-0.3, 0.3] rad | femur-tibia; held folded in clips (no strike) |
| `mn_R1_coxa_root` | `mn_body_offset` | translation (0.650, 9.200, 9.700) | - | pivot |
| `mn_R1_coxa_swing` | `mn_R1_coxa_root` | (0.9910, 0.0393, -0.1279) | [-0.4, 0.4] rad | + = coxa swings (raptorial leg raised/lowered as a unit) |
| `mn_R1_femur_root` | `mn_R1_coxa_offset` | translation (1.050, 6.600, 12.000) | - | pivot |
| `mn_R1_femur_flex` | `mn_R1_femur_root` | (0.9910, 0.0393, -0.1279) | [-0.3, 0.3] rad | coxa-femur; held in clips |
| `mn_R1_tibia_root` | `mn_R1_femur_offset` | translation (1.200, 10.300, 14.300) | - | pivot |
| `mn_R1_tibia_flex` | `mn_R1_tibia_root` | (0.9910, 0.0393, -0.1279) | [-0.3, 0.3] rad | femur-tibia; held folded in clips (no strike) |
| `mn_L2_hip_root` | `mn_body_offset` | translation (-0.800, 4.400, 1.600) | - | pivot |
| `mn_L2_hip_yaw` | `mn_L2_hip_root` | (0.0000, 1.0000, 0.0000) | [-0.7, 0.7] rad | + = protraction (tip forward) |
| `mn_L2_hip_lift` | `mn_L2_hip_yaw` | (-0.4961, 0.0000, -0.8682) | [-0.5, 0.7] rad | + = levation (leg raised) |
| `mn_L2_knee_root` | `mn_L2_hip_offset` | translation (-6.200, 6.800, 4.400) | - | pivot |
| `mn_L2_knee_flex` | `mn_L2_knee_root` | (-0.4961, 0.0000, -0.8682) | [-0.8, 0.8] rad | + = distal part up |
| `mn_L2_ankle_root` | `mn_L2_knee_offset` | translation (-9.200, 0.150, 6.400) | - | pivot |
| `mn_L2_ankle_flex` | `mn_L2_ankle_root` | (-0.4961, 0.0000, -0.8682) | [-1.0, 1.0] rad | + = distal part up |
| `mn_L3_hip_root` | `mn_body_offset` | translation (-0.800, 4.300, -0.900) | - | pivot |
| `mn_L3_hip_yaw` | `mn_L3_hip_root` | (0.0000, 1.0000, 0.0000) | [-0.7, 0.7] rad | + = protraction (tip forward) |
| `mn_L3_hip_lift` | `mn_L3_hip_yaw` | (0.6910, 0.0000, -0.7228) | [-0.5, 0.7] rad | + = levation (leg raised) |
| `mn_L3_knee_root` | `mn_L3_hip_offset` | translation (-6.800, 7.100, -4.400) | - | pivot |
| `mn_L3_knee_flex` | `mn_L3_knee_root` | (0.6910, 0.0000, -0.7228) | [-0.8, 0.8] rad | + = distal part up |
| `mn_L3_ankle_root` | `mn_L3_knee_offset` | translation (-9.900, 0.150, -9.600) | - | pivot |
| `mn_L3_ankle_flex` | `mn_L3_ankle_root` | (0.6910, 0.0000, -0.7228) | [-1.0, 1.0] rad | + = distal part up |
| `mn_R2_hip_root` | `mn_body_offset` | translation (0.800, 4.400, 1.600) | - | pivot |
| `mn_R2_hip_yaw` | `mn_R2_hip_root` | (0.0000, -1.0000, 0.0000) | [-0.7, 0.7] rad | + = protraction (tip forward) |
| `mn_R2_hip_lift` | `mn_R2_hip_yaw` | (-0.4961, 0.0000, 0.8682) | [-0.5, 0.7] rad | + = levation (leg raised) |
| `mn_R2_knee_root` | `mn_R2_hip_offset` | translation (6.200, 6.800, 4.400) | - | pivot |
| `mn_R2_knee_flex` | `mn_R2_knee_root` | (-0.4961, 0.0000, 0.8682) | [-0.8, 0.8] rad | + = distal part up |
| `mn_R2_ankle_root` | `mn_R2_knee_offset` | translation (9.200, 0.150, 6.400) | - | pivot |
| `mn_R2_ankle_flex` | `mn_R2_ankle_root` | (-0.4961, 0.0000, 0.8682) | [-1.0, 1.0] rad | + = distal part up |
| `mn_R3_hip_root` | `mn_body_offset` | translation (0.800, 4.300, -0.900) | - | pivot |
| `mn_R3_hip_yaw` | `mn_R3_hip_root` | (0.0000, -1.0000, 0.0000) | [-0.7, 0.7] rad | + = protraction (tip forward) |
| `mn_R3_hip_lift` | `mn_R3_hip_yaw` | (0.6910, -0.0000, 0.7228) | [-0.5, 0.7] rad | + = levation (leg raised) |
| `mn_R3_knee_root` | `mn_R3_hip_offset` | translation (6.800, 7.100, -4.400) | - | pivot |
| `mn_R3_knee_flex` | `mn_R3_knee_root` | (0.6910, -0.0000, 0.7228) | [-0.8, 0.8] rad | + = distal part up |
| `mn_R3_ankle_root` | `mn_R3_knee_offset` | translation (9.900, 0.150, -9.600) | - | pivot |
| `mn_R3_ankle_flex` | `mn_R3_ankle_root` | (0.6910, -0.0000, 0.7228) | [-1.0, 1.0] rad | + = distal part up |

## Files

`tools/assets/build_env_rigs.py --animal spider|mantis` writes `env_<animal>_rig.glb`, `.blend` and
`env_<animal>_rig_joints.json` (this table, machine-readable, with per-leg hip/knee/ankle/tip points) into
a new versioned output directory. The static props and their outputs are only read.
