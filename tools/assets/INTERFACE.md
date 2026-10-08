# Presentation assets: interface contract

Presentation only. These assets change how the fly and the arena *look* in the
browser, never what is simulated, measured, or claimed. They are opt-in
(`?assets=hq`), off by default, and not yet adopted: adoption needs a visible
browser review by the orchestrator.

## 1. Hook points

| Where | Hook | What it does with `?assets=hq` | Without the flag |
| --- | --- | --- | --- |
| `web/app.js`, `ArticulatedFly3DViewport.init()` right after `this.buildFlyMesh()` | `if (window.NeuroflyHQAssets) window.NeuroflyHQAssets.attachViewport(this);` | Loads `assets/hq/fly_hq_lod0.glb`, hides the procedural fly's materials (contact spheres excepted) and parents the GLB meshes to the existing groups. Loads `assets/hq/arena_shell.glb` and wraps `updateAssayGeometry` to keep a plinth under the arena. | Returns `null` at once. Nothing is fetched, injected or changed. |
| `web/embodied_replay.js`, end of the IIFE, before `resize()` | `if (window.NeuroflyHQAssets) window.NeuroflyHQAssets.attachReplay({ THREE, scene, state });` | Adds an `hq-replay-rig` group with its own lights and stretches leg meshes between the recorded joint positions. The bone lines, joint points and trail stay drawn. | Returns `null` at once. |
| `web/index.html`, `web/embodied_replay.html` | `<script src="hq_assets.js"></script>` | — | The module only defines `window.NeuroflyHQAssets` and `window.NeuroflyHQAssetsLib`. |
| `web/vendor/GLTFLoader.js` | Injected by `hq_assets.js` only when the flag is set. | — | Never requested. |
| `neurofly_daemon.DASHBOARD_WEB_ASSETS` | Lists `web/hq_assets.js`. | — | Changes only the web-build identity hash. |

`web/asset_preview.html` is a standalone page. It loads the GLBs and PNGs from
`web/assets/hq/` (git-ignored, filled by `tools/assets/build_all.sh --stage`).

Every failure falls back to the procedural fly and says so in an amber
`hq-assets-label`: no loader, an HTTP error, a non-glTF file, a missing node
(see §3), or a thorax outside 0.5–10 viewport-mm (wrong units or axes). The arena
shell fails silently and independently of the fly.

## 2. Frames, units and transforms

### Dashboard 3D viewport (`ArticulatedFly3DViewport`)

* World frame: three.js, **Y up**. Arena `(x, y)` mm maps to `(x − cx, ·, −(y − cy))`
  (`arenaPointFor3D`), with the arena-bounds centre at the origin. The floor sits at
  y = −0.02, wall boxes are 2.0 high and centred at y = 1.0.
* `flyGroup` (the fly root) sits at `(x, zBody + 1.2, −y)` and is rotated about Y by
  `−heading + π/2`. Its local frame has **+Z forward (head), +Y up**. The units are
  "viewport-mm": the fly is drawn about 8.3 long, roughly 3× life size, as an
  illustrative scale that the procedural fly already used.
* `thoraxMesh` sits at `(0, 2.2, 0)` in `flyGroup`. All body meshes (head, eyes,
  antennae, proboscis, abdomen, wings, halteres, bristles) are authored **in the
  thorax frame** and are static.
* Six legs, in viewport order `L1 L2 L3 R1 R2 R3`, which the loader maps to the
  FlyGym prefixes `lf lm lh rf rm rh`. The joint-angle stream is
  `joint_angles_rad[i*3 + {0: coxa, 1: femur, 2: tibia}]` and the contact stream is
  `leg_contacts[i]` in the same order.

| Group (parent) | Local origin | Rotation applied by `updatePose` | HQ meshes attached |
| --- | --- | --- | --- |
| `coxa` (thoraxMesh) | `(side·1.0, −0.2, zOff)`, zOff = 0.8 / 0 / −0.8 | `rotation.y = baseAngle + coxa·0.8` | `<leg>_coxa` (length 1.0 along −Y) |
| `femur` (coxa) | `(0, −1.0, 0)` | `rotation.z = side·(0.35 + femur·0.5)` | `<leg>_trochanterfemur` (length 2.2 along −Y) |
| `tibia` (femur) | `(0, −2.2, 0)` | `rotation.z = −side·(0.6 + (tibia − 1.4)·0.6)` | `<leg>_tibia` (0 → −1.55) and `<leg>_tarsus` (−1.55 → −2.4, five tarsomeres plus claws) |
| `contact` sphere (tibia) | `(0, −2.4, 0)` | colour = stance / swing / unavailable | **kept visible**; not replaced |

`side = −1` for L legs and `+1` for R legs; `baseAngle` is ±π/4, ±π/2, ±3π/4.
The GLB repeats every origin and the coxa yaw exactly. The loader resets each
attached mesh to the identity transform under its group, so the pose comes only
from the code above.

Observed, not changed: the viewport puts the L legs at local −X while the fly faces
+Z with +Y up, which is anatomically the right side under a right-handed frame. The
assets follow the viewport convention so that nothing moves. Whether this is a
mirror in the procedural fly is a question for the viewport owner, not something
these assets should settle.

### GLB files (`fly_hq_lod{0,1}.glb`)

* glTF 2.0, **+Y up, metres-as-units = viewport-mm** (one unit is one viewport-mm).
  Node hierarchy: `neurofly_fly` (= flyGroup) → `thorax_frame` at (0, 2.2, 0)
  (= thoraxMesh) → body meshes and `<leg>_coxa_joint` → `<leg>_femur_joint` →
  `<leg>_tibia_joint`, each holding its segment meshes at identity.
* The joint empties carry a **display pose** (legs splayed, knees bent) for
  previews and 2D sprites only. The dashboard ignores the empties.
* Blender is Z-up. The scripts compute everything in the three.js frame and convert
  with `(X, Y, Z) → (X, −Z, Y)`; the exporter's +Y-up option inverts this exactly.

### Static preview placement (`asset_preview.html`, `arena_shell_preview.png`)

The GLB root frame is the viewport's `flyGroup` frame. In the display pose, the
feet hang **below** that root's origin, so a model placed at the origin pokes
through the floor and the plinth. The previews have no physics. They stand the
model from its own geometry by lifting the root by
`surface_top − min(tarsus vertex y)`, where the minimum is an exact scan of every
`<leg>_tarsus` mesh vertex (claws included) in the root frame. The scan is done by
`lowestFootY` in `hq_assets.js`, by `build_arena.py` in Blender and by
`foot_placement` in `measure_assets.py`, and the three agree:

| | lf | lm | lh | rf | rm | rh | mesh minimum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lowest tarsus y, LOD0 and LOD1 | −2.1315 | −1.8798 | −2.1054 | −2.1315 | −1.8798 | −2.1054 | −2.1315 (an lf/rf claw) |

The stand height is −0.02 − (−2.1315) = **2.1115** viewport-mm on the dashboard
floor top (−0.02). With the round shell the floor is hidden, so the preview uses
the shell top instead: −0.06 + 2.1315 = 2.0715. This placement exists only in the
previews. It is never applied to recorded or streamed poses, and the GLB root frame
is unchanged.

### Where the feet meet the floor in the live views (reported, not changed)

* **Dashboard viewport.** The illustrative rig puts the contact spheres (and so
  the HQ tarsus tips, which end at the sphere centre at −2.4 in the tibia group)
  below the floor top. With the illustrative lift (zBody 0.5) and no joint stream,
  the sphere bottoms sit at y ≈ −2.15. With sample joint angles they sit between
  −1.39 and −2.14. The floor top is −0.02. This comes from the procedural rig's
  fixed segment lengths and joint mapping, not from the HQ loader, so it is left
  for the viewport owner.
* **Embodied replay.** Each HQ tarsus mesh ends exactly on the recorded
  `<leg>_tarsus5` body point. In a 5 s FlyGym walk those points sit 0.023–0.624 mm
  above the ground plane (z = 0), and 0.023–0.095 mm during recorded stance. The
  drawn feet therefore show the recorded data as it is.

### Embodied replay (`embodied_replay.js`)

* World frame: MuJoCo, **Z up, millimetres, real size**. The skeleton comes from
  `header.skeleton.segments` (FlyGym body names, for example `c_thorax`,
  `lf_coxa`, `lf_trochanterfemur`, `lf_tibia`, `lf_tarsus1..5`, `l_eye`) and
  `frame.pos` is the world position of each segment's body origin.
* Legs: `<leg>_coxa` spans coxa→trochanterfemur, `<leg>_trochanterfemur` spans
  trochanterfemur→tibia, `<leg>_tibia` spans tibia→tarsus1 and `<leg>_tarsus` spans
  tarsus1→tarsus5. Each mesh is stretched along its recorded edge. The radial scale
  `s` is the median recorded femur length divided by 2.2, or 0.3 when no femur is
  recorded.
* Body: a rigid cluster in the thorax frame, scaled by `s`. Its heading is the
  horizontal component of thorax→head, where head is `c_head`, else `l_eye` or
  `r_eye` (FlyGym 2 records no `c_head` body). If neither is recorded, the heading
  falls back to `frame.yaw`. The drawn head is anchored on the recorded head point.
  The body is drawn level: **no pitch or roll is invented**, and recorded abdomen
  or head articulation is not shown (less than the data, never more).

### 2D views (PNG)

* Orthographic Cycles renders of the **same objects that were exported**, 768×768,
  `ortho_scale` = 12 viewport-mm, so 64 px per viewport-mm. `*_top.png` has the head
  toward the image top and the image centre at flyGroup x = 0, z = −0.6 (the
  thorax centre is 0.6 viewport-mm, or 38 px, above the image centre). `*_side.png` looks along −X and `*_front.png` looks along −Z.
* `arena_shell_{rect,round}_top.png`: the unit footprint fills 1/1.04 of the frame.
  Scale it to the arena's own bounds plus the margin used by the loader.

## 3. Required node names

The validator (`validateFly`) requires: `c_thorax c_head l_eye r_eye c_abdomen12
l_wing r_wing` and, for each of `lf lm lh rf rm rh`, `<leg>_coxa
<leg>_trochanterfemur <leg>_tibia <leg>_tarsus`. Optional parts that are attached
when present: `c_scutum c_thorax_bristles c_head_bristles c_proboscis
{l,r}_pedicel {l,r}_funiculus {l,r}_arista c_abdomen3..6 {l,r}_wing_veins
{l,r}_haltere`. The arena GLB needs `shell_rect`, and `shell_round` for circular
arenas.

## 4. What must not change

* **Stimulus geometry, physics, collision geometry and arena dimensions.** The
  shell is scaled from `assayGeometryDescriptor` (the simulation's bounds) plus a
  margin of 2 + 6 % of its size. Its top sits at y = −0.06, below the floor at
  −0.02. It has no rim, wall, tick, marker or glow, so it can never read as a
  boundary, a cue or a scale. Floors, walls, outlines and the cue layer
  (`assay_cues_3d.js`) are untouched.
* **Every honesty label and banner.** `v3dDataStatus` ("Illustrative 3D view · …"),
  the HELD/UNAVAILABLE pose status, the geometry status, the cue status and the
  replay's "Reference fly (not connectome)" banner are all produced unchanged. The
  HQ mode only *adds* its own label stating that the surface is illustrative and
  that wings, halteres, antennae and bristles are static and not simulated.
* **Measured state stays visible.** Contact spheres keep their colours
  (stance / swing / gray "unavailable"). In the replay the recorded bone lines,
  joint points and trail stay drawn on top of the surface.
* **Biology that is not simulated is never implied.** Wings and halteres do not
  flap, antennae and bristles do not move, and eye facets are a flat-shading style
  rather than an ommatidia model. No neural activity is mapped onto the surface.
* **Default behaviour.** No flag means no request, no injected script, no changed
  object, material, transform or label (checked by `tests/test_hq_assets.js`).
* **The 2D canvas arenas** (`app.js` and `live_assays.js`) are not wired to the
  PNGs yet. Using the PNGs later must keep the drawn arena geometry, stimuli and
  labels exactly as they are: a sprite may replace the fly glyph and a shell image
  may sit behind the arena, never on top of it.
