# Female and male appearance variants

These variants change only how the fly looks: abdomen shape and pigmentation,
the male sex comb, and the display size.

**Selecting an appearance never switches the brain dataset, never changes
physiology or body mechanics, and never implies sex-specific behaviour.** The
connectome, the controller, the body model, the stimuli and every measurement are
the same whichever variant is drawn. The UI that offers the choice (owned by the
integrator, not these scripts) must say so wherever the choice appears. The
existing neutral fly (`fly_hq_lod*.glb`) stays the default and is rebuilt
byte-identically.

## What differs, and the sources

| Feature | Female | Male | Source |
| --- | --- | --- | --- |
| Visible abdominal segments | Seven (A1/2 to A7); A7 is merged into the `c_abdomen6` mesh | Six; A7 is externally absent | Wang W, Kidd BJ, Carroll SB, Yoder JH (2011) *PNAS* 108:11139–11144, doi:10.1073/pnas.1108431108 ("males develop fewer abdominal segments than females"). Williams et al. 2008 (below): "the female-specific segment A7 (this segment is greatly reduced in males)". |
| Tergite pigment | A dark posterior band on every tergite | Bands on A2–A4; A5 and A6 fully dark, read as one fused dark tip | Williams TM, Selegue JE, Werner T, Gompel N, Kopp A, Carroll SB (2008) *Cell* 134:610–623, doi:10.1016/j.cell.2008.06.052: males "typically have fully pigmented dorsal cuticular plates (tergites) on abdominal segments A5 and A6"; "in females, A5 and A6 tergite pigmentation is restricted to a posterior stripe". Kopp A, Duncan I, Carroll SB (2000) *Nature* 408:553–559, doi:10.1038/35046017. |
| Abdomen shape | Longer (3.95 vs 3.45 neutral), wider (radius 1.12), tapering to a point | Shorter (3.05), narrower (0.98), blunt rounded tip | A **stylisation** that follows from the extra tapered A7 segment in females and the lost A7 in males (sources above). The exact profile is not measured. |
| Sex comb | — | One row of 10 short dark teeth along the axis of the first tarsal segment (basitarsus) of both front legs (`lf_tarsus`, `rf_tarsus`) | Kopp A (2011) *Evol Dev* 13:504–522, doi:10.1111/j.1525-142X.2011.00507.x (a male-specific array of modified bristles on the prothoracic leg). Doucet D et al. (2023) *microPublication Biology*, doi:10.17912/micropub.biology.000884: in *D. melanogaster* "the sex comb typically originates from the most distal transverse rows on the first tarsal segment (ta1)"; the same paper discusses the comb's rotation into its final, roughly vertical orientation in *D. melanogaster*. The tooth count and the side of the segment are stylised, not measured. |
| Size | Display scale 1.0 (reference) | Display scale 1/1.15 = **0.8696** | David JR, Gibert P, Mignon-Grasteau S, Legout H, Pétavy G, Beaumont C, Moreteau B (2003) *J Genet* 82:79–88, doi:10.1007/BF02715810: at 25 °C the average female/male ratios are 1.16 (wing length) and 1.15 (thorax length). The thorax ratio is used here. |

Not modelled (would need further sources): male genital arch detail, female
ovipositor plates beyond a small dark terminal cone, eye-size or colour
differences, and variation with rearing temperature.

## One rig interface

* Same node names, hierarchy and joint origins as the neutral fly (see
  [INTERFACE.md](INTERFACE.md)), checked by `tests/test_hq_assets.js`. The abdomen
  nodes keep the FlyGym body names (`c_abdomen12`, `c_abdomen3`–`c_abdomen6`); A7
  and the terminalia live inside `c_abdomen6`. The sex comb lives inside
  `lf_tarsus` and `rf_tarsus`. `hq_assets.js` therefore attaches either variant
  without any change.
* The geometry is authored in the shared rig frame (viewport-mm, 1 unit =
  1 viewport-mm). The size difference is **metadata**, not baked into the
  geometry. The root node `neurofly_fly` carries glTF `extras`: `nf_rig =
  "neurofly-viewport-fly-v1"`, `nf_units`, `nf_variant`, `nf_display_scale`,
  `nf_display_scale_basis` and `nf_appearance_only`. An integrator scales the root
  (`flyGroup`) uniformly by `nf_display_scale`, which keeps every joint origin
  proportional. It must not scale individual parts.
* The neutral fly carries no extras. Treat it as `nf_display_scale = 1`.

## Floor bounds (both sexes, both LODs)

The scan is the same exact vertex scan as the preview foot fix (`foot_placement`
in `measure_assets.py`, `lowestFootY` in `hq_assets.js`). It runs in the root frame
with the display leg pose, and the values are recorded in
`PROVENANCE_SEX_VARIANTS.json`.

| GLB | lowest tarsus y: lf / lm / lh (rf / rm / rh identical) | mesh minimum y | bounds min (x, y, z) | bounds max (x, y, z) | stand height on floor top −0.02 | with display scale |
| --- | --- | --- | --- | --- | --- | --- |
| fly_female_lod0 | −2.1315 / −1.8798 / −2.1054 | −2.1315 | (−3.6387, −2.1315, −4.95) | (3.6387, 3.5119, 2.9842) | 2.1115 | same (scale 1) |
| fly_female_lod1 | −2.1315 / −1.8798 / −2.1054 | −2.1315 | (−3.6232, −2.1315, −4.95) | (3.6232, 3.47, 2.9842) | 2.1115 | same |
| fly_male_lod0 | −2.1315 / −1.8798 / −2.1054 | −2.1315 | (−3.6387, −2.1315, −4.05) | (3.6387, 3.5119, 2.9842) | 2.1115 (unscaled) | lowest −1.8535, stand **1.8335** |
| fly_male_lod1 | −2.1315 / −1.8798 / −2.1054 | −2.1315 | (−3.6232, −2.1315, −4.05) | (3.6232, 3.47, 2.9842) | 2.1115 (unscaled) | lowest −1.8535, stand **1.8335** |

The derivation is `stand = floor_top − display_scale × min(tarsus y)`, with the
root scaled about its origin. The feet are the lowest geometry in every variant,
because neither abdomen reaches below them. As before, these are static preview
placements and are never applied to recorded or streamed poses.

## Files and build

`build_fly.py --sex female|male --lod 0|1 [--render 1]` writes
`fly_<sex>_lod<N>.glb` and `.blend` (editable). With `--render 1` it also writes
transparent orthographic `_top`, `_side` and `_front` PNGs and a perspective
`_hero` PNG from the same objects, at 64 px per female-referenced viewport-mm,
with the root scaled by the display scale for the renders only.
`build_sex_variants.sh OUT` builds all four, adds a close-up of the male sex comb
(`render_closeup.py`), measures them and writes
`fly_sex_variants_contact_sheet.png` (`contact_sheet.py`, needs Pillow). The
left/right leg naming question in INTERFACE.md is unchanged and stays separately
scoped.
