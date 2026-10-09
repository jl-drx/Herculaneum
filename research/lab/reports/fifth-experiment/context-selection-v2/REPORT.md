# Geometry-informed v2 context plan

**One feasible crop:** source bbox `[768, 256, 768, 768]` on PHerc0800 segment `20251028220955`, at 8.64 µm native spacing. The canvas spans about 6.64 mm. This is a post hoc context extension containing the entire fourth-experiment field, not held-out validation or an independently selected new coverage experiment.

The unchanged v1 result remains a stop under its full-valid-crop requirement. The root agent explicitly authorized v2 after seeing v1 geometry coverage, before any v2 geometry evaluation or new intensity/model inspection. V2 permits limited invalid boundary geometry while excluding the complete support of every potentially contributing invalid-geometry model tile. The v2 methodology and selector were frozen on disk before evaluating v2 candidates. No new intensities, predictions, network requests, acquisition or inference were used here.

## Frozen gates and actual result

Origins remain on the 128-pixel grid, inside the `[31, 2100, 2100]` source. Every candidate must contain fourth bbox `[1024, 384, 384, 384]` and give both fixed source points `(1278, 504)` and `(1100, 628)` at least 128 pixels of crop-edge margin.

V2 requires at least 95% whole-crop geometry validity (560,333 pixels) and at least 95% strict geometry coverage in the 512-square analysis interior (249,037 pixels). Schedule all 121 complete 128-square tiles at stride 64, without intensity-dependent skips. If any geometry pixel in a tile is invalid, mark its entire 128-square footprint bad. Strict geometry is valid geometry inside local `[128:640, 128:640]`, excluding the union of those bad footprints. Each fixed point and its four cardinal neighbors must also lie inside strict geometry.

All 121 inside-source grid origins were enumerated; nine meet the containment and point-margin constraints. Only `[768, 256, 768, 768]` passes all v2 gates:

- Whole geometry: 567,072 / 589,824 valid pixels (**96.1426%**), with 22,752 invalid pixels.
- Strict geometry interior: 258,048 / 262,144 pixels (**98.4375%**).
- Ten scheduled tiles contain invalid geometry. Their complete footprint union contains 94,208 pixels and removes 4,096 pixels from the otherwise valid analysis interior.
- Both points and all eight cardinal-neighbor checks pass.

Ranking is descending strict interior count, descending total valid count, ascending squared center distance to the old field center `(1216, 576)`, then Y and X. The sole feasible candidate wins without a tie. Evaluation took 0.580 seconds, including verification of 25 frozen input records before and after selection.

## Acquisition contract and remaining gates

`result/selected-plan.json` preserves the geometry decision, masks, XYZ crop, diagnostic records, input hashes and chunk list. `result/acquisition-plan.json` also provides the flat acquisition fields and references the selected-plan hash. It records the source metadata, global geometry maps, exact render provenance and fourth acquisition provenance.

The 768 crop covers 36 uncompressed `[31, 128, 128]` uint8 chunks (18,284,544 bytes). Nine already observed HTTP-200 chunks have verified source URL, index, length and SHA-256 identities and can be reused; at most 27 new chunks require 13,713,408 raw bytes. The plan bounds all new response bodies, including two metadata responses, at 13,844,480 bytes with zero automatic retries. A separate frozen 768/cache-aware acquisition adapter must preserve observed-versus-fill provenance and accept this explicitly partial geometry mask. The original 384 adapter remains unchanged.

The plan's native CT mapping gate and future adapter hash remain null. Actual acquisition is conditional on the root's mapping investigation. Geometry validity and the strict geometry mask do not certify observed selected-depth data: subsequent analysis must also exclude complete contributing tile support containing unobserved or missing selected-depth pixels. Geometry masks must not be mixed into model normalization.

The old and new origins share the stride-64 tile phase. An eventual same-model forward comparison on the fourth interior can check source preservation and context consistency; it would be an implementation diagnostic, not new scientific validation.

Local XYZ continuity diagnostics are finite without detected finite-difference degeneracy. Archived renderer normals and full interpolation/accumulation footprints were not reconstructed here. No independent physical surface, ink, stroke sequence, reading or translation is established by this plan.
