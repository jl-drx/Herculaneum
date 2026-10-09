# Fifth experiment: native CT checks and conditional context expansion

This is a follow-up to two detector-selected responses from the fourth experiment, at source X/Y `(1278,504)` and `(1100,628)` on PHerc0800 segment `20251028220955`, volume `20250521135224`. The selections are post hoc diagnostics, not new held-out validation. Preserve the previous results and all negative or partial outcomes.

## Native source and rendering check

Freeze each response peak and its four cardinal neighbours at one source pixel distance, giving ten points. Reconstruct all 31 surface samples using the recorded renderer commit and parameters: half-pixel/grid mapping, float32 tile/halo arithmetic, nearest cached mesh normal, recorded normal flip, fixed offsets −15 through +15, trilinear native CT sampling, clipping and uint8 truncation. Do not search offsets, signs, transforms or contrast to improve agreement.

Require every trilinear corner to be supplied by an observed native CT chunk from the exact volume. Missing-object fill cannot pass the observation requirement. Report exact matches, within-one-count matches and residual locations. The practical mapping gate requires every sample within one uint8 intensity count; any larger error or source/metadata/convention mismatch pauses expansion for diagnosis. This tolerance is not proof of bit-identical rendering. Matching a generated surface to its CT source does not establish correct papyrus-layer placement or ink.

Freeze the exact chunk list before network acquisition. At most eight 128-cubed native uint8 chunks (16 MiB) plus 256 KiB of metadata are permitted, with no automatic retries. Reuse only previously retained same-volume chunks with verified source identity and hashes. The geometry preflight currently requires two new chunks (4 MiB); retain all interpolation support and provenance.

For each peak, produce two fixed CT cross-sections using the recorded base coordinate, unit normal and orthogonal in-plane frame. Normal offsets are integers −24 through +24 native voxels; tangent offsets are −48 through +48. Use only the already acquired native cubes, marking any sample without all eight observed corners as unknown. No extra downloads, fitted contrast or preferred slice selection are allowed for these figures. Show 0–255 grayscale with the base surface, selected input depth ±8 and full rendered depth ±15 indicated. These are local geometry diagnostics, not independent papyrological readings.

## Conditional surrounding context

The first frozen 768-square rule required complete geometry validity. It failed: no such crop contained the previous 384-square region with both peaks at least 128 pixels from the edge. Preserve `context-selection/` as a completed negative geometry result. The deterministic rule was specified before candidate inspection; a secondary geometry inspection preceded its on-disk methodology freeze, which is disclosed in its report.

A separate geometry-informed revision may accept a partially valid outer boundary while preserving strict interior support. Before new intensities or predictions, freeze the following rule in `context-selection-v2/`: same containment and point margins, at least 95% whole-crop geometry validity, and at least 95% of the fixed 512-square interior remaining after excluding the complete support of every scheduled 128-square, stride-64 tile containing any invalid geometry. Both response peaks and their cardinal neighbours must be inside that strict interior. Rank by strict interior support, then total valid pixels, then nearest crop center to the old crop center, then source Y/X. If none qualify, stop this revised plan without another automatic relaxation.

Acquisition is conditional on a successful native-source check. If selected, acquire a fixed 768-square, 31-layer field, retaining the old nine verified chunks and fetching only the other 27. Maximum additional surface payload is 13,713,408 bytes; total assembled chunk payload is 18,284,544 bytes. Retain an observed-source mask distinct from geometry and any declared missing fill.

If acquisition proceeds, use the same unchanged two checkpoints on center `7:24`, and the same three seed42 depth controls (`5:22`, `9:26`, reversed `7:24`). Freeze the execution plan before acquisition. Report all supported outputs and compare the overlapping old interior directly, expecting identical source bytes and matching scores where the tile context is unchanged. Do not interpret shared training, overlapping depth slices or repeated input pixels as independent replication.

No target training, transcription, translation, external publication, GPU resource or provider charge is part of this experiment. New cloud results remain local until a supported persistence/export path is completed.
