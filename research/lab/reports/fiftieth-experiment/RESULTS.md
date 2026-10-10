# Study50: independent audit verified; smaller full-recipe core rendered and tested

The VM audit is correct and useful. Its192x192 candidate passes the stated single
native grid, while the original conservative128-minimum result remains unchanged.
The padded Villa grid rejects part of that candidate. A prospectively permitted
contained128x128 subcore **x1024:1152,y280:408** on segment20251028225813 passes all
four model/phase footprints under both arithmetic conventions. Actual rendering
and a separately frozen16-map centre-depth comparison now pass their execution
checks. This is a usable controlled region; it is not ink accuracy or a reading.

## Independent audit assessment

The uploaded archive SHA256 is
`059701de670216bde9ed180c032708aa511cab20dcec4e1674d27906f0587443`.
All108 members were safely extracted and all107 internal file checksums pass.
All52 input-manifest source files match their actual retained local sources.
The unmodified locked audit was run in a separate output directory with matching
NumPy/SciPy/tifffile versions. Its five meshes,40 masks, all40 component results
and every non-environment JSON value reproduce exactly. That full replay took
106.05s and peaked at1,651,140KiB RSS on this host.

The nine point-level arithmetic discrepancies are real and preserved. They do
not change either full conservative context masks or exact single-grid masks on
any mesh. All40 conservative component ranks, areas, square sizes and positions
match the sealed phase48 reference. The accompanying exhaustive square checks,
500 direct pixel/context tests and10,000-normal/66-layer envelope checks pass.
Post-reference convention reconciliation is correctly labeled as reconciliation;
it is not additional independent evidence.

The audit's units correction is valid: treating0.01 as a per-coordinate error
changes the two-position Euclidean allowance to0.02sqrt(3). The reported resulting
sufficient bound50.998742631 remains below51.1, so the unchanged gate is adequate.
Failure of that sufficient bound does not prove overlap, the same sheet or no ink.

## Extending the candidate to actual model footprints

Before new intensity access, study50 froze the proposed192-square core and a
fallback limited to its largest common supported square, with topmost/leftmost
tie-breaking and a128 minimum. No other component, offset, depth or detector
response was searched. Hecate preparation and output interpolation use the full
canvas, not a locally reset crop grid.

| Contract on the original192-square core | Contributing patches | Fully supported patches | Supported core pixels |
| --- | ---: | ---: | ---: |
| Villa plain |16|16|36,864|
| Villa padded |25|16|25,600|
| Hecate plain |56|56|36,864|
| Hecate padded |49|49|36,864|

The common core's largest contained square is128x128 at(1024,280), identically
under both arithmetic implementations. The new rule does not alter phase48's
result or promote the entire192-square proposal. Only the padded Villa context
reduces it. Six newly acquired study49 chunks were added to the current exposure
inventory; they exclude no previously admitted point in this candidate segment.
All required current context still clears the expanded inventory.

For the selected128-square, each phase needs12 native Villa patches and36 Hecate
patches. Their exact combined native source footprint is98,360 pixels; every one
passes both arithmetic conventions and all current exclusions. Minimum archived
nearest-old-centre distance remains53.70381249 voxels,2.60381249 above the51.1 bound.
The code checks full contexts, not merely scored pixels or patch centres.

## Observed CT and actual renderer

The bounded acquisition downloaded26 new level0 chunks,52MiB. Seven verified
study49 chunks were reused only for the output-metadata control; all33 available
chunks have byte/SHA receipts. No missing chunk or unplanned raw request was
accepted as Zarr fill. No larger-field source set was acquired.

A global-tile-aligned bounding rectangle would include invalid geometry outside
required inputs, and the tighter bounding rectangle has an invalid8x8 corner.
Both preliminary plans stopped before acquisition. The successful plan renders
two explicit valid rectangles, leaving an8x24 strip unknown in the assembled
320x320 context canvas. All98,360 required pixels avoid that strip. The renderer
observed102,208 context pixels;3,848 extras are not part of the required model
footprint. All extra acquisitions are retained for future exposure accounting.

Across **6,745,728 observed66-plane samples**,6,724,366 (**99.6833%**) match the
independent float32 raw-CT sampler exactly. Every remaining discrepancy is1 uint8
level (mean0.003167); the original frozen gate passes. The explicit8.64um output
metadata is correct, and adding it to the study49 control render preserves every
intensity byte. This fixes the earlier generic-metadata hazard without changing
scientific input values. Matching CT mechanics does not establish the ink surface.

## Fixed sixteen-map comparison

A separately frozen stage ran dnative, seed42, seed43 and Hecate at nominal d0,
with both depth orders and both padding phases. All288 singleton forwards ran in
fp32 CPU; no training, contrast fitting or ensemble was introduced. Every map
uses the original global grid and sampling/window contracts. Villa and Hecate
retain the released half-voxel difference in their physical window centres.

Original Villa dataset/probability/accumulation code was used on contributing
patches. Original Hecate predict code retained the full global denominator grid,
with unneeded model inputs explicitly suppressed outside the scored footprint.
No unknown input contributes to the scored core. Batch1 fixes cohort identity;
bit equivalence to batch4 or GPU inference is not claimed. The comparison took
**387.4s (6.46min)** including saved-logit replays. A controlled stop after two
committed native forwards resumed successfully; all16 maps were reproduced
exactly from immutable logits with zero extra model forwards. Completion receipts
were written last. This demonstrates the chosen restart path, not arbitrary
platform-failure behavior or cloud snapshot persistence.

All16 fixed0..255 panels are in ALL_MAPS.png. Descriptive padding comparisons:

| Model | Forward correlation / mean absolute difference | Reverse correlation / mean absolute difference |
| --- | --- | --- |
| dnative |0.353 /3.69 levels|0.652 /4.85 levels|
| seed42 |0.725 /12.85 levels|0.433 /10.17 levels|
| seed43 |0.578 /9.49 levels|0.742 /18.17 levels|
| Hecate |0.816 /6.13 levels|0.520 /0.94 levels|

Forward/reverse correlations range from−0.159 to0.477. Hecate responds more
strongly in forward order on this region, but this does not identify the reading
side. The low reverse-order Hecate absolute difference partly reflects near-zero
responses, not established accuracy. Cross-model means and calibration differ;
no model is ranked as more accurate. MAP_COMPARISONS.json retains all16 summaries
and40 pair comparisons.

On reviewing every fixed-scale panel, I do not see defensible lettering. The
maps contain diffuse or localized responses that vary with model, padding and
order. This approximately1.1mm-square region is too small to establish readable
text, and a dark or unstable response does not establish absence of ink. It is
now detector-review exposed and is not a pristine held-out benchmark, a new
independent specimen, or evidence of complete training exclusion.

## Decision and next work

Retain the exact region, full depth input, masks and all responses as a fixed
regional comparison. The independent review has changed our usable-region
assessment without weakening the earlier gate. Continue the larger existing
PHerc0800 development field with both directions/phases, fixed raw display and
restartable work; do not choose settings by this tiny crop's apparent brightness.
See NEXT_STUDY_DRAFT.md. That longer worker has not been launched in study50.

The user's new delegation preference is recorded: keep short work in this chat;
use the other VM when an hour-plus local run or similar occupation makes parallel
independent work useful. No new Actions dispatch or VM assignment was made here.
GPU spend remains£0; there is no transcription, translation or prize claim.
