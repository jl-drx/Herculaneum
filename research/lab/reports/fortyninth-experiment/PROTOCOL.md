# Study 49 — PHerc0800 development transfer qualification

Frozen before new source-intensity access, rendering or inference. The other VM's
independent phase48 geometry check continues separately. This is a development
study on previously inspected segment20251028222030, not a fresh accuracy test.

## Fixed inputs and scope

Use the original 129x129 mesh and recorded flipped normal convention on the full
2580x2580 canvas. The eventual scoring field remains source x512:2048,y512:2048.
No M7 snapping or fitted geometry. First qualify only x1216:1344,y1216:1344,
chosen around the three archived source-check points at1280, not model responses.
Run the original PHerc343 VC3D41701aa executable on this crop at31 and66 layers,
unit layer spacing, flip-normals, no rotation/affine, scale1, group0. At31 layers
compare every output byte to the retained published31-layer surface. At66 layers
compare every byte to an independent raw-CT trilinear reconstruction using the
archived centered-canvas float32 geometry; offsets are k-32.5.

Prepare the conservative source-chunk whitelist from geometric corner envelopes,
with0.01 voxel arithmetic margin. Require all crop base/normal supports valid.
Cap actual needed native CT payload at32MiB (metadata at2MiB). Reuse verified
retained chunks; fetch only whitelisted public level0 chunks. Missing or unplanned
chunks are fatal; a renderer's silent Zarr fill cannot establish observed support.
Record received requests, byte lengths and SHA256. Actual renderer cache requests
may be more conservative than interpolation support; stop and document if outside
the whitelist, rather than silently widening the source scope.

## Gates, decided before looking at new outputs

1. The crop has full finite bilinear and normal support and all planned raw chunks
   are present, correctly sized and verified against acquisition receipts.
2.31-layer reproduction: max absolute discrepancy <=1 uint8 level and >=99% exact
   bytes.66-layer independent sampler: max discrepancy <=1, mean <=0.1 and >=95%
   exact bytes. Float32 contraction and crop-origin arithmetic may change rounding;
   no fitted offsets or post-result tolerance widening. Report failures explicitly.
3. If rendering passes, qualify the three Villa models on the fixed128x128 tile
   and Hecate on global prepared-grid tile y1120:x1120 of size64. Keep the full
   2580 canvas's0.9 resize grid; require every input resampling contributor known.
   For each model run one tile and a batch of four copies; finite fp32 outputs,
   strict checkpoint loading and max sigmoid discrepancy<1e-4. Total20 forwards.
   Stop inference if any support or rendering gate fails.

The nominal centre d0 preserves author contracts: Villa native planes25..41
(reversed41..25), physically centred+0.5 voxel relative to the mesh; Hecate
16 interpolated planes centred32.5 (physical0), spacing9.6/8.64, reversed. This is
not an identical physical centre. Reverse describes array order only; matching
export orientation does not establish the papyrus reading side. No forward-order
or depth sweep occurs in this qualification; these need a subsequent frozen plan.

## Compute planning and interpretation

Count complete-canvas and scoring-field-contributing tiles for both plain and
40-row/200-column top/left padding. Hecate counts include the global0.9 area
resampling and final linear upsampling kernel. Villa occupancy can only lower
an all-grid bound; call bounds bounds until actual occupancy is observed. Preserve
the global grids and original batching when planning any cropped evaluation.
Calculate conservative raw-chunk union for all required patch context, working
array sizes and measured CPU estimates; do not fetch that larger source set now.
No full target map, depth sweep, training, GPU resource or reading claim in49.

Prospective display for a later target map: individual raw uint8 outputs on a
fixed0..255 display, with support masks and both phases separate. PHerc343's
joined-mesh ownership rule does not apply to this single mesh. Its all-depth
fitted ensemble statistics are not automatic PHerc0800 calibration. Do not fit
contrast or propose letters from this qualification tile.

A pass licenses preparation of a centre-depth development run with checkpointed
CPU scheduling. A failure licenses diagnosing rendering/support before inference;
it says nothing about whether the target contains ink. Seal and privately publish
all outcomes and the next-study decision. No paid resources.
