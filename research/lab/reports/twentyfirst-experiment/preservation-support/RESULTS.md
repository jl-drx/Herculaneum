# Preservation support audit

**The existing schedule has reproducible manual and retired-validation participation masks, but no authenticated geometry-valid output mask.** Full-valid-output preservation remains blocked. A narrower experiment could explicitly use the supplied retired-validation membership for teacher consistency, under a separately frozen root protocol. That would change the role of known reference participation; it would not establish new ink labels or geometry validity.

This audit read only the eight retained PHerc0139 acquisition, supervision, validation and diagnostic occupancy masks, plus source/code/schedule metadata. It read no CT arrays, ink-label arrays, teacher logits, predictions or model checkpoints. No model, optimizer or network request ran. Both original 384 × 384 contexts and the exact 32 batch-two schedule were reproduced, including crop coordinates and all D4 transforms.

## What the sources establish

The source crops are entirely within the level-2 surface canvas. Nine complete 109 × 128 × 128 byte chunks cover each crop according to the retained acquisition receipts; the assembler stopped on missing objects and never filled missing chunks. There is no local crop padding or locally substituted missing-source fill. This does **not** establish whether the upstream surface renderer sampled an unmapped point, an invalid normal or a masked native voxel.

Both original provenance records explicitly state `geometry_validity = {"status": "not_acquired", "mask_file": null}`. `acquisition_observed_mask` records successful source transport. `source_observed_nonzero_mask` is explicitly diagnostic occupancy, and every retained context pixel happens to have a nonzero profile. Neither is an authenticated mapping-validity mask. Geometry-valid and geometry-invalid counts therefore remain unknown (`null`), rather than assumed zero or assumed all pixels.

The released supervision attributes describe a relaxed mapping-validity policy with proximity and cleanup operations. That prose does not supply the original per-pixel validity raster or certify every output pixel. Existing supervision is conditional supplied-reference participation, not independently authenticated ink truth or geometric validity.

## Exact participation and repeated exposures

Let `O` be acquired source coverage, `S` supplied supervision and `V` supplied validation membership. The existing manual mask is `M = O & S & ~V`. Retired validation is `O & V`; annotation-unknown participation is `O & ~S & ~V`. Those three disjoint groups partition acquired outputs. Supervision and validation have **zero overlap** in both entire contexts, so expanding to `S` alone would add nothing to the existing manual support.

| Role | Entire two contexts | Unique scheduled pixels | Scheduled output exposures |
|---|---:|---:|---:|
| Acquired source coverage | 294,912 | 102,400 | 1,048,576 |
| Existing manual participation | 57,166 | 54,701 | 618,208 |
| Retired validation, all outside supervision | 139,516 | 43,515 | 409,448 |
| Annotation-unknown participation | 98,230 | 4,184 | 20,920 |

The 54,701 unique manual pixels split into 25,967 in rank01 and 28,734 in rank02. Rank02 contains an additional 2,465 manual pixels outside the scheduled tile union. Repeated exposure counts include overlapping tiles and repeated scheduled visits; they are not independent observations. Every scheduled manual pixel is visited at least five times, and some source pixels appear 22 times.

Each batch contains 8,372–30,732 manual pixels and 2,036–24,396 retired-validation pixels. Each individual patch also has both groups: at least 4,008 manual and 464 retired-validation pixels. All 4,184 scheduled annotation-unknown pixels are in rank01 and appear five times each. They are separate from retired validation and are not background targets.

`audit-v2/support-contract.json` retains every per-batch count and per-region multiplicity histogram. Its Boolean batch masks have shape `[32,2,1,128,128]`. `batch-source-index.npy` records the transformed local source index `y*384+x`; the second axis identifies the context. Exposure maps retain source-space visit counts. No file is named or presented as a geometry-valid mask.

## Synthetic objective checks

The broad-support preparation checked a fixed manual BCE plus three illustrative KL reductions: manual mean, uniform supplied-support mean and a half/half stratified mean. The final fixture passed 39 checks, including independent NumPy loss/gradient formulas and scalar finite differences. A uniform wider-support mean dilutes each manual KL derivative by `|M|/|support|`; keeping the same scalar coefficient does not keep manual preservation strength unchanged. For the actual hypothetical `M | V` participation union, that ratio varies from 0.25549 to 0.93787 across batches.

A separate 32-check fixture validates the exact root-proposed comparison:

- A: `mean BCE_M + 2 * mean KL_M`.
- B: `mean BCE_M + mean KL_M + mean KL_V`.

Each group mean pools its participants across the two patches in a batch. BCE remains FP32 with targets `0.25 + 0.5*y`; both KL means are FP64 Bernoulli `KL(seed42 || student)` with a detached teacher. Only explicit manual labels enter BCE. Retired-validation target values and all other nonmanual target values are ignored. Both role masks must be nonempty and disjoint; empty or overlapping masks are rejected. Unsupported output gradients are exactly zero. Teacher gradients are absent.

This comparison reallocates one unit of preservation from manual support to retired validation. Both sums of group coefficients are two, but B halves the manual KL derivative. It is not unchanged manual protection, and it is not twice a uniform union mean when group sizes differ. Lower KL alone would not establish better ink detection. No scientific training or final study choice is made by these fixtures.

## Preserved corrections and limits

The first support audit passed but contained an unused stale constant for the top-level training-plan hash. Its actual input pin was correctly recorded. The second audit replaces that constant with the verified pin and enforces it before mask reads. All batch arrays and counts are identical. Both receipts and the original source bytes are retained.

The initial broad-support synthetic fixture stopped because it demanded bitwise zero from a mathematically zero stratified KL derivative. Floating point summation left tiny roundoff. The corrected fixture uses an explicit absolute tolerance of `1e-15` for that FP64 identity derivative; identity loss and gradients outside support remain exact-zero checks. The objective kernel did not change. The stopped fixture and its source are retained; no scientific run failed or was retried.

The audit itself literally read no ink-label arrays. A future runner that reads whole historical label or pseudo arrays for exact fingerprint verification must instead state that retired labels do not participate in targets, objectives, teachers or selection; it must not claim their bytes were unread.

## Geometry prerequisite lead

The exact retained surface is `PHerc0139/segments/20250108000004-w029_2025010827/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr`, with level-2 canvas `[109,7020,7220]` and crop boxes `[1792,4864,384,384]` and `[3584,5120,384,384]`. Model inputs use pooled planes `[2,19)`, corresponding to published source-depth `[21,89)` in groups of four. The source attributes name a local masked native Zarr path; that path is not itself an authenticated public native-volume mapping contract.

No PPM or mapping-validity raster is linked by these exact acquisition records. A useful next metadata step is to bind the original segment, renderer revision, native volume and mapping artifact from the retained full metadata, then determine the renderer's invalid-coordinate, normal and masked-source handling before proposing a bounded local mapping-support extraction. The root's separate PHerc0139 geometry trace handles this prerequisite. Re-reading surface intensity or broadening the mask because pixels are nonzero would not resolve it.

All earlier evidence remains unchanged. The support artifacts and synthetic fixtures are ready for independent review; any scientific follow-on requires its own root-owned frozen protocol.
