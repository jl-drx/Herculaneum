# Offline rendering and registration mechanics passed

We now have a tested prototype for two parts of a new reference-construction workflow: sampling explicitly specified 3D points from bounded source chunks, and checking an affine landmark registration on points excluded from fitting. This is useful development while the maintainer request is pending. It is not a new ink model, a completed renderer/storage integration, or a qualified Frag6 benchmark.

All **37 tests passed on the first run**, with no failures, errors or skips. The complete validation and diagnostic run took **2.36 seconds** and peaked at **71,790,592 bytes RSS**. Root separately checked the saved evidence and re-enumerated the chunk plan with scalar arithmetic. No independent-agent review is claimed.

## Sampling behavior

[reference_mechanics.py](reference_mechanics.py) makes voxel-center XYZ coordinates and array/chunk ZYX order explicit. A batch plans all unique required chunks before calling the source loader, rejects budget overflow before a read, and loads each planned chunk once. Exact edge-chunk shape and dtype are checked; there is no silent source conversion, axis swap, boundary clamping, intensity normalization or half-voxel shift.

The sampler returns FP64 interpolated values plus a separate validity mask. A missing chunk or nonfinite contributing voxel invalidates the sample and leaves NaN. A genuine zero intensity remains valid. Only corners with positive interpolation weight matter, so exact outer integer voxel centers are valid and an unused missing neighbor does not cause an unnecessary rejection. Tests cover chunk faces/edges/corners, two different chunk partitions, valid zeros, missing chunks, nonfinite values, repeated reads, invalid geometry, budget violations and source shape/dtype errors.

The declared per-batch ceilings are 100,000 points, 128 logical chunks and 64 MiB of uncompressed source arrays. The current loader is an in-memory callback used with synthetic volumes. No live S3/Zarr transport, compressed/sharded decoder, persistent cache or full-surface output writer is implemented here. FP64 reference arithmetic is not a compatibility claim about a historical renderer or a future FP32 model input pipeline.

Ray construction also requires an explicit signed-offset and vector convention. It supports physical-unit vectors and unit voxel-coordinate direction vectors, with spacing-aware displacement. It rejects nonunit mapped vectors and preserves invalid records. Under anisotropic voxel spacing, a surface-normal covector may require an inverse-transpose conversion before it can be used as a physical direction. The prototype does not infer that conversion, an anatomical normal, the ink-bearing side or the historical depth origin.

## Known-truth controls

The synthetic volume is an analytic multilinear function with different X/Y/Z coefficients. Trilinear interpolation should reproduce it exactly up to arithmetic rounding. A separate direct scalar interpolator supplies an additional oracle.

| Diagnostic | Mean absolute error | Maximum absolute error |
|---|---:|---:|
| Correct coordinate contract | 5.45 × 10⁻¹⁴ | 2.27 × 10⁻¹³ |
| Deliberate X/Z swap | 189.542 | 549.699 |
| Deliberate half-voxel shift | 59.833 | 61.785 |

These are arbitrary synthetic intensity units, not scroll-image accuracy. The negative controls demonstrate that the checks detect concrete coordinate mistakes instead of merely accepting plausible-looking outputs. The frozen analytic/scalar agreement tolerance was 1e-10; it did not change after results.

![Synthetic coordinate and registration controls](run-v1/synthetic-controls.png)

Root viewed the whole figure: the sampled analytic profile coincides with its truth curve; both deliberate mistakes produce visible error; the held-out registration displacement is visible. The figure contains no scroll image or prediction and was rendered once.

## Registration behavior

The affine helper accepts uniquely named optical-XY/target-XY landmarks with disjoint fitting and check IDs. It rejects overlap, duplicate or unknown IDs, nonfinite landmarks and degenerate source fitting geometry. Source-coordinate normalization limits numerical conditioning problems; the normalized fitting-design condition ceiling is 1e8. This is a landmark-fit/check utility, not an image correspondence finder or nonlinear registration method.

For the synthetic known affine transformation, the maximum fitting residual was **6.36 × 10⁻¹⁵ target pixels**. A deliberately injected `[3,4]`-pixel error in one excluded check landmark remained a **5-pixel check residual**, with check RMS **3.535534 pixels** across two check landmarks. It did not change the fitted transform. This demonstrates why low fitting error cannot establish the accuracy of registration outside the fitting points.

No real landmarks were annotated or inspected. The helper does not decide which anatomical landmarks are trustworthy, establish a residual threshold for Frag6, assess nonlinear deformation, certify transform invertibility over an image domain, propagate registration uncertainty into labels, or warp optical/label images. Those are separate integration and validation requirements before using a real reference.

## Geometry-only chunk-budget rehearsal

The dry run used **all 729 previously retained sparse PPM records**, including 558 mapped records and 171 all-zero sentinels. It acquired no new geometry, source CT, optical image or annotation pixels. No sites were selected or replaced.

For this planner diagnostic only, the declared offsets were `[-3.24, 0, +3.24] µm`, with isotropic 3.24 µm spacing and the candidate volume bounds from prior metadata. These are newly declared probe assumptions, not a reconstruction of the old 65-plane stack. A hypothetical uint16/64³ chunk layout produced:

| Planned quantity | Result |
|---|---:|
| Point slots | 2,187 |
| Geometrically valid point slots | 1,674 |
| Preserved invalid point slots | 513 |
| Positive-weight corner references | 13,392 |
| Unique logical source chunks | 78 |
| Uncompressed source-array budget | 40,894,464 bytes = **39 MiB** |
| Actual source-loader calls / network requests | **0 / 0** |

The scalar second pass reproduced the complete 78-chunk schedule and byte total exactly. This fits the proposed 64 MiB batch budget under the stated assumptions. It is not an estimate of compressed transfer, request count, egress cost or the actual native Zarr layout. Bounds agreement does not identify which energy scan produced the original PPM or authenticate its coordinate frame. Frag6 Stage C remains closed.

## What follows

The next useful work is to bind a real source-storage contract and test the storage adapter using synthetic and already permitted data before any candidate evaluation. [NEXT_STUDY_DRAFT.md](NEXT_STUDY_DRAFT.md) separates that implementation work from historical provenance and annotation requirements. A curator response may enable the historical route; otherwise a newly declared render and independently checked optical-registration/annotation workflow is a distinct route with its own validation burden.

The user intends to submit the prepared maintainer request; actual submission has not been confirmed. We sent no message. All candidate parents remain reserved and unscored, the forward seed42 baseline and native contrasting reference remain unchanged, and there is no accuracy, transcription, translation or model-promotion claim.

The cloud workspace became ready and all 19,026 phase25 retained records passed the startup check before this study. The initial Matplotlib environment probe used a temporary cache because the home directory is read-only; the actual run explicitly placed both Matplotlib and XDG caches under this phase and required no installation. All earlier evidence and tracked repository files remain intact.

No network, credential operation, GPU, optimizer, upload, publication or prize submission was used. **GPU spend remains £0**, with the **£10 total first-pilot cap** unchanged. Configuration persistence and local integrity checks remain distinct from publication, external backup and verified fresh-task restoration.

Full evidence: [test log](run-v1/tests.log), [run receipt](run-v1/RUN_RECEIPT.json), [synthetic measurements](run-v1/SYNTHETIC_DIAGNOSTICS.json), [complete dry plan](run-v1/SPARSE_GEOMETRY_DRY_PLAN.json) and [second-pass verification](SAVED_RESULT_VERIFICATION.json).
