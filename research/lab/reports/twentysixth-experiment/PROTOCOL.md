# Offline mechanics for a new auditable reference workflow

The user is handling the maintainer request and authorized useful work in parallel. This study prepares mechanics for a newly declared rendering/registration workflow. It does not reconstruct the missing historical Frag6 recipe or admit a benchmark. Root owns implementation and review; no independent-agent review is claimed.

Preserve all phase1–25 records and the checkout at49fabf3e890d5b880f391b27d581f1e7a1ebdbcc. The phase25 readiness check passed after workspace startup:19,026records/6,501,709,440bytes. New outputs belong only here. No packages or services are needed. Use existing Python3.12.14/NumPy2.5.3, offline CPU; set MPLCONFIGDIR inside this phase for plotting. No network, secrets, new candidate image/CT/label pixels, model calls, gradients or GPU resources.

## Sampling contract

Implement an in-memory chunk-source interface and point-batch trilinear sampler. Declare native voxel-center XYZ coordinates, array/chunk ZYX ordering, positive XYZ voxel spacing and exact volume shape/dtype. No half-voxel shift, implicit axis permutation, quantization, intensity normalization or boundary clamping. Output FP64 samples and an explicit valid mask. Out-of-bounds/nonfinite coordinates, missing chunks or nonfinite contributing voxels are invalid and remain NaN; a valid zero intensity remains valid. Only strictly positive-weight interpolation corners contribute. Exact outer integer voxel centers are valid; zero-weight missing neighbors must not invalidate them.

Plan all required unique chunks before invoking the loader. Validate exact edge-chunk dimensions and dtype. Cap each batch at100,000points,128chunks and64MiB of uncompressed planned source arrays; reject an exceeded budget before any load. Cache each chunk once per batch, including a missing response; no retries or substitutions. These are application memory/layout estimates, not compressed S3 bytes or an implemented S3 transport.

Construct ray coordinates from explicit signed offsets in micrometers. Require the caller to name the vector convention: physical-unit XYZ vectors, or unit voxel-coordinate direction vectors whose physical length is computed with voxel spacing. Neither convention infers anatomical inward/outward normals. Under anisotropy a surface-normal covector may need a separate inverse-transpose transform; do not silently treat one as a direction vector. Mapped vectors must be finite/unit within1e-5. Explicitly invalid records are preserved as invalid, never replaced.

## Registration contract

Implement a 2D optical-XY to surface-XY affine baseline using finite, uniquely named landmarks. Use disjoint named fitting and check sets; require at least three non-collinear fitting points and at least one check point. Report fit and held-out Euclidean residuals separately in target pixels. Reject duplicate/overlapping/unknown IDs and rank-deficient or severely ill-conditioned normalized fitting designs. This is not an image-registration algorithm, nonlinear warp, measured Frag6 residual or proof that affine registration will suffice.

## Tests fixed before implementation results

Use analytic multilinear volumes with known values to expose XYZ/ZYX mistakes and interpolation error; compare against a separate direct scalar interpolation oracle, tolerance1e-10. Include exact and fractional chunk boundaries, integer outer boundaries, anisotropic physical displacement, reversed offset ordering, invalid geometry, valid zero intensities, missing data and nonfinite positive-versus-zero-weight neighbors. Require identical values/validity across two chunk partitions, and exact loader-call deduplication. Resource-limit and shape/dtype failures must be demonstrated. Explicitly test shifted-axis and half-voxel negative controls against the analytic truth.

For registration use a known affine map, held-out errors injected only into the check set, and collinear/overlap/duplicate controls. Fit residuals must not conceal injected check residuals. Synthetic success tests implementation mechanics only.

Then run one offline planning exercise over ALL729 retained phase21 sparse PPM records, preserving171all-zero sentinels and558mapped records. Use the already qualified little-endian geometry bytes and its exact SHA, not new PPM acquisition. For this hypothetical planner probe only, offsets are[-3.24,0,+3.24]micrometers, spacing3.24micrometers isotropic, candidate boundsZYX[8855,2260,6300], hypothetical chunks64³ anduint16. These assumptions are explicitly new diagnostics, not authenticated historical rendering or actual Zarr metadata. Generate the complete chunk schedule and uncompressed upper bound without invoking a source loader. No site selection, replacement or source-volume identification by bounds.

Retain source/test freezes, failures, runtime/count receipts and a static synthetic diagnostic plot. Maximum study runtime120seconds per offline command, artifacts16MiB including final integrity manifest. Do not tune tolerances after results. Complete root report, next work recommendation, registerv22, integrity check and additive startup save. Current candidate reservation and £10 TOTAL firstpilot cap remain unchanged; spend£0. The user intends to submit the request; do not record it as actually sent without confirmation.
