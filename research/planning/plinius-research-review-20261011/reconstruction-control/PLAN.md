# Same-scan reconstruction qualification — frozen before chunk reads

Question: are three published PHerc0139 reconstruction/export volumes locally
consistent in coordinates, and how different are their observed CT arrays?
This is a bounded physical-data qualification, not ink inference or a new model.

Sites are the **first three destination landmarks in recorded order** in the
retained fine78→native113 transform. Convert their XYZ coordinates to ZYX, then
use the containing 128³ chunk at level0 of volumes20250728140407,
20251107132835 and20251107135911. Keep all three sites, including uninformative
or failed sites. No replacement or adaptive extension. Nine public uncompressed
uint8 chunk objects, each2097152 bytes: total18874368bytes (18MiB). Request each
once using inherited proxy/verifiedTLS, no redirects; retain source hashes and
HTTP failures. Metadata is already fetched separately.

All sites must lie strictly outside the publisher's unreliable sibling masking
ranges z1144–1310 and z≥20448; use whole-chunk bounds. Compare all three volume
pairs at all three sites (nine comparisons). Report full cube and common-nonzero
support separately. Zero intensity does not prove absent tissue or missing data.
Record each cube's nonzero and255 fractions, value quantiles and standard
deviation. Report Pearson and Spearman on both supports (undefined when variance
or support is insufficient), mean absolute uint8 difference, binary nonzero-mask
agreement, and gradient correlation on edges with both endpoints nonzero in both
volumes. Common support can itself select easier material; never hide full-cube
results.

Estimate translation with a fixed separable Hann-tapered, demeaned3D phase
correlation on full cubes. Report the full periodic-grid integer peak as the
shift applied to the second array to align to the first. Also report the strongest
peak outside the ±2cube around it and their ratio; ties/flat inputs are ambiguous.
This is a gross coordinate diagnostic, not a subvoxel residual estimate, posterior
uncertainty or sheet identification. Synthetic rolled arrays must verify sign and
axis order before live analysis. No tuning, subpixel search, intensity fitting,
registration fitting to ink maps, selected frequency bands or model predictions.

Save fixed0–255 orthogonal midplane slices for every volume/site, with neither
contrast fitting nor cherry-picked planes. Assess whether apparent numerical
agreement is dominated by mask/blankness; show all panels and scores. Different
uint8 export windows, phase retrieval and possible detector-distortion handling
confound a pure Paganin comparison. Volumes come from the same acquisition and
share noise; agreement is not independent physical/ink evidence.

Stop after nine comparisons. Successful object checks and interpretable local
structure make this a usable *supporting control at these sites*, not an approved
training/evaluation benchmark. A nonzero/ambiguous translation is retained and
blocks treating the corresponding site as an exact pair. Do not fix coordinates
or fetch neighboring chunks in response. No GPU, paid resources, training or
active study52 change; preserve its frozen review gate and do not inspect partial
shifted outputs. This study runs with one numerical thread beside that worker.
