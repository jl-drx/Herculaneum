# Fourth experiment: geometry-selected PHerc0800 screen

This bounded private study searches for additional source-linked evidence on eligible candidate scan PHerc0800 / `20250521135224`, at native 8.640 µm sampling. The immediate outcome is an observed image and reproducible detector comparison, with any reading hypothesis left for later source checks and specialist review.

## Fixed selection and execution

Use only retained full XYZ maps and geometry-validity masks for the five larger segments when selecting the crop. Select a chunk-aligned, fully valid 384×384 region; exclude all previously acquired pixels on the same source. Require the exact minimum nearest-base-coordinate distance to previously acquired geometry-valid PHerc0800 coverage to exceed 40 native voxels. Among qualifying crops maximize that minimum distance, then break ties by segment ID, source Y and source X ascending. The selector must retain all candidates and exclusions. If none qualify, stop this plan without silently relaxing it.

The 40-voxel rule describes base-coordinate separation. It does not reconstruct renderer normal/depth/interpolation support, identify a new papyrus layer, or establish statistically independent data. Report physical independence as unresolved. Check finite geometry and neighboring-coordinate discontinuities; geometry-valid pixels do not establish correctly placed papyrus.

Acquire only the selected nine uncompressed `[31,128,128]` uint8 chunks (4,571,136 bytes maximum source payload), after confirming metadata against the frozen plan. Retain original payloads and SHA-256, source URLs/ETags, provenance, source coordinates and masks. Metadata and error responses count toward a separate 6 MiB total response-body cap. There are no automatic retries. Reconstruct the declared zero fill only after an explicit S3 `NoSuchKey` response, mark that support unobserved, and exclude it from analysis.

`inference-plan.json` freezes the unchanged model adapter, checkpoints, dependency lock, four CPU threads and conditions before acquisition: both seeds on forward `7:24`; seed42 on forward `5:22`, forward `9:26`, and reversed `7:24`. Maximum scheduled inference is 125 tiles; no training, fitted preprocessing, GPU or paid provider resource is involved. Keep any disappointing result rather than selecting a substitute region from the predictions.

## Fixed analysis and interpretation

Retain full raw maps and show the complete field. Analyze a fixed 64-pixel crop margin with common geometry, observed source, evaluated support and nonzero selected-depth profiles across all conditions. Also report a strict domain excluding the complete support of every scored tile containing invalid, unobserved or all-zero-profile input. These masks describe support, not ink/background truth.

Report raw score distributions, seed agreement, components and depth sensitivity. The midpoint 0.5 is a diagnostic for models trained with smoothed targets, not calibrated ink confidence. Both checkpoints share training data. There are no target labels and no accuracy estimate. Source images use fixed 0–255 grayscale and score maps use fixed 0.25–0.75 limits, with max projections explicitly labeled as projections.

The full field spans approximately 3.32 mm and the fixed interior 2.21 mm. Any candidate requires broader spatial context and native source/geometry validation. Keep transcription and translation null unless reproducible source-supported strokes warrant specialist review. No external publication or prize submission is authorized by this experiment.

Preserve the source, code, plans, raw outputs and evidence privately in the lab. Git remains for reusable code/configuration, not scans, weights or private novel findings. R2 access is verified locally but its credentials are not installed in this cloud process. No R2 publication or remote backup is implied by local result creation.
