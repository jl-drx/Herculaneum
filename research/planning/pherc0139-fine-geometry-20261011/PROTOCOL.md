# Fixed-cohort fine geometry and teacher context qualification

11 October 2026. Frozen before new coordinate acquisition/analysis. Keep the
previous256-region intake unchanged:192 proposed training cores,32 w024 alternate
reference reservations and32 w053 native blank reservations. No replacement
regions and no new CT, annotation or prediction pixels in this step.

Acquire exactly the12 fine XYZ TIFFs in the previous finite plan, at most
164,736,024 bytes. Conditional GET against its ETags, exact lengths and SHA-256
receipts. Read only those coordinates, retained native coordinates, metadata
and source code. No neural inference or paid compute. Use one numerical CPU
thread while the unchanged native-learning study runs.

Validate common normalized-mesh ancestry and declared mesh-transform/render
contracts. Implement the source renderer's canvas origin, mesh centre,128-pixel
render-tile phase, FP32 arithmetic, complete bilinear and normal stencils and
source_group scaling. Fine78 uses group0; w024 fine59 uses source_group1,
render_scale0.5 and base-coordinate scaling0.5. Normals remain unit vectors in
the sampled source-level coordinates. Convert points/rays to full-resolution
source XYZ before applying the fine-to-native affine chain.

For every fixed128-square core, map every native XY location into the fine
render by composing native render-to-mesh coordinates, the common-source mesh
resampling convention qF=(qN+0.5)*NF/NN-0.5, and the inverse fine canvas mapping.
Sample the fine rendered geometry bilinearly at this continuous query using
the four integer render neighbors. Require complete source coordinate/normal
support. Record centre-surface residual vectors, normal angles and the naive
full-raster resize alternative as a diagnostic. No fitted translation or target
values. A geometric screen requires complete core support, maximum position
residual<=1 native voxel and maximum normal angle<=10degrees. These are source
parameterization gates, not CT registration or proven supervision accuracy.

Use the pinned v42/v43 teacher source contracts and catalog depth/tile settings
to enumerate every256-square inference tile contributing to any fine integer
sample used by the bilinear transfer, including the special last-border tile.
Retain full tiles even where a blending edge could have zero weight: this is a
conservative complete spatial-input envelope, not a minimal receptive field.
Record reverse-depth handling; it does not change the selected source slab.
Require the complete union's geometry and source-volume extent. Enclose all
source depth offsets plus trilinear neighbors, transform bounds into native XYZ,
then reject possible cross-role overlap. Reject invalid or ambiguous stencils;
do not use zeros, repairs or clamping as valid source support.

Training support must clear the268 older reserved native contexts and all64
new reference native contexts. It must also clear every qualified w024 teacher
input envelope; w024 evaluation envelopes must clear the older reserved native
contexts and w053 native contexts. Report any w024 reference-envelope overlaps
instead of pretending those references are independent observations. Current
w035/w039/w044 native contexts remain development references after the ongoing
study. Upstream teacher training ancestry and binary provenance are separate
limitations from spatial input disjointness.

Freeze a proposed CT-only pilot to selection-order0,16,32,48 in each of the
three training windings. Geometry failures are retained without replacements.
For supported pilot regions, produce explicit full28-native and full109-fine
group2 chunk requests with byte ceilings; group2 means XY-pooled fine CT with
unchanged depth sampling. Include a small fixed spatial halo around the mapped
core for the later local correspondence test. This step does not execute those
requests or relax the earlier failed w040 pointwise-transfer screens.

Verification: group0 regression to the previously audited renderer, independent
scalar group0/group1 mapping and plane/ray checks, teacher-grid boundary/coverage
tests, reproducible physical exclusions, and a relocated offline replay. Pin
all inputs, code and results. Preserve any implementation correction and its
effect on results. A pass admits geometry for a bounded CT check only; actual
local CT correspondence, source masks and transferred supervision remain gates.
