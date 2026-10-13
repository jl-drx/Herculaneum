# Proposed bounded grid-origin comparison

Prepared after review of the completed PHerc343 result. No inference launched.
This is an exposed positive-control diagnostic, not model selection or a fresh
scientific validation set. The original four-model result is fixed and retained.

Question: when the CT source coordinates stay fixed, how much of the seed42
response is sensitive to the inference grid, and does the modulation move with
that grid? A near-line association alone cannot answer that question.

- Model: the same qualified seed42 step075000 checkpoint, CPU fp32, original
  17-plane reversed depth26..42, robust normalization, occupancy, 128-pixel patch,
  stride64 and floored Hann weights. No training or depth change.
- Fixed native region: rows768:1280, columns2816:3328, centred on the original
  debugging point `(1024,3072)`. It is within the already inspected field. Retain
  source CT and the valid mask; never create artificial cropped-volume padding.
- Offsets `(row,column)` in native pixels: `(0,0)`, `(0,32)`, `(32,0)`, `(32,32)`.
  Record the actual global tile origins in advance. Read all complete128px input
  patches whose contribution intersects the region; the necessary context is up
  to127px on each side. Preserve any normalization statistics/occupancy inputs
  whose support exceeds an individual patch. Do not recompute them from the ROI.
- Baseline: offset0 must reproduce the saved seed42 plain region. Establish
  exactly which global batching and occupancy conventions affect its arithmetic.
  Keep full patch outputs until their original weighted contributions have been
  summed and converted to uint8. No altered batch layout or precision is admitted
  merely to make the ROI run smaller.
- Report: every aligned response, absolute change and correlation; rank-support
  Dice at fixed5/10/20%; all symmetric within0/2/4/8px fractions; disagreement
  binned by both source coordinates and the declared grid phases. Use the same
  fixed display transforms; include invalid support and context boundaries.

Stopping rule: stop before the offset comparison if the zero-origin baseline
fails exact uint8 equality, an input/source hash changes, CT is incomplete, or
the original preprocessing cannot be reproduced locally. Otherwise stop after
the four fixed origins; do not search more offsets, checkpoints or regions.

Decision relevance: a verified zero-offset baseline followed by changing
high-response support establishes operational grid sensitivity at fixed CT.
If a spatial modulation translates with the imposed grid, grid locking becomes
a supported mechanism to investigate; if that tracking is absent, do not infer
grid locking from output differences alone. Stable source-coordinate structure
across origins warrants reporting a stability envelope, not an accuracy gain.
If this bounded ROI gives an uninformative result, record it and proceed with the
fixed target-transfer design rather than extending the search on this reference.

No result here selects the best-looking offset or automatically replaces author
ownership with averaging. A proposed deployment change would require a separate
prospective evaluation. PHerc0800 geometry/exposure and workload preparation can
continue independently; a full31-depth, two-direction sweep still needs explicit
sizing. GPU work remains deferred under the existing £10 total cap and readiness
requirements.
