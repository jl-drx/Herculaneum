# Assessment of Sam's follow-up and the completed PHerc343 reproduction

10 October 2026. The supplied analysis is supported. Both primary analyses and
their verification scripts replayed successfully: **25,070 numeric values match
exactly**, including all supports, ranks, grid bins and amplitude strata. All
40 supplied files are retained unchanged; 28 submitted checksum entries passed.
The archive identity and per-file hashes are in `SOURCE_MANIFEST.json`.

## Assessment of the three claims

- **Strong-response locations change with padding.** The reported top-10% pooled
  within-eight-pixel fractions are 94.6% for dense_native, 76.3% for seed42 and
  76.1% for seed43. We independently verified all six whole-crop pair comparisons
  with KD trees, closing the explicitly disclosed gap in the supplied whole-crop
  EDT verification. The existing 324 box/rank/pair distance checks also pass.
  These fractions pool both directions, weighted by selected-pixel counts; they
  are not feature tracking, displacement estimates or accuracy scores. The
  supplemental radii were declared after initial results and remain descriptive.
- **Disagreement is broadly distributed.** The exact finite tile-start schedules,
  appended terminal starts, full crop, interior exclusion, union of boxes and all
  18 boxes reproduce. Near-line area is 26.29%; it accounts for 26.92%, 27.27% and
  27.36% of absolute disagreement. This argues against a narrow seam-only remedy.
  It does not exclude an effect of tiling across the whole field or establish its
  causal mechanism. Hann blending makes a tile start different from an ownership
  switch. The CT-dependent occupancy pattern was not independently reconstructed
  by Sam's analysis; potential start lines are the correctly stated scope.
- **The completion checker is useful and passes real outputs.** All 21 supplied
  synthetic cases pass. We then ran it unchanged on the actual eight final maps,
  with the ownership-source hash taken from the previous workspace manifest.
  File/pixel hashes, dimensions, saved-map preservation, model identities,
  Hecate metadata and all four exact ownership merges pass. Hecate uses the
  36/180 crop and nearest-neighbour ownership mask on its own grid. The checker's
  constant `original_composition_still_required: true` reports its limited scope;
  the original composition has actually completed and is verified separately.

The supplied `infer.py` is byte-identical to the pinned runtime source. Copies of
the runner, preparation script and qualification JSON have one extra trailing
newline; their contents otherwise match. No source substitution was needed.
All five submitted figures were inspected, including all 18 boxes. Some long
column labels touch adjacent text, but the numerical tables and region identities
are intact; the original figures are preserved.

## Completed four-model result

The unchanged d=+1, reverse-side CPU pipeline finished at **14:51:20 BST**. All
eight maps and the original composition exited successfully. The separate
integrity check verified nine frozen files, all map receipts, four composition
outputs and all 60 saved checkpoint prefixes against the finished padded Hecate
map, bit-for-bit. The observer did not resume or alter inference: after the cloud
replacement, padded Hecate ran from its beginning to completion.

This review independently reconstructs the final ink component from the saved
merges using the original float32 operation order, transforms, equal z-score
weights and fixed stretch. Its decoded pixels match exactly. Actual CT layer34
again matches the published decoded hash; the independently stretched CT
component also matches exactly. Orientation, area, overlay and normalisation
metadata match the reference. The single surface/CT-basename warning is not a
shape or fixed-statistics mismatch.

| Fixed published-panel comparison | Result |
| --- | ---: |
| Valid pixels | 8,887,441 |
| RGB mean absolute difference, 0–255 scale | 0.0180978 |
| Yellow-display mask intersection / union | 1,122,372 / 1,124,302 |
| Yellow-display mask IoU | 99.8283% |
| Mean RGB difference on reference-yellow pixels | 0.109691 |
| Exact encoded PNG equality | No |
| Candidate-box RGB MAE range, all 18 | 0.015870–0.073091 |
| Candidate-box yellow-display IoU range | 99.7405%–99.9597% |
| Largest RGB difference within any candidate box | 5 |

The global maximum RGB difference of166 occurs at native `(y=832,x=3410)`, outside
the candidate boxes, on the recorded cyan row annotation. The original algorithm
derives row paths partly from responses; the input annotation file is identical
but derived paths differ slightly. A post-hoc check using both saved row paths
and the original rasterizer places the maximum inside their union. Outside that
recorded row coverage, the maximum is6. This diagnosis does not replace or amend
the original full-panel metrics.

We inspected the full display and every candidate box with unchanged contrast,
plus fixed ×16 difference views. The highlighted structures recur throughout the
18-box set; there is no visible missing/extra major structure between these two
composites at the reviewed scales. This supports **published-image reproduction
and qualification of the acquisition-domain positive control** under the existing
protocol. No numerical success cutoff was invented after seeing the output.

Hecate's aligned phases, separately linearly resized to native resolution, have
whole-crop correlation0.940887, uint8-equivalent MAE6.63199 and top-10% Dice0.82000.
Region correlations range0.6949–0.9712. This extension is newly declared and
descriptive. Resampling and different model calibration preclude treating these
as a direct accuracy ranking against the Villa values. Hecate's broad response
patterns recur, but response intensity/connectivity still changes with phase.

## Decision and next work

Retain the unchanged four-model recipe. The numerical and visual correspondence
is sufficient to move forward with a prospectively specified PHerc0800 transfer;
it is not evidence for reweighting models, choosing a prettier phase or accepting
every response as ink. CPU fp32, the rebuilt dense_native container and absent
raw author prediction maps remain explicit limitations. No AP, calibrated
probability, authenticated character, transcription or translation is established.
These boxes were already inspected and overlap; none is a fresh test region.

Accept one bounded grid-origin follow-up as an uncertainty diagnostic, alongside
target-transfer preparation. Use the existing debugging coordinate, one fixed
region, seed42 at75k, unchanged depth/preprocessing and four predetermined origins.
Keep enough actual CT context to reproduce all contributing patches. First
require a zero-offset reconstruction of the saved baseline; otherwise stop and
diagnose the harness. Report every offset, never promote an offset or averaged
map using this exposed reference. See `NEXT_GRID_COMPARISON.md` for the proposed
design and what would change the decision. No new inference was run in this audit.

## Evidence and replay

- `REPLAY.json`: path-only adaptations, source identities and exact numeric replay.
- `WHOLE_CROP_KDTREE.json`: independent whole-crop distance verification.
- `SYNTHETIC_REPLAY.json`, `ACTUAL_FINAL_MAP_CHECK.json`: synthetic and actual gates.
- `completed-review/COMPLETION_REVIEW.json` and four PNGs: all-region assessment.
- `ANNOTATION_DIFFERENCE.json`: maximum-difference diagnosis, explicitly post-hoc.
- [Completed outputs and portable bundle](../../in-progress/pherc343-reproduction/completion-20261010/README.md).

`replay.py` accepts `--lab`, `--repo`, `--python` and a fresh `--out`; use the
existing historical CPU environment with NumPy, SciPy, Pillow and Matplotlib.
`whole_crop_kdtree.py` needs NumPy/SciPy/Pillow. The final-map checker and
`review_completion.py` use the qualified PHerc343 environment (NumPy, Pillow,
OpenCV and Zarr). The latter also requires the existing rendered CT, which is
excluded from the small completed-results bundle. All scripts are read-only
with respect to inference inputs and write new analysis outputs separately.

Completed results and this review are retained as an append-only supplement to
the earlier phase46 working snapshot. They do not rewrite its original manifest
or the cumulative sealed phase45 archive/register. GPU spend remains £0.
