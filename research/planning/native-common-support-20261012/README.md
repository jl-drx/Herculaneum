# Common-support native model comparison

**Follow-up:** the [executed input packet](../../in-progress/native-common-support-20261012/README.md)
preserves the original union-panel admission failure and a separately frozen
supervision-supported development panel. Inference remains unrun. The text below
is the original metadata-only preparation record, not the current run status.

12 October 2026. This intake prepares the next strong-model development
comparison. It runs no model and opens no new CT or annotation array payload.
The current native0139 experiment is unchanged. The next comparison is useful
regardless of whether that small learner passes its own advancement gate.

## What is resolved

The published Reader suite provides a concrete two-parent development comparison:
all three PHerc0841 segments and PHerc0009B segment 20250919125754, the four
segments used for its reported human-reference endpoints. Selection here uses
that complete declared set, not whichever segment gives attractive model maps.
The transferred references are on the fine-render canvas; the intended deployed
inputs are the actual native scans, not synthetically degraded fine CT.

| Parent / segment | Native volume D×H×W | Native spacing | Fine-reference Y×X |
|---|---|---|---|
| 0841 / 20260220213127 | 28×4220×4760 | 9.366 µm | 16460×18560 |
| 0841 / 20260220214732 | 28×4640×4720 | 9.366 µm | 18080×18400 |
| 0841 / 20260221022814 | 28×3760×4900 | 9.366 µm | 14660×19100 |
| 0009B / 20250919125754 | 31×7040×5700 | 8.64 µm | 25360×20520 |

The source is hash-bound in [METADATA_RESULTS.json](METADATA_RESULTS.json).
All twelve ink/supervision/validation arrays have the expected YX axes, uint8
type and exact fine-render shape. They use sharded Zarr v3, while the render
volumes use Zarr v2. The initial v2 label-header requests returned 24 HTTP 404s;
actual directory listings identified v3 and all 24 corrected header requests
succeeded. There were 80 requests, 56 successful metadata bodies totalling
106,897 bytes, and zero array bodies. An absent v2 header was not a missing label
package. Catalogue transform pairs are algebraic inverses; this does not verify
local physical registration.

## What stays unresolved, without repeating old investigations

Studies [11](../../lab/reports/eleventh-experiment/reader-preparation/READER_DESIGN.md)
and [22](../../lab/reports/twentysecond-experiment/benchmark-metadata/v2/RESULTS.md)
already established the important limits: transferred and prediction-reviewed
annotations, reconstructed letter forms in part of 0841, unknown broader model
ancestry, missing published mapping-validity and exact transfer provenance.
Those findings are inherited. This refresh adds array-level contracts and links
them to the newer Reader comparison; it does not discover a pristine reference.
No further search for the same missing provenance is part of this intake.

Recorded source-canvas offsets are not new corrections to apply to the output
labels. The earlier transfer-code audit explains why doing so could apply the
source correction twice. Do not fit offsets to detector predictions. The
published Reader evaluator resizes the fine-canvas masks onto the native canvas
using area interpolation and a 128/255 cutoff. That is a reproducible **supplied
reference agreement** target, not proof of exact physical correspondence.

PHerc0841 was also used for upstream checkpoint/operating-point decisions;
PHerc0009B has other Reader training segments and upstream exposure. Both are
development material under the approved broader programme. The new prospective
confirmation families remain PHerc0175 A/B and PHerc1203. Their payloads are
unopened. No current native0139 held-out output selects these four segments.

## Next bounded experiment

1. Decode supervision and validation separately, retaining their observed
   overlap and value encodings. Define the published comparison support as their
   union; do not turn unlabelled pixels or absent mapping-validity into negatives.
   Keep ink values unavailable to crop selection. Reuse the qualified indexed
   shard reader after checking the actual level's shape/codec contract.
2. On each segment select six nonoverlapping 128² cores from a fixed 128-pixel
   grid, with full model context and at least 4,096 supported pixels. Use a
   deterministic spatially spread selection from participation masks only.
   Preserve all four segments; if six eligible cores do not exist, report the
   deficiency before inference rather than choosing another segment. Freeze the
   exact algorithm, IDs and chunk union at that stage. Twenty-four cores are a
   finite implementation/development pilot, not a powered final accuracy study.
3. Fetch only the union of contributing native chunks. Retain all 28/31 source
   planes. Qualify each model's full-render tile origin and context using the
   existing exact-context tooling; a cropped tensor with a reset grid is a
   different input. No depth or origin sweep and no response-selected crop.
4. Compare Reader v2, dense-native, Hecate and the two released seed controls on
   exactly those score pixels. Respect the hybrid 17-plane/MAD contract and
   Hecate's 16-plane, 9.6 µm XYZ contract. Fix direction, tiling, blending, precision
   and output representation before running. Do not silently use the fine CT or
   another model's preprocessing. Keep each model's result, not just the winner.
5. Report per-segment and equal-parent AP/AUC, calibration, annotation-negative
   response and declared model exposure. Repeat-use 0139 results can be an
   additional retention endpoint once sealed, never an untouched test. A claim
   of better physical ink needs separately supported references and fresh tissue.

No model has passed this comparison yet. The first stop is input/participation
failure, a single-class pooled endpoint, or incompatible sampling—not a failure
of coarse CT. Successful execution establishes a strong baseline and measures
whether rankings depend on the parent/reference. Small, label-sensitive or
parent-opposed differences do not justify selecting a universal winner.

This comparison precedes the already specified matched-coverage training pair;
it is not another small adapter continuation. Freeze training populations,
teacher tier and equal original-source exposure separately. Meaningful training
exposure will require a measured GPU budget. The £10 all-in pilot cap and explicit
approval requirement remain; neither this document nor a positive CPU result
launches paid compute.

## Reproduction

`metadata.zip` preserves every received metadata body and receipt, including the
404 records, under its checksum manifest. `SOURCE_ARCHIVE.json` pins the archive.
No credential, CT array, label array, model weight or current-run result is in it.
Run `summarize_metadata.py --packet <this directory> --metadata <extracted metadata>
--output <new file>` with the retained NumPy environment to rebuild the admission
report offline. No network or inference occurs during summarization.
