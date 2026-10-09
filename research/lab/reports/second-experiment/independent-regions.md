# Prediction-independent region controls

Three disjoint contexts were selected with a frozen geometry/data-only rule and scored with both verified `ink_9um` checkpoints. Two central review areas remain below the fixed 0.5 score midpoint. The third has elevated shared scores concentrated close to the geometry boundary; these panels support no coherent letter reading.

**These are prediction-independent selections on a previously scored field, not unseen or held-out validation regions.** The full PHerc0800 segment was already scored in the first experiment, and a reviewer had seen its panels. The reproducible selection code did not read detector maps. Disjoint render coordinates do not make these independent observations of model generalization: they share one scroll/segment, source acquisition and model training corpus.

## Selection frozen before new inference

The [frozen selection plan](independent-selection-plan.json) was saved at `2026-10-08T06:18:53.361870Z`, SHA-256 `32741ba9d330940f077129e07e1d6e33f844824fa9d1b589de15592643ff01c2`. [Selection code](select_independent_regions.py) and the plan record input hashes, all ranked candidates and exact rules.

- Scan 256 × 256 contexts at source origins on a 64-pixel grid, from `(0,0)` through `(704,704)`.
- Reject any positive-area overlap with the initial pilot `[256,256,256,256]` or selected hotspot context `[512,448,256,256]`; touching rectangle edges are permitted.
- Require at least 50% conservative geometry-valid pixels and at least 95% of those pixels with any nonzero raw data in selected depth layers `7:24`. No contrast or intensity ranking enters selection.
- Rank by geometry-valid fraction descending, observed-data fraction descending, then y/x ascending. Greedily retain three disjoint contexts.

There were 21 eligible candidates. All three selections met the rule without new acquisition or changing the rule. The third is only 66.67% geometry-valid across its complete context and retains less central coverage; that is a coverage limitation, not a negative ink label. [Overview](independent-selection-overview.png) shows the selected and excluded boxes.

Inputs: full retained uint8 surface SHA-256 `47e18003d1d3a0118cdd9ade63d9b21f2a8d17ba7d31dcfbc872b50b7417a570`; conservative geometry-mask SHA-256 `a5ec3b003135871f517528f789cf62f112a79b4ac99f5e5324bb4162faea41f8`. Each derived crop/mask/provenance is retained under `data/experiments/pherc0800-independent-01` through `-03` and carries the frozen plan hash.

## Fixed processing and results

All contexts used forward depth layers `7:24`, exact embedded 17 × 128 × 128 patches, stride 64, unmodified upstream robust-MAD normalization, Hann blending and CPU FP32. Both final seed checkpoints passed pinned checksum/size, `weights_only=True` loading and strict state matching. No runner, setup, dependency or root documentation changes were made. [Run records](independent-run-records.json), per-context metadata/logs and float32 score/mask outputs preserve commands and provenance.

Summaries below cover only the **central 128 × 128 area intersected with geometry validity, both evaluated masks, and any observed raw selected-depth data**. Values are unscaled model scores. The fixed **0.5 midpoint** is a behavior diagnostic; no target labels calibrate it, and it supplies no accuracy or ink-certainty estimate.

| Context / source origin | Complete-context geometry valid | Central common pixels | Seed42 range / mean | Seed43 range / mean | Both scores >0.5 |
| --- | ---: | ---: | --- | --- | ---: |
| independent-01 `(256,512)` | 100.00% | 16,384 | 0.251–0.415 / 0.303 | 0.227–0.323 / 0.267 | 0 / 16,384 |
| independent-02 `(512,192)` | 98.93% | 16,384 | 0.237–0.399 / 0.274 | 0.199–0.349 / 0.261 | 0 / 16,384 |
| independent-03 `(0,320)` | 66.67% | 12,796 | 0.241–0.639 / 0.329 | 0.223–0.709 / 0.446 | 830 / 12,796 (6.49%) |

The third central area has 13,632 geometry-valid pixels; requiring observed data and both evaluated masks leaves 12,796. All remaining values are finite. The two seeds disagree substantially there: mean absolute score difference 0.132 and maximum 0.437. Agreement elsewhere or agreement on the 830 elevated pixels is not independent authentication.

[Machine-readable summaries](independent-regions-summary.json) retain full distributions, coverage and correlations. [Review 01](independent-01-review.png), [review 02](independent-02-review.png) and [review 03](independent-03-review.png) use the same source layers 11/15/19, shared raw CT display limits 0–155, and identical display-only `clip((p−0.25)/0.5,0,1)` scaling. Raw summaries never use this display correction.

## Evidence and consistency checks

The observed views show substantial longitudinal/transverse texture, irregular gaps and bright points. The first two central model panels are weak and do not resolve a coherent stroke sequence. The third shows diffuse responses adjoining the stepped valid-geometry boundary, with the two seeds differing in location and extent. No letter identity, language or text is supplied by these observations.

A **post-freeze geometry diagnostic**, recorded separately in [the boundary diagnostic](independent-boundary-diagnostic.json), found every one of the 830 joint-above-midpoint central pixels in context 03 within **30 source pixels** of invalid geometry; median distance 14 pixels. This is approximately at most 0.259 mm using the nominal planar 8.64 µm sampling. All would be excluded by a 64-pixel geometry margin. This association motivates a boundary/normalization diagnostic; it does not prove either an artifact or genuine ink, and the boundary analysis was not used to select contexts or change the frozen summaries.

After selections were frozen and all new runs completed, original full-field maps were loaded for a consistency comparison. **Both seeds' scores match the corresponding full-field central scores bit for bit in all three contexts**, with maximum absolute difference 0 over the common domains. This confirms the aligned cropping/tiling computation, not model accuracy.

All selected origins are multiples of 64, so tile phase aligns with the full-field regular grid. None of these central areas is reached by the special full-field edge tile at 852. A crop origin off that grid would change patch composition and robust normalization; changes then could reflect tiling rather than physical evidence. Reviewing the central 128 pixels avoids the crop's missing outside-context contribution, while retaining dependence on the same field and fixed patchwise normalization.

## Next diagnostic

Interpret the root task's labeled known-ink control alongside these unlabeled behaviors. For the third context, compare predefined depth/orientation and geometry-margin controls at fixed coordinates and common valid pixels before investigating another rendering. Keep the current selection and results frozen; any boundary-treatment or normalization change belongs in a separately documented experiment. These results neither establish recovered letters nor establish absence of ink in the quieter contexts.
