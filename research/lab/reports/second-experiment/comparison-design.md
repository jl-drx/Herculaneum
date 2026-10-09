# Audit of the known-label and PHerc0800 comparison

Prepared 8 October 2026. This is a design audit, not an ink result. The proposed experiment can establish whether the retained detector follows a released annotation aligned to genuine scan data in a region overlapping the stated training corpus, and describe its behavior on additional PHerc0800 contexts. It cannot measure PHerc0800 ink accuracy or validate a letter reading.

## Fixed comparison and units

Use the settings in `PREDECLARED_COMPARISON.md`: both pinned final-step checkpoints, CPU float32, complete 17×128×128 model patches, upstream robust-MAD normalization over each whole patch, stride 64, and the same nonperiodic Hann blend map. Do not tune depth, thresholds, checkpoints, source contrast or crop positions after viewing the new predictions.

Retain a 256×256 source context for each case. Analyze local rows and columns **64:192**, the central 128×128 area. Show its boundary explicitly in context panels. This gives the same model context and pixel dimensions for the comparison, while retaining a 64-pixel margin from the crop boundary. The numerical region is not the full 256×256 display.

Define the primary analysis mask explicitly and retain it:

```
interior
AND caller-verified geometry/source-valid mask
AND seed42 evaluated mask
AND seed43 evaluated mask
AND observed nonzero source-data support
```

For the reference, additionally intersect with its documented **supervision/annotation-validity mask**. Specify whether source-data support means any nonzero value in the selected 17 input layers or in all retained layers; use the same stated rule for every case. Selected-layer support is the more direct rule for the actual model input. Report counts removed at every stage and known-positive/known-negative counts after all intersections. Do not substitute an annotation mask for proof of geometry, and do not count empty-tile policy zeros as learned background responses.

The current runner's `interior_score_summary` is only its central rectangular finite-score summary; it does not apply the above combined support/annotation mask. Quantitative comparison must use the retained combined analysis mask rather than copying that convenience summary. Missing, NaN, invalid, or unevaluated scores are excluded, never converted into negative predictions.

## Selected reference, depth and physical scale

The selected reference is **PHerc1667/w013**, released segment `20240304141531-w013_20240304141531_flatboi`, using the 2.399 µm source volume `20251217075048`. The paired aligned-label metadata maps to source XY level 2, shape `[109,10400,19900]`, and source depth slice **13:97**. The selected crop is source XY **[3968,1408,256,256]**, or columns 3968:4224 and rows 1408:1664. Apply the documented rounded mean of successive four-plane groups to these 84 genuine source planes, producing **21 layers at nominal 9.596 µm isotropic sampling**. Preserve the source hashes, grouping rule and label metadata in the [frozen provenance](../../data/experiments/known-ink-control/provenance.frozen.json) and [selection record](../../data/experiments/known-ink-control/selection.json).

The fixed reference model input is pooled layers **2:19**, the centered 17-of-21 window. It corresponds to raw source layers **21:89**, with no depth padding. Aligned flat labels and supervision are at pooled layer **10**. Four-plane means have centers at source indices 14.5, 18.5, …, 94.5; the middle pooled layer therefore has center 54.5. Do not silently identify these means with one of their constituent raw slices.

PHerc0800 retains 31 genuine layers with verified local rendering offsets −15 through +15 and a one-native-voxel step. Its primary model input remains **7:24**, centered offsets −8 through +8 in the documented forward order. These windows have the same number of layers and relative center, but different physical spacings. The PHerc0800 depth order was reproduced from native CT at three local points in the first experiment; that establishes those rendering samples, not equivalence to the reference's trained normal orientation everywhere.

| Quantity | PHerc0800 target | PHerc1667/w013 reference |
| --- | ---: | ---: |
| Nominal model-input sampling | 8.640 µm | 9.596 µm |
| 256-pixel context side | 2.211840 mm | 2.456576 mm |
| 128-pixel analysis side | 1.105920 mm | 1.228288 mm |
| First-to-last center span of 17 layers | 138.240 µm | 153.536 µm |

The target sampling is **9.96% smaller relative to this reference**. The 8.36% figure only describes a different native 9.362 µm reference and must not be reused for w013. These nominal canvas dimensions do not independently measure papyrus area or flattening distortion. Report the scale mismatch and preserve native data in this first comparison. A fitted resampling chosen to make target responses resemble labels would change the experiment. Any later resolution-matched experiment should predeclare and apply its physical XY and Z resampling, interpolation, label handling and antialiasing together.

For future predefined depth controls, reference windows 0:17 and 4:21 correspond to offsets −2 and +2 pooled planes; PHerc0800 controls 5:22 and 9:26 correspond to −2 and +2 native planes. These are matched index offsets, not exactly matched micrometre offsets. Keep all controls at the same source XY coordinates and analyze their fixed windows; do not select the most text-like output. Reverse-depth results are diagnostic sensitivity controls and do not override independently documented rendering order.

## Label encoding and ignore checks

A binary-looking image does not establish that every zero is a known negative. The pinned upstream data path loads **inklabels and a separate supervision mask**. `_flat_sample` reads each independently; `_tensor_sample` passes inklabels to float tensors without automatically dividing by 255. The matching upstream README requires paired ink/supervision assets and describes explicit version matching. Unknown pixels outside supervision must remain ignored.

Before computing any class metric:

1. Retain exact ink and supervision source URLs, retrieval metadata and hashes, annotation version when available, full shape, selected level/Z plane and crop XY coordinates. Confirm that both annotations refer to the same source representation and transform.
2. Record actual unique values and counts for the original ink and supervision arrays. Decode only using the producer's documented scheme. A documented 0/1 scheme permits direct binary checks; a documented 0/255 image requires an explicit equality-based conversion. Arbitrary grayscale or a third/ignore code must be explained, not thresholded automatically.
3. Assert matching label/supervision/source XY shapes and crop registration without a guessed transpose, flip or resize. Preserve the original annotation bytes. Visual overlay should support exact coordinate correspondence.
4. Define known positive as valid supervision AND the documented ink-positive code, and known negative as valid supervision AND the documented ink-negative code. Zero ink outside valid supervision is unknown. Report contradictory codes or ink marked outside supervision as a data-quality observation; exclude them from class metrics.
5. If the annotation supplies positives only, measure coverage of marked positives and their score distribution. Precision, specificity, false-positive rate, IoU and total classification accuracy require documented known negatives and are unavailable in that case.

The selected source arrays have now been checked directly: `inklabels.npy` is uint8 with unique values **0 and 1**, while `supervision-mask.npy` is uint8 with unique value **1** throughout the 256×256 crop. The aligned Boolean `annotation-valid-mask.npy` is all true. The complete context has **26,364 supplied positive labels, 39,172 supplied negative labels, zero unknown labels and zero positive labels outside supervision**. These are whole-context annotation counts; central-128 analysis counts must be computed after intersecting the source and evaluated masks, rather than copying the whole-context counts.

The selected annotation files match these SHA-256 hashes: inklabels `b79711215419a724d61c8a075d8314f942ba886a5ef0ed1c230c8247842b1c61`, raw supervision `8876b9c90ac76ee46dee87c3a62d74aabeb413c986967499ad54694b58669d42`, Boolean annotation validity `72bf42bd9c376d68820d774b6ce353f9fd4f71523379d25ad5aca0af4510e04e`. Supervision establishes the producer's annotation participation semantics, not independent physical proof that every supplied positive is authentic ink or every supplied negative is free of ink.

## Scores, thresholds and allowed measurements

Both embedded checkpoint configs have `bce_label_smoothing=0.5` and unsmoothed Dice. Under the BCE smoothing rule, target **0 maps to 0.25**, and target **1 maps to 0.75**. These are training targets; the combined loss and model generalization do not guarantee that outputs equal a calibrated ink posterior or stay within that interval.

Keep the unquantized float32 sigmoid/blended score arrays as quantitative inputs. The recorded display transform `clip((p−0.25)/0.5,0,1)` is suitable for a common display scale, but it does not calibrate probabilities and must not replace the raw arrays. Apply the same score scale on every panel. Mark all source-image percentile/contrast changes separately and retain an original fixed-intensity view.

Use the predeclared strict comparison **p > 0.5** consistently, including the two-seed conjunction. This is the midpoint of the smoothed BCE targets, not a 50% probability that a pixel contains authentic ink. Preserve its operator (`>` rather than silently switching to `>=`) and report any exact-threshold ties. Do not choose an optimal threshold from this reference crop or the target maps.

For every context, on the common analysis mask, report each seed's pixel count, mean, median, 1st/99th percentiles and maximum, plus the fraction with both scores >0.5. Add mean/median absolute seed difference and, if both maps have nonzero variance, their Pearson correlation. Correlation is undefined for constant maps; represent that as unavailable. The target's conjunction fraction is a model-response fraction, not an ink fraction.

For a reference with documented positives and negatives, report class-specific raw-score distributions and the confusion counts TP, FP, FN and TN at the fixed threshold. Descriptive precision, recall, specificity, IoU or balanced accuracy can be derived from those counts with their denominators disclosed. A zero denominator yields an unavailable value, not an invented zero or perfect score. State the exact estimator if reporting a ranking measure: average precision and trapezoidal precision–recall area are different. Such ranking is descriptive within this selected reference crop. A matched known-negative distribution is useful texture behavior evidence; it does not estimate target false-positive rate across the scan/domain shift.

Call these **current annotation agreement on a training-overlapped region**, with descriptive threshold metrics where justified. Do not call them in-sample accuracy, held-out accuracy, cross-scroll validation, calibration, or authenticated ancient ink. PHerc0800 has no trusted labels for these contexts, so it has no precision/recall, accuracy or transcription metric.

## Dependence, selection and interpretation

The retained metadata from both final checkpoints lists physical segment **1667:w013** and representation **public_2p4_level2_zmean4:pherc1667-w013**. The pinned model card states that the two runs changed only the seed. Thus physical-segment and representation overlap with training are confirmed, and the two seeds share the same corpus. The exact historical training-label bytes and exact pixel-level training supervision remain unproved: the current supplied label objects are dated **18 August 2026**, whereas the pinned model revision is dated **14 August 2026**. Those object/revision dates do not establish that the released labels are identical to, or different from, the annotations used in training. Retain this distinction rather than calling the present metrics in-sample accuracy. The crop remains unsuitable as an independent benchmark even without proof that every selected pixel was supervised in the historical run.

Retain the frozen PHerc0800 geometry/source-coverage selection record before opening the new outputs. Their selection can be prediction-independent while remaining inside the small segment already scored in the first experiment. Label-informed reference crop selection is appropriate for a positive control but does not make a representative benchmark. Keep the earlier model-selected hotspot visibly designated as post hoc.

Crops, overlapping contexts, adjacent layers, alternate renders, the two checkpoints, and individual nearby pixels are not independent biological or archaeological observations. Count physical provenance units, disclose crop overlap and source-area reuse, and avoid pixel-bootstrap confidence intervals, seed-based significance tests, or a claim of n=2 validation from the two models. A handful of contexts in one segment provides descriptive comparison, not population-level inference.

The central crop margin removes crop-border contributions only under the matched local tiling schedule. Equality to the original full-segment scores also requires source XY origin aligned to the full-run stride and preservation of every contributing full-run tile. The 980×980 full runner adds an irregular final tile start **852**; a context whose interior receives such a tile may differ from a locally scheduled context. Verify equality directly before claiming that new crop scores reproduce the first experiment's full-segment map. Identically processed local contexts remain comparable even if this stronger equality is unavailable.

Source texture, fibers, cracks, voids and rendering artifacts must remain visible alongside predictions and the reference annotations. A convincing response on released annotations in the training-overlapped reference establishes detector operation and alignment. An equally strong target response remains a candidate whose image-supported stroke structure and independent specialist review must precede diplomatic transcription, word division or translation.

## Runner and source audit

Read-only comparison of `run_ink_baseline.py` and the separately added `run_ink_reference.py` found AST-identical implementations of `infer_surface`, `hann_blend_map`, `sliding_positions`, `review_interior`, `build_checkpoint_model`, `load_focused_sources` and `score_summary`. Therefore both use the same verified upstream normalizer, checkpoint loading, model calls, score averaging and mask semantics. The reference changes source acceptance from exactly 31 to 17–31 genuine layers, validates the dynamic 17-layer bounds, requires an explicit layer start and updates help/limitations. It introduces no synthetic depth padding or altered inference function.

Audited hashes:

- Baseline adapter: `2cbdb280777e19794fc78857969f9c1d43965da17772f6a1659264e6f5a48a59`.
- Reference adapter: `16bc6e9fb080eb5c2fcc1f20975481881e513b36b7533c1987bdb422340fea9f`.

Evidence inspected:

- [Predeclared comparison](PREDECLARED_COMPARISON.md), [methodology](../../METHODOLOGY.md), [model audit](../model-audit.md) and [first results](../first-experiment/RESULTS.md).
- [Retained pinned model card](../../data/models/ink9um-model-card.md), model revision `7109667e2607db1b90c37c8b09cb876ea7fe7bb1`.
- [Both embedded checkpoint configs](../first-experiment/ink9um-full-center-forward/metadata.json), including smoothing and physical/representation training keys.
- Pinned villa recipe `aligned21_hybrid_3d2d.json`, flat dataset/normalization code and rounded mean pooling implementation at commit `e0bbb8b40a2db58b1d71864f286eb85717e59e64`, retained under `/tmp/herculaneum-model-source`.
- [Native CT depth consistency](../first-experiment/native-ct-consistency.md) and retained PHerc0800 source/geometry provenance.

No inference, training, source-image modification or external publication was performed by this audit. Only this comparison-design report was added.
