# Scan-derived supervised reference: PHerc1667 / w013

Audit and acquisition date: October 8, 2026.

**A genuine scan-derived, aligned training control is ready as uint8 `[21,256,256]`.** It contains 26,364 annotated positive pixels and 39,172 annotated negative pixels under the supplied supervision mask. The crop was fixed from released labels before any model prediction or source-image inspection. This is a supervised training control, with known segment overlap; it is not independent physical ground truth or held-out validation.

## Exact sources and alignment

| Field | Retained value |
| --- | --- |
| Scroll / physical training segment | PHerc1667 / `1667:w013` |
| Public segment | `20240304141531-w013_20240304141531_flatboi` |
| Native CT volume associated with surface | `20251217075048`, 2.399 µm |
| Source surface array | Level 2, uint8 `[109,10400,19900]`, chunks `[109,128,128]` |
| Source surface spacing Z/Y/X | `[2.399,9.596,9.596]` µm |
| Fixed source level-2 crop X/Y | `3968:4224` / `1408:1664`, half-open pixel indices |
| Prepared surface | uint8 `[21,256,256]`, `[9.596,9.596,9.596]` µm |
| Supplied annotation plane | Z = 10 in the 21-plane label arrays |
| Model input window | Prepared planes `2:19`, exactly 17 real planes |
| Prepared surface SHA-256 | `47c741b4b2f12764c83ae34f98935e7b4cf93bad1f2dff0e4e78135cd374574c` |

The [official ink_9um dataset README](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/README.md) identifies the exact public surface volume for `pherc1667-w013`. Its [label attributes](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_inklabels.zarr/.zattrs) and [supervision attributes](https://huggingface.co/buckets/scrollprize/datasets/resolve/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc1667-w013/pherc1667-w013_supervision_mask.zarr/.zattrs) independently agree on the source URL, level 2, source shape, depth crop `[13,97]`, 21 output channels and 9.596 µm output spacing. Both label arrays have the same XY shape as source level 2. The [source array metadata](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/2/.zarray) and source physical attributes were downloaded and checked against those fields.

Alignment provenance is a transferred annotation reviewed with offsets, rather than direct infrared measurement. The label attributes record `annotation_source_results=results-fixed` and a relaxed mapping policy: additions within 128 pooled pixels of strict supervision coverage, followed by specified island and hole cleanup. Internal source-manifest/report paths are retained as provenance strings; those private paths were not accessed. No unverified private source was substituted for the public scan.

## Selection and annotation semantics

Four predefined crops near the canvas center had no supervised labels. Their raw label and supervision chunks are retained, and those pixels were treated as unknown. A single bounded official chunk listing then identified the first compressed label chunk larger than an observed all-zero chunk, in server path order: `0.11.31`. Its 256 × 256 crop was accepted because it contained at least 256 supervised positive and negative pixels. Selection stopped at that first adequate region. This procedure uses the annotation support to choose a positive control; it does not estimate typical performance on the scroll.

| Released value or combination | Interpretation | Pixels in selected crop |
| --- | --- | ---: |
| Ink label > 0 and supervision > 0 | Annotated positive | 26,364 |
| Ink label = 0 and supervision > 0 | Annotated negative | 39,172 |
| Supervision = 0 | Unknown; exclude from comparison | 0 |
| Ink label > 0 and supervision = 0 | Inconsistent positive outside supervision | 0 |

Observed ink-label values are exactly `{0,1}`; observed supervision values are exactly `{1}`. The source label and supervision arrays are zero outside Z = 10 in every acquired label chunk. [Pinned training documentation](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/docs/ink_detection.md#L533) states that the supervision mask defines which pixels participate in loss and metrics. [Pinned training code](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/training/train.py#L169) uses positive ink values and excludes nonpositive supervision.

Separate retained files preserve the distinction:

- `inklabels.npy` and `supervision-mask.npy`: original center-plane uint8 annotation values.
- `annotation-valid-mask.npy`: Boolean `supervision > 0`.
- `ink-positive-mask.npy`: Boolean `(inklabels > 0) & annotation-valid-mask`.
- `source-observed-nonzero-mask.npy`: Boolean nonzero CT occupancy over source planes `13:97`; all 65,536 crop pixels have observed CT. This is an occupancy diagnostic, not a guarantee of mesh validity. No independent geometry-validity mask was acquired.

## Depth preparation and orientation

The prepared array follows the [pinned upstream preprocessing algorithm](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/preprocessing/prepare_9um_isotropic_input.py): select source planes `13:97`, reshape into 21 consecutive groups of four, calculate each mean in float32, apply `np.rint`, then cast to uint8. `np.rint` uses nearest-even rounding. There is no added plane, spatial resampling, reversal or fitted depth offset.

Prepared plane `k` uses source planes `[13+4k:17+4k]`. The annotated center plane 10 therefore averages source indices 53, 54, 55 and 56, with mean index 54.5. The centered model window `2:19` uses source depth `21:89`. These half-open indices preserve the released annotation convention exactly.

Full official metadata independently records the June 17, 2026 surface render, source commit `ae2ecce9dded502b0cc24127e9fceabd04e1257d`, 109 slices, scale 1 and `flip-normals=true`. Increasing published source Z is retained. The rendering record and its association with the public surface were extracted from the previously retained, hashed official `metadata.json`; it was not inferred from the image or model output.

The reference spacing is 9.596 µm versus the PHerc0800 target's 8.640 µm. The target pixels are approximately 9.96% smaller than reference pixels; neither source was fitted or rescaled for visual similarity. The selected reference covers 2.456576 × 2.456576 mm nominally in its flattened canvas. It is a small control area, not an award-area claim.

## Provenance, acquisition limits and checks

Four source CT chunks, Z/Y/X `0/11/31`, `0/11/32`, `0/12/31` and `0/12/32`, supplied the crop. Their decoded total is 7,143,424 bytes. Scan payloads transferred **7,119,599 bytes**; all retained HTTP payloads, including metadata, label probes, one listing and terms evidence, total **7,467,219 bytes (approximately 7.12 MiB)**, below the 64 MiB cap. No native 3D volume, whole surface or new checkpoint was downloaded.

All raw chunks remain under `data/experiments/known-ink-control/`. `acquisition-sources.json` retains exact requested/final URLs, retrieval times, HTTP headers, ETags, Last-Modified when supplied, byte lengths and SHA-256 hashes. `provenance.json` includes model-compatible `array_sha256` and `shape_zyx`, source index ranges, aggregation, render orientation, derived-file hashes and selection records. `acquire_reference.py` implements bounded acquisition; upstream preprocessing source and its MIT license are retained for attribution.

Acquisition checks reproduced pooling independently with a per-plane loop, executed the pinned upstream centered-slice function to confirm `[13,97]`, verified chunk decode dimensions, compared every mask conversion, and confirmed annotation encodings and absence of unsupervised positives. Evidence is in `acquisition-validation.json`. No model inference or transcription was performed by this acquisition.

## Training overlap and terms

The [published hybrid training recipe](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/configs/aligned21_hybrid_3d2d.json) explicitly includes `1667:w013` and representation `public_2p4_level2_zmean4:pherc1667-w013`; both locally verified checkpoint configurations also include them. The source attributes mark this case `online_validation=false`. The acquired label objects are dated August 18, 2026, while the pinned model repository was last modified August 14, 2026. Named segment/representation overlap is established; byte identity with the labels actually used during checkpoint training is not established. A successful response here can check pipeline behavior, not demonstrate cross-scroll accuracy.

The [official curated dataset description](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/02_data_datasets.md#L103) states that no infrared ground truth exists in intact scrolls: labels begin with human ink strokes and iterative pseudo-labeling. This limits independence from earlier models and reinforces the training-control interpretation.

Terms evidence is retained for review. The [public data portal](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/02_data.md#L143) defaults to CC-BY-NC 4.0 unless an asset has other terms. The ink_9um README provides no explicit license declaration, while the [official curated ink-labels README](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/static/data/datasets/ink-labels-README.md#L37) links to [legacy data terms](https://dl.ash2txt.org/LICENSE.txt), including written approval for redistribution and restrictions on publishing textual revelations. Their exact application to this bucket is not resolved by its README. Code/model MIT does not assign a license to CT or annotations. This acquisition remains outside the repository for local analysis; no data redistribution, textual revelation or prize submission was performed.
