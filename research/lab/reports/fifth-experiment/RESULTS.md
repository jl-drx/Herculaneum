# Fifth experiment: native rendering and larger context

**The source-to-render check passed a separately recorded offline revalidation, and the frozen context experiment completed. No authenticated ink, letter sequence, transcription or translation is established.** The larger view continues to show broad detector response patches. Further expansion around these peaks is not the next priority; a controlled comparison with newer ink models and verified labels is more informative.

Open the [context review](context-run/analysis/review.html), [source and score panels](context-run/analysis/screen-overview.png), [depth controls](context-run/analysis/condition-sensitivity.png), or [native CT cross-sections](native-sections/native-cross-sections.png). These are private research diagnostics.

## Native source check and integrity recovery

We checked the two fourth-experiment response peaks and their four cardinal neighbours, ten points in total, across all 31 rendered layers. Two observed native CT chunks supplied every interpolation corner. The fixed archived rendering conventions reproduced 309 of 310 samples exactly; the remaining sample differed by one uint8 intensity count. All samples passed the predeclared within-one-count technical tolerance. No sign, offset, contrast or transform was fitted.

The original run correctly stopped when a concurrent review edit changed its code during execution. It remains a stopped run, not a successful clean execution. The original code and plan were restored byte for byte. A separate offline revalidation verified the frozen inputs, retained chunk identity and hashes, and recomputed the same 310 comparisons with network access trapped. Its result passed. The [native review](NATIVE_REVIEW.md), [offline record](native-check/offline-revalidation.json) and [recovery manifest](native-check/recovery-manifest.json) retain this distinction. The original manifest describes the interrupted attempt and is historical evidence, not the current recovery manifest.

Four fixed local cross-sections contain 17,801 observed and 1,211 unknown samples. Unknown interpolation support is explicitly masked; no extra chunks were acquired for these figures. Structured, layered material is visible near the sampling locations. The reference lines represent fixed local tangent planes and normal offsets, not the curved mesh throughout each plane. These checks establish source correspondence at the selected points, not correct papyrus-wrap placement, recto/verso, or ink.

Native payload was 4,194,304 bytes plus 239 metadata bytes. The revalidation and cross-sections added no network traffic.

## Context selection and acquisition

The first 768-square geometry rule found no fully valid crop containing the earlier field with the required margins. That negative result remains preserved. An explicit second rule, frozen before new intensities or predictions, allowed at least 95% whole-crop geometry validity while requiring at least 95% of the fixed interior to survive conservative exclusion of tiles touching invalid geometry.

Exactly one of nine spatially eligible candidates passed: source bbox `[768,256,768,768]` on PHerc0800 volume `20250521135224`, segment `20251028220955`. Geometry covers 567,072 of 589,824 pixels (96.14%). Geometry alone preserves 258,048 of the 262,144 interior pixels (98.44%). Selection is explicitly post hoc to the earlier peaks and is not held-out validation.

Nine old chunks were reused after hash, source identity and conditional HEAD verification. Twenty-seven new chunks all returned HTTP 200. New payload was 13,713,408 bytes, with 3,668 metadata bytes: 13,717,076 response-body bytes across 38 requests, including nine bodyless HEAD requests. Reused payload was 4,571,136 bytes. There was no missing-object fill, automatic retry, whole-volume download or new model download. Combined native and context acquisitions used 17,911,619 response-body bytes, excluding protocol overhead and the separately recorded literature audit.

## Fixed CPU execution and evidence

The unchanged adapter and two pinned checkpoints ran center layers `7:24`, followed by seed42 controls `5:22`, `9:26` and reversed `7:24`. CPU FP32 inference used four threads. All 605 scheduled tiles were scored, taking 44.63 seconds of model inference and 56.21 seconds across the four subprocesses including startup and verification. No target training or paid compute occurred.

The nominal full field spans 6.64 mm per side; the fixed 512-square interior spans 4.42 mm. Actual strict support is 253,952 pixels (96.875% of that interior), after excluding every contributing tile that touches invalid geometry, unobserved source or a condition-specific all-zero selected-depth profile. It is smaller than the geometry-only preflight support, as expected. Primary-domain support is 262,144 pixels. All source objects were observed; a zero profile is not missing-object fill and is not evidence of ink absence.

| Descriptive measurement | Strict domain |
| --- | ---: |
| Both center raw scores above 0.5 | 18,297 / 253,952 pixels (7.20%) |
| Connected shared-response components | 14 |
| Center seed correlation | 0.537 |
| Above-midpoint seed Jaccard | 0.346 |
| Center vs minus-two depth correlation | 0.881 |
| Center vs plus-two depth correlation | 0.897 |
| Center vs reversed-depth correlation | 0.402 |

The entire old 31×384×384 source region is byte-identical. Both center predictions over the old 256-square interior are bit-identical, with identical evaluation/validity masks. This validates implementation consistency where tile context is unchanged. It is not an independent biological observation.

The fixed threshold is a model-behaviour diagnostic, not calibrated ink probability. Both seeds share architecture and training data. Shifted windows share 15 of 17 input slices. Neither agreement nor depth sensitivity establishes ink. Connected components can be truncated at the analysis boundary and are not letters or independent samples.

Root visual inspection of both complete figures found broad, partly seed-dependent response islands and structured source material without a coherent source-supported stroke sequence. The wider view does not justify a reading. The separate [independent review](context-run/INDEPENDENT_REVIEW.md) and [verification record](context-run/independent-verification.json) document the additional read-only checks; assistant reviews are not specialist papyrology.

## Research decision and persistence

Retain these leads and stop automatic crop expansion or training on these predictions. The next priority is the [frontier methods research program](../frontier-methods-audit-20261008/RESEARCH_PROGRAM.md): verify newer model contracts, separate authentic annotation from teacher targets, and design physically separated evaluation before claiming improvement. Surface geometry methods become a separate study if a measurable geometry bottleneck is demonstrated.

Plans, source chunks, code, raw predictions, masks, logs, negative outcomes and hashes remain local in this cloud workspace. These new results have not been uploaded to R2 or published externally. GPU spend remains £0; the £10 total first-pilot cap and its launch prerequisites remain in force.
