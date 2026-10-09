# Eleventh study: optical-derived Frag4 reference

The unchanged baseline has the highest equal-context AUC and average precision among all five states on the predeclared forward input. The continuation variants do not demonstrate a consistent transferable benefit. Reversing depth makes every model worse on these macro ranking measures. This is a descriptive result on two regions of one fragment, with unresolved historical model exposure and optical registration error; it establishes neither unseen-test accuracy nor a new scroll reading.

All 20 fixed model/context/direction runs completed: 500 tile forwards, no training, calibration or retries. Independent checks reproduced the input pooling bit for bit, all blended score maps exactly, and the metrics and frozen-threshold confusion counts.

## Primary result

The table uses the fixed primary masks and gives each context equal weight. Sensitivity and specificity use each model’s own previously frozen primary cutoff. Those operating points are not matched-specificity comparisons on Frag4.

| State | AUC | Average precision | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Unchanged baseline | 0.734809 | 0.186118 | 3.03% | 95.16% |
| A: manual + smoothing | 0.568181 | 0.120993 | 8.94% | 86.67% |
| B: manual + pseudo + smoothing | 0.642271 | 0.142314 | 22.16% | 84.23% |
| A + prediction preservation | 0.564788 | 0.129121 | 6.14% | 86.06% |
| B + prediction preservation | 0.626298 | 0.134257 | 13.20% | 85.11% |

The baseline’s two regions differ substantially: AUC is 0.887179 in context 1 and 0.582439 in context 2. At its unchanged cutoff, sensitivity is 6.05% and 0.00%, respectively. The continuations sometimes recover more supplied positives, but at substantially lower specificity. Higher sensitivity at different cutoffs is not evidence that they learned a better ranking.

| Context | State | AUC | Average precision |
| --- | --- | ---: | ---: |
| frag4-context1 | Unchanged baseline | 0.887179 | 0.161482 |
| frag4-context1 | A: manual + smoothing | 0.601046 | 0.046845 |
| frag4-context1 | B: manual + pseudo + smoothing | 0.690955 | 0.064070 |
| frag4-context1 | A + prediction preservation | 0.536058 | 0.041540 |
| frag4-context1 | B + prediction preservation | 0.669552 | 0.058212 |
| frag4-context2 | Unchanged baseline | 0.582439 | 0.210754 |
| frag4-context2 | A: manual + smoothing | 0.535316 | 0.195142 |
| frag4-context2 | B: manual + pseudo + smoothing | 0.593587 | 0.220559 |
| frag4-context2 | A + prediction preservation | 0.593518 | 0.216701 |
| frag4-context2 | B + prediction preservation | 0.583045 | 0.210303 |

Context 1 has 3,818 supplied positives and 87,626 negatives (4.18% positive prevalence); context 2 has 15,252 positives and 62,599 negatives (19.59%). All 169,295 primary pixels remained evaluated and finite for all states and both directions. Average precision must be interpreted with these differing prevalences; the macro average is descriptive, not a pooled specimen estimate.

## Matched preservation comparison

Adding prediction preservation to A changes forward macro AUC by −0.003393 and AP by +0.008128. The AUC change is −0.064988 in context 1 and +0.058202 in context 2: a mixed effect, rather than a consistent rescue. Its macro sensitivity falls by 2.79 percentage points and specificity by 0.61 points at the retained cutoffs.

Adding preservation to B changes forward macro AUC by −0.015972 and AP by −0.008057. Macro sensitivity falls by 8.97 percentage points while specificity rises by 0.89 points. Neither preservation state exceeds the baseline on forward macro AUC or AP. These observations do not identify a single mechanism of training failure.

## Predeclared reversed-depth diagnostic

Reverse was evaluated for every state before any orientation selection was possible. It remains secondary; no primary result was replaced.

| State | Reverse AUC | Reverse average precision |
| --- | ---: | ---: |
| Unchanged baseline | 0.457600 | 0.106357 |
| A: manual + smoothing | 0.348484 | 0.085030 |
| B: manual + pseudo + smoothing | 0.404380 | 0.097730 |
| A + prediction preservation | 0.250573 | 0.080785 |
| B + prediction preservation | 0.323753 | 0.094898 |

The large direction effect confirms sensitivity to depth order in this representation. Reversal does not rescue the comparison. This is not independent authentication of anatomical normal orientation or optical alignment, and below-0.5 AUC is not a reason to invert scores after inspection.

## Reference and preparation

The specimen is PHercParis1Fr39 (legacy Frag4), not the PHercParis4 scroll listed in the current models’ training configurations. It is absent from the 29 explicit representations audited for their initializers and from our continuation/calibration work; upstream historical exposure remains unknown. The EduceLab-Scrolls paper, arXiv 2304.02084v4, describes 1000 nm IR, manual registration, manual ink tracing under papyrology supervision, and a 65-plane rendering with surface index 32. The exact released region descriptor associates the CT stack, labels and support mask. This supports conditional agreement with an optical-derived reference, not perfect ground truth.

All three downloaded PNGs share a 9,791 × 6,064 canvas. Palette polarity was checked, and every one of the 4,610,281 marked ink pixels lies inside the surface mask. The candidate registered IR was retained but not used to fit a warp, select a region or change a label. No registration residual was measured. The original optical TIFF, large alignment PSD and full PPM were not acquired.

Label-only rules selected two complete native 1,152 × 1,152 regions from 40 candidates (25 eligible): YX origins (0, 1152) and (8064, 3456). Selection used fixed class/support gates and lexicographic first/last-separated order. It did not maximize model agreement or ink amount. We excluded a symmetric 16-native-pixel neighborhood around label/support boundaries and a 32-coarse-pixel context border. This is an explicit conservative variant of the paper’s boundary exclusion, not an exact reproduction of its historical loader.

We fetched only the required full-width row bands in source layers 07–57. Each of 51 TIFFs was independently checked, and metadata/ranges were bound by strong ETags and If-Match. Exact uint32 sums over 3 × 3 × 3 voxels were divided in float64 by 27 × 257 and cast once to float32, producing 17 × 384 × 384 inputs at nominal 9.72 µm. There was no uint8 quantization. This is our newly declared adaptation for the unchanged robust-MAD detector, not the historical ink-ID preprocessing. The exact 2023 renderer execution was not reproduced.

Modern PHerc0009B remains unused: intended transfer code applies its offset during transfer, but the actual dirty publisher is unreproduced, the advertised mapping-validity asset is absent from the complete inspected label prefix, and annotation depth remains unresolved. No additional shift, validity mask or pooling window was invented.

## Verification and resources

- CT acquisition: 255 successful HTTP 206 requests, 1,425,096,672 retained response bytes, no retry or redirect; peak reader RSS 57,462,784 bytes. Full-width TIFF rows cause unused-column transfer, but no full plane or native scan was downloaded.
- CPU comparison: 500 real tile forwards, 62.20 seconds controller wall time, maximum worker RSS 616,030,208 bytes; 117,754,953 bytes of run and analysis artifacts. No backward calls, new checkpoint, threshold fit or state mutation.
- Independent CT reconstruction matched both float32 inputs bit for bit. All 20 saved score maps reconstructed from the retained logits with maximum error 0; 100 unique normalized input fingerprints matched across all five states.
- A separate scorer verified 403 file pins, tie-aware AUC/AP, exact confusion counts, context and macro deltas. Earlier evidence is retained unchanged.
- Paid GPU spend remains £0; the £10 total pilot cap is unchanged. No R2 write, GitHub data upload, external publication, submission, outreach or terms acceptance occurred.

The initial TIFF header probe read eight incidental encoded CT-strip bytes beyond its required eight-byte classic header. Those zero bytes were retained without pixel decoding. This small scope deviation is documented; subsequent structural requests used exact header bounds. All successful response bytes are accounted for.

## Decision and next study

Pause this continuation recipe and preservation-coefficient tuning. The tenth study’s limited benefit on reused development references did not become a consistent gain here. More GPU time is not justified by these results alone.

The next useful step is a bounded input-domain and learning-dynamics diagnosis: distinguish acquisition/preprocessing effects from the observed failure of the original training objective to improve on matched online batches. Keep optimizer, representation and supervision changes separate, use synthetic and existing retained evidence first, and freeze any new real-data/model comparison before execution. Frag4 has now been inspected and must be treated as seen diagnostic evidence, not a fresh prospective holdout for further tuning. Keep it outside training, teacher targets and calibration unless an explicit future design changes that role and records the loss of that reservation. Reserve additional physical specimens before making future generalization claims.

This study improves the evidence base and rejects an unsupported improvement claim. It does not produce a first-in-class model, translation or authenticated new text.

## Reproducible evidence

- [Scientific design](RESEARCH_DESIGN.md), [root inference review](INFERENCE_REVIEW.md), [primary-source annotation/depth addendum](optical-reference/annotation-provenance/ANNOTATION_AND_DEPTH_ADDENDUM.md).
- [Frozen CT plan](ct-acquisition/acquisition-plan.json), [CT provenance](ct-acquisition/run/provenance.json), [frozen comparison plan](comparison/final-plan/comparison-plan.json), [complete metrics](comparison/analysis/metrics.json).
- [Independent technical review](review/comparison/REVIEW.md), [independent numeric review](result-verification/final-review/REVIEW.md).
- Legacy source terms and EduceLab citation requirements are preserved in [LICENSE](optical-reference/legacy-LICENSE.current.txt). Raw data remains private; configuration-draft saving is not backup, publication or verified fresh-task restoration.

Numeric figures: [ranking agreement](visualization/results-v2/ranking-agreement.png) and [fixed-threshold operating points](visualization/results-v2/fixed-thresholds.png). Initial renders are preserved; version 2 only corrects layout.
