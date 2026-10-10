# Fixed depth averaging did not improve the Frag6 panel

The first fixed Frag6 CPU comparison is complete. **Retain unchanged forward seed42 for this experiment.** Equal averaging of three neighboring depth windows did not meet the predeclared improvement criterion. No additional model, depth, crop or threshold was searched after seeing the scores.

## Primary result

| AP summary, equal weighting of four contexts | Unchanged seed42 | Three-window average | Paired change |
|---|---:|---:|---:|
| Mean over shifts −1, 0, +1 | 0.557754 | 0.555493 | **−0.002261** |
| Clean input, shift 0 | 0.560409 | 0.556660 | **−0.003749** |

The frozen screen required a mean AP improvement of at least **+0.01** over the three shifts, with mean clean AP loss no greater than **0.01**. The gain condition failed; the mean clean-loss condition passed. This is a descriptive screening result, not a statistical rejection of every possible depth-aware method. It does not support promoting this fixed averaging candidate or renting a GPU for it.

## Every context is retained

| Context | Positive prevalence | Clean baseline AP | Clean candidate AP | Clean baseline AUROC | Clean candidate AUROC | Mean AP change over shifts |
|---|---:|---:|---:|---:|---:|---:|
| r544-c4128 | 72.85% | 0.974140 | 0.986637 | 0.938946 | 0.966973 | +0.000466 |
| r2080-c3616 | 86.39% | 0.992932 | 0.989117 | 0.957105 | 0.933772 | −0.000196 |
| r4640-c1056 | 3.17% | 0.117563 | 0.093233 | 0.879686 | 0.855254 | −0.010317 |
| r6688-c1568 | 9.60% | 0.157000 | 0.157652 | 0.723944 | 0.710623 | +0.001003 |

There is useful agreement with supplied labels in some contexts, but performance varies greatly. The two small-positive regions have much lower precision-recall performance than the positive-dominated regions. Prevalence differs sharply, so compare paired changes within each context rather than treating the absolute APs as directly comparable measures of difficulty. The largest loss is in r4640-c1056, whose scoring core contains only 129 positive coarse pixels. Those pixels are not independent examples.

The [all-context figure](figures-v1/fixed-core-comparison.png) shows that averaging makes relatively small changes to the probability maps. The weak sparse-ink cases remain weak. [Paired changes](figures-v1/paired-AP-changes.png) and the complete [machine-readable outcomes](scores-v1/RESULTS.json) retain every displacement and metric. No best-looking window or image was substituted for the primary result.

## What ran

The four source crops and reference masks were selected and fixed in phase31, before inference. Each crop contains 63 historical uint16 CT layers, pooled exactly into a 21×128×128 array using 3×3×3 averaging. The clean baseline's 17-plane input corresponds to raw layers 7–57. Five window starts per context support all three displacement comparisons without duplicate model execution.

The verified seed42 checkpoint ran unchanged in CPU FP32 evaluation mode, with robust-MAD normalization over each full patch. All 20 normalized inputs were stored and hashed before the first forward. There were **exactly 20 forwards**, no training, and an unchanged before/after model-state hash. Original logits and sigmoid probabilities are retained. A single patch needs no overlapping-tile blend; no redundant Hann multiply/divide was added. Candidate probabilities were averaged equally in float64 from retained FP32 outputs, as frozen before inference.

The common scoring support contains 16,153 coarse pixels: 6,922 supplied positives and 9,231 supplied negatives. Another 231 pixels remain unknown/excluded. This homogeneous-block convention omits mixed boundaries. It does not validate boundary precision, exhaustive negatives, calibrated probabilities or anatomical ink depth.

## Acquisition failure and explicit recovery

The original serial acquisition hit its 600-second budget and is **preserved as a failed attempt**. It completed all 252 metadata requests and 252 pixel batches covering layers 1–42. The next batch, layer 43 batch 0, timed out with 196,608 partial response bytes retained. It has no completion manifest and was never treated as a complete source.

Before more requests, a separate recovery protocol was frozen: reuse the 252 verified complete batches, fetch only the 126 missing batches with four subprocess workers, and allow one new 600-second attempt. The original cumulative byte, request and disk caps remained unchanged. Recovery finished in **48.856 seconds**. Its completion and combined accounting are separate from the failed attempt. There was **one deliberate repeat of an interrupted request**, no hidden automatic retries and no redirects. The partial response was excluded from source arrays.

Across both attempts: 631 requests, 84,575,721 returned body bytes and 149,429,372 reserved body bytes. Useful CT payload is 74,317,824 bytes. Aggregate active acquisition time was 649.609 seconds, so the original 600-second total did not pass. No complete TIFF layers or whole scans were downloaded.

The recovery monitor measured aggregate HTTP-worker RSS, but did not capture peak RSS of its supervising/assembling process. Do not claim a verified whole-recovery memory peak. This instrumentation limitation does not replace the exact data checks below. Future acquisition supervision should cover the parent process as well as workers.

## Verification and compute

All 37,158,912 source uint16 values were reconciled against retained HTTP bytes, using the frozen byte-range plan separately from the recovery producer's physical-address assembly. All 1,376,256 pooled FP32 values matched a second exact pooling formulation. No missing source rows were filled or inferred.

Five transport tests and four scoring tests passed. The original acquisition runtime failure remains distinct from those nine passing fixtures. All 48 AP/AUROC values matched separate ranking formulations within 2.23×10⁻¹⁶. All 20 saved probability maps agreed with their original logits' sigmoid within 7.64×10⁻⁸. All executable/input freezes remained unchanged.

The externally supervised CPU job took **7.509 seconds**, with measured peak RSS **582,586,368 bytes**. Its worker reported 5.515 seconds including model loading and inference work. There were no extra model forwards during review. These measurements support CPU sufficiency for this small comparison; they are not a GPU-training cost estimate.

The restored phase31 evidence passed all 19,539 file records before work began. Repository tracked files remain unchanged. Prior failed and successful evidence is preserved. **GPU spend remains £0; the total first-pilot cap remains £10.**

## Interpretation and next action

This is agreement with supplied annotations in four selected contexts of **one PHerc51 parent**. Frag6 is now model-scored development data. Optical registration, annotation exhaustiveness and model/teacher ancestry remain qualified. The result is neither an independent ink-accuracy estimate nor a new-model or transcription claim. PHerc0009B, PHerc0500P2 and PHerc0841 remain untouched.

Keep the baseline and use the existing cached data to diagnose the weak sparse-ink regions, especially sensitivity to reference boundaries and small registration uncertainty. Preserve this negative primary result while doing so. Freeze any follow-on diagnostic before execution, report all perturbations, and never shift labels to maximize model scores. This is the next useful CPU step before proposing a new architecture or paid training run.
