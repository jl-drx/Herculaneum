# Tenth study: preservation helps one comparison, but not baseline retention

Adding fixed initial-prediction preservation improves the manual-only continuation on both diagnostic regions, passing that predeclared mitigation screen. It worsens sensitivity with pseudo-labels, and **both new arms still fail retention against frozen seed42**. Keep the frozen baseline. This is a mixed result for a conventional preservation control, not a new model advance or independent ink-accuracy result.

Manual-only mean primary sensitivity rises **73.63% → 76.94%** (+3.31 percentage points); manual-plus-pseudo falls **69.06% → 66.31%** (−2.75 points). Frozen seed42 remains **90.32%**. The shared-mitigation criterion is false. The matched late-batch training losses also remain worse than the initializer, so the result does not establish useful adaptation merely because one retention comparison improves.

## Fixed design and execution

The [protocol](PROTOCOL.md), [lead review](ROOT_REVIEW.md), [independent pre-run review](review/CODE_REVIEW.md) and code were frozen before new model calls. Exactly two new 32-update CPU arms were executed once. Historical ninth A_smooth/B_smooth and frozen seed42 were reused by literal artifact identities. The unchanged initializer predicted the exact same normalized/augmented training tensors as the student, with independent storage, no teacher gradients and unchanged teacher state/RNG. No PHerc0814 data entered gradients or preservation targets.

Original manual/pseudo BCE targets remain 0.25/0.75. A adds mean Bernoulli KL from the initializer on manual participation; B also adds 0.25 times KL on the original pseudo support. Lambda is one, temperature one, with float64 KL arithmetic from FP32 logits. Masks, teacher ancestry, sampling, optimizer, learning rate, update count and calibration remain unchanged. The teacher target is model-derived, not a new annotation. No unknown pixel outside the original active support contributes to the loss.

Ready plan SHA-256: `9e2c9f4f778c3f794e103191fd413937b521fa0d0b72cab0e3905778ce3e3a55`. Metrics SHA: `ff8e549c93ae1430373ff956a5448b5e10abf7202ede858828b0f8192362b751`. There are64 new student updates and64 extra frozen-teacher batch forwards, plus the specified final student evaluation. No extra probe, seed, coefficient search, intermediate checkpoint selection or scientific retry occurred.

## Retention measurements

The two PHerc0814 contexts are reused supplied-supervision development diagnostics from one specimen in a known historical model-development corpus. Equal-context means are descriptive. Calibration uses separate fixed regions and a separately selected threshold for each model; achieved specificity is constrained above95%, not equal across models.

| Model | Mean AUC | Mean AP | Primary sensitivity | Primary specificity |
|---|---:|---:|---:|---:|
| Frozen seed42 | 0.999608 | 0.998510 | 90.32% | 99.967% |
| Manual, ninth | 0.994717 | 0.988662 | 73.63% | 99.179% |
| Manual + pseudo, ninth | 0.995598 | 0.990528 | 69.06% | 99.759% |
| Manual + preservation | 0.996026 | 0.991256 | 76.94% | 99.281% |
| Manual + pseudo + preservation | 0.995775 | 0.989587 | 66.31% | 99.803% |

![Preservation comparison](visualization/render/retention-comparison.png)

### pherc0814-46527-fresh-rank01

Source XYWH `[1152, 1152, 384, 384]`; 49,033 shared participating pixels, 19,545 supplied positives and 29,488 supplied negatives; prevalence 0.398609. The legacy ID contains `fresh`; it is now reused development evidence.

| Model | AUC | AP | Primary sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen seed42 | 0.999882 | 0.999824 | 0.919621 | 1.000000 | 17974 | 0 | 29488 | 1571 |
| Manual, ninth | 0.993594 | 0.990130 | 0.896137 | 0.983587 | 17515 | 484 | 29004 | 2030 |
| Manual + pseudo, ninth | 0.996075 | 0.994427 | 0.884421 | 0.995184 | 17286 | 142 | 29346 | 2259 |
| Manual + preservation | 0.996214 | 0.994056 | 0.931287 | 0.985621 | 18202 | 424 | 29064 | 1343 |
| Manual + pseudo + preservation | 0.997408 | 0.995750 | 0.884216 | 0.996066 | 17282 | 116 | 29372 | 2263 |
### pherc0814-46527-fresh-rank14

Source XYWH `[128, 1152, 384, 384]`; 21,328 shared participating pixels, 4,767 supplied positives and 16,561 supplied negatives; prevalence 0.223509. The legacy ID contains `fresh`; it is now reused development evidence.

| Model | AUC | AP | Primary sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen seed42 | 0.999334 | 0.997196 | 0.886721 | 0.999336 | 4227 | 11 | 16550 | 540 |
| Manual, ninth | 0.995839 | 0.987193 | 0.576463 | 1.000000 | 2748 | 0 | 16561 | 2019 |
| Manual + pseudo, ninth | 0.995120 | 0.986630 | 0.496748 | 1.000000 | 2368 | 0 | 16561 | 2399 |
| Manual + preservation | 0.995837 | 0.988456 | 0.607510 | 1.000000 | 2896 | 0 | 16561 | 1871 |
| Manual + pseudo + preservation | 0.994143 | 0.983424 | 0.441997 | 1.000000 | 2107 | 0 | 16561 | 2660 |

Manual preservation improves sensitivity by3.515 and3.105 percentage points relative to ninth, with ranking changes inside the fixed allowances. Against frozen, it gains1.167 points in rank01 but loses27.921 points in rank14, failing baseline retention. The rank01 gain also has lower specificity than frozen; it is not a free dominance improvement.

Pseudo preservation changes sensitivity by−0.020 and−5.475 points relative to ninth, failing the mean-gain and rank14 criteria. Against frozen it loses3.541 and44.472 points, with additional rank14 AUC/AP margin failures. No screen was relaxed or favorable arm silently promoted. Full-precision per-context checks are in the metrics JSON.

### Calibration and interaction

Primary remains the maximum of three per-region conservative empirical95%-specificity cutoffs. Secondary uses PHerc1667/w029 only. Threshold receipts precede diagnostic inference, and historical maps/thresholds remain unchanged.

| Model | Primary cutoff | Secondary cutoff | Secondary mean sensitivity |
|---|---:|---:|---:|
| Frozen seed42 | 0.755894720554 | 0.755894720554 | 0.903171 |
| Manual, ninth | 0.563550233841 | 0.563550233841 | 0.736300 |
| Manual + pseudo, ninth | 0.627626538277 | 0.625621020794 | 0.695604 |
| Manual + preservation | 0.528907239437 | 0.528907239437 | 0.769398 |
| Manual + pseudo + preservation | 0.675920963287 | 0.675920963287 | 0.663106 |

The two new arms have identical primary/secondary cutoffs, both determined by w029. Secondary reporting therefore leaves their result unchanged. Historical B_smooth secondary sensitivity is slightly higher than its primary, making the new B comparison slightly worse under that rule. All per-region secondary confusion counts are retained in the full metrics JSON; for every role except historical B_smooth they equal the primary table above.

| Contrast | ΔAUC | ΔAP | Δprimary sensitivity (pp) | Δprimary specificity (pp) |
|---|---:|---:|---:|---:|
| interaction difference of differences | -0.001132 | -0.003536 | -6.058 | -0.058 |
| preservation with manual only | +0.001309 | +0.002594 | +3.310 | +0.102 |
| preservation with pseudo | +0.000178 | -0.000941 | -2.748 | +0.044 |
| pseudo with preservation | -0.000250 | -0.001669 | -10.629 | +0.522 |
| pseudo without preservation | +0.000881 | +0.001867 | -4.572 | +0.580 |

The sensitivity difference of differences is−6.058 percentage points. This is a descriptive preservation/pseudo interaction under one continuation schedule, not evidence of a general mechanism. The [retained score audit](diagnostic-audit/DIAGNOSTIC_REPORT.md) separately demonstrates changed ordering, class-specific score redistribution and substantial effects of already fixed cutoffs. These observations cannot locate internal representation damage; diagnostic-specific threshold fitting would not resolve that uncertainty.

## Adaptation cost on exactly matched batches

The table averages preupdate batches25–32 from evolving models. It compares the original target objective, excluding KL, with historical ninth receipts and with the unchanged initializer's actual predictions on those identical inputs. These are online optimization diagnostics, not final-model accuracy. Lower is better; positive deltas versus the initializer mean worse target fit.

| New arm | Manual BCE | Historical manual BCE | Initializer manual BCE | Original objective | Historical objective | Initializer objective |
|---|---:|---:|---:|---:|---:|---:|
| Manual + preservation | 0.610155 | 0.610353 | 0.604241 | 0.610155 | 0.610353 | 0.604241 |
| Manual + pseudo + preservation | 0.608442 | 0.613104 | 0.604241 | 0.757613 | 0.762656 | 0.753955 |

Manual-only preservation changes original loss by−0.000198 versus ninth but remains+0.005913 above the initializer. Pseudo preservation improves original loss by−0.005043 versus ninth yet remains+0.003658 above the initializer. Both new manual losses are worse than their same-input initializer reference. This is stronger reason to pause loss-variant tuning than the isolated mitigation pass is to scale it.

Late-batch mean manual KL is0.046585/0.045041 for A/B; mean absolute manual probability drift is0.102333/0.088891. B's pseudo KL is0.035316 and pseudo probability drift0.064353. These values measure remaining drift, not a demonstrated reduction against ninth: ninth did not retain its per-step raw logits for that comparison. All current raw logits and per-step curves are retained for reproducibility.

## Independent verification and resources

Both first actual student/teacher outputs match exactly, and first full parameter-gradient norms match their historical counterparts exactly. Initial hard/smoothed losses and normalized input/target/mask hashes agree. Frozen teacher predictions are bitwise identical across the two arms and its semantic state remains unchanged. Each final student checkpoint is a safe raw state dictionary with finite, strictly matching schema/storage aliases and actual parameter changes.

The completed-output independent review reconstructed all64 losses from raw logits and original masks, along with fingerprints, teacher identity, online cost curves, saved checkpoints and calibration/metrics/screens. Its maximum complete-objective discrepancy is1.247e-7, KL discrepancy2.78e-17 and metric discrepancy2.22e-16, within the fixed tolerances. It checked312 unique records and used three safe checkpoint reads plus one factory schema inspection with no forward/backward/inference. See [independent verification](review/result/independent-verification.json). The first reviewer attempt misread the explicit eight-step list as a two-endpoint interval; its failure and isolated checker correction are preserved. No scientific code, result or model run was changed or repeated.

The two workers used30.99/30.13 internal wall seconds,62.62 seconds combined controller time, at most1497.36 MiB peak RSS, and307,780,548 bytes of run/analysis output, below400 MiB. Final checkpoints are138,351,655 bytes each; four raw logit arrays are4,194,432 bytes each. Runtime versions and all prior frozen inputs are unchanged. No new scan/model/label-array acquisition, dependencies, paid compute or automatic scientific retry occurred.

## Reference lineage and next decision

The separate [lineage audit](reference-audit/LINEAGE.md) maps existing adaptation/calibration/diagnostic support and confirms all29 curated representations occur in the four original configurations. It identified **PHerc0009B** and **PHerc0500P2**, absent from those explicit roots, as metadata-only candidates. Eight bounded HTTPS reads totaled36,579 response-body bytes, all successful, with no arrays or predictions inspected. Matching CT/label XY metadata establishes only canvas compatibility. The releases are transferred/model-assisted annotations with unknown original annotation source, incomplete teacher exposure, unresolved label transform/depth relation and a dirty package-commit identifier. They are not independently authenticated ground truth.

Keep both candidate specimens outside current adaptation, teacher-target generation and calibration. PHerc0009B is the first mapping/depth-contract lead;109 planes alone does not justify borrowing the existing pooling range. For independent ink accuracy, an exposed-fragment IR/CT pair with verified alignment and annotation lineage is a stronger path. The catalog also identifies legacy Frag4/PHercParis1Fr39 as an optical-reference provenance lead; its past Kaggle-withheld status does not establish holdout from current models. The reservation proposal is unexecuted, and standing ordinary-research authorization remains in place.

Retain frozen seed42. **Pause further loss/coefficient tuning on this reused panel.** Next complete a defensible fragment reference contract and reserve physical support before new predictions; audit the adaptation task/optimization fit before another training comparison. Do not turn a new crop into an independence claim or add replay/calibration changes to rescue this study. Research, not prize compatibility, drives that decision, while actual source terms and attribution remain binding. No authentic letter, transcription, translation, independent accuracy, significance, novelty or frontier-performance claim follows here.

Paid GPU spend remains£0 and the total first-pilot cap remains£10 including failures/retries and attributable overhead. No R2 write, credential import, publication, submission or outreach occurred. Saved local evidence/startup guidance is not an external backup or proof of fresh-task restoration.
