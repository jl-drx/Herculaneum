# Dense supervision improves ranking; threshold transfer remains uneven

The seventh study completed all 16 fixed model/context runs on CPU. Russo's dense ex016 checkpoint improves ROC AUC over its matched manual-label control in three of four new contexts and average precision in all four. Equal-weight averages across the two scroll specimens improve by **0.0756 AUC and 0.1057 AP**. The negative comparison on the sixth study's PHerc1667 crop was local; it does not justify rejecting dense supervision.

The useful next question concerns the operating point. At thresholds selected on the previously used PHerc1667 development crop and frozen before acquiring this panel, dense ex016 has lower sensitivity in three contexts and ties the control at zero in the fourth. Its achieved specificity falls below the development 95% target on one context. Better ranking alone therefore does not establish a reliable detector or calibrated ink probabilities.

![Four-context comparison](panel-summary.png)

## Fixed comparison

Two 384×384 contexts from PHerc0139 and two from PHerc0814 were selected using frozen metadata ranking, annotation class gates and full-context XY nonoverlap. Each primary region is the fixed 256×256 interior intersected with supplied validation membership, acquired source support and common finite evaluated predictions. The four primary masks contain 187,348 annotated pixels: 50,538 positives and 136,810 negatives. These pixels are correlated, and annotation-rich selection does not estimate representative prevalence.

The scan input is the pinned fine acquisition pooled into 21 planes at 9.596 µm, using source depth [13,97), float32 means of consecutive groups of four, nearest-even rounding and uint8 storage. Inference uses planes [2,19), unchanged reference normalization, 128-pixel tiles, stride 64, Hann blending and CPU FP32 with four threads. All four checkpoints load strictly into the unchanged 34,546,498-parameter factory. No checkpoint, window, mask or threshold was selected using the new predictions.

| Context (source XY origin) | Dense AUC | Control AUC | Dense AP | Control AP | Dense sensitivity | Control sensitivity | Dense specificity | Control specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PHerc0139 A (1792,4864) | .8528 | .8069 | .5988 | .5398 | 10.57% | 16.67% | 99.81% | 98.77% |
| PHerc0139 B (3584,5120) | .9514 | .8653 | .8056 | .5149 | 0.00% | 0.00% | 100.00% | 100.00% |
| PHerc0814 C (256,1664) | .8339 | .6592 | .6099 | .5445 | 8.03% | 14.53% | 92.30% | 95.93% |
| PHerc0814 D (640,1536) | .9773 | .9815 | .9757 | .9681 | 58.69% | 59.61% | 100.00% | 99.64% |

Sensitivity and specificity use each model's **development-locked threshold**; the reserved contexts were not recalibrated to 95% specificity. Both development thresholds yield 468 false positives among 9,378 supplied negatives (95.0096% specificity). Their score cutoffs are 0.7234724164 for dense ex016 and 0.7311767340 for control. All crossing ties are excluded by the next representable float32 cutoff.

PHerc0139 B illustrates the distinction: dense AUC is .9514, yet no supplied positive reaches its locked cutoff. On PHerc0814 C, dense has better global ranking but both lower sensitivity and lower specificity than control at their respective transferred cutoffs. Regional score tails deserve direct study.

| Model | Equal-specimen AUC | AP | Locked sensitivity | Locked specificity | Contexts achieving ≥95% specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense ex01675k | .903846 | .747519 | .193243 | .980275 | 3/4 |
| Matched control75k | .828220 | .641845 | .227031 | .985842 | 4/4 |
| Official seed42 | .791627 | .633767 | .121788 | .999913 | 4/4 |
| Official seed43 | .893778 | .743273 | .337280 | .958693 | 3/4 |

Contexts receive equal weight within a specimen; specimens then receive equal weight. These are descriptive averages, without pixel confidence intervals. The complete four-model table, full-validation secondary domain, score curves, occupancy strata, raw predictions and masks are retained in [benchmark results](benchmark/run/analysis/RESULTS.md) and [machine-readable metrics](benchmark/run/analysis/metrics.json). Seed43's lower specificity on PHerc0814 C is retained. No arm or failed region is hidden by the averages.

## Verification and resources

All 16 workers completed once, without retries, producing 400 inference tiles. Summed worker wall time was 73.80 seconds; final benchmark output size was 52,945,226 bytes. The 36 scan chunks plus before/after metadata consumed 64,305,530 measured response-body bytes in 44 successful requests, within the frozen 64 MiB acquisition cap. Annotation selection consumed 121,261 measured body bytes across 216 requests, including redirects. Source metadata discovery and the separate native-reference audit have their own retained ledgers. These are body-byte measurements, not total network-wire accounting.

- [Independent data reconstruction](data-review/DATA_REVIEW.md) matched all 12,386,304 pooled voxels, all 108 retained compressed annotation objects and all source/mask counts; 251 input receipts stayed unchanged.
- [Independent metric reconstruction](metric-review/INDEPENDENT_REVIEW.md) matched both domains, tied-score AUC/AP, threshold selection, confusion counts, strata and aggregation. Maximum absolute numerical discrepancy was 2.22×10⁻¹⁶; 421 receipts stayed unchanged.
- The four-model exposure ledger was derived from the actual checkpoint configurations. The strict model contract, frozen plans, source receipts and raw outputs remain available. Separate Astra interpretation review informed the next experiment.

No paid GPU, R2 upload, external publication or prize submission occurred. Paid GPU spend remains £0; the first pilot's £10 total cap is unchanged. Local files and saved startup guidance are not a durable R2 backup or a verified fresh-environment restore.

## Scientific limits and decision

The contexts were reserved from **our tuning**, not globally held out from these models. All three reference families occur in the models' recorded training/online-validation universe. Exact historical pixel exposure, teacher ancestry and complete human/optical truth remain unresolved. PHerc0139 manual supervision is unchanged in ex016, so its gains cannot all be attributed to dense training directly on that evaluated region. Current supervision/validation overlap is zero, but this does not prove historical exclusion.

Distinct scrolls provide two physical specimen groups. Within-specimen XY nonoverlap does not establish independence of complete 3D sampling support. Observed zero-valued profiles are retained and are not treated as verified background; no such profile occurs in the primary masks. No geometry-validity mask was acquired. This is public-annotation agreement, with no new authenticated letters, transcription or translation and no first-in-class performance claim.

The [native-reference audit](native-reference/NATIVE_FEASIBILITY.md) identified PHerc0139/w035 with delivered 9.362 µm/113 keV data and native-origin reference annotations. Its fine and native canvases differ, and a measured interscan registration is absent. It is a useful later diagnostic lead, not a pixel-paired native/fine experiment. The present pooled panel establishes no native-domain causal mechanism.

**Proceed to a controlled training prototype aimed at retaining dense-label ranking gains while stabilizing regional positive and negative behavior.** Compare common-initialization manual-only continuation, ordinary hard-pseudo continuation and dense continuation with region-balanced manual positive/negative anchors. Freeze ordered patches, augmentation, optimizer, updates and manual exposure across arms. Include a frozen-model calibration comparator; training must earn its additional complexity. Familiar ingredients do not yet constitute a novel method.

This panel is now development evidence for that new hypothesis. Reserve fresh physical support, including aliases and context, before new adaptation; describe it as held out from our adaptation unless stronger historical independence is demonstrated. Predeclare per-region ranking and sensitivity/specificity criteria so suppressing all positive predictions cannot count as success. A one-seed pilot is a feasibility result; later replication and a factorial ablation must separate regional balancing from manual anchoring.

The separate [training contract draft](../training-prototype-readiness/TRAINING_CONTRACT_DRAFT.md) records the next design and remaining choices. Its [synthetic CPU mechanics check passed](../training-prototype-readiness/RESULTS.md): one strict seed42 forward/backward/update took 3.123 seconds and 781.23 MiB peak RSS, with finite gradients, an actual parameter update and exactly zero output gradients outside known participation. Updated weights were discarded; this did not train a research model. Real data, teacher, split and training budget still need freezing. Paid execution additionally requires verified scoped worker credentials, a pinned CUDA environment, current pricing and independent shutdown within the £10 aggregate cap. No further user decision is needed for the remaining unpaid design work.

Evidence entry points: [protocol](PROTOCOL.md), [accepted design](DESIGN_DECISIONS.md), [frozen panel plan](benchmark/panel-plan.json), [threshold verification](threshold-verification.json), [exposure ledger](model-exposure-ledger.json), [interpretation review](RESULT_INTERPRETATION_REVIEW.md).
