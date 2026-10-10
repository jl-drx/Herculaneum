# Dense-supervision reference comparison — 8 October 2026

**The four-model CPU comparison completed. Russo's ex016 dense-supervision checkpoint performed worse than its matched manual-label control on this fixed PHerc1667/w029 crop. We retain the existing baselines and move to testing supervision reliability across physical regions and acquisition domains.** This is a local result against public reference annotations, not a general rejection of pseudo-labeling or a new ink-reading result.

The [research charter](../../RESEARCH_CHARTER.md) governs the program: develop leading models and techniques; record prize eligibility as an optional downstream consequence. Astra reviewed both the strategy and the completed results. Actual licenses, publication permissions and the £10 total first-GPU-pilot cap remain binding.

## Results

The primary common domain contains 16,384 pixels: 7,006 supplied positives and 9,378 negatives. At 9.596 µm per pixel, this fixed 128 × 128 interior spans approximately 1.23 mm square inside a 256 × 256 crop. Every model evaluated the whole crop; no primary pixels were lost to common support.

| Fixed checkpoint | ROC AUC | Average precision | Balanced accuracy at score ≥ 0.5 |
| --- | ---: | ---: | ---: |
| Russo ex016 dense | 0.912934 | 0.874074 | 0.538334 |
| Russo manual control | 0.947220 | 0.920451 | 0.750845 |
| Official seed42 / 75k | 0.924449 | 0.895401 | 0.840397 |
| Official seed43 / 75k | 0.947973 | 0.912323 | 0.864291 |

The prespecified paired comparison is ex016 versus its matched control: **−0.034287 AUC and −0.046377 AP**. At the diagnostic midpoint, ex016 labels all 7,006 supplied positives positive, but also 8,659 of 9,378 supplied negatives. Its specificity is 7.67%, versus the control's 51.48%. The mean score rises by 0.077. Lower AUC/AP shows that threshold placement alone does not explain the result; a strictly monotonic recalibration cannot recover the lost ranking.

The secondary full validation domain contains 51,580 pixels. Its AUC is 0.792465 for ex016 and 0.903113 for control. This overlapping domain includes different boundary context and is a sensitivity check, not an independent replication. Seed43 has the highest primary AUC, while control has the highest AP; this crop does not select a generally superior model.

[Fixed-scale comparison](benchmark/run/analysis/comparison.png) · [ROC and precision–recall curves](benchmark/run/analysis/primary-curves.png) · [Exact metrics and paired diagnostics](benchmark/run/analysis/metrics.json) · [Common support](benchmark/run/analysis/common-support.json)

## What was verified

The [protocol](PROTOCOL.md) fixed four arms, crop-selection rule, 64-pixel margin, 17-plane window, forward orientation, unchanged normalization, tiled inference and metrics before predictions. Selection used metadata and supplied labels, not image intensities or model responses. No model, crop, threshold, ensemble or depth window was fitted after inspecting scores.

[Independent data reconstruction](data-contract/DATA_REVIEW.md) reproduced all 1,376,256 pooled voxels exactly using an integer nearest-even calculation independent of the acquisition implementation, decoded all 12 annotation chunks, and verified 72 input records. Source depth [13,97) becomes 21 planes; the model uses [2,19). All source pixels were acquired, with no missing-object fill. Acquisition support and nonzero occupancy remain distinct from authenticated geometry validity.

All four SHA-pinned checkpoints passed strict loading into the same 34,546,498-parameter architecture. The two Russo embedded configurations differ at the six intended label roots and output directory. Stored checkpoint step is 74,999; the release filenames say 75k. The initial audit failure is preserved: it counted aliased tensor names twice and assumed the filename was the stored integer. The [additive v2 correction](model-contract/README-v2.md) retains the storage limit and verifies exact aliases, tensor layout and the observed step.

Root executed each model once, on CPU with four threads and FP32. Each scored nine tiles in approximately 0.72–0.78 seconds; model workers including strict loading took approximately 2.68–2.73 seconds and peaked near 560 MiB RSS. These small-crop measurements do not estimate GPU training throughput or memory. Final code, input, checkpoint and output integrity checks passed.

[Independent numerical review](independent-metric-review/result/independent-verification.json) verified 149 unique input/output records before and after, reconstructed both common masks, and reproduced all four arms' AUC, AP, fixed-threshold confusion, occupancy and paired diagnostics to 1e-12 without importing the benchmark metrics implementation. [Astra's separate interpretation review](RESULT_INTERPRETATION_REVIEW.md) supports the bounded conclusions and next-study decision below.

## Interpretation and next study

The supplied annotations may include iterative/model-derived refinement. Historical trainer exposure, teacher ancestry and complete physical sampling overlap remain unresolved. The crop is selected for annotation support and belongs to a segment present in training configurations. It is a developmental comparison with correlated pixels, not a physically held-out accuracy benchmark. It neither reproduces the author's whole-region scores nor measures transfer to native PHerc0800. No target labels, training, new letters, transcription or translation were produced.

The next study will freeze a small reference panel of physically distinct regions and record annotation provenance, scan energy/resolution, physical support and teacher lineage before prediction. Keep development and reserved regions separate. Select thresholds on development support under one common specificity requirement; report sensitivity at that operating point, AUC and AP on reserved regions. Current w029 is already used development data. Only genuinely native coarse-scan references can test the acquisition-domain hypothesis; pooled fine scans cannot substitute for them.

If ranking or matched-specificity sensitivity repeatedly degrades under dense supervision, test manual-only, hard pseudo-label and reliability-aware supervision with matched initialization, sampling, compute and controlled coverage. If the practical deficit is mainly operating-point transfer, prioritize calibration. If results vary by acquisition domain, test domain × supervision reliability. If this local disadvantage does not recur, reject the broader degradation hypothesis. Prior-art review must precede any novelty claim. A geometry-uncertainty robustness study remains a separate candidate, not a prerequisite to every model prototype.

## Resources and persistence

Two new public checkpoints totaled 276,734,604 payload bytes. Label/source acquisition used 7,164,184 response-body bytes across 32 HTTP requests. Initial metadata research recorded 4,440,477 final-response bytes, but its redirect overhead was unmeasured; these figures are not total wire traffic. All execution was CPU-only, with **£0 paid GPU spend** and no new R2 upload. Runpod secret import and worker delivery remain unverified; the £10 total pilot cap remains intact.

Reports, plans, raw maps, source chunks and review evidence are retained in this cloud workspace. Saved environment startup guidance does not publish a snapshot or back up these artifacts. The tracked repository remains unchanged during onboarding.
