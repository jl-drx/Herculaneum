# Independent Astra interpretation review

Reviewed 8 October 2026 by the user-authorized `frontier_strategy_review` agent using gpt-6-astra, read-only after completed inference. Root records its conclusions here; the reviewer performed no training, downloads or output edits. Numerical recomputation is a separate review.

The diagnostic argues against adopting ex016 as an upgrade on this evidence. Ex016's excessive positive responses at the fixed midpoint accompany poorer supplied-label ranking: primary AUC falls by 0.0343 and AP by 0.0464 relative to control, while mean score rises by 0.077. Monotonic recalibration cannot recover lost ranking. This is not evidence that dense supervision generally harms ink recovery.

The secondary domain favors control too, but overlaps the primary and includes different boundary context. It is a sensitivity result, not independent replication. Seed43 leads primary AUC and balanced accuracy; control leads AP. Define the intended operating criterion before choosing a model.

Contract and source checks complete a useful integration experiment without resolving historical exposure, teacher independence, label authenticity or geometry validity. The 16,384 correlated pixels are not 16,384 independent observations. This is neither a reproduction of whole-region author scores nor a native PHerc0800 transfer test. No frontier performance or new method is demonstrated.

The next falsifiable study should freeze physically distinct reference regions, record annotation lineage/acquisition domain before predictions, and keep development threshold selection separate from reserved evaluation. Select thresholds under one common specificity requirement, then assess sensitivity, AUC and AP on reserved regions. Genuine native coarse-scan references are required to test acquisition-domain interaction.

Decision branches:

- If independent threshold selection removes the practical deficit and ranking is competitive across regions, prioritize operating-point transfer and calibration.
- If ranking or matched-specificity sensitivity repeatedly declines, proceed to manual-only, hard pseudo-label and reliability-aware supervision with sampling/coverage controlled.
- If behavior varies by acquisition domain, prioritize the domain × supervision-quality interaction.
- If the disadvantage disappears outside this crop, retain the local negative result and reject a broad degradation claim.

Complete and retain this benchmark, keep existing baselines, and prioritize the reference panel and bounded supervision experiment over further checkpoint shopping. Prize eligibility remains a recorded downstream consequence of research choices.
