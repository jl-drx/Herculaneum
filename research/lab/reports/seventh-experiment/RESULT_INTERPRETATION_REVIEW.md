# Astra interpretation review and lead decision

Reviewed after the frozen seventh-panel run, 8 October 2026. Reviewer: the existing `frontier_strategy_review` agent, gpt-6-astra, requested by the user as a high-level second opinion. The lead accepted the following interpretation. This is scientific judgment based on retained results, separate from the independent data and numerical checks.

The seventh panel reverses a broad reading of the sixth study's local negative result. Dense ex016 exceeds control AUC in three contexts and AP in all four, with equal-specimen gains of .0756 and .1057. Preserve dense supervision as a candidate. Do not call it a universally improved detector: frozen-threshold sensitivity falls in three contexts and ties at zero in one, and specificity transfer fails on PHerc0814 rank01.

PHerc0139 rank02 has .9514 AUC but zero locked sensitivity. PHerc0814 rank01 combines better global ranking with worse sensitivity and specificity at transferred cutoffs. Global ranking does not determine the useful score tail. Annotation ancestry and limited physical independence restrict causal and generalization claims.

The next mechanistic hypothesis is that dense teacher targets improve ranking while their contribution to regional negative-score tails is uneven. Preserve reliable supplied-positive and supplied-negative behavior during dense training. Compare (1) manual-only continuation, (2) ordinary hard-pseudo continuation and (3) dense continuation with region-balanced manual anchors. All arms need common initialization, ordered patches, augmentation, optimizer, update count and manual exposure; masks and normalization must make the intended differences explicit.

One seed can test mechanics and feasibility. A promising result needs additional seeds, and a later factorial ablation must separate regional balancing from additional manual anchoring. Include a cheap frozen-model calibration comparator. Reduced positive detection everywhere is not improvement. Require a useful per-region sensitivity/specificity tradeoff with retained ranking, rather than a favorable aggregate hiding a failed region.

The existing panel is development evidence for this hypothesis. Freeze fresh support before adaptation, exclude aliases and complete context from new updates, and state the limited historical independence. Do not delay the pooled prototype for missing native/fine registration. This is a concrete experiment using familiar ideas, not an established novel method. Prior-art comparison and later independent evidence determine whether a frontier contribution emerges.

Lead action: prepare the bounded training/data contract and execute one synthetic CPU training-readiness check. No real-data adaptation, checkpoint adoption or paid launch is implied by this review. Worker credentials, pinned CUDA image, current price and independent shutdown remain paid-pilot prerequisites under the £10 total cap.
