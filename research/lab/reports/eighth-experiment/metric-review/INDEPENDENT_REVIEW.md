# Independent eighth-pilot verification

PASS. One completed-output review reconstructed the frozen masks, training schedule and exposures, development cutoffs, raw-score ranking and confusion metrics, equal-context summaries and predeclared gates. It verified the initializer and three saved checkpoints offline. No training, forward, backward, inference or network calls occurred during this review.

The frozen training plan SHA256 is `eadb8406f594446e62840366c280358ded25b93874bd04e07b3e0e2482ee96f3`; the reviewed analyzer metrics SHA256 is `ce8354f3a023b105b5e02aeabd4fbf512915ec81d3b76e7b999b2e16a9ddb89b`. All 223 reviewed input/output records retained their sizes and SHA256 values before and after verification. The machine-readable receipt is [result/independent-verification.json](result/independent-verification.json).

## Training and checkpoint evidence

All four workers used the same actual initializer semantic state hash, `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`. A, B and C each completed 32 recorded updates using the exact common batch schedule, region order and D4 choices. The independently reconstructed masks use supplied manual participation `S & ~V & observed`; teacher participation covers observed, evaluated, finite-score cells outside that manual mask, including the openly retired historical V support. Teacher targets come only from the retained seed43 scores at 0.5.

Each arm's repeated patch exposure was 230,340 manual-positive and 387,868 manual-negative cells, plus 40,620 teacher-positive and 389,748 teacher-negative cells. These are repeated exposures, not independent examples. A assigns no loss coefficient to the teacher term. The eligible grid has six tiles per region and 6,782 unique teacher-positive cells across the two adaptation contexts; the whole-context teacher-positive count is larger, as already disclosed in the design audit.

The saved scalar arithmetic agrees exactly with A=`U`, B=`U + 0.25 P`, C=`0.5 U + 0.5 R + 0.25 P`; all recorded unknown-output gradients are zero and all 96 parameter-update probes change. Raw training logits and the four class BCE means were not retained. Consequently this review verifies coefficient arithmetic and receipts; it cannot independently recompute the numerical U/P/R losses. The earlier frozen implementation review and synthetic checks cover the masked BCE construction and gradient behavior.

Each saved final raw state was loaded with `weights_only=True` on CPU, after bounded uncompressed ZIP checks and SHA256/size verification. A single unchanged CPU factory construction supplied canonical parameter and buffer names; it received no input and performed no forward call. All final checkpoints have the exact 508 tensor keys/shapes/dtypes, 264 unique storages, 244 alias groups, matching strides/offsets/allocation sizes, finite values and bitwise consistent aliases. Unique tensor storage is 138,185,992 bytes, beneath the fixed 160 MiB guard. All 264 canonical parameter tensors change in each adapted checkpoint; this counts tensors containing changes, not changed individual elements. Semantic state and parameter hashes match the worker receipts. The independent review checks strict schema compatibility; the workers' separately recorded persisted roundtrip and strict reload remain their operation.

## Independent numerical result

Ranking uses the immutable sixth independent implementation: integer pair concordance with half credit for tied positive/negative scores, and AP from score-group suffix counts. Benchmark metric code is not imported. Fresh masks are reconstructed from supplied supervision, observed source, the fixed 64-pixel margin and common evaluated/finite support. Primary counts are exactly 49,033 and 21,328 cells, with 19,545 and 4,767 supplied positives. All midpoint and locked-threshold confusion counts, score tie counts, occupancy strata and quantiles match. The maximum numerical difference is `1.1102230246251565e-16`.

Every local development cutoff was reconstructed from its negative-score crossing boundary and the next representable float32 value, excluding the full boundary tie. The primary cutoff is the maximum of the three fixed development cutoffs; secondary is the fixed w029-only cutoff. Neither rule consults fresh labels or scores. The frozen worker source writes and hashes the threshold lock before fresh inference. For C, the primary cutoff is `0.859406054019928`, determined by old PHerc0814 rank01; its w029-only cutoff is `0.8128029108047485`. For the other arms, both rules coincide.

| Model | Mean AUC | Mean AP | Primary sensitivity | Primary specificity |
|---|---:|---:|---:|---:|
| Frozen seed42 | 0.999608 | 0.998510 | 0.903171 | 0.999668 |
| A, manual | 0.989429 | 0.976560 | 0.688869 | 0.998946 |
| B, manual + teacher | 0.987941 | 0.979877 | 0.771325 | 0.997219 |
| C, balanced anchors + teacher | 0.984691 | 0.965987 | 0.559924 | 0.999678 |

These are equal-context averages over two fresh PHerc0814 contexts. Exact rational confusion-count calculations reproduce the sensitivity and specificity gates. All three C-comparator screens fail. C minus B mean primary sensitivity is `-0.21140045538840593`; C minus A is `-0.1289445167040415`; C minus frozen is `-0.34324682949949614`. The candidate gate is false. Every adapted arm also has lower mean AUC, AP and primary sensitivity than the frozen baseline. C has a higher mean primary specificity accompanied by substantially lower sensitivity.

## Scope

This is a negative one-seed, 32-update continuation screen under a frozen recipe. It provides no evidence to promote C under the predeclared criterion. The C package mixes region and class balancing, and all adapted arms share a change from inherited smoothed training targets to unsmoothed targets and the continuation optimizer; the result cannot isolate a regional causal mechanism or decide the value of other dense-training recipes.

Both fresh contexts belong to one PHerc0814 specimen and are reserved only from this study's new updates. The baseline and teacher configurations include the source representation; exact historical pixel exposure and annotation ancestry remain unresolved. The exceptionally strong frozen baseline therefore supports a supplied-annotation retention diagnostic. It does not establish independent ink accuracy, calibrated ink confidence, physical independence, optical truth, a novelty claim or statistical significance. Repeated cells, tiles and the two contexts do not provide independent pixel replicates.

The verifier ran once, exited zero and preserved all frozen data, code, plans and results. Only new files in this metric-review directory were created. No further edits or verification runs are planned.
