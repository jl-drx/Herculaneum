PHerc343 saved-response spatial persistence
2026-10-10 | Six saved maps, three model identities, no new inference

BOTTOM LINE
Rank-selected response locations recur across saved outputs, but overlap and connectivity depend on model, phase, region and selected fraction. Dnative is more phase-stable descriptively; that does not establish greater accuracy. Cross-model strongest-response support often differs substantially. This adds a stability characterization, not ink validation, letter readings, or completion of the pending four-model reproduction.

WHOLE VALID PUBLISHED CROP (8,887,441 pixels)
Dice at top 5%, 10%, 20% (boundary ties included):
dnative phase          0.853 / 0.852 / 0.838
seed42 phase           0.597 / 0.623 / 0.661
seed43 phase           0.597 / 0.632 / 0.670
dnative / seed42       0.501 / 0.495 / 0.504
dnative / seed43       0.466 / 0.473 / 0.493
seed42 / seed43        0.426 / 0.427 / 0.472

TOP 10% POSITION SUMMARY, WHOLE VALID CROP
Pair                   mean/p95 nearest distance (native px); within 0/2/4/8 px
dnative phase            2.3/  8.6; 85.2%/88.9%/91.5%/94.6%
seed42 phase             8.6/ 45.9; 62.3%/66.8%/70.5%/76.3%
seed43 phase             9.9/ 58.4; 63.2%/67.2%/70.6%/76.1%
dnative / seed42        26.8/133.4; 49.5%/52.5%/55.0%/59.5%
dnative / seed43        29.5/147.9; 47.3%/50.2%/52.7%/57.3%
seed42 / seed43         24.6/108.6; 42.7%/45.8%/48.6%/53.6%

The 0/2/4/8-pixel summary was explicitly added after initial metrics, following a review request; it is supplemental, not a prespecified pass/fail criterion. Both directions separately are in metrics.json.

ALL 18 BOXES, TOP 10%
Columns: box; phase Dice dnative/seed42/seed43; merged-pair Dice dnative-seed42/dnative-seed43/seed42-seed43.
 1  0.900 / 0.764 / 0.568;  0.644 / 0.501 / 0.506
 2  0.777 / 0.581 / 0.667;  0.485 / 0.815 / 0.522
 3  0.917 / 0.333 / 0.585;  0.246 / 0.007 / 0.063
 4  0.900 / 0.691 / 0.740;  0.622 / 0.561 / 0.686
 5  0.742 / 0.755 / 0.752;  0.562 / 0.563 / 0.680
 6  0.837 / 0.287 / 0.472;  0.413 / 0.280 / 0.244
 7  0.931 / 0.697 / 0.850;  0.705 / 0.830 / 0.716
 8  0.800 / 0.582 / 0.640;  0.548 / 0.769 / 0.520
 9  0.746 / 0.385 / 0.271;  0.444 / 0.063 / 0.209
10  0.869 / 0.399 / 0.434;  0.218 / 0.239 / 0.114
11  0.835 / 0.556 / 0.761;  0.001 / 0.012 / 0.534
12  0.816 / 0.519 / 0.673;  0.396 / 0.302 / 0.394
13  0.704 / 0.835 / 0.613;  0.744 / 0.677 / 0.515
14  0.836 / 0.697 / 0.755;  0.553 / 0.567 / 0.347
15  0.814 / 0.545 / 0.613;  0.000 / 0.365 / 0.027
16  0.682 / 0.573 / 0.618;  0.538 / 0.560 / 0.399
17  0.536 / 0.775 / 0.413;  0.247 / 0.220 / 0.251
18  0.807 / 0.627 / 0.545;  0.408 / 0.408 / 0.304

These boxes were predefined and sometimes overlap. Their median and ranges below summarize this fixed set, not independent replicates. All boxes and every declared rank are displayed in all_supports_dice.png; all 18 have overlap contact sheets at the predeclared central 10% rank.

CONNECTIVITY
8-connected binary selected masks; components are clipped by each support. Largest-component share changes at top 10%, median (range) over all 18 boxes:
dnative phase          4.7 percentage points (0.0 to 37.4)
seed42 phase           6.1 percentage points (0.2 to 39.5)
seed43 phase           9.7 percentage points (0.0 to 34.3)
dnative / seed42       21.2 percentage points (0.0 to 54.7)
dnative / seed43       11.1 percentage points (0.1 to 58.8)
seed42 / seed43        7.2 percentage points (0.1 to 48.5)

Counts of all components, largest-component share, and components needed to cover 90% of selected mass are provided for every mask in selections.csv. No component-size filter was fitted. Tiny quantization speckles affect component counts; bridges can change connectivity abruptly. A change in largest share does not identify which component split or merged, and an unchanged summary does not prove unchanged topology. These are intensity-rank selection effects, not ground-truth errors.

METHOD
Frozen PLAN.json precedes new metrics; SUPPLEMENT_PLAN.json records the later distance-radius addition. Source TIFF and decoded hashes were rechecked against retained PASS receipts, along with valid-mask pins. Source hashes and previous-review provenance are embedded in metrics.json. Native padded alignment [40:2600,200:6920], 20x20 source ownership (IDs 1 or 4 use padded), and published crop [105:2560,380:6021] are unchanged. Region bounds use prior floor/ceil exclusive maxima. Visuals reverse both axes only for published display orientation; calculations use native pixels.

Each map is ranked separately within each support. Therefore a box top-10% mask is not the restriction of the whole-crop top-10% mask. Ranking reduces dependence on differing raw levels but is not calibration or a probability. Fractions 5/10/20% span a small declared high-response grid without tuning to shapes. All boundary ties are included to avoid arbitrary coordinate tie-breaking. Exact-mass weighted Dice uses unit weight above the threshold and a common fractional weight at ties, with min-membership intersection.

Target 5%: actual tie-inclusive fraction ranges 5.003% to 5.875%.
Target 10%: actual tie-inclusive fraction ranges 10.000% to 11.460%.
Target 20%: actual tie-inclusive fraction ranges 20.001% to 21.498%.
Maximum absolute binary-versus-weighted Dice difference over all comparisons: 0.0417. The full sensitivity grid is retained; no threshold is declared best.

DISTANCE AND CONTROLS
For A-to-B, each selected pixel in A is assigned the Euclidean pixel-center distance to the nearest selected pixel in B within that same support. Reverse distances are pooled with forward distances, including zeros, and mean/p95 computed across the pooled array. Directions are weighted by their selected counts, not averaged equally. Straight lines can cross invalid geometry; these are not geodesic distances. A whole-crop search can find a selected response outside a particular candidate box; a per-box search cannot. Distances are native pixels only, not physical distances, tracked motion, registration shifts, or feature identity. Centroid drift can reflect changing rank prominence or remote components and can conceal local differences.

Self-overlap controls are exactly one; all-box nearest self distances are exactly zero. Uniform-independent occupancy expected Dice 2*pA*pB/(pA+pB) is recorded only as an arithmetic scale (near the selected fraction). Spatial autocorrelation and shared training invalidate treating this as a realistic spatial null, a p-value, or independent evidence. No significance test treats pixels independently.

VERIFICATION AND VISUAL REVIEW
All 342 support/rank/pair overlaps were recomputed from reloaded arrays; an independent cKDTree method reproduced both-direction nearest distances, p95 and centroids for all 324 box comparisons, including the supplemental radius fractions. The whole-crop distance array uses the primary EDT implementation and was not independently recomputed. Component area sums, boundary-rank inequalities, exact weighted mass and source hashes passed. verification.json records the scope. All four generated figures were visually inspected: labels, box inclusion/order, overlap colors and native-aspect-ratio rendering are readable, and the visual pattern agrees with the descriptive numerical results.

LIMITS AND COMPLETION ASSESSMENT
Same saved inputs, model families and shared training create dependent evidence. Quantization, ties, selected fraction, specimen geometry and box clipping affect the summaries. No ground truth, CT-volume reinspection, Hecate map, independent training replication, transcription accuracy, or physical-ink validation is supplied. Reproducible saved-response locations and a completed saved-map analysis do not by themselves finish the four-model reproduction.

FILES
PLAN.json / SUPPLEMENT_PLAN.json: declared analysis scopes
metrics.json: all selections, comparisons, provenance and hashes
comparisons.csv / selections.csv: machine-readable compact tables
all_supports_dice.png: every support and all fractions
all_boxes_overlap_1.png through _3.png: boxes 1-18, all six pair comparisons; phase A=plain/B=padded, cross-model A/B follow name order
analyze.py / verify.py / write_report.py: reproducible code
verification.json: independent numerical-check scope and outcome
