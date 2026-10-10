# Saved calibration-transfer results

Metrics source: `/workspace/herculaneum-lab/reports/sixteenth-experiment/inference/analysis/metrics.json`; SHA-256 `c911803a07ac44741eb9724b2e4b78dbfd8cb32f92848378960960f17d1ee175` (86092 bytes).

This helper only formats saved metrics. Independent numerical verification is a separate prerequisite for interpretation.

AUC and AP use 0–1 units. Markdown rates are percentages rounded to two decimals; CSV rates retain 0–1 fractions. Counts are exact integers. `undefined` preserves missing metrics and never means zero.

## Locked model cutoffs

The Frag3-only cutoff is the maximum of two negative-only calibration cutoffs from Paris1. The retained 14 cutoff is each model’s own earlier cutoff. Threshold changes leave its ranking scores unchanged.

| Model | Retained 14 cutoff | Frag3-only cutoff |
| --- | --- | --- |
| seed42 baseline | 0.7558947205543518 | 0.726689338684082 |
| A preserve | 0.7450202703475952 | 0.7076279520988464 |
| B preserve | 0.7422624826431274 | 0.7000159621238708 |


## Paris2 context ranking and support

| Context | Model | ROC AUC | AP | Positive prevalence | Positive count | Negative count |
| --- | --- | --- | --- | --- | --- | --- |
| frag1-context1 | seed42 baseline | 0.4461 | 0.1860 | 21.91% | 19400 | 69159 |
| frag1-context1 | A preserve | 0.4433 | 0.1849 | 21.91% | 19400 | 69159 |
| frag1-context1 | B preserve | 0.4436 | 0.1847 | 21.91% | 19400 | 69159 |
| frag1-context2 | seed42 baseline | 0.9037 | 0.5001 | 15.16% | 12633 | 70681 |
| frag1-context2 | A preserve | 0.9008 | 0.4903 | 15.16% | 12633 | 70681 |
| frag1-context2 | B preserve | 0.9016 | 0.4920 | 15.16% | 12633 | 70681 |
| frag2-context1 | seed42 baseline | 0.5013 | 0.2698 | 23.60% | 19049 | 61660 |
| frag2-context1 | A preserve | 0.5211 | 0.2756 | 23.60% | 19049 | 61660 |
| frag2-context1 | B preserve | 0.5242 | 0.2754 | 23.60% | 19049 | 61660 |
| frag2-context2 | seed42 baseline | 0.7795 | 0.5812 | 23.39% | 20248 | 66318 |
| frag2-context2 | A preserve | 0.7815 | 0.5706 | 23.39% | 20248 | 66318 |
| frag2-context2 | B preserve | 0.7825 | 0.5627 | 23.39% | 20248 | 66318 |


## Paris2 context operating points

TP/FP/TN/FN refer to the fixed primary support. “No positives” means zero predicted-positive pixels.

| Context | Model | Cutoff rule | Sensitivity | Specificity | TP | FP | TN | FN | No positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frag1-context1 | seed42 baseline | retained 14 | 0.00% | 95.17% | 0 | 3341 | 65818 | 19400 | no |
| frag1-context1 | seed42 baseline | Frag3 only | 0.31% | 93.07% | 61 | 4796 | 64363 | 19339 | no |
| frag1-context1 | A preserve | retained 14 | 0.00% | 95.07% | 0 | 3407 | 65752 | 19400 | no |
| frag1-context1 | A preserve | Frag3 only | 0.00% | 92.94% | 0 | 4886 | 64273 | 19400 | no |
| frag1-context1 | B preserve | retained 14 | 0.00% | 95.22% | 0 | 3308 | 65851 | 19400 | no |
| frag1-context1 | B preserve | Frag3 only | 0.00% | 92.88% | 0 | 4924 | 64235 | 19400 | no |
| frag1-context2 | seed42 baseline | retained 14 | 1.45% | 99.58% | 183 | 296 | 70385 | 12450 | no |
| frag1-context2 | seed42 baseline | Frag3 only | 5.72% | 98.71% | 723 | 914 | 69767 | 11910 | no |
| frag1-context2 | A preserve | retained 14 | 0.74% | 99.64% | 94 | 252 | 70429 | 12539 | no |
| frag1-context2 | A preserve | Frag3 only | 4.95% | 98.70% | 625 | 920 | 69761 | 12008 | no |
| frag1-context2 | B preserve | retained 14 | 0.51% | 99.65% | 65 | 247 | 70434 | 12568 | no |
| frag1-context2 | B preserve | Frag3 only | 4.79% | 98.72% | 605 | 905 | 69776 | 12028 | no |
| frag2-context1 | seed42 baseline | retained 14 | 2.58% | 99.45% | 491 | 342 | 61318 | 18558 | no |
| frag2-context1 | seed42 baseline | Frag3 only | 2.59% | 98.51% | 494 | 920 | 60740 | 18555 | no |
| frag2-context1 | A preserve | retained 14 | 2.58% | 99.57% | 491 | 266 | 61394 | 18558 | no |
| frag2-context1 | A preserve | Frag3 only | 2.58% | 98.54% | 491 | 898 | 60762 | 18558 | no |
| frag2-context1 | B preserve | retained 14 | 2.58% | 99.60% | 491 | 244 | 61416 | 18558 | no |
| frag2-context1 | B preserve | Frag3 only | 2.58% | 98.55% | 491 | 892 | 60768 | 18558 | no |
| frag2-context2 | seed42 baseline | retained 14 | 6.37% | 99.85% | 1289 | 98 | 66220 | 18959 | no |
| frag2-context2 | seed42 baseline | Frag3 only | 9.56% | 99.54% | 1935 | 303 | 66015 | 18313 | no |
| frag2-context2 | A preserve | retained 14 | 4.68% | 99.91% | 948 | 60 | 66258 | 19300 | no |
| frag2-context2 | A preserve | Frag3 only | 8.15% | 99.54% | 1650 | 303 | 66015 | 18598 | no |
| frag2-context2 | B preserve | retained 14 | 3.97% | 99.90% | 803 | 66 | 66252 | 19445 | no |
| frag2-context2 | B preserve | Frag3 only | 7.83% | 99.45% | 1586 | 367 | 65951 | 18662 | no |


## Paris2 equal-fragment summary

Each fragment first averages its two context metrics equally; Paris2 then averages the two fragment means equally. This is a macro average, not pooled pixels. Counts are therefore reported per context above. A summary metric remains undefined if any contributing metric is undefined.

| Model | Cutoff rule | ROC AUC | AP | Positive prevalence | Sensitivity | Specificity | Precision | F1 | Predicted positive | Balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed42 baseline | retained 14 | 0.6576 | 0.3843 | 21.02% | 2.60% | 98.51% | 47.52% | 4.91% | 1.75% | 50.56% |
| seed42 baseline | Frag3 only | 0.6576 | 0.3843 | 21.02% | 4.55% | 97.46% | 41.70% | 8.17% | 2.95% | 51.00% |
| A preserve | retained 14 | 0.6617 | 0.3804 | 21.02% | 2.00% | 98.55% | 46.52% | 3.83% | 1.59% | 50.27% |
| A preserve | Frag3 only | 0.6617 | 0.3804 | 21.02% | 3.92% | 97.43% | 40.07% | 7.12% | 2.84% | 50.67% |
| B preserve | retained 14 | 0.6630 | 0.3787 | 21.02% | 1.76% | 98.59% | 45.01% | 3.39% | 1.51% | 50.18% |
| B preserve | Frag3 only | 0.6630 | 0.3787 | 21.02% | 3.80% | 97.40% | 39.19% | 6.91% | 2.84% | 50.60% |


## Specificity transfer checks

Each context passes exactly when `20 × FP ≤ negative count`. All four must pass; the primary rule is Frag3 only. Passing specificity does not establish useful ink detection; read sensitivity and zero-positive flags alongside it.

| Model | Cutoff rule | frag1-context1 | frag1-context2 | frag2-context1 | frag2-context2 | All four |
| --- | --- | --- | --- | --- | --- | --- |
| seed42 baseline | retained 14 | PASS | PASS | PASS | PASS | PASS |
| seed42 baseline | Frag3 only | FAIL | PASS | PASS | PASS | FAIL |
| A preserve | retained 14 | PASS | PASS | PASS | PASS | PASS |
| A preserve | Frag3 only | FAIL | PASS | PASS | PASS | FAIL |
| B preserve | retained 14 | PASS | PASS | PASS | PASS | PASS |
| B preserve | Frag3 only | FAIL | PASS | PASS | PASS | FAIL |


Both evaluation fragments come from one parent papyrus, Paris2. Labels were inspected for fixed support gates; complete upstream model exposure is unknown. These are conditional supplied-reference agreement results, without independent-scroll replication, significance tests, automatic model promotion, authenticated text or translation.
