# Completed native common-support comparison

Completed 12 October 2026. Dense-native has the highest equal-parent average
precision (AP), while Hecate leads on PHerc0841. Their ordering reverses on
PHerc0009B. Reader v2 does not lead on either parent. These findings establish
useful development comparators; they do not identify a universally better model
or demonstrate new ink or text recovery.

All five frozen models finished: 20 jobs, 1,698 forwards and 120 core maps, in
58.2 minutes of summed job time on CPU. Every job replayed exactly from saved
logits. The separate verifier checked 238 metric records, map derivatives and
comparison flags. GPU spend was £0. No Actions job or spare-VM task was needed.

## Reference agreement on identical score support

| Model | Equal-parent AP | Equal-parent AUC | PHerc0841 AP | PHerc0009B AP |
|---|---:|---:|---:|---:|
| Reader v2 | 0.424019 | 0.844860 | 0.299046 | 0.548992 |
| Dense-native | **0.560821** | **0.870564** | 0.311644 | **0.809998** |
| Seed42 | 0.271557 | 0.691259 | 0.263400 | 0.279715 |
| Seed43 | 0.487258 | 0.812393 | 0.360173 | 0.614344 |
| Hecate | 0.385107 | 0.793982 | **0.447020** | 0.323193 |
| Fixed five-model mean | 0.554800 | 0.863867 | 0.441332 | 0.668268 |
| Constant 0.5 control | 0.117153 | 0.500000 | 0.061446 | 0.172860 |

The primary statistic first averages the three PHerc0841 segments, then gives
each physical parent equal weight. It is not pooled pixel AP. All models use
the same 232,250 scoring pixels, including 20,137 supplied positives. The 14
one-class cores remain in the analysis with undefined AP/AUC and reported
probability/negative-response metrics. They were not silently dropped.

The five-model mean is 0.006021 AP below dense-native in the primary statistic
and requires all five models. It has a different parent tradeoff; these data do
not justify tuning its weights or selecting an ensemble after the fact.

Segment heterogeneity matters:

| Segment | Positive pixels | Reader v2 AP | Dense-native AP | Seed42 AP | Seed43 AP | Hecate AP | Mean AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0841 / 20260220213127 | 465 | 0.009725 | 0.023923 | 0.025728 | 0.353986 | 0.520296 | 0.461772 |
| 0841 / 20260220214732 | 2,699 | 0.189203 | 0.274644 | 0.184216 | 0.144393 | 0.111686 | 0.196580 |
| 0841 / 20260221022814 | 7,386 | 0.698210 | 0.636365 | 0.580256 | 0.582139 | 0.709078 | 0.665643 |
| 0009B / 20250919125754 | 9,587 | 0.548992 | 0.809998 | 0.279715 | 0.614344 | 0.323193 | 0.668268 |

Hecate's PHerc0841 advantage is not uniform. It is lowest among the five models
on the second segment. On the first, only two cores contain positives; 342 of
465 positives occur in the x3840/y1664 core, where Hecate AP is 0.958642.
The other positive core has Hecate AP 0.145585. Reader v2 also leads some
individual cores despite its weaker parent aggregates. The full
[per-core table](evaluation/metrics.csv) preserves every case; no additional
crop selection or contrast fitting followed these observations.

Seed43 exceeds seed42 by 0.215701 equal-parent AP, including 0.096772 on 0841
and 0.334629 on 0009B. This is variation between released checkpoints, not a
controlled training-seed variance estimate, confidence interval or noise floor.
It does not justify setting a future training gate to twice that gap. It does
show why beating one weak checkpoint is insufficient and why paired training
replicates are needed for later claims.

## What the maps and scores can establish

All four fixed-scale panels were inspected, covering all 24 cores and all five
models plus the frozen mean. The supplied positives often occupy broad regions,
and hybrid predictions are frequently smooth responses. Agreement with these
areas is not a test of resolving individual strokes. Hecate's display is much
darker on 0009B, with sparse localized responses; darkness alone is not better
specificity or proof of absent ink. Context outside the score mask is unscored.

- [0841 / 20260220213127](evaluation/pherc0841-20260220213127.png)
- [0841 / 20260220214732](evaluation/pherc0841-20260220214732.png)
- [0841 / 20260221022814](evaluation/pherc0841-20260221022814.png)
- [0009B / 20250919125754](evaluation/pherc0009b-20250919125754.png)

Calibration also differs. Equal-parent BCE is 0.421373 for Reader, 0.496373 for
dense-native and 0.543229 for Hecate, versus 0.394322 for the fixed mean. Hecate's
BCE is 0.166791 on 0841 and 0.919666 on 0009B. Ranking, probability loss and
threshold behavior therefore answer different questions. The recorded fraction
of reference-negative pixels above 0.5 is not a chemically validated false
positive rate; no threshold was fitted on this panel.

This is adaptive, exposed development evidence. An initial union-mask panel had
zero positive references in all 18 PHerc0841 cores and failed admission. The
retained replacement used only supervision masks to select coordinates, but
that correction followed inspection of class counts. It was frozen before
model inference, not blind at the project level. References were transferred
from the fine canvas using the published resizing contract; physical local
registration, model-review ancestry and upstream training exposure remain
unresolved. Four segments from two parents do not supply independent validation
of generalization, authenticated ink or letters.

## Decision and next work

1. **Keep strong, complementary controls.** Dense-native is a mandatory frozen
   comparator for the proposed coverage study; Hecate remains an architectural
   control with a different parent response. Keep Reader's executable training
   recipe and frozen inference result, without presuming its newer release
   makes it the strongest initializer. Choose and freeze the actual paired-arm
   initializer with the training cohort and development evidence.
2. **Prepare the matched coverage pair.** Admit physical sources and annotation
   tiers, freeze original-source draws and separately normalized stream losses,
   then build the restartable GPU workload. The completed
   [training preflight](../../native-training-preflight-20261012/README.md)
   shows that fixed patch draws alone do not prevent original-source gradient
   dilution. Its synthetic fix is a control, not a new learning result. Do not
   extend the closed native0139 fits or require another tiny CPU improvement.
3. **Strengthen the endpoint without spending the reserve.** Use separate
   development material to establish label provenance and supported stroke or
   negative references alongside area AP. Report acquisition and parent strata.
   Do not open PHerc0175 A/B or PHerc1203 to choose the recipe; confirmation
   requires physically separate support and multiple paired seeds. No claim of
   upstream-unseen material follows from our local reserve alone.
4. **Continue correct-sheet reference construction independently.** Establish
   accepted CT-only continuation/branch references before scoring surface-field
   and tracing methods. This is genuine surface recovery, distinct from local
   depth adjustment. Do not choose geometry by maximizing these ink responses.

The training protocol, actual corpus, exposure, attainable runtime and delivered
artifacts must be concrete before a paid-compute proposal. The £10 first-pilot
total remains a cap, not launch permission or a promise that two 20,000-update
arms fit. No new paid compute is launched by this result.

## Verification and portability

[VERIFICATION.json](evaluation/VERIFICATION.json) records separate arithmetic
checks of 1,698 logit files, 124 output files, all 120 core maps and all 238
metric records. The original runner separately performed 20 exact logit replays.
An injected exception after two committed forwards tested reuse; this was not
a process-crash or lost-success-response test. No metric inspection occurred
before all jobs completed. A pre-result verifier ordering correction and its
original code are preserved in the [evaluation record](../README.md).

[FINALIZER_COMPLETE.json](FINALIZER_COMPLETE.json) records final private progress
publication: 3,567 files in 13 generations, backup branch
`codex/plinius-native-common-comparison-checkpoints`, commit
`752bbdbdd7976c04be3e59ed45bbcebf703b126f`.

The [portable review archive](../../../workspace/bundles/native-common-comparison-review-20261012.tar.gz)
contains 3,600 manifest entries, 79,494,713 compressed bytes, SHA256
`4ae4082fad47e336c6b408fb88fab411cd81002202b480ca2f616611d8a399ac`.
It includes saved logits/maps, all scoring references, receipts and the separate
verifier. It excludes model weights and source CT; use the separate input
bundle to audit source construction. See [PORTABILITY.md](../PORTABILITY.md)
for selective download and offline verification. Replaying this bundle on
another machine does not create a new physical evaluation set.
