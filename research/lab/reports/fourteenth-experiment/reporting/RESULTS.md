# Development-only calibration transfer

This calibration-only control changes operating cutoffs while retaining the literal thirteenth diagnostic score maps. Ranking is unchanged. Its primary comparisons use each model’s own development-only cutoff; common-baseline and secondary cutoffs remain separately reported. Root owns the scientific decision.

The specificity rule was fit only to the same three calibration regions’ supplied negatives, using the unchanged conservative float32 tie selector. It constrains empirical calibration-region specificity; achieved specificity on a diagnostic context must be checked separately. These are already seen contexts, with unresolved physical/teacher-history independence.

## Frozen operating cutoffs

| State | Common baseline | Own primary | Secondary PHerc1667 only |
| --- | ---: | ---: | ---: |
| frozen_seed42 | 0.75589472055435181 | 0.75589472055435181 | 0.75589472055435181 |
| A_preserve | 0.75589472055435181 | 0.74502027034759521 | 0.74502027034759521 |
| B_preserve | 0.75589472055435181 | 0.74226248264312744 | 0.74226248264312744 |

Primary is the maximum of all three regional cutoffs. Secondary is fixed PHerc1667-w029 only and is not selected instead of primary after viewing diagnostics. The canonical negative counts are 9,378, 40,400 and 14,644; Frag4 and PHerc0139 adaptation labels are excluded from cutoff fitting.

## Per-context achieved operating points

![Sensitivity and achieved specificity](render/calibration-transfer.png)

| Context | State | Common sensitivity / specificity | Primary sensitivity / specificity | Primary 95% floor |
| --- | --- | ---: | ---: | --- |
| pherc0814-46527-fresh-rank01 | frozen_seed42 | 91.962% / 100.000% | 91.962% / 100.000% | pass |
| pherc0814-46527-fresh-rank01 | A_preserve | 91.016% / 100.000% | 92.638% / 100.000% | pass |
| pherc0814-46527-fresh-rank01 | B_preserve | 91.087% / 100.000% | 92.858% / 100.000% | pass |
| pherc0814-46527-fresh-rank14 | frozen_seed42 | 88.672% / 99.934% | 88.672% / 99.934% | pass |
| pherc0814-46527-fresh-rank14 | A_preserve | 86.847% / 99.946% | 87.980% / 99.934% | pass |
| pherc0814-46527-fresh-rank14 | B_preserve | 86.742% / 99.946% | 88.211% / 99.934% | pass |
| frag4-context1 | frozen_seed42 | 6.050% / 96.268% | 6.050% / 96.268% | pass |
| frag4-context1 | A_preserve | 0.707% / 97.296% | 2.907% / 96.570% | pass |
| frag4-context1 | B_preserve | 0.367% / 97.597% | 2.410% / 96.675% | pass |
| frag4-context2 | frozen_seed42 | 0.000% / 94.054% | 0.000% / 94.054% | fail |
| frag4-context2 | A_preserve | 0.000% / 96.490% | 0.000% / 95.401% | pass |
| frag4-context2 | B_preserve | 0.000% / 96.823% | 0.000% / 95.816% | pass |

Floor flags use exact counts (`20*FP <= Nnegative`), not rounded percentages. The CSV retains all three operating rules, exact confusion counts, precision and unchanged AUC/AP.

## Predeclared primary recovery screens

| Arm | Specimen | All contexts meet 95% specificity | Mean sensitivity delta vs baseline (points) | No context loses more than 2 points | Full screen |
| --- | --- | --- | ---: | --- | --- |
| A_preserve | PHerc0814 | pass | -0.0084 | pass | fail |
| A_preserve | PHercParis1Fr39 | pass | -1.5715 | fail | fail |
| B_preserve | PHerc0814 | pass | +0.2169 | pass | pass |
| B_preserve | PHercParis1Fr39 | pass | -1.8203 | fail | fail |

The frozen baseline fails the specificity floor in frag4-context2.

The screen also requires exact equal-context mean sensitivity to be at least the baseline mean. Its Fraction-based flags are taken from the saved analysis; sensitivity improvement versus the same model at the common cutoff does not replace a failed primary screen. Meeting a weak baseline does not establish a useful ink detector.

## Domain macros and fixed secondary analysis

| Specimen | State | Common sensitivity / specificity | Primary sensitivity / specificity | Secondary sensitivity / specificity |
| --- | --- | ---: | ---: | ---: |
| PHerc0814 | frozen_seed42 | 90.317% / 99.967% | 90.317% / 99.967% | 90.317% / 99.967% |
| PHerc0814 | A_preserve | 88.931% / 99.973% | 90.309% / 99.967% | 90.309% / 99.967% |
| PHerc0814 | B_preserve | 88.915% / 99.973% | 90.534% / 99.967% | 90.534% / 99.967% |
| PHercParis1Fr39 | frozen_seed42 | 3.025% / 95.161% | 3.025% / 95.161% | 3.025% / 95.161% |
| PHercParis1Fr39 | A_preserve | 0.354% / 96.893% | 1.454% / 95.985% | 1.454% / 95.985% |
| PHercParis1Fr39 | B_preserve | 0.183% / 97.210% | 1.205% / 96.245% | 1.205% / 96.245% |

Macros give the two contexts in each specimen equal weight. There is no pooled-pixel or cross-specimen headline, pixel-IID interval, diagnostic fit, score transformation or new ranking evidence. Public validation naming, disjoint 2D boxes and sequential computation do not create prospective validation. Optical registration, annotation transfer and upstream-exposure limits remain those of the earlier studies.

Independent numerical verification status and final input/output identities are recorded separately before this reporting package is sealed. This reporting code reads only completed JSON and does not load arrays, checkpoints or models.
