# Retained score audit: ranking and threshold transfer

The retained outputs show **both changed ordering and large operating-point effects**. They do not support interpreting every sensitivity loss as representation damage. This audit uses six existing roles, the same three calibration domains and the two previously examined PHerc0814 diagnostic domains. It selects no new threshold and makes no model or network call.

All 112 used input identities retain their SHA256/size before and after analysis. Recomputed AUC/AP match the frozen eighth/ninth reports within `1.1102230246251565e-16`; existing locked confusion counts agree exactly. Full class distributions, paired shifts, within-class correlations and counts are in [score-diagnostics.json](result/score-diagnostics.json) and [score-distributions.csv](result/score-distributions.csv). The [fixed-scale interval figure](result/class-score-intervals.png) displays descriptive q05–q95 intervals and already locked cutoffs; these intervals are not confidence intervals.

## Fixed calibration still matters

The original primary rule takes the maximum of three local negative-tail cutoffs. Its controlling region changes for C and B_smooth. Values below are the already stored empirical cutoffs, rounded only for this table; the calculations retain their full float32-derived values.

| Role | w029 local | Old 0814 rank01 local | Old rank07 local | Primary |
|---|---:|---:|---:|---:|
| Frozen | .755895 | .641786 | .310271 | .755895 |
| A | .829144 | .824476 | .689913 | .829144 |
| B | .625098 | .354646 | .286109 | .625098 |
| C | .812803 | .859406 | .649517 | .859406 |
| A_smooth | .563550 | .526118 | .411032 | .563550 |
| B_smooth | .625621 | .627627 | .361398 | .627627 |

These cutoffs meet the fixed calibration-negative guarantee. Under `score >= t`, the negative fraction strictly below t is exactly specificity; boundary ties belong to false positives. The two diagnostic regions are much easier for the frozen model than old calibration rank01: its primary produces 0/29,488 and 11/16,561 diagnostic false positives, while its old calibration rank01 sensitivity is only 474/24,582 at that same cutoff. Calibration support and diagnostic support therefore have very different score distributions.

## Ranking and operating points separate

The table averages the two diagnostic contexts equally. “Same frozen number” applies the already fixed `0.7558947205543518` to all roles to expose scale differences; it is not a proposed calibrated policy. Each role's own-primary column preserves its original calibration rule.

| Role | AUC | AP | Own-primary sensitivity | Own-primary specificity | Sensitivity at same frozen number |
|---|---:|---:|---:|---:|---:|
| Frozen | .999608 | .998510 | .903171 | .999668 | .903171 |
| A | .989429 | .976560 | .688869 | .998946 | .844895 |
| B | .987941 | .979877 | .771325 | .997219 | .553129 |
| C | .984691 | .965987 | .559924 | .999678 | .805583 |
| A_smooth | .994717 | .988662 | .736300 | .991793 | .002993 |
| B_smooth | .995598 | .990528 | .690585 | .997592 | .047278 |

Two contrasts are especially informative:

- **Identical C scores, different existing rules:** switching only the reported operating point from its primary .859406 to its retained w029-only secondary .812803 changes mean sensitivity .559924→.718683 and specificity .999678→.998571. AUC/AP are unchanged. This quantifies a cutoff effect without fitting a diagnostic threshold.
- **B_smooth versus B on rank14:** AUC/AP improve .979147/.963994→.995120/.986630, while own-primary sensitivity falls 3,039/4,767→2,368/4,767 (.637508→.496748). Both have zero false positives. The positive median moves .678594→.626510, near the nearly unchanged primary .625098→.627627, even as positive q05 improves .255839→.405565. Thus better ordering at the low-score tail can coexist with fewer positives above the fixed calibration-derived operating point.

## Score shifts are regional and class-dependent

The smoothed models move positives down and negatives up relative to frozen in both diagnostic regions. The table reports **median of paired score differences**, not subtraction of separate medians. IQR ratios compare each class's spread with frozen on exactly the same paired mask.

| Role / region | Paired positive shift | Paired negative shift | Positive IQR ratio | Negative IQR ratio |
|---|---:|---:|---:|---:|
| A_smooth / rank01 | −.105550 | +.038475 | 2.64 | 7.70 |
| A_smooth / rank14 | −.190004 | +.037427 | 9.64 | .96 |
| B_smooth / rank01 | −.069340 | +.048897 | 2.26 | 6.43 |
| B_smooth / rank14 | −.154741 | +.045547 | 8.48 | 1.79 |

The median gap between classes shrinks, but positive within-class dispersion expands relative to the unusually tight frozen positive distribution. Describing this as uniform score compression would miss the tails and regional changes. Unsmooothed A also illustrates the distinction: its rank01 positive median rises .782907→.893650, while positive q05 falls .711170→.565351 and AUC/AP decline. Higher typical positive scores do not ensure retained discrimination or operating-point sensitivity.

Overall diagnostic Spearman correlation with frozen averages .839 for A, .752 for B, .857 for C, .755 for A_smooth and .750 for B_smooth. The paired within-class correlations in the JSON are substantially lower; for example B_smooth's positive Spearman is −.011 on rank01 and −.153 on rank14, despite high AUC. Pooled correlation partly reflects between-class separation, and within-positive ordering need not track label discrimination. No correlation here establishes probability calibration.

## What the scores establish

A strictly increasing score transform preserves AUC/AP and ordering. Their observed changes and paired rank changes show that adaptation is more than such a transform on these fixed domains. Score summaries establish output-ordering drift, class-specific redistribution and cutoff placement; they cannot locate an internal feature change, distinguish head/feature/optimization mechanisms or identify causal representation damage. A preservation continuation remains an experimental hypothesis for the lead to specify, rather than a conclusion or recipe chosen by this audit.

The diagnostic counts remain 49,033 (19,545 positive) and 21,328 (4,767 positive) supplied-supervision cells on two contexts from one specimen. Historical model/teacher exposure and annotation ancestry remain unresolved. These are development retention diagnostics, without independent ink truth, fresh holdout claims, pixel confidence intervals or significance. All earlier studies remain immutable. No training code, recipe, coefficient or new diagnostic threshold was written or frozen.
