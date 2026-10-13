# Complete fixed-map numeric summary

All 48 map views and 168 distinct pairs were included. Each pair has one
full-strip score and three fixed-cell scores, for 672 scope scores. Every score
uses the inherited common support: 786,240 known pixels and 192 unknown pixels.
Differences are in uint8-equivalent probability levels; no contrast was fitted.

## Full-strip ranges across all prescribed pairs

| Comparison | Pairs | Pearson range | Mean absolute difference range |
|---|---:|---:|---:|
| padding | 24 | 0.590 to 0.940 | 2.39 to 10.14 |
| order | 24 | -0.046 to 0.418 | 7.60 to 29.52 |
| cross_model | 72 | -0.035 to 0.505 | 10.76 to 85.33 |
| depth | 48 | -0.117 to 0.157 | 8.89 to 32.76 |

## Padding comparison by model

Each range contains both input orders at all three depths (six comparisons).

| Model | Pearson range | Mean absolute difference range |
|---|---:|---:|
| dnative | 0.811 to 0.933 | 4.19 to 5.69 |
| hybrid_3d2d-seed42 | 0.590 to 0.768 | 7.36 to 10.14 |
| hybrid_3d2d-seed43 | 0.592 to 0.829 | 5.93 to 7.95 |
| hecate | 0.779 to 0.940 | 2.39 to 9.67 |

## Cross-model comparison

Each range contains three depths, both padding phases and both orders (12
comparisons). Shared training and calibration limit interpretation.

| Models | Pearson range | Mean absolute difference range |
|---|---:|---:|
| dnative / hybrid_3d2d-seed42 | 0.154 to 0.505 | 13.68 to 20.06 |
| dnative / hybrid_3d2d-seed43 | 0.203 to 0.463 | 13.74 to 22.82 |
| dnative / hecate | 0.080 to 0.395 | 76.07 to 85.33 |
| hybrid_3d2d-seed42 / hybrid_3d2d-seed43 | 0.115 to 0.357 | 10.76 to 15.16 |
| hybrid_3d2d-seed42 / hecate | -0.007 to 0.204 | 65.25 to 74.40 |
| hybrid_3d2d-seed43 / hecate | -0.035 to 0.378 | 63.24 to 71.11 |

The full-precision machine summary also contains all fixed-cell group ranges
and each model’s order/depth ranges. Individual comparisons remain in
`METRICS.json`. Responses change substantially between sampled depths; that
does not identify the ink-bearing surface. Padding agreement alone is not
accuracy, and reversal invariance was not required.

## Deterministic verification

- All 21 checkpoint archives and 7,080 restored members passed checksums.
- All 32 shifted map combinations, 3,096 batch receipts, 200 output hashes
  and 16 baseline hashes passed independent completion validation.
- Independent reporting verification recomputed all 672 scope scores; maximum
  differences were 3.33e−16 in Pearson and 1.42e−14 in absolute-difference levels.
- Every source pixel in all 64 exact panels (192 depth-region views) matched
  fixed rounding and brown unknown pixels. All 48 overview tiles also matched
  their declared uniform spatial reduction.
- The original reporting output manifest contains 71 verified files; later
  verification and scientific-review documents need their own package manifest.

No new model inference, acquisition, paid compute or scientific accuracy
measurement was performed. Visual conclusions are recorded separately.
