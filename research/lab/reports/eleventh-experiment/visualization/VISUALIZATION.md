# Numeric figures for the completed Frag4 comparison

The corrected figures show every fixed state and both predeclared directions on primary reference support:

- [Ranking agreement — PNG](results-v2/ranking-agreement.png), [PDF](results-v2/ranking-agreement.pdf), [SVG](results-v2/ranking-agreement.svg): ROC AUC and average precision, per context and equal-context mean.
- [Retained operating thresholds — PNG](results-v2/fixed-thresholds.png), [PDF](results-v2/fixed-thresholds.pdf), [SVG](results-v2/fixed-thresholds.svg): sensitivity and specificity at each state's previously fixed **primary** threshold, for both directions.
- [Exact numeric table](results-v2/numeric-results.csv) and [compact numeric input](results-v2/numeric-input.json).

Forward is the primary direction; reverse is secondary. Each panel includes the frozen seed42 baseline, A_smooth, B_smooth, A_preserve and B_preserve with consistent state colors. Circles/squares represent the two contexts; diamonds represent their equally weighted mean, rather than pooled pixels. Axes remain 0–1 throughout, means in state labels are rounded to three decimals, and the CSV retains full precision. No uncertainty intervals are claimed from two contexts in one specimen. Average precision also depends on positive prevalence, which is preserved for each context in the compact input. The threshold figure does not make sensitivity comparisons at matched specificity; the thresholds were fixed separately for each state before Frag4 predictions.

The interpretation remains descriptive supplied-reference agreement on one exposed specimen, with unresolved historical exposure and registration uncertainty. These are numeric results only: no source-image overlays, hidden text, readings or model selection appear in the figures.

Root confirmed that the actual twenty-run comparison and analysis were complete before rendering. The exact 166,792-byte metrics input has SHA256 `ea510202c8c95cdea09377213e0103fb6d0a57818b6b57f6699d20289d83d3f7`. Both independently completed verification reports are PASS and reference that same metrics hash:

- `review/comparison/completed-comparison-verification.json`: SHA256 `299967b698ed0b6db808901d46cd6df6a53d943dedd0989bcdeec9b51c104c6a`.
- `result-verification/final-review/independent-verification.json`: SHA256 `d586796714ee14b51838da1a03d1eb83b771bf5477ab9ff5608d1a43b90760ea`.

The plotter independently verifies all five roles, twenty runs, complete primary participation, finite metrics and exact consistency between per-context values and equal-context means. It records source and code hashes before and after rendering. No source arrays/checkpoints/models or network were accessed by the visualization task.

The initial figures had a legend/title collision and some mean labels close to context markers. They and their original code/receipts remain under `results/` and `plot_comparison.py`. The additive `plot_comparison_layout_v2.py` and `results-v2/` move the legend into a separate header band and place mean values under state labels. **Both numeric input JSON and CSV are byte-identical between layouts.** The initial render emitted nonfatal Fontconfig cache messages; the corrected render uses a writable private cache and completed without them. Caches are incidental runtime files, excluded from the evidence manifest.

[OUTPUT_RECEIPT.json](results-v2/OUTPUT_RECEIPT.json) pins the corrected code, input, Matplotlib version and all eight numeric artifacts. [REVIEW_RECEIPT.json](REVIEW_RECEIPT.json) captures final numeric/visual checks and independent evidence. [MANIFEST.json](MANIFEST.json) seals the presentation package while preserving the initial version.
