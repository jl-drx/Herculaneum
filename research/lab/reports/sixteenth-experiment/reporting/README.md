# Saved-results presentation helper

`render_comparison.py` reads the sealed inference analyzer's future `metrics.json` and formats saved results using Python's standard library only. It does not load arrays, checkpoints, model code or network clients. Root will execute it only after the independent numerical verification passes.

```sh
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/sixteenth-experiment/reporting/render_comparison.py \
  --metrics /absolute/path/to/metrics.json \
  --expected-metrics-sha256 VERIFIED_METRICS_SHA256 \
  --output-dir /absolute/path/to/new/presentation-directory
```

The output directory must not exist. The helper emits `tables.md`, `cutoffs.csv`, `contexts.csv`, `paris2-equal-fragment.csv`, `specificity-transfer.csv` and a source/output hash receipt. It checks the expected source hash, exact evaluation scope, count totals and specificity flags before formatting; these checks do not replace independent numerical verification.

Markdown rates use percentages with two decimal places; ROC AUC and AP use 0–1 values with four decimal places. CSV rates are unrounded 0–1 fractions, cutoff values use the input Python float representation, counts are exact integers and missing metrics use the literal `undefined`. Zero and false retain distinct values. Paris2 summaries copy the saved equal-fragment macro averages; the helper does not pool counts or silently omit undefined values.

`formatting-check.json` records six tiny synthetic display checks only. It contains no fabricated scientific results. The code freeze pins the helper, these instructions, formatting receipt and the inspected research-design/analyzer schema sources. No real metrics, models or CT inputs have been read during preparation.
