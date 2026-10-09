The verifier independently scores the saved Frag4 comparison. It uses ascending grouped positive/negative concordance for AUC and grouped suffix precision for average precision, rather than the experiment's curve-based metric implementation. Both implementations must agree within an absolute tolerance of 1e-12; frozen-threshold confusion counts must agree exactly.

It checks all 20 fixed runs, file hashes and byte lengths before and after review, source and label-selection pins, retained checkpoint and semantic-state receipts, frozen cutoffs, captured normalized-input fingerprints and tile placement, raw-logit shape and finiteness, and evaluation coverage. Primary participation cannot shrink. Secondary supplied-known participation uses the explicitly reconstructed common evaluated and finite support across all five arms. The two-context macros and all matched preservation and baseline-relative deltas are recomputed from individual contexts.

The checker never imports the experiment metric modules or model code. It does not deserialize checkpoints, decode CT planes, run a model, recalibrate a threshold or access the network. It stream-hashes checkpoint files and checks semantic-state hashes against the frozen worker and source receipts. It does not independently reproduce a model forward, the input normalizer or the numeric Torch sigmoid/Hann blend; those limits are recorded in the result receipt. Actual selected labels and masks are decoded only when root-owned inference and analysis have completed.

Synthetic checks passed: 1,134 exhaustive tied-score fixtures, four boundary/rejection cases, and 13 artifact, tile-placement and support checks. No scientific arrays were read by these checks. Their receipts are retained in this directory.

Run after the root-owned comparison and analysis finish, using their exact final paths:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/result-verification/verify_reference.py \
  --plan /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/final-plan/comparison-plan.json \
  --results /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/run/results.json \
  --metrics /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/analysis/metrics.json \
  --output-dir /workspace/herculaneum-lab/reports/eleventh-experiment/result-verification/final-review
```

The output directory must be new and inside this directory. A failed review preserves a STOPPED receipt and makes no partial verification claim. The supplied optical-reference, unresolved historical exposure and one-specimen limitations remain in force even when numeric verification passes.
