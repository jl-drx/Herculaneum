# Fixed Frag4 comparison

This package prepares CPU inference and descriptive supplied-reference agreement for five retained states on two preselected contexts. It performs no training or reference-based calibration. Forward depth is primary; reversed depth is a separately reported, predeclared diagnostic. Both frozen cutoff rules apply unchanged. All twenty context/model/direction runs must be retained; any failure stops without retry, domain shrinkage, model substitution or choosing the better orientation.

Root supplies the pooled descriptor and executes the following after code/data review:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/prepare_reference.py \
  --descriptor /ABSOLUTE/POOLED-DESCRIPTOR.json --output-dir /NEW/FINAL-PLAN \
  --review /ABSOLUTE/DATA-REVIEW.json --review /ABSOLUTE/CODE-REVIEW.json --ready

PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/run_comparison.py \
  --plan /NEW/FINAL-PLAN/comparison-plan.json --preflight

PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/run_comparison.py \
  --plan /NEW/FINAL-PLAN/comparison-plan.json --output-dir /NEW/RUN

PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/eleventh-experiment/comparison/analyze_reference.py \
  --plan /NEW/FINAL-PLAN/comparison-plan.json --run-dir /NEW/RUN --output-dir /NEW/ANALYSIS
```

The descriptor status is `pooled_reference_complete_no_inference`, and its preprocessing contains the exact `PREPROCESSING` fields from `prepare_reference.py`. Each case supplies `id`, `native_bbox_yxhw` and file records for `surface`, `labels`, `primary_mask`, `known_mask`, `surface_mask`. Records have absolute `path`, `sha256` and a size under `size_bytes`, `byte_length` or `bytes`. Root preprocessing uses exact uint32 sums of 27 source uint16 values, float64 division by 27×257, and one float32 cast; the runner preserves float32 input and imports unchanged `infer_surface`, rather than its uint8-only CLI. Layer start is zero on the new 17-plane input.

For each run, `scores.npy` is the unchanged Hann-blended sigmoid map, `tile-logits.npy` contains original FP32 scored-tile logits in metadata order, and `evaluated-mask.npy` distinguishes skipped tiles. Captured normalized-input SHA256s are taken before the model call and checked afterward. These are raw tile logits, not a logit transformation of a blended probability. State and source identities are checked before and after inference.

`analysis/metrics.json` uses `per_context[direction][case].domains[primary|secondary_known].models[role]`, `equal_context_macro[direction][domain][role]`, and `macro_deltas[direction][domain]`. It retains model-specific primary/secondary operating metrics, matched preservation deltas and all baseline-relative deltas. A fixed primary pixel must be finite and evaluated for all states; secondary-known coverage is reported separately. No success gate or confidence interval is fitted to two contexts of one historically uncertain specimen.

Preparation performed syntax/static review only. Actual pooled arrays, checkpoint deserialization, factory construction and inference are root-owned execution steps. Source provenance and interpretation are governed by the root's frozen `RESEARCH_DESIGN.md`.
