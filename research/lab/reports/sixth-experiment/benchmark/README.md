# Fixed CPU comparison package

Prepared only. No model inference, acquisition, training or paid compute was performed by this package during preparation. `comparison-plan.draft.json` is deliberately incomplete and `ready_for_execution=false`.

The final plan must bind the actual corrected w029 data/provenance, explicit decoded binary foreground values, selected source XYWH, verified spacing, a passing four-model contract audit, model artifact pins and all execution sources. The root reviewer freezes a separate final plan after those records pass. Do not edit code or inputs during a run; changes require another version.

The exact four arms are `russo_ex01675k`, `russo_control75k`, `seed42`, and `seed43`. All use the unchanged reference `infer_surface` on pooled layers `[2:19]`, forward, CPU FP32, batch one and four threads. Each runs in a fresh offline subprocess with a hard 120-second wall timeout, 120 CPU-second limit and no retry. The aggregate output bound is 32 MiB, checked after each arm and analysis; per-worker file size is bounded at 8 MiB. There is no GPU launch or fitted checkpoint selection.

Source support must be completely acquired; missing input pixels stop rather than becoming filled context. Fixed primary support is validation membership and actual acquisition support within a 64-pixel crop margin, then common evaluated/finite support across all models. It needs at least 256 supplied positives and negatives before predictions and again after the common support intersection. No inadequate-domain relaxation is allowed. Raw supervision overlap and observed zero profiles are retained; current loader validation subtraction is conditional evidence, not certified historical exposure. Occupancy never serves as geometry validity.

The driver saves raw float32 scores, Boolean evaluated masks, pre-inference support and checkpoint/source/code/runtime metadata. The analyzer saves common masks, tie-grouped ROC/PR curves, exact AUC and noninterpolated average precision, fixed `score >= 0.5` diagnostics, zero/nonzero-profile strata and paired ex016/control differences. Figures use source intensity 0–255 and score 0–1 scales. Integrity is checked before inference and after analysis.

After the final plan is frozen and all inputs exist, the reviewer can run:

```sh
/workspace/.venvs/herculaneum/bin/python -B /workspace/herculaneum-lab/reports/sixth-experiment/benchmark/run_comparison.py --plan /absolute/path/to/final-plan.json
/workspace/.venvs/herculaneum/bin/python -B /workspace/herculaneum-lab/reports/sixth-experiment/benchmark/run_comparison.py --plan /absolute/path/to/final-plan.json --output-dir /absolute/path/to/fresh-output --execute
```

The first command verifies contracts and class counts only. The second performs the four fixed arms sequentially, stops on the first failure and preserves partial outputs. A separate analyzer call requires all four original model outputs and an absent analysis subdirectory; existing analysis is preserved. It does not select a new crop or fit a threshold.

`preparation-checks.json` records 12 passing synthetic metric/mask/gate checks with zero model calls. The root additionally verified 100 direct positive-versus-negative pairwise AUC comparisons. These checks verify the analyzer; they are not scientific model validation.

One label-supported, correlated crop provides a descriptive comparison. It cannot establish untouched test accuracy, authentic ink, correct layer geometry, teacher/physical independence, letters or a reading. Existing w013 integration evidence is not rerun or pooled into this comparison.
