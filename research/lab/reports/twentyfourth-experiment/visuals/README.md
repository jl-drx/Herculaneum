# Phase24 fixed depth-reversal panels

Preparation only. No source arrays, model outputs, checkpoints, plotting imports or networks have been accessed. The plotter derives from the sealed phase23 `plot_contexts.py` (SHA256 `a0cc443167fa06ceb310eae83043e97d53dca52cce10ea898aec38dc4380eaf9`); it writes exclusively under this new phase24 visual directory. Its phase24 protocol binding must be finalized before root releases execution.

Four PNGs cover seed42/native × Paris1/Paris2. Paris1 has both complete contexts; Paris2 has all four. Each row displays supplied reference labels, forward-depth raw scores, reversed-depth raw scores and reverse-minus-forward differences. Probability scales remain 0..1 and difference scales −1..1 for both models and every context. All four panels use the original primary mask with gray outside it. No crop, contrast adaptation, fitted threshold, model, metric or letter interpretation is produced.

Forward seed42 scores remain bound by the sealed phase23 input contract. Forward native scores come from the exact sealed phase23 result (`c38ad03a287af2c20f365eb1a5edd63f40639b2a0c7dd61922a60abd1493ecf9`). The phase24 implementation agent specified separate reverse results at `implementation/run-v1/{seed42,native}/result.json`; each contains `runs[case_id]`, `id`, `model`, `direction="reverse"` and `arrays.raw_scores/evaluated_mask` with the same full file/raw-array descriptor schema used in phase23.

After root release, the plotter requires both verified reverse-result hashes on its CLI. It checks the exact six-case sets/order, model/direction identities, source-file and raw-array hashes, shape/dtype, original support counts, full evaluated coverage and finite [0,1] scores. It binds 60 consumed arrays, metadata, code and all four images in a receipt and rechecks source hashes after rendering. The entire visual stage, including source files, caches, panels and receipts, is capped at 32 MiB.

Expected output names are `seed42-paris1-depth-reversal.png`, `seed42-paris2-depth-reversal.png`, `native-paris1-depth-reversal.png`, `native-paris2-depth-reversal.png` and `PLOT_RECEIPT.json`. Root reviews the images and retains scientific interpretation. The figures will be marked supplied-reference/reused-development diagnostics.

No runtime is authorized by this preparation note. The intended command after root release is:

```bash
PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python \
  /workspace/herculaneum-lab/reports/twentyfourth-experiment/visuals/plot_depth_reversal.py \
  --reverse-seed42-result /workspace/herculaneum-lab/reports/twentyfourth-experiment/implementation/run-v1/seed42/result.json \
  --reverse-seed42-result-sha256 ACTUAL_VERIFIED_SEED42_REVERSE_RESULT_SHA256 \
  --reverse-native-result /workspace/herculaneum-lab/reports/twentyfourth-experiment/implementation/run-v1/native/result.json \
  --reverse-native-result-sha256 ACTUAL_VERIFIED_NATIVE_REVERSE_RESULT_SHA256 \
  --output-dir /workspace/herculaneum-lab/reports/twentyfourth-experiment/visuals/render-v1 \
  --root-release
```
