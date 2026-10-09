# Paired normalized-depth reversal

This reuses the phase23 inference, strict-state inspection, source binding and saved-output metric structure. The intervention occurs inside `Capture`: verify the unchanged forward-normalized tensor, flip only normalized depth axis 2, make it contiguous, verify the prospective reversed hash, and call the model. Both tensors are checked for mutation afterward. The unchanged adapter always reads forward. Raw-reversal normalization is a separate input-audit diagnostic, never the actual model path.

Two fresh sequential workers construct the fixed seed42 and native models once each, with 150 forwards per model. Seed42 uses one `torch.load(weights_only=True,map_location='cpu')` of its pinned original artifact, exact `model` container, exact bound config/step and original semantic state. Native uses one qualified safetensors load and must match its retained forward-state identity. Both use the qualified current factory and exact 508-key FP32 state/244-alias inspection before and after. Unreviewed deserialization and all network calls are blocked.

The analyzer reuses the sealed forward controls and reports all four model/direction combinations on the unchanged primary masks. It retains reverse-minus-forward AUC/AP for each model, native-minus-seed42 difference of those changes, paired probability drift and separate equal-context parent means. No direction is selected, and no quantized metrics, thresholds, training, extra forwards or checkpoint copies are introduced.

Forty-nine preparation checks passed on the first attempt using generated arrays, toy models and metadata only. They include exact permutation/value preservation, both hash guards, wrong-axis rejection, mutation detection, counts, state aliases, missing primary support, tied metrics, imbalance and signed difference of changes. No real factory, checkpoint, scientific array or model output was used in preparation.

After root GO and independent technical review:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/workspace/.venvs/herculaneum/bin/python run_native.py \
  --plan PLAN.json --release ../PAIRED_REVERSAL_GO.json --output run-v1

PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python analyze_saved.py \
  --plan PLAN.json --release ../PAIRED_REVERSAL_GO.json \
  --results run-v1/controller-result.json --output analysis-v1
```
