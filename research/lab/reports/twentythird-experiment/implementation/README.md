# Fixed native reference comparison

The worker uses the unchanged phase22 current-factory helper and one unchanged extracted `state_receipt` function. It strictly loads the same qualified native safetensors, verifies every parameter byte and actual alias, then runs the retained `run_ink_reference.infer_surface` on the six input-audited surfaces. The returned factory normalizer is bound to its actual pinned source path. All150 normalized hashes must match the retained seed42 capture before their native forward calls. Primary masks remain unchanged and missing support stops execution.

One construction, one strict load and150 model calls are authorized only after root releases the exact frozen plan. There are no warmups, training, thresholds, network requests, copied checkpoints or seed42 model calls. All inputs, code and native state remain immutable; pre/post state receipts and tile logits/probabilities, weights/sums, raw scores, evaluated masks and separate encoded/decoded diagnostics are saved.

The saved-output analyzer computes raw AUC/AP on each original primary mask, separate equal-context Paris1/Paris2 means and native-minus-seed42 differences. Both models receive the prespecified quantization diagnostics. Paired differences use FP64 arithmetic on the original FP32 scores and NumPy linear95th percentile. Ties are grouped exactly; no pixel independence, causal effect or promotion claim follows.

Forty-two first-attempt toy checks cover tied metrics, imbalance, inversion, invalid class/support/shape/value inputs, quantization, pre-forward fingerprint rejection, alias/state corruption and Hann blending. No real factory, checkpoint, scientific array or native forward was used in preparation.

After root GO:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/workspace/.venvs/herculaneum/bin/python run_native.py \
  --plan PLAN.json --release ../NATIVE_COMPARISON_GO.json --output run-v1

PYTHONDONTWRITEBYTECODE=1 /workspace/.venvs/herculaneum/bin/python analyze_saved.py \
  --plan PLAN.json --release ../NATIVE_COMPARISON_GO.json \
  --results run-v1/native/result.json --output analysis-v1
```
