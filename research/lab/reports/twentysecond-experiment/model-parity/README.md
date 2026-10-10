# Native synthetic CPU compatibility harness

This preparation constructs no real model until root releases the exact frozen plan. Each fresh offline worker builds one source-bound factory, strictly loads the Stage1-qualified native safetensors, checks every named value and actual storage alias, and runs the nine fixed fabricated cases. There are eleven full model forwards per worker, no warmups, no optimization and no scientific image input.

The historical comparator executes unchanged selected AST definitions from the pinned historical inference source. Safe library globals replace its broad module imports. Its reader receives an in-memory NumPy array, its original dataset normalizes and checks raw occupancy, and its original run loop, chunk accumulator and uint8 serialization operate over memory. The source extraction deliberately excludes unsafe checkpoint loading, network/Zarr opening, multiprocessing startup, TTA, the CLI and compressed TIFF writer. Instrumentation captures logits before the historical in-place sigmoid. The current comparator executes the retained baseline `infer_surface` with the current focused factory and native authored configuration. This is an explicitly bounded extraction comparison, not the complete old application or its CUDA runtime.

Saved model inputs are canonical NumPy values. Worker metadata separately preserves actual input strides/contiguity and the synthetic uint8 source identity. Normalization ownership differences are covered by toy fixtures: the historical float32 normalizer can alias its input whereas the current version copies. Each actual path receives fresh synthetic data and neither may alter its uint8 source.

The 94 preparation checks use generated arrays and toy models only. Two initial fixture failures are retained: one omitted standard-library binding in the extraction namespace, and one mathematically overstrong assumption about all nonperiodic Hann overlap sums. Both were corrected before freeze without historical source or protocol changes.

All pre/post loaded per-key hashes and exact alias partitions are saved. Inference arrays include selected indices, normalized tensors, pre-sigmoid logits, probabilities, Hann map, schedules, accumulated sums/weights, evaluated mask, raw blended score, uint8 encoding and the author's clipped decoding. Cross-factory encoded discrepancies remain diagnostic if raw floating parity passes. No accuracy, transfer, physical-ink or decipherment claim follows.

Execution after exact root GO:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/workspace/.venvs/herculaneum/bin/python run_parity.py \
  --plan PLAN.json --release ../NATIVE_PARITY_GO.json --output run-v1
```
