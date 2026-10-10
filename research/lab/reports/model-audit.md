# `ink_9um` audit for a bounded PHerc0800 CPU experiment

Audit date: 7 October 2026. Initial work was metadata and source inspection only; the standalone adapter validation added below subsequently verified checkpoint loading and a synthetic forward pass. No target surface pixels were scored by this audit. The root setup task handled dependency installation and checkpoint downloads.

## Recommendation

The official hybrid detector has a CPU execution path and is a reasonable baseline candidate for a **256 × 256 surface ROI**, retaining all 31 depth layers. Use a chunk-aligned ROI from the observed `[31, 980, 980]` uint8 surface array; run 17-layer, 128 × 128 patches with batch size 1, no compilation, and no data-loader subprocesses. Do not pass the full remote surface to the stock CLI: it has no XY crop option and schedules the whole input. A bounded local crop must come first.

The focused adapter now passes model construction, strict seed42 state-key loading and a finite synthetic CPU forward pass. Target-surface inference, resource use and checkpoint agreement still require validation by the experiment runner before scientific interpretation. An arbitrary raw CT chunk is not suitable model input: this detector requires a surface-aligned volume whose geometry, physical sampling and depth orientation are documented.

## Pinned official artifacts

Hugging Face model repository: [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um), revision **`7109667e2607db1b90c37c8b09cb876ea7fe7bb1`**. API metadata reports public/non-gated access and last modification `2026-08-14T01:02:42Z`. The model card declares **MIT**. There is no separate license file in the inspected file inventory; retain the card declaration and attribution. This model declaration does not change CT or label dataset terms.

| Suggested checkpoint | Advertised bytes | Advertised LFS SHA-256 |
| --- | ---: | --- |
| `hybrid_3d2d-seed42/step-075000.pth` | 138,360,039 | `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab` |
| `hybrid_3d2d-seed43/step-075000.pth` | 138,360,231 | `2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f` |

Each file is about 132 MiB; together about 264 MiB. These hashes initially came from the official model API; the root fetch subsequently verified both files, and the adapter validation independently verified seed42 before loading. The files use PyTorch checkpoint serialization; pinned upstream `load_checkpoint` explicitly calls `torch.load(..., map_location="cpu", weights_only=False)`. Our adapter uses `weights_only=True` instead.

Pin download URLs to this HF revision, for example:

```text
https://huggingface.co/scrollprize/ink_9um/resolve/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/hybrid_3d2d-seed42/step-075000.pth
https://huggingface.co/scrollprize/ink_9um/resolve/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/hybrid_3d2d-seed43/step-075000.pth
```

Both seeds also provide steps 10k, 20k, 30k, 40k, 50k and 60k. Choosing a different checkpoint because it looks more like expected text would bias evaluation; declare selection criteria and use held-out evidence. Final-step checkpoints are a simple initial default, not a claim that they are best.

## Architecture and input contract

Code revision: [`ScrollPrize/villa@e0bbb8b40a2db58b1d71864f286eb85717e59e64`](https://github.com/ScrollPrize/villa/tree/e0bbb8b40a2db58b1d71864f286eb85717e59e64). Exact recipe: [`vesuvius/src/vesuvius/ink_detection/configs/aligned21_hybrid_3d2d.json`](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/configs/aligned21_hybrid_3d2d.json).

- Model type: `vesuvius_unet_3d_stem_2d`; input **`[batch, 1, 17, 128, 128]`**, float32 after normalization. There is no 160- or 256-pixel model tile in this recipe.
- Stem: two 3 × 3 × 3 Conv3d layers, 16 channels, affine InstanceNorm3d and LeakyReLU. A learned 1 × 1 × 1 attention head produces depth-softmax weights. Attention-weighted sum and max over depth concatenate into **32 2D channels**.
- Body: autoconfigured 2D U-Net, `BasicBlockD` encoder and `ConvBlock` decoder; the published tutorial describes six stages. For the shipped 128-pixel isotropic recipe, the builder's channel schedule is 32, 64, 128, 256, 320, 320. One `ink` output channel, no model activation; inference applies sigmoid to logits. Treat the actual embedded checkpoint configuration and loaded state as authoritative.
- Nominal input: approximately 9 µm **isotropic surface sampling**. Native training examples are 9.362 µm; pooled 2.399 µm sources use level-2 XY and mean pooling of four Z slices, approximately 9.596 µm. The observed PHerc0800 8.640 µm input is close but differs by roughly 8–10% from these training scales. Record this domain shift. The model does not automatically verify or resample physical spacing. Do not apply the 2.4 µm pooling recipe to an already-native 8.640 µm surface.
- Recipe spacing is `(1,1,1)` for model autoconfiguration, not a declaration that physical voxel spacing equals one micrometre.
- Training used jittered **17-of-21** depth windows, offset ±2, without padding. Preserve enough real layers for comparison; padded short-depth inputs change the training contract even though the inference reader can center-pad them.

### Exact normalization

The embedded/shipped configuration uses `robust_mad`, percentiles 1 and 99. For each complete 17 × 128 × 128 patch, the shared `normalize_robust` function:

1. Copies to float32, computes pre-clipping standard deviation/minimum/maximum, and clips at the patch's 1st/99th percentiles.
2. Subtracts the clipped patch median and divides by **1.4826 × median absolute deviation**.
3. If that scale is non-finite or below `1e-6`, uses the first suitable fallback: pre-clipping standard deviation, then half the percentile span, then 1.
4. Replaces non-finite results with zero.

This is patchwise normalization over depth and XY together; it is not independent normalization of each slice, division by 255, or the display helper's percentile contrast operation. Preserve original uint8 data. Compute occupancy before normalization as the stock reader does, and do not infer ink from normalized texture alone.

## Bounded ROI and depth controls

Observed surface metadata supplied by the root task: shape `[31,980,980]`, chunks `[31,128,128]`, uint8, uncompressed. It is correctly depth-first for the upstream heuristic (the smallest dimension is axis 0).

| Local ROI | Chunk-aligned source chunks | Raw source bytes | 128-pixel windows at stride 64 |
| --- | ---: | ---: | ---: |
| 31 × 256 × 256 | 4 | 2,031,616 | 9 |
| 31 × 384 × 384 | 9 | 4,571,136 | 25 |

Counts assume full 128 × 128 chunks away from the last array boundary. Keep the crop's source origin, volume/segment identity, Zarr metadata and mesh/surface mapping. Choose a nonempty mesh-valid region rather than arbitrary texture. A 256-pixel ROI is sufficient for a low-cost runtime test; review its central 128 × 128 area to reduce edge effects. A 384-pixel ROI provides a larger interior if that initial crop contains a useful candidate. Neither area supports a prize-area claim without verified physical surface scale and geometry.

For all 31 layers, the upstream default centered selection is **`7:24`**, indices 7 through 23. Explicitly record `--layer-start 7 --layer-end 24`. Before a reading claim, compare predefined neighboring windows **`5:22`** and **`9:26`**, then inspect forward/reverse depth direction when orientation is unresolved. Do not silently pick the most text-like control result. The CLI clamps layer ranges and can fall back to all layers for an invalid range; prevalidate exact bounds and depth instead of relying on that behavior.

Suggested upstream execution flags, after runtime/source compatibility has been validated:

```text
python -m vesuvius.ink_detection.inference.infer LOCAL_BOUNDED_SURFACE.zarr CHECKPOINT.pth FRESH_OUTPUT.tif
  --resolution 0 --layer-start 7 --layer-end 24
  --batch-size 1 --num-workers 0 --overlap 0.5 --blend-mode hann --no-compile
```

This is a recipe, not an executed command. The stock CLI selects CPU when CUDA is unavailable; it has no explicit `--device cpu` switch. A focused adapter can explicitly select CPU. CUDA autocast is disabled on the CPU path, so prediction is FP32. Avoid `--gpus`. For a small ROI, disable compilation and initially omit mirror TTA to keep the validation interpretable and inexpensive.

## Runtime compatibility and pins

The complete `vesuvius` package declares **Python `>=3.14,<3.15`** and models extra `torch>=2.12,<3`. Its standard installation therefore does not target the existing Python 3.12 lab. That package-level restriction does **not** establish a requirement to port the small hybrid model to Python 3.14: all inspected focused sources parse under Python 3.12, and official CPU wheels for PyTorch 2.12 exist for CPython 3.12.

A conservative candidate for a separate focused CPU adapter is:

| Dependency | Candidate pin / evidence |
| --- | --- |
| Python | Existing 3.12; focused source imports and a synthetic CPU forward pass now verified |
| torch | `2.12.0+cpu`, official Linux x86_64 CPython 3.12 wheel |
| torchvision | `0.27.0+cpu`; PyPI 0.27.0 requires `torch==2.12.0` |
| timm | `1.0.20`, matching the pinned villa lower bound |
| einops | `0.8.1`, matching the pinned villa lower bound |
| huggingface-hub | `0.35.0`, matching the pinned villa lower bound |
| NumPy, TIFF/Zarr/imagecodecs | Retain/reconcile the existing lab pins; resolve and hash-pin the complete adapter dependency graph |

The general network builder eagerly imports alternative backbone wrappers even for this small U-Net. Those imports pull in `timm`, `einops`, and `huggingface_hub`; timm in turn requires torchvision, PyYAML, and safetensors. Upstream CLI/package initialization additionally touches core data utilities and can require requests, pynrrd, aiohttp and related dependencies. A source-preserving focused loader should avoid unnecessary package initialization and retain upstream MIT attribution, or use the official full-package runtime. Do not bypass the full package's declared Python requirement by pretending it is a standard supported 3.12 installation.

Official CPU wheel index evidence:

- [`torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`](https://download.pytorch.org/whl/cpu/torch/), advertised SHA-256 `5e3dc83725581fa38b7b2e45c58692e30b2a3cde19191af54b675ffcac3840a6`.
- [`torchvision-0.27.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`](https://download.pytorch.org/whl/cpu/torchvision/), advertised SHA-256 `912a8b008d95fc9f8f8507e0663238aa01abfeb29f38e218116217560a1b6401`.

No CPU throughput or peak-memory claim is warranted yet. One FP32 input patch is about 1.06 MiB; activations, model parameters and deserialized checkpoint payload use substantially more. Benchmark one patch first, then the nine-window ROI, record thread count and memory, and extrapolate only from observed timings.

## Output interpretation and compatibility gates

- `configure_model` reconstructs `InkConfig` from the checkpoint. Its weight-selection priority is `ema_model`, `state_dict`, `model_state_dict`, then `model`. Record the selected container.
- **Material compatibility gate:** `load_flat_inference_state` uses `strict=False` and logs counts. Require **zero missing and zero unexpected keys** before treating any output as a recovered signal. A command can otherwise complete with partly random or ignored weights.
- Inference applies sigmoid and averages overlapping patch outputs with the specified blending map. Default overlap is 0.5; `auto` chooses Hann for overlapping patches. TTA is optional, not default.
- **The current stock flat CLI writes uint8 TIFF, not lossless float probabilities.** It divides accumulation by weights, clips to `[0,1]`, multiplies by 255 and truncates to uint8. Recover only an approximate probability with `p≈value/255`. A focused adapter should retain float32 probabilities separately for quantitative comparisons.
- BCE label smoothing is 0.5: target levels are 0.25 background and 0.75 ink. For display only, use `clip((p-0.25)/0.5,0,1)` and record it. These outputs are not calibrated empirical probabilities of authentic ink, and display values are not quantitative outputs.
- Preserve separate seed maps, their mean, absolute seed difference, and declared depth/orientation control results. Agreement between two seeds trained on the same corpus is a stability diagnostic, not independent validation of writing. Bright strokes need image/geometry review against fibers and artifacts.
- Use fresh output paths: the stock TIFF writer can replace an existing output. Input sources should remain immutable and byte-verified.

## Training provenance and leakage

The model card and pinned recipe identify four training scrolls: **PHerc0139, PHerc1667, PHercParis4 and PHerc0814**. The released recipe contains 29 representations across 25 physical segments: 24 pooled 2.4 µm representations (9/6/8/1 by scroll) plus 5 native 9.362 µm PHerc0139 representations, four of which duplicate physical segments in the pooled set. Batches use per-scroll quotas 29/22/11/2 of 64. The two seeds change the random seed, not the underlying corpus.

PHerc0800 is absent from the stated/shipped training list, which supports calling it a candidate cross-scroll target. It does not establish independent accuracy or complete checkpoint provenance. Inspect actual embedded configurations after verified download; retain segment/representation identity and check labels/pretraining exposure before claiming a held-out benchmark. The official worked example PHerc0139/w035 is explicitly in the training set and only demonstrates pipeline operation.

On an unread PHerc0800 ROI with no trusted ink labels or vetted reading, report only model behavior, stability and image-supported review observations. Do not report ink precision/recall, transcription accuracy, or recovered letters from model agreement alone. Keep all crops, overlapping layers, alternate renders and transforms of the same physical region in the same evaluation group.

## Sources inspected

- [HF model API including file sizes and LFS hashes](https://huggingface.co/api/models/scrollprize/ink_9um?blobs=true), resolved to the revision above.
- [Pinned HF model card](https://huggingface.co/scrollprize/ink_9um/blob/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/README.md). Its `koine_machines` CLI names the earlier branch; use the pinned maintained `vesuvius.ink_detection` namespace instead.
- [Pinned official ink tutorial](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/docs/07_tutorial5.md).
- [Pinned flat inference implementation](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/inference/infer.py), [CPU/runtime selection](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/inference/inference_runtime.py), [checkpoint loader](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/models/checkpoint.py).
- [Pinned hybrid model](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/models/hybrid_3d2d.py), [network builder](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/models/build/build_network_from_config.py), [exact robust normalization](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/image_proc/intensity/normalization.py).
- [Pinned package requirements](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/pyproject.toml), [torch 2.12 metadata](https://pypi.org/pypi/torch/2.12.0/json), [torchvision 0.27 metadata](https://pypi.org/pypi/torchvision/0.27.0/json), and the linked official CPU indexes.

Small pinned source downloads were stored under `/tmp/herculaneum-model-source`; the HF API response is `/tmp/herculaneum-ink9um-model-api.json`. The 30 unmodified modules required by the focused adapter are now retained under `/workspace/herculaneum-lab/vendor/villa-ink-baseline`, together with the MIT license and `source-manifest.json`. Each retained source file was checked against the pinned official Git tree's blob SHA-1 and has a recorded SHA-256.

## Standalone adapter validation update

`/workspace/herculaneum-lab/run_ink_baseline.py` imports these focused modules through an isolated namespace. It does not install the full `vesuvius` package or execute upstream package initialization. The adapter requires source-manifest integrity, input/provenance SHA agreement, pinned final-checkpoint SHA/size, `torch.load(weights_only=True)`, and an exact `strict=True` state match. It explicitly uses CPU FP32, four threads by default, deterministic algorithms, the embedded checkpoint config, upstream robust normalization, nine 128-pixel windows at stride 64, and the upstream Hann blend semantics. Per-seed float32 scores remain unquantized; optional two-seed results include a mean and absolute difference. It rejects an existing output directory and has no remote-data or training operation.

Validated in a separate subprocess with Python 3.12 and the proposed CPU dependencies:

- Focused source imports succeeded without additional package installation.
- The pinned seed42 checkpoint passed its size and SHA check, loaded with `weights_only=True`, and matched every upstream model key with `strict=True`.
- The reconstructed model contains **34,546,498 parameters**. Its actual embedded configuration has `patch_size=crop_size=[17,128,128]`, `stitch_factor=1`, `use_stitched_forward=false`, and the expected `robust_mad` 1/99 normalization.
- One all-zero synthetic `[1,1,17,128,128]` input produced finite float32 `[1,1,128,128]` logits. Its measured forward time was **0.261 seconds at four threads**; this single synthetic timing does not establish whole-experiment throughput or peak memory.
- No target image pixels were read or scored in this validation. A synthetic forward pass validates runtime operation, not ink recovery or ancient writing.

The root task has separately reported both checkpoint downloads verified. The adapter still validates each supplied file on every run. Example from the lab directory:

```bash
python run_ink_baseline.py data/experiments/pherc0800-roi256/surface.npy \
  --provenance data/experiments/pherc0800-roi256/provenance.json \
  --output-dir reports/ink9um-center-forward \
  --seed42-checkpoint data/models/ink9um-seed42-step075000.pth \
  --seed43-checkpoint data/models/ink9um-seed43-step075000.pth
```

Default depth indices are 7:24, forward. Use `--layer-start 5` or `--layer-start 9` and `--direction reverse` for predefined controls, each with a fresh output directory. Model scores and score stability remain hypotheses until source geometry and image evidence support an independently reviewed reading.

## Bounded full-segment runner update

The original 256-pixel adapter is retained unchanged as `run_ink_baseline.roi256.py`, SHA-256 **`3df9720c825337760c9899bd780a52cc9b6115dadf9a3d923880c9bade385a0a`**, so existing experiment metadata can be matched to its source. The current `run_ink_baseline.py` accepts uint8 **`[31,H,W]`** inputs with **128 ≤ H,W ≤ 1024**, retaining exact embedded 17 × 128 × 128 model patches. It uses stride 64 and appends a final complete tile at `dimension - 128` when needed. A 980 × 980 surface therefore has 15 starts per axis, including 852, and **225 scheduled tiles per checkpoint**. No source pixels are silently cropped and no source-edge padding is introduced.

An optional `--mask MASK.npy` accepts only a Boolean `[H,W]` array. It records the input mask path/hash and retains `validity-mask.npy`. Completely masked tiles may be skipped; the mask never changes the raw patch or its upstream robust normalization. Invalid output pixels are **NaN**, excluded from reported finite-value summaries. Mask/source alignment and mesh validity remain explicit caller-supplied prerequisites.

Entirely zero raw selected-layer tiles are skipped following the upstream empty-input policy. Valid pixels receiving no model contribution remain zero, with **`evaluated-mask=false`**, and are explicitly counted as unscored. Per-seed evaluated masks distinguish this policy output from a learned background score. Model summaries include a separate summary restricted to evaluated pixels; interpreting a skipped pixel as a confident no-ink prediction would be incorrect.

Coverage invariants passed for 128 × 128, 129 × 140, 256 × 256, 980 × 980 and 1024 × 1024 synthetic dimensions: every source pixel is covered by a scheduled complete tile, and the final tile reaches each boundary. Strict verified seed42 loading and a finite `[1,1,128,128]` synthetic CPU forward also passed after generalization. Evidence: `reports/runner-generalization-validation.json`. No target surface pixels were read or scored during these checks. No environment dependencies were changed.

The generalized adapter captures its code hash at process startup, retains fresh-directory protection and source hashes, and preserves per-seed float32 scores and raw seed differences. For a caller-verified full-segment mask, add `--mask PATH` to the earlier command and pass the separately retained full surface/provenance paths.
