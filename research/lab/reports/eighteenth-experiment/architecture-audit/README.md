# Seed42 architecture and phase18 residual-branch audit

This is a source and retained-JSON audit only. No checkpoint or array was decoded, model constructed or run, numerical training performed, or network accessed. All writes are confined to this directory. The parent selected the 801-parameter additive-logit branch below; the executable experiment plan is authoritative for training, data, resource limits and gates.

## Exact existing contract

The retained strict loader is `reports/sixth-experiment/model-contract/model_contract_v2.py`, functions `load_runtime` (line 82) and `build_model` (line 157). It verifies the reference adapter and focused vendor source manifest before importing `InkConfig`, `make_model` and `normalize_robust`. It verifies the published checkpoint's size and SHA, requires a bounded PyTorch ZIP, uses `torch.load(map_location='cpu', weights_only=True)`, selects only the `model` container, validates embedded configuration, shape/dtype/alias equivalence, finite tensors and strict state loading, then returns a CPU FP32 model in evaluation mode. There is no EMA, pretrained-backbone substitution, added depth padding, or non-strict repair.

Seed42 is `/workspace/herculaneum-lab/data/models/ink9um-seed42-step075000.pth`, 138,360,039 bytes, SHA-256 `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`, released revision `7109667e2607db1b90c37c8b09cb876ea7fe7bb1`. The filename says 75000; the retained embedded step is 74999. Retained metadata records 34,546,498 unique parameters, 138,185,992 unique tensor-storage bytes, 508 named state tensors and 264 unique storage groups; 244 groups contain exact aliases. Count `model.parameters()`, not duplicated state-dict entries. The initial semantic state SHA is `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`.

Inputs are FP32 `[B,1,17,128,128]` patches, normalized once over the entire 17-plane patch. `normalize_robust` clips to the patch's 1st/99th intensity percentiles, subtracts the clipped median and divides by 1.4826 times clipped MAD, with the exact retained fallback for near-zero MAD. It does not normalize each depth plane independently. Preserve this normalizer unchanged; do not normalize the derived difference and coordinate channels again. Raw CT means the unaltered CT pathway through this established normalizer, not removal of that existing clipping operation.

The phase13 schedule reads acquired 21-plane surfaces at `[2:19]`, leaving all 17 selected model planes in increasing published source Z. The nominal surface/label center is acquired index 10, hence model index 8. Phase16 Paris inputs already have 17 pooled planes, selected `[0:17]`; their source centers are 8,11,...,56 with nominal center 32 and pooled pitch 9.72 micrometres. No spatial or depth support change is needed.

The actual model is `Local3DStem2DUNet` in `vendor/villa-ink-baseline/vesuvius/ink_detection/models/hybrid_3d2d.py`:

1. Conv3d 1→16, kernel 3, padding 1, no bias; affine InstanceNorm3d; LeakyReLU(0.01).
2. Conv3d 16→16, kernel 3, padding 1, no bias; affine InstanceNorm3d; LeakyReLU(0.01).
3. Conv3d 16→1, kernel 1, produces depth-softmax attention. Weighted sum and depth maximum of the 16 feature channels are concatenated to `[B,32,128,128]`.
4. A 2D U-Net has six encoder stages, widths `[32,64,128,256,320,320]`, BasicBlockD counts `[1,3,4,6,6,6]`, stride `[1,2,2,2,2,2]`, 3×3 kernels, affine InstanceNorm2d and LeakyReLU(0.01). Five ConvBlock decoder stages use transposed-convolution upsampling and skip features. The final 1×1 head is 32→1, with bias and no output activation.

The stem already learns local axial filters and depth attention. The proposed branch tests an explicit difference/coordinate representation; it is not the first depth-aware operation in the model. The released attention projection was initialized to zero when originally constructed, but loaded checkpoint values must never be reset to that original initialization.

The output mapping is `{'ink': logits}` where the selected tensor is `[B,1,128,128]`. Inference applies sigmoid per tile, then fixed Hann probability blending on stride-64 tiles, including final edge-aligned tiles. A residual must be added to each tile's logits before sigmoid. Adding to blended probabilities or a globally normalized image would change the contract.

## Selected matched-capacity branch

Let `X` denote the single normalized raw channel, with depth index `j=0..16`. Candidate inputs are `U_depth = stack(X, D, r)` with `[B,3,17,128,128]` shape:

- `D[0] = X[1] - X[0]` and `D[16] = X[16] - X[15]`.
- `D[j] = (X[j+1] - X[j-1])/2` for interior planes.
- `r[j] = (j-8)/8`, spatially broadcast, fixed before data execution.

The raw-only control uses `U_raw = stack(X_prev, X, X_next)`, where missing endpoint neighbors replicate the endpoint plane. This gives both arms three active input channels, identical acquired-plane support, identical architecture and identical parameter count. It avoids claiming a capacity match using zero-valued channels whose weights receive no gradients. The control can itself learn axial differences from raw neighbors: the causal comparison is the explicit feature/coordinate package under a fixed tiny architecture and training budget, not information present only in one raw scan.

Use Conv3d(3→h, kernel 3, padding 1, bias=True), LeakyReLU(0.01), concatenate depth mean and depth maximum, then Conv2d(2h→1, kernel 3, padding 1, bias=True). With `h=8`, the hidden activation is `[B,8,17,128,128]`, pooling produces `[B,16,128,128]`, and the residual is `[B,1,128,128]`. There is no extra normalization, spatial subtraction, learned depth shift, resampling or clipping.

The parameter formula is `(3*h*27 + h) + (2*h*9 + 1) = 100*h + 1`; at h=8 there are 801 trainable parameters in each arm and 34,547,299 total parameters including the frozen seed42 model. Give both arms identical hidden-convolution parameter tensors from a dedicated fixed RNG seed. Initialize every final Conv2d weight and bias to zero. Then `z_arm = z_seed42.detach() + residual_arm` equals seed42 at initialization for every finite input. Do not initialize both layers to zero: that would prevent the hidden representation from learning.

All seed42 parameters have `requires_grad=False`, remain in evaluation mode and retain their original values. Only the 801 branch parameters go to the optimizer. Detached seed42 logits may be cached once for the exact training tensors and reused identically by both arms. Under zero final projection the first hidden-convolution gradient is necessarily zero; the first final-projection gradient/update should be nonzero, followed by hidden-layer gradients after a projection update. Correctness checks should recognize this chain-rule behavior.

The chosen scalar `r` is a signed nominal surface-stack coordinate. It is not an independently validated physical surface-normal distance or learned geometric correction. The source-frame audit must support any stronger interpretation. The one-sided endpoint rule preserves a linear profile's full slope at the endpoints; it is a fixed convention and does not fit a depth shift. The two arms use the same real endpoint planes.

## Integration and cache requirements

Construct and freeze the strict seed42 reference once. Obtain its logits under `no_grad`; clone/detach into ordinary tensors if they will be used by autograd loss code after an `inference_mode` region. Build the new branch in a phase18-only module. Existing vendor and reference sources need no edits. The natural API is `branch(normalized_image, role) -> residual` and `adapted_logits = frozen_logits + residual`.

For training, reuse the exact phase13 schedule, patch normalization, augmentation and manual masks if selected by the parent. Use the normalized and augmented tensor to derive D/r; 2D flips/rotations do not change depth indexing. A reference cache identity must bind normalized tensor bytes, checkpoint state, mode, dtype and output shape. Cached preupdate teacher logits from historical arms are reusable only when these identities agree; they must not be confused with adapted postupdate logits.

For phase16/17 development contexts, retained baseline results include per-tile input hashes and raw-logit artifacts. Reuse requires identical tile coordinates/order, normalization, full input tensor hashes, all 17 planes, direction, geometry/skip rules and unchanged state. The branch's tile logits are then passed through the same sigmoid/Hann blending. Baseline raw logits are not replaceable by logits reconstructed from blended probabilities.

The output-logit insertion avoids any backward pass through the 34.5-million-parameter U-Net. A residual inserted at `depth_fusion(X)` would be a different experiment: it requires the U-Net input-gradient graph and a 32-channel output projection. For the current cheap pilot, retain the additive-logit insertion. Its scientific claim is frozen-backbone residual CT adaptation, not full backbone representation learning. Candidate-versus-raw isolates the selected feature package; it does not separately identify the marginal effect of coordinates and differences. Any such marginal claim requires a later prespecified ablation.

## Retained cost evidence and bounded feasibility

Phase13 trained every seed42 parameter using AdamW, learning rate 1e-6, batch size 2, 32 updates, four CPU threads, deterministic FP32. It performed 192 batch forwards plus 32 backwards per arm, including isolated repeated fixed-risk measurements. Arm A took 108.123 seconds and peaked at 1,530,109,952 RSS bytes; arm B took 108.924 seconds and peaked at 1,512,841,216 bytes. Non-checkpoint-measurement steps averaged 2.107 and 2.065 seconds, respectively; those times include immediate postupdate forward, hashes and logging, so they are not bare optimizer-step timings.

Phase16 unchanged seed42 calibration performed 50 batch-one tile forwards in 7.188 seconds at 591,613,952 RSS bytes; evaluation performed 100 in 10.169 seconds at 591,716,352 bytes, both including loading/validation overhead. These are observations on this retained CPU workflow, not guarantees for new code.

The proposed branch has about 182.845 million convolution multiply-accumulates per 128×128 patch: 180.486 million for its 3D convolution and 2.359 million for its 2D projection, excluding activation/pooling. Its largest hidden tensor is 8×17×128×128 FP32 values, 8.5 MiB per patch, or 17 MiB at batch two. All 32 batch-two training inputs occupy 68 MiB and their one-channel baseline-logit cache 4 MiB. Three-channel features can be derived on demand to avoid retaining threefold copies. Optimizer state for 801 parameters is negligible compared with the frozen model and activations.

A prespecified 32-step matched-arm CPU pilot is credible within the previous 180-second per-worker / 420-second total envelope because seed42 can be cached and only the small branch is differentiated. This is a planning estimate, not a measured phase18 runtime. Use explicit wall/CPU/RSS/output caps, no unplanned retries, preserve failed-worker evidence, and perform only the frozen correctness and experiment calls. Six development contexts contain 150 stride-64 tiles total, so cached seed42 logits can also avoid duplicate baseline inference across the two branches. Do not use the retained timings to claim paid GPU readiness or scientific effectiveness.

## Evidence files

`source-pins.json` records the text sources and retained metadata used here, plus copied checkpoint identity and scalar historical resource observations. It does not certify a new checkpoint load or a new model run. No executable model was authored by this audit.
