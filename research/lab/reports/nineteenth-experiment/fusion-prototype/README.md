# Matched depth-feature fusion prototype

This is an engineering prototype, not a trained ink detector or an efficacy result. The phase 18 output residual failed its predeclared improvement screen; its evidence remains unchanged. This proposal changes the insertion point to the original model's 3D-to-2D feature boundary. The released model already has learned axial convolutions and depth attention, so this is not a claim of newly introducing depth awareness.

## Exact integration

The verified `Local3DStem2DUNet` computes `network(depth_fusion(X))`. Its original `depth_fusion` maps normalized FP32 `[B,1,17,H,W]` to `[B,32,H,W]`: 16 attention-weighted feature channels followed by 16 depth-maximum channels. The proposed output is:

```
F = original_depth_fusion(X).detach()
R = adapter(features(X, arm))
ink_logits = original_network(F + R)["ink"]
```

The 32-channel feature residual enters the complete 2D U-Net, before its encoder. It is not added merely to the final decoder block or the ink logit. The original depth stem, 2D encoder, decoder and output head are all frozen and stay in evaluation mode. The unchanged model remains available as the reference. No vendor or tracked repository source is modified.

The original fused features are computed under `torch.no_grad`; the original 2D network must run with autograd enabled. Frozen weights can transmit derivatives with respect to input without acquiring parameter gradients. Wrapping that network call in `no_grad`, detaching `F + R`, or using inference-mode tensors as the differentiable cache would break the learning path. The wrapper rejects inference-mode caches and non-eval nested backbone modules before use. Ordinary detached feature caches can be reused only for exactly identified normalized inputs, original checkpoint and runtime.

## Matched representations and capacity

Both arms use exactly the same branch architecture and initial state:

| Operation | Output | Parameters |
|---|---|---:|
| Conv3d, 3 → 8, kernel 3, padding 1, with bias | `[B,8,17,H,W]` | 656 |
| LeakyReLU, slope 0.01 | same | 0 |
| Concatenate depth mean and maximum | `[B,16,H,W]` | 0 |
| Conv2d, 16 → 32, kernel 3, padding 1, with bias | `[B,32,H,W]` | 4,640 |
| **Total trainable per arm** | | **5,296** |

The complete model would have 34,551,794 unique parameters, of which 34,546,498 are frozen. This branch has 4,495 more trainable parameters than the phase 18 output residual. Thus a cross-phase comparison alone would confound insertion location and capacity; the within-phase raw/depth comparison remains capacity matched.

For `raw_control`, the three channels are `[X_previous, X, X_next]`, with endpoint neighbors clamped to the endpoint. For `explicit_depth`, they are `[X, D, r]`: centered differences with factor 1/2 in the interior, full one-sided differences at indices 0 and 16, and `r[j] = (j - 8) / 8`. Keep published increasing plane order and the whole-patch robust-MAD normalization unchanged; do not separately normalize the derivative or coordinate. Apply existing XY augmentations first, then derive features. No extra planes, learned offsets or interpolation are introduced.

The coordinate is a nominal relative plane index. It is not certified physical distance, signed surface-normal orientation or measured ink depth. Differences are per plane index, not per micrometre. The raw control can itself learn differences; the contrast concerns the explicit feature package under this architecture, not information exclusive to one scan. Coordinates and derivatives are not separately identified without an additional prespecified ablation.

Initialization uses a forked CPU RNG seeded at 20261008, leaving the caller's RNG unchanged. The random first convolution is identical in both arms. The final projection is exactly zero, so finite fused features and logits initially equal the baseline. The first hidden-layer gradient must be zero; the projection gradient can be nonzero. After one projection update, a second backward pass must reach the hidden representation. Both feature arms retain the original CT channel and preserve acquired depth support.

## Executed synthetic checks

`synthetic-results-v1.json` records 45 passing checks: analytic features including endpoints and reversal, XY covariance, invalid inputs, exact matching and zero identity, native-size projection shape, gradient propagation through a frozen synthetic decoder with affine normalization, optimizer membership, immutable parameters and buffers, and inference-cache/mode rejection. It took 0.96 seconds and peaked at 378,494,976 bytes RSS with four CPU threads. There were two synthetic toy SGD updates and four toy backwards. No actual seed42 model, checkpoint or scientific array was loaded.

Technical review requested rejecting nested backbone training mode before computing the frozen stem. A finite-derivative overflow guard was also added. The original receipt and exact tested adapter source are retained as `fusion_adapter.v1-tested.py`; `v1-source-reconstruction.json` records the exact textual reversal of those two additions. The unchanged original fixture source is retained as `check_synthetic.py`. Only the new guards and their valid paths were rerun: `synthetic-results-v2.json` records four passing targeted checks in 0.003 seconds, with no backward or optimizer calls. The final adapter source is bound in that receipt. These tests validate mechanics against toy feature/decoder fixtures, not the actual released network.

## Prepared actual-backbone smoke, not executed by this author

`actual_backbone_smoke.py` and `smoke-plan.json` are for root review before execution. The controller requires the exact externally supplied execution-plan SHA, verifies pinned local code before construction, and runs one bounded worker with no implicit retry. The unchanged strict phase 6 loader verifies the released seed42 artifact, source manifest, embedded configuration, aliases and strict state loading, using CPU `weights_only=True` deserialization.

One fixed synthetic Gaussian FP32 input of shape `[1,1,17,128,128]`, generated with seed 20261009, is shared by both arms. Its values mimic standardized intensities only; they are neither a scan nor a normalization fit. A fixed spatial `linspace(-1,1)` target supplies an engineering mean-squared-error objective. Each arm performs exactly one SGD step at learning rate 0.001 and a second backward without a further step. This validates the projection-to-hidden learning path; it is not scientific training or a candidate model. No resulting weights are saved.

The frozen schedule observes hooks rather than inferring calls: one original full-model forward, two original depth-fusion forwards, five original 2D-network forwards, four adapter forwards, four backwards and two synthetic SGD steps. It demands exact original logits at zero initialization, matched arm state, finite nonzero projection then hidden gradients, no backbone gradients, and exact before/after state-dict digests covering named parameters and buffers. Buffer names and all child evaluation modes are recorded.

Limits are 300 seconds controller wall time, 295 seconds worker CPU/alarm, four Torch threads plus one interop thread, sampled 4 GiB RSS termination, 8 GiB virtual address space, and 64 MiB output. Virtual address space is not RSS. The controller samples memory every 50 ms, so an instantaneous overshoot is possible; final worker `ru_maxrss` must also be below the limit for a pass. Failures leave command, worker log and controller failure receipt. No network, GPU, scientific array reads, labels, metrics, external publication or checkpoint writes are permitted.

A passing actual-network synthetic smoke would establish integration readiness only. A subsequent scientific study still needs its frozen training-only sampling/optimizer/endpoint contract and a defensible independent confirmation route. Earlier six Paris contexts remain reused development, and Paris2 remains excluded from training, teacher and cutoff fitting. The next scientific architecture question should not be combined with new pseudo-label recipes or expanded data if the representation contrast is to remain interpretable.

GPU expenditure is zero. Nothing in this prototype authorizes a paid launch or changes the £10 total first-pilot cap or the unresolved worker-secret delivery requirements.
