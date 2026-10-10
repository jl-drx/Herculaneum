# BCE smoothing source audit — 8 October 2026

**The published BCE target convention is verified:** `y′ = (1−s)y + 0.5s`. At `s=0.5`, hard targets 0/1 become **0.25/0.75**, equivalently `y′ = 0.25 + 0.5y`. The ninth A_smooth/B_smooth continuation can accurately be described as restoring this BCE target convention. It does **not** reproduce the original combined objective or full historical trainer.

## Primary evidence

The actual retained seed42 checkpoint configuration, recorded in eighth `run/frozen_seed42/result.json`, specifies `bce_label_smoothing=0.5` and `dice_label_smoothing=0.0`. The [published model card](https://huggingface.co/scrollprize/ink_9um) independently explains the BCE smoothing convention and approximate 0.25 background score scale. The exact historic trainer source revision is not established by these records.

At immutable official Villa commit **e0bbb8b40a2db58b1d71864f286eb85717e59e64**:

- [training/losses.py](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/training/losses.py#L24), `_smooth_bce_targets`, casts targets to float, returns them unchanged for smoothing zero, otherwise applies `target_regions * (1.0 − smoothing) + 0.5 * smoothing`. Retained plaintext SHA-256: **5fdbf11cc66c80dfcf30de7abf1a2455ece12cb80b2f98e8c0c06907a8c7b39a**.
- The same class's `forward` passes original `target_regions` to Dice, smooths only the BCE targets, masks BCE element losses and divides by participating-mask count. Its return is `weight_ce * ce_loss + weight_dice * dc_loss`.
- [nnunet_losses.py](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/models/training/loss/nnunet_losses.py#L316) constructs BCEWithLogitsLoss and sigmoid MemoryEfficientSoftDiceLoss. With the ignore-label convention, BCE reduction is `none` before the masked reduction. Dice is a masked soft-overlap objective, not another BCE term. Retained plaintext SHA-256: **c3807408b807d801a2b0aac0332fdfaf70e5fe135258942f03fb9269a340ea47**.
- Retained [config.py](https://github.com/ScrollPrize/villa/blob/e0bbb8b40a2db58b1d71864f286eb85717e59e64/vesuvius/src/vesuvius/ink_detection/config.py#L992) resolves the checkpoint's simple loss mapping to LabelSmoothedDCAndBCELoss with default **BCE weight 1.0 / Dice weight 0.25**, absent explicit overrides. The initializer's recorded loss mapping omits such overrides. These are the verified published factory defaults, rather than proof of every historical training detail.

The combined Dice term means 0.25/0.75 are BCE target endpoints, not a guarantee that the original combined-loss model's probabilities attain those exact values or are calibrated ink probabilities.

## Ninth control scope

The eighth A/B pilot used masked BCE on hard targets 0/1. Ninth changes the participating manual and hard-pseudo **BCE targets** to 0.25/0.75 while retaining hard membership/class identity. A_smooth is the smoothed manual-only continuation; B_smooth adds the same fixed pseudo term. Compare these to the already frozen eighth hard-target A/B results with identical initializer, teacher bytes, support, ordered patches, XY transforms, optimizer, learning rate, coefficients and 32 updates.

The ninth implementation's smoothing helper, inspected read-only during preparation, uses the same algebra. Its zero-smoothing branch returns the unchanged hard-loss implementation. Final implementation freezing, fixtures and execution belong to the other agents/root; this audit does not certify an unfinished package or execute tests.

Neither ninth arm introduces the published Dice component, reproduces its composite-loss/ignore-channel wiring, or reconstructs the original optimizer/schedule, batch size, number of iterations, augmentations, sampling or full data/teacher lineage. Those choices remain our controlled continuation design. Use **“published BCE target-smoothing control”** or **“restored BCE target convention”**; avoid **“restored original training recipe.”** Any later factorial experiment adding Dice is a separate question.

## CPU environment and determinism

[ENVIRONMENT_READONLY.json](ENVIRONMENT_READONLY.json) records the read-only compatibility check. The intended lab interpreter is `/workspace/.venvs/herculaneum/bin/python`, not the unrelated default interpreter. Current metadata is Python **3.12.14**, PyTorch **2.12.0+cpu**, NumPy **2.5.3**, numcodecs **0.17.0**, architecture **x86_64**. Eighth A and B both record PyTorch **2.12.0+cpu** and four threads; these fields match.

The retained eighth worker calls deterministic algorithms, seed **20261008**, four intra-op threads and one inter-op thread. Its runner sets OMP/MKL/OpenBLAS threads to four and disables CUDA. Reusing the frozen eighth schedule directly preserves patch/augmentation order without regenerating it. Ninth should retain those settings and record/check its runtime versions before execution.

The eighth frozen plan/results do not include a full historical Python/NumPy/numcodecs environment receipt or a verified lockfile entry. Full package equality and cross-hardware bitwise determinism are therefore not established by this read-only check. Matching seed and version settings is a reproducibility condition, not proof of equal training outputs. No tensor/model execution, repeated training or numerical determinism test was performed here.

## Retrieval and immutable receipts

Root authorized exact pinned public-source reads after local search showed the formula implementation was absent. Two source files were retrieved over verified HTTPS: **2 requests / 21,098 response-body bytes**, under **18 requests / 128 KiB**. No redirects, retries, failures, repository clone, model weights or datasets were retrieved. The helper explicitly disables automatic redirects and counts bodies/attempts; failures would be retained. Headers/TLS are outside the body budget.

[SOURCE_RECEIPTS.json](SOURCE_RECEIPTS.json) pins both retrieved files, the local config/trainer/model-card evidence, eighth configuration/results, environment receipt and this report. No older scientific file or tracked repository file was changed. This audit closes the BCE-formula ambiguity while retaining uncertainty about the complete historical trainer.
