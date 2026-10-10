---
license: mit
---

# 9 µm ink detection

Ink-detection models for Herculaneum scroll surface volumes at ~9 µm isotropic resolution, trained on aligned labels across four scrolls. The labels live in the [ink_9um dataset](https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um). Training and inference code: villa @ merge-ink-pipelines - see ink-detection/configs/README.md

## Layout

```
hybrid_3d2d-seed42/   step-010000.pth ... step-075000.pth  (7 checkpoints)
hybrid_3d2d-seed43/   step-010000.pth ... step-075000.pth  (7 checkpoints)
```

## Models

The whole recipe is one JSON config in the code branch, configs/aligned21_hybrid_3d2d.json. The model is a small local 3D stem feeding a 2D U-Net, trained on 128×128 patches sampled every 32 px. Every training batch draws a fixed number of patches from each scroll (29/22/11/2 of 64 across PHerc. 0139, Scroll 1667, PHerc. Paris 4, and PHerc. 0814), and the z window jitters over 17 of the 21 slices so the models don't lock onto one exact depth

We trained it twice, changing only the seed, and we're sharing both runs in full: hybrid_3d2d-seed42/ and hybrid_3d2d-seed43/, seven checkpoints each along the trajectory, named by training step (step-010000.pth up to step-075000.pth). Different steps behave a bit differently on different segments, so it's worth trying a few

Checkpoints embed their training config; inference rebuilds the model and its normalization from the checkpoint automatically

## Quick start

Inference takes a surface volume and a checkpoint:

```
uv run python -m koine_machines.inference.infer <input.zarr> <checkpoint.pth> <output.tif>
```

The models expect ~9 µm isotropic surface volumes. Native ~9 µm renders work directly, local or by URL:

```
uv run python -m koine_machines.inference.infer \
  https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr \
  <checkpoint.pth> w043.tif
```

For 2.4 µm surface volumes, first pool them to the ~9.6 µm isotropic representation the models were trained on (XY pyramid level 2, 4x z mean-pooling):

```
uv run python scripts/prepare_9um_isotropic_input.py <surface-volume-2p4um.zarr> <pooled.zarr>
```

## Tips

- If a checkpoint is not responding well on your data, it might just be a z layer offset; the models can be quite sensitive to it, and picking a different z window (--layer-start/--layer-end) can help. Averaging predictions over a few nearby z windows also works as a simple ensemble. Training on jittered 17-of-21 windows makes the models handle small offsets reasonably well, but larger ones can still throw them off
- The models train with BCE label smoothing 0.5, so their most confident no-ink output sits near 0.25 rather than 0. For display, rescale with (p - 0.25) / 0.5; keep the raw TIFFs raw for anything quantitative

## Training data

We trained on the public z-anisotropic 2.399 µm volumes downsampled to ~9.6 µm isotropic (level-2 XY, 4x z mean-pooling), with segments from PHerc. 0139 (9), Scroll 1667 (6), PHerc. Paris 4 (8), and PHerc. 0814 (1), plus 5 native 9.362 µm segments from PHerc. 0139

The labels, the per-segment source-volume tables, and how they plug into the training config all live in the [ink_9um dataset](https://huggingface.co/buckets/scrollprize/datasets/tree/ink_9um)

## Train it yourself

One command:

```
uv run python -m koine_machines.training.train configs/aligned21_hybrid_3d2d.json
```

These models are *far* from optimized. Better augmentation, longer training, other architectures, ensembling, and more data are all open directions, and we still expect substantial gains to be made to the models
