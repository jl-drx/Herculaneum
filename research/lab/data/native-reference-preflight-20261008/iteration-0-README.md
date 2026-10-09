---
library_name: transformers
license: mit
tags:
- vesuvius-challenge
- ink-detection
- herculaneum
- resnet3d
- u-net
- 3d-segmentation
- volumetric-imaging
pipeline_tag: image-segmentation
---

# PHerc.1667-iteration-0

> **Trained on segment l_2 with (cross-segment baseline; no l_2 labels) (20,075 tiles).**

Cross-segment baseline (no l_2 labels in training). Trained on 500p2a, 658, 20250910185200, and 20250919125754-auto_grown_20250919055754487_inp_hr. l_2 inference uses reverse_layers=True to match the training-segment convention.

This is one of **six sibling models** released together — five label
ablations on segment `l_2` (`ink1`–`ink5`, increasing label coverage)
and one cross-segment baseline (`ink0`).  The full family is listed
at the bottom of this card.

## Preview

`l_2` (training segment) prediction with the training label overlaid in
magenta, and `l_5` (held-out segment) prediction. All panels are
downsampled 16× and rotated 180° to match the publication-figure
convention. The full-resolution `last.ckpt` outputs are at 43008 × ~30000
voxels.

| training label | l_2 prediction | l_5 prediction |
|----------------|----------------|----------------|
| ![label](./preview_label.png) | ![l_2 pred](./preview_l_2.png) | ![l_5 pred](./preview_l_5.png) |


## Architecture in one paragraph

A 3-D volumetric input `(B, 1, 62, 256, 256)` is encoded by a
**ResNet3D-50** backbone (Hara, Kataoka & Satoh, 2018; initialised from
the Kinetics-700 release `r3d50_KM_200ep.pth` with conv1 weights
summed across RGB → 1 grayscale channel).  Each of the four backbone
stages is collapsed along the z (depth) axis with `torch.max`,
producing a 2-D feature pyramid `{(256,64,64), (512,32,32),
(1024,16,16), (2048,8,8)}`.  A small **2-D U-Net decoder** upsamples
coarse-to-fine with concatenated skip connections; a 1×1 conv head
produces a single sigmoid logit channel at quarter resolution
`(B, 1, 64, 64)`.  Training uses `0.5·Dice + 0.5·SoftBCE` against the
label down-interpolated to 64×64.

## Quick start

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "YoussefMoNader/PHerc.1667-iteration-0",
    trust_remote_code=True,
).eval().cuda()

# Input: float32, shape (B, 1, D=62, H=256, W=256).
# Intensity should already be in roughly [0, 1] (the training pipeline
# clipped raw uint8 layers to [0, 200] then applied Normalize(mean=0, std=1)
# which keeps the magnitude small).
x = torch.randn(1, 1, 62, 256, 256, device="cuda")

with torch.no_grad():
    out = model(x)

print(out.logits.shape)                        # torch.Size([1, 1, 64, 64])
prob = torch.sigmoid(out.logits)               # ink probability per pixel
```

## Full-segment inference (tiling)

The model only sees 256×256 windows.  For a full scroll segment you
need to slide the window across the (padded) layer stack and average
overlapping predictions:

```python
import numpy as np, cv2, torch
import torch.nn.functional as F
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "YoussefMoNader/PHerc.1667-iteration-0", trust_remote_code=True,
).eval().cuda()

WINDOW, STRIDE = 256, 128       # 128 = 2x oversample; 64 for 8x oversample
D = 62                           # number of z-layers

# image: (H, W, D) uint8 stack of the 62 layers, padded to multiples of 256.
# fmask: (H, W) uint8 fragment mask (0 = outside, 255 = inside).
H, W, _ = image.shape
mask_pred  = np.zeros((H, W), dtype=np.float32)
mask_count = np.zeros((H, W), dtype=np.float32)

with torch.no_grad():
    for y in range(0, H - WINDOW + 1, STRIDE):
        for x in range(0, W - WINDOW + 1, STRIDE):
            if np.any(fmask[y:y+WINDOW, x:x+WINDOW] == 0):
                continue
            tile = image[y:y+WINDOW, x:x+WINDOW]            # (256,256,62)
            t = torch.from_numpy(tile).permute(2, 0, 1)     # (62,256,256)
            t = t.unsqueeze(0).unsqueeze(0).float().cuda()  # (1,1,62,256,256)
            logits = model(t).logits                        # (1,1,64,64)
            prob   = torch.sigmoid(logits)
            prob   = F.interpolate(prob, scale_factor=4,
                                   mode="bilinear").squeeze().cpu().numpy()
            mask_pred[y:y+WINDOW, x:x+WINDOW]  += prob
            mask_count[y:y+WINDOW, x:x+WINDOW] += 1.0

pred = np.divide(mask_pred, mask_count,
                 out=np.zeros_like(mask_pred),
                 where=mask_count != 0)
cv2.imwrite("prediction.png", np.clip(pred * 255, 0, 255).astype(np.uint8))
```

## Training summary

| | |
|---|---|
| **Backbone** | ResNet3D-50 (3-D conv, BN, ReLU residual blocks) |
| **Encoder init** | `r3d50_KM_200ep.pth` (Kinetics-700), conv1 summed across RGB |
| **Decoder** | 2-D U-Net (3 up-blocks: bilinear 2× + concat skip + 3×3 conv + BN + ReLU) |
| **Output** | 1 channel, sigmoid logit, quarter-resolution (64×64) |
| **Loss** | 0.5 × Dice + 0.5 × SoftBCE (smooth = 0.25) |
| **Optimizer** | AdamW, OneCycle lr 2e-5 → 3e-4, pct_start = 0.15 |
| **Batch** | 2 (effective 8 via accumulate 4), 16-mixed, grad-clip 1.0 |
| **Max steps** | 12,396 (= 3 epochs over the densest ablation label) |
| **Training segment(s)** | `500p2a + 658 + 20250910185200 + 20250919125754*` |
| **Training label** | `(cross-segment baseline; no l_2 labels)` |
| **Training tiles** (256×256 sub-tiles at stride 64) | **20,075** |
| **Final train loss (`_epoch`)** | **0.5432** |
| **Final train loss (`_step`, single-batch noise)** | 0.5207 |
| **Wandb** | [vesuvius-challenge/paper/l2_ink0_l5infer_baseline](https://wandb.ai/vesuvius-challenge/paper/runs/d68uhctc) |
| **Random seed** | 130697 |
| **Determinism** | `cudnn.deterministic = True`, `cudnn.benchmark = False` |
| **Hardware** | 1 × NVIDIA H100 80 GB; ≈ 2 h end-to-end (load + train + inference) |

## Files

| file | size | description |
|------|------|-------------|
| `config.json` | 1 KB | architecture + provenance metadata; loaded by `AutoConfig` |
| `configuration_inkdetection.py` | 2 KB | `InkDetectionConfig(PretrainedConfig)` |
| `modeling_inkdetection.py` | 9 KB | self-contained `InkDetectionModel(PreTrainedModel)` |
| `model.safetensors` | 319 MB | converted weights (338 tensors) |
| `last.ckpt` | 963 MB | original PyTorch-Lightning checkpoint (incl. optimizer + LR-scheduler state) — load with `torch.load(...)["state_dict"]` |
| `preview_l_2.png` | ~700 KB | low-res preview of the l_2 prediction (1/16 scale, 180° rotated) |
| `preview_l_5.png` | ~2 MB | low-res preview of the l_5 (held-out) prediction |
| `preview_label.png` | ~50 KB | the training label, same scale + rotation |

The HuggingFace weights are **bit-perfect identical** to the original
PyTorch-Lightning checkpoint (verified `max abs diff = 0.0e+00` on
identical inputs).  Use `model.safetensors` for `AutoModel.from_pretrained`;
use `last.ckpt` only if you want to resume training from the saved
optimizer / scheduler state.

## The model family

| model | training segment(s) | label | tiles | effective epochs |
|-------|---------------------|-------|-------|-------------------|
| [`PHerc.1667-iteration-0`](https://huggingface.co/scrollprize/PHerc.1667-iteration-0) | 500p2a + 658 + 20250910185200 + 20250919125754* | (cross-segment baseline) | 20,075 | ~5 |
| [`PHerc.1667-iteration-1`](https://huggingface.co/scrollprize/PHerc.1667-iteration-1) | `l_2` | `l_2_inklabels.png`  | 3,396 | ~30 |
| [`PHerc.1667-iteration-2`](https://huggingface.co/scrollprize/PHerc.1667-iteration-2) | `l_2` | `l_2_inklabels2.png` | 8,970 | ~12 |
| [`PHerc.1667-iteration-3`](https://huggingface.co/scrollprize/PHerc.1667-iteration-3) | `l_2` | `l_2_inklabels3.png` | 15,286 | ~7 |
| [`PHerc.1667-iteration-4`](https://huggingface.co/scrollprize/PHerc.1667-iteration-4) | `l_2` | `l_2_inklabels4.png` | 24,773 | ~5 |
| [`PHerc.1667-iteration-5`](https://huggingface.co/scrollprize/PHerc.1667-iteration-5) | `l_2` | `l_2_inklabels5.png` | 33,061 | 3 |

All six share the architecture, hyperparameters, and a fixed step
budget of 12,396 optimizer steps; the only thing that varies between
rows is the supervising label (or, for ink0, the training segments).

## Citation

If you use this model in published work, please cite the Vesuvius
Challenge and the underlying ResNet3D paper:

```bibtex
@inproceedings{hara2018can,
  title  = {Can spatiotemporal 3D CNNs retrace the history of 2D CNNs and ImageNet?},
  author = {Hara, Kensho and Kataoka, Hirokatsu and Satoh, Yutaka},
  booktitle = {CVPR}, year = {2018},
}
```

## Licence

MIT.
