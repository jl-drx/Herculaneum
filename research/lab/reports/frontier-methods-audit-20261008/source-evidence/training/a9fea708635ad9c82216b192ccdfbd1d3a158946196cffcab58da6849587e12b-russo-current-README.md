---
license: mit
tags:
  - vesuvius-challenge
  - ink-detection
---

# 9 um ink checkpoints trained with dense pseudo-labels

Four checkpoints in the same format as `scrollprize/ink_9um` (`model` /
`config` / `step`, 138 MB each), so they load with the released inference CLI
unchanged:

```
python -m koine_machines.inference.infer <segment_9um_iso.zarr> \
  dense9um-w016excluded-step075000.pth out.tif \
  --overlap 0.5 --blend-mode hann --batch-size 8
```

| file | what it is |
|---|---|
| `dense9um-w016excluded-step075000.pth` | **the one to use.** Legible on w016, beats the control on all three held-out regions, beats the best of the 14 released checkpoints on pherc0814-46527 |
| `dense9um-w016excluded-step060000-best.pth` | same run, best online-validation step |
| `dense9um-all7-step075000.pth` | scores higher on w016 and 0814 but renders w016's letters visibly worse. Included so the comparison is reproducible |
| `control-manuallabels-step075000.pth` | the matched control, needed to verify the ablation |

## Results

Balanced accuracy at threshold 0.5 / AUC, on the three `_validation_mask.zarr`
regions that ship with `ink_9um`, which are the only held-out ground truth in
the release. Scored with the team's own `BalancedAccuracy`, masked.

| model | pherc0139-w016 | pherc0814-46527 | pherc1667-w029 |
|---|---|---|---|
| control, manual labels only | 0.7016 / 0.8707 | 0.7539 / 0.8323 | 0.7814 / 0.8945 |
| dense, all 7 segments | 0.7783 / 0.8962 | **0.8539** / **0.9350** | 0.8070 / 0.9044 |
| dense, w016 excluded | 0.7496 / **0.9070** | 0.8329 / 0.9298 | 0.7882 / 0.8853 |
| best of 14 released | 0.8194 / 0.9366 | 0.8129 / 0.8970 | 0.8434 / 0.9334 |

Dense supervision beats the matched control on all three regions, by +0.077,
+0.100 and +0.026. It exceeds the best released checkpoint on pherc0814-46527 by
+0.041 and does not on the other two regions.

The larger effect is on overfitting. Supervised score minus held-out score falls
from 0.21 to 0.29 for the control and the released checkpoints, to 0.007 to 0.099
for the dense models.

## Read this before comparing against "the released model"

The 14 released checkpoints vary by up to **0.14** in held-out balanced accuracy
depending on which one you pick, and `seed42/step-075000`, the most natural
default, ranks last of 14 on w016. seed43 checkpoints are systematically better
on two of three regions. Any comparison needs to name the checkpoint. Full
per-region table for all 14 is in [the repo](https://github.com/DomRusso2/ink9um-dense).

Same-step seed noise is 0.061 to 0.130 on w016, 0.013 to 0.021 on 0814, and
0.066 to 0.072 on w029. Only the pherc0814 improvement above clearly exceeds it.

## Balanced accuracy disagreed with the picture

On w016 the all-7 model scores +0.029 above the w016-excluded model and renders
the same held-out letters as merged blobs, while AUC ranks them the other way
round. The two runs differ only in one segment's labels. The checkpoint
recommended above is the legible one, not the higher-scoring one. Figures showing
this are in [the repo](https://github.com/DomRusso2/ink9um-dense).

## Training

Unmodified `aligned21_hybrid_3d2d.json` from villa except `out_dir` and
`datasets`: seed 42, 78,125 iterations, batch 64, SGD lr 0.01, fp16,
`robust_mad` normalization, `bce_label_smoothing` 0.5, 17-of-21 jittered z
window, `fixed_scroll_prior` quotas 0139:29 1667:22 Paris4:11 0814:2. Full
29-representation corpus. 9.5 hours per run on one RTX 5070 Ti at 2.3 it/s.
Labels: [domenicor046/ink9um-dense-labels](https://huggingface.co/domenicor046/ink9um-dense-labels). Code and full results: [https://github.com/DomRusso2/ink9um-dense](https://github.com/DomRusso2/ink9um-dense).

## Provenance and licence

Scripts and documentation are MIT. The checkpoints are derived from Vesuvius
Challenge data and models (`scrollprize/ink_9um`, `scrollprize/ink_canonical_2um`,
the open-data S3 bucket) and remain subject to the terms of those sources.
