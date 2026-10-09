---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
base_model: scrollprize/ink_canonical_2um
buckets:
  - scrollprize/datasets
tags:
  - hecate
  - experimental
  - vesuvius-challenge
  - ink-detection
  - 3d-segmentation
  - knowledge-distillation
inference: false
---

# Hecate

**Hecate is an experimental architecture designed to locate ink on the middle papyrus sheet of a surface-conditioned render, while also producing the usual 2D ink map.**

Its name evokes Hecate, the three-formed goddess of thresholds and crossroads. Here, the threshold lies between a volume and a page: traces scattered through depth must become writing on a surface. The model looks into both spaces, seeking the hidden layer while gathering its marks into a readable image.

This release contains checkpoints for **2.4 µm** and **9.6 µm** inputs.

## From a volume to a page

A **surface-conditioned render (surface volume)** is a CT scan resampled around a mesh that follows a papyrus sheet. Although flattened, it still contains depth—and often several neighbouring sheets. The intended sheet can also wander above and below the centre of the render.

A conventional 2D ink detector must therefore learn two things together: which sheet to follow, and where that sheet carries ink.

We started from the **[canonical 2 µm ink detector](https://huggingface.co/scrollprize/ink_canonical_2um)** and modified it to expose a volumetric ink prediction. Hecate predicts ink throughout the render, then uses learned attention to combine its predictions across depth into a 2D map.

The two outputs share the same backbone. Supervision applied to the 2D map can therefore change how the model locates ink in 3D. The attention is learned, without a fixed Gaussian preference for the centre.

## Learning without new labels

**We created no additional manual labels for this work.** We used existing 2D annotations and predictions from frozen models, which served as teachers for the student's 3D output.

We began with three sources of guidance:

- **Strong 2D supervision** from the existing ink annotations.
- **Moderate 3D supervision on Paris4**, supplied by a precise **[Paris4 ink model](https://huggingface.co/scrollprize/ink_3d_dino_guided)**. We oversampled examples carrying these targets to give the student frequent exposure to reliable volumetric guidance.
- **Weak 3D supervision on other scrolls**, supplied by a less precise model.

These sources did not always describe the same task. A 3D teacher could detect ink on several neighbouring sheets, while the 2D annotations described writing on the intended middle sheet.

During training, we began to see the student's volumetric predictions concentrate on that middle sheet, even where its teacher responded to multiple layers. The shared architecture appeared to let the stronger 2D guidance shape a more selective 3D prediction.

We then froze selected student snapshots and used their predictions as new teaching targets. We reduced the special treatment of Paris4 and repeatedly adjusted the balance between 2D and 3D supervision. Dark background voxels were removed from positive teacher targets. When volumetric guidance became too restrictive and suppressed real ink, we weakened its influence.

The models reached these checkpoints through a gradual drift: **no new manual annotations, but a changing balance of imperfect teachers**. Their agreement improved in some places and remained imperfect in others.

## The 2.4 µm model

We preserved as much of the canonical model's architecture and pretrained weights as possible.

We removed its CT intensity clipping but retained division by **200**. This allowed brighter values to remain distinguishable while keeping the input scale familiar to the pretrained representation.

The model predicts ink on a coarser internal grid and interpolates the volumetric output back to the input grid. Its full-sized output should therefore not be mistaken for equally fine localization.

![Hecate 2.4 µm architecture](assets/hecate-2.4um.png)

*Hecate at 2.4 µm. The output images in the diagram are illustrative, not measured predictions. Tensor dimensions are depth × height × width; feature channels are omitted.*

## The 9.6 µm model

We next transferred knowledge from the 2.4 µm model into a model operating at **9.6 µm**. Both its 2D and 3D teaching targets came from the finer-resolution model, rescaled to the new sampling.

The smaller input covers approximately the same physical region with fewer voxels. This model divides CT intensities by **255** and removes the canonical model's initial spatial reduction, allowing its decoder to predict on the input grid. Its encoder still uses internal multiscale stages.

Initially, its inputs were derived by downsampling fine CT. We subsequently introduced surface-conditioned renders from native coarse-resolution scans. Where both scans were available, we aligned their renders so that predictions from the fine scan could supervise the corresponding native CT. Blur and contrast augmentation helped bridge their different appearances. Where native patches failed the alignment checks, we retained fine-derived inputs instead.

**Surface-conditioned renders supplied to this model must be sampled at 9.6 µm in both the surface plane and the depth direction.** A scan described as “approximately 9 µm” may have a different acquisition spacing. Use its actual spacing when rendering or resampling.

![Hecate 9.6 µm architecture](assets/hecate-9.6um.png)

*Hecate at 9.6 µm. The output images in the diagram are illustrative, not measured predictions. Tensor dimensions are depth × height × width; feature channels are omitted.*

## A view through the sheet

<img src="assets/pherc139-cross-section.png" alt="PHerc139 CT cross-section with Hecate's predicted ink shown as a coloured overlay along the papyrus surface" width="360" />

*A cross-section from a PHerc139 segment, with Hecate's predicted ink shown as a coloured overlay. The prediction follows a thin layer along the papyrus surface. The measurement shown is a local viewer measurement, not an estimate of the model's accuracy or spatial resolution.*

## Outputs and limitations

Both models produce:

- **A 3D ink probability volume**, intended to localize ink on the middle sheet.
- **A 2D ink probability map**, obtained by combining volumetric ink logits across depth using learned attention, then converting the result to probabilities.

The 3D output is not a papyrus-surface segmentation. It can miss faint ink, respond to fibres, or include neighbouring sheets. Middle-sheet localization is an intended behaviour rather than a guarantee.

Results also depend on the surface geometry, depth orientation, scan appearance, and whether the intended sheet lies within the surface-conditioned render.

## Running inference

Download `hecate.py`, `requirements.txt`, and the checkpoint you need from this repository. No training code, W&B account, or other model downloads are required.

```bash
python -m pip install -r requirements.txt
python hecate.py --checkpoint hecate_9.6um.pth --input surface.zarr \
  --spacing-um 9.6 --output prediction.png --device cuda
```

For the finer model, use `--checkpoint hecate_2.4um.pth --spacing-um 2.4`. Add `--reverse` to reverse the **full render depth before selecting the central planes**. If the correct direction is unknown, run both directions into different output files.

Inputs must be **unnormalized uint8 arrays in Z, Y, X order**, stored as a local Zarr array, a Zarr group containing array `0` (override with `--array`), or a `.npy` file. Supply a surface-conditioned render, not a raw scroll volume. The render must already have the model's sampling in all three axes; the script does not resample. `--spacing-um` declares the render spacing and checks it against the checkpoint. It does not infer spacing from scan metadata.

| Checkpoint | Input patch, Z × Y × X | CT normalization |
|---|---|---|
| `hecate_2.4um.pth` | 64 × 256 × 256 | Divide by 200; no clipping |
| `hecate_9.6um.pth` | 16 × 64 × 64 | Divide by 255; no clipping |

The script selects the central input depth, slides over XY with half-patch overlap, and blends probabilities using a floored Hann window. Small XY inputs are zero-padded. Empty CT columns receive zero output. It saves a losslessly compressed PNG with **pixel value = round(255 × probability)**, plus a small JSON describing the inference settings. There is no per-image min–max scaling. Output XY dimensions match the input.

The default is float32 with batch size one. On compatible NVIDIA GPUs, `--precision bf16` enables the mixed precision used for production inference; increase `--batch-size` if memory permits. `--device cpu` also works, but this remains a large 3D network. Temporary disk-backed accumulators use approximately nine bytes per output pixel; set `TMPDIR` to a disk with enough free space for large surfaces.

### Saving a 3D probability volume

Use `--output-3d` to save a compressed uint8 Zarr:

```bash
python hecate.py --checkpoint hecate_9.6um.pth --input surface.zarr --spacing-um 9.6 --output-3d ink_3d.zarr --reverse --device cuda
```

Add `--output prediction.png` to also save the 2D map. When only `--output-3d` is requested, attention and the 2D projection are not executed. Requesting both outputs runs two inference passes to keep the implementation and memory use simple.

The 3D Zarr is a **single-resolution array**, not a multiscale OME-Zarr. It stores `round(255 × probability)` with lossless Zstandard compression. Its ZYX shape and voxel order match the input, including after reversed inference: predictions are flipped back before saving. No cropping or per-volume min–max scaling is applied.

Inference tiles over XY and evaluates only the checkpoint's **central depth window**, not sliding windows through the entire Z axis. Unevaluated planes, including the finer model's excluded edge planes, are zero. The Zarr attributes record the half-open `evaluated_z_interval`, spacing, orientation, and inference settings so those zeros can be distinguished from evaluated predictions. Empty CT columns are zeroed; no additional voxel-intensity threshold is applied. Source world-coordinate transforms are not copied; the output is aligned to the input's voxel indices.

Temporary disk use for 3D blending is approximately `(4 × input_patch_depth + 4) × height × width` bytes, in addition to the compressed output. Set `TMPDIR` appropriately for large renders.

### Accessing the 3D output

The same file provides a small Python API. `model(image)` returns 2D logits; `model.forward_3d(image)` returns 3D logits without executing the learned projection. The CLI can write either output or both.

```python
import numpy as np
import torch
from hecate import load_model

model = load_model("hecate_9.6um.pth", "cuda")
# One correctly oriented uint8 patch, with shape model.patch_size.
ct = np.load("patch.npy").astype(np.float32)
ct /= model.divisor
image = torch.from_numpy(ct[None, None]).to("cuda")
with torch.inference_mode():
    ink_3d = model.forward_3d(image).sigmoid()[0, 0].cpu().numpy()
np.save("ink_3d.npy", ink_3d)
```

The 2.4 µm model excludes its first and last input planes from the network and pads their 3D logits with −20; these planes are not valid learned predictions. Its interior 3D logits are interpolated to the input grid. The 9.6 µm model uses all sixteen planes. The Python API returns model predictions without applying an additional CT-background threshold.

The standalone runtime was checked against the original inference implementation on real CT crops with both released checkpoints: 2D and 3D patch logits, tiled forward/reverse predictions, padded edges, blank inputs, lossless PNG decoding, and compressed 3D Zarr round trips. Tiled 3D exports matched the original model in both depth directions, and checks confirmed that 3D-only inference bypasses attention. Float32 and CUDA bfloat16 were tested. The checkpoints load strictly, including their saved EMA batch-normalization buffers.

## Sources and acknowledgements

This work builds on the **[canonical 2 µm ink detector](https://huggingface.co/scrollprize/ink_canonical_2um)** and the **[Paris4 3D ink teacher](https://huggingface.co/scrollprize/ink_3d_dino_guided)**.

Development used the **[Scroll Prize ink dataset](https://huggingface.co/buckets/scrollprize/datasets/tree/ink)** and scans from the **[Vesuvius Challenge open-data collection](https://vesuvius-challenge-open-data.s3.amazonaws.com/index.html)**.

Hecate is released for experimentation and evaluation. It is a step toward finding writing in depth, not a finished solution to reliable 3D ink localization.
