---
pretty_name: "Optical profilometry of opened Herculaneum papyri"
license: cc-by-nc-4.0
task_categories: [image-segmentation]
size_categories: [n<1K]
tags: [image, 3d, archaeology]
---

# Optical profilometry of opened Herculaneum papyri

Dataset accompanying [“Ink detection from surface topography of the Herculaneum papyri”](https://doi.org/10.1038/s41598-026-58467-1).

The release contains 14 co-registered profilometry regions encompassing 16 letters from PHerc. 248, PHerc. 250, and PHerc. 500P2.

- Native lateral sampling: 0.68793625 µm/pixel in X and Y
- Vertical sensitivity: 8 nm

> **NOTICE — 3 September 2026:** We identified an inconsistency between the lateral sampling recorded in this dataset’s metadata and the value reported in the manuscript. We are investigating the discrepancy and will apply any necessary corrections.

## Data

Each sample under `data/` has four files with identical dimensions and a top-left origin:

- `<sample>.txt`: semicolon-delimited X, Y, and calibrated Z values in µm; missing reconstructions are `nan`
- `<sample>_z.tif`: per-sample normalized uint16 height map
- `<sample>_stack.tif`: co-registered RGB brightfield image
- `<sample>_label.png`: grayscale ink annotation; experiment preprocessing treats nonzero values as foreground

Use the text files when calibrated physical heights are required. Exact height-map preprocessing is specified in `configs/experiment.json`.

## Metadata and configurations

- `manifest.csv`: dimensions, physical extents, sampling, acquisition dates, and file paths
- `configs/case_manifest.json`: stable machine-learning case IDs and sample mapping
- `configs/sampling.json`: native, vertical, and factor-derived sampling values
- `configs/experiment.json`: preprocessing, experiments, training settings, and software versions
- `configs/nnunet/`: dataset descriptor and nine resolution-specific nnU-Net plans
- `folds/splits_final.json`: five cross-validation and three leave-one-papyrus-out splits

## License

Data, annotations, folds, and configurations are CC BY-NC 4.0. See `LICENSE`. Please cite the associated paper above.
