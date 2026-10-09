# Larger-segment native CT / surface consistency

**92 of 93 tested uint8 samples match exactly, and all 93 agree within one intensity count.** Three geometry-fixed source pixels near `(1280,1280)` on segment `20251028222030` were reproduced using the archived renderer's recorded settings. Forward maximum errors were 0, 0 and 1; the reversed-order control disagreed substantially. This supports local CT/render correspondence and depth order without authenticating ink.

## Fixed points and source calculation

The check froze source X/Y `(1280,1280)`, `(1279,1280)` and `(1280,1281)` from the geometry mask before examining source profiles or detector maps. All three were valid, so the deterministic nearest-valid fallback was unnecessary. The selected larger ROI begins at `(512,512)`; observed profiles are therefore local `(768,768)`, `(767,768)` and `(768,769)`. The freeze record, geometry hashes and initial cap are retained in `data/experiments/pherc0800-larger-native-consistency/point-selection.json`.

The official, revalidated export record identifies commit `530584554c031c982ba3aa243e1ea39c4bef9818`, the full 2580 × 2580 source canvas, scale 1, 31 slices and `flip-normals=true`. The earlier native-consistency helper was copied into a separate directory and adapted only for this canvas, source crop and fixed points. It preserves the archived float32 tile/halo arithmetic, bilinear base position, nearest cached grid normal, fixed offsets −15 through +15, float32 trilinear sampling, clipping to 0–255 and truncation to uint8. No normal sign, transform, offset or contrast parameter was fitted.

At source `(1280,1280)`, the archived base CT XYZ is `[3418.668457,6300.133789,11988.391602]`, with recorded flipped normal `[0.764139,-0.601337,0.233421]`. All points sample the cached normal at mesh column/row `(64,64)`. The float64-derived geometry map differs from the archived float32 base calculation by at most 0.000977 CT voxel at these points; the check uses the archived calculation for sampling.

## Observed comparison

| Source X/Y | Exact matches | Forward MAE / maximum error | Reverse-order MAE / maximum error |
| --- | ---: | ---: | ---: |
| 1280 / 1280 | 31 / 31 | 0.0000 / 0 | 37.29 / 96 |
| 1279 / 1280 | 31 / 31 | 0.0000 / 0 | 39.29 / 92 |
| 1280 / 1281 | 30 / 31 | 0.0323 / 1 | 32.10 / 94 |

The single discrepancy is depth index 28 at `(1280,1281)`: source value 124, reproduced float value approximately 125.0016, truncated prediction 125. The cause of this one-count residual remains unresolved. No transform, offset or sampling parameter was changed to eliminate it. This check is therefore not an assertion of bit-identical rendering at every pixel.

The reverse control reverses the same fixed samples and is not an alternative fitted transform. Each reversed profile matches only its central value exactly.

## Budget and retained evidence

Only one native level 0 CT chunk, Z/Y/X `93/49/26`, supplies every required trilinear corner. Its raw payload is **2,097,152 bytes (2 MiB)**, below the 16 MiB cap. The object existed, so no missing-chunk fill reconstruction was used.

```text
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr/0/93/49/26
SHA-256: e1b76914d28dd93b535b31d2354327a5bb6b7fefe896126c32de0bb077a245b3
```

The raw chunk, metadata, exact source URL, ETag, Last-Modified, helper hashes, fixed point plan and full 31-layer profiles are retained. [Machine-readable evidence](native-ct-consistency.json) and [profile plot](native-ct-consistency.png) summarize the result. Before/after hashes confirm that the selected surface, its provenance and geometry inputs remained unchanged; the frozen selection files were not edited.

This checks three nearby profiles in one cached-normal neighborhood. It does not validate the entire larger crop, all partitions, papyrus-layer identity, detector calibration, ink, readable letters or translation. Matching native intensities tests scientific input integrity; it does not turn a model score into writing.
