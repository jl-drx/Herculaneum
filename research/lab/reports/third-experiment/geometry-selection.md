# Frozen geometry selection for the larger PHerc0800 experiment

**Use native segment `20251028222030`, source X/Y `[512:2048,512:2048]`, as a 1536 × 1536 crop.** Its conservative base-coordinate mask contains 2,196,160 valid pixels, or 93.09% of the crop. Selection was frozen at October 8, 2026, 06:55:06 UTC, before acquisition or prediction on this larger crop.

The first experiment's 980 × 980 segment contained 497,600 coordinate-supported pixels. The selected crop offers 4.41 times as many, and 6.25 times as many pixels with a complete 128 × 128 geometry neighborhood inside the crop. These are gains in usable geometry and context; they do not establish ink or an independent physical surface.

## Ranking and the size preference

All 188 chunk-aligned 1536 × 1536 windows and 32 chunk-aligned 2048 × 2048 windows fitting the five other public surfaces were ranked using geometry alone. The ranking maximizes pixels having a complete 128 × 128 coordinate-supported neighborhood inside the crop, then fully valid scheduled 128 × 128 model tiles, then total valid pixels. Remaining ties use size, segment ID, Y and X in the recorded order. No intensity image, detector map or transcription informed the choice.

The global ranking pools both sizes and therefore places larger windows first when their absolute supported area is greater. **The recommendation adds the parent's explicit size preference: choose the highest-ranked 1536 × 1536 window first.** It is rank 1 among the 188 windows of that size and rank 30 in the combined 220-window list. The frozen JSON's `selection_basis` describes the global ranking; its separate `recommendation_scope` records this size-first recommendation. The selected coordinates and frozen files have not been changed.

| Quantity | Recommended 1536 crop | Best 2048 alternative |
| --- | ---: | ---: |
| Segment | 20251028222030 | 20251028222030 |
| Source X/Y origin | 512 / 512 | 256 / 256 |
| Valid coordinate pixels | 2,196,160 | 2,936,960 |
| Valid coordinate fraction | 93.09% | 70.02% |
| Complete 128 × 128 neighborhood inside crop | 1,820,439 | 2,432,231 |
| Fully valid / scheduled 128 × 128 tiles | 468 / 529 | 606 / 961 |
| Native surface chunk count | 144 | 256 |
| Exact uncompressed payload bound | 73,138,176 bytes | 130,023,424 bytes |
| Payload in MiB | 69.75 | 124.00 |

The 2048 alternative adds 33.6% more supported interior for 77.8% more payload and 81.7% more scheduled tiles, with a larger proportion of unsupported boundary geometry. The 1536 crop is the first bounded experiment; the retained 2048 ranking supports a later scope decision without selecting from predictions.

## Five public surfaces inspected

| Segment ID | Native surface Y × X | Conservative valid pixels | Whole-canvas valid fraction |
| --- | --- | ---: | ---: |
| 20251028213516 | 2160 × 2240 | 2,313,200 | 47.81% |
| 20251028220042 | 2040 × 2000 | 2,150,800 | 52.72% |
| 20251028220955 | 2100 × 2100 | 2,067,600 | 46.88% |
| 20251028222030 | 2580 × 2580 | 3,159,600 | 47.47% |
| 20251028225813 | 2300 × 2260 | 2,484,000 | 47.79% |

Each has one connected component under four-neighbor connectivity. All five use the same target CT volume `20250521135224`, native 8.640 µm spacing, 31 real depth planes and uncompressed uint8 chunks `[31,128,128]`. Only coordinate TIFFs and metadata were read during ranking. Their total HTTP payload was 844,642 bytes, below the 2 MiB cap; surface and native CT chunk reads during selection were zero.

The recommended source array is:

```text
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028222030-auto_grown_20251028222030940/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/0
```

Its exact chunk plan is depth chunk 0, Y chunks 4 through 15 and X chunks 4 through 15: 144 objects of 507,904 bytes each. The crop spans 13.27104 × 13.27104 mm nominally in the flattened canvas. This is an exploratory crop, not proof of an award-sized readable region.

## Geometry and interpretation

The archived renderer's centered-canvas convention maps native pixel centers to the stored mesh with `grid_x=(x+0.5)*scale_x`, `grid_y=(y+0.5)*scale_y`. Each base-coordinate pixel requires all four bilinear XYZ corners to be positive and finite and inside the stored grid. The five scale pairs are `[0.05,0.05]`. Float64 interpolation is stored as float32 CT XYZ; invalid pixels are NaN and have a separate Boolean mask. The last 20 rows and columns lack a complete stored bilinear cell and are conservatively excluded.

The 128 × 128 support score removes a 64-pixel crop border and checks a complete coordinate neighborhood. It does not guarantee that every overlapping inference tile contributing to an output pixel is fully valid. The explicit full-tile counts use the runner's 128-pixel tiles at stride 64. Base-coordinate support also does not independently verify normal validity, CT occupancy, papyrus-layer identity or ink.

All five CT bounding boxes contain the old tiny segment's bounding box. Treat the selected crop as broader spatial context until the dense mapping comparison is reviewed. [The separate overlap review](design-review.md) quantifies corresponding CT base points and states the limits of physical-independence claims.

Frozen machine-readable evidence is [geometry-selection.json](geometry-selection.json). The source arrays, hashes, full masks/maps and recommended crop masks/maps are retained under `data/experiments/pherc0800-larger-geometry/`; [source-audit.md](source-audit.md) records inventory and render-provenance checks. This prose clarifies the frozen ranking without editing its inputs or coordinates.
