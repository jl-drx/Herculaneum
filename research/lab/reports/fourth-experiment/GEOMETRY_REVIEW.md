# Fourth experiment: frozen geometry-only selection

The selector chose **segment `20251028220955`, native source bbox `[1024,384,384,384]`**. The complete 384 × 384 crop contains 147,456 geometry-valid base-coordinate pixels. Its minimum exact nearest distance from all previously acquired valid base coordinates is **52.89808653091241 native CT voxels**, approximately **457.04 µm** at the recorded nominal 8.64 µm spacing. The median is 71.0365 voxels and maximum 86.8561.

[The selected plan](geometry-selection/selected-plan.json) has SHA-256 `f7ae57e533af2b0411d874d1ab931ab33ffcd16494f59b869f1e5fd15fbe8d3a`. It preserves the source metadata/attributes, render provenance, global and selected geometry maps, nearest-coordinate witness, input hashes, chunk requests and frozen controls. Acquisition is a separate parent-owned stage after review; this selector performed no network requests, intensity reads or model runs.

## Frozen rule and prior inventory

[Methodology](methodology.json), SHA-256 `5986365a56b1a49fb007257a5bbaa699f7e65a0b21705896b86dc2c77d2c4860`, was frozen before selector execution. It enumerates 384-square native crops on the 128-pixel chunk grid, requires complete geometry validity, excludes positive-area overlap with previously acquired same-source extents, and requires minimum nearest base-coordinate distance strictly greater than 40 voxels. Ranking maximizes that minimum, then uses segment ID, source Y and source X ascending. No criterion was relaxed.

The exact prior tree contains **2,693,760 points**: 497,600 valid acquired pixels from the entire initial 980-square surface and 2,196,160 from the larger 1536-square acquisition. Pilot, candidate, independent-region and central-control crops are subsets of these extents. The tree retains all valid centers without intensity/prediction filtering, subsampling or deduplication.

`cKDTree.query` used Euclidean XYZ distance with `k=1`, `p=2`, `eps=0` and four workers. Retained float32 coordinates were cast to float64. Querying each unique pixel in eligible crop unions avoids repeated work; every crop's score still reduces all 147,456 exact distances. The selected crop was queried again independently and matched its saved global distance crop bitwise.

| Segment | Grid positions | Excluded prior extent | Not fully valid | Eligible | Distance passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20251028213516 | 210 | 0 | 150 | 60 | 0 |
| 20251028220042 | 169 | 0 | 115 | 54 | 0 |
| 20251028220955 | 196 | 0 | 147 | 49 | 29 |
| 20251028222030 | 324 | 196 | 128 | 0 | 0 |
| 20251028225813 | 225 | 0 | 160 | 65 | 0 |
| **Total** | **1,124** | **196** | **700** | **228** | **29** |

The winner uniquely maximizes the minimum distance. The full run took **62.59 seconds**, including a 0.64-second tree build. All 42 verified inputs remained unchanged at completion. [Candidate scores](geometry-selection/candidate-scores.json), [prior inventory](geometry-selection/prior-acquisition-inventory.json), [outcome](geometry-selection/outcome.json) and [artifact manifest](geometry-selection/artifact-manifest.json) preserve the evidence.

## Geometry diagnostics and review

Selected neighboring XYZ step lengths range from 0.96695–1.02964 voxels horizontally and 0.97140–1.02088 vertically. There are no zero/near-zero neighbor steps. Central finite-difference tangent cross-product norms range from 0.94997–1.04832 voxel², with no nonfinite or degenerate values at the frozen threshold. These are smoothness/degeneracy diagnostics, not recto or surface-placement authentication.

The whole crop's minimum Euclidean clearance from invalid original-global geometry is **41.40 pixels**; 498 pixels are below 64. The central 256-square core, after the predeclared 64-pixel crop margin, has minimum clearance **131.64 pixels**. Complete crop validity means contributing 128-square inference patches fitting this crop have valid base geometry. Data coverage and evaluated support still require acquisition/inference checks.

Independent read-only review recomputed all candidate validity/exclusion statuses and eligible distance reductions, confirmed ranking and the nearest-coordinate witness, verified all 57 unique recorded file hashes/sizes, and checked the selected mask/XYZ against the exact source slice. It found no material issue. No model or intensity evidence informed that review.

## Bounded acquisition and scientific limits

The source is native uint8 `[31,2100,2100]` with uncompressed chunks `[31,128,128]`. The crop requires **nine objects**, depth chunk 0, Y chunks 3–5 and X chunks 8–10: **4,571,136 bytes** maximum source payload. The separate acquisition contract caps all remote responses including metadata at **6 MiB**, with zero automatic retries and metadata revalidation before chunk reads.

Controls were predeclared before source acquisition: center forward `7:24` with both seeds; seed 42 minus-two `5:22`, plus-two `9:26`, and reversed `7:24`. These schedule at most **125 checkpoint tiles** on the same crop. Full-field and 64-pixel-interior summaries use common valid/evaluated/observed support; scores are behavior diagnostics.

The >40-voxel test is descriptive base-coordinate separation. **Independent physical papyrus layers and disjoint through-depth/interpolation support remain unresolved**, because archived renderer normals and complete sample footprints were not reconstructed. Geometry validity does not establish ink, and the nominal 3.31776 mm field cannot provide text-line context. No reading, translation or target accuracy is established.

Reproduce the selector in a fresh output directory under this fourth-experiment directory:

```bash
/workspace/.venvs/herculaneum/bin/python select_geometry.py \
  --methodology methodology.json \
  --methodology-sha256 5986365a56b1a49fb007257a5bbaa699f7e65a0b21705896b86dc2c77d2c4860 \
  --output-dir geometry-selection-reproduction
```

Run from `/workspace/herculaneum-lab/reports/fourth-experiment`. The command verifies frozen script/provenance hashes, preserves existing outputs, and performs geometry-only CPU work.
