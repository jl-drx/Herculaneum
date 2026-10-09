# Native CT / surface consistency at the first experiment hotspot

**The archived surface and native CT agree exactly at all 93 tested uint8 intensities.** At surface pixel x/y 652/563 and two immediate neighbors, a reproduction using the archived renderer geometry, normal sign and fixed 31 slice offsets returned exactly the observed source profile. The reversed-order control disagreed. This supports local source/render correspondence and depth order; it does not establish ink or readable letters.

## Export provenance recovered before sampling

- The small `metadata.min.json` projection omitted rendering provenance. A bounded read of the full public https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json recovered it: 1,391,330 gzip wire bytes, October 5, 2026 Last-Modified. Raw bytes, SHA256, HTTP headers and the relevant export record are retained under `data/experiments/pherc0800-native-ct-consistency/`.
- Exact export source commit: `530584554c031c982ba3aa243e1ea39c4bef9818`, rendering process September 7, 2026. Historical source files are retained with their upstream license; SHA256 and source URLs are in the JSON report.
- Recorded settings: eligible native volume 20250521135224; transformed mesh 20251029010146; scale 1; group 0; 31 slices; `flip-normals=true`; rotate 0; `flip=-1`; zero crop; no affine or segmentation-scale overrides; output uncompressed. Surface attrs independently specify `slice_step=1`. Historical source has no smooth surface-interpolation mode; its disabled-accumulation default is used because no accumulation-step override is recorded.

## Reproduced sampling convention

1. [vc_render_tifxyz.cpp](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/apps/src/vc_render_tifxyz.cpp) and [QuadSurface.cpp](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/core/src/QuadSurface.cpp) center the 980 × 980 canvas over a 49 × 49 coordinate grid: native pixel centers map to grid `(x+0.5)*0.05`, `(y+0.5)*0.05`. Tile and halo arithmetic was reproduced in float32. It confirms the earlier full geometry map to within 0.000489 voxel at these points.
2. [Geometry.cpp](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/core/src/Geometry.cpp) computes cached grid normal from the cross product of central X/Y coordinate differences. The normal is sampled at the nearest grid vertex (column 33, row 28), then negated by recorded `flip-normals=true` and normalized. At the hotspot, native CT XYZ is `[3466.528809, 6424.378906, 12004.220703]`; normal is `[0.752489, -0.578353, 0.315068]`.
3. The archived renderer centers the 31 depth offsets at `index−15`, hence -15 through +15 native voxels at 8.64 µm steps. [Slicing.cpp](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/core/src/Slicing.cpp) performs trilinear CT sampling, clips to 0–255 and **truncates** to uint8. Python reproduced the float32 fused trilinear operations. No spatial transform, normal direction, depth offset or contrast parameter was fitted.

## Observed result

| Surface pixel X/Y | Quantized samples matching | Forward MAE / maximum error | Reverse-order MAE / maximum error |
| --- | --- | --- | --- |
| 652/563 | 31/31 | 0.00 / 0 | 24.65 / 55 |
| 651/563 | 31/31 | 0.00 / 0 | 33.03 / 62 |
| 652/564 | 31/31 | 0.00 / 0 | 27.74 / 55 |

The reverse control reverses the same fixed samples; it is not an alternative transform fitted to the data. Model inference with reversed input remains a separate experimental control.

## Native data budget and retained evidence

- Only native level 0 chunks Z/Y/X 93/50/26 and 93/50/27 were needed to supply every trilinear corner for all three 31 point profiles. Both objects existed; no missing-chunk fill reconstruction was required.
- Native source payload: **4,194,304 bytes (4 MiB)**, within the 16 MiB cap. Each source chunk is 128³ uint8, uncompressed. Raw chunks, exact public URLs, SHA256, ETags and Last-Modified are retained. Metadata and the source-derived pre-download plan are retained separately.
- [Machine-readable profiles and provenance](native-ct-consistency.json) contain all observed/predicted intensities and source inputs hashes. [Profile plot](native-ct-consistency.png) displays the source comparison. The new local helper `data/experiments/pherc0800-native-ct-consistency/check_native_ct.py` reproduces the fixed calculation without model inference.
- Source surface, mesh, coordinate map and all existing inputs remained unchanged; before/after SHA256 checks passed.

## What this establishes and leaves open

The native scan intensity data, transformed surface coordinates, normal sign and depth stack agree at these three nearby pixels. Thus a generic depth-order mistake or coordinate mismatch is not supported at this hotspot. This comparison covers one small neighborhood and one nearest-normal cell, not all surface pixels or render partitions.

It makes no claim about ink, letterforms, text or translation. The neighboring profiles also are not independent held-out validation of an ink model. The first experiment still requires image-supported letters and papyrological assessment before any reading claim.
