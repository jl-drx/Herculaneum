# PHerc0139 coordinate-TIFF structure

**PASS:** all three pinned coordinate objects reconcile with the published surface canvas. Nine exact HTTP206 requests retained/charged402 body bytes; reservation total9,267 bytes, elapsed7.6904 seconds. No retries, fallback or redirects. Forty primary and eight additional synthetic cases passed before source freeze.

| Observed field | x.tif, y.tif and z.tif |
|---|---|
| TIFF format / byte order | Classic TIFF42 / little endian |
| Width × height | 1444 ×1404 |
| Representation | One IEEE754 float32 sample per pixel |
| Compression / orientation | Uncompressed / default top-left |
| Storage | One strip, rows-per-strip1404 |
| Coordinate storage | `[8,8109512)`;8,109,504 bytes |
| IFD | `[8109512,8109638)`; ten entries and zero next-page pointer |
| Object length | 8,109,638 bytes |

The retained scale `.05000000074505806` and observed dimensions produce `lround(1444/scale)=28880`, `lround(1404/scale)=28080`, exactly matching the published full-resolution raster. This resolves the apparent discrepancy with the earlier unflattened mesh dimensions without substituting coordinates or changing training contexts.

Each request covered only the8-byte header,2-byte IFD count or124-byte IFD table/next pointer. Offset and bytecount tags were inline, so no additional metadata-array requests were needed. The post-run receipt independently checks that all nine retained ranges lie within observed metadata extents and are disjoint from the coordinate strip. No coordinate samples, labels, masks, native CT chunks, scientific model states or predictions were read.

The canonical source ETags remain the inventory-pinned values. Per-object contracts and all response validators/raw metadata are retained under `run/`; result SHA `1de22c517b956121b9f2a48a6679db7bc8a0d035b2eeacde323667b6e9c63288`. Source freeze SHA `e8831df281d9d0f2734679f7c9de744d5cc5dc4f936edfdb551f8918d941e6ff`. Original text-phase manifest entries were rehashed and remain unchanged.

Geometry-valid support is still **not established**. Required remaining observations are bounded XYZ coordinate samples covering every bilinear/nearest-normal dependency for the fixed training extents, then a declared reconstruction with finite/nondegenerate normal checks and selected-depth CT interpolation bounds. Missing/all-fill native masked CT data and historical private optional-mask/render identity remain distinct limitations even after a public computational mesh-validity reconstruction. This result neither changes the separate frozen S/V allocation study nor authorizes coordinate-payload acquisition.
