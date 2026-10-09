# Proposed public-mesh computational support for the two fixed training contexts

**Proposal only: no coordinate payload has been acquired.** This contract is separate from the frozen S/V allocation study and cannot change its masks, labels, objective or decision rules. Prior text and structural evidence remains immutable.

## Exact request budget and support extent

The observed transformed XYZ TIFFs are1444×1404, little-endian IEEEfloat32, uncompressed single strip at byte8. Their exact object lengths and strong ETags come from the completed structural pass. The following applies identically to each of x.tif, y.tif and z.tif:

| Fixed context | Fine raster XYWH | Conservative mesh XYWH | Inclusive byte range per TIFF | Body bytes |
|---|---|---|---|---:|
| PHerc0139-w016-rank01 |7168,19456,1536,1536 |357,972,80,80 |5615708–6072331 |456624 |
| PHerc0139-w016-rank02 |14336,20480,1536,1536 |716,1023,80,80 |5911720–6368343 |456624 |

Exactly six sequential single-range GETs total2,739,744 body bytes; reserving one additional byte per request for strict framing gives2,739,750 bytes, below4MiB. There are no multipart requests, retries, redirects, full-object fallbacks or new header requests. Require the original If-Match ETag, exact206 Content-Range and unchanged total length. Any changed source stops; refreshing headers or extending support requires another contract.

Only153,600 of those bytes encode the six intended80² windows. Each response starts at the first window column of its first row and ends at the last window column of its last row. Decode exactly320 bytes at relative offsets `j*5776`, j=0..79. Preserve all2,586,144 intervening gap bytes as opaque bytes, never float-decode them or use them as new support. The overlap of the two requested broad row spans does not authorize observing additional coordinates.

## Raster-to-mesh dependency derivation

`derive_plan.py` enumerates synthetic indices, not coordinates. It uses observed mesh dimensions, declared FP32 scale `.05000000074505806`, exact128-square source-render tiles, the full-canvas half-pixel origin, FP32 center and offset arithmetic from commit650bd566. Both contexts are aligned to complete12×12 source-render tiles.

For each axis and tile, with render scale1 and crop0:

1. `center = float32(double(mesh_count)/2/double(scale32))`.
2. `origin = float32(-0.5f * float32(canvas_count - 1.f))`; tile offset is `float32(origin + float32(tile_start))`.
3. `ul = float32(float32(offset + center) * scale32)`.
4. Scratch mapping begins at `ox = float32(double(ul) - 4*double(scale32))`.
5. Retained output j uses scratch index j+4 and position `float32(ox + float32(float32(j+4)*scale32))`. The alternative single-FMA final multiply-add is also evaluated for an index-dependency envelope.

The union contains bilinear floor/+1 coordinate vertices, nearest `lround` validity/normal centers and each normal center's four axial ±1 neighbors. It yields exactly the80×80 windows above. Both evaluated multiply-add forms select identical bilinear and nearest indices for all relevant samples; maximum position difference is0.0001220703125 mesh units in Y. This is a bounded arithmetic-dependency check, not a proof of every historical compiler optimization or private image binary.

The source's temporary four-output-pixel padding is cropped away; the actual mesh dependency halo comes from bilinear and normal stencils. All16 aligned fine XY samples underlying each level2 output are included. The source performs two rounded2×2 reductions, so a pooled center or nearest fine sample is not a valid replacement for this dependency union.

## Proposed conservative reconstruction and outputs

A later implementation must pin its NumPy/runtime and arithmetic order. The synthetic prototype demonstrates the definitions; it is not yet an approved real-data driver.

- Load only the selected rows' little-endian float32 XYZ triplets. Apply explicit finite checks. Treat z≤0 and x=-1 as invalid, matching the loader/renderer sentinel route. Reject infinity conservatively even though the renderer's cached vertex-validity test does not universally reject it.
- A fine sample requires finite valid coordinates at all four bilinear vertices, the nearest mask/normal center, and its four axial normal neighbors. Require interior normal-grid bounds. This is deliberately conservative and does not claim to reproduce every renderer acceptance edge case.
- Compute bilinear position in staged FP32, preserving the declared raster mapping. Construct the nearest grid normal from axial differences and cross product; require finite positive squared norm, normalize, negate once for flip-normals, and apply the renderer's second normalization. Reject nonfinite base/direction. No new fitted transform, normal reversal, depth shift or mesh smoothing.
- Save an explicit fine-to-coarse **mesh/stencil support** mask: logical all of each4×4 fine block. Keep it separate from the native-frame-bound mask below. Both outputs are384×384 booleans; optionally retain fixed rejection-reason counts, but not a reinterpreted ink/background label.
- For each fine sample with valid geometry, the selected68 raw source depths21..88 correspond to offsets−33..+34 along that fixed flipped direction. Test full eight-corner trilinear-index support in the **declared original** native frame XYZ `[26511,26511,76953]`. The strict condition on every computed coordinate v is `0 <= v < dimension-1`. This rejects even an edge with a mathematically zero interpolation weight on an out-of-bounds corner; it is a conservative complete-stencil definition.
- Save a separate coarse **mesh-plus-declared-native-bounds** mask by logical all over all16 fine samples and the complete68-depth interval. The source catalog is pinned in NATIVE_FRAME_METADATA.json; the remasked successor must not replace the declared original volume.

Geometry base and directions remain constant across depth for a fine sample. Do not use a normalized intensity, nonzero voxel, source-acquisition mask, S/V membership or teacher prediction anywhere in this reconstruction.

## Why endpoint depth checks cover all68 depths

For each fixed finite component b,d, the declared CPU computation is `v(o)=RN32(b + RN32(d*o))`, where o runs through consecutive exactly represented integers−33..34. Correctly rounded multiplication by a fixed d is monotone in o (increasing for d≥0, decreasing for d<0). Adding fixed b and applying round-to-nearest remain monotone. Consequently every computed intermediate depth lies between the two computed endpoint values. If both endpoints are finite and satisfy `0 <= v < dimension-1`, all68 values do; if any intermediate violates a bound, an endpoint does too. Constant/signed-zero directions are included.

A fused multiply-add has the same monotonicity property, but this proposal does **not** silently mix it into the staged computation or assert bitwise private-render equivalence. Any alternate arithmetic path must be declared and independently checked before use. `floor(v)+1 < dimension` follows from the strict upper bound, so all eight integer corners are in range. No native CT data are required to establish this coordinate-index bound; it cannot establish that those corners were observed, unmasked or valid tissue.

## Synthetic evidence and required production checks

The retained synthetic receipt covers planar interpolation and normal sign, all68 depth support, NaN/Inf/x=-1/z≤0 vertices, a bad normal neighbor outside the bilinear cell, degenerate normals, mesh borders, exact/adjacent-float native boundaries, all16 fine-contributor reduction, opaque gap handling, and all four fixed-context axis footprints. An independent full68 enumeration matches endpoint decisions for2,048 fixed synthetic base/direction cases. These fixtures read no real coordinates.

Before any real execution: implement the bounded driver, pin all six range receipts/source identities, test truncation/oversize/changed validators/ignored Range and gaps on fake responses, and obtain root source review/freeze. During reconstruction assert every actual indexing operation stays in the derived window; abort rather than expand the range. Process128×128 fine tiles sequentially, with at most512MiB RSS and32MiB outputs. Freeze outputs/receipt names and resource enforcement before acquiring payloads; the current JSON remains `ready=false`.

## Limits on interpretation

The output would describe **reconstructed current public mesh geometric/computational support under the declared conservative arithmetic**. The current complete public prefix has no mask.tif; whether a mask existed in historical private staging remains unknown. The archived orchestration is marked dirty. No native CT chunks are read, so missing/all-fill blocks and masked-material/tissue validity remain unknown. A coordinate bound is not observed scan coverage, papyrus support, ink, transcription, model accuracy or an authenticated historical private renderer-validity mask.
