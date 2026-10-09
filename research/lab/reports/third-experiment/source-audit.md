# PHerc0800 larger-surface source audit

Audit date: October 8, 2026. This audit covers geometry acquisition and selection, before larger-surface chunk reads by the root experiment.

**The public inventory, five source meshes, native surface metadata and archived render provenance match the preserved snapshot.** Geometry selection used coordinate data only. The chosen 1536 × 1536 crop remains segment `20251028222030`, X/Y origin `(512,512)`.

## Snapshot preservation and current-source checks

- A bounded fresh read of [metadata.min.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.min.json) returned the same 67,375 gzip bytes and SHA-256 `f6f1d6a7f917d60fa32edb9986a87a7aeafd2e47510f4d438de0b3142bc8a0f4` as the prior snapshot. The PHerc0800 inventory compares equal after decoding. The old snapshot was preserved.
- A fresh HEAD of [full metadata.json](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json) matched the retained ETag `b4788ae14c64e13e57346c07c3f0b8a5` and Last-Modified October 5, 2026, 15:58:43 UTC. The retained full snapshot's SHA-256 is `7ec8dcaa3000b9716724c26a1991ee416491d8a6b3615238ca90ae11540abdd3`; it was checked before extracting render records. No new full-metadata body was downloaded.
- Each newly retrieved native surface `.zarray`, surface `.zattrs` and mesh `meta.json` matched its prior SHA-256. All 15 full coordinate TIFF downloads matched their previously retained object ETags. Exact source URLs, SHA-256, byte sizes, headers and retrieval times are retained in `data/experiments/pherc0800-larger-geometry/source-records.json`.
- The combined metadata and mesh HTTP payload was 844,642 bytes. No source surface chunk, CT chunk or model output was read by the geometry-selection helper.

These checks establish that the inspected source objects and published metadata snapshot remain unchanged. They do not independently prove the underlying scan's quality or every rendered pixel.

## Geometry-to-volume mapping

All five transformed meshes explicitly target volume `20250521135224`; source surface metadata records 8.640 µm Z/Y/X spacing at native level 0, 31 slices, slice step 1, uint8, C order, no compressor, fill value 0 and chunk shape `[31,128,128]`. Coordinate TIFFs decode as float32. Their positive finite vertex bounds match each mesh's reported CT XYZ bounding box within 0.002 voxel. Stored grid dimensions divided by scale `[0.05,0.05]` match the native canvas dimensions.

The archived [C++ renderer](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/apps/src/vc_render_tifxyz.cpp) and [QuadSurface](https://github.com/ScrollPrize/villa/blob/530584554c031c982ba3aa243e1ea39c4bef9818/volume-cartographer/core/src/QuadSurface.cpp) use native pixel centers `(x+0.5,y+0.5)` before multiplication by mesh scale. The helper preserves that convention. It requires all four neighboring bilinear vertices, excludes unsupported edges and writes NaN outside the mask. It does not use the inconsistent Python convenience interpolation convention identified earlier.

Conservative mask here means base-coordinate support. The archived normal cache additionally requires neighboring grid vertices and nondegenerate central differences; boundary support in the coordinate mask alone does not guarantee a sampled depth direction. The complete 128 × 128 geometry-neighborhood score provides stronger interior context, while source occupancy and observed intensity remain separate checks.

The selected mask and full CT base mapping were frozen before source acquisition. Their hashes are in `geometry-selection.json`. No frozen JSON, metadata, source TIFF, mask or coordinate map was changed after the root selected the crop.

## Exact render provenance

Each surface's official render record specifies source commit `530584554c031c982ba3aa243e1ea39c4bef9818`, binary `vc_render_tifxyz`, group 0, scale 1, 31 slices, `flip-normals=true`, rotate 0, `flip=-1`, zero crop, no affine or segmentation-scale overrides, no composite accumulation, and an uncompressed Zarr with `/` chunk separator. The full records are copied into each segment's geometry folder with a link to the revalidated full metadata snapshot. Internal volume-path strings are retained as historical provenance, not accessed as private data sources.

The expected depth stack follows fixed offsets −15 through +15 native voxels along the recorded flipped normal. The archived sampler uses bilinear base geometry, a nearest cached grid normal, trilinear CT interpolation and clipping then uint8 truncation. The separate [native CT consistency report](native-ct-consistency.md) checks three geometry-fixed pixels in this selected segment without fitting any transform or depth offset.

## Interpretation and scope

The parent requested the best 1536-square crop first; the global 2048/1536 ranking was retained separately. [geometry-selection.md](geometry-selection.md) clarifies this size preference and the exact 69.75 MiB surface payload bound. The larger crop is broader spatial context in the same scan. Same-volume boxes overlap the first surface, and adjacent wraps or the same through-plane slab remain possible; consult [design-review.md](design-review.md) for dense correspondence evidence.

The PHerc0800 scan's published inventory declares CC-BY-NC 4.0. Coordinate and scan data remain outside the repository in the local lab. Geometry support, source consistency and detector agreement do not authenticate ink, letterforms or translation. No new prize-eligibility or reading claim is made by this source audit.
