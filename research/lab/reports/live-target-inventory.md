# Live PHerc0800 target inventory

Retrieved 2026-10-07T22:31:42.535358+00:00. Verified read-only HTTP metadata, HEAD requests, and bounded TIFF directory ranges. No CT/surface chunks, model weights, or whole TIFF coordinates were downloaded by this inventory task.

**PHerc0800 / volume 20250521135224 remains a defensible provisional target.** Live inventory confirms six surface-volume Zarrs and six transformed TIFXYZ meshes explicitly mapped to that volume. The smallest segment gives a reproducible first ROI costing 507,904 bytes, plus 29,214 bytes for its three coordinate TIFFs. No text or ink is inferred.

## Source, eligibility, and snapshot correction

- Live inventory: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.min.json . Downloaded gzip wire bytes: 67,375; SHA256 `f6f1d6a7f917d60fa32edb9986a87a7aeafd2e47510f4d438de0b3142bc8a0f4`. The live JSON has no explicit top-level update timestamp; the retrieval time and asset Last-Modified headers are our provenance.
- Exact eligibility source: https://raw.githubusercontent.com/ScrollPrize/villa/e0bbb8b40a2db58b1d71864f286eb85717e59e64/scrollprize.org/src/data/prizeEligibility.json. Read-only Git HEAD still matched this commit during this check. The pair `PHerc0800` / `20250521135224` is present in both `first-letters-2027` and `grand-prize-2027`.
- The previous website bundled index had `updated=2026-06-15`, six segments but a curated `segmented=false` stage flag. The live origin records and object probes resolve availability; neither timestamp nor curated stage flag establishes segment quality.
- Acquisition scan `20250510225703`, ESRF Grenoble, 8.640 µm and 116 keV. Volume license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ .

## Verified surface volumes and meshes

All native surface arrays: Zarr v2, axes Z/Y/X = depth/row/column, dtype uint8, chunks 31×128×128, C order, uncompressed, slash-separated chunk keys, fill value 0. All six native arrays contain 31 depth slices; their metadata advertise six pyramid levels (0–5). Only Y/X are downsampled; depth remains 31 with metadata scale 8.64 µm. A single native chunk has HTTP Content-Length 507,904 bytes. All JSON and HEAD object probes returned HTTP 200; bounded TIFF directory reads returned HTTP 206.

| Segment ID | Native surface shape Z×Y×X | Transformed X-TIFF grid Y×X | Bytes per X/Y/Z TIFF | Surface Zarr and volume-mapped mesh |
| --- | --- | --- | --- | --- |
| `20251028213516` | 31×2160×2240 | 108×112 | 48,518 | [Zarr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028213516-auto_grown_20251028213516907/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/) · [TIFXYZ meta](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028213516-auto_grown_20251028213516907/mesh/20251028213516-on-20250521135224-8.64um.tifxyz/meta.json) |
| `20251028220042` | 31×2040×2000 | 102×100 | 40,934 | [Zarr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028220042-auto_grown_20251028220042762/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/) · [TIFXYZ meta](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028220042-auto_grown_20251028220042762/mesh/20251028220042-on-20250521135224-8.64um.tifxyz/meta.json) |
| `20251028220955` | 31×2100×2100 | 105×105 | 44,234 | [Zarr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028220955-auto_grown_20251028220955262/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/) · [TIFXYZ meta](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028220955-auto_grown_20251028220955262/mesh/20251028220955-on-20250521135224-8.64um.tifxyz/meta.json) |
| `20251028222030` | 31×2580×2580 | 129×129 | 66,698 | [Zarr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028222030-auto_grown_20251028222030940/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/) · [TIFXYZ meta](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028222030-auto_grown_20251028222030940/mesh/20251028222030-on-20250521135224-8.64um.tifxyz/meta.json) |
| `20251028225813` | 31×2300×2260 | 115×113 | 52,114 | [Zarr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028225813-auto_grown_20251028225813045/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/) · [TIFXYZ meta](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028225813-auto_grown_20251028225813045/mesh/20251028225813-on-20250521135224-8.64um.tifxyz/meta.json) |
| `20251029010146` | 31×980×980 | 49×49 | 9,738 | [Zarr](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251029010146-auto_grown_20251029010146642/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/) · [TIFXYZ meta](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251029010146-auto_grown_20251029010146642/mesh/20251029010146-on-20250521135224-8.64um.tifxyz/meta.json) |

No ink predictions, binary ink labels, or standalone flattened-image renders are listed for PHerc0800 in the live projection. The available `layers-zarr` volumes are ready surface-aligned data for subsequent visualization/inference. Intermediate original/flattened/normalized OBJ and TIFXYZ artifacts are also listed, but the **transformed** TIFXYZ paths above explicitly declare `target_volume=20250521135224` and are the correct geometry to preserve for this target.

Each coordinate TIFF was HEAD checked, and X-TIFF dimensions/sample metadata were read via HTTP 206 byte ranges: float32, one channel, uncompressed. Their TIFF directories sit at the end of the objects; the bounded header probe followed the IFD offset. Y/Z grids were HEAD checked only; their exact shape/content still needs reading when mapping the ROI.

Full object URLs, ETags, Content-Length, Last-Modified, parsed JSON, mesh bounding boxes and successful TIFF metadata ranges are retained in [live-target-inventory.json](live-target-inventory.json). All checked transformed-coordinate and surface artifacts show Last-Modified September 7, 2026.

## First bounded ROI and exact provenance mapping

Use segment `20251029010146` (`20251029010146-auto_grown_20251029010146642`), native level `0`, array ROI `[:,384:512,384:512]` (Z=0:31, Y=384:512, X=384:512). Its single chunk is HEAD verified:

- Native surface chunk: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251029010146-auto_grown_20251029010146642/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/0/0/3/3 ; Content-Length **507,904 bytes**.
- Start visualization at depth index 15 while retaining all 31 slices for inspection. This does not establish the physical zero-depth offset or identify ink.
- Surface nominal planar scale: 8.64 µm/pixel, so the 128×128 ROI spans 1.10592×1.10592 mm before any flattening distortion assessment.
- Coordinate mapping metadata: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251029010146-auto_grown_20251029010146642/mesh/20251029010146-on-20250521135224-8.64um.tifxyz/meta.json ; `scale=[0.05,0.05]`, transformed grid 49×49, render canvas 980×980. The recorded scale and dimensions indicate a 20-render-pixel grid spacing; preserve these values and use the maintained TIFXYZ interpolation conventions rather than treating the 49×49 grid as 980×980 direct coordinates.
- Coordinates: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251029010146-auto_grown_20251029010146642/mesh/20251029010146-on-20250521135224-8.64um.tifxyz/x.tif ; `y.tif` and `z.tif` in the same directory. Each is **9,738 bytes** by HEAD; all three total **29,214 bytes**.
- Native source-coordinate bounding box (X/Y/Z from transformed mesh metadata): `[3089.9785,5907.0840,11617.8838]` to `[3703.0190,6614.2456,12497.9766]`. This fits within the verified native CT shape (Z/Y/X) `[24298,9867,9867]`. Validate coordinate grid masks/missing values before choosing any region for analysis.
- The catalog associates both surface data and transformed mesh with the exact eligible volume. Surface `.zattrs` also records internal `source_zarr=/volumes/esrf/20250506/SCROLLS_HEL_4.320um_1.0m_116keV_binmean_2_PHerc0800_TA_B_0001_masked.zarr/`, `source_group=0`, `slice_step=1.0`; this is an internal provenance string, not a public path or authorization to switch target scans.

A coarse overview of the **entire smallest surface** costs slightly less than the native tile and can help choose a fiber-visible, valid geometry region:

- Level 3 metadata is verified as shape `[31,123,123]`, one uncompressed chunk `[31,123,123]`, Y/X scale 69.12 µm; depth remains 8.64 µm.
- Overview chunk: https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251029010146-auto_grown_20251029010146642/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/3/0/0/0 ; HEAD Content-Length **468,999 bytes**. This is a diagnostic overview, not sufficient evidence to judge letter legibility.

## Scientific limits and next step

The first ROI is selected to bound cost and preserve complete source mapping. Neither papyrus-fiber visibility, mesh validity, ink presence, letter legibility nor any translation has been established by these metadata/HEAD checks. Inspect the coarse surface and coordinate masks first, then refine the native ROI and compare it to the original CT using the transformed mesh. Record actual fetched-data hashes and exact slice/UV/voxel coordinates for every later artifact.

The relevant released future ink baseline remains `scrollprize/ink_9um`, whose stated training scrolls are PHerc0139, PHerc1667, PHercParis4 and PHerc0814. PHerc0800 is not in that list; verify checkpoint/training provenance before calling results held out. The public PHerc0139/w035 tutorial is in its training set. Display enhancement or agreement across model seeds cannot establish true letters.

The discovery rules still require technical reproduction, held-out validation, false-positive mitigation and papyrological review. Keep original recovered text private until official prize announcement. This inventory resolves data access and target availability, not award qualification.
