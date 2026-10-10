# PHerc0139 fixed-training geometry trace

Status: **geometry-valid support remains unestablished**. This is a retained-source audit and a proposed text-only acquisition contract, not a geometry mask or authorization to decode coordinates. Root owns scientific decisions and network execution. No arrays, weights, CT samples, labels, coordinate payloads or predictions were read here.

## Provenance bridge

The curated alias `pherc0139-w016` explicitly names public surface segment `20250108000004-w029_2025010827`, target volume `20260102150214`, in the seventh-study supervision/validation `.zattrs`. Thus the alias difference itself is not a mismatched specimen. See the pinned records in [SOURCE_BRIDGE.json](SOURCE_BRIDGE.json) and the research agent's [literal export extract](../research-context/PHERC0139_RENDER_LEAD.json).

The fixed contexts are level-2 XYWH rank01 `[1792,4864,384,384]` and rank02 `[3584,5120,384,384]`. Their source shape is ZYX `[109,7020,7220]`. The published surface attributes declare canvas width/height `[28880,28080]`, XY mean downsampling and source slice step 1.0. Existing depth construction uses source `[13,97)` in four-plane groups, followed by pooled planes `[2,19)`, therefore model input depends on source planes `[21,89)` exactly. Geometry must cover every constituent plane and every constituent XY sample, not merely a central plane or a pooled center.

The 32-step schedule's six eligible 128-square tiles per context touch 53,248 unique rank01 pixels and 49,152 rank02 pixels (102,400 total). Rank01's local union is `[256,64,128,320]` plus `[192,192,64,192]`; rank02 is `[0,0,192,256]`. These are coordinate arithmetic from the fixed tile list, not a newly observed validity mask. Full contexts remain the conservative proposal extent; no support expansion or reduced acquisition has been executed.

The recorded transformed mesh applies affine diag(4,4,4) to a normalized mesh. The renderer consumes that **already transformed** mesh with no additional affine, scale-segmentation unset, group 0, scale 1, rotation 0, flip −1 and flip-normals true. Do not apply the geometry affine a second time. The original unflattened mesh properties width/height `[37679,38799]` and original grid `[1884,1940]` at scale approximately .05 do not define the normalized/transformed raster. Actual transformed `meta.json`, coordinate TIFF dimensions and exact rendering arithmetic must reconcile the published canvas.

## Exact-source blocker

The locally retained implementation is villa commit `530584554c031c982ba3aa243e1ea39c4bef9818`, authenticated by `reports/first-experiment/native-ct-consistency.json` lines 618–648. This differs from the PHerc0139 declared renderer commit `650bd56663efb7225fdba73fe267ba86b8e29368`. It is an investigation lead only, not the exact implementation contract.

In that retained, different version:

- `QuadSurface.cpp:2064–2197` reads mesh scale, x/y/z TIFFs, invalidates z≤0 and optionally applies `mask.tif` by default; an absent mask cannot be assumed without checking the mesh inventory and exact loader.
- `QuadSurface.cpp:862–913,948–1130` uses a vertex-validity cache, coordinate interpolation, separately sampled validity and cached normals. Nearest validity is not automatically equivalent to all contributing interpolation vertices being valid.
- `vc_render_tifxyz.cpp:1496–1555` replaces x=-1 sentinels, calculates raster dimensions from the loaded mesh scale and handles crop; `:263–289` prepares positions/normals. None of these details may be imported silently into the older commit.

The exact-source query in [TEXT_QUERY_PLAN.json](TEXT_QUERY_PLAN.json) resolves those defaults, plus geometry/normal and pyramid helper dependencies. A raw GitHub path returning an error stops the bounded phase; it does not authorize branch fallback or unpinned replacement. Archived deployment orchestration is marked dirty, so a declared image commit supports intended renderer identity but is not an independent proof of every historical output byte.

## Dependency extent to establish before binary acquisition

If the exact pyramid implementation confirms aligned 4×4 level-0 support per level-2 pixel, the full contexts require fine-raster boxes rank01 `[7168,19456,1536,1536]` and rank02 `[14336,20480,1536,1536]`. These are **conditional constituent bounds**, not authenticated mesh-grid ranges. A level-2 pixel's mean footprint differs from evaluating one point at four times its coordinate; zero OME translation does not remove that distinction.

Derive the mesh-grid footprint by applying the exact renderer origin/scale/rounding to every required fine-raster sample and taking the union of coordinate interpolation vertices, validity-mask samples and all normal-estimation neighbor stencils. Then add the exact CT interpolation neighbors for each selected depth offset. Halo size is deliberately unset: a generic one-vertex or four-pixel halo borrowed from another commit would be unjustified. Determine optional mask scale and first-channel aggregation independently. Future TIFF header/IFD reads must authenticate endian, float representation, compression, tile/strip extents, canvas and validators before choosing byte ranges; all payload/decompression caps require a separate plan.

## Separate validity claims

Keep these separate in any later receipt:

1. Released annotation-transfer participation: supervision `S`, validation `V`, or explicitly declared `S|V`. Their shared relaxed-mapping/cleanup policy does not establish geometry outside that participation.
2. Mesh-coordinate/raster validity under the exact loader and interpolation rules, including optional mesh mask and sentinel handling.
3. Finite nondegenerate normal and complete selected-depth interpolation support inside the exact native CT volume frame/bounds.
4. Native masked-volume or material support: coordinates being in bounds does not establish valid tissue or papyrus everywhere.
5. Local acquisition completeness and nonzero intensities, which the older pipeline recorded but which establish none of items 2–4 by themselves.

Unknown or unmapped pixels cannot become background or teacher-preservation support by negating an ink label. A future conservative strict-all-constituents mask would also need to be explicitly distinguished from the renderer's own acceptance/fill behavior. This audit does not choose the training support or a new objective.

## Proposed bounded next source query

Maximum ten sequential attempts, 2 MiB total response-body cap, 30 s per request, zero retries/redirects; proposed individual reservations total 2,031,616 bytes. Eight exact-commit source texts plus one constrained immediate-prefix inventory and conditional `meta.json`. No binary geometry, mask or CT object is requested. Full-cap failure charging, partial evidence retention and strict complete-body parsing are required. The proposal is marked `ready=false` until root reviews and freezes its transport implementation and execution plan.
