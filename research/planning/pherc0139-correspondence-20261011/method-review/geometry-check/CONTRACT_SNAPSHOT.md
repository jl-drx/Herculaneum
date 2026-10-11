# PHerc0139 w040 fine/native coordinate contract

Independent geometry review, 11 October 2026. Exact segment:
`20250831000000-w040_2025083102`. The initial retained-source review acquired
no network objects. The authorized addendum below fetched four small pinned
source files and read root's coordinate TIFFs. Neither pass read CT samples,
labels, or prediction pixels. No inference, active-study change, or Git
operation was performed.

**Decision:** the two public surface volumes have a documented common-source
surface and compatible depth coverage. This permits a bounded CT-only local
correspondence experiment. It does not yet establish pixelwise registered
supervision. The index mapping between the two transformed coordinate grids
must be checked; resizing one complete raster to the other's shape is not an
independent registration measurement.

## Identity and pinned implementation

The retained PHerc0139 catalog is
`reports/fortyfifth-experiment/source-evidence/PHerc0139-retained-catalog.json`,
SHA256 `2a460d1f57ca0d3535ce2a8a8ab8b80fe4141f143e8de486394a150bc93f094d`.
Within `segments.20250831000000`, both transformed meshes derive from
`mesh/intermediate/tifxyz_normalized/`, generated from the same normalized OBJ.
Do not use the original traced grid's 431×615 dimensions or its original
8619×12299 nominal canvas as either transformed raster's dimensions.

| Item | Fine78 | Native113 |
|---|---|---|
| Volume | `20260102150214` | `20250728140407` |
| Scan | `20251205005141` | `20250720065842` |
| Catalog voxel size | 2.399 µm | 9.362 µm |
| Group-0 surface ZYX | 109×24880×31040 | 28×6400×7980 |
| Group-2 surface ZYX | 109×6220×7760 | Not proposed as input |
| Transformed mesh scale XY | .05000000074505806 each | .05000000074505806 each |
| Mesh-transform commit | `650bd56663efb7225fdba73fe267ba86b8e29368` | `d9a472ea2f50d9ce4ce56303cc1d682d962490d9` |
| Render commit | `650bd56663efb7225fdba73fe267ba86b8e29368` | `530584554c031c982ba3aa243e1ea39c4bef9818` |

The newly fetched metadata underlying this table are the seven JSON bodies
under `../metadata/responses/`; their source receipts are in root's metadata
ledger. At the time of this review, coordinate TIFF dimensions still require
header confirmation. The renderer and published dimensions imply grids
1552×1244 (fine) and 399×320 (native). These are inferred width×height values,
not measurements of downloaded TIFFs.

Fine78 mesh coordinates are `4 * normalized_source_coordinate`. Native113
coordinates apply the catalog's fine-to-native affine after that scaling:

```
p_native = A p_fine + b
A = [[-0.2569036650583705, -0.009715391643719536,  0.002790733227688403],
     [ 0.009850885045974334,-0.25680016838938274, -0.0014215065457306453],
     [ 0.0029113360690184954,-0.0014844301309079428,0.257134581099402]]
b = [6677.364709056732, 6689.120376657908, 624.1652625002474]
```

The native mesh-transform command records the linear block `4A`, with the
same `b`, because its input is the normalized source grid. **Do not apply the
affine a second time to an already transformed native mesh.**

Catalog production orchestration is marked dirty. Commit/image identity is
the declared implementation contract, not proof that every current public
byte reproduces the historical internal rendering. Current mesh metadata UUID
`out` is generic; exact paths and creation provenance supply identity.

## Raster XY mapping and the resize pitfall

Both renders record group 0, scale 1, no additional segmentation scaling or
affine, no crop, rotate 0, flip −1 (no image flip), and flip-normals true.
Both source renderers compute canvas origin `u0 = -(W-1)/2` and use the loaded
surface center `columns/(2*scale_x)`. In real arithmetic, at render pixel `x`,

```
mesh_x(x) = scale_x * (x - (W-1)/2) + mesh_columns/2
```

When `W = mesh_columns/scale_x` exactly, this simplifies to
`scale_x*(x+0.5)`. The Y equation is analogous. The source implementation
stages FP32 arithmetic at 128-pixel tile origins and inside the scratch halo;
preserve that arithmetic for faithful sampling/support checks. Do not
silently replace it with a float64 simplification at nearest-sample ties.

XYZ positions are bilinear interpolation of four mesh vertices. Validity and
cached normals are sampled independently at rounded nearest vertices. The
normal cache uses ±1 X and ±1 Y neighbors on the coordinate grid. Thus a point
needs a four-vertex coordinate footprint plus a possibly different
five-vertex normal footprint. Conservative geometry validity should require
all constituent finite vertices, valid normal, and complete CT interpolation
footprints. This is stricter than the renderer's nearest-validity test and
must be named accordingly.

The transformed grids demonstrably differ in sampling density: their nominal
width ratio is 1552/399 and height ratio 1244/320. Equal integer mesh indices
therefore cannot mean equal source-surface parameters. A shared segment ID
only establishes intended common tissue. It does not establish that the two
rendered columns hit the same fibers or the same sheet after registration.

`cv2.INTER_AREA` on a full fine image to native `(H,W)` implements an image
area resampling convention. Applying it to an already same-shaped image is
an identity operation; it cannot correct alignment. Resizing full rasters to
the same shape cannot prove their physical correspondence either. Width and
height ratios differ slightly because transformed mesh dimensions are rounded.

The transformation command's own coordinate-grid resampling matters. For
example, a conventional half-pixel coordinate resize maps a destination
mesh index `q` to `(q+0.5)*source_count/dest_count - 0.5`. Composing this with
the renderer mapping is generally **not** the same as applying a half-pixel
resize directly to the output rasters: there can be a fixed translation as
well as scale. This is a concrete mechanism to test, not an observed error
in w040. The exact `vc_transform_geom.cpp` at native commit `d9a472...` was
not retained as source during this independent review. Do not import the
newer `QuadSurface::resample` behavior without checking its caller/version.

The decisive metadata/coordinate check is to evaluate corresponding native
and fine surface coordinates using each renderer contract, map fine XYZ
through `A,b`, and report the residual vector, normal angle, and validity.
Use a source-derived index correspondence; if fitting a correspondence to
coordinate grids is necessary, distinguish fit points from unused validation
points. This tests the surface parameterization and transformation pipeline.
It is **not** a CT registration measurement: the same erroneous global
affine can be used consistently in both coordinate arrays.

## Depth, units, and normals

With group 0 and recorded slice_step 1, exact source offsets are:

```
fine depth k:   (k - 54)   * 2.399 micrometers, k=0..108
native depth j: (j - 13.5) * 9.362 micrometers, j=0..27
```

Offsets follow each renderer's **flipped unit geometric normal**. Plane54
is the fine surface center; the native center lies halfway between planes13
and14. Native plane14 is +4.681 µm, not zero. Consequently, a simple physical
depth correspondence is `k = 54 + (j-13.5)*9.362/2.399`. It sends j0/j27 to
k1.316799/k106.683201. The fine full range is ±129.546 µm and native range is
±126.387 µm. All fine109 depth planes remain present at group2: group2 is
9.596 µm in XY but **2.399 µm in Z**, not an isotropic coarse input.

Keep two different comparisons separate:

1. **Nominal physical sampling:** the equation above uses the catalog voxel
   sizes and matched distance along each actual geometric normal.
2. **Mapping through the empirical affine:** the vector `A*n_fine` and the
   independently reconstructed native normal specify how source points
   move under the released transform. These need not exactly equal the
   nominal micron ratio.

Indeed `det(A)=0.016991152204927057`; its cubic-root scale is
0.2570835431431352, versus nominal 2.399/9.362 = 0.2562486648152104, a
0.326% difference. Column norms also differ slightly, so `A` is not an exact
scaled rotation. Under a general affine the transformed geometric normal is
proportional to `det(A)*inverse(A).T*n`, not simply `A*n`. Compare the actual
normals generated from the coordinate grids. Do not infer biological
recto/verso or ink side from `flip-normals=true`; catalog `text_side` is null.

Fine group2 pixels are rounded means over aligned 4×4 source XY footprints,
implemented as two successive 2×2 integer rounded reductions. A pooled pixel
is not a CT sample at one center point, nor generally an exact unrounded
16-pixel mean. For coordinate attribution report its footprint/support or a
clearly declared representative center. For CT correspondence use explicit
antialiasing/resampling and distinguish this blur from geometric alignment.

## Reuse and qualification limits

Useful existing files under the lab root:

- `reports/twentyfirst-experiment/pherc0139-geometry/text-run/`: exact fine
  renderer, loader, normals, CT sampling, and pyramid source with source
  URLs/hashes in `ledger.json`.
- `data/experiments/pherc0800-native-ct-consistency/export-source/`: exact
  native renderer, `QuadSurface.cpp`, `Geometry.cpp`, and `Slicing.cpp`.
  Their source identities are in
  `reports/first-experiment/native-ct-consistency.json`.
- `reports/twentyfirst-experiment/pherc0139-geometry/structural/`: bounded TIFF
  structural readers and transport fixtures. They are reusable machinery;
  old w029 dimensions, payload lengths, validators, and offsets are not w040
  facts.
- `reports/twentyfirst-experiment/pherc0139-geometry/public-geometry-implementation/geometry_kernels.py`:
  staged raster mapping, vertex/normal validity, and source support arithmetic.
  It hard-codes old w029 dimensions, 80×80 windows and depth [21,89); reuse
  the reasoning, not the unmodified executable or its frozen query plans.
- `research/planning/plinius-research-review-20261011/paired-data/` in the
  repository: exact global affine residual replay, exposure matrix, and
  pinned teacher/native scripts. Those supplied-landmark residuals are fit
  residuals, not independent local registration errors.

The catalog's w040 overlap ratios of 1 are axis-aligned bounding-box proxies,
not surface-validity or masked-material proofs. Nonzero CT maximum is useful
acquired support, but cannot prove the intended papyrus sheet or absence of
ink. Missing samples remain unknown. A successful geometry check should
precede independent CT structure correspondence; a successful correspondence
check on one w040 area admits that area only, not a whole-surface guarantee,
teacher independence, or a pristine evaluation benchmark.

Key exact source hashes:

| File | SHA256 |
|---|---|
| Fine renderer | `0d3423489028443151c62e735222357a5a17293a43633860ac05425eac1b5d57` |
| Fine loader/raster | `14d250f20a17ac7a34728fbb99e0aa12c533a79be00e10c30e4439495f50a897` |
| Fine pyramid | `65bf6d1e41c0d1c3f2f344d2b632a5ad6460b6312f98e35a2b82b0b1c657a217` |
| Native renderer | `cda11ab76be310aa7e07fb09d9e01234f4aecc6f2977b1ee40966e77812fd002` |
| Native loader/raster | `4adeeb75ff144c032efc8db56cee4f2b5778de047fb36f42b717a96458ab0406` |
| Shared normal code | `5919bea504e3e22329b3f9c41c1de4020a0c5781435fc83be84692abdaaa50bc` |
| Shared CT sampler | `31aa0fe39d171eed0ebb3316a598658c7ae6f846aa222aa9137256c84732a83d` |

No recommendation here changes study52 or uses prediction quality to select
the correspondence region.

## Addendum: transformed-grid mapping resolved and independently checked

This addendum supersedes the transformation-helper and TIFF-dimension
unknowns above. Four bounded public source requests, with no retries or
redirects, returned 200 and consumed **36,792 bytes**. Plan and receipt:
`SOURCE_REQUEST_PLAN.json` and `SOURCE_REQUEST_LEDGER.json`. Each request's
cap was 512 KiB. The helper implementation and declaration are byte-identical
at the fine650bd and natived9a472 commits:

- `AffineTransform.cpp`: SHA256
  `b4664a1ad4883558083f68edf5131f6e07734f260bcc0cd370033c6353dcd93a`.
- `AffineTransform.hpp`: SHA256
  `ca586a7a45e9d13597526fc0f55873ff78d23dd4913c8e3b3c4d802287d9f4b4`.

`transformSurfacePoints` (helper lines366–440) measures median neighboring
vertex spacing along each grid axis before and after applying the affine.
It calls `surface->resample(spacingScaleX, spacingScaleY, INTER_LINEAR)` and
then **restores the original mesh scale**. The transform app calls this
helper at line242. This explains why the transformed grid dimensions change
while both stored scales remain .05. It also explains why the X and Y ratios
are not precisely equal: measured axiswise changes and rounded dimensions
matter, not merely the affine determinant.

Both exact `QuadSurface` implementations resample using invalid-preserving
bilinear interpolation with the usual pixel-center mapping:

```
q_source = (q_destination + 0.5) * N_source / N_destination - 0.5
```

The native helper appears at lines99–141 of `d9a472e-QuadSurface.cpp`; the
fine counterpart appears at lines104–146 of the retained fine loader. The
new TIFFs confirm actual fine shape1244×1552 and native shape320×399 (YX).
Both descend from the same normalized source. Therefore the common source
count cancels, giving the **source-derived** relation between the transformed
grid parameters, away from clamped boundaries:

```
q_fine = (q_native + 0.5) * N_fine / N_native - 0.5
```

This relation requires no fitted CT shift and no ink/teacher information.
It equates the underlying source parameter location, although nested
bilinear interpolation of a curved surface can still produce small XYZ
differences. Near an original grid-cell kink, invalid footprint, or clamped
border, equality of the continuous parameters does not imply identical
interpolated coordinates or identical normal stencils.

Composing this equation with each renderer's mapping produces, in ideal
arithmetic with equal mesh scale `s=.05` and exact canvas dimensions:

```
r = N_fine / N_native
x_fine = (x_native + 0.5)*r - 0.5 + 0.5*(r-1)/s
```

The first two terms alone are the ordinary full-raster center resize.
The final term is missing from that resize. For w040 its value is:

| Axis | r | Additional fine pixels | Same distance in native raster pixels |
|---|---:|---:|---:|
| X | 1552/399 = 3.88972431 | +28.89724311 | +7.42912371 |
| Y | 1244/320 = 3.8875 | +28.875 | +7.42765273 |

For a representative center of a fine level2 footprint, convert full fine
coordinate `x_fine` by `(x_fine-1.5)/4`; the level2 mapping has additional
offsets +7.22431078 (X), +7.21875 (Y) compared with directly resizing native
to fine-level2 canvas centers. Actual code should compose the full
source/renderer equations with recorded scales and staged arithmetic, not
hard-code these rounded numerical offsets. INTER_AREA has area footprints;
these center equations alone do not claim exact equality of resampled CT or
teacher values.

The coordinate-only check `verify_source_mapping.py` uses a fixed 3×3 grid
at quarter, half and three-quarter positions of each native mesh dimension
minus1. It reads the six XYZ TIFFs, applies the saved affine, and compares
the source-derived mapping against the naive ratio-only mesh mapping.
All nine sites have complete finite bilinear coordinate footprints.

| Check | XYZ residual range, native voxels |
|---|---:|
| Source-derived half-pixel mesh mapping | 0.02724–0.33677 |
| Naive ratio-only mesh mapping, equivalent ideal center resize | 10.46604–10.65520 |

Every residual vector, input hash and coordinate is in
`SOURCE_MAPPING_COORDINATE_CHECK.json`. This is a float64 representative
geometry calculation, not an exact staged FP32 renderer replay; it sampled
no normals, CT, labels, or model outputs. It validates the predicted
parameterization correction at these nine sites. It neither establishes
local fine/native acquisition registration nor measures ink-label accuracy.

**Recommendation:** use the source-derived XY correspondence as the fixed
baseline for local CT correspondence testing. Retain ordinary resize as a
named geometry-control comparison if useful. Do not estimate away this
known coordinate shift with a CT optimizer and call that improvement new
registration evidence. Evaluate any remaining CT displacement separately,
with independent support and checks against ambiguous neighboring sheets.
