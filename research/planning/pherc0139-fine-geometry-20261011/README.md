# PHerc0139 fine geometry and teacher input support

11 October 2026. **102 of the 192 fixed training candidates clear the source
geometry and native-reference exclusion screens. Eight of the twelve fixed
CT pilot sites are ready for a bounded CT-only correspondence check.** No CT,
annotation or prediction pixels were acquired in this qualification.

This makes the next experiment concrete without treating a shared surface ID
or an image resize as verified local scan alignment. The original native CPU
learning study continues with its own frozen inputs and endpoint rules.

## Fixed-cohort outcome

| Winding | Fixed candidates | Core geometry passes | Also clears teacher-input exclusions | Fixed CT pilot sites passing |
|---|---:|---:|---:|---:|
| w040 | 64 | 39 | 39 | 2 / 4 |
| w041 | 64 | 36 | 35 | 3 / 4 |
| w043 | 64 | 28 | 28 | 3 / 4 |
| Total training | 192 | 103 | **102** | **8 / 12** |
| w024 fine59 reference reservation | 32 | 2 | Not admitted | Not a training pilot |

The frozen core screen requires complete coordinate/normal stencils, maximum
centre-surface disagreement at most one native voxel and maximum normal angle
at most 10 degrees across the whole 128-square core. All 192 fine78 cores have
complete coordinate support; the rejected cores fail position and/or normal
agreement, often because their worst local discrepancy exceeds these bounds.
The original 192 coordinates and all failures remain recorded. No replacement
sites, fitted translations or teacher-response selection were used.

These are conservative source-parameterization gates. They select a geometry
stratum; they do not prove that rejected tissue is unreadable or that the
accepted tissue is representative of the entire scroll. All learner arms must
later share the same admitted support and declared exclusions.

## What the mapping check establishes

The fine and native meshes share the same normalized-mesh ancestry. Their
recorded transform matrices compose consistently. We implement the renderer's
128-pixel tile phase, staged float32 index calculations, complete coordinate and
normal stencils, and the correct source-level coordinate units. Continuous
native-to-fine queries compose the mesh-resampling half-pixel convention with
the inverse fine renderer, then interpolate the four integer rendered neighbors.

For the three training windings, the median of per-core median source-position
disagreements is **0.108–0.138 native voxels** with this mapping, compared with
**10.48–10.51 voxels** for naive full-raster resizing. At 9.362 µm per native
voxel, the latter is about 98 µm. These are coordinate-array comparisons, not
measurements of CT image registration. Both coordinate arrays can consistently
inherit the same imperfect acquisition transform.

Local worst-case disagreements remain substantial: maximum per-core position
errors reach 3.24 / 5.30 / 7.55 native voxels for w040/w041/w043. Some normal
disagreements are also large. We preserve those failures instead of inferring
pointwise supervision from the small typical residual.

## Complete teacher input support changes one decision

The retained v42 inference source is pinned to
`d4615b134ce1b62368bed1029ed47e5569ca03fb`; its current public tag agrees. It reads
256-square tiles on a stride-128 grid with a special final border tile, uses
fixed clipping/scaling within each tile, reverses depth where configured, and
blends predictions over that tile. The public fine78 teacher uses source layers
[23,85). We enumerate every contributing tile for the bilinear transfer, retain
entire tiles conservatively, and include every selected depth and trilinear
neighbor in physical bounds. This is a spatial input envelope, not a minimal
network receptive field or a replay of historical predictions.

**w041-x1088-y1088** passes the core mapping screen, but its conservative full
teacher-input box intersects the reserved native contexts **w035-x00640-y00768**
and **w035-x00768-y00768**. Its native input context alone had cleared the earlier
screen. The new intake excludes it. Box intersection means possible shared
input support; it does not establish that actual sampled voxels are shared.

The remaining 102 clear all 265 older native reference contexts, the three
previously inspected w040 domains, and the 64 new native diagnostic reservations.
They also clear the computable conditional w024 fine59 envelopes. Training
contexts are not asserted independent of one another. Upstream teacher training
exposure remains a separate limitation.

The catalog identifies production images by tag and records dirty orchestration
provenance. Source-code qualification does not cryptographically bind the
historical ECR container or weights to a reproduced output.

## w024 is still not an admitted alternate reference

Fine59 uses source group1: its mesh coordinates are full resolution, its rendered
base coordinates scale by 0.5, and its unit depth directions stay in source-level
voxels. Its rendered spacing is 2.258 µm and depth is 116 planes. Geometry is
mapped through fine59 → fine78 → native, with source-level scaling handled
explicitly. Reusing the group0 renderer would be incorrect.

Of the 32 unchanged w024 reservations, 22 have complete conditional teacher
geometry and 10 fit the conservative source extent. Only two pass the core
position/normal screen; **one**, w024-x2368-y2112, also passes the complete
conditional footprint/extent screen. That is still not reference admission.

The catalog's **v43** teacher implementation cannot currently be pinned: its
public tag returns 404 and the public matching-ref listing stops at v42. The
[source-scope addendum](SOURCE_SCOPE_ADDENDUM.md) preserves this finding. We used
only a clearly conditional, phase-independent 256-tile envelope for w024 planning;
unknown v43 processing remains unqualified. No v42 code is presented as a v43
reproduction. Local CT correspondence and valid reference pixels are also untested.

This narrows w024's immediate usefulness. Keep it auxiliary; the native
annotation references and author-designated w053 blank diagnostic do not depend
on v43. No new maintainer message or request was sent.

## Next bounded CT experiment

The original pilot rule selects order 0,16,32,48 in each training winding.
The four geometry failures remain failures. The eight survivors are:

| Winding | Core IDs |
|---|---|
| w040 | x2368-y5440; x6720-y4672 |
| w041 | x7488-y3136; x4672-y576; x320-y576 |
| w043 | x7488-y2880; x4928-y5440; x3648-y2624 |

[NEXT_CT_PILOT_PLAN.json](NEXT_CT_PILOT_PLAN.json) pins all URLs, metadata hashes,
ETags and expected lengths. All 72 live HEAD checks passed. The exact ceiling is
**86,114,304 bytes (82.125 MiB)**: 32 native chunks with all 28 depth planes and
40 fine group2 chunks with all 109 depth planes. Fine group2 is XY pooled to
9.596 µm; its depth sampling remains 2.399 µm. The earlier 448 MiB whole-intake
plan remains unexecuted; this smaller request is sufficient for the fixed pilot.

The next useful run is CT-only local correspondence on these eight sites, before
any fine teacher or annotation payload is examined. Freeze the CT scoring/split
protocol before GET, retain all failures and require source masks and held-out
local checks. It will determine whether correspondence is strong enough for
paired teaching on these specific supports. This geometry packet does not clear
that gate or overrule the earlier w040 pointwise-transfer failures.

## Verification and resource use

- 62 geometry/source checks pass: group0 regression, scalar group0/group1
  tile-phase indexing, analytic plane/unit-normal checks, every teacher depth's
  interpolation support, exact teacher border-grid coverage and physical exclusions.
- Independent per-pixel enumeration reproduces all pilot chunk coverage and the
  byte ceiling; all 72 HEAD receipts and metadata hashes verify.
- All 12 fine coordinate rasters were conditionally fetched and checksummed:
  164,736,024 bytes. The three new fine group2 array metadata bodies total 618 bytes.
- No new CT, label or teacher image bodies, neural inference, GPU rental,
  credential changes, Actions job or spare-VM assignment occurred.

See [PORTABILITY.md](PORTABILITY.md) for the private checksummed archive and
offline replay. Native geometry, source support, local CT correspondence,
teacher agreement and authenticated ink remain different claims.
