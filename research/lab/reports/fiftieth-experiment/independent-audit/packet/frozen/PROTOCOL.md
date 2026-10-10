# Phase48 — PHerc0800 sampling-support admission (CPU, geometry only)

Freeze before evaluating new candidate support. No detector outputs, CT intensity
acquisition, model inference, training, paid resources or surface snapping.
Use the five retained larger meshes and phase47 separated components. Preserve
all earlier studies and classify geometry separately from scientific accuracy.

1. Reconstruct the pinned unrotated, uncropped half-pixel base positions and
nearest-grid, cross-product, flipped unit normals from retained TIFXYZ, using
the archived 128-pixel tile / four-pixel halo float32 convention. Check bases
against the prior full coordinate maps (maximum allowed absolute error 0.01 CT
voxel). Require every bilinear and normal-stencil vertex positive and finite,
nondegenerate normals, and every proposed interpolation corner inside native CT.
Invalid/unknown normal support is excluded, not filled.
2. Proposed support is a fixed 66-layer slab at offsets k-32.5, k=0..65, along
the original unit normal. A later depth-window sweep stays within this slab;
this study does not approve extra slab translations. Both depth directions have
the same support. No claim of executing the PHerc343 renderer on this target.
3. Earlier surface exposures are the three maximal extents in phase47. Their
31-layer offsets are -15..15. Any trilinear source corner is less than sqrt(3)
voxels from its sample. The conservative nearest-old-centre exclusion is 51.1
voxels: greater than 32.5+15+2*sqrt(3)+0.02 for archived base rounding. Passing
this bound proves disjoint voxel support for these bounded slabs; failing it is
unresolved proximity, not proof of shared papyrus or exact shared voxels.
4. Include both earlier level3 CT cubes and all native-CT chunks used in the
first/third/fifth source checks. Treat entire retained chunks as previously
available, a conservative exclusion stricter than the actually plotted pixels.
For each candidate centre, bound all 66 trilinear-corner indices by the integer
AABB of its two normal endpoints, with an extra 0.01 voxel rounding allowance.
Exclude intersection with any previous CT box. Record this reason separately
from previous surface proximity. Published level3 cube support uses recorded
8x scaling and one extra native voxel of boundary uncertainty. This inventory
cannot establish no undocumented human exposure; scope all claims accordingly.
5. Require all native source pixels within Chebyshev radius128 of each scored
pixel to pass geometry and exposure checks (257x257 neighbourhood; outside
canvas is invalid). This covers any contributing 128px Villa patch under either
0/32 grid origin and the smaller native support of a 64px Hecate patch at9.6um.
It conservatively exceeds the exact fixed-grid union. Report lost area separately
for normal validity, old CT boxes, surface proximity and context erosion.
6. Candidate cores are restricted to nearest20px cells of the existing phase47
far components, with true full-resolution support rechecked. Consider the25813
largest component first, then every other component ordered by descending
sampled size and segment-ID/component-ID tie breaks. Require a contiguous
128x128 core minimum for a useful initial transfer pilot (not a prize-area gate).
Select the largest fully supported square in the first qualifying component;
break equal squares by smallest y then x. Retain masks and dimensions whether
or not anything qualifies. Failure stops target inference and prompts a coverage
or deliberately reused-development-field decision, not a weaker hidden gate.
7. Verify source identities against the phase47 cumulative seal, falling back
only to the existing immutable Git workspace manifest for early omitted records.
Independently test normal/base reconstruction at fixed sites against archived
point receipts, validate geometric bound/context operations, and replay the
selection from saved masks. Prepare raw-mesh inputs and explicit instructions
for a spare-VM independent implementation if that can challenge the conclusion.

A qualified geometry core remains neither ink ground truth, proof of a unique
physical sheet, nor a frozen complete model study. Cross-section / M7 ambiguity,
public CT availability and execution-time rendering qualification remain separate.
No source intensity is opened here. All new files are write-once until closure.
