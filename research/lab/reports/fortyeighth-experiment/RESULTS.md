# PHerc0800: fresh-field support gate not met

The full-resolution geometry check completed on all five retained larger meshes.
No component contains the frozen minimum 128×128 evaluation core after requiring
valid normals, separated depth/interpolation support and complete model context.
The largest remaining square is 69×69 on segment20251028225813 at x1093,y318.
The earlier20px screen therefore did not establish a usable fresh model field.

| Mesh | Valid base pixels | Valid normal pixels | Point support after old-surface/CT exclusions | Full-context pixels | Largest full-context square |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20251028213516 | 2,313,200 | 2,282,000 | 251,463 | 0 | 0 |
| 20251028220042 | 2,150,800 | 2,111,200 | 184,696 | 0 | 0 |
| 20251028220955 | 2,067,600 | 2,044,900 | 690,233 | 11,754 | 44 |
| 20251028222030 | 3,159,600 | 3,130,900 | 710,588 | 8,160 | 35 |
| 20251028225813 | 2,484,000 | 2,454,100 | 369,449 | 11,442 | 69 |

All40 earlier separated components were considered in the frozen order. Even
without restricting cores to those components, no128px square survives. The69px
lead is about0.60mm wide at8.64um; the128px criterion was an operational initial
pilot requirement, not a prize-rule minimum. No detector or intensity was read.

## What this establishes

New bases and nearest-grid normals were reconstructed from the source TIFFs using
the archived centered-canvas, tile/halo and half-pixel convention. Bilinear masks
match all five historical masks; maximum base difference is0.001953125 native CT
voxel, inside the frozen0.01 bound. All valid normals' proposed66-layer envelopes
and trilinear corners are inside the declared native CT dimensions.

The screen queries every valid native base against3,260,832 earlier surface
centres. The51.1-voxel bound exceeds both slabs' normal lengths, both interpolation
corner radii and the admitted base-rounding allowance. This is a conservative
sufficient condition for disjoint sampling, not a physical-sheet classifier.
Seven previous CT boxes are also excluded: two level3 inspections plus all five
retained native chunks from earlier source checks. Excluding whole available
chunks is stricter than excluding only the pixels that were visually plotted.
Level3 support uses the recorded8x scaling plus a boundary allowance; its full
multiscale reconstruction kernel was not rederived. No fresh-support claim rests
on an unverified admission here.

Requiring a257×257 support neighbourhood covers every possible128px patch origin
and the smaller native extent of Hecate's64px patch. It is deliberately more
conservative than an exact, fixed patch schedule. The small surviving cores mean
**the frozen sufficient test fails**, not that all possible target inference is
impossible or that these margins contain no ink. The51.1 sphere, integer CT-box
envelopes, all-origin context and128px core requirement have distinct meanings;
none should be silently changed into an anatomical fact.

## Verification and decision

The16 archived native-checked point receipts match the reconstructed bases and
normals within their stated tolerances. Analytic normal, missing-stencil, slab
bound, discrete interpolation-corner and context-edge fixtures pass. A separate
integral-image implementation exactly reproduces the context masks for all five
meshes; all stage counts and minimum-square outcomes agree. The full support
figure was inspected. Synthetic and retained-output checks verify the computation
and its bounds; they do not verify physical-layer identity or ink accuracy.

No fresh target field is admitted. Preserve this result without weakening its
criteria after observing the outcome. A compact independent task is now prepared
for the user's spare VM: derive the geometry from original TIFFs, challenge the
assumptions, and separately compare the conservative context with one exact
native128px/stride64 origin. That diagnostic cannot alone admit the full
four-model/plain+padded recipe. Inputs contain no raw CT or model weights.

The main reading programme need not wait for an untouched evaluation field.
Previously inspected PHerc0800 tissue can support a clearly labelled development
transfer of the qualified four-model recipe, without fresh-test or accuracy
claims. Next prepare that transfer contract and actual render/compute estimates;
keep independent coverage work separate. Do not resume arbitrary training or
another PHerc343 contrast/tiling search.

GPU spend£0. No target inference, new CT download, R2 upload, public release,
transcription or translation. Actions packaging is operationally separate from
the scientific seal; its run/download receipt is retained in the repository.
