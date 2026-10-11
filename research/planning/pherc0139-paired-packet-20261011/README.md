# Broader PHerc0139 paired-acquisition intake

11 October 2026. **256 prospective regions selected from native geometry; paired
training and alternate-reference admission remain conditional.** This work ran
alongside the unchanged native CPU learning study, without its held-out outputs.

The packet replaces an open-ended search with explicit coordinates, exclusions,
source contracts and bounded next acquisitions. It does not claim that these
regions contain ink or that global registration makes pointwise teaching safe.

## What is ready

| Role | Winding | Geometry-valid grid contexts | Clear declared native exclusions | Selected |
|---|---|---:|---:|---:|
| Prospective training | w040 | 606 | 390 | 64 |
| Prospective training | w041 | 595 | 364 | 64 |
| Prospective training | w043 | 603 | 368 | 64 |
| Alternate-reference reservation | w024 | 529 | 502 | 32 |
| Author-designated blank reservation | w053 | 921 | 859 | 32 |

Every selected core is 128 × 128 pixels inside a full 256 × 256 native context,
with all 28 depth planes. Selection uses a regular context grid and deterministic
physical spacing, before new CT/annotation/teacher values. Exact native renderer
stencils and conservative full-depth/trilinear-support boxes pass for all 256.
All selected boxes also fit inside the native scan's catalog extent; that does
not establish nonzero CT support.

The 192 training contexts clear all 265 existing w035/w039/w044 context boxes,
the complete three previously inspected w040 domains, and the 64 new reference
reservations. The new reference boxes are also mutually disjoint and clear the
declared older contexts. This result concerns **native input support**; it does
not yet cover the fine teacher's entire contributing tiles. Training contexts
are not asserted mutually independent, and windings belong to the same scroll.

The [region table](selection/SELECTED_REGIONS.csv) and
[full selection record](selection/RESULTS.json) preserve exact coordinates and
boxes. All rejected grid positions remain in the bundled `CANDIDATES.json`.
Quotas define an intake, not a training-power calculation or 192 known labelled
examples. Later support failures reduce admissible support; do not replace
regions after seeing target responses.

## Acquisition and frame evidence

This packet made 72 public metadata/HEAD requests, receiving 45,059 JSON bytes,
then fetched 15 native XYZ coordinate rasters totalling 8,239,134 bytes. No new
CT intensities, teacher prediction pixels or annotation values were fetched.
Receipts bind metadata, object ETags, lengths and acquired SHA-256 checksums.
The source catalog is the retained release, with selected URLs checked live;
this is not a fresh exhaustive catalog crawl.

The [native CT plan](selection/NEXT_NATIVE_CT_PLAN.json) names exactly 1,024
uncompressed chunks: **469,762,048 bytes (448 MiB)**. It is not executed. Recheck
metadata and freeze object lengths/ETags before acquisition; save SHA receipts,
then audit complete source support. Missing data must not become synthetic zeros.

The [next fine-coordinate plan](NEXT_FINE_GEOMETRY_PLAN.json) names 12 coordinate
TIFFs for the three training windings and w024: **164,736,024 bytes**, also not
executed. It is the next geometric step before defining finite fine-CT and
teacher TIFF-range requests. There is no plan to download whole fine CT scans.

| Observation | Source volume | Source group | Rendered depth | Render group0 spacing, Z/Y/X µm |
|---|---|---:|---:|---|
| Native 113 keV | 20250728140407 | 0 | 28 | 9.362 / 9.362 / 9.362 |
| Fine 78 keV | 20260102150214 | 0 | 109 | 2.399 / 2.399 / 2.399 |
| w024 fine59 | 20260413113053 | 1 | 116 | 2.258 / 2.258 / 2.258 |

Fine78 render group2 pools XY to 9.596 µm while depth remains 2.399 µm. It is
not native 9.362 µm CT. Fine59 is nominally a 1.129 µm acquisition rendered from
source group1; its actual rendered spacing is 2.258 µm. The pinned C++ renderer
sets both the canvas sampling scale and base-coordinate factor to 0.5 for that
source group. Reusing the native group0 mapper unchanged would be wrong.

[SOURCE_CONTRACTS.json](SOURCE_CONTRACTS.json) records the two-affine chain from
full-resolution fine59 XYZ through fine78 into native XYZ. Only **10 of the32
w024 reservations** have their entire native context box inside fine59's catalog
extent under that affine. This is an approximate acquisition-extent screen,
not a local correspondence result; the other22 are not silently replaced, and
partial overlap does not mean a complete valid reference. Exact group1 renderer
mapping, masks and local CT correspondence still need qualification.

## Supervision and evaluation admission

| Candidate | Defensible use now | Still required |
|---|---|---|
| w040/w041 | Broader native-annotation candidates; public native and fine label assets are inventoried | Mask/provenance/source-support audit for selected regions; transferred targets additionally require correspondence |
| w043 | Fine-teacher/fine-annotation training candidate | No native label set appears in the retained catalog; do not present transferred labels as native human truth |
| w024 fine59 | Different acquisition/model diagnostic reservation | Group1 mapping, common support, local registration, complete teacher-context exclusions and training-history caveat |
| w053 | Author-designated blank diagnostic reservation | Native support; preserve uncertainty that this is universally ink-free; no AP against an invented all-negative ground truth |
| w035/w039/w044 | Current study's frozen native-annotation endpoints; development references afterward | Keep current endpoint lock; future reuse is not an untouched test |

The fine78 teacher is model20260417190342, v42, with depth window[23,85) and
256/128 tile/stride. The alternate fine59 teacher is model20260709123958, v43,
declared ResNet3D-152, depth[27,89), reverse enabled. A different teacher/scan
does not prove independent training history or physical ink accuracy. w024 and
w053 were excluded from the author's reported native-dense training config,
but were repeatedly used as benchmarks; broader teacher ancestry is unresolved.

The author's blank designation is useful for measuring response distributions
and strong-response area under a training-fixed threshold. It is not sufficient
to call every response a proven false positive. Keep native annotation agreement,
alternate-teacher agreement and blank-region response as separate endpoints.

## Decision for the next learning comparison

Finish the existing nine native fits and assess their frozen transfer result.
If fitting succeeds but transfer fails, broader coverage is a stronger next
intervention than another small depth/fusion adapter. w040/w041 provide a native
annotation route that does not depend on fine registration; w043 cannot join
that route under a fictitious native-label status.

For a paired-information comparison, qualify the selected fine geometry and
common support first. Freeze patch draws, update budget, mask support and loss
weights across arms. Adding w043 supervision only to a fine-teacher arm would
mix coverage with information quality. Either give both arms that coverage under
an explicitly controlled target policy, or keep the primary comparison on common
labelled support and test coverage separately. A released high-resolution teacher
also changes model/training ancestry; gains against an ordinary learner would
not isolate acquisition quality by themselves. Matched native/fine teachers on
the same eligible physical training regions are the cleaner eventual contrast.

The prior w040 correspondence failures and soft-target controls remain in force.
No additional uncertainty architecture is justified by this intake. The current
native comparison needs no alignment with fine CT to finish and remains unchanged.

## Verification and portability

`verify_packet.py` passes62 checks spanning bound inputs, exact cohorts/exclusions,
complete native chunk coverage, analytic plane geometry, all28-depth support,
edge/invalid-stencil rejection and explicit refusal of paired-training admission.
Selection is deterministic and has been replayed without target pixels. See
[PORTABILITY.md](PORTABILITY.md) for the checksummed private bundle and replay.
These checks establish geometry/plumbing claims, not annotation or ink validity.

The separate [GPU qualification packet](../gpu-qualification-20261011/README.md)
contains a digest-pinned prospective CUDA image and a CPU-tested timing/restart
harness. No paid GPU, cloud secret, Actions workflow or external VM job was
created for this work. The user's £10 total pilot cap remains in force.
