# Prospective native regions for broader paired acquisition

Frozen before processing the new coordinate rasters on 11 October 2026.
This selects a **candidate intake**, not an admitted paired-training dataset.
No CT intensities, annotation values or teacher predictions select the regions.
The ongoing nine-arm native CPU study and its endpoint rules are unchanged.

Use the retained source catalog and the live metadata/ETag receipts. Fetch only
the fifteen native XYZ coordinate TIFFs, at most 8,239,134 bytes. They describe
surface geometry; they are not CT or prediction rasters. Every native render must
match the previously audited commit/scale/group/orientation/crop contract.

Candidate central cores are 128 square, with 256-square native input contexts
and all28 depth planes. Core origins are (64+256i,64+256j), with the complete
context inside the source canvas. Require the entire bilinear coordinate and
nearest-normal stencil, and agreement of the previously audited FP32/F64 index
calculations. An inexpensive mesh-vertex bounding box with a conservative
floating-point allowance screens every candidate. Exact staged FP32 rendering,
the F64 alternative, all-depth endpoint bounds and native trilinear support are
rechecked for every selected context. Reject rather than clamp or repair.

Reserve all265 existing w035/w039/w044 context boxes and the complete three
previously inspected w040 272-square domains. The latter exclusion preserves
their training/validation/prospective-scoring roles; no new intake quietly
absorbs the old correspondence sites. This is a declared exposure inventory,
not a claim to know every upstream teacher's training history.

First choose up to32 w024 and32 w053 contexts, in that order, excluding reserved
boxes and previously selected reference boxes. Then choose up to64 contexts
each from w040,w041,w043, excluding all reserved and newly selected reference
boxes. Within each winding, start at the eligible candidate nearest its XYZ
centroid; greedily maximize the minimum squared XYZ-centre distance to already
selected candidates. Break ties by row then column. Do not replace a selected
region because its later CT/label response is weak. Record all rejected grids.
These fixed quotas define a bounded intake, not a power calculation. Training
contexts can overlap other training contexts physically; they are not claimed
to be independent samples or different papyrus specimens.

Publish selected coordinates, per-context physical boxes, a native CT chunk
request plan, byte ceilings, and an admission table. The native CT plan is
prospective and must fail closed on changed metadata/lengths and missing chunks;
it is not executed in this packet. Chunk responses need SHA-256 receipts at
actual acquisition. Source identity is not guaranteed by an absent ETag pin.

**Not yet admitted:** local fine/native CT registration, common nonzero/source
support, transferred supervision, and the fine teacher's complete contributing
input contexts. A256 fine tile is not a256 native tile; take the union of every
contributing tile, depth window, and interpolation support before repeating the
cross-split physical exclusions. The present overlap result covers native input
contexts only. No affine/raster resize alone establishes local registration.

The alternate w024 fine59 scan's group1 rendering must use its correct geometry
and composed transforms to the native frame. Its teacher is a different model,
not independent human truth. w053 is an author-designated blank control, not an
authenticated all-negative label. Both remain diagnostic reservations until
their required support and exclusions pass. w035/w039/w044 are reused development
references after the current study; no claim of an untouched test is made.

No training, new model forward, paid resource or spare-VM delegation belongs to
this packet. A failed qualification is retained; enlarging/replacing the cohort
requires a new version made before accessing that version's targets.
