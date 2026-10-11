# Study52 independent seed43 review

Reviewed 11 October 2026. Reviewer: `/root/supervision_design_review`.

## Judgment

**I do not identify a candidate meeting the frozen stroke/text-like-group criterion in the seed43 contribution.** The most conspicuous nonzero response is the lower part of cell1 at d+15 in forward order. It recurs at the same broad location in plain and padded seed43, and Hecate also responds strongly over overlapping tissue at d+15. That is real visual co-location of model responses, not evidence that the response is accurate. I cannot identify a common stroke-group organization across those families: seed43 is a broad branched mass, whereas Hecate produces a larger multi-lobed/partly enclosed connected mass with different internal gaps and branches. I would retain this as an **uncertain broad response**, not promote it to a resolved stroke/text-like candidate merely because it is conspicuous in both families.

This conclusion is limited to my assigned seed43 review plus the Hecate views listed below. It is not a complete independent judgment of seed42 or dense_native. I read the frozen `PROTOCOL.md` before viewing maps and have not read other reviewers' conclusions. No inference, new data, fitted contrast, image editing, map aggregation or new crop was used.

## View-by-view observations

Each referenced image places d−15, d0 and d+15 vertically in that order, with fixed 0–255 display and brown unknown support. The three fixed cells span x512:1024, x1024:1536 and x1536:2048, all y1024:1536, in the existing development region.

| Seed43 phase/order | Cell1 | Cell2 | Cell3 |
|---|---|---|---|
| Plain / forward | d−15 has separated diffuse patches. d0 has a broad central response and isolated edge responses. d+15 creates a conspicuous broad branch in the lower half and scattered upper/side spots; no resolved neighbouring stroke group. | d−15 has an extended lower-left diffuse structure. d0 is mostly scattered blobs. d+15 has several separated broader/short elongated responses; they do not organize into a recurring text-like group. | All depths show separated soft patches or short diffuse elongations. Locations and strengths change; no coherent new group. |
| Padded / forward | The lower-cell1 d+15 branch remains at the same broad location, while its left extension and nearby separate patches change. d−15/d0 shapes and strengths also change. Same weak broad-response lead, not a positive stroke-group judgment. | Several plain-phase patches persist broadly, but connectivity and dominance differ. The d+15 lower-left response persists; no repeated text-like arrangement. | Some d−15 patches strengthen and d+15 responses weaken/reorganize relative to plain. No group meeting the criterion. |
| Plain / reverse | Nonzero depths show several diffuse elongated/rounded responses. The forward d+15 lower-cell1 branch is not reproduced in that form; that fact is descriptive, not a required invariance failure. | d−15 gives the strongest connected-looking central/lower forms in this order, including a thin lower horizontal extension, but the arrangement differs from forward and is diffuse. d+15 remains separated patches. No cross-phase/family stroke group identified. | Isolated patches, with a strong lower-right d+15 response near the strip edge; no linked new group. |
| Padded / reverse | d−15 becomes more diffuse; d0 patches become more distinct. d+15 retains dispersed responses, not the forward branch. | d−15 retains broad central forms but the thin plain-phase horizontal extension is not comparably resolved. d+15 has isolated lower/side blobs. No qualifying group. | The lower-right d+15 response persists broadly; other responses vary. No recurring text-like group. |

The d+15 forward lower-cell1 feature is approximately centred around local cell x200–280, y350–460 (global roughly x712–792, y1374–1484); these are descriptive visual coordinates, not a new fitted box or a precise localisation measurement. Hecate's d+15 forward response extends across a substantially larger part of the lower cell, with two dark internal gaps in the forward maps that are not preserved as the same stroke topology in seed43. Plain/padded Hecate reproduce that broad configuration well. Hecate reverse d+15 retains large connected responses over the left cell, but changes their organisation. The order change is not a known negative and is not being used as a veto.

At d−15, seed43 reverse has relatively organised-looking diffuse responses in cell2. The corresponding Hecate reverse full strips do not show a corroborating stroke group at those central positions; their conspicuous responses are largely elsewhere. Across the remaining full-strip locations, isolated bright responses occur in both families, but I do not find a same-location, newly resolved multi-stroke organisation that survives the two padding phases.

## Uncertainty and stopping interpretation

There is enough shared broad response to avoid saying that the models found “nothing.” Its physical cause is unresolved: papyrus texture/geometry, a neighbouring wrap, artefacts, real weak ink or combinations remain possible. The fixed review criterion asks for structure, not response amplitude or agreement alone. My assessment is **no demonstrated qualifying structure**, with the lower-cell1 d+15 forward response explicitly recorded as uncertain.

These qualitative judgments have no measured sensitivity or false-positive rate. A negative judgment here cannot exclude ink, other intermediate depths, other regions or methods. I would follow the protocol's stopping rule for this seed43 evidence: retain the maps, do not select a brightest depth or add intermediate depths/crops, and combine this record with the separate dense_native/seed42/Hecate reviews before making the study-level decision. No authenticated character, transcription or accuracy claim follows.

If another reviewer independently identifies a specific stroke group in the already fixed evidence, it should be reconciled against these exact panels and the relevant retained CT context; my uncertainty does not authorize a new search or inference.

## Exact images actually inspected

The 4 seed43 full-strip images and all 12 seed43 fixed-cell images were viewed, so every seed43 depth/phase/order map was inspected both in full-strip context and in its three predetermined cells. All 4 Hecate full strips and its 4 cell1 panels were additionally viewed. Each file contains all three depths. Full-strip tool output was automatically displayed at 1504×1632 from source1568×1702; fixed-cell images were requested at original detail. No new resampling or enhancement was saved.

Seed43 full strips:

- `full-strips/hybrid_3d2d-seed43-plain-forward-full.png`
- `full-strips/hybrid_3d2d-seed43-padded-forward-full.png`
- `full-strips/hybrid_3d2d-seed43-plain-reverse-full.png`
- `full-strips/hybrid_3d2d-seed43-padded-reverse-full.png`

Seed43 fixed cells:

- `cells/hybrid_3d2d-seed43-plain-forward-cell1.png`
- `cells/hybrid_3d2d-seed43-plain-forward-cell2.png`
- `cells/hybrid_3d2d-seed43-plain-forward-cell3.png`
- `cells/hybrid_3d2d-seed43-padded-forward-cell1.png`
- `cells/hybrid_3d2d-seed43-padded-forward-cell2.png`
- `cells/hybrid_3d2d-seed43-padded-forward-cell3.png`
- `cells/hybrid_3d2d-seed43-plain-reverse-cell1.png`
- `cells/hybrid_3d2d-seed43-plain-reverse-cell2.png`
- `cells/hybrid_3d2d-seed43-plain-reverse-cell3.png`
- `cells/hybrid_3d2d-seed43-padded-reverse-cell1.png`
- `cells/hybrid_3d2d-seed43-padded-reverse-cell2.png`
- `cells/hybrid_3d2d-seed43-padded-reverse-cell3.png`

Hecate corroboration views:

- `full-strips/hecate-plain-forward-full.png`
- `full-strips/hecate-padded-forward-full.png`
- `full-strips/hecate-plain-reverse-full.png`
- `full-strips/hecate-padded-reverse-full.png`
- `cells/hecate-plain-forward-cell1.png`
- `cells/hecate-padded-forward-cell1.png`
- `cells/hecate-plain-reverse-cell1.png`
- `cells/hecate-padded-reverse-cell1.png`

I did not inspect other reviewers' notes, the seed42/dense_native maps, additional CT panels, or the numerical comparison summary as part of this visual judgment. The parent reports completion and byte checks passed; this review does not duplicate or extend those execution-verification claims.
