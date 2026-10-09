# Larger context and the PHerc1667 method comparison

**The bounded research workflow is feasible, but this larger PHerc0800 view still provides no supported letter reading.** Both detectors and four fixed depth controls completed. The source and model panels show material texture and scattered diffuse responses, with limited seed agreement. Transcription and translation remain empty.

The user also asked whether we are learning from the June 2026 PHerc1667 team. We now have a primary-source [method-by-method comparison](../methods-review/PHERC1667_METHODS.md). We inherit their mesh-based surface sampling and ink-recovery approach, and use their later official 9 µm baseline. The June result additionally required finer observations, a different network, target adaptation, extensive geometry correction and specialist reading. Those missing ingredients guide the next research decision.

## Geometry-selected acquisition

The frozen 1536×1536 crop on segment `20251028222030` begins at native source X/Y `(512,512)`. Its nominal canvas spans 13.271 mm per side, with 2,196,160 conservatively geometry-valid pixels (93.09%)—4.41 times the valid coordinate coverage of the initial tiny segment. Its selection used only geometry, with no new surface intensities or detector maps consulted.

The acquisition retained all 31 depth layers. Of 144 chunk positions, 142 stored objects supplied 72,122,368 bytes (68.78 MiB); two explicit S3 `NoSuchKey` responses used the documented Zarr fill value zero. Authentication or generic errors never become reconstructed source data. Original chunks, remote metadata, meshes, coordinate maps, hashes and the immutable acquisition plan are preserved.

Dense same-volume comparison found no base-coordinate matches to the initial surface within the predeclared 1- or 3-voxel tolerances. Minimum separation was 7.462 voxels, about 64.47 µm. This does not establish separate papyrus layers or independent CT information: adjacent surfaces and overlapping through-depth sampling remain possible. See [geometry selection](geometry-selection.md) and [design review](design-review.md).

At three geometry-fixed center points, the archived renderer calculation reproduced 92 of 93 surface intensity values exactly; all 93 were within one count. The single residual remains unresolved, without a fitted correction. One 2 MiB native CT chunk supplied the check. This is local source/render agreement, not global geometry or ink validation; details are in [the native consistency report](native-ct-consistency.md).

## Model and support results

Both pinned checkpoints passed checksum verification and strict state matching. Each scheduled 529 tiles, scored 523 and skipped six without valid geometry. The two full-field CPU inference times were 95.17 and 84.28 seconds. Normalization, model inference and blending are unchanged from the original adapter; the separate larger adapter widens only size and tile-budget guards.

Primary analysis requires valid geometry, evaluated scores, selected-depth nonzero support, a 64-pixel crop margin and a global geometry margin. The stricter analysis also removes the entire support of any scored tile containing invalid geometry or a completely zero selected-depth profile. Such exclusions are conservative support checks, not ink-negative labels.

| Domain | Analyzed pixels | Both raw seed scores >0.5 | Connected components |
| --- | ---: | ---: | ---: |
| Primary interior | 1,849,282 | 2,758 (0.149%) | 12 |
| Strict contributing-tile support | 1,728,512 | 1,356 (0.078%) | 6 |

The threshold is the fixed midpoint of smoothed training targets; it is not calibrated ink confidence. Components and their coordinate-linked peaks are unresolved image-analysis leads, with null language, text and translation fields. The largest primary component is outside the fixed central control crop; the central sensitivity results do not explain that entire area.

## Fixed depth controls

Four seed42 conditions were run on the independently fixed central 1024×1024 context: forward `7:24`, forward `5:22`, forward `9:26`, and reversed `7:24`. Every run scored 225 tiles. The cropped forward output exactly matches the full-field seed42 output over all 802,816 common central pixels, verifying the crop/tiling computation.

In that fully supported comparison area, seed correlation is 0.187. Only 885 pixels exceed the midpoint in both seeds, versus 31,528 in either; Jaccard overlap is 0.0281. The minus-two and plus-two controls correlate 0.651 and 0.637 with the seed42 forward result; reverse-order correlation is 0.154. These are model sensitivity measurements, not target accuracy. Full measurements and their scope are in [fixed-controls-evidence.md](fixed-controls-evidence.md).

Direct inspection of source layers, full maps and the two already-ranked closeups finds fine striations, gaps and irregular dark features. The shared model responses are broad and rounded, without a coherent source-supported stroke sequence. A [separate assistant visual review](visual-evidence-review.md) agrees; this is not specialist papyrological assessment and does not establish absence of ink.

## What we learned from the June team

The [primary preprint](https://arxiv.org/html/2606.29085v1) documents approximately 2.4 µm scans, a 3D ResNet/2D U-Net detector using 62×256×256 input, fragment supervision followed by five target-adaptation rounds, corrected meshes and SLIM flattening, and reading by eight papyrologists. The released iteration checkpoints are a separate-column ablation family, with no demonstrated identity to the final three-column full-scroll detector.

Our later `ink_9um` model uses 17×128×128 inputs at approximately 9 µm. That matches the available target resolution and is the official tutorial's First Letters baseline. Our PHerc1667 supplied-label control overlaps training and cannot replace the paper's separate-region validation or reading criteria.

The refreshed inventory lists only an 8.640 µm public volume for PHerc0800. All 22 matched First Letters eligible scan pairs in this inventory are 8.640 or 9.362 µm; only PHerc0800 currently lists released surface stacks on those exact pairs. We have not verified a finer eligible substitute. Interpolating these observations cannot restore native 2.4 µm information. The [metadata-only compatibility screen](../../data/experiments/eligible-resolution-inventory/compatible-targets.json) records this limit and its pinned eligibility source.

## Decision and review

Continue method validation before claiming new text. A useful next milestone is a bounded native-resolution PHerc1667 control using the paper's documented ablation and separate-region validation, after exact checkpoint, labels and preprocessing audits. This would teach us how the published adaptation changes a known signal; it would be reference research, with no prize claim. On PHerc0800, target adaptation should wait for an independently reviewed seed signal and a separate physical validation region. Fine-tuning today's diffuse responses could reinforce texture.

Review [the visual package](review.html), [machine-readable field analysis](large-analysis/large-analysis-summary.json), [the depth comparisons](fixed-control-analysis/summary.json), [the method comparison](../methods-review/PHERC1667_METHODS.md) and [feasibility assessment](FEASIBILITY.md). Original source arrays, weights and earlier adapters remain unchanged. No target letters, translation, external publication or submission have been produced.
