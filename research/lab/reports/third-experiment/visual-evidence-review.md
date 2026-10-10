# Assistant visual review of the larger PHerc0800 field

Review date: October 8, 2026. Review role: assistant image review, with no specialist papyrological authority.

**I did not identify a coherent stroke sequence supported by the inspected source layers. Transcription: `null`. Translation: `null`.** The larger context and two source-aligned diagnostic closeups show structural texture and broad detector responses, without a defensible reading in these views. This finding does not establish that ink is absent.

## Images inspected directly

The reviewed field is segment `20251028222030`, target volume `20250521135224`, native source X/Y `[512:2048,512:2048]`, at 8.640 µm sampling. Source layers 11, 15 and 19 correspond to recorded normal offsets −4, 0 and +4 native voxels in the forward 31-layer stack.

| Image | Evidence reviewed |
| --- | --- |
| [first-look.png](first-look.png) | Observed layers 11/15/19 with shared source contrast; seed 42, seed 43 and their minimum on the same raw score scale |
| [observed-layers-11-15-19.png](large-analysis/observed-layers-11-15-19.png) | Source-only layer comparison with native source axes |
| [large-field-overview.png](large-analysis/large-field-overview.png) | Aligned source layer, separate seed maps, minimum score, absolute seed difference and support domains |
| [diagnostic-closeups.png](large-analysis/diagnostic-closeups.png) | Two already-ranked diagnostic areas, each shown in all three observed layers and the seed-minimum map with surrounding context |

I viewed the source-layer comparison before the support-domain overview, then inspected the closeups. The closeup locations came from the recorded component ranking; they were not selected by proposing a glyph. No expected wording or glyph identities were used.

The source-only and closeup displays use previously recorded full-field 1st–99th percentile limits: layer 11 `[36,153]`, layer 15 `[36,157]` and layer 19 `[32,162]`, in uint8 intensity units. The closeups reuse these limits without a new local contrast stretch. Model maps use fixed raw score limits `[0.25,0.75]`; the overview's seed-difference map uses `[0,0.25]`. These are display settings, not ink authentication or calibrated likelihoods.

## Field-level observations

The observed source planes contain extensive fine vertical striations, long horizontal gaps or bands, and irregular dark patches. Some local features persist through the planes, while others change size or shape with depth. I cannot trace a set of narrow connected strokes into repeated groups with a convincing common baseline or text-line organization.

The separate seed maps contain bright, compact islands and broader diffuse patches. Several conspicuous responses change in prominence between seeds. Taking their minimum leaves sparse rounded responses, without resolving a coherent stroke sequence at overview scale. The shared support and strict support panels define where comparisons are meaningful; they do not supply evidence that the retained patches are writing.

Overview downsampling limits assessment of small features. The two closer views below were therefore checked before concluding that no supported reading was identified.

## Recorded diagnostics and closeup observations

Coordinates below are native rendered source indices; all boxes are X, Y, width, height with half-open bounds. These remain diagnostic response areas, not identified letters.

| Diagnostic area | Component box | Inspected context box |
| --- | --- | --- |
| Largest primary shared-response component | `[1845,1816,42,60]` | `[1781,1752,170,188]` |
| Largest strict-support shared-response component | `[1699,957,43,39]` | `[1635,893,171,167]` |

At the primary area, layer 11 predominantly shows fine striations inside the marked box. Layer 15 contains an irregular dark patch, and layer 19 shows a larger irregular dark region extending into its upper surroundings. The detector minimum covers a broad vertical area. I do not see a resolved sequence of narrow source-supported strokes or a surrounding group that makes this interpretable as writing.

At the strict-support area, the source contains fine vertical texture intersected by horizontal dark gaps and neighboring irregular patches. The detector minimum is a broad rounded response. I cannot isolate a continuous stroke sequence from the surrounding structural texture across the three displayed planes. Removing poor input support has not made a defensible letter sequence visible here.

The closeup sheet uses native source axes and nearest-pixel display. Its final retained SHA-256 is `ecad4ee4a8ea3b93d04676b0307c143c42e019e9aaa6310d21d23b00124bda35`; the layout was polished without changing pixels, bounds or scales, and I reopened that final image. Its producer verified unchanged source, run and analysis-summary hashes. Raw scan arrays remain preserved.

## Scope and uncertainty

This review covers the displayed larger field and the two recorded closeups, using three observed depth planes. It is not an exhaustive examination of all 31 planes or all other public surfaces. A learned detector can respond to cues that these views do not resolve, so a null visual reading is not a finding of no ink. The detected patches also do not establish a reading merely because they survive support masks or appear in both seeds.

The separate native CT check supports local data correspondence: 92 of 93 samples match exactly and all agree within one intensity count, with the residual unresolved. That check establishes input consistency at its fixed neighborhood; it does not authenticate these diagnostic patches. No source-supported glyph hypothesis, transcription or translation is populated by this assistant review.
