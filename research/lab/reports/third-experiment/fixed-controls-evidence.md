# Fixed central controls: evidence and limits

The four predefined seed-42 controls completed successfully on the frozen 1024 × 1024 context at native origin `(768,768)`. Every control scored all 225 scheduled tiles, used the original runner with strict checkpoint matching and four-thread CPU FP32 inference, and preserved its source hashes. Inference took 39.7–41.8 seconds per condition; process wall time was 45.0–47.3 seconds. The retained commands, logs and execution records are in [fixed-controls-execution](fixed-controls-execution/), and input hashes and conditions are in [the frozen plan](fixed-controls-plan.json).

The centered forward control is **bitwise identical** to the full-context seed-42 forward map over all **802,816** comparison pixels: zero unequal pixels and maximum absolute difference zero. Tile phase and local patch inputs match in this interior. This verifies consistency between crop and full-context processing; it does not validate ink.

## Shared support

Both the primary common domain and stricter contributing-tile domain retain **802,816 pixels**, an 896 × 896 core. The entire control canvas is geometry-valid, and every XY location has at least one nonzero intensity under each selected 17-depth window. The core also satisfies the 64-pixel margin from the crop edge and Euclidean distance ≥64 from invalid geometry in the original global mask.

The stricter domain additionally excludes any pixel covered by a scored tile containing invalid geometry or an XY location without selected-depth intensity support. None of the control tiles violate this rule. The primary maps have 56 such scored tiles each elsewhere in the larger context; their footprints do not intersect this common core. Thus boundary exclusions do not account for the seed/depth differences reported here. Nonzero intensity remains an operational data-present convention, not evidence of ink or all-layer coverage.

## Score behavior

The following summaries all use the same 802,816-pixel domain. The 0.5 midpoint is a fixed diagnostic threshold, without target accuracy or ink-confidence meaning.

| Condition | Mean score | Maximum | Pixels >0.5 | Fraction >0.5 |
| --- | ---: | ---: | ---: | ---: |
| Seed 42 center forward `7:24` | 0.302 | 0.778 | 20,898 | 2.603% |
| Seed 43 center forward `7:24` | 0.290 | 0.693 | 11,515 | 1.434% |
| Seed 42 minus 2 forward `5:22` | 0.305 | 0.762 | 28,722 | 3.578% |
| Seed 42 plus 2 forward `9:26` | 0.294 | 0.767 | 10,072 | 1.255% |
| Seed 42 center reverse `7:24` | 0.298 | 0.768 | 8,489 | 1.057% |

Between the two center-forward seeds, mean absolute score difference is **0.0488**, maximum difference **0.4717**, and pixel correlation **0.187**. Only **885 pixels** exceed 0.5 in both maps, compared with **31,528** in either; midpoint Jaccard overlap is **0.0281**. This describes limited agreement between these checkpoints on this field.

Relative to seed-42 center forward, minus-2 and plus-2 controls have correlations **0.651** and **0.637**, mean absolute changes **0.0324** and **0.0305**, and midpoint Jaccard overlaps **0.311** and **0.171**. Reversing the same selected layers has correlation **0.154**, mean absolute change **0.0486**, and midpoint Jaccard **0.0255**. The reversed input changes the model's expected depth ordering; it is a sensitivity diagnostic, not an alternative authenticated interpretation. Pixel counts and correlations do not provide independent sample sizes or statistical accuracy estimates.

## Visual observations

[The fixed-scale overview](fixed-control-analysis/fixed-controls-overview.png) shows observed CT layers 11, 15 and 19 at display range 0–155 and all score maps at display range 0.25–0.75. The source layers contain dense horizontal and vertical material texture, broad dark bands and scattered irregular dark features. Their appearances vary with depth. These panels do not expose a clearly traceable sequence of repeated letter strokes.

The detector maps contain scattered rounded, block-like and elongated diffuse patches. The centered seed-42 crop reproduces the primary seed-42 map as expected. Several prominent responses, including the bright upper-right cluster, weaken or move substantially between seeds and controls. Some broad patches persist across neighboring depth windows, but persistence alone does not distinguish ink from material or rendering structure. The reversal changes the response pattern strongly. No panel supports a reliable coherent letterform or a text-line reading in this context.

These observations leave the physical cause of the responses unresolved. The next research gate is independent confirmation of spatially coherent stroke structure in the measured depth stack and its native geometry, with blinded specialist review and matched background contexts. Score magnitude, seed agreement, or favorable depth selection should not supply a transcription or translation.

Machine-readable paired metrics, input/output hashes, contributing-tile exclusions and domain masks are retained in [the analysis summary](fixed-control-analysis/summary.json). The fixed crop shares source material with the primary run; these controls are reproducible sensitivity checks, not held-out model validation or independent observations.
