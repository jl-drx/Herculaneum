# Independent visual evidence review

Reviewed `full-surface-baseline.png` and `candidate-context.png` on 7 October 2026. This review used the supplied panels only; no model runs, labels, recovered reading, or suggested ancient text were used.

**The displayed evidence does not currently support a coherent letterform or transcription.** The candidate is an ambiguous model response that merits diagnostics. This does not establish that ink is absent: ink-sensitive volumetric signals need not be plainly visible in a single CT slice.

In the roughly 2.212 mm context around source XY ≈652,563:

- The observed layers contain repeated approximately horizontal bands, several near-vertical structures, irregular dark gaps and isolated bright points. These structures extend through the context and suggest substantial material/geometry texture against which an ink interpretation must be tested. Their chemical or physical cause cannot be identified from these views alone.
- The strongest shared seed response is a broad, soft-edged patch around XY ≈650,570–590. It does not resolve a narrow connected stroke, stable junction, recognizable closed contour, or an ordered sequence of separated character shapes. Small bright source points occur within or near this patch; their presence is not evidence that the surrounding model score is writing.
- Seed42 also has a stronger upper response around XY ≈630,470 than seed43. The central response is shared, but the extent and nearby responses differ. Shared training data means this agreement is a stability observation rather than independent authentication.
- The source configuration and contrast change between layers 11, 15 and 19, particularly around the left-side dark gaps and vertical structures. The panels do not yet show that a proposed stroke follows one physical surface consistently through depth.

The hotspot was selected after inspecting joint model scores, using minimum seed score above 0.5 and an eroded validity mask. That makes this exploratory review, not a blind test. A score threshold has not been calibrated on target labels. The corrected display `(p−0.25)/0.5` emphasizes the model output visually; it does not validate ink. The eroded mask reduces direct boundary concerns but does not validate the local flattening, material texture, or surface-normal direction. No accuracy estimate or letter identification is supported by these images.

**Next diagnostic:** complete the predefined depth-window and direction controls before choosing another rendering. At the same fixed source coordinates, compare raw score maps and source slices using common display limits; overlay the candidate on the independently checked surface/mesh mapping. Check whether its position and geometry persist across small depth changes, whether it follows a strong material band or a flattening discontinuity, and how similarly textured nearby regions score. Retain all control results, including disagreement. A persistent localized response would justify closer stroke-level review; persistence alone would still not supply a letter reading.
