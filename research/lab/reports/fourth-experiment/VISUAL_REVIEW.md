# Independent visual evidence review

I inspected `analysis/screen-overview.png` and `analysis/condition-sensitivity.png` directly. I do not see a coherent, source-supported sequence of strokes or letterforms in these displays. There are reproducible areas of detector response worth investigating, but they do not establish ink or a reading. This is an image review, not specialist papyrological authentication.

## Visible structure

The center source layer shows dense material striations, broad dark irregular regions, and isolated bright specks. The fixed maximum projection across layers 7:24 shows a lighter, woven-looking texture; several broad dark features in the center layer largely disappear in that projection. Neither display reveals an unambiguous sequence of narrow strokes corresponding to the detector response. A depth maximum is a composite and cannot by itself identify a written surface.

Both seeds produce a rounded response near native source X1280, Y500 and a smaller, irregular response near X1105, Y625. Their extensions differ: seed 43 has a broader upper horizontal ridge, while seed 42 has a more localized upper lobe. Thresholding their common strict domain leaves separated islands rather than a joined letter sequence. Other conspicuous responses lie near the displayed crop edges and are outside the primary interior domain. The central responses remain ambiguous between ink-related structure and material, rendering, or model effects.

## Fixed controls

The minus-two and plus-two depth controls retain the main response locations, with changes in strength and breadth. This is more encouraging than an entirely unstable detector pattern, but the shifted windows share 15 of their 17 layers with the center window. The reported correlations of 0.939 and 0.878 therefore measure strongly overlapping inputs, not independent confirmation of ink.

Reversing the layer order substantially weakens the bright islands and changes the upper response into a faint broad ridge. The difference image concentrates around the original islands. The reverse correlation of 0.250 shows sensitivity to depth order; a model trained with a fixed orientation may exhibit that sensitivity on either ink or nuisance structure, so this control cannot authenticate or reject a feature by itself.

In the common strict interior, the seeds share 3,019 pixels above the fixed 0.5 diagnostic threshold out of 65,536 pixels (4.61%); correlation is 0.510 and threshold Jaccard is 0.302. These are model-behavior summaries, not accuracy or calibrated ink certainty. There are no target labels or independently supplied readings.

## Limits and next gate

The field is only about 3.32 mm across and its primary interior is smaller. It supplies little surrounding line context. Geometry-based selection preceded inspection of the new intensities and predictions, but separation of base coordinates does not establish independence of complete through-depth sampling footprints.

The strongest next gate is to check the native CT placement and depth profiles of these response areas against the rendered surface geometry, including normals and interpolation support. If that check supports a plausible surface feature, acquire a larger, fixed surrounding field to test whether the response develops into coherent strokes with consistent spatial context. Any follow-up centered on these newly observed response islands must be recorded as a post hoc diagnostic selection, not held-out validation. These displays do not justify training labels, glyph identities, or a translation.
