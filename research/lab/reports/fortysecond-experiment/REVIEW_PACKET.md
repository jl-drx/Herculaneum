# Phase42 paired CT review packet

All six centers were fixed from geometry before CT inspection. Every region below has now been visually inspected for structure and possible reference evidence by the root assistant. They are reference-review exposed, not untouched tests. No ink labels were produced. This packet makes the admission decision reviewable; it is not an annotation ground truth.

| Region | Original nominal role | Compact views | Current finding |
| --- | --- | --- | --- |
| pair01 | Training | [Views](review-compact/pair01.png) | Broad edge and bright compact features; local correspondence ambiguous |
| pair02 | Training | [Views](review-compact/pair02.png) | Folded laminae; plane intersects an air/sheet boundary; local correspondence ambiguous |
| pair03 | Selection | [Views](review-compact/pair03.png) | Curved sheet; frozen diagnostic lacks full coverage |
| pair04 | Selection | [Views](review-compact/pair04.png) | Repeated laminae; local correlation alternatives remain ambiguous |
| pair05 | Comparison | [Views](review-compact/pair05.png) | Separated sheet bands; nominal plane sometimes lies in a gap or at an edge |
| pair06 | Comparison | [Views](review-compact/pair06.png) | Repeated laminae and bright feature; frozen diagnostic lacks full blur context |

Each figure shows native high-resolution CT, its fixed Gaussian blur, real coarse CT at the public transform, and real coarse CT with the diagnostic shift. In pairs03/06 the final row repeats the public mapping because no fit was admitted. Columns are three parallel planes and two normal cross-sections. Each panel spans about 307 µm. Display intensity scales are fixed within each acquisition; these are not calibrated cross-scanner attenuation comparisons. White wedges denote missing coverage, not tears or air. The blur row is displayed on the fine grid; the correlation diagnostic also samples a coarse grid.

Bright features are not identified as ink or assigned a chemical composition. Air cannot be a non-ink writing-surface label, and visually quiet papyrus is not a verified negative. No surface recentering follows from these views. Fine structure is visibly resolved differently, but this packet cannot decide whether the difference affects recoverable ink.

Exact old-grid and old/new volume coordinates are in [SELECTION.json](geometry/SELECTION.json); block origins, raw chunk URLs, hashes and shapes are in [CT_PLAN.json](CT_PLAN.json) and [CT receipt](ct/COMPLETE.json). The original 150dpi figures, native blocks, view arrays and coverage masks remain in the cloud lab. Compact figures are replotted from those same arrays for GitHub. Bulk arrays and original large PNGs are not in the selected Git export.

For a qualified reference review, the useful deliverable is a list of ink-positive, non-ink-papyrus and unknown regions with native coordinates, surface/wrap identity, supporting observation and uncertainty. Do not infer these classes from brightness or correlation alone. Three spatially separated structural correspondences, including evidence withheld from fitting, would be needed per admitted real-scan region; the current automated supports are not established physical landmarks.

See [results](RESULTS.md), [recorded root observations](REFERENCE_REVIEW.json), [numerical audit](ADMISSION_VERIFICATION.json), and [next task](NEXT_STUDY_DRAFT.md).
